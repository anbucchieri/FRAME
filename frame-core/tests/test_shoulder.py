"""ISB humerothoracic shoulder coordinates (frame_core.shoulder)."""
import numpy as np
import pytest

from frame_core import shoulder as sh

D = np.deg2rad


def _angles(side, clav, gh):
    k = sh.kinematics(side, np.atleast_2d(clav).astype(float), np.atleast_2d(gh).astype(float))
    return np.rad2deg(k["angles"][0]), k


def test_arm_hanging_is_zero_elevation():
    a, _ = _angles("right", [0, 0], [D(90), 0, 0])
    assert a[1] == pytest.approx(0.0, abs=1e-9)


def test_pure_abduction_is_plane_zero():
    for alpha in (30, 60, 90, 120):
        a, _ = _angles("right", [0, 0], [D(90 - alpha), 0, 0])
        assert a[1] == pytest.approx(alpha, abs=1e-6)
        assert a[0] == pytest.approx(0.0, abs=1e-6)
        assert a[2] == pytest.approx(0.0, abs=1e-6)


def test_forward_flexion_is_plane_ninety():
    # from arm-hanging (X = 90), the humerus z axis is the mediolateral axis: Z turns it in the
    # sagittal plane. Whichever sign moves the hand forward must read plane = +90.
    for beta in (30, 60, 90):
        for sgn in (1, -1):
            a, k = _angles("right", [0, 0], [D(90), 0, sgn * D(beta)])
            H = k["angles"]
            if abs(a[0] - 90) < 1e-6:
                assert a[1] == pytest.approx(beta, abs=1e-6)
                break
        else:
            pytest.fail("no sign of shoulder_Z produced forward flexion")


def test_axial_rotation_about_the_long_axis():
    # arm hanging, rotate about the humerus long axis (humerus Y in the DHM)
    a, _ = _angles("right", [0, 0], [D(90), D(25), 0])
    assert a[1] == pytest.approx(0.0, abs=1e-6)
    # at 0 elevation phi and psi are degenerate; at 60 deg abduction psi is well defined
    a, _ = _angles("right", [0, 0], [D(30), D(25), 0])
    assert a[1] == pytest.approx(60, abs=1e-6)
    assert abs(a[2]) == pytest.approx(25, abs=1e-6)


def test_left_arm_is_the_mirror_of_the_right():
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = rng.uniform(-0.3, 0.3, 2); g = rng.uniform([-0.5, -0.8, -0.8], [2.0, 0.6, 1.5])
        r, _ = _angles("right", c, g)
        l, _ = _angles("left", [-c[0], -c[1]], [-g[0], g[1], -g[2]])
        assert np.allclose(r, l, atol=1e-9)


def test_projected_torques_are_power_consistent():
    """sum_k tau_k qdot_k == M . omega, with omega the humerothoracic angular velocity."""
    rng = np.random.default_rng(2)
    dt = 1e-6
    for side in ("right", "left"):
        for _ in range(40):
            c0 = rng.uniform(-0.2, 0.2, 2); g0 = rng.uniform([-0.5, -0.6, -0.6], [1.2, 0.5, 1.2])
            cd = rng.normal(size=2); gd = rng.normal(size=3)
            k0 = sh.kinematics(side, c0[None], g0[None])
            if np.rad2deg(k0["angles"][0, 1]) < 25:
                continue
            kp = sh.kinematics(side, (c0 + cd * dt)[None], (g0 + gd * dt)[None])
            km = sh.kinematics(side, (c0 - cd * dt)[None], (g0 - gd * dt)[None])
            rates = (kp["angles"][0] - km["angles"][0]) / (2 * dt)
            M = rng.normal(size=3)                                   # GH moment, DHM thorax
            tau_gh = np.einsum("ij,i->j", k0["A"][0], M)[None]       # tau_i = a_i . M
            tau_e = sh.project_torque(k0, tau_gh)[0]
            omega = k0["B"][0] @ np.r_[cd, gd]                       # DHM thorax
            assert tau_e @ rates == pytest.approx(M @ omega, rel=1e-5, abs=1e-7)


def test_torque_capacity_bounds_every_feasible_moment():
    rng = np.random.default_rng(3)
    effort = np.array([69.0, 90.0, 51.0])
    k = sh.kinematics("right", np.zeros((1, 2)), np.array([[D(40), D(10), D(30)]]))
    cap = sh.torque_capacity(k, effort)[0]
    for _ in range(2000):
        tau = rng.uniform(-1, 1, 3) * effort
        te = sh.project_torque(k, tau[None])[0]
        assert np.all(np.abs(te) <= cap + 1e-9)
    # and it is attained at a vertex of the effort box
    best = max(abs(sh.project_torque(k, (np.array(s) * effort)[None])[0][1])
               for s in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3))
    assert best == pytest.approx(cap[1])


def test_transform_replaces_columns_masks_and_derives_limits():
    from frame_core.types import Capture, Repetition
    names = ["right_clavicular_X", "right_clavicular_Z", "right_shoulder_X", "right_shoulder_Y",
             "right_shoulder_Z", "right_elbow_Z"]
    N = 200
    t = np.linspace(0, 1, N)
    ang = np.zeros((N, 6))
    ang[:, 2] = D(90) - D(100) * np.sin(np.pi * t)        # abduct 0 -> 100 -> 0 deg
    tau = np.random.default_rng(0).normal(size=(N, 6))
    lim = {n: {"lower": -1.0, "upper": 1.0, "effort": 50.0, "velocity": 10.0} for n in names}
    lim["right_shoulder_X"].update(lower=-1.57, upper=2.4)
    cap = Capture(reps=[Repetition(angles=ang, torques=tau, gravity_torques=tau.copy())],
                  dof_names=names, urdf_limits=lim)
    out, info = sh.to_elevation(cap)
    assert out.dof_names == ["right_clavicular_X", "right_clavicular_Z", "right_shoulder_plane",
                             "right_shoulder_elevation", "right_shoulder_rotation", "right_elbow_Z"]
    el = np.rad2deg(out.reps[0].angles[:, 3])
    assert el.max() == pytest.approx(100, abs=0.1)
    masked = el < 20
    assert np.all(np.isnan(out.reps[0].angles[masked, 2]))
    assert np.all(np.isfinite(out.reps[0].angles[~masked, 2]))
    assert np.all(np.isfinite(out.reps[0].torques[:, 2:5]))          # torques never masked
    np.testing.assert_allclose(out.reps[0].torques[:, [0, 1, 5]], tau[:, [0, 1, 5]])
    assert info["right"]["valid_frac"][0] == pytest.approx(np.mean(~masked))
    for c in ("plane", "elevation", "rotation"):
        e = out.urdf_limits[f"right_shoulder_{c}"]
        assert e["effort"] > 0 and e["velocity"] > 0 and "capacity_basis" in e
    assert "right_shoulder_X" not in out.urdf_limits
    assert out.urdf_limits["right_shoulder_elevation"]["upper"] > D(90)
    # idempotent on a capture that no longer has the raw chain
    again, info2 = sh.to_elevation(out)
    assert again is out and info2 == {}


def test_rename_maps_the_dof_list():
    raw = ["right_clavicular_X", "right_clavicular_Z", "right_shoulder_X", "right_shoulder_Y",
           "right_shoulder_Z"]
    assert sh.rename(raw)[2:] == ["right_shoulder_plane", "right_shoulder_elevation",
                                  "right_shoulder_rotation"]
    assert sh.rename(raw, "dhm") == raw


def _shoulder_capture(n_reps=3, N=600, fs=240.0, seed=0):
    """Clavicle + GH + elbow, abducting 5 -> 110 deg and back (dips below the 20 deg mask)."""
    from frame_core.types import Capture, Repetition
    names = ["right_clavicular_X", "right_clavicular_Z", "right_shoulder_X", "right_shoulder_Y",
             "right_shoulder_Z", "right_elbow_Z"]
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_reps):
        t = np.arange(N) / fs
        ang = np.zeros((N, 6))
        ang[:, 2] = D(90) - D(5) - D(105) * np.sin(np.pi * t / t[-1]) ** 2
        ang[:, 4] = D(20) * np.sin(2 * np.pi * t)
        ang[:, 5] = D(60)
        tau = np.column_stack([np.zeros(N), np.zeros(N), 8 * np.sin(D(90) - ang[:, 2]),
                               rng.normal(0, 0.1, N), 2 * np.cos(2 * np.pi * t), np.full(N, 2.0)])
        reps.append(Repetition(angles=ang, torques=tau, gravity_torques=0.9 * tau))
    lim = {n: {"lower": -1.0, "upper": 1.0, "effort": 50.0, "velocity": 10.0} for n in names}
    lim["right_shoulder_X"].update(lower=-1.57, upper=2.4)
    lim["right_elbow_Z"].update(lower=-0.08, upper=3.14)
    return Capture(reps=reps, dof_names=names, fs=fs, urdf_limits=lim)


def test_analyze_tolerates_masked_angles():
    """The whole pipeline runs on elevation coordinates whose plane / rotation are partly NaN,
    and every metric of the masked DOFs is computed over the defined samples only."""
    import warnings
    from frame_core import analyze
    from frame_core.synthetic import default_config
    cap, info = sh.to_elevation(_shoulder_capture())
    assert 0 < info["right"]["valid_frac"][0] < 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # no all-NaN reductions
        res = analyze(cap, default_config(), urdf_limits=cap.urdf_limits)
    js = res["joint_space"]["per_dof"]
    for d in ("right_shoulder_plane", "right_shoulder_rotation", "right_shoulder_elevation"):
        for k in ("rom_deg", "peak_vel_dps", "rms_vel_dps"):
            assert np.isfinite(js[d][k]["median"]), (d, k)
    assert 95 < js["right_shoulder_elevation"]["rom_deg"]["median"] < 115     # ~5 -> 110 deg
    pa = res["passivity"]["per_dof"]["right_shoulder_plane"]
    assert np.isfinite(pa["G"]["median"]) and np.isfinite(pa["n_reversals"]["median"])
    assert res["pca"]["ok"] and 0 < res["pca"]["frames_used_frac"] < 1
    assert res["dynamics"]["per_dof"]["right_shoulder_elevation"]["torque_capacity_Nm"] > 0


def test_curve_bands_tolerate_masked_samples():
    from frame_core.curves import trial_curves, stack_curves, bands
    cap, _ = sh.to_elevation(_shoulder_capture())
    b = bands(stack_curves([trial_curves(cap, ["right_shoulder_plane"], n_resample=100)]))
    band = b["right_shoulder_plane"]["angle_deg"]
    assert band["n"] == 3
    assert np.isnan(band["mean"][0]) and np.isfinite(band["mean"][50])   # masked at the start
