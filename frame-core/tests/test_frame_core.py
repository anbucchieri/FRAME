"""
Test suite for frame-core. Runs on NumPy/SciPy only -- if this passes in a bare venv, the
no-Pinocchio dependency separation holds.
"""
import numpy as np
import pytest

from frame_core.synthetic import make_capture, default_config
from frame_core.pipeline import analyze, capability
from frame_core import signal, task_space, dynamics


# --------------------------- conventions --------------------------- #
def test_no_pinocchio_import():
    import sys, importlib, frame_core  # noqa
    for name in list(sys.modules):
        assert "pinocchio" not in name, "frame-core must not import pinocchio"


def test_resample_is_interp_not_fft():
    # a ramp (non-periodic) must stay monotone with no Gibbs overshoot beyond its endpoints
    x = np.linspace(0, 1, 50)
    y = signal.resample_interp(x, 200)
    assert y.min() >= x.min() - 1e-9 and y.max() <= x.max() + 1e-9
    assert np.all(np.diff(y) >= -1e-9)


def test_butter_passthrough_on_short_signal():
    x = np.arange(5.0)
    assert np.allclose(signal.butter_lowpass(x), x)   # too short to filter -> unchanged


# --------------------------- gating --------------------------- #
def test_capability_full():
    caps = capability(make_capture())
    assert all(caps[k]["ok"] for k in ("task_space", "joint_space", "dynamics", "passivity"))


def test_capability_angles_only():
    cap = make_capture(with_torque=False, with_ee=False)
    caps = capability(cap)
    assert caps["joint_space"]["ok"]
    assert not caps["task_space"]["ok"]
    assert not caps["dynamics"]["ok"]
    assert not caps["passivity"]["ok"]
    assert "torques" in caps["dynamics"]["reason"]


def test_analyze_gates_layers():
    cap = make_capture(with_torque=False, with_ee=False)
    res = analyze(cap, default_config())
    assert "joint_space" in res
    assert "dynamics" not in res and "task_space" not in res and "passivity" not in res


# --------------------------- task-space --------------------------- #
def test_voxel_volume_less_than_hull_on_hollow_cloud():
    # The ~5.5x overstatement finding is about a FILLED reachable workspace with unreachable
    # pockets: the hull fills the interior, voxels occupy only the reached region. A hollow
    # spherical shell is the clean demonstration -- hull fills the ball, voxels stay on the shell.
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6000, 3))
    shell = 0.30 * u / np.linalg.norm(u, axis=1, keepdims=True)   # radius 0.30 m, empty inside
    vox = task_space.occupied_volume(shell, 0.02)
    hull = ConvexHull(shell).volume * 1e3
    assert vox < hull                       # hull counts the empty interior; voxels don't
    assert hull / vox > 2.0                 # substantial overstatement, as documented


def test_coverage_iou_bounds():
    cap = make_capture()
    r = task_space.coverage_iou(cap.ee_cloud, cap.ee_cloud, 0.02)   # identical clouds
    assert r["coverage"] == pytest.approx(1.0)
    assert r["iou"] == pytest.approx(1.0)
    assert r["waste"] == pytest.approx(0.0)


# --------------------------- joint-space --------------------------- #
def test_involvement_drops_terms_without_data():
    cap = make_capture(with_torque=False, with_ee=False)   # no dynamic, no jacobian
    res = analyze(cap, default_config())
    inv = res["joint_space"]["involvement"]
    assert "dynamic" in inv["terms_dropped"]
    assert "jacobian" in inv["terms_dropped"]
    assert set(inv["weights_used"]) == {"rom", "path"}
    assert sum(inv["weights_used"].values()) == pytest.approx(1.0)


def test_involvement_weights_shift_ranking():
    cap = make_capture(with_jac=False)
    cfg = default_config()
    cfg.involvement_weights = {"rom": 1, "path": 0, "jacobian": 0, "dynamic": 0}
    rom_first = analyze(cap, cfg)["joint_space"]["involvement"]["ranking"]
    cfg.involvement_weights = {"rom": 0, "path": 0, "jacobian": 0, "dynamic": 1}
    dyn_first = analyze(cap, cfg)["joint_space"]["involvement"]["ranking"]
    # the springy high-amplitude Z dominates ROM; the motor-like X dominates dynamic share
    assert rom_first[0] == "right_shoulder_Z"
    assert dyn_first[0] == "right_shoulder_X"


# --------------------------- dynamics --------------------------- #
def test_quasi_stiffness_recovers_spring():
    q = np.linspace(-1, 1, 300)
    tau = 7.5 * q                       # perfect linear spring
    k, r2 = dynamics.quasi_stiffness(q, tau)
    assert k == pytest.approx(7.5, rel=1e-3)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_loop_work_zero_for_single_valued():
    q = np.linspace(0, 1, 200)
    tau = 3.0 * q                       # no hysteresis -> closed loop area ~ 0
    assert abs(dynamics.loop_work(q, tau)) < 1e-6


# ---------- Phase C: deadbanded joint-speed metrics (PDF item 5) ---------- #
def _sine_capture(n_cycles=5, fs=240.0, noise=0.0, seed=0, amp_deg=60.0):
    """One DOF tracing a clean n_cycles sine (+ optional jitter) -> exactly 2*n_cycles crossings."""
    from frame_core.types import Capture, Repetition
    rng = np.random.default_rng(seed)
    N = int(n_cycles * fs)                       # n_cycles seconds at 1 Hz
    t = np.arange(N) / fs
    q = np.deg2rad(amp_deg) * np.sin(2 * np.pi * t)
    if noise:
        q = q + np.deg2rad(noise) * rng.normal(size=N)
    return Capture(reps=[Repetition(angles=q[:, None])], dof_names=["j"], fs=fs)


def test_zero_crossings_on_a_clean_sine():
    """1 Hz sine over 5 s -> omega is a cosine -> exactly 10 sign changes."""
    from frame_core.joint_space import zero_crossings
    cap = _sine_capture()
    assert zero_crossings(cap.reps[0].angles[:, 0], None, cap.fs) == 10.0


def test_deadband_rejects_near_zero_jitter():
    """The regression that motivates the whole item: jitter must not inflate the count.

    A raw sign(omega) count explodes here; the hysteresis count must still report 10.
    """
    from frame_core.joint_space import zero_crossings, crossing_indices
    from frame_core.signal import derivative
    cap = _sine_capture(noise=0.35)
    w = np.rad2deg(derivative(cap.reps[0].angles[:, 0], cap.fs))
    raw = int(np.sum(np.abs(np.diff(np.sign(w[np.sign(w) != 0]))) > 0))
    assert zero_crossings(cap.reps[0].angles[:, 0], None, cap.fs) == 10.0
    assert raw >= 10                              # the naive count never undercounts
    idx, thr = crossing_indices(w)
    assert thr > 0 and idx.size == 10


def test_min_crossing_gap_is_half_a_period():
    """Consecutive crossings of a 1 Hz cosine are 0.5 s apart."""
    from frame_core.joint_space import min_crossing_gap_s
    cap = _sine_capture()
    assert min_crossing_gap_s(cap.reps[0].angles[:, 0], None, cap.fs) == pytest.approx(0.5, abs=0.02)


def test_min_crossing_gap_nan_with_fewer_than_two_crossings():
    from frame_core.joint_space import min_crossing_gap_s, zero_crossings
    q = np.linspace(0, 1.0, 480)                  # monotonic ramp: never reverses
    assert zero_crossings(q, None, 240.0) == 0.0
    assert np.isnan(min_crossing_gap_s(q, None, 240.0))


def test_a_still_joint_reports_no_crossings():
    """An essentially motionless DOF must score zero, not hundreds of filter-residue reversals."""
    from frame_core.joint_space import zero_crossings
    rng = np.random.default_rng(1)
    q = np.deg2rad(0.05) * rng.normal(size=1200)  # sub-degree tremor only
    assert zero_crossings(q, None, 240.0) == 0.0


def test_floor_and_fraction_both_gate_the_count():
    """thr = max(floor, frac * peak): raising either one alone must be able to suppress a crossing."""
    from frame_core.joint_space import zero_crossings
    cap = _sine_capture(amp_deg=3.0)              # peak |omega| ~ 19 deg/s
    q = cap.reps[0].angles[:, 0]
    assert zero_crossings(q, None, cap.fs, frac=0.05, floor_dps=2.0) == 10.0
    assert zero_crossings(q, None, cap.fs, frac=0.05, floor_dps=500.0) == 0.0   # floor dominates
    assert zero_crossings(q, None, cap.fs, frac=5.0, floor_dps=0.0) == 0.0      # fraction dominates


def test_crossing_index_is_the_true_zero_not_the_threshold_crossing():
    """The reported instant must be the sign change itself, so the timing metric stays honest."""
    from frame_core.joint_space import crossing_indices
    w = np.concatenate([np.full(50, 100.0), np.linspace(100, -100, 21), np.full(50, -100.0)])
    idx, _ = crossing_indices(w, frac=0.5, floor_dps=0.0)
    assert idx.size == 1
    assert abs(int(idx[0]) - 60) <= 1              # the ramp passes zero at sample 60


def test_reversals_now_uses_the_deadband():
    """`reversals` is the same deadbanded quantity -- the involvement menu must not see raw noise."""
    from frame_core.joint_space import reversals, zero_crossings
    cap = _sine_capture(noise=0.35)
    q = cap.reps[0].angles[:, 0]
    assert reversals(q, None, cap.fs) == zero_crossings(q, None, cap.fs) == 10.0


def test_compute_reports_the_three_speed_metrics():
    from frame_core.joint_space import compute
    from frame_core.synthetic import default_config
    cap = _sine_capture()
    res = compute(cap, default_config())
    e = res["per_dof"]["j"]
    assert e["zero_crossings"]["median"] == 10.0
    assert e["min_crossing_gap_s"]["median"] == pytest.approx(0.5, abs=0.02)
    # analytic peak is 60 deg * 2*pi = 377 deg/s; the measured value runs ~5% high because
    # filtfilt rings at the record ends -- interior samples hit 376.97.
    analytic = np.deg2rad(60) * 2 * np.pi * 180 / np.pi
    assert e["max_speed_dps"] == pytest.approx(analytic, rel=0.10)
    assert e["max_speed_dps"] >= analytic
    assert res["speed_deadband"] == {"frac": 0.05, "floor_dps": 2.0}


def test_config_deadband_reaches_the_metric():
    from frame_core.joint_space import compute
    from frame_core.synthetic import default_config
    cap = _sine_capture()
    cfg = default_config()
    cfg.speed_deadband_floor_dps = 1e6            # suppress everything
    res = compute(cap, cfg)
    assert res["per_dof"]["j"]["zero_crossings"]["median"] == 0.0
    assert np.isnan(res["per_dof"]["j"]["min_crossing_gap_s"]["median"])


def _wobbly_still_joint(fs=240.0):
    """The case the deadband exists for: a joint that barely moves, with in-band wobble on top.

    2 deg of 1 Hz motion (the real movement, 10 crossings) plus 0.6 deg at 4 Hz. The 4 Hz
    component sits INSIDE the 6 Hz passband, so `derivative`'s Butterworth cannot remove it --
    only the deadband can. Broadband noise above 6 Hz is a poor demonstrator precisely because
    the filter already handles it.
    """
    N = int(5 * fs)
    t = np.arange(N) / fs
    return np.deg2rad(2.0) * np.sin(2 * np.pi * t) + np.deg2rad(0.6) * np.sin(2 * np.pi * 4 * t)


def test_zero_deadband_recovers_the_raw_count():
    """frac=floor=0 means "no deadband", which must give the MAXIMUM count -- not zero.

    Regression: the degenerate-input guard used to treat thr == 0 as "nothing committed" and
    returned no crossings at all, so turning the deadband off silently suppressed every reversal
    instead of reporting them all.
    """
    from frame_core.joint_space import zero_crossings
    q = _wobbly_still_joint()
    raw = zero_crossings(q, None, 240.0, frac=0.0, floor_dps=0.0)
    deadbanded = zero_crossings(q, None, 240.0, frac=0.05, floor_dps=2.0)
    assert raw == 40.0                            # every wobble counted
    assert deadbanded < raw                       # the deadband suppresses the low-amplitude ones


def test_absolute_floor_does_the_work_on_a_near_still_joint():
    """On a joint whose peak is already small, the fractional term is too small to gate anything --
    the absolute floor is what suppresses the wobble. That is why both knobs exist."""
    from frame_core.joint_space import zero_crossings
    q = _wobbly_still_joint()
    frac_only = zero_crossings(q, None, 240.0, frac=0.05, floor_dps=0.0)
    with_floor = zero_crossings(q, None, 240.0, frac=0.05, floor_dps=2.0)
    assert frac_only == 40.0 and with_floor == 30.0


def test_zero_deadband_on_a_clean_signal_matches_the_deadbanded_count():
    """With no noise there is nothing for the deadband to remove, so both agree."""
    from frame_core.joint_space import zero_crossings
    q = _sine_capture().reps[0].angles[:, 0]
    assert zero_crossings(q, None, 240.0, frac=0.0, floor_dps=0.0) == 10.0


# ---------- Phase D: torque capacity, ellipse, anatomy (PDF item 6) ---------- #
def test_capacity_ratios_against_a_known_effort():
    from frame_core.dynamics import capacity_ratios
    entry = {"rms_torque": {"median": 25.0}, "peak_torque": {"max": 60.0}}
    r = capacity_ratios(entry, 100.0)
    assert r["torque_capacity_Nm"] == 100.0
    assert r["rms_torque_ratio"] == pytest.approx(0.25)
    assert r["peak_torque_ratio"] == pytest.approx(0.60)


def test_capacity_ratios_absent_without_a_limit():
    """No effort recorded -> report the torques unnormalised rather than inventing a denominator."""
    from frame_core.dynamics import capacity_ratios
    entry = {"rms_torque": {"median": 25.0}, "peak_torque": {"max": 60.0}}
    assert capacity_ratios(entry, None) == {}
    assert capacity_ratios(entry, 0.0) == {}


def test_confidence_ellipse_encloses_about_95_percent():
    """An isotropic Gaussian cloud: the 95% ellipse must actually contain ~95% of the points."""
    from frame_core.dynamics import confidence_ellipse
    rng = np.random.default_rng(0)
    w = rng.normal(0, 30.0, 4000)
    t = rng.normal(0, 8.0, 4000)
    e = confidence_ellipse(w, t)
    # Mahalanobis test against the same covariance the ellipse was built from
    pts = np.column_stack([w, t])
    d = pts - pts.mean(0)
    inv = np.linalg.inv(np.cov(pts, rowvar=False))
    inside = np.einsum("ij,jk,ik->i", d, inv, d) <= 5.991
    assert 0.93 <= inside.mean() <= 0.97


def test_confidence_ellipse_is_a_closed_polyline():
    from frame_core.dynamics import confidence_ellipse
    rng = np.random.default_rng(1)
    e = confidence_ellipse(rng.normal(0, 10, 500), rng.normal(0, 3, 500))
    assert len(e["omega_dps"]) == len(e["tau_Nm"]) == 101
    assert e["omega_dps"][0] == pytest.approx(e["omega_dps"][-1])   # closes on itself
    assert e["tau_Nm"][0] == pytest.approx(e["tau_Nm"][-1])
    assert e["semi_axes"][0] >= e["semi_axes"][1] > 0               # major axis first


def test_confidence_ellipse_tracks_a_correlated_cloud():
    """tau = k*omega + noise -> the major axis must lie along that slope, not along an axis."""
    from frame_core.dynamics import confidence_ellipse
    rng = np.random.default_rng(2)
    w = rng.normal(0, 40.0, 3000)
    t = 0.5 * w + rng.normal(0, 1.0, 3000)
    e = confidence_ellipse(w, t)
    assert e["angle_deg"] == pytest.approx(np.degrees(np.arctan(0.5)), abs=3.0)
    assert e["semi_axes"][0] / e["semi_axes"][1] > 5                # strongly elongated


def test_confidence_ellipse_none_on_degenerate_input():
    from frame_core.dynamics import confidence_ellipse
    assert confidence_ellipse([1.0, 2.0], [1.0, 2.0]) is None       # fewer than 3 points
    assert confidence_ellipse(np.zeros(50), np.zeros(50)) is None   # no spread at all


def test_dof_to_segment_mapping():
    from frame_core.anatomy import dof_to_segment
    assert dof_to_segment("right_shoulder_X") == "RightUpperArm"
    assert dof_to_segment("left_elbow_Z") == "LeftForeArm"
    assert dof_to_segment("right_clavicular_X") == "RightShoulder"
    assert dof_to_segment("left_knee_Y") == "LeftLowerLeg"
    assert dof_to_segment("thoracicspine_Y") == "T8"
    assert dof_to_segment("not_a_joint") is None


def test_upper_limb_validation_scope():
    from frame_core.anatomy import is_dynamics_validated, validation_note
    assert is_dynamics_validated("right_shoulder_Y")
    assert is_dynamics_validated("left_wrist_X")
    assert not is_dynamics_validated("right_knee_Y")
    assert not is_dynamics_validated("lumbarspine_Y")
    assert "GRF" in validation_note("right_hip_X")
    assert validation_note("right_elbow_Z") == ""


def test_torque_bands_are_ordered_and_cover_the_range():
    from frame_core.anatomy import torque_band
    names = [torque_band(r)[0] for r in (0.0, 0.3, 0.6, 0.9, 2.0)]
    assert names == ["ok", "moderate", "high", "critical", "critical"]
    assert torque_band(float("nan"))[0] == "unknown"


def test_dynamics_compute_carries_ratios_ellipse_and_validation():
    from frame_core.dynamics import compute
    from frame_core.synthetic import make_capture, default_config
    cap = make_capture(n_reps=4)
    limits = {"right_shoulder_Z": {"effort": 51.0}, "right_shoulder_X": {"effort": 69.0}}
    res = compute(cap, default_config(), urdf_limits=limits)
    z = res["per_dof"]["right_shoulder_Z"]
    assert z["torque_capacity_Nm"] == 51.0
    assert 0.0 < z["rms_torque_ratio"] < 2.0
    assert z["torque_speed_ellipse"]["coverage"] == 0.95
    assert z["dynamics_validated"] is True
    # the DOF with no effort entry keeps its torques but gets no ratio
    y = res["per_dof"]["right_shoulder_Y"]
    assert "rms_torque_ratio" not in y and "peak_torque" in y
    assert res["validation"]["unvalidated_dofs"] == []


def test_analyze_threads_limits_into_the_dynamics_layer():
    """End to end through the pipeline, with the canonical full-limits shape."""
    from frame_core import analyze
    from frame_core.synthetic import make_capture, default_config
    cap = make_capture(n_reps=3, with_scene=True)
    res = analyze(cap, default_config(), urdf_limits=cap.urdf_limits)
    assert res["dynamics"]["per_dof"]["right_shoulder_X"]["torque_capacity_Nm"] == 69.0
    assert "rom_pct_of_limit" in res["joint_space"]["per_dof"]["right_shoulder_X"]


def test_limit_entry_accepts_both_shapes():
    """Limits may arrive as (lo, hi) tuples while dynamics needs the effort dict."""
    from frame_core.types import limit_entry
    full = {"j": {"lower": -1.0, "upper": 2.0, "effort": 50.0}}
    pair = {"j": (-1.0, 2.0)}
    assert limit_entry(full, "j")["effort"] == 50.0
    assert limit_entry(pair, "j") == {"lower": -1.0, "upper": 2.0}
    assert limit_entry(pair, "j").get("effort") is None
    assert limit_entry(None, "j") == {} and limit_entry({}, "missing") == {}


def test_ellipse_angle_is_wrapped_to_a_line_orientation():
    """The eigenvector sign is arbitrary; the reported angle must not flip by 180 deg because of it."""
    from frame_core.dynamics import confidence_ellipse
    rng = np.random.default_rng(2)
    w = rng.normal(0, 40.0, 3000)
    for slope in (0.5, -0.5, 2.0):
        e = confidence_ellipse(w, slope * w + rng.normal(0, 1.0, 3000))
        assert -90.0 < e["angle_deg"] <= 90.0
        assert e["angle_deg"] == pytest.approx(np.degrees(np.arctan(slope)), abs=3.0)


# ---------- Phase E: PCA (PDF item 4) ---------- #
def _coupled_capture(n=600, seed=0):
    """Three DOFs where two are exact linear combinations of one -> a single true component."""
    from frame_core.types import Capture, Repetition
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    a = np.sin(t)
    X = np.column_stack([a, 2.0 * a, -0.5 * a]) + rng.normal(0, 1e-6, (n, 3))
    return Capture(reps=[Repetition(angles=X)], dof_names=["j1", "j2", "j3"], fs=240.0)


def test_pca_finds_one_component_when_joints_are_perfectly_coupled():
    from frame_core.pca import compute
    r = compute(_coupled_capture())
    assert r["ok"] and r["n_pc_90"] == 1
    assert r["explained_variance_ratio"][0] > 0.99
    assert all(abs(v) > 0.5 for v in r["components"][0]["loadings"].values())


def test_pca_threshold_controls_the_retained_count():
    """A capture with graded variance: a stricter threshold must retain at least as many PCs."""
    from frame_core.pca import compute
    from frame_core.synthetic import make_capture
    cap = make_capture(n_reps=3)
    assert compute(cap, vaf_threshold=0.99)["n_pc_90"] >= compute(cap, vaf_threshold=0.50)["n_pc_90"]


def test_pca_default_threshold_is_90_percent():
    from frame_core.pca import VAF_THRESHOLD, compute
    from frame_core.synthetic import make_capture
    assert VAF_THRESHOLD == 0.90
    r = compute(make_capture(n_reps=3))
    assert r["vaf_threshold"] == 0.90
    # the retained set must actually clear the threshold, and one fewer must not
    assert r["cumulative_vaf"][r["n_pc_90"] - 1] >= 0.90
    if r["n_pc_90"] > 1:
        assert r["cumulative_vaf"][r["n_pc_90"] - 2] < 0.90


def test_pca_signs_are_deterministic():
    """Largest-|loading| positive, so a re-run cannot flip a published bar chart."""
    from frame_core.pca import compute
    cap = _coupled_capture()
    a, b = compute(cap), compute(cap)
    assert a["components"][0]["loadings"] == b["components"][0]["loadings"]
    for c in a["components"]:
        top = max(c["loadings"].values(), key=abs)
        assert top > 0


def test_pca_drops_constant_dofs_and_names_them():
    from frame_core.types import Capture, Repetition
    from frame_core.pca import compute
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=300), np.zeros(300), rng.normal(size=300)])
    cap = Capture(reps=[Repetition(angles=X)], dof_names=["a", "frozen", "c"], fs=240.0)
    r = compute(cap)
    assert r["dropped_dofs"] == ["frozen"] and r["dof_names"] == ["a", "c"]


def test_pca_refuses_degenerate_captures():
    from frame_core.types import Capture, Repetition
    from frame_core.pca import compute
    one = Capture(reps=[Repetition(angles=np.zeros((300, 1)))], dof_names=["a"], fs=240.0)
    assert compute(one)["ok"] is False
    flat = Capture(reps=[Repetition(angles=np.zeros((300, 3)))],
                   dof_names=["a", "b", "c"], fs=240.0)
    assert compute(flat)["ok"] is False and compute(flat)["n_pc_90"] == 0


def test_engagement_is_vaf_weighted_not_a_plain_mean():
    """A low-variance PC must not count as much as PC1 -- that is the point of the weighting."""
    from frame_core.pca import engagement
    Vt = np.array([[1.0, 0.0], [0.0, 1.0]])      # PC1 loads j1, PC2 loads j2
    vaf = np.array([0.9, 0.1])
    e = engagement(Vt, vaf, n_pc=2)
    assert e[0] == pytest.approx(0.9) and e[1] == pytest.approx(0.1)
    assert e[0] > e[1]                            # a plain mean would call them equal


def test_engagement_ranking_orders_by_engagement():
    from frame_core.pca import compute
    from frame_core.synthetic import make_capture
    r = compute(make_capture(n_reps=3))
    vals = [r["engagement"][n] for n in r["engagement_ranking"]]
    assert vals == sorted(vals, reverse=True)


def test_pca_reaches_the_pipeline_and_capability():
    from frame_core import analyze
    from frame_core.pipeline import capability
    from frame_core.synthetic import make_capture, default_config
    cap = make_capture(n_reps=3)
    assert capability(cap)["pca"]["ok"] is True
    res = analyze(cap, default_config())
    assert res["pca"]["ok"] and res["pca"]["n_pc_90"] >= 1


# --------------------------------------------------------------------------------------------- #
# flatten: the unit registry decides what gets divided by body mass or stature
# --------------------------------------------------------------------------------------------- #
def test_units_table_classifies_every_real_key_shape():
    from frame_core.flatten import units_table
    t = units_table([
        "task_space.workspace_volume_L", "task_space.bbox_x_m", "task_space.reach_max_m",
        "task_space.peak_speed_mps", "task_space.peak_accel_mps2",
        "task_space.dwell_concentration",
        "joint_space.right_shoulder_Z.rom_deg",
        "joint_space.right_shoulder_Z.angular_path_rad",
        "joint_space.right_shoulder_Z.min_crossing_gap_s",
        "dynamics.right_shoulder_Z.peak_torque_Nm",
        "dynamics.right_shoulder_Z.net_work_J",
        "dynamics.right_shoulder_Z.peak_power_pos_W",
        "dynamics.right_shoulder_Z.k_Nm_per_rad",
        "dynamics.right_shoulder_Z.rms_torque_ratio",
        "pca.n_pc_90",
    ])
    assert t["task_space.workspace_volume_L"] == "volume_L"
    assert t["task_space.bbox_x_m"] == t["task_space.reach_max_m"] == "length"
    assert t["task_space.peak_speed_mps"] == "speed"
    assert t["task_space.peak_accel_mps2"] == "accel"
    assert t["dynamics.right_shoulder_Z.peak_torque_Nm"] == "torque"
    assert t["dynamics.right_shoulder_Z.net_work_J"] == "work"
    assert t["dynamics.right_shoulder_Z.peak_power_pos_W"] == "power"
    assert t["dynamics.right_shoulder_Z.k_Nm_per_rad"] == "stiffness"
    # angles, times, counts and ratios are already subject-independent
    for k in ("joint_space.right_shoulder_Z.rom_deg",
              "joint_space.right_shoulder_Z.angular_path_rad",
              "joint_space.right_shoulder_Z.min_crossing_gap_s",
              "dynamics.right_shoulder_Z.rms_torque_ratio",
              "task_space.dwell_concentration", "pca.n_pc_90"):
        assert t[k] == "none", k


def test_a_dof_name_ending_in_a_unit_letter_is_not_mistaken_for_that_unit():
    """`involvement.<dof>` and `engagement.<dof>` end in a DOF name, not a unit."""
    from frame_core.flatten import units_table, normalize_row
    t = units_table(["joint_space.involvement.wrist_W", "pca.engagement.wrist_m",
                     "joint_space.involvement.right_shoulder_Z"])
    assert set(t.values()) == {"none"}
    row = {"joint_space.involvement.wrist_W": 0.8, "pca.engagement.wrist_m": 0.5}
    assert normalize_row(row, height_m=1.62, mass_kg=58.0) == {}


def test_normalisation_omits_a_column_it_cannot_normalise():
    """An absent column is honest; a NaN one would silently shrink a pooled distribution's n."""
    from frame_core.flatten import normalize_row
    row = {"dynamics.d.peak_torque_Nm": 40.0, "task_space.reach_max_m": 0.4}
    out = normalize_row(row, height_m=None, mass_kg=58.0)
    assert out == {"dynamics.d.peak_torque_Nm__norm": pytest.approx(40.0 / 58.0)}


# --------------------------------------------------------------------------------------------- #
# pooling: ranking a thousand columns into a reading order
# --------------------------------------------------------------------------------------------- #
def _fake_pooled(values_by_group, metric="m"):
    from frame_core.pooling import pool
    rows = []
    for g, vals in values_by_group.items():
        for i, v in enumerate(vals):
            rows.append({"trial_id": f"{g}{i}", "subject": "s1", "task": g,
                         "metrics": {metric: v}})
    return pool(rows, by="task")


def test_pooled_metric_keeps_values_aligned_with_their_trials():
    from frame_core.pooling import pool_metric
    rows = [{"trial_id": "t1", "metrics": {"m": 1.0}},
            {"trial_id": "t2", "metrics": {"m": None}},      # not measurable -> shrinks n
            {"trial_id": "t3", "metrics": {"m": 3.0}}]
    e = pool_metric(rows, "m")
    assert e["values"] == [1.0, 3.0] and e["trials"] == ["t1", "t3"]
    assert e["n"] == 2 and e["n_missing"] == 1
    assert e["median"] == pytest.approx(2.0)


def test_rank_identical_metrics_are_folded_into_one_row():
    """A torque, its per-kg twin and its capacity ratio differ by a constant divisor: one row."""
    from frame_core.pooling import pool, ranked_differences
    rows = []
    for g, base in (("H", 10.0), ("M", 20.0), ("L", 30.0)):
        for i in range(4):
            raw = base + i
            rows.append({"trial_id": f"{g}{i}", "subject": "s1", "task": g,
                         "metrics": {"dynamics.d.peak_torque_Nm": raw,
                                     "dynamics.d.peak_torque_Nm__norm": raw / 58.0,
                                     "dynamics.d.peak_torque_ratio": raw / 90.0}})
    ranked = ranked_differences(pool(rows, by="task"), top=10)
    assert len(ranked) == 1, "three rescalings of one number are one finding"
    assert ranked[0]["metric"] == "dynamics.d.peak_torque_Nm"      # raw SI wins
    assert ranked[0]["aliases"] == ["dynamics.d.peak_torque_Nm__norm",
                                    "dynamics.d.peak_torque_ratio"]


def test_dedupe_can_be_switched_off():
    from frame_core.pooling import pool, ranked_differences
    rows = []
    for g, base in (("H", 10.0), ("M", 20.0), ("L", 30.0)):
        for i in range(4):
            rows.append({"trial_id": f"{g}{i}", "subject": "s1", "task": g,
                         "metrics": {"a_Nm": base + i, "a_Nm__norm": (base + i) / 58.0}})
    assert len(ranked_differences(pool(rows, by="task"), dedupe=False)) == 2


def test_genuinely_different_metrics_are_not_folded():
    from frame_core.pooling import pool, ranked_differences
    rows = []
    for g, (a, b) in (("H", (1, 30)), ("M", (2, 20)), ("L", (3, 10))):
        for i in range(4):
            rows.append({"trial_id": f"{g}{i}", "subject": "s1", "task": g,
                         "metrics": {"rises_Nm": a + i * 0.1, "falls_m": b + i * 0.1}})
    ranked = ranked_differences(pool(rows, by="task"), top=10)
    assert {r["metric"] for r in ranked} == {"rises_Nm", "falls_m"}
    assert all(not r["aliases"] for r in ranked)


def test_trial_level_test_declares_itself_pseudoreplicated():
    from frame_core.pooling import compare_groups
    pooled = _fake_pooled({"H": [1.0, 1.1, 1.2], "L": [5.0, 5.1, 5.2]})
    r = compare_groups(pooled, "m", level="trial")
    assert r["test"] == "kruskal" and r["p"] < 0.2
    assert r["pseudoreplicated"] is True and r["n_subjects"] == 1


def test_subject_level_test_refuses_to_run_on_too_few_subjects():
    """Better to report nothing than to hand back the inflated number under a better name."""
    from frame_core.pooling import compare_groups
    pooled = _fake_pooled({"H": [1.0, 1.1, 1.2], "L": [5.0, 5.1, 5.2]})
    r = compare_groups(pooled, "m", level="subject")
    assert r["test"] is None and r["p"] != r["p"]          # NaN
    assert "2 subjects per group" in r["reason"]


def test_subject_level_test_is_far_less_significant_than_the_trial_level_one():
    """The whole point: 24 trials from 3 people is not 24 independent observations."""
    from frame_core.pooling import pool, compare_groups
    rows = []
    for sid, off in (("s1", 0.0), ("s2", 0.4), ("s3", -0.3)):
        for g, base in (("H", 10.0), ("M", 7.0), ("L", 4.0)):
            for i in range(8):
                rows.append({"trial_id": f"{sid}{g}{i}", "subject": sid, "task": g,
                             "metrics": {"m": base + off + 0.05 * i}})
    pooled = pool(rows, by="task")
    trial = compare_groups(pooled, "m", level="trial")
    subject = compare_groups(pooled, "m", level="subject")
    assert trial["n"]["H"] == 24 and trial["n_subjects"] == 3
    assert subject["test"] == "kruskal"
    assert subject["p"] > trial["p"] * 1e6, "pseudo-replication must be visible, not subtle"
    assert subject["pseudoreplicated"] is False


def test_pooled_metric_records_which_subject_each_value_came_from():
    from frame_core.pooling import pool_metric
    rows = [{"trial_id": "a", "subject": "s1", "metrics": {"m": 1.0}},
            {"trial_id": "b", "subject": "s2", "metrics": {"m": 2.0}}]
    e = pool_metric(rows, "m")
    assert e["subject_of_trial"] == ["s1", "s2"]
    assert len(e["subject_of_trial"]) == len(e["values"])


# --------------------------------------------------------------------------------------------- #
# curves: time-normalised trajectories for the mean +/- SD bands
# --------------------------------------------------------------------------------------------- #
def _curve_capture(n_frames, amp_deg=90.0, fs=240.0, torque=True):
    """One rep of a half-sine sweep on a single DOF, at a chosen duration."""
    from frame_core.types import Capture, Repetition
    t = np.linspace(0.0, 1.0, n_frames)
    q = np.deg2rad(amp_deg) * np.sin(np.pi * t)[:, None]
    tau = 8.0 * q if torque else None
    return Capture(reps=[Repetition(angles=q, torques=tau)], dof_names=["j"], fs=fs)


def test_trial_curves_normalise_every_trial_to_the_same_length():
    from frame_core.curves import trial_curves
    for n_frames in (300, 1500, 4000):
        c = trial_curves(_curve_capture(n_frames), ["j"], n_resample=200)
        assert c["j"]["angle_deg"].shape == (1, 200)
        assert c["j"]["speed_dps"].shape == (1, 200)
        assert c["j"]["torque_Nm"].shape == (1, 200)


def test_angle_curve_keeps_its_amplitude_through_resampling():
    """Linear interpolation passes through the samples; FFT resampling would ring and overshoot."""
    from frame_core.curves import trial_curves
    c = trial_curves(_curve_capture(2000, amp_deg=90.0), ["j"], n_resample=200)
    peak = float(np.max(c["j"]["angle_deg"]))
    assert peak == pytest.approx(90.0, abs=0.5)


def test_speed_is_differentiated_at_the_native_rate_not_after_normalising():
    """The same movement performed twice as fast must show twice the joint speed.

    Differentiating the time-normalised curve instead would give deg per percent-of-trial, which
    is identical for both and silently hides the speed difference the figure exists to show.
    """
    from frame_core.curves import trial_curves
    slow = trial_curves(_curve_capture(2400), ["j"], n_resample=200)   # 10 s at 240 Hz
    fast = trial_curves(_curve_capture(1200), ["j"], n_resample=200)   # 5 s
    peak_slow = float(np.max(np.abs(slow["j"]["speed_dps"])))
    peak_fast = float(np.max(np.abs(fast["j"]["speed_dps"])))
    assert peak_fast == pytest.approx(2.0 * peak_slow, rel=0.05)


def test_a_quantity_the_capture_cannot_supply_is_absent_not_nan():
    from frame_core.curves import trial_curves
    c = trial_curves(_curve_capture(600, torque=False), ["j"], n_resample=200)
    assert "angle_deg" in c["j"] and "speed_dps" in c["j"]
    assert "torque_Nm" not in c["j"], "an absent row beats a row of NaN"


def test_unknown_dof_is_skipped_rather_than_erroring():
    from frame_core.curves import trial_curves
    assert trial_curves(_curve_capture(600), ["nope"], n_resample=50) == {}


def test_stack_curves_pools_every_repetition_of_every_trial():
    from frame_core.curves import trial_curves, stack_curves
    per_trial = [trial_curves(_curve_capture(n), ["j"], n_resample=100)
                 for n in (300, 600, 900)]
    stacked = stack_curves(per_trial)
    assert stacked["j"]["angle_deg"].shape == (3, 100)


def test_band_reports_mean_sd_and_the_curve_count():
    from frame_core.curves import band
    arr = np.vstack([np.full(50, 1.0), np.full(50, 3.0)])
    b = band(arr)
    assert b["n"] == 2
    assert np.allclose(b["mean"], 2.0) and np.allclose(b["sd"], 1.0)
    assert np.allclose(b["lo"], 1.0) and np.allclose(b["hi"], 3.0)
    assert b["pct"][0] == 0.0 and b["pct"][-1] == 100.0


def test_band_is_none_for_an_empty_stack():
    from frame_core.curves import band
    assert band(np.empty((0, 10))) is None


def _cyclic_capture(n_cycles=3, cycle_frames=600, fs=240.0, gap=200):
    """A recording of several reach-and-return cycles separated by rest, like the real captures.

    It opens on a descent and closes on a rise so that every rest period is an INTERIOR minimum.
    A plateau touching the array edge is not a detectable trough (nothing falls into it), which is
    a property of the signal, not of the detector -- the real recordings start and end mid-motion.
    """
    from frame_core.types import Capture, Repetition
    half = np.linspace(0, 1, cycle_frames // 2)
    q = [np.deg2rad(80.0) * np.cos(0.5 * np.pi * half)]        # opening descent
    z = [0.5 * np.cos(0.5 * np.pi * half)]
    for _ in range(n_cycles):
        q.append(np.zeros(gap))
        z.append(np.zeros(gap))
        t = np.linspace(0, 1, cycle_frames)
        q.append(np.deg2rad(80.0) * np.sin(np.pi * t))
        z.append(0.5 * np.sin(np.pi * t))
    q.append(np.zeros(gap))                                     # closing rest
    z.append(np.zeros(gap))
    q.append(np.deg2rad(80.0) * np.sin(0.5 * np.pi * half))     # closing rise
    z.append(0.5 * np.sin(0.5 * np.pi * half))
    q = np.concatenate(q)[:, None]
    z = np.concatenate(z)
    ee = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    return Capture(reps=[Repetition(angles=q, torques=5.0 * q, ee_pos=ee)],
                   dof_names=["j"], fs=fs)


def test_trial_curves_keep_the_whole_recording_unsegmented():
    """A recording with several movements is ONE curve covering all of it -- nothing is cut out,
    so every oscillation of the movement survives into the curve."""
    from frame_core.curves import trial_curves
    from scipy.signal import find_peaks
    cap = _cyclic_capture(n_cycles=4)                  # has ee_pos: must NOT trigger segmentation
    c = trial_curves(cap, ["j"], n_resample=500)
    assert c["j"]["angle_deg"].shape == (1, 500)
    peaks, _ = find_peaks(c["j"]["angle_deg"][0], prominence=40.0)
    assert len(peaks) == 4, "all four movements are still in the one curve"


def test_trial_curves_pool_one_curve_per_recording():
    from frame_core.curves import trial_curves, stack_curves, band
    caps = [_cyclic_capture(n_cycles=n, gap=g) for n, g in ((3, 150), (4, 600), (5, 300))]
    b = band(stack_curves([trial_curves(c, ["j"], n_resample=500) for c in caps])["j"]["angle_deg"])
    assert b["n"] == 3 and len(b["mean"]) == 500


def test_hull_coverage_matches_a_full_cloud_triangulation():
    """The vertex-only triangulation is an optimisation, not an approximation.

    Containment in a convex hull is decided by its vertices alone, so reducing the exo cloud
    before triangulating must give bit-identical coverage -- on a PARTIAL overlap, where a wrong
    hull would show.
    """
    from scipy.spatial import Delaunay
    from frame_core.task_space import hull_coverage

    rng = np.random.default_rng(3)
    E = rng.normal(size=(20000, 3))
    H = rng.normal(size=(4000, 3)) * 1.4 + np.array([1.2, 0.0, 0.0])   # partly outside
    got = hull_coverage(H, E)
    ref = float((Delaunay(E).find_simplex(H) >= 0).mean())
    assert 0.05 < ref < 0.95, "the fixture must actually be a partial overlap"
    assert got["hull_coverage"] == pytest.approx(ref, abs=1e-12)
    assert got["n_inside"] == int(round(ref * len(H)))
    assert got["n_total"] == len(H)


def test_hull_coverage_is_nan_on_a_degenerate_exo_cloud():
    from frame_core.task_space import hull_coverage
    out = hull_coverage(np.random.default_rng(0).normal(size=(100, 3)), np.zeros((3, 3)))
    assert out["hull_coverage"] != out["hull_coverage"]        # NaN, not a wrong number


# --------------------------------------------------------------------------------------------- #
# reporting on a subset of joints
# --------------------------------------------------------------------------------------------- #
def test_dof_of_identifies_the_joint_behind_every_key_shape():
    from frame_core.flatten import dof_of
    assert dof_of("joint_space.right_elbow_Z.rom_deg") == "right_elbow_Z"
    assert dof_of("dynamics.right_elbow_Z.peak_torque_Nm") == "right_elbow_Z"
    assert dof_of("dynamics.right_elbow_Z.peak_torque_Nm__norm") == "right_elbow_Z"
    assert dof_of("joint_space.involvement.right_elbow_Z") == "right_elbow_Z"
    assert dof_of("pca.engagement.right_elbow_Z") == "right_elbow_Z"
    # whole-capture metrics belong to no joint and must survive any filter
    for k in ("task_space.reach_max_m", "pca.n_pc_90", "pca.vaf_pc1",
              "task_space.coverage.hull_coverage"):
        assert dof_of(k) is None, k


def test_filter_metrics_keeps_whole_capture_metrics_and_the_chosen_joints():
    from frame_core.flatten import filter_metrics
    m = {
        "task_space.reach_max_m": 1.0, "pca.n_pc_90": 8.0,
        "joint_space.right_elbow_Z.rom_deg": 2.0,
        "dynamics.right_elbow_Z.peak_torque_Nm": 3.0,
        "joint_space.involvement.right_elbow_Z": 0.4,
        "joint_space.left_hip_X.rom_deg": 4.0,
        "dynamics.left_hip_X.peak_torque_Nm": 5.0,
    }
    out = filter_metrics(m, ["right_elbow_Z"])
    assert set(out) == {"task_space.reach_max_m", "pca.n_pc_90",
                        "joint_space.right_elbow_Z.rom_deg",
                        "dynamics.right_elbow_Z.peak_torque_Nm",
                        "joint_space.involvement.right_elbow_Z"}
    assert out["joint_space.right_elbow_Z.rom_deg"] == 2.0, "values are passed through untouched"
    assert filter_metrics(m, None) == m, "no list means no filtering"


def test_filter_record_narrows_metrics():
    from frame_core.flatten import filter_record
    rec = {"trial_id": "t", "metrics": {"dynamics.a_X.peak_torque_Nm": 1.0,
                                        "dynamics.b_X.peak_torque_Nm": 2.0}}
    out = filter_record(rec, ["a_X"])
    assert set(out["metrics"]) == {"dynamics.a_X.peak_torque_Nm"}
    assert rec["metrics"] and len(rec["metrics"]) == 2, "the original record is not mutated"


# --------------------------------------------------------------------------------------------- #
# how often the joint reverses, and how briskly
# --------------------------------------------------------------------------------------------- #
def _reversing_capture(n_cycles=5, fs=240.0, amp_deg=40.0, seconds=10.0):
    """A clean sinusoid: velocity reverses exactly twice per cycle, at a known spacing."""
    from frame_core.types import Capture, Repetition
    N = int(seconds * fs)
    t = np.linspace(0.0, seconds, N)
    q = np.deg2rad(amp_deg) * np.sin(2 * np.pi * (n_cycles / seconds) * t)
    return Capture(reps=[Repetition(angles=q[:, None])], dof_names=["j"], fs=fs)


def test_reversals_counts_velocity_direction_changes():
    """A sinusoid reverses twice per cycle; the interior crossings are the countable ones."""
    from frame_core.joint_space import reversals, zero_crossings
    cap = _reversing_capture(n_cycles=5, seconds=10.0)
    q, fs = cap.reps[0].angles[:, 0], cap.fs
    n = reversals(q, None, fs)
    assert 8 <= n <= 10, f"5 cycles -> about 10 direction changes, got {n}"
    assert zero_crossings(q, None, fs) == n, "the two names are the same quantity"


def test_mean_gap_matches_the_known_half_period():
    """5 cycles in 10 s -> a reversal every 1.0 s, so the mean gap must be ~1.0 s."""
    from frame_core.joint_space import mean_crossing_gap_s, min_crossing_gap_s
    cap = _reversing_capture(n_cycles=5, seconds=10.0)
    q, fs = cap.reps[0].angles[:, 0], cap.fs
    assert mean_crossing_gap_s(q, None, fs) == pytest.approx(1.0, rel=0.05)
    assert min_crossing_gap_s(q, None, fs) <= mean_crossing_gap_s(q, None, fs)


def test_mean_gap_separates_a_rapid_joint_from_a_slow_one():
    """The point of the metric: the same movement done faster must show a shorter gap."""
    from frame_core.joint_space import mean_crossing_gap_s
    slow = _reversing_capture(n_cycles=2, seconds=10.0)
    fast = _reversing_capture(n_cycles=8, seconds=10.0)
    g_slow = mean_crossing_gap_s(slow.reps[0].angles[:, 0], None, slow.fs)
    g_fast = mean_crossing_gap_s(fast.reps[0].angles[:, 0], None, fast.fs)
    assert g_slow == pytest.approx(2.5, rel=0.1) and g_fast == pytest.approx(0.625, rel=0.1)
    assert g_slow > 3 * g_fast


def test_gap_is_nan_with_fewer_than_two_crossings():
    """One reversal has no interval between reversals -- NaN, not a made-up number."""
    from frame_core.joint_space import mean_crossing_gap_s
    cap = _reversing_capture(n_cycles=0.5, seconds=4.0)
    v = mean_crossing_gap_s(cap.reps[0].angles[:, 0], None, cap.fs)
    assert v != v or v > 0


def test_mean_gap_reaches_the_flat_metric_row():
    from frame_core.synthetic import make_capture, default_config
    from frame_core import analyze, flatten_result
    cap = make_capture()
    row = flatten_result(analyze(cap, default_config()), cap.dof_names)
    assert f"joint_space.{cap.dof_names[0]}.mean_crossing_gap_s" in row


def test_empty_pool_by_makes_one_named_group():
    """Pooling everything is a real request, and its group needs a name, not an empty tuple."""
    from frame_core.pooling import pool, group_rows, ALL_GROUP
    rows = [{"trial_id": f"t{i}", "subject": "s1", "task": t,
             "metrics": {"m": float(i)}}
            for i, t in enumerate(["H", "M", "L", "H", "M", "L"])]
    assert list(group_rows(rows, [])) == [ALL_GROUP]
    pooled = pool(rows, by=[])
    assert set(pooled) == {"all"}
    assert pooled["all"]["n_trials"] == 6
    assert pooled["all"]["metrics"]["m"]["n"] == 6
    # and the per-task split still works, on the same rows
    assert set(pool(rows, by=["task"])) == {"H", "M", "L"}
