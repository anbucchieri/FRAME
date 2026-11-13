"""
frame-kinematics tests -- run entirely WITHOUT Pinocchio using injected fakes. Passing here
proves the heavy-path wiring (segment -> IK -> ID -> Jacobian -> assemble -> Capture ->
frame_core.analyze) is correct; only the Pinocchio internals remain to validate on real data.
"""
import numpy as np
import pandas as pd
import pytest

from frame_kinematics.interfaces import KinematicsJob
from frame_kinematics.jacobian import jacobian_contribution, _dof_vindex
from frame_kinematics.assemble import (
    ee_in_trunk, build_capture, segments_in_trunk, build_skeleton,
    available_segments, XSENS_BONES,
)
from frame_kinematics.serialize import capture_to_npz_bytes, capture_from_npz_bytes
from frame_kinematics.worker import run_job
from frame_core import analyze


DOFS = ["right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y"]


# ---------- fakes ---------- #
class FakeModel:
    """Minimal Pinocchio-model stand-in for DOF->velocity-index mapping."""
    def __init__(self, dof_names):
        self._id = {n: i + 1 for i, n in enumerate(dof_names)}   # 1-based joint ids
        class _J:
            def __init__(self, iv): self.idx_v = iv
        # velocity indices 6.. (after a free-flyer's 6 base DOFs)
        self.joints = [None] + [_J(6 + i) for i in range(len(dof_names))]
        self.nv = 6 + len(dof_names)

    def getJointId(self, name):
        return self._id[name]


class FakeRobot:
    def __init__(self, model): self.model = model; self.data = object()


class FakeIK:
    """Returns canned angle trajectories + states so the whole chain runs offline.

    Designed (like frame-core's synthetic) so Z is spring-like, X motor-like, Y idle.
    """
    def __init__(self, urdf_path, mv):
        self.mv = mv
        self.N = len(mv)
        self.xsens_frequency = 240.0
        self.robot = FakeRobot(FakeModel(DOFS))
        self.saved_configurations = np.zeros((self.N, 6 + len(DOFS) + 1))   # nq (free-flyer quat)
        self.saved_velocity = np.zeros((self.N, 6 + len(DOFS)))
        self.saved_accelerations = np.zeros((self.N, 6 + len(DOFS)))
        self._log_q = None

    def uploadMocapFile(self): pass

    def runFullIK(self):
        t = np.linspace(0, 1, self.N)
        qz = np.deg2rad(90) * np.sin(np.pi * t)
        qx = np.deg2rad(35) * (1 - np.cos(2 * np.pi * t)) / 2
        qy = np.deg2rad(10) * np.sin(4 * np.pi * t)
        self._log_q = {"right_shoulder_Z": qz, "right_shoulder_X": qx, "right_shoulder_Y": qy}
        for k, q in enumerate((qz, qx, qy)):
            self.saved_velocity[:, 6 + k] = np.gradient(q, 1.0 / self.xsens_frequency)
        return self._log_q

    def temp_save_log_q(self):
        return self._log_q

    def butterworth_filter(self, x):
        return np.asarray(x, float)


def fake_run_id(ik, fs=None):
    lq = ik._log_q
    fs = ik.xsens_frequency
    tau_z = 8.0 * lq["right_shoulder_Z"]                                    # spring-like
    om_x = np.gradient(lq["right_shoulder_X"], 1.0 / fs)
    tau_x = 20.0 * lq["right_shoulder_X"] + 6.0 * om_x                      # hysteretic / high work
    tau_y = 0.6 * lq["right_shoulder_Y"]                                    # tiny
    return {"right_shoulder_Z": tau_z, "right_shoulder_X": tau_x, "right_shoulder_Y": tau_y}, {}


def fake_jac(model, data, q, ee_frame):
    """Fixed translational Jacobian (3 x nv): Z drives most Cartesian motion."""
    nv = model.nv
    J = np.zeros((3, nv))
    J[2, 6] = 0.40   # Z column -> vertical authority
    J[0, 7] = 0.20   # X column
    J[1, 8] = 0.05   # Y column
    return J


def _fake_ee_df(N, seed=0):
    """Dataframe with the segment/quaternion columns ee_in_trunk needs (identity trunk frame)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    d = {}
    for c, base in zip("XYZ", (0.30, 0.0, 0.10)):
        d[f"RightForeArm_{c}"] = base + 0.4 * np.sin(np.pi * t) * (c == "Z") + rng.normal(0, 1e-3, N)
    for c in "XYZ":
        d[f"T8_{c}"] = np.zeros(N)
    d["T8_x_q"] = d["T8_y_q"] = d["T8_z_q"] = np.zeros(N)
    d["T8_w_q"] = np.ones(N)                                                # identity quaternion
    return pd.DataFrame(d)


def make_df(n_reps=6, base_len=200):
    parts = []
    for m in range(1, n_reps + 1):
        N = base_len + m * 3
        sub = _fake_ee_df(N, seed=m)
        sub["MovementNumber"] = m
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def make_job(tmp_urdf=None):
    return KinematicsJob(dof_names=DOFS,
                         ee_segment="RightForeArm", trunk_segment="T8", fs=240.0, min_len=30,
                         height_m=1.78, mass_kg=74.0,
                         urdf_path=tmp_urdf, subject_id="S_test",
                         models_dir="/tmp/frame_models", xacro_base="/tmp/dhm_new.xacro",
                         mvnx_path="/tmp/S_test.mvnx")


def fake_urdf_builder(mvnx_path, height, mass, urdf_folder=None, xacro_base_file=None):
    """Stand-in for create_urdf_from_Xsens_new: writes a stub URDF at {folder}/{trial}.urdf."""
    import os
    trial = os.path.basename(str(mvnx_path)).split(".")[0]
    os.makedirs(urdf_folder, exist_ok=True)
    with open(os.path.join(urdf_folder, trial + ".urdf"), "w") as f:
        f.write("<robot name='stub'/>")


# ---------- tests ---------- #
def test_no_pinocchio_import():
    import sys
    for name in list(sys.modules):
        assert "pinocchio" not in name, "frame-kinematics tests must not import pinocchio"


def test_recording_is_never_segmented(tmp_path):
    """Even with a MovementNumber column, the whole recording is ONE continuous repetition."""
    df = make_df(n_reps=5)
    cap = run_job(df, make_job(), make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder)
    assert len(cap.reps) == 1 and cap.reps[0].n == len(df)


def test_dof_vindex_mapping():
    m = FakeModel(DOFS)
    assert [_dof_vindex(m, n) for n in DOFS] == [6, 7, 8]


def test_jacobian_accumulation_matches_hand_calc():
    m = FakeModel(DOFS)
    N = 100
    Q = np.zeros((N, m.nv + 1))
    QD = np.zeros((N, m.nv))
    QD[:, 6] = 1.0                              # constant unit velocity on Z only
    fs = 240.0
    contrib = jacobian_contribution(m, None, Q, QD, "ee", DOFS, fs, jac_fn=fake_jac)
    # Z: sum_i ||J[:,6]||*|1|/fs = N * 0.40 / fs ; X,Y: zero velocity -> 0
    assert contrib[0] == pytest.approx(N * 0.40 / fs)
    assert contrib[1] == 0.0 and contrib[2] == 0.0


def test_ee_in_trunk_identity_frame():
    df = _fake_ee_df(50)
    ee = ee_in_trunk(df, "RightForeArm", "T8")
    # identity trunk frame at origin -> ee equals the raw RightForeArm columns
    assert np.allclose(ee[:, 2], df["RightForeArm_Z"].to_numpy())


def test_run_job_end_to_end_feeds_frame_core():
    """The money test: full heavy path with fakes -> Capture -> frame_core metrics."""
    df = make_df(n_reps=1)
    job = make_job()
    cap = run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac, urdf_builder=fake_urdf_builder)

    assert cap.n_dof == 3 and len(cap.reps) == 1                # one continuous recording
    assert cap.has_torque and cap.has_ee and cap.has_jac        # all four layers enabled

    from frame_core.synthetic import default_config
    cfg = default_config()
    res = analyze(cap, cfg)
    assert set(res) >= {"task_space", "joint_space", "dynamics", "passivity"}
    pk = res["dynamics"]["per_dof"]
    assert pk["right_shoulder_X"]["peak_torque"]["median"] > pk["right_shoulder_Y"]["peak_torque"]["median"]
    # the Jacobian involvement term is now available (heavy path supplied it)
    assert "jacobian" in res["joint_space"]["involvement"]["terms_available"]


def test_angles_only_job_skips_torque_and_jac():
    df = make_df(n_reps=4)
    job = make_job()
    cap = run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder, want_torque=False, want_jac=False)
    assert not cap.has_torque and not cap.has_jac
    assert cap.has_ee                                            # ee cloud still built
    res = analyze(cap, __import__("frame_core.synthetic", fromlist=["default_config"]).default_config())
    assert "dynamics" not in res and "passivity" not in res
    assert "task_space" in res and "joint_space" in res


def test_serialize_roundtrip():
    df = make_df(n_reps=3)
    cap = run_job(df, make_job(), make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac, urdf_builder=fake_urdf_builder)
    blob = capture_to_npz_bytes(cap)
    cap2 = capture_from_npz_bytes(blob)
    assert cap2.n_dof == cap.n_dof and len(cap2.reps) == len(cap.reps)
    assert cap2.has_torque and cap2.has_jac and cap2.has_ee
    assert np.allclose(cap2.reps[0].angles, cap.reps[0].angles)
    assert np.allclose(cap2.reps[0].jac_contribution, cap.reps[0].jac_contribution)
    # a reconstructed capture analyses identically
    from frame_core.synthetic import default_config
    r1 = analyze(cap, default_config())["dynamics"]["per_dof"]["right_shoulder_X"]["rms_torque"]
    r2 = analyze(cap2, default_config())["dynamics"]["per_dof"]["right_shoulder_X"]["rms_torque"]
    assert r1["median"] == pytest.approx(r2["median"])


# ---------- DHM URDF build + cache ---------- #
def test_dhm_builds_on_cache_miss(tmp_path):
    from frame_kinematics.dhm import build_or_load_urdf, urdf_path_for
    models = tmp_path / "models"
    mvnx = tmp_path / "S9.mvnx"
    mvnx.write_text("<mvnx/>")                       # content irrelevant to the fake builder
    out = build_or_load_urdf(models, tmp_path / "dhm.xacro", 1.80, 80.0,
                             mvnx_path=str(mvnx), builder=fake_urdf_builder)
    assert out == urdf_path_for(models, mvnx)
    assert out.exists()


def test_dhm_cache_hit_skips_build(tmp_path):
    from frame_kinematics.dhm import build_or_load_urdf, urdf_path_for
    models = tmp_path / "models"; models.mkdir()
    mvnx = tmp_path / "S9.mvnx"
    cached = urdf_path_for(models, mvnx)
    cached.write_text("<robot name='cached'/>")      # pretend a prior run built it
    called = {"n": 0}
    def spy(*a, **k): called["n"] += 1
    out = build_or_load_urdf(models, tmp_path / "dhm.xacro", 1.8, 80.0,
                             mvnx_path=str(mvnx), builder=spy)
    assert out == cached and called["n"] == 0        # builder NOT called on a hit


def test_dhm_errors_without_mvnx_or_prebuilt(tmp_path):
    from frame_kinematics.dhm import build_or_load_urdf
    with pytest.raises(FileNotFoundError):
        build_or_load_urdf(tmp_path / "models", tmp_path / "dhm.xacro", 1.8, 80.0,
                           mvnx_path=None, subject_urdf=None, builder=fake_urdf_builder)


def test_dhm_subject_cache_survives_without_mvnx(tmp_path):
    """First run builds models/<mvnx-stem>.urdf from the .mvnx; a CSV-only repeat run reuses it by
    pointing subject_urdf (job.urdf_path) at the cached file -- no rebuild, no mvnx."""
    from frame_kinematics.dhm import build_or_load_urdf, urdf_path_for
    models = tmp_path / "models"
    xacro = tmp_path / "dhm.xacro"; xacro.write_text("<robot/>")
    mvnx = tmp_path / "S9.mvnx"; mvnx.write_text("<mvnx/>")
    calls = {"n": 0}
    def counting_builder(*a, **k):
        calls["n"] += 1
        return fake_urdf_builder(*a, **k)
    # first run: mvnx present -> build S9.urdf (named after the mvnx stem)
    out1 = build_or_load_urdf(models, xacro, 1.8, 80.0, mvnx_path=str(mvnx), builder=counting_builder)
    assert out1 == urdf_path_for(models, mvnx) and out1.exists() and calls["n"] == 1
    # second run: no mvnx, reuse via explicit urdf_path -> cache hit, no build
    out2 = build_or_load_urdf(models, xacro, 1.8, 80.0, mvnx_path=None,
                              subject_urdf=str(out1), builder=counting_builder)
    assert str(out2).endswith("S9.urdf") and calls["n"] == 1


# ---------- scene metadata: skeleton, segments, N-pose ---------- #
SKEL_WORLD = {                       # world positions held constant over the trial
    "Pelvis": (1.0, 2.0, 0.50), "L5": (1.0, 2.0, 0.62), "L3": (1.0, 2.0, 0.74),
    "T12": (1.0, 2.0, 0.88), "T8": (1.0, 2.0, 1.02), "Neck": (1.0, 2.0, 1.24),
    "Head": (1.0, 2.0, 1.38), "RightShoulder": (1.0, 1.82, 1.20),
    "RightUpperArm": (1.0, 1.78, 1.18), "RightForeArm": (1.0, 1.75, 0.92),
    "RightHand": (1.0, 1.73, 0.68),
}
YAW = np.deg2rad(30.0)


def _fake_skeleton_df(N=40):
    """Several Xsens segments with the trunk rotated by a known yaw about world Z."""
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_euler("z", YAW).as_quat()                  # [x, y, z, w]
    d = {}
    for seg, (x, y, z) in SKEL_WORLD.items():
        d[f"{seg}_X"] = np.full(N, x)
        d[f"{seg}_Y"] = np.full(N, y)
        d[f"{seg}_Z"] = np.full(N, z)
    for k, c in zip("xyzw", q):
        d[f"T8_{k}_q"] = np.full(N, c)
    return pd.DataFrame(d)


def test_segments_in_trunk_matches_hand_calculation():
    """p_trunk = R_trunk^T (p_world - o_trunk), for a known 30 deg yaw."""
    df = _fake_skeleton_df()
    names = ["Pelvis", "RightHand"]
    P = segments_in_trunk(df, names, "T8")
    o = np.array(SKEL_WORLD["T8"])
    c, s = np.cos(-YAW), np.sin(-YAW)
    Rt = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # R(-yaw) == R(yaw)^T
    for k, seg in enumerate(names):
        assert np.allclose(P[0, k], Rt @ (np.array(SKEL_WORLD[seg]) - o))


def test_ee_in_trunk_is_the_single_segment_case():
    """The ee cloud and the skeleton must land in the same frame -- same code path, so they do."""
    df = _fake_skeleton_df()
    assert np.allclose(ee_in_trunk(df, "RightForeArm", "T8"),
                       segments_in_trunk(df, ["RightForeArm"], "T8")[:, 0, :])


def test_available_segments_uses_xsens_order_and_skips_missing():
    df = _fake_skeleton_df()
    names = available_segments(df)
    assert names[:5] == ["Pelvis", "L5", "L3", "T12", "T8"]      # canonical order, not column order
    assert "LeftHand" not in names                                # absent from the dataframe


def test_build_skeleton_bones_reference_present_segments_only():
    df = _fake_skeleton_df()
    sk = build_skeleton(df, "T8")
    n = len(sk["segment_names"])
    assert sk["pose_mean"].shape == (n, 3)
    assert sk["pose_npose"] is None                               # no calibration frame supplied
    assert sk["bones"], "expected a connected skeleton"
    assert all(0 <= a < n and 0 <= b < n for a, b in sk["bones"])
    # the left-arm bones are dropped because those segments are absent
    assert len(sk["bones"]) < len(XSENS_BONES)
    # T8 is the origin of its own frame
    t8 = sk["segment_names"].index("T8")
    assert np.allclose(sk["pose_mean"][t8], [0, 0, 0], atol=1e-9)


def test_npose_puts_the_arms_down_not_out():
    """The .mvnx calibration frames are all T-POSES (arms lateral). The N-pose keeps their bone
    lengths but rebuilds the posture with the arms hanging, which is what the torque map wants."""
    df = _fake_skeleton_df()
    sk = build_skeleton(df, "T8", npose_world=SKEL_WORLD)
    P, names = sk["pose_npose"], sk["segment_names"]
    hand, t8 = P[names.index("RightHand")], P[names.index("T8")]
    assert np.allclose(t8, [0, 0, 0])                            # trunk frame origin
    assert hand[2] < -0.3, "the hand must hang well below T8"
    # in the source T-pose the hand is 0.75 m lateral; in the N-pose it is tucked in
    assert abs(hand[1]) < 0.3


def test_npose_preserves_measured_bone_lengths():
    """Only the directions are canonical -- the subject's own segment lengths must survive."""
    df = _fake_skeleton_df()
    sk = build_skeleton(df, "T8", npose_world=SKEL_WORLD)
    P, names = sk["pose_npose"], sk["segment_names"]
    for a, b in [("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
                 ("T8", "Neck"), ("Pelvis", "L5")]:
        measured = np.linalg.norm(np.array(SKEL_WORLD[b]) - np.array(SKEL_WORLD[a]))
        rebuilt = np.linalg.norm(P[names.index(b)] - P[names.index(a)])
        assert rebuilt == pytest.approx(measured, abs=1e-9)


def test_npose_spine_stacks_upward():
    df = _fake_skeleton_df()
    sk = build_skeleton(df, "T8", npose_world=SKEL_WORLD)
    P, names = sk["pose_npose"], sk["segment_names"]
    chain = [P[names.index(n)][2] for n in ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head"]]
    assert all(b > a for a, b in zip(chain, chain[1:])), "spine must ascend"


def test_npose_marks_segments_missing_from_the_calibration_frame():
    df = _fake_skeleton_df()
    partial = {k: v for k, v in SKEL_WORLD.items() if k != "RightHand"}
    sk = build_skeleton(df, "T8", npose_world=partial)
    i = sk["segment_names"].index("RightHand")
    assert np.all(np.isnan(sk["pose_npose"][i]))       # NaN, not a silent zero at the origin


def test_read_calibration_pose_from_mvnx(tmp_path):
    """'identity' is the neutral pose; the reader pairs its 3-tuples with the segment labels."""
    from frame_kinematics.mvnx import read_calibration_pose
    mvnx = tmp_path / "S1.mvnx"
    mvnx.write_text(
        '<?xml version="1.0"?>'
        '<mvnx xmlns="http://www.xsens.com/mvn/mvnx"><subject frameRate="240">'
        '<segments><segment label="Pelvis"/><segment label="T8"/></segments>'
        '<frames>'
        '<frame type="identity"><position>1 2 3 4 5 6</position></frame>'
        '<frame type="tpose"><position>9 9 9 9 9 9</position></frame>'
        '<frame type="normal"><position>0 0 0 0 0 0</position></frame>'
        '</frames></subject></mvnx>'
    )
    assert read_calibration_pose(mvnx) == {"Pelvis": (1.0, 2.0, 3.0), "T8": (4.0, 5.0, 6.0)}
    assert read_calibration_pose(mvnx, pose="tpose")["Pelvis"] == (9.0, 9.0, 9.0)
    assert read_calibration_pose(mvnx, pose="npose") == {}       # absent -> empty, not an error


def test_scene_survives_the_npz_boundary():
    """Skeleton + URDF limits must survive the npz cache intact -- figures rebuild from it alone."""
    from frame_core.synthetic import make_scene
    skeleton, limits = make_scene()
    cap = build_capture(
        [{"angles": np.zeros((50, 3))}], DOFS, 240.0, skeleton=skeleton, urdf_limits=limits)
    back = capture_from_npz_bytes(capture_to_npz_bytes(cap))
    assert back.urdf_limits == limits
    assert back.skeleton["segment_names"] == skeleton["segment_names"]
    assert back.skeleton["bones"] == skeleton["bones"]
    assert np.allclose(back.skeleton["pose_mean"], skeleton["pose_mean"])
    assert np.allclose(back.skeleton["pose_npose"], skeleton["pose_npose"])


def test_capture_without_scene_roundtrips_as_none():
    cap = build_capture([{"angles": np.zeros((50, 3))}], DOFS, 240.0)
    back = capture_from_npz_bytes(capture_to_npz_bytes(cap))
    assert back.skeleton is None and back.urdf_limits is None


def test_run_job_attaches_scene_metadata():
    """The worker folds the model limits and the .mvnx N-pose into the Capture."""
    df = make_df(n_reps=2)
    limits = {"right_shoulder_Z": {"lower": -1.0, "upper": 2.0, "effort": 51.0}}
    cap = run_job(df, make_job(), make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder,
                  model_fn=lambda p: {"joints": DOFS, "limits": limits},
                  npose_fn=lambda p: SKEL_WORLD)
    assert cap.urdf_limits == limits
    # make_df carries only RightForeArm + T8, so the skeleton is those two segments
    assert cap.skeleton is not None
    assert "T8" in cap.skeleton["segment_names"]


def test_model_metadata_failure_does_not_fail_the_job():
    """A missing .mvnx or an unreadable URDF degrades the torque map, never the metrics."""
    def boom(*a, **k):
        raise RuntimeError("no model")
    df = make_df(n_reps=2)
    cap = run_job(df, make_job(), make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder, model_fn=boom, npose_fn=boom)
    assert cap.urdf_limits is None
    assert cap.has_torque and cap.n_dof == 3          # the metrics still ran


def test_hull_coverage_is_resolution_independent():
    """The reason we moved off voxel intersection: a SPARSE exo lattice under-counts overlap at a
    fine voxel and recovers at a coarse one, while hull containment gives the same answer."""
    from frame_core.task_space import hull_coverage, coverage_iou
    rng = np.random.default_rng(0)
    exo = rng.uniform(-1, 1, (4000, 3))                      # sparse but spans the cube
    human = rng.uniform(-0.2, 0.2, (500, 3))                 # well inside it
    hull = hull_coverage(human, exo)
    assert hull["hull_coverage"] > 0.95
    assert hull["n_total"] == 500 and hull["exo_hull_volume_L"] > 0
    fine = coverage_iou(human, exo, 0.02)["coverage"]
    coarse = coverage_iou(human, exo, 0.20)["coverage"]
    assert fine < 0.5 < coarse                               # the voxel metric swings; the hull does not


def test_hull_coverage_degenerate_inputs():
    from frame_core.task_space import hull_coverage
    r = hull_coverage(np.zeros((5, 3)), np.zeros((2, 3)))    # too few exo points to triangulate
    assert np.isnan(r["hull_coverage"])


def test_unknown_dof_name_is_caught_before_ik_with_a_suggestion():
    from frame_kinematics.worker import check_dof_names
    with pytest.raises(ValueError, match="right_shoulder_Z"):
        check_dof_names(["right_shoulder_ZZ"], ["right_shoulder_Z", "right_elbow_Y"])


def test_unknown_ee_frame_is_caught():
    from frame_kinematics.worker import check_dof_names
    with pytest.raises(ValueError, match="ee_frame"):
        check_dof_names(["a"], ["a"], ee_frame="right_forarm", model_links=["right_forearm"])


def test_dof_check_is_skipped_when_introspection_is_unavailable():
    from frame_kinematics.worker import check_dof_names
    check_dof_names(["anything"], None)                     # must not raise


def test_run_job_rejects_dof_names_the_model_lacks():
    """The check fires from inside run_job, before any IK runs."""
    df = make_df(n_reps=2)
    job = make_job()
    job.dof_names = ["right_shoulder_Z", "not_a_joint"]
    with pytest.raises(ValueError, match="not_a_joint"):
        run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                urdf_builder=fake_urdf_builder,
                model_fn=lambda p: {"joints": DOFS, "links": ["right_forearm"], "limits": {}})


def test_run_job_builds_urdf_then_runs(tmp_path):
    """Heavy path with no prebuilt URDF: run_job builds it from the mvnx, then runs IK."""
    df = make_df(n_reps=3)
    job = KinematicsJob(dof_names=DOFS, ee_segment="RightForeArm",
                        height_m=1.78, mass_kg=74.0, subject_id="S9",
                        models_dir=str(tmp_path / "models"),
                        xacro_base=str(tmp_path / "dhm.xacro"),
                        mvnx_path=str(tmp_path / "S9.mvnx"))
    (tmp_path / "S9.mvnx").write_text("<mvnx/>")
    cap = run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder)
    assert cap.n_dof == 3 and cap.has_torque
    assert str(job.urdf_path).endswith("S9.urdf")     # resolved + cached on the job


# ---------- canonical frame convention ---------- #
def test_rotation_from_axes_builds_a_proper_rotation():
    from frame_core.frames import rotation_from_axes
    R = rotation_from_axes(left=[0, 1, 0], up=[0, 0, 1])
    assert np.allclose(R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_from_axes_orthonormalises_sloppy_input():
    """Measured anatomical axes are never exactly orthogonal; the result must still be a rotation."""
    from frame_core.frames import rotation_from_axes
    R = rotation_from_axes(left=[0.02, 0.99, 0.05], up=[0.03, -0.04, 0.998])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_from_axes_needs_two_axes():
    from frame_core.frames import rotation_from_axes
    with pytest.raises(ValueError):
        rotation_from_axes(up=[0, 0, 1])


def test_nearest_axis_rotation_snaps_the_exo_permutation():
    """The measured exo frame (X=up, Y=forward, Z=left) must snap to the exact cyclic permutation."""
    from frame_core.frames import rotation_from_axes, nearest_axis_rotation, describe_axes
    R = rotation_from_axes(left=[-0.009, -0.045, 0.999], up=[1.0, -0.024, 0.003])
    P, residual = nearest_axis_rotation(R)
    assert np.allclose(P, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert residual < 5.0
    assert describe_axes(P) == "X=up, Y=forward, Z=left"


def test_snapping_preserves_real_geometry():
    """A subject leaning forward 10 deg is anatomy, not a frame error -- it must survive."""
    from frame_core.frames import rotation_from_axes, nearest_axis_rotation
    lean = np.deg2rad(10.0)
    up = [np.sin(lean), 0.0, np.cos(lean)]            # tilted 'up', axes otherwise canonical
    P, residual = nearest_axis_rotation(rotation_from_axes(left=[0, 1, 0], up=up))
    assert np.allclose(P, np.eye(3))                   # snapped to identity: nothing rotated away
    assert 9.0 < residual < 11.0                       # the lean is reported, not applied


def test_all_axis_rotations_are_proper_and_unique():
    from frame_core.frames import _AXIS_ROTATIONS
    assert len(_AXIS_ROTATIONS) == 24                  # the octahedral group
    assert all(np.isclose(np.linalg.det(P), 1.0) for P in _AXIS_ROTATIONS)


def test_xsens_convention_measures_identity_on_canonical_data():
    """The T8 body frame is already X-forward/Y-left/Z-up, so the correction is the identity."""
    from frame_kinematics.canonical import xsens_trunk_convention
    N = 60
    d = {}
    layout = {"Pelvis": (0, 0, -0.5), "Head": (0, 0, 0.35),
              "RightShoulder": (0, -0.18, 0.15), "LeftShoulder": (0, 0.18, 0.15)}
    for seg, (x, y, z) in layout.items():
        d[f"{seg}_X"] = np.full(N, x); d[f"{seg}_Y"] = np.full(N, y); d[f"{seg}_Z"] = np.full(N, z)
    for c in "XYZ":
        d[f"T8_{c}"] = np.zeros(N)
    d["T8_x_q"] = d["T8_y_q"] = d["T8_z_q"] = np.zeros(N); d["T8_w_q"] = np.ones(N)
    conv = xsens_trunk_convention(pd.DataFrame(d), "T8")
    assert conv.is_identity and conv.residual_deg < 1.0


def test_xsens_convention_detects_a_swapped_export():
    """A capture exported with Y up / Z right must be detected and corrected, not trusted."""
    from frame_kinematics.canonical import xsens_trunk_convention
    N = 60
    d = {}
    # same body, but written with X=forward, Y=UP, Z=RIGHT
    layout = {"Pelvis": (0, -0.5, 0), "Head": (0, 0.35, 0),
              "RightShoulder": (0, 0.15, 0.18), "LeftShoulder": (0, 0.15, -0.18)}
    for seg, (x, y, z) in layout.items():
        d[f"{seg}_X"] = np.full(N, x); d[f"{seg}_Y"] = np.full(N, y); d[f"{seg}_Z"] = np.full(N, z)
    for c in "XYZ":
        d[f"T8_{c}"] = np.zeros(N)
    d["T8_x_q"] = d["T8_y_q"] = d["T8_z_q"] = np.zeros(N); d["T8_w_q"] = np.ones(N)
    conv = xsens_trunk_convention(pd.DataFrame(d), "T8")
    assert not conv.is_identity
    # head must come out ABOVE the pelvis after correction
    head = conv.apply(np.array(layout["Head"]))
    pelvis = conv.apply(np.array(layout["Pelvis"]))
    assert head[2] > pelvis[2]
    left = conv.apply(np.array(layout["LeftShoulder"]))
    assert left[1] > 0                                  # left shoulder on the +Y side


def test_convention_falls_back_to_identity_without_landmarks():
    from frame_kinematics.canonical import xsens_trunk_convention
    df = pd.DataFrame({"T8_X": [0.0], "T8_Y": [0.0], "T8_Z": [0.0],
                       "T8_x_q": [0.0], "T8_y_q": [0.0], "T8_z_q": [0.0], "T8_w_q": [1.0]})
    assert xsens_trunk_convention(df, "T8").is_identity


def test_convention_apply_is_a_noop_when_identity():
    from frame_kinematics.canonical import FrameConvention
    pts = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(FrameConvention(np.eye(3), 0.0, "t").apply(pts), pts)


def test_convention_apply_handles_stacked_shapes():
    """segments_in_trunk passes (N, S, 3); the rotation must broadcast over both leading axes."""
    from frame_kinematics.canonical import FrameConvention
    R = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
    conv = FrameConvention(R, 0.0, "exo")
    pts = np.random.default_rng(0).normal(size=(7, 4, 3))
    out = conv.apply(pts)
    assert out.shape == pts.shape
    assert np.allclose(out[3, 2], R @ pts[3, 2])


# ---------- N-pose torque map ---------- #
def test_npose_torque_map_collapses_dofs_onto_segments():
    """Three shoulder DOFs share one segment -> the WORST ratio wins, not the last one seen."""
    from frame_kinematics.torque_map import npose_torque_map
    from frame_core.synthetic import make_capture
    cap = make_capture(n_reps=2, with_scene=True)
    m = npose_torque_map(cap)
    assert set(m) == {"RightUpperArm"}                 # all three DOFs are shoulder joints
    assert m["RightUpperArm"]["capacity"] == 51.0      # the tightest of the three limits
    assert 0.0 < m["RightUpperArm"]["ratio"] < 1.0
    assert m["RightUpperArm"]["has_capacity"] is True


def test_npose_map_falls_back_to_relative_when_no_capacity():
    from frame_kinematics.torque_map import npose_torque_map
    from frame_core.synthetic import make_capture
    cap = make_capture(n_reps=2, with_scene=True)
    cap.urdf_limits = None
    m = npose_torque_map(cap)
    assert m["RightUpperArm"]["has_capacity"] is False
    assert m["RightUpperArm"]["ratio"] == pytest.approx(1.0)   # normalised to the largest RMS


def test_npose_map_empty_without_torques():
    from frame_kinematics.torque_map import npose_torque_map
    from frame_core.synthetic import make_capture
    assert npose_torque_map(make_capture(n_reps=2, with_torque=False, with_scene=True)) == {}


def test_torque_map_excludes_the_legs_by_default():
    """Display filter, not an analysis one: leg DOFs stay in the Capture, just get no sphere."""
    from frame_kinematics.torque_map import npose_torque_map, UPPER_BODY_PATTERN
    from frame_core.synthetic import make_capture
    import numpy as _np
    cap = make_capture(n_reps=2, with_scene=True)
    cap.dof_names = ["right_shoulder_Z", "right_knee_Y", "lumbarspine_Y"]
    for r in cap.reps:
        r.torques = _np.abs(r.torques) + 10.0
    cap.urdf_limits = {"right_shoulder_Z": {"effort": 51.0},
                       "right_knee_Y": {"effort": 164.0},
                       "lumbarspine_Y": {"effort": 228.0}}
    segs = set(npose_torque_map(cap))
    assert "RightLowerLeg" not in segs                    # knee filtered out
    assert {"RightUpperArm", "L5"} <= segs                # lumbar spine and above kept
    assert "RightLowerLeg" in set(npose_torque_map(cap, pattern=None))   # opt back in


def test_upper_body_pattern_covers_the_chain_from_the_lumbar_spine_up():
    import re
    from frame_kinematics.torque_map import UPPER_BODY_PATTERN
    rx = re.compile(UPPER_BODY_PATTERN, re.I)
    for keep in ("lumbarspine_Y", "thoracicspine_X", "neck_Z", "head_Y",
                 "right_clavicular_X", "left_shoulder_Y", "right_elbow_Z", "left_wrist_X"):
        assert rx.search(keep), keep
    for drop in ("right_hip_X", "left_knee_Y", "right_ankle_Y"):
        assert not rx.search(drop), drop


# ---------- edge trimming ---------- #
def test_trim_edges_removes_the_requested_seconds():
    from frame_kinematics.segment import trim_edges
    df = pd.DataFrame({"a": np.arange(2400)})            # 10 s at 240 Hz
    out, (a, b) = trim_edges(df, 240.0, 3.0, 3.0)
    assert (a, b) == (720, 1680) and len(out) == 960     # 4 s kept
    assert out["a"].iloc[0] == 720 and out["a"].iloc[-1] == 1679


def test_trim_edges_is_a_noop_at_zero():
    from frame_kinematics.segment import trim_edges
    df = pd.DataFrame({"a": np.arange(500)})
    out, w = trim_edges(df, 240.0, 0.0, 0.0)
    assert len(out) == 500 and w == (0, 500)


def test_trim_refuses_to_leave_a_sliver():
    """The guard that matters: 3+3 s on a 6.5 s record clears any frame floor but keeps 8%."""
    from frame_kinematics.segment import trim_edges
    df = pd.DataFrame({"a": np.arange(1566)})            # 6.53 s
    with pytest.raises(ValueError, match="8%"):
        trim_edges(df, 240.0, 3.0, 3.0)
    assert len(trim_edges(df, 240.0, 1.0, 1.0)[0]) == 1086      # a sane trim still works


def test_trim_frame_floor_also_applies():
    from frame_kinematics.segment import trim_edges
    df = pd.DataFrame({"a": np.arange(100)})
    with pytest.raises(ValueError):
        trim_edges(df, 240.0, 0.1, 0.1, min_frames=80, min_keep_frac=0.0)


def test_trim_index_is_reset():
    """Downstream code indexes rows positionally; a stale index would silently misalign IK."""
    from frame_kinematics.segment import trim_edges
    df = pd.DataFrame({"a": np.arange(2400)})
    out, _ = trim_edges(df, 240.0, 1.0, 1.0)
    assert list(out.index[:3]) == [0, 1, 2]


def test_run_job_trims_before_ik_and_records_the_window():
    """The trim must happen BEFORE IK, so spikes never reach the angles at all."""
    df = make_df(n_reps=1, base_len=2400)
    job = make_job()
    job.trim_start_s, job.trim_end_s = 1.0, 1.0
    cap = run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder)
    n_raw = len(df)
    assert cap.reps[0].angles.shape[0] == n_raw - 480      # 1 s off each end at 240 Hz
    assert cap.trim["raw_frames"] == n_raw
    assert cap.trim["kept_frames"] == n_raw - 480
    assert cap.trim["window_s"] == pytest.approx([1.0, (n_raw - 240) / 240.0])


def test_trim_window_survives_the_npz_boundary():
    df = make_df(n_reps=1, base_len=2400)
    job = make_job()
    job.trim_start_s, job.trim_end_s = 1.0, 1.0
    cap = run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder)
    back = capture_from_npz_bytes(capture_to_npz_bytes(cap))
    assert back.trim == cap.trim


def test_run_job_refuses_an_oversized_trim():
    df = make_df(n_reps=1, base_len=1566)
    job = make_job()
    job.trim_start_s, job.trim_end_s = 3.0, 3.0
    with pytest.raises(ValueError, match="trim_start_s"):
        run_job(df, job, make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                urdf_builder=fake_urdf_builder)


# ---------- gravity torque + consistent ID states (ID_VERSION 2) ---------- #
def fake_run_id_with_gravity(ik, fs=None):
    log_torque, grf = fake_run_id(ik, fs)
    log_tau_g = {n: 0.5 * np.asarray(v, float) for n, v in log_torque.items()}
    return log_torque, grf, log_tau_g


def test_gravity_torque_is_carried_into_the_capture_and_through_npz():
    df = make_df()
    cap = run_job(df, make_job(), make_ik=FakeIK, run_id=fake_run_id_with_gravity,
                  jac_fn=fake_jac, urdf_builder=fake_urdf_builder)
    assert cap.has_gravity
    r = cap.reps[0]
    np.testing.assert_allclose(r.gravity_torques, 0.5 * r.torques)
    back = capture_from_npz_bytes(capture_to_npz_bytes(cap))
    assert back.has_gravity
    np.testing.assert_allclose(back.reps[0].gravity_torques, r.gravity_torques)
    assert back.decimate == 1


def test_run_id_without_gravity_leaves_it_absent():
    cap = run_job(make_df(), make_job(), make_ik=FakeIK, run_id=fake_run_id, jac_fn=fake_jac,
                  urdf_builder=fake_urdf_builder)
    assert not cap.has_gravity
    back = capture_from_npz_bytes(capture_to_npz_bytes(cap))
    assert back.reps[0].gravity_torques is None


def test_id_version_is_in_the_batch_fingerprint():
    from frame_kinematics.kinematics import ID_VERSION
    assert ID_VERSION >= 2


def _pendulum_model():
    """Free-flyer base + one revolute joint carrying a point-ish mass at distance l."""
    pin = pytest.importorskip("pinocchio")
    model = pin.Model()
    base = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
    model.appendBodyToJoint(base, pin.Inertia.FromSphere(1.0, 0.05), pin.SE3.Identity())
    j = model.addJoint(base, pin.JointModelRY(), pin.SE3.Identity(), "hinge")
    m, l = 2.0, 0.4
    model.appendBodyToJoint(j, pin.Inertia(m, np.array([0.0, 0.0, -l]), np.eye(3) * 1e-4),
                            pin.SE3.Identity())
    return pin, model, m, l


def test_filtered_states_recover_analytic_velocity_and_acceleration():
    pin, model, m, l = _pendulum_model()
    from frame_kinematics.pinocchio_backend.id_grf import filtered_states
    fs, T = 240.0, 4.0
    t = np.arange(int(fs * T)) / fs
    A, w = 0.8, 2 * np.pi * 0.5
    q = A * np.sin(w * t)
    Q = np.tile(pin.neutral(model), (t.size, 1))
    Q[:, -1] = q
    qf, dq, ddq = filtered_states(model, Q, fs)
    mid = slice(120, -120)                  # 0.5 s clear of the filtfilt edge transient
    np.testing.assert_allclose(dq[mid, -1], A * w * np.cos(w * t[mid]), atol=0.01 * A * w)
    np.testing.assert_allclose(ddq[mid, -1], -A * w * w * np.sin(w * t[mid]),
                               atol=0.02 * A * w * w)
    assert np.allclose(dq[:, :6], 0.0) and np.allclose(ddq[:, :6], 0.0)


def test_rnea_on_filtered_states_has_the_inertial_term_and_gravity_matches():
    """tau = I q'' + m g l sin q for the hinge; g(q) from computeGeneralizedGravity."""
    pin, model, m, l = _pendulum_model()
    from frame_kinematics.pinocchio_backend.id_grf import filtered_states
    data = model.createData()
    fs = 240.0
    t = np.arange(960) / fs
    A, w = 0.8, 2 * np.pi * 1.0
    Q = np.tile(pin.neutral(model), (t.size, 1))
    Q[:, -1] = A * np.sin(w * t)
    qf, dq, ddq = filtered_states(model, Q, fs)
    I = m * l * l + 1e-4
    for i in range(100, 860, 37):
        tau = pin.rnea(model, data, qf[i], dq[i], ddq[i])[-1]
        g = pin.computeGeneralizedGravity(model, data, qf[i])[-1]
        qi = qf[i, -1]
        assert g == pytest.approx(m * 9.81 * l * np.sin(qi), abs=1e-6)
        assert tau == pytest.approx(I * ddq[i, -1] + g, abs=1e-6)
        # and the inertial part is not negligible at this frequency -- the bug hid exactly this
    assert np.max(np.abs(I * ddq[100:860, -1])) > 0.2 * np.max(np.abs(m * 9.81 * l * np.sin(qf[:, -1])))


# ---------- shoulder elevation coordinates: the hard-coded chain matches the DHM ---------- #
def test_shoulder_chain_matches_the_urdf():
    """frame_core.shoulder hard-codes thorax -> clavicle -> GH -> humerus. Check every joint
    against the DHM: parent/child continuity, axis, zero rotation offset, and the humerus long
    axis (shoulder -> elbow origin) along -Y (right) / +Y (left), i.e. the T-pose zero."""
    import glob
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from frame_core.shoulder import CHAINS
    root = Path(__file__).resolve().parents[2] / "models"
    urdfs = sorted(glob.glob(str(root / "subject_*.urdf")))
    if not urdfs:
        pytest.skip("no subject URDF in models/")
    joints = {j.get("name"): j for j in ET.parse(urdfs[0]).getroot().findall("joint")}
    unit = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}
    for side, ch in CHAINS.items():
        chain = list(ch["clav"]) + list(ch["gh"])
        assert joints[chain[0][0]].find("parent").get("link") == "thoracic_spine"
        prev_child = None
        for name, ax in chain:
            j = joints[name]
            assert " ".join(j.find("axis").get("xyz").split()) == unit[ax], name
            rpy = (j.find("origin").get("rpy") or "0 0 0").split()
            assert all(abs(float(v)) < 1e-12 for v in rpy), name
            if prev_child:
                assert j.find("parent").get("link") == prev_child, name
            prev_child = j.find("child").get("link")
        elbow = joints[f"{side}_elbow_Z"]
        assert elbow.find("parent").get("link") == prev_child
        xyz = np.array([float(v) for v in elbow.find("origin").get("xyz").split()])
        assert np.allclose(xyz / np.linalg.norm(xyz), ch["long_axis"])
