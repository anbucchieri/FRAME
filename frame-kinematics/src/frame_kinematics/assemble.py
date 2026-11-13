"""
Capture assembly -- the bridge from heavy-path outputs to the frame-core data model.

Takes per-repetition payload dicts (angles / torques / ee_pos / jac) and builds a
`frame_core.Capture` that the pure engine consumes unchanged. No Pinocchio here: this is the
seam where the heavy path hands off to the light engine.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from frame_core.types import Capture, Repetition


# Xsens 23-segment skeleton connectivity, by segment NAME so a partial segment set still draws.
# A tree: 23 segments -> 22 bones. Fixed by the Xsens body model, not per-subject.
XSENS_BONES = [
    ("Pelvis", "L5"), ("L5", "L3"), ("L3", "T12"), ("T12", "T8"),
    ("T8", "Neck"), ("Neck", "Head"),
    ("T8", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("T8", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Pelvis", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"), ("RightFoot", "RightToe"),
    ("Pelvis", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
]

# Canonical Xsens ordering, used to sort whatever subset the dataframe actually carries.
XSENS_SEGMENTS = [
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe",
]

# N-pose bone directions in the trunk frame: +Z up, +X forward, +Y to the subject's LEFT
# (the Xsens convention -- right-side segments sit at negative Y in the capture).
#
# Why a direction table rather than a captured frame: the .mvnx calibration frames ('identity',
# 'tpose', 'tpose-isb') are all T-POSES -- 'identity' zeroes segment ORIENTATIONS, which in the
# Xsens model leaves the arms pointing straight out to the sides (measured: hand 0.77 m lateral,
# level with T8). None of them is the arms-down N-pose. Bone LENGTHS, however, are pose-invariant,
# so the subject-scaled N-pose is those measured lengths laid out along these fixed directions.
NPOSE_DIRECTIONS = {
    ("Pelvis", "L5"): (0, 0, 1), ("L5", "L3"): (0, 0, 1), ("L3", "T12"): (0, 0, 1),
    ("T12", "T8"): (0, 0, 1), ("T8", "Neck"): (0, 0, 1), ("Neck", "Head"): (0, 0, 1),
    ("T8", "RightShoulder"): (0, -1, 0), ("RightShoulder", "RightUpperArm"): (0, -1, 0),
    ("RightUpperArm", "RightForeArm"): (0, 0, -1), ("RightForeArm", "RightHand"): (0, 0, -1),
    ("T8", "LeftShoulder"): (0, 1, 0), ("LeftShoulder", "LeftUpperArm"): (0, 1, 0),
    ("LeftUpperArm", "LeftForeArm"): (0, 0, -1), ("LeftForeArm", "LeftHand"): (0, 0, -1),
    ("Pelvis", "RightUpperLeg"): (0, -1, 0), ("RightUpperLeg", "RightLowerLeg"): (0, 0, -1),
    ("RightLowerLeg", "RightFoot"): (0, 0, -1), ("RightFoot", "RightToe"): (1, 0, 0),
    ("Pelvis", "LeftUpperLeg"): (0, 1, 0), ("LeftUpperLeg", "LeftLowerLeg"): (0, 0, -1),
    ("LeftLowerLeg", "LeftFoot"): (0, 0, -1), ("LeftFoot", "LeftToe"): (1, 0, 0),
}


def npose_from_lengths(calib_world, segment_names, trunk_segment="T8"):
    """The subject's neutral (N) pose -> (S, 3) in the trunk frame, or None.

    `calib_world` is {segment: (x, y, z)} from any .mvnx calibration frame (see
    mvnx.read_calibration_pose). Only the bone lengths it implies are used; the posture is rebuilt
    along NPOSE_DIRECTIONS, so the result is a genuine arms-down N-pose scaled to this subject.

    Segments the calibration frame cannot reach (missing, or disconnected from the root) come back
    as NaN so the renderer skips them rather than piling them on the origin.
    """
    if not calib_world:
        return None
    index = {n: i for i, n in enumerate(segment_names)}
    out = np.full((len(segment_names), 3), np.nan)

    root = "Pelvis" if "Pelvis" in index else segment_names[0]
    if root in index:
        out[index[root]] = (0.0, 0.0, 0.0)

    # walk the bone tree outward from the root; XSENS_BONES is already parent-before-child
    progressed = True
    while progressed:
        progressed = False
        for a, b in XSENS_BONES:
            if a not in index or b not in index:
                continue
            ia, ib = index[a], index[b]
            if np.isnan(out[ia]).any() or not np.isnan(out[ib]).any():
                continue
            if a not in calib_world or b not in calib_world:
                continue
            length = float(np.linalg.norm(np.asarray(calib_world[b], float)
                                          - np.asarray(calib_world[a], float)))
            out[ib] = out[ia] + length * np.asarray(NPOSE_DIRECTIONS.get((a, b), (0, 0, -1)), float)
            progressed = True

    if trunk_segment in index and not np.isnan(out[index[trunk_segment]]).any():
        out -= out[index[trunk_segment]]                 # express in the trunk frame
    return out


def _pcols(seg):
    return [f"{seg}_X", f"{seg}_Y", f"{seg}_Z"]


def _qcols(seg):
    return [f"{seg}_x_q", f"{seg}_y_q", f"{seg}_z_q", f"{seg}_w_q"]


def segments_in_trunk(mv, segment_names, trunk_segment="T8", convention=None):
    """Cartesian trajectories of several segments in the CANONICAL trunk frame -> (N, S, 3).

    Two steps: p_trunk = R_trunk^T (p - o_trunk) from the Xsens columns (positions [m] + trunk
    body->world quaternion [x,y,z,w]), then the trunk frame's own axis convention is corrected to
    +X forward / +Y left / +Z up. `convention` is a canonical.FrameConvention; omitted, it is
    measured from this capture's anatomy. For Xsens it comes out as the identity -- the T8 body
    frame is already canonical -- so this costs nothing and guards against a re-exported capture
    that is not.

    `ee_in_trunk` is the single-segment case, so the end-effector cloud, the skeleton and the exo
    reach are all guaranteed to share one frame.
    """
    if convention is None:
        from .canonical import xsens_trunk_convention
        convention = xsens_trunk_convention(mv, trunk_segment)
    o = mv[_pcols(trunk_segment)].to_numpy(float)
    R = Rot.from_quat(mv[_qcols(trunk_segment)].to_numpy(float)).inv()   # world -> trunk body
    out = np.empty((len(mv), len(segment_names), 3), float)
    for k, seg in enumerate(segment_names):
        out[:, k, :] = R.apply(mv[_pcols(seg)].to_numpy(float) - o)
    return convention.apply(out)


def ee_in_trunk(mv, ee_segment, trunk_segment, convention=None):
    """End-effector Cartesian trajectory in the canonical trunk frame (frame_core.frames
    world->trunk transform, plus the axis-convention correction)."""
    return segments_in_trunk(mv, [ee_segment], trunk_segment, convention)[:, 0, :]


def available_segments(mv):
    """Segment names the dataframe carries full X/Y/Z position columns for, in Xsens order.

    Anything outside XSENS_SEGMENTS (a custom prop, say) is kept and appended after the known
    ones, so an unusual capture still renders.
    """
    cols = set(mv.columns)
    have = [s for s in XSENS_SEGMENTS if all(c in cols for c in _pcols(s))]
    known = set(XSENS_SEGMENTS)
    extra = sorted({c[:-2] for c in cols if c.endswith("_X")}
                   - known - {"COM"})
    extra = [s for s in extra if all(c in cols for c in _pcols(s))]
    return have + extra


def build_skeleton(mv, trunk_segment="T8", npose_world=None, convention=None):
    """Skeleton for the N-pose torque map: segment names, bone index pairs, and two poses.

    pose_mean  : per-segment median of the measured trajectory (the posture the workspace cloud
                 was produced from -- what plot_workspace() draws the stick figure at).
    pose_npose : the subject's neutral posture, rebuilt from the bone lengths implied by the
                 .mvnx calibration frame `npose_world` = {segment: (3,) world xyz}. See
                 npose_from_lengths -- the calibration frames themselves are T-poses.

    Returns the dict that goes on `Capture.skeleton`, or None if no segment has position columns.
    """
    names = available_segments(mv)
    if not names:
        return None
    index = {n: i for i, n in enumerate(names)}
    P = segments_in_trunk(mv, names, trunk_segment, convention)      # (N, S, 3)

    return {
        "segment_names": names,
        "bones": [[index[a], index[b]] for a, b in XSENS_BONES
                  if a in index and b in index],
        "pose_mean": np.median(P, axis=0),
        # NPOSE_DIRECTIONS is already written in canonical axes, so no correction here
        "pose_npose": npose_from_lengths(npose_world, names, trunk_segment),
    }


def build_capture(rep_payloads, dof_names, fs, exo_cloud=None, height=None, mass=None,
                  skeleton=None, urdf_limits=None):
    """Assemble a frame_core.Capture from a list of per-rep payload dicts.

    Each payload: {'angles': (N, nDOF), 'torques': (N, nDOF)|None,
                   'gravity_torques': (N, nDOF)|None,
                   'ee_pos': (N, 3)|None, 'jac': (nDOF,)|None, 'skeleton': dict|None}

    `skeleton` defaults to the first repetition that produced one -- it is presentation metadata for the
    whole capture, not a per-rep quantity.
    """
    reps, ee_all = [], []
    for p in rep_payloads:
        reps.append(Repetition(
            angles=np.asarray(p["angles"], float),
            torques=None if p.get("torques") is None else np.asarray(p["torques"], float),
            gravity_torques=(None if p.get("gravity_torques") is None
                             else np.asarray(p["gravity_torques"], float)),
            ee_pos=None if p.get("ee_pos") is None else np.asarray(p["ee_pos"], float),
            jac_contribution=None if p.get("jac") is None else np.asarray(p["jac"], float),
        ))
        if p.get("ee_pos") is not None:
            ee_all.append(np.asarray(p["ee_pos"], float))
        if skeleton is None and p.get("skeleton") is not None:
            skeleton = p["skeleton"]
    return Capture(
        reps=reps, dof_names=list(dof_names), fs=float(fs),
        ee_cloud=(np.vstack(ee_all) if ee_all else None),
        exo_cloud=exo_cloud, height_m=height, mass_kg=mass,
        skeleton=skeleton, urdf_limits=urdf_limits,
    )
