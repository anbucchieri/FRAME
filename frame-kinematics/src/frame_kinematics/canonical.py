"""
Putting the human and the exoskeleton into the same frame, for any capture and any model.

Both clouds are drawn in "the trunk frame", but that phrase meant two different things:

  * Xsens T8 body frame  -- measured: X=forward, Y=left, Z=up (already canonical).
  * exo trunk_attachment -- measured: X=up,      Y=forward, Z=left (a cyclic permutation).

So the exo reach was landing rotated a quarter turn against the human workspace it is supposed to
be compared with, and `exo_workspace`'s claim that "the rpy in J_trunk_attachment is what makes
those axes match the Xsens T8 axes" does not hold for this model.

Rather than hardcode a fix for one file pair, each source reports its own convention by measuring
ANATOMY it is guaranteed to carry -- head above pelvis, left side opposite right -- and the result
is snapped to the nearest quarter-turn (see frame_core.frames.nearest_axis_rotation). That is
permanent: a different MVNX, a different exo URDF, or a re-exported model with different rpy all
get measured rather than assumed, and the residual angle says when the measurement disagrees with
the axes badly enough to need a human look.
"""
from __future__ import annotations
import numpy as np

from frame_core.frames import rotation_from_axes, nearest_axis_rotation, describe_axes

# anatomical landmark pairs, most reliable first -- the fallbacks keep partial exports working
UP_PAIRS = [("Pelvis", "Head"), ("Pelvis", "Neck"), ("T8", "Head"), ("L5", "T8")]
LEFT_PAIRS = [("RightShoulder", "LeftShoulder"), ("RightUpperArm", "LeftUpperArm"),
              ("RightUpperLeg", "LeftUpperLeg"), ("RightHand", "LeftHand")]


class FrameConvention:
    """A source's measured convention: the snapped rotation into the canonical frame."""

    def __init__(self, rotation, residual_deg, source):
        self.rotation = np.asarray(rotation, float)      # source -> canonical
        self.residual_deg = float(residual_deg)
        self.source = source

    @property
    def is_identity(self):
        return bool(np.allclose(self.rotation, np.eye(3)))

    @property
    def description(self):
        return describe_axes(self.rotation)

    def apply(self, points):
        """Rotate (..., 3) points from the source frame into the canonical frame."""
        P = np.asarray(points, float)
        if self.is_identity:
            return P
        return P @ self.rotation.T

    def as_dict(self):
        return {"source": self.source, "rotation": self.rotation.tolist(),
                "residual_deg": round(self.residual_deg, 3),
                "axes": self.description, "identity": self.is_identity}

    def __repr__(self):
        return (f"<FrameConvention {self.source}: {self.description} "
                f"(residual {self.residual_deg:.1f} deg)>")


def _first_available(mv_dirs, pairs):
    for a, b in pairs:
        v = mv_dirs(a, b)
        if v is not None and np.linalg.norm(v) > 1e-6:
            return v
    return None


def xsens_trunk_convention(mv, trunk_segment="T8"):
    """Measure the Xsens trunk BODY frame's convention from the capture itself.

    Anatomical directions are rotated into the trunk body frame sample by sample and averaged, so
    the result describes the frame's axis convention, not the subject's momentary posture -- and
    the snap then discards everything but the quarter turns, leaving posture untouched.
    """
    from scipy.spatial.transform import Rotation as Rot

    pcols = lambda s: [f"{s}_X", f"{s}_Y", f"{s}_Z"]
    qcols = lambda s: [f"{s}_x_q", f"{s}_y_q", f"{s}_z_q", f"{s}_w_q"]
    if not all(c in mv.columns for c in qcols(trunk_segment)):
        return FrameConvention(np.eye(3), 0.0, f"xsens:{trunk_segment}")
    R_trunk = Rot.from_quat(mv[qcols(trunk_segment)].to_numpy(float)).inv()   # world -> body

    def direction(a, b):
        if not all(c in mv.columns for c in pcols(a) + pcols(b)):
            return None
        d = mv[pcols(b)].to_numpy(float) - mv[pcols(a)].to_numpy(float)
        return R_trunk.apply(d).mean(axis=0)

    up = _first_available(direction, UP_PAIRS)
    left = _first_available(direction, LEFT_PAIRS)
    if up is None or left is None:                  # not enough landmarks -> assume canonical
        return FrameConvention(np.eye(3), 0.0, f"xsens:{trunk_segment}")
    P, residual = nearest_axis_rotation(rotation_from_axes(left=left, up=up))
    return FrameConvention(P, residual, f"xsens:{trunk_segment}")


def exo_trunk_convention(model, data, trunk_frame, arm_tips, down_frame_candidates=
                         ("base_link", "upper_back", "back_imu")):
    """Measure the exo `trunk_frame`'s convention from the model's own neutral geometry.

    left = left tip - right tip; down = trunk frame -> whichever of `down_frame_candidates` the
    model has (the pack body sits below the attachment). Pinocchio only -- no rpy is trusted.
    """
    import pinocchio as pin

    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    TA = data.oMf[model.getFrameId(trunk_frame)]

    def rel(name):
        fid = model.getFrameId(name)
        if fid >= model.nframes:
            return None
        return TA.actInv(data.oMf[fid].translation)

    rt, lt = rel(arm_tips.get("right", "")), rel(arm_tips.get("left", ""))
    down = next((v for v in (rel(n) for n in down_frame_candidates)
                 if v is not None and np.linalg.norm(v) > 1e-6), None)
    if rt is None or lt is None or down is None:
        return FrameConvention(np.eye(3), 0.0, f"exo:{trunk_frame}")
    P, residual = nearest_axis_rotation(rotation_from_axes(left=lt - rt, up=-down))
    return FrameConvention(P, residual, f"exo:{trunk_frame}")
