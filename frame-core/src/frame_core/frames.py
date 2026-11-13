"""
Reference-frame transforms -- pure NumPy/SciPy, no Pinocchio.

Xsens exports segment world positions plus per-segment body->world quaternions
([x, y, z, w], matching scipy). To analyse the end-effector "arm-only" (relative to
the trunk) we express its world position in the trunk body frame:

        p_trunk = R_trunk^T (p_world - o_trunk)

It lives in frame-core so the task-space layer can build the trunk-frame cloud without any
kinematics engine.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def to_body_frame(p_world, origin_world, quat_xyzw):
    """Express p_world (N, 3) in the body frame defined by origin_world (N, 3) and
    body->world quaternion quat_xyzw (N, 4, order [x, y, z, w]).

    Returns (N, 3) in the body frame.
    """
    p = np.asarray(p_world, float)
    o = np.asarray(origin_world, float)
    R = Rot.from_quat(np.asarray(quat_xyzw, float))  # body -> world
    return R.inv().apply(p - o)


# ============================== canonical analysis frame ============================== #
# One convention for every source, fixed project-wide:
#
#       +X forward (the direction the subject faces)
#       +Y to the subject's LEFT
#       +Z up
#
# Right-handed, and the same convention the Xsens segment frames already use. Nothing downstream
# -- clouds, skeletons, exo reach, coverage -- is comparable unless every source is expressed here.
#
# The correction is derived from each source's own ANATOMY (head-above-pelvis, left-of-right),
# never from a hardcoded matrix or a URDF's rpy, so it holds for any capture and any exo model.
# It is then SNAPPED to the nearest signed axis permutation: a frame convention differs from the
# canonical one by whole 90 degree steps, so anything left over is real geometry -- the subject's
# trunk lean, a tilted exo mount -- and must be preserved, not rotated away.
CANONICAL_AXES = ("forward", "left", "up")
CANONICAL_DESCRIPTION = "+X forward, +Y left, +Z up"


def rotation_from_axes(forward=None, left=None, up=None):
    """Rotation taking SOURCE-frame coordinates into the canonical frame.

    Pass any two of the three anatomical directions, expressed in the source frame; the third is
    completed by a cross product. Rows of the result are the canonical basis vectors written in
    source coordinates, so `R @ v_source` gives canonical components. The result is orthonormalised
    (nearest true rotation, via SVD) because measured anatomical axes are never exactly orthogonal.
    """
    def _n(v):
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("degenerate axis: the two reference points coincide")
        return v / n

    f = None if forward is None else _n(forward)
    l = None if left is None else _n(left)
    u = None if up is None else _n(up)
    given = sum(x is not None for x in (f, l, u))
    if given < 2:
        raise ValueError("give at least two of forward/left/up")
    if f is None:
        f = np.cross(l, u)
    elif l is None:
        l = np.cross(u, f)
    elif u is None:
        u = np.cross(f, l)

    M = np.vstack([_n(f), _n(l), _n(u)])
    U, _, Vt = np.linalg.svd(M)                 # nearest orthogonal matrix
    R = U @ Vt
    if np.linalg.det(R) < 0:                    # keep it a rotation, never a reflection
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _signed_permutations():
    """The 24 rotations that map axes onto axes (the octahedral group)."""
    from itertools import permutations, product
    out = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for row, col in enumerate(perm):
                P[row, col] = signs[row]
            if np.linalg.det(P) > 0:
                out.append(P)
    return out


_AXIS_ROTATIONS = _signed_permutations()


def nearest_axis_rotation(R):
    """Snap R to the nearest axis-permuting rotation -> (P, residual_degrees).

    A frame-convention mismatch is always a whole-quarter-turn relabelling of axes. Snapping keeps
    exactly that and leaves everything else alone, so a subject's forward lean or an exo's tilted
    mount survives the correction instead of being silently straightened. `residual_degrees` is how
    far the measured rotation sat from the snapped one -- small is healthy, large means the
    anatomical landmarks disagree with the axes and the frame should be checked by hand.
    """
    R = np.asarray(R, float)
    best = max(_AXIS_ROTATIONS, key=lambda P: float(np.sum(P * R)))
    cos = (np.trace(best.T @ R) - 1.0) / 2.0
    return best, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def describe_axes(R):
    """Read a source frame's convention back out of its correction, e.g. 'X=up, Y=forward, Z=left'.

    R maps source -> canonical, so column j says where source axis j lands.
    """
    names = {0: "forward", 1: "left", 2: "up"}
    parts = []
    for j, axis in enumerate("XYZ"):
        col = np.asarray(R, float)[:, j]
        k = int(np.argmax(np.abs(col)))
        parts.append(f"{axis}={'-' if col[k] < 0 else ''}{names[k]}")
    return ", ".join(parts)


def is_canonical(R, tol_deg=1.0):
    """True when the source frame already matches the canonical convention."""
    P, _ = nearest_axis_rotation(R)
    return bool(np.allclose(P, np.eye(3))) and nearest_axis_rotation(R)[1] <= 90.0
