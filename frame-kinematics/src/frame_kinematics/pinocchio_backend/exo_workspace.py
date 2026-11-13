"""
Exoskeleton arm reachable workspace, expressed in the human trunk (T8) frame.

Rationale
---------
The human forearm workspace is built in the T8
trunk frame: every forearm sample is R_T8^T (p - o_T8), so the origin is T8 and the
axes are the Xsens T8 body axes.

To overlay the EXOSKELETON arm workspace in that same frame we anchor the exo on its
`trunk_attachment` link, which is fixed to `upper_back` by `J_trunk_attachment`. We treat
that frame as coincident with the human trunk: every exo tip sample is expressed as
    p_tip_TA = M_TA^{-1} . p_tip            (= R_TA^T (p_tip - o_TA))
i.e. relative to the trunk_attachment frame. Because that frame is declared identical to
T8, the resulting cloud lives in the same coordinate system as the human forearm cloud and
the two superimpose directly. This is base-invariant: any joints between the exo base and
`upper_back` cancel out, so it does not matter where the exo root is placed.

Workspace
---------
For each side we sweep every joint that moves the corresponding `*_arm_tip` relative to
`trunk_attachment`, from its URDF lower to its URDF upper limit, and collect the tip
position in the trunk_attachment frame -> the full cloud of reachable points. The convex
hull of that cloud is the exo arm reach envelope.

Axis alignment -- MEASURED, not assumed
--------------------------------------
The earlier note here claimed the rpy in J_trunk_attachment already matched the Xsens T8
axes. It does not, at least for exo1_full: measured from the model's own geometry the
attachment frame is X=up, Y=forward, Z=left, a cyclic permutation away from T8's
X=forward, Y=left, Z=up. Overlaying the two without correcting it rotated the exo reach a
quarter turn against the human workspace it is meant to be compared with.

So `extra_R` is now DERIVED per model (canonical.exo_trunk_convention): left comes from
left_arm_tip - right_arm_tip, up from the pack body below the attachment, and the result is
snapped to the nearest quarter turn so a genuinely tilted mount survives the correction.
Pass `extra_R` explicitly to override the measurement.
"""

import os
from itertools import product
import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

# ------------------------------- configuration ------------------------------- #
# Default location inside the container: <FRAME_MODELS_DIR>/exoskeleton/exo1_full.urdf. Pass an
# explicit path (job.exo_urdf) to use another model.
EXO_URDF       = os.path.join(os.environ.get("FRAME_MODELS_DIR", "/opt/frame/models"),
                              "exoskeleton", "exo1_full.urdf")
TRUNK_FRAME    = 'trunk_attachment'                                   # exo link coincident with T8
ARM_TIPS       = {'right': 'right_arm_tip', 'left': 'left_arm_tip'}   # frame at each arm extremity
GRID_BUDGET    = 200_000        # max FK evaluations per arm before switching to Monte-Carlo
GRID_KMAX      = 15             # max grid samples per joint
MC_SAMPLES     = 200_000        # samples if the dense grid would exceed the budget


# --------------------------------- internals --------------------------------- #
def _joint_limits(model, jid):
    """(idx_q, lower, upper) for a 1-DoF joint; unbounded/continuous -> [-pi, pi]."""
    iq = model.joints[jid].idx_q
    nq = model.joints[jid].nq
    lo = float(model.lowerPositionLimit[iq])
    hi = float(model.upperPositionLimit[iq])
    if nq != 1 or not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = -np.pi, np.pi
    return iq, lo, hi


def _joints_moving_tip(model, tip_frame, ta_parent_joint):
    """Joints on the path that move `tip_frame` but NOT `trunk_attachment`
    (i.e. the arm joints; anything proximal to the attachment cancels out)."""
    fid = model.getFrameId(tip_frame)
    tip_joint = model.frames[fid].parentJoint
    sup_ta = set(model.supports[ta_parent_joint])
    return [j for j in model.supports[tip_joint] if j not in sup_ta and j != 0]


def _tip_cloud(model, data, fid_ta, tip_frame, swept_joints, extra_R,
               grid_budget=GRID_BUDGET, kmax=GRID_KMAX, mc_samples=MC_SAMPLES):
    """Cloud of `tip_frame` positions in the trunk_attachment frame over the full
    joint range. Dense grid if it fits the budget, else Monte-Carlo + limit corners."""
    fid_tip = model.getFrameId(tip_frame)
    lims    = [_joint_limits(model, j) for j in swept_joints]
    n       = len(lims)
    q0      = pin.neutral(model)

    def fk(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = data.oMf[fid_ta].actInv(data.oMf[fid_tip].translation)   # R_TA^T (p - o_TA)
        return extra_R @ p

    pts = []
    k = min(kmax, max(2, int(np.floor(grid_budget ** (1.0 / max(n, 1)))))) if n else 1
    if n == 0:                                   # tip rigidly fixed to the attachment
        return fk(q0)[None, :]

    if k ** n <= grid_budget:                    # dense grid (endpoints included)
        grids = [(iq, np.linspace(lo, hi, k)) for (iq, lo, hi) in lims]
        for combo in product(*[g for _, g in grids]):
            q = q0.copy()
            for (iq, _), v in zip(grids, combo):
                q[iq] = v
            pts.append(fk(q))
    else:                                        # Monte-Carlo + the 2^n limit corners
        rng = np.random.default_rng(0)
        for _ in range(mc_samples):
            q = q0.copy()
            for (iq, lo, hi) in lims:
                q[iq] = rng.uniform(lo, hi)
            pts.append(fk(q))
        for combo in product(*[(lo, hi) for (_, lo, hi) in lims]):
            q = q0.copy()
            for (iq, _, _), v in zip(lims, combo):
                q[iq] = v
            pts.append(fk(q))
    return np.asarray(pts)


# ----------------------------------- public ---------------------------------- #
def compute_exo_workspace(urdf_path=EXO_URDF, trunk_frame=TRUNK_FRAME,
                          arm_tips=ARM_TIPS, extra_R=None, verbose=True,
                          return_convention=False):
    """Reachable point clouds of each arm tip in the trunk_attachment (== T8) frame.

    Returns {side: (N, 3) ndarray}. Compute this once; it is independent of the
    movement / pick-height category, so it can be reused for every plot_workspace call.
    """
    model   = pin.buildModelFromUrdf(str(urdf_path))     # fixed base (no free-flyer needed)
    data    = model.createData()

    conv = None
    if extra_R is None:
        # Measure this model's own convention instead of trusting J_trunk_attachment's rpy. On the
        # reference exo the attachment frame is X=up, Y=forward, Z=left -- a quarter turn away from
        # the Xsens trunk frame it gets overlaid on.
        from ..canonical import exo_trunk_convention
        conv = exo_trunk_convention(model, data, trunk_frame, arm_tips)
        extra_R = conv.rotation
        if verbose:
            print(f"exo frame  | {conv.description} -> canonical "
                  f"(residual {conv.residual_deg:.1f} deg)")
    else:
        extra_R = np.asarray(extra_R, float)

    fid_ta = model.getFrameId(trunk_frame)
    if fid_ta == model.nframes:                          # getFrameId returns nframes if absent
        raise ValueError(f"'{trunk_frame}' frame not found in {urdf_path}")
    ta_parent_joint = model.frames[fid_ta].parentJoint

    clouds = {}
    for side, tip in arm_tips.items():
        if model.getFrameId(tip) == model.nframes:
            raise ValueError(f"'{tip}' frame not found in {urdf_path}")
        swept = _joints_moving_tip(model, tip, ta_parent_joint)
        cloud = _tip_cloud(model, data, fid_ta, tip, swept, extra_R)
        clouds[side] = cloud
        if verbose:
            names = [model.names[j] for j in swept]
            ext   = cloud.max(0) - cloud.min(0)
            try:
                vol = ConvexHull(cloud).volume * 1e3
            except Exception:
                vol = float('nan')
            print(f"exo {side:>5s} | {len(swept)} DoF {names} | {cloud.shape[0]} pts "
                  f"| hull vol = {vol:6.2f} L | bbox = {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")
    return (clouds, conv) if return_convention else clouds


def compute_exo_arm_directions(urdf_path=EXO_URDF, trunk_frame=TRUNK_FRAME,
                               arm_tips=ARM_TIPS, extra_R=None, n_samples=300_000, seed=0):
    """Directions the exo's upper-arm segment can point, as ISB plane of elevation / elevation.

    For each side every joint that moves the arm tip (relative to the trunk attachment) is sampled
    uniformly over its URDF range (`n_samples` draws plus every joint-limit corner): a coarse grid
    leaves bands of elevation unsampled, which would read as unreachable. The arm segment's long axis u = pivot -> tip, where the pivot
    is the origin of the LAST swept joint (the arm's own rotation axis) and the tip is the
    `*_arm_tip` frame, expressed in the canonical trunk frame (X forward, Y left, Z up) with the
    same measured correction `compute_exo_workspace` uses. Then, exactly as for the human
    (frame_core.shoulder):

        elevation         theta = arccos(-u_z)
        plane of elevation phi  = atan2(u_x, -u_y)  right arm  /  atan2(u_x, +u_y)  left arm

    Returns {side: {"phi_deg": (N,), "theta_deg": (N,), "joints": [names]}}. The exo arm
    direction stands for the human humerus only as far as the cuff keeps them aligned.
    """
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    if extra_R is None:
        from ..canonical import exo_trunk_convention
        extra_R = exo_trunk_convention(model, data, trunk_frame, arm_tips).rotation
    extra_R = np.asarray(extra_R, float)
    fid_ta = model.getFrameId(trunk_frame)
    if fid_ta == model.nframes:
        raise ValueError(f"'{trunk_frame}' frame not found in {urdf_path}")
    ta_parent_joint = model.frames[fid_ta].parentJoint
    q0 = pin.neutral(model)
    out = {}
    for side, tip in arm_tips.items():
        fid_tip = model.getFrameId(tip)
        swept = _joints_moving_tip(model, tip, ta_parent_joint)
        if not swept:
            continue
        pivot = swept[-1]
        lims = [_joint_limits(model, j) for j in swept]
        rng = np.random.default_rng(seed)
        draws = [np.column_stack([rng.uniform(lo, hi, n_samples) for (_, lo, hi) in lims]),
                 np.array(list(product(*[(lo, hi) for (_, lo, hi) in lims])))]
        us = []
        for vals in np.vstack(draws):
            q = q0.copy()
            for (iq, _, _), v in zip(lims, vals):
                q[iq] = v
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            ta = data.oMf[fid_ta]
            d = ta.rotation.T @ (data.oMf[fid_tip].translation - data.oMi[pivot].translation)
            us.append(extra_R @ d)
        u = np.asarray(us)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        lat = -1.0 if side == "right" else 1.0
        out[side] = {"phi_deg": np.degrees(np.arctan2(u[:, 0], lat * u[:, 1])),
                     "theta_deg": np.degrees(np.arccos(np.clip(-u[:, 2], -1, 1))),
                     "joints": [model.names[j] for j in swept]}
    return out


if __name__ == '__main__':                               # quick self-test
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else EXO_URDF
    compute_exo_workspace(path)