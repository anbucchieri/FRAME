"""
Model introspection: what the person-scaled URDF offers, read before IK runs.

`run_job` uses it to fail fast on a mistyped DOF or frame name, and to read the joint limits and
torque capacities the metrics normalise against:

  * movable joints  -- every actuated DOF (nq/nv > 0), i.e. the candidate `dof_names`. Fixed
                       joints and the universe/free-flyer root are excluded.
  * link frames     -- body frames only (Pinocchio FrameType.BODY), excluding sensor/motor frames
                       and anything containing 'virtual'. These are the end-effector candidates.

Building the URDF is the expensive part and is cached per subject; introspection itself is
cheap. Pinocchio is imported lazily so importing this module stays free.
"""
from __future__ import annotations


def _pin():
    import pinocchio as pin
    return pin


def movable_joints(model):
    """Actuated joint names in model order (skip 'universe' and the free-flyer root)."""
    pin = _pin()
    out = []
    for jid in range(1, model.njoints):          # 0 = universe
        j = model.joints[jid]
        name = model.names[jid]
        if name == "root_joint":                 # free-flyer base, not an analysed DOF
            continue
        if j.nq == 0:                            # fixed joint
            continue
        out.append(name)
    return out


def link_frames(model, exclude_substrings=("sensor", "motor", "virtual")):
    """Body (link) frame names only, excluding sensor/motor/virtual frames.

    Uses Pinocchio's FrameType.BODY so joint/op frames are dropped, then applies the name filter
    (case-insensitive) for the sensor/motor/virtual conventions in the DHM URDF.
    """
    pin = _pin()
    body = pin.FrameType.BODY
    out = []
    for f in model.frames:
        if f.type != body:
            continue
        low = f.name.lower()
        if any(sub in low for sub in exclude_substrings):
            continue
        if f.name in ("universe",):
            continue
        out.append(f.name)
    return out


def joint_limits(model, names=None):
    """Per-joint limits -> {joint: {"lower","upper","effort","velocity"}}, SI (rad, N.m, rad/s).

    `effort` is the URDF's per-joint torque ceiling. In the DHM models it is populated with human
    joint torque capacities (shoulder_X 69, elbow_Z 75, knee_Y 164, hip_Y 186 N.m), which is what
    Layer 3 normalises RMS/peak torque against. Multi-DOF joints and joints without finite
    position limits are skipped rather than guessed at.
    """
    out = {}
    for name in (names if names is not None else movable_joints(model)):
        jid = model.getJointId(name)
        j = model.joints[jid]
        if j.nq != 1:                            # only 1-DOF joints have a scalar limit row
            continue
        iq, iv = int(j.idx_q), int(j.idx_v)
        entry = {
            "lower": float(model.lowerPositionLimit[iq]),
            "upper": float(model.upperPositionLimit[iq]),
            "effort": float(model.effortLimit[iv]),
            "velocity": float(model.velocityLimit[iv]),
        }
        # non-finite / unset limits are noise downstream -- drop the individual fields
        out[name] = {k: v for k, v in entry.items() if v == v and abs(v) != float("inf")}
    return out


def introspect_model(urdf_path, root_free_flyer=True):
    """Load the URDF and return {'joints', 'links', 'limits', 'nq', 'nv'}."""
    pin = _pin()
    if root_free_flyer:
        model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
    else:
        model = pin.buildModelFromUrdf(str(urdf_path))
    joints = movable_joints(model)
    return {
        "joints": joints,
        "links": link_frames(model),
        "limits": joint_limits(model, joints),
        "nq": int(model.nq),
        "nv": int(model.nv),
    }
