"""
Jacobian task-contribution -- the fourth involvement term, the one that needs a kinematic model.

For each DOF k it accumulates the Cartesian end-effector travel attributable to that joint:

        contribution_k = integral || J_ee[:3, col_k] || * |qd_k|  dt        [m]

i.e. how much of the tool's motion this joint actually drove over the trial. This links
joint-space back to Layer-1 task-space (a DOF with high Cartesian authority is "involved" even
if its ROM is modest). frame-core normalises it across DOFs; here we produce the raw per-DOF
scalar and hand it over as `Repetition.jac_contribution`.

The single Pinocchio touch (the frame Jacobian) is injected as `jac_fn`, so the accumulation
logic is unit-testable with a fake -- no Pinocchio needed to verify the maths or the DOF->column
index mapping.
"""
from __future__ import annotations
import numpy as np


def _dof_vindex(model, name):
    """Velocity-space column index of a 1-DOF joint by name (Pinocchio model)."""
    jid = model.getJointId(name)
    return int(model.joints[jid].idx_v)


def _pin_frame_jac(model, data, q, ee_frame):
    """Default jac_fn: translational (3 x nv) frame Jacobian in LOCAL_WORLD_ALIGNED."""
    import pinocchio as pin
    fid = model.getFrameId(ee_frame)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
    return np.asarray(J[:3, :], float)


def jacobian_contribution(model, data, Q, QD, ee_frame, dof_names, fs, jac_fn=None):
    """Per-DOF task-contribution (m), ordered like `dof_names`.

    Q  : (N, nq) configurations (filtered, as fed to ID).
    QD : (N, nv) velocities (filtered).
    jac_fn(model, data, q, ee_frame) -> (3, nv); defaults to the Pinocchio implementation.
    """
    jac_fn = jac_fn or _pin_frame_jac
    ivs = [_dof_vindex(model, n) for n in dof_names]
    contrib = np.zeros(len(dof_names))
    N = Q.shape[0]
    for i in range(N):
        J = jac_fn(model, data, Q[i], ee_frame)           # (3, nv)
        for k, iv in enumerate(ivs):
            contrib[k] += np.linalg.norm(J[:, iv]) * abs(QD[i, iv]) / fs
    return contrib
