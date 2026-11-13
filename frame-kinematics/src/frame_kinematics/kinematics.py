"""
Per-repetition kinematics orchestration.

For one repetition dataframe: run IK (injected engine), inverse dynamics + GRF (the vendored
id_grf), the Jacobian task-contribution, and the trunk-frame end-effector cloud, then package a
payload dict for assemble.build_capture. Every heavy dependency is injected, so this whole
module runs under test with fakes and reaches for Pinocchio only in production defaults.
"""
from __future__ import annotations
import numpy as np

from .assemble import ee_in_trunk, build_skeleton
from .jacobian import jacobian_contribution

# Bump whenever the torque a trial produces changes for a reason the config cannot see. It rides in
# the batch cache key (frame_batch.config.fingerprint), so old cached torques are recomputed.
#   1 : ID_Xsens states (zero q-ddot, q-dot 2.08x too fast, filter assuming fs = 500)
#   2 : id_grf.filtered_states -- consistent q, q-dot, q-ddot at the true capture rate, 6 Hz;
#       plus the gravity torque g(q) per joint
ID_VERSION = 2


def _default_make_ik(urdf_path, mv):
    """Production IK factory. Replace/inject with the project's ID_Xsens."""
    from utils import inverse_dynamics as _id           # the project's module
    return _id.ID_Xsens(urdf_path, mv)


def _default_run_id(ik, fs=None):
    """Production ID+GRF runner (vendored), with the gravity torque."""
    from .pinocchio_backend.id_grf import run_id_with_grf
    return run_id_with_grf(ik, fs=fs, return_gravity=True)


def _filtered_states(ik, fs=None):
    """(q, dq) for the Jacobian pass, built the same way id_grf builds them.

    Falls back to filtering ID_Xsens' own states only where Pinocchio is unavailable (the fakes
    under test) -- in production that fallback would reintroduce the 2.08x velocity error.
    """
    try:
        import pinocchio as pin
        from .pinocchio_backend.id_grf import filtered_states
        real_model = isinstance(ik.robot.model, pin.Model)
    except ImportError:
        real_model = False
    if real_model:
        qf, dqf, _ = filtered_states(ik.robot.model, ik.saved_configurations,
                                     fs or getattr(ik, "xsens_frequency", 240.0))
        return qf, dqf
    Q = np.asarray(ik.saved_configurations, float)
    DQ = np.asarray(ik.saved_velocity, float)
    qf = np.column_stack([ik.butterworth_filter(Q[:, j]) for j in range(Q.shape[1])])
    dqf = np.column_stack([ik.butterworth_filter(DQ[:, j]) for j in range(DQ.shape[1])])
    return qf, dqf


def process_rep(mv, job, make_ik=None, run_id=None, jac_fn=None, want_torque=True, want_jac=True,
                npose_world=None, convention=None):
    """Run the heavy chain on one repetition dataframe -> payload dict."""
    make_ik = make_ik or _default_make_ik
    run_id = run_id or _default_run_id

    ik = make_ik(job.urdf_path, mv)
    ik.uploadMocapFile()
    ik.runFullIK()
    log_q = ik.temp_save_log_q()                                    # {joint: rad}
    angles = np.column_stack([np.asarray(log_q[n], float) for n in job.dof_names])
    payload = {"angles": angles, "torques": None, "gravity_torques": None, "ee_pos": None,
               "jac": None, "skeleton": None}

    # end-effector cloud (trunk frame) straight from the dataframe (no Pinocchio)
    try:
        payload["ee_pos"] = ee_in_trunk(mv, job.ee_segment, job.trunk_segment, convention)
    except KeyError:
        payload["ee_pos"] = None                                   # segment columns absent

    # full-body skeleton for the torque map -- same transform, same frame as the ee cloud
    try:
        payload["skeleton"] = build_skeleton(mv, job.trunk_segment, npose_world=npose_world,
                                             convention=convention)
    except KeyError:
        payload["skeleton"] = None                                 # trunk columns absent

    states = None
    if want_torque:
        out = run_id(ik, fs=job.fs)
        log_torque, grf = out[0], (out[1] or {})
        payload["torques"] = np.column_stack([np.asarray(log_torque[n], float) for n in job.dof_names])
        if len(out) > 2 and out[2] is not None:
            payload["gravity_torques"] = np.column_stack(
                [np.asarray(out[2][n], float) for n in job.dof_names])
        states = grf.get("states")

    if want_jac:
        model, data = ik.robot.model, ik.robot.data
        qf, dqf = (states[0], states[1]) if states is not None else _filtered_states(ik, job.fs)
        payload["jac"] = jacobian_contribution(model, data, qf, dqf, job.ee_frame,
                                               job.dof_names, job.fs, jac_fn=jac_fn)
    return payload
