"""
inverse_dynamics_grf.py
=======================
Floating-base inverse dynamics with the modelled GRF/CoP from grf_model.py.

This is the corrected version of ID_Xsens.runInverseDynamics. Two fixes vs. the original:

  1. SIGN. The equation of motion is   M a + b = tau + sum J^T f_ext.
     pin.rnea(q, dq, ddq) returns M a + b (no external force), so the required actuation is
            tau = rnea(q, dq, ddq) - sum J^T f_ext
     with f_ext the PHYSICAL ground-on-foot wrench (pushing up). Hence MINUS, not plus.

  2. WRENCH. The GRF moved from the CoP to the foot frame is (force, (cop - p_foot) x force).
     The force is KEPT - a pure moment cannot transmit the vertical support up the leg.

Frame consistency: the foot Jacobian is taken in LOCAL_WORLD_ALIGNED and the wrench from
grf_model is already expressed that way (world axes, referenced to the foot origin), so they
pair correctly.

Note on the upper limb: the foot Jacobian has zero columns for every joint not on the
base->foot path, so the clavicular/shoulder torques here are identical with or without the
GRF. The GRF only affects the base (first 6) and the leg joints - it is wired in for the
future lower-limb analysis.

  3. STATES. q-dot and q-ddot are NOT taken from ID_Xsens. Its `saved_accelerations` are
     identically zero (q_prev / q_prev2 alias q_revised, so (q - 2 q_prev + q_prev2) = 0), and its
     `saved_velocity` is the IK step over dt = 1/500 s on 240 Hz frames (2.08x too fast); its
     Butterworth also assumes fs = 500. `filtered_states` rebuilds all three from the IK
     configurations alone, with the project convention (6 Hz, order 4, filtfilt, true fs) and
     pin.difference for the free-flyer, so rnea finally sees the inertial term M(q) q-ddot.

Validation handles returned in `grf`:
    grf['base_residual']  : saved_torques[:, :6]  - should be ~0 if the GRF model is good.
    F_total[:, 2] ~ m_total * 9.81 at quiet stance.
"""

import numpy as np
import pinocchio as pin

from frame_core.signal import butter_lowpass, CUTOFF_HZ, ORDER
from .grf_model import compute_grf_cop


def _continuous_quaternions(Q, model):
    """Flip free-flyer quaternions onto one hemisphere so filtering does not average q and -q."""
    Q = np.array(Q, float, copy=True)
    if model.nq == model.nv:                    # no free-flyer
        return Q
    quat = Q[:, 3:7]
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    return Q


def filtered_states(model, Q, fs, cutoff=CUTOFF_HZ, order=ORDER):
    """(q, q-dot, q-ddot) from IK configurations alone, at the true capture rate.

    q is low-passed (zero-phase Butterworth, project convention) and the free-flyer quaternion
    renormalised; q-dot is the central difference on the configuration manifold
    (pin.difference, so the base rotation is handled properly); q-ddot is the central
    difference of q-dot. One-sided at the ends.
    """
    Q = _continuous_quaternions(np.asarray(Q, float), model)
    qf = butter_lowpass(Q, cutoff, fs, order)
    if model.nq != model.nv:
        qf[:, 3:7] /= np.linalg.norm(qf[:, 3:7], axis=1, keepdims=True)
    N = qf.shape[0]
    dt = 1.0 / float(fs)
    dqf = np.zeros((N, model.nv))
    for i in range(N):
        a, b = max(i - 1, 0), min(i + 1, N - 1)
        if b > a:
            dqf[i] = pin.difference(model, qf[a], qf[b]) / ((b - a) * dt)
    ddqf = np.gradient(dqf, dt, axis=0) if N > 1 else np.zeros_like(dqf)
    return qf, dqf, ddqf


def run_id_with_grf(ik, reference=pin.LOCAL_WORLD_ALIGNED,
                    foot_frames=("right_foot", "left_foot"),
                    contact_fs=None, grf_kwargs=None, fs=None, return_gravity=False):
    """
    Parameters
    ----------
    ik          : an ID_Xsens instance AFTER runFullIK() (saved_configurations / _velocity /
                  _accelerations populated).
    reference   : Jacobian reference frame; must match grf_model's convention -> LOCAL_WORLD_ALIGNED.
    contact_fs  : capture rate for foot-velocity contact detection (defaults to fs).
    grf_kwargs  : extra kwargs forwarded to compute_grf_cop (thresholds, cop_forward_offset, ...).
    fs          : true capture rate [Hz] (defaults to ik.xsens_frequency).
    return_gravity : also return the gravity torque g(q) = computeGeneralizedGravity(q), on the
                  same filtered q that feeds rnea.

    Returns
    -------
    log_torque  : dict joint_name -> list of torque [N.m] over the trial (base 6 DOF dropped).
    grf         : the full dict from compute_grf_cop, plus 'base_residual' (N, 6) and
                  'states' = (q, dq, ddq), the filtered states actually used.
    log_tau_g   : (only with return_gravity) dict joint_name -> list of gravity torque [N.m],
                  named and base-dropped exactly like log_torque.
    """
    model, data = ik.robot.model, ik.robot.data
    grf_kwargs = dict(grf_kwargs or {})
    fs = float(fs or getattr(ik, "xsens_frequency", 240.0))
    contact_fs = contact_fs or fs

    # ---- consistent states from the IK configurations (see fix 3) ----
    qf, dqf, ddqf = filtered_states(model, ik.saved_configurations, fs)

    # ---- modelled GRF / CoP / foot-frame wrench ----
    grf = compute_grf_cop(model, data, qf, dqf, ddqf,
                          contact_fs=contact_fs, foot_frames=foot_frames, **grf_kwargs)

    fidR = model.getFrameId(foot_frames[0])
    fidL = model.getFrameId(foot_frames[1])

    N, nv = ddqf.shape
    saved_torques = np.zeros((N, nv))
    gravity = np.zeros((N, nv)) if return_gravity else None
    for i in range(N):
        q, dq, ddq = qf[i], dqf[i], ddqf[i]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        J_R = pin.computeFrameJacobian(model, data, q, fidR, reference)   # 6 x nv, rows [lin; ang]
        J_L = pin.computeFrameJacobian(model, data, q, fidL, reference)

        w_R = np.concatenate([grf["force_R"][i], grf["moment_R"][i]])      # [force; moment]
        w_L = np.concatenate([grf["force_L"][i], grf["moment_L"][i]])

        tau_ext = J_R.T @ w_R + J_L.T @ w_L
        tau = pin.rnea(model, data, q, dq, ddq)
        saved_torques[i, :] = tau - tau_ext                               # MINUS (ground-on-foot)
        if return_gravity:
            gravity[i, :] = pin.computeGeneralizedGravity(model, data, q)

    grf["base_residual"] = saved_torques[:, :6]
    grf["states"] = (qf, dqf, ddqf)

    names = list(model.names[2:])          # drop 'universe' + free-flyer root
    tj = saved_torques[:, 6:]              # drop the 6 base DOFs (nv side)
    log_torque = {n: tj[:, k].tolist() for k, n in enumerate(names)}
    if return_gravity:
        gj = gravity[:, 6:]
        log_tau_g = {n: gj[:, k].tolist() for k, n in enumerate(names)}
        return log_torque, grf, log_tau_g
    return log_torque, grf