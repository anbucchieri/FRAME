"""
grf_model.py
============
Ground-reaction-force (GRF) and centre-of-pressure (CoP) estimation for floating-base
inverse dynamics on motion-capture trials recorded WITHOUT force plates.

What is determined vs. what is modelled
----------------------------------------
The *total* GRF is NOT a free choice - Newton's 2nd law for the whole body fixes it:

        F_total = m_total * (a_com - g),     g = (0, 0, -9.81)

with a_com the whole-body CoM acceleration and m_total the total model mass. Both already
include the 5 kg load, because the load is part of the URDF. The only genuinely *modelled*
quantity is how F_total is shared between the two feet during double support (statically
indeterminate from kinematics alone). Here:

  * contact state per foot  : foot height above the floor  AND  foot speed, each thresholded
                              (direction-agnostic, so lateral or forward steps both work);
  * double-support split    : CoM ground-projection weighting along the inter-foot line;
  * CoP                      : mid-foot, i.e. the foot-frame origin projected to the floor.

Frames / convention
--------------------
force_* and moment_* are returned in WORLD axes, referenced to the FOOT-FRAME ORIGIN, i.e.
the LOCAL_WORLD_ALIGNED convention. Feed them to a foot Jacobian computed with
pin.LOCAL_WORLD_ALIGNED (see inverse_dynamics_grf.py).

    moment_foot = (cop - p_foot) x force          # the GRF moved from the CoP to the ankle
                                                   # (force is KEPT, not zeroed)

Returns (dict), every array shape (N, 3) unless noted
-----------------------------------------------------
    force_R, force_L      GRF on each foot                         [N]
    moment_R, moment_L    GRF moment about the foot-frame origin   [N.m]
    cop_R, cop_L          centre of pressure under each foot       [m]
    contact_R, contact_L  smoothed contact score in [0, 1]         (N,)
    F_total               total GRF from Newton                    (N, 3)
    a_com, com            CoM acceleration / position (world)      (N, 3)
    ground_z              estimated floor level (scalar)

CAVEAT on scale: a_com is taken from centerOfMass(q, dq, ddq) using the SAME (q, dq, ddq)
that feed rnea, so the GRF and the inverse dynamics stay mutually consistent. Its absolute
scale therefore inherits whatever dt/representation produced dq, ddq upstream. Verify it with
the quiet-stance check: at a still double-stance frame, F_total[:, 2] should equal
m_total * 9.81. Foot velocity for contact detection is computed geometrically (finite
difference of foot position at the capture rate), so it is independent of that convention.
"""

import numpy as np
import pinocchio as pin
from scipy.signal import butter, filtfilt

GRAVITY = np.array([0.0, 0.0, -9.81])


def _lowpass(x, cutoff, fs, order=4):
    x = np.asarray(x, float)
    if x.shape[0] <= 3 * (order + 1):
        return x
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, x, axis=0)


def compute_grf_cop(model, data, q, dq, ddq, contact_fs=240.0,
                    foot_frames=("right_foot", "left_foot"),
                    height_thr=0.04, speed_thr=0.30,
                    ground_pctile=2.0, contact_cutoff=8.0,
                    cop_forward_offset=0.0):
    """
    Parameters
    ----------
    model, data        : pinocchio model + data
    q, dq, ddq         : (N, nq), (N, nv), (N, nv) filtered states (the ones fed to rnea)
    contact_fs         : capture rate [Hz] used for the geometric foot-velocity estimate
    foot_frames        : (right, left) frame names in the URDF
    height_thr         : foot is "down" if (z - ground_z) < height_thr           [m]
    speed_thr          : foot is "still" if |v_foot| < speed_thr                 [m/s]
    ground_pctile      : floor level = this percentile of min(z_R, z_L) over the trial
    contact_cutoff     : low-pass [Hz] applied to the boolean contact to keep GRF continuous
    cop_forward_offset : optional shift of the CoP forward along the foot heading [m]
    """
    q = np.asarray(q, float)
    N = q.shape[0]
    fidR = model.getFrameId(foot_frames[0])
    fidL = model.getFrameId(foot_frames[1])
    m_total = pin.computeTotalMass(model)

    pR = np.zeros((N, 3)); pL = np.zeros((N, 3))
    RR = np.zeros((N, 3, 3)); RL = np.zeros((N, 3, 3))
    com = np.zeros((N, 3)); a_com = np.zeros((N, 3))

    for i in range(N):
        # CoM acceleration consistent with what rnea will see
        pin.centerOfMass(model, data, q[i], dq[i], ddq[i])
        com[i] = np.array(data.com[0])
        a_com[i] = np.array(data.acom[0])
        # foot placements (positions only -> geometric, convention-independent)
        pin.framesForwardKinematics(model, data, q[i])
        pR[i] = data.oMf[fidR].translation
        pL[i] = data.oMf[fidL].translation
        RR[i] = data.oMf[fidR].rotation
        RL[i] = data.oMf[fidL].rotation

    # ---- contact detection (height + geometric speed) ----
    ground_z = float(np.percentile(np.minimum(pR[:, 2], pL[:, 2]), ground_pctile))
    vR = np.gradient(pR, 1.0 / contact_fs, axis=0)
    vL = np.gradient(pL, 1.0 / contact_fs, axis=0)
    spdR = np.linalg.norm(vR, axis=1)
    spdL = np.linalg.norm(vL, axis=1)
    rawR = (pR[:, 2] - ground_z < height_thr) & (spdR < speed_thr)
    rawL = (pL[:, 2] - ground_z < height_thr) & (spdL < speed_thr)
    # smooth boolean -> continuous score so the force has no step discontinuities
    cR = np.clip(_lowpass(rawR.astype(float), contact_cutoff, contact_fs), 0.0, 1.0)
    cL = np.clip(_lowpass(rawL.astype(float), contact_cutoff, contact_fs), 0.0, 1.0)

    # ---- total GRF from Newton (load already in m_total) ----
    F_total = m_total * (a_com - GRAVITY)

    # ---- double-support split: project CoM onto the inter-foot line ----
    d = pL[:, :2] - pR[:, :2]                                   # R -> L, in the ground plane
    L2 = np.sum(d * d, axis=1)
    t = np.sum((com[:, :2] - pR[:, :2]) * d, axis=1) / np.maximum(L2, 1e-9)
    alpha_L = np.clip(t, 0.0, 1.0)                              # 0 at R foot, 1 at L foot

    # gate by contact, then normalise so force_R + force_L == F_total whenever grounded
    wL = cL * alpha_L
    wR = cR * (1.0 - alpha_L)
    s = wL + wR
    grounded = s > 1e-6
    wL_n = np.where(grounded, wL / np.maximum(s, 1e-9), 0.0)
    wR_n = np.where(grounded, wR / np.maximum(s, 1e-9), 0.0)
    force_L = wL_n[:, None] * F_total
    force_R = wR_n[:, None] * F_total

    # ---- CoP at mid-foot (foot origin projected to the floor, optional forward shift) ----
    cop_R = pR.copy(); cop_R[:, 2] = ground_z
    cop_L = pL.copy(); cop_L[:, 2] = ground_z
    if cop_forward_offset:
        fwdR = RR[:, :, 0].copy(); fwdR[:, 2] = 0.0
        fwdL = RL[:, :, 0].copy(); fwdL[:, 2] = 0.0
        fwdR /= np.maximum(np.linalg.norm(fwdR, axis=1, keepdims=True), 1e-9)
        fwdL /= np.maximum(np.linalg.norm(fwdL, axis=1, keepdims=True), 1e-9)
        cop_R += cop_forward_offset * fwdR
        cop_L += cop_forward_offset * fwdL

    # ---- move each GRF from the CoP to the foot-frame origin (force KEPT) ----
    moment_R = np.cross(cop_R - pR, force_R)
    moment_L = np.cross(cop_L - pL, force_L)

    return dict(
        force_R=force_R, force_L=force_L,
        moment_R=moment_R, moment_L=moment_L,
        cop_R=cop_R, cop_L=cop_L,
        contact_R=cR, contact_L=cL,
        F_total=F_total, a_com=a_com, com=com,
        ground_z=ground_z, m_total=m_total,
    )