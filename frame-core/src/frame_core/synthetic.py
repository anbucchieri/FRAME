"""
Synthetic capture generator -- lets you exercise the whole pipeline with zero real data and
zero Pinocchio. Produces a `Capture` of repeated reach-lift-lower cycles with plausible joint
angles, torques (a spring-like DOF, a motor-like DOF, and a near-idle DOF), and an end-effector
Cartesian trajectory. Used by the test suite and as a runnable smoke demo.
"""
from __future__ import annotations
import numpy as np
from .types import Capture, Repetition, Config


# A plausible upper-body skeleton + the matching URDF rows, for exercising the 3D scene and the
# capacity/limit metrics offline. Segment names are real Xsens labels so assemble.XSENS_BONES
# resolves against them; effort/limit values are the DHM URDF's own numbers for these joints.
_SCENE_SEGMENTS = ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
                   "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand"]
_SCENE_POSE = np.array([
    [0.00, 0.00, -0.52], [0.00, 0.00, -0.40], [0.00, 0.00, -0.28], [0.00, 0.00, -0.14],
    [0.00, 0.00, 0.00], [0.00, 0.00, 0.22], [0.00, 0.00, 0.36],
    [0.00, -0.06, 0.18], [0.00, -0.19, 0.16], [0.00, -0.22, -0.10], [0.00, -0.24, -0.34],
])
_SCENE_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                [4, 7], [7, 8], [8, 9], [9, 10]]
_SCENE_LIMITS = {
    "right_shoulder_Z": {"lower": -1.57, "upper": 3.14, "effort": 51.0, "velocity": 10.0},
    "right_shoulder_X": {"lower": -1.57, "upper": 2.40, "effort": 69.0, "velocity": 10.0},
    "right_shoulder_Y": {"lower": -1.57, "upper": 0.78, "effort": 90.0, "velocity": 10.0},
}


def make_scene():
    """Synthetic skeleton + URDF limits (see Capture.skeleton / Capture.urdf_limits)."""
    npose = _SCENE_POSE.copy()
    npose[7:, 2] = [0.18, 0.02, -0.24, -0.48]        # arm hanging down = neutral pose
    return {"segment_names": list(_SCENE_SEGMENTS), "bones": [list(b) for b in _SCENE_BONES],
            "pose_mean": _SCENE_POSE.copy(), "pose_npose": npose}, dict(_SCENE_LIMITS)


def make_capture(n_reps=12, fs=240.0, seed=0, with_torque=True, with_ee=True, with_jac=False,
                 with_scene=False):
    rng = np.random.default_rng(seed)
    dof_names = ["right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y"]  # Z=springy, X=motor, Y=idle
    reps = []
    ee_all = []
    for _ in range(n_reps):
        N = int(rng.integers(160, 260))         # variable-length reps
        t = np.linspace(0, 1, N)
        # a lift-then-lower angle profile (rad), with per-rep amplitude jitter
        a = np.deg2rad(90 + rng.normal(0, 6))
        q_z = a * np.sin(np.pi * t) + np.deg2rad(rng.normal(0, 2))            # 0->peak->0
        q_x = np.deg2rad(35) * (1 - np.cos(2 * np.pi * t)) / 2 + np.deg2rad(rng.normal(0, 2))
        q_y = np.deg2rad(10) * np.sin(4 * np.pi * t) + np.deg2rad(rng.normal(0, 1))
        angles = np.column_stack([q_z, q_x, q_y])

        torques = None
        if with_torque:
            # Z: torque ~ linear in angle -> spring-like (high R^2, small loop) -> PASSIVE
            tau_z = 8.0 * q_z + rng.normal(0, 0.3, N)
            # X: torque leads angle (hysteresis) + high level -> fat loop / high work -> ACTIVE
            om_x = np.gradient(q_x, 1.0 / fs)
            tau_x = 20.0 * q_x + 6.0 * om_x + rng.normal(0, 0.5, N)
            # Y: tiny torque -> below the free floor -> FREE
            tau_y = 0.6 * q_y + rng.normal(0, 0.1, N)
            torques = np.column_stack([tau_z, tau_x, tau_y])

        ee = None
        if with_ee:
            # end-effector traces an arc that rises during the lift (Z up), some XY sweep
            x = 0.30 + 0.12 * np.sin(np.pi * t) + rng.normal(0, 0.002, N)
            y = 0.05 * np.sin(2 * np.pi * t) + rng.normal(0, 0.002, N)
            z = 0.10 + 0.45 * np.sin(np.pi * t) + rng.normal(0, 0.002, N)     # peak height mid-rep
            ee = np.column_stack([x, y, z])
            ee_all.append(ee)

        jac = None
        if with_jac:
            jac = np.array([0.9, 0.6, 0.2]) * (1 + rng.normal(0, 0.05, 3))

        reps.append(Repetition(angles=angles, torques=torques, ee_pos=ee, jac_contribution=jac))

    skeleton, urdf_limits = make_scene() if with_scene else (None, None)
    cap = Capture(reps=reps, dof_names=dof_names, fs=fs,
                  ee_cloud=(np.vstack(ee_all) if ee_all else None),
                  height_m=1.78, mass_kg=74.0,
                  skeleton=skeleton, urdf_limits=urdf_limits)
    return cap


def default_config():
    cfg = Config()
    cfg.dof_groups = {
        "upper-limb": ["right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y"],
        "whole-body": ["right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y"],
    }
    return cfg
