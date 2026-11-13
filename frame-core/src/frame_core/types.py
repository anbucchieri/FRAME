"""
Core data model.

A `Capture` is one recording. In the pooled tool it holds exactly one `Repetition`: the whole
continuous movement, never segmented. Metrics are computed per repetition; the distributions the
tool reports are taken ACROSS recordings (see pooling).

Data availability is graceful:
  * `angles`   (rad) : always present (IK output).
  * `torques`  (N.m) : optional -> gates Layers 3-4.
  * `gravity_torques` (N.m) : optional gravity part g(q) of `torques` -> the gravity share G
                       (reported NaN without it).
  * `ee_pos`   (m)   : end-effector Cartesian trajectory in the trunk frame -> gates Layer 1.
  * `jac_contribution` : optional per-DOF scalar -> enables the Jacobian involvement term
                       (the term drops out when it is absent).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Repetition:
    """One continuous movement (a whole recording). Arrays are native-rate."""
    angles: np.ndarray                      # (N, nDOF) rad
    torques: Optional[np.ndarray] = None    # (N, nDOF) N.m
    gravity_torques: Optional[np.ndarray] = None   # (N, nDOF) N.m, g(q) on the ID's filtered q
    ee_pos: Optional[np.ndarray] = None     # (N, 3) m, analysis frame
    jac_contribution: Optional[np.ndarray] = None  # (nDOF,) unitless

    @property
    def n(self) -> int:
        return self.angles.shape[0]


@dataclass
class Capture:
    """A whole recording: shared metadata + a list of repetitions."""
    reps: list                              # list[Repetition]
    dof_names: list                         # list[str], length nDOF
    fs: float = 240.0
    ee_cloud: Optional[np.ndarray] = None   # (M, 3) pooled raw ee points across reps
    exo_cloud: Optional[np.ndarray] = None  # (K, 3) exo tip reachable cloud (assessment)
    height_m: Optional[float] = None
    mass_kg: Optional[float] = None
    # --- presentation only (the N-pose torque map); never touches a metric --- #
    skeleton: Optional[dict] = None         # see build_skeleton(); all poses in the trunk frame:
                                            #   segment_names : list[str], length S
                                            #   bones         : list[[i, j]] index pairs into it
                                            #   pose_mean     : (S, 3) median measured posture
                                            #   pose_npose    : (S, 3) | None  neutral (N-pose)
    urdf_limits: Optional[dict] = None      # {joint: {"lower","upper","effort","velocity"}} SI:
                                            # rad, rad, N.m, rad/s. `effort` is the joint's maximum
                                            # torque capacity -> Layer-3 capacity ratios.

    @property
    def n_dof(self) -> int:
        return len(self.dof_names)

    @property
    def has_torque(self) -> bool:
        return all(r.torques is not None for r in self.reps) and len(self.reps) > 0

    @property
    def has_gravity(self) -> bool:
        return all(r.gravity_torques is not None for r in self.reps) and len(self.reps) > 0

    @property
    def has_ee(self) -> bool:
        return all(r.ee_pos is not None for r in self.reps) and len(self.reps) > 0

    @property
    def has_jac(self) -> bool:
        return all(r.jac_contribution is not None for r in self.reps) and len(self.reps) > 0


def _passivity_config():
    from .passivity import PassivityConfig
    return PassivityConfig()


@dataclass
class Config:
    """Metric-engine settings. Defaults reproduce the paper's baseline."""
    ee_segment: str = "RightForeArm"
    voxel_m: float = 0.02
    # Speed-metric deadband (see joint_space): a crossing counts only once |omega| commits past
    # thr = max(floor_dps, frac * max|omega|) in one direction and then the other.
    speed_deadband_frac: float = 0.05
    speed_deadband_floor_dps: float = 2.0
    involvement_weights: dict = field(default_factory=lambda: {
        "rom": 0.25, "path": 0.25, "jacobian": 0.25, "dynamic": 0.25,
    })
    dof_groups: dict = field(default_factory=dict)  # group_name -> list[dof_name]
    passivity: object = field(default_factory=_passivity_config)            # PassivityConfig
    fs: float = 240.0


def limit_entry(limits, name):
    """One joint's limits as a dict, from either shape the pipeline may be handed.

    The canonical shape is {joint: {"lower","upper","effort","velocity"}} (what introspect_model
    produces). Older callers pass {joint: (lo_rad, hi_rad)} because that is all joint_space needed.
    Accepting both at the point of use keeps one parameter name meaning one thing -- the
    alternative was dynamics.compute receiving a tuple and failing on .get("effort").
    """
    e = (limits or {}).get(name)
    if e is None:
        return {}
    if isinstance(e, dict):
        return e
    try:
        lo, hi = e
        return {"lower": float(lo), "upper": float(hi)}
    except (TypeError, ValueError):
        return {}


def summarize(x) -> dict:
    """Distribution summary of a per-rep (or per-sample) scalar array."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("median", "iqr_lo", "iqr_hi", "p05", "p95",
                                          "min", "max", "mean", "std", "n")}
    return {
        "median": float(np.median(x)),
        "iqr_lo": float(np.percentile(x, 25)),
        "iqr_hi": float(np.percentile(x, 75)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "n": int(x.size),
    }
