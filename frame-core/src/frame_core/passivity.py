"""
Gravity share and direction-reversal metrics -- exo-independent requirements, per repetition.

  * gravity share      G = RMS(tau_g) / RMS(tau)
        How much of the joint torque is due to gravity. G ~ 1: the joint mostly holds the arm
        against gravity (a spring or cam could target it); low: inertia or external load dominate.
        tau_g = g(q) from the inverse dynamics (frame_kinematics, `gravity_torques`); without it
        G is NaN with the reason "no gravity torque provided".

  * velocity reversals and their speed
        n_reversals          sign changes of qdot outside a deadband (default 5 deg/s), at least
                             `reversal_min_sep_s` (0.1 s) apart
        latency_med/min_s    time between the |qdot| peaks that bracket each reversal
        alpha_req_med/max    required angular acceleration (|w_before| + |w_after|) / latency
        peak_acc_rad_s2      measured peak |qddot|
        Frequent, fast reversals call for responsive actuation (bandwidth, low inertia).

All computed per repetition at the native rate on the human joint torque / angle; nothing here
reads a time-normalised curve.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .signal import derivative
from .joint_space import crossing_indices


@dataclass
class PassivityConfig:
    """Settings of the reversal metrics."""
    reversal_deadband_dps: float = 5.0
    reversal_min_sep_s: float = 0.1


def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def gravity_share(tau, tau_g):
    """G = RMS(tau_g) / RMS(tau). (NaN, reason) when tau_g is absent or tau is zero."""
    if tau_g is None:
        return float("nan"), "no gravity torque provided"
    den = _rms(tau)
    if not np.isfinite(den) or den <= 1e-12:
        return float("nan"), "zero torque"
    return _rms(tau_g) / den, ""


def reversal_metrics(omega, fs, deadband_dps=5.0, min_sep_s=0.1):
    """Velocity reversals with deadband + minimum separation, and the reversal latencies.

    Reuses `joint_space.crossing_indices` (frac=0, floor=deadband) -- a crossing counts only once
    |qdot| has committed past the deadband on both sides -- then drops crossings closer than
    `min_sep_s` to the previous kept one.
    """
    w = np.asarray(omega, float)
    acc = np.gradient(w, 1.0 / fs) if w.size > 1 else np.zeros_like(w)
    out = {"n_reversals": 0.0, "latency_med_s": float("nan"), "latency_min_s": float("nan"),
           "alpha_req_med_rad_s2": float("nan"), "alpha_req_max_rad_s2": float("nan"),
           "peak_acc_rad_s2": (float(np.nanmax(np.abs(acc))) if np.isfinite(acc).any()
                               else float("nan"))}
    idx, _ = crossing_indices(np.rad2deg(w), frac=0.0, floor_dps=deadband_dps)
    kept = []
    for i in idx:
        if not kept or (i - kept[-1]) / fs >= min_sep_s:
            kept.append(int(i))
    out["n_reversals"] = float(len(kept))
    if not kept:
        return out
    bounds = [0] + kept + [w.size]
    lat, alpha = [], []
    for k, c in enumerate(kept):
        a, b = bounds[k], bounds[k + 2]
        if c - a < 1 or b - c < 1:
            continue
        seg0, seg1 = np.abs(w[a:c]), np.abs(w[c:b])
        if not (np.isfinite(seg0).any() and np.isfinite(seg1).any()):
            continue
        i0 = a + int(np.nanargmax(seg0))
        i1 = c + int(np.nanargmax(seg1))
        dt = (i1 - i0) / fs
        if dt <= 0:
            continue
        lat.append(dt)
        alpha.append((abs(w[i0]) + abs(w[i1])) / dt)
    if lat:
        out.update({"latency_med_s": float(np.median(lat)), "latency_min_s": float(np.min(lat)),
                    "alpha_req_med_rad_s2": float(np.median(alpha)),
                    "alpha_req_max_rad_s2": float(np.max(alpha))})
    return out


METRIC_KEYS = ("G", "n_reversals", "latency_med_s", "latency_min_s", "alpha_req_med_rad_s2",
               "alpha_req_max_rad_s2", "peak_acc_rad_s2")


def rep_metrics(q, tau, tau_g, fs, cfg=None):
    """G and the reversal metrics of ONE repetition of ONE DOF (native rate, SI)."""
    cfg = cfg or PassivityConfig()
    G, reason = gravity_share(tau, tau_g)
    out = {"G": G, "_G_reason": reason}
    out.update(reversal_metrics(derivative(np.asarray(q, float), fs), fs,
                                cfg.reversal_deadband_dps, cfg.reversal_min_sep_s))
    return out


def compute(cap, cfg):
    """Per-capture layer: {"per_dof": {dof: {metric: summary}}}. Requires cap.has_torque."""
    from .types import summarize
    pcfg = getattr(cfg, "passivity", None) or PassivityConfig()
    per_dof = {}
    for j, name in enumerate(cap.dof_names):
        reps = [rep_metrics(r.angles[:, j], r.torques[:, j],
                            None if r.gravity_torques is None else r.gravity_torques[:, j],
                            cap.fs, pcfg) for r in cap.reps]
        reasons = {p.pop("_G_reason") for p in reps} - {""}
        entry = {k: summarize([p[k] for p in reps]) for k in METRIC_KEYS}
        entry["G_reason"] = "; ".join(sorted(reasons))
        per_dof[name] = entry
    return {"per_dof": per_dof, "has_gravity": bool(cap.has_gravity)}
