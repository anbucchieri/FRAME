"""
Layer 2 -- joint-space requirements.

Per-DOF, per-repetition:
  * ROM              : angle range (max - min) [deg]; also % of the URDF limit span if given.
  * peak |velocity|  : from filtered, native-rate differentiation [deg/s].
  * angular path     : integral |omega| dt [rad] -- captures repetitive small motions ROM misses.
  * reversals        : direction changes (velocity zero-crossings) per rep -> duty cycle.

Involvement is a WEIGHTED MENU of four candidate definitions, blended into one score. Each
component is normalised to [0, 1] across DOFs (busiest DOF = 1). The blend uses only the
terms whose data is available; missing terms (no Jacobian, no torques) drop out and the
remaining weights renormalise.
"""
from __future__ import annotations
from functools import partial
import numpy as np
from .signal import derivative
from .types import summarize, limit_entry

_TERMS = ("rom", "path", "jacobian", "dynamic")


def _nanmedian0(a):
    """Column medians over reps, NaN (silently) for a column with no defined value."""
    a = np.asarray(a, float)
    out = np.full(a.shape[1], np.nan)
    ok = np.isfinite(a).any(axis=0)
    out[ok] = np.nanmedian(a[:, ok], axis=0)
    return out


def _nan_or(f, x):
    """f() if x has any finite sample, else NaN -- nan-reductions warn on all-NaN input."""
    return f() if np.isfinite(np.asarray(x, float)).any() else float("nan")


def _per_rep_dof(cap, fn):
    """Apply fn(angles_col, torque_col_or_None, fs) -> scalar for every (rep, dof).
    Returns (n_rep, n_dof)."""
    out = np.full((len(cap.reps), cap.n_dof), np.nan)
    for i, r in enumerate(cap.reps):
        for j in range(cap.n_dof):
            tau = None if r.torques is None else r.torques[:, j]
            out[i, j] = fn(r.angles[:, j], tau, cap.fs)
    return out


def rom_deg(q_rad, tau, fs):
    q = np.rad2deg(np.asarray(q_rad, float))
    if not np.isfinite(q).any():
        return float("nan")
    return float(np.nanmax(q) - np.nanmin(q))


def peak_vel_dps(q_rad, tau, fs):
    w = derivative(q_rad, fs)               # rad/s, filtered
    return _nan_or(lambda: float(np.rad2deg(np.nanmax(np.abs(w)))), w)


def rms_vel_dps(q_rad, tau, fs):
    """Root-mean-square joint velocity [deg/s] -- the CONTINUOUS speed demand.

    `peak_vel_dps` sizes a drive's top speed; this sizes what it does all day. The two can differ
    by an order of magnitude on a task that is mostly slow with brief fast transitions, which is
    exactly the shape of a tracing task.
    """
    w = derivative(q_rad, fs)               # rad/s, filtered
    return _nan_or(lambda: float(np.rad2deg(np.sqrt(np.nanmean(w ** 2)))), w)


def angular_path_rad(q_rad, tau, fs):
    w = derivative(q_rad, fs)               # rad/s
    return float(np.nansum(np.abs(w)) / fs)    # integral |omega| dt (over defined samples)


def crossing_indices(w_dps, frac=0.05, floor_dps=2.0):
    """Sample indices of the accepted zero crossings of omega, plus the threshold used.

    A crossing is accepted only when omega commits past +thr and later past -thr (or vice versa).
    The index returned is the true sign change between those two committed excursions.
    """
    w = np.asarray(w_dps, float)
    if w.size < 2:
        return np.empty(0, int), 0.0
    if not np.isfinite(w).any():
        return np.empty(0, int), float(floor_dps)
    peak = float(np.nanmax(np.abs(w)))
    thr = max(float(floor_dps), float(frac) * peak)
    # thr == 0 is legitimate -- it means "no deadband", and must yield the RAW sign-change count,
    # not zero. Only a motionless signal, or one that never commits past the threshold, has none.
    if peak <= 0 or peak <= thr:
        return np.empty(0, int), thr

    committed = np.flatnonzero(np.abs(w) > thr)
    if committed.size < 2:
        return np.empty(0, int), thr
    signs = np.sign(w[committed])
    flips = np.flatnonzero(np.diff(signs) != 0)

    out = []
    for f in flips:
        a, b = committed[f], committed[f + 1]
        new = signs[f + 1]
        seg = np.sign(w[a:b + 1])
        k = np.flatnonzero(seg == new)          # first sample carrying the new sign
        out.append(a + int(k[0]) if k.size else b)
    return np.asarray(out, int), thr


def _omega_dps(q_rad, fs):
    return np.rad2deg(derivative(q_rad, fs))


def zero_crossings(q_rad, tau, fs, frac=0.05, floor_dps=2.0):
    """Number of accepted velocity zero crossings in this repetition."""
    idx, _ = crossing_indices(_omega_dps(q_rad, fs), frac, floor_dps)
    return float(idx.size)


def min_crossing_gap_s(q_rad, tau, fs, frac=0.05, floor_dps=2.0):
    """Shortest time between consecutive accepted crossings [s]; NaN with fewer than two.

    This is the fastest direction reversal the task demands of the joint -- the quantity that
    sizes actuator bandwidth, and the one a raw sign count destroys by reporting noise intervals
    of a single sample."""
    idx, _ = crossing_indices(_omega_dps(q_rad, fs), frac, floor_dps)
    if idx.size < 2:
        return float("nan")
    return float(np.min(np.diff(idx)) / fs)


def mean_crossing_gap_s(q_rad, tau, fs, frac=0.05, floor_dps=2.0):
    """Average time between consecutive direction changes [s]; NaN with fewer than two.

    `min_crossing_gap_s` is the worst case -- the fastest reversal the task ever demands, which is
    what sizes actuator bandwidth. This is the typical case: it says whether the joint reverses
    briskly all the time or drifts for seconds between changes.

    It is also the duration-INDEPENDENT companion to `reversals`. The raw count rises simply
    because a recording is longer (these run 25-65 s), so counts are only comparable between
    trials of similar length; the mean gap is comparable between any two.
    """
    idx, _ = crossing_indices(_omega_dps(q_rad, fs), frac, floor_dps)
    if idx.size < 2:
        return float("nan")
    return float(np.mean(np.diff(idx)) / fs)


def reversals(q_rad, tau, fs, frac=0.05, floor_dps=2.0):
    """Direction changes per repetition -- the deadbanded count (see the module docstring).

    Previously a raw sign(omega) change count, which counted the near-zero oscillation the spec
    explicitly excludes. Same quantity as `zero_crossings`, kept under its original name because
    the involvement menu and the flattened metric names refer to it."""
    return zero_crossings(q_rad, tau, fs, frac, floor_dps)


def dynamic_share(q_rad, tau, fs):
    """Per-rep dynamic involvement proxy: peak |mechanical power| = |tau * omega| [W].
    Returns 0 when torque is absent (term will be dropped upstream)."""
    if tau is None:
        return 0.0
    w = derivative(q_rad, fs)               # rad/s
    p = np.abs(np.asarray(tau, float) * w)
    return float(np.nanmax(p)) if np.isfinite(p).any() else 0.0


def _norm01(v):
    v = np.asarray(v, float)
    m = np.nanmax(v)
    return v / m if (m and np.isfinite(m) and m > 0) else np.zeros_like(v)


def involvement(cap, weights):
    """Blended per-DOF involvement in [0, 1], plus the raw components and the weights that
    were actually used (after dropping unavailable terms and renormalising)."""
    # per-DOF component vectors = median across reps of each per-rep quantity
    comp = {
        "rom": _nanmedian0(_per_rep_dof(cap, rom_deg)),
        "path": _nanmedian0(_per_rep_dof(cap, angular_path_rad)),
    }
    if cap.has_jac:
        comp["jacobian"] = np.nanmedian(
            np.vstack([r.jac_contribution for r in cap.reps]), axis=0)
    if cap.has_torque:
        comp["dynamic"] = _nanmedian0(_per_rep_dof(cap, dynamic_share))

    available = [t for t in _TERMS if t in comp]
    w = {t: float(weights.get(t, 0.0)) for t in available}
    wsum = sum(w.values())
    if wsum <= 0:                           # user zeroed everything -> equal weights
        w = {t: 1.0 / len(available) for t in available}
    else:
        w = {t: v / wsum for t, v in w.items()}

    comp01 = {t: _norm01(comp[t]) for t in available}
    score = np.zeros(cap.n_dof)
    for t in available:
        score += w[t] * comp01[t]

    order = np.argsort(score)[::-1]
    return {
        "dof_names": list(cap.dof_names),
        "score": score.tolist(),
        "ranking": [cap.dof_names[k] for k in order],
        "components_raw": {t: comp[t].tolist() for t in available},
        "components_norm": {t: comp01[t].tolist() for t in available},
        "weights_used": w,
        "terms_available": available,
        "terms_dropped": [t for t in _TERMS if t not in available],
    }


def compute(cap, cfg, urdf_limits=None, urdf_vel_limits=None):
    """Layer-2 result dict. urdf_limits: {dof: (lo_rad, hi_rad)}; urdf_vel_limits: {dof: rad/s}."""
    frac = getattr(cfg, "speed_deadband_frac", 0.05)
    floor = getattr(cfg, "speed_deadband_floor_dps", 2.0)
    deadband = {"frac": frac, "floor_dps": floor}

    rom = _per_rep_dof(cap, rom_deg)
    vel = _per_rep_dof(cap, peak_vel_dps)
    vrms = _per_rep_dof(cap, rms_vel_dps)
    path = _per_rep_dof(cap, angular_path_rad)
    rev = _per_rep_dof(cap, partial(reversals, frac=frac, floor_dps=floor))
    xing = _per_rep_dof(cap, partial(zero_crossings, frac=frac, floor_dps=floor))
    gap = _per_rep_dof(cap, partial(min_crossing_gap_s, frac=frac, floor_dps=floor))
    mgap = _per_rep_dof(cap, partial(mean_crossing_gap_s, frac=frac, floor_dps=floor))

    per_dof = {}
    for j, name in enumerate(cap.dof_names):
        entry = {
            "rom_deg": summarize(rom[:, j]),
            "peak_vel_dps": summarize(vel[:, j]),
            "rms_vel_dps": summarize(vrms[:, j]),
            "angular_path_rad": summarize(path[:, j]),
            "reversals": summarize(rev[:, j]),
            "zero_crossings": summarize(xing[:, j]),
            "min_crossing_gap_s": summarize(gap[:, j]),
            "mean_crossing_gap_s": summarize(mgap[:, j]),
            "max_speed_dps": float(np.nanmax(vel[:, j])) if np.any(np.isfinite(vel[:, j]))
                             else float("nan"),
        }
        lim = limit_entry(urdf_limits, name)
        if "lower" in lim and "upper" in lim:
            lo, hi = lim["lower"], lim["upper"]
            span = np.rad2deg(hi - lo)
            entry["rom_pct_of_limit"] = float(100.0 * entry["rom_deg"]["median"] / span) if span else float("nan")
            entry["urdf_limit_deg"] = [float(np.rad2deg(lo)), float(np.rad2deg(hi))]
        vlim = urdf_vel_limits.get(name) if urdf_vel_limits else lim.get("velocity")
        if vlim is not None:
            entry["urdf_vel_limit_dps"] = float(np.rad2deg(vlim))
        per_dof[name] = entry

    return {
        "per_dof": per_dof,
        "involvement": involvement(cap, cfg.involvement_weights),
        "speed_deadband": deadband,
    }
