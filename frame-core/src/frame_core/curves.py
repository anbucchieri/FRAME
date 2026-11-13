"""
Time-normalised joint trajectories -- the curve behind the scalar.

Every other layer reduces a recording to numbers. This one keeps the shape: ONE curve per
recording per DOF, spanning the whole continuous movement (no segmentation), linearly
time-normalised so that recordings of different duration can be averaged sample-by-sample across
subjects and repetitions and drawn as a mean +/- SD band.

Two things it is careful about, both of which quietly corrupt curve averages otherwise:

  * **Velocity is differentiated at the NATIVE rate, then normalised** -- never the other way
    round. Differentiating a time-normalised curve gives degrees per percent-of-trial, which
    looks like a speed, is not one, and silently rescales every trial by its own duration.

  * **Resampling is linear interpolation** (`signal.resample_interp`), not FFT resampling, which
    rings at the endpoints of non-periodic trajectories and inflates exactly the peaks these
    curves are read for.

This is a PRESENTATION layer: it never feeds a spec scalar. Time-normalising throws
away duration, so a curve here answers "what shape is the movement", not "how big is the demand".
"""
from __future__ import annotations
import numpy as np

from .signal import resample_interp, derivative

# What each quantity is called and what unit its curve carries.
QUANTITIES = {
    "angle_deg": "joint angle [deg]",
    "speed_dps": "joint speed [deg/s]",
    "torque_Nm": "joint torque [N.m]",
}


def trial_curves(cap, dof_names=None, n_resample=500, quantities=None):
    """Time-normalised curves for one Capture: ONE curve per recording, covering all of it.

    Returns {dof: {quantity: (n_reps, n_resample)}}. Each repetition (a whole recording, in the
    pooled tool) is kept as a single continuous movement from its first to its last sample and
    linearly resampled to `n_resample` points -- it is never cut into pieces. A quantity the
    capture cannot supply (torque, on a kinematics-only run) is absent rather than full of NaN.
    """
    want = set(quantities or QUANTITIES)
    names = list(dof_names) if dof_names is not None else list(cap.dof_names)
    index = {n: i for i, n in enumerate(cap.dof_names)}
    fs = float(cap.fs or 240.0)

    out = {}
    for name in names:
        j = index.get(name)
        if j is None:
            continue
        per_q = {q: [] for q in want}
        for rep in cap.reps:
            if rep.n < 4:
                continue
            q_rad = np.asarray(rep.angles[:, j], float)
            if "angle_deg" in want:
                per_q["angle_deg"].append(resample_interp(np.rad2deg(q_rad), n_resample))
            if "speed_dps" in want:
                w_dps = np.rad2deg(derivative(q_rad, fs))      # at the native rate, then resampled
                per_q["speed_dps"].append(resample_interp(w_dps, n_resample))
            if "torque_Nm" in want and rep.torques is not None:
                tau = np.asarray(rep.torques[:, j], float)
                per_q["torque_Nm"].append(resample_interp(tau, n_resample))
        built = {q: np.vstack(v) for q, v in per_q.items() if v}
        if built:
            out[name] = built
    return out


def stack_curves(per_trial):
    """Merge many trials' curve dicts into one. Returns {dof: {quantity: (n_curves, n_resample)}}.

    Every recording in the group is one row, so the band is taken across all subjects and all
    repetitions at once rather than as a mean of per-subject means.
    """
    merged: dict = {}
    for curves in per_trial:
        for dof, qs in (curves or {}).items():
            slot = merged.setdefault(dof, {})
            for q, arr in qs.items():
                slot.setdefault(q, []).append(np.asarray(arr, float))
    return {dof: {q: np.vstack(v) for q, v in qs.items()} for dof, qs in merged.items()}


def band(curves, sd=1.0):
    """Mean +/- SD envelope of a (n_curves, n_resample) stack.

    Population SD (ddof=0): the curves in hand ARE the observed set for this task, not a sample
    drawn to estimate a wider population's spread. Returns None for an empty stack.
    """
    a = np.asarray(curves, float)
    if a.ndim != 2 or a.size == 0:
        return None
    # A curve may be partly undefined (the shoulder plane of elevation is masked near gimbal lock),
    # so mean and SD are taken per sample over the curves defined THERE; n counts curves with any
    # defined sample, and `n_at` is the per-sample count.
    a = a[np.isfinite(a).any(axis=1)]
    if a.shape[0] == 0:
        return None
    n_at = np.isfinite(a).sum(axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(a, axis=0)
            std = np.nanstd(a, axis=0, ddof=0)
    return {
        "mean": mean, "sd": std,
        "lo": mean - sd * std, "hi": mean + sd * std,
        "n": int(a.shape[0]), "n_at": n_at,
        "pct": np.linspace(0.0, 100.0, a.shape[1]),
    }


def bands(stacked, sd=1.0):
    """`band` applied across a whole {dof: {quantity: stack}} tree."""
    out = {}
    for dof, qs in (stacked or {}).items():
        built = {q: band(arr, sd) for q, arr in qs.items()}
        built = {q: b for q, b in built.items() if b is not None}
        if built:
            out[dof] = built
    return out
