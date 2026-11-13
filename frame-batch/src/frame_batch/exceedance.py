"""
Torque exceedance -- how often, and for how long, a joint's torque stays above a fraction of its
capacity (default 15%, `BatchConfig.capacity_fraction`).

Per trial and DOF, with thr = fraction x URDF effort:

    exceed<P>_count       number of excursions: maximal runs of samples with |tau| > thr
                          (runs shorter than `exceedance_min_duration_s` are not counted)
    exceed<P>_mean_dur_s  mean duration of those excursions [s] (NaN with none)
    exceed<P>_time_frac   fraction of the recording spent above thr

<P> is the percentage (15). Computed at REPORT time from the cached Captures (native rate), and
merged into each trial's metric row before pooling, so it reaches trials.csv, pooled.json and the
ranked differences without re-running IK. Changing the fraction costs one `report`.

The box figure draws the count; the number in brackets beside each box is the POOLED mean
duration over every excursion of that group (total time above / total count), which weights each
excursion equally rather than each recording.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

FIGURE_NAME = "torque_exceedance_by_task"


def runs_above(x, thr):
    """(starts, lengths) of the maximal runs where x > thr."""
    m = np.asarray(x, float) > thr
    if not m.any():
        return np.empty(0, int), np.empty(0, int)
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def trial_exceedance(tau, effort, fs, fraction=0.15, min_duration_s=0.0):
    """{count, mean_dur_s, time_frac, total_s} for one torque trace."""
    t = np.abs(np.asarray(tau, float))
    if not effort or not np.isfinite(effort) or effort <= 0 or t.size == 0:
        return None
    _, lengths = runs_above(t, fraction * float(effort))
    dur = lengths / float(fs)
    dur = dur[dur >= float(min_duration_s)] if min_duration_s else dur
    return {"count": float(dur.size),
            "mean_dur_s": float(dur.mean()) if dur.size else float("nan"),
            "time_frac": float(dur.sum() * fs / t.size) if t.size else float("nan"),
            "total_s": float(dur.sum())}


def prefix(fraction):
    return f"exceed{100 * float(fraction):g}".replace(".", "p")


def augment(records, cache, cfg, dofs=None, progress=None):
    """Return copies of `records` with the exceedance metrics added to `metrics`.

    Records without a cached capture are returned unchanged (and counted in the second value).
    """
    from .captures import load

    frac = float(getattr(cfg, "capacity_fraction", 0.15))
    mind = float(getattr(cfg, "exceedance_min_duration_s", 0.0) or 0.0)
    pre = prefix(frac)
    dofs = set(dofs) if dofs is not None else set(cfg.selected_dofs())
    out, missing = [], 0
    for rec in records:
        cap = load(cache, rec, cfg)
        if cap is None:
            missing += 1
            out.append(rec)
            continue
        if not cap.has_torque:
            out.append(rec)
            continue
        lim = cap.urdf_limits or {}
        new = dict(rec)
        m = dict(rec.get("metrics") or {})
        for j, d in enumerate(cap.dof_names):
            if d not in dofs:
                continue
            e = (lim.get(d) or {}).get("effort") if isinstance(lim.get(d), dict) else None
            tau = np.concatenate([r.torques[:, j] for r in cap.reps])
            x = trial_exceedance(tau, e, cap.fs, frac, mind)
            if x is None:
                continue
            m[f"dynamics.{d}.{pre}_count"] = x["count"]
            m[f"dynamics.{d}.{pre}_mean_dur_s"] = x["mean_dur_s"]
            m[f"dynamics.{d}.{pre}_time_frac"] = x["time_frac"]
            m[f"dynamics.{d}.{pre}_total_s"] = x["total_s"]
        new["metrics"] = m
        out.append(new)
        if progress:
            progress(rec.get("trial_id"))
    return out, missing


def pooled_mean_durations(pooled, pre):
    """{(group, dof): total time above / total excursions} over every trial of the group."""
    out = {}
    for g, entry in (pooled or {}).items():
        mets = entry.get("metrics", {})
        for key, blk in mets.items():
            if not key.endswith(f".{pre}_count"):
                continue
            dof = key.split(".")[1]
            tot = mets.get(f"dynamics.{dof}.{pre}_total_s")
            n = float(np.nansum(blk.get("values") or []))
            s = float(np.nansum((tot or {}).get("values") or []))
            out[(g, dof)] = s / n if n > 0 else float("nan")
    return out


def build_figure(pooled, cfg, out_dir):
    """Box chart of the excursion count per DOF and group, mean duration in brackets."""
    from .plots import per_dof_figure
    frac = float(getattr(cfg, "capacity_fraction", 0.15))
    pre = prefix(frac)
    notes = {k: (f"({v:.2f} s)" if np.isfinite(v) else "(–)")
             for k, v in pooled_mean_durations(pooled, pre).items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return per_dof_figure(
        pooled, cfg.figure_path(out_dir, FIGURE_NAME), metric_suffix=f"{pre}_count",
        layer="dynamics", dof_pattern=cfg.dof_regex(fallback=cfg.plot_dofs),
        ylabel=f"excursions above {100 * frac:g}% of torque capacity [count per recording]",
        title=(f"How often |τ| exceeds {100 * frac:g}% of the joint's torque capacity "
               f"(mean excursion duration in brackets)"),
        max_points=getattr(cfg, "box_max_points", None), box_notes=notes)
