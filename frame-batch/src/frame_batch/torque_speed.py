"""
Torque-speed operating plane per joint and task -- the demand an actuator has to cover.

One file per task (`torque_speed_by_task_<task>.<fmt>`). Each file is a DOF-type x side grid
(left `tab:blue`, right `tab:red`), and each panel is a log-density hexbin of every
(joint speed q̇ [rad/s], human joint torque τ [N.m]) sample of every recording of that task.
Speed is recomputed from the angles at the native rate, never read from a time-normalised curve.

Torque and speed with the same sign is positive power (the joint drives); opposite signs is
negative power (the joint brakes). The axes are drawn through zero for that reason.

The view is clipped to the central `torque_speed_clip_pct` % of samples so that the few
edge-ringing / IK spikes do not squash the cloud; the true extremes are the peak_torque /
peak_vel columns of trials.csv.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np

from frame_core.signal import derivative

from .plots import SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, _style, _order_groups

FIGURE_NAME = "torque_speed_by_task"
DPI = 300
SIDE_COLORS = {"left": "tab:blue", "right": "tab:red", "": "0.3"}


def select_dofs(dof_names, pattern):
    if not pattern:
        return list(dof_names)
    rx = re.compile(pattern)
    return [d for d in dof_names if rx.search(d)]


def _split(dof):
    m = re.match(r"^(left|right)_(.+)$", dof)
    return (m.group(1), m.group(2)) if m else ("", dof)


def dof_grid(dofs):
    """(rows = DOF types in first-seen order, cols = sides present) and {dof: (i, j)}."""
    types, sides = [], []
    for d in dofs:
        s, t = _split(d)
        if t not in types:
            types.append(t)
        if s not in sides:
            sides.append(s)
    order = [s for s in ("left", "right", "") if s in sides]
    pos = {d: (types.index(_split(d)[1]), order.index(_split(d)[0])) for d in dofs}
    return types, order, pos


def collect(records, cache, cfg, dofs, progress=None):
    """{task: {dof: {"omega": (N,) rad/s, "tau": (N,) N.m}}} from the cached Captures."""
    from .captures import load

    raw: dict = {}
    missing = 0
    for rec in records:
        cap = load(cache, rec, cfg)
        if cap is None:
            missing += 1
            continue
        index = {n: i for i, n in enumerate(cap.dof_names)}
        fs = float(cap.fs or 240.0)
        mass = float(rec.get("mass_kg") or 0.0)
        slot = raw.setdefault(rec.get("task"), {})
        for dof in dofs:
            j = index.get(dof)
            if j is None:
                continue
            for rep in cap.reps:
                if rep.torques is None:
                    continue
                w = derivative(np.asarray(rep.angles[:, j], float), fs)
                t = np.asarray(rep.torques[:, j], float)
                if cfg.torque_speed_normalise and mass > 0:
                    t = t / mass
                d = slot.setdefault(dof, {"omega": [], "tau": []})
                d["omega"].append(w)
                d["tau"].append(t)
        if progress:
            progress(rec.get("trial_id"))

    out: dict = {}
    for task, per_dof in raw.items():
        out[task] = {}
        for dof, d in per_dof.items():
            w, t = np.concatenate(d["omega"]), np.concatenate(d["tau"])
            m = np.isfinite(w) & np.isfinite(t)
            if m.any():
                out[task][dof] = {"omega": w[m], "tau": t[m]}
    return out, missing


def _robust_limits(v, pct, pad=0.08):
    """Limits set by the central `pct` % of samples, zero always in view."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, [100.0 - pct, pct])
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    if span <= 0:
        return None
    return lo - pad * span, hi + pad * span


def figure(clouds, dofs, task, out_path, normalised=False, clip_pct=99.5, panel=(3.4, 2.4)):
    """One task: a DOF-type x side grid of (q̇, τ) hexbins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dofs = [d for d in dofs if d in clouds]
    if not dofs:
        return None
    types, sides, pos = dof_grid(dofs)
    fig, axes = plt.subplots(len(types), len(sides),
                             figsize=(panel[0] * len(sides), panel[1] * len(types)),
                             facecolor=SURFACE, squeeze=False)
    for j, s in enumerate(sides):
        axes[0][j].annotate(s or "", xy=(0.5, 1.22), xycoords="axes fraction", ha="center",
                            fontsize=10, color=SIDE_COLORS[s], weight="bold")
    unit = "N·m/kg" if normalised else "N·m"
    used = set()
    for d in dofs:
        i, j = pos[d]
        used.add((i, j))
        ax = axes[i][j]
        c = clouds[d]
        xl, yl = _robust_limits(c["omega"], clip_pct), _robust_limits(c["tau"], clip_pct)
        extent = (xl + yl) if (xl and yl) else None
        ax.hexbin(c["omega"], c["tau"], gridsize=40, mincnt=1, bins="log", cmap="Greys",
                  linewidths=0, rasterized=True, extent=extent)
        if extent:
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_title(d, fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel("q̇ [rad/s]", fontsize=7, color=INK_SECONDARY)
        ax.set_ylabel(f"τ [{unit}]", fontsize=7, color=INK_SECONDARY)
        _style(ax)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if (i, j) not in used:
                ax.set_visible(False)
    fig.suptitle(f"Torque–speed operating plane — {task}", x=0.01, ha="left", fontsize=11,
                 color=INK, y=1.04)
    fig.text(0.01, -0.01,
             f"Hexbin: (q̇, τ) samples of every recording of the task (human joint torque, log "
             f"density). Same sign = the joint drives; opposite signs = it brakes. Axes clipped to "
             f"the central {clip_pct:g}% of samples.",
             fontsize=7, color=INK_MUTED, ha="left", va="top", wrap=True)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def build(records, cache, cfg, out_dir, progress=None):
    """One torque-speed figure per task. Returns (paths | None, note)."""
    dofs = select_dofs(cfg.analysed_dofs(), cfg.dof_regex(fallback=cfg.torque_speed_dofs))
    if not dofs:
        return None, f"no DOF matches torque_speed_dofs={cfg.torque_speed_dofs!r}"

    clouds, missing = collect(records, cache, cfg, dofs, progress=progress)
    if not clouds:
        return None, ("no cached Capture found -- the torque-speed figure needs "
                      "`cache_captures: true`.")

    out_dir = Path(out_dir)
    stem = Path(cfg.figure_path(out_dir, FIGURE_NAME))
    paths = []
    for task in _order_groups(list(clouds)):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(task))
        p = figure(clouds[task], dofs, task, stem.with_name(f"{stem.stem}_{safe}{stem.suffix}"),
                   normalised=cfg.torque_speed_normalise,
                   clip_pct=float(cfg.torque_speed_clip_pct))
        if p:
            paths.append(p)
    note = (f"{missing} of {len(records)} trials had no cached Capture and are missing from the "
            f"torque-speed planes") if missing else ""
    return (paths or None), note
