"""
Mean +/- SD joint trajectories per task type, from the cached Captures.

The scalar figures answer "how big"; this one answers "what shape". For each selected DOF it
draws the joint angle, joint speed and joint torque against percent-of-trial, as a mean line with
a +/- 1 SD band, one band per task type.

It reads the per-trial `Capture` npz files the sweep already cached, so it costs seconds and no
IK. That also means it needs `cache_captures: true` (the default); with captures turned off the
figure is skipped with a message rather than silently omitted.

Every curve is ONE WHOLE RECORDING -- the full continuous movement, never segmented -- linearly
time-normalised to percent of trial and pooled into its task's band across all subjects and
repetitions. With
several subjects the band therefore mixes body sizes: angles and speeds are already
subject-independent, but TORQUE is not, so `timeseries_normalise_torque` switches the torque row
to N.m/kg. It is off by default because within one subject it only relabels the axis.
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import numpy as np

from frame_core.curves import trial_curves, stack_curves, bands, QUANTITIES

from .plots import (SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, SERIES,
                    group_colors, _order_groups, _style)

ROW_ORDER = ("angle_deg", "speed_dps", "torque_Nm")


def select_dofs(dof_names, pattern):
    """The DOFs a pattern selects, in the config's order (not the regex's or a dict's)."""
    if not pattern:
        return list(dof_names)
    rx = re.compile(pattern)
    return [d for d in dof_names if rx.search(d)]


def collect(records, cache, cfg, dofs=None, progress=None):
    """{task: {dof: {quantity: band}}} from the cached Captures of these trial records.

    Returns (bands_by_group, n_missing). `n_missing` counts records whose Capture was not cached
    -- reported rather than swallowed, because a band quietly built from half the trials is worse
    than no band.
    """
    from .captures import load

    dofs = dofs if dofs is not None else select_dofs(
        cfg.analysed_dofs(), cfg.dof_regex(fallback=cfg.timeseries_dofs))
    n = int(cfg.timeseries_n_resample)

    per_group: dict = {}
    missing = 0
    for rec in records:
        cap = load(cache, rec, cfg)
        if cap is None:
            missing += 1
            continue
        curves = trial_curves(cap, dofs, n_resample=n)
        if cfg.timeseries_normalise_torque and rec.get("mass_kg"):
            m = float(rec["mass_kg"])
            for qs in curves.values():
                if "torque_Nm" in qs:
                    qs["torque_Nm"] = qs["torque_Nm"] / m
        per_group.setdefault(rec.get("task"), []).append(curves)
        if progress:
            progress(rec.get("trial_id"))

    return ({g: bands(stack_curves(v)) for g, v in per_group.items()}, missing)


def _ylabel(quantity, normalised):
    if quantity == "torque_Nm" and normalised:
        return "joint torque [N.m/kg]"
    return QUANTITIES.get(quantity, quantity)


# Which joint capacity each row is read against: (limit keys, legend label).
#   angle -> the URDF range-of-motion bounds; speed -> +/- velocity capacity; torque -> +/- effort.
CAPACITY_ROWS = {
    "angle_deg": (("lower_deg", "upper_deg"), "ROM limits"),
    "speed_dps": (("velocity_dps",), "speed capacity"),
    "torque_Nm": (("effort",), "torque capacity"),
}


def _capacity_levels(limits, dof, quantity, normalised=False):
    """The horizontal capacity lines for one panel, as a list of y values (may be empty)."""
    keys, _ = CAPACITY_ROWS.get(quantity, ((), ""))
    e = (limits or {}).get(dof) or {}
    if quantity == "torque_Nm" and normalised:
        return []                      # an N.m capacity cannot be drawn on an N.m/kg axis
    if quantity == "angle_deg":
        return [e[k] for k in keys if e.get(k) is not None]
    v = e.get(keys[0])
    return [v, -v] if v else []


# A capacity further than this multiple of the data's own extent is not drawn as a line -- it
# would compress the curves into a flat band -- but written at the panel edge with an arrow.
CAPACITY_HEADROOM = 1.6


def _data_extent(bands_by_group, groups, dof, quantity):
    """(min, max) of every group's mean +/- SD band for one panel; NaN if none."""
    lo, hi = np.inf, -np.inf
    for g in groups:
        b = ((bands_by_group.get(g) or {}).get(dof) or {}).get(quantity)
        if b is None:
            continue
        lo = min(lo, float(np.nanmin(b["lo"])))
        hi = max(hi, float(np.nanmax(b["hi"])))
    return (lo, hi) if np.isfinite(lo) else (float("nan"), float("nan"))


def draw_panel(ax, bands_by_group, groups, colors, dof, quantity, limits=None, normalised=False,
               capacity_fraction=None, label_lines=False):
    """One (DOF, quantity) panel: a mean +/- SD band per group, the zero line, the capacity lines.

    Returns False when no group has this curve (the caller hides the axes). `label_lines` gives the
    bands legend labels, which the caller collects once for the whole figure."""
    drew = False
    for g in groups:
        b = ((bands_by_group.get(g) or {}).get(dof) or {}).get(quantity)
        if b is None:
            continue
        ax.fill_between(b["pct"], b["lo"], b["hi"], color=colors[g], alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(b["pct"], b["mean"], color=colors[g], linewidth=2.0, zorder=3,
                solid_capstyle="round", label=f"{g}  (n={b['n']})" if label_lines else None)
        drew = True
    if not drew:
        return False
    # a zero line matters for speed and torque, where the sign is the story
    if quantity in ("speed_dps", "torque_Nm"):
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, alpha=0.5, zorder=1)
    # The joint's capacity. Range-of-motion limits share the angle's scale and are always
    # drawn. Speed and torque capacities usually sit far beyond the data: drawn as lines
    # they flatten the curves, so beyond CAPACITY_HEADROOM x the data's extent they are
    # written at the panel edge instead and the axis stays fitted to the data.
    lo_d, hi_d = _data_extent(bands_by_group, groups, dof, quantity)
    span = max(abs(lo_d), abs(hi_d)) if np.isfinite(lo_d) else 0.0
    levels = _capacity_levels(limits, dof, quantity, normalised)
    off_axis = []
    for y in levels:
        if quantity == "angle_deg" or not span or abs(y) <= CAPACITY_HEADROOM * span:
            ax.axhline(y, color=INK, linewidth=1.2, linestyle=(0, (4, 2.5)), zorder=4)
            ax.annotate(f"{y:.4g}", xy=(100, y), xytext=(-2, 2),
                        textcoords="offset points", fontsize=6.5, color=INK,
                        ha="right", va="bottom")
        else:
            off_axis.append(y)
    lo_v, hi_v = lo_d, hi_d
    if quantity == "torque_Nm" and capacity_fraction and not normalised:
        e = ((limits or {}).get(dof) or {}).get("effort")
        if e:
            fe = capacity_fraction * e
            for y in (fe, -fe):
                ax.axhline(y, color=SERIES[6], linewidth=1.1,
                           linestyle=(0, (1.5, 1.5)), zorder=4)
            ax.annotate(f"{100 * capacity_fraction:g}%: \u00b1{fe:.3g}",
                        xy=(0, fe), xytext=(2, 2), textcoords="offset points",
                        fontsize=6.3, color=SERIES[6], ha="left", va="bottom")
            lo_v, hi_v = min(lo_v, -fe), max(hi_v, fe)
    if off_axis and np.isfinite(lo_v):
        pad = 0.08 * max(hi_v - lo_v, 1e-9)
        ax.set_ylim(lo_v - pad, hi_v + 2.2 * pad)          # room for the edge label
        unit = {"speed_dps": "deg/s", "torque_Nm": "N.m"}.get(quantity, "")
        ax.text(0.99, 0.985, f"capacity \u00b1{max(abs(v) for v in off_axis):.4g} {unit} \u2191",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=GRID, lw=0.6))
    return True


def figure(bands_by_group, dofs, out_path, normalised=False, n_resample=500, limits=None,
           capacity_fraction=None,
           title="Joint trajectories by task type — mean \u00b11 SD over trials"):
    """Rows = angle / speed / torque, columns = DOF, one mean+SD band per task.

    Grouped this way because the question the figure answers is "how does this joint's trajectory
    differ between H, M and L" -- which needs the three bands on one pair of axes, sharing a scale.
    Faceting by task instead would put the comparison across a gutter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _order_groups(list(bands_by_group))
    colors = group_colors(groups)

    rows = [q for q in ROW_ORDER
            if any(q in (bands_by_group.get(g) or {}).get(d, {}) for g in groups for d in dofs)]
    cols = [d for d in dofs
            if any(d in (bands_by_group.get(g) or {}) for g in groups)]
    if not rows or not cols:
        return None

    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.05 * len(cols) + 0.8, 2.55 * len(rows) + 0.9),
                             facecolor=SURFACE, squeeze=False, sharex=True)

    for ri, quantity in enumerate(rows):
        for ci, dof in enumerate(cols):
            ax = axes[ri][ci]
            if not draw_panel(ax, bands_by_group, groups, colors, dof, quantity, limits=limits,
                              normalised=normalised, capacity_fraction=capacity_fraction,
                              label_lines=(ri == 0 and ci == 0)):
                ax.set_visible(False)
                continue
            _style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            if ri == 0:
                ax.set_title(dof, fontsize=9, color=INK, loc="left", pad=6)
            if ci == 0:
                ax.set_ylabel(_ylabel(quantity, normalised), fontsize=8.5, color=INK_SECONDARY)
            if ri == len(rows) - 1:
                ax.set_xlabel("% of trial", fontsize=8.5, color=INK_MUTED)
            ax.set_xlim(0, 100)

    fig.suptitle(title, fontsize=12.5, color=INK, x=0.008, ha="left", y=0.996)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if limits:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color=INK, lw=1.2, ls=(0, (4, 2.5))))
        labels.append("capacity (ROM / speed / torque; off-axis values written in the panel)")
        if any(((limits or {}).get(d) or {}).get("effort") for d in dofs):
            handles.append(Line2D([], [], color=SERIES[6], lw=1.1, ls=(0, (1.5, 1.5))))
            labels.append(f"{100 * (capacity_fraction or 0.15):g}% of torque capacity")
    if handles:
        leg = fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(handles),
                         loc="upper right", bbox_to_anchor=(0.995, 1.004), title="task")
        leg.get_title().set_fontsize(8)
        leg.get_title().set_color(INK_MUTED)
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    fig.text(0.008, 0.004,
             f"line = mean over every trial of every subject (n in the legend); "
             f"shaded = \u00b11 SD. Each trial (a whole recording, or one pick-to-place movement) "
             f"is linearly time-normalised to {n_resample} samples, so the x-axis is percent of "
             f"trial.",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


FIGURE_NAME = "joint_timeseries_by_task"


def build(records, cache, cfg, out_dir, progress=None):
    """The whole time-series figure. Returns (path | None, note) -- `note` explains a None."""
    dofs = select_dofs(cfg.analysed_dofs(), cfg.dof_regex(fallback=cfg.timeseries_dofs))
    if not dofs:
        return None, (f"no DOF matches timeseries_dofs={cfg.timeseries_dofs!r}")

    bands_by_group, missing = collect(records, cache, cfg, dofs=dofs, progress=progress)
    if not bands_by_group:
        return None, ("no cached Capture found -- the time-series figure needs "
                      "`cache_captures: true` (it reads the per-trial .npz the sweep stored). "
                      "Re-run the sweep with it enabled to get these curves.")

    # Own the directory rather than inheriting it from whichever figure ran first: the scalar
    # figures are allowed to fail without taking this one down with them.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.figure_path(out_dir, "joint_timeseries_by_task")
    try:
        from . import limits as limits_mod
        lims = limits_mod.collect(records, cache, dofs=set(dofs), cfg=cfg)
    except Exception:
        lims = None
    path = figure(bands_by_group, dofs, out, normalised=cfg.timeseries_normalise_torque,
                  n_resample=int(cfg.timeseries_n_resample), limits=lims,
                  capacity_fraction=getattr(cfg, "capacity_fraction", None))
    note = ""
    if missing:
        note = (f"{missing} of {len(records)} trials had no cached Capture and are missing from "
                f"the time-series bands")
    return path, note
