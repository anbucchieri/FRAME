"""
Distribution figures -- the debugging view of a pooled run.

Every figure answers the same question in a different slice: *how does this metric distribute
within a task type, and does it differ between task types?* So every figure is built from the
same primitive -- a box for the median and IQR with the actual trial points drawn on top of it.

Showing the raw points is not decoration. With 24-36 trials per group, a box alone hides whether
a shifted median is a real shift or one outlier, and that distinction is exactly what a
requirements number lives or dies on. If a point looks wrong you can find the recording behind it
in `trials.csv`, because the pooled entry keeps its values and trial ids index-aligned.

Colour: three categorical slots (blue / orange / aqua), assigned to the task types in fixed sorted
order so a task keeps its colour across every figure and across runs. The aqua slot sits below 3:1
against the light surface, so each group is also labelled directly on the axis and its n is
printed -- identity is never carried by colour alone.
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import numpy as np

# --- design tokens (validated categorical slots 1-3; see references/palette.md) -------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a84"
GRID = "#e5e4e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Task types in a fixed order so H is always slot 1 regardless of dict iteration order. Anything
# outside this list is appended alphabetically -- a study with other labels still gets stable
# colours, it just does not get this study's preferred ordering.
PREFERRED_ORDER = ["H", "M", "L"]


def _order_groups(groups):
    """H, M, L first; compound labels ("H_N", "M_O") follow their task's slot, then alphabetical."""
    def key(g):
        head, _, tail = str(g).partition("_")
        return ((PREFERRED_ORDER.index(head), tail) if head in PREFERRED_ORDER
                else (len(PREFERRED_ORDER), str(g)))
    return sorted(groups, key=key)


def group_colors(groups):
    """{group: hex} -- assigned by entity, in fixed order, never by rank or by current count."""
    return {g: SERIES[i % len(SERIES)] for i, g in enumerate(_order_groups(groups))}


def _style(ax):
    """Recessive axes: horizontal grid only, no top/right spines, muted ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3, width=0.8)


def _short(metric):
    """`dynamics.right_shoulder_Z.peak_torque_Nm` -> `right_shoulder_Z\\npeak_torque_Nm`."""
    parts = metric.split(".")
    if len(parts) >= 3:
        return f"{parts[1]}\n{'.'.join(parts[2:])}"
    return metric.replace(".", "\n", 1)


def _values(pooled, group, metric):
    entry = (pooled.get(group) or {}).get("metrics", {}).get(metric)
    if not entry:
        return []
    return [v for v in entry.get("values", []) if v == v and abs(v) != math.inf]


def thin(values, max_points):
    """Evenly spaced subset of `values`, preserving the ends. Deterministic.

    Only the DRAWING is thinned. The box, median and IQR beside it are computed from every value,
    so this changes how crowded the panel looks and nothing about what it reports. An even stride
    rather than a random sample keeps the same figure on every run and cannot accidentally drop a
    tail.
    """
    n = len(values)
    if not max_points or n <= max_points:
        return list(values)
    idx = np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))
    ordered = np.argsort(np.asarray(values, float))      # thin across the DISTRIBUTION, not time
    return [values[ordered[i]] for i in idx]


def box_strip(ax, pooled, metric, groups, colors, rng=None, max_points=None):
    """One panel: a box per group with its trials drawn over it. Returns True if anything drew."""
    rng = rng or np.random.default_rng(0)
    data = [_values(pooled, g, metric) for g in groups]
    if not any(len(d) for d in data):
        return False

    positions = np.arange(len(groups), dtype=float)
    bp = ax.boxplot([d or [np.nan] for d in data], positions=positions, widths=0.52,
                    patch_artist=True, showfliers=False, zorder=2,
                    medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
                    capprops=dict(color=INK_MUTED, linewidth=1.0))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.16)
        patch.set_edgecolor(colors[g])
        patch.set_linewidth(1.4)

    for i, (g, d) in enumerate(zip(groups, data)):
        if not d:
            continue
        shown = thin(d, max_points)
        # deterministic jitter: the same figure twice is the same figure
        x = positions[i] + rng.uniform(-0.16, 0.16, size=len(shown))
        ax.scatter(x, shown, s=13, color=colors[g], alpha=0.75, linewidths=0.6,
                   edgecolors=SURFACE, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{g}\nn={len(d)}" for g, d in zip(groups, data)], color=INK_SECONDARY)
    ax.set_xlabel("")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _style(ax)
    return True


def headline_figure(pooled, metrics, out_path, title="Metric distributions by task type",
                    ncols=4, max_points=None):
    """Small multiples: one box+strip panel per headline metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _order_groups(list(pooled))
    colors = group_colors(groups)
    metrics = [m for m in metrics if any(_values(pooled, g, m) for g in groups)]
    if not metrics:
        return None

    ncols = min(ncols, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows),
                             facecolor=SURFACE, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes.ravel(), metrics):
        box_strip(ax, pooled, metric, groups, colors, rng, max_points)
        ax.set_title(_short(metric), fontsize=8.5, color=INK, loc="left", pad=6)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, color=INK, x=0.012, ha="left", y=0.995)
    fig.text(0.012, 0.002, "box = median + IQR across trials; each dot is one recording",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.015, 1, 0.975))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


# How far above the data a capacity line may sit before drawing it would squash the distributions
# into a band at the bottom. Past this the limit is annotated instead of plotted -- the fact
# ("you are nowhere near it") survives, the readability of the data is not sacrificed for it.
LIMIT_HEADROOM = 2.2


def per_dof_figure(pooled, out_path, metric_suffix="rom_deg", layer="joint_space",
                   dof_pattern=r"shoulder|elbow|lumbarspine", title=None, ylabel=None,
                   max_points=None, limits=None, limit_label="capacity", fraction=None,
                   box_notes=None):
    """One DOF per x-slot, one box per task within it -- the per-joint comparison across tasks.

    Grouped rather than faceted because the question is 'which joints differ between H, M and L',
    and that comparison has to happen inside one shared y-axis to be legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _order_groups(list(pooled))
    colors = group_colors(groups)
    rx = re.compile(dof_pattern) if dof_pattern else None

    dofs, keys = [], {}
    for g in groups:
        for m in (pooled.get(g) or {}).get("metrics", {}):
            parts = m.split(".")
            if len(parts) != 3 or parts[0] != layer or parts[2] != metric_suffix:
                continue
            dof = parts[1]
            if rx and not rx.search(dof):
                continue
            if dof not in keys:
                keys[dof] = m
                dofs.append(dof)
    if not dofs:
        return None
    dofs.sort()

    fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * len(dofs) * len(groups)), 4.4),
                           facecolor=SURFACE)
    rng = np.random.default_rng(0)
    width = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        offs = (gi - (len(groups) - 1) / 2) * width
        data = [_values(pooled, g, keys[d]) for d in dofs]
        pos = np.arange(len(dofs)) + offs
        bp = ax.boxplot([d or [np.nan] for d in data], positions=pos,
                        widths=width * 0.78, patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color=INK, linewidth=1.3),
                        whiskerprops=dict(color=INK_MUTED, linewidth=0.9),
                        capprops=dict(color=INK_MUTED, linewidth=0.9))
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[g]); patch.set_alpha(0.18)
            patch.set_edgecolor(colors[g]); patch.set_linewidth(1.2)
        for i, d in enumerate(data):
            if not d:
                continue
            shown = thin(d, max_points)
            ax.scatter(pos[i] + rng.uniform(-width * 0.22, width * 0.22, len(shown)), shown,
                       s=8, color=colors[g], alpha=0.7, linewidths=0.4,
                       edgecolors=SURFACE, zorder=3)
            if box_notes and (g, dofs[i]) in box_notes:
                # a per-box note (e.g. the mean excursion duration), just above the box's data
                ax.annotate(box_notes[(g, dofs[i])], xy=(pos[i], max(d)), xytext=(0, 3),
                            textcoords="offset points", rotation=90, fontsize=6.2,
                            color=INK_SECONDARY, ha="center", va="bottom", zorder=6)
        ax.plot([], [], color=colors[g], linewidth=6, alpha=0.5, label=f"{g}")

    # --- the joint's own limit, drawn over its x-slot ---------------------------------------- #
    note = ""
    if limits:
        from .limits import for_metric

        data_top = max((max(_values(pooled, g, keys[d]) or [0.0])
                        for g in groups for d in dofs), default=0.0)
        above = []
        for i, dof in enumerate(dofs):
            lim, unit = for_metric(limits, dof, metric_suffix)
            if not lim:
                continue
            if fraction:
                # the fraction-of-capacity line sits well inside the data range, so it is drawn
                # even when the full capacity is too far above to show
                fl = fraction * lim
                ax.plot([i - 0.46, i + 0.46], [fl, fl], color=SERIES[6], linewidth=1.5,
                        linestyle=(0, (1.5, 1.5)), zorder=5, solid_capstyle="butt")
                ax.annotate(f"{fl:.3g}", xy=(i - 0.46, fl), xytext=(-2, 2),
                            textcoords="offset points", fontsize=6.5, color=SERIES[6],
                            ha="right", va="bottom", zorder=6)
            if data_top and lim > LIMIT_HEADROOM * data_top:
                above.append((dof, lim, unit))
                continue
            ax.plot([i - 0.46, i + 0.46], [lim, lim], color=INK, linewidth=1.6,
                    linestyle=(0, (4, 2.5)), zorder=5, solid_capstyle="butt")
            # 4 significant figures, and offset in POINTS so the label never sits on the line it
            # annotates -- a rad/s limit converted to deg/s is 572.9577..., which is not a number
            # anyone needs to four decimal places.
            ax.annotate(f"{lim:.4g}", xy=(i + 0.46, lim), xytext=(2, 2),
                        textcoords="offset points", fontsize=6.8, color=INK,
                        ha="left", va="bottom", zorder=6)
        if above:
            lo = min(l for _, l, _ in above)
            unit = above[0][2] or ""
            note = (f"{len(above)} joint limit(s) lie above this axis — lowest {lo:.4g} {unit}, "
                    f"more than {LIMIT_HEADROOM:g}x the largest value shown, so they are not drawn")

    ax.set_xticks(np.arange(len(dofs)))
    ax.set_xticklabels(dofs, rotation=38, ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel(ylabel or metric_suffix, fontsize=9, color=INK_SECONDARY)
    ax.set_title(title or f"{metric_suffix} per DOF, by task type", fontsize=11,
                 color=INK, loc="left", pad=10)
    _style(ax)
    leg = ax.legend(frameon=False, fontsize=9, ncol=len(groups), loc="upper right",
                    title="task")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(INK_MUTED)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    if limits:
        ax.plot([], [], color=INK, linewidth=1.6, linestyle=(0, (4, 2.5)),
                label=limit_label)
        if fraction:
            ax.plot([], [], color=SERIES[6], linewidth=1.5, linestyle=(0, (1.5, 1.5)),
                    label=f"{100 * fraction:g}% of {limit_label}")
        handles, labels = ax.get_legend_handles_labels()
        leg2 = ax.legend(handles, labels, frameon=False, fontsize=9, ncol=len(handles),
                         loc="upper right", title="task")
        leg2.get_title().set_fontsize(8)
        leg2.get_title().set_color(INK_MUTED)
        for t in leg2.get_texts():
            t.set_color(INK_SECONDARY)
    if note:
        fig.text(0.008, 0.004, note, fontsize=7.6, color=INK_MUTED, ha="left")

    fig.tight_layout(rect=(0, 0.035, 1, 1)) if note else fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


# Active / Passive / Free is a classification, not a magnitude -- three fixed categorical slots,
# always with a legend, never colour alone (the counts are printed inside the bars).
def differences_figure(ranked, pooled, out_path, top=20,
                       title="Metrics that separate the task types most"):
    """Profile plot of the top discriminating metrics: each row's medians, scaled within that row.

    Scaling per row is what lets a torque in N.m and a volume in litres share one axis honestly --
    the axis is explicitly 'share of this metric's own largest median', not a value. The absolute
    medians are printed on the right so nothing has to be read off the scaled position.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = [r for r in ranked if r.get("p") == r.get("p")][:top]
    if not ranked:
        return None
    groups = _order_groups(list(pooled))
    colors = group_colors(groups)

    fig, ax = plt.subplots(figsize=(10.5, max(3.5, 0.42 * len(ranked))), facecolor=SURFACE)
    labels = []
    for i, row in enumerate(reversed(ranked)):
        med = row["medians"]
        finite = [abs(med[g]) for g in groups if med.get(g) == med.get(g)]
        scale = max(finite) if finite and max(finite) > 0 else 1.0
        xs = [med.get(g, float("nan")) / scale for g in groups]
        ax.plot([min(x for x in xs if x == x), max(x for x in xs if x == x)], [i, i],
                color=GRID, linewidth=1.6, zorder=1, solid_capstyle="round")
        for g, x in zip(groups, xs):
            if x != x:
                continue
            ax.scatter(x, i, s=54, color=colors[g], zorder=3, linewidths=1.2,
                       edgecolors=SURFACE)
        labels.append(f"{row['metric']}   (p={row['p']:.2g})")

    for g in groups:
        ax.scatter([], [], s=54, color=colors[g], label=g)

    ax.set_yticks(np.arange(len(ranked)))
    ax.set_yticklabels(labels, fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("group median, scaled within each metric", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    leg = ax.legend(frameon=False, fontsize=9, ncol=len(groups), loc="lower right", title="task")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(INK_MUTED)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.text(0.008, 0.004, "Kruskal-Wallis p, screening only -- with one subject these are "
                           "repeated measures, not a population test",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


def build_all(pooled, ranked, cfg, out_dir, limits=None):
    """The scalar figures a pooled run produces, filtered by `cfg.figures`. Returns paths written."""
    # `joints_of_interest`, when set, is THE list -- it overrides the per-figure default patterns
    # so one line in the config governs every per-joint panel.
    cfg_dofs = cfg.dof_regex(fallback=cfg.plot_dofs)
    cap = getattr(cfg, "box_max_points", None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def keep(p):
        if p:
            written.append(p)

    if cfg.wants("headline_distributions"):
        keep(headline_figure(pooled, cfg.headline_metrics,
                             cfg.figure_path(out_dir, "headline_distributions"), max_points=cap))

    for suffix, layer, label, name, limit_label in (
        ("rom_deg", "joint_space", "range of motion [deg]", "rom_by_task", None),
        ("peak_vel_dps", "joint_space", "peak joint speed [deg/s]", "joint_speed_by_task",
         "URDF speed limit"),
        ("peak_torque_Nm", "dynamics", "peak torque [N.m]", "peak_torque_by_task",
         "URDF torque capacity"),
        ("rms_torque_Nm", "dynamics", "RMS torque [N.m]", "rms_torque_by_task",
         "URDF torque capacity"),
        ("net_work_J", "dynamics", "net work over the trial [J]", "net_work_by_task", None),
    ):
        if cfg.wants(name):
            keep(per_dof_figure(pooled, cfg.figure_path(out_dir, name), metric_suffix=suffix,
                                layer=layer, dof_pattern=cfg_dofs, ylabel=label, max_points=cap,
                                limits=(limits if limit_label else None),
                                limit_label=limit_label or "capacity",
                                fraction=(getattr(cfg, "capacity_fraction", None)
                                          if layer == "dynamics" else None)))

    if cfg.wants("top_differences"):
        keep(differences_figure(ranked, pooled, cfg.figure_path(out_dir, "top_differences"),
                                top=min(20, cfg.top_differences)))
    return written
