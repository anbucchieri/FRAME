"""
The averaged 3D path of a tracked point, per task type.

For a trajectory-tracing task the shape of the path IS the task, and no scalar carries it. This
draws the mean path of one segment -- the palm holding the tool, by default -- in the trunk frame,
averaged over every full recording of every subject, with the individual recordings behind it.

Three choices worth stating:

  * **The point drawn is not the point measured.** The analysis tracks the FOREARM, because that is
    where an exoskeleton's end-effector attaches, and every task-space metric stays there. The palm
    is drawn because that is what holds the tool and traces the shape. Both come from the same
    .mvnx, in the same trunk frame; only the figure differs.

  * **Each recording is kept whole.** No segmentation: every recording is one continuous path,
    linearly time-normalised exactly as the joint trajectory bands are, so the mean at 40% is the
    mean of every recording's 40% point.

  * **A mean path is a fiction if the recordings disagree.** So the individual recordings are
    drawn underneath it, and the spread is reported as the median distance from each recording to
    the mean path. A tight mean over a scattered bundle is visible rather than implied.

Everything is in the trunk frame: +X forward, +Y to the subject's left, +Z up, origin at T8. So the
picture does not move when the subject shifts their feet.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from frame_core.signal import resample_interp

from .plots import (SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID,
                    group_colors, _order_groups)
from .segments import cached_segment_trajectory

AXES = ("X (forward)", "Y (left)", "Z (up)")


def collect(records, trials_by_id, cache, cfg, progress=None):
    """{task: {"trials": (n_trials, n, 3), "mean": (n, 3), "spread_m": float}}."""
    n = int(cfg.timeseries_n_resample)
    per_group: dict = {}
    missing = 0

    for rec in records:
        trial = trials_by_id.get(rec.get("trial_id"))
        if trial is None:
            continue
        try:
            xyz = cached_segment_trajectory(trial, rec.get("subject"), cache,
                                            cfg.trajectory_segment, cfg.trunk_segment)
        except (ValueError, OSError):
            missing += 1
            continue
        if len(xyz) < 4:
            missing += 1
            continue
        # the WHOLE recording, one path, never segmented
        per_group.setdefault(rec.get("task"), []).append(resample_interp(xyz, n))
        if progress:
            progress(rec.get("trial_id"))

    out = {}
    for task, paths in per_group.items():
        stack = np.stack(paths)                       # (n_trials, n, 3)
        mean = stack.mean(axis=0)
        # how far a typical recording sits from the mean path, in metres -- the honesty number
        spread = float(np.median(np.linalg.norm(stack - mean, axis=2)))
        out[task] = {"trials": stack, "mean": mean, "spread_m": spread,
                     "n": int(stack.shape[0])}
    return out, missing


def _equal_bounds(paths, pad=0.06):
    """One cube that contains every path, so the three projections share a scale and a path's
    shape cannot be distorted by an axis stretching to fit it."""
    pts = np.vstack([p for p in paths if len(p)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo)) * (1.0 + pad)
    return centre - half, centre + half


def figure(paths_by_group, out_path, segment="RightHand", show_trials=True,
           title=None):
    """A 3D view plus the three orthogonal projections, one mean path per task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    groups = _order_groups(list(paths_by_group))
    if not groups:
        return None
    colors = group_colors(groups)
    lo, hi = _equal_bounds([paths_by_group[g]["mean"] for g in groups])

    fig = plt.figure(figsize=(15.5, 4.6), facecolor=SURFACE)
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    ax3d.set_facecolor(SURFACE)

    for g in groups:
        d = paths_by_group[g]
        if show_trials:
            for p in d["trials"][:: max(1, d["trials"].shape[0] // 18)]:
                ax3d.plot(p[:, 0], p[:, 1], p[:, 2], color=colors[g], alpha=0.08, linewidth=0.6)
        m = d["mean"]
        ax3d.plot(m[:, 0], m[:, 1], m[:, 2], color=colors[g], linewidth=2.4,
                  label=f"{g}  (n={d['n']})")
        ax3d.scatter(*m[0], color=colors[g], s=26, edgecolors=SURFACE, linewidths=1.0, zorder=5)

    ax3d.set_xlim(lo[0], hi[0]); ax3d.set_ylim(lo[1], hi[1]); ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_box_aspect((1, 1, 1))
    # The z-axis is drawn on the right of the 3D box; a padded label there runs into the next
    # panel's y-label, so it is kept tight. The projections carry the full axis names anyway.
    ax3d.set_xlabel("X fwd [m]", fontsize=8, color=INK_MUTED, labelpad=6)
    ax3d.set_ylabel("Y left [m]", fontsize=8, color=INK_MUTED, labelpad=6)
    ax3d.set_zlabel("Z up [m]", fontsize=8, color=INK_MUTED, labelpad=-2)
    ax3d.tick_params(colors=INK_SECONDARY, labelsize=7)
    ax3d.view_init(elev=18, azim=-62)
    ax3d.set_title("3D (dot = start of recording)", fontsize=9, color=INK, loc="left", pad=4)

    for k, (i, j, name) in enumerate(((0, 2, "side:  X–Z"), (1, 2, "front:  Y–Z"),
                                      (0, 1, "top:  X–Y"))):
        ax = fig.add_subplot(1, 4, k + 2)
        ax.set_facecolor(SURFACE)
        for g in groups:
            d = paths_by_group[g]
            if show_trials:
                for p in d["trials"][:: max(1, d["trials"].shape[0] // 25)]:
                    ax.plot(p[:, i], p[:, j], color=colors[g], alpha=0.09, linewidth=0.6)
            m = d["mean"]
            ax.plot(m[:, i], m[:, j], color=colors[g], linewidth=2.2, zorder=3)
            ax.scatter(m[0, i], m[0, j], color=colors[g], s=26, edgecolors=SURFACE,
                       linewidths=1.0, zorder=5)
        ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j])
        ax.set_aspect("equal")
        ax.set_xlabel(f"{AXES[i]} [m]", fontsize=8, color=INK_MUTED)
        ax.set_ylabel(f"{AXES[j]} [m]", fontsize=8, color=INK_SECONDARY)
        ax.set_title(name, fontsize=9, color=INK, loc="left", pad=6)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=3, width=0.8)

    fig.suptitle(title or f"{segment} path per task type — mean over full recordings",
                 fontsize=12.5, color=INK, x=0.006, ha="left", y=0.995)
    handles, labels = ax3d.get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(groups),
                     loc="upper right", bbox_to_anchor=(0.997, 1.005), title="task")
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(INK_MUTED)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    spread = "   ".join(f"{g}: {paths_by_group[g]['spread_m'] * 100:.1f} cm"
                        for g in groups)
    fig.text(0.006, 0.005,
             f"bold = mean path in the trunk frame (origin T8); faint = individual recordings. "
             f"Median distance from a recording to its mean path — {spread}. All panels share one "
             f"scale, so shapes are comparable.",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


FIGURE_NAME = "ee_trajectory_3d_by_task"


def build(records, cache, cfg, out_dir, subjects=(), progress=None):
    """The 3D trajectory figure. Returns (path | None, note)."""
    trials_by_id = {t.trial_id: t for s in subjects for t in s.trials}
    if not trials_by_id:
        return None, "no trials available for the 3D trajectory figure"

    paths, missing = collect(records, trials_by_id, cache, cfg, progress=progress)
    if not paths:
        return None, (f"could not extract any {cfg.trajectory_segment!r} trajectory -- the raw "
                      f".mvnx files must be readable (they are re-read for this figure).")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = figure(paths, cfg.figure_path(out_dir, "ee_trajectory_3d_by_task"),
                  segment=cfg.trajectory_segment,
                  show_trials=cfg.trajectory_show_trials)
    note = (f"{missing} of {len(records)} trials contributed no 3D trajectory"
            if missing else "")
    return path, note
