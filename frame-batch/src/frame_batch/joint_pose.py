"""
One figure per task type and joint of interest: the joint's curves under the posture they were
produced in.

Bottom: the joint's angle, speed and torque against percent of trial, mean +/- 1 SD over every
trial of the task (the panels of `joint_timeseries_by_task`, drawn by the same code).

Top: three stick figures of the torso and right arm -- the MEAN pose over the task's trials at 0 %,
50 % and 100 % of the trial -- with a sphere on each joint coloured (and sized) by |tau| / torque
capacity at that instant, averaged over the trials. DOFs sharing a segment are collapsed to the
worst of them, as in the N-pose map; the sphere of the joint the figure is about is ringed in black
and that DOF's own value is printed under the panel.

The tasks are drawn separately because a mean pose is only a posture within one task: averaged
over the H, M and L shelves it would describe none of them.

The poses are segment positions, not IK output: they come straight from the .mvnx the trial was
analysed from, trimmed exactly as the IK input was, and expressed in the PELVIS frame (+X forward,
+Y left, +Z up) so that trunk flexion shows as the torso leaning rather than the pelvis swinging.
The three poses per trial are cached beside the trial's record, so only the first report reads the
recordings (~1 s each).
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np

from .plots import SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID, group_colors, _style

FIGURE_NAME = "joint_pose_timeseries"

INSTANTS = (0.0, 0.5, 1.0)                          # fractions of the trial
INSTANT_LABELS = ("t = 0", "t = T/2", "t = T")
ANCHOR = "Pelvis"
TORSO = ["Pelvis", "L5", "L3", "T12", "T8", "Neck"]
ARM = ["T8", "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand"]
SEGMENTS = list(dict.fromkeys(TORSO + ARM))
BONES = [(a, b) for chain in (TORSO, ARM) for a, b in zip(chain, chain[1:])]
# The DOFs that get a sphere: the trunk and the right-arm chain.
SPHERE_DOFS = r"^(lumbarspine|thoracicspine|right_clavicular|right_shoulder|right_elbow|right_wrist)"


def _at(x, fracs=INSTANTS):
    """`x` (N, ...) linearly interpolated at fractions of its length along axis 0."""
    x = np.asarray(x, float)
    t = np.asarray(fracs, float) * (len(x) - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, len(x) - 1)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    w = (t - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[i0] + w * x[i1]


def _trial_frame(rec, cfg):
    """The segment columns of the trial's .mvnx, trimmed the way its IK input was."""
    from frame_kinematics.segment import trim_edges
    from .segments import segment_dataframe
    df = segment_dataframe(str(rec.get("path")))
    fs = float(rec.get("fs") or cfg.fs or 240.0)
    df, _ = trim_edges(df, fs, cfg.trim_start_s, cfg.trim_end_s, cfg.min_len)
    return df


def trial_poses(rec, cache, cfg):
    """(len(INSTANTS), len(SEGMENTS), 3) segment positions in the pelvis frame, cached."""
    from frame_kinematics.assemble import segments_in_trunk
    key = f"{ANCHOR}|{','.join(SEGMENTS)}|{INSTANTS}|{cfg.trim_start_s}|{cfg.trim_end_s}"
    path = Path(cache.record_path(rec.get("subject"), rec.get("trial_id"), create=True)) \
        .with_suffix(".pose_instants.npz")
    if path.is_file():
        try:
            with np.load(path) as z:
                if str(z["key"]) == key:
                    return z["pose"]
        except (OSError, KeyError, ValueError):
            pass                                    # corrupt or stale -> just redo it
    pose = _at(segments_in_trunk(_trial_frame(rec, cfg), SEGMENTS, ANCHOR))
    try:
        np.savez_compressed(path, pose=pose, key=key)
    except OSError:
        pass                                        # a read-only cache must not fail the figure
    return pose


def trial_load(cap):
    """{segment: (len(INSTANTS),) worst |tau| / capacity}, {dof: same} for the sphere DOFs."""
    from frame_core.anatomy import dof_to_segment
    if not cap.reps or cap.reps[0].torques is None:
        return {}, {}
    T = np.vstack([r.torques for r in cap.reps])
    limits = cap.urdf_limits or {}
    rx = re.compile(SPHERE_DOFS)
    per_dof, per_seg = {}, {}
    for j, name in enumerate(cap.dof_names):
        seg = dof_to_segment(name)
        eff = (limits.get(name) or {}).get("effort")
        if seg not in SEGMENTS or not eff or not rx.search(name):
            continue
        r = np.abs(_at(T[:, j])) / float(eff)
        per_dof[name] = r
        per_seg[seg] = r if seg not in per_seg else np.fmax(per_seg[seg], r)
    return per_seg, per_dof


def collect(records, cache, cfg):
    """{"pose": (I, S, 3) mean, "seg": {seg: (I,)}, "dof": {dof: (I,)}, "n": int}, n_missing."""
    from .captures import load
    poses, segs, dofs, missing = [], {}, {}, 0
    for rec in records:
        cap = load(cache, rec, cfg)
        try:
            pose = trial_poses(rec, cache, cfg)
        except (ValueError, OSError, KeyError):
            pose = None
        if cap is None or pose is None:
            missing += 1
            continue
        poses.append(pose)
        per_seg, per_dof = trial_load(cap)
        for k, v in per_seg.items():
            segs.setdefault(k, []).append(v)
        for k, v in per_dof.items():
            dofs.setdefault(k, []).append(v)
    if not poses:
        return None, missing
    import warnings
    with warnings.catch_warnings():                 # a masked shoulder DOF can be all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"pose": np.nanmean(np.stack(poses), axis=0),
                "seg": {k: np.nanmean(np.stack(v), axis=0) for k, v in segs.items()},
                "dof": {k: np.nanmean(np.stack(v), axis=0) for k, v in dofs.items()},
                "n": len(poses)}, missing


def _stick(ax, P, seg_ratio, k, cmap, norm, vmax, focus_seg, lims):
    """One stick figure: the mean pose at instant `k`, spheres by |tau| / capacity."""
    idx = {s: i for i, s in enumerate(SEGMENTS)}
    for a, b in BONES:
        pa, pb = P[idx[a]], P[idx[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            ax.plot(*zip(pa, pb), color="0.35", lw=2.4, solid_capstyle="round", zorder=1)
    good = P[np.isfinite(P).all(axis=1)]
    ax.scatter(good[:, 0], good[:, 1], good[:, 2], s=8, c="0.45", depthshade=False, zorder=2)
    o = P[idx[ANCHOR]]
    ax.quiver(*o, 0.12, 0, 0, color=INK_MUTED, lw=1.0, arrow_length_ratio=0.3)
    ax.text(o[0] + 0.15, o[1], o[2], "fwd", fontsize=6.5, color=INK_MUTED)
    (x0, x1), (y0, y1), (z0, z1) = lims
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=1.5)
    ax.view_init(elev=14, azim=-35)                 # from the front-right
    ax.computed_zorder = False                      # labels over spheres over bones, whatever the depth
    # Labels are placed in screen space: the trunk's to one side, the arm's to the other, so the
    # four joints stacked at shoulder height do not print over each other or their spheres.
    from mpl_toolkits.mplot3d import proj3d
    proj = ax.get_proj()
    mid = proj3d.proj_transform(*P[idx["T8"]], proj)[0]
    # least-loaded first, so the joints that matter are drawn on top
    for seg, r in sorted(seg_ratio.items(), key=lambda kv: np.nan_to_num(kv[1][k])):
        q, v = P[idx[seg]], float(r[k])
        if not (np.isfinite(q).all() and np.isfinite(v)):
            continue
        ax.scatter(*q, s=110 + 480 * min(v / vmax, 1.0), color=cmap(norm(v)),
                   edgecolors=INK if seg == focus_seg else "white",
                   linewidths=2.0 if seg == focus_seg else 1.0, depthshade=False, zorder=6)
        sx, sy, _ = proj3d.proj_transform(*q, proj)
        arm = seg in ARM[1:]
        right = (sx >= mid) if arm else (sx < mid)
        ax.annotate(f"{100 * v:.0f}%", xy=(sx, sy), xycoords="data",
                    xytext=(13 if right else -13, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=7.5, color=INK,
                    fontweight="bold" if seg == focus_seg else "normal", zorder=7)
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()


def figure(dof, bands_by_group, pose, out_path, limits=None, normalised=False,
           capacity_fraction=None, n_resample=500, task=None, colors=None):
    """Three stick figures over the joint's angle / speed / torque panels.

    `colors` ({group: colour}) keeps a task's colour the same as in the other figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from frame_core.anatomy import dof_to_segment
    from .timeseries import ROW_ORDER, draw_panel, _ylabel

    groups = list(bands_by_group)
    colors = colors or group_colors(groups)
    rows = [q for q in ROW_ORDER if any(q in (bands_by_group[g].get(dof) or {}) for g in groups)]
    if not rows:
        return None

    vals = [float(x) for r in pose["seg"].values() for x in r if np.isfinite(x)]
    vmax = max(1.0, max(vals, default=0.0))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn_r")
    P = pose["pose"]
    good = P[np.isfinite(P).all(axis=-1)]
    lo, hi = good.min(axis=0) - 0.08, good.max(axis=0) + 0.08     # one box for all three
    lims = list(zip(lo, hi))

    fig = plt.figure(figsize=(12.5, 5.6 + 2.3 * len(rows)), facecolor=SURFACE)
    gs = fig.add_gridspec(1 + len(rows), 3, height_ratios=[2.6] + [1.0] * len(rows),
                          hspace=0.16, wspace=0.0, left=0.07, right=0.93, top=0.96, bottom=0.07)
    focus = dof_to_segment(dof)
    own = pose["dof"].get(dof)
    sticks = []
    for k, lab in enumerate(INSTANT_LABELS):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        _stick(ax, P[k], pose["seg"], k, cmap, norm, vmax, focus, lims)
        ax.text2D(0.5, 0.97, f"{lab}  ({100 * INSTANTS[k]:.0f} % of trial)", transform=ax.transAxes,
                  ha="center", va="top", fontsize=10, color=INK)
        if own is not None and np.isfinite(own[k]):
            ax.text2D(0.5, 0.03, f"{dof}: {100 * own[k]:.0f}% of capacity", transform=ax.transAxes,
                      ha="center", fontsize=8.5, color=INK)
        sticks.append(ax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=sticks, shrink=0.75, pad=0.04, aspect=22)
    cb.set_label("|τ| / torque capacity (mean over trials)", fontsize=8.5, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=7.5, colors=INK_SECONDARY)

    first = None
    for ri, q in enumerate(rows):
        ax = fig.add_subplot(gs[1 + ri, :], sharex=first)
        first = first or ax
        draw_panel(ax, bands_by_group, groups, colors, dof, q, limits=limits,
                   normalised=normalised, capacity_fraction=capacity_fraction,
                   label_lines=(ri == 0))
        for g in groups:                            # the three instants on the mean curve
            b = (bands_by_group[g].get(dof) or {}).get(q)
            if b is not None:
                ax.scatter(100 * np.asarray(INSTANTS), _at(b["mean"]), s=28, color=INK,
                           zorder=5, clip_on=False)
        ax.axvline(50, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_ylabel(_ylabel(q, normalised), fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlim(0, 100)
        if ri < len(rows) - 1:
            ax.tick_params(labelbottom=False)
    first.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_xlabel("% of trial", fontsize=8.5, color=INK_MUTED)

    which = f"task {task}" if task is not None else "all trials pooled"
    fig.suptitle(f"{dof} — posture and trajectories, {which} (n={pose['n']})",
                 fontsize=12.5, color=INK, x=0.01, ha="left", y=0.985)
    import textwrap
    fig.text(0.01, 0.005, textwrap.fill(
        "Top: mean torso and right-arm pose over the trials at 0, 50 and 100 % of the trial, in "
        "the pelvis frame (seen from the front-right). Sphere colour and area = |τ| / torque "
        "capacity at that instant, averaged over the trials; a segment with several DOFs shows the "
        "worst of them, and the ringed sphere is this joint's segment. Bottom: mean ±1 SD over "
        f"the trials, each time-normalised to {n_resample} samples; dots mark the three instants.",
        width=200), fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


def build(records, cache, cfg, out_dir, progress=None):
    """One figure per task type and joint of interest. Returns (paths | None, note)."""
    from . import timeseries
    from .plots import _order_groups
    dofs = timeseries.select_dofs(cfg.analysed_dofs(), cfg.dof_regex(fallback=cfg.timeseries_dofs))
    if not dofs:
        return None, f"{FIGURE_NAME}: no DOF matches timeseries_dofs={cfg.timeseries_dofs!r}"
    bands_by_group, _ = timeseries.collect(records, cache, cfg, dofs=dofs, progress=progress)
    try:
        from . import limits as limits_mod
        lims = limits_mod.collect(records, cache, dofs=set(dofs), cfg=cfg)
    except Exception:
        lims = None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.figure_path(out_dir, FIGURE_NAME))
    paths, notes = [], []
    tasks = _order_groups({str(r.get("task")) for r in records})
    colors = group_colors(tasks)
    for task in tasks:
        recs = [r for r in records if str(r.get("task")) == task]
        pose, missing = collect(recs, cache, cfg)
        if task not in bands_by_group or pose is None:
            notes.append(f"task {task}: no trial had both a cached Capture and readable segment "
                         "positions")
            continue
        if missing:
            notes.append(f"task {task}: {missing} of {len(recs)} trials had no capture or segment "
                         "positions and are missing from the stick figures")
        for dof in dofs:
            p = figure(dof, {task: bands_by_group[task]}, pose,
                       stem.with_name(f"{stem.stem}_{task}_{dof}{stem.suffix}"),
                       limits=lims, normalised=cfg.timeseries_normalise_torque,
                       capacity_fraction=getattr(cfg, "capacity_fraction", None),
                       n_resample=int(cfg.timeseries_n_resample), task=task, colors=colors)
            if p:
                paths.append(p)
    return (paths or None), "; ".join(f"{FIGURE_NAME}: {n}" for n in notes)
