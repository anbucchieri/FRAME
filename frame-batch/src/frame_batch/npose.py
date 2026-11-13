"""
N-pose torque map, pooled per task type (and orientation -- see `BatchConfig.npose_by`).

A neutral-standing skeleton with a coloured sphere at every actuated joint: green when the joint
loafs, through yellow and orange, to red when its RMS torque approaches the joint's torque
capacity. One panel per task, so where the load sits at a high shelf versus a low one is a
side-by-side comparison rather than three separate pictures.

It does not reimplement the map. `frame_kinematics.torque_map.npose_torque_map` already reduces one
Capture to {segment: RMS, capacity, ratio}, collapsing the DOFs that share a segment by taking the
worst of them; this module calls that per trial and pools the results. What is added here is the
pooling and the side-by-side layout.

Three things worth stating:

  * **The colour scale is shared across tasks.** Per-panel scaling would make every task look
    equally loaded, which is the one comparison the figure exists to make.

  * **Pooled by MEDIAN.** A mean over trials lets one bad inverse-dynamics spike repaint a joint;
    the per-trial ratios are also right-skewed, like the torques they come from.

  * **Colour is never the only channel.** A red/green ramp is the worst case for colour-vision
    deficiency, so the sphere AREA grows with the ratio and the number is printed beside it. The
    figure is readable in greyscale.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .plots import SURFACE, INK, INK_SECONDARY, INK_MUTED, _order_groups

FIGURE_NAME = "npose_torque_map_by_task"

# Which joints get a sphere. The legs are still analysed; they are simply not painted, because an
# upper-limb exoskeleton's spec sheet is read off the chain it attaches to.
UPPER_BODY = r"lumbarspine|thoracicspine|spine|neck|head|clavicular|shoulder|elbow|wrist"


def group_label(rec, by):
    """The panel a record belongs to: its `by` fields joined by "_" ("H_N"), or "all"."""
    by = list(by or [])
    if not by:
        return "all"
    return "_".join(str(rec.get(f)) for f in by)


def collect(records, cache, cfg, pattern=UPPER_BODY, progress=None):
    """{group: {"segments": {seg: {...}}, "skeleton": {...}, "n": int}} from the cached Captures.

    The group is `cfg.npose_by` (default task x orientation)."""
    by = getattr(cfg, "npose_by", None) or ["task"]
    from frame_kinematics.torque_map import npose_torque_map
    from .captures import load

    raw: dict = {}
    missing = 0
    for rec in records:
        cap = load(cache, rec, cfg)
        if cap is None:
            missing += 1
            continue
        per_seg = npose_torque_map(cap, pattern)
        if not per_seg:
            missing += 1
            continue
        slot = raw.setdefault(group_label(rec, by),
                              {"per_seg": {}, "poses": [], "skeleton": None})
        for seg, v in per_seg.items():
            d = slot["per_seg"].setdefault(seg, {"ratio": [], "rms": [], "peak": [],
                                                 "capacity": v.get("capacity"),
                                                 "has_capacity": v.get("has_capacity", False),
                                                 "dof": v.get("dof")})
            if v.get("ratio") is not None:
                d["ratio"].append(float(v["ratio"]))
            d["rms"].append(float(v["rms"]))
            d["peak"].append(float(v["peak"]))
        sk = cap.skeleton or {}
        pose = sk.get("pose_npose")
        if pose is None:
            pose = sk.get("pose_mean")
        if pose is not None and slot["skeleton"] is None:
            slot["skeleton"] = {"segment_names": list(sk.get("segment_names", [])),
                                "bones": [list(b) for b in sk.get("bones", [])]}
        if pose is not None:
            slot["poses"].append(np.asarray(pose, float))
        if progress:
            progress(rec.get("trial_id"))

    out = {}
    for task, slot in raw.items():
        if slot["skeleton"] is None or not slot["poses"]:
            continue
        shapes = {p.shape for p in slot["poses"]}
        if len(shapes) != 1:                      # subjects with different segment sets
            ref = slot["poses"][0].shape
            slot["poses"] = [p for p in slot["poses"] if p.shape == ref]
        segments = {}
        for seg, d in slot["per_seg"].items():
            segments[seg] = {
                "ratio": float(np.median(d["ratio"])) if d["ratio"] else None,
                "rms": float(np.median(d["rms"])),
                "peak": float(np.median(d["peak"])),
                "capacity": d["capacity"], "has_capacity": d["has_capacity"],
                "dof": d["dof"], "n": len(d["rms"]),
            }
        out[task] = {
            "segments": segments,
            # a representative body: the median neutral pose over the contributing subjects
            "pose": np.median(np.stack(slot["poses"]), axis=0),
            "skeleton": slot["skeleton"],
            "n": max((v["n"] for v in segments.values()), default=0),
        }
    return out, missing


def figure(by_task, out_path, title="Joint load in the N-pose, by task type and orientation"):
    """One skeleton per task, spheres coloured by median RMS torque over joint capacity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors

    groups = _order_groups(list(by_task))
    if not groups:
        return None

    # ONE scale for every panel: the comparison between tasks is the point of the figure.
    vals = [v["ratio"] for g in groups for v in by_task[g]["segments"].values()
            if v["ratio"] is not None]
    if not vals:
        return None
    vmax = max(1.0, float(np.max(vals)))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn_r")
    any_cap = any(v["has_capacity"] for g in groups for v in by_task[g]["segments"].values())

    # One row per task, one column per further factor ("H_N", "H_P" -> row H). Plain labels
    # ("H", "M", "L") stay on a single row, as before.
    heads = []
    for g in groups:
        h = str(g).partition("_")[0] if "_" in str(g) else ""
        if h not in heads:
            heads.append(h)
    rows = [[g for g in groups if (str(g).partition("_")[0] if "_" in str(g) else "") == h]
            for h in heads]
    nrows, ncols = len(rows), max(len(r) for r in rows)
    fig = plt.figure(figsize=(4.0 * ncols + 1.5, 4.6 * nrows + 0.4), facecolor=SURFACE)
    axes = []
    slot_of = {g: (i, j) for i, r in enumerate(rows) for j, g in enumerate(r)}
    for k, g in enumerate(groups):
        d = by_task[g]
        i, j = slot_of[g]
        ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1, projection="3d")
        ax.set_facecolor(SURFACE)
        P = np.asarray(d["pose"], float)
        names = list(d["skeleton"]["segment_names"])

        pts = []
        for a, b in d["skeleton"]["bones"]:
            if a >= len(P) or b >= len(P):
                continue
            pa, pb = P[a], P[b]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                    color="0.35", lw=2.0, solid_capstyle="round", zorder=1)
            pts.extend([pa, pb])
        good = P[np.isfinite(P).all(axis=1)]
        if good.size:
            ax.scatter(good[:, 0], good[:, 1], good[:, 2], s=8, c="0.45", zorder=2)
            pts.extend(list(good))

        # least-loaded first, so the joints that matter are drawn on top
        painted, midline_seen = [], 0
        for seg, v in sorted(d["segments"].items(), key=lambda kv: kv[1]["ratio"] or 0.0):
            if seg not in names:
                continue
            q = P[names.index(seg)]
            if not np.isfinite(q).all():
                continue
            r = v["ratio"] or 0.0
            ax.scatter(*q, s=170 + 620 * min(r / vmax, 1.0),      # area = redundant encoding
                       color=cmap(norm(r)), edgecolors="white", linewidths=1.1,
                       depthshade=False, zorder=6)
            painted.append((q, v, r))

        # Labels last, and SHORT: just the percentage of capacity, because the N.m figure is in
        # joint_summary.csv and two numbers per sphere is what made the shoulder row unreadable.
        # They are staggered vertically in the order the joints appear across the body, which is
        # what separates the four or five that sit almost on one line at shoulder height.
        for i, (q, v, r) in enumerate(sorted(painted, key=lambda t: t[0][1])):
            label = f"{100 * r:.0f}%" if v["has_capacity"] else f"{v['rms']:.0f}"
            dz = 0.055 if i % 2 == 0 else -0.055
            ax.text(q[0], q[1], q[2] + dz, label, fontsize=7.0, ha="center",
                    va="bottom" if dz > 0 else "top", color=INK, zorder=7)
        painted = [q for q, _, _ in painted]

        # Crop to the painted region. The legs are drawn for context but carry no sphere, and at
        # full body height they take two thirds of the panel and squeeze the upper limb -- which
        # is the only part this figure is about -- into an unreadable strip.
        A = np.array(painted) if painted else np.array(pts)
        zlo, zhi = A[:, 2].min() - 0.18, A[:, 2].max() + 0.14
        ylo, yhi = A[:, 1].min() - 0.30, A[:, 1].max() + 0.30
        half = 0.5 * max(yhi - ylo, zhi - zlo)
        ycen, zcen = 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)
        ax.set_ylim(ycen - half, ycen + half)
        ax.set_zlim(zcen - half, zcen + half)
        ax.set_box_aspect((0.16, 1.0, 1.0))
        ax.set_xticks([])
        ax.tick_params(labelsize=6.5, colors=INK_SECONDARY)
        ax.view_init(elev=8, azim=180)
        label = "task " + str(g).replace("_", " · ", 1)
        ax.set_title(f"{label}   (n={d['n']} trials)", fontsize=10.5, color=INK,
                     loc="left", pad=2)
        axes.append(ax)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.52, pad=0.02, aspect=26)
    cb.set_label("median RMS torque / joint capacity" if any_cap
                 else "median RMS torque / largest RMS (no capacity recorded)",
                 fontsize=8.5, color=INK_SECONDARY)
    cb.ax.tick_params(labelsize=7.5, colors=INK_SECONDARY)
    if any_cap and vmax >= 1.0:
        cb.ax.axhline(1.0 / vmax, color="black", lw=1.4)      # the capacity limit itself

    fig.suptitle(title, fontsize=12.5, color=INK, x=0.006, ha="left", y=0.985)
    observed = f"{100 * float(np.max(vals)):.0f}%"
    import textwrap
    fig.text(0.006, 0.0, textwrap.fill(
             "sphere colour AND area = median RMS torque as a fraction of that joint's capacity, "
             "and the number beside each is that percentage (the N.m values are in "
             "joint_summary.csv). "
             + f"The scale runs to 100% of capacity on purpose, so green really does mean "
               f"'far from the limit' -- the highest value anywhere here is {observed}. "
               "All panels share it. Skeleton = median neutral pose of the contributing subjects, "
               "in the T8 frame; the legs carry no sphere and are drawn for orientation only.",
             width=int(15 * fig.get_figwidth())),
             fontsize=8, color=INK_MUTED, ha="left", va="top")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def build(records, cache, cfg, out_dir, progress=None):
    """The pooled N-pose torque map. Returns (path | None, note)."""
    by_task, missing = collect(records, cache, cfg, progress=progress)
    if not by_task:
        return None, ("no N-pose torque map could be built -- it needs cached Captures with "
                      "torques and a stored skeleton (`cache_captures: true`).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = figure(by_task, cfg.figure_path(out_dir, FIGURE_NAME))
    note = (f"{missing} of {len(records)} trials contributed no torque map"
            if missing else "")
    return path, note
