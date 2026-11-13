"""
One-page schematic of FRAME -> FRAME_schematic.pdf

Laid out left to right across the CACHE: everything left of it runs once per recording and costs
hours, everything right of it re-runs from the cache in seconds. Which side a setting feeds is the
most useful fact about it, so that is the axis the page is organised on.

Run:  ./.venv/bin/python scripts/make_schematic.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Same tokens as the report figures.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a84"
GRID = "#e5e4e0"
PANEL = "#f5f5f2"
BLUE = "#2a78d6"     # analysis: runs once
ORANGE = "#eb6834"   # the cache
AQUA = "#15966a"     # reporting: re-runs freely (darkened aqua for text contrast)
VIOLET = "#4a3aa7"   # what you provide

W, H = 180.0, 112.0
LINE = 3.2           # text line pitch inside boxes


def box(ax, x, y, w, h, title, edge, lw=1.5, face=SURFACE, title_color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))
    if title:
        ax.text(x + 1.8, y + h - 2.4, title, fontsize=9.0, color=title_color or edge,
                va="top", ha="left", fontweight="bold", zorder=4)


def lines(ax, x, y, rows, fs=7.8, color=INK2, mono=False, pitch=LINE):
    """Rows of text top-down from (x, y). A row may be a (text, color) pair."""
    for i, row in enumerate(rows):
        text, c = row if isinstance(row, tuple) else (row, color)
        ax.text(x, y - i * pitch, text, fontsize=fs, color=c, va="top", ha="left", zorder=4,
                family="monospace" if mono else None)


def arrow(ax, p0, p1, color=INK2, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                                 color=color, zorder=2, linestyle=ls, shrinkA=1, shrinkB=1))


def main(out="FRAME_schematic.pdf"):
    fig, ax = plt.subplots(figsize=(16.8, 16.8 * H / W), facecolor=SURFACE)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ------------------------------------------------------------------------------ header ---
    ax.text(2, H - 2.0, "FRAME — exoskeleton requirements from motion capture", fontsize=17,
            color=INK, fontweight="bold", va="top")
    ax.text(2, H - 7.6, "A folder of Xsens recordings in; one distribution per task type out. "
                        "Read left to right: the cache in the middle decides whether a change "
                        "costs hours or seconds.", fontsize=9.4, color=INK2, va="top")

    # ---------------------------------------------------------------------- column headers ---
    hy = 99.0
    ax.text(2, hy, "YOU PROVIDE", fontsize=9.4, color=VIOLET, fontweight="bold")
    ax.text(40, hy, "ANALYSE  ·  once per recording, ~80 s each", fontsize=9.4, color=BLUE,
            fontweight="bold")
    ax.text(108, hy, "CACHE", fontsize=9.4, color=ORANGE, fontweight="bold", ha="center")
    ax.text(124, hy, "REPORT  ·  seconds, re-run any time", fontsize=9.4, color=AQUA,
            fontweight="bold")
    sub = 95.8
    ax.text(40, sub, "frame-kinematics + utils/  (Pinocchio, in Docker)  →  frame-core",
            fontsize=7.6, color=MUTED)
    ax.text(108, sub, "frame-batch", fontsize=7.6, color=MUTED, ha="center")
    ax.text(124, sub, "frame-batch  ·  frame_core.pooling", fontsize=7.6, color=MUTED)

    # the cache band
    ax.add_patch(Rectangle((97.0, 25.0), 22.0, 69.0, facecolor="#fff5f0", edgecolor="none",
                           zorder=0))
    for xb in (97.0, 119.0):
        ax.plot([xb, xb], [25.0, 94.0], color=ORANGE, lw=1.0, ls=(0, (5, 3)), zorder=1)

    # ------------------------------------------------------------------------------ inputs ---
    box(ax, 2, 79.0, 32, 13.5, "models/", VIOLET)
    lines(ax, 3.8, 86.7, [
        "dhm_new.xacro       DHM template",
        "exoskeleton/*.urdf  for coverage",
    ], mono=True, fs=7.2)

    box(ax, 2, 53.0, 32, 21.5, "<data_root>/   read-only", VIOLET)
    lines(ax, 3.8, 68.7, [
        "<subject>/",
        "  <subject>_<task>_…_<rep>.mvnx",
        "  anthropo_data_<subject>.csv",
        ("  height (cm) · weight (kg)", MUTED),
    ], mono=True, fs=7.2, pitch=3.6)

    box(ax, 2, 19.0, 32, 30.5, "batch.json", VIOLET)
    lines(ax, 3.8, 43.7, [
        "data_root · cache_root",
        "output_root",
        "trial_pattern · pool_by",
        "joints_of_interest",
        "figures · figure_format",
        "coverage_exo_urdf · jobs",
    ], mono=True, fs=7.2)
    lines(ax, 3.8, 23.4, [("every key here: re-report only", VIOLET)], fs=7.2)

    # ---------------------------------------------------------------------------- analysis ---
    box(ax, 40, 79.0, 52, 13.5, "①  per SUBJECT — build the body model once", BLUE, lw=1.7)
    lines(ax, 41.8, 86.7, [
        "height + mass + one .mvnx + dhm_new.xacro",
        "→  models/subject_<id>.urdf     one model per person",
    ])

    box(ax, 40, 34.0, 52, 40.5, "②  per TRIAL — the heavy chain", BLUE, lw=1.7)
    chain = [
        (".mvnx  →  movement CSV", "utils/process_mvnx_file"),
        ("  →  inverse kinematics", "utils/ID_Xsens  (Pink)"),
        ("  →  inverse dynamics + GRF", "Pinocchio"),
        ("  =  Capture", "angles · torques · hand path"),
    ]
    for i, (left, right) in enumerate(chain):
        yy = 68.7 - i * LINE
        ax.text(41.8, yy, left, fontsize=7.8, color=INK2, va="top", zorder=4)
        ax.text(68.5, yy, right, fontsize=7.4, color=MUTED, va="top", zorder=4)
    ax.plot([41.8, 90.2], [55.0, 55.0], color=GRID, lw=1.0, zorder=4)
    lines(ax, 41.8, 53.2, [
        "frame_core.analyze  →  5 metric layers",
        ("  task space · joint space · dynamics · synthesis · PCA", MUTED),
        "flatten  →  one row of ~1000 metrics",
        ("  + subject-normalised __norm twins", MUTED),
    ])
    lines(ax, 41.8, 39.2, [("no Pinocchio past this line: frame-core is pure NumPy/SciPy", MUTED)],
          fs=7.0)

    # ------------------------------------------------------------------------------ cache ---
    box(ax, 98.5, 34.0, 19.0, 58.5, "per trial", ORANGE, lw=1.8)
    lines(ax, 100.3, 86.7, [
        ("<trial>.json", INK),
        "  every metric",
        "  ~60 KB",
        "",
        ("<trial>.npz", INK),
        "  the Capture",
        "  ~5 MB",
        "",
        ("keyed by", INK),
        "  recording",
        "  + analysis settings",
        "  + height & mass",
        "",
        ("restartable:", ORANGE),
        ("  resumes where", ORANGE),
        ("  it stopped", ORANGE),
    ], fs=7.4)

    # ----------------------------------------------------------------------------- report ---
    box(ax, 124, 80.0, 54, 12.5, "③  keep only joints_of_interest", AQUA, lw=1.7)
    lines(ax, 125.8, 86.7, [
        "a reporting filter: all 43 joints stay analysed,",
        "so widening the list later costs no compute",
    ])

    box(ax, 124, 63.0, 54, 13.0, "④  pool_by  →  one group per task", AQUA, lw=1.7)
    lines(ax, 125.8, 70.2, [
        "median · IQR · min · max · n for every metric,",
        "each value traceable to its recording",
    ])

    box(ax, 124, 30.0, 26, 29.5, "⑤  tables", AQUA)
    lines(ax, 125.8, 53.7, [
        "summary.md", "trials.csv", "pooled.json", "pooled_summary.csv",
        "differences.csv", "joint_summary.csv", "exo_eval/requirements",
    ], mono=True, fs=7.0, pitch=3.3)

    box(ax, 152, 30.0, 26, 29.5, "⑤  figures  ·  PDF", AQUA)
    lines(ax, 153.8, 53.7, [
        "ee_trajectory_3d", "joint_timeseries", "npose_torque_map", "torque_speed",
        "peak/rms_torque", "joint_speed", "shoulder_globe",
    ], mono=True, fs=7.0, pitch=3.3)

    # ----------------------------------------------------------------------------- arrows ---
    arrow(ax, (34, 85.8), (40, 85.8), color=VIOLET)          # xacro   -> per subject
    arrow(ax, (34, 72.0), (40, 81.5), color=VIOLET)          # dataset -> per subject
    arrow(ax, (34, 63.0), (40, 63.0), color=VIOLET)          # dataset -> per trial
    arrow(ax, (66, 79.0), (66, 74.5), color=BLUE)
    ax.text(67.5, 76.8, "the model every trial's IK uses", fontsize=7.2, color=BLUE, va="center")
    arrow(ax, (92, 56.0), (98.5, 56.0), color=BLUE, lw=1.7)
    ax.text(94.3, 54.8, "writes\nonce", fontsize=6.9, color=BLUE, ha="center", va="top")
    arrow(ax, (117.5, 86.0), (124, 86.0), color=AQUA, lw=1.7)
    arrow(ax, (117.5, 69.5), (124, 69.5), color=AQUA, lw=1.7)
    arrow(ax, (151, 80.0), (151, 76.0), color=AQUA)
    arrow(ax, (137, 63.0), (137, 59.5), color=AQUA)
    arrow(ax, (165, 63.0), (165, 59.5), color=AQUA)

    # the cached Captures feed the curve figures directly, bypassing the pooled scalars
    ax.plot([108, 108, 176], [34.0, 26.5, 26.5], color=ORANGE, lw=1.2, ls=(0, (4, 2)),
            zorder=2)
    arrow(ax, (176, 26.5), (176, 30.0), color=ORANGE, ls=(0, (4, 2)))
    ax.text(142, 25.2, "the .npz files also feed the curve, torque-speed, N-pose and coverage "
                       "figures directly", fontsize=7.2, color=ORANGE, ha="center", va="top")

    # ------------------------------------------------------------------- cost + commands ---
    ax.add_patch(FancyBboxPatch((2, 1.0), 115, 16.0, boxstyle="round,pad=0,rounding_size=1.3",
                                linewidth=1.0, edgecolor=GRID, facecolor=PANEL, zorder=3))
    ax.text(3.8, 15.4, "What a change costs", fontsize=9.0, color=INK, fontweight="bold",
            va="top", zorder=4)
    ax.text(3.8, 11.2, "re-analysis (hours)", fontsize=8.0, color=BLUE, fontweight="bold",
            va="top", zorder=4)
    lines(ax, 30.0, 11.2, [
        "an advanced analysis setting (dof_names, ee_segment, trim_*, voxel_m, "
        "exo_urdf …),",
        "or a corrected height / weight — then only that subject is re-analysed",
    ], fs=7.6, pitch=2.7)
    ax.text(3.8, 4.8, "re-report (seconds)", fontsize=8.0, color=AQUA, fontweight="bold",
            va="top", zorder=4)
    ax.text(30.0, 4.8, "every key in batch.json, and every figure / table option",
            fontsize=7.6, color=INK2, va="top", zorder=4)

    ax.add_patch(FancyBboxPatch((124, 1.0), 54, 16.0, boxstyle="round,pad=0,rounding_size=1.3",
                                linewidth=1.0, edgecolor=GRID, facecolor=PANEL, zorder=3))
    cmds = [("./frame build", "build the image, once"),
            ("./frame list", "check the dataset"),
            ("./frame run --jobs 6", "analyse + report"),
            ("./frame report", "re-report from the cache")]
    for i, (c, what) in enumerate(cmds):
        yy = 14.6 - i * 3.3
        ax.text(125.8, yy, c, fontsize=7.6, color=INK, family="monospace", va="top", zorder=4)
        ax.text(151.0, yy, what, fontsize=7.4, color=INK2, va="top", zorder=4)

    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return str(Path(out).resolve())


if __name__ == "__main__":
    print("wrote", main())
