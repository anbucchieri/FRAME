"""
Capture-frame helpers -- pure pandas, no Pinocchio.

Loading and edge trimming only. A recording is never segmented: it is analysed as one continuous
movement from its first to its last kept sample.
"""
from __future__ import annotations
import pandas as pd


def load_capture_frame(csv_path, drop_units_header=True):
    """Load a capture CSV. The Xsens export's first data row is a units header (dropped)."""
    df = pd.read_csv(csv_path)
    if drop_units_header and len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def trim_edges(df, fs, start_s=0.0, end_s=0.0, min_frames=30, min_keep_frac=0.25):
    """Drop the first `start_s` and last `end_s` seconds of a capture.

    Mocap records routinely open and close with the subject settling, the operator in frame, or
    the suit still converging, and those frames produce spikes that are instrumental rather than
    behavioural. Trimming HERE -- before IK -- means nothing downstream ever sees them: not the
    angles, not the torques, not the metrics, not the figures.

    Two guards, because `min_frames` alone is not enough: on a 6.5 s record a 3 + 3 s trim leaves
    126 frames, which clears any sane frame floor while being half a second of data. The FRACTION
    guard is the one that catches a trim sized for a different recording length.

    Raises ValueError rather than returning a stub -- a silently near-empty capture still produces
    plausible-looking numbers, which is the worst outcome.
    """
    n = len(df)
    a = int(round(max(0.0, start_s) * fs))
    b = n - int(round(max(0.0, end_s) * fs))
    kept = max(b - a, 0)
    if kept < min_frames or kept < min_keep_frac * n:
        raise ValueError(
            f"trim_start_s={start_s:g} + trim_end_s={end_s:g} removes {start_s + end_s:g} s from a "
            f"{n / fs:.2f} s recording, leaving {kept} of {n} frames ({kept / fs:.2f} s, "
            f"{100 * kept / n:.0f}%). Keep at least {100 * min_keep_frac:.0f}% and {min_frames} "
            f"frames: lower job.trim_start_s / job.trim_end_s, or record for longer.")
    return df.iloc[a:b].reset_index(drop=True), (a, b)
