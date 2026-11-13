"""
Segment trajectories straight from the .mvnx -- the cheap way to look at a point the analysis
does not track.

The sweep stores a per-frame Cartesian trajectory for exactly ONE point: the configured
`ee_segment`, which is the forearm, because the forearm is where the exoskeleton's end-effector
attaches. Looking at any other point -- the palm holding the tool, say -- would normally mean
re-running the whole heavy path.

It does not have to. Segment positions are raw .mvnx content: no IK, no inverse dynamics, no URDF.
A targeted `iterparse` pulls all 23 segments' positions and orientations out of a 110 MB capture in
well under a second, against ~90 s for a full analysis. So this module reads what it needs
directly and caches it beside the trial, leaving the analysis cache -- and every metric in it --
untouched.

The frame transform is NOT reimplemented here. The extracted columns are shaped exactly like the
converted movement CSV, then handed to `frame_kinematics`'s own `xsens_trunk_convention` and
`ee_in_trunk`, so a trajectory from this module lands in the same trunk frame, with the same axis
convention, as the one the sweep produced.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MVNX_NS = "{http://www.xsens.com/mvn/mvnx}"

# Xsens writes orientation as (w, x, y, z); the movement CSV names those columns in that order,
# and the transforms read them by name. Keeping the same names is what lets the existing code run.
QUAT_SUFFIXES = ("_w_q", "_x_q", "_y_q", "_z_q")


def read_segment_frames(mvnx_path):
    """(segment_names, positions (N, S, 3), orientations (N, S, 4)) from an .mvnx.

    Only `type="normal"` frames are kept, which is the same set the project's converter keeps when
    it slices past 'identity', 'tpose' and 'tpose-isb' -- so frame i here is frame i there.
    Elements are cleared as they are consumed, so peak memory is the output, not the document.
    """
    names, pos, ori = None, [], []
    for _event, el in ET.iterparse(str(mvnx_path), events=("end",)):
        if el.tag == MVNX_NS + "segments" and names is None:
            names = [s.get("label") for s in el.iter(MVNX_NS + "segment")]
            el.clear()
        elif el.tag == MVNX_NS + "frame":
            if el.get("type") == "normal":
                p = el.find(MVNX_NS + "position")
                o = el.find(MVNX_NS + "orientation")
                if p is not None and p.text:
                    pos.append(np.fromstring(p.text, sep=" "))
                if o is not None and o.text:
                    ori.append(np.fromstring(o.text, sep=" "))
            el.clear()
    if not names or not pos:
        raise ValueError(f"{mvnx_path}: no segment frames found (is this an .mvnx?)")
    S = len(names)
    P = np.vstack(pos).reshape(-1, S, 3)
    O = np.vstack(ori).reshape(-1, S, 4) if ori else np.zeros((P.shape[0], S, 4))
    return names, P, O


def segment_dataframe(mvnx_path):
    """The extracted frames as the movement-CSV-shaped dataframe the transforms expect."""
    import pandas as pd

    names, P, O = read_segment_frames(mvnx_path)
    data = {}
    for i, name in enumerate(names):
        for k, axis in enumerate("XYZ"):
            data[f"{name}_{axis}"] = P[:, i, k]
        for k, suffix in enumerate(QUAT_SUFFIXES):
            data[f"{name}{suffix}"] = O[:, i, k]
    return pd.DataFrame(data)


def segment_in_trunk(mvnx_path, segment, trunk_segment="T8"):
    """One segment's trajectory in the canonical trunk frame, (N, 3).

    Uses frame-kinematics' own convention measurement and transform, so this is the same frame the
    sweep's `ee_pos` lives in -- not a parallel implementation that could drift from it.
    """
    from frame_kinematics.assemble import ee_in_trunk
    from frame_kinematics.canonical import xsens_trunk_convention

    df = segment_dataframe(mvnx_path)
    if f"{segment}_X" not in df.columns:
        have = sorted({c[:-2] for c in df.columns if c.endswith("_X")})
        raise ValueError(f"{mvnx_path}: no segment named {segment!r}. Available: {have}")
    convention = xsens_trunk_convention(df, trunk_segment)
    return np.asarray(ee_in_trunk(df, segment, trunk_segment, convention), float)


def cached_segment_trajectory(trial, subject, cache, segment, trunk_segment="T8"):
    """`segment_in_trunk` with a per-trial disk cache beside the analysis cache.

    Keyed by segment name, so switching which point you look at re-extracts that point only and
    leaves the rest -- and never touches the analysis records.
    """
    path = Path(cache.record_path(subject, trial.trial_id, create=True)).with_suffix(f".seg_{segment}.npz")
    if path.is_file():
        try:
            with np.load(path) as z:
                return z["xyz"]
        except (OSError, KeyError, ValueError):
            pass                                    # corrupt or truncated -> just redo it
    xyz = segment_in_trunk(trial.path, segment, trunk_segment)
    try:
        np.savez_compressed(path, xyz=xyz)
    except OSError:
        pass                                        # a read-only cache must not fail the figure
    return xyz
