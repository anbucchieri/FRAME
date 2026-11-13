"""
Per-segment torque load for the N-pose torque map.

Reduces one Capture to {segment: RMS torque / capacity}, which `frame_batch.npose` pools across
the trials of a task and paints onto the subject's neutral skeleton.
"""
from __future__ import annotations
import re

import numpy as np

UPPER_BODY_PATTERN = r"lumbarspine|thoracicspine|spine|neck|head|clavicular|shoulder|elbow|wrist"


def npose_torque_map(cap, pattern=UPPER_BODY_PATTERN):
    """Per-SEGMENT torque summary for the N-pose map: {segment: {...}}.

    `pattern` filters which DOFs get a sphere (None/"" for every mapped joint).

    Torque is per DOF, but the figure paints body segments, so DOFs sharing a segment are
    collapsed by taking the WORST one -- a shoulder whose X axis runs at 80% of capacity is a
    loaded shoulder even if its Y axis is idle. `ratio` is RMS torque over the URDF effort limit;
    when no capacity is recorded it falls back to RMS normalised by the largest RMS present, and
    `has_capacity` says which of the two the colour means.
    """
    from frame_core.anatomy import dof_to_segment

    if not cap.reps or cap.reps[0].torques is None:
        return {}
    T = np.vstack([r.torques for r in cap.reps])
    limits = cap.urdf_limits or {}
    rx = re.compile(pattern, re.I) if pattern else None
    per_seg = {}
    for j, name in enumerate(cap.dof_names):
        seg = dof_to_segment(name)
        if seg is None or (rx is not None and not rx.search(name)):
            continue
        rms = float(np.sqrt(np.mean(T[:, j] ** 2)))
        peak = float(np.max(np.abs(T[:, j])))
        eff = (limits.get(name) or {}).get("effort")
        ratio = (rms / eff) if eff else None
        cur = per_seg.get(seg)
        key = ratio if ratio is not None else rms
        if cur is None or key > cur["_key"]:
            per_seg[seg] = {"_key": key, "dof": name, "rms": rms, "peak": peak,
                            "capacity": eff, "ratio": ratio}
    if per_seg and all(v["ratio"] is None for v in per_seg.values()):
        top = max(v["rms"] for v in per_seg.values()) or 1.0
        for v in per_seg.values():
            v["ratio"] = v["rms"] / top
        for v in per_seg.values():
            v["has_capacity"] = False
    else:
        for v in per_seg.values():
            v["has_capacity"] = v["capacity"] is not None
    return per_seg
