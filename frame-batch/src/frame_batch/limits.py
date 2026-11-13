"""
Joint capacity limits, read back from the analysed captures.

The DHM's URDF carries a torque capacity (`effort`, N.m) and a speed limit (`velocity`, rad/s) for
every joint. Those are what turn a torque in newton-metres into a statement about whether the task
is near the limit of what the joint can do -- which is the comparison a box plot of torques alone
cannot make.

They are NOT in the flat metric table (only the derived ratios are), but they ride in every cached
Capture, so a reporting pass can read them without re-running anything.

One thing this module checks rather than assumes: whether the limits are the same for every
subject. In the current models they are -- they come from the fixed xacro base, not from the
per-subject scaling, so a 52 kg and a 98 kg person get the same 51 N.m shoulder capacity. A single
reference line is therefore honest. If that ever stops being true the figure would be drawing one
subject's limit across everyone's data, so `varies` is reported and the caller says so.
"""
from __future__ import annotations

import numpy as np

RAD_TO_DEG = 57.29577951308232

# Which limit a plotted metric should be compared against.
LIMIT_FOR_METRIC = {
    "peak_torque_Nm": ("effort", "N.m"),
    "rms_torque_Nm": ("effort", "N.m"),
    "peak_vel_dps": ("velocity_dps", "deg/s"),
    "rms_vel_dps": ("velocity_dps", "deg/s"),
}


def collect(records, cache, dofs=None, max_subjects=None, cfg=None):
    """{dof: {"effort", "velocity_dps", "lower_deg", "upper_deg", "n_subjects", "varies"}} from the
    cached Captures (effort N.m, velocity deg/s, range-of-motion bounds deg).

    Reads ONE capture per subject -- the limits are a property of the model, not of the recording,
    so there is nothing to gain from opening 998 files.
    """
    from .captures import load

    seen_subjects = set()
    derived = False          # pose-dependent capacities (shoulder elevation coordinates) differ
    per_dof: dict = {}       # per recording, so then every recording is read, not one per subject
    for rec in records:
        sid = rec.get("subject")
        if sid in seen_subjects and not derived:
            continue
        cap = load(cache, rec, cfg)
        if cap is None:
            continue
        lim = cap.urdf_limits or {}
        if not lim:
            continue
        derived = derived or any(isinstance(e, dict) and "capacity_basis" in e
                                 for d, e in lim.items() if dofs is None or d in dofs)
        seen_subjects.add(sid)
        for dof, e in lim.items():
            if dofs is not None and dof not in dofs:
                continue
            if not isinstance(e, dict):
                continue
            slot = per_dof.setdefault(dof, {"effort": [], "velocity_dps": [],
                                            "lower_deg": [], "upper_deg": []})
            for k in ("lower", "upper"):
                if e.get(k) is not None:
                    slot[f"{k}_deg"].append(float(e[k]) * RAD_TO_DEG)
            if e.get("effort"):
                slot["effort"].append(float(e["effort"]))
            if e.get("velocity"):
                slot["velocity_dps"].append(float(e["velocity"]) * RAD_TO_DEG)
        if max_subjects and len(seen_subjects) >= max_subjects and not derived:
            break

    out = {}
    for dof, slot in per_dof.items():
        entry = {"n_subjects": len(seen_subjects), "varies": False}
        for key in ("lower_deg", "upper_deg"):
            vals = [v for v in slot[key] if np.isfinite(v)]
            entry[key] = float(np.median(vals)) if vals else None
            if vals and max(vals) - min(vals) > 1e-6 * max(1.0, max(abs(v) for v in vals)):
                entry["varies"] = True
        for key in ("effort", "velocity_dps"):
            vals = [v for v in slot[key] if np.isfinite(v) and v > 0]
            if not vals:
                entry[key] = None
                continue
            # median, and a flag if the subjects disagree by more than a rounding error
            entry[key] = float(np.median(vals))
            if max(vals) - min(vals) > 1e-6 * max(abs(v) for v in vals):
                entry["varies"] = True
        entry["derived"] = derived
        out[dof] = entry
    return out


def for_metric(limits, dof, metric_suffix):
    """The limit value and its unit for one (dof, metric), or (None, None) if there is none."""
    kind_unit = LIMIT_FOR_METRIC.get(metric_suffix)
    if not kind_unit or not limits:
        return None, None
    kind, unit = kind_unit
    value = (limits.get(dof) or {}).get(kind)
    # A unit without a value is a half-answer the caller would have to guard against separately.
    return (value, unit) if value else (None, None)


def any_varies(limits):
    return any((e or {}).get("varies") for e in (limits or {}).values())
