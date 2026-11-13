"""
One way to read a cached Capture back for reporting.

The cache stores the RAW Capture the heavy path produced (DHM joint coordinates). Everything that
reports on it goes through `load`, which applies the configured shoulder coordinates
(`BatchConfig.shoulder_coords`), so no figure or table can end up mixing the DHM Euler angles with
the ISB elevation coordinates.
"""
from __future__ import annotations


def load(cache, rec, cfg=None):
    """The capture of one trial record, transformed for reporting, or None if not cached."""
    from frame_kinematics.serialize import capture_from_npz_bytes
    blob = cache.load_capture_bytes(rec.get("subject"), rec.get("trial_id"))
    if not blob:
        return None
    cap = capture_from_npz_bytes(blob)
    return cfg.transform_capture(cap) if cfg is not None else cap
