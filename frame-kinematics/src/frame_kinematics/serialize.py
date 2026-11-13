"""
Capture serialization -- how a trial's Capture is stored in the result cache.

npz is compact, exact and needs no Pinocchio to read back. A small manifest carries dof_names,
fs, availability flags and per-rep shapes so reconstruction needs no guessing. Figures are rebuilt
from these files, which is why re-reporting never re-runs IK.
"""
from __future__ import annotations
import io
import json
import numpy as np

from frame_core.types import Capture, Repetition


def capture_to_npz_bytes(cap, float32=False, decimate=1):
    """Serialize a Capture to npz bytes.

    `float32` and `decimate` trade stored precision for disk. They exist because a cached Capture
    is a means of REBUILDING FIGURES without re-running IK -- every scalar metric was already
    computed from the full-precision arrays before this is called. float32 keeps ~7 significant
    digits, which is well past what the angles and torques resolve. `decimate=N` keeps every Nth
    sample AND divides the stored `fs` by N, so a velocity derived from the reloaded capture is
    still in the right units; without that rescaling every speed would come back N times too small.
    """
    step = max(1, int(decimate))
    dtype = np.float32 if float32 else None

    def _store(a):
        a = np.asarray(a)
        if a.ndim >= 1 and step > 1:
            a = a[::step]
        return a.astype(dtype, copy=False) if dtype is not None and a.dtype.kind == "f" else a

    arrays, manifest = {}, {
        "dof_names": list(cap.dof_names), "fs": float(cap.fs) / step,
        "n_reps": len(cap.reps), "height_m": cap.height_m, "mass_kg": cap.mass_kg,
        "has_exo": cap.exo_cloud is not None, "has_ee_cloud": cap.ee_cloud is not None,
        "reps": [], "decimate": step, "float32": bool(float32),
    }
    for i, r in enumerate(cap.reps):
        flags = {}
        arrays[f"angles_{i}"] = _store(r.angles)
        flags["torques"] = r.torques is not None
        flags["gravity_torques"] = r.gravity_torques is not None
        flags["ee_pos"] = r.ee_pos is not None
        flags["jac"] = r.jac_contribution is not None
        if r.torques is not None:
            arrays[f"torques_{i}"] = _store(r.torques)
        if r.gravity_torques is not None:
            arrays[f"tau_g_{i}"] = _store(r.gravity_torques)
        if r.ee_pos is not None:
            arrays[f"ee_pos_{i}"] = _store(r.ee_pos)
        if r.jac_contribution is not None:
            # per-DOF scalars, not a time series -- never decimated
            arrays[f"jac_{i}"] = np.asarray(r.jac_contribution)
        manifest["reps"].append(flags)
    if cap.ee_cloud is not None:
        arrays["ee_cloud"] = _store(cap.ee_cloud)
    if cap.exo_cloud is not None:
        arrays["exo_cloud"] = cap.exo_cloud

    # skeleton for the torque map: names/bones are JSON, the poses ride as arrays
    manifest["urdf_limits"] = cap.urdf_limits
    manifest["trim"] = getattr(cap, "trim", None)
    sk = cap.skeleton or {}
    manifest["skeleton"] = None if not sk else {
        "segment_names": list(sk["segment_names"]),
        "bones": [list(b) for b in sk["bones"]],
        "has_pose_mean": sk.get("pose_mean") is not None,
        "has_pose_npose": sk.get("pose_npose") is not None,
    }
    if sk.get("pose_mean") is not None:
        arrays["skeleton_pose_mean"] = np.asarray(sk["pose_mean"], float)
    if sk.get("pose_npose") is not None:
        arrays["skeleton_pose_npose"] = np.asarray(sk["pose_npose"], float)

    buf = io.BytesIO()
    np.savez_compressed(buf, manifest=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8), **arrays)
    return buf.getvalue()


def capture_from_npz_bytes(b):
    """Reconstruct a Capture from npz bytes."""
    z = np.load(io.BytesIO(b), allow_pickle=False)
    manifest = json.loads(bytes(z["manifest"]).decode())
    reps = []
    for i, flags in enumerate(manifest["reps"]):
        reps.append(Repetition(
            angles=z[f"angles_{i}"],
            torques=z[f"torques_{i}"] if flags["torques"] else None,
            # absent in captures cached before the gravity torque existed -> None (G is NaN)
            gravity_torques=z[f"tau_g_{i}"] if flags.get("gravity_torques") else None,
            ee_pos=z[f"ee_pos_{i}"] if flags["ee_pos"] else None,
            jac_contribution=z[f"jac_{i}"] if flags["jac"] else None,
        ))
    sk_meta = manifest.get("skeleton")
    skeleton = None if not sk_meta else {
        "segment_names": sk_meta["segment_names"],
        "bones": sk_meta["bones"],
        "pose_mean": z["skeleton_pose_mean"] if sk_meta["has_pose_mean"] else None,
        "pose_npose": z["skeleton_pose_npose"] if sk_meta["has_pose_npose"] else None,
    }
    cap = Capture(
        reps=reps, dof_names=manifest["dof_names"], fs=manifest["fs"],
        ee_cloud=z["ee_cloud"] if manifest["has_ee_cloud"] else None,
        exo_cloud=z["exo_cloud"] if manifest["has_exo"] else None,
        height_m=manifest["height_m"], mass_kg=manifest["mass_kg"],
        skeleton=skeleton, urdf_limits=manifest.get("urdf_limits"),
    )
    cap.trim = manifest.get("trim")
    # what the stored samples are: exo-eval refuses decimated captures (native-rate rule)
    cap.decimate = int(manifest.get("decimate", 1) or 1)
    return cap
