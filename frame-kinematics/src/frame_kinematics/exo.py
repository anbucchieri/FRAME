"""
Exoskeleton reach (assessment add-on) -- thin wrapper over the vendored exo_workspace, pooling
both arm-tip clouds into one (K, 3) array in the trunk_attachment (== T8) frame for the
Capture. Pinocchio is imported lazily inside the call, so importing this module is free.
"""
from __future__ import annotations
import numpy as np


def compute_exo_cloud(job):
    """Pooled exo tip reach cloud (both arms), or None if no exo URDF was supplied."""
    if not job.exo_urdf:
        return None
    from .pinocchio_backend.exo_workspace import compute_exo_workspace
    clouds, _conv = compute_exo_workspace(job.exo_urdf, job.exo_trunk_frame, job.exo_arm_tips,
                                          extra_R=job.exo_extra_R, verbose=False,
                                          return_convention=True)
    return np.vstack(list(clouds.values())) if clouds else None
