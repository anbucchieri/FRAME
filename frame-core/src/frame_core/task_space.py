"""
Layer 1 -- task-space (Cartesian) requirements.

Quantitative counterparts to the 3D workspace visualisation. Everything is derived from the
end-effector Cartesian trajectory (analysis frame). The occupied-volume metric is voxel-based,
NOT convex-hull: the hull fills unreachable pockets and overstates the volume ~5.5x (measured),
so the reported number must be voxel-occupied volume. The hull stays a visual only.
"""
from __future__ import annotations
import numpy as np
from .signal import speed, accel_mag
from .types import summarize


def voxel_keys(cloud, voxel):
    """Integer voxel indices for each point; the set of unique rows is the occupied set."""
    return np.floor(np.asarray(cloud, float) / voxel).astype(np.int64)


def voxel_downsample(cloud, voxel):
    """One representative point per occupied voxel (uniform-density reach map)."""
    if voxel is None or voxel <= 0:
        return np.asarray(cloud, float)
    keys = voxel_keys(cloud, voxel)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.asarray(cloud, float)[idx]


def occupied_volume(cloud, voxel):
    """Occupied workspace volume [L] = (#occupied voxels) * voxel^3 * 1000."""
    keys = voxel_keys(cloud, voxel)
    n_occ = np.unique(keys, axis=0).shape[0]
    return n_occ * voxel ** 3 * 1e3


def bbox(cloud):
    """Axis-aligned bounding-box extents (dx, dy, dz) [m]."""
    cloud = np.asarray(cloud, float)
    return cloud.max(0) - cloud.min(0)


def coverage_iou(human_cloud, exo_cloud, voxel):
    """Voxelised coverage of the human end-effector cloud by the exo tip cloud.

    Returns dict with:
      coverage : fraction of human voxels that the exo can also reach   (|H n E| / |H|)
      iou      : Jaccard overlap of the two occupied voxel sets         (|H n E| / |H u E|)
      waste    : fraction of exo voxels outside the human demand        (|E \\ H| / |E|)
    """
    H = set(map(tuple, np.unique(voxel_keys(human_cloud, voxel), axis=0)))
    E = set(map(tuple, np.unique(voxel_keys(exo_cloud, voxel), axis=0)))
    if not H or not E:
        return {"coverage": float("nan"), "iou": float("nan"), "waste": float("nan")}
    inter = len(H & E)
    union = len(H | E)
    return {
        "coverage": inter / len(H),
        "iou": inter / union,
        "waste": len(E - H) / len(E),
    }


def hull_coverage(human_cloud, exo_cloud):
    """Fraction of human end-effector samples that fall INSIDE the exo reach hull.

    This is the headline task-space overlap. It replaces voxel intersection for the human-vs-exo
    comparison because the exo cloud is a SAMPLED lattice (a joint-space grid pushed through FK),
    not a filled volume: on the reference model it occupies only ~15% of its own bounding box at
    3.8 points per 2 cm voxel, so |H n E| counts gaps between samples as unreachable. The symptom
    is a coverage that moves with the voxel size (29% at 2 cm -> 58% at 5 cm) rather than with the
    geometry. Containment in the hull is resolution-independent.

    The trade-off is the one this module's header already flags for the human volume: a convex
    hull fills concave pockets, so this is an UPPER bound on reachability. Report it alongside the
    voxel numbers, never instead of them.

    Returns {"hull_coverage", "n_inside", "n_total", "exo_hull_volume_L"}; NaN if the exo cloud is
    too small or degenerate to trianglulate.
    """
    H = np.asarray(human_cloud, float)
    E = np.asarray(exo_cloud, float)
    out = {"hull_coverage": float("nan"), "n_inside": 0, "n_total": int(H.shape[0]),
           "exo_hull_volume_L": float("nan")}
    if H.size == 0 or E.shape[0] < 4:
        return out
    try:
        from scipy.spatial import Delaunay, ConvexHull
        hull = ConvexHull(E)
        # Triangulate the hull's VERTICES, not the whole cloud. Containment in a convex hull is
        # decided entirely by its vertices, so this is the same test -- but the exo cloud is a
        # joint-space sweep of up to a few hundred thousand samples whose hull has a few hundred
        # vertices, and a 3-D Delaunay of the full cloud costs minutes and gigabytes where the
        # reduced one costs milliseconds. Same answer, three orders of magnitude cheaper.
        inside = Delaunay(E[hull.vertices]).find_simplex(H) >= 0
        out["hull_coverage"] = float(inside.mean())
        out["n_inside"] = int(inside.sum())
        out["exo_hull_volume_L"] = float(hull.volume * 1e3)
    except Exception:
        pass                                   # degenerate hull -> leave the NaNs
    return out


def dwell_concentration(cloud, voxel, top_frac=0.10):
    """How concentrated the end-effector residence is: fraction of samples that fall in the
    busiest `top_frac` of occupied voxels. ~top_frac => uniform; >>top_frac => the task dwells
    in a small region (where assistance/stiffness matters most)."""
    keys = voxel_keys(cloud, voxel)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    counts = np.sort(counts)[::-1]
    k = max(1, int(np.ceil(top_frac * len(counts))))
    return float(counts[:k].sum() / counts.sum())


def compute(cap, cfg):
    """Layer-1 result dict. Requires cap.has_ee."""
    voxel = cfg.voxel_m
    # pooled cloud across reps (analysis frame)
    pooled = np.vstack([r.ee_pos for r in cap.reps])

    # per-rep scalars -> distributions
    peak_speed, peak_acc, reach_max, rep_vol = [], [], [], []
    for r in cap.reps:
        peak_speed.append(float(np.max(speed(r.ee_pos, cap.fs))))
        peak_acc.append(float(np.max(accel_mag(r.ee_pos, cap.fs))))
        reach_max.append(float(np.max(np.linalg.norm(r.ee_pos, axis=1))))
        rep_vol.append(occupied_volume(r.ee_pos, voxel))

    ext = bbox(pooled)
    out = {
        "ee_segment": cfg.ee_segment,
        "workspace_volume_L": occupied_volume(pooled, voxel),   # pooled envelope
        "bbox_m": {"x": float(ext[0]), "y": float(ext[1]), "z": float(ext[2])},
        "dwell_concentration": dwell_concentration(pooled, voxel),
        "peak_speed_mps": summarize(peak_speed),
        "peak_accel_mps2": summarize(peak_acc),
        "reach_max_m": summarize(reach_max),
        "per_rep_volume_L": summarize(rep_vol),
        "cloud_points": int(pooled.shape[0]),
    }

    # assessment overlay: hull containment is the headline, voxel numbers stay alongside it
    if cap.exo_cloud is not None:
        cov = coverage_iou(pooled, cap.exo_cloud, voxel)
        cov.update(hull_coverage(pooled, cap.exo_cloud))
        cov["headline"] = "hull_coverage"
        out["coverage"] = cov
    return out


def convex_hull_mesh(cloud):
    """Triangulated convex hull of a point cloud, for rendering.

    Returns {"vertices": [[x,y,z], ...], "faces": [[i,j,k], ...], "volume_L": float} with faces
    indexing into `vertices`, or None when the cloud is too small or degenerate to triangulate.

    This is the SAME hull `hull_coverage` measures containment against, so the surface drawn in
    the scene and the coverage number quoted beside it describe one object. Only the hull's own
    vertices are emitted (typically a few hundred out of tens of thousands of samples), which is
    why the surface can be sent at full precision where the raw cloud has to be downsampled.
    """
    P = np.asarray(cloud, float)
    P = P[np.isfinite(P).all(axis=1)]
    if P.shape[0] < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(P)
    except Exception:
        return None                              # coplanar / degenerate -> no surface to draw
    # compact the vertex list: hull.simplices index into the full cloud, which we do not send
    keep = np.unique(hull.simplices)
    remap = np.full(P.shape[0], -1, int)
    remap[keep] = np.arange(keep.size)
    return {
        "vertices": P[keep].tolist(),
        "faces": remap[hull.simplices].tolist(),
        "volume_L": float(hull.volume * 1e3),
        "area_m2": float(hull.area),
        "n_source_points": int(P.shape[0]),
    }
