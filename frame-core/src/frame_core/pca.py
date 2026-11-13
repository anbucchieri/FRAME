"""
Layer 5 -- principal component analysis of the joint configuration.

Which joints move TOGETHER, and how many independent coordination patterns the task actually
needs. A task whose variance collapses into two components is asking for far less than its DOF
count suggests -- directly relevant to how many actuated axes an exoskeleton needs.

Method:
    centre the (frames x nDOF) matrix, optionally z-score the columns, SVD, and read the variance
    accounted for from the singular values as  ev = S^2 / (n - 1),  VAF = ev / sum(ev).
    Component signs are fixed deterministically (largest-magnitude loading made positive) so a
    re-run cannot flip a bar chart.

Column z-scoring is ON by default here (correlation PCA). Without it a joint with large ROM
dominates purely because it is measured in the same units as everything else but swings further --
which answers "which joint moves most", not "which joints are recruited together".

`n_pc_90` is the number of components needed to reach VAF_THRESHOLD; every component is
returned so the threshold stays a reporting choice.
"""
from __future__ import annotations
import numpy as np

VAF_THRESHOLD = 0.90


def _flip_signs(U, Vt):
    """Deterministic sign: make each component's largest-magnitude loading positive."""
    idx = np.argmax(np.abs(Vt), axis=1)
    s = np.sign(Vt[np.arange(Vt.shape[0]), idx])
    s[s == 0] = 1.0
    return U * s, Vt * s[:, None]


def pca_svd(X, vaf_threshold=VAF_THRESHOLD, standardize=True):
    """Centred-SVD PCA. Returns (n_pc, Vt, vaf).

    X   : (n_frames, n_dof)
    Vt  : (k, n_dof) loadings, one row per component
    vaf : (k,) fraction of variance per component
    """
    a = np.asarray(X, float)
    a = a - a.mean(0)
    if standardize:
        sd = a.std(0)
        sd[sd < 1e-9] = 1.0
        a = a / sd
    U, S, Vt = np.linalg.svd(a, full_matrices=False)
    U, Vt = _flip_signs(U, Vt)
    ev = (S ** 2) / max(len(a) - 1, 1)
    total = ev.sum()
    vaf = ev / total if total > 0 else np.zeros_like(ev)
    n_pc = int(np.searchsorted(np.cumsum(vaf), vaf_threshold) + 1)
    return min(n_pc, len(vaf)), Vt, vaf


def engagement(Vt, vaf, n_pc):
    """How strongly each joint participates across the RETAINED components.

    VAF-weighted mean |loading| over PC1..PC_n_pc:

        engagement_j = sum_k vaf_k * |V_kj|  /  sum_k vaf_k

    Weighting by VAF matters: an unweighted mean lets PC7, which may explain 1% of the variance,
    count as much as PC1. This is the single number that answers "which joints are most engaged in
    the task", and it is what the debug figure sorts by.
    """
    k = max(1, int(n_pc))
    w = np.asarray(vaf, float)[:k]
    L = np.abs(np.asarray(Vt, float)[:k, :])
    wsum = w.sum()
    return (w @ L) / wsum if wsum > 0 else L.mean(0)


def compute(cap, cfg=None, vaf_threshold=VAF_THRESHOLD, standardize=True):
    """Layer-5 result dict from a Capture's pooled joint angles.

    Repetitions are stacked (they are samples of the same movement class), giving one coordination
    structure for the capture. Constant DOFs are dropped before the SVD -- a column with no
    variance contributes nothing but makes the correlation scaling singular -- and named in
    `dropped_dofs` so their absence is visible rather than silent.
    """
    names = list(cap.dof_names)
    X = np.vstack([r.angles for r in cap.reps]) if cap.reps else np.empty((0, len(names)))
    # Masked angles (shoulder plane / rotation near gimbal lock) -> listwise deletion: PCA needs
    # every DOF at the same instant. The fraction of frames used is reported.
    n_all = X.shape[0]
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] < 2 or X.shape[1] < 2:
        return {"ok": False, "reason": "PCA needs at least 2 DOFs and 2 frames",
                "dof_names": names, "components": [], "n_pc_90": 0}

    sd = X.std(0)
    keep = sd > 1e-9
    dropped = [n for n, k in zip(names, keep) if not k]
    kept = [n for n, k in zip(names, keep) if k]
    if len(kept) < 2:
        return {"ok": False, "reason": "fewer than 2 DOFs vary in this capture",
                "dof_names": names, "dropped_dofs": dropped, "components": [], "n_pc_90": 0}

    n_pc, Vt, vaf = pca_svd(X[:, keep], vaf_threshold, standardize)
    cum = np.cumsum(vaf)
    eng = engagement(Vt, vaf, n_pc)
    order = np.argsort(eng)[::-1]

    components = []
    for i in range(len(vaf)):
        load = Vt[i, :]
        rank = np.argsort(np.abs(load))[::-1]
        components.append({
            "pc": i + 1,
            "vaf": float(vaf[i]),
            "cumulative_vaf": float(cum[i]),
            "retained": bool(i < n_pc),
            "loadings": {kept[j]: float(load[j]) for j in range(len(kept))},
            "top_joints": [{"dof": kept[j], "loading": float(load[j]),
                            "abs": float(abs(load[j]))} for j in rank[:5]],
        })

    return {
        "ok": True,
        "dof_names": kept,
        "dropped_dofs": dropped,
        "standardized": bool(standardize),
        "vaf_threshold": float(vaf_threshold),
        "n_pc_90": int(n_pc),
        "frames_used_frac": float(X.shape[0] / n_all) if n_all else float("nan"),
        "explained_variance_ratio": vaf.tolist(),
        "cumulative_vaf": cum.tolist(),
        "components": components,
        "engagement": {kept[j]: float(eng[j]) for j in range(len(kept))},
        "engagement_ranking": [kept[j] for j in order],
        "n_frames": int(X.shape[0]),
    }
