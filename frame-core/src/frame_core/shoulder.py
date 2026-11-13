"""
Shoulder complex in ISB humerothoracic coordinates: plane of elevation, elevation, axial rotation.

Why: the DHM shoulder is an X -> Y -> Z Euler chain whose ZERO POSE IS A T-POSE (the humerus points
laterally along -Y for the right arm). What `shoulder_X/Y/Z` mean anatomically therefore changes with
the pose -- `shoulder_Y` is axial rotation at zero but not once X has turned -- so their angles,
speeds and torques cannot be read as flexion / abduction. This module re-expresses the shoulder in
the ISB recommendation for the humerus relative to the THORAX (Wu et al. 2005, Y-X-Y sequence):

    plane of elevation  phi    0 = abduction (frontal plane), +90 deg = forward flexion
    elevation           theta  0 = arm hanging, 90 = horizontal, 180 = overhead
    axial rotation      psi    0 = neutral (elbow flexion axis mediolateral), + = internal

H = Ry(phi) Rx(-theta) Ry(psi) is the humerus orientation relative to its anatomical neutral (arm
hanging, elbow axis mediolateral), in ISB thorax axes (X forward, Y up, Z lateral). The left arm is
MIRRORED into the same convention, so positive means the same movement on both sides.

Humerothoracic: the clavicle DOFs are included in the orientation (the exo spans thorax -> upper
arm). The TORQUES are the glenohumeral moment M (recovered exactly from the three GH joint torques,
tau_i = a_i . M) projected onto the three Euler axes,

    tau_phi = e_phi . M,  tau_theta = e_theta . M,  tau_psi = e_psi . M,

which are the generalised forces conjugate to the angle rates: M . omega = sum tau_k qdot_k. The
gravity torque is transformed identically.

Gimbal lock: at theta = 0 (and 180) phi and psi are undefined and their rates diverge. Below
`mask_deg` (default 20) the plane and rotation ANGLES are set to NaN; elevation and all three
torques are always kept.

Capacities are DERIVED from the DHM's per-axis URDF limits, never invented:
  * torque  max e_k . M subject to |tau_i| <= effort_i  ->  sum_i |(A^-1 e_k)_i| effort_i
  * speed   max |qdot_k| subject to |qdot_j| <= velocity_j over the 5 thorax->humerus joints
  * ROM     min / max of each coordinate over a grid of the 5-joint limit box
Torque and speed capacities depend on the pose; the per-recording MEDIAN is stored in
`urdf_limits` (key `capacity_basis` says so).
"""
from __future__ import annotations
from dataclasses import replace
from itertools import product

import numpy as np

COORDS = ("plane", "elevation", "rotation")
_SUFFIX_OF = {"X": "plane", "Y": "elevation", "Z": "rotation"}   # which column each replaces
MASK_DEG = 20.0

# The DHM chain thorax -> humerus, per side: (joint, axis) in order. Every joint origin carries no
# rotation (rpy 0), so the chain is a pure product of elementary rotations -- checked against the
# URDF in the tests (frame-kinematics: test_shoulder_chain_matches_the_urdf).
CHAINS = {
    "right": {"clav": (("right_clavicular_X", "x"), ("right_clavicular_Z", "z")),
              "gh": (("right_shoulder_X", "x"), ("right_shoulder_Y", "y"), ("right_shoulder_Z", "z")),
              "long_axis": (0.0, -1.0, 0.0), "neutral_x": np.pi / 2},
    "left": {"clav": (("left_clavicular_X", "x"), ("left_clavicular_Z", "z")),
             "gh": (("left_shoulder_X", "x"), ("left_shoulder_Y", "y"), ("left_shoulder_Z", "z")),
             "long_axis": (0.0, 1.0, 0.0), "neutral_x": -np.pi / 2},
}
# DHM thorax axes (X forward, Y left, Z up) -> ISB (X forward, Y up, Z lateral). The left map is a
# reflection: it mirrors the left arm into the right-arm convention.
_P = {"right": np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
      "left": np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])}


def _solve(A, B):
    """Batched solve that survives exactly singular samples (gimbal lock): pseudo-inverse there."""
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ B


def names(side):
    return [f"{side}_shoulder_{c}" for c in COORDS]


def _rot(axis, a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        R[..., 0, :] = np.stack([o, z, z], -1); R[..., 1, :] = np.stack([z, c, -s], -1)
        R[..., 2, :] = np.stack([z, s, c], -1)
    elif axis == "y":
        R[..., 0, :] = np.stack([c, z, s], -1); R[..., 1, :] = np.stack([z, o, z], -1)
        R[..., 2, :] = np.stack([-s, z, c], -1)
    else:
        R[..., 0, :] = np.stack([c, -s, z], -1); R[..., 1, :] = np.stack([s, c, z], -1)
        R[..., 2, :] = np.stack([z, z, o], -1)
    return R


def kinematics(side, clav, gh):
    """Everything per sample for one side.

    clav : (N, 2) clavicle angles [X, Z]; gh : (N, 3) shoulder angles [X, Y, Z] (rad).
    Returns dict with angles (N, 3) [phi, theta, psi] (unmasked), E (N, 3, 3) Euler axes in ISB
    thorax coordinates (columns e_phi, e_theta, e_psi), A (N, 3, 3) GH joint axes in DHM thorax
    coordinates, B (N, 3, 5) all five joint axes (clavicle X, Z, GH X, Y, Z), P, detP.
    """
    ch = CHAINS[side]
    cx, cz = clav[:, 0], clav[:, 1]
    sx, sy, sz = gh[:, 0], gh[:, 1], gh[:, 2]
    Rcx = _rot("x", cx)
    Rc = Rcx @ _rot("z", cz)
    R1 = Rc @ _rot("x", sx)
    R2 = R1 @ _rot("y", sy)
    Rh = R2 @ _rot("z", sz)
    Rn = _rot("x", np.array(ch["neutral_x"]))
    P = _P[side]
    detP = float(np.linalg.det(P))
    H = P @ (Rh @ Rn.T) @ P.T
    theta = np.arccos(np.clip(H[:, 1, 1], -1.0, 1.0))
    phi = np.arctan2(-H[:, 0, 1], -H[:, 2, 1])
    psi = np.arctan2(-H[:, 1, 0], H[:, 1, 2])
    E = np.zeros_like(H)
    E[:, 1, 0] = 1.0                                               # e_phi   = thorax Y
    E[:, 0, 1] = -np.cos(phi); E[:, 2, 1] = np.sin(phi)            # e_theta = Ry(phi) (-X)
    E[:, :, 2] = H[:, :, 1]                                        # e_psi   = humerus Y
    A = np.stack([Rc[:, :, 0], R1[:, :, 1], R2[:, :, 2]], axis=2)
    ex = np.broadcast_to(np.array([1.0, 0, 0]), (len(cx), 3))
    B = np.stack([ex, Rcx[:, :, 2], A[:, :, 0], A[:, :, 1], A[:, :, 2]], axis=2)
    return {"angles": np.column_stack([phi, theta, psi]), "E": E, "A": A, "B": B,
            "P": P, "detP": detP}


def project_torque(k, tau_gh):
    """GH joint torques (N, 3) -> generalised torques (N, 3) on (phi, theta, psi)."""
    M = _solve(np.transpose(k["A"], (0, 2, 1)), tau_gh[..., None])[..., 0]   # tau = A^T M
    M_isb = k["detP"] * (M @ k["P"].T)
    return np.einsum("nij,ni->nj", k["E"], M_isb)


def torque_capacity(k, effort_gh):
    """(N, 3) pose-dependent torque capacity per coordinate from the GH efforts (3,)."""
    e_dhm = k["detP"] * np.einsum("ij,njk->nik", k["P"].T, k["E"])        # E back in DHM coords
    C = _solve(k["A"], e_dhm)                                             # columns: A^-1 e_k
    return np.einsum("nik,i->nk", np.abs(C), np.asarray(effort_gh, float))


def speed_capacity(k, velocity5):
    """(N, 3) pose-dependent speed capacity [rad/s] from the five joints' velocity limits."""
    B_isb = k["detP"] * np.einsum("ij,njk->nik", k["P"], k["B"])
    with np.errstate(all="ignore"):
        G = _solve(k["E"], B_isb)                                         # rates per unit joint rate
    return np.einsum("nkj,j->nk", np.abs(G), np.asarray(velocity5, float))


def rom_from_limits(side, lim, n=7, mask_deg=MASK_DEG):
    """{coord: (lo, hi)} [rad] over a grid of the 5-joint URDF limit box."""
    ch = CHAINS[side]
    joints = [j for j, _ in ch["clav"]] + [j for j, _ in ch["gh"]]
    axes = []
    for j in joints:
        e = lim.get(j) or {}
        if e.get("lower") is None or e.get("upper") is None:
            return None
        axes.append(np.linspace(float(e["lower"]), float(e["upper"]), n))
    G = np.array(list(product(*axes)))
    k = kinematics(side, G[:, :2], G[:, 2:])
    ang = k["angles"]
    ok = ang[:, 1] >= np.deg2rad(mask_deg)
    out = {"elevation": (float(ang[:, 1].min()), float(ang[:, 1].max()))}
    for c, i in (("plane", 0), ("rotation", 2)):
        v = ang[ok, i]
        out[c] = (float(v.min()), float(v.max())) if v.size else (None, None)
    return out


def _unwrap_masked(x, valid):
    """Unwrap an angle over its valid samples, then NaN the rest."""
    y = np.full_like(x, np.nan)
    if valid.any():
        y[valid] = np.unwrap(x[valid])
    return y


def side_available(dof_names, side):
    ch = CHAINS[side]
    need = [j for j, _ in ch["clav"]] + [j for j, _ in ch["gh"]]
    return all(n in dof_names for n in need)


def to_elevation(cap, mask_deg=MASK_DEG):
    """A new Capture whose shoulder X/Y/Z columns are plane / elevation / rotation.

    Returns (capture, info) where info = {side: {"valid_frac": [per rep], ...}}. A capture that
    already carries elevation coordinates, or lacks the chain, is returned unchanged.
    """
    names_in = list(cap.dof_names)
    sides = [s for s in ("right", "left") if side_available(names_in, s)]
    if not sides:
        return cap, {}
    idx = {n: i for i, n in enumerate(names_in)}
    new_names = list(names_in)
    lim_in = cap.urdf_limits or {}
    lim_out = {k: v for k, v in lim_in.items()}
    info = {}
    reps_out = [dict(angles=r.angles.copy(),
                     torques=None if r.torques is None else r.torques.copy(),
                     gravity_torques=None if r.gravity_torques is None else r.gravity_torques.copy())
                for r in cap.reps]
    thr = np.deg2rad(mask_deg)
    for side in sides:
        ch = CHAINS[side]
        ci = [idx[j] for j, _ in ch["clav"]]
        gi = [idx[j] for j, _ in ch["gh"]]
        eff = [(lim_in.get(j) or {}).get("effort") for j, _ in ch["gh"]]
        vel = [(lim_in.get(j) or {}).get("velocity") for j, _ in ch["clav"] + ch["gh"]]
        tq_caps, sp_caps, valid_fracs = [], [], []
        for r, out in zip(cap.reps, reps_out):
            k = kinematics(side, r.angles[:, ci], r.angles[:, gi])
            ang = k["angles"]
            valid = ang[:, 1] >= thr
            valid_fracs.append(float(valid.mean()) if valid.size else float("nan"))
            out["angles"][:, gi[0]] = _unwrap_masked(ang[:, 0], valid)
            out["angles"][:, gi[1]] = ang[:, 1]
            out["angles"][:, gi[2]] = _unwrap_masked(ang[:, 2], valid)
            if r.torques is not None:
                out["torques"][:, gi] = project_torque(k, r.torques[:, gi])
            if r.gravity_torques is not None:
                out["gravity_torques"][:, gi] = project_torque(k, r.gravity_torques[:, gi])
            if all(e for e in eff):
                tc = torque_capacity(k, eff)
                tq_caps.append(np.where(np.c_[valid, np.ones_like(valid), valid], tc, np.nan))
            if all(v for v in vel):
                sc = speed_capacity(k, vel)
                sp_caps.append(np.where(np.c_[valid, np.ones_like(valid), valid], sc, np.nan))
        rom = rom_from_limits(side, lim_in, mask_deg=mask_deg)
        for c, col in zip(COORDS, range(3)):
            new = f"{side}_shoulder_{c}"
            new_names[gi[col]] = new
            entry = {"capacity_basis": "DHM URDF limits mapped to ISB humerothoracic coordinates; "
                                       "effort / velocity = median of the pose-dependent capacity "
                                       "over this recording"}
            if rom and rom.get(c) and rom[c][0] is not None:
                entry["lower"], entry["upper"] = rom[c]
            # a coordinate masked for the whole recording (an arm that never rises above the mask
            # angle) has no defined capacity: the key is left out rather than set to NaN
            for key, caps in (("effort", tq_caps), ("velocity", sp_caps)):
                if caps:
                    v = np.concatenate([t[:, col] for t in caps])
                    v = v[np.isfinite(v)]
                    if v.size:
                        entry[key] = float(np.median(v))
            lim_out[new] = entry
        for j, _ in ch["gh"]:
            lim_out.pop(j, None)
        info[side] = {"valid_frac": valid_fracs, "mask_deg": mask_deg}

    reps = [replace(r, angles=o["angles"], torques=o["torques"],
                    gravity_torques=o["gravity_torques"]) for r, o in zip(cap.reps, reps_out)]
    new_cap = replace(cap, reps=reps, dof_names=new_names, urdf_limits=lim_out)
    for attr in ("trim", "decimate"):
        if hasattr(cap, attr):
            setattr(new_cap, attr, getattr(cap, attr))
    new_cap.shoulder_coords = "elevation"
    new_cap.shoulder_info = info
    return new_cap, info


def rename(dof_names, coords="elevation"):
    """The analysed DOF list for a raw DHM list: shoulder X/Y/Z -> plane/elevation/rotation."""
    if coords != "elevation":
        return list(dof_names)
    out = list(dof_names)
    for side in ("right", "left"):
        if side_available(dof_names, side):
            for j, ax in CHAINS[side]["gh"]:
                out[out.index(j)] = f"{side}_shoulder_{_SUFFIX_OF[j[-1]]}"
    return out
