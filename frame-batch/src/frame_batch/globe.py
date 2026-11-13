"""
Shoulder globe -- the task's arm directions on a Mollweide map of (plane of elevation, elevation).

After Stienen & Keemink (2015), as used by Zimmermann et al. (2023, ANYexo 2.0, Figs. 7-9): the
humerus direction is a point on a sphere, so the shoulder's range, its pose-dependent limits and
the task's trajectory are drawn on one map instead of as per-angle time series with constant
bands, which are wrong almost everywhere for a pose-dependent shoulder.

    horizontal = plane of elevation phi, -90 (left edge) .. 180 deg (right edge), centred on 45
    vertical   = elevation theta, 0 (bottom pole, arm hanging) .. 180 deg (top pole)
(the drawn ellipse of Zimmermann Fig. 7; internally cells are binned on lon = phi - 45, lat =
theta - 90 over the whole sphere, so the metrics see every sample).

The bottom pole is where phi is undefined (gimbal lock): on this map it is a single point, so the
samples masked below `shoulder_mask_deg` in the time series sit naturally near the pole here and
are drawn with their unmasked phi.

Three panels per figure (one figure per task x orientation, every recording of the group pooled):
  (a) how long the arm stays in each 15 x 15 deg zone (seconds per recording, averaged over the
      group), with the Boone & Azen (1979) human ROM points and the RULA upper-arm zones;
  (b) demand / capacity: per cell, the 95th percentile across recordings of each recording's peak
      |tau_theta| / elevation torque capacity there (`capacity_overrides` when set, else the
      DHM-derived pose-dependent capacity);
  (d) how long |tau_theta| stays above `capacity_fraction` x capacity in each 15 deg zone
      (seconds per recording);
  (e) the gravity share G = RMS(tau_g_theta) / RMS(tau_theta) per cell, all samples pooled;
  (c) the exoskeleton's reachable arm directions (its URDF swept over its real joint limits; the
      arm line = motor pivot -> `*_arm_tip`) and the task's cells, inside vs outside it.

Per-recording metrics (report time, merged into each trial's row before pooling):
    shoulder_globe.<side>.exo_inside_frac    share of samples whose direction the exo arm reaches
    shoulder_globe.<side>.norm_inside_frac   share of samples inside the normative region
    shoulder_globe.<side>.peak_demand_ratio  max |tau_theta| / capacity over the recording

Axial rotation is not drawn. Manipulability is not drawn: for the DHM it would describe the
conditioning of its X-Y-Z Euler chain, which has no physiological meaning.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .plots import SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID

CENTRE_PHI = 45.0
BIN_DEG = 5.0
FIGURE_NAME = "shoulder_globe_by_task"

# Normative active shoulder ROM, Boone & Azen (1979), J Bone Joint Surg Am 61(5):756-759, all
# subjects, mean + 1 SD -- the values Zimmermann et al. (2023, Fig. 7) plot. Placed on the globe:
#   horizontal flexion  140.7 + 5.9   -> phi = +146.6, theta = 90   (arm horizontal, swept forward)
#   horizontal extension 45.4 + 6.2   -> phi =  -51.6, theta = 90
#   neutral abduction   184.0 + 7.0   -> phi =    0,   theta = 180 (capped at the pole)
#   forward flexion     166.7 + 4.7   -> phi =   90,   theta = 171.4
#   backward extension   62.3 + 9.5   -> phi =  -90,   theta = 71.8
BOONE_POINTS = [("horizontal flexion", 146.6, 90.0), ("horizontal extension", -51.6, 90.0),
                ("neutral abduction", 0.0, 180.0), ("forward flexion", 90.0, 171.4),
                ("backward extension", -90.0, 71.8)]
# Region used for the normative human range (user-specified, from the points above):
# plane of elevation -55 .. 150 deg, elevation 0 .. 180 deg.
NORM_PHI = (-55.0, 150.0)
NORM_THETA = (0.0, 180.0)
NORM_SOURCE = "Boone & Azen 1979, mean + 1 SD"


# --------------------------------------------------------------------------------------------- #
# map geometry
# --------------------------------------------------------------------------------------------- #
def lonlat(phi_deg, theta_deg):
    """Map coordinates in radians: lon = wrap(phi - 45), lat = theta - 90."""
    lon = (np.asarray(phi_deg, float) - CENTRE_PHI + 180.0) % 360.0 - 180.0
    lat = np.asarray(theta_deg, float) - 90.0
    return np.deg2rad(lon), np.deg2rad(lat)


def _cells(phi_deg, theta_deg, bin_deg=BIN_DEG):
    lon, lat = lonlat(phi_deg, theta_deg)
    nlon, nlat = int(360 / bin_deg), int(180 / bin_deg)
    i = np.clip(((np.rad2deg(lon) + 180.0) / bin_deg).astype(int), 0, nlon - 1)
    j = np.clip(((np.rad2deg(lat) + 90.0) / bin_deg).astype(int), 0, nlat - 1)
    ok = np.isfinite(lon) & np.isfinite(lat)
    return i, j, ok, (nlat, nlon)


def occupancy(phi_deg, theta_deg, values=None, bin_deg=BIN_DEG):
    """Grid over the map (lat x lon bins of `bin_deg`): bool occupancy, or max of `values`."""
    i, j, ok, shape = _cells(phi_deg, theta_deg, bin_deg)
    if values is None:
        g = np.zeros(shape, bool)
        g[j[ok], i[ok]] = True
        return g
    v = np.asarray(values, float)
    ok &= np.isfinite(v)
    g = np.full(shape, -np.inf)
    np.maximum.at(g, (j[ok], i[ok]), v[ok])
    g[~np.isfinite(g)] = np.nan
    return g


def counts(phi_deg, theta_deg, bin_deg=BIN_DEG):
    i, j, ok, shape = _cells(phi_deg, theta_deg, bin_deg)
    g = np.zeros(shape, float)
    np.add.at(g, (j[ok], i[ok]), 1.0)
    return g


def sums(phi_deg, theta_deg, values, bin_deg=BIN_DEG):
    """Per-cell sum of `values` (non-finite values skipped)."""
    i, j, ok, shape = _cells(phi_deg, theta_deg, bin_deg)
    v = np.asarray(values, float)
    ok &= np.isfinite(v)
    g = np.zeros(shape, float)
    np.add.at(g, (j[ok], i[ok]), v[ok])
    return g


def inside(grid, phi_deg, theta_deg):
    """Per sample: does it fall in a cell of `grid`?"""
    i, j, ok, _ = _cells(phi_deg, theta_deg)
    out = np.zeros(i.shape, bool)
    out[ok] = grid[j[ok], i[ok]]
    return out


def fill_holes(grid):
    """Close single-cell gaps of a sampled region (random sampling leaves pinholes)."""
    from scipy.ndimage import binary_closing
    return binary_closing(grid, structure=np.ones((3, 3), bool), border_value=0) | grid


def edges(bin_deg=BIN_DEG):
    return (np.deg2rad(np.arange(-180, 180 + bin_deg / 2, bin_deg)),
            np.deg2rad(np.arange(-90, 90 + bin_deg / 2, bin_deg)))


# RULA upper-arm score (McAtamney & Corlett 1993), from the arm direction alone:
#   1  within 20 deg of hanging (flexion or extension)
#   2  extension > 20 deg (arm below horizontal, behind the frontal plane), or 20-45 deg in
#      front / to the side
#   3  45-90 deg in front / to the side
#   4  above 90 deg (any plane: an arm raised overhead is scored as flexion > 90)
# The +1 / -1 adjustments (shoulder raised, arm abducted, arm supported) need more than the arm
# direction and are not applied. Zones: 2 = risk, 3 = high risk, 4 = very high risk.
RULA_LABELS = {2: "risk", 3: "high risk", 4: "very high risk"}
RULA_SOURCE = "RULA upper-arm score, McAtamney & Corlett 1993 (no adjustments)"


def rula_score(phi_deg, theta_deg):
    """RULA upper-arm score 1-4 per sample (arm direction only)."""
    phi = np.asarray(phi_deg, float)
    th = np.asarray(theta_deg, float)
    behind = np.sin(np.deg2rad(phi)) < 0               # arm behind the frontal plane
    score = np.select([th < 20, th > 90, behind, th < 45], [1, 4, 2, 2], 3)
    return score.astype(int)


def rula_grid(bin_deg=BIN_DEG):
    """RULA score at every cell centre of the map."""
    lon_e, lat_e = edges(bin_deg)
    lon_c = np.rad2deg((lon_e[:-1] + lon_e[1:]) / 2)
    lat_c = np.rad2deg((lat_e[:-1] + lat_e[1:]) / 2)
    P, T = np.meshgrid((lon_c + CENTRE_PHI + 180.0) % 360.0 - 180.0, lat_c + 90.0)
    return rula_score(P, T)


def norm_region(bin_deg=BIN_DEG, cap_deg=20.0):
    """The normative region as cells whose centre lies in NORM_PHI x NORM_THETA, plus the polar
    cap below `cap_deg` elevation: with the arm near hanging, the plane of elevation is undefined
    (gimbal lock), so any phi there is a normal posture rather than one outside the range."""
    lon_e, lat_e = edges(bin_deg)
    lon_c = np.rad2deg((lon_e[:-1] + lon_e[1:]) / 2)
    lat_c = np.rad2deg((lat_e[:-1] + lat_e[1:]) / 2)
    phi_c = (lon_c + CENTRE_PHI + 180.0) % 360.0 - 180.0
    th_c = lat_c + 90.0
    P, T = np.meshgrid(phi_c, th_c)
    band = (P >= NORM_PHI[0]) & (P <= NORM_PHI[1]) & (T >= NORM_THETA[0]) & (T <= NORM_THETA[1])
    return band | (T < cap_deg)


# --------------------------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------------------------- #
def trial_series(cap, side="right", capacity_override=None):
    """From a RAW capture: phi, theta (unmasked, deg), tau_theta, its gravity part tau_g_theta
    (NaN without gravity torques) and capacity_theta, per sample."""
    from frame_core import shoulder as sh
    ch = sh.CHAINS[side]
    idx = {n: i for i, n in enumerate(cap.dof_names)}
    if not sh.side_available(cap.dof_names, side):
        return None
    ci = [idx[j] for j, _ in ch["clav"]]
    gi = [idx[j] for j, _ in ch["gh"]]
    lim = cap.urdf_limits or {}
    eff = [(lim.get(j) or {}).get("effort") for j, _ in ch["gh"]]
    out = {"phi": [], "theta": [], "tau": [], "tau_g": [], "cap": []}
    for r in cap.reps:
        k = sh.kinematics(side, r.angles[:, ci], r.angles[:, gi])
        out["phi"].append(np.rad2deg(k["angles"][:, 0]))
        out["theta"].append(np.rad2deg(k["angles"][:, 1]))
        tau = sh.project_torque(k, r.torques[:, gi])[:, 1] if r.torques is not None \
            else np.full(r.n, np.nan)
        out["tau"].append(tau)
        out["tau_g"].append(sh.project_torque(k, r.gravity_torques[:, gi])[:, 1]
                            if r.gravity_torques is not None else np.full(r.n, np.nan))
        if capacity_override:
            out["cap"].append(np.full(r.n, float(capacity_override)))
        elif all(eff):
            out["cap"].append(sh.torque_capacity(k, eff)[:, 1])
        else:
            out["cap"].append(np.full(r.n, np.nan))
    return {k: np.concatenate(v) for k, v in out.items()}


def exo_directions(cfg, cache_dir=None):
    """{side: (phi, theta)} [deg] the exo arm can point, swept over its URDF limits; cached.

    Needs Pinocchio (inside the container). Returns (None, reason) when unavailable.
    """
    urdf = cfg.any_exo_urdf
    if not urdf:
        return None, "no exoskeleton URDF configured (coverage_exo_urdf)"
    path = Path(cache_dir) / "exo_arm_directions.npz" if cache_dir else None
    stamp = f"{urdf}|{Path(urdf).stat().st_mtime if Path(urdf).exists() else 'missing'}"
    if path and path.is_file():
        try:
            with np.load(path) as z:
                if str(z["stamp"]) == stamp:
                    return {s: (z[f"{s}_phi"], z[f"{s}_theta"]) for s in ("right", "left")
                            if f"{s}_phi" in z}, ""
        except (OSError, KeyError, ValueError):
            pass
    try:
        from frame_kinematics.pinocchio_backend.exo_workspace import compute_exo_arm_directions
    except ImportError as e:
        return None, f"exo directions need Pinocchio ({e})"
    d = compute_exo_arm_directions(urdf, trunk_frame=cfg.exo_trunk_frame)
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, stamp=stamp,
                                **{f"{s}_{k}": v[f"{k}_deg"] for s, v in d.items()
                                   for k in ("phi", "theta")})
        except OSError:
            pass
    return {s: (v["phi_deg"], v["theta_deg"]) for s, v in d.items()}, ""


def working_arm(rec, side="right", min_ratio=0.5):
    """Is `side` a working arm in THIS recording? Returns "right" | "left" | None.

    Subjects may switch hands between recordings, and many recordings are two-handed. `side` counts
    as working unless it is essentially idle: its shoulder-elevation range is below `min_ratio` x
    the other arm's. Two-handed recordings therefore keep `side`; only recordings done with the
    other arm alone return the other side.
    """
    other = "left" if side == "right" else "right"
    m = rec.get("metrics") or {}
    a = m.get(f"joint_space.{side}_shoulder_elevation.rom_deg")
    b = m.get(f"joint_space.{other}_shoulder_elevation.rom_deg")
    if a is None or not np.isfinite(a):
        return None
    if b is None or not np.isfinite(b):
        return side
    return side if float(a) >= min_ratio * float(b) else other


def collect(records, cache, cfg, exo=None, side="right",
            bys=(("task", "orientation"), ("task",)), working_only=True, arm_records=None):
    """Per group: sample counts, per-recording peak-ratio grids, inside counts; per trial metrics.

    side         : the arm analysed ("right" by default; "left" is in the mirrored convention).
    working_only : keep only recordings whose working arm (`working_arm`) is `side`. Every
                   recording still gets `shoulder_globe.<side>.is_working_arm` (1 / 0), and the
                   globe metrics only when it is kept.
    bys          : the groupings to pool by -- every kept recording is added to one group of each
                   (default: task x orientation, and task alone).
    exo          : {side: (phi, theta)} or one (phi, theta) used for both sides.
    arm_records  : records with BOTH arms' metrics, used for the working-arm check -- the reported
                   records may be narrowed by `joints_of_interest` to one arm, which would make
                   every recording look right-handed. Defaults to `records`.
    Returns (groups, metrics, missing) with groups keyed by label ("H_N", "H", ...).
    """
    from frame_kinematics.serialize import capture_from_npz_bytes
    if exo is not None and not isinstance(exo, dict):
        exo = {"right": exo, "left": exo}
    exo_g = fill_holes(occupancy(*exo[side])) if exo and side in exo else None
    mask = float(getattr(cfg, "shoulder_mask_deg", 20.0))
    norm_g = norm_region(cap_deg=mask)
    over = ((getattr(cfg, "capacity_overrides", None) or {})
            .get(f"{side}_shoulder_elevation") or {}).get("effort")
    key = f"shoulder_globe.{side}"
    frac = float(getattr(cfg, "capacity_fraction", 0.15))

    def labels(rec):
        return ["_".join(str(rec.get(f)) for f in by) if by else "all" for by in bys]

    def slot(label):
        return groups.setdefault(label, {"counts": 0.0, "seconds": 0.0, "seconds_over": 0.0,
                                         "tau2": 0.0, "tau_g2": 0.0, "n_g": 0.0,
                                         "ratio": [], "n": 0,
                                         "in_exo": 0,
                                         "in_norm": 0, "samples": 0, "excluded": [],
                                         "rula": np.zeros(5)})

    groups, metrics, missing = {}, {}, 0
    by_id = {r.get("trial_id"): r for r in (arm_records or records)}
    for rec in records:
        tid = rec.get("trial_id")
        arm = working_arm(by_id.get(tid, rec), side,
                          float(getattr(cfg, "globe_min_rom_ratio", 0.5)))
        is_working = arm == side
        metrics[tid] = {f"{key}.is_working_arm": float(is_working)}
        if working_only and not is_working:
            for lab in labels(rec):
                slot(lab)["excluded"].append(tid)
            continue
        blob = cache.load_capture_bytes(rec.get("subject"), tid)
        if not blob:
            missing += 1
            continue
        cap = capture_from_npz_bytes(blob)
        fs = float(cap.fs or 240.0)
        ser = trial_series(cap, side, over)
        if ser is None:
            continue
        ratio = np.abs(ser["tau"]) / ser["cap"]
        c = counts(ser["phi"], ser["theta"])
        above = ratio > frac
        c_over = counts(ser["phi"][above], ser["theta"][above])
        both = np.isfinite(ser["tau"]) & np.isfinite(ser["tau_g"])       # G: same samples only
        t2 = sums(ser["phi"][both], ser["theta"][both], ser["tau"][both] ** 2)
        g2 = sums(ser["phi"][both], ser["theta"][both], ser["tau_g"][both] ** 2)
        n_g = counts(ser["phi"][both], ser["theta"][both])
        r_grid = occupancy(ser["phi"], ser["theta"], ratio)
        in_norm = inside(norm_g, ser["phi"], ser["theta"]) | (ser["theta"] < mask)
        in_exo = inside(exo_g, ser["phi"], ser["theta"]) if exo_g is not None else None
        rula = np.bincount(rula_score(ser["phi"], ser["theta"]), minlength=5)
        for lab in labels(rec):
            g = slot(lab)
            g["counts"] = g["counts"] + c
            g["seconds"] = g["seconds"] + c / fs
            g["seconds_over"] = g["seconds_over"] + c_over / fs
            g["tau2"] = g["tau2"] + t2
            g["tau_g2"] = g["tau_g2"] + g2
            g["n_g"] = g["n_g"] + n_g / fs
            g["ratio"].append(r_grid)
            g["n"] += 1
            g["samples"] += ser["phi"].size
            g["in_norm"] += int(in_norm.sum())
            g["rula"] = g["rula"] + rula
            if in_exo is not None:
                g["in_exo"] += int(in_exo.sum())
        m = metrics[tid]
        m[f"{key}.norm_inside_frac"] = float(in_norm.mean())
        n = max(int(rula.sum()), 1)
        for sc in (2, 3, 4):
            m[f"{key}.rula{sc}_time_frac"] = float(rula[sc] / n)
        m[f"{key}.peak_demand_ratio"] = (float(np.nanmax(ratio)) if np.isfinite(ratio).any()
                                         else float("nan"))
        if in_exo is not None:
            m[f"{key}.exo_inside_frac"] = float(in_exo.mean())
    for lab in list(groups):
        g = groups[lab]
        if not g["ratio"]:
            groups.pop(lab)
            continue
        stack = np.stack(g.pop("ratio"))
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                g["ratio_p95"] = np.nanpercentile(stack, 95, axis=0)
    return groups, metrics, missing


def augment(records, metrics):
    """Copies of `records` with the globe metrics merged into `metrics`."""
    out = []
    for r in records:
        m = metrics.get(r.get("trial_id"))
        if m:
            r = dict(r)
            r["metrics"] = {**(r.get("metrics") or {}), **m}
        out.append(r)
    return out


# --------------------------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------------------------- #
# The drawn map is the one of Zimmermann et al. (2023, Fig. 7): an ellipse spanning the plane of
# elevation -90 .. 180 deg (centred on 45 deg) and the elevation 0 (bottom pole, arm hanging) ..
# 180 deg, with a 15 deg graticule. It is an equal-area Mollweide ellipse whose longitude is the
# plane of elevation stretched so that those 270 deg fill it (phi -90 -> left edge, 180 -> right).
# Samples with phi outside -90 .. 180 deg (arm behind AND across the body -- almost only near
# hanging, where phi is undefined anyway) are not drawn; the footer says how many there are.
PHI_SPAN = (-90.0, 180.0)
GRAT_DEG = 15.0                       # graticule, and the zones of panel (a)
TIME_BOUNDS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]   # s per recording, panel (a)
BOONE_C = "#b02fc4"                   # violet, as in the paper
EXO_C, EXO_EDGE = "#c6dbef", "#4292c6"
GRAT_C = "#3a3a38"


def project(phi_deg, theta_deg):
    """(x, y) of the drawn map; x in [-2 sqrt 2, 2 sqrt 2], y in [-sqrt 2, sqrt 2]."""
    phi = np.asarray(phi_deg, float)
    lam = np.deg2rad((phi - CENTRE_PHI) * 180.0 / ((PHI_SPAN[1] - PHI_SPAN[0]) / 2.0))
    lat = np.deg2rad(np.asarray(theta_deg, float) - 90.0)
    t = lat.copy()
    for _ in range(30):                                  # 2t + sin 2t = pi sin(lat)
        den = 2.0 + 2.0 * np.cos(2.0 * t)
        step = np.where(den > 1e-9, (2.0 * t + np.sin(2.0 * t) - np.pi * np.sin(lat))
                        / np.where(den > 1e-9, den, 1.0), 0.0)
        t = t - step
    return 2.0 * np.sqrt(2.0) / np.pi * lam * np.cos(t), np.sqrt(2.0) * np.sin(t)


def _drawn_cols(bin_deg=BIN_DEG):
    """Columns of the internal (lat x lon) grid whose plane of elevation lies in PHI_SPAN."""
    lon_e, _ = edges(bin_deg)
    phi_c = np.rad2deg((lon_e[:-1] + lon_e[1:]) / 2) + CENTRE_PHI      # unwrapped: -135 .. 225
    return np.flatnonzero((phi_c > PHI_SPAN[0]) & (phi_c < PHI_SPAN[1]))


def _mesh(bin_deg=BIN_DEG):
    """Projected cell corners (X, Y) of the drawn columns, and the cell centres (Xc, Yc)."""
    phi_e = np.arange(PHI_SPAN[0], PHI_SPAN[1] + bin_deg / 2, bin_deg)
    th_e = np.arange(0.0, 180.0 + bin_deg / 2, bin_deg)
    X, Y = project(*np.meshgrid(phi_e, th_e))
    Xc, Yc = project(*np.meshgrid((phi_e[:-1] + phi_e[1:]) / 2, (th_e[:-1] + th_e[1:]) / 2))
    return X, Y, Xc, Yc


def _drawn(grid):
    return np.asarray(grid)[:, _drawn_cols()]


def zone_seconds(g, zone_deg=GRAT_DEG, key="seconds"):
    """Mean time per recording [s] spent in each zone_deg x zone_deg zone of the drawn map,
    spread back onto the internal BIN_DEG grid so it is drawn on the same mesh. NaN where no time
    was spent (and outside the drawn span)."""
    k = int(round(zone_deg / BIN_DEG))
    sec = _drawn(g[key]) / max(g["n"], 1)
    nr, nc = sec.shape
    z = sec.reshape(nr // k, k, nc // k, k).sum(axis=(1, 3))
    full = np.full(np.shape(g[key]), np.nan)
    full[:, _drawn_cols()] = np.repeat(np.repeat(z, k, 0), k, 1)
    return np.where(full > 0, full, np.nan)


G_MIN_S = 0.5      # a cell's gravity share is drawn only with >= this much data (s, pooled)
G_RANGE = (0.5, 1.0)


def gravity_share(g, min_s=G_MIN_S):
    """Per cell G = RMS(tau_g_theta) / RMS(tau_theta) over every sample of the group there."""
    with np.errstate(all="ignore"):
        G = np.sqrt(g["tau_g2"] / g["tau2"])
    return np.where((np.asarray(g["n_g"]) >= min_s) & np.isfinite(G), G, np.nan)


def _globe_axes(ax, title):
    """Graticule, outline and readable labels; the data go underneath (zorder < 5)."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.2, foreground=SURFACE)]
    ax.set_aspect("equal")
    ax.axis("off")
    R = 2.0 * np.sqrt(2.0)
    ax.set_xlim(-R - 0.42, R + 0.12)
    ax.set_ylim(-np.sqrt(2) - 0.5, np.sqrt(2) + 0.16)
    th = np.linspace(0.0, 180.0, 181)
    ph = np.linspace(*PHI_SPAN, 271)
    for p in np.arange(PHI_SPAN[0], PHI_SPAN[1] + 1, GRAT_DEG):     # meridians
        edge = p in PHI_SPAN
        ax.plot(*project(np.full_like(th, p), th), color=GRAT_C if not edge else INK,
                lw=1.2 if edge else 0.45, alpha=1.0 if edge else 0.75, zorder=6)
    for t in np.arange(GRAT_DEG, 180.0, GRAT_DEG):                    # parallels
        ax.plot(*project(ph, np.full_like(ph, t)), color=GRAT_C, lw=0.45, alpha=0.75, zorder=6)
    for p in np.arange(PHI_SPAN[0], PHI_SPAN[1] + 1, GRAT_DEG):       # phi labels on the equator
        x, y = project(p, 90.0)
        ha = "left" if p == PHI_SPAN[0] else "right" if p == PHI_SPAN[1] else "center"
        dx = 0.03 if p == PHI_SPAN[0] else -0.03 if p == PHI_SPAN[1] else 0.0
        ax.text(x + dx, y - 0.035, f"{p:.0f}", fontsize=6.6, color=INK, ha=ha, va="top",
                zorder=9, path_effects=halo)
    for t in np.arange(GRAT_DEG, 180.0, GRAT_DEG):                    # theta labels, left edge
        x, y = project(PHI_SPAN[0], t)
        ax.text(x - 0.06, y, f"{t:.0f}", fontsize=6.6, color=INK, ha="right", va="center",
                zorder=9)
    for t, dy, va in ((0.0, -0.07, "top"), (180.0, 0.07, "bottom")):  # the poles
        x, y = project(CENTRE_PHI, t)
        ax.text(x - 0.08, y + dy, f"{t:.0f}", fontsize=6.6, color=INK, ha="right", va=va,
                zorder=9)
    ax.text(-R - 0.36, 0.0, "elevation θ [°]", rotation=90, fontsize=7.5, color=INK_SECONDARY,
            ha="center", va="center")
    ax.text(R, -np.sqrt(2) - 0.1, "plane of elevation φ [°]\n(labels on the 90° parallel)",
            fontsize=7.5, color=INK_SECONDARY, ha="right", va="top")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=4)


def _paint(ax, grid, **kw):
    """pcolormesh of an internal-grid array on the drawn map."""
    X, Y, _, _ = _mesh()
    return ax.pcolormesh(X, Y, _drawn(grid), shading="flat", rasterized=True, **kw)


def _region(ax, grid, color, alpha, zorder=1):
    from matplotlib.colors import ListedColormap
    _paint(ax, np.where(grid, 1.0, np.nan), cmap=ListedColormap([color]), vmin=0, vmax=1,
           alpha=alpha, zorder=zorder)


def _outline(ax, grid, color, lw=1.4, zorder=4):
    _, _, Xc, Yc = _mesh()
    ax.contour(Xc, Yc, _drawn(grid).astype(float), levels=[0.5], colors=[color], linewidths=lw,
               zorder=zorder)


def _boone(ax, zorder=8):
    """Boone & Azen (1979) human ROM (mean + 1 SD) as violet dots, as in Zimmermann Fig. 7:
    the two poles and the horizontal extension / flexion limits on the 90 deg parallel."""
    for p, t in ((CENTRE_PHI, 180.0), (CENTRE_PHI, 0.0), (-53.0, 90.0), (150.0, 90.0)):
        ax.plot(*project(p, t), marker="o", ms=8, color=BOONE_C, mec=SURFACE, mew=1.0,
                zorder=zorder, clip_on=False)


def _rula(ax, g):
    """RULA upper-arm zone boundaries (dashed) with small labels; returns the time shares."""
    def line(phi, theta):
        ax.plot(*project(phi, theta), color=INK, lw=1.0, ls="--", zorder=7)

    around = np.linspace(*PHI_SPAN, 271)
    front = np.linspace(0.0, 180.0, 181)
    line(around, np.full_like(around, 20.0))           # 1 | 2
    line(front, np.full_like(front, 45.0))             # 2 | 3 (in front / to the side)
    line(around, np.full_like(around, 90.0))           # 3 | 4 and 2 | 4
    th = np.linspace(45.0, 90.0, 46)
    line(np.zeros_like(th), th)                        # 2 (behind) | 3 (in front), 45-90 deg
    for sc, t in ((2, 32), (3, 67), (4, 125)):
        ax.text(*project(165.0, t), f"RULA {sc}: {RULA_LABELS[sc]}", fontsize=6.8, color=INK,
                ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    tot = max(float(g["rula"].sum()), 1.0)
    return " · ".join(f"{RULA_LABELS[sc]} {100 * g['rula'][sc] / tot:.0f}%" for sc in (2, 3, 4))


def _time_panel(fig, ax, g, key, cmap_name, title, what):
    """A seconds-per-recording zone map on the fixed TIME_BOUNDS scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    _globe_axes(ax, title)
    cmap = plt.get_cmap(cmap_name, len(TIME_BOUNDS))
    zs = zone_seconds(g, key=key)
    top = max(TIME_BOUNDS[-1], float(np.nanmax(zs)) if np.isfinite(zs).any() else 0.0)
    bounds = TIME_BOUNDS + ([top] if top > TIME_BOUNDS[-1] else [TIME_BOUNDS[-1] * 2])
    pm = _paint(ax, zs, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), alpha=0.8, zorder=2,
                linewidth=0)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=45,
                      ticks=bounds[:-1])
    cb.ax.set_xticklabels([f"{b:g}" for b in bounds[:-1]], fontsize=7)
    cb.set_label(what, fontsize=8, color=INK_SECONDARY)


def group_figure(g, label, out_path, exo_g, capacity_label, side="right", fraction=0.15):
    """Five globe panels for one group of recordings: (a)-(c) in the left column, (d), (e) right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    task_g = g["counts"] > 0
    rec_s = float(np.sum(g["seconds"])) / max(g["n"], 1)
    hidden = 1.0 - float(_drawn(g["counts"]).sum()) / max(float(np.sum(g["counts"])), 1.0)
    fig, axes = plt.subplots(3, 2, figsize=(18.0, 15.0), facecolor=SURFACE)
    boone = Line2D([], [], marker="o", ls="", ms=7, color=BOONE_C,
                   label="human ROM, Boone & Azen 1979 (mean + 1 SD)")

    # (a) how long the arm stays in each direction zone
    ax = axes[0][0]
    _time_panel(fig, ax, g, "seconds", "YlGnBu",
                f"(a) Time spent per {GRAT_DEG:.0f}° × {GRAT_DEG:.0f}° zone, mean per recording",
                f"seconds per recording in the zone   (mean recording length {rec_s:.0f} s; "
                f"last bin open-ended)")
    _boone(ax)
    rula_share = _rula(ax, g)
    ax.legend(handles=[boone, Line2D([], [], color=INK, lw=1.0, ls="--",
                                     label=f"RULA upper-arm zones; share of time: {rula_share}")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7.3, frameon=False)

    # (b) demand / capacity
    ax = axes[1][0]
    _globe_axes(ax, "(b) Peak |τ_θ| / elevation torque capacity, 95th percentile across "
                    "recordings")
    r = g["ratio_p95"]
    vmax = max(0.3, float(np.nanmax(r)) if np.isfinite(r).any() else 0.3)
    pm = _paint(ax, r, cmap="YlOrRd", vmin=0, vmax=vmax, alpha=0.85, zorder=2, linewidth=0)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=45)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"peak |τ_θ| / capacity   (capacity: {capacity_label})", fontsize=8,
                 color=INK_SECONDARY)

    # (c) exo vs task
    ax = axes[2][0]
    _globe_axes(ax, "(c) Exoskeleton reachable directions vs the task")
    if exo_g is not None:
        _region(ax, exo_g, EXO_C, 0.8, zorder=1)
        _outline(ax, exo_g, EXO_EDGE, lw=1.4, zorder=5)
        _region(ax, task_g & exo_g, "#2a78d6", 0.85, zorder=3)
        _region(ax, task_g & ~exo_g, "#e34948", 0.9, zorder=4)
        frac_exo = g["in_exo"] / max(g["samples"], 1)
        handles = [Patch(fc=EXO_C, ec=EXO_EDGE, label="exo arm reachable (URDF limits)"),
                   Patch(color="#2a78d6", label="task cells inside the exo region"),
                   Patch(color="#e34948", label="task cells outside the exo region")]
        ax.text(1.0, 1.0, f"{100 * frac_exo:.1f}% of samples inside the exo region",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
    else:
        _region(ax, task_g, "#2a78d6", 0.85, zorder=3)
        handles = [Patch(color="#2a78d6", label="task cells (exo region unavailable)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7.3,
              frameon=False)

    # (d) where, and for how long, the elevation torque is above the reference fraction
    ax = axes[0][1]
    over_s = float(np.sum(g["seconds_over"])) / max(g["n"], 1)
    _time_panel(fig, ax, g, "seconds_over", "PuRd",
                f"(d) Time with |τ_θ| > {100 * fraction:g}% of capacity, per "
                f"{GRAT_DEG:.0f}° × {GRAT_DEG:.0f}° zone, mean per recording",
                f"seconds per recording above {100 * fraction:g}% of capacity in the zone   "
                f"(total {over_s:.1f} s of {rec_s:.0f} s; last bin open-ended)")
    _boone(ax)
    ax.legend(handles=[boone], loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7.3,
              frameon=False)

    # (e) gravity share of the elevation torque
    ax = axes[1][1]
    _globe_axes(ax, "(e) Gravity share of the elevation torque, G = RMS(τ_g,θ) / RMS(τ_θ)")
    G = gravity_share(g)
    # G clusters tightly around 1 for the shoulder elevation: the darkest colour is "all gravity".
    pm = _paint(ax, G, cmap="YlGn", vmin=G_RANGE[0], vmax=G_RANGE[1], alpha=0.85, zorder=2,
                linewidth=0)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=45,
                      extend="both")
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"G per {BIN_DEG:g}° cell, all samples pooled   (1: the torque is all gravity; "
                 f"<1: inertia / load add to it; >1 (arrow): they partly offset it; cells with "
                 f"< {G_MIN_S:g} s of data blank)", fontsize=8, color=INK_SECONDARY)
    if not np.isfinite(G).any():
        ax.text(0.5, 0.5, "no gravity torque in the cached captures", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=INK_MUTED, zorder=10)

    axes[2][1].axis("off")
    excl = len(g.get("excluded") or [])
    arm_txt = (f"{side} arm as working arm; {excl} recording(s) done with the other arm excluded"
               if excl else f"{side} arm as working arm")
    title_label = label.replace("_", " · ", 1)
    fig.suptitle(f"Shoulder globe · {title_label}   ({g['n']} recordings; {arm_txt})",
                 x=0.01, ha="left", fontsize=12, color=INK, y=0.995)
    fig.text(0.01, 0.004,
             f"Cells {BIN_DEG:g}° (panels b, c, e). {100 * hidden:.1f}% of samples have a plane of "
             f"elevation outside −90…180° (arm behind and across the body) and are not drawn. "
             f"τ_θ: shoulder elevation torque; τ_g,θ: its gravity part. "
             f"Map after Stienen & Keemink 2015 / Zimmermann et al. 2023.",
             fontsize=7.3, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300 if str(out_path).endswith(".pdf") else 110, facecolor=SURFACE)
    plt.close(fig)
    return str(out_path)


def build(records, cache, cfg, out_dir, side="right", cache_dir=None, arm_records=None):
    """Collect, draw one figure per group, return (paths, per-trial metrics, notes).

    Groups: `npose_by` (default task x orientation) and task alone, so H, M, L each also get a
    figure pooling their orientations."""
    exo, why = exo_directions(cfg, cache_dir)
    notes = [why] if why else []
    fine = tuple(getattr(cfg, "npose_by", None) or ("task",))
    bys = (fine, ("task",)) if fine != ("task",) else (fine,)
    groups, metrics, missing = collect(records, cache, cfg, exo, side, bys,
                                       arm_records=arm_records)
    if missing:
        notes.append(f"{missing} recording(s) had no cached capture")
    others = sorted(t for t, m in metrics.items() if not m.get(f"shoulder_globe.{side}.is_working_arm"))
    if others:
        notes.append(f"{len(others)} recording(s) done with the {'left' if side == 'right' else 'right'} "
                     f"arm were excluded from the {side}-arm globe: "
                     + ", ".join(others[:8]) + (" ..." if len(others) > 8 else ""))
    over = ((getattr(cfg, "capacity_overrides", None) or {})
            .get(f"{side}_shoulder_elevation") or {})
    src = str(over.get("source", "capacity_overrides")).split(",")[0]     # citation only
    cap_label = (f"{over['effort']:g} N\u00b7m, {src}" if over.get("effort")
                 else "DHM-derived, pose-dependent")
    exo_g = fill_holes(occupancy(*exo[side])) if exo and side in exo else None
    fmt = (getattr(cfg, "figure_format", "pdf") or "pdf").lower()
    from .plots import _order_groups
    paths = []
    for label in _order_groups(list(groups)):
        paths.append(group_figure(groups[label], label,
                                  Path(out_dir) / f"{FIGURE_NAME}_{label}.{fmt}", exo_g,
                                  cap_label, side,
                                  float(getattr(cfg, "capacity_fraction", 0.15))))
    return paths, metrics, notes
