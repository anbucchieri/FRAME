"""
Flatten one `analyze()` result into a single flat metric row.

The pooled tool analyses many trials and then reports the DISTRIBUTION of each metric across
them. That needs one row of scalars per trial, not the nested per-layer dict `analyze()`
returns -- so this module is the seam between "one capture's result" and "a table of trials".

Two rules make the table trustworthy:

  1. Every `summarize()` block collapses to its **median**. With the pooled tool's convention of
     one trial = one repetition that median IS the measured value (n=1), so nothing is lost; if a
     trial is ever segmented into several reps, the median stays the right per-trial
     representative. `n_reps` is carried alongside so the reader can tell the two cases apart.

  2. Every key is registered with a physical DIMENSION (`UNITS`). Torque, work, power and
     stiffness scale with body mass; lengths, speeds and volumes scale with stature. Pooling
     across subjects of different size without saying so would put body-size variance inside
     what looks like task variance, so `normalize_row` emits a mass/height-normalised twin of
     every dimensional column and leaves the raw SI column untouched beside it.

"""
from __future__ import annotations

# Bump this whenever the SET of metrics changes -- a column added, removed or renamed, or a
# formula altered. It rides in the batch cache key, so a cached trial computed by an older metric
# set is recomputed instead of being served with the new columns silently missing. The config
# fingerprint cannot catch this on its own: the config did not change, the code did.
#
#   1 : initial metric set
#   2 : added joint_space.<dof>.rms_vel_dps (continuous joint speed)
#   3 : added joint_space.<dof>.mean_crossing_gap_s (average time between direction changes)
#   4 : added the passivity layer; verdict is now ACTIVE / QUASI_PASSIVE / PASSIVE / FREE
#   5 : passivity reduced to G + reversal metrics; the verdict (synthesis layer) removed
METRICS_VERSION = 5

# --- physical dimensions -> how a quantity is made subject-independent -------------------- #
#   torque/work/power/stiffness : divide by body mass   [per kg]
#   length/speed/accel          : divide by body height [per m of stature]
#   volume_L                    : divide by height^3
#   none                        : dimensionless, angular, temporal or a count -> left alone
#
# Angles are deliberately NOT normalised: a joint's range of motion in degrees is already
# subject-independent, and dividing it by anything would destroy that.
BY_MASS = ("torque", "work", "power", "stiffness")
BY_HEIGHT = ("length", "speed", "accel")


# Key families whose last segment is a DOF NAME rather than a unit-bearing metric name. They are
# dimensionless scores, and they must be recognised BEFORE the suffix table: a DOF called
# `wrist_W` would otherwise make `joint_space.involvement.wrist_W` look like watts and get divided
# by body mass. Nothing would error -- the column would just be quietly wrong.
DOF_SUFFIXED_PREFIXES = ("joint_space.involvement.", "pca.engagement.")


def _dim_for(key: str) -> str:
    """Physical dimension of a flat metric key, from its registered suffix."""
    if key.startswith(DOF_SUFFIXED_PREFIXES):
        return "none"
    for suffix, dim in UNITS.items():
        if key.endswith(suffix):
            return dim
    return "none"


# Suffix -> dimension. Matched longest-first so `_peak_torque_Nm` cannot be shadowed by `_Nm`.
UNITS = {
    "_Nm_per_rad": "stiffness",
    "_mps2": "accel",
    "_mps": "speed",
    "_volume_L": "volume_L",
    "_Nm": "torque",
    "_W": "power",
    "_J": "work",
    "_m": "length",
}
UNITS = dict(sorted(UNITS.items(), key=lambda kv: -len(kv[0])))

# Raw dynamics key -> the flat column name that carries its unit.
DYNAMICS_KEYS = {
    "peak_torque": "peak_torque_Nm",
    "rms_torque": "rms_torque_Nm",
    "peak_power_pos": "peak_power_pos_W",
    "peak_power_neg": "peak_power_neg_W",
    "net_work": "net_work_J",
    "k": "k_Nm_per_rad",
    "r2": "r2",
}

# Gravity share and reversal metrics carried into each trial's row (`passivity.<dof>.<key>`).
PASSIVITY_KEYS = ("G", "n_reversals", "latency_med_s", "latency_min_s", "alpha_req_med_rad_s2",
                  "alpha_req_max_rad_s2", "peak_acc_rad_s2")

JOINT_SPACE_KEYS = ("rom_deg", "peak_vel_dps", "rms_vel_dps", "angular_path_rad",
                    "reversals", "zero_crossings",
                    "min_crossing_gap_s", "mean_crossing_gap_s")


def _med(block):
    """The representative scalar of a `summarize()` block (see rule 1 in the module docstring)."""
    if isinstance(block, dict) and "median" in block:
        return float(block["median"])
    try:
        return float(block)
    except (TypeError, ValueError):
        return float("nan")


def _n(block):
    return int(block["n"]) if isinstance(block, dict) and "n" in block else 1


def flatten_result(result, dof_names=None) -> dict:
    """One `analyze()` result -> {flat_key: float}. Missing layers simply contribute no keys.

    `dof_names` restricts (and orders) the per-DOF columns; without it every DOF the result
    mentions is emitted. Restricting matters when pooling across subjects: a column that exists
    for some trials and not others silently changes what `n` means for that metric.
    """
    row: dict = {}

    ts = result.get("task_space")
    if ts:
        row["task_space.workspace_volume_L"] = float(ts.get("workspace_volume_L", float("nan")))
        bb = ts.get("bbox_m") or {}
        for ax in ("x", "y", "z"):
            row[f"task_space.bbox_{ax}_m"] = float(bb.get(ax, float("nan")))
        row["task_space.dwell_concentration"] = float(ts.get("dwell_concentration", float("nan")))
        row["task_space.peak_speed_mps"] = _med(ts.get("peak_speed_mps"))
        row["task_space.peak_accel_mps2"] = _med(ts.get("peak_accel_mps2"))
        row["task_space.reach_max_m"] = _med(ts.get("reach_max_m"))
        row["task_space.per_rep_volume_L"] = _med(ts.get("per_rep_volume_L"))
        for k, v in (ts.get("coverage") or {}).items():
            if isinstance(v, (int, float)):
                row[f"task_space.coverage.{k}"] = float(v)

    js = result.get("joint_space")
    if js:
        per_dof = js.get("per_dof") or {}
        names = dof_names if dof_names is not None else list(per_dof)
        for d in names:
            e = per_dof.get(d)
            if not e:
                continue
            for k in JOINT_SPACE_KEYS:
                if k in e:
                    row[f"joint_space.{d}.{k}"] = _med(e[k])
            if "rom_pct_of_limit" in e:
                row[f"joint_space.{d}.rom_pct_of_limit"] = float(e["rom_pct_of_limit"])
        involvement = js.get("involvement") or {}
        inv = dict(zip(involvement.get("dof_names") or (), involvement.get("score") or ()))
        for d in (names if dof_names is not None else list(inv)):
            if d in inv:
                row[f"joint_space.involvement.{d}"] = float(inv[d])

    dy = result.get("dynamics")
    if dy:
        per_dof = dy.get("per_dof") or {}
        names = dof_names if dof_names is not None else list(per_dof)
        for d in names:
            e = per_dof.get(d)
            if not e:
                continue
            for raw, col in DYNAMICS_KEYS.items():
                if raw in e:
                    row[f"dynamics.{d}.{col}"] = _med(e[raw])
            for ratio in ("rms_torque_ratio", "peak_torque_ratio"):
                if ratio in e:
                    row[f"dynamics.{d}.{ratio}"] = float(e[ratio])

    pa = result.get("passivity")
    if pa:
        per_dof = pa.get("per_dof") or {}
        names = dof_names if dof_names is not None else list(per_dof)
        for d in names:
            e = per_dof.get(d)
            if not e:
                continue
            for k in PASSIVITY_KEYS:
                if k in e:
                    row[f"passivity.{d}.{k}"] = _med(e[k])

    pc = result.get("pca")
    if pc and pc.get("ok"):
        row["pca.n_pc_90"] = float(pc.get("n_pc_90", float("nan")))
        vaf = pc.get("explained_variance_ratio") or []
        for i in range(min(3, len(vaf))):
            row[f"pca.vaf_pc{i + 1}"] = float(vaf[i])
        eng = pc.get("engagement") or {}
        for d in (dof_names if dof_names is not None else list(eng)):
            if d in eng:
                row[f"pca.engagement.{d}"] = float(eng[d])

    return row


def n_reps(result) -> int:
    """How many repetitions produced this row (1 under the one-trial-one-rep convention)."""
    ts = result.get("task_space") or {}
    if "reach_max_m" in ts:
        return _n(ts["reach_max_m"])
    js = (result.get("joint_space") or {}).get("per_dof") or {}
    for e in js.values():
        if "rom_deg" in e:
            return _n(e["rom_deg"])
    return 1


def normalize_row(row, height_m=None, mass_kg=None, suffix="__norm") -> dict:
    """Subject-normalised twins of every dimensional column in `row`.

    Returns ONLY the new columns, so the caller decides how to merge them; the raw SI values are
    never modified. A column whose normaliser is unknown (no height/mass for that subject) is
    omitted rather than emitted as NaN -- an absent column is honest, a NaN one silently shrinks
    the `n` of a pooled distribution.
    """
    out = {}
    for key, val in row.items():
        dim = _dim_for(key)
        if dim == "none":
            continue
        if dim in BY_MASS:
            denom = mass_kg
        elif dim in BY_HEIGHT:
            denom = height_m
        elif dim == "volume_L":
            denom = height_m ** 3 if height_m else None
        else:
            denom = None
        if denom and denom > 0:
            out[key + suffix] = float(val) / float(denom)
    return out


def units_table(keys) -> dict:
    """{flat_key: dimension} for the given keys -- what `normalize_row` will and will not touch."""
    return {k: _dim_for(k) for k in keys}


# Layers whose middle key is a DOF name: `joint_space.<dof>.<metric>`, `dynamics.<dof>.<metric>`.
PER_DOF_LAYERS = ("joint_space", "dynamics", "passivity")


def dof_of(key: str):
    """The DOF a flat metric key belongs to, or None if the metric is whole-capture.

    `task_space.workspace_volume_L` and `pca.n_pc_90` describe the capture, not a joint, so they
    survive any joint filter. `joint_space.right_elbow_Z.rom_deg`,
    `joint_space.involvement.right_elbow_Z` and `pca.engagement.right_elbow_Z` are all that joint's.
    """
    for prefix in DOF_SUFFIXED_PREFIXES:                 # involvement / engagement: dof is LAST
        if key.startswith(prefix):
            return key[len(prefix):].split("__")[0]
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] in PER_DOF_LAYERS:
        return parts[1]
    return None


def filter_metrics(metrics, dofs):
    """Keep every whole-capture metric, plus the per-joint ones for `dofs`.

    Reporting on a subset of joints is a filter, never a recomputation: the values are exactly the
    ones the sweep produced, so narrowing (or widening) the list costs nothing but a re-report.
    `dofs=None` keeps everything.
    """
    if dofs is None:
        return dict(metrics)
    keep = set(dofs)
    return {k: v for k, v in metrics.items()
            if (d := dof_of(k)) is None or d in keep}


def filter_record(record, dofs, metrics_field="metrics"):
    """`filter_metrics` applied to a whole trial record."""
    if dofs is None:
        return record
    keep = set(dofs)
    out = dict(record)
    out[metrics_field] = filter_metrics(record.get(metrics_field) or {}, dofs)
    return out
