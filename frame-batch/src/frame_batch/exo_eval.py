"""
Exoskeleton requirements per recording: gravity share G and the direction-reversal metrics.

The numbers are computed by the `passivity` layer of frame_core during the analysis (one recording
= one repetition) and already sit in every trial's metric row as `passivity.<dof>.<key>`. This
module only regroups them into one long table, one row per recording x DOF, so they can be read
without the hundreds of other trials.csv columns:

    exo_eval/requirements_per_rep.csv
        trial_id, subject, task, <factors>, rep, dof,
        G, n_reversals, latency_med_s, latency_min_s,
        alpha_req_med_rad_s2, alpha_req_max_rad_s2, peak_acc_rad_s2

Units: SI (s, rad/s^2). G is dimensionless, RMS(tau_g) / RMS(tau).
"""
from __future__ import annotations
import csv
from pathlib import Path

FILE_NAME = "requirements_per_rep.csv"
ID_COLUMNS = ["trial_id", "subject", "task"]


def rows(records, dofs, factors=()):
    """One dict per (recording, DOF) that has at least one passivity value."""
    from frame_core.flatten import PASSIVITY_KEYS
    out = []
    for r in records:
        m = r.get("metrics") or {}
        for d in dofs:
            vals = {k: m.get(f"passivity.{d}.{k}") for k in PASSIVITY_KEYS}
            if all(v is None for v in vals.values()):
                continue
            out.append({**{c: r.get(c) for c in ID_COLUMNS},
                        **{f: r.get(f) for f in factors},
                        "rep": r.get("rep"), "dof": d, **vals})
    return out


def write(records, cfg, out_dir):
    """Write exo_eval/requirements_per_rep.csv. Returns {"requirements_per_rep": path} or {}."""
    from frame_core.flatten import PASSIVITY_KEYS
    import re
    factors = [f for f in re.compile(cfg.trial_pattern).groupindex
               if f not in ID_COLUMNS + ["rep"]]
    table = rows(records, cfg.selected_dofs(), factors)
    if not table:
        return {}
    path = Path(out_dir) / "exo_eval" / FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ID_COLUMNS + factors + ["rep", "dof"] + list(PASSIVITY_KEYS)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    return {"requirements_per_rep": str(path)}
