"""
Writing a pooled run down -- the table, the distributions, and a summary you can read first.

The outputs are deliberately layered, because a thousand-column table is not a result:

    results/
      summary.md                 read this first: what ran, what failed, what differs
      trials.csv                 one row per recording, every metric wide -- the analysis input
      trials.json                the same rows plus provenance (capability, urdf, durations)
      pooled.json                per task type: every metric's median/IQR/min/max/n + the values
      pooled_summary.csv         the flat version of pooled.json, one row per (task, metric)
      differences.csv            metrics ranked by how much they separate the task types
      failures.json              every recording that could not be analysed, with its traceback
      config.json                the exact config this run used
      figures/*.png              the distribution figures

`trials.csv` is the file to take into R or a notebook: it is tidy-ish (one row per trial, metrics
as columns), it carries the factor columns, and it has both raw SI and normalised twins.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

from .cache import json_safe


META_COLUMNS = ["trial_id", "subject", "task", "rep", "height_m", "mass_kg",
                "fs", "n_frames", "n_reps", "duration_s", "path"]


def _factor_columns(records):
    """Factor columns come from the filename pattern, so they are discovered, not hard-coded."""
    known = set(META_COLUMNS) | {"metrics", "capability", "urdf", "cache_key",
                                 "cached", "ok"}
    cols = []
    for r in records:
        for k in r:
            if k not in known and k not in cols:
                cols.append(k)
    return cols


def write_trials_csv(records, path):
    """One row per recording: metadata, then every metric column."""
    factors = _factor_columns(records)
    metric_cols = []
    for r in records:
        for k in (r.get("metrics") or {}):
            if k not in metric_cols:
                metric_cols.append(k)
    metric_cols.sort()

    header = META_COLUMNS + factors + metric_cols
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            metrics = r.get("metrics") or {}
            w.writerow([r.get(c) for c in META_COLUMNS]
                       + [r.get(c) for c in factors]
                       + [metrics.get(c) for c in metric_cols])
    return str(path)


def write_pooled_summary_csv(pooled, path):
    """pooled.json flattened to one row per (group, metric) -- the plottable long form."""
    fields = ["median", "iqr_lo", "iqr_hi", "min", "max", "mean", "std", "n", "n_missing"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "metric"] + fields)
        for group in sorted(pooled, key=str):
            for metric, entry in sorted(pooled[group]["metrics"].items()):
                w.writerow([group, metric] + [entry.get(k) for k in fields])
    return str(path)


def write_differences_csv(ranked, pooled, path):
    groups = sorted(pooled, key=str)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "test", "p_trial", "p_subject", "H", "n_subjects"]
                   + [f"median_{g}" for g in groups] + [f"n_trials_{g}" for g in groups]
                   + ["rank_identical_with"])
        for r in ranked:
            w.writerow([r["metric"], r.get("test"), r.get("p"), r.get("p_subject"), r.get("H"),
                        r.get("n_subjects")]
                       + [r["medians"].get(g) for g in groups]
                       + [r.get("n", {}).get(g) for g in groups]
                       + [" ".join(r.get("aliases") or ())])
    return str(path)


# The four numbers an actuator spec is read from, in the order a reader wants them.
SUMMARY_METRICS = [
    ("joint_space", "peak_vel_dps", "peak velocity [deg/s]"),
    ("joint_space", "rms_vel_dps", "RMS velocity [deg/s]"),
    ("dynamics", "peak_torque_Nm", "peak torque [N.m]"),
    ("dynamics", "rms_torque_Nm", "RMS torque [N.m]"),
    # How often the joint reverses (fixed 5 deg/s deadband, >= 0.1 s apart), and how long a
    # reversal takes (speed peak before -> speed peak after). The count scales with recording
    # length; the durations do not.
    ("passivity", "n_reversals", "velocity reversals [count]"),
    ("passivity", "latency_med_s", "median reversal duration [s]"),
    ("passivity", "latency_min_s", "shortest reversal duration [s]"),
]


def write_joint_summary_csv(pooled, path, dofs=None, metrics=SUMMARY_METRICS):
    """One row per (task, joint): mean +/- SD of peak/RMS velocity and peak/RMS torque.

    This is the table, not the figures: the four numbers per joint that an actuator or elastic
    element is sized from. Mean and SD as asked, with the median and IQR alongside -- the
    distributions are right-skewed and small, so a mean can sit outside the middle half of the
    data and the reader should be able to see when it does. `n` is the number of recordings behind
    each cell.
    """
    import csv

    groups = sorted(pooled, key=str)
    seen = []
    for g in groups:
        for key in (pooled.get(g) or {}).get("metrics", {}):
            parts = key.split(".")
            if len(parts) == 3 and parts[2] in {m for _, m, _ in metrics} and parts[1] not in seen:
                seen.append(parts[1])
    order = [d for d in (dofs or seen) if d in seen] or seen

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "joint", "metric", "unit", "mean", "std",
                    "median", "iqr_lo", "iqr_hi", "min", "max", "n"])
        for g in groups:
            per = (pooled.get(g) or {}).get("metrics", {})
            for dof in order:
                for layer, metric, label in metrics:
                    e = per.get(f"{layer}.{dof}.{metric}")
                    if not e:
                        continue
                    unit = label[label.find("[") + 1:label.rfind("]")] if "[" in label else ""
                    w.writerow([g, dof, metric, unit,
                                e.get("mean"), e.get("std"), e.get("median"),
                                e.get("iqr_lo"), e.get("iqr_hi"),
                                e.get("min"), e.get("max"), e.get("n")])
    return str(path)


def _fmt(v, nd=3):
    if v is None or v != v:
        return "--"
    return f"{v:.{nd}g}"


def write_summary_md(records, failures, pooled, ranked, cfg, subjects, figures, path,
                     skipped=()):
    """The page to read first. Everything else is a drill-down from it."""
    groups = sorted(pooled, key=str)
    lines = []
    A = lines.append

    A("# FRAME pooled analysis\n")
    A(f"- **subjects**: {len(subjects)} — " + "; ".join(s.describe() for s in subjects))
    A(f"- **trials analysed**: {len(records)}"
      + (f"  |  **failed**: {len(failures)}" if failures else ""))
    A(f"- **pooled by**: {', '.join(cfg.pool_by)}")
    A(f"- **DOFs**: {len(cfg.dof_names)}  |  **end-effector**: {cfg.ee_segment} "
      f"({cfg.ee_frame})  |  **exo assessment**: {cfg.exo_urdf or 'off (requirements only)'}")
    A(f"- **trim**: {cfg.trim_start_s:g} s head / {cfg.trim_end_s:g} s tail  "
      f"|  **config fingerprint**: `{cfg.fingerprint()}`\n")

    A("## Trials per group\n")
    A("| task | trials | subjects |")
    A("|---|---|---|")
    for g in groups:
        A(f"| {g} | {pooled[g]['n_trials']} | {', '.join(pooled[g]['subjects'])} |")
    A("")

    if skipped:
        A("## Recordings NOT analysed\n")
        A("These files are in the dataset but do not match `trial_pattern`, so they are not in any "
          "number on this page. Rename them or widen the pattern if they should be included.\n")
        for f in skipped:
            A(f"- `{f}`")
        A("")

    if failures:
        A("## Failed recordings\n")
        A("| trial | error |")
        A("|---|---|")
        for f in failures:
            A(f"| `{f['trial_id']}` | {str(f.get('error', ''))[:160]} |")
        A("\nFull tracebacks in `failures.json`. Rerun them with `--retry-failed`.\n")

    A("## Headline metrics — median [IQR] per task\n")
    head = [m for m in cfg.headline_metrics
            if any(m in pooled[g]["metrics"] for g in groups)]
    if head:
        A("| metric | " + " | ".join(groups) + " |")
        A("|---" * (len(groups) + 1) + "|")
        for m in head:
            cells = []
            for g in groups:
                e = pooled[g]["metrics"].get(m)
                cells.append(f"{_fmt(e['median'])} [{_fmt(e['iqr_lo'])}–{_fmt(e['iqr_hi'])}]"
                             if e else "--")
            A(f"| `{m}` | " + " | ".join(cells) + " |")
        A("")

    if ranked:
        n_subjects = len(subjects)
        has_subject_test = any(r.get("subject_test") for r in ranked)
        A(f"## Metrics that separate the task types most (top {min(15, len(ranked))})\n")
        A("Kruskal-Wallis across the groups, ranked by the trial-level `p`.\n")
        A(f"⚠️ **`p_trial` is pseudo-replicated.** It counts every recording as an observation, but "
          f"these {len(records)} trials come from **{n_subjects} subject"
          f"{'s' if n_subjects != 1 else ''}** — roughly "
          f"{len(records) // max(n_subjects, 1)} repeated measures each. The `n` behind it is "
          f"inflated by about that factor and the p is correspondingly optimistic. Use it to rank "
          f"which metrics separate the tasks cleanly; do not quote it as a result.\n")
        if has_subject_test:
            A("`p_subject` collapses each subject to one median per task and tests across "
              "subjects. That is the honest unit of replication — quote that one.\n")
        else:
            A(f"`p_subject` — the defensible test — needs at least 2 subjects per group and "
              f"cannot be computed from {n_subjects}. It appears here once more subjects are "
              f"added; nothing else about the run needs to change.\n")
        A("Rank-identical columns are folded into one row — within a single subject a torque, its "
          "mass-normalised twin and its capacity ratio differ only by a constant divisor, so they "
          "share a p and say the same thing. The folded names are in the `rank_identical_with` "
          "column of `differences.csv`.\n")
        head = ["metric", "p_trial"] + (["p_subject"] if has_subject_test else []) \
            + [f"median {g}" for g in groups] + ["folded"]
        A("| " + " | ".join(head) + " |")
        A("|---" * len(head) + "|")
        for r in ranked[:15]:
            cells = [f"`{r['metric']}`", _fmt(r["p"], 2)]
            if has_subject_test:
                cells.append(_fmt(r.get("p_subject"), 2))
            cells += [_fmt(r["medians"].get(g)) for g in groups]
            cells.append(str(len(r.get("aliases") or ())))
            A("| " + " | ".join(cells) + " |")
        A("")

    if figures:
        A("## Figures\n")
        for p in figures:
            A(f"- `figures/{Path(p).name}`")
        A("")

    A("## Caveats that still apply\n")
    A("These are properties of the underlying pipeline, not of the pooling — see "
      "\"Known limitations\" in the FRAME README:\n")
    A("- Inverse dynamics runs through `id_grf.py` with states rebuilt from the IK configurations "
      "(`filtered_states`, ID_VERSION 2): the earlier zero-acceleration / 2.08x-velocity defect "
      "of `ID_Xsens` no longer reaches the torques. Only the **arm chain** is validated; spine "
      "and leg torques rest on an unvalidated GRF model (vertical base residual ~40% of body "
      "weight on the checked trial).")
    A("- `filtfilt` edge ringing inflates peak metrics by ~5%.")

    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def write_all(records, failures, pooled, ranked, cfg, subjects, out_dir, figures=(),
              skipped=()):
    """Write every artifact of a pooled run. Returns {name: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "trials_csv": write_trials_csv(records, out / "trials.csv"),
        "pooled_summary_csv": write_pooled_summary_csv(pooled, out / "pooled_summary.csv"),
        "differences_csv": write_differences_csv(ranked, pooled, out / "differences.csv"),
        "joint_summary_csv": write_joint_summary_csv(
            pooled, out / "joint_summary.csv",
            dofs=(cfg.selected_dofs() if getattr(cfg, "joints_of_interest", None) else None)),
    }
    (out / "trials.json").write_text(json.dumps(json_safe(records), indent=1))
    (out / "pooled.json").write_text(json.dumps(json_safe(pooled), indent=1))
    (out / "failures.json").write_text(json.dumps(json_safe(failures), indent=1))
    cfg.save(out / "config.json")
    paths["trials_json"] = str(out / "trials.json")
    paths["pooled_json"] = str(out / "pooled.json")
    paths["failures_json"] = str(out / "failures.json")
    paths["config_json"] = str(out / "config.json")
    paths["summary_md"] = write_summary_md(records, failures, pooled, ranked, cfg, subjects,
                                           figures, out / "summary.md", skipped=skipped)
    return paths
