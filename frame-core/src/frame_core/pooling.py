"""
Pooling -- many per-trial metric rows -> one distribution per group.

This is the layer the pooled tool exists for. `frame_core.analyze` already reports a distribution
ACROSS REPETITIONS WITHIN one recording; pooling reports the distribution ACROSS RECORDINGS,
grouped by whatever factor the study varies (here: the task type H / M / L).

Three deliberate choices:

  * **Per-trial scalars, then pool.** Each trial is analysed on its own and contributes one value
    per metric. The alternative -- concatenating every trial's frames into one Capture -- would
    mix subjects inside a single metric and make a bad trial impossible to find or drop.

  * **Non-parametric summaries.** `summarize` reports median and IQR, not mean +- SD, because
    n per group is small (24-36 trials for one subject) and peak-torque style metrics are
    right-skewed. The mean and SD are still emitted for anyone who wants them.

  * **Missing values shrink `n`, they do not become zero.** Every summary carries its own `n`, so
    a metric that only some trials could produce is never silently averaged against zeros.

Nothing here does I/O or plotting -- it is the citable half of the pooled analysis.
"""
from __future__ import annotations
import math
import numpy as np

from .types import summarize


# What the single group is called when nothing is being split by. A name, not an empty tuple,
# because it becomes an axis tick, a CSV cell and a legend entry.
ALL_GROUP = "all"


def group_rows(rows, by=("task",)):
    """Bucket trial records by one or more metadata fields. Returns {key: [row, ...]}.

    A single field gives a plain string key ("H"); several give a tuple ("H", "1817").

    An EMPTY `by` is meaningful and supported: every record lands in one group called "all". That
    is how you ask for one distribution over the whole dataset rather than one per task -- note
    that filtering (`--tasks H M L`) selects which recordings take part, while `by` decides how
    the ones that took part are split up. They are different questions.
    """
    by = (by,) if isinstance(by, str) else tuple(by)
    out = {}
    for r in rows:
        if not by:
            out.setdefault(ALL_GROUP, []).append(r)
            continue
        key = tuple(r.get(f) for f in by)
        out.setdefault(key[0] if len(key) == 1 else key, []).append(r)
    return out


def metric_keys(rows, metrics_field="metrics"):
    """Every metric column present in any row, in first-seen order (stable across runs)."""
    seen = {}
    for r in rows:
        for k in (r.get(metrics_field) or {}):
            seen.setdefault(k, None)
    return list(seen)


def pool_metric(rows, key, metrics_field="metrics"):
    """Distribution of one metric across trials, with the contributing trials named.

    `values` and `trials` stay index-aligned so an outlier in the summary can always be traced
    back to the recording that produced it -- the single most useful thing when a pooled number
    looks wrong.
    """
    values, trials, subjects = [], [], []
    for r in rows:
        v = (r.get(metrics_field) or {}).get(key)
        if v is None:
            continue
        v = float(v)
        if not math.isfinite(v):
            continue
        values.append(v)
        trials.append(r.get("trial_id"))
        subjects.append(r.get("subject"))
    out = summarize(values)
    out["values"] = values
    out["trials"] = trials
    # Index-aligned with `values`, so a subject-level test can collapse trials to one value per
    # person without going back to trials.csv -- see compare_groups(level="subject").
    out["subject_of_trial"] = subjects
    out["n_missing"] = len(rows) - len(values)
    return out


def pool(rows, by=("task",), metrics_field="metrics"):
    """The whole pooled result: {group: {n_trials, trials, subjects, metrics}}.

    `rows` are trial records shaped like
        {"trial_id": str, "subject": str, "task": str, ..., "metrics": {k: v}}
    which is exactly what `frame_batch.runner` emits.
    """
    grouped = group_rows(rows, by)
    out = {}
    for key, members in grouped.items():
        keys = metric_keys(members, metrics_field)
        out[key] = {
            "n_trials": len(members),
            "trials": [m.get("trial_id") for m in members],
            "subjects": sorted({m.get("subject") for m in members if m.get("subject")}),
            "metrics": {k: pool_metric(members, k, metrics_field) for k in keys},
        }
    return out


def _subject_medians(pooled, metric, group):
    """One value per subject for this (group, metric): that subject's median across its trials."""
    entry = (pooled.get(group) or {}).get("metrics", {}).get(metric)
    if not entry:
        return {}
    by_subject = {}
    for v, t in zip(entry.get("values", ()), entry.get("subject_of_trial", ())):
        by_subject.setdefault(t, []).append(v)
    return {s: float(np.median(v)) for s, v in by_subject.items() if v}


def compare_groups(pooled, metric, groups=None, level="trial"):
    """Kruskal-Wallis across the pooled groups for one metric, plus each group's median.

    Non-parametric on purpose (see the module docstring): these are small, skewed samples.

    **`level` is the difference between a screening aid and a defensible test.**

      * `level="trial"` (default) treats every recording as an observation. Trials are repeated
        measures of the same few people, so this is PSEUDO-REPLICATED: with 24 trials per subject
        the `n` is inflated ~24x and `p` is correspondingly optimistic. It is a useful ranking of
        which metrics separate the tasks most cleanly. It is not a population result, at any
        number of subjects.

      * `level="subject"` collapses each subject to one median per group first, then tests across
        subjects. That is the honest unit of replication, and it needs >= 2 subjects per group to
        run at all -- which is exactly why it reports `None` rather than a number on a one-subject
        dataset, instead of quietly handing back the inflated version.

    `n_subjects` is reported either way, because "p = 1e-10 from 2 people" is a sentence that
    should never be readable without the second half.
    """
    groups = list(groups or pooled)
    samples, medians = [], {}
    for g in groups:
        entry = (pooled.get(g) or {}).get("metrics", {}).get(metric)
        vals = list(entry["values"]) if entry else []
        medians[g] = float(np.median(vals)) if vals else float("nan")
        if level == "subject":
            vals = list(_subject_medians(pooled, metric, g).values())
        if len(vals) >= 2:
            samples.append(vals)
    n_subjects = len({s for g in groups for s in (pooled.get(g) or {}).get("subjects", [])})
    result = {"metric": metric, "groups": groups, "medians": medians, "level": level,
              "n": {g: len((pooled.get(g) or {}).get("metrics", {}).get(metric, {}).get("values", []))
                    for g in groups},
              "n_subjects": n_subjects,
              # Trials are ALWAYS clustered within a subject. The trial-level test ignores that.
              "pseudoreplicated": level == "trial"}
    if len(samples) < 2:
        why = ("needs >= 2 subjects per group -- a subject-level test cannot be run on this many "
               "subjects" if level == "subject" else "needs >= 2 groups with >= 2 finite values")
        result.update({"test": None, "H": float("nan"), "p": float("nan"), "reason": why})
        return result
    try:
        from scipy.stats import kruskal
        H, p = kruskal(*samples)
        result.update({"test": "kruskal", "H": float(H), "p": float(p)})
    except Exception as e:                       # identical values -> scipy raises, not a failure
        result.update({"test": "kruskal", "H": float("nan"), "p": float("nan"),
                       "reason": str(e)})
    return result


def _canonical_first(names):
    """Of several rank-identical metrics, the one to show: raw SI over its derived twins.

    `peak_torque_Nm`, `peak_torque_Nm__norm` (divided by a constant body mass) and
    `peak_torque_ratio` (divided by a constant joint effort limit) are the same numbers rescaled,
    so within one subject they carry one piece of information between them.
    """
    return sorted(names, key=lambda n: ("__norm" in n, n.endswith("_ratio"), len(n), n))[0]


def ranked_differences(pooled, metrics=None, groups=None, top=25, dedupe=True):
    """The metrics that separate the groups most, p-ascending -- a reading order for the report.

    Without this a pooled run is a thousand-column table nobody reads. With it, the first screen
    answers "what actually changes between H, M and L".

    `dedupe` collapses metrics that are RANK-IDENTICAL. Kruskal-Wallis depends only on ranks, so
    two metrics sharing a p to full precision have the same rank structure and one tells you
    nothing the other does not -- which is exactly what happens to every torque column, since its
    `__norm` twin and its `_ratio` sibling differ only by a constant divisor. Left in, they push
    the top of the list down to a third of the metrics it appears to show. The collapsed names are
    kept on the surviving row as `aliases`, so nothing is hidden, only folded.
    """
    groups = list(groups or pooled)
    if metrics is None:
        metrics = []
        for g in groups:
            for k in (pooled.get(g) or {}).get("metrics", {}):
                if k not in metrics:
                    metrics.append(k)
    rows = []
    for m in metrics:
        r = compare_groups(pooled, m, groups, level="trial")
        # The subject-level test rides along on every row. It is the defensible one, and it must
        # not be something you have to know to ask for -- if it is computable it is reported, and
        # if it is not, its absence says why (too few subjects) instead of being silently omitted.
        sub = compare_groups(pooled, m, groups, level="subject")
        r["p_subject"] = sub.get("p")
        r["subject_test"] = sub.get("test")
        r["subject_reason"] = sub.get("reason")
        rows.append(r)
    rows = [r for r in rows if r.get("p") == r.get("p")]          # drop NaN p
    rows.sort(key=lambda r: r["p"])

    if dedupe:
        by_p, order = {}, []
        for r in rows:
            # p and H alone are NOT enough: Kruskal-Wallis is symmetric under reversal, so a
            # metric that RISES across H/M/L and one that FALLS the same way share both. Folding
            # those would hide a real, opposite finding. The ordering of the group medians
            # separates them, and a positive constant divisor (what `__norm` and `_ratio` are)
            # leaves that ordering untouched -- so the real rescalings still fold.
            meds = [r["medians"].get(g, float("nan")) for g in groups]
            direction = tuple(sorted(range(len(meds)),
                                     key=lambda i: (meds[i] != meds[i], meds[i])))
            key = (round(r["p"], 12),
                   round(r.get("H") if r.get("H") == r.get("H") else 0.0, 9),
                   direction)
            if key not in by_p:
                by_p[key] = []
                order.append(key)
            by_p[key].append(r)
        deduped = []
        for key in order:
            family = by_p[key]
            keep_name = _canonical_first([r["metric"] for r in family])
            row = next(r for r in family if r["metric"] == keep_name)
            row = dict(row)
            row["aliases"] = sorted(r["metric"] for r in family if r["metric"] != keep_name)
            deduped.append(row)
        rows = deduped

    return rows[:top] if top else rows
