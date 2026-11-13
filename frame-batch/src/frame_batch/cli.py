"""
`python -m frame_batch` -- the whole pooled tool from one command.

    frame-batch run                       analyse data/, pool by task, write results/
    frame-batch run --subjects 1817 --tasks H --reps 1 2    a two-trial smoke test first
    frame-batch list                      what the dataset looks like, without analysing anything
    frame-batch report                    rebuild the pooling + figures from cached trials (free)

`run` is restartable: every finished trial is cached, so re-running after an interruption picks up
where it stopped, and `report` re-does the cheap half (pooling, tables, figures) without touching
IK at all. That split matters when the expensive half takes hours.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from .config import BatchConfig
from .dataset import discover, all_trials, DatasetError


def _add_common(p):
    p.add_argument("--config", help="batch config JSON (default: batch.json if present)")
    p.add_argument("--data", dest="data_root", help="dataset root (default: data)")
    p.add_argument("--out", dest="output_root", help="results directory (default: results)")
    p.add_argument("--subjects", nargs="+", help="analyse only these subject ids")
    p.add_argument("--tasks", nargs="+", help="analyse only these task types (e.g. H M)")
    p.add_argument("--reps", nargs="+", type=int, help="analyse only these repetition numbers")


def _load_config(args):
    """batch.json, then the environment, then explicit flags -- in that order of authority.

    The environment layer exists for the container: a device mount point can contain characters
    Docker refuses in a volume spec, so the wrapper mounts the dataset and the cache at fixed
    internal paths and announces them through FRAME_DATA_ROOT / FRAME_CACHE_ROOT. batch.json keeps
    the real host paths, which is what a human reads and what a local (non-Docker) run uses.
    """
    import os

    path = args.config
    if path is None and Path("batch.json").is_file():
        path = "batch.json"
    overrides = {}
    for field, env in (("data_root", "FRAME_DATA_ROOT"), ("cache_root", "FRAME_CACHE_ROOT"),
                       ("output_root", "FRAME_OUTPUT_ROOT")):
        if os.environ.get(env):
            overrides[field] = os.environ[env]
    for k in ("data_root", "output_root", "jobs"):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    return BatchConfig.load(path, **overrides)


def _discover(cfg, args, skipped=None):
    """Discover the dataset and SAY what was skipped.

    A recording that does not match the trial pattern is dropped from the analysis. That is the
    right behaviour -- datasets hold notes, exports and aborted takes -- but it must never be
    silent, or the report quietly describes fewer recordings than the folder contains.
    """
    skipped = [] if skipped is None else skipped
    subjects = discover(cfg.data_root, cfg.trial_pattern,
                        subjects=args.subjects, tasks=args.tasks, reps=args.reps,
                        strict=cfg.strict_filenames, skipped=skipped)
    if skipped:
        print(f"! {len(skipped)} file(s) did not match the trial pattern and were NOT analysed:",
              file=sys.stderr)
        for path in skipped[:10]:
            print(f"    {path}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more", file=sys.stderr)
    return subjects


# --------------------------------------------------------------------------------------------- #
def cmd_list(args):
    cfg = _load_config(args)
    subjects = _discover(cfg, args)
    total_mb = 0.0
    for s in subjects:
        print(f"\n{s.describe()}")
        by_task = {}
        for t in s.trials:
            by_task.setdefault(t.task, []).append(t)
            total_mb += t.size_mb if t.size_mb == t.size_mb else 0.0
        for task in sorted(by_task):
            trials = sorted(by_task[task], key=lambda t: t.trial_id)
            factors = sorted({tuple(sorted(t.factors.items())) for t in trials})
            print(f"   {task}: {len(trials):3d} trials   "
                  + "  ".join("/".join(v for _, v in f) for f in factors))
    n = len(all_trials(subjects))
    print(f"\n{n} trials, {total_mb / 1000:.1f} GB total.")
    print(f"pooled by: {', '.join(cfg.pool_by)}   |   config fingerprint: {cfg.fingerprint()}")
    return 0


def _apply_joint_filter(records, cfg):
    """Narrow the reported metrics to `joints_of_interest`, leaving the cache untouched.

    The filter is applied to COPIES, once, here -- so trials.csv, pooled.json, differences.csv and
    every figure describe the same joints, and the cached records still hold the full chain for
    the next time the list changes.
    """
    from frame_core.flatten import filter_record

    bad_figs = cfg.unknown_figures()
    if bad_figs:
        print(f"! figures names {len(bad_figs)} unknown figure(s): {', '.join(bad_figs)}",
              file=sys.stderr)
    unknown = cfg.unknown_joints()
    if unknown:
        print(f"! joints_of_interest names {len(unknown)} joint(s) the model does not have: "
              f"{', '.join(unknown)}", file=sys.stderr)
    if not cfg.joints_of_interest:
        return records
    dofs = cfg.selected_dofs()
    if not dofs:
        print("! joints_of_interest matched no analysed joint; reporting on all of them.",
              file=sys.stderr)
        return records
    print(f"  reporting on {len(dofs)} of {len(cfg.dof_names)} joints: {', '.join(dofs)}")
    return [filter_record(r, dofs) for r in records]


def _pool_and_report(records, failures, cfg, subjects, out_dir, make_figures=True,
                     cache_root=None, skipped=()):
    from frame_core.pooling import pool, ranked_differences
    from . import report

    unfiltered = records          # the globe's working-arm check needs BOTH arms' ROM
    records = _apply_joint_filter(records, cfg)
    # Report-time metrics read from the cached Captures (no re-analysis): torque exceedance of a
    # fraction of the joint capacity. Merged into each record BEFORE pooling so they reach every
    # table like any other metric.
    if cache_root:
        try:
            from .cache import TrialCache
            from . import exceedance
            records, n_missing = exceedance.augment(records, TrialCache(cache_root), cfg)
            if n_missing:
                print(f"  ! exceedance: {n_missing} trial(s) had no cached capture",
                      file=sys.stderr)
        except Exception as e:
            print(f"  ! exceedance metrics skipped: {type(e).__name__}: {e}", file=sys.stderr)
    # Shoulder globe (report time, raw captures): per-recording coverage / demand metrics, merged
    # before pooling; its figures are drawn here too because they come from the same pass.
    globe_figs = []
    if cache_root and cfg.wants("shoulder_globe_by_task"):
        try:
            from .cache import TrialCache
            from . import globe
            globe_figs, gmetrics, notes = globe.build(
                records, TrialCache(cache_root), cfg, Path(out_dir) / "figures",
                cache_dir=cache_root, arm_records=unfiltered)
            records = globe.augment(records, gmetrics)
            for n in notes:
                print(f"  ! shoulder globe: {n}", file=sys.stderr)
        except Exception as e:
            print(f"  ! shoulder globe skipped: {type(e).__name__}: {e}", file=sys.stderr)
    pooled = pool(records, by=cfg.pool_by)
    ranked = ranked_differences(pooled, top=cfg.top_differences)

    # Joint capacities come from the cached Captures, so the torque and speed figures can show the
    # limit the data should be read against without anything being re-analysed.
    joint_limits = None
    if make_figures and cache_root:
        try:
            from .cache import TrialCache
            from . import limits as limits_mod
            joint_limits = limits_mod.collect(records, TrialCache(cache_root),
                                              dofs=set(cfg.selected_dofs()), cfg=cfg)
            if limits_mod.any_varies(joint_limits):
                print("  ! joint limits differ between subjects or recordings (the shoulder "
                      "elevation capacities are pose-dependent); the figures draw the median",
                      file=sys.stderr)
        except Exception as e:
            print(f"  ! joint limits unavailable: {type(e).__name__}: {e}", file=sys.stderr)

    figures = list(globe_figs) if make_figures else []
    if make_figures:
        try:
            from . import plots
            figures += plots.build_all(pooled, ranked, cfg, Path(out_dir) / "figures",
                                       limits=joint_limits)
        except Exception as e:                    # figures are debugging aids, never the result
            print(f"  ! figures skipped: {type(e).__name__}: {e}", file=sys.stderr)

        if cfg.wants("torque_exceedance_by_task"):
            try:
                from . import exceedance
                p = exceedance.build_figure(pooled, cfg, Path(out_dir) / "figures")
                if p:
                    figures.append(p)
            except Exception as e:
                print(f"  ! torque_exceedance_by_task skipped: {type(e).__name__}: {e}",
                      file=sys.stderr)

        # These read the cached Captures rather than the pooled scalars, so they are built
        # separately and are each allowed to be absent (captures can be turned off).
        if cache_root:
            from .cache import TrialCache
            from . import timeseries, torque_speed, trajectory3d, npose, joint_pose
            cache = TrialCache(cache_root)
            figs_dir = Path(out_dir) / "figures"
            for mod, kwargs in ((timeseries, {}), (torque_speed, {}),
                                (trajectory3d, {"subjects": subjects}), (npose, {}),
                                (joint_pose, {})):
                if not cfg.wants(mod.FIGURE_NAME):
                    continue
                try:
                    path, note = mod.build(records, cache, cfg, figs_dir, **kwargs)
                    if isinstance(path, (list, tuple)):
                        figures.extend(path)
                    elif path:
                        figures.append(path)
                    if note:
                        print(f"  ! {note}", file=sys.stderr)
                except Exception as e:
                    print(f"  ! {mod.FIGURE_NAME} skipped: {type(e).__name__}: {e}",
                          file=sys.stderr)

    # The figures above saw every metric; the written tables leave out `exclude_metrics`.
    records = [dict(r, metrics={k: v for k, v in (r.get("metrics") or {}).items()
                                if cfg.keep_metric(k)}) for r in records]
    pooled = {g: dict(e, metrics={k: v for k, v in e["metrics"].items() if cfg.keep_metric(k)})
              for g, e in pooled.items()}
    ranked = ranked_differences(pooled, top=cfg.top_differences)
    paths = report.write_all(records, failures, pooled, ranked, cfg, subjects, out_dir,
                             figures=figures, skipped=skipped)
    # G and the reversal metrics per recording x DOF -- only while `passivity.*` is reported.
    from . import exo_eval
    if any(cfg.keep_metric(f"passivity.x.{k}") for k in ("G", "n_reversals")):
        paths.update({f"exo_eval.{k}": v
                      for k, v in exo_eval.write(records, cfg, out_dir).items()})
    return pooled, ranked, paths, figures


def cmd_run(args):
    cfg = _load_config(args)
    skipped = []
    subjects = _discover(cfg, args, skipped)
    out_dir = Path(cfg.output_root)
    cache_root = args.cache or cfg.resolved_cache_root()

    from .cache import TrialCache
    from .runner import run_sweep

    if args.retry_failed:
        n = TrialCache(cache_root).clear_failures()
        print(f"cleared {n} cached failure(s); they will be retried.")

    trials = all_trials(subjects)
    print(f"{len(subjects)} subject(s), {len(trials)} trial(s), {args.jobs or cfg.jobs} worker(s)")
    for s in subjects:
        print(f"  {s.describe()}")

    t0 = time.time()
    done = {"n": 0}

    def progress(stage, what, extra):
        if stage == "urdf":
            print(f"  [model] subject {what}", flush=True)
            return
        done["n"] += 1
        tag = {"cached": "cache", "cached-failure": "FAIL*", "done": "ok", "failed": "FAIL"}.get(
            stage, stage)
        suffix = f"  {extra}s" if stage == "done" else (f"  {extra}" if extra else "")
        print(f"  [{done['n']:3d}/{len(trials)}] {tag:6s} {what}{suffix}", flush=True)

    records, failures = run_sweep(subjects, cfg, cache_root=cache_root,
                                  jobs=args.jobs, progress=progress,
                                  skip_cached_failures=not args.retry_failed)
    dt = time.time() - t0
    print(f"\n{len(records)} analysed, {len(failures)} failed, {dt / 60:.1f} min")

    if not records:
        print("nothing analysed -- see failures.json", file=sys.stderr)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=1))
        return 1

    pooled, ranked, paths, figures = _pool_and_report(
        records, failures, cfg, subjects, out_dir, make_figures=not args.no_figures,
        cache_root=cache_root, skipped=skipped)
    print(f"\npooled into {len(pooled)} group(s): "
          + ", ".join(f"{g} (n={pooled[g]['n_trials']})" for g in sorted(pooled, key=str)))
    print(f"wrote {paths['summary_md']}")
    for name in ("trials_csv", "pooled_json", "differences_csv"):
        print(f"      {paths[name]}")
    for f in figures:
        print(f"      {f}")
    return 0 if not failures else 2


def cmd_report(args):
    """Re-pool and re-plot from the cache -- no IK, no MVNX parsing, seconds not hours."""
    cfg = _load_config(args)
    skipped = []
    subjects = _discover(cfg, args, skipped)
    out_dir = Path(cfg.output_root)
    cache_root = args.cache or cfg.resolved_cache_root()

    from .cache import TrialCache, trial_key
    from .runner import refresh_metadata
    cache = TrialCache(cache_root)
    fp = cfg.fingerprint()

    records, failures, missing = [], [], []
    for subject, trial in all_trials(subjects):
        key = trial_key(trial.path, fp, subject.fingerprint())
        hit = cache.load(subject.subject_id, trial.trial_id, key)
        if hit is not None:
            records.append(refresh_metadata(hit, trial, subject))
            continue
        bad = cache.load_failure(subject.subject_id, trial.trial_id, key)
        (failures if bad else missing).append(bad or trial.trial_id)

    if missing:
        subjects_missing = sorted({t.split("_")[0] for t in missing if isinstance(t, str)})
        print(f"{len(missing)} trial(s) have not been analysed yet "
              f"({len(subjects_missing)} subject(s): {', '.join(subjects_missing)}). "
              f"`report` only reads the cache -- run `./frame run --jobs 6` to analyse them. "
              f"Reporting on the {len(records)} already done.", file=sys.stderr)
    if not records:
        print("nothing cached for this config.", file=sys.stderr)
        return 1

    records.sort(key=lambda r: (r.get("subject", ""), r.get("task", ""), r.get("trial_id", "")))
    pooled, ranked, paths, figures = _pool_and_report(
        records, failures, cfg, subjects, out_dir, make_figures=not args.no_figures,
        cache_root=cache_root, skipped=skipped)
    print(f"re-pooled {len(records)} cached trial(s) into "
          + ", ".join(f"{g} (n={pooled[g]['n_trials']})" for g in sorted(pooled, key=str)))
    print(f"wrote {paths['summary_md']}")
    for f in figures:
        print(f"      {f}")
    return 0


def cmd_reanalyse(args):
    """Rebuild every trial's metrics from its cached Capture under the CURRENT config.

    For settings that act after inverse dynamics -- `shoulder_coords`, capacity overrides, the metric
    set -- this replaces a full `run` (hours of IK) with minutes. Then pools and reports.
    """
    from .runner import run_reanalyse
    cfg = _load_config(args)
    skipped = []
    subjects = _discover(cfg, args, skipped)
    cache_root = args.cache or cfg.resolved_cache_root()
    n = len(all_trials(subjects))
    print(f"re-analysing {n} trial(s) from the cached captures "
          f"(shoulder coordinates: {cfg.shoulder_coords}), {args.jobs or cfg.jobs} worker(s)")
    done = {"n": 0}

    def progress(status, tid, why):
        done["n"] += 1
        if status in ("missing", "failed") or done["n"] % 50 == 0:
            print(f"  [{done['n']:4d}/{n}] {status:7s} {tid}" + (f"  {why}" if why else ""),
                  flush=True)

    out = run_reanalyse(subjects, cfg, cache_root, jobs=args.jobs or cfg.jobs,
                        force=args.force, progress=progress)
    print(f"{len(out['done'])} re-analysed, {len(out['current'])} already current, "
          f"{len(out['missing'])} without a usable cache entry, {len(out['failed'])} failed")
    if args.no_report:
        return 0 if not out["failed"] else 2
    return cmd_report(args)


def cmd_config(args):
    """Print the effective config -- what a run would actually use, after file + flags."""
    print(json.dumps(_load_config(args).to_dict(), indent=2))
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="frame-batch",
        description="FRAME pooled analysis: analyse every recording in a dataset and report each "
                    "metric's distribution per task type.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="show the dataset without analysing it")
    _add_common(pl)
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("run", help="analyse every trial, pool, and write the report")
    _add_common(pr)
    pr.add_argument("--jobs", type=int, help="parallel trials (default: config `jobs`)")
    pr.add_argument("--cache", help="cache directory (default: <out>/cache)")
    pr.add_argument("--retry-failed", action="store_true",
                    help="drop cached failures and analyse them again")
    pr.add_argument("--no-figures", action="store_true", help="tables only, skip the plots")
    pr.set_defaults(func=cmd_run)

    pp = sub.add_parser("report", help="re-pool and re-plot from cached trials (no IK)")
    _add_common(pp)
    pp.add_argument("--cache", help="cache directory (default: <out>/cache)")
    pp.add_argument("--no-figures", action="store_true")
    pp.set_defaults(func=cmd_report)

    pa = sub.add_parser("reanalyse", help="recompute metrics from cached captures (no IK), "
                                           "then report")
    _add_common(pa)
    pa.add_argument("--jobs", type=int, help="parallel trials (default: config `jobs`)")
    pa.add_argument("--cache", help="cache directory (default: <out>/cache)")
    pa.add_argument("--force", action="store_true", help="also redo trials already current")
    pa.add_argument("--no-report", action="store_true", help="skip the pooling + report step")
    pa.add_argument("--no-figures", action="store_true")
    pa.set_defaults(func=cmd_reanalyse)

    pc = sub.add_parser("config", help="print the effective configuration")
    _add_common(pc)
    pc.set_defaults(func=cmd_config)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ("jobs", "cache", "retry_failed", "no_figures", "force", "no_report"):
        if not hasattr(args, name):
            setattr(args, name, None)
    try:
        return args.func(args)
    except DatasetError as e:
        print(f"dataset error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"missing file: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
