"""
The per-trial heavy run, and the sweep that drives it.

One trial goes:

    .mvnx --(utils converter)--> movement CSV --(IK + ID, Pinocchio)--> Capture
          --(frame_core.analyze)--> nested result --(flatten)--> ONE ROW of scalars

and that row, plus the trial's metadata, is what `frame_core.pooling` later turns into the H / M / L
distributions.

Three things this module is careful about, all of them consequences of running 84 recordings
instead of one:

  * **The subject's DHM is built once, before the pool starts.** Building it inside parallel
    workers would race several processes onto the same output path, and keying it per recording
    would let calibration jitter masquerade as task variance.

  * **The converter writes next to its input**, so each trial runs against a symlink in its own
    temp directory. The 100 MB CSVs never land in `data/`, and two trials can never collide.

  * **One trial's failure is not the sweep's failure.** An unreadable recording is recorded,
    reported at the end, and the other 83 still produce a distribution.
"""
from __future__ import annotations
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

from .cache import TrialCache, trial_key, json_safe


# --------------------------------------------------------------------------------------------- #
# subject-level prepass
# --------------------------------------------------------------------------------------------- #
def ensure_subject_urdf(subject, cfg, builder=None, force=False):
    """Build (or find) the one DHM URDF for this subject. Returns its path.

    Runs BEFORE any parallelism: `create_urdf_from_Xsens_new` writes a fixed path derived from the
    .mvnx it is given, so two workers building the same subject would write the same file at the
    same time. Doing it once here also means the remaining trials never need to touch the .mvnx
    for anything but their own motion data.
    """
    from frame_kinematics.dhm import build_or_load_urdf

    if not subject.trials:
        raise ValueError(f"subject {subject.subject_id} has no trials to build a model from.")
    seed = sorted(subject.trials, key=lambda t: t.trial_id)[0]
    urdf = build_or_load_urdf(
        models_dir=cfg.models_dir, xacro_base=cfg.xacro_base,
        height_m=subject.height_m, mass_kg=subject.mass_kg,
        mvnx_path=seed.path, subject_id=subject.subject_id,
        urdf_stem=subject.urdf_stem, builder=builder, force=force)
    return str(urdf)


# --------------------------------------------------------------------------------------------- #
# one trial
# --------------------------------------------------------------------------------------------- #
def _kinematics_job(trial, subject, cfg, urdf_path, work_mvnx):
    from frame_kinematics.interfaces import KinematicsJob
    return KinematicsJob(
        dof_names=list(cfg.dof_names),
        ee_segment=cfg.ee_segment, ee_frame=cfg.ee_frame, trunk_segment=cfg.trunk_segment,
        fs=cfg.fs or 240.0, min_len=cfg.min_len,
        trim_start_s=cfg.trim_start_s, trim_end_s=cfg.trim_end_s,
        height_m=subject.height_m, mass_kg=subject.mass_kg,
        subject_id=subject.subject_id, urdf_stem=subject.urdf_stem, urdf_path=urdf_path,
        models_dir=cfg.models_dir, xacro_base=cfg.xacro_base, mvnx_path=work_mvnx,
        exo_urdf=cfg.exo_urdf, exo_trunk_frame=cfg.exo_trunk_frame,
    )


def _link_or_copy(src, dst):
    """Symlink the recording into the work dir, falling back to a copy where links are refused."""
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src, dst)


def analyse_capture(cap, cfg, subject):
    """Capture -> (flat metric row, nested result, n_reps). The pure half; no Pinocchio.

    `cap` is the RAW capture (DHM coordinates, as cached); the configured shoulder coordinates are
    applied here, so the cache never has to be rewritten to change them.
    """
    from frame_core.pipeline import analyze
    from frame_core.flatten import flatten_result, normalize_row, n_reps

    cap = cfg.transform_capture(cap)
    core_cfg = cfg.core_config()
    core_cfg.fs = cap.fs
    result = analyze(cap, core_cfg, urdf_limits=cap.urdf_limits)

    metrics = flatten_result(result, cfg.analysed_dofs())
    # how much of each shoulder plane / rotation trace is defined (elevation >= the mask angle)
    for side, inf in (getattr(cap, "shoulder_info", None) or {}).items():
        vf = inf.get("valid_frac") or []
        if vf:
            for c in ("plane", "rotation"):
                metrics[f"joint_space.{side}_shoulder_{c}.valid_frac"] = float(sum(vf) / len(vf))
    if cfg.normalize:
        metrics.update(normalize_row(metrics, height_m=subject.height_m, mass_kg=subject.mass_kg))
    return metrics, result, n_reps(result)


METADATA_FIELDS = ("trial_id", "subject", "task", "rep", "path", "height_m", "mass_kg")


def refresh_metadata(record, trial, subject):
    """Re-apply the CURRENT filename metadata to a record loaded from the cache.

    The cache stores expensive computation, not labels. Renaming a factor group -- or dropping one
    the study turned out not to need -- must not cost a re-analysis, so the task, repetition and
    factor columns are re-derived from the current `trial_pattern` every time a cached record is
    read. Stale factor keys from an older pattern are dropped rather than left alongside the new
    ones, or `trials.csv` would slowly accumulate columns nobody can explain.

    Anything that could change a NUMBER is deliberately not refreshed here: height and mass are in
    the cache key instead, so a changed anthropometry produces a miss and a real re-analysis.
    """
    known = set(METADATA_FIELDS) | {"metrics", "capability", "urdf", "cache_key",
                                    "cached", "ok", "fs", "n_frames", "n_reps", "duration_s"}
    out = {k: v for k, v in record.items() if k in known}
    out.update({
        "trial_id": trial.trial_id, "subject": subject.subject_id,
        "task": trial.task, "rep": trial.rep, "path": trial.path,
        "height_m": subject.height_m, "mass_kg": subject.mass_kg,
    })
    out.update(trial.factors)
    return out


def run_trial(trial, subject, cfg, urdf_path, cache=None, make_ik=None, converter=None,
              run_id=None, jac_fn=None):
    """Analyse ONE recording end to end and return its trial record.

    The record is the unit everything downstream consumes: flat metrics and
    enough provenance (subject, task, factors, model, frame counts, duration) that any pooled
    number can be traced back to the recordings behind it.

    `make_ik` / `run_id` / `jac_fn` / `converter` are the same injection seams frame-kinematics
    uses, so the whole pooled path is exercised under test with fakes and touches Pinocchio only
    through the production defaults.
    """
    import pandas as pd
    from frame_kinematics.mvnx import mvnx_to_csv
    from frame_kinematics.worker import run_job
    from frame_kinematics.serialize import capture_to_npz_bytes

    t0 = time.time()
    key = trial_key(trial.path, cfg.fingerprint(), subject.fingerprint())
    if cache is not None:
        hit = cache.load(subject.subject_id, trial.trial_id, key)
        if hit is not None:
            return refresh_metadata(hit, trial, subject) | {"cached": True}

    work_dir = tempfile.mkdtemp(prefix=f"frame_{trial.trial_id}_")
    try:
        work_mvnx = os.path.join(work_dir, f"{trial.trial_id}.mvnx")
        _link_or_copy(trial.path, work_mvnx)

        csv_path, frame_rate = mvnx_to_csv(work_mvnx, converter=converter)
        fs = float(cfg.fs or frame_rate or 240.0)
        df = pd.read_csv(csv_path)          # the CONVERTED csv has no units row to drop

        job = _kinematics_job(trial, subject, cfg, urdf_path, work_mvnx)
        job.fs = fs
        cap = run_job(df, job, make_ik=make_ik, run_id=run_id, jac_fn=jac_fn,
                      want_torque=cfg.want_torque, want_jac=cfg.want_jac)

        metrics, result, n_rep = analyse_capture(cap, cfg, subject)

        record = {
            "trial_id": trial.trial_id,
            "subject": subject.subject_id,
            "task": trial.task,
            "rep": trial.rep,
            **{k: v for k, v in trial.factors.items()},
            "path": trial.path,
            "height_m": subject.height_m,
            "mass_kg": subject.mass_kg,
            "fs": fs,
            "n_frames": int(sum(r.n for r in cap.reps)),
            "n_reps": int(n_rep),
            "urdf": urdf_path,
            "capability": result.get("capability"),
            "metrics": metrics,
            "duration_s": round(time.time() - t0, 1),
            "cache_key": key,
            "cached": False,
            "ok": True,
        }
        # Sanitise here, not only on write: a metric that was not measurable must look the same
        # whether this record came from the cache or was just computed.
        record = json_safe(record)
        if cache is not None:
            blob = (capture_to_npz_bytes(cap, float32=cfg.cache_float32,
                                         decimate=cfg.cache_decimate)
                    if cfg.cache_captures else None)
            cache.save(subject.subject_id, trial.trial_id, record, capture_bytes=blob)
        return record
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def source_keys(trial, subject, cfg):
    """Cache keys whose record shares this trial's RAW capture with the current config: the same
    analysis settings under either shoulder coordinate system (the capture is always raw DHM)."""
    import dataclasses
    from frame_core.flatten import METRICS_VERSION
    keys = []
    for mv in range(METRICS_VERSION, 0, -1):     # the capture does not depend on the metric set
        for coords in ("dhm", "elevation"):
            for over in (cfg.capacity_overrides, {}):
                c = dataclasses.replace(cfg, shoulder_coords=coords, capacity_overrides=over)
                k = trial_key(trial.path, c.fingerprint(metrics_version=mv), subject.fingerprint())
                if k not in keys:
                    keys.append(k)
    return keys


def reanalyse_trial(trial, subject, cfg, cache_root, force=False):
    """Rebuild one trial's record from its cached raw Capture -- no MVNX, no IK, no ID.

    Returns (status, record_or_reason): status in {"done", "current", "missing"}.
    """
    from frame_kinematics.serialize import capture_from_npz_bytes
    cache = TrialCache(cache_root)
    key = trial_key(trial.path, cfg.fingerprint(), subject.fingerprint())
    if not force and cache.load(subject.subject_id, trial.trial_id, key) is not None:
        return "current", None
    src = None
    for k in source_keys(trial, subject, cfg):
        src = cache.load(subject.subject_id, trial.trial_id, k)
        if src is not None:
            break
    blob = cache.load_capture_bytes(subject.subject_id, trial.trial_id) if src else None
    if src is None or not blob:
        return "missing", "no cached record + capture for these analysis settings"
    t0 = time.time()
    cap = capture_from_npz_bytes(blob)
    if getattr(cap, "decimate", 1) != 1 or (src.get("n_frames") and
                                           sum(r.n for r in cap.reps) != src["n_frames"]):
        return "missing", "cached capture is decimated -- metrics need the native rate; re-run"
    metrics, result, n_rep = analyse_capture(cap, cfg, subject)
    record = dict(src)
    record.update({"metrics": metrics, "capability": result.get("capability"),
                   "n_reps": int(n_rep), "cache_key": key, "cached": False,
                   "reanalysed_from_capture": True,
                   "reanalyse_s": round(time.time() - t0, 1)})
    record = json_safe(record)
    cache.save(subject.subject_id, trial.trial_id, record)
    return "done", record


def _reanalyse_worker(args):
    trial, subject, cfg, cache_root, force = args
    try:
        return trial.trial_id, reanalyse_trial(trial, subject, cfg, cache_root, force)
    except BaseException as e:                  # noqa: BLE001
        return trial.trial_id, ("failed", f"{type(e).__name__}: {e}")


def run_reanalyse(subjects, cfg, cache_root, jobs=1, force=False, progress=None):
    """Re-derive every cached trial's record under the current config. Returns {status: [ids]}."""
    from .dataset import all_trials
    work = [(t, s, cfg, cache_root, force) for s, t in all_trials(subjects)]
    out = {"done": [], "current": [], "missing": [], "failed": []}
    if int(jobs or 1) <= 1:
        results = map(_reanalyse_worker, work)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=int(jobs))
        results = pool.map(_reanalyse_worker, work, chunksize=4)
    for tid, (status, payload) in results:
        out[status].append(tid if status in ("done", "current") else (tid, payload))
        if progress:
            progress(status, tid, payload if status in ("missing", "failed") else None)
    return out


def _failure_record(trial, subject, key, exc, t0):
    return {
        "trial_id": trial.trial_id, "subject": subject.subject_id, "task": trial.task,
        "rep": trial.rep, "path": trial.path,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "duration_s": round(time.time() - t0, 1),
        "cache_key": key, "ok": False,
    }


# --------------------------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------------------------- #
def _worker(args):
    """Process-pool entry point. Returns (record, ok) and never raises, so one bad recording
    cannot take the pool down with it."""
    trial, subject, cfg, urdf_path, cache_root = args
    t0 = time.time()
    cache = TrialCache(cache_root) if cache_root else None
    key = trial_key(trial.path, cfg.fingerprint(), subject.fingerprint())
    try:
        return run_trial(trial, subject, cfg, urdf_path, cache=cache), True
    except BaseException as e:                  # noqa: BLE001 -- deliberately total
        rec = json_safe(_failure_record(trial, subject, key, e, t0))
        if cache is not None:
            cache.save_failure(subject.subject_id, trial.trial_id, rec)
        return rec, False


def run_sweep(subjects, cfg, cache_root=None, jobs=None, progress=None,
              skip_cached_failures=True):
    """Analyse every trial of every subject. Returns (records, failures).

    Sequential when `jobs <= 1`, which is also what you want while debugging: a traceback from a
    process pool is a worse thing to read than one from the main process.
    """
    from .dataset import all_trials

    cache = TrialCache(cache_root) if cache_root else None
    jobs = int(jobs or cfg.jobs or 1)

    # --- prepass: one DHM per subject, built serially --------------------------------------- #
    urdfs = {}
    for s in subjects:
        if progress:
            progress("urdf", s.subject_id, None)
        urdfs[s.subject_id] = ensure_subject_urdf(s, cfg)

    work = []
    records, failures = [], []
    for subject, trial in all_trials(subjects):
        key = trial_key(trial.path, cfg.fingerprint(), subject.fingerprint())
        if cache is not None:
            hit = cache.load(subject.subject_id, trial.trial_id, key)
            if hit is not None:
                records.append(refresh_metadata(hit, trial, subject) | {"cached": True})
                if progress:
                    progress("cached", trial.trial_id, None)
                continue
            if skip_cached_failures:
                bad = cache.load_failure(subject.subject_id, trial.trial_id, key)
                if bad is not None:
                    failures.append(bad)
                    if progress:
                        progress("cached-failure", trial.trial_id, bad.get("error"))
                    continue
        work.append((trial, subject, cfg, urdfs[subject.subject_id], cache_root))

    if not work:
        return records, failures

    if jobs <= 1:
        for args in work:
            rec, ok = _worker(args)
            (records if ok else failures).append(rec)
            if progress:
                progress("done" if ok else "failed", rec["trial_id"],
                         rec.get("duration_s") if ok else rec.get("error"))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_worker, a): a[0].trial_id for a in work}
            for fut in as_completed(futures):
                rec, ok = fut.result()
                (records if ok else failures).append(rec)
                if progress:
                    progress("done" if ok else "failed", rec["trial_id"],
                             rec.get("duration_s") if ok else rec.get("error"))

    # Sorted whichever way the sweep ran: with a pool, records arrive in completion order, and a
    # report whose row order depends on which worker finished first is not reproducible.
    records.sort(key=lambda r: (r.get("subject", ""), r.get("task", ""), r.get("trial_id", "")))
    failures.sort(key=lambda r: (r.get("subject", ""), r.get("trial_id", "")))
    return records, failures
