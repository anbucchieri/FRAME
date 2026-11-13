"""
frame-batch -- the pooled entry point: a DATASET in, one distribution per task type out.

This package answers "what does this task demand of an exoskeleton, across every recording of
it". It walks a dataset tree of subjects, analyses every .mvnx through the frame-kinematics heavy
path and the frame-core metric engine, then hands the per-trial scalars to `frame_core.pooling`.

    from frame_batch import BatchConfig, discover, run_sweep
    cfg = BatchConfig.load("batch.json")
    subjects = discover(cfg.data_root, cfg.trial_pattern)
    records, failures = run_sweep(subjects, cfg, cache_root="results/cache")

Everything expensive is cached per trial, so a sweep is restartable and re-reporting is free.
"""
from .config import BatchConfig
from .dataset import discover, all_trials, Subject, Trial, DatasetError, read_anthropometry
from .runner import run_sweep, run_trial, ensure_subject_urdf
from .cache import TrialCache, trial_key

__all__ = [
    "BatchConfig",
    "discover", "all_trials", "Subject", "Trial", "DatasetError", "read_anthropometry",
    "run_sweep", "run_trial", "ensure_subject_urdf",
    "TrialCache", "trial_key",
]
__version__ = "0.1.0"
