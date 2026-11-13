"""
frame-core -- pure NumPy/SciPy metric engine for the FRAME tool.

No Pinocchio, no heavy dependencies: this is the citable package. `frame-kinematics` runs
Pinocchio IK/ID on the raw MVNX captures and feeds the resulting arrays into this engine.

Two levels of distribution live here:
  * `analyze` summarises each metric ACROSS THE REPETITIONS of one recording;
  * `pooling.pool` summarises it ACROSS RECORDINGS, grouped by task type -- which is what the
    pooled (multi-trial, multi-subject) tool reports. `flatten` is the seam between them.

Quick start:
    from frame_core.synthetic import make_capture, default_config
    from frame_core.pipeline import analyze
    res = analyze(make_capture(), default_config())
"""
from .types import Capture, Repetition, Config, summarize
from .pipeline import analyze, capability
from .flatten import (flatten_result, normalize_row, n_reps,
                      units_table, filter_metrics, filter_record, dof_of,
                      METRICS_VERSION)
from .pooling import pool, pool_metric, compare_groups, ranked_differences

__all__ = [
    "Capture", "Repetition", "Config", "summarize",
    "analyze", "capability",
    # one analysed trial -> one flat row of scalars
    "flatten_result", "normalize_row", "n_reps", "units_table",
    "filter_metrics", "filter_record", "dof_of", "METRICS_VERSION",
    # many rows -> one distribution per task type
    "pool", "pool_metric", "compare_groups", "ranked_differences",
]
__version__ = "0.1.0"
