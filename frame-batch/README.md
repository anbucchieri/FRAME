# frame-batch

The command-line entry point of [FRAME](../README.md). It discovers a dataset, analyses every
recording through `frame-kinematics` and `frame-core`, caches each trial, pools the results per
task type, and writes the report and figures.

```bash
frame-batch list | run | report | config      # normally invoked through ../frame
```

```python
from frame_batch import BatchConfig, discover, run_sweep
cfg = BatchConfig.load("batch.json")
subjects = discover(cfg.data_root, cfg.trial_pattern)
records, failures = run_sweep(subjects, cfg, cache_root=cfg.resolved_cache_root())
```

| module | contents |
|---|---|
| `config.py` | `BatchConfig`: every setting, its default, and the cache fingerprint |
| `dataset.py` | subject/trial discovery, filename grammar, anthropometry |
| `runner.py` | one trial end to end, and the parallel sweep |
| `cache.py` | the per-trial result cache |
| `cli.py` | `list`, `run`, `report`, `config` |
| `report.py` | `summary.md` and the CSV/JSON tables |
| `plots.py` | distribution figures |
| `timeseries.py`, `torque_speed.py`, `npose.py`, `trajectory3d.py`, `globe.py`, `exceedance.py` | figures (and report-time metrics) built from the cached captures |
| `exo_eval.py` | `exo_eval/requirements_per_rep.csv`: G and reversal metrics per recording × joint |

```bash
pytest -q
```
