# frame-core

The pure **NumPy / SciPy** metric engine of [FRAME](../README.md): it derives exoskeleton design
requirements from joint angles, torques and end-effector trajectories. It never imports Pinocchio,
and its test suite checks this.

```python
from frame_core import Capture, Repetition, Config, analyze

res = analyze(capture, Config())    # runs exactly the layers the data supports
res["capability"]                    # which layers ran, and why others did not
res["task_space"], res["joint_space"], res["dynamics"], res["passivity"], res["pca"]
```

Distributions work at two levels:

- `analyze` summarises each metric **across the repetitions of one recording**;
- `pooling.pool` summarises it **across recordings**, grouped by task type.

`flatten` connects the two: it turns one `analyze` result into one flat row of scalars and adds
subject-normalised (`__norm`) twins of the dimensional columns.

| module | contents |
|---|---|
| `types.py` | `Capture`, `Repetition`, `Config`, `summarize()` |
| `task_space.py` | Layer 1: workspace volume, reach, end-effector speed, exo coverage |
| `joint_space.py` | Layer 2: ROM, velocity, reversals, weighted involvement |
| `dynamics.py` | Layer 3: peak/RMS torque, power, net work, quasi-stiffness |
| `passivity.py` | exo requirements: gravity share G = RMS(τ_g)/RMS(τ), velocity reversals, reversal latency, required / peak acceleration |
| `pca.py` | coordination structure |
| `pipeline.py` | `analyze()`: gated orchestration and the capability report |
| `flatten.py`, `pooling.py` | one row per trial → one distribution per group |
| `curves.py`, `signal.py`, `frames.py`, `anatomy.py` | shared helpers |
| `synthetic.py` | a synthetic capture for tests and demos |

```bash
pip install -e ".[test]" && pytest -q
```

