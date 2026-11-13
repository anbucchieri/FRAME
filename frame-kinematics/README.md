# frame-kinematics

The Pinocchio heavy path of [FRAME](../README.md). It turns one Xsens movement dataframe into a
`frame_core.Capture`: IK (via the project's `ID_Xsens`), floating-base inverse dynamics with a
modelled ground-reaction force, the Jacobian involvement term, and the exoskeleton's reach.

```python
from frame_kinematics import run_job, KinematicsJob
cap = run_job(df, KinematicsJob(urdf_path=..., dof_names=[...], ee_segment="RightForeArm"))
```

**Pinocchio is isolated and injected.** The IK engine, the inverse-dynamics runner, the Jacobian
and the exo builder are all injectable (see `interfaces.IKEngine`). Importing the package needs
no Pinocchio, and the test suite runs the full wiring with fakes. Production runs inside the
FRAME Docker image.

| module | contents |
|---|---|
| `worker.py` | `run_job()`: model resolution, trimming, per-repetition IK/ID, assembly |
| `kinematics.py` | one repetition: IK → ID → Jacobian → end-effector cloud |
| `dhm.py` | build or reuse the per-subject scaled URDF |
| `mvnx.py` | MVNX → movement CSV, calibration pose |
| `canonical.py`, `assemble.py` | trunk-frame conventions, `Capture` assembly |
| `serialize.py` | `Capture` ⇄ npz (the result cache format) |
| `torque_map.py` | per-segment torque ÷ capacity for the N-pose map |
| `pinocchio_backend/` | inverse dynamics + GRF model, exoskeleton workspace |

**Inverse-dynamics states (ID_VERSION 2).** `id_grf.run_id_with_grf` does not use `ID_Xsens`'
`saved_velocity` / `saved_accelerations`: the accelerations are identically zero (`q_prev` /
`q_prev2` alias `q_revised`) and the velocities are the IK step over dt = 1/500 s on 240 Hz frames
(≈2.08× too fast). `id_grf.filtered_states` rebuilds q, q̇, q̈ from the IK configurations at the
true capture rate (6 Hz, order 4, `filtfilt`; `pin.difference` for the free-flyer). The same
filtered q feeds `rnea`, the GRF model and `computeGeneralizedGravity`, whose output is exported
per joint as `Repetition.gravity_torques`. `kinematics.ID_VERSION` rides in the batch cache key.

```bash
pip install -e ../frame-core && pip install -e . --no-deps && pip install pandas pytest
pytest -q
```

To run it outside Docker, install the `ik` extra (`pip install -e ".[ik]"`) and put the
repository root on `PYTHONPATH` so that `utils` can be imported.
