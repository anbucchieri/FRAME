"""
frame-kinematics -- the Pinocchio/TSID heavy path for FRAME.

Runs IK (via the project's ID_Xsens, injected), floating-base inverse dynamics + modelled GRF
(id_grf/grf_model), the Jacobian involvement term, and exoskeleton reach, then hands a
`frame_core.Capture` to the pure engine. Importing this package is free; Pinocchio is reached
only when a real job runs.

    from frame_kinematics import run_job, KinematicsJob
    cap = run_job(df, KinematicsJob(urdf_path=..., dof_names=[...], ee_segment='RightForeArm'))
    from frame_core import analyze
    results = analyze(cap, config)
"""
from .interfaces import KinematicsJob, IKEngine
from .worker import run_job
from .serialize import capture_to_npz_bytes, capture_from_npz_bytes

__all__ = ["KinematicsJob", "IKEngine", "run_job",
           "capture_to_npz_bytes", "capture_from_npz_bytes"]
__version__ = "0.1.0"
