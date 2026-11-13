"""
Interfaces for frame-kinematics.

frame-kinematics does NOT reimplement IK. It calls an injected engine that satisfies `IKEngine`,
which is exactly the surface the project's `ID_Xsens` (utils/inverse_dynamics.py) exposes.
Production wires in the real class; tests wire in a fake -> the whole heavy path runs with zero
Pinocchio.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, Any, runtime_checkable


@runtime_checkable
class IKEngine(Protocol):
    """Minimal surface frame-kinematics needs from an IK engine (ID_Xsens-compatible)."""
    robot: Any                 # .robot.model, .robot.data  (Pinocchio)
    saved_configurations: Any  # (N, nq)  after runFullIK()
    saved_velocity: Any        # (N, nv)
    saved_accelerations: Any   # (N, nv)
    xsens_frequency: float

    def uploadMocapFile(self) -> None: ...
    def runFullIK(self) -> Any: ...
    def temp_save_log_q(self) -> dict: ...        # {joint_name: angle trajectory [rad]}
    def butterworth_filter(self, x) -> Any: ...


@dataclass
class KinematicsJob:
    """One heavy-path job: which model, which joints, which segments, and the subject."""
    dof_names: list = field(default_factory=list)   # URDF joint names -> the analysed DOFs
    ee_segment: str = ""                    # Xsens segment for the task-space cloud (dataframe cols)
    trunk_segment: str = "T8"               # frame the ee cloud is expressed in
    ee_frame: Optional[str] = None          # URDF frame for the Jacobian term (defaults to ee_segment)
    fs: float = 240.0
    min_len: int = 30
    # Discard the settling frames at each end of the recording, BEFORE IK, so instrumental spikes
    # never reach the angles, the torques, the metrics or the figures.
    #
    # Default 0 = no trim. A non-zero LIBRARY default is the wrong choice: the right number depends
    # on the recording length, and a default sized for a 30 s capture silently refuses (or guts) a
    # short one.
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0
    height_m: Optional[float] = None
    mass_kg: Optional[float] = None
    # --- DHM URDF: build-once-per-subject from a fixed xacro base + the subject's .mvnx ---
    urdf_path: Optional[str] = None         # prebuilt URDF; if set & exists, used directly
    subject_id: Optional[str] = None        # cache key / label
    # Cache the built URDF under this stem instead of the .mvnx stem. The pooled runner sets it to
    # `subject_<id>` so every recording of one person shares ONE model -- see dhm.subject_urdf_path.
    urdf_stem: Optional[str] = None
    models_dir: str = "models"              # where scaled URDFs are cached
    xacro_base: Optional[str] = None        # fixed dhm_new.xacro reused for everyone
    mvnx_path: Optional[str] = None         # needed ONLY on a cache miss (segment dimensions)
    # exoskeleton reach (assessment add-on)
    exo_urdf: Optional[str] = None
    exo_trunk_frame: str = "trunk_attachment"
    exo_arm_tips: dict = field(default_factory=lambda: {"right": "right_arm_tip", "left": "left_arm_tip"})
    exo_extra_R: Optional[Any] = None

    def __post_init__(self):
        if self.ee_frame is None:
            self.ee_frame = self.ee_segment
