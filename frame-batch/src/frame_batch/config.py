"""
Batch configuration -- one file that fixes every choice a pooled run makes.

A pooled run is expensive (tens of recordings, minutes each) and its output is a distribution
that only means something if every trial in it was analysed the same way. So the settings live in
one serialisable object, that object is hashed into the result cache key, and a copy of it is
written next to the results. A run whose config changed is a different run, and the cache knows it.

`batch.json` at the repo root is the default; `--config` points elsewhere. Every field below has a
default, so batch.json only needs the ones you want to change.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

# Filename grammar for this study: <subject>_<task>_<orientation>_<x>_<rep>.mvnx, e.g.
# 1817_H_N_E_1.mvnx.
#
#   task        H / M / L -- shelf height. The pooling key.
#   orientation N / P / O -- how the task is oriented at that height. Carried as metadata, so it
#               can be sliced (or pooled on, via `pool_by`) later without re-running anything.
#   <x>         E / T     -- MATCHED BUT NOT CAPTURED. The study treats these as the same
#               condition, so recording it would invent a factor that does not exist; the trials
#               simply pool together. Give it a name here if that ever stops being true.
#   rep         1..6
#
# A named group becomes a column automatically, so renaming a factor is a config change. It does
# not invalidate the result cache -- see BatchConfig.fingerprint.
DEFAULT_TRIAL_PATTERN = (r"^(?P<subject>[^_]+)_(?P<task>[A-Za-z])_(?P<orientation>[A-Za-z])"
                         r"_(?:[A-Za-z])_(?P<rep>\d+)$")

# The full 43-DOF chain of the DHM. Kept here rather than in a separate job.json so that the DOF
# set and the metrics computed from it can never disagree about which run they belong to.
DEFAULT_DOF_NAMES = [
    "lumbarspine_X", "lumbarspine_Y", "lumbarspine_Z",
    "thoracicspine_X", "thoracicspine_Y", "thoracicspine_Z",
    "neck_X", "neck_Y", "neck_Z", "head_X", "head_Y",
    "right_clavicular_X", "right_clavicular_Z",
    "right_shoulder_X", "right_shoulder_Y", "right_shoulder_Z",
    "right_elbow_Z", "right_elbow_Y", "right_wrist_X", "right_wrist_Z",
    "left_clavicular_X", "left_clavicular_Z",
    "left_shoulder_X", "left_shoulder_Y", "left_shoulder_Z",
    "left_elbow_Z", "left_elbow_Y", "left_wrist_X", "left_wrist_Z",
    "right_hip_X", "right_hip_Y", "right_hip_Z", "right_knee_Y", "right_knee_Z",
    "right_ankle_X", "right_ankle_Y",
    "left_hip_X", "left_hip_Y", "left_hip_Z", "left_knee_Y", "left_knee_Z",
    "left_ankle_X", "left_ankle_Y",
]

# Every figure the tool can produce. `BatchConfig.figures` selects from this list.
FIGURE_NAMES = [
    "headline_distributions",       # box+strip of the headline metrics
    "rom_by_task",                  # range of motion per DOF
    "joint_speed_by_task",          # peak joint speed per DOF
    "peak_torque_by_task",
    "rms_torque_by_task",
    "net_work_by_task",
    "top_differences",
    "joint_timeseries_by_task",     # mean +/- SD angle / speed / torque curves
    "torque_speed_by_task",         # (speed, torque) density per joint, one file per task
    "ee_trajectory_3d_by_task",     # mean 3D hand path
    "npose_torque_map_by_task",     # N-pose skeleton, joints coloured by RMS torque / capacity
    "torque_exceedance_by_task",    # count / mean duration of |tau| above capacity_fraction x effort
    "shoulder_globe_by_task",       # shoulder directions on a (plane, elevation) globe, per npose_by
    "joint_pose_timeseries",        # per task x joint: mean pose at 0 / 50 / 100 % over its curves
]

# The set actually asked for. Everything else stays available by naming it in `figures`.
DEFAULT_FIGURES = [
    "ee_trajectory_3d_by_task",
    "joint_speed_by_task",
    "joint_timeseries_by_task",
    "peak_torque_by_task",
    "rms_torque_by_task",
    "torque_speed_by_task",
    "npose_torque_map_by_task",
]

# Headline metrics that get their own distribution figure. Everything else is still in the table
# and in pooled.json; this is the reading order, not the scope.
DEFAULT_HEADLINE_METRICS = [
    "task_space.workspace_volume_L",
    "task_space.reach_max_m",
    "task_space.peak_speed_mps",
    "task_space.dwell_concentration",
    "joint_space.right_shoulder_Z.rom_deg",
    "joint_space.right_elbow_Z.rom_deg",
    "joint_space.lumbarspine_Y.rom_deg",
    "dynamics.right_shoulder_Z.peak_torque_Nm",
    "dynamics.right_shoulder_Z.rms_torque_Nm",
    "dynamics.right_elbow_Z.peak_torque_Nm",
    "dynamics.lumbarspine_Y.peak_torque_Nm",
    "pca.n_pc_90",
]


# Metric columns left out of the written results by default (see BatchConfig.exclude_metrics).
DEFAULT_EXCLUDE_METRICS = [
    r"\.exceed\d+_time_frac$", r"\.exceed\d+_total_s$",     # exceedance: keep count + mean duration
    r"\.k_Nm_per_rad(__norm)?$", r"\.r2$",                    # quasi-stiffness
    r"\.net_work_J",                                          # net work (+ __norm)
    r"\.peak_power_",                                         # peak positive / negative power
    r"\.peak_torque_Nm__norm$", r"\.rms_torque_Nm__norm$",    # mass-normalised torques
    # joint_space reversal detector (5 % of peak deadband): superseded by passivity.<dof>.n_reversals
    # (fixed 5 deg/s deadband, >= 0.1 s apart), whose latency_* columns give the reversal duration
    r"^joint_space\.[^.]+\.(zero_crossings|reversals|min_crossing_gap_s|mean_crossing_gap_s)$",
    r"^pca\.",
]


# Fields that no longer exist but were part of the cache fingerprint when they did. Their last value
# is hashed in as a constant so removing them did not invalidate every finished trial.
_RETIRED_FIELDS = {"align": "none", "thresholds": {}}


@dataclass
class BatchConfig:
    """Everything a pooled run needs to be reproducible."""
    # --- dataset ---
    # Where the recordings live. An ABSOLUTE path outside this project is the normal case --
    # captures are large and should not be copied in beside the code. The `./frame` wrapper reads
    # this value and bind-mounts that exact path into the container at the same location, so one
    # line here is the only place the path is written.
    data_root: str = "data"
    output_root: str = "results"
    # Where the per-trial IK/ID cache goes. Defaults to <output_root>/cache; point it at a roomier
    # disk when the dataset is large. It is the thing that makes a re-plot free.
    cache_root: Optional[str] = None
    models_dir: str = "models"
    xacro_base: str = "models/dhm_new.xacro"
    trial_pattern: str = DEFAULT_TRIAL_PATTERN
    strict_filenames: bool = False      # unparseable file -> skip rather than abort the sweep

    # --- what to analyse ---
    dof_names: list = field(default_factory=lambda: list(DEFAULT_DOF_NAMES))
    ee_segment: str = "RightForeArm"    # Xsens segment (dataframe column prefix)
    ee_frame: str = "right_forearm"     # URDF link, for the Jacobian involvement term
    trunk_segment: str = "T8"
    fs: Optional[float] = None          # None -> take the capture rate the .mvnx reports
    min_len: int = 30
    trim_start_s: float = 0.0           # no trim: analyse the full recording
    trim_end_s: float = 0.0
    want_torque: bool = True
    want_jac: bool = True
    # --- shoulder coordinates ---
    # "elevation": the shoulder complex is reported as ISB humerothoracic plane of elevation,
    #   elevation and axial rotation (frame_core.shoulder), replacing {side}_shoulder_X/Y/Z in every
    #   metric, table and figure. The cached Captures keep the raw DHM angles; the transform is
    #   applied when one is loaded, and `./frame reanalyse` rebuilds the records without IK.
    # "dhm": the raw Euler angles of the DHM (zero = T-pose; axes change meaning with the pose).
    shoulder_coords: str = "elevation"
    shoulder_mask_deg: float = 20.0     # plane / rotation undefined below this elevation (gimbal)
    # Joint capacities that replace the model's (or the derived shoulder) values, per DOF:
    # {"right_shoulder_elevation": {"effort": 77.0, "source": "..."}}. Keys: effort [N.m],
    # velocity [rad/s], lower / upper [rad]; "source" is carried into the limits as provenance.
    capacity_overrides: dict = field(default_factory=dict)
    # --- exoskeleton assessment ---
    # TWO keys, because there are two different costs.
    #
    #   coverage_exo_urdf : report time only -- the exo's reachable arm directions drawn on the
    #       shoulder globe (panel c). It needs nothing from the heavy path, so setting it is free.
    #       This is the one to use.
    #
    #   exo_urdf : additionally folds the coverage numbers into every trial's metric row inside
    #       analyze(), so they appear in trials.csv and pooled.json beside the other metrics. That
    #       changes the cached record, so it DOES force a full re-analysis.
    coverage_exo_urdf: Optional[str] = None
    exo_urdf: Optional[str] = None
    exo_trunk_frame: str = "trunk_attachment"   # the exo link that sits at the human's T8

    @property
    def any_exo_urdf(self):
        """The exo model to assess against: the figure-only one if set, else the analysed one."""
        return self.coverage_exo_urdf or self.exo_urdf

    # --- metric config (frame_core.Config) ---
    voxel_m: float = 0.02
    speed_deadband_frac: float = 0.05
    speed_deadband_floor_dps: float = 2.0
    involvement_weights: dict = field(default_factory=lambda: {
        "rom": 0.25, "path": 0.25, "jacobian": 0.25, "dynamic": 0.25})
    dof_groups: dict = field(default_factory=dict)

    # --- which joints get reported -----------------------------------------------------------
    # THE list. Name the joints you want and every per-joint figure and metric column follows it:
    # angles, speeds, torques, the torque-speed panels, the trajectory bands and the ranked
    # differences. Leave it empty to fall back to the per-figure regexes below.
    #
    # It does NOT change what is analysed -- `dof_names` still drives IK and inverse dynamics,
    # because involvement and PCA are computed over the whole chain and would change meaning if
    # joints were dropped from the model. So this is a reporting filter, it costs no re-analysis,
    # and you can change it and re-plot in seconds.
    joints_of_interest: list = field(default_factory=list)

    # --- which metric columns get WRITTEN ----------------------------------------------------
    # Regexes matched against each metric column name (e.g. `dynamics.right_elbow_Z.r2`). A match
    # is left out of trials.csv / trials.json / pooled.json / pooled_summary.csv /
    # differences.csv / summary.md. It is a reporting filter only: the metrics are still computed
    # and cached, and the figures still see them -- so changing this costs one `./frame report`.
    exclude_metrics: list = field(default_factory=lambda: list(DEFAULT_EXCLUDE_METRICS))

    # --- pooling + reporting ---
    pool_by: list = field(default_factory=lambda: ["task"])
    normalize: bool = True              # emit mass/height-normalised twins alongside raw SI
    headline_metrics: list = field(default_factory=lambda: list(DEFAULT_HEADLINE_METRICS))
    plot_dofs: str = r"shoulder|elbow|lumbarspine"   # regex: which DOFs get per-DOF figures
    # --- which figures to produce, and how -----------------------------------------------------
    # Name the ones you want; empty means all of them. Every name in FIGURE_NAMES is valid, and an
    # unrecognised name is reported rather than silently producing nothing.
    figures: list = field(default_factory=lambda: list(DEFAULT_FIGURES))
    figure_format: str = "pdf"          # "pdf" (vector, for papers) or "png"
    # Box plots draw every trial as a dot. Past a few dozen that is a smear rather than a
    # distribution, so the dots are thinned to this many per group -- deterministically, and only
    # the DRAWING is thinned: the box, the median and the IQR always use every value.
    box_max_points: int = 40
    top_differences: int = 25
    # --- mean +/- SD joint trajectories (joint_timeseries_by_task.png) ---
    # Which DOFs get a column. The default is the right arm chain: the task is right-handed
    # (ee_segment = RightForeArm), so the left side is a postural response, not the movement.
    timeseries_dofs: str = r"^right_(shoulder|elbow)_"
    # Each curve is one WHOLE recording (never segmented), linearly time-normalised to this many
    # samples and averaged across every subject and repetition of the group.
    timeseries_n_resample: int = 500
    # Torque scales with body mass. Off by default because within ONE subject it only relabels
    # the axis; turn it on once the bands pool several subjects together.
    timeseries_normalise_torque: bool = False
    # --- torque-speed operating plane (torque_speed_by_task_<task>) ---
    torque_speed_dofs: str = r"clavicular|shoulder|elbow|wrist"   # the upper limbs, both sides
    torque_speed_normalise: bool = False  # plot N.m/kg instead of N.m
    # View clipping. Edge ringing and IK spikes put a few samples ten times beyond the working
    # range; without this they set every axis and hide the data.
    torque_speed_clip_pct: float = 99.5
    # --- averaged 3D path of a tracked point (ee_trajectory_3d_by_task.png) ---
    # The Xsens segment to DRAW. Independent of `ee_segment`, which is what gets MEASURED: the
    # metrics stay on the forearm (the exoskeleton's attachment point) while the figure shows the
    # hand that holds the tool. Read straight from the .mvnx, so changing it costs no re-analysis.
    trajectory_segment: str = "RightHand"
    trajectory_show_trials: bool = True   # draw the individual recordings behind the mean path
    # --- N-pose torque map (npose_torque_map_by_task) ---
    # The metadata fields that split its panels. Default: one panel per task x orientation
    # (H {N, P}, M {N, O, P}, L {N, P}), laid out one row per task.
    npose_by: list = field(default_factory=lambda: ["task", "orientation"])
    # --- shoulder globe (shoulder_globe_by_task) ---
    # The right arm counts as working in a recording unless its shoulder-elevation range is below
    # this fraction of the left arm's (two-handed recordings keep the right arm); recordings done
    # with the left arm alone are excluded from the right-arm globe and its metrics.
    globe_min_rom_ratio: float = 0.5
    # --- fraction of the joint torque capacity used as a reference level ---
    # Drawn on the peak / RMS torque figures, and the threshold of the exceedance metric
    # (`torque_exceedance_by_task`: how often, and for how long, |tau| stays above it).
    capacity_fraction: float = 0.15
    # Excursions shorter than this are not counted (0 = every excursion counts).
    exceedance_min_duration_s: float = 0.0

    # --- execution ---
    jobs: int = 4                       # parallel trials; each one parses a ~100 MB XML
    cache_captures: bool = True         # keep the per-trial .npz so figures can be redone free
    # --- how much disk the IK/ID cache is allowed to cost ---
    # The cached Capture exists ONLY so figures can be rebuilt without re-running IK and inverse
    # dynamics. Every scalar metric is computed once at full precision during the sweep and stored
    # in the trial's JSON record, which is tiny and never reduced. So the capture can be stored
    # more cheaply than it was computed:
    #
    #   cache_float32  : halves it. float32 carries ~7 significant digits, far more than the
    #                    angles and torques resolve, so figures are unaffected.
    #   cache_decimate : keeps every Nth sample and stores fs/N with it, dividing the size again.
    #                    4 turns 240 Hz into 60 Hz: still thousands of samples per recording,
    #                    plenty for a curve normalised to 500. It DOES slightly change anything re-derived
    #                    from the sample cloud (voxel coverage), so it is
    #                    off by default -- turn it on when the disk matters more than that.
    cache_float32: bool = True
    cache_decimate: int = 1

    # ---------------------------------------------------------------------------------------- #
    @classmethod
    def load(cls, path=None, **overrides):
        """Read a config file (if given), then apply explicit overrides. Unknown keys are named."""
        data = {}
        if path:
            with open(path) as f:
                data = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{path}: unknown config key(s) {sorted(unknown)}. Known keys: {sorted(known)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def fingerprint(self, metrics_version=None) -> str:
        """Hash of everything that can change a NUMBER, so the cache invalidates when it should.

        Presentation-only and execution-only fields are excluded on purpose: changing `jobs`,
        `headline_metrics` or `plot_dofs` must not throw away hours of finished IK.

        `trial_pattern` is excluded too, which is less obvious. It decides how a filename is split
        into a task and its factors -- pure metadata, never read by IK, the metric engine or the
        DHM. Renaming a factor group would otherwise cost a full re-analysis to change a column
        heading. The metadata itself is re-derived from the current pattern whenever a cached
        record is loaded (see runner.refresh_metadata), so the labels still track the config.

        The subject's height and mass are NOT here -- they are not config. They enter the cache key
        separately through Subject.fingerprint, because they do change the numbers.
        """
        import hashlib
        from frame_core.flatten import METRICS_VERSION
        volatile = {"output_root", "jobs", "headline_metrics", "plot_dofs",
                    "top_differences", "cache_captures", "normalize",
                    "pool_by", "strict_filenames", "trial_pattern",
                    "timeseries_dofs", "timeseries_n_resample", "timeseries_normalise_torque",
                    "torque_speed_dofs",
                    "torque_speed_normalise",
                    "torque_speed_clip_pct", "trajectory_segment",
                    "trajectory_show_trials", "coverage_exo_urdf", "npose_by", "capacity_fraction", "globe_min_rom_ratio",
                    "exceedance_min_duration_s",
                    # Reporting-only, and the two that decide where things live on disk. A path
                    # is not a number: moving the dataset or the cache must not invalidate hours
                    # of finished IK.
                    "joints_of_interest", "exclude_metrics", "data_root", "cache_root",
                    "cache_float32", "cache_decimate",
                    "figures", "figure_format", "box_max_points"}
        payload = {k: v for k, v in sorted(self.to_dict().items()) if k not in volatile}
        payload.update(_RETIRED_FIELDS)
        # A setting that CANNOT affect the result must not key the cache. The exoskeleton
        # parameters only reach a number when there is an exoskeleton, so with `exo_urdf` unset
        # they are dropped -- otherwise merely adding an exo option to the config would invalidate
        # every finished trial for a field that changed nothing.
        if not self.exo_urdf:
            payload.pop("exo_trunk_frame", None)
        # The raw DHM coordinates are what every trial analysed before the shoulder transform
        # existed was computed in, so "dhm" hashes exactly as before (and `reanalyse` can find
        # those records); the mask only matters for the elevation coordinates.
        if self.shoulder_coords == "dhm":
            payload.pop("shoulder_coords", None)
            payload.pop("shoulder_mask_deg", None)
        if not self.capacity_overrides:           # absent before overrides existed
            payload.pop("capacity_overrides", None)
        # The metric set is part of what a cached record IS, and it lives in the code rather than
        # in this file -- so it has to be hashed in explicitly or a new column would come back
        # silently absent on every previously analysed trial.
        payload["_metrics_version"] = METRICS_VERSION if metrics_version is None else metrics_version
        # Same for the torques themselves: a fix inside inverse dynamics changes every number
        # without touching the config. Imported from the Pinocchio-free module on purpose.
        from frame_kinematics.kinematics import ID_VERSION
        payload["_id_version"] = ID_VERSION
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def analysed_dofs(self):
        """The DOF names metrics are reported under: `dof_names` with the shoulder renamed to the
        elevation coordinates when `shoulder_coords` is "elevation"."""
        from frame_core.shoulder import rename
        return rename(list(self.dof_names), self.shoulder_coords)

    def transform_capture(self, cap):
        """Apply the configured shoulder coordinates to a (raw, cached) Capture."""
        if self.shoulder_coords == "elevation":
            from frame_core.shoulder import to_elevation
            cap = to_elevation(cap, mask_deg=self.shoulder_mask_deg)[0]
        if self.capacity_overrides:
            lim = {k: (dict(v) if isinstance(v, dict) else v)
                   for k, v in (cap.urdf_limits or {}).items()}
            for dof, over in self.capacity_overrides.items():
                if dof not in cap.dof_names:
                    continue
                e = lim.setdefault(dof, {})
                for key in ("effort", "velocity", "lower", "upper"):
                    if over.get(key) is not None:
                        e[key] = float(over[key])
                        e[f"{key}_source"] = over.get("source", "capacity_overrides (batch.json)")
            cap.urdf_limits = lim
        return cap

    def keep_metric(self, key):
        """False for a metric column `exclude_metrics` leaves out of the written results."""
        import re as _re
        return not any(_re.search(p, key) for p in self.exclude_metrics)

    def selected_dofs(self):
        """The joints to REPORT on: `joints_of_interest` if set, else every analysed DOF.

        Names that are not in `dof_names` are dropped rather than silently producing empty panels,
        and the order follows `dof_names` so figures do not reshuffle when the list is reordered.
        """
        names = self.analysed_dofs()
        if not self.joints_of_interest:
            return names
        wanted = set(self.joints_of_interest)
        return [d for d in names if d in wanted]

    def unknown_joints(self):
        """Names in `joints_of_interest` that no analysed DOF matches -- a typo, said out loud."""
        return [j for j in self.joints_of_interest if j not in set(self.analysed_dofs())]

    def dof_regex(self, fallback=None):
        """`joints_of_interest` as an exact-match regex, so the figure code keeps one interface.

        Returns `fallback` when no joints are named, which is how the per-figure defaults survive.
        """
        dofs = self.selected_dofs() if self.joints_of_interest else None
        if not dofs:
            return fallback
        import re as _re
        return "^(?:" + "|".join(_re.escape(d) for d in dofs) + ")$"

    def wants(self, name):
        """Should this figure be produced? An empty `figures` list means every one of them."""
        return (not self.figures) or (name in set(self.figures))

    def unknown_figures(self):
        """Names in `figures` that no builder answers to -- a typo, said out loud."""
        return [f for f in self.figures if f not in set(FIGURE_NAMES)]

    def figure_path(self, out_dir, name):
        """<out_dir>/<name>.<figure_format>, so one setting changes every file's format."""
        import os
        ext = (self.figure_format or "pdf").lower().lstrip(".")
        return os.path.join(str(out_dir), f"{name}.{ext}")

    def resolved_cache_root(self):
        import os
        return self.cache_root or os.path.join(self.output_root, "cache")

    def core_config(self):
        """The `frame_core.Config` these settings imply."""
        from frame_core.types import Config
        return Config(
            ee_segment=self.ee_segment, voxel_m=self.voxel_m,
            speed_deadband_frac=self.speed_deadband_frac,
            speed_deadband_floor_dps=self.speed_deadband_floor_dps,
            involvement_weights=dict(self.involvement_weights),
            dof_groups=dict(self.dof_groups),
            fs=self.fs or 240.0,
        )
