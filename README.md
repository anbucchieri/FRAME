**FRAME — Framework for Requirements and Assessment of Movement and Exoskeletons**  
   
 FRAME turns a dataset of **Xsens motion captures** (`.mvnx`) into **exoskeleton design**  
   
    
   
  **requirements**. For every recording it runs inverse kinematics and inverse dynamics on a  
   
    
   
  person-scaled digital human model, computes \~1000 task-space, joint-space and dynamics metrics,  
   
    
   
  then pools them into **one distribution per task type** (median, IQR, min, max, n), with tables  
   
    
   
  and publication-ready figures.  
```  
<data_root>/                                   results/    
    S1/                                          summary.md          read this first    
      anthropo_data_S1.csv   height, mass  ──▶   trials.csv          one row per recording    
      S1_H_N_E_1.mvnx  ┐                         pooled.json         distribution per group    
      ...                │  every recording        differences.csv     what separates the groups    
      S1_L_P_T_6.mvnx  ┘                         figures/*.pdf    
    S2/ ...    
      
   
```  
[**FRAME_schematic.pdf** shows the whole pipeline on one page.](FRAME_schematic.pdf "FRAME_schematic.pdf")  
**Paper:** the method implemented in this repository is described in the paper XXX. If you use  
   
 FRAME, please cite it (see [Citation).](#anchor-1 "#anchor-1")  
  
---  
**Contents**  
- [How it works](#anchor-2 "#anchor-2")  
- [Installation](#anchor-3 "#anchor-3")  
- [Quick start: the example data](#anchor-4 "#anchor-4")  
- [Dataset layout](#anchor-5 "#anchor-5")  
- [Configuration (`batch.json`)](#anchor-6 "#anchor-6")  
- [Commands](#anchor-7 "#anchor-7")  
- [Outputs](#anchor-8 "#anchor-8")  
- [Caching: what costs hours and what costs seconds](#anchor-9 "#anchor-9")  
- [Metrics](#anchor-10 "#anchor-10")  
- [Known limitations](#anchor-11 "#anchor-11")  
- [Troubleshooting](#anchor-12 "#anchor-12")  
- [Development](#anchor-13 "#anchor-13")  
- [Citation](#anchor-1 "#anchor-1")  
- [License](#anchor-14 "#anchor-14")  
  
---  
**How it works**  
```  
 per SUBJECT   anthropometry CSV + one .mvnx ──▶ ONE person-scaled DHM (URDF)    
                                                            │    
   per TRIAL     .mvnx ──▶ movement CSV ──▶ IK + inverse dynamics ──▶ Capture ──▶ metrics    
                                             (Pinocchio)                         (one row)    
                                                                           ── cache ──    
   per GROUP     all rows of task H / M / L ──▶ median · IQR · min · max · n  + figures    
      
   
```  
The expensive part (IK and inverse dynamics, \~80 s per recording) runs once per recording and is  
   
    
   
  cached. Pooling, tables and figures are rebuilt from that cache in seconds, so changing what you  
   
    
   
  report never means re-running the analysis.  
   
 The code is split into three packages:  
  
|  |  |  |  
| - | - | - |  
|   |    |    |  
| **Package** |  **Role** |  **Heavy dependencies** |  
| [`frame-core`](frame-core "frame-core") |  the metric engine and the pooling layer |  none (NumPy / SciPy) |  
| [`frame-kinematics`](frame-kinematics "frame-kinematics") |  IK + inverse dynamics + GRF + exo reach → `Capture` |  Pinocchio |  
| [`frame-batch`](frame-batch "frame-batch") |  dataset discovery, the sweep, the cache, the report, the figures |  none (pandas, matplotlib) |  
  
   
   
   
 plus [`utils/`, which holds the Xsens tooling that the heavy path calls: `ID_Xsens` (IK)  
   
    
   
  and the MVNX → CSV / URDF converters.](utils "utils")  
  
---  
**Installation**  
   
 FRAME runs inside **Docker**. Pinocchio and its IK stack are installed in the image, so you do  
   
    
   
  not need them on your machine. It runs on Linux, on macOS, and on Windows through WSL 2.  
   
 You need:  
- **Docker**. On Linux, add yourself to the `docker` group  
   
    
   
  (`sudo usermod -aG docker $USER`, then log out and back in) so that it runs without `sudo`.  
- **Git** and **Git LFS**. The example recordings are stored with Git LFS.  
- **Python 3**. The `./frame` launcher uses it to read `batch.json`.  
- About 2 GB of RAM for each parallel worker.  
```  
git lfs install                         # once per machine    
  git clone <repository-url> FRAME    
  cd FRAME    
  ./frame build                           # builds the Docker image frame:local (a few minutes, once)    
      
   
```  
If you cloned without Git LFS, the `.mvnx` files in `data/` are small pointer files. Fetch the  
   
    
   
  recordings with `git lfs pull`.  
   
 Rebuild with `./frame build` whenever you change code under `frame-*/src/`. You do not need to  
   
    
   
  rebuild for changes to `utils/`, `models/` or `batch.json`: these are mounted into the container  
   
    
   
  at run time.  
  
---  
**Quick start: the example data**  
   
 The repository includes two recordings (\~60 s each, 240 Hz) of one pseudonymised subject in  
   
    
   
  `data/S1/`, and `batch.json` already points at them.  
```  
./frame list              # 1. check the dataset is read correctly (takes seconds)    
  ./frame run               # 2. analyse both recordings, then write the report (~2 min)    
  ./frame report            # 3. optional: rebuild tables and figures from the cache (seconds)    
      
   
```  
`./frame list` should print:  
```  
S1 1.62 m  58.0 kg  2 trials (H:2)    
     H:   2 trials   N    
  2 trials, 0.2 GB total.    
      
   
```  
When `run` finishes, the results are in `results/`:  
- `results/summary.md` is the report; read it first.  
- `results/figures/` holds the figures as PDFs.  
- `results/trials.csv` has one row per recording, with every metric.  
 **Using your own data:** set `data_root` in `batch.json` to your dataset folder (laid out as  
   
    
   
  below), then run `./frame list` and `./frame run --jobs 6`.  
  
---  
**Dataset layout**  
```  
<data_root>/<subject_id>/    
      anthropo_data_<subject_id>.csv         must contain 'Body Height' (cm) and 'Weight' (kg)    
      <subject_id>_<task>_<orientation>_<x>_<rep>.mvnx    
      
   
```  
The filename grammar is the `trial_pattern` regex in `batch.json`. Every **named group** in it  
   
    
   
  becomes a column in `trials.csv` and can be used in `pool_by`. With the default pattern:  
  
|  |  |  |  
| - | - | - |  
|   |    |    |  
| **field** |  **example** |  **role** |  
| `subject` |  `S1` |  subject id (must match the folder name) |  
| `task` |  `H` / `M` / `L` |  task type: the default pooling key |  
| `orientation` |  `N` / `P` / `O` |  a factor: a column in `trials.csv`, poolable via `pool_by` |  
| `<x>` |  `E` / `T` |  matched but not captured (an unnamed group), so these trials pool together |  
| `rep` |  `1`–`6` |  repetition number |  
  
   
   
   
 The anthropometry CSV is a simple `label,value` file. FRAME reads its `Body Height` (cm) and  
   
    
   
  `Weight` (kg) rows. See `data/S1/anthropo_data_S1.csv` for an example.  
   
 Files that don't match the pattern are skipped, and every skip is listed on screen. Each subject  
   
    
   
  gets **one** scaled model, built from their anthropometry and their first recording, and all of  
   
    
   
  that subject's trials share it.  
   
 **Adding a subject:** create `<data_root>/<id>/`, add the recordings and  
   
    
   
  `anthropo_data_<id>.csv`, run `./frame list` to check the names, then `./frame run`. Subjects  
   
    
   
  that were already analysed are loaded from the cache.  
  
---  
**Configuration (**`batch.json`**)**  
   
 `batch.json` contains only the settings you are likely to change. Every other setting has a  
   
    
   
  sensible default (see [Advanced settings).](#anchor-15 "#anchor-15")  
```  
{    
    "data_root": "data",    
    "cache_root": null,    
    "output_root": "results",    
      
    "trial_pattern": "^(?P<subject>[^_]+)_(?P<task>[A-Za-z])_(?P<orientation>[A-Za-z])_(?:[A-Za-z])_(?P<rep>\\d+)$",    
    "pool_by": ["task"],    
      
    "joints_of_interest": [    
      "right_shoulder_plane", "right_shoulder_elevation", "right_shoulder_rotation",    
      "right_elbow_Y", "right_elbow_Z", "lumbarspine_Y"    
    ],    
      
    "figures": ["npose_torque_map_by_task", "torque_speed_by_task", "joint_timeseries_by_task",    
                "peak_torque_by_task", "rms_torque_by_task", "joint_speed_by_task",    
                "torque_exceedance_by_task", "shoulder_globe_by_task", "joint_pose_timeseries"],    
    "figure_format": "pdf",    
    "coverage_exo_urdf": "models/exoskeleton/exo1_full.urdf",    
      
    "capacity_overrides": {    
      "right_shoulder_elevation": {"effort": 53.9, "velocity": 3.002, "source": "Zimmermann et al. 2023 ..."},    
      "...": {}    
    },    
      
    "jobs": 2    
  }    
      
   
```  
  
|  |  |  
| - | - |  
|   |    |  
| **key** |  **what it does** |  
| `data_root` |  the dataset folder, mounted **read-only** (default: the bundled `data/`) |  
| `cache_root` |  where the per-trial IK/ID cache goes (\~5 MB per recording). `null` = `<output_root>/cache` |  
| `output_root` |  where tables and figures are written |  
| `trial_pattern` |  filename grammar. Named groups become columns (see [Dataset layout)](#anchor-5 "#anchor-5") |  
| `pool_by` |  how trials are grouped: `["task"]` gives H / M / L; `[]` gives one group over everything; `["task", "orientation"]` gives H_N, H_P, …; `["subject"]` gives one group per person |  
| `joints_of_interest` |  the joints every per-joint table and figure reports on. `[]` = all 43 |  
| `figures` |  which figures to draw (see [Figures). `[]` = all of them](#anchor-16 "#anchor-16") |  
| `figure_format` |  `"pdf"` or `"png"` |  
| `coverage_exo_urdf` |  exoskeleton model for the coverage figure. Keep it under `models/` so the container can see it. `null` turns coverage off |  
| `capacity_overrides` |  ⟳ joint capacities that replace the model's, per joint: `effort` \[N·m\], `velocity` \[rad/s\], `lower` / `upper` \[rad\], plus a free-text `source`. The bundled file sets the shoulder plane and elevation to the mean human values of Zimmermann et al. (2023), Table III |  
| `jobs` |  recordings analysed in parallel. Choose by memory (\~2 GB each), not by core count |  
  
   
   
   
 **Apart from `capacity_overrides` (⟳), none of these keys triggers a re-analysis.** Changing any of them costs one `./frame report`.  
   
 **Advanced settings**  
   
 Any field of `frame_batch.config.BatchConfig` can be added to `batch.json`, and an unknown key is  
   
    
   
  reported as an error rather than ignored. `./frame config` prints the effective configuration.  
   
    
   
  Fields marked ⟳ change the numbers, so changing them **invalidates the cache** and forces a  
   
    
   
  re-analysis.  
  
|  |  |  |  
| - | - | - |  
|   |    |    |  
| **key** |  **default** |    |  
| `dof_names` |  the 43-DOF DHM chain |  ⟳ joints analysed by IK / ID |  
| `ee_segment` / `ee_frame` |  `RightForeArm` / `right_forearm` |  ⟳ end-effector for task-space metrics / Jacobian |  
| `trunk_segment` |  `T8` |  ⟳ frame the end-effector cloud is expressed in |  
| `trim_start_s` / `trim_end_s` |  `0` / `0` |  ⟳ seconds dropped at each end before IK |  
| `fs` |  `null` (read from the .mvnx) |  ⟳ sampling rate override |  
| `voxel_m` |  `0.02` |  ⟳ workspace voxel size |  
| `involvement_weights` |  0.25 each |  ⟳ blend of the joint-involvement score |  
| `exo_urdf` |  `null` |  ⟳ fold exo coverage into every trial's metric row (prefer `coverage_exo_urdf`) |  
| `models_dir` / `xacro_base` |  `models` / `models/dhm_new.xacro` |  ⟳ where per-subject DHMs live / the DHM template |  
| `headline_metrics` |  12 metrics |  which metrics get their own panel in `headline_distributions` |  
| `normalize` |  `true` |  emit subject-normalised `…__norm` columns |  
| `top_differences` |  `25` |  rows in `differences.csv` |  
| `shoulder_coords` |  `"elevation"` |  ⟳ `"elevation"`: shoulder reported as ISB humerothoracic plane of elevation / elevation / axial rotation (`*_shoulder_plane`, `*_elevation`, `*_rotation`); `"dhm"`: the raw DHM Euler angles. Re-derive with `./frame reanalyse` (minutes), not `run` |  
| `shoulder_mask_deg` |  `20` |  ⟳ plane and rotation are undefined (NaN) below this elevation (gimbal lock) |  
| `capacity_fraction` |  `0.15` |  reference fraction of torque capacity: the dotted line on the torque figures and the threshold of `torque_exceedance_by_task` |  
| `exceedance_min_duration_s` |  `0` |  excursions above the fraction shorter than this are not counted |  
| `npose_by` |  `["task", "orientation"]` |  panels of the N-pose torque map (one row per task) |  
| `coverage_exo_urdf` |  `null` |  exo URDF whose reachable arm directions are drawn on the shoulder globe (panel c); report time only |  
| `globe_min_rom_ratio` |  `0.5` |  a recording is excluded from the right-arm globe when its right shoulder-elevation range is below this fraction of the left one |  
| `timeseries_n_resample` |  `500` |  points each full recording is time-normalised to for the curve figures |  
| `timeseries_normalise_torque` / `torque_speed_normalise` |  `false` |  plot N·m/kg instead of N·m |  
| `torque_speed_clip_pct` |  `99.5` |  axis clipping of the torque–speed planes (every sample is still binned) |  
| `trajectory_segment` |  `RightHand` |  segment drawn in `ee_trajectory_3d_by_task` |  
| `box_max_points` |  `40` |  dots drawn per box (the statistics use every trial) |  
| `cache_float32` / `cache_decimate` |  `true` / `1` |  shrink the cached captures (figures only; metrics are never reduced) |  
| `strict_filenames` |  `false` |  abort instead of skipping files that don't match `trial_pattern` |  
  
   
   
   
 When `joints_of_interest` is empty, the per-figure regexes `plot_dofs`, `timeseries_dofs` and  
   
    
   
  `torque_speed_dofs` decide which joints each figure shows.  
  
---  
**Commands**  
```  
./frame build                       # build the Docker image    
  ./frame list                        # show the dataset without analysing anything    
  ./frame run [--jobs N]              # analyse, pool, write the report (restartable)    
  ./frame report                      # re-pool + re-plot from the cache, no IK    
  ./frame reanalyse [--jobs N]        # recompute every metric from the cached captures (no IK), then report    
  ./frame config                      # print the effective configuration    
      
   
```  
Common options for `list`, `run` and `report`:  
  
|  |  |  
| - | - |  
|   |    |  
| **option** |    |  
| `--subjects S1 S2` |  only these subjects |  
| `--tasks H M` |  only these task types |  
| `--reps 1 2` |  only these repetitions |  
| `--config other.json` |  use another config file |  
| `--no-figures` |  tables only (`run`, `report`) |  
| `--retry-failed` |  re-analyse trials that failed before (`run`) |  
  
   
   
   
 `--subjects`, `--tasks` and `--reps` choose **which recordings take part**. `pool_by` in  
   
    
   
  `batch.json` decides **how those recordings are grouped**.  
   
 `run` exits with code `0` when every trial succeeded, `2` when some failed (details in  
   
    
   
  `failures.json`), and `1` when nothing could be analysed.  
  
---  
**Outputs**  
```  
results/    
    summary.md            the report: groups, headline distributions, top differences, caveats    
    trials.csv            one row per recording, every metric column (+ subject/task/factors)    
    trials.json           the same rows plus provenance (model, frame counts, timing)    
    pooled.json           per group: median/IQR/min/max/n + raw values with their trial ids    
    pooled_summary.csv    long form: one row per (group, metric)    
    differences.csv       metrics ranked by how strongly they separate the groups    
    joint_summary.csv     mean ± SD of peak/RMS joint speed and torque, per group and joint    
    exo_eval/requirements_per_rep.csv    
                          gravity share G and reversal metrics, one row per recording x joint    
    failures.json         recordings that could not be analysed, with tracebacks    
    config.json           the exact configuration used    
    figures/    
      
   
```  
```  
import pandas as pd    
  df = pd.read_csv("results/trials.csv")    
  df.groupby("task")["dynamics.right_shoulder_Z.peak_torque_Nm"].describe()    
      
   
```  
Every dimensional metric also has a **subject-normalised twin** (`…__norm`: per kg for torque,  
   
    
   
  work and power; per metre of stature for lengths). Compare subjects using the `__norm` columns.  
   
    
   
  Angles are never normalised.  
   
 **Figures**  
  
|  |  |  
| - | - |  
|   |    |  
| **name** |  **what it shows** |  
| `torque_exceedance_by_task` |  per joint and task: how many times \|τ\| exceeds `capacity_fraction` × capacity per recording (box), mean excursion duration in brackets |  
| `npose_torque_map_by_task` |  neutral skeleton per task × orientation (`npose_by`), each joint coloured green→red by median RMS torque ÷ joint capacity (shared scale; sphere size and printed % carry the same value) |  
| `torque_speed_by_task` |  one file per task (`torque_speed_by_task_<task>`): log-density hexbin of every (q̇ \[rad/s\], human τ \[N·m\]) sample, one panel per joint (left / right columns) |  
| `joint_timeseries_by_task` |  mean ± 1 SD angle / speed / torque over the **full recording** (% of trial), across all subjects and repetitions; dashed lines = URDF ROM limits / speed capacity / torque capacity |  
| `joint_pose_timeseries` |  one file per task and joint of interest (`joint_pose_timeseries_<task>_<joint>.pdf`): that task's mean ± 1 SD angle, speed and torque for the joint, under three stick figures of the torso and right arm (mean pose at 0, 50 and 100 % of the trial, pelvis frame), with a sphere per joint coloured and sized by \|τ\| ÷ torque capacity at that instant. The first report reads each recording's segment positions once (\~1 s per .mvnx); the three poses are then cached as `<trial>.pose_instants.npz` |  
| `peak_torque_by_task`, `rms_torque_by_task`, `joint_speed_by_task` |  per-joint distributions, with the URDF capacity (dashed) and, for torque, `capacity_fraction` × capacity (dotted) |  
| `shoulder_globe_by_task` |  right-arm shoulder directions on a Zimmermann-style globe (plane of elevation −90…180°, elevation 0…180°, 15° grid), per task × orientation and per task: (a) mean seconds per recording in each 15° × 15° zone, with the Boone & Azen human ROM points and RULA zones, (b) peak \|τ_elevation\| ÷ capacity, (c) directions the exoskeleton can reach (`coverage_exo_urdf`), (d) seconds per recording with \|τ_elevation\| > `capacity_fraction` × capacity in each 15° zone, (e) gravity share G = RMS(τ_g)/RMS(τ) of the elevation torque per cell |  
| `rom_by_task`, `net_work_by_task`, `headline_distributions`, `top_differences` |  further distributions |  
| `ee_trajectory_3d_by_task` |  mean 3D hand path per task over full recordings, individual recordings behind it |  
  
   
   
  
---  
**Caching: what costs hours and what costs seconds**  
   
 Each analysed trial is stored under `cache_root/<subject>/` and keyed by **(recording, analysis**  
   
    
   
  **settings, subject height and mass)**. As a result:  
- an interrupted `run` resumes where it stopped;  
- correcting a subject's height or weight re-analyses **that subject only**;  
- changing a ⟳ setting (see [Advanced settings) re-analyses everything;](#anchor-15 "#anchor-15")  
- changing anything else (joints, grouping, figures, paths, `jobs`) costs one `./frame report`.  
   
 Each subject's model is cached as `models/subject_<id>.urdf`. If you correct an anthropometry  
   
    
   
  file, delete that URDF so the model is rebuilt:  
   
    
   
  `rm models/subject_<id>.urdf models/subject_<id>.xacro`.  
 **Budget:** \~80–90 s per recording at one worker. About 1000 recordings take \~4–5 h at  
   
    
   
  `--jobs 6` and use \~5 GB of cache (`cache_decimate: 4` cuts that to \~1.4 GB).  
  
---  
**Metrics**  
   
 All metrics are computed in `frame-core`, per repetition, then pooled across recordings.  
1. **Task space**: voxel workspace volume and bounding box, reach, end-effector speed and  
   
    
   
  acceleration, dwell concentration, exoskeleton coverage.  
2. **Joint space**: ROM against the model's limits, peak and RMS velocity, direction reversals,  
   
    
   
  and a weighted involvement score (ROM / path / Jacobian / dynamic).  
3. **Dynamics**: peak and RMS torque, torque ÷ joint capacity, peak positive and negative power,  
   
    
   
  net work ∮τ dq over the recording, quasi-stiffness.  
4. **Exoskeleton requirements** (`passivity` layer): gravity share G = RMS(τ_g)/RMS(τ), with  
   
    
   
  τ_g the gravity torque from the inverse dynamics, and the velocity reversals (5°/s deadband,  
   
    
   
  0.1 s apart) with their latency, required acceleration and peak acceleration. Reported per  
   
    
   
  joint in `trials.csv` (`passivity.<dof>.*`) and in `exo_eval/requirements_per_rep.csv`.  
 **Shoulder coordinates**  
   
 The DHM shoulder is an X → Y → Z Euler chain whose zero pose is a **T-pose**, so `shoulder_X/Y/Z`  
   
    
   
  change anatomical meaning with the pose. By default (`shoulder_coords: "elevation"`) every result  
   
    
   
  reports the shoulder in ISB **humerothoracic** coordinates (thorax → humerus, clavicle included):  
   
    
   
  **plane of elevation** (0° = abduction plane, 90° = forward flexion), **elevation** (0° = arm  
   
    
   
  hanging) and **axial rotation** (+ = internal). The left arm is mirrored into the same convention.  
   
    
   
  Torques are the glenohumeral moment projected onto these three axes (power-consistent with the  
   
    
   
  angle rates), and the gravity torque likewise. Capacities (ROM, speed, torque) are derived from  
   
    
   
  the DHM's per-axis URDF limits; speed and torque capacity depend on the pose, so the per-recording  
   
    
   
  median is used. Below `shoulder_mask_deg` elevation the plane and rotation are undefined (gimbal  
   
    
   
  lock) and are NaN; `joint_space.<side>_shoulder_plane.valid_frac` reports how much of each  
   
    
   
  recording is defined (a hanging left arm is often fully masked). The cache keeps the raw DHM  
   
    
   
  angles, so `./frame reanalyse` switches coordinate systems without re-running IK.  
 **Statistics:** `differences.csv` has two p-values. `p_trial` treats every recording as  
   
    
   
  independent, so it overstates significance when each subject contributes many trials; use it for  
   
    
   
  ranking only. `p_subject` first collapses each subject to one median per group. It is the  
   
    
   
  defensible test, and it needs at least 2 subjects per group.  
  
---  
**Known limitations**  
   
 These are properties of the underlying pipeline. `summary.md` repeats them with every report.  
- **Inverse dynamics states were rebuilt (ID_VERSION 2).** `ID_Xsens` stores a zero acceleration  
   
    
   
  (`q_prev` aliasing) and a velocity \~2.08× too fast (`dt = 1/500 s` on 240 Hz data);  
   
  `id_grf.filtered_states` now derives q̇ and q̈ from the IK configurations at the true rate, so  
   
    
   
  the torques include their inertial term. Trials cached before this fix are re-analysed.  
- **Only the arm chain's inverse dynamics is validated.** Spine and leg torques depend on a  
   
    
   
  modelled ground-reaction force that has not been validated.  
- **`filtfilt` edge ringing** inflates peak metrics by about 5%. The torque–speed figures clip  
   
    
   
  their axes to the central 99.5% of samples for this reason.  
- **The GRF model leaves a large vertical base residual** (\~40% of body weight on the checked  
   
    
   
  trial, unchanged by the ID fix). Arm torques do not depend on it; spine and leg torques do.  
- **Recordings are never segmented.** Each one is analysed and plotted as one continuous  
   
    
   
  movement. The curve figures time-normalise every recording to % of trial and average across  
   
    
   
  all subjects and repetitions. Recordings of different duration or pace are therefore compared  
   
    
   
  at the same *fraction* of the trial, not at the same movement event.  
   
 Pooling does not remove these biases: it shows how repeatable the numbers are, not whether  
   
    
   
  they are unbiased.  
  
---  
**Troubleshooting**  
  
|  |  |  
| - | - |  
|   |    |  
| **symptom** |  **cause / fix** |  
| `data_root does not exist` |  the drive is not mounted, or the path in `batch.json` is wrong |  
| `N file(s) did not match the trial pattern` |  rename the files or adjust `trial_pattern` |  
| `has no anthropometry CSV` |  add `anthropo_data_<id>.csv` to the subject folder |  
| `is not a plausible stature` |  the height in the CSV is in m; it must be in cm |  
| `dof_names lists N joint(s) the URDF does not have` |  a typo; the error suggests the nearest name |  
| some trials `FAIL`, the rest fine |  see `failures.json`; fix, then `./frame run --retry-failed` |  
| code change has no effect |  rebuild the image: `./frame build` |  
| everything re-analyses unexpectedly |  a ⟳ setting changed (`./frame config` shows the fingerprint) |  
  
   
   
  
---  
**Development**  
   
 The test suites need neither Docker nor Pinocchio. They exercise the whole chain with fakes.  
```  
python3 -m venv .venv    
  ./.venv/bin/pip install -e ./frame-core    
  ./.venv/bin/pip install -e ./frame-kinematics --no-deps     # before frame-batch: not on PyPI    
  ./.venv/bin/pip install -e ./frame-batch pytest matplotlib    
  ./.venv/bin/python -m pytest frame-core frame-kinematics frame-batch -q    
      
   
```  
`frame-core` asserts that it never imports Pinocchio, which keeps the metric engine usable on its  
   
    
   
  own. To regenerate the schematic, run `./.venv/bin/python scripts/make_schematic.py`.  
   
 The same suites run on every push and pull request through GitHub Actions  
   
    
   
  (`.github/workflows/tests.yml`, Python 3.11 and 3.12).  
```  
frame                  launcher: build / list / run / report / reanalyse / config    
  batch.json             your configuration    
  Dockerfile             the analysis image    
  frame-core/            metric engine + pooling            (pure NumPy / SciPy)    
  frame-kinematics/      IK / ID / GRF / exo reach           (Pinocchio, in the image)    
  frame-batch/           dataset, sweep, cache, report, figures    
  utils/                 ID_Xsens and the MVNX / URDF converters    
  models/                dhm_new.xacro (DHM template), exoskeleton/ URDF + meshes    
  data/S1/             the example subject (recordings in Git LFS)    
  scripts/               make_schematic.py    
      
   
```  
  
---  
**Citation**  
If you use FRAME in your scientific works, please cite it *e.g.* as follows:  
```  
@software{frame,  
   title   = {{FRAME}: Framework for Requirements and Assessment of Movement and Exoskeletons},  
   author  = {Anna Bucchieri},  
   year    = {2026},  
   version = {0.1.0},  
   url     = {XXX}  
 }  
   
```  
Please give the version (or the commit hash) you used: metrics and cache formats can change  
   
 between versions.  
  
---  
**License**  
   
 FRAME is released under the [MIT License.  
   
 The shoulder figures draw on published reference data: the human range of motion of Boone \&  
   
    
   
  Azen (1979), the mean human shoulder capacities of Zimmermann et al. (2023, ANYexo 2.0,  
   
    
   
  Table III) set in `batch.json`, and the RULA upper-arm zones.](LICENSE "LICENSE")  
