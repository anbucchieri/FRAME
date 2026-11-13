"""
frame-batch tests -- the whole pooled path with injected fakes, no Pinocchio.

Passing here proves the pooled wiring is right: filenames parsed, anthropometry read in the right
units, one DHM per subject, per-trial analysis flattened to a row, rows pooled into per-task
distributions, the cache honoured and invalidated, a bad recording isolated, and the report and
figures written. Only the Pinocchio internals remain to validate on real data -- exactly the split
frame-kinematics' own suite uses.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frame_batch.config import BatchConfig
from frame_batch.dataset import (
    read_anthropometry, make_trial, parse_trial_name, discover, DatasetError, Subject, Trial,
)
from frame_batch.cache import TrialCache, trial_key
from frame_batch import runner


DOFS = ["right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y"]


# --------------------------------------------------------------------------------------------- #
# fakes (same shape as frame-kinematics' suite)
# --------------------------------------------------------------------------------------------- #
class FakeModel:
    def __init__(self, dof_names):
        self._id = {n: i + 1 for i, n in enumerate(dof_names)}

        class _J:
            def __init__(self, iv):
                self.idx_v = iv
        self.joints = [None] + [_J(6 + i) for i in range(len(dof_names))]
        self.nv = 6 + len(dof_names)

    def getJointId(self, name):
        return self._id[name]


class FakeRobot:
    def __init__(self, model):
        self.model = model
        self.data = object()


class FakeIK:
    def __init__(self, urdf_path, mv):
        self.N = len(mv)
        self.xsens_frequency = 240.0
        self.robot = FakeRobot(FakeModel(DOFS))
        self.saved_configurations = np.zeros((self.N, 6 + len(DOFS) + 1))
        self.saved_velocity = np.zeros((self.N, 6 + len(DOFS)))
        self.saved_accelerations = np.zeros((self.N, 6 + len(DOFS)))
        self._log_q = None

    def uploadMocapFile(self):
        pass

    def runFullIK(self):
        t = np.linspace(0, 1, self.N)
        qz = np.deg2rad(90) * np.sin(np.pi * t)
        qx = np.deg2rad(35) * (1 - np.cos(2 * np.pi * t)) / 2
        qy = np.deg2rad(10) * np.sin(4 * np.pi * t)
        self._log_q = {"right_shoulder_Z": qz, "right_shoulder_X": qx, "right_shoulder_Y": qy}
        for k, q in enumerate((qz, qx, qy)):
            self.saved_velocity[:, 6 + k] = np.gradient(q, 1.0 / self.xsens_frequency)
        return self._log_q

    def temp_save_log_q(self):
        return self._log_q

    def butterworth_filter(self, x):
        return np.asarray(x, float)


def fake_run_id(ik, fs=None):
    lq, fs = ik._log_q, ik.xsens_frequency
    om_x = np.gradient(lq["right_shoulder_X"], 1.0 / fs)
    return {"right_shoulder_Z": 8.0 * lq["right_shoulder_Z"],
            "right_shoulder_X": 20.0 * lq["right_shoulder_X"] + 6.0 * om_x,
            "right_shoulder_Y": 0.6 * lq["right_shoulder_Y"]}, {}


def fake_jac(model, data, q, ee_frame):
    J = np.zeros((3, model.nv))
    J[2, 6], J[0, 7], J[1, 8] = 0.40, 0.20, 0.05
    return J


def fake_capture_df(N=300, amplitude=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    d = {}
    for c, base in zip("XYZ", (0.30, 0.0, 0.10)):
        d[f"RightForeArm_{c}"] = base + amplitude * 0.4 * np.sin(np.pi * t) * (c == "Z") \
            + rng.normal(0, 1e-3, N)
    for c in "XYZ":
        d[f"T8_{c}"] = np.zeros(N)
    d["T8_x_q"] = d["T8_y_q"] = d["T8_z_q"] = np.zeros(N)
    d["T8_w_q"] = np.ones(N)
    return pd.DataFrame(d)


# --------------------------------------------------------------------------------------------- #
# fixtures: a miniature dataset on disk
# --------------------------------------------------------------------------------------------- #
ANTHRO = """Identifiant,1817
Age,20.0
Gender,F
Weight,58.0

Segment,Size (cm)
Body Height,162.0
Shoulder Width,31.2

Setting high,175.4
"""


@pytest.fixture
def dataset(tmp_path):
    """data/1817/ with 6 tiny stand-in .mvnx files across 3 task types."""
    sub = tmp_path / "data" / "1817"
    sub.mkdir(parents=True)
    (sub / "anthropo_data_1817.csv").write_text(ANTHRO)
    for task in ("H", "M", "L"):
        for rep in (1, 2):
            (sub / f"1817_{task}_N_E_{rep}.mvnx").write_text("<mvnx/>")
    return tmp_path


@pytest.fixture
def cfg(tmp_path):
    return BatchConfig(
        data_root=str(tmp_path / "data"), output_root=str(tmp_path / "results"),
        models_dir=str(tmp_path / "models"), xacro_base=str(tmp_path / "models" / "dhm.xacro"),
        dof_names=list(DOFS), ee_segment="RightForeArm", ee_frame="right_forearm",
        jobs=1, cache_captures=False,
        headline_metrics=["task_space.workspace_volume_L",
                          "dynamics.right_shoulder_Z.peak_torque_Nm"],
        plot_dofs="shoulder",
    )


def make_converter(amplitude_by_task=None):
    """Stands in for write_Xsens_movement_data_to_csv: writes a movement CSV next to the mvnx.

    The amplitude is driven by the task letter in the filename, so the pooled groups genuinely
    differ and a test can assert that the pooling SEES that difference rather than just that it
    ran.
    """
    amplitude_by_task = amplitude_by_task or {"H": 1.0, "M": 0.6, "L": 0.3}

    def convert(mvnx_path):
        import os
        stem = os.path.basename(mvnx_path).split(".")[0]
        task = stem.split("_")[1]
        out = str(mvnx_path).split(".")[0] + "_com_mov.csv"
        fake_capture_df(300, amplitude_by_task.get(task, 1.0), seed=hash(stem) % 1000).to_csv(
            out, index=False)
        return out, 240.0
    return convert


def fake_urdf_builder(mvnx, h, m, urdf_folder=None, xacro_base_file=None):
    import os
    stem = os.path.basename(str(mvnx)).split(".")[0]
    with open(os.path.join(str(urdf_folder), f"{stem}.urdf"), "w") as f:
        f.write(f"<robot name='{stem}' height='{h}' mass='{m}'/>")


# --------------------------------------------------------------------------------------------- #
# anthropometry
# --------------------------------------------------------------------------------------------- #
def test_anthropometry_reads_height_in_metres_and_mass_in_kg(dataset):
    a = read_anthropometry(dataset / "data" / "1817" / "anthropo_data_1817.csv")
    assert a["height_m"] == pytest.approx(1.62)          # the file says 162.0 cm
    assert a["mass_kg"] == pytest.approx(58.0)
    assert a["Gender"] == "F"                            # every other row survives verbatim


def test_anthropometry_rejects_an_implausible_stature(tmp_path):
    """The cm->m conversion is the one place a factor of 100 can hide; it must not hide."""
    p = tmp_path / "a.csv"
    p.write_text("Weight,58.0\nBody Height,1.62\n")      # already in metres -> reads as 1.6 cm
    with pytest.raises(DatasetError, match="plausible stature"):
        read_anthropometry(p)


def test_anthropometry_rejects_an_implausible_mass(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("Weight,5800\nBody Height,162.0\n")
    with pytest.raises(DatasetError, match="plausible body mass"):
        read_anthropometry(p)


# --------------------------------------------------------------------------------------------- #
# filenames
# --------------------------------------------------------------------------------------------- #
def test_trial_filename_splits_into_task_and_metadata_factors():
    t = make_trial("/d/1817_H_N_E_4.mvnx", BatchConfig().trial_pattern)
    assert (t.subject, t.task, t.rep) == ("1817", "H", 4)
    assert t.factors == {"orientation": "N"}      # kept as metadata, not discarded


def test_an_uncaptured_group_is_matched_but_not_recorded():
    """E/T is the same condition either way, so it must not become a factor column."""
    e = make_trial("/d/1817_H_N_E_4.mvnx", BatchConfig().trial_pattern)
    t = make_trial("/d/1817_H_N_T_4.mvnx", BatchConfig().trial_pattern)
    assert e.factors == t.factors == {"orientation": "N"}
    assert e.trial_id != t.trial_id        # still distinct recordings, they just pool together


def test_unparseable_filename_names_the_file_and_the_pattern():
    with pytest.raises(DatasetError, match="does not match the trial pattern"):
        make_trial("/d/random_export.mvnx", BatchConfig().trial_pattern)


def test_pattern_without_a_task_group_is_rejected():
    with pytest.raises(DatasetError, match="no 'task' group"):
        make_trial("/d/1817_1.mvnx", r"^(?P<subject>[^_]+)_(?P<rep>\d+)$")


def test_parse_trial_name_returns_none_on_a_miss():
    assert parse_trial_name("nope", BatchConfig().trial_pattern) is None


# --------------------------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------------------------- #
def test_discover_attaches_anthropometry_and_every_trial(dataset, cfg):
    subs = discover(cfg.data_root, cfg.trial_pattern)
    assert len(subs) == 1
    s = subs[0]
    assert s.subject_id == "1817" and len(s.trials) == 6
    assert (s.height_m, s.mass_kg) == (pytest.approx(1.62), pytest.approx(58.0))
    assert s.urdf_stem == "subject_1817"


def test_discover_filters_do_not_touch_the_tree(dataset, cfg):
    subs = discover(cfg.data_root, cfg.trial_pattern, tasks=["H"], reps=[1])
    assert [t.trial_id for t in subs[0].trials] == ["1817_H_N_E_1"]


def test_missing_anthropometry_is_an_error_that_says_what_is_missing(dataset, cfg):
    (dataset / "data" / "1817" / "anthropo_data_1817.csv").unlink()
    with pytest.raises(DatasetError, match="anthropometry"):
        discover(cfg.data_root, cfg.trial_pattern)


def test_unparseable_files_are_skipped_when_not_strict(dataset, cfg):
    (dataset / "data" / "1817" / "calibration_export.mvnx").write_text("<mvnx/>")
    assert len(discover(cfg.data_root, cfg.trial_pattern, strict=False)[0].trials) == 6
    with pytest.raises(DatasetError):
        discover(cfg.data_root, cfg.trial_pattern, strict=True)


# --------------------------------------------------------------------------------------------- #
# one DHM per subject
# --------------------------------------------------------------------------------------------- #
def test_subject_urdf_is_built_once_and_named_for_the_subject(dataset, cfg):
    calls = {"n": 0}

    def counting(mvnx, h, m, urdf_folder=None, xacro_base_file=None):
        calls["n"] += 1
        fake_urdf_builder(mvnx, h, m, urdf_folder=urdf_folder, xacro_base_file=xacro_base_file)

    s = discover(cfg.data_root, cfg.trial_pattern)[0]
    first = runner.ensure_subject_urdf(s, cfg, builder=counting)
    second = runner.ensure_subject_urdf(s, cfg, builder=counting)

    assert first == second and first.endswith("subject_1817.urdf")
    assert calls["n"] == 1, "84 recordings of one person must not build 84 models"


def test_subject_urdf_carries_that_subjects_height_and_mass(dataset, cfg):
    s = discover(cfg.data_root, cfg.trial_pattern)[0]
    text = open(runner.ensure_subject_urdf(s, cfg, builder=fake_urdf_builder)).read()
    assert "height='1.62'" in text and "mass='58.0'" in text


# --------------------------------------------------------------------------------------------- #
# one trial end to end
# --------------------------------------------------------------------------------------------- #
def _run_one(dataset, cfg, trial_index=0, cache=None):
    s = discover(cfg.data_root, cfg.trial_pattern)[0]
    urdf = runner.ensure_subject_urdf(s, cfg, builder=fake_urdf_builder)
    trial = sorted(s.trials, key=lambda t: t.trial_id)[trial_index]
    rec = runner.run_trial(trial, s, cfg, urdf, cache=cache, make_ik=FakeIK,
                           converter=make_converter(), run_id=fake_run_id, jac_fn=fake_jac)
    return s, trial, rec


def test_run_trial_produces_a_flat_row_with_provenance(dataset, cfg):
    _, trial, rec = _run_one(dataset, cfg)
    assert rec["ok"] and rec["trial_id"] == trial.trial_id
    assert rec["task"] == trial.task and rec["orientation"] == "N"
    assert rec["height_m"] == pytest.approx(1.62) and rec["mass_kg"] == pytest.approx(58.0)
    assert rec["n_frames"] == 300 and rec["n_reps"] == 1
    m = rec["metrics"]
    assert m["task_space.workspace_volume_L"] > 0
    assert "dynamics.right_shoulder_Z.peak_torque_Nm" in m
    assert "joint_space.right_shoulder_Z.rom_deg" in m
    assert all(v is None or isinstance(v, float) for v in m.values())


def test_metrics_carry_normalised_twins_of_the_dimensional_columns(dataset, cfg):
    _, _, rec = _run_one(dataset, cfg)
    m = rec["metrics"]
    tau = "dynamics.right_shoulder_Z.peak_torque_Nm"
    assert m[tau + "__norm"] == pytest.approx(m[tau] / 58.0)
    assert m["task_space.reach_max_m__norm"] == pytest.approx(m["task_space.reach_max_m"] / 1.62)
    # an angle is already subject-independent and must NOT be divided by anything
    assert "joint_space.right_shoulder_Z.rom_deg__norm" not in m


def test_normalisation_can_be_switched_off(dataset, cfg):
    cfg.normalize = False
    _, _, rec = _run_one(dataset, cfg)
    assert not any(k.endswith("__norm") for k in rec["metrics"])


def test_the_converted_csv_never_lands_in_the_dataset(dataset, cfg):
    """The converter writes beside its input, so the run must feed it a temp copy."""
    _run_one(dataset, cfg)
    assert list((dataset / "data" / "1817").glob("*_com_mov.csv")) == []


def test_gravity_share_and_reversals_reach_the_numeric_row(dataset, cfg):
    _, _, rec = _run_one(dataset, cfg)
    assert "verdicts" not in rec
    assert "passivity.right_shoulder_Z.n_reversals" in rec["metrics"]
    # the fakes carry no gravity torque, so G is undefined (None after json_safe)
    assert rec["metrics"].get("passivity.right_shoulder_Z.G") is None


# --------------------------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------------------------- #
def test_a_finished_trial_is_reused_on_the_next_run(dataset, cfg, tmp_path):
    cache = TrialCache(tmp_path / "cache")
    _, _, first = _run_one(dataset, cfg, cache=cache)
    assert first["cached"] is False
    _, _, second = _run_one(dataset, cfg, cache=cache)
    assert second["cached"] is True
    # exact equality, which only holds because non-finite metrics are None and not NaN
    assert second["metrics"] == first["metrics"]


def test_changing_a_setting_that_changes_a_number_invalidates_the_cache(dataset, cfg, tmp_path):
    cache = TrialCache(tmp_path / "cache")
    _run_one(dataset, cfg, cache=cache)
    before = cfg.fingerprint()
    cfg.voxel_m = 0.05                                   # changes workspace_volume_L
    assert cfg.fingerprint() != before
    _, _, again = _run_one(dataset, cfg, cache=cache)
    assert again["cached"] is False


def test_changing_a_presentation_setting_keeps_the_cache(cfg):
    """Hours of finished IK must not be thrown away because a figure list changed."""
    before = cfg.fingerprint()
    cfg.headline_metrics = ["task_space.reach_max_m"]
    cfg.jobs = 16
    cfg.plot_dofs = "elbow"
    assert cfg.fingerprint() == before


def test_cache_key_follows_the_recording_not_just_its_name(tmp_path):
    p = tmp_path / "t.mvnx"
    p.write_text("a")
    k1 = trial_key(p, "fp")
    p.write_text("a much longer recording")
    assert trial_key(p, "fp") != k1


def test_failures_are_cached_and_clearable(tmp_path):
    cache = TrialCache(tmp_path / "cache")
    cache.save_failure("1817", "t1", {"cache_key": "k", "error": "boom", "ok": False})
    assert cache.load_failure("1817", "t1", "k")["error"] == "boom"
    assert cache.load_failure("1817", "t1", "other") is None      # different config -> retry
    assert cache.clear_failures() == 1
    assert cache.load_failure("1817", "t1", "k") is None


# --------------------------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------------------------- #
# The originals are captured at import time, ONCE. Capturing them inside the patch helper would
# re-wrap an already-patched function on the second call and silently ignore the new fake.
_REAL_RUN_TRIAL = runner.run_trial
_REAL_ENSURE_URDF = runner.ensure_subject_urdf


def _patched_sweep(monkeypatch, converter=None):
    """Wire the fakes into the sweep, which calls run_trial in-process when jobs=1."""
    conv = converter or make_converter()

    def patched(trial, subject, cfg, urdf_path, cache=None, **kw):
        return _REAL_RUN_TRIAL(trial, subject, cfg, urdf_path, cache=cache, make_ik=FakeIK,
                               converter=conv, run_id=fake_run_id, jac_fn=fake_jac)

    monkeypatch.setattr(runner, "run_trial", patched)
    monkeypatch.setattr(runner, "ensure_subject_urdf",
                        lambda s, c, builder=None, force=False:
                        _REAL_ENSURE_URDF(s, c, builder=fake_urdf_builder))


@pytest.fixture
def sweep(monkeypatch):
    _patched_sweep(monkeypatch)
    return runner.run_sweep


def test_sweep_analyses_every_trial_and_groups_them(dataset, cfg, sweep):
    subs = discover(cfg.data_root, cfg.trial_pattern)
    records, failures = sweep(subs, cfg, jobs=1)
    assert len(records) == 6 and not failures
    assert sorted(r["task"] for r in records) == ["H", "H", "L", "L", "M", "M"]
    # a reproducible report needs a row order that does not depend on completion order
    assert [r["trial_id"] for r in records] == sorted(r["trial_id"] for r in records)


def test_one_bad_recording_does_not_take_the_sweep_down(dataset, cfg, sweep, monkeypatch):
    def flaky(mvnx_path):
        if "_M_" in str(mvnx_path):
            raise RuntimeError("corrupt recording")
        return make_converter()(mvnx_path)
    _patched_sweep(monkeypatch, converter=flaky)

    subs = discover(cfg.data_root, cfg.trial_pattern)
    records, failures = sweep(subs, cfg, jobs=1)
    assert len(records) == 4 and len(failures) == 2
    assert all("corrupt recording" in f["error"] for f in failures)
    assert all("traceback" in f for f in failures)


def test_sweep_resumes_from_the_cache(dataset, cfg, sweep, tmp_path):
    subs = discover(cfg.data_root, cfg.trial_pattern)
    cache_root = str(tmp_path / "cache")
    first, _ = sweep(subs, cfg, cache_root=cache_root, jobs=1)
    second, _ = sweep(subs, cfg, cache_root=cache_root, jobs=1)
    assert len(second) == len(first) == 6
    assert all(r["cached"] for r in second)


# --------------------------------------------------------------------------------------------- #
# pooling + report + figures
# --------------------------------------------------------------------------------------------- #
@pytest.fixture
def pooled_run(dataset, cfg, sweep):
    from frame_core.pooling import pool, ranked_differences
    subs = discover(cfg.data_root, cfg.trial_pattern)
    records, failures = sweep(subs, cfg, jobs=1)
    pooled = pool(records, by=cfg.pool_by)
    ranked = ranked_differences(pooled, top=cfg.top_differences)
    return subs, records, failures, pooled, ranked


def test_pooling_reports_one_distribution_per_task(pooled_run):
    _, _, _, pooled, _ = pooled_run
    assert set(pooled) == {"H", "M", "L"}
    for g in pooled:
        entry = pooled[g]["metrics"]["task_space.workspace_volume_L"]
        assert entry["n"] == 2 and pooled[g]["n_trials"] == 2
        assert entry["iqr_lo"] <= entry["median"] <= entry["iqr_hi"]


def test_pooled_values_stay_traceable_to_their_recordings(pooled_run):
    _, _, _, pooled, _ = pooled_run
    e = pooled["H"]["metrics"]["task_space.workspace_volume_L"]
    assert len(e["values"]) == len(e["trials"]) == 2
    assert all(t.startswith("1817_H_") for t in e["trials"])


def test_pooling_sees_a_difference_the_data_actually_has(pooled_run):
    """The fake converter makes H the largest movement; the pooled medians must say so."""
    _, _, _, pooled, _ = pooled_run
    med = {g: pooled[g]["metrics"]["task_space.reach_max_m"]["median"] for g in pooled}
    assert med["H"] > med["M"] > med["L"]


def test_ranked_differences_orders_by_p(pooled_run):
    _, _, _, _, ranked = pooled_run
    ps = [r["p"] for r in ranked]
    assert ps == sorted(ps)
    # one subject: the trial-level p is pseudo-replicated and the honest test cannot run at all
    assert all(r["pseudoreplicated"] and r["n_subjects"] == 1 for r in ranked)
    assert all(r["subject_test"] is None for r in ranked)


def test_report_writes_the_table_the_summary_and_the_config(pooled_run, cfg, tmp_path):
    from frame_batch import report
    subs, records, failures, pooled, ranked = pooled_run
    out = tmp_path / "results"
    paths = report.write_all(records, failures, pooled, ranked, cfg, subs, out)

    header = open(paths["trials_csv"]).readline().strip().split(",")
    assert header[:4] == ["trial_id", "subject", "task", "rep"]
    assert "orientation" in header and "factor_3" not in header
    assert "task_space.workspace_volume_L" in header
    assert not any(c.startswith("verdict.") for c in header)
    assert sum(1 for _ in open(paths["trials_csv"])) == 7          # header + 6 trials

    summary = open(paths["summary_md"]).read()
    assert "FRAME pooled analysis" in summary and "1.62 m" in summary
    assert "Known limitations" in summary, "the known caveats must travel with the numbers"

    assert json.loads(open(paths["config_json"]).read())["ee_segment"] == "RightForeArm"
    assert json.loads(open(paths["pooled_json"]).read())["H"]["n_trials"] == 2


def test_figures_are_written(pooled_run, cfg, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import plots
    _, _, _, pooled, ranked = pooled_run
    cfg.figures = []                                  # empty = every figure
    written = plots.build_all(pooled, ranked, cfg, tmp_path / "figures")
    names = {p.rsplit("/", 1)[-1] for p in written}
    assert "headline_distributions.pdf" in names and "rom_by_task.pdf" in names
    assert "verdicts_by_task.pdf" not in names
    assert all((tmp_path / "figures" / n).stat().st_size > 1000 for n in names)


def test_task_colours_are_stable_and_follow_the_task_not_the_count():
    from frame_batch.plots import group_colors
    full = group_colors(["H", "M", "L"])
    assert full["H"] == "#2a78d6"
    filtered = group_colors(["H", "L"])
    assert filtered["H"] == full["H"], "filtering a group must not repaint the survivors"


# --------------------------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------------------------- #
def test_unknown_config_key_is_named(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps({"data_rooot": "data"}))
    with pytest.raises(ValueError, match="data_rooot"):
        BatchConfig.load(p)


def test_config_round_trips_through_disk(cfg, tmp_path):
    p = tmp_path / "c.json"
    cfg.save(p)
    assert BatchConfig.load(p).fingerprint() == cfg.fingerprint()


# --------------------------------------------------------------------------------------------- #
# what the cache key must and must not cover
# --------------------------------------------------------------------------------------------- #
def test_changed_anthropometry_invalidates_that_subjects_trials(dataset, cfg, tmp_path):
    """Height and mass scale the DHM, so correcting them must not be served from cache."""
    cache = TrialCache(tmp_path / "cache")
    _, _, first = _run_one(dataset, cfg, cache=cache)
    assert first["cached"] is False

    (dataset / "data" / "1817" / "anthropo_data_1817.csv").write_text(
        ANTHRO.replace("Weight,58.0", "Weight,72.0"))
    _, _, again = _run_one(dataset, cfg, cache=cache)
    assert again["cached"] is False, "a re-measured subject must be re-analysed"
    assert again["mass_kg"] == pytest.approx(72.0)


def test_renaming_a_factor_does_not_throw_away_the_analysis(dataset, cfg, tmp_path):
    """A column heading is not a number. Re-labelling must cost seconds, not hours."""
    cache = TrialCache(tmp_path / "cache")
    _, _, first = _run_one(dataset, cfg, cache=cache)
    assert first["orientation"] == "N"

    cfg.trial_pattern = (r"^(?P<subject>[^_]+)_(?P<task>[A-Za-z])_(?P<posture>[A-Za-z])"
                         r"_(?:[A-Za-z])_(?P<rep>\d+)$")
    _, _, again = _run_one(dataset, cfg, cache=cache)
    assert again["cached"] is True, "the numbers are unchanged, so the cache must hold"
    assert again["posture"] == "N"
    assert "orientation" not in again, "the stale column must not linger beside the new one"
    assert again["metrics"] == first["metrics"]


def test_refresh_metadata_keeps_the_expensive_half_untouched():
    from frame_batch.runner import refresh_metadata
    rec = {"trial_id": "old", "task": "H", "factor_2": "N", "metrics": {"a": 1.0},
           "n_frames": 300, "duration_s": 12.3,
           "cache_key": "k", "urdf": "/m/subject_1817.urdf"}
    trial = Trial(path="/d/1817_L_P_E_2.mvnx", trial_id="1817_L_P_E_2", subject="1817",
                  task="L", rep=2, factors={"orientation": "P"})
    subject = Subject(subject_id="1817", root="/d", height_m=1.62, mass_kg=58.0)
    out = refresh_metadata(rec, trial, subject)
    assert out["task"] == "L" and out["rep"] == 2 and out["orientation"] == "P"
    assert "factor_2" not in out
    assert out["metrics"] == {"a": 1.0}
    assert out["n_frames"] == 300 and out["urdf"] == "/m/subject_1817.urdf"


def test_subject_fingerprint_tracks_height_and_mass():
    a = Subject(subject_id="1817", root="/d", height_m=1.62, mass_kg=58.0)
    b = Subject(subject_id="1817", root="/d", height_m=1.62, mass_kg=58.0)
    c = Subject(subject_id="1817", root="/d", height_m=1.62, mass_kg=58.5)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


# --------------------------------------------------------------------------------------------- #
# mean +/- SD joint trajectories
# --------------------------------------------------------------------------------------------- #
def test_timeseries_dof_selection_follows_the_config_order():
    from frame_batch.timeseries import select_dofs
    cfg = BatchConfig()
    assert select_dofs(cfg.dof_names, cfg.timeseries_dofs) == [
        "right_shoulder_X", "right_shoulder_Y", "right_shoulder_Z",
        "right_elbow_Z", "right_elbow_Y"]
    assert select_dofs(["a_x", "b_x"], None) == ["a_x", "b_x"]


def test_timeseries_settings_do_not_invalidate_the_cache(cfg):
    """A figure's DOF list or sample count is presentation -- it must never cost a re-analysis."""
    before = cfg.fingerprint()
    cfg.timeseries_dofs = "elbow"
    cfg.timeseries_n_resample = 500
    cfg.timeseries_normalise_torque = True
    assert cfg.fingerprint() == before


@pytest.fixture
def cached_run(dataset, cfg, sweep, tmp_path):
    """A sweep that kept its Captures, so the trajectory bands have something to read."""
    cfg.cache_captures = True
    cfg.timeseries_dofs = "right_shoulder"
    subs = discover(cfg.data_root, cfg.trial_pattern)
    cache_root = str(tmp_path / "cache")
    records, _ = sweep(subs, cfg, cache_root=cache_root, jobs=1)
    return records, TrialCache(cache_root), cfg


def test_timeseries_bands_pool_every_trial_of_a_task(cached_run):
    from frame_batch import timeseries
    records, cache, cfg = cached_run
    bands, missing = timeseries.collect(records, cache, cfg)
    assert missing == 0
    assert set(bands) == {"H", "M", "L"}
    b = bands["H"]["right_shoulder_Z"]["angle_deg"]
    assert b["n"] == 2                        # 2 trials x 1 rep each
    assert len(b["mean"]) == len(b["sd"]) == cfg.timeseries_n_resample
    assert b["pct"][0] == 0.0 and b["pct"][-1] == 100.0
    assert np.all(b["hi"] >= b["mean"]) and np.all(b["lo"] <= b["mean"])


def test_timeseries_figure_is_written(cached_run, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import timeseries
    records, cache, cfg = cached_run
    path, note = timeseries.build(records, cache, cfg, tmp_path / "figs")
    assert path and path.endswith("joint_timeseries_by_task.pdf")
    assert (tmp_path / "figs" / "joint_timeseries_by_task.pdf").stat().st_size > 5000
    assert note == ""


STANDING = {"Pelvis": (0, 0, 1.0), "L5": (0, 0, 1.05), "L3": (0, 0, 1.12), "T12": (0, 0, 1.2),
            "T8": (0, 0, 1.32), "Neck": (0, 0, 1.5), "RightShoulder": (0, -0.05, 1.45),
            "RightUpperArm": (0, -0.18, 1.42), "RightForeArm": (0, -0.2, 1.12),
            "RightHand": (0.05, -0.2, 0.9)}


def fake_segment_frame(rec, cfg, n=200):
    """Stands in for the .mvnx read: a standing body whose right hand rises by 0.4 m."""
    d = {}
    for seg, (x, y, z) in STANDING.items():
        lift = np.linspace(0.0, 0.4, n) if seg == "RightHand" else np.zeros(n)
        d.update({f"{seg}_X": np.full(n, x), f"{seg}_Y": np.full(n, y), f"{seg}_Z": z + lift,
                  f"{seg}_x_q": np.zeros(n), f"{seg}_y_q": np.zeros(n), f"{seg}_z_q": np.zeros(n),
                  f"{seg}_w_q": np.ones(n)})
    return pd.DataFrame(d)


def test_joint_pose_poses_are_in_the_pelvis_frame_and_cached(cached_run, monkeypatch):
    from frame_batch import joint_pose
    records, cache, cfg = cached_run
    monkeypatch.setattr(joint_pose, "_trial_frame", fake_segment_frame)
    pose = joint_pose.trial_poses(records[0], cache, cfg)
    assert pose.shape == (3, len(joint_pose.SEGMENTS), 3)
    hand = joint_pose.SEGMENTS.index("RightHand")
    assert pose[:, hand, 2] == pytest.approx([-0.1, 0.1, 0.3])      # pelvis at z = 1.0
    # the second read comes from the cache, not from the recording
    def no_read(rec, cfg):
        raise AssertionError("the recording was read again")
    monkeypatch.setattr(joint_pose, "_trial_frame", no_read)
    assert np.array_equal(joint_pose.trial_poses(records[0], cache, cfg), pose)


def test_joint_pose_figure_per_task_and_joint(cached_run, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import joint_pose
    records, cache, cfg = cached_run
    monkeypatch.setattr(joint_pose, "_trial_frame", fake_segment_frame)
    paths, note = joint_pose.build(records, cache, cfg, tmp_path / "figs")
    names = sorted(Path(p).name for p in paths)
    assert names == sorted(f"joint_pose_timeseries_{t}_{d}.pdf"
                           for t in ("H", "M", "L") for d in DOFS)
    assert all(Path(p).stat().st_size > 5000 for p in paths)
    assert note == ""
    pose, missing = joint_pose.collect([r for r in records if r["task"] == "H"], cache, cfg)
    assert pose["n"] == 2 and missing == 0


def test_missing_captures_are_reported_not_silently_averaged_away(cached_run):
    """A band built from half the trials, with no warning, is worse than no band."""
    from frame_batch import timeseries
    records, cache, cfg = cached_run
    cache.capture_path(records[0]["subject"], records[0]["trial_id"]).unlink()
    bands, missing = timeseries.collect(records, cache, cfg)
    assert missing == 1


def test_timeseries_says_why_it_produced_nothing_when_captures_are_off(dataset, cfg, sweep,
                                                                      tmp_path):
    from frame_batch import timeseries
    cfg.cache_captures = False
    subs = discover(cfg.data_root, cfg.trial_pattern)
    cache_root = str(tmp_path / "cache")
    records, _ = sweep(subs, cfg, cache_root=cache_root, jobs=1)
    path, note = timeseries.build(records, TrialCache(cache_root), cfg, tmp_path / "figs")
    assert path is None and "cache_captures" in note


def test_torque_normalisation_divides_by_that_subjects_mass(cached_run):
    from frame_batch import timeseries
    records, cache, cfg = cached_run
    raw, _ = timeseries.collect(records, cache, cfg)
    cfg.timeseries_normalise_torque = True
    norm, _ = timeseries.collect(records, cache, cfg)
    a = raw["H"]["right_shoulder_Z"]["torque_Nm"]["mean"]
    b = norm["H"]["right_shoulder_Z"]["torque_Nm"]["mean"]
    assert np.allclose(b, a / 58.0)


def test_a_skipped_recording_is_reported_not_silently_dropped(dataset, cfg):
    """A file that does not match the pattern is excluded from every number -- say so."""
    (dataset / "data" / "1817" / "1817_M_P_E_ERROR.mvnx").write_text("<mvnx/>")
    skipped = []
    subs = discover(cfg.data_root, cfg.trial_pattern, strict=False, skipped=skipped)
    assert len(subs[0].trials) == 6
    assert skipped and skipped[0].endswith("1817_M_P_E_ERROR.mvnx")


def test_summary_names_the_recordings_it_did_not_analyse(pooled_run, cfg, tmp_path):
    from frame_batch import report
    subs, records, failures, pooled, ranked = pooled_run
    paths = report.write_all(records, failures, pooled, ranked, cfg, subs, tmp_path / "r",
                             skipped=["/data/1817/1817_M_P_E_ERROR.mvnx"])
    summary = open(paths["summary_md"]).read()
    assert "Recordings NOT analysed" in summary
    assert "1817_M_P_E_ERROR.mvnx" in summary


# --------------------------------------------------------------------------------------------- #
# torque-speed envelopes
# --------------------------------------------------------------------------------------------- #
def test_torque_speed_selects_every_upper_limb_joint():
    from frame_batch.torque_speed import select_dofs
    cfg = BatchConfig()
    dofs = select_dofs(cfg.dof_names, cfg.torque_speed_dofs)
    assert len(dofs) == 18
    assert "right_shoulder_Z" in dofs and "left_wrist_X" in dofs
    assert not any("hip" in d or "knee" in d or "spine" in d for d in dofs)


def test_torque_speed_settings_do_not_invalidate_the_cache(cfg):
    before = cfg.fingerprint()
    cfg.torque_speed_dofs = "elbow"
    cfg.torque_speed_normalise = True
    cfg.torque_speed_clip_pct = 95.0
    assert cfg.fingerprint() == before


def test_torque_speed_collects_every_sample_in_si(cached_run):
    from frame_batch import torque_speed
    records, cache, cfg = cached_run
    clouds, missing = torque_speed.collect(records, cache, cfg, ["right_shoulder_Z"])
    assert missing == 0
    c = clouds["H"]["right_shoulder_Z"]
    n = sum(rec["n_frames"] for rec in records if rec["task"] == "H")
    assert c["omega"].size == c["tau"].size == n, "nothing is thinned"
    assert np.nanmax(np.abs(c["omega"])) < 50, "speed is in rad/s"


def test_torque_speed_writes_one_figure_per_task(cached_run, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import torque_speed
    records, cache, cfg = cached_run
    cfg.torque_speed_dofs = "right_shoulder"
    paths, note = torque_speed.build(records, cache, cfg, tmp_path / "figs")
    assert sorted(Path(p).name for p in paths) == [
        "torque_speed_by_task_H.pdf", "torque_speed_by_task_L.pdf", "torque_speed_by_task_M.pdf"]
    assert all(Path(p).stat().st_size > 5000 for p in paths)


# --------------------------------------------------------------------------------------------- #
# the three things batch.json must let you change cheaply
# --------------------------------------------------------------------------------------------- #
def test_data_and_cache_paths_do_not_invalidate_the_cache(cfg):
    """Moving the dataset or the cache is not a change to any number."""
    before = cfg.fingerprint()
    cfg.data_root = "/mnt/some/other/disk/recordings"
    cfg.cache_root = "/mnt/elsewhere/cache"
    assert cfg.fingerprint() == before


def test_joints_of_interest_does_not_invalidate_the_cache(cfg):
    """The whole point: change the list, re-plot in seconds, never re-run IK."""
    before = cfg.fingerprint()
    cfg.joints_of_interest = ["right_shoulder_Z"]
    assert cfg.fingerprint() == before


def test_cache_size_settings_do_not_invalidate_the_cache(cfg):
    before = cfg.fingerprint()
    cfg.cache_float32 = False
    cfg.cache_decimate = 4
    assert cfg.fingerprint() == before


def test_selected_dofs_follows_dof_names_order_and_drops_unknowns():
    cfg = BatchConfig(dof_names=["a_X", "b_Y", "c_Z"])
    assert cfg.selected_dofs() == ["a_X", "b_Y", "c_Z"], "empty list means all of them"
    cfg.joints_of_interest = ["c_Z", "a_X", "nope"]
    assert cfg.selected_dofs() == ["a_X", "c_Z"], "order follows dof_names, not the request"
    assert cfg.unknown_joints() == ["nope"]


def test_dof_regex_matches_exactly_and_falls_back():
    import re
    cfg = BatchConfig(dof_names=["right_elbow_Z", "right_elbow_Y"])
    assert cfg.dof_regex(fallback="shoulder") == "shoulder", "no list -> the figure's own default"
    cfg.joints_of_interest = ["right_elbow_Z"]
    rx = re.compile(cfg.dof_regex(fallback="shoulder"))
    assert rx.search("right_elbow_Z") and not rx.search("right_elbow_Y")


def test_resolved_cache_root_defaults_under_the_output_dir():
    assert BatchConfig(output_root="out").resolved_cache_root() == "out/cache"
    assert BatchConfig(output_root="out", cache_root="/big/disk").resolved_cache_root() == "/big/disk"


def test_cached_capture_is_smaller_but_still_usable(dataset, cfg, tmp_path):
    """float32 halves the cache; decimation divides it again and rescales fs with it."""
    from frame_kinematics.serialize import capture_to_npz_bytes, capture_from_npz_bytes
    import numpy as np
    from frame_core.types import Capture, Repetition

    rng = np.random.default_rng(0)
    cap = Capture(reps=[Repetition(angles=rng.normal(size=(2400, 6)),
                                   torques=rng.normal(size=(2400, 6)),
                                   ee_pos=rng.normal(size=(2400, 3)))],
                  dof_names=[f"j{i}" for i in range(6)], fs=240.0)
    full = capture_to_npz_bytes(cap)
    small = capture_to_npz_bytes(cap, float32=True)
    tiny = capture_to_npz_bytes(cap, float32=True, decimate=4)
    assert len(small) < 0.65 * len(full)
    assert len(tiny) < 0.35 * len(small)

    back = capture_from_npz_bytes(tiny)
    assert back.reps[0].angles.shape[0] == 600
    assert back.fs == pytest.approx(60.0), "fs must follow the decimation or every speed is wrong"
    assert capture_from_npz_bytes(small).fs == pytest.approx(240.0)


def test_decimated_capture_still_yields_correct_joint_speeds():
    """The fs rescaling is the thing that makes decimation safe -- prove it on a known signal."""
    from frame_kinematics.serialize import capture_to_npz_bytes, capture_from_npz_bytes
    from frame_core.curves import trial_curves
    import numpy as np
    from frame_core.types import Capture, Repetition

    fs, N = 240.0, 2400
    t = np.arange(N) / fs
    q = np.deg2rad(30.0) * np.sin(2 * np.pi * 0.5 * t)          # 0.5 Hz, peak 30 deg
    cap = Capture(reps=[Repetition(angles=q[:, None])], dof_names=["j"], fs=fs)

    peaks = []
    for kw in ({}, {"float32": True}, {"float32": True, "decimate": 4}):
        back = capture_from_npz_bytes(capture_to_npz_bytes(cap, **kw))
        c = trial_curves(back, ["j"], n_resample=200)
        peaks.append(float(np.max(np.abs(c["j"]["speed_dps"]))))
    # analytic peak = 30 deg * 2*pi*0.5 rad/s ~= 94 deg/s
    for p in peaks:
        assert p == pytest.approx(94.2, rel=0.05)


# --------------------------------------------------------------------------------------------- #
# figure selection, output format, and crowding
# --------------------------------------------------------------------------------------------- #
def test_only_the_named_figures_are_produced(pooled_run, cfg, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import plots
    _, _, _, pooled, ranked = pooled_run
    cfg.figures = ["rms_torque_by_task", "joint_speed_by_task"]
    names = {p.rsplit("/", 1)[-1] for p in plots.build_all(pooled, ranked, cfg, tmp_path / "f")}
    assert names == {"rms_torque_by_task.pdf", "joint_speed_by_task.pdf"}


def test_figure_format_switches_every_file(pooled_run, cfg, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import plots
    _, _, _, pooled, ranked = pooled_run
    cfg.figures = ["rms_torque_by_task"]
    cfg.figure_format = "png"
    names = {p.rsplit("/", 1)[-1] for p in plots.build_all(pooled, ranked, cfg, tmp_path / "p")}
    assert names == {"rms_torque_by_task.png"}


def test_unknown_figure_name_is_reported(cfg):
    cfg.figures = ["rms_torque_by_task", "not_a_figure"]
    assert cfg.unknown_figures() == ["not_a_figure"]


def test_figure_settings_do_not_invalidate_the_cache(cfg):
    before = cfg.fingerprint()
    cfg.figures = ["rom_by_task"]
    cfg.figure_format = "png"
    cfg.box_max_points = 5
    assert cfg.fingerprint() == before


def test_thinning_reduces_drawn_points_without_touching_the_statistics():
    """Only the strip is thinned -- the box beside it still summarises every trial."""
    from frame_batch.plots import thin
    vals = list(np.linspace(0.0, 100.0, 300))
    shown = thin(vals, 40)
    assert len(shown) <= 40
    assert min(shown) == pytest.approx(0.0) and max(shown) == pytest.approx(100.0)
    assert thin(vals, 40) == shown, "deterministic: the same figure on every run"
    assert thin(vals, None) == vals and thin(vals, 1000) == vals


# --------------------------------------------------------------------------------------------- #
# N-pose torque map
# --------------------------------------------------------------------------------------------- #
def test_npose_map_pools_per_task_and_writes_a_figure(cached_run, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import npose
    records, cache, cfg = cached_run
    # default: one panel per task x orientation (the fixture's files are all orientation N)
    by_group, missing = npose.collect(records, cache, cfg, pattern="shoulder")
    assert set(by_group) == {"H_N", "M_N", "L_N"} and missing == 0
    cfg.npose_by = ["task"]
    by_task, missing = npose.collect(records, cache, cfg, pattern="shoulder")
    assert set(by_task) == {"H", "M", "L"} and missing == 0
    entry = by_task["H"]
    assert entry["segments"], "at least one segment carries a torque"
    assert entry["pose"].ndim == 2 and entry["pose"].shape[1] == 3
    for v in entry["segments"].values():
        assert v["rms"] >= 0 and v["n"] == 2

    path, note = npose.build(records, cache, cfg, tmp_path / "figs")
    assert path and path.endswith("npose_torque_map_by_task.pdf")
    assert (tmp_path / "figs" / "npose_torque_map_by_task.pdf").stat().st_size > 5000


# --------------------------------------------------------------------------------------------- #
# the joint summary table
# --------------------------------------------------------------------------------------------- #
def test_joint_summary_csv_carries_the_sizing_numbers(pooled_run, cfg, tmp_path):
    import csv
    from frame_batch import report
    subs, records, failures, pooled, ranked = pooled_run
    paths = report.write_all(records, failures, pooled, ranked, cfg, subs, tmp_path / "r")
    rows = list(csv.DictReader(open(paths["joint_summary_csv"])))
    assert rows, "the summary must not be empty"
    assert {r["metric"] for r in rows} == {
        "peak_vel_dps", "rms_vel_dps", "peak_torque_Nm", "rms_torque_Nm",
        "n_reversals", "latency_med_s", "latency_min_s"}
    assert {r["task"] for r in rows} == {"H", "M", "L"}
    for r in rows:
        assert r["n"], "every row reports how many recordings are behind it"
        assert r["unit"] in {"deg/s", "N.m", "count", "s"}
    assert {r["joint"] for r in rows} <= set(DOFS)


# --------------------------------------------------------------------------------------------- #
# joint capacity limits drawn on the torque and speed figures
# --------------------------------------------------------------------------------------------- #
def test_limits_are_read_from_the_cached_captures(cached_run):
    """No re-analysis: the URDF limits ride in every cached Capture."""
    from frame_batch import limits as L
    records, cache, cfg = cached_run
    got = L.collect(records, cache, dofs=set(DOFS))
    assert set(got) <= set(DOFS)
    for dof, e in got.items():
        assert e["n_subjects"] >= 1
        assert e["varies"] is False, "one subject cannot disagree with itself"


def test_rad_per_s_limit_is_converted_to_deg_per_s():
    """The URDF stores rad/s; the speed figure plots deg/s. Mixing them is a 57x error."""
    from frame_batch.limits import collect, RAD_TO_DEG

    class FakeCache:
        def load_capture_bytes(self, s, t):
            from frame_core.types import Capture, Repetition
            from frame_kinematics.serialize import capture_to_npz_bytes
            cap = Capture(reps=[Repetition(angles=np.zeros((10, 1)))], dof_names=["j"], fs=240.0)
            cap.urdf_limits = {"j": {"effort": 51.0, "velocity": 10.0}}
            return capture_to_npz_bytes(cap)

    got = collect([{"subject": "s1", "trial_id": "t1"}], FakeCache())
    assert got["j"]["effort"] == pytest.approx(51.0)
    assert got["j"]["velocity_dps"] == pytest.approx(10.0 * RAD_TO_DEG)
    assert got["j"]["velocity_dps"] == pytest.approx(572.96, abs=0.1)


def test_each_metric_is_matched_to_the_right_limit():
    from frame_batch.limits import for_metric
    lim = {"j": {"effort": 51.0, "velocity_dps": 573.0}}
    assert for_metric(lim, "j", "peak_torque_Nm") == (51.0, "N.m")
    assert for_metric(lim, "j", "rms_torque_Nm") == (51.0, "N.m")
    assert for_metric(lim, "j", "peak_vel_dps") == (573.0, "deg/s")
    assert for_metric(lim, "j", "rms_vel_dps") == (573.0, "deg/s")
    # a metric with no corresponding limit must not borrow one
    assert for_metric(lim, "j", "rom_deg") == (None, None)
    assert for_metric(lim, "other", "peak_torque_Nm") == (None, None)


def test_a_limit_far_above_the_data_is_annotated_not_drawn(pooled_run, cfg, tmp_path):
    """Drawing a limit 20x the data would flatten every box into a line at the bottom."""
    pytest.importorskip("matplotlib")
    from frame_batch import plots
    _, _, _, pooled, ranked = pooled_run
    huge = {d: {"effort": 1e5, "velocity_dps": 1e5} for d in DOFS}
    p = plots.per_dof_figure(pooled, tmp_path / "far.pdf", metric_suffix="peak_torque_Nm",
                             layer="dynamics", dof_pattern="shoulder", limits=huge)
    assert p and (tmp_path / "far.pdf").stat().st_size > 1000


def test_limit_in_range_is_drawn(pooled_run, cfg, tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import plots
    _, _, _, pooled, ranked = pooled_run
    near = {d: {"effort": 20.0, "velocity_dps": 300.0} for d in DOFS}
    p = plots.per_dof_figure(pooled, tmp_path / "near.pdf", metric_suffix="peak_torque_Nm",
                             layer="dynamics", dof_pattern="shoulder", limits=near,
                             limit_label="URDF torque capacity")
    assert p and (tmp_path / "near.pdf").stat().st_size > 1000


# --------------------------------------------------------------------------------------------- #
# exoskeleton requirements table (G + reversals, from the trial records)
# --------------------------------------------------------------------------------------------- #
def test_exo_eval_writes_one_row_per_recording_and_dof(pooled_run, cfg, tmp_path):
    import csv as _csv
    from frame_batch import exo_eval
    from frame_core.flatten import PASSIVITY_KEYS
    _, records, _, _, _ = pooled_run
    paths = exo_eval.write(records, cfg, tmp_path)
    rows = list(_csv.DictReader(open(paths["requirements_per_rep"])))
    assert len(rows) == len(records) * len(cfg.selected_dofs())
    assert set(PASSIVITY_KEYS) | {"trial_id", "task", "orientation", "rep", "dof"} <= set(rows[0])
    assert {r["dof"] for r in rows} == set(cfg.selected_dofs())


def test_compound_groups_follow_their_task_order():
    from frame_batch.plots import _order_groups
    assert _order_groups(["L_P", "M_O", "H_P", "L_N", "H_N", "M_N", "M_P"]) == \
        ["H_N", "H_P", "M_N", "M_O", "M_P", "L_N", "L_P"]
    assert _order_groups(["L", "H", "M", "X"]) == ["H", "M", "L", "X"]


# --------------------------------------------------------------------------------------------- #
# torque exceedance of a fraction of capacity
# --------------------------------------------------------------------------------------------- #
def test_exceedance_counts_runs_and_durations():
    from frame_batch.exceedance import trial_exceedance
    fs = 100.0
    tau = np.zeros(1000)
    tau[100:150] = 20.0          # 0.5 s above
    tau[300:310] = -20.0         # 0.1 s above (|tau| counts)
    tau[500:700] = 9.0           # below 15% of 100 N.m -> not counted
    x = trial_exceedance(tau, 100.0, fs, 0.15)
    assert x["count"] == 2 and x["mean_dur_s"] == pytest.approx(0.3)
    assert x["time_frac"] == pytest.approx(60 / 1000)
    y = trial_exceedance(tau, 100.0, fs, 0.15, min_duration_s=0.2)
    assert y["count"] == 1 and y["mean_dur_s"] == pytest.approx(0.5)
    assert trial_exceedance(tau, None, fs) is None


def test_exceedance_reaches_the_records_and_the_figure(cached_run, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import frame_kinematics.serialize as ser
    from frame_batch import exceedance
    from frame_core.pooling import pool
    records, cache, cfg = cached_run
    # the fake captures carry no URDF limits: give them a capacity of 10 N.m -> threshold 1.5 N.m
    real = ser.capture_from_npz_bytes

    def with_limits(b):
        cap = real(b)
        cap.urdf_limits = {d: {"effort": 10.0} for d in cap.dof_names}
        return cap
    monkeypatch.setattr(ser, "capture_from_npz_bytes", with_limits)
    out, missing = exceedance.augment(records, cache, cfg)
    assert missing == 0
    m = out[0]["metrics"]
    # Z is the spring-like 8 q torque (well above 1.5 N.m for most of the movement); Y is tiny
    assert m["dynamics.right_shoulder_Z.exceed15_count"] >= 1
    assert m["dynamics.right_shoulder_Y.exceed15_count"] == 0
    assert 0 < m["dynamics.right_shoulder_Z.exceed15_time_frac"] <= 1
    assert "dynamics.right_shoulder_Z.exceed15_count" not in records[0]["metrics"]   # copies
    pooled = pool(out, by=cfg.pool_by)
    notes = exceedance.pooled_mean_durations(pooled, "exceed15")
    assert np.isfinite(notes[("H", "right_shoulder_Z")])
    assert exceedance.build_figure(pooled, cfg, tmp_path)


# --------------------------------------------------------------------------------------------- #
# shoulder elevation coordinates + reanalyse (no IK)
# --------------------------------------------------------------------------------------------- #
def _raw_shoulder_capture():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ts", Path(__file__).resolve().parents[2] / "frame-core" / "tests" / "test_shoulder.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod._shoulder_capture()


def test_analyse_capture_reports_elevation_coordinates():
    from frame_batch.runner import analyse_capture
    cap = _raw_shoulder_capture()
    cfg = BatchConfig(dof_names=list(cap.dof_names))
    subj = type("S", (), {"height_m": 1.7, "mass_kg": 70.0})()
    metrics, result, _ = analyse_capture(cap, cfg, subj)
    assert "joint_space.right_shoulder_elevation.rom_deg" in metrics
    assert "dynamics.right_shoulder_plane.peak_torque_Nm" in metrics
    assert not any("right_shoulder_X" in k for k in metrics)
    assert 0 < metrics["joint_space.right_shoulder_plane.valid_frac"] < 1
    assert cfg.selected_dofs()[2:5] == ["right_shoulder_plane", "right_shoulder_elevation",
                                        "right_shoulder_rotation"]
    # "dhm" keeps the raw Euler DOFs and is the cache fingerprint of the pre-transform era
    raw_cfg = BatchConfig(dof_names=list(cap.dof_names), shoulder_coords="dhm")
    m2, *_ = analyse_capture(cap, raw_cfg, subj)
    assert "joint_space.right_shoulder_X.rom_deg" in m2
    assert raw_cfg.fingerprint() != cfg.fingerprint()


def test_reanalyse_rebuilds_records_from_the_cache(dataset, cfg, sweep, tmp_path):
    from frame_batch.runner import run_reanalyse
    from frame_batch.cache import trial_key
    cfg.cache_captures = True
    cfg.shoulder_coords = "dhm"
    cache_root = str(tmp_path / "cache")
    subs = discover(cfg.data_root, cfg.trial_pattern)
    first, _ = sweep(subs, cfg, cache_root=cache_root, jobs=1)
    cfg.shoulder_coords = "elevation"
    out = run_reanalyse(subs, cfg, cache_root, jobs=1)
    assert len(out["done"]) == 6 and not out["missing"] and not out["failed"]
    cache = TrialCache(cache_root)
    for s in subs:
        for t in s.trials:
            rec = cache.load(s.subject_id, t.trial_id,
                             trial_key(t.path, cfg.fingerprint(), s.fingerprint()))
            assert rec is not None and rec["reanalysed_from_capture"]
            old = next(r for r in first if r["trial_id"] == t.trial_id)
            # no clavicle in these fakes -> the transform is a no-op; float32 cache -> ~1e-6 rel.
            for k, v in old["metrics"].items():
                if v is None:                 # undefined metric (e.g. no reversal): stays None
                    assert rec["metrics"][k] is None, k
                    continue
                assert rec["metrics"][k] == pytest.approx(v, rel=1e-4, abs=1e-6, nan_ok=True), k
    again = run_reanalyse(subs, cfg, cache_root, jobs=1)
    assert len(again["current"]) == 6 and not again["done"]


def test_capacity_overrides_replace_the_derived_capacity_with_provenance():
    from frame_batch.runner import analyse_capture
    cap = _raw_shoulder_capture()
    cfg = BatchConfig(dof_names=list(cap.dof_names), capacity_overrides={
        "right_shoulder_elevation": {"effort": 77.0, "source": "test"}})
    out = cfg.transform_capture(cap)
    e = out.urdf_limits["right_shoulder_elevation"]
    assert e["effort"] == 77.0 and e["effort_source"] == "test"
    assert out.urdf_limits["right_shoulder_plane"]["effort"] != 77.0       # only the one named
    subj = type("S", (), {"height_m": 1.7, "mass_kg": 70.0})()
    m, result, _ = analyse_capture(cap, cfg, subj)
    peak_max = result["dynamics"]["per_dof"]["right_shoulder_elevation"]["peak_torque"]["max"]
    assert m["dynamics.right_shoulder_elevation.peak_torque_ratio"] == pytest.approx(peak_max / 77.0)
    assert BatchConfig(dof_names=list(cap.dof_names)).fingerprint() != cfg.fingerprint()


# --------------------------------------------------------------------------------------------- #
# shoulder globe
# --------------------------------------------------------------------------------------------- #
def test_globe_map_places_known_directions():
    from frame_batch import globe
    lon, lat = globe.lonlat([45.0, 0.0, 90.0], [90.0, 0.0, 180.0])
    assert np.allclose(np.rad2deg(lon), [0.0, -45.0, 45.0])       # centred on phi = 45
    assert np.allclose(np.rad2deg(lat), [0.0, -90.0, 90.0])       # hanging = bottom pole
    g = globe.norm_region()
    assert globe.inside(g, np.array([0.0, 90.0, 140.0]), np.array([90.0, 60.0, 120.0])).all()
    assert not globe.inside(g, np.array([-90.0, 170.0]), np.array([90.0, 90.0])).any()


def test_globe_collects_metrics_and_draws(tmp_path):
    pytest.importorskip("matplotlib")
    from frame_batch import globe
    from frame_batch.cache import TrialCache
    from frame_kinematics.serialize import capture_to_npz_bytes
    cap = _raw_shoulder_capture()
    cache = TrialCache(tmp_path / "c")
    cache.save("s1", "t1", {"cache_key": "k"}, capture_bytes=capture_to_npz_bytes(cap))
    cfg = BatchConfig(dof_names=list(cap.dof_names),
                      capacity_overrides={"right_shoulder_elevation": {"effort": 77.0}})
    rec = {"subject": "s1", "trial_id": "t1", "task": "H", "orientation": "N",
           "metrics": {"joint_space.right_shoulder_elevation.rom_deg": 90.0,
                       "joint_space.left_shoulder_elevation.rom_deg": 15.0}}
    lefty = dict(rec, trial_id="t2",
                 metrics={"joint_space.right_shoulder_elevation.rom_deg": 12.0,
                          "joint_space.left_shoulder_elevation.rom_deg": 80.0})
    cache.save("s1", "t2", {"cache_key": "k"}, capture_bytes=capture_to_npz_bytes(cap))
    exo = (np.array([0.0, 30.0, 60.0]), np.array([60.0, 60.0, 60.0]))   # a narrow exo region
    groups, metrics, missing = globe.collect([rec, lefty], cache, cfg, exo)
    m = metrics["t1"]
    assert missing == 0 and set(groups) == {"H_N", "H"}                # fine + pooled-by-task
    assert groups["H"]["n"] == 1 and groups["H"]["excluded"] == ["t2"]  # left-arm recording out
    assert m["shoulder_globe.right.is_working_arm"] == 1.0
    assert metrics["t2"] == {"shoulder_globe.right.is_working_arm": 0.0}
    assert 0 <= m["shoulder_globe.right.exo_inside_frac"] < 1         # most samples elsewhere
    assert m["shoulder_globe.right.norm_inside_frac"] == pytest.approx(1.0)
    assert m["shoulder_globe.right.peak_demand_ratio"] > 0
    assert globe.working_arm(lefty) == "left" and globe.working_arm(rec) == "right"
    both = {"metrics": {"joint_space.right_shoulder_elevation.rom_deg": 134.0,
                        "joint_space.left_shoulder_elevation.rom_deg": 137.0}}
    assert globe.working_arm(both) == "right"                         # two-handed: kept
    # (d) time above the capacity fraction never exceeds the time spent; (e) G is a ratio or NaN
    h = groups["H"]
    assert np.all(h["seconds_over"] <= h["seconds"] + 1e-12) and h["seconds"].sum() > 0
    G = globe.gravity_share(h)
    assert np.all(np.isnan(G) | (G >= 0))
    # the working-arm check reads the unfiltered records (one-arm joints_of_interest)
    narrowed = [dict(r, metrics={k: v for k, v in r["metrics"].items() if "left" not in k})
                for r in (rec, lefty)]
    _, m2, _ = globe.collect(narrowed, cache, cfg, exo, arm_records=[rec, lefty])
    assert m2["t2"] == {"shoulder_globe.right.is_working_arm": 0.0}
    exo_g = globe.fill_holes(globe.occupancy(*exo))
    assert globe.group_figure(groups["H_N"], "H_N", tmp_path / "g.png", exo_g, "77 N.m")
    assert globe.group_figure(groups["H"], "H", tmp_path / "g2.png", exo_g, "77 N.m")
    assert globe.augment([rec], metrics)[0]["metrics"]["shoulder_globe.right.exo_inside_frac"] == \
        m["shoulder_globe.right.exo_inside_frac"]


def test_rula_upper_arm_score_from_direction():
    from frame_batch.globe import rula_score
    #            forward 10/30/60/100 deg | behind 30/100 deg | side 60 | overhead tilted back
    phi = [90, 90, 90, 90, -90, -90, 0, -10]
    th = [10, 30, 60, 100, 30, 100, 60, 130]
    assert rula_score(phi, th).tolist() == [1, 2, 3, 4, 2, 4, 3, 4]


def test_excluded_metrics_are_left_out_of_the_written_results(cfg):
    keep = cfg.keep_metric
    for k in ("dynamics.d.exceed15_time_frac", "dynamics.d.exceed15_total_s",
              "dynamics.d.k_Nm_per_rad", "dynamics.d.k_Nm_per_rad__norm", "dynamics.d.r2",
              "dynamics.d.net_work_J", "dynamics.d.net_work_J__norm",
              "dynamics.d.peak_power_pos_W", "dynamics.d.peak_power_neg_W__norm",
              "dynamics.d.peak_torque_Nm__norm", "dynamics.d.rms_torque_Nm__norm",
              "joint_space.d.zero_crossings", "joint_space.d.reversals",
              "joint_space.d.mean_crossing_gap_s", "joint_space.d.min_crossing_gap_s",
              "pca.n_pc_90", "pca.engagement.d"):
        assert not keep(k), k
    for k in ("dynamics.d.exceed15_count", "dynamics.d.exceed15_mean_dur_s",
              "dynamics.d.peak_torque_Nm", "dynamics.d.rms_torque_Nm",
              "dynamics.d.rms_torque_ratio", "passivity.d.G", "passivity.d.n_reversals",
              "passivity.d.latency_med_s", "passivity.d.alpha_req_max_rad_s2",
              "joint_space.d.peak_vel_dps", "task_space.workspace_volume_L__norm"):
        assert keep(k), k
    cfg.exclude_metrics = []
    assert keep("pca.n_pc_90")
