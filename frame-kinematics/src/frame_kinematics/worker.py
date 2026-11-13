"""
Heavy-path entry point: one movement dataframe in, one `frame_core.Capture` out.

`run_job(df, job, ...)` trims the capture, runs the per-rep heavy chain, folds in the exo
reach (if requested), and returns a `frame_core.Capture` ready for `frame_core.analyze`. Every
heavy dependency (IK factory, ID runner, Jacobian fn, exo builder) is injectable, so the entire
path is exercised under test with fakes and touches Pinocchio only via the production defaults.
"""
from __future__ import annotations

from .segment import trim_edges
from .kinematics import process_rep
from .assemble import build_capture
from .exo import compute_exo_cloud
from .dhm import build_or_load_urdf


def resolve_urdf(job, builder=None):
    """Resolve the DHM URDF for this job: use a prebuilt one, hit the per-subject cache, or build
    it once from the subject's .mvnx + the fixed xacro base. Returns a path (str)."""
    if job.urdf_path:                                   # explicit prebuilt URDF wins
        from pathlib import Path
        if Path(job.urdf_path).exists():
            return str(job.urdf_path)
    urdf = build_or_load_urdf(
        models_dir=job.models_dir, xacro_base=job.xacro_base,
        height_m=job.height_m, mass_kg=job.mass_kg,
        mvnx_path=job.mvnx_path, subject_urdf=job.urdf_path,
        subject_id=job.subject_id, builder=builder,
        urdf_stem=getattr(job, "urdf_stem", None))
    return str(urdf)


def check_dof_names(dof_names, model_joints, ee_frame=None, model_links=None):
    """Fail fast on a DOF or frame name the built URDF does not have.

    Without this the mismatch surfaces only once IK has already run -- as a KeyError on the
    log_q dict, minutes into a job, naming nothing useful.
    """
    known = set(model_joints or ())
    if not known:
        return                                   # introspection unavailable; nothing to check against
    missing = [n for n in dof_names if n not in known]
    if missing:
        import difflib
        hints = []
        for n in missing:
            near = difflib.get_close_matches(n, known, n=2, cutoff=0.6)
            hints.append(f"{n!r}" + (f" (did you mean {' or '.join(map(repr, near))}?)" if near else ""))
        raise ValueError(
            f"job.dof_names lists {len(missing)} joint(s) the built URDF does not have: "
            + "; ".join(hints)
            + f". The model has {len(known)} movable joints: {', '.join(sorted(known))}."
        )
    if ee_frame and model_links and ee_frame not in set(model_links):
        import difflib
        near = difflib.get_close_matches(ee_frame, model_links, n=2, cutoff=0.6)
        raise ValueError(
            f"job.ee_frame {ee_frame!r} is not a link frame in the built URDF"
            + (f" (did you mean {' or '.join(map(repr, near))}?)" if near else "")
            + "."
        )


def resolve_scene_inputs(job, urdf_path, model_fn=None, npose_fn=None):
    """Model + .mvnx metadata that the movement dataframe cannot supply.

    `model_fn(urdf_path)` follows introspect_model's contract: {"joints", "links", "limits", ...}.
    Returns (model_info, npose_world). The N-pose is best-effort (it only feeds the torque map), but
    a model that introspects successfully IS used to validate job.dof_names -- see check_dof_names.
    """
    model_info, npose_world = None, None
    try:
        if model_fn is None:
            from .introspect import introspect_model
            model_fn = introspect_model
        model_info = model_fn(urdf_path)
    except Exception:
        model_info = None
    try:
        if job.mvnx_path:
            if npose_fn is None:
                from .mvnx import read_calibration_pose
                npose_fn = read_calibration_pose
            npose_world = npose_fn(job.mvnx_path) or None
    except Exception:
        npose_world = None
    return model_info, npose_world


def run_job(df, job, make_ik=None, run_id=None, jac_fn=None,
            exo_fn=None, urdf_builder=None, want_torque=True, want_jac=True, progress=None,
            model_fn=None, npose_fn=None):
    """Heavy-path entry: capture dataframe + KinematicsJob -> frame_core.Capture.

    df           : loaded capture dataframe (units header already dropped).
    make_ik/... : injectable heavy deps (default to the Pinocchio-backed implementations).
    urdf_builder: injectable create_urdf_from_Xsens_new (defaults to the project's utils).
    want_torque : run inverse dynamics (Layers 3-4). want_jac: compute the Jacobian term.
    progress    : optional callback(stage, i, n).
    model_fn/npose_fn : injectable model-introspection / calibration readers
                  (see resolve_scene_inputs).
    """
    # ---- resolve/build the person-scaled DHM URDF (cached per subject) ----
    if progress:
        progress("urdf", 0, 1)
    resolved_urdf = resolve_urdf(job, builder=urdf_builder)
    job.urdf_path = resolved_urdf                       # per-rep IK reads job.urdf_path
    if progress:
        progress("urdf", 1, 1)

    model_info, npose_world = resolve_scene_inputs(job, resolved_urdf, model_fn, npose_fn)
    urdf_limits = (model_info or {}).get("limits")
    check_dof_names(job.dof_names, (model_info or {}).get("joints"),
                    job.ee_frame, (model_info or {}).get("links"))

    # ---- drop the settling frames at both ends before anything reads the data ----
    n_raw = len(df)
    df, window = trim_edges(df, job.fs, job.trim_start_s, job.trim_end_s, job.min_len)
    trim_info = {"raw_frames": n_raw, "kept_frames": len(df),
                 "start_s": float(job.trim_start_s), "end_s": float(job.trim_end_s),
                 "window_frames": list(window),
                 "window_s": [window[0] / job.fs, window[1] / job.fs]}
    if progress:
        progress("trim", len(df), n_raw)

    # Measure the capture's trunk-frame convention ONCE, from anatomy, and reuse it for every
    # repetition so the ee cloud, the skeleton and the exo reach cannot drift apart.
    from .canonical import xsens_trunk_convention
    convention = xsens_trunk_convention(df, job.trunk_segment)

    # The recording is analysed as ONE continuous movement -- it is never segmented.
    if len(df) < job.min_len:
        raise ValueError(f"the recording has {len(df)} samples, fewer than min_len={job.min_len}.")
    reps = [df.reset_index(drop=True)]
    payloads = []
    for i, mv in enumerate(reps):
        payloads.append(process_rep(mv, job, make_ik=make_ik, run_id=run_id, jac_fn=jac_fn,
                                    want_torque=want_torque, want_jac=want_jac,
                                    npose_world=npose_world, convention=convention))
        if progress:
            progress("ik", i + 1, len(reps))

    exo_cloud = (exo_fn or compute_exo_cloud)(job) if job.exo_urdf else None

    cap = build_capture(payloads, job.dof_names, job.fs, exo_cloud=exo_cloud,
                        height=job.height_m, mass=job.mass_kg, urdf_limits=urdf_limits)
    cap.trim = trim_info
    return cap
