"""
DHM URDF build stage -- person-scaled digital human model from a fixed xacro base + subject
height/mass, cached per subject.

IMPORTANT (discovered from the real create_urdf_from_Xsens_new):
  * It reads anatomical SEGMENT DIMENSIONS from the .mvnx (get_body_dimensions_from_Xsens),
    NOT from the pick_to_place CSV -- those calibration landmarks aren't in the CSV. So building
    a subject's URDF needs the .mvnx ONCE. IK itself runs from the CSV.
  * It writes '{urdf_folder}/{trial_name}.urdf' (trial_name = mvnx basename) and returns None.
    So we derive the path ourselves and check the cache, mirroring kinematic-analysis.py's
    `if not urdf_path.exists()` behaviour.

Because the URDF is cached per subject, the .mvnx is only required on the FIRST run for a given
subject (cache miss). Later runs for the same subject need just the CSV.

The builder itself (process_mvnx_file.create_urdf_from_Xsens_new) lives in the project's `utils`
and is imported lazily / injectable, so this module stays importable (and testable) without it.
"""
from __future__ import annotations
import os
from pathlib import Path


def _default_builder():
    """Production builder from the project's utils package."""
    from utils import process_mvnx_file
    return process_mvnx_file.create_urdf_from_Xsens_new


def urdf_path_for(models_dir, mvnx_path):
    """The URDF path create_urdf_from_Xsens_new would write for this mvnx (trial_name = basename)."""
    trial = os.path.basename(str(mvnx_path)).split(".")[0]
    return Path(models_dir) / f"{trial}.urdf"


def subject_urdf_path(models_dir, subject_id):
    """The per-SUBJECT model path: `models_dir/subject_<id>.urdf`.

    The pooled tool analyses tens of recordings of the same person. Anthropometry is a property of
    the SUBJECT, not of the recording, so all of them must share one model -- keying the cache on
    the .mvnx stem (as `urdf_path_for` does) would build 84 near-identical URDFs for subject 1817
    and, worse, let per-recording calibration jitter leak into the torques as if it were task
    variance. One model per subject makes every trial of that subject dimensionally comparable.
    """
    return Path(models_dir) / f"subject_{subject_id}.urdf"


def build_or_load_urdf(models_dir, xacro_base, height_m, mass_kg,
                       mvnx_path=None, subject_urdf=None, subject_id=None,
                       builder=None, force=False, urdf_stem=None):
    """Return a path to a person-scaled DHM URDF, building it once if absent.

    By default naming follows create_urdf_from_Xsens_new: the URDF is named after the .mvnx stem,
    so `S1.mvnx` -> `models_dir/S1.urdf`, and `subject_id` is only a label.

    Pass `urdf_stem` to cache per SUBJECT instead (see `subject_urdf_path`): the builder still
    writes its mvnx-named file, which is then renamed to `models_dir/<urdf_stem>.urdf`, so every
    later recording of that subject is a cache hit and needs no .mvnx at all.

    Resolution order (first hit wins, unless force):
      1. `subject_urdf` given and exists -> use it (explicit prebuilt model; how a CSV-only repeat
         run reuses a cached URDF -- point urdf_path at models_dir/<name>.urdf).
      2. cached `<urdf_stem>.urdf`, else `<mvnx-stem>.urdf`, exists -> reuse it (no rebuild).
      3. otherwise BUILD from mvnx_path + xacro_base (the .mvnx is required: the CSV can't supply
         the anatomical segment dimensions).
    """
    models_dir = str(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    # 1) explicit prebuilt URDF (also the CSV-only reuse path)
    if subject_urdf:
        p = Path(subject_urdf)
        if p.exists() and not force:
            return p

    # 2a) per-subject cache -- checked BEFORE the mvnx-stem cache and without needing an .mvnx,
    #     which is what lets trials 2..84 of a subject skip the build entirely.
    target = Path(models_dir) / f"{urdf_stem}.urdf" if urdf_stem else None
    if target is not None and target.exists() and not force:
        return target

    # 2b) cache derived from the mvnx basename (the exact name the builder writes)
    if mvnx_path is not None:
        cached = urdf_path_for(models_dir, mvnx_path)
        if cached.exists() and not force:
            return _adopt(cached, target)
    else:
        # no mvnx and no prebuilt URDF -> we cannot build
        raise FileNotFoundError(
            "no .mvnx provided and no prebuilt urdf_path given: the DHM is built once from the "
            ".mvnx (it carries the anatomical segment dimensions the CSV lacks). Send the .mvnx "
            "on the first run; on later runs either send the .mvnx again or set urdf_path to the "
            "cached models/<name>.urdf.")

    # 3) build -- named after the mvnx stem by create_urdf_from_Xsens_new
    builder = builder or _default_builder()
    folder = models_dir if models_dir.endswith(os.sep) else models_dir + os.sep
    builder(str(mvnx_path), height_m, mass_kg, urdf_folder=folder, xacro_base_file=str(xacro_base))

    out = urdf_path_for(models_dir, mvnx_path)
    if not out.exists():
        raise RuntimeError(f"URDF build reported success but {out} is missing")
    return _adopt(out, target)


def _adopt(built, target):
    """Rename a freshly built (or mvnx-named) URDF to its per-subject name, sidecar .xacro too.

    Returns `built` unchanged when no per-subject name was asked for, so the single-trial
    behaviour is byte-for-byte what it was.
    """
    if target is None or Path(built).resolve() == Path(target).resolve():
        return Path(built)
    built = Path(built)
    target = Path(target)
    built.replace(target)
    sidecar = built.with_suffix(".xacro")
    if sidecar.exists():
        sidecar.replace(target.with_suffix(".xacro"))
    return target
