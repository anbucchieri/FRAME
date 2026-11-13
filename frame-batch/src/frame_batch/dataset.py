"""
Dataset discovery -- a `data/` tree of subjects and recordings becomes typed records.

Expected layout (one directory per subject, named by the subject's id):

    data/
      1817/
        anthropo_data_1817.csv          <- Body Height [cm] + Weight [kg] for the DHM
        1817_H_N_E_1.mvnx               <- <subject>_<task>_<f2>_<f3>_<rep>
        1817_H_N_E_2.mvnx
        ...

Nothing about that layout is hard-coded except "one directory per subject". The filename grammar
is a configurable regex (`BatchConfig.trial_pattern`) with named groups, so a study that encodes
its factors differently is a config change, not a code change. Every named group becomes trial
metadata; `subject` and `rep` are special only in that they are also used for identity.

The anthropometry CSV is read per subject, not per recording: height and mass are properties of
the person, and one DHM per person is what makes that person's trials comparable to each other.
"""
from __future__ import annotations
import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class DatasetError(Exception):
    """A dataset the user can fix: a missing anthropometry file, an unparseable filename."""


@dataclass(frozen=True)
class Trial:
    """One .mvnx recording and everything its filename says about it."""
    path: str
    trial_id: str                     # the filename stem -- unique within a subject
    subject: str
    task: str                         # the pooling key (H / M / L)
    rep: Optional[int] = None
    factors: dict = field(default_factory=dict)   # every other named group, kept as metadata

    @property
    def size_mb(self) -> float:
        try:
            return os.path.getsize(self.path) / 1e6
        except OSError:
            return float("nan")


@dataclass
class Subject:
    """One subject directory: anthropometry + the recordings under it."""
    subject_id: str
    root: str
    height_m: Optional[float]
    mass_kg: Optional[float]
    trials: list = field(default_factory=list)
    anthropometry: dict = field(default_factory=dict)   # every row of the CSV, verbatim

    @property
    def urdf_stem(self) -> str:
        """One model per subject -- see frame_kinematics.dhm.subject_urdf_path."""
        return f"subject_{self.subject_id}"

    def fingerprint(self) -> str:
        """Hash of what this subject contributes to a NUMBER: the height and mass the DHM is
        scaled from. Feeds the trial cache key -- see cache.trial_key."""
        import hashlib
        payload = f"{self.subject_id}|{self.height_m!r}|{self.mass_kg!r}"
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def describe(self) -> str:
        h = f"{self.height_m:.2f} m" if self.height_m else "height?"
        m = f"{self.mass_kg:.1f} kg" if self.mass_kg else "mass?"
        tasks = {}
        for t in self.trials:
            tasks[t.task] = tasks.get(t.task, 0) + 1
        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(tasks.items()))
        return f"{self.subject_id}  {h}  {m}  {len(self.trials)} trials ({breakdown})"


# --- anthropometry ------------------------------------------------------------------------ #

# The CSV is a flat "label,value" sheet with blank lines and a section header in the middle; it is
# not a rectangular table, so it is read as key/value pairs rather than with pandas. Keys are
# matched case- and space-insensitively because these files are typed by hand.
_HEIGHT_KEYS = ("body height", "height", "bodyheight")
_MASS_KEYS = ("weight", "mass", "body mass", "bodymass")


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def read_anthropometry(csv_path) -> dict:
    """Every label/value pair in an anthropometry CSV, plus derived `height_m` and `mass_kg`.

    Height is stored in the file in CENTIMETRES ('Body Height,162.0'), and is converted here --
    exactly once, at the edge of the system. Everything downstream is SI, so a model is never
    built 100x too tall because a unit conversion was done twice or not at all.
    """
    raw: dict = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            key, val = row[0].strip(), row[1].strip()
            if not key or not val:
                continue
            raw[key] = val

    lookup = {_norm_key(k): v for k, v in raw.items()}

    def _num(keys):
        for k in keys:
            v = lookup.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except ValueError:
                continue
        return None

    height_cm = _num(_HEIGHT_KEYS)
    mass = _num(_MASS_KEYS)
    out = dict(raw)
    out["height_m"] = height_cm / 100.0 if height_cm else None
    out["mass_kg"] = mass
    if out["height_m"] is not None and not (0.5 < out["height_m"] < 2.6):
        raise DatasetError(
            f"{csv_path}: body height reads as {height_cm} cm -> {out['height_m']:.2f} m, which is "
            f"not a plausible stature. Check the units in the file (it is expected in cm).")
    if mass is not None and not (20.0 < mass < 250.0):
        raise DatasetError(
            f"{csv_path}: weight reads as {mass} kg, which is not a plausible body mass.")
    return out


def find_anthropometry(subject_dir) -> Optional[str]:
    """The anthropometry CSV inside a subject directory, whatever the id suffix is."""
    cands = sorted(Path(subject_dir).glob("anthropo*.csv"))
    if not cands:
        cands = sorted(Path(subject_dir).glob("*.csv"))
    return str(cands[0]) if cands else None


# --- trial filenames ---------------------------------------------------------------------- #

def parse_trial_name(stem, pattern) -> Optional[dict]:
    """Named groups of `pattern` matched against a filename stem, or None if it does not match."""
    m = re.match(pattern, stem)
    return m.groupdict() if m else None


def make_trial(path, pattern, subject_fallback=None) -> Trial:
    """Build a Trial from a .mvnx path, raising a message that names the actual file on a miss."""
    stem = Path(path).stem
    groups = parse_trial_name(stem, pattern)
    if groups is None:
        raise DatasetError(
            f"filename {stem!r} does not match the trial pattern {pattern!r}. Either rename the "
            f"file or set `trial_pattern` in the batch config to the grammar this study uses "
            f"(named groups become trial metadata; 'task' is the pooling key).")
    subject = groups.pop("subject", None) or subject_fallback
    task = groups.pop("task", None)
    rep = groups.pop("rep", None)
    if not task:
        raise DatasetError(
            f"trial_pattern has no 'task' group, so {stem!r} cannot be pooled. Add (?P<task>...) "
            f"to the pattern.")
    return Trial(path=str(path), trial_id=stem, subject=str(subject), task=str(task),
                 rep=int(rep) if rep and str(rep).isdigit() else None,
                 factors={k: v for k, v in groups.items() if v is not None})


# --- discovery ---------------------------------------------------------------------------- #

def discover(data_root, pattern, subjects=None, tasks=None, reps=None, strict=True,
             skipped=None):
    """Walk `data_root` and return [Subject, ...] with their trials attached.

    `subjects` / `tasks` / `reps` filter the sweep without touching the tree -- the normal way to
    smoke-test a change on two recordings before committing to the full multi-hour run.

    `strict=False` downgrades an unparseable filename from an error to a skip, which is what you
    want when a directory also holds notes, exports or a stray recording. Pass a list as `skipped`
    to receive those filenames; the CLI prints them, because a silent skip is how you end up
    reporting on less data than you think you have.

    A subject directory that holds recordings but yields NO parseable ones is always an error,
    even in non-strict mode. That is the shape of a newly added subject whose files follow a
    different naming convention, and quietly dropping the whole person would produce a
    perfectly plausible report built on everyone except them.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise DatasetError(f"data root {root} is not a directory.")

    out = []
    skipped = [] if skipped is None else skipped
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = sub_dir.name
        if subjects and sid not in set(subjects):
            continue
        anthro_path = find_anthropometry(sub_dir)
        if anthro_path is None:
            raise DatasetError(
                f"subject directory {sub_dir} has no anthropometry CSV (expected something like "
                f"anthropo_data_{sid}.csv carrying 'Body Height' in cm and 'Weight' in kg). "
                f"Without it there is no height/mass to scale this subject's DHM with.")
        anthro = read_anthropometry(anthro_path)

        present = sorted(sub_dir.glob("*.mvnx"))
        trials, parsed = [], 0
        for mvnx in present:
            try:
                t = make_trial(mvnx, pattern, subject_fallback=sid)
            except DatasetError:
                if strict:
                    raise
                skipped.append(str(mvnx))
                continue
            parsed += 1
            if tasks and t.task not in set(tasks):
                continue
            if reps and (t.rep is None or t.rep not in set(reps)):
                continue
            trials.append(t)

        if present and parsed == 0:
            raise DatasetError(
                f"subject {sid}: none of its {len(present)} recording(s) match the trial pattern "
                f"{pattern!r} -- e.g. {present[0].name!r}. This subject would be dropped from the "
                f"analysis entirely and the report would look complete without them. Rename the "
                f"files, or widen `trial_pattern` in the batch config to cover both conventions.")

        if not trials:
            continue                        # nothing left after the --tasks/--reps filters
        if anthro.get("height_m") is None or anthro.get("mass_kg") is None:
            raise DatasetError(
                f"{anthro_path}: could not read both a body height and a weight "
                f"(got height_m={anthro.get('height_m')}, mass_kg={anthro.get('mass_kg')}). "
                f"The DHM cannot be scaled without them.")
        out.append(Subject(subject_id=sid, root=str(sub_dir),
                           height_m=anthro["height_m"], mass_kg=anthro["mass_kg"],
                           trials=trials, anthropometry=anthro))

    if not out:
        raise DatasetError(
            f"no analysable subject found under {root}"
            + (f" (skipped {len(skipped)} unparseable file(s))" if skipped else "")
            + ". Expected data/<subject_id>/<recordings>.mvnx + an anthropometry CSV.")
    return out


def all_trials(subjects):
    """Flat [(Subject, Trial), ...] across every subject -- the unit of work for the runner."""
    return [(s, t) for s in subjects for t in s.trials]
