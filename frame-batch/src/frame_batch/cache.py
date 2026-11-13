"""
Per-trial result cache -- what makes a multi-hour sweep restartable.

One trial is a ~100 MB XML parse followed by IK and inverse dynamics: minutes of work. A sweep
over 84 of them cannot be something you have to redo because the machine slept, a single
recording was corrupt, or you wanted a different figure. So every finished trial is written to
disk as soon as it succeeds, keyed by (this recording, this config), and a rerun picks up exactly
where it stopped.

The key deliberately covers the recording's size and mtime as well as its path: replacing a file
in place with a new capture must invalidate its result, and comparing content hashes of 100 MB
files on every run would cost more than it saves.

Failures are cached too, in a separate file. A trial that cannot be analysed is a fact about the
dataset -- it should be reported at the end of the sweep next to the successes, not rediscovered
(slowly) on every rerun. `--retry-failed` clears them.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path


def json_safe(obj):
    """Replace NaN/Inf with None, recursively, so what is written is valid JSON.

    Two reasons, both bites-you-later:

      * `json.dumps` emits bare `NaN`, which Python reads back but strict parsers (R's jsonlite,
        most JS) reject -- and these files exist to be read by other tools.
      * NaN != NaN, so a record that round-trips through the cache would never compare equal to
        the one that produced it, and "is this the cached result?" becomes unanswerable.

    None is also the honest encoding: a NaN here means the metric was not measurable for this
    recording, and `pooling.pool_metric` already skips missing values rather than averaging them.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def trial_key(mvnx_path, config_fingerprint, subject_fingerprint="") -> str:
    """Stable id for 'this recording, analysed with this config, for this subject'.

    `subject_fingerprint` carries the height and mass. They are not config, but they scale the DHM
    and therefore every torque in the result -- so correcting a typo in an anthropometry CSV has to
    invalidate that subject's cached trials. Leaving them out would mean the fix silently changed
    nothing, which is the worst kind of wrong: plausible numbers from a model you already replaced.
    """
    try:
        st = os.stat(mvnx_path)
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        stamp = "missing"
    payload = f"{os.path.basename(mvnx_path)}|{stamp}|{config_fingerprint}|{subject_fingerprint}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class TrialCache:
    """Directory-backed store: <root>/<subject>/<trial_id>.{json,npz,failed.json}."""

    def __init__(self, root):
        self.root = Path(root)

    def _dir(self, subject, create=False) -> Path:
        """This subject's cache directory. Only CREATED when something is about to be written.

        Creating it on a read would leave an empty directory for every subject ever looked up,
        which makes "which subjects are actually analysed?" unanswerable from a directory listing.
        """
        d = self.root / str(subject)
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def record_path(self, subject, trial_id, create=False) -> Path:
        return self._dir(subject, create) / f"{trial_id}.json"

    def capture_path(self, subject, trial_id, create=False) -> Path:
        return self._dir(subject, create) / f"{trial_id}.npz"

    def failure_path(self, subject, trial_id, create=False) -> Path:
        return self._dir(subject, create) / f"{trial_id}.failed.json"

    # --- reads ---------------------------------------------------------------------------- #
    def load(self, subject, trial_id, key):
        """The cached record, but only if it was produced by this exact (recording, config)."""
        p = self.record_path(subject, trial_id)
        if not p.is_file():
            return None
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if rec.get("cache_key") != key:
            return None
        rec["cached"] = True
        return rec

    def load_failure(self, subject, trial_id, key):
        p = self.failure_path(subject, trial_id)
        if not p.is_file():
            return None
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return rec if rec.get("cache_key") == key else None

    def load_capture_bytes(self, subject, trial_id):
        p = self.capture_path(subject, trial_id)
        return p.read_bytes() if p.is_file() else None

    # --- writes --------------------------------------------------------------------------- #
    def save(self, subject, trial_id, record, capture_bytes=None):
        self.record_path(subject, trial_id, create=True).write_text(
            json.dumps(json_safe(record), indent=1))
        self.failure_path(subject, trial_id).unlink(missing_ok=True)
        if capture_bytes:
            self.capture_path(subject, trial_id).write_bytes(capture_bytes)

    def save_failure(self, subject, trial_id, record):
        self.failure_path(subject, trial_id, create=True).write_text(
            json.dumps(json_safe(record), indent=1))

    def clear_failures(self):
        """Drop every cached failure so the next sweep retries them. Returns how many."""
        n = 0
        for p in self.root.rglob("*.failed.json"):
            p.unlink(missing_ok=True)
            n += 1
        return n
