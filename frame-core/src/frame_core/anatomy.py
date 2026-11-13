"""
Joint naming: DHM URDF joints <-> Xsens segments, and what the dynamics layer is validated for.

Two things the torque map needs and nothing else in frame-core carries:

  1. Which body segment to paint for a given DOF. The URDF joint `right_elbow_Z` actuates the
     forearm, so its torque belongs on the RightForeArm marker. Derived by name convention with an
     explicit table, because a wrong mapping paints the load on the wrong limb and nothing about
     the picture would look obviously broken.

  2. Which DOFs the inverse dynamics is actually trustworthy for. The supervisor's position is
     that the dynamic layer is reliable for the UPPER LIMB only; `id_grf` says the same thing from
     the other direction -- the foot Jacobian has zero columns off the base->foot path, so the
     arm-chain torques are identical with or without the modelled ground reaction, while the
     spine and leg torques depend entirely on a GRF model that has not been validated here.
"""
from __future__ import annotations
import re

# Joint-name stem -> the Xsens segment the joint drives (its CHILD, i.e. where the load acts).
JOINT_SEGMENT = {
    "lumbarspine": "L5",
    "thoracicspine": "T8",
    "neck": "Neck",
    "head": "Head",
    "clavicular": "{Side}Shoulder",
    "shoulder": "{Side}UpperArm",
    "elbow": "{Side}ForeArm",
    "wrist": "{Side}Hand",
    "hip": "{Side}UpperLeg",
    "knee": "{Side}LowerLeg",
    "ankle": "{Side}Foot",
}

# Inverse dynamics is validated for the arm chain only -- see the module docstring.
VALIDATED_STEMS = ("clavicular", "shoulder", "elbow", "wrist")
UNVALIDATED_REASON = {
    "lumbarspine": "trunk torque depends on the unvalidated GRF model",
    "thoracicspine": "trunk torque depends on the unvalidated GRF model",
    "neck": "not covered by the upper-limb validation",
    "head": "not covered by the upper-limb validation",
    "hip": "lower limb: torque is dominated by the unvalidated GRF model",
    "knee": "lower limb: torque is dominated by the unvalidated GRF model",
    "ankle": "lower limb: torque is dominated by the unvalidated GRF model",
}

_SIDE = re.compile(r"^(right|left)[_\-]?", re.I)
# A DOF's axis suffix: the DHM Euler axes (_X, _Y, _Z) or the ISB shoulder coordinates
# (_plane, _elevation, _rotation -- see frame_core.shoulder).
_AXIS = re.compile(r"[_\-]?(?:[XYZ]\d*|plane|elevation|rotation)$")


def split_dof(dof_name):
    """('right', 'shoulder') from 'right_shoulder_X'. Side is None for midline joints."""
    name = str(dof_name).strip()
    m = _SIDE.match(name)
    side = m.group(1).lower() if m else None
    stem = name[m.end():] if m else name
    stem = _AXIS.sub("", stem).strip("_-").lower()
    return side, stem


def dof_to_segment(dof_name):
    """The Xsens segment a DOF's torque should be painted on, or None if unmapped."""
    side, stem = split_dof(dof_name)
    template = JOINT_SEGMENT.get(stem)
    if template is None:
        return None
    if "{Side}" in template:
        if side is None:
            return None
        return template.replace("{Side}", "Right" if side == "right" else "Left")
    return template


def is_dynamics_validated(dof_name):
    """True when the inverse dynamics for this DOF is inside the validated (upper-limb) scope."""
    _, stem = split_dof(dof_name)
    return stem in VALIDATED_STEMS


def validation_note(dof_name):
    """Why a DOF sits outside the validated scope, or '' when it is inside it."""
    _, stem = split_dof(dof_name)
    if stem in VALIDATED_STEMS:
        return ""
    return UNVALIDATED_REASON.get(stem, "outside the validated upper-limb scope")


# --- torque-capacity status bands -------------------------------------------------------- #
# Green -> red as the supervisor asked, but as four RESERVED status steps rather than a
# continuous rainbow, and never carried by colour alone: the renderer also scales the marker with
# the ratio and the table prints the number, so the ordering survives any colour-vision deficiency.
TORQUE_BANDS = [
    (0.25, "ok",       "#4a9d7f", "under a quarter of capacity"),
    (0.50, "moderate", "#c9a227", "a quarter to half of capacity"),
    (0.75, "high",     "#c4622a", "half to three quarters of capacity"),
    (None, "critical", "#9b2226", "at or above three quarters of capacity"),
]


def torque_band(ratio):
    """(name, hex, description) for a torque/capacity ratio; the 'unknown' band when it is None."""
    if ratio is None or ratio != ratio:                    # None or NaN
        return ("unknown", "#879199", "no capacity recorded for this joint")
    for upper, name, color, desc in TORQUE_BANDS:
        if upper is None or ratio < upper:
            return (name, color, desc)
    return TORQUE_BANDS[-1][1:]
