"""
MVNX -> movement CSV conversion.

Each .mvnx is converted to the movement CSV with the project's
`write_Xsens_movement_data_to_csv(mvnx_file) -> (csv_path, frame_rate)` (it writes
`<stem>_com_mov.csv` next to the mvnx and returns the path + capture rate), then IK/ID run on that
CSV. The converter is injected so this module and its tests run without the project's utils.

The whole recording is analysed as one continuous movement; it is never segmented.
"""
from __future__ import annotations
import os

MVNX_NS = "{http://www.xsens.com/mvn/mvnx}"


def read_calibration_pose(mvnx_path, pose="identity"):
    """Segment world positions from one of the .mvnx calibration frames -> {segment: (x, y, z)}.

    The converter drops these frames (it slices `[3:]` past 'identity', 'tpose' and 'tpose-isb')
    because they are not movement -- but 'identity' is exactly the subject's NEUTRAL pose: every
    segment orientation is identity, so the body stands with the arms down along the global axes.
    That is the N-pose the torque map wants, already scaled to this subject and carrying the same
    23 segments in the same order as the movement columns.

    Pure stdlib XML -- no Pinocchio, no URDF link-name mapping. Returns {} if the frame is absent.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(mvnx_path)).getroot()
    labels = [s.get("label") for s in root.iter(MVNX_NS + "segment")]
    for frame in root.iter(MVNX_NS + "frame"):
        if frame.get("type") != pose:
            continue
        el = frame.find(MVNX_NS + "position")
        if el is None:
            break
        vals = [float(v) for v in el.text.split()]
        return {name: tuple(vals[3 * i:3 * i + 3])
                for i, name in enumerate(labels) if 3 * i + 3 <= len(vals)}
    return {}


def _default_converter():
    from utils.process_mvnx_file import write_Xsens_movement_data_to_csv
    return write_Xsens_movement_data_to_csv


def mvnx_to_csv(mvnx_path, converter=None):
    """Convert an .mvnx to the movement CSV. Returns (csv_path, frame_rate).

    frame_rate may be None if the converter doesn't report it; callers fall back to the job fs.
    """
    converter = converter or _default_converter()
    result = converter(str(mvnx_path))
    if isinstance(result, tuple):
        csv_path, frame_rate = (list(result) + [None])[:2]
    else:
        csv_path, frame_rate = result, None
    if not csv_path or not os.path.isfile(csv_path):
        raise RuntimeError(f"MVNX->CSV conversion did not produce a file (got {csv_path!r})")
    return csv_path, frame_rate
