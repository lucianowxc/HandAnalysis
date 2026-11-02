from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .hand_kinematics import JOINT_NAMES


def read_landmarks_csv(path: str | Path) -> np.ndarray:
    """Read long-format landmarks CSV into ndarray [frames, 21, 3].

    Expected columns:
    - frame: int
    - joint: one of JOINT_NAMES
    - x, y, z: float (z can be relative depth)
    """
    rows: Dict[int, Dict[str, np.ndarray]] = {}
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "joint", "x", "y", "z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            frame = int(row["frame"])
            joint = row["joint"]
            if joint not in JOINT_NAMES:
                raise ValueError(f"Unknown joint name: {joint}")
            rows.setdefault(frame, {})[joint] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])], dtype=float
            )

    if not rows:
        raise ValueError("Input CSV has no landmark rows")

    frame_ids = sorted(rows)
    frames = []
    for frame_id in frame_ids:
        joints = rows[frame_id]
        missing_joints = [j for j in JOINT_NAMES if j not in joints]
        if missing_joints:
            raise ValueError(
                f"Frame {frame_id} is missing joints: {', '.join(missing_joints)}"
            )
        frames.append(np.stack([joints[name] for name in JOINT_NAMES], axis=0))
    return np.stack(frames, axis=0)


def write_measurements_csv(path: str | Path, matrix: np.ndarray, columns: Iterable[str]) -> None:
    columns = list(columns)
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", *columns])
        for idx, row in enumerate(matrix):
            writer.writerow([idx, *[f"{v:.6f}" for v in row]])


def write_summary_csv(path: str | Path, matrix: np.ndarray, columns: Iterable[str]) -> None:
    columns = list(columns)
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for i, name in enumerate(columns):
            series = matrix[:, i]
            writer.writerow(
                [
                    name,
                    f"{np.nanmean(series):.6f}",
                    f"{np.nanstd(series):.6f}",
                    f"{np.nanmin(series):.6f}",
                    f"{np.nanmax(series):.6f}",
                ]
            )
