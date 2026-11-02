from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

JOINT_NAMES: List[str] = [
    "W",
    "I-CMC",
    "II-MCP",
    "III-MCP",
    "IV-MCP",
    "V-MCP",
    "I-MCP",
    "I-IP",
    "I-TIP",
    "II-PIP",
    "II-DIP",
    "II-TIP",
    "III-PIP",
    "III-DIP",
    "III-TIP",
    "IV-PIP",
    "IV-DIP",
    "IV-TIP",
    "V-PIP",
    "V-DIP",
    "V-TIP",
]

ANGLE_NAMES: List[str] = [
    "F-I-CMC",
    "F-II-MCP",
    "F-III-MCP",
    "F-IV-MCP",
    "F-V-MCP",
    "F-I-MCP",
    "F-II-PIP",
    "F-III-PIP",
    "F-IV-PIP",
    "F-V-PIP",
    "F-I-IP",
    "F-II-DIP",
    "F-III-DIP",
    "F-IV-DIP",
    "F-V-DIP",
    "A-II-TIP",
    "A-III-TIP",
    "A-IV-TIP",
    "A-V-TIP",
    "OP-I-MCP",
    "OP-II-MCP",
    "OP-III-MCP",
    "OP-IV-MCP",
]


@dataclass(frozen=True)
class SkeletonFrame:
    points: np.ndarray  # shape: (21, 3)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v)
    return v / n


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    u1 = _unit(v1)
    u2 = _unit(v2)
    if np.linalg.norm(u1) == 0 or np.linalg.norm(u2) == 0:
        return float("nan")
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _angle(p0: np.ndarray, midpoint: np.ndarray, p1: np.ndarray, mult: int = 1) -> float:
    v1 = midpoint - p0
    v2 = mult * (p1 - midpoint)
    return _angle_between(v1, v2)


def _midpoint(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    return 0.5 * (p0 + p1)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Center at wrist and scale by palm width (II-MCP <-> V-MCP)."""
    centered = frame - frame[0]
    palm_width = np.linalg.norm(centered[2] - centered[5])
    if palm_width == 0:
        return centered
    return centered / palm_width


def skeleton_measures(frame: np.ndarray) -> Dict[str, float]:
    """Compute clinically-oriented hand measures on normalized 3D joints."""
    pts = dict(zip(JOINT_NAMES, frame))

    measures: Dict[str, float] = {}
    measures["F-I-CMC"] = _angle(pts["W"], pts["I-CMC"], pts["I-MCP"])
    measures["F-II-MCP"] = _angle(pts["W"], pts["II-MCP"], pts["II-PIP"])
    measures["F-III-MCP"] = _angle(pts["W"], pts["III-MCP"], pts["III-PIP"])
    measures["F-IV-MCP"] = _angle(pts["W"], pts["IV-MCP"], pts["IV-PIP"])
    measures["F-V-MCP"] = _angle(pts["W"], pts["V-MCP"], pts["V-PIP"])
    measures["F-I-MCP"] = _angle(pts["I-CMC"], pts["I-MCP"], pts["I-IP"])
    measures["F-II-PIP"] = _angle(pts["II-MCP"], pts["II-PIP"], pts["II-DIP"])
    measures["F-III-PIP"] = _angle(pts["III-MCP"], pts["III-PIP"], pts["III-DIP"])
    measures["F-IV-PIP"] = _angle(pts["IV-MCP"], pts["IV-PIP"], pts["IV-DIP"])
    measures["F-V-PIP"] = _angle(pts["V-MCP"], pts["V-PIP"], pts["V-DIP"])
    measures["F-I-IP"] = _angle(pts["I-MCP"], pts["I-IP"], pts["I-TIP"])
    measures["F-II-DIP"] = _angle(pts["II-PIP"], pts["II-DIP"], pts["II-TIP"])
    measures["F-III-DIP"] = _angle(pts["III-PIP"], pts["III-DIP"], pts["III-TIP"])
    measures["F-IV-DIP"] = _angle(pts["IV-PIP"], pts["IV-DIP"], pts["IV-TIP"])
    measures["F-V-DIP"] = _angle(pts["V-PIP"], pts["V-DIP"], pts["V-TIP"])

    # Distances are reported in normalized hand units.
    measures["A-II-TIP"] = float(np.linalg.norm(pts["I-TIP"] - pts["II-TIP"], ord=2))
    measures["A-III-TIP"] = float(np.linalg.norm(pts["II-TIP"] - pts["III-TIP"], ord=2))
    measures["A-IV-TIP"] = float(np.linalg.norm(pts["III-TIP"] - pts["IV-TIP"], ord=2))
    measures["A-V-TIP"] = float(np.linalg.norm(pts["IV-TIP"] - pts["V-TIP"], ord=2))

    measures["OP-I-MCP"] = _angle(pts["I-IP"], _midpoint(pts["II-MCP"], pts["I-MCP"]), pts["II-PIP"], -1)
    measures["OP-II-MCP"] = _angle(pts["II-PIP"], _midpoint(pts["III-MCP"], pts["II-MCP"]), pts["III-PIP"], -1)
    measures["OP-III-MCP"] = _angle(pts["III-PIP"], _midpoint(pts["IV-MCP"], pts["III-MCP"]), pts["IV-PIP"], -1)
    measures["OP-IV-MCP"] = _angle(pts["IV-PIP"], _midpoint(pts["V-MCP"], pts["IV-MCP"]), pts["V-PIP"], -1)

    return measures


def compute_measure_matrix(frames: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
    results: List[List[float]] = []
    for frame in frames:
        work_frame = normalize_frame(frame) if normalize else frame
        measures = skeleton_measures(work_frame)
        results.append([measures[name] for name in ANGLE_NAMES])
    return np.asarray(results, dtype=float), ANGLE_NAMES
