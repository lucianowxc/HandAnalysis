from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Iterable

import numpy as np

SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 6),
    (6, 7),
    (7, 8),
    (2, 9),
    (9, 10),
    (10, 11),
    (3, 12),
    (12, 13),
    (13, 14),
    (4, 15),
    (15, 16),
    (16, 17),
    (5, 18),
    (18, 19),
    (19, 20),
]

DEFAULT_MOTION_METRICS: list[str] = [
    "F-II-MCP",
    "F-III-MCP",
    "F-IV-MCP",
    "F-V-MCP",
    "A-II-TIP",
    "A-III-TIP",
    "A-IV-TIP",
    "A-V-TIP",
]


def _read_landmarks_by_frame(path: str | Path) -> dict[int, np.ndarray]:
    rows: dict[int, dict[int, np.ndarray]] = {}
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "joint", "x", "y", "z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in landmarks CSV: {sorted(missing)}")

        from .hand_kinematics import JOINT_NAMES

        joint_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        for row in reader:
            frame = int(row["frame"])
            joint = row["joint"]
            if joint not in joint_to_idx:
                continue
            rows.setdefault(frame, {})[joint_to_idx[joint]] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])], dtype=float
            )

    out: dict[int, np.ndarray] = {}
    for frame, joints in rows.items():
        if len(joints) != 21:
            continue
        arr = np.zeros((21, 3), dtype=float)
        for idx, val in joints.items():
            arr[idx] = val
        out[frame] = arr
    return out


def _load_angles_csv(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in (reader.fieldnames or []) if c != "frame"]
        if not columns:
            raise ValueError("Angles CSV must contain 'frame' and metric columns")

        frame_ids: list[int] = []
        values: list[list[float]] = []
        for row in reader:
            frame_ids.append(int(row["frame"]))
            values.append([float(row[c]) for c in columns])

    return np.asarray(frame_ids, dtype=int), columns, np.asarray(values, dtype=float)


def parse_metrics_arg(metrics: str | None, available_columns: Iterable[str]) -> list[str]:
    available = list(available_columns)
    if metrics is None or metrics.strip() == "":
        return [m for m in DEFAULT_MOTION_METRICS if m in available]

    picked = [m.strip() for m in metrics.split(",") if m.strip()]
    valid = [m for m in picked if m in available]
    if not valid:
        raise ValueError("None of the requested metrics exist in angles CSV")
    return valid


def render_skeleton_overlay_video(
    input_video: str,
    landmarks_csv: str,
    output_video: str,
    point_radius: int = 3,
    line_thickness: int = 2,
) -> Path:
    cv2 = importlib.import_module("cv2")

    landmarks_by_frame = _read_landmarks_by_frame(landmarks_csv)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pts = landmarks_by_frame.get(frame_idx)
            if pts is not None:
                for i, j in SKELETON_EDGES:
                    p1 = (int(pts[i, 0]), int(pts[i, 1]))
                    p2 = (int(pts[j, 0]), int(pts[j, 1]))
                    cv2.line(frame, p1, p2, (0, 220, 255), line_thickness)

                for i in range(pts.shape[0]):
                    p = (int(pts[i, 0]), int(pts[i, 1]))
                    cv2.circle(frame, p, point_radius, (70, 255, 90), -1)

            cv2.putText(
                frame,
                f"frame: {frame_idx}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return out_path


def plot_motion_metrics(
    angles_csv: str,
    output_png: str,
    metrics: str | None = None,
    title: str = "Hand motion metrics over time",
) -> Path:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = importlib.import_module("matplotlib.pyplot")

    frames, columns, matrix = _load_angles_csv(angles_csv)
    selected = parse_metrics_arg(metrics, columns)

    col_idx = {name: i for i, name in enumerate(columns)}

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for name in selected:
        plt.plot(frames, matrix[:, col_idx[name]], label=name, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

    return out_path
