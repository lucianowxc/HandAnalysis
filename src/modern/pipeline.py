from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hand_kinematics import compute_measure_matrix
from .io import read_landmarks_csv, write_measurements_csv, write_summary_csv


@dataclass(frozen=True)
class PipelineOutputs:
    measurements_csv: Path
    summary_csv: Path
    n_frames: int
    n_measures: int


def run_pipeline(
    input_csv: str,
    output_csv: str,
    summary_csv: str | None = None,
    normalize: bool = True,
) -> PipelineOutputs:
    frames = read_landmarks_csv(input_csv)
    matrix, angle_names = compute_measure_matrix(frames, normalize=normalize)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_measurements_csv(out_path, matrix, angle_names)

    summary_path = Path(summary_csv) if summary_csv else out_path.with_name(out_path.stem + "_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary_path, matrix, angle_names)

    return PipelineOutputs(
        measurements_csv=out_path,
        summary_csv=summary_path,
        n_frames=int(matrix.shape[0]),
        n_measures=int(matrix.shape[1]),
    )
