import csv
from pathlib import Path

from modern.hand_kinematics import JOINT_NAMES
from modern.pipeline import run_pipeline


def test_pipeline_generates_output_files(tmp_path: Path):
    in_csv = tmp_path / "landmarks.csv"
    out_csv = tmp_path / "angles.csv"

    with in_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "joint", "x", "y", "z"])
        for frame in [0, 1]:
            for i, joint in enumerate(JOINT_NAMES):
                writer.writerow([frame, joint, i * 0.1 + frame * 0.01, i * 0.2, i * 0.03])

    result = run_pipeline(str(in_csv), str(out_csv))

    assert out_csv.exists()
    assert result.summary_csv.exists()
    assert result.n_frames == 2
    assert result.n_measures > 5
