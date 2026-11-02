import csv

import numpy as np

from modern.hand_kinematics import JOINT_NAMES
from modern.landmarks_extractor import (
    JOINT_TO_MEDIAPIPE_INDEX,
    _as_pixel_relative_depth,
    _normalize_handside,
    _select_hand,
)
from modern.io import write_landmarks_csv


def test_joint_mapping_covers_all_project_joints():
    assert set(JOINT_TO_MEDIAPIPE_INDEX.keys()) == set(JOINT_NAMES)
    assert JOINT_TO_MEDIAPIPE_INDEX["W"] == 0
    assert JOINT_TO_MEDIAPIPE_INDEX["V-TIP"] == 20


def test_normalize_handside_accepts_expected_values():
    assert _normalize_handside("left") == "left"
    assert _normalize_handside("RIGHT") == "right"
    assert _normalize_handside("any") == "any"


def test_select_hand_filters_and_picks_best_score():
    hands = [
        {"label": "Left", "score": 0.4, "coords": np.zeros((21, 3))},
        {"label": "Left", "score": 0.9, "coords": np.zeros((21, 3))},
        {"label": "Right", "score": 0.95, "coords": np.zeros((21, 3))},
    ]
    selected_left = _select_hand(hands, "left")
    assert selected_left is not None
    assert float(selected_left["score"]) == 0.9

    selected_any = _select_hand(hands, "any")
    assert selected_any is not None
    assert float(selected_any["score"]) == 0.95


def test_relative_depth_conversion_uses_frame_size():
    coords_norm = np.array([[0.5, 0.25, -0.1]])
    out = _as_pixel_relative_depth(coords_norm, width=640, height=480)
    assert np.allclose(out[0], np.array([320.0, 120.0, -64.0]))


def test_write_landmarks_csv_generates_expected_columns(tmp_path):
    out_csv = tmp_path / "landmarks.csv"
    rows = [
        {
            "frame": 0,
            "joint": "W",
            "x": 100.0,
            "y": 120.0,
            "z": -3.5,
            "confidence": 0.98,
            "handedness": "right",
        }
    ]
    write_landmarks_csv(out_csv, rows)

    with out_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)

    assert reader.fieldnames == ["frame", "joint", "x", "y", "z", "confidence", "handedness"]
    assert len(read_rows) == 1
    assert read_rows[0]["joint"] == "W"
