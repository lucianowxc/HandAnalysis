import numpy as np

from modern.hand_kinematics import (
    ANGLE_NAMES,
    JOINT_NAMES,
    compute_measure_matrix,
    normalize_frame,
    skeleton_measures,
)


def _synthetic_open_hand() -> np.ndarray:
    # 21x3 synthetic pose with no collinearity issues.
    pts = np.array(
        [
            [0.0, 0.0, 0.0],    # W
            [-0.8, -0.3, 0.1],  # I-CMC
            [-0.4, 0.8, 0.0],   # II-MCP
            [0.0, 0.9, 0.0],    # III-MCP
            [0.4, 0.8, 0.0],    # IV-MCP
            [0.8, 0.6, 0.0],    # V-MCP
            [-1.1, 0.1, 0.1],   # I-MCP
            [-1.4, 0.5, 0.15],  # I-IP
            [-1.7, 0.9, 0.2],   # I-TIP
            [-0.5, 1.3, 0.05],  # II-PIP
            [-0.55, 1.75, 0.1], # II-DIP
            [-0.6, 2.1, 0.1],   # II-TIP
            [0.0, 1.45, 0.05],  # III-PIP
            [0.0, 1.9, 0.1],    # III-DIP
            [0.0, 2.3, 0.1],    # III-TIP
            [0.45, 1.35, 0.05], # IV-PIP
            [0.5, 1.75, 0.1],   # IV-DIP
            [0.55, 2.1, 0.12],  # IV-TIP
            [0.85, 1.1, 0.05],  # V-PIP
            [1.0, 1.45, 0.1],   # V-DIP
            [1.15, 1.75, 0.12], # V-TIP
        ],
        dtype=float,
    )
    assert pts.shape == (len(JOINT_NAMES), 3)
    return pts


def test_normalize_frame_centers_wrist():
    frame = _synthetic_open_hand() + np.array([20.0, -3.0, 7.0])
    normalized = normalize_frame(frame)
    assert np.allclose(normalized[0], np.zeros(3))


def test_skeleton_measures_has_all_expected_metrics():
    frame = normalize_frame(_synthetic_open_hand())
    measures = skeleton_measures(frame)
    assert set(measures) == set(ANGLE_NAMES)
    for value in measures.values():
        assert np.isfinite(value)


def test_compute_measure_matrix_shape():
    frame_a = _synthetic_open_hand()
    frame_b = _synthetic_open_hand() * np.array([1.02, 0.98, 1.01])
    frames = np.stack([frame_a, frame_b], axis=0)
    matrix, names = compute_measure_matrix(frames, normalize=True)
    assert matrix.shape == (2, len(ANGLE_NAMES))
    assert names == ANGLE_NAMES
