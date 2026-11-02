from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence

import numpy as np

from .hand_kinematics import JOINT_NAMES
from .io import write_landmarks_csv

HandSide = Literal["left", "right", "any"]

# MediaPipe hand landmark indices mapped to this project's joint names.
JOINT_TO_MEDIAPIPE_INDEX: Dict[str, int] = {
    "W": 0,
    "I-CMC": 1,
    "II-MCP": 5,
    "III-MCP": 9,
    "IV-MCP": 13,
    "V-MCP": 17,
    "I-MCP": 2,
    "I-IP": 3,
    "I-TIP": 4,
    "II-PIP": 6,
    "II-DIP": 7,
    "II-TIP": 8,
    "III-PIP": 10,
    "III-DIP": 11,
    "III-TIP": 12,
    "IV-PIP": 14,
    "IV-DIP": 15,
    "IV-TIP": 16,
    "V-PIP": 18,
    "V-DIP": 19,
    "V-TIP": 20,
}


@dataclass(frozen=True)
class ExtractionStats:
    total_frames: int
    detected_frames: int
    output_csv: Path


class MediaPipeHandDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        model_asset_path: str | None = None,
    ) -> None:
        try:
            mp = importlib.import_module("mediapipe")
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for RGB landmark extraction. "
                "Install with: pip install mediapipe"
            ) from exc

        self._mp = mp
        self._backend = "legacy" if hasattr(mp, "solutions") else "tasks"
        self._hands = None
        self._landmarker = None

        if self._backend == "legacy":
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            if not model_asset_path:
                raise RuntimeError(
                    "MediaPipe Tasks backend requires a hand landmarker model (.task). "
                    "Provide --model-asset-path (CLI) or model_asset_path (API)."
                )
            model_path = Path(model_asset_path)
            if not model_path.exists():
                raise RuntimeError(f"Hand landmarker model not found: {model_path}")

            tasks = importlib.import_module("mediapipe.tasks.python")
            vision = importlib.import_module("mediapipe.tasks.python.vision")

            options = vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._backend == "legacy" and self._hands is not None:
            self._hands.close()
        if self._backend == "tasks" and self._landmarker is not None:
            self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> List[Dict[str, object]]:
        cv2 = importlib.import_module("cv2")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hands: List[Dict[str, object]] = []
        if self._backend == "legacy":
            result = self._hands.process(frame_rgb)
            if not result.multi_hand_landmarks:
                return []

            handedness_list = result.multi_handedness or []
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                label = "Unknown"
                score = 0.0
                if idx < len(handedness_list):
                    classification = handedness_list[idx].classification[0]
                    label = classification.label
                    score = float(classification.score)

                coords = np.zeros((len(JOINT_NAMES), 3), dtype=float)
                for j, joint_name in enumerate(JOINT_NAMES):
                    mp_idx = JOINT_TO_MEDIAPIPE_INDEX[joint_name]
                    lm = hand_landmarks.landmark[mp_idx]
                    coords[j, 0] = lm.x
                    coords[j, 1] = lm.y
                    coords[j, 2] = lm.z

                hands.append({"label": label, "score": score, "coords": coords})
            return hands

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int(timestamp_ms if timestamp_ms is not None else 0)
        result = self._landmarker.detect_for_video(mp_image, ts)
        if not result.hand_landmarks:
            return []

        handedness_list = result.handedness or []
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            label = "Unknown"
            score = 0.0
            if idx < len(handedness_list) and handedness_list[idx]:
                category = handedness_list[idx][0]
                label = str(getattr(category, "category_name", "Unknown"))
                score = float(getattr(category, "score", 0.0))

            coords = np.zeros((len(JOINT_NAMES), 3), dtype=float)
            for j, joint_name in enumerate(JOINT_NAMES):
                mp_idx = JOINT_TO_MEDIAPIPE_INDEX[joint_name]
                lm = hand_landmarks[mp_idx]
                coords[j, 0] = lm.x
                coords[j, 1] = lm.y
                coords[j, 2] = lm.z
            hands.append({"label": label, "score": score, "coords": coords})
        return hands


def _normalize_handside(side: str) -> HandSide:
    normalized = side.strip().lower()
    if normalized in {"left", "right", "any"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("hand_side must be one of: left, right, any")


def _select_hand(hands: Sequence[Dict[str, object]], hand_side: HandSide) -> Dict[str, object] | None:
    if not hands:
        return None

    if hand_side == "any":
        return max(hands, key=lambda h: float(h["score"]))

    selected = [h for h in hands if str(h["label"]).lower() == hand_side]
    if not selected:
        return None
    return max(selected, key=lambda h: float(h["score"]))


def _as_pixel_relative_depth(coords_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros_like(coords_norm, dtype=float)
    out[:, 0] = coords_norm[:, 0] * float(width)
    out[:, 1] = coords_norm[:, 1] * float(height)
    out[:, 2] = coords_norm[:, 2] * float(width)
    return out


def extract_landmarks_from_video(
    input_video: str,
    output_csv: str,
    hand_side: str = "any",
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_frames: int | None = None,
    model_asset_path: str | None = None,
) -> ExtractionStats:
    cv2 = importlib.import_module("cv2")

    normalized_side = _normalize_handside(hand_side)
    detector = MediaPipeHandDetector(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_asset_path=model_asset_path,
    )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        detector.close()
        raise ValueError(f"Cannot open input video: {input_video}")

    output_rows: List[Dict[str, object]] = []
    total_frames = 0
    detected_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            if max_frames is not None and total_frames > max_frames:
                break

            h, w = frame.shape[:2]
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_ms = int(pos_msec) if pos_msec and pos_msec > 0 else int((total_frames - 1) * 33)
            hands = detector.detect(frame, timestamp_ms=timestamp_ms)
            selected = _select_hand(hands, normalized_side)
            if selected is None:
                continue

            detected_frames += 1
            coords_norm = selected["coords"]
            assert isinstance(coords_norm, np.ndarray)
            coords = _as_pixel_relative_depth(coords_norm, width=w, height=h)
            confidence = float(selected["score"])
            handedness = str(selected["label"]).lower()

            for j, joint_name in enumerate(JOINT_NAMES):
                output_rows.append(
                    {
                        "frame": total_frames - 1,
                        "joint": joint_name,
                        "x": float(coords[j, 0]),
                        "y": float(coords[j, 1]),
                        "z": float(coords[j, 2]),
                        "confidence": confidence,
                        "handedness": handedness,
                    }
                )
    finally:
        cap.release()
        detector.close()

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_landmarks_csv(out_path, output_rows)

    return ExtractionStats(
        total_frames=total_frames,
        detected_frames=detected_frames,
        output_csv=out_path,
    )
