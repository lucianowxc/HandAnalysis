from __future__ import annotations

import argparse

from .landmarks_extractor import extract_landmarks_from_camera, extract_landmarks_from_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 21 hand landmarks from RGB video to CSV for modern hand-analysis pipeline."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-video", help="Path to input RGB video file.")
    source_group.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera index for webcam capture (e.g., 0 for default camera).",
    )
    parser.add_argument("--output-csv", required=True, help="Path to output landmarks CSV.")
    parser.add_argument(
        "--hand-side",
        default="any",
        choices=["left", "right", "any"],
        help="Select which hand to track when multiple hands are visible.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence for hand detector.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence for hand detector.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for number of frames to process.",
    )
    parser.add_argument(
        "--model-asset-path",
        type=str,
        default=None,
        help="Path to MediaPipe hand landmarker .task model (required for Tasks backend).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.camera_index is not None:
        stats = extract_landmarks_from_camera(
            output_csv=args.output_csv,
            camera_index=args.camera_index,
            hand_side=args.hand_side,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            max_frames=args.max_frames,
            model_asset_path=args.model_asset_path,
        )
        source_label = f"camera:{args.camera_index}"
    else:
        stats = extract_landmarks_from_video(
            input_video=args.input_video,
            output_csv=args.output_csv,
            hand_side=args.hand_side,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            max_frames=args.max_frames,
            model_asset_path=args.model_asset_path,
        )
        source_label = args.input_video

    print(
        f"Source {source_label} | processed {stats.total_frames} frames, detected hand in "
        f"{stats.detected_frames} frames -> {stats.output_csv}"
    )


if __name__ == "__main__":
    main()
