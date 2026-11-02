from __future__ import annotations

import argparse
from pathlib import Path

from .visualization import plot_motion_metrics, render_skeleton_overlay_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate motion plot and skeleton overlay video from HandAnalysis outputs."
    )
    parser.add_argument("--input-video", required=True, help="Path to original RGB video.")
    parser.add_argument("--landmarks-csv", required=True, help="Path to landmarks CSV.")
    parser.add_argument("--angles-csv", required=True, help="Path to angles CSV.")
    parser.add_argument(
        "--overlay-video-out",
        default=None,
        help="Output MP4 path for overlay video. Default: <angles_csv_stem>_overlay.mp4",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="Output PNG path for motion plot. Default: <angles_csv_stem>_motion.png",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metrics to plot. Example: F-II-MCP,F-III-MCP,A-II-TIP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    angles_path = Path(args.angles_csv)
    overlay_out = args.overlay_video_out or str(angles_path.with_name(angles_path.stem.replace("_angles", "") + "_overlay.mp4"))
    plot_out = args.plot_out or str(angles_path.with_name(angles_path.stem.replace("_angles", "") + "_motion.png"))

    overlay_path = render_skeleton_overlay_video(
        input_video=args.input_video,
        landmarks_csv=args.landmarks_csv,
        output_video=overlay_out,
    )
    plot_path = plot_motion_metrics(
        angles_csv=args.angles_csv,
        output_png=plot_out,
        metrics=args.metrics,
    )

    print(f"Overlay video -> {overlay_path}")
    print(f"Motion plot   -> {plot_path}")


if __name__ == "__main__":
    main()
