from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modern monocular hand-analysis pipeline (offline v1)."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input landmarks CSV (frame,joint,x,y,z).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output measurements CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path to summary statistics CSV.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-frame normalization by palm width.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
        normalize=not args.no_normalize,
    )
    print(
        f"Processed {outputs.n_frames} frames, "
        f"generated {outputs.n_measures} measures -> {outputs.measurements_csv}"
    )
    print(f"Summary -> {outputs.summary_csv}")


if __name__ == "__main__":
    main()
