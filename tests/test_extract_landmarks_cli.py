import pytest

from modern.extract_landmarks_cli import parse_args


def test_parse_args_accepts_input_video():
    args = parse_args(["--input-video", "sample.mp4", "--output-csv", "out.csv"])
    assert args.input_video == "sample.mp4"
    assert args.camera_index is None
    assert args.output_csv == "out.csv"


def test_parse_args_accepts_camera_index():
    args = parse_args(["--camera-index", "0", "--output-csv", "out.csv"])
    assert args.camera_index == 0
    assert args.input_video is None
    assert args.output_csv == "out.csv"


def test_parse_args_requires_source():
    with pytest.raises(SystemExit):
        parse_args(["--output-csv", "out.csv"])
