import csv
from pathlib import Path

from modern.visualization import DEFAULT_MOTION_METRICS, parse_metrics_arg


def test_parse_metrics_default_subset():
    available = ["frame", "F-II-MCP", "F-III-MCP", "A-II-TIP", "X-OTHER"]
    selected = parse_metrics_arg(None, available)
    assert selected == ["F-II-MCP", "F-III-MCP", "A-II-TIP"]


def test_parse_metrics_explicit_and_invalid_filtered():
    available = ["F-II-MCP", "A-II-TIP"]
    selected = parse_metrics_arg("F-II-MCP,NOPE,A-II-TIP", available)
    assert selected == ["F-II-MCP", "A-II-TIP"]


def test_parse_metrics_raises_if_none_valid():
    available = ["F-II-MCP", "A-II-TIP"]
    try:
        parse_metrics_arg("NOPE,ALSO_NO", available)
    except ValueError as exc:
        assert "requested metrics" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid metrics")
