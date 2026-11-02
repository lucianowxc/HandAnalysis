import csv
import importlib
import io

import pytest

pytest.importorskip("fastapi")
TestClient = importlib.import_module("fastapi.testclient").TestClient

from modern.hand_kinematics import JOINT_NAMES
from modern.webserver import app


def _landmarks_csv_bytes() -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["frame", "joint", "x", "y", "z"])
    for i, joint in enumerate(JOINT_NAMES):
        writer.writerow([0, joint, 10.0 + i, 20.0 + i * 0.5, i * 0.1])
    return stream.getvalue().encode("utf-8")


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_page_available():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "HandAnalysis Modern" in response.text
    assert "Analisar CSV de landmarks" in response.text
    assert "Analisar vídeo RGB" in response.text


def test_analyze_csv_endpoint_and_download():
    client = TestClient(app)
    payload = _landmarks_csv_bytes()

    response = client.post(
        "/analyze/csv",
        files={"file": ("landmarks.csv", payload, "text/csv")},
        data={"normalize": "true"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["n_frames"] == 1
    assert data["n_measures"] > 10

    angles_resp = client.get(data["files"]["angles"])
    summary_resp = client.get(data["files"]["summary"])
    assert angles_resp.status_code == 200
    assert summary_resp.status_code == 200
    assert "frame" in angles_resp.text.splitlines()[0]
    assert "metric" in summary_resp.text.splitlines()[0]
