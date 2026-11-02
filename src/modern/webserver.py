from __future__ import annotations

import importlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

from .landmarks_extractor import extract_landmarks_from_video
from .pipeline import run_pipeline
from .visualization import plot_motion_metrics, render_skeleton_overlay_video

RunKind = Literal["landmarks", "angles", "summary", "overlay", "motion"]

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = BASE_DIR / "web_runs"

_fastapi = importlib.import_module("fastapi")
_fastapi_responses = importlib.import_module("fastapi.responses")
_fastapi_cors = importlib.import_module("fastapi.middleware.cors")

FastAPI = _fastapi.FastAPI
File = _fastapi.File
Form = _fastapi.Form
HTTPException = _fastapi.HTTPException
UploadFile = _fastapi.UploadFile
FileResponse = _fastapi_responses.FileResponse
HTMLResponse = _fastapi_responses.HTMLResponse
CORSMiddleware = _fastapi_cors.CORSMiddleware

app = FastAPI(
    title="HandAnalysis Modern API",
    version="0.1.0",
    description="Web API para pipeline monocular de análise de mão.",
)

_origins_env = os.getenv("HANDANALYSIS_CORS_ALLOW_ORIGINS", "*")
_origins = [x.strip() for x in _origins_env.split(",") if x.strip()]
if _origins == ["*"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>HandAnalysis Modern Web UI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
        .container { max-width: 980px; margin: 0 auto; padding: 24px; }
        .card { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
        h1 { margin-top: 0; }
        h2 { margin-top: 0; font-size: 1.1rem; }
        label { display: block; margin-top: 10px; font-size: 0.95rem; }
        input, select, button { margin-top: 6px; padding: 8px; border-radius: 8px; border: 1px solid #475569; background: #0b1220; color: #e2e8f0; }
        button { cursor: pointer; background: #2563eb; border: none; }
        button:hover { background: #1d4ed8; }
        .muted { color: #94a3b8; font-size: 0.9rem; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .result { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #020617; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; }
    .preview { margin-top: 12px; }
    .preview video, .preview img { width: 100%; border-radius: 8px; border: 1px solid #1e293b; margin-top: 8px; }
    .links a { display: block; margin-top: 4px; }
        a { color: #60a5fa; }
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>HandAnalysis Modern</h1>
        <p class=\"muted\">Upload CSV de landmarks ou vídeo RGB para gerar medidas clínicas de mão.</p>

        <div class=\"card\">
            <h2>1) Analisar CSV de landmarks</h2>
            <form id=\"csvForm\">
                <label>Arquivo CSV
                    <input type=\"file\" name=\"file\" accept=\".csv\" required />
                </label>
                <label>
                    <input type=\"checkbox\" name=\"normalize\" checked /> Normalizar por largura da palma
                </label>
                <button type=\"submit\">Executar análise CSV</button>
            </form>
        </div>

        <div class=\"card\">
            <h2>2) Analisar vídeo RGB</h2>
            <form id=\"videoForm\">
                <label>Arquivo de vídeo
                    <input type=\"file\" name=\"file\" accept=\".mp4,.mov,.avi,.mkv\" required />
                </label>
                <div class=\"row\">
                    <label>Lado da mão
                        <select name=\"hand_side\">
                            <option value=\"any\" selected>any</option>
                            <option value=\"left\">left</option>
                            <option value=\"right\">right</option>
                        </select>
                    </label>
                    <label>Máximo de frames (opcional)
                        <input type=\"number\" name=\"max_frames\" min=\"1\" placeholder=\"ex: 300\" />
                    </label>
                </div>
                <div class=\"row\">
                    <label>Min detection confidence
                        <input type=\"number\" name=\"min_detection_confidence\" step=\"0.05\" min=\"0\" max=\"1\" value=\"0.5\" />
                    </label>
                    <label>Min tracking confidence
                        <input type=\"number\" name=\"min_tracking_confidence\" step=\"0.05\" min=\"0\" max=\"1\" value=\"0.5\" />
                    </label>
                </div>
                <label>
                    <input type=\"checkbox\" name=\"normalize\" checked /> Normalizar por largura da palma
                </label>
                <button type=\"submit\">Executar análise de vídeo</button>
            </form>
            <p class=\"muted\">Esse endpoint requer extras de visão instalados (`opencv-python` + `mediapipe`).</p>
        </div>

        <div class=\"card\">
            <h2>Resultados</h2>
            <div id=\"result\" class=\"result\">Aguardando execução...</div>
            <div id=\"downloads\" class=\"links preview\"></div>
            <div id=\"preview\" class=\"preview\"></div>
            <p class=\"muted\">Você também pode usar <a href=\"/docs\">/docs</a> para testar a API.</p>
        </div>
    </div>

    <script>
    const resultEl = document.getElementById('result');
    const downloadsEl = document.getElementById('downloads');
    const previewEl = document.getElementById('preview');

        function renderSuccess(data) {
            const files = data.files || {};
            resultEl.textContent = JSON.stringify(data, null, 2);

            downloadsEl.innerHTML = '<h3>Downloads</h3>' + Object.entries(files)
                .map(([k, v]) => `<a href="${v}" target="_blank" rel="noopener">${k}: ${window.location.origin}${v}</a>`)
                .join('');

            let previewHtml = '';
            if (files.overlay) {
                previewHtml += `<h3>Overlay (esqueleto)</h3><video controls src="${files.overlay}"></video>`;
            }
            if (files.motion) {
                previewHtml += `<h3>Gráfico de movimento</h3><img alt="motion plot" src="${files.motion}" />`;
            }
            previewEl.innerHTML = previewHtml;
        }

        function renderError(err) {
            resultEl.textContent = `Erro: ${err}`;
            downloadsEl.innerHTML = '';
            previewEl.innerHTML = '';
        }

        async function postForm(form, url) {
            resultEl.textContent = 'Processando...';
            const formData = new FormData(form);

            const normalizeChecked = form.querySelector('input[name="normalize"]')?.checked;
            formData.set('normalize', normalizeChecked ? 'true' : 'false');

            const maxFrames = formData.get('max_frames');
            if (maxFrames === '') {
                formData.delete('max_frames');
            }

            const resp = await fetch(url, { method: 'POST', body: formData });
            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data?.detail || JSON.stringify(data));
            }
            return data;
        }

        document.getElementById('csvForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const data = await postForm(e.target, '/analyze/csv');
                renderSuccess(data);
            } catch (err) {
                renderError(err);
            }
        });

        document.getElementById('videoForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const data = await postForm(e.target, '/analyze/video');
                renderSuccess(data);
            } catch (err) {
                renderError(err);
            }
        });
    </script>
</body>
</html>
"""


def _run_dir(run_id: str) -> Path:
    return DEFAULT_RUNS_DIR / run_id


def _ensure_csv(filename: str) -> None:
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expected a .csv file")


def _ensure_video(filename: str) -> None:
    valid = (".mp4", ".mov", ".avi", ".mkv")
    if not filename.lower().endswith(valid):
        raise HTTPException(status_code=400, detail=f"Expected one of: {', '.join(valid)}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/analyze/csv")
def analyze_from_csv(
    file: UploadFile = File(...),
    normalize: bool = Form(True),
) -> dict[str, object]:
    _ensure_csv(file.filename or "")

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    input_csv = run_dir / "landmarks.csv"
    angles_csv = run_dir / "angles.csv"
    summary_csv = run_dir / "angles_summary.csv"
    motion_png = run_dir / "motion.png"
    warnings: list[str] = []

    with input_csv.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        outputs = run_pipeline(
            input_csv=str(input_csv),
            output_csv=str(angles_csv),
            summary_csv=str(summary_csv),
            normalize=normalize,
        )
        try:
            plot_motion_metrics(
                angles_csv=str(angles_csv),
                output_png=str(motion_png),
            )
        except Exception as viz_exc:  # noqa: BLE001
            warnings.append(f"motion_plot_unavailable: {viz_exc}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to analyze CSV: {exc}") from exc

    response: dict[str, object] = {
        "run_id": run_id,
        "n_frames": outputs.n_frames,
        "n_measures": outputs.n_measures,
        "files": {
            "landmarks": f"/runs/{run_id}/landmarks",
            "angles": f"/runs/{run_id}/angles",
            "summary": f"/runs/{run_id}/summary",
        },
    }
    if motion_png.exists():
        response["files"]["motion"] = f"/runs/{run_id}/motion"  # type: ignore[index]
    if warnings:
        response["warnings"] = warnings
    return response


@app.post("/analyze/video")
def analyze_from_video(
    file: UploadFile = File(...),
    hand_side: str = Form("any"),
    normalize: bool = Form(True),
    min_detection_confidence: float = Form(0.5),
    min_tracking_confidence: float = Form(0.5),
    max_frames: int | None = Form(None),
    model_asset_path: str | None = Form(None),
) -> dict[str, object]:
    _ensure_video(file.filename or "")

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    input_video = run_dir / "input_video" / (file.filename or "video.mp4")
    input_video.parent.mkdir(parents=True, exist_ok=True)
    landmarks_csv = run_dir / "landmarks.csv"
    angles_csv = run_dir / "angles.csv"
    summary_csv = run_dir / "angles_summary.csv"
    overlay_mp4 = run_dir / "overlay.mp4"
    motion_png = run_dir / "motion.png"
    warnings: list[str] = []

    with input_video.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        extraction = extract_landmarks_from_video(
            input_video=str(input_video),
            output_csv=str(landmarks_csv),
            hand_side=hand_side,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_frames=max_frames,
            model_asset_path=model_asset_path,
        )
        outputs = run_pipeline(
            input_csv=str(landmarks_csv),
            output_csv=str(angles_csv),
            summary_csv=str(summary_csv),
            normalize=normalize,
        )
        try:
            render_skeleton_overlay_video(
                input_video=str(input_video),
                landmarks_csv=str(landmarks_csv),
                output_video=str(overlay_mp4),
            )
        except Exception as viz_exc:  # noqa: BLE001
            warnings.append(f"overlay_unavailable: {viz_exc}")

        try:
            plot_motion_metrics(
                angles_csv=str(angles_csv),
                output_png=str(motion_png),
            )
        except Exception as viz_exc:  # noqa: BLE001
            warnings.append(f"motion_plot_unavailable: {viz_exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to analyze video: {exc}") from exc

    response: dict[str, object] = {
        "run_id": run_id,
        "total_frames": extraction.total_frames,
        "detected_frames": extraction.detected_frames,
        "n_frames": outputs.n_frames,
        "n_measures": outputs.n_measures,
        "files": {
            "landmarks": f"/runs/{run_id}/landmarks",
            "angles": f"/runs/{run_id}/angles",
            "summary": f"/runs/{run_id}/summary",
        },
    }
    if overlay_mp4.exists():
        response["files"]["overlay"] = f"/runs/{run_id}/overlay"  # type: ignore[index]
    if motion_png.exists():
        response["files"]["motion"] = f"/runs/{run_id}/motion"  # type: ignore[index]
    if warnings:
        response["warnings"] = warnings
    return response


@app.get("/runs/{run_id}/{kind}")
def download_result(run_id: str, kind: RunKind) -> FileResponse:
    run_dir = _run_dir(run_id)
    mapping = {
        "landmarks": run_dir / "landmarks.csv",
        "angles": run_dir / "angles.csv",
        "summary": run_dir / "angles_summary.csv",
        "overlay": run_dir / "overlay.mp4",
        "motion": run_dir / "motion.png",
    }
    target = mapping[kind]
    if not target.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    media_type = {
        ".csv": "text/csv",
        ".mp4": "video/mp4",
        ".png": "image/png",
    }.get(target.suffix.lower(), "application/octet-stream")

    return FileResponse(path=target, filename=target.name, media_type=media_type)


def main() -> None:
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run web server. Install with: pip install uvicorn") from exc

    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    host = os.getenv("HANDANALYSIS_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("HANDANALYSIS_PORT", "8000")))
    uvicorn.run("modern.webserver:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
