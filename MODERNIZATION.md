# HandAnalysis Modernization Track (v2)

Este documento adiciona uma trilha moderna em **Python 3** para evoluir o projeto da tese sem quebrar o baseline original.

## Objetivo

Criar um pipeline monocular offline (`v1`) que:

1. lê landmarks de mão por frame (`x, y, z relativo`),
2. reconstrói uma pose 3D normalizada por mão,
3. calcula medidas clínicas (flexão, oposição e distâncias entre pontas dos dedos),
4. exporta séries temporais e resumo estatístico.

## Estrutura adicionada

- `src/modern/hand_kinematics.py`: geometria e cálculo de medidas
- `src/modern/io.py`: leitura/escrita de CSV
- `src/modern/pipeline.py`: orquestração do processamento
- `src/modern/cli.py`: runner CLI
- `src/modern/landmarks_extractor.py`: extração de landmarks 21-juntas de vídeo RGB
- `src/modern/extract_landmarks_cli.py`: CLI para gerar CSV de landmarks
- `tests/`: testes unitários e smoke test
- `pyproject.toml`: configuração moderna de pacote e testes

## Formato do CSV de entrada

CSV longo com colunas:

- `frame` (int)
- `joint` (nome de junta: `W`, `I-CMC`, ..., `V-TIP`)
- `x`, `y`, `z` (float)

## Uso rápido

### 1) Extrair landmarks de vídeo RGB

```bash
python -m modern.extract_landmarks_cli \
	--input-video data/hand_video.mp4 \
	--output-csv data/landmarks.csv \
	--hand-side any \
	--model-asset-path models/hand_landmarker.task
```

> Dependências opcionais para esta etapa: `pip install -e .[vision]`

> Na API mais nova do MediaPipe (Tasks), o modelo `.task` de mão é obrigatório.

No backend, se `model_asset_path` não for enviado, o sistema tenta resolver automaticamente nesta ordem:

1. `HANDANALYSIS_MEDIAPIPE_MODEL_PATH` (env)
2. `models/hand_landmarker.task` no projeto
3. download automático do modelo oficial (se `HANDANALYSIS_AUTO_DOWNLOAD_MODEL=true`)

### 1.1) Extrair landmarks direto da webcam

```bash
python -m modern.extract_landmarks_cli \
	--camera-index 0 \
	--output-csv data/landmarks_camera.csv \
	--max-frames 600 \
	--model-asset-path models/hand_landmarker.task
```

> Use `--camera-index 1` (ou outro valor) se a câmera padrão não for a desejada.

### 2) Calcular medidas clínicas a partir dos landmarks

```bash
python -m modern.cli --input-csv data/landmarks.csv --output-csv out/angles.csv
```

### 2.1) Visualizar movimento (gráfico + esqueleto sobreposto)

```bash
python -m modern.visualize_cli \
	--input-video data/hand_video.mp4 \
	--landmarks-csv data/landmarks.csv \
	--angles-csv out/angles.csv
```

Saídas:

- `<stem>_overlay.mp4` com esqueleto sobre o vídeo
- `<stem>_motion.png` com séries temporais (flexão/abdução por padrão)

### 3) Subir webserver para uso fácil

```bash
pip install -e .[web]
python -m modern.webserver
```

Para uso com frontend em domínio diferente (ex.: GitHub Pages), configure CORS:

```bash
export HANDANALYSIS_CORS_ALLOW_ORIGINS="https://<usuario>.github.io"
python -m modern.webserver
```

API disponível em `http://localhost:8000`.

Interface web amigável disponível em `http://localhost:8000/`.

Na interface (`/`), após cada análise, você verá:

- links de download dos resultados;
- preview embutido do vídeo com esqueleto (quando disponível);
- preview do gráfico temporal de flexão/abdução.

- `GET /health` — status da API
- `POST /analyze/csv` — upload de `landmarks.csv`, retorna links dos resultados
- `POST /analyze/video` — upload de vídeo RGB (`.mp4`, `.mov`, `.avi`, `.mkv`), roda extração + análise
- `POST /analyze/video` — upload de vídeo RGB (`.mp4`, `.mov`, `.avi`, `.mkv`), roda extração + análise (aceita `model_asset_path`)
- `GET /runs/{run_id}/{kind}` — download de resultado (`landmarks|angles|summary`)

Fluxo sugerido para uso clínico rápido:

1. Abrir `http://localhost:8000/` no navegador.
2. Fazer upload do CSV de landmarks ou vídeo.
3. Baixar `angles.csv` e `angles_summary.csv` pelos links gerados.

> Para usar `POST /analyze/video`, instale também os extras de visão:

```bash
pip install -e .[vision]
```

### 4) Publicar frontend no GitHub Pages

O frontend estático está em `docs/` e é publicado pelo workflow `.github/workflows/pages.yml`.

Passos:

1. No repositório do GitHub, abra **Settings → Pages**.
2. Em **Build and deployment**, selecione **GitHub Actions**.
3. Faça push de mudanças em `docs/` ou rode manualmente o workflow.
4. Acesse a URL publicada e configure o campo **API base URL** com a URL do backend FastAPI.

> Importante: GitHub Pages é estático; o backend Python precisa rodar em host externo (Render/Railway/Fly/VM).

### 5) Publicar backend FastAPI (produção)

#### Opção A — Render (recomendada)

O repositório já inclui `render.yaml` com configuração padrão.

1. No Render, clique em **New + → Blueprint**.
2. Conecte este repositório.
3. Confirme o serviço `handanalysis-backend`.
4. Após deploy, copie a URL pública (ex.: `https://handanalysis-backend.onrender.com`).
5. No GitHub Pages, preencha essa URL no campo **API base URL**.

Variável importante:

- `HANDANALYSIS_CORS_ALLOW_ORIGINS=https://lucianowxc.github.io`
- `HANDANALYSIS_MEDIAPIPE_MODEL_PATH=/opt/render/project/src/models/hand_landmarker.task`

Se usar domínio customizado no frontend, inclua-o na variável, separado por vírgulas.

#### Opção B — Docker (qualquer VM/cloud)

Arquivos prontos:

- `Dockerfile`
- `.dockerignore`

Build e execução local:

```bash
docker build -t handanalysis-backend .
docker run --rm -p 8000:8000 \
	-e HANDANALYSIS_CORS_ALLOW_ORIGINS="https://lucianowxc.github.io" \
	handanalysis-backend
```

Health check:

```bash
curl http://localhost:8000/health
```

Arquivos de saída:

- `out/angles.csv` (série temporal por frame)
- `out/angles_summary.csv` (média, desvio, min, max por medida)

## Etapas de evolução (v1 -> v3)

### v1 — Monocular offline (implementado)

- Pipeline de análise usando landmarks de entrada
- Medidas clínicas em lote
- Exportação tabular reprodutível

### v2 — Validação clínica automatizada

- Extração automática RGB -> CSV (implementado)
- Comparador `monocular vs depth-baseline`
- Métricas por articulação (MAE angular, jitter, taxa de falha)
- Relatório por sessão/paciente

### v3 — Deploy para câmera comum

- Execução em tempo real (webcam/celular)
- Filtro temporal robusto (OneEuro/Kalman)
- Relatório clínico orientado a acompanhamento longitudinal

## Notas

- Esta trilha é **paralela** ao legado e não altera os scripts de 2021.
- A normalização por largura da palma reduz dependência de escala absoluta do monocular.
- Para uso clínico final, recomenda-se calibração de escala e validação contra baseline depth.
