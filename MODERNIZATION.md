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
- `tests/`: testes unitários e smoke test
- `pyproject.toml`: configuração moderna de pacote e testes

## Formato do CSV de entrada

CSV longo com colunas:

- `frame` (int)
- `joint` (nome de junta: `W`, `I-CMC`, ..., `V-TIP`)
- `x`, `y`, `z` (float)

## Uso rápido

```bash
python -m modern.cli --input-csv data/landmarks.csv --output-csv out/angles.csv
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
