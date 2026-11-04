# TrueLens Backend

FastAPI image-forensics service. See the [root README](../README.md) for architecture and
[REPORT.md](../REPORT.md) for method and limitations.

## Run locally

```bash
uv sync --extra dev --extra redis --system-certs
uv run uvicorn app.main:app --reload --port 8000
```

## Test

```bash
uv run pytest -q                     # fast suite (in-process analysers)
uv run pytest -q -m pool             # real spawn ProcessPool (catches pickling regressions)
```

## Layout

| Path | Responsibility |
|---|---|
| `app/analysis/techniques/` | One module per technique. Each exports `run(...) -> (score, visual, details)`. |
| `app/analysis/worker.py` | The process boundary. Minimal import closure — never import torch here. |
| `app/core/orchestrator.py` | Fan-out, progress publishing, deadlines, cancellation, pool recovery. |
| `app/core/pipeline.py` | Decode, validate, build the two-resolution-track `JobArrays`. |
| `app/ml/` | CNN runtime, Grad-CAM, YuNet face detector, CLIP probe. |
| `app/schemas/` | Wire models for the API response. |
| `scripts/` | Fusion export/training, warm-up probe. |
