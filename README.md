# TrueLens

An image-authenticity workbench: upload a photograph and get two probabilities — that it was
generated, that it was manipulated — each with the evidence that produced it, drawn on the image.

It answers "what does this look like it was made by, and how sure are we" and never "this is fake".
No forensic tool can prove an image is authentic; a negative would require ruling out every
manipulation method that exists and every one that does not yet. So every number here is a
likelihood, every likelihood carries a reliability figure beside it, and every page says what the
analysis cannot tell you.

```
   browser ─────────────────────────────┐
      │  static site (Vercel / web image)│
      ▼                                  ▼
 ┌──────────────┐  POST /api/analyze   ┌───────────────────────────────────────────┐
 │  Next.js 16  │─────────────────────▶│  FastAPI (uvicorn, 1 worker)              │
 │  App Router  │◀───── job_id ────────│                                           │
 │  React 19    │                      │  security.py   magic bytes, size, format  │
 └──────┬───────┘                      │  pipeline.py   decode → 3 resolution      │
        │                              │                tracks (analysis / native / │
        │ GET /api/jobs/{id}           │                container)                  │
        │  progress + result           │  orchestrator.py  fan-out, deadlines,      │
        ▼                              │                     progress, cancellation │
 ┌──────────────┐                      └───────┬───────────────────────────┬───────┘
 │  overlays    │◀── signed URLs ──────────────│                           │
 │  PNG, 10 min │        /api/overlays/…       │ parent process            │ spawn pool
 └──────────────┘                              │  torch: CNN + Grad-CAM    │  8 workers, 1 thread each
                                               │  loaded once at startup   │  cv2 / scipy / FFT
        ┌──────────────────────────────────────┴───────────────┬────────────────┐
        │  10 analysers, each → (score, visual, details)       │                │
        │  metadata  c2pa  ela  jpeg_ghost  block_grid  noise  │  registry.py   │
        │  copy_move  face  frequency  cnn                     │  the catalogue │
        └───────────────────────┬──────────────────────────────┴────────────────┘
                                ▼
              features.py → 48-column vector  →  fusion.py
                              (learned heads, or the labelled expert prior)
                                ▼
              reliability.py  × penalties, never a subtraction
                                ▼
              verdict.py      two sub-scores, flags, reasons, limitations
                                ▼
              JSON result → gauge, viewer, cards, metadata panel, PDF
```

## Run it

Two terminals, or `make dev` where GNU make exists.

```bash
# 1. API  (http://127.0.0.1:8010/api/health)
cd backend && uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8010

# 2. Web  (http://localhost:3000)
cd frontend && npm ci
echo 'NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8010' > .env.local
npm run dev
```

`NEXT_PUBLIC_*` is inlined into the browser bundle at build time, so it must exist before `dev` or
`build` starts — a value added later is silently absent in the bundle that is already running.

On Windows, `8000` may already belong to another service, which is why the local instructions use
`8010` and `frontend/.env.local` ships with that address. If npm hangs on a TLS handshake on a
machine with an intercepting proxy, run node with `NODE_OPTIONS=--use-system-ca`.

Verify the product rather than trusting this file:

```bash
cd backend
uv run pytest -q                      # 466 tests: analysers, fusion, verdict, API, ML contracts
uv run pytest -q -m pool              # the real spawn pool, which is where pickling regressions live
uv run python scripts/smoke_techniques.py   # each analyser responds to the artefact it claims to detect
uv run python scripts/smoke_fusion.py        # ten scores fuse into the verdict the fixtures imply
uv run python scripts/smoke_api.py           # a multipart upload becomes a job becomes a schema
```

## The pages

| Route | What it is for |
|---|---|
| `/` | What the product does, inline drop, the technique grid, promises and refusals, FAQ |
| `/analyze` | Drop, paste (`Ctrl+V`) or import by URL; live per-technique progress; verdict, reliability, interactive overlay viewer, per-technique cards, metadata |
| `/report/{job}` | The same analysis laid out for printing and for `GET /api/report/{job}` — a server-rendered PDF, not a screenshot of the DOM |
| `/methodology` | Each check's diagram, the fusion weights, and the trained model's confusion matrix, ROC, PR, calibration and robustness — read from `metrics.json`, not restated |
| `/batch` | Up to 20 files, one table, sortable, CSV export |
| `/history` | Past analyses, kept in this browser's IndexedDB and nowhere else |
| `/about` | Privacy, limitations, source of the numbers |

## How a number is produced

Each analyser returns `(score, visual, details)` — the prototype's contract, kept. `score` is
`None` when the check had nothing to measure, and that is the most important rule in the system:
**`0.0` means "measured, found nothing"; `NaN`/`null` means "could not measure"**. Collapsing them is
how a forensics tool quietly decides that every PNG is authentic, because ELA cannot run on a PNG.
The distinction survives to the wire, the UI, the CSV and the PDF.

1. **Three resolution tracks.** `analysis` (≤2048 px, resampled) for learned and spatial checks,
   `native` (8-aligned, never resampled) for the checks that read compression artefacts — resampling
   destroys exactly what they measure — and `container` for metadata and C2PA.
2. **`features.py`** builds a 48-column vector: two columns per technique (its score and whether the
   score means anything), explanation columns for *why*, and file context (dimensions, bits-per-pixel,
   container, estimated JPEG quality). One definition, imported by both the trainer and the server,
   because a saved model cannot notice that column 14 changed meaning.
3. **`fusion.py`** either scores that vector with two calibrated gradient-boosted heads
   (`fusion.joblib`) or falls back to the documented noisy-OR prior, and says which it used in
   `fusion.source`. The hand-tuned weights and the `fused * 1.6` stretch are gone. Deterministic
   findings — a signed C2PA generative manifest, an identified generator tag — stay as rules that can
   *floor* a score, so a fact is not outvoted by statistics.
4. **`reliability.py`** multiplies penalties: a check that timed out costs less than one that could
   not run. It is separate from likelihood on purpose. `Likely authentic` at 30% reliability is a
   different answer than the same verdict at 85%, and a single blended number would hide that.
5. **`verdict.py`** turns the two sub-scores into one of four states plus reasons, flags and
   limitations. `overall` is `max(ai, edited)`, not an average: a file can be neither, and averaging
   two "no"s into a low number is not the same claim as measuring them.

## Artifacts, and what happens without them

`artifacts/` ships empty on purpose. The only data this repository can prove anything about is
synthetic constructions whose forgery mechanism is known by definition, and a confident detector
trained on those would corrupt every verdict while *raising* apparent reliability.

| File | Produced by | Absent behaviour |
|---|---|---|
| `artifacts/model.pt` | `ml/scripts/train_cnn.py` | CNN reports unavailable → `cnn.*` columns NaN → reliability penalised |
| `artifacts/fusion.joblib` | `ml/scripts/calibrate_fusion.py` | labelled expert prior, and `fusion.source = "expert"` in every response |
| `artifacts/metrics.json` | `ml/scripts/evaluate_cnn.py` | Methodology page shows its "not trained yet" panel |
| `artifacts/clip_probe.joblib` | `ml/scripts/train_clip_probe.py` | probe disabled, `cnn.clip_score` NaN |

[`ml/README.md`](ml/README.md) has the corpus layout, the dataset list with its licensing cautions,
and the exact commands. `make ml-smoke` runs the whole pipeline against generated data in a few
minutes, and `ml/scripts/check_artifacts.py` proves each produced file satisfies the contract the
running service expects — the failures there are the ones that never raise.

## Security and privacy

- Magic bytes decide the format, never the extension; decoded-pixel ceiling before allocation
  (`MAX_ANALYSIS_PIXELS`), so a 200-megapixel file is refused rather than fatal.
- Uploads, decoded arrays and overlays live in a scratch directory that is swept with the job — at
  most ten minutes, and in Docker that directory is a `tmpfs`. Nothing is persisted.
- Overlay URLs are HMAC-signed over path and expiry (`OVERLAY_SIGNING_KEY`), so a job id alone is not
  a way to read someone else's heatmaps.
- Rate limits per address, CORS locked to the frontend origin, no analytics of image content, and
  `REDACT_PII` strips camera location and personal names from the *returned* metadata while the
  analysed copy keeps everything — EXIF is evidence.
- History is IndexedDB. There is no account, no server-side store of past analyses, and no way to
  list what another visitor analysed.

## Deploy

**Frontend to Vercel, backend to a container host (Render, Fly, Railway) — not both to Vercel.**
The API is a long-lived process: a `spawn` analyser pool started once at boot, a torch model kept
warm in the parent, a background sweeper, an in-memory job store shared across polls, and overlay
PNGs on a filesystem that outlives the request. A Python Vercel Function gives you none of that, and
importing this repo at its root fails before any of it matters — Vercel matches
`backend/app/main.py` as a FastAPI entrypoint and bundles ~1.17 GB of torch past the 500 MB limit
with `LAMBDA_SIZE_EXCEEDED`. [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) §0 has the four mismatches
and the exact settings for both halves, including the two things that bite:
`NEXT_PUBLIC_API_BASE_URL` is a build-time value on Vercel, and Render's small instances need
`PERF_PROFILE=render-1cpu` or the analysis finishes after the request deadline.

```bash
docker compose up --build          # web on :3000, api on :8010, needs OVERLAY_SIGNING_KEY
docker compose --profile redis up  # + redis, for STORE_BACKEND=redis and more than one web worker
```

The images are written for CPU-only torch (`~900 MB`, not `~6 GB`) and run as a non-root user. They
were **not built on the authoring machine** — no Docker daemon was available here — so treat the
first `docker build` as a verification step. Everything else in this repository was run: the test
suite, the four smoke scripts, the ML pipeline, and the browser pass at 1440 px and 390 px in both
themes.

## Layout

```
backend/    FastAPI service: analysis techniques, fusion, reliability, verdict, jobs, PDF
ml/         corpus layout and splitting, training, evaluation, fusion calibration, artifact checks
frontend/   Next.js App Router: six pages, overlay viewer, charts, IndexedDB history
docs/       deployment, architecture notes
samples/    the fixture set, with the ground truth each file was constructed to demonstrate
artifacts/  trained models and metrics — empty until you train something
```

## License

MIT for the code in this repository. That grant covers exactly what is here: the analysers, the API,
the frontend, the training scripts. It does not reach the datasets, because it cannot — the corpora
in [`ml/README.md`](ml/README.md) are research-licensed, non-commercial, and gated behind agreements
their authors control, and none of them are distributed here. Anyone training weights for anything
beyond personal study is responsible for reading those terms and agreeing to them directly.

## Read next

- [`REPORT.md`](REPORT.md) — problem statement, method, what was measured, what was not, limitations
- [`ml/README.md`](ml/README.md) — datasets, licences, training commands
- [`backend/README.md`](backend/README.md) — the service's own layout and test commands
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the process model, and why the CNN never enters the pool
