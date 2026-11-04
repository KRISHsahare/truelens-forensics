# TrueLens — technical report

Written to be read as an account of what was built and what was actually measured. Where a number is
absent, the reason is stated rather than papered over; where a check was not run, it says so.

## 1. Problem statement

A reader who uploads a photograph wants to know whether it is what it appears to be: a capture from a
camera, a picture produced by a model, or an altered version of a capture. The interesting part of the
problem is not the classifier. It is that the honest answer is usually "here is the evidence, and here
is how little of it was measurable".

Three failure modes make most implementations of this wrong:

1. **A single blended score.** "74% fake" merges two different hypotheses. A file can be neither
   generated nor edited and still score low on both, and a generated image with a spliced region is
   not the average of those two facts.
2. **Silence reported as cleanliness.** An analyser that cannot run returns 0.0 because that is the
   path of least resistance. ELA on a PNG returns 0. Every PNG then looks authentic, and the tool is
   confidently wrong about the majority of images people actually upload.
3. **A number without its conditions.** A hand-tuned weighted sum presented as a measurement, an
   accuracy figure without its operating threshold, a validation set that shares frames with the
   training set. Each of these produces a plausible-looking product whose claims are unfounded.

So the deliverable is an engine that keeps the two hypotheses separate, keeps "not measured"
distinguishable from "measured clean" all the way to the wire and the PDF, and reports reliability as
a separate figure with an itemised reason for every penalty — in a UI that shows the evidence over the
image rather than asking for trust.

## 2. What was built

**Engine.** Ten analysers behind one contract, `run(...) -> TechniqueResult(score, visual, details)`,
which is the prototype's `(score, visual, details)` with the score allowed to be `None`:

| # | Technique | Feeds | Track | Overlay | What it measures |
|---|---|---|---|---|---|
| 01 | Metadata & Provenance | ai | container | none | EXIF/XMP/PNG chunks; generator parameter blocks. Its *score* does not vote on AI — platforms strip EXIF |
| 02 | Content Credentials | ai | container | none | C2PA manifest verified / invalid / absent |
| 03 | Error Level Analysis | edited | native | heatmap | Regions whose re-encode error disagrees with the frame, with a content confound check |
| 04 | JPEG Ghost | edited | native | heatmap | DCT striation revealing a second compression history in part of the image |
| 05 | Block-Grid Alignment | edited | native | heatmap | Whether every region shares one 8×8 block phase |
| 06 | Noise Inconsistency | edited | analysis | heatmap | Per-block sensor noise floor; a spliced region carries another camera's noise |
| 07 | Copy-Move Cloning | edited | analysis | vector | SIFT self-matching, Lowe ratio, displacement voting |
| 08 | Face Consistency | edited | analysis | vector | YuNet faces + landmarks: blending boundaries, lighting symmetry, specular consistency |
| 09 | Frequency Spectrum | ai | native | spectrum | Amplitude-spectrum decay plus periodicity from checkerboard upsampling |
| 10 | Learned Detector | ai | analysis | heatmap | EfficientNet-B0, one logit, Grad-CAM via tensor hooks; optional frozen-CLIP linear probe reported alongside |

Above them: a 48-column feature vector (`features.py`, one definition shared by trainer and server),
two calibrated gradient-boosted fusion heads with a documented noisy-OR prior as the fallback, a
multiplicative reliability layer, and a verdict layer that keeps deterministic findings as *flags*
that can floor a score. Around them: three resolution tracks, a spawn process pool with the CNN
pinned to the parent, signed overlay URLs with a ten-minute lifetime, SHA-256 result caching that
refuses to replay a partial run, per-address rate limits, and a server-rendered PDF.

**Product.** Six routes — landing, analyzer, printable report, methodology, batch, history — plus the
privacy and limitations pages. Drag-drop, clipboard paste, URL import, live per-technique progress,
an overlay viewer with opacity/zoom/pan/swipe/side-by-side, per-technique cards with "why this
matters", a metadata panel with red flags, sortable batch table with CSV export, and IndexedDB-only
history.

**ML pipeline.** Corpus discovery and group-atomic, class-stratified splitting; augmentation that
matches the serving preprocessing (JPEG/resize/blur round trips, not just crops); training with class
weighting, best-val checkpointing and early stopping; evaluation that writes `metrics.json`,
`roc.json`, `confusion_matrix.json` through a validator that refuses a file a browser cannot parse;
fusion calibration with a disjoint calibration split; the CLIP probe trainer; and an artifact checker
that tests each produced file against the contract the running service expects.

## 3. Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the reasoning behind the process model, the
tracks, the deadlines and the cache. The short version:

```
upload → magic-byte/format/pixel checks → decode into 3 tracks
       → 4 groups (container, native, analysis, learned) in parallel
       → outcomes (score | None, visual, details)
       → 48-column feature vector → fusion (learned heads | labelled expert prior)
       → reliability (× penalties) → verdict (ai, edited, overall = max) + flags + reasons
       → signed overlays, timings, limitations → JSON / PDF
```

## 4. Datasets

The product's claims depend on data this repository does not contain, and cannot: the corpora that
would justify them are research-licensed, non-commercial, and gated behind agreements that have to be
signed by the person who will use them.

| Needed for | Corpus | Status here |
|---|---|---|
| CNN `real/` | COCO, FF++ real frames | not downloaded (FF++ requires a signed EULA) |
| CNN `fake/<family>/` | Real & Fake Faces (140k), DiffusionForensics, GenImage | not downloaded (gated forms; some subsets are private) |
| Fusion `edited/` | CASIA v2, Columbia, Coverage, IMD2020 | not downloaded |
| Fusion `ai_and_edited/` | constructed from the above | constructed synthetically only |
| Pipeline verification | generated textures + known constructions | **built and run here** |

What exists in this repository instead: `ml/truelens_ml/synthetic.py` (spectral textures whose
sensor-noise floor and frequency slope differ by class) and
`backend/tests/fixtures/make_fixtures.py` (splices with known offsets and quality histories,
copy-move with a known displacement, double compression, generator metadata, a period-2 checkerboard
"generated" frame). Both are labelled as what they are, in the files themselves and in every artifact
they produce.

`ml/README.md` lists the layout, the acquisition route for each corpus, the size targets before a
model is worth shipping (≥20k per CNN class across ≥3 generator families; ≥1k per fusion class across
≥50 groups), and the licensing caution.

## 5. Evaluation

### 5.1 What was measured, here

| Check | Result |
|---|---|
| Backend test suite | **453 passed** in 15.7 s, including 13 new ML-pipeline contract tests |
| Spawn-pool tests (`-m pool`) | pass — the pickling path that an in-process test cannot reach |
| `smoke_techniques.py` | all expectations met (each analyser fires ≥0.35 on its artefact, ≤0.25 on a quiet file) |
| `smoke_fusion.py` | all 10 fixtures produce the expected verdict *and* the expected score band, including the two that must come back `inconclusive` |
| `smoke_api.py`, `smoke_orchestrator.py` | pass |
| Artifact contracts (`check_artifacts.py`) | **22 passed, 0 failed** across checkpoint, metrics, fusion blob, CLIP probe |
| Browser: 7 routes × {1440 px, 390 px} × {dark, light} | **0 console errors, 0 uncaught exceptions** |
| Analyzer flow (upload → progress → verdict → overlays → swipe/side-by-side) | completes at both widths; overlay geometry verified inside the frame box |
| Batch: 3 files, sort, CSV, PDF | verdicts and four-decimal scores in the CSV; 205 KB server-rendered PDF downloaded |
| End-to-end latency, 12-core CPU, no CNN weights | 311-672 ms per image (`total_ms` from the API's own timings); 48 images × 10 analysers in 9.7 s |

### 5.2 What the trained-model numbers mean

`make ml-smoke` (`ml/README.md`) runs the full pipeline on generated data. Reproduced output:

```
train_cnn      3 epochs, --no-pretrained → val AUC 0.500, checkpoint at epoch 1
evaluate_cnn   test acc 0.375-0.5, AUC 0.5-0.5625, n=8; metrics/roc/confusion written
calibrate_fusion  ai     auc 0.9688  brier 0.1163 → 0.0932  ece 0.1211 → 0.1064
                  edited auc 1.0     brier 0.0037 → 0.0138  ece 0.0405 → 0.1096  (WARNING printed)
                  24-27 of 48 columns constant on this corpus, dropped
train_clip_probe --self-test: AUC 0.84 val / 0.81 test on separable synthetic embeddings
```

Read these as plumbing, not performance. The CNN numbers sit at chance because the checkpoint was
randomly initialised on texture images — that is the point of the run, and `metrics.json` carries a
`dataset.note` saying exactly this, which the Methodology page renders. The fusion AUCs describe how
well two heads separate *the constructions in `ml/truelens_ml/fusionset.py`* from each other; a
generator that leaves no spectral trace and a splice that matches the plate's compression history
would not be separated by anything in that corpus.

The `edited` head's calibration *worsened* ECE (0.0405 → 0.1096) on 16 held-out rows, and the script
now says so and records it in the blob's `meta.warnings` rather than letting a worse-calibrated model
ship behind a "calibrated: true" flag. That is the kind of finding a smoke run exists to surface.

### 5.3 What was not measured, and why it matters

- **No accuracy claim about real photographs or real generators.** Not estimated, not asserted. The
  gated datasets are the one step a reviewer must do, and the commands for it are in `ml/README.md`.
- **No deepfake/video evaluation.** The face analyser works on single frames; a swapped face in a
  still is in scope, temporal inconsistency is not, and the UI says so.
- **No adversarial robustness.** Re-encoding, resizing and blurring are measured (`robustness` in
  `metrics.json`); deliberate evasion (diffusion-based re-rendering, GAN-specific anti-forensics) is
  not, and would defeat several of these checks by design.
- **No C2PA ecosystem coverage beyond the library installed here.** A valid manifest is read; a
  manifest from an issuer the trust list does not include is reported as such rather than trusted.
- **No Docker image build.** The authoring machine had the Docker CLI but no daemon, so
  `docker-compose.yml` is config-validated (`docker compose config` parses, the redis profile
  resolves) and both Dockerfiles are written against the same commands that were run locally —
  `uv sync --frozen`, `npm ci && npm run build`, `node .next/standalone/server.js` (which served 200
  for `/` and `/analyze` here) — but the images themselves were not built. Treat the first build as a
  verification step.

### 5.4 Defects found by verification, and what came of them

Each of these was caught by measurement rather than reading, which is the only reason any of them were
caught at all:

1. **Hydration mismatch on `/history`** — a render-time `typeof indexedDB` check disagreed between
   server and client, React discarded the tree, and the resulting warning looked like a script-tag bug
   in the layout. Fixed by deciding that state in an effect.
2. **Report headline wrapping one word per line** — a viewport breakpoint plus a fixed-width gauge row
   squeezed the text column inside a narrower report layout. Fixed with a container query.
3. **Batch rows on a cache hit showed "succeeded" with no scores, no links and a CSV of blanks** —
   `/api/batch` returns a terminal status for a cached image, and the poll loop skipped terminal rows,
   so the result was never fetched. Fixed by making "settled" mean *terminal and holding a result*.
4. **A page that looked healthy and did nothing when opened on `http://127.0.0.1:3000`** — the backend
   allowlisted `http://localhost:3000` and Next blocked dev resources from the other spelling, while
   this repo's own `.env.local` recommended `127.0.0.1`. Fixed on both sides: the loopback sibling is
   accepted in dev, and `allowedDevOrigins` names both hosts.
5. **Methodology schematics unreadable on a phone** — a fixed `h-24` meant `preserveAspectRatio`
   pinned the drawing to its literal 220×96 units at every width, so 6.5-unit labels stayed 6.5 px.
   Fixed by letting the height follow the width.
6. **The report's SHA-256 ran off the card** — a 64-hex token has no break opportunity. Fixed with
   `break-all`, because the digest is the report's identity and an ellipsis hides the part that cannot
   be guessed.
7. **`evaluate_cnn` refused to write its own metrics file** — scikit-learn returns PR precision/recall
   one element longer than its thresholds, and ROC's first threshold is `inf`. The validator was right;
   the writer was wrong. Now aligned and sentinel-mapped, with a regression test.

## 6. Limitations

**Of the method.** Every analyser here measures an artefact of a *process*, and processes can be
imitated or destroyed. A photograph re-encoded the way a generator's output is re-encoded looks like a
generator. A generated image passed through a phone camera (or a screen and a second camera) looks
like a photograph. Social platforms resize and re-compress, which erases the block-phase and
compression-history evidence — that is why `heavy_compression` and `screenshot` are expected to come
back `inconclusive` rather than authentic. Content confounds are measured, not assumed: ELA and noise
both report when a high score is explainable by the image's own texture rather than by an edit.

**Of the numbers.** A likelihood is not a proof, and the two sub-scores are not probabilities of
guilt — they are calibrated to whatever corpus trained them, which is stated in `dataset.note` and
displayed on the page. Calibration is only as good as the calibration split: the fusion heads are
fitted on one set of groups and calibrated on a disjoint set, so the reported ECE describes rows the
base head never saw, but on a small corpus that figure is noisy by construction.

**Of this build.** No trained weights ship, because nothing here was trained on data that could
justify shipping them. The default fusion is the labelled expert prior — an explicit, auditable
statement of trust in each analyser, marked as such in every response and in the PDF. The YuNet face
model is vendored (232 KB, from the OpenCV Zoo); when it is absent, or when an image contains no
face, the technique reports not-measured rather than clean, which is why `face.measured` is 0 across a
corpus of texture images and of photographs without people. The Docker images are unbuilt here (§5.3).
And the product's own honest answer to "is this proof" is no, printed on every page rather than buried
in a terms of service.

## 7. Making the claims real

One reviewer, one afternoon, with dataset access:

```bash
# 1. lay out the corpora (ml/README.md: which dataset goes in which directory, and why groups matter)
# 2. train and evaluate the detector
backend/.venv/bin/python ml/scripts/train_cnn.py --corpus data/corpus --out artifacts/model.pt --epochs 30
backend/.venv/bin/python ml/scripts/evaluate_cnn.py --corpus data/corpus --weights artifacts/model.pt --out-dir artifacts
# 3. fit the fusion on labelled authentic / generated / edited / both
backend/.venv/bin/python ml/scripts/calibrate_fusion.py --corpus data/fusion_corpus --weights artifacts/model.pt
# 4. optional second opinion
uv sync --extra clip && backend/.venv/bin/python ml/scripts/train_clip_probe.py --corpus data/corpus
# 5. prove the artifacts satisfy the service
backend/.venv/bin/python ml/scripts/check_artifacts.py --weights artifacts/model.pt --metrics artifacts/metrics.json \
  --fusion artifacts/fusion.joblib --clip-probe artifacts/clip_probe.joblib
```

Then the Methodology page renders the confusion matrix, ROC, PR, calibration and robustness figures
from the run, and `fusion.source` flips from `expert` to `learned` without a code change.

## 8. Reproducing everything in this report

```bash
cd backend && uv sync --extra dev
uv run pytest -q                        # 453 passed
uv run pytest -q -m pool
uv run python scripts/smoke_techniques.py
uv run python scripts/smoke_fusion.py
uv run python scripts/smoke_api.py

cd .. && make ml-smoke PY=backend/.venv/bin/python   # or the five commands in ml/README.md
backend/.venv/bin/python ml/scripts/check_artifacts.py --weights ml/.smoke/model.pt \
  --metrics ml/.smoke/metrics.json --fusion ml/.smoke/fusion.joblib --clip-probe ml/.smoke/clip_probe.joblib

cd frontend && npm ci && npm run build  # 9 routes, type-checks clean
```
