# TrueLens ML

Training and evaluation for the two learned detectors the API serves: an EfficientNet-B0
fake/generated classifier with Grad-CAM, and — optionally — a linear probe on frozen CLIP features.
Plus the calibrated fusion model that turns ten analyser scores into two probabilities.

```
ml/
  truelens_ml/          corpus layout, splitting, augmentation, model, evaluation writers
    corpus.py           real/fake tree -> leak-free train/val/test (group-atomic, class-stratified)
    fusionset.py        authentic/ai_generated/edited/ai_and_edited tree -> 48-column feature rows
    augment.py          letterbox, train-time JPEG/resize/blur/crop jitter (same pad as the runtime)
    model.py            build / save_checkpoint / load_checkpoint (weights_only=True)
    evalout.py          metrics.json, roc.json, confusion_matrix.json + a validator that refuses bad ones
    synthetic.py        texture corpus generator, for exercising the pipeline with no downloads
  scripts/
    make_corpus.py      write a scaffold or a generated corpus
    train_cnn.py        fine-tune EfficientNet-B0, best-val checkpoint, early stopping
    evaluate_cnn.py     test + robustness + calibration -> the three JSON files the Methodology page reads
    calibrate_fusion.py two calibrated heads over the fusion vector -> fusion.joblib
    train_clip_probe.py linear probe on frozen ViT-L/14 features -> clip_probe.joblib
    check_artifacts.py  does each produced artifact satisfy the contract the running service expects
```

## Nothing here is checked in, and that is the state you start from

`artifacts/` ships empty. No weights and no fusion blob are committed, because the only data this
repository can prove anything about is synthetic constructions whose forgery mechanism is known by
definition — and a *confident* detector trained on those would corrupt every verdict the product
emits while raising the apparent reliability of the analysis. The service's fallback for each absent
artifact is deliberate and labelled:

| Artifact | Absent behaviour |
|---|---|
| `artifacts/model.pt` | CNN technique reports unavailable; `cnn.*` columns are NaN, not 0.0; reliability is penalised |
| `artifacts/fusion.joblib` | documented **expert prior** is used, and the response says `fusion.source = "expert"` |
| `artifacts/metrics.json` | Methodology page renders its "not trained yet" panel rather than invented numbers |
| `artifacts/clip_probe.joblib` | probe disabled; `cnn.clip_score` stays NaN |

The gap between "the pipeline works" and "the detector works on photographs" is exactly one dataset
download, and it is yours to make: the licence terms of every corpus below are research-only,
non-commercial, and gated behind a form that has to be signed as you.

## Run it with no data at all

Exercises every stage — corpus, split, training, evaluation, feature extraction, fusion fitting,
probe fitting, artifact contracts — against generated textures and constructions. Takes a few
minutes on CPU and writes only into `ml/.smoke/`.

```bash
cd backend && uv sync --extra dev
uv run python ../ml/scripts/make_corpus.py --out ../ml/.smoke/corpus --generated \
    --images 48 --per-group 2
uv run python ../ml/scripts/train_cnn.py --corpus ../ml/.smoke/corpus \
    --out ../ml/.smoke/model.pt --epochs 3 --batch 8 --no-pretrained
uv run python ../ml/scripts/evaluate_cnn.py --corpus ../ml/.smoke/corpus \
    --weights ../ml/.smoke/model.pt --out-dir ../ml/.smoke --generated --limit 48
uv run python ../ml/scripts/calibrate_fusion.py --generated --corpus ../ml/.smoke/fusion_corpus \
    --out ../ml/.smoke/fusion.joblib --report ../ml/.smoke/fusion_report.json \
    --weights ../ml/.smoke/model.pt
uv run python ../ml/scripts/train_clip_probe.py --self-test --out ../ml/.smoke/clip_probe.joblib
uv run python ../ml/scripts/check_artifacts.py --weights ../ml/.smoke/model.pt \
    --metrics ../ml/.smoke/metrics.json --fusion ../ml/.smoke/fusion.joblib \
    --clip-probe ../ml/.smoke/clip_probe.joblib
```

Expect `check_artifacts` to print `22 passed, 0 failed`, and expect the accuracy numbers to be
worthless — `evaluate_cnn` writes that into `metrics.json` itself, in `dataset.note`, and the
Methodology page renders it. With `--no-pretrained` the CNN sits at chance (val AUC 0.5) by
construction, and `calibrate_fusion` will print a `WARNING` that its Platt map made expected
calibration error *worse* on the manipulation head; both are the small-corpus behaviour, and the
script says so rather than letting the numbers look chosen. What these runs prove is the plumbing: a
checkpoint that loads into `CnnRuntime`, Grad-CAM that survives `inference_mode`, metrics that
`JSON.parse` accepts, and a fusion blob `_Learned.load()` refuses to reject.

`--generated` writes into an empty directory and refuses to mix corpora, so a second run of the
sequence above needs `rm -rf ml/.smoke/fusion_corpus` first — or drop the flag and reuse the corpus
the previous run left there.

## Corpus layout

Both detectors read a directory tree rather than a CSV, because the directory *is* the annotation
that matters: what family produced an image, and which scene it belongs to.

**CNN** — `truelens_ml/corpus.py`:

```
corpus/
  real/<group>/<image>                 label 0, a camera capture that never saw a generator
  fake/<generator>/<group>/<image>     label 1, one directory per generator family
```

**Fusion** — `truelens_ml/fusionset.py`. Each sample carries an `(ai, edited)` pair, so the four
combinations are four directories and an image can be positive for both heads:

```
fusion_corpus/
  authentic/<group>/<image>            (0, 0)
  ai_generated/<group>/<image>         (1, 0)
  edited/<group>/<image>               (0, 1)   splices, copy-move, double compression
  ai_and_edited/<group>/<image>        (1, 1)   generated plate with a region replaced
```

**`<group>` is load-bearing.** Frames of one video, crops of one scan, and the burst from one prompt
are near-duplicates. A split that puts frame 12 in train and frame 13 in val measures memorisation
and reports it as accuracy — so splitting is group-atomic and class-stratified, and both scripts
*raise* when a class has too few groups to hold any of it back. A loose file sitting at a class root
becomes a group of one, which is the honest treatment rather than silently pooling it with its
neighbours.

`<generator>` is load-bearing too: a detector trained near one family generalises worse to the next
than its pooled accuracy suggests, which is why `evaluate_cnn.py` and `train_clip_probe.py` report
per-family numbers and the pooled one is shown as the summary rather than the answer.

## Datasets

Verify the licence and the access terms yourself before downloading, redistributing, or training a
product on any of these. Counts are as published and move between releases.

| For | Dataset | What it provides | Access |
|---|---|---|---|
| `real/` | COCO (`val2017`, `train2017`) | Camera photography, wide device/scene spread, per-image source URLs | Public; images keep their Flickr terms |
| `real/` | FF++ / DF420 real frames | Faces at known compression levels, needed for the face-consistency path | Research request form, signed EULA |
| `fake/<family>/` | Real and Fake Faces (Yahoo, 140k) | 4 GAN families against FF++ real faces, the classic UnivFD/DF2K split | Research-only licence on the distribution page |
| `fake/<family>/` | DiffusionForensics | ~a dozen diffusion samplers, several resolutions; the diversity that makes per-family reporting mean something | Hosted on Hugging Face; several subsets gated |
| `fake/<family>/` | GenImage | Large generator-by-resolution grid built for exactly this transfer question | Gated download, research-only |
| `edited/` | CASIA v2, Columbia, Coverage | Splice and copy-move forgeries with region masks | Public research mirrors; originals flickr-sourced |
| `edited/` | IMD2020, Apache-Midata | Large, diverse manipulations with masks | Request forms |
| `ai_and_edited/` | constructed | A generated plate with a real region pasted in, or the reverse | Build it with `fusionset.build_generated_corpus`, keep the offsets |

What to aim for, in round numbers: **≥20k real and ≥20k fake across at least three generator
families** for the CNN, **≥1k per fusion class across ≥50 groups each** before the fusion heads are
worth shipping. Below that, `calibrate_fusion.py` refuses to write and the expert prior keeps
running — which is the correct outcome, not an inconvenience.

## Training

```bash
uv run python ../ml/scripts/train_cnn.py --corpus ../data/corpus \
    --out ../artifacts/model.pt --epochs 30 --batch 32 --lr 3e-4 --patience 5
```

BCE-with-logits on one output, `pos_weight = n_real / n_fake` so a lopsided corpus cannot be fit by
predicting the majority class, AdamW with cosine decay, gradient clipping, best-validation-AUC
checkpoint restored at the end, early stopping on `--patience`. Checkpoints are saved as
`{"state_dict", "meta"}` and loaded with `weights_only=True` — unpickling a checkpoint is running
someone else's code, so a model file you did not produce is a supply-chain decision, not a config
value.

```bash
uv run python ../ml/scripts/evaluate_cnn.py --corpus ../data/corpus \
    --weights ../artifacts/model.pt --out-dir ../artifacts --tta
```

Scores the test split through the *deployed* preprocessing path (`CnnRuntime.encode_for_resize`),
not a reimplementation of it, then writes `metrics.json`, `roc.json`, `confusion_matrix.json`. It
also re-scores under JPEG q70/q50, 3px blur and 256/512 downscaling, and per generator family, into
`robustness` — the section a reviewer should read first, because it is the one that predicts what
happens to the image they just uploaded from a messenger app. Thresholds come from maximum
validation F1 and travel with the accuracy figures that were measured at them.

```bash
uv run python ../ml/scripts/calibrate_fusion.py --corpus ../data/fusion_corpus \
    --weights ../artifacts/model.pt --out ../artifacts/fusion.joblib \
    --report ../artifacts/fusion_report.json
```

Two `HistGradientBoostingClassifier` heads over the 48 columns of `app.analysis.features.FEATURES`,
each followed by a Platt (or `--method isotonic`) calibrator fitted on groups the base head never
saw. It refuses to write when the corpus is too small, when the calibration half is missing a class,
when a technique measured under half the corpus, or when a `--generated` corpus is pointed at
`artifacts/`. The rules that were fused by hand do not disappear — they stay in `verdict.py` as
named, deterministic evidence flags that can floor a score, so a signed C2PA generative manifest is
still a *fact* in the report rather than one more feature the model might discount.

```bash
uv run python ../ml/scripts/train_clip_probe.py --corpus ../data/corpus \
    --features-cache ../data/clip_vitl14.npz --out ../artifacts/clip_probe.joblib
```

Needs `uv sync --extra clip` and ~900 MB of ViT-L/14 weights on first use. The cache is keyed by
path and records the encoder, so a second probe run costs seconds and a cache from a different
architecture is refused rather than trusted. Enable at serve time with `ENABLE_CLIP_PROBE=true`;
until then nothing in the product depends on it.

## Where artifacts are found

`app.config` resolves each path against the working directory, then `backend/artifacts`, then the
repository root — so a training run that writes `../artifacts/model.pt` is picked up without an env
var. `TRULENS_ARTIFACTS` overrides the directory for every artifact, which is what `ml/scripts/_bootstrap.py`
honours too, so a script and a server can be pointed at the same run. Related settings:
`CNN_WEIGHTS_PATH`, `FUSION_PATH`, `METRICS_PATH`, `CLIP_PROBE_PATH`, `ENABLE_CLIP_PROBE`.

## What the numbers do not mean

`evaluate_cnn.py` writes a `dataset.note` into `metrics.json` and the Methodology page displays it,
because a metric without its corpus is a claim about a dataset nobody agreed to. AUC is a ranking:
it says nothing about whether 0.8 means 80%, which is why expected-calibration error is reported
next to it and why `calibration` is *measured and described* rather than silently applied to the
deployed head. Neither detector can tell a generated image from a heavily edited photograph, a
screenshot of one, or a real photo of a screen — and the reliability layer exists to say "this file
could not be inspected" instead of producing a confident answer about it.
