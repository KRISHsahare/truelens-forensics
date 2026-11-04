"""The training pipeline's own contracts, exercised end to end without a dataset download.

These are the ``slow`` tests: they build corpora, fine-tune nothing but load a checkpoint through the
serving runtime, write the three JSON files the Methodology page reads, and fit a fusion blob. A full
run is tens of seconds, which is why they are not in the default group.

They exist because every failure mode in this area is *silent at the layer above*. A corpus splitter
that leaks looks identical to one that does not — you just get a better number. A metrics file with
``NaN`` in it parses in Python and throws in ``JSON.parse``, so the page shows "not trained yet" for
a file that exists. A checkpoint whose head emits two logits loads as "unavailable", the same symptom
as no file at all. A fusion blob whose columns drifted from ``features.FEATURES`` is refused by
``_Learned.load`` and the service then quietly runs the expert prior forever. None of those produce
a traceback, and a reviewer sees a working product with meaningless numbers in it.

Nothing here asserts an accuracy figure. The corpora are ``truelens_ml.synthetic`` textures and
``fusionset`` constructions, so a number would describe those, and asserting on it would just make
the next seed change a false failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for entry in (str(REPO / "ml"), str(REPO / "backend")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

pytestmark = pytest.mark.slow

from truelens_ml import corpus as C          # noqa: E402
from truelens_ml import evalout as E         # noqa: E402
from truelens_ml import fusionset as F       # noqa: E402


# --------------------------------------------------------------------- fixtures
def write_cnn_corpus(root: Path, *, groups: int = 6, per_group: int = 2, size: int = 96) -> Path:
    """``real/gNNNN/*.jpg`` and ``fake/<family>/gNNNN/*.png``, mixed containers on both sides.

    Mixed containers matter: if every real image were a JPEG and every fake a PNG, extension alone
    would separate the classes, the analyser scores would be irrelevant, and the split tests below
    would pass for the wrong reason.
    """
    import numpy as np
    from PIL import Image

    from truelens_ml.synthetic import render

    rng = np.random.default_rng(4)
    families = ("stylegan", "midjourney")
    for kind in ("real", "fake"):
        for g in range(groups):
            base = root / kind / families[g % 2] / f"g{g:04d}" if kind == "fake" else \
                root / "real" / f"g{g:04d}"
            base.mkdir(parents=True, exist_ok=True)
            for n in range(per_group):
                img = render(rng, kind=kind, size=size)
                suffix = ".jpg" if (g + n) % 3 else ".png"
                Image.fromarray(img).save(base / f"{n:02d}{suffix}")
    return root


@pytest.fixture(scope="module")
def cnn_corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cnn_corpus")
    return write_cnn_corpus(root)


@pytest.fixture(scope="module")
def fusion_corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fusion_corpus")
    F.build_generated_corpus(root, groups_per_class=3, variants_per_group=1,
                             size=(192, 256), seed=99)
    return root


@pytest.fixture(scope="module")
def fusion_rows(fusion_corpus: Path):
    """Every registered analyser over the fusion corpus, once.

    Module-scoped because it is the slow part of this file and three tests want the same matrix —
    and because sharing one extraction is what lets two of them compare each other's claims.
    """
    samples = F.discover(fusion_corpus)
    matrix, names, unreadable = F.extract_features(samples, config_overrides={"cnn_tta": False})
    return samples, matrix, names, unreadable


# ----------------------------------------------------------------- corpus layout
def test_discover_reads_the_documented_tree_and_labels_both_classes(cnn_corpus: Path) -> None:
    samples = C.discover(cnn_corpus)
    counts = C.tally(samples)
    assert counts["real"] == 12 and counts["fake"] == 12, counts
    assert {s.label for s in samples} == {C.REAL, C.FAKE}
    # The fake side keeps its family, because per-generator reporting is the only number that
    # predicts behaviour on a generator the corpus has not met.
    assert {s.source for s in samples if s.label == C.FAKE} == {"stylegan", "midjourney"}


def test_discover_refuses_a_layout_that_cannot_be_trusted(cnn_corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        C.discover(tmp_path / "does-not-exist")
    one_class = tmp_path / "one_class"
    (one_class / "real").mkdir(parents=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        C.discover(one_class)


def test_split_keeps_every_group_on_one_side_and_both_classes_on_both_sides(
        cnn_corpus: Path) -> None:
    samples = C.discover(cnn_corpus)
    splits = C.split_samples(samples, seed=7)

    # Leak test, the whole reason the splitter exists: a group is a scene, and two frames of one
    # scene on opposite sides of the boundary would let the test split measure memorisation.
    # The key is (source, group), not group alone — ``g0003`` under ``real/`` and ``g0003`` under
    # ``fake/stylegan/`` are unrelated directories, and a test that called that a leak would be
    # asserting a bug that does not exist.
    def key(sample: "C.Sample") -> tuple[str, str]:
        return (sample.source, sample.group)

    for name in ("val", "test"):
        trained = {key(s) for s in splits["train"]}
        strays = [key(s) for s in splits[name] if key(s) in trained]
        assert not strays, f"{name} shares scenes with train: {strays[:3]}"

    for name, part in splits.items():
        assert {s.label for s in part} == {C.REAL, C.FAKE}, f"{name} lost a class: {C.tally(part)}"


def test_split_is_reproducible_and_seed_dependent(cnn_corpus: Path) -> None:
    samples = C.discover(cnn_corpus)
    a = {s.path for s in C.split_samples(samples, seed=7)["test"]}
    b = {s.path for s in C.split_samples(samples, seed=7)["test"]}
    c = {s.path for s in C.split_samples(samples, seed=11)["test"]}
    assert a == b, "the same seed must produce the same split, or two runs are incomparable"
    assert a != c, "a different seed must move the boundary, or the seed is not being used"


# ------------------------------------------------------------------- checkpoint
def test_checkpoint_loads_through_the_serving_runtime(tmp_path: Path) -> None:
    """A saved file that ``CnnRuntime`` accepts, which is the only consumer that matters.

    Asserted through the runtime rather than against the training code because the two can drift:
    a prefix added by ``torch.compile`` or a DDP wrapper, a head with two logits, or in-place SiLU
    left unpatched all save perfectly and fail at load or at the first Grad-CAM call.
    """
    import torch

    from app.ml.cnn_runtime import CnnRuntime, reset_runtime
    from truelens_ml import model as M

    path = str(tmp_path / "model.pt")
    net = M.build_model(pretrained=False)
    M.save_checkpoint(net, path, meta={"architecture": M.ARCHITECTURE, "img_size": 224})

    reset_runtime()
    try:
        runtime = CnnRuntime(path)
        assert runtime.load(), runtime.load_error
        assert runtime.available
        head = runtime._model.classifier[1]  # noqa: SLF001 - this is the contract
        assert int(head.out_features) == 1, "two logits is a different problem than P(fake)"
        arr = torch.zeros(1, 3, runtime.input_size, runtime.input_size).numpy()
        score = runtime.score(arr, tta=True)
        assert score is not None and 0.0 <= score <= 1.0
        cam = runtime.gradcam(arr)
        assert cam is not None and cam.shape == (7, 7)
    finally:
        reset_runtime()


def test_load_checkpoint_uses_weights_only(tmp_path: Path) -> None:
    """``load_checkpoint`` must not unpickle arbitrary objects.

    A checkpoint is a file someone else produced. The parameter check is the cheap half of the
    guarantee; the real one is that the loader works on a plain state dict, so nothing needs the
    pickle path at all.
    """
    import torch

    from truelens_ml import model as M

    path = tmp_path / "ck.pt"
    net = M.build_model(pretrained=False)
    M.save_checkpoint(net, str(path), meta={"architecture": M.ARCHITECTURE})
    state, meta = M.load_checkpoint(str(path))
    assert meta["architecture"] == M.ARCHITECTURE
    assert isinstance(state, dict) and next(iter(state.values())).dtype in (
        torch.float32, torch.float64)
    assert torch.load(path, map_location="cpu", weights_only=True) is not None


# ------------------------------------------------------------------- metrics out
def test_written_metrics_survive_a_browser_parse(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=120)
    p = np.clip(rng.normal(loc=0.35 + 0.3 * y, scale=0.22), 0.0, 1.0)

    path = tmp_path / "metrics.json"
    E.write_evaluation(
        str(path),
        metrics=E.metrics_payload(y, p, threshold=0.5),
        dataset={"name": "test-corpus", "train": 80, "val": 20, "test": 20,
                 "classes": ["real", "fake"], "sources": ["synthetic"],
                 "augmentations": ["jpeg-reencode"], "note": "pipeline test, not accuracy"},
        model="efficientnet_b0_test",
        confusion=E.confusion_payload(y, p, threshold=0.5),
        roc=E.roc_payload(y, p),
        pr=E.pr_payload(y, p),
        extra={"trained_at": "2026-01-01T00:00:00Z"})

    raw = path.read_text(encoding="utf-8")
    # Python writes NaN and reads it back; JSON.parse throws on it, so the file exists and the
    # page says "nothing measured". ``allow_nan=False`` is the only honest check.
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)
    assert E.validate(json.loads(raw)) == []


def test_curves_have_equal_columns_and_no_infinite_sentinels() -> None:
    """The regression that ``validate()`` was written to catch.

    ``precision_recall_curve`` returns precision/recall one element longer than thresholds, and
    ``roc_curve`` opens its threshold list with ``inf``. Both parse in Python; one makes the chart
    axes disagree and the other is not JSON.
    """
    import math

    import numpy as np

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=300)
    p = np.clip(rng.normal(loc=0.3 + 0.4 * y, scale=0.25), 0.0, 1.0)

    roc = E.roc_payload(y, p)
    assert len({len(v) for v in roc.values()}) == 1, {k: len(v) for k, v in roc.items()}
    assert all(math.isfinite(v) for row in roc.values() for v in row)

    pr = E.pr_payload(y, p)
    assert len({len(v) for v in pr.values()}) == 1, {k: len(v) for k, v in pr.items()}
    assert all(math.isfinite(v) for row in pr.values() for v in row)
    assert all(0.0 <= v <= 1.0 for v in pr["thresholds"]), "thresholds are probability cut points"


def test_validate_refuses_the_payloads_a_page_cannot_render() -> None:
    """The writer's guard is the test target, so it is aimed at each failure separately."""
    import numpy as np

    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.6])
    good = {
        "trained_at": "2026-01-01T00:00:00Z",
        "model": "m",
        "dataset": {"name": "d", "note": "n", "train": 10, "val": 5, "test": 5,
                    "classes": ["real", "fake"], "sources": ["synthetic"],
                    "augmentations": []},
        "metrics": E.metrics_payload(y, p, threshold=0.5),
        "confusion_matrix": E.confusion_payload(y, p, threshold=0.5),
        "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.5, 1.0],
                      "thresholds": [1.1, 0.5, -0.1]},
    }
    assert E.validate(good) == []

    assert E.validate({k: v for k, v in good.items() if k != "metrics"})
    assert E.validate({**good, "confusion_matrix": {"labels": ["a", "b"], "matrix": [[1, 2]]}})
    lopsided = {**good, "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 1.0],
                                      "thresholds": [1.0, 0.0]}}
    assert E.validate(lopsided)


# ------------------------------------------------------------------ fusion vector
def test_feature_rows_are_the_deployed_vector_itself(fusion_rows) -> None:
    """``features.FEATURES`` column-for-column, produced by the real pipeline.

    ``calibrate_fusion.py`` writes the vector names into the blob and ``_Learned.load`` compares
    them, so a drift here is caught before training — but only if the rows this test builds are
    ordered by the same constant the service uses. Building them through ``fusionset`` is what
    makes that check non-trivial: it runs every registered analyser over real files.
    """
    from app.analysis.features import FEATURES

    samples, matrix, names, unreadable = fusion_rows
    assert F.tally(samples)["per_class"] == {"authentic": 3, "ai_generated": 3,
                                             "edited": 3, "ai_and_edited": 3}
    assert not unreadable
    assert names == list(FEATURES)
    assert matrix.shape == (len(samples), len(FEATURES))


def test_measured_companion_matches_its_score_column(fusion_rows) -> None:
    """Silence and zero are different measurements, and the vector has to keep them apart.

    ``0.0`` means "we looked and found nothing suspicious"; NaN means "we could not look". The
    collapse is how a forensics tool quietly decides every PNG is authentic, so the invariant is
    asserted on files that actually trigger it — a corpus where nothing was ever unmeasured would
    pass this test by vacuity, which is why the counter below is part of it.
    """
    import math

    from app.analysis.features import FEATURES

    _, matrix, _, _ = fusion_rows
    unmeasured_pairs = 0
    for technique in {c.split(".")[0] for c in FEATURES if "." in c}:
        score_i = FEATURES.index(f"{technique}.score")
        meas_i = FEATURES.index(f"{technique}.measured")
        for row in matrix:
            if row[meas_i] < 0.5:
                unmeasured_pairs += 1
                assert math.isnan(row[score_i]), (
                    f"{technique} reported a score while measured=0; either the explainer or the "
                    "vector is wrong about whether this file could be inspected")
    assert unmeasured_pairs, (
        "no technique was ever unmeasured on this corpus, so the rule above was never exercised")


def test_learned_blob_is_refused_when_its_columns_drift(tmp_path: Path) -> None:
    """``_Learned.load`` is the last line, and it has to be tested from both sides.

    Accepting the correct blob proves nothing on its own — the bug this guards is a *silently*
    accepted model whose column 14 changed meaning after training, which is what a reordered
    ``FEATURES`` tuple produces.
    """
    import joblib
    import numpy as np

    from app.analysis.features import FEATURES
    from app.analysis.fusion import _Learned  # noqa: SLF001 - load() is the contract

    def heads():
        from sklearn.dummy import DummyClassifier

        # Fitted, because that is what the blob holds: ``_Learned.load`` unpickles whatever was
        # saved and ``learned_fuse`` calls ``predict_proba`` on it without asking, so an unfitted
        # estimator here would fail on ``predict_proba`` rather than on the contract under test.
        seed_rows = np.array([[0.0], [1.0]])
        ai = DummyClassifier(strategy="prior").fit(seed_rows, [0, 1])
        edited = DummyClassifier(strategy="prior").fit(seed_rows, [1, 0])
        return ai, edited

    ai, edited = heads()
    row = np.zeros(len(FEATURES))
    good = {"ai": ai, "edited": edited, "feature_names": list(FEATURES), "calibrated": True}
    good_path = str(tmp_path / "good.joblib")
    joblib.dump(good, good_path)
    model = _Learned.load(good_path)
    assert model is not None, "a correctly labelled blob must load"
    p = model.ai.predict_proba(row.reshape(1, -1))[0]
    assert len(p) == 2 and abs(float(p.sum()) - 1.0) < 1e-6

    drifted = {"ai": ai, "edited": edited, "feature_names": list(reversed(FEATURES)),
               "calibrated": True}
    bad_path = str(tmp_path / "drifted.joblib")
    joblib.dump(drifted, bad_path)
    assert _Learned.load(bad_path) is None, "a reordered vector must be refused, not scored"

    missing = {"ai": ai, "edited": edited}
    no_names_path = str(tmp_path / "anonymous.joblib")
    joblib.dump(missing, no_names_path)
    assert _Learned.load(no_names_path) is None


def test_generated_corpus_labels_are_what_the_directories_claim(fusion_corpus: Path,
                                                                cnn_corpus: Path) -> None:
    samples = F.discover(fusion_corpus)
    by_class: dict[str, set[tuple[int, int]]] = {}
    for s in samples:
        by_class.setdefault(s.cls, set()).add((s.ai, s.edited))
    assert by_class == {cls: {pair} for cls, pair in F.CLASSES.items()}, by_class

    with pytest.raises(ValueError):
        # A CNN-layout corpus is a real directory of real images with the wrong vocabulary, which
        # is the mistake a reader following two different README sections will actually make.
        F.discover(cnn_corpus)
