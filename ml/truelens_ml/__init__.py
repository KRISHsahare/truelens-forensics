"""Training and evaluation code for TrueLens's learned detectors.

Lives outside ``backend/`` on purpose: the API ships CPU inference and must never depend on a
training loop, while this package depends on the API in one direction only — it imports
``app.analysis.features`` so that the fusion model is fitted against *exactly* the columns the
running service builds. That single shared definition is the reason a saved fusion artifact can
refuse to load rather than silently mis-score.

Layout::

    truelens_ml/
        corpus.py      on-disk layout, discovery, leak-free splits, the torch Dataset
        synthetic.py   a generated corpus, for verifying this pipeline without a download
        augment.py     the augmentation the generalisation claims rest on
        model.py       EfficientNet-B0 with the head the runtime expects
        evalout.py     metrics.json / roc.json / confusion_matrix.json writers
    scripts/
        make_corpus.py        write a real/fake corpus tree (downloaded or generated)
        train_cnn.py          fine-tune the detector -> artifacts/model.pt
        evaluate_cnn.py       held-out numbers -> artifacts/metrics.json
        calibrate_fusion.py   fitted fusion classifiers -> artifacts/fusion.joblib
        train_clip_probe.py   optional UnivFD-style linear probe -> artifacts/clip_probe.joblib
        check_artifacts.py    assert every produced file satisfies the runtime's contract

Every script is importable and CLI-runnable; nothing here runs at API startup.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
