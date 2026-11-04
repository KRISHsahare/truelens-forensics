"""Path setup shared by every script in this directory.

``python ml/scripts/train_cnn.py`` puts ``ml/scripts`` on ``sys.path`` and nothing else, so a
script that wants ``truelens_ml`` (in ``ml/``) and ``app`` (in ``backend/``) has to say so.
Doing that inline in five scripts is how one of them ends up importing a stale copy after a
restructure; doing it here means the repo layout is stated exactly once.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["BACKEND", "ML_ROOT", "REPO_ROOT", "add_paths", "artifact"]

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
BACKEND = REPO_ROOT / "backend"


def add_paths() -> None:
    """Make ``truelens_ml`` and ``app`` importable, and pin the working directory.

    ``backend`` is added rather than installed because the API is not a library: the training
    script needs ``app.analysis.features`` for the fusion column definition and nothing else from
    it, and a half-installed copy of the service in a venv is a worse trap than a path entry.
    """
    for path in (str(ML_ROOT), str(BACKEND)):
        if path not in sys.path:
            sys.path.insert(0, path)


def artifact(name: str, *, default_dir: str = "artifacts") -> str:
    """Where a produced artifact goes, resolved the way ``app.config`` resolves it.

    The backend looks for ``artifacts/model.pt`` against the current directory, then
    ``backend/artifacts``, then the repository root — so writing into either of the last two is
    picked up without an env var. Writing into a third place is the one mistake that makes a
    successful training run look like it did nothing, which is why the choice is made here rather
    than left to whoever types ``--out``.
    """
    base = Path(os.environ.get("TRULENS_ARTIFACTS", "")).resolve() if os.environ.get(
        "TRULENS_ARTIFACTS") else None
    if base is None:
        base = BACKEND / default_dir if (BACKEND / default_dir).exists() else REPO_ROOT / default_dir
    base.mkdir(parents=True, exist_ok=True)
    return str(base / name)
