"""Shared test fixtures.

Two decisions shape everything here.

**Techniques are driven through ``worker.run_group``, not by importing each module.** The binder, the
field trimmer and the outcome wrapper are where the interesting bugs live — a test that called
``ela.run(gray)`` directly would have passed through every one of the defects this suite now catches,
because none of them were inside an analyser.

**Nothing in here touches the network or a trained model.** The images are the synthetic set from
``tests/fixtures/make_fixtures.py`` and the heavy (torch) techniques are excluded from the default
group run, so the suite is seconds rather than minutes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2
from app.analysis import worker
from app.analysis.types import JobArrays, SinkConfig, TaskOutcome
from app.config import get_settings
from app.core.job import new_job_id
from app.core.orchestrator import worker_config
from app.core.pipeline import build_arrays
from app.core.security import probe_header
from tests.support import FAST_NAMES, fixture_bytes

GROUP_TIMEOUT_S = 120.0


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def config(settings) -> dict:
    """The ``FromConfig`` namespace, from the same function the orchestrator uses.

    Restating it here is how a test drifts away from production: an inline copy once left out
    ``copy_move_features``, and the copy-move analyser failed in the test while working in the app.
    """
    return worker_config(settings)


@pytest.fixture(scope="session")
def arrays_factory(settings):
    """``fixture name -> JobArrays``, built once per name for the whole session.

    Decoding and building the two resolution tracks is the slowest thing a test can do, and every
    analyser test wants the same arrays.
    """
    cache: dict[str, JobArrays] = {}

    def build(name: str) -> JobArrays:
        if name not in cache:
            data = fixture_bytes(name)
            cache[name] = build_arrays(
                data, probe_header(data),
                analysis_max_dim=settings.analysis_max_dim,
                native_max_dim=settings.native_max_dim)
        return cache[name]

    return build


@pytest.fixture(scope="session")
def group_runner(config, tmp_path_factory):
    """Run a set of techniques in-process and return ``{name: TaskOutcome}``.

    ``worker.configure_parent`` is what makes this honest. Without it the ``FromConfig`` bindings
    resolve to nothing, which is exactly the misconfiguration that made the CNN report a timeout in
    production while succeeding in a hand-called test.
    """
    root = tmp_path_factory.mktemp("scratch")
    worker.configure_parent(config, threads=1)

    def run(arrays: JobArrays, names: list[str], *, max_dim: int = 512) -> dict[str, TaskOutcome]:
        # A real minted id, because ``scratch.path_safe`` gates overlay writes and an id that does
        # not look like ours means every overlay in the suite is refused and only logged.
        job_id = new_job_id()
        request = worker.GroupRequest(
            job_id=job_id, names=tuple(names),
            arrays=worker.trim_arrays(arrays, tuple(names)),
            sink=SinkConfig(root=str(root), job_id=job_id, max_dim=max_dim,
                            expires_at=time.time() + 3600),
            deadline_at=time.monotonic() + GROUP_TIMEOUT_S)
        return {o.name: o for o in worker.run_group(request).outcomes}

    return run


@pytest.fixture
def outcomes_factory(arrays_factory, group_runner):
    """``fixture name -> outcomes for every fast technique``."""

    def run(name: str) -> dict[str, TaskOutcome]:
        return group_runner(arrays_factory(name), FAST_NAMES)

    return run
