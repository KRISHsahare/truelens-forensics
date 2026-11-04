"""The worker: what crosses the process boundary, and what must not.

Three properties hold this module together, and each is a bug worth the code that prevents it. Work
is grouped by resolution track so a multi-megabyte array set crosses the pipe once instead of ten
times, which means trimming the fields a group cannot read — and trimming one a group *does* need is
a technique that fails on its own fixture. Second, a technique that throws must not fail the job, so
every path out of ``_run_one`` is an outcome, including the NaN that means "I could not measure".
Third, this module is re-imported by every spawn child, so nothing in its import closure may drag
torch or pydantic-settings into a process that was only asked to run ELA.
"""

from __future__ import annotations

import ast
import math
import sys
import time
import types as pytypes
from pathlib import Path
from queue import SimpleQueue

import numpy as np
import pytest

from app.analysis import techniques as _techniques_pkg
from app.analysis import worker
from app.analysis.registry import BY_NAME, TECHNIQUES
from app.analysis.registry import names as technique_names
from app.analysis.types import JobArrays, SinkConfig, TaskOutcome
from app.core.job import new_job_id

TECHNIQUE_DIR = Path(_techniques_pkg.__file__).parent
WORKER_SOURCE = Path(worker.__file__)

CONFIG = {"redact_pii": True, "face_model_path": None, "copy_move_features": 4000,
          "cnn_tta": False, "gradcam_min_ai_score": 0.5, "cnn_weights_path": None,
          "clip_probe_path": None, "clip_probe_enabled": False}


@pytest.fixture(autouse=True)
def _restore_worker_globals():
    """These are process-wide module globals; a test that sets them must hand them back."""
    snapshot = (worker._PROGRESS_QUEUE, worker._PROGRESS_CALLBACK, worker._CONFIG,
                worker._IN_CHILD, worker._THREADS)
    yield
    (worker._PROGRESS_QUEUE, worker._PROGRESS_CALLBACK, worker._CONFIG,
     worker._IN_CHILD, worker._THREADS) = snapshot


@pytest.fixture()
def events():
    """Collect parent-side progress events, so the stepper's input is observable."""
    seen: list[worker.ProgressEvent] = []
    worker.configure_parent(dict(CONFIG), threads=1, on_event=seen.append)
    return seen


@pytest.fixture()
def stub(monkeypatch):
    """Replace one technique module with a scripted ``run``.

    The worker imports techniques by dotted name at call time, which is exactly the seam a fake
    needs to sit in: the dispatcher is under test here, not the maths.
    """
    def apply(name: str, run):
        spec = BY_NAME[name]
        module = pytypes.ModuleType(f"app.analysis.techniques.{spec.module}")
        module.run = run
        monkeypatch.setitem(sys.modules, f"app.analysis.techniques.{spec.module}", module)
        return module
    return apply


def _arrays(**overrides) -> JobArrays:
    """A ``JobArrays`` whose fields are individually recognisable."""
    base = JobArrays(
        sha256="e" * 64, mime="image/jpeg", orig_w=640, orig_h=480, file_size=90_000,
        analysis_scale=1.0, native_box=(0, 0, 640, 480),
        rgb_u8=np.zeros((480, 640, 3), np.uint8),
        gray_u8=np.zeros((480, 640), np.uint8),
        native_rgb_u8=np.zeros((480, 640, 3), np.uint8),
        native_gray_u8=np.zeros((480, 640), np.uint8),
        orig_bytes=b"\xff\xd8\xff" + b"j" * 100,
        est_jpeg_quality=82, bpp=2.0, round_dimensions=False)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _sink(root: str = ".", **kwargs) -> SinkConfig:
    return SinkConfig(root=root, job_id=new_job_id(), max_dim=2048,
                      expires_at=time.time() + 600, **kwargs)


def _request(names, arrays=None, **kwargs) -> worker.GroupRequest:
    defaults = dict(job_id=new_job_id(), names=tuple(names), arrays=arrays or _arrays(),
                    sink=_sink(), deadline_at=time.monotonic() + 120.0, write_overlays=False)
    return worker.GroupRequest(**{**defaults, **kwargs})


def _only(result: worker.GroupResult, name: str) -> TaskOutcome:
    return next(o for o in result.outcomes if o.name == name)


# ------------------------------------------------------------------- field demand


def test_needed_fields_reads_only_the_array_bindings():
    assert worker.needed_fields(BY_NAME["ela"]) == frozenset(
        {"native_rgb_u8", "mime", "est_jpeg_quality"})
    assert worker.needed_fields(BY_NAME["copy_move"]) == frozenset(
        {"rgb_u8", "gray_u8", "analysis_scale"})
    # A FromConfig binding is a knob, not a payload; it must not keep anything alive.
    assert worker.needed_fields(BY_NAME["metadata"]) == frozenset({"orig_bytes", "mime"})


def test_every_field_a_technique_binds_exists_on_job_arrays():
    """A typo in the registry is a technique that silently fails in production."""
    declared = set(JobArrays.__dataclass_fields__)
    for spec in TECHNIQUES:
        missing = worker.needed_fields(spec) - declared
        assert not missing, f"{spec.name} binds {missing}, which JobArrays does not have"


# ------------------------------------------------------------------------ grouping


def test_heavy_techniques_are_routed_to_the_learned_group():
    assert worker.group_for("cnn") == "learned"
    assert worker.group_for("metadata") == "container"
    assert worker.group_for("ela") == "native"
    assert worker.group_for("noise") == "analysis"


def test_groups_cover_every_technique_exactly_once():
    buckets = worker.groups_for(list(technique_names()))
    flattened = [n for group in buckets.values() for n in group]
    assert sorted(flattened) == sorted(technique_names())
    assert len(flattened) == len(set(flattened))


def test_the_container_group_goes_first_and_order_within_a_group_is_registry_order():
    buckets = worker.groups_for(list(technique_names()))
    assert list(buckets) == ["container", "native", "analysis", "learned"]
    assert buckets["native"] == ["ela", "jpeg_ghost", "block_grid", "frequency"]
    assert buckets["analysis"] == ["noise", "copy_move", "face"]
    assert buckets["learned"] == ["cnn"]


def test_a_subset_is_grouped_without_strays():
    assert worker.groups_for(["ela", "noise", "metadata"]) == {
        "container": ["metadata"], "native": ["ela"], "analysis": ["noise"]}


def test_a_group_of_one_technique_still_names_its_group():
    assert worker.groups_for(["cnn"]) == {"learned": ["cnn"]}


# -------------------------------------------------------------------------- trimming


def test_a_group_carries_only_the_pixel_payload_it_can_read():
    trimmed = worker.trim_arrays(_arrays(), ("metadata", "c2pa"))
    assert trimmed.orig_bytes is not None, "the container group is the bytes"
    for field in ("rgb_u8", "gray_u8", "native_rgb_u8", "native_gray_u8"):
        assert getattr(trimmed, field) is None, f"{field} crossed the pipe for nothing"


def test_the_native_group_is_not_given_the_resampled_track():
    trimmed = worker.trim_arrays(_arrays(), ("ela", "jpeg_ghost", "block_grid", "frequency"))
    assert trimmed.native_rgb_u8 is not None and trimmed.native_gray_u8 is not None
    assert trimmed.rgb_u8 is None and trimmed.gray_u8 is None
    assert trimmed.orig_bytes is None


def test_the_analysis_group_is_not_given_the_native_track():
    trimmed = worker.trim_arrays(_arrays(), ("noise", "copy_move", "face"))
    assert trimmed.rgb_u8 is not None and trimmed.gray_u8 is not None
    assert trimmed.native_rgb_u8 is None and trimmed.native_gray_u8 is None


def test_scalars_are_never_trimmed():
    """``est_jpeg_quality = None`` is a *measurement* (a PNG), not a missing payload.

    Trimming scalars made every non-JPEG file report ``ela`` and ``noise`` as errored techniques,
    when the honest answer was that they ran and had nothing to read.
    """
    trimmed = worker.trim_arrays(_arrays(), ("noise",))
    assert trimmed.est_jpeg_quality == 82
    assert trimmed.analysis_scale == 1.0 and trimmed.native_box == (0, 0, 640, 480)
    assert trimmed.bpp == 2.0 and trimmed.round_dimensions is False
    assert worker.trim_arrays(_arrays(est_jpeg_quality=None), ("noise",)).est_jpeg_quality is None


def test_sha_is_kept_even_for_a_group_that_never_asks_for_it():
    assert worker.trim_arrays(_arrays(), ("ela",)).sha256 == "e" * 64
    assert worker.trim_arrays(_arrays(), ("ela",)).mime == "image/jpeg"


def test_keeping_a_field_keeps_the_same_array_rather_than_a_copy():
    source = _arrays()
    trimmed = worker.trim_arrays(source, ("ela",))
    assert trimmed.native_rgb_u8 is source.native_rgb_u8, "trimming must not duplicate pixels"
    assert trimmed is not source


def test_a_technique_whose_payload_was_trimmed_reports_the_gap_rather_than_raising():
    """The group that needs a field it was not sent, run through the real dispatcher.

    ``run_group`` cannot raise — a group that dies takes its finished members with it — so this
    arrives as an error outcome whose message names the field and the group.
    """
    result = worker.run_group(_request(["ela"],
                                       arrays=worker.trim_arrays(_arrays(), ("noise",))))
    outcome = result.outcomes[0]
    assert outcome.status == "error"
    assert "native_rgb_u8" in outcome.error_msg and "was not sent to this group" in outcome.error_msg


# -------------------------------------------------------------------------- binding


def test_a_config_knob_that_was_never_configured_is_an_error_not_a_none():
    """``None`` reached SIFT's ``n_features`` and the technique died later, elsewhere."""
    worker.configure_parent({}, threads=1)
    with pytest.raises(ValueError, match="copy_move_features"):
        worker._bind(BY_NAME["copy_move"], _arrays())


def test_literal_and_config_and_array_bindings_all_resolve(events):
    args, kwargs = worker._bind(BY_NAME["cnn"], _arrays())
    assert len(args) == 1 and isinstance(args[0], np.ndarray)
    assert kwargs["want_gradcam"] is True, "a Lit binding is passed through untouched"
    assert kwargs["tta"] is False and kwargs["gradcam_threshold"] == 0.5


def test_with_arrays_keeps_the_rest_of_the_request():
    request = _request(["ela", "noise"])
    trimmed = worker.trim_arrays(request.arrays, request.names)
    again = request.with_arrays(trimmed)
    assert again.arrays is trimmed
    assert again.job_id == request.job_id and again.names == request.names
    assert again is not request


# ---------------------------------------------------------------------- execution


def test_a_deadline_passed_skips_every_member_without_running_them(events):
    request = _request(list(technique_names()), deadline_at=time.monotonic() - 1)
    result = worker.run_group(request)
    assert {o.status for o in result.outcomes} == {"skipped"}
    assert all(o.error_code == "deadline" for o in result.outcomes)
    assert all(o.score is None for o in result.outcomes)
    assert len(result.outcomes) == len(request.names)


def test_the_deadline_is_checked_before_each_member_not_once_per_group(stub):
    """The first technique's cost is the last one's deadline risk."""
    def slow(*_args, **_kwargs):
        time.sleep(0.05)
        return 0.1, None, {}
    stub("ela", slow)
    result = worker.run_group(_request(["ela", "ela"], deadline_at=time.monotonic() + 0.02))
    assert [o.status for o in result.outcomes] == ["ok", "skipped"]


def test_an_unknown_technique_name_is_reported_not_raised(events):
    request = _request(["not_a_technique"])
    result = worker.run_group(request)
    assert result.job_id == request.job_id
    outcome = result.outcomes[0]
    assert outcome.status == "error" and outcome.error_code == "unknown_technique"


def test_a_technique_that_raises_produces_an_error_outcome_and_the_group_continues(stub):
    def boom(*_args, **_kwargs):
        raise RuntimeError("failed at D:\\app\\secrets\\config.py")
    stub("ela", boom)
    stub("noise", lambda *_a, **_k: (0.2, None, {}))
    result = worker.run_group(_request(["ela", "noise"]))
    assert _only(result, "ela").status == "error"
    assert "D:\\app" not in _only(result, "ela").error_msg
    assert _only(result, "noise").status == "ok", "one technique's exception is not the job's"


def test_a_nan_score_is_preserved_as_nan_and_still_reports_ok(stub):
    """NaN means "I could not measure". Zero means "I measured and it is clean"."""
    stub("ela", lambda *_a, **_k: (float("nan"), None, {"applicable": False}))
    outcome = worker.run_group(_request(["ela"])).outcomes[0]
    assert outcome.status == "ok"
    assert math.isnan(outcome.score)
    assert outcome.details["applicable"] is False


def test_a_score_of_zero_is_left_at_zero(stub):
    stub("ela", lambda *_a, **_k: (0.0, None, {}))
    assert worker.run_group(_request(["ela"])).outcomes[0].score == 0.0


def test_progress_events_are_running_then_a_terminal_carrying_the_outcome(stub, events):
    stub("noise", lambda *_a, **_k: (0.3, None, {}))
    result = worker.run_group(_request(["noise"]))
    assert [(e.name, e.status) for e in events] == [("noise", "running"), ("noise", "done")]
    assert events[0].outcome is None, "there is nothing to report while it is still running"
    assert events[-1].outcome is result.outcomes[0]


def test_a_technique_skipped_for_the_deadline_never_reports_running(stub, events):
    def never_called(*_a, **_k):  # pragma: no cover - the point is that this does not run
        raise AssertionError("ran past the deadline")
    stub("noise", never_called)
    worker.run_group(_request(["noise"], deadline_at=time.monotonic() - 1))
    assert [(e.name, e.status) for e in events] == [("noise", "skipped")]


def test_a_failed_technique_sets_its_stage_to_failed(stub, events):
    def boom(*_a, **_k):
        raise ValueError("no")
    stub("ela", boom)
    worker.run_group(_request(["ela"]))
    assert [e.status for e in events] == ["running", "failed"]


def test_a_progress_listener_that_throws_does_not_stop_the_analysis(stub, events):
    stub("noise", lambda *_a, **_k: (0.1, None, {}))
    def explode(_event):
        raise RuntimeError("the sink is on fire")
    worker._PROGRESS_CALLBACK = explode
    assert worker.run_group(_request(["noise"])).outcomes[0].status == "ok"


def test_the_group_return_is_a_fallback_for_a_dropped_event(stub, events):
    """The queue can be full; the return value is the second delivery of the same outcomes."""
    stub("noise", lambda *_a, **_k: (0.1, None, {}))
    stub("copy_move", lambda *_a, **_k: (0.2, None, {}))
    result = worker.run_group(_request(["noise", "copy_move"]))
    assert {o.name for o in result.outcomes} == {"noise", "copy_move"}
    assert [e.outcome for e in events if e.outcome is not None] == list(result.outcomes)


def test_wall_time_and_per_technique_ms_are_reported(stub, events):
    stub("metadata", lambda *_a, **_k: (0.1, None, {}))
    result = worker.run_group(_request(["metadata"]))
    assert 0 <= result.wall_s < 60
    assert result.outcomes[0].ms >= 0


# ----------------------------------------------------------------------- overlays


def _visual() -> np.ndarray:
    image = np.zeros((32, 32, 3), np.uint8)
    image[..., 0] = 200
    return image


def test_an_overlay_is_written_when_the_sink_allows_it(tmp_path, stub, events):
    stub("ela", lambda *_a, **_k: (0.5, _visual(), {}))
    sink = _sink(root=str(tmp_path))
    outcome = worker.run_group(_request(["ela"], sink=sink, write_overlays=True)).outcomes[0]
    assert outcome.overlay is not None
    assert outcome.overlay.rel_path == f"{sink.job_id}/ela.png"
    assert (tmp_path / outcome.overlay.rel_path).is_file()
    assert outcome.overlay.media_type == "image/png"
    assert outcome.overlay.expires_at == sink.expires_at
    assert (outcome.overlay.width, outcome.overlay.height) == (32, 32)


def test_a_sink_that_writes_no_overlays_scores_the_same_and_writes_nothing(tmp_path, stub, events):
    stub("ela", lambda *_a, **_k: (0.5, _visual(), {}))
    outcome = worker.run_group(_request(["ela"], sink=_sink(root=str(tmp_path)),
                                        write_overlays=False)).outcomes[0]
    assert outcome.score == 0.5 and outcome.overlay is None
    assert list(tmp_path.iterdir()) == []


def test_a_malformed_sink_job_id_degrades_to_no_overlay_rather_than_a_failed_technique(
        tmp_path, stub, events, caplog):
    stub("ela", lambda *_a, **_k: (0.5, _visual(), {}))
    sink = SinkConfig(root=str(tmp_path), job_id="../../outside", max_dim=2048,
                      expires_at=time.time() + 600)
    with caplog.at_level("WARNING", logger="truelens.worker"):
        outcome = worker.run_group(_request(["ela"], sink=sink,
                                           write_overlays=True)).outcomes[0]
    assert outcome.status == "ok" and outcome.overlay is None
    assert not (tmp_path.parent / "outside").exists()
    assert any("refused overlay path" in r.message for r in caplog.records)


def test_a_disabled_sink_writes_nothing_even_with_a_visual(tmp_path, stub, events):
    stub("ela", lambda *_a, **_k: (0.5, _visual(), {}))
    outcome = worker.run_group(_request(["ela"], sink=_sink(root=str(tmp_path),
                                                            enabled=False))).outcomes[0]
    assert outcome.overlay is None and list(tmp_path.iterdir()) == []


def test_an_overlay_write_that_fails_does_not_lose_the_score(tmp_path, stub, events, monkeypatch):
    stub("ela", lambda *_a, **_k: (0.5, _visual(), {}))

    def refuse(*_args, **_kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("app.core.scratch.write_overlay", refuse)
    outcome = worker.run_group(_request(["ela"], sink=_sink(root=str(tmp_path)),
                                        write_overlays=True)).outcomes[0]
    assert outcome.status == "ok" and outcome.score == 0.5 and outcome.overlay is None


# ------------------------------------------------------------------ child hygiene


def test_a_heavy_technique_is_refused_inside_a_spawn_child(events):
    """Torch in a child means N copies of a 300 MB runtime for no reason."""
    worker._IN_CHILD = True
    outcome = worker.run_group(_request(["cnn"])).outcomes[0]
    assert outcome.status == "error" and outcome.error_code == "misrouted"
    assert "parent" in outcome.error_msg


def test_the_same_name_is_not_refused_in_the_parent(stub, events):
    """The refusal is about the process, not the technique."""
    calls = []
    stub("cnn", lambda *_a, **_k: (calls.append(1) or 0.4, None, {"available": True}))
    assert worker.run_group(_request(["cnn"])).outcomes[0].status == "ok"
    assert calls == [1]


def test_init_child_installs_the_queue_and_the_config():
    queue = SimpleQueue()
    worker.init_child(queue, {"redact_pii": False}, 2)
    assert worker._PROGRESS_QUEUE is queue
    assert worker._IN_CHILD is True
    assert worker._CONFIG == {"redact_pii": False}
    assert worker._THREADS == 2


def test_a_child_emits_through_the_queue_it_was_given(stub):
    queue = SimpleQueue()
    worker.init_child(queue, dict(CONFIG), 1)
    stub("metadata", lambda *_a, **_k: (0.1, None, {}))
    worker.run_group(_request(["metadata"]))
    got = [queue.get(timeout=1), queue.get(timeout=1)]
    assert [(e.name, e.status) for e in got] == [("metadata", "running"), ("metadata", "done")]


def test_a_broken_queue_does_not_stall_the_technique(stub):
    class Broken:
        def put_nowait(self, _item):
            raise ValueError("Queue is closed")
    worker.init_child(Broken(), dict(CONFIG), 1)
    stub("metadata", lambda *_a, **_k: (0.1, None, {}))
    assert worker.run_group(_request(["metadata"])).outcomes[0].status == "ok"


# ---------------------------------------------------------------- import closure


def _top_level_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            found.add(node.module or "")
    return found


#: Modules a spawn child must not import transitively. ``torch`` is the memory and startup cost the
#: ``learned`` group exists to avoid; ``app.config`` and ``app.core.store`` drag pydantic-settings
#: (a .env read) and a Redis client into every worker.
FORBIDDEN_TOP_LEVEL = ("torch", "torchvision", "app.config", "app.core.store",
                       "app.core.orchestrator", "pydantic_settings", "redis")


@pytest.mark.parametrize("module", sorted(p.stem for p in TECHNIQUE_DIR.glob("*.py")))
def test_a_technique_module_keeps_expensive_imports_inside_the_function(module):
    imports = _top_level_imports(TECHNIQUE_DIR / f"{module}.py")
    offending = {m for m in imports if any(m == f or m.startswith(f + ".")
                                           for f in FORBIDDEN_TOP_LEVEL)}
    assert not offending, f"{module}.py imports {sorted(offending)} at module level"


def test_the_registry_imports_no_technique_module():
    """The registry is in every child's closure; a technique import there defeats the routing."""
    from app.analysis import registry
    imports = _top_level_imports(Path(registry.__file__))
    assert not [m for m in imports if "techniques" in m], imports


def test_the_worker_module_itself_stays_light():
    """Everything ``worker`` imports at module level is paid for by every spawn child."""
    imports = _top_level_imports(WORKER_SOURCE)
    assert "torch" not in imports and "pydantic" not in imports
    assert "cv2" not in imports, "cv2 belongs to the techniques, not to the dispatcher"


def test_sanitise_keeps_one_line_and_no_filesystem():
    assert "D:\\app" not in worker._sanitise(RuntimeError("boom at D:\\app\\x.py"))
    assert worker._sanitise(ValueError("")) == "ValueError"
    long = worker._sanitise(RuntimeError("x" * 900))
    assert len(long) <= 200 and "\n" not in long
    assert worker._sanitise(RuntimeError("/etc/passwd could not open")) == \
        "RuntimeError: <path> could not open"
