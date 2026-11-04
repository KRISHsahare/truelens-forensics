"""The job record: stage transitions are one-way, and progress always reaches its denominator.

Two callbacks race across a thread boundary here — a pool worker saying "this technique is done" and
the watchdog saying "this technique timed out" — and whichever arrives second must not rewrite
history. Without that rule the stepper flickers backwards from Done to Running, and a finished result
gets overwritten by a timeout the client already saw resolve. The tests below are the race, written
as pairs of writes.
"""

from __future__ import annotations

import time

import pytest

from app.core.job import JobRecord, new_job_id

ALL = ["metadata", "ela", "noise", "cnn"]


def record(status: str = "queued", stages: dict[str, str] | None = None) -> JobRecord:
    job = JobRecord(id=new_job_id(), sha256="a" * 64, mime="image/jpeg", size_bytes=10)
    job.status = status
    job.stages = stages if stages is not None else {n: "pending" for n in ALL}
    return job


# ------------------------------------------------------------------------ ids


def test_new_ids_are_unique_and_sorted_by_mint_time():
    ids = [new_job_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert int(ids[0].split("-")[0], 16) <= int(ids[-1].split("-")[0], 16)


# ---------------------------------------------------------------- init_stages


def test_init_stages_gives_progress_a_denominator():
    job = JobRecord(id=new_job_id(), sha256="b" * 64, mime="image/jpeg", size_bytes=1)
    assert job.stages == {}
    job.init_stages(ALL)
    assert job.stages == {n: "pending" for n in ALL}
    assert job.progress == 0.0


# ------------------------------------------------------------------- transitions


@pytest.mark.parametrize("path", [
    ("pending", "running"),
    ("pending", "done"),
    ("running", "done"),
    ("running", "failed"),
    ("pending", "skipped"),
    ("pending", "blocked"),
])
def test_forward_transitions_are_accepted(path):
    first, second = path
    job = record(stages={"ela": first})
    assert job.set_stage("ela", second) is True
    assert job.stages["ela"] == second


@pytest.mark.parametrize("attempt", [
    ("running", "pending"),
    ("done", "running"),
    ("done", "pending"),
    ("failed", "running"),
])
def test_backward_transitions_are_refused(attempt):
    """A late "still running" from a worker must not resurrect a finished stage."""
    current, next_state = attempt
    job = record(stages={"ela": current})
    assert job.set_stage("ela", next_state) is False
    assert job.stages["ela"] == current, "the stage keeps the state it had"


def test_the_first_writer_to_reach_terminal_wins():
    """The race the whole rule exists for, both orderings.

    The result is what the client is shown, so the analyser's "done" beating the watchdog's
    "timed out" is the correct outcome; the reverse ordering must not lose the timeout either.
    """
    analyser_first = record(stages={"ela": "running"})
    assert analyser_first.set_stage("ela", "done")
    assert analyser_first.set_stage("ela", "failed") is False
    assert analyser_first.stages["ela"] == "done"

    watchdog_first = record(stages={"ela": "running"})
    assert watchdog_first.set_stage("ela", "failed")
    assert watchdog_first.set_stage("ela", "done") is False
    assert watchdog_first.stages["ela"] == "failed"


def test_a_stage_for_a_technique_that_was_never_registered_is_refused():
    """The registry is the denominator. An event for a name outside it means a stale worker."""
    job = record()
    assert job.set_stage("some_old_technique", "done") is False
    assert set(job.stages) == set(ALL)


def test_a_provisional_terminal_can_be_corrected_by_real_evidence():
    """The rule is rank-sticky, not simply terminal-sticky.

    ``blocked`` and ``skipped`` are the orchestrator's guess that a technique will not run; a result
    that arrives afterwards is better evidence than the guess, so it is allowed to replace it.
    """
    job = record(stages={"ela": "blocked"})
    assert job.set_stage("ela", "done", ms=5) is True
    assert job.stages["ela"] == "done"
    assert job.set_stage("ela", "skipped") is False, "and the weaker claim cannot re-take it"


def test_stage_ms_is_only_recorded_on_an_accepted_transition():
    job = record(stages={"ela": "done", "noise": "pending"})
    job.set_stage("ela", "running", ms=120)
    assert job.stage_ms == {}, "a refused write must not leave timing behind"
    job.set_stage("noise", "done", ms=88)
    assert job.stage_ms == {"noise": 88}


# --------------------------------------------------------------------- progress


def test_every_terminal_state_counts_towards_completion():
    """``skipped`` and ``blocked`` are still "nothing left to wait for".

    A job that ran out of deadline on half its techniques still has to report 1.0, or the client
    polls a finished job forever.
    """
    job = record(status="succeeded", stages={"metadata": "done", "ela": "failed",
                                             "noise": "skipped", "cnn": "blocked"})
    assert job.progress == 1.0


def test_progress_is_the_fraction_of_settled_stages():
    job = record(stages={"metadata": "done", "ela": "done", "noise": "running", "cnn": "pending"})
    assert job.progress == 0.5


def test_a_record_with_no_stages_still_has_a_defensible_progress():
    """Before the registry seeds, or for a job rejected at ingest: nothing to wait for only once
    the job itself is terminal."""
    assert record(status="queued", stages={}).progress == 0.0
    assert record(status="running", stages={}).progress == 0.0
    assert record(status="succeeded", stages={}).progress == 1.0
    assert record(status="expired", stages={}).progress == 1.0


# ------------------------------------------------------------------ serialisation


def test_the_record_survives_a_round_trip_through_dict():
    """The store is a dict-shaped sink (in-process today, Redis in production).

    A field dropped here reads back as a different job on the next poll — the client would see a
    technique that finished reappear as pending.
    """
    job = record(status="succeeded")
    job.stage_ms = {"ela": 41}
    job.result = {"verdict": "edited"}
    job.from_cache = True
    job.client = "web"
    job.started_at = time.time()
    job.finished_at = time.time()
    restored = JobRecord.from_dict(job.to_dict())
    assert restored.to_dict() == job.to_dict()


def test_a_record_read_from_an_older_shape_does_not_raise():
    """A Redis instance survives a deploy; its keys do not."""
    restored = JobRecord.from_dict({"id": "1-2", "sha256": "c" * 64})
    assert restored.mime == "" and restored.size_bytes == 0
    assert restored.status == "queued" and restored.stages == {}
    assert restored.from_cache is False


def test_the_stages_dict_is_copied_on_read():
    """Two handles on one dict would let a poll response mutate the live job."""
    job = record()
    restored = JobRecord.from_dict(job.to_dict())
    restored.set_stage("ela", "done")
    assert job.stages["ela"] == "pending"


def test_is_terminal_and_age():
    job = record(status="running")
    job.created_at = time.time() - 30
    assert not job.is_terminal()
    assert 29 < job.age() < 31
    job.status = "expired"
    assert job.is_terminal()
    assert job.age(now=job.created_at + 5) == 5.0


def test_error_and_result_are_both_carried():
    """A failed job still needs its ``error`` block readable after the store round trip."""
    job = record(status="failed")
    job.error = {"code": "no_measurements", "message": "nothing ran"}
    assert JobRecord.from_dict(job.to_dict()).error["code"] == "no_measurements"
