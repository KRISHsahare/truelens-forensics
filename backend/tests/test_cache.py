"""The SHA-256 result cache: replay, TTL, LRU, and the overlay deadline that outranks both.

Every analyser here is a deterministic function of the bytes, which is the only reason keying on the
digest is safe — no analyser reads the clock, the filename or the job id. What the cache must not do
is outlive the PNGs its entries point at: ``Settings`` refuses a result TTL longer than the overlay
TTL, and this module is the second half of that promise, re-signing each replayed URL so it expires
no later than the files do and degrading to "scores without pictures" once they are gone.
"""

from __future__ import annotations

import time

import pytest

from app.core.cache import ResultCache
from app.core.job import new_job_id
from app.core.signing import OverlaySigner

SHA = "a" * 64
OTHER = "b" * 64
KEY = "unit-test-key"


@pytest.fixture()
def signer() -> OverlaySigner:
    return OverlaySigner(KEY, ttl_s=540)


def _result(overlay_url: str | None = None) -> dict:
    return {"verdict": "edited", "overall": 0.62,
            "techniques": {"ela": {"score": 0.7, "overlay_url": overlay_url},
                           "noise": {"score": 0.3, "overlay_url": None}},
            "timings": {"total_ms": 4200}, "notes": []}


@pytest.fixture()
def cache() -> ResultCache:
    return ResultCache(ttl_s=300, max_entries=8)


# ----------------------------------------------------------------------- replay


def test_a_stored_analysis_is_served_again_with_a_working_overlay_url(cache, signer):
    job = new_job_id()
    now = time.time()
    cache.put(SHA, _result(), {"ela": f"{job}/ela"}, overlay_deadline=now + 600)
    hit = cache.get(SHA, signer, now=now)
    assert hit is not None
    assert hit.result["verdict"] == "edited"
    url = hit.result["techniques"]["ela"]["overlay_url"]
    assert signer.check_url(url, now=now)
    assert hit.overlays_available == ["ela"]
    assert not hit.overlays_expired


def test_an_unknown_digest_is_a_miss_and_says_so(cache, signer):
    assert cache.get("c" * 64, signer) is None
    assert cache.stats()["misses"] == 1


def test_an_empty_digest_is_not_stored(cache, signer):
    """A file with no sha has no identity; caching it would serve one answer to every upload."""
    cache.put("", _result(), {}, overlay_deadline=time.time() + 60)
    assert cache.stats()["entries"] == 0


def test_a_replay_does_not_pretend_the_timings_are_fresh(cache, signer):
    """The honesty the module docstring promises: the numbers came from an earlier run."""
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 600)
    hit = cache.get(SHA, signer)
    assert hit.result["timings"] == {"total_ms": 4200}


def test_the_cached_answer_is_independent_of_the_returned_copy(cache, signer):
    """A client-facing dict handed out twice must not be one object with two owners."""
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 600)
    first = cache.get(SHA, signer).result
    first["techniques"]["ela"]["score"] = 0.99
    assert cache.get(SHA, signer).result["techniques"]["ela"]["score"] == 0.7


def test_a_mutation_of_the_callers_dict_afterwards_does_not_rewrite_the_answer(cache, signer):
    payload = _result()
    cache.put(SHA, payload, {}, overlay_deadline=time.time() + 600)
    payload["verdict"] = "authentic"
    assert cache.get(SHA, signer).result["verdict"] == "edited"


# ---------------------------------------------------------------------- the TTL


def test_an_entry_past_its_ttl_is_not_servable_and_is_gone(cache, signer):
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 600)
    assert cache.get(SHA, signer, now=time.time() + 301) is None
    # And not revived: a later read at an earlier clock still finds nothing.
    assert cache.get(SHA, signer, now=time.time()) is None
    assert cache.stats()["entries"] == 0


def test_the_ttl_boundary_is_exclusive(cache, signer):
    """At exactly ``ttl_s`` the entry is dead, so no two components can disagree by a second."""
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 10_000)
    stored_at = cache._entries[SHA][0]
    assert cache.get(SHA, signer, now=stored_at + cache.ttl_s - 1) is not None
    assert cache.get(SHA, signer, now=stored_at + cache.ttl_s) is None


def test_reading_an_expired_entry_counts_as_a_miss(cache, signer):
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 600)
    assert cache.get(SHA, signer, now=time.time() + 400) is None
    stats = cache.stats()
    assert stats["hits"] == 0 and stats["misses"] == 1


# ----------------------------------------------------------------- overlay expiry


def test_a_replay_after_the_files_are_gone_returns_scores_and_says_the_pictures_expired(
        cache, signer):
    job = new_job_id()
    now = 1_000_000.0
    cache.put(SHA, _result(), {"ela": f"{job}/ela"}, overlay_deadline=now + 60)
    hit = cache.get(SHA, signer, now=now + 120)
    assert hit is not None, "the verdict still stands; only the images went"
    assert hit.result["techniques"]["ela"]["overlay_url"] is None
    assert hit.expired_overlays == ["ela"]
    assert hit.overlays_expired
    assert any("expired" in note for note in hit.result["notes"])


def test_the_note_is_not_repeated_by_replays(cache, signer):
    job = new_job_id()
    now = 1_000_000.0
    cache.put(SHA, _result(), {"ela": f"{job}/ela"}, overlay_deadline=now + 10)
    notes = [cache.get(SHA, signer, now=now + 500).result["notes"] for _ in range(3)]
    assert all(len(n) == 1 for n in notes), notes
    assert "expired" in notes[0][0]


def test_a_re_signed_url_never_outlives_the_file_it_points_at(cache, signer):
    """The clamped TTL is the whole reason a cache entry cannot advertise a deleted PNG."""
    from urllib.parse import parse_qs, urlparse

    job = new_job_id()
    now = 1_000_000.0
    deadline = now + 40  # shorter than the signer's own 540
    cache.put(SHA, _result(), {"ela": f"{job}/ela"}, overlay_deadline=deadline)
    url = cache.get(SHA, signer, now=now).result["techniques"]["ela"]["overlay_url"]
    exp = int(parse_qs(urlparse(url).query)["exp"][0])
    assert exp <= deadline
    assert signer.check_url(url, now=now)
    assert not signer.check_url(url, now=deadline)


def test_a_reference_to_a_technique_that_is_not_in_the_result_is_ignored(cache, signer):
    job = new_job_id()
    cache.put(SHA, _result(), {"ghost": f"{job}/ghost"}, overlay_deadline=time.time() + 600)
    hit = cache.get(SHA, signer)
    assert hit.overlays_available == [] and hit.expired_overlays == []


# ---------------------------------------------------------------------------- LRU


def test_the_cache_is_bounded_and_recency_based(signer):
    cache = ResultCache(ttl_s=300, max_entries=2)
    now = time.time()
    for sha in (SHA, OTHER):
        cache.put(sha, _result(), {}, overlay_deadline=now + 600)
    assert cache.get(SHA, signer) is not None          # SHA is now the most recent
    cache.put("c" * 64, _result(), {}, overlay_deadline=now + 600)
    assert cache.stats()["entries"] == 2
    assert cache.get(OTHER, signer) is None, "the least recently used went, not the oldest written"
    assert cache.get(SHA, signer) is not None


def test_re_putting_a_digest_refreshes_it_without_doubling_it(cache, signer):
    now = time.time()
    cache.put(SHA, _result("u"), {"ela": f"{new_job_id()}/ela"}, overlay_deadline=now + 60)
    cache.put(SHA, _result(), {}, overlay_deadline=now + 600)
    assert cache.stats()["entries"] == 1
    assert cache.get(SHA, signer, now=now + 100).expired_overlays == []


def test_a_non_positive_capacity_is_clamped_to_one(signer):
    cache = ResultCache(ttl_s=300, max_entries=0)
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 60)
    cache.put(OTHER, _result(), {}, overlay_deadline=time.time() + 60)
    assert cache.stats()["entries"] == 1
    assert cache.get(SHA, signer) is None


# ------------------------------------------------------------------- invalidation


def test_dropping_a_job_invalidates_every_entry_that_pointed_at_its_files(cache, signer):
    job_a, job_b = new_job_id(), new_job_id()
    now = time.time()
    cache.put(SHA, _result(), {"ela": f"{job_a}/ela"}, overlay_deadline=now + 600)
    cache.put(OTHER, _result(), {"ela": f"{job_b}/ela"}, overlay_deadline=now + 600)
    assert cache.drop_overlays_for(job_a) == 1
    assert cache.get(SHA, signer) is None
    assert cache.get(OTHER, signer) is not None


def test_dropping_an_unknown_job_is_not_an_error(cache, signer):
    assert cache.drop_overlays_for(new_job_id()) == 0


def test_clear_empties_the_cache_but_keeps_the_counters(cache, signer):
    cache.put(SHA, _result(), {}, overlay_deadline=time.time() + 60)
    cache.get(SHA, signer)
    cache.clear()
    stats = cache.stats()
    assert stats["entries"] == 0 and stats["hits"] == 1


def test_hit_rate_is_reported_as_none_before_any_traffic(signer):
    assert ResultCache(ttl_s=1).stats()["hit_rate"] is None
