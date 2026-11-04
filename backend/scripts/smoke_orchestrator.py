"""Boot the orchestrator, run the fixtures through it, print what came back.

Exercises the parts a per-technique smoke script cannot: the spawn pool, the live event queue, the
deadlines, fusion assembly into a validated ``AnalysisResult``, overlay signing, the SHA-256 cache and
the store round trip. It drives the orchestrator directly rather than over HTTP — the routes are a
thin shell and have their own tests — so a failure here points at the pipeline, not at FastAPI.

    uv run python scripts/smoke_orchestrator.py --fixture spliced --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2

from app.config import get_settings  # noqa: E402
from app.core.cache import ResultCache  # noqa: E402
from app.core.orchestrator import Orchestrator, Rejected  # noqa: E402
from app.core.signing import OverlaySigner  # noqa: E402
from app.core.scratch import Scratch  # noqa: E402
from app.core.store import build_store  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tests", "fixtures", "images")

#: What each fixture must come back as. Deliberately narrower than the fusion smoke's expectations:
#: here we are checking that the *plumbing* delivers a coherent result, so the score ranges are wide
#: and the assertions are about structure. Verdict equality is still checked, because a pipeline that
#: reports "authentic" for a spliced image through the API layer is a product bug, not a plumbing bug.
EXPECTED: dict[str, tuple[str, float, float]] = {
    # fixture: (verdict, min overall, max overall)
    "spliced": ("edited", 0.50, 0.95),
    "copy_move": ("edited", 0.50, 0.95),
    "double_compressed": ("edited", 0.50, 0.95),
    "authentic": ("authentic", 0.00, 0.35),
    "no_exif": ("authentic", 0.00, 0.35),
    "generator_metadata": ("ai_generated", 0.70, 1.00),
    "heavy_compression": ("inconclusive", 0.00, 0.35),
    "screenshot": ("inconclusive", 0.00, 0.35),
    "ai_generated": ("inconclusive", 0.00, 0.60),
}


def _load(name: str) -> bytes:
    for ext in (".jpg", ".png"):
        path = os.path.join(FIXTURE_DIR, name + ext)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
    raise FileNotFoundError(f"{name} is not in {FIXTURE_DIR}")


async def _wait_for(orch: Orchestrator, job_id: str, timeout_s: float = 90.0) -> dict:
    """Poll the store like a client does, so the publish path is exercised too."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        record = await orch.fetch(job_id)
        if record is None:
            raise AssertionError(f"job {job_id} vanished from the store")
        progress = round(record.progress, 2)
        if last != progress:
            last = progress
            done = [n for n, s in record.stages.items() if s == "done"]
            print(f"    progress {progress:>5.2f}  done={sorted(done)}")
        if record.is_terminal():
            if record.status != "succeeded":
                raise AssertionError(f"job failed: {record.error}")
            return record.result
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s")


async def run_one(orch: Orchestrator, name: str, *, verbose: bool) -> tuple[bool, str]:
    data = _load(name)
    expected, lo, hi = EXPECTED[name]
    t0 = time.perf_counter()
    record = await orch.create_job(data, client="smoke", label=name + ".jpg")
    result = await _wait_for(orch, record.id)
    elapsed = time.perf_counter() - t0

    problems: list[str] = []
    verdict = result["verdict"]
    overall = result["overall"]
    # A terminal job settles every stage. Outcomes and stage events are separate deliveries, and a
    # group that returns before its last event lands would otherwise freeze a technique at
    # "running" on a job the client has been told is finished.
    unsettled = sorted(n for n, s in record.stages.items() if s in ("pending", "running"))
    if unsettled:
        problems.append(f"terminal job left {unsettled} unsettled")
    if record.progress != 1.0:
        problems.append(f"progress {record.progress:.2f} on a finished job")
    if verdict != expected:
        problems.append(f"verdict {verdict!r}, expected {expected!r}")
    if not (lo <= overall <= hi):
        problems.append(f"overall {overall:.3f} outside [{lo}, {hi}]")
    # The invariant the whole honesty story rests on: every registered technique is accounted for.
    missing = set(record.stages) - set(result["techniques"])
    if missing:
        problems.append(f"no technique view for {sorted(missing)}")
    if result["sha256"] != record.sha256:
        problems.append("sha256 mismatch between record and result")
    for view in result["techniques"].values():
        if view["score"] is not None and not (0.0 <= view["score"] <= 1.0):
            problems.append(f"{view['name']} score out of range: {view['score']}")
        if view["measured"] and view["score"] is None:
            problems.append(f"{view['name']} claims to have measured nothing at all")
        # The other half of the same rule, and the one a UI colours: silence must arrive as ``null``,
        # because a 0.0 renders as a green "this check passed" bar for a check that had nothing to say.
        if not view["measured"] and view["score"] is not None:
            problems.append(f"{view['name']} did not measure but reports {view['score']:.3f}")
        if view["score"] is None and view["band"] != "not_measured":
            problems.append(f"{view['name']} has no score but is banded {view['band']!r}")
    if not (0.0 <= result["reliability"] <= 1.0):
        problems.append("reliability out of range")
    if not result["reasons"]:
        problems.append("no reasons supplied for the verdict")
    if not result["limitations"]:
        problems.append("limitations block is empty")

    if verbose:
        print(f"    ai={result['ai_score']:.3f} edited={result['edited_score']:.3f} "
              f"overall={overall:.3f} reliability={result['reliability']:.3f} "
              f"band={result['band']} fusion={result['fusion']['source']}")
        for n, v in sorted(result["techniques"].items(),
                           key=lambda kv: -(kv[1]["score"] or 0)):
            mark = "measured" if v["measured"] else ("ran" if v["ran"] else "silent")
            score = "  n/a" if v["score"] is None else f"{v['score']:.3f}"
            print(f"      {n:<14} {score}  {mark:<8} {v['status']:<7} "
                  f"{'overlay' if v['overlay_url'] else ''}")
        for flag in result["flags"]:
            print(f"      flag[{flag['kind']}] {flag['id']}"
                  + (f" -> {flag['target']}={flag['floor']}" if flag["target"] else ""))
        for pen in result["reliability_detail"]["reasons"]:
            print(f"      reliability[-{pen['cost']:.2f}] {pen['reason'][:70]}")
        overlays = [v["overlay_url"] for v in result["techniques"].values() if v["overlay_url"]]
        print(f"      timings: total={result['timings']['total_ms']}ms "
              f"decode={result['timings']['decode_ms']}ms "
              f"assembly={result['timings']['assembly_ms']}ms")
        print(f"      signed overlays: {len(overlays)}")
        for reason in result["reasons"]:
            print(f"      why: {reason[:100]}")

    if elapsed > 20.0:
        problems.append(f"took {elapsed:.1f}s, far outside the budget")
    note = f"{elapsed:.2f}s"
    return (not problems), (note + (" | " + "; ".join(problems) if problems else ""))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--serial", action="store_true", help="force ANALYZER_PARALLEL=0")
    args = ap.parse_args()

    if args.serial:
        os.environ["ANALYZER_PARALLEL"] = "0"

    settings = get_settings()
    scratch = Scratch(settings.scratch_root or None)
    store = build_store(settings)
    signer = OverlaySigner(settings.overlay_signing_key, ttl_s=settings.overlay_ttl_s)
    cache = ResultCache(ttl_s=settings.result_cache_ttl_s,
                        max_entries=settings.result_cache_max_entries)
    orch = Orchestrator(settings, store=store, cache=cache, signer=signer, scratch=scratch)

    names = args.fixture or sorted(EXPECTED)
    await orch.startup()
    failures = 0
    try:
        for name in names:
            print(f"\n=== {name} ({settings.perf_profile}, "
                  f"{'pool' if settings.analyzer_parallel else 'threads'}) ===")
            ok, note = await run_one(orch, name, verbose=args.verbose)
            print(f"    {'PASS' if ok else 'FAIL'}  {note}")
            failures += 0 if ok else 1

        # The cache is only observable if a second upload of the same bytes skips the workers. A file
        # that is not in the fixture loop is used so this section starts cold and exercises
        # compute -> store -> replay, rather than replaying something already computed above.
        print("\n=== cache replay ===")
        data = _load("small")
        first = await orch.create_job(data, client="smoke")
        await _wait_for(orch, first.id)
        if first.from_cache:
            print("    FAIL  the first upload of this file was already cached")
            failures += 1
        second = await orch.create_job(data, client="smoke")
        if not second.from_cache:
            print("    FAIL  second identical upload was recomputed")
            failures += 1
        else:
            replay = await orch.fetch(second.id)
            urls = [v["overlay_url"] for v in replay.result["techniques"].values()
                    if v["overlay_url"]]
            still_signed = all(signer.check_url(u) for u in urls)
            same_sha = replay.sha256 == first.sha256
            print(f"    {'PASS' if still_signed and same_sha else 'FAIL'}  served from cache, "
                  f"{len(urls)} overlay URL(s) re-signed, valid={still_signed} "
                  f"sha_match={same_sha}")
            failures += 0 if (still_signed and same_sha) else 1

        # A non-image must be refused before any job exists.
        print("\n=== rejection ===")
        for payload, why in [(b"%PDF-1.4\n", "PDF disguised as an image"),
                             (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "truncated PNG"),
                             (b"", "empty body")]:
            try:
                await orch.create_job(payload, client="smoke")
            except Rejected as exc:
                print(f"    PASS  {why}: {exc.status} {exc.code}")
            except Exception as exc:  # noqa: BLE001
                print(f"    FAIL  {why}: raised {type(exc).__name__}: {exc}")
                failures += 1
            else:
                print(f"    FAIL  {why}: accepted")
                failures += 1
    finally:
        await orch.shutdown()
        await store.close()
        scratch.discard()

    print(f"\n{'all fixtures passed' if not failures else f'{failures} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
