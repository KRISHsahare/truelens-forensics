# Architecture notes

The shape of the system is on the [front page](../README.md). This document is about the four
decisions that are not visible in that diagram and that each exist because the obvious alternative
failed: the process model, the resolution tracks, the deadline, and the way a job's answer is
assembled from work that may not all have arrived.

## The process model

Python's GIL makes threads useless for the CPU-bound part of an analysis, so the analysers run in a
`ProcessPoolExecutor`. That choice has one expensive consequence and one dangerous one.

**Expensive:** under `spawn` — which is forced everywhere, because `fork` plus a threaded parent is a
way to deadlock in a child that inherited a lock it never took — every child re-imports the worker
module and everything it touches. So `app/analysis/registry.py` contains *declarations only*:
strings, numbers, and binding descriptors. It imports no technique module. If it did, every child
would import torch before deciding whether it needed to, costing 1.5-3 s and a few hundred MB per
worker on machines where most jobs never run the CNN. Technique modules are imported by name inside
`worker.py`, in the child, only for the tasks that child was given.

**Dangerous:** torch in a forked or spawned child is the same mistake multiplied. So the CNN never
enters the pool. `Orchestrator._submit_groups` routes exactly one group — `learned` — to a dedicated
single-thread executor in the parent process, where `CnnRuntime` is a process-wide singleton loaded
once at startup. The other three groups (`container`, `native`, `analysis`) go to the children.

Thread budgets are pinned to 1 per process by `app/envbootstrap.py`, which must be imported before
numpy, cv2 or torch: with a pool of N workers each letting BLAS/OpenMP size itself to `cpu_count()`,
total thread count is N×C and the pooled run is *slower* than the serial one. Torch gets its own
budget because CNN inference is the one stage that genuinely benefits from intra-op threads, and it
runs in the parent, never in a worker.

Grouping is by track rather than by technique, so a job submits four futures rather than ten: the
arrays a group needs are pickled once for that group, and `trim_arrays` empties the pixel planes and
the container bytes that group will not read. Scalars are never trimmed — `est_jpeg_quality` is
legitimately `None` for a PNG, and treating a null scalar as a missing one made every non-JPEG file
report `ela` and `noise` as *errored* techniques when the honest answer is that they ran and had
nothing to read.

## Three resolution tracks

One decode produces three views, and which one a technique receives is a correctness requirement, not
a performance knob:

| Track | Built as | For |
|---|---|---|
| `container` | the original bytes | metadata, C2PA — the file as delivered |
| `native` | 8-aligned crop, never resampled, up to `NATIVE_MAX_DIM` | ELA, JPEG ghost, block grid, frequency — every check that reads compression artefacts |
| `analysis` | resampled to ≤`ANALYSIS_MAX_DIM` | noise, copy-move, face, CNN |

Resampling destroys exactly what the `native` techniques measure: an 8×8 block phase is not preserved
by an arbitrary resize, and a re-encode erases the second compression history that `jpeg_ghost` looks
for. Running those on the downscaled copy is not slower-but-equivalent; it silently returns a clean
result for a doctored file. The binding is declarative — `registry.py` records the track, and the
worker's binder hands each group the fields its members read — so `backend/tests/test_analysers.py`
drives techniques through `worker.run_group` rather than calling their `run()` directly. A test that
imported the module and passed an array itself would have passed through every defect in the binder,
the trimmer and the outcome wrapper, which is where the interesting bugs lived.

## Deadlines bound the wait, not the work

`JOB_SOFT_DEADLINE_S` is passed into each group as `deadline_at`, so a child can stop *starting*
techniques it has no time left for. `JOB_HARD_DEADLINE_S` is the parent's patience: on expiry the
unfinished futures are cancelled, which stops queued work and abandons running work whose results are
then dropped.

What survives is assembled anyway, and the result says so: the missing techniques appear as checks
that did not complete, reliability is penalised for them, and the verdict is computed from what was
measured. A deadline cannot make a slow machine faster; it converts "the request hung" into "three of
ten checks arrived, and here is what they say". `smoke_orchestrator.py --serial` exercises the
degraded path with `ANALYZER_PARALLEL=0`.

Pool recovery is part of the same story: a child that dies (OOM, a segfault inside cv2) is noticed,
the pool is rebuilt, and the groups that had not returned are re-submitted once. `_submit_groups`
mutates the futures dict in place rather than returning a new one precisely so that a retry cannot
run the learned group twice for one job.

## Assembly, cache, and the overlay lifetime

`_assemble` turns outcomes into the response: scores, explanations from the same `Explanation`
objects the UI renders (one interpreter of a result dict, not two), evidence flags, reliability,
limitations, timings, and the signed overlay URLs.

The result cache is keyed on the SHA-256 of the uploaded bytes, which is safe because every analyser
here is a deterministic function of those bytes — none reads the clock, the filename or the job id. On
a hit, a new job id and new timestamps are minted and overlay URLs are re-signed, but the cached entry
records the wall-clock deadline after which *its* PNGs are gone: `RESULT_CACHE_TTL_S` must stay
strictly below `OVERLAY_TTL_S`, and once the deadline has passed the cache returns the scores with
overlays stripped and a note saying so. That is why a cache hit can legitimately show overlay URLs
bearing a different job id than the page's own — the pictures come from the run that measured them.

`_cacheable` refuses to cache a run that had a *skipped* or *timed-out* technique: a skip is a
function of load, not of the file, and caching it would pin a thin analysis to a hash that could
support a full one — the next uploader would inherit the first user's busy minute. An *errored*
technique is different and is cached on purpose: it failed because of what is in the file, so it will
fail identically, and replaying that is the honest answer rather than a bug.

## Where the honesty is enforced

Not in prose. In code that fails loudly:

- `features.py` is the single definition of the 48-column vector, and the fusion blob stores the
  names it was trained on; `_Learned.load()` returns `None` on a mismatch, which falls back to the
  labelled expert prior rather than scoring with the wrong columns.
- `fusion.source` and `calibrated` are fields of the response, so "these numbers came from a prior,
  not a model" reaches the UI and the PDF.
- `Settings.validate_prod` refuses to start with an unsigned overlay key, a cache that outlives its
  files, or a web concurrency the store cannot support.
- `metrics.json` is validated by the writer that produces it (`ml/truelens_ml/evalout.py`) and again
  by `ml/scripts/check_artifacts.py`, because a `NaN` that Python writes and `JSON.parse` rejects
  shows up as a blank page rather than an error.
