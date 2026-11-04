"""Thread-budget bootstrap. MUST be imported before numpy / cv2 / torch.

Every numerical library independently sizes its own thread pool to ``os.cpu_count()``.
With a process pool of N workers each spawning C BLAS/OpenMP/cv2 threads, total thread
count is N*C and context switching makes the pooled run *slower* than the serial one.

We therefore pin all of them to 1 thread per process and buy parallelism from the
process pool instead. Torch gets its own separate budget (see ``lifespan``), because
CNN inference is the one stage that genuinely benefits from intra-op threads and it
runs in the parent process, never in a pool worker.

Setting these in a pool initializer would be too late under ``fork`` (libgomp reads the
env var at first use, in the parent). We force ``spawn`` everywhere, so children
re-import this module from scratch and the values take effect.
"""

from __future__ import annotations

import os

__all__ = ["POOL_MAX_WORKERS", "apply_child_thread_budget"]

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _resolve_pool_size() -> int:
    """Pool size is a *setting*, not ``cpu_count()``.

    On a 1-2 vCPU container a wide pool is pure overhead (pickling + scheduling).
    """
    cpus = os.cpu_count() or 2
    requested = int(os.getenv("POOL_MAX_WORKERS", "8"))
    return max(1, min(requested, cpus - 1))


POOL_MAX_WORKERS = _resolve_pool_size()


def apply_child_thread_budget() -> None:
    """Pin BLAS/OpenMP to a single thread. Idempotent; safe to call from a pool initializer."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"


apply_child_thread_budget()
