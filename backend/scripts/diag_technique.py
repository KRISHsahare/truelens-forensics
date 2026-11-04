"""Run every technique against one fixture, in-process, printing full tracebacks.

``run_group`` swallows exceptions into a sanitised one-line message, which is what the API must
return but is useless for debugging. This binds the same arguments and calls the same ``run()``
without the guard, so a failure shows its real traceback.

    uv run python scripts/diag_technique.py tests/fixtures/images/screenshot.png --traceback
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2

from app.analysis import worker
from app.analysis.registry import BY_NAME, names as technique_names
from app.config import get_settings
from app.core.pipeline import build_arrays
from app.core.security import probe_header


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    data = Path(args.fixture).read_bytes()
    head = probe_header(data, max_pixels=s.max_analysis_pixels)
    arrays = build_arrays(data, head, analysis_max_dim=s.analysis_max_dim,
                          native_max_dim=s.native_max_dim)
    print(f"{args.fixture}\n  {arrays.mime} {arrays.orig_w}x{arrays.orig_h} "
          f"bpp={arrays.bpp} est_q={arrays.est_jpeg_quality} "
          f"native_box={arrays.native_box} scale={arrays.analysis_scale} "
          f"round={arrays.round_dimensions}")

    cfg = {
        "redact_pii": s.redact_pii,
        "face_model_path": s.yunet_model_path,
        "copy_move_features": s.copy_move_features,
        "cnn_tta": s.cnn_tta,
        "gradcam_min_ai_score": s.gradcam_min_ai_score,
        "clip_probe_enabled": s.enable_clip_probe,
    }
    worker.configure_parent(cfg, s.torch_threads)

    failures = 0
    for name in technique_names():
        if args.only and name not in args.only:
            continue
        spec = BY_NAME[name]
        t0 = time.perf_counter()
        try:
            bind_args, kwargs = worker._bind(spec, arrays)
            module = importlib.import_module(f"app.analysis.techniques.{spec.module}")
            score, _visual, details = module.run(*bind_args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  {name:14} RAISED {type(exc).__name__}: {exc}")
            if args.traceback:
                import traceback

                traceback.print_exc()
            continue
        ms = int((time.perf_counter() - t0) * 1000)
        shown = "nan" if score is None or not math.isfinite(score) else f"{score:.3f}"
        print(f"  {name:14} {shown:>6}  {ms:>5}ms  "
              f"applicable={details.get('applicable')} available={details.get('available')} "
              f"{(details.get('note') or details.get('reason') or '')[:55]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
