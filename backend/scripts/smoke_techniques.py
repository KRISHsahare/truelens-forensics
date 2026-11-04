"""Run every classical analyser against every fixture and print the score matrix.

This is the fastest way to catch a detector that is inverted, saturating, or silently returning
zero. It is a diagnostic, not a test: the assertions live in ``tests/``.

    python scripts/smoke_techniques.py
    python scripts/smoke_techniques.py --only spliced heavy_compression --dump noise block_grid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.envbootstrap  # noqa: F401,E402  must precede numpy/cv2

import numpy as np  # noqa: E402

from app.core.pipeline import build_arrays  # noqa: E402
from app.core.security import probe_header  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "tests", "fixtures", "images")

EXPECTED = {
    # fixture: {technique: direction}  "hi" = should fire, "lo" = should stay quiet
    # `spliced` is deliberately absent from jpeg_ghost: its pasted region has been through two
    # q88 encodes, which converges it onto the plate's lattice and erases the q62 periodicity.
    # That is correct behaviour, not a miss — double_compressed is the fixture that exercises it.
    "spliced": {"ela": "hi", "noise": "hi", "block_grid": "hi"},
    "copy_move": {"copy_move": "hi"},
    "double_compressed": {"jpeg_ghost": "hi", "ela": "hi", "block_grid": "lo"},
    "ai_generated": {"frequency": "hi"},
    "generator_metadata": {"metadata": "hi"},
    "authentic": {"ela": "lo", "noise": "lo", "copy_move": "lo", "block_grid": "lo",
                  "jpeg_ghost": "lo", "frequency": "lo"},
    "heavy_compression": {"noise": "lo"},
    "screenshot": {"ela": "lo"},
    "no_exif": {"ela": "lo", "noise": "lo"},
}
FIRE = 0.35
QUIET = 0.25


def _load(name: str):
    for ext in (".jpg", ".png"):
        path = os.path.join(FIXTURE_DIR, name + ext)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            probe = probe_header(data)
            return build_arrays(data, probe, analysis_max_dim=2048, native_max_dim=2560)
    raise FileNotFoundError(name)


CALLS = {
    "ela": lambda a: ("ela", a.native_rgb_u8, a.mime, a.est_jpeg_quality),
    "metadata": lambda a: ("metadata", a.orig_bytes, a.mime),
    "noise": lambda a: ("noise", a.gray_u8, a.analysis_scale, a.est_jpeg_quality),
    "frequency": lambda a: ("frequency", a.native_gray_u8),
    "copy_move": lambda a: ("copy_move", a.rgb_u8, a.gray_u8, a.analysis_scale),
    "face": lambda a: ("face", a.rgb_u8, a.gray_u8),
    "jpeg_ghost": lambda a: ("jpeg_ghost", a.native_rgb_u8, a.mime, a.native_box),
    "block_grid": lambda a: ("block_grid", a.native_gray_u8, a.mime),
    "c2pa": lambda a: ("c2pa", a.orig_bytes, a.mime),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dump", nargs="*", default=None, help="print details for these techniques")
    args = ap.parse_args()

    from app.analysis.techniques import (block_grid, c2pa_verify, copy_move, ela, face, frequency,
                                         jpeg_ghost, metadata, noise)
    mods = {"ela": ela, "metadata": metadata, "noise": noise, "frequency": frequency,
            "copy_move": copy_move, "face": face, "jpeg_ghost": jpeg_ghost,
            "block_grid": block_grid, "c2pa": c2pa_verify}

    names = args.only or sorted(
        os.path.splitext(f)[0] for f in os.listdir(FIXTURE_DIR)
        if f.endswith((".jpg", ".png"))
    )
    order = list(CALLS)
    fails: list[str] = []
    manifest = json.load(open(os.path.join(FIXTURE_DIR, "manifest.json"), encoding="utf-8"))
    labels = {e["name"]: e["label"] for e in manifest["images"]}

    print(f"{'fixture':22s} {'label':>13s}" + "".join(f"{t[:9]:>10s}" for t in order)
          + f"{'ms':>8s}")
    for name in names:
        arrays = _load(name)
        row: list[str] = []
        t0 = time.perf_counter()
        scores: dict[str, float] = {}
        for tech in order:
            mod_name, *positional = CALLS[tech](arrays)
            try:
                res = getattr(mods[mod_name], "run")(*positional)
                scores[tech] = res.score
                row.append(f"{res.score:10.3f}")
                if args.dump and tech in args.dump:
                    print(f"\n--- {name} / {tech} ---")
                    print(json.dumps(_plain(res.details), indent=2))
            except Exception as exc:  # noqa: BLE001  diagnostics must show, not swallow
                scores[tech] = float("nan")
                row.append(f"{'ERR:' + type(exc).__name__[:5]:>10s}")
                print(f"\n!! {name}/{tech}: {exc}", file=sys.stderr)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{name:22s} {labels.get(name, '?'):>13s}" + "".join(row) + f"{ms:8.0f}")

        for tech, direction in EXPECTED.get(name, {}).items():
            got = scores.get(tech, float("nan"))
            if np.isnan(got):
                fails.append(f"{name}/{tech} errored")
            elif direction == "hi" and got < FIRE:
                fails.append(f"{name}/{tech} = {got:.3f}, expected >= {FIRE}")
            elif direction == "lo" and got > QUIET:
                fails.append(f"{name}/{tech} = {got:.3f}, expected <= {QUIET}")

    print()
    if fails:
        print(f"{len(fails)} expectation(s) not met:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("all expectations met "
          f"(fire >= {FIRE}, quiet <= {QUIET})")


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
