"""Write a corpus tree: either the real layout for you to fill, or a generated one to verify with.

    python ml/scripts/make_corpus.py --out ml/.smoke/corpus --images 240 --generated
    python ml/scripts/make_corpus.py --out datasets --scaffold

``--scaffold`` creates the empty ``real/`` and ``fake/<generator>/`` directories the loaders expect
and stops; that is the command a person running this for real executes first, and having it in the
same script as the generator keeps the layout definition in one place.

``--generated`` writes ``truelens_ml.synthetic`` frames in the documented shape — groups of
siblings, mixed containers, mixed sizes — so ``corpus.discover`` and the group-aware split are
exercised on data that looks like data rather than on a flat list of files.

The generated corpus is a pipeline check. See ``truelens_ml/synthetic.py`` for why its metrics
must not be read as accuracy on photographs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import add_paths  # noqa: E402

add_paths()

from truelens_ml.synthetic import render  # noqa: E402


def scaffold(out: Path, *, families: list[str]) -> list[str]:
    created: list[str] = []
    (out / "real" / "put-camera-photos-here").mkdir(parents=True, exist_ok=True)
    for family in families:
        (out / "fake" / family).mkdir(parents=True, exist_ok=True)
        created.append(f"fake/{family}")
    created.append("real")
    return created


def generate(out: Path, *, images: int, per_group: int, size: int, seed: int,
              families: list[str]) -> dict[str, int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    counts = {"real": 0, "fake": 0}
    # One group index shared by both classes: a generated "burst" is the closest analogue to the
    # sibling frames a real corpus has, and the splitter needs groups on both sides to test against.
    groups = max(1, images // 2 // max(1, per_group))
    for kind in ("real", "fake"):
        for g in range(groups):
            root = out / kind / f"g{g:04d}" if kind == "real" else \
                out / "fake" / families[g % len(families)] / f"g{g:04d}"
            root.mkdir(parents=True, exist_ok=True)
            for n in range(per_group):
                img = render(rng, kind=kind, size=size)
                _write(root / f"{n:02d}.jpg" if (g + n) % 3 else root / f"{n:02d}.png", img)
                counts[kind] += 1
    return counts


def _write(path: Path, rgb) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="corpus root to write")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generated", action="store_true", help="fill it with synthetic frames")
    mode.add_argument("--scaffold", action="store_true", help="create the empty layout only")
    ap.add_argument("--images", type=int, default=240, help="total generated images (both classes)")
    ap.add_argument("--per-group", type=int, default=4, help="sibling images per group directory")
    ap.add_argument("--size", type=int, default=224, help="rendered edge length")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--families", default="midjourney,stable-diffusion",
                    help="comma-separated generator family directories under fake/")
    args = ap.parse_args()

    out = Path(args.out)
    families = [f.strip() for f in args.families.split(",") if f.strip()] or ["generated"]
    if args.scaffold:
        for line in scaffold(out, families=families):
            print(f"created {out / line}")
        print("\nDrop camera photographs under real/ and generator output under fake/<family>/.")
        print("ml/README.md documents the accepted layouts and the licensing terms that matter.")
        return 0

    counts = generate(out, images=args.images, per_group=args.per_group, size=args.size,
                      seed=args.seed, families=families)
    print(f"wrote {counts['real']} real + {counts['fake']} fake images to {out}")
    print("This corpus is a pipeline check, not an evaluation of the product.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
