"""Synthesise test images with *known* ground truth.

Real labelled datasets (CASIA v2, FaceForensics++, DFDC) are gated multi-GB downloads that need
credentials and signed agreements, so the test suite cannot depend on them. Instead we generate
images whose forgery mechanism is known by construction, which is what a unit test actually
needs: ``spliced.jpg`` must raise the noise and block-grid scores, ``copy_move.jpg`` must raise
the copy-move score, and so on.

These fixtures validate that each analyser responds to the artifact it claims to detect. They do
**not** validate real-world accuracy — that needs the real datasets, and ``ml/`` documents the
one-command path from download to ``metrics.json``.

Usage::

    python tests/fixtures/make_fixtures.py --out tests/fixtures/images
    python tests/fixtures/make_fixtures.py --out ../samples --include-samples
"""

from __future__ import annotations

import argparse
import io
import os
from typing import Any

import numpy as np
from PIL import Image
from PIL.ExifTags import Base as ExifBase

SEED = 20260103

#: A plausible camera EXIF block. Real capture metadata, so the "authentic" fixture does not
#: trip the missing-metadata findings.
CAMERA_EXIF: dict[int, Any] = {
    ExifBase.Make: "NIKON CORPORATION",
    ExifBase.Model: "NIKON D750",
    ExifBase.Software: "Ver.1.30",
    ExifBase.Orientation: 1,
    ExifBase.DateTime: "2025:06:14 11:32:07",
    ExifBase.DateTimeOriginal: "2025:06:14 11:32:07",
    ExifBase.DateTimeDigitized: "2025:06:14 11:32:07",
    ExifBase.ExposureTime: 0.008,
    ExifBase.FNumber: 5.6,
    ExifBase.ISOSpeedRatings: 200,
    ExifBase.FocalLength: 50.0,
    ExifBase.ExposureProgram: 3,
    ExifBase.MeteringMode: 5,
    ExifBase.Flash: 16,
    ExifBase.ColorSpace: 1,
}


# --------------------------------------------------------------------- texture
def natural_image(h: int, w: int, rng: np.random.Generator, *, hue: float = 0.0) -> np.ndarray:
    """A synthetic image with a natural ``1/f`` amplitude spectrum.

    Real photographs are not white noise: their power falls off roughly as ``1/f^2``. Generating
    with that spectrum matters, because the FFT analyser fits exactly this slope — white noise
    would be flagged as anomalous by every technique at once and prove nothing.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    radius = np.hypot(yy - cy, xx - cx) + 1.0
    amplitude = 1.0 / np.power(radius, 1.05)
    amplitude[0, 0] = 0.0

    channels = []
    for c in range(3):
        phase = rng.uniform(0, 2 * np.pi, size=(h, w)).astype(np.float32)
        spectrum = amplitude * np.exp(1j * phase)
        # Suppress the very high frequencies so the result looks like an optical image rather
        # than film grain.
        spectrum *= np.exp(-radius / (0.42 * max(h, w)))
        real = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum))).astype(np.float32)
        real -= real.min()
        real /= real.max() + 1e-8
        channels.append(real)

    img = np.stack(channels, axis=-1)
    # Give it some colour cast and a smooth vignette so it is not grey mush.
    tint = np.array([0.62 + 0.30 * np.cos(hue), 0.66 + 0.26 * np.sin(hue), 0.70],
                    dtype=np.float32)
    img = img * tint
    vignette = 1.0 - 0.32 * (np.hypot(yy - cy, xx - cx) / (np.hypot(cy, cx)))[:, :, None]
    img = np.clip(img * vignette, 0, 1)
    return (img * 255).astype(np.uint8)


def add_structures(img: np.ndarray, rng: np.random.Generator, n: int = 7) -> np.ndarray:
    """Add blobs and edges so keypoint detectors have something stable to lock onto."""
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    for _ in range(n):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = int(rng.integers(24, max(25, w // 6))), int(rng.integers(24, max(25, h // 6)))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        shade = rng.uniform(-70, 70)
        out += shade * blob[:, :, None]
    # Sharp linear features: SIFT needs corners.
    for _ in range(n // 2):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        out = _draw_line(out, (int(x0), int(y0)), (int(x1), int(y1)),
                         float(rng.uniform(-90, 90)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], value: float) -> np.ndarray:
    h, w = img.shape[:2]
    n = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]), 1)
    for i in range(n + 1):
        t = i / n
        x = int(round(p0[0] + (p1[0] - p0[0]) * t))
        y = int(round(p0[1] + (p1[1] - p0[1]) * t))
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = np.clip(img[y, x] + value, 0, 255)
    return img


def add_sensor_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 2.6) -> np.ndarray:
    """Photon shot noise (signal-dependent) plus read noise (constant)."""
    out = img.astype(np.float32)
    shot = rng.normal(0, 1, out.shape).astype(np.float32) * np.sqrt(np.maximum(out, 1) * 0.35)
    read = rng.normal(0, sigma, out.shape).astype(np.float32)
    return np.clip(out + shot + read, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------- save
def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG and decode back, so the array carries that encode's quantisation imprint."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def save_jpeg(img: np.ndarray, path: str, quality: int = 88,
              exif: dict[int, Any] | None = CAMERA_EXIF) -> str:
    pil = Image.fromarray(img)
    kwargs: dict[str, Any] = {"quality": quality, "subsampling": 0, "optimize": False}
    if exif:
        ex = Image.Exif()
        for tag, value in exif.items():
            ex[tag] = value
        kwargs["exif"] = ex.tobytes()
    pil.save(path, "JPEG", **kwargs)
    return path


def save_png(img: np.ndarray, path: str, text_chunks: dict[str, str] | None = None) -> str:
    Image.fromarray(img).save(path, "PNG", pnginfo=_pnginfo(text_chunks))
    return path


def _pnginfo(chunks: dict[str, str] | None):
    if not chunks:
        return None
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for k, v in chunks.items():
        info.add_text(k, v)
    return info


# ------------------------------------------------------------------- fixtures
def make_authentic(h: int = 720, w: int = 960, seed: int = 1) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(SEED + seed)
    img = add_structures(natural_image(h, w, rng, hue=0.7), rng)
    img = add_sensor_noise(img, rng, sigma=2.6)
    return img, "Single-source photograph. Saved as JPEG once, with full camera EXIF."


def make_spliced(h: int = 720, w: int = 960, seed: int = 2) -> tuple[np.ndarray, str]:
    """Paste a region from a *different* photograph into an already-shared one.

    The construction mirrors how splices actually happen, and each detail matters:

    * The plate has **already been through one JPEG encode** (a photo downloaded from a site or
      re-shared by a messenger). The pasted region then gets only the final encode, so it has a
      different compression history from its surroundings. Measured on controlled fixtures, a
      plate and region with the *same* encode count give an ELA in/out ratio of 0.5 — no signal
      at all — while this construction gives 5-16x.
    * The paste offset is deliberately **not** a multiple of 8. Real selections are not aligned
      to the DCT grid, and an 8-aligned paste makes the region's block phase identical to the
      plate's, which silently disables ``block_grid``.
    * The region comes from a source compressed at a *lower* quality, so it carries a coarser
      quantisation imprint than the plate.

    Together these give ``ela``, ``noise``, ``block_grid`` and ``jpeg_ghost`` something real to
    detect, without any of them having to fire on content differences alone.
    """
    rng_base = np.random.default_rng(SEED + seed)
    rng_src = np.random.default_rng(SEED + seed + 500)
    base = add_structures(natural_image(h, w, rng_base, hue=0.4), rng_base)
    base = add_sensor_noise(base, rng_base, sigma=2.6)

    src = add_structures(natural_image(h // 2, w // 2, rng_src, hue=2.4), rng_src, n=9)
    # Different sensor, different noise level.
    src = add_sensor_noise(src, rng_src, sigma=6.5)
    # Source arrived as a downloaded JPEG at a coarser quality.
    src = jpeg_roundtrip(src, quality=62)

    # Plate has already been encoded once at the final quality.
    plate = jpeg_roundtrip(base, quality=88)

    ph, pw = src.shape[:2]
    # Non-8-aligned on purpose: a real marquee selection does not snap to the DCT grid.
    py, px = h // 4 + 5, w // 2 - 77
    ph, pw = min(ph, h - py), min(pw, w - px)
    out = plate.copy()
    out[py:py + ph, px:px + pw] = src[:ph, :pw]
    ground = (f"Region [{px}:{px + pw}, {py}:{py + ph}] pasted from a different image with "
              f"6.5-sigma noise, sourced from a JPEG q62, into a plate already encoded at q88. "
              f"Offset is not 8-aligned.")
    return out, ground


def make_copy_move(h: int = 720, w: int = 960, seed: int = 3) -> tuple[np.ndarray, str]:
    """Clone a patch to another location in the same image (clone-stamp forgery)."""
    rng = np.random.default_rng(SEED + seed)
    img = add_structures(natural_image(h, w, rng, hue=1.4), rng, n=10)
    img = add_sensor_noise(img, rng, sigma=2.4)

    ph, pw = h // 4, w // 5
    sy, sx = h // 2, w // 8
    dy, dx = h // 8, w // 2
    out = img.copy()
    out[dy:dy + ph, dx:dx + pw] = img[sy:sy + ph, sx:sx + pw]
    ground = (f"Patch copied from ({sx},{sy}) to ({dx},{dy}); displacement "
              f"({dx - sx},{dy - sy}) px.")
    return out, ground


def make_ai_generated(h: int = 720, w: int = 960, seed: int = 4) -> tuple[np.ndarray, str]:
    """Simulate the artifacts a generative model leaves.

    Three things, all of which real diffusion/GAN output exhibits: an anomalously smooth
    high-frequency tail, a periodic checkerboard from stride-2 transposed convolution, and no
    camera metadata. This is a *simulation* of those artifacts, not a real generated image —
    which is exactly why the metrics produced from it are labelled as a smoke test.
    """
    rng = np.random.default_rng(SEED + seed)
    img = natural_image(h, w, rng, hue=3.6).astype(np.float32)
    # Over-smooth: generators reproduce low frequencies well and lose fine detail.
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=(1.4, 1.4, 0))

    # Transposed-convolution checkerboard. Stride-2 upsampling with a k*k kernel modulates the
    # gain with period 2 *independently along each axis*, so the artifact has spectral components
    # at (0.5, 0) and (0, 0.5) cycles/px — i.e. on the horizontal and vertical axes at Nyquist,
    # where the analyser scans. A diagonal `(x+y) % 2` pattern puts its energy at (0.5, 0.5), the
    # corner of the spectrum, which an axis scan can never see.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gain = (1.0 + 0.035 * np.cos(np.pi * xx)) * (1.0 + 0.035 * np.cos(np.pi * yy))
    img *= gain[:, :, None]

    img = np.clip(img, 0, 255).astype(np.uint8)
    # Very low, spatially-uniform noise: no real sensor.
    img = add_sensor_noise(img, rng, sigma=0.8)
    ground = ("Synthetic stand-in for a generated image: smoothed spectrum, period-2 "
              "checkerboard, near-zero sensor noise, no EXIF.")
    return img, ground


def make_heavy_compression(h: int = 720, w: int = 960, seed: int = 5) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(SEED + seed)
    img = add_structures(natural_image(h, w, rng, hue=5.0), rng)
    img = add_sensor_noise(img, rng, sigma=2.6)
    return img, "Authentic content saved at JPEG q32 — traces largely destroyed by compression."


def make_small(seed: int = 6) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(SEED + seed)
    img = add_structures(natural_image(128, 128, rng, hue=2.0), rng, n=3)
    img = add_sensor_noise(img, rng, sigma=3.0)
    return img, "128x128 — below the resolution floor for block statistics."


def make_screenshot(h: int = 1080, w: int = 1920, seed: int = 7) -> tuple[np.ndarray, str]:
    """Round dimensions, flat regions, no EXIF, lossless PNG."""
    rng = np.random.default_rng(SEED + seed)
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    for _ in range(14):
        x0, y0 = rng.integers(0, w - 200), rng.integers(0, h - 120)
        bw, bh = rng.integers(80, 400), rng.integers(40, 200)
        colour = rng.integers(40, 240, size=3).astype(np.uint8)
        img[y0:y0 + bh, x0:x0 + bw] = colour
    return img, "Screenshot-like: 1920x1080, flat regions, PNG, no metadata."


def make_no_exif(h: int = 720, w: int = 960, seed: int = 8) -> tuple[np.ndarray, str]:
    """Authentic pixels with EXIF stripped — the social-media case."""
    rng = np.random.default_rng(SEED + seed)
    img = add_structures(natural_image(h, w, rng, hue=0.9), rng)
    img = add_sensor_noise(img, rng, sigma=2.6)
    return img, "Authentic content with EXIF stripped, as a messenger or platform would."


def make_double_compressed(h: int = 720, w: int = 960, seed: int = 9) -> tuple[np.ndarray, str]:
    """Save at q90, then a sub-region at q55 — two quantisation histories on one grid.

    The region is 8-aligned on purpose. Unlike ``spliced``, this fixture isolates *quantisation
    history* from *grid phase*: ``jpeg_ghost`` should find a minority of blocks best explained by
    q55 while ``block_grid`` correctly finds nothing, because one encoder laid down the whole grid.
    """
    rng = np.random.default_rng(SEED + seed)
    img = add_structures(natural_image(h, w, rng, hue=1.9), rng)
    img = add_sensor_noise(img, rng, sigma=2.6)
    once = jpeg_roundtrip(img, quality=90)

    ry, rx = h // 3, w // 3
    ry, rx = ry // 8 * 8, rx // 8 * 8
    region = jpeg_roundtrip(once[ry:ry + h // 3, rx:rx + w // 3], quality=55)
    out = once.copy()
    out[ry:ry + region.shape[0], rx:rx + region.shape[1]] = region
    ground = (f"Whole frame saved at q90, region [{rx}:{rx + w // 3}, {ry}:{ry + h // 3}] "
              f"re-saved at q55, grid-aligned.")
    return out, ground


def make_generator_metadata(h: int = 512, w: int = 512, seed: int = 10) -> tuple[np.ndarray, str]:
    """A PNG carrying Stable-Diffusion-style ``parameters`` text chunks."""
    rng = np.random.default_rng(SEED + seed)
    img = natural_image(h, w, rng, hue=4.2)
    img = add_sensor_noise(img, rng, sigma=0.9)
    return img, "PNG with generator parameters embedded in a text chunk."


GENERATOR_CHUNKS = {
    "parameters": (
        "prompt: a professional headshot of a person, studio lighting\n"
        "negative prompt: blurry, deformed\n"
        "steps: 30, sampler: DPM++ 2M Karras, cfg scale: 7.5, seed: 4281137, "
        "size: 512x512, model hash: 6ce0161689, model: v1-5-pruned-emaonly"
    ),
    "Software": "Automatic1111",
}


FIXTURE_SPECS = {
    # name: (builder, label, format, save kwargs)
    "authentic": (make_authentic, "authentic", "jpeg", {"quality": 88}),
    "spliced": (make_spliced, "spliced", "jpeg", {"quality": 88}),
    "copy_move": (make_copy_move, "copy_move", "jpeg", {"quality": 88}),
    "ai_generated": (make_ai_generated, "ai_generated", "png", {}),
    "double_compressed": (make_double_compressed, "edited", "jpeg", {"quality": 90}),
    "heavy_compression": (make_heavy_compression, "authentic", "jpeg", {"quality": 32}),
    "small": (make_small, "authentic", "jpeg", {"quality": 85}),
    "screenshot": (make_screenshot, "authentic", "png", {}),
    "no_exif": (make_no_exif, "authentic", "jpeg", {"quality": 85, "exif": None}),
    "generator_metadata": (make_generator_metadata, "ai_generated", "png",
                           {"text_chunks": GENERATOR_CHUNKS}),
}

LABELS = {"authentic": 0, "edited": 0, "spliced": 1, "copy_move": 1, "ai_generated": 1}


def build_fixture(name: str) -> tuple[np.ndarray, str, str]:
    """Return ``(array, label, ground_truth_description)``."""
    builder, label, fmt, _ = FIXTURE_SPECS[name]
    img, truth = builder()
    return img, label, truth


def write_fixture(name: str, out_dir: str) -> str:
    builder, _, fmt, kwargs = FIXTURE_SPECS[name]
    img, _truth = builder()
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "jpeg":
        return save_jpeg(img, os.path.join(out_dir, f"{name}.jpg"), **kwargs)
    return save_png(img, os.path.join(out_dir, f"{name}.png"), **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tests/fixtures/images")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    names = args.only or list(FIXTURE_SPECS)
    written = []
    for name in names:
        path = write_fixture(name, args.out)
        size = os.path.getsize(path)
        builder, label, _fmt, _kw = FIXTURE_SPECS[name]
        _img, truth = builder()
        written.append((name, label, size, truth))
        print(f"  {name:22s} label={label:14s} {size / 1024:7.1f} KB")
    print(f"\nwrote {len(written)} fixtures to {args.out}")

    manifest = {
        "generated_by": "tests/fixtures/make_fixtures.py",
        "seed": SEED,
        "warning": "Synthetic fixtures. They prove each analyser responds to the artifact it "
                   "claims to detect; they do NOT establish real-world accuracy.",
        "images": [{"name": n, "label": l, "bytes": s, "ground_truth": t}
                   for n, l, s, t in written],
    }
    import json
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
