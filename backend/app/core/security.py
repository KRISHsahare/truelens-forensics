"""Upload validation and privacy redaction.

Three defences, applied in order of cost:

1. **Magic bytes**, never the client-supplied filename or content type.
2. **Header-only probe** (``Image.open`` without ``.load()``) to read dimensions and reject
   decompression bombs *before* any pixel is decoded.
3. **Warning-to-error promotion** during decode, because PIL's default only *warns* between
   ``MAX_IMAGE_PIXELS`` and ``2 * MAX_IMAGE_PIXELS`` — a 150 MP image would otherwise decode
   happily and OOM the container.

``Image.MAX_IMAGE_PIXELS = None`` (the usual StackOverflow fix) disables the check entirely.
We set an explicit, stricter number instead.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass
from typing import Final

from PIL import Image, ImageFile

__all__ = [
    "ALLOWED_MAGIC",
    "MAX_IMAGE_PIXELS",
    "PII_TAGS",
    "GeneratorHit",
    "ImageProbe",
    "PayloadTooLargeError",
    "RedactionError",
    "SecurityError",
    "UnsupportedMediaError",
    "decode_verified",
    "estimate_jpeg_quality",
    "probe_header",
    "redact_metadata",
    "sniff_mime",
]

# Explicit and stricter than PIL's default (~89 MP). Never set this to None.
MAX_IMAGE_PIXELS: Final[int] = 60_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
# Reject truncated uploads rather than silently analysing a partial image.
ImageFile.LOAD_TRUNCATED_IMAGES = False

ALLOWED_MAGIC: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"BM", "image/bmp"),
)

#: EXIF/XMP fields that identify a person, device or location. Stripped by default.
PII_TAGS: Final[frozenset[str]] = frozenset({
    "GPSInfo", "GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSTimeStamp", "GPSDateStamp",
    "GPSLatitudeRef", "GPSLongitudeRef", "GPSAltitudeRef", "GPSDestLatitude",
    "GPSDestLongitude", "Artist", "Copyright", "OwnerName", "Make_note",
    "BodySerialNumber", "LensSerialNumber", "UserComment",
    "Iptc4xmpCore:CreatorContactInfo", "dc:creator", "photoshop:AuthorsPosition",
    "XPAuthor", "XPComment",
})

#: Words that reveal generator parameters. We report the *matched keyword* and the chunk
#: name, never the chunk body: Stable Diffusion ``parameters`` chunks contain the full
#: prompt, which frequently describes real people.
GENERATOR_KEYWORDS: Final[tuple[str, ...]] = (
    "steps:", "sampler", "cfg scale", "cfg:", "seed:", "seed ", "diffusion",
    "midjourney", "stable diffusion", "comfyui", "automatic1111", "dall-e", "dall·e",
    "flux", "negative prompt", "model hash", "lora:", "checkpoint:",
)

#: Cap on a single PNG text chunk forwarded to the API. A ComfyUI ``workflow`` chunk is a
#: multi-kilobyte JSON graph; the forensically interesting part is the prompt block, and the
#: response should not carry the whole thing.
MAX_CHUNK_CHARS: Final[int] = 1200

#: Software tags that indicate editing or generation.
EDITING_SOFTWARE: Final[tuple[str, ...]] = (
    "photoshop", "gimp", "lightroom", "snapseed", "canva", "stable diffusion",
    "midjourney", "dall", "firefly", "comfyui", "automatic1111", "affinity",
    "pixelmator", "facetune", "pixlr", "darktable", "rawtherapee",
)

_ROUND_DIMS: Final[frozenset[int]] = frozenset({
    240, 320, 480, 600, 640, 720, 768, 800, 1024, 1080, 1200, 1280, 1366, 1440,
    1536, 1600, 1920, 2048, 2160, 2560, 3440, 3840, 4096, 7680,
})

# JPEG Annex K standard luminance quantisation table, used to invert the quality formula.
_STD_LUMA_Q: Final[tuple[int, ...]] = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)


class SecurityError(Exception):
    """Base class for rejected uploads. Carries an HTTP status and a safe message.

    ``code`` doubles as the wire error code, so it must be a member of the API's ``ErrorCode`` enum:
    a value outside it makes the error envelope fail validation and turns a 400 into a 500.
    """

    status_code: int = 400
    code: str = "invalid_image"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedMediaError(SecurityError):
    status_code = 415
    code = "unsupported_media"


class PayloadTooLargeError(SecurityError):
    status_code = 413
    code = "payload_too_large"


class RedactionError(SecurityError):
    status_code = 400
    code = "unredactable"


@dataclass(slots=True, frozen=True)
class ImageProbe:
    """Header-only findings. No pixel data was decoded to produce this."""

    mime: str
    fmt: str
    width: int
    height: int
    has_alpha: bool
    n_frames: int
    quantization: dict[int, list[int]] | None
    info_keys: tuple[str, ...]

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass(slots=True, frozen=True)
class GeneratorHit:
    """A generator fingerprint found in metadata."""

    chunk: str
    keyword: str


def sniff_mime(data: bytes) -> str | None:
    """Identify the format from magic bytes. Returns None if not an allowed type."""
    if len(data) < 12:
        return None
    # WebP needs a RIFF container check, not a simple prefix match.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in ALLOWED_MAGIC:
        if data.startswith(magic):
            return mime
    return None


def probe_header(data: bytes, max_pixels: int = MAX_IMAGE_PIXELS) -> ImageProbe:
    """Parse the header only (~1 ms) and enforce every cap before decoding any pixels."""
    mime = sniff_mime(data)
    if mime is None:
        raise UnsupportedMediaError(
            "File is not a JPEG, PNG, WebP or BMP image. Content was checked by magic "
            "bytes, not by file extension."
        )
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            fmt = im.format or "UNKNOWN"
            has_alpha = "A" in (im.getbands() or ())
            n_frames = getattr(im, "n_frames", 1)
            quant = getattr(im, "quantization", None)
            info_keys = tuple(im.info.keys())
    except UnsupportedMediaError:
        raise
    except Image.DecompressionBombError as exc:
        # Pillow refuses outright above 2x its own limit, inside ``Image.open``, so the dimension
        # check below never sees the most extreme bombs. Letting the generic handler report one as
        # "header could not be parsed" is a lie — the header parsed perfectly; the file is simply
        # enormous. Both spellings of "too big" must come back as 413.
        raise PayloadTooLargeError(
            f"Image header declares far more than the {max_pixels / 1e6:.0f} MP cap "
            "(decompression-bomb protection)."
        ) from exc
    except Exception as exc:
        raise UnsupportedMediaError(f"Image header could not be parsed ({type(exc).__name__}).") from exc

    if w * h > max_pixels:
        raise PayloadTooLargeError(
            f"Image is {w * h / 1e6:.1f} MP, above the {max_pixels / 1e6:.0f} MP cap "
            "(decompression-bomb protection)."
        )
    if n_frames > 1:
        raise UnsupportedMediaError("Animated and multi-frame images are not supported.")
    return ImageProbe(
        mime=mime, fmt=fmt, width=w, height=h, has_alpha=has_alpha,
        n_frames=n_frames, quantization=quant, info_keys=info_keys,
    )


def decode_verified(data: bytes, probe: ImageProbe) -> Image.Image:
    """Decode pixels with the decompression-bomb *warning* promoted to a hard error.

    PIL only raises between 1x and 2x ``MAX_IMAGE_PIXELS``; below 2x it warns and continues.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        im = Image.open(io.BytesIO(data))
        im.load()
    return im


def estimate_jpeg_quality(quantization: dict[int, list[int]] | None) -> int | None:
    """Invert the JPEG quality formula from the luminance quantisation table.

    Standard tables scale as ``q[i] = clamp(std[i] * s)`` where ``s = 200 - 2*Q`` for Q >= 50
    and ``s = 5000 / Q`` below that. We solve for the median ratio, which is robust to
    encoder-specific table tweaks.
    """
    if not quantization:
        return None
    table = quantization.get(0) or next(iter(quantization.values()), None)
    if not table or len(table) < len(_STD_LUMA_Q):
        return None
    ratios = []
    for actual, std in zip(table[: len(_STD_LUMA_Q)], _STD_LUMA_Q, strict=True):
        if std > 0 and actual > 0:
            ratios.append(actual / std)
    if not ratios:
        return None
    ratios.sort()
    s = ratios[len(ratios) // 2]
    if s <= 0:
        return None
    # s = (200 - 2Q)/100 for Q >= 50 ; s = 5000/(100Q) for Q < 50
    q = (200 - s * 100) / 2 if s <= 1.0 else 5000 / (s * 100)
    return int(round(max(1, min(100, q))))


def is_screenshot_shape(w: int, h: int) -> bool:
    """Both dimensions are round screen/resolution values — typical of screenshots and exports."""
    return w in _ROUND_DIMS and h in _ROUND_DIMS


def redact_metadata(entries: dict[str, object], *, redact: bool = True) -> dict[str, object]:
    """Strip GPS entirely and mask identity fields.

    GPS EXIF carries lat/long to sub-metre precision — a home address. We drop the whole
    ``GPSInfo`` IFD rather than individual sub-tags, because a partially-redacted IFD still
    leaks position via ``GPSMapDatum`` + partial coordinates.
    """
    if not redact:
        return {k: _stringify(v) for k, v in entries.items()}
    out: dict[str, object] = {}
    for k, v in entries.items():
        if k == "GPSInfo" and isinstance(v, dict):
            out[k] = "[redacted: location data removed]"
        elif k in PII_TAGS:
            out[k] = "[redacted]"
        else:
            out[k] = _stringify(v)
    return out


def scan_generator_fingerprints(
    chunks: dict[str, object]
) -> tuple[list[GeneratorHit], dict[str, object]]:
    """Look for generator parameters in PNG text chunks / XMP / comment fields.

    Returns the hits (keyword + chunk name) and a length-capped view of the chunk text.

    Generator parameters are **not** PII-redacted, deliberately. The distinction this codebase
    draws is between *evidence* and *incidental personal data*:

    * A Stable Diffusion ``parameters`` block is the finding itself. Scrubbing it would leave the
      UI able to say "generator metadata detected" but unable to show the prompt, seed, sampler
      and model hash that justify it — the tool would be least useful at the exact moment it is
      most confident.
    * GPS coordinates and camera-owner identity carry no weight in the verdict and can locate a
      person's home, so ``redact_metadata`` strips those.

    Nor does redaction here protect anyone: the person reading the report already possesses the
    file, and results are cached by SHA-256 of that same file, so no chunk can reach a reader who
    does not already hold the bytes. Uploads are never persisted, so we are not the ones storing
    the prompt.

    Values are capped at ``MAX_CHUNK_CHARS`` to keep a multi-kilobyte ComfyUI workflow blob out
    of the API response.
    """
    hits: list[GeneratorHit] = []
    safe: dict[str, object] = {}
    for chunk, value in chunks.items():
        text = _stringify(value)
        lowered = text.lower()
        found = next((kw for kw in GENERATOR_KEYWORDS if kw in lowered), None)
        if found:
            hits.append(GeneratorHit(chunk=str(chunk), keyword=found.strip()))
        safe[chunk] = text[:MAX_CHUNK_CHARS]
    return hits, safe


def sanitise_error(exc: BaseException, limit: int = 200) -> str:
    """Make an exception message safe to send to a browser.

    Analyser exceptions contain absolute filesystem paths, and a ``BrokenProcessPool``
    carries the child's full traceback. Neither belongs in an API response.
    """
    msg = str(exc) or type(exc).__name__
    msg = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<path>", msg)          # windows paths
    msg = re.sub(r"(?:/[\w.\-]+){2,}", "<path>", msg)             # posix paths
    msg = re.sub(r"0x[0-9a-fA-F]{6,}", "<addr>", msg)             # memory addresses
    msg = " ".join(msg.split())
    return msg[:limit]


def _stringify(value: object) -> object:
    """Coerce EXIF values (bytes, IFDRational, tuples, nested IFDs) to JSON-safe text."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return repr(value[:32])
    if isinstance(value, dict):
        return {str(k): _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_stringify(v) for v in value]
        return items if len(items) <= 32 else items[:32] + [f"...(+{len(items) - 32} more)"]
    if isinstance(value, float):
        return round(value, 6)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):  # IFDRational
        try:
            return round(float(value), 6)
        except Exception:
            return str(value)
    return value if isinstance(value, (str, int, bool, type(None))) else str(value)[:200]
