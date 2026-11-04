"""Upload-boundary security: what we accept, what we refuse, and what we never repeat back.

These are the checks that stand between "someone POSTed bytes" and "a CPU-bound pipeline decoded
them". The order matters as much as the rules: format is decided by magic bytes before any
extension is believed, dimensions are capped before a single pixel is decoded, and every message
that leaves the process has been stripped of the filesystem paths that exception text carries.
"""

from __future__ import annotations

import io
import struct
import warnings
import zlib

import pytest
from PIL import Image

from app.core.security import (
    MAX_IMAGE_PIXELS,
    PII_TAGS,
    GeneratorHit,
    PayloadTooLargeError,
    UnsupportedMediaError,
    estimate_jpeg_quality,
    is_screenshot_shape,
    probe_header,
    redact_metadata,
    sanitise_error,
    scan_generator_fingerprints,
    sniff_mime,
)


def _png_header(width: int, height: int) -> bytes:
    """A real PNG with its IHDR dimensions rewritten.

    Pillow's parser wants a well-formed container with a correct chunk CRC, so this patches a
    generated 8x8 file rather than assembling one from nothing. The body is now inconsistent with
    the declared size, which is fine: ``probe_header`` reads the header only, and that is exactly
    the property under test — 40 gigapixels must be refused without decoding a single pixel.
    """
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, "PNG")
    data = bytearray(image.getvalue())
    struct.pack_into(">II", data, 16, width, height)
    struct.pack_into(">I", data, 29, zlib.crc32(bytes(data[12:29])) & 0xFFFFFFFF)
    return bytes(data)


def _jpeg(width: int = 64, height: int = 48, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 200, 90)).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def test_magic_bytes_decide_not_the_extension():
    assert sniff_mime(_jpeg()) == "image/jpeg"
    assert sniff_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24) == "image/png"
    assert sniff_mime(b"BM" + b"\x00" * 30) == "image/bmp"
    assert sniff_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    # A PDF with a .jpg name is the attack the filename-based check would have waved through.
    assert sniff_mime(b"%PDF-1.7\n%\xc7\xec\x8f\xd2\n" + b"\x00" * 32) is None
    assert sniff_mime(b"GIF89a" + b"\x00" * 32) is None
    assert sniff_mime(b"\xff\xd8") is None  # too short to be anything


def test_disguised_container_refused_before_decode():
    with pytest.raises(UnsupportedMediaError):
        probe_header(b"%PDF-1.7\n" + b" " * 40)


def test_truncated_container_refused():
    data = _png_header(8, 8)
    with pytest.raises(UnsupportedMediaError):
        probe_header(data[:12])


def test_decompression_bomb_refused_at_the_header():
    """40 gigapixels must cost one comparison, not one decode."""
    with pytest.raises(PayloadTooLargeError) as exc:
        probe_header(_png_header(200_000, 200_000), max_pixels=MAX_IMAGE_PIXELS)
    assert "MP" in str(exc.value)


def test_bomb_between_one_and_two_times_the_cap_is_still_413():
    """Pillow warns, rather than raising, between 1x and 2x its limit.

    That band is where a naive guard leaks: the header probe returns happily and the decode eats the
    memory. Ours reads the declared size, so 70 MP against a 60 MP cap is refused before pixels.
    """
    probe = _png_header(10_000, 7_000)
    assert 2 * MAX_IMAGE_PIXELS < 200_000 * 200_000, "the test above covers Pillow's own refusal"
    assert 2 * MAX_IMAGE_PIXELS > 10_000 * 7_000, "this one must land in the warn-only band"
    with pytest.raises(PayloadTooLargeError) as exc, warnings.catch_warnings():
        # Pillow's own contribution in this band is a warning, not a refusal. Silence the noise;
        # the assertion is that our cap check turns that warning band into a 413.
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        probe_header(probe, max_pixels=MAX_IMAGE_PIXELS)
    assert "70.0 MP" in str(exc.value)


def test_gif_is_not_an_accepted_container():
    frames = io.BytesIO()
    Image.new("P", (8, 8)).save(
        frames, "GIF", save_all=True, append_images=[Image.new("P", (8, 8))], duration=10)
    assert sniff_mime(frames.getvalue()) is None


def test_probe_reports_real_dimensions():
    probe = probe_header(_jpeg(320, 200))
    assert (probe.width, probe.height) == (320, 200)
    assert probe.mime == "image/jpeg"
    assert probe.pixels == 320 * 200
    assert probe.quantization, "a JPEG must expose its quantisation tables"
    assert estimate_jpeg_quality(probe.quantization) is not None
    assert estimate_jpeg_quality(None) is None, "a PNG genuinely has no JPEG quality"


@pytest.mark.parametrize("quality", [30, 60, 92])
def test_estimated_quality_tracks_the_encoder(quality):
    probe = probe_header(_jpeg(160, 120, quality=quality))
    assert abs((estimate_jpeg_quality(probe.quantization) or 0) - quality) <= 12


def test_gps_is_removed_wholesale_not_partially():
    entries = {
        "GPSInfo": {"GPSLatitude": (19.1, 2.0), "GPSMapDatum": "WGS-84"},
        "Artist": "Someone",
        "Make": "Canon",
        "ImageWidth": 4000,
    }
    out = redact_metadata(entries)
    assert out["GPSInfo"] == "[redacted: location data removed]"
    assert "GPSMapDatum" not in str(out["GPSInfo"]), "a surviving datum still locates a person"
    assert out["Artist"] == "[redacted]"
    assert out["Make"] == "Canon", "device make/model is provenance evidence, not identity"
    assert out["ImageWidth"] == 4000


def test_every_pii_tag_is_masked():
    out = redact_metadata({tag: "value" for tag in PII_TAGS})
    assert all(str(v).startswith("[redacted") for v in out.values()), out


def test_redaction_can_be_disabled_by_configuration():
    """``REDACT_PII=0`` is a real deployment switch, so it has to switch the behaviour off."""
    entries = {"GPSInfo": {"GPSLatitude": (19.1, 2.0)}, "Artist": "Someone"}
    kept = redact_metadata(entries, redact=False)
    assert kept["GPSInfo"] != "[redacted: location data removed]"
    assert kept["Artist"] == "Someone"


def test_generator_parameters_are_evidence_not_pii():
    """The prompt and seed are the finding; redacting them would hide the reason for the score."""
    chunks = {
        "parameters": "steps: 35, cfg: 7, seed: 1029384, model: sd_xl_base_1.0 (ComfyUI)",
        "Software": "Stable Diffusion",
        "description": "a wholly ordinary caption",
    }
    hits, safe = scan_generator_fingerprints(chunks)
    assert isinstance(hits[0], GeneratorHit)
    assert {h.chunk for h in hits} >= {"parameters", "Software"}
    assert all(h.keyword for h in hits)
    assert "sd_xl_base" in str(safe["parameters"]), "the evidence must stay readable"


def test_chunk_values_are_length_capped():
    """A multi-kilobyte workflow blob must not ride into the API response."""
    from app.core.security import MAX_CHUNK_CHARS
    huge = {"prompt": "midjourney " + ("x" * 40_000)}
    _, safe = scan_generator_fingerprints(huge)
    assert len(safe["prompt"]) <= MAX_CHUNK_CHARS


def test_sanitise_error_strips_paths_and_addresses():
    class Boom(Exception):
        pass

    win = Boom("cannot open C:\\Users\\someone\\scratch\\job\\ela.png")
    posix = Boom("BrokenProcessPool: /home/app/backend/app/core/scratch.py line 3")
    addr = Boom("array at 0x7f2a1b3c4d50 is invalid")
    for exc in (win, posix, addr):
        out = sanitise_error(exc)
        assert len(out) <= 200
        assert "\\" not in out and "/home/" not in out and "0x7f" not in out, out
        assert "Traceback" not in out


def test_sanitise_error_falls_back_to_the_type_name():
    assert sanitise_error(ValueError("")) == "ValueError"


def test_screenshot_shape_detector_recognises_common_resolutions():
    assert is_screenshot_shape(1920, 1080)
    assert is_screenshot_shape(2560, 1440)
    assert not is_screenshot_shape(3024, 4032), "a phone photo is not a screen grab"
