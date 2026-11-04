"""Metadata / provenance analysis.

Reads EXIF, XMP and PNG text chunks from the **pristine** file bytes. This must never run on
a decoded-and-re-encoded image: that destroys the C2PA manifest and rewrites every timestamp.

Scoring philosophy: *presence of a generator fingerprint is strong evidence; absence of
metadata is weak evidence.* The prototype added a flat ``0.35`` for missing EXIF, but Instagram,
WhatsApp, X and every screenshot tool strip EXIF from authentic photos too. Missing metadata is
now worth ``0.12`` and is reported as a reliability penalty rather than as suspicion.
"""

from __future__ import annotations

import io
import re

from PIL import Image
from PIL.ExifTags import TAGS

from app.analysis.contract import TechniqueResult, clip01
from app.core.security import (
    EDITING_SOFTWARE,
    redact_metadata,
    scan_generator_fingerprints,
)

__all__ = ["run"]

_XMP_RE = re.compile(rb"<x[:.]xmpmeta.*?</x[:.]xmpmeta>", re.DOTALL | re.IGNORECASE)
#: XMP tags worth surfacing to the reviewer.
_XMP_INTEREST = (
    "CreatorTool", "tool", "DigitalSourceType", "SourceModified", "DocumentID",
    "InstanceID", "OriginalDocumentID", "History", "softwareAgent", "action",
    "derivedFrom", "format", "ai:", "CompositeWith", "trainedAlgorithmicMedia",
)
#: C2PA / IPTC digital source types that declare AI involvement.
_AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia", "compositeai", "algorithmicmedia", "aigenerated",
)

#: ``im.info`` keys that hold structural or binary payloads rather than author text.
#:
#: ``exif`` is the important one. PIL surfaces a PNG ``eXIf`` chunk as raw bytes, which are
#: *not* text. Forwarding them would put a kilobyte of mojibake in the API response — and,
#: worse, would ship unredacted EXIF past ``redact_metadata``, which only ever sees the parsed
#: dict. The parsed and redacted form is already reported separately under ``details["exif"]``,
#: so nothing is lost by dropping the raw chunk here.
_NON_TEXT_CHUNKS = frozenset({
    "dpi", "gamma", "chromaticity", "background", "transparency", "exif", "icc_profile",
    "aspect", "rendering_intent", "srgb", "applied_gamma",
})

#: Fraction of a chunk that must be printable for it to be treated as text. Catches any binary
#: payload a future PIL version surfaces under a key not listed above.
_PRINTABLE_RATIO = 0.92


def _is_printable_text(value: str | bytes) -> bool:
    """True when a chunk value looks like author text rather than a binary blob."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if not value:
        return False
    printable = sum(1 for ch in value if ch.isprintable() or ch in "\n\r\t")
    return printable / len(value) >= _PRINTABLE_RATIO


def run(orig_bytes: bytes, mime: str = "", *, redact_pii: bool = True) -> TechniqueResult:
    im = Image.open(io.BytesIO(orig_bytes))
    score = 0.0
    findings: list[dict[str, object]] = []

    def add(points: float, code: str, text: str, severity: str = "info") -> None:
        nonlocal score
        score += points
        findings.append({"code": code, "text": text, "severity": severity,
                         "points": round(points, 3)})

    # --- EXIF -------------------------------------------------------------------
    try:
        raw_exif = im.getexif()
        info = {TAGS.get(tag_id, str(tag_id)): value for tag_id, value in raw_exif.items()}
        # IFD1 (thumbnail) and the Exif/GPS sub-IFDs hold most of the interesting tags.
        for ifd_key in ("Exif", "GPS", "Interop"):
            try:
                sub = raw_exif.get_ifd({"Exif": 0x8769, "GPS": 0x8825,
                                        "Interop": 0xA005}[ifd_key])
            except Exception:
                continue
            if ifd_key == "GPS" and sub and redact_pii:
                info["GPSInfo"] = "[redacted: location data removed]"
                continue
            for tag_id, value in sub.items():
                info.setdefault(TAGS.get(tag_id, f"{ifd_key}_{tag_id}"), value)
    except Exception:
        info = {}

    has_exif = bool(info)
    software = str(info.get("Software", "")).strip()
    make = str(info.get("Make", "")).strip()
    model = str(info.get("Model", "")).strip()

    if not has_exif:
        add(0.12, "no_exif",
            "No EXIF metadata. This is expected for screenshots and for anything re-uploaded "
            "to a social platform, so it is weak evidence only.", "weak")
    else:
        lowered = software.lower()
        if any(s in lowered for s in EDITING_SOFTWARE):
            gen = any(s in lowered for s in
                      ("stable diffusion", "midjourney", "dall", "firefly", "comfyui",
                       "automatic1111"))
            add(0.55 if gen else 0.30,
                "generator_software" if gen else "editing_software",
                f"Software tag names {'an AI generator' if gen else 'an editor'}: {software!r}",
                "strong" if gen else "moderate")
        if not make and not model:
            add(0.08, "no_camera_identity",
                "No camera make/model recorded.", "weak")
        if "DateTimeOriginal" not in info:
            add(0.08, "no_capture_time",
                "No original capture timestamp.", "weak")
        mod, orig = str(info.get("DateTime", "")), str(info.get("DateTimeOriginal", ""))
        if mod and orig and mod != orig:
            add(0.18, "timestamp_mismatch",
                f"Modification time ({mod}) differs from capture time ({orig}).", "moderate")
        # A real camera photo carries an exposure triangle. Generated images almost never do,
        # and editors rarely synthesise a self-consistent one.
        exposure_tags = ("ExposureTime", "FNumber", "ISOSpeedRatings", "ShutterSpeedValue",
                         "ApertureValue", "FocalLength")
        n_exposure = sum(1 for t in exposure_tags if t in info)
        if n_exposure == 0 and (make or model):
            add(0.10, "missing_exposure_triple",
                "Camera is identified but no exposure parameters were recorded.", "weak")

    # --- PNG text chunks (Stable Diffusion / ComfyUI / NovelAI write these) --------
    text_chunks = {k: v for k, v in im.info.items()
                   if isinstance(v, (str, bytes)) and k not in _NON_TEXT_CHUNKS
                   and _is_printable_text(v)}
    gen_hits, safe_chunks = scan_generator_fingerprints(text_chunks)
    if gen_hits:
        kws = sorted({h.keyword for h in gen_hits})
        add(0.60, "generator_parameters",
            f"Generator parameters embedded in {', '.join(h.chunk for h in gen_hits)} "
            f"(matched: {', '.join(kws[:4])}).", "strong")

    # --- XMP ---------------------------------------------------------------------
    xmp = _extract_xmp(orig_bytes, im)
    xmp_findings: list[str] = []
    if xmp:
        low = xmp.lower()
        for tag in _XMP_INTEREST:
            for m in re.finditer(re.escape(tag.lower()) + r'"?\s*(?:rdf:parseType=)?"?', low):
                snippet = xmp[max(0, m.start() - 40): m.start() + 160]
                snippet = " ".join(re.sub(r"<[^>]+>", " ", snippet).split())[:120]
                if snippet:
                    xmp_findings.append(snippet)
        if any(t in low for t in _AI_SOURCE_TYPES):
            add(0.62, "xmp_ai_source_type",
                "XMP declares an AI-trained or composite-AI digital source type.", "strong")
        if "trainedalgorithmicmedia" in low or "compositeai" in low:
            xmp_findings.insert(0, "IPTC DigitalSourceType declares AI generation.")
        tool_match = re.search(r'(?:CreatorTool|xmpMM:softwareAgent)[^>]*>([^<]{2,80})<', xmp)
        if tool_match:
            tool = tool_match.group(1).strip()
            if any(s in tool.lower() for s in EDITING_SOFTWARE):
                add(0.35, "xmp_editing_tool",
                    f"XMP records an editing/generation tool: {tool!r}", "moderate")
            xmp_findings.append(f"CreatorTool: {tool}")
        if re.search(r'st:(?:save|edit)\b', low):
            add(0.15, "xmp_edit_history",
                "XMP contains an explicit save/edit history (the file was re-written by a tool).",
                "moderate")

    if not findings:
        add(0.0, "clean",
            "Metadata is internally consistent and shows no editing or generation markers.",
            "ok")

    details = {
        "findings": findings,
        "has_exif": has_exif,
        "exif": redact_metadata(dict(list(info.items())[:40]), redact=redact_pii),
        "exif_tag_count": len(info),
        "software": software or None,
        "camera_make": make or None,
        "camera_model": model or None,
        "png_text_chunks": safe_chunks,
        "generator_hits": [{"chunk": h.chunk, "keyword": h.keyword} for h in gen_hits],
        "has_xmp": bool(xmp),
        "xmp_snippets": xmp_findings[:8],
        "mime": mime or None,
        #: Scope is the EXIF table only. Generator-parameter chunks are deliberately *not*
        #: redacted — they are the evidence, not incidental personal data. See
        #: ``security.scan_generator_fingerprints``.
        "exif_pii_redacted": redact_pii,
        "deterministic_flag": _deterministic_flag(findings),
    }
    im.close()
    return TechniqueResult(clip01(score), None, details)


def _deterministic_flag(findings: list[dict[str, object]]) -> str | None:
    """A *fact about the file* (as opposed to a statistical inference) may floor the AI score."""
    for f in findings:
        if f.get("code") in ("generator_parameters", "xmp_ai_source_type"):
            return str(f["code"])
        if f.get("code") == "generator_software":
            return "generator_software"
    return None


def _extract_xmp(orig_bytes: bytes, im: Image.Image) -> str:
    """Pull the XMP packet out of a JPEG APP1 segment or a PNG iTXt chunk."""
    for key in ("xmp", "XML:com.adobe.xmp"):
        val = im.info.get(key)
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8", errors="replace")
            except Exception:
                continue
        if isinstance(val, str) and val:
            return val
    m = _XMP_RE.search(orig_bytes[: min(len(orig_bytes), 4_000_000)])
    return m.group(0).decode("utf-8", errors="replace") if m else ""
