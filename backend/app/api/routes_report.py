"""``GET /api/report/{job_id}`` — the forensic report as a PDF.

Why the PDF is built server-side: the on-screen report and the downloaded one must not disagree. If
the browser rendered the document and a library flattened that DOM, every font substitution, page
break and colour profile would be the reviewer's machine's answer — and this document is meant to be
emailed to a lawyer. Reportlab lays it out from the same stored result the poll returns, so the
numbers, the bands and the disclaimers are the engine's, once.

What the report contains, and what it refuses to contain:

- **No filename.** The job record has one internally for the table label; a report is a document
  that leaves the building, and the uploader's file path is their business. The identity is the
  SHA-256, which is what a verifier compares.
- **Overlays only if they still exist.** They are composites of the uploaded photograph, and they are
  deleted with the job. A missing heatmap is printed as a caption saying it expired, never as a blank
  frame that a reader could mistake for "no evidence found".
- **An unmeasured score is the words "not measured".** A zero in a risk column is the single most
  misleading cell this document could contain.
- **The limitations block is on page one.** Not in an appendix: a report that is forwarded without its
  qualifications is how a probability becomes an accusation.

The route is signed-off by the same rules as the job poll — allowlisted id, store lookup, one 404 for
every reason a record might not be there. It is rate-limited with the *job* window, because a report
per poll would let a client make the server render PDFs all day for a job it already has the result of.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import Response as PlainResponse
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, Paragraph, Spacer, Table, TableStyle

from app.api.deps import error_response, get_orchestrator, get_settings
from app.api.limits import limit_jobs, limiter
from app.core.job import JobRecord
from app.core.scratch import job_id_safe

__all__ = ["router"]

log = logging.getLogger("truelens.api.report")

router = APIRouter(prefix="/api", tags=["report"])

_GONE = (
    "Not found. Jobs and their results are deleted within ten minutes of upload, so this report may "
    "simply have expired; nothing is stored longer than that."
)

# Palette mirrors the product's own bands, so a printed bar means the same thing as an on-screen one.
_INK = colors.HexColor("#1b2027")
_MUTED = colors.HexColor("#5b6672")
_RULE = colors.HexColor("#d7dce2")
_LOW = colors.HexColor("#1f9d6b")
_MODERATE = colors.HexColor("#c98a12")
_HIGH = colors.HexColor("#c93a4e")
_NONE = colors.HexColor("#7b8794")

_BAND_COLORS = {"low": _LOW, "moderate": _MODERATE, "high": _HIGH, "not_measured": _NONE}

_DISCLAIMER = (
    "This report states a likelihood, not a finding of fact. It was produced by automated analysis of "
    "the file's pixels, container and metadata. No human reviewed the image. Detection methods lag "
    "the generators they target, re-encoding destroys the evidence they rely on, and a clean result "
    "is not a certificate of authenticity. Treat every score below as one piece of evidence alongside "
    "whatever you know about the image's origin."
)


@router.get(
    "/report/{job_id}",
    summary="Download the forensic report for a finished job as a PDF",
    responses={
        404: {"description": "Unknown, expired, or already deleted job id"},
        409: {"description": "The job exists but has not produced a result"},
    },
)
@limiter.limit(limit_jobs)
async def get_report(request: Request, job_id: str, response: Response):
    if not job_id_safe(job_id):
        return error_response(404, "not_found", _GONE)

    record = await get_orchestrator(request).fetch(job_id)
    if record is None:
        return error_response(404, "not_found", _GONE)
    if record.status != "succeeded" or not record.result:
        message = (
            "This job did not finish successfully, so there is no report to write."
            if record.status in {"failed", "expired"}
            else "The analysis is still running. Poll the job and request the report when it succeeds."
        )
        return error_response(409, "job_not_ready", message,
                              detail=f"status={record.status}")

    settings = get_settings(request)
    scratch = get_orchestrator(request).scratch
    overlays = {
        name: scratch.overlay_path(job_id, name)
        for name, view in (record.result.get("techniques") or {}).items()
        if view.get("overlay_url")
    }
    present = {name: path for name, path in overlays.items() if isinstance(path, Path) and path.is_file()}

    try:
        pdf = build_pdf(record, present, app_version=routes_version(), ttl_s=settings.overlay_ttl_s)
    except Exception:
        log.exception("report rendering failed for job %s", job_id)
        return error_response(500, "internal", "The report could not be rendered.")

    sha8 = (record.sha256 or "unknown")[:8]
    return PlainResponse(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="truelens-report-{sha8}.pdf"',
            # The body is an analysis of someone's uploaded photograph. No cache, no referral, and
            # nothing that lets a proxy hand it to the next visitor of this deployment.
            "Cache-Control": "private, no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


# --------------------------------------------------------------------------- rendering


def routes_version() -> str:
    """Kept as a call so the version comes from the same place as ``/api/health``."""
    from app.api import routes_health

    return routes_health.app_version()


def build_pdf(
    record: JobRecord,
    overlays: dict[str, Path],
    *,
    app_version: str,
    ttl_s: int,
) -> bytes:
    from reportlab.platypus import SimpleDocTemplate

    result = record.result or {}
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=f"TrueLens report {record.sha256[:12]}",
        author="TrueLens",
        subject="Automated image authenticity analysis",
    )
    styles = _styles()
    story: list[Any] = [_header(record, result, styles, app_version)]
    story.extend(_verdict_block(result, styles))
    story.extend(_reasons(result, styles))
    story.extend(_technique_table(result, styles))
    story.extend(_overlays(result, overlays, styles, ttl_s))
    story.extend(_provenance(result, styles))
    story.extend(_timings(record, result, styles))
    story.extend(_limitations(result, styles))
    story.append(Paragraph(_DISCLAIMER, styles["disclaimer"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} · "
        f"job {record.id} · TrueLens {app_version}",
        styles["footer"],
    ))

    doc.build(story)
    return buffer.getvalue()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    make = ParagraphStyle
    return {
        "h1": make("TLH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                   leading=19, textColor=_INK, spaceAfter=2 * mm),
        "h2": make("TLH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11,
                   leading=13, textColor=_INK, spaceBefore=5 * mm, spaceAfter=2 * mm),
        "label": make("TLLabel", parent=base["Normal"], fontName="Helvetica", fontSize=7.5,
                      leading=9, textColor=_MUTED, spaceAfter=1 * mm),
        "body": make("TLBody", parent=base["Normal"], fontName="Helvetica", fontSize=9,
                     leading=11.5, textColor=_INK),
        "small": make("TLSmall", parent=base["Normal"], fontName="Helvetica", fontSize=8,
                      leading=10, textColor=_MUTED),
        "mono": make("TLMono", parent=base["Normal"], fontName="Helvetica", fontSize=8,
                     leading=10, textColor=_INK),
        "cell": make("TLCell", parent=base["Normal"], fontName="Helvetica", fontSize=8,
                     leading=10, textColor=_INK),
        "disclaimer": make(
            "TLDisclaimer", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8.5,
            leading=11, textColor=_INK, spaceBefore=4 * mm, spaceAfter=2 * mm,
            borderPadding=4, borderColor=_RULE, borderWidth=0.5, alignment=TA_LEFT),
        "footer": make("TLFooter", parent=base["Normal"], fontName="Helvetica", fontSize=7,
                       leading=9, textColor=_MUTED, spaceBefore=2 * mm),
    }


def _header(record: JobRecord, result: dict[str, Any], styles, app_version: str):
    image = result.get("image") or {}
    width = image.get("width")
    height = image.get("height")
    identity = Table(
        [
            [Paragraph("File hash (SHA-256)", styles["label"]),
             Paragraph("Dimensions", styles["label"]),
             Paragraph("Container", styles["label"]),
             Paragraph("Engine", styles["label"])],
            [Paragraph(_mono(str(result.get("sha256") or record.sha256)), styles["mono"]),
             Paragraph(
                 f"{width} × {height} px" if width and height else "not decoded",
                 styles["cell"]),
             Paragraph(
                 f"{record.mime}, {int(record.size_bytes):,} bytes", styles["cell"]),
             Paragraph(f"v{app_version}", styles["cell"])],
        ],
        colWidths=[78 * mm, 34 * mm, 34 * mm, 26 * mm],
    )
    identity.setStyle(_table_style())
    return KeepTogether([
        Paragraph("Image authenticity report", styles["h1"]),
        Paragraph(
            "Automated analysis. A likelihood, not a finding of fact.", styles["small"]),
        Spacer(1, 3 * mm),
        identity,
    ])


def _verdict_block(result: dict[str, Any], styles):
    band = str(result.get("band") or "not_measured")
    colour = _BAND_COLORS.get(band, _NONE)
    verdict = str(result.get("verdict") or "inconclusive")
    rows = [
        [Paragraph("Conclusion", styles["label"]), Paragraph("Confidence in the evidence",
                                                             styles["label"])],
        [Paragraph(_escape(_verdict_sentence(verdict)), styles["h2"]),
         Paragraph(_pct(result.get("reliability")), styles["h2"])],
    ]
    table = Table(rows, colWidths=[110 * mm, 62 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, _RULE),
        ("TEXTCOLOR", (0, 1), (0, 1), colour),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    headline = str(result.get("headline") or "")
    detail = str(result.get("detail") or "")
    scores = Table(
        [
            [Paragraph("Synthetic likelihood", styles["label"]),
             Paragraph("Manipulation likelihood", styles["label"])],
            [Paragraph(_pct(result.get("ai_score")), styles["body"]),
             Paragraph(_pct(result.get("edited_score")), styles["body"])],
            [Paragraph(_prob(result.get("ai_score")), styles["small"]),
             Paragraph(_prob(result.get("edited_score")), styles["small"])],
        ],
        colWidths=[86 * mm, 86 * mm],
    )
    scores.setStyle(_table_style())

    blocks: list[Any] = [Paragraph("Conclusion", styles["h2"]), table]
    if headline:
        blocks.append(Spacer(1, 2 * mm))
        blocks.append(Paragraph(_escape(headline), styles["body"]))
    if detail:
        blocks.append(Spacer(1, 1.5 * mm))
        blocks.append(Paragraph(_escape(detail), styles["small"]))
    blocks.extend([Spacer(1, 3 * mm), scores])

    penalties = (result.get("reliability_detail") or {}).get("reasons") or []
    if penalties:
        blocks.append(Spacer(1, 2 * mm))
        lines = [Paragraph("Why the reliability is not higher", styles["label"])]
        for item in penalties:
            lines.append(Paragraph(
                f"• {_escape(str(item.get('reason', '')))} "
                f"(×{float(item.get('cost', 1.0)):.2f} on "
                f"{_escape(str(item.get('affects') or 'the total'))})",
                styles["small"]))
        blocks.append(Table([[lines]], colWidths=[172 * mm], style=_plain_style()))
    return blocks


def _reasons(result: dict[str, Any], styles):
    reasons = result.get("reasons") or []
    flags = result.get("flags") or []
    if not reasons and not flags:
        return []
    blocks: list[Any] = [Paragraph("Evidence relied upon", styles["h2"])]
    for reason in reasons:
        blocks.append(Paragraph(f"• {_escape(str(reason))}", styles["body"]))
    for flag in flags:
        target = flag.get("target")
        floor = flag.get("floor")
        tail = ""
        if target is not None and floor is not None:
            tail = f" — sets a floor of {_pct(floor)} on {_escape(str(target))}"
        elif flag.get("forces"):
            tail = f" — forces the conclusion to <b>{_escape(str(flag['forces']))}</b>"
        blocks.append(Paragraph(
            f"▪ <b>{_escape(str(flag.get('id', 'flag')))}</b> ({_escape(str(flag.get('kind', '')))}):"
            f" {_escape(str(flag.get('text', '')))}{tail}",
            styles["small"]))
    return blocks


_GROUP_ORDER = {"ai": 0, "edited": 1, "context": 2}
_GROUP_NAMES = {"ai": "synthetic signals", "edited": "manipulation signals",
                "context": "provenance"}


def _ordered(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Techniques in the order the on-screen report uses: track first, then label."""
    techniques = result.get("techniques") or {}
    return sorted(
        ((str(name), dict(view or {})) for name, view in techniques.items()),
        key=lambda kv: (
            _GROUP_ORDER.get(str(kv[1].get("group")), 3),
            str(kv[1].get("label") or kv[0]),
        ),
    )


def _technique_table(result: dict[str, Any], styles):
    ordered = _ordered(result)
    rows: list[list[Any]] = [[
        Paragraph("Check", styles["label"]),
        Paragraph("Result", styles["label"]),
        Paragraph("Band", styles["label"]),
        Paragraph("Weight", styles["label"]),
        Paragraph("What it found", styles["label"]),
    ]]
    colours: list[tuple[int, colors.Color]] = []
    for index, (name, view) in enumerate(ordered, start=1):
        band = str(view.get("band") or "not_measured")
        colour = _BAND_COLORS.get(band, _NONE)
        if not view.get("ran"):
            outcome = "not run"
        elif view.get("score") is None:
            outcome = "not measured"
        else:
            outcome = _pct(view.get("score"))
        explanation = view.get("explanation") or {}
        rows.append([
            Paragraph(
                f"{_escape(str(view.get('label') or name))}"
                f"<br/><font size=6.5 color='#5b6672'>{_escape(_group_name(view))}</font>",
                styles["cell"]),
            Paragraph(outcome, styles["cell"]),
            Paragraph(band.replace("_", " "), styles["cell"]),
            Paragraph(_escape(str(explanation.get("weight") or "—")), styles["cell"]),
            Paragraph(_escape(str(explanation.get("found") or view.get("error") or "—")),
                      styles["cell"]),
        ])
        colours.append((index, colour))

    table = Table(rows, colWidths=[36 * mm, 18 * mm, 20 * mm, 20 * mm, 78 * mm], repeatRows=1)
    style = _table_style()
    for row, colour in colours:
        style.add("TEXTCOLOR", (1, row), (1, row), colour)
    table.setStyle(style)
    return [Paragraph("Per-technique results", styles["h2"]), table]


def _group_name(view: dict[str, Any]) -> str:
    return _GROUP_NAMES.get(str(view.get("group")), "untracked")


def _overlays(result: dict[str, Any], overlays: dict[str, Path], styles, ttl_s: int):
    ordered = [
        (name, view) for name, view in _ordered(result)
        if view.get("overlay_url") or overlays.get(name)
    ]
    if not ordered:
        return [
            Paragraph("Visual evidence", styles["h2"]),
            Paragraph("The engine produced no overlay for this image — nothing was localised enough"
                      " to draw. That is an absence of a finding, not a finding of absence.",
                      styles["body"]),
        ]

    blocks: list[Any] = [Paragraph("Visual evidence", styles["h2"])]
    width = 84 * mm
    max_height = 92 * mm
    for name, view in ordered:
        label = str(view.get("label") or name)
        path = overlays.get(name)
        if path is None:
            blocks.append(Paragraph(
                f"{_escape(label)}: the heatmap for this check has expired with the upload "
                f"(links live about {int(ttl_s / 60)} minutes). The score above stands.",
                styles["small"]))
            continue
        prepared = _prepare_overlay(path)
        if prepared is None:
            blocks.append(Paragraph(
                f"{_escape(label)}: the stored image could not be read.", styles["small"]))
            continue
        draw_w, draw_h = _fit(prepared["width"], prepared["height"], width, max_height)
        image = RLImage(io.BytesIO(prepared["data"]), draw_w, draw_h)
        caption = Paragraph(
            f"{_escape(label)} — {_escape(str((view.get('explanation') or {}).get('why') or ''))}",
            styles["small"])
        blocks.append(KeepTogether([image, Spacer(1, 1 * mm), caption, Spacer(1, 3 * mm)]))
    return blocks


#: A heatmap printed at 84 mm needs about 1000 px; the scratch files are full resolution.
_OVERLAY_MAX_PX = 1000


def _prepare_overlay(path: Path) -> dict[str, Any] | None:
    """Shrink and re-encode one overlay for print, or ``None`` when it cannot be read.

    Embedding the scratch PNG verbatim produced a 6 MB document, and a forensic report that cannot be
    emailed is not a report. The image is capped at 1000 px — above what 84 mm of paper resolves — and
    flattened onto white and re-encoded as JPEG, because a heat map's edges are smooth gradients and
    survive that loss while its caption stays crisp.
    """
    import io

    from PIL import Image

    try:
        with Image.open(path) as source:
            source.load()
            if source.mode in {"RGBA", "LA", "P"}:
                source = source.convert("RGBA")
                flat = Image.new("RGB", source.size, (255, 255, 255))
                flat.paste(source, mask=source.getchannel("A"))
                source = flat
            elif source.mode != "RGB":
                source = source.convert("RGB")
            longest = max(source.size)
            if longest > _OVERLAY_MAX_PX:
                ratio = _OVERLAY_MAX_PX / longest
                source = source.resize(
                    (max(1, round(source.width * ratio)), max(1, round(source.height * ratio))),
                    Image.LANCZOS)
            out = io.BytesIO()
            source.save(out, format="JPEG", quality=86, optimize=True)
            return {"data": out.getvalue(), "width": source.width, "height": source.height}
    except Exception:
        log.debug("overlay %s could not be prepared", path.name, exc_info=True)
        return None


def _fit(pixels_w: int, pixels_h: int, max_w: float, max_h: float) -> tuple[float, float]:
    """Landscape- or portrait-fit one image inside the print box, keeping its aspect ratio."""
    if pixels_w <= 0 or pixels_h <= 0:
        return max_w, max_w
    scale = min(max_w / pixels_w, max_h / pixels_h)
    return pixels_w * scale, pixels_h * scale


def _provenance(result: dict[str, Any], styles):
    metadata = result.get("metadata") or {}
    c2pa = result.get("c2pa") or {}
    blocks: list[Any] = [Paragraph("What the file claims about itself", styles["h2"])]

    camera = " / ".join(
        value for value in (metadata.get("camera_make"), metadata.get("camera_model")) if value)
    software = metadata.get("software")
    rows = [
        [Paragraph("Metadata", styles["label"]), Paragraph("Camera", styles["label"]),
         Paragraph("Software", styles["label"]), Paragraph("Content credentials", styles["label"])],
        [Paragraph(_escape(str(metadata.get("status") or "—")), styles["cell"]),
         Paragraph(_escape(camera) or "none recorded", styles["cell"]),
         Paragraph(_escape(str(software)) if software else "none recorded", styles["cell"]),
         Paragraph(_escape(str(c2pa.get("credential") or c2pa.get("status") or "—")),
                   styles["cell"])],
    ]
    table = Table(rows, colWidths=[40 * mm, 46 * mm, 42 * mm, 44 * mm])
    table.setStyle(_table_style())
    blocks.append(table)

    lines = [_finding_line(f) for f in (metadata.get("findings") or []) if isinstance(f, dict)]
    for hit in metadata.get("generator_hits") or []:
        lines.append(
            f"generator fingerprint “{hit.get('keyword')}” found in {hit.get('chunk')}")
    if c2pa.get("issuer"):
        lines.append(f"credential issued by {c2pa.get('issuer')}")
    if metadata.get("pii_redacted"):
        lines.append("location fields were withheld before reporting")
    if lines:
        blocks.append(Spacer(1, 2 * mm))
        for item in lines:
            blocks.append(Paragraph(f"• {_escape(item)}", styles["small"]))
        blocks.append(Paragraph(
            "A generator tag or a missing credential is a fact about the file's container. Both can"
            " be added, edited or removed by hand, and their absence proves nothing.", styles["small"]))
    return blocks


def _finding_line(finding: dict[str, Any]) -> str:
    """One analyser finding as a sentence, with the weight the fusion gave it named out."""
    text = str(finding.get("text") or finding.get("code") or "").strip()
    severity = str(finding.get("severity") or "info")
    return f"{text} — {severity} signal" if severity in ("weak", "moderate", "strong") else text


def _timings(record: JobRecord, result: dict[str, Any], styles):
    timings = result.get("timings") or {}
    per = timings.get("per_technique_ms") or {}
    rows = [[
        Paragraph("Total", styles["label"]),
        Paragraph("Decode", styles["label"]),
        Paragraph("Queue", styles["label"]),
        Paragraph("Analysis", styles["label"]),
        Paragraph("Reused", styles["label"]),
    ], [
        Paragraph(_ms(timings.get("total_ms")), styles["cell"]),
        Paragraph(_ms(timings.get("decode_ms")), styles["cell"]),
        Paragraph(_ms(timings.get("queue_ms")), styles["cell"]),
        Paragraph(_ms(timings.get("analysis_ms")), styles["cell"]),
        Paragraph("yes (identical hash)" if record.from_cache else "no", styles["cell"]),
    ]]
    table = Table(rows, colWidths=[34 * mm, 30 * mm, 30 * mm, 34 * mm, 44 * mm])
    table.setStyle(_table_style())
    blocks: list[Any] = [Paragraph("How this was computed", styles["h2"]), table]
    if per:
        blocks.append(Spacer(1, 1.5 * mm))
        blocks.append(Paragraph(
            "per check: " + ", ".join(f"{k} {_ms(v)}" for k, v in sorted(per.items())),
            styles["small"]))
    return blocks


def _limitations(result: dict[str, Any], styles):
    items = result.get("limitations") or []
    blocks: list[Any] = [Paragraph("Limitations of this method", styles["h2"])]
    if not items:
        blocks.append(Paragraph(
            "Detection methods lag the generators they target; re-encoding destroys evidence; a"
            " clean result is not proof of authenticity.", styles["body"]))
        return blocks
    for item in items:
        blocks.append(Paragraph(
            f"<b>{_escape(str(item.get('title', '')))}</b> — {_escape(str(item.get('body', '')))}",
            styles["body"]))
    return blocks


# --------------------------------------------------------------------------- small helpers


def _verdict_sentence(verdict: str) -> str:
    return {
        "authentic": "No manipulation found",
        "ai_generated": "Likely AI-generated",
        "edited": "Likely manipulated",
        "inconclusive": "Inconclusive",
    }.get(verdict, verdict.replace("_", " "))


def _pct(value: Any) -> str:
    if value is None:
        return "not measured"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "not measured"


def _prob(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"probability {float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def _ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.0f} ms" if number < 1000 else f"{number / 1000:.2f} s"


def _escape(text: str) -> str:
    """Reportlab parses paragraph markup, and every string here came from a stranger's file."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _mono(text: str) -> str:
    # A 64-character hash has to fit its cell; the head and tail are what a verifier compares.
    cleaned = _escape(text)
    if len(cleaned) <= 34:
        return cleaned
    return f"{cleaned[:18]}…{cleaned[-12:]}"


def _table_style() -> TableStyle:
    return TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, _RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, _INK),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f4f7")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])


def _plain_style() -> TableStyle:
    return TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ])
