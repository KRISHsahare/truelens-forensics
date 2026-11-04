"""Content Credentials (C2PA) verification.

C2PA is the only technique in the pipeline that reads a **cryptographically signed assertion**
rather than inferring from pixels. That makes it epistemically different, and we treat it
differently:

* **No manifest** — the overwhelmingly common case. *Not* evidence of authenticity. Essentially
  no consumer camera and no social platform signs images. Score 0, and the UI must say
  "no credentials found", never "verified authentic".
* **Valid manifest naming an AI generator** — a signed fact, not a statistic. Sets the
  deterministic evidence flag that floors ``ai_score``.
* **Manifest present but failing validation** — evidence that *something* happened to the
  provenance chain, but not evidence of generation. Lowers reliability and raises a flag; it
  must **not** raise ``ai_score``. Conflating the two would punish a legitimately-signed photo
  that was re-saved by a non-conformant tool.

The c2pa crate is Rust doing untrusted-input parsing, so this runs inside the same
failure-isolation wrapper as everything else: a panic in the parser degrades one technique
rather than the job.
"""

from __future__ import annotations

import io
import json
import re

from app.analysis.contract import TechniqueResult, clip01

__all__ = ["run"]

#: IPTC digital source types that declare algorithmic/AI creation.
_AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia",
    "compositeai",
    "algorithmicmedia",
)
#: Substrings in ``claim_generator`` that identify a generative tool.
_AI_GENERATORS = (
    "firefly", "dall", "midjourney", "stable diffusion", "stability", "flux",
    "comfyui", "automatic1111", "novelai", "runway", "pika", "ideogram",
    "recraft", "krea", "leonardo", "openai", "anthropic", "google ai",
    "imagen", "gemini", "copilot", "bing image", "dream studio",
)
#: Actions that indicate the pixels were changed after capture.
_EDIT_ACTIONS = (
    "c2pa.edited", "c2pa.opened", "c2pa.converted", "c2pa.cropped",
    "c2pa.resized", "c2pa.adjusted", "c2pa.filtered", "com.adobe.photoshop.",
)
_CREATE_ACTIONS = ("c2pa.created", "c2pa.captured", "c2pa.published")


def run(orig_bytes: bytes, mime: str = "image/jpeg") -> TechniqueResult:
    try:
        import c2pa
    except ImportError:
        return TechniqueResult(0.0, None, {
            "status": "unavailable", "present": False,
            "note": "The c2pa library is not installed; Content Credentials were not checked.",
        })

    try:
        reader = c2pa.Reader(format_or_path=mime, stream=io.BytesIO(orig_bytes))
    except c2pa.C2paError as exc:
        if isinstance(exc, c2pa.C2paError.ManifestNotFound):
            return _no_manifest_result()
        return TechniqueResult(0.0, None, {
            "status": "unreadable", "present": False,
            "error": _safe(str(exc)),
            "note": "A manifest may be present but could not be parsed.",
            "reliability_penalty": "c2pa_unreadable",
        })
    except Exception as exc:
        return TechniqueResult(0.0, None, {
            "status": "error", "present": False, "error": _safe(str(exc)),
            "reliability_penalty": "c2pa_unreadable",
        })

    try:
        payload = json.loads(reader.json())
    except Exception as exc:
        return TechniqueResult(0.0, None, {
            "status": "malformed_json", "present": True, "error": _safe(str(exc)),
            "reliability_penalty": "c2pa_invalid",
        })

    validation_state = _validation_state(reader)
    is_valid = _is_valid(reader, validation_state)
    claim_generator = str(payload.get("claim_generator") or "").strip()
    title = str(payload.get("title") or "").strip()
    fmt = str(payload.get("format") or "").strip()
    signature = _signature_info(payload)
    assertions = _walk_assertions(payload)
    blob = json.dumps(payload).lower()

    ai_source_declared = any(t in blob for t in _AI_SOURCE_TYPES)
    generator_named = any(g in claim_generator.lower() for g in _AI_GENERATORS)
    edit_actions = sorted({a for a in _EDIT_ACTIONS if a in blob})
    create_actions = sorted({a for a in _CREATE_ACTIONS if a in blob})

    # ---------------------------------------------------------------- scoring
    deterministic_flag = None
    score = 0.0
    summary: list[str] = []

    if not is_valid:
        # Present but not trustworthy: do NOT raise the AI score.
        summary.append("A Content Credentials manifest is present but does not validate.")
        deterministic_flag = "c2pa_invalid"
        score = 0.25
    elif ai_source_declared or generator_named:
        summary.append(
            f"A valid C2PA manifest declares AI generation"
            f"{' via ' + claim_generator if claim_generator else ''}."
        )
        deterministic_flag = "c2pa_ai_generator"
        score = 0.98
    elif edit_actions:
        summary.append(
            f"A valid manifest records post-capture edits ({', '.join(edit_actions[:3])})."
        )
        deterministic_flag = "c2pa_edit_declared"
        score = 0.70
    elif create_actions and signature:
        summary.append(
            "A valid manifest asserts capture/creation provenance with no edit history."
        )
        deterministic_flag = "c2pa_authentic_capture"
        score = 0.02
    else:
        summary.append("A valid manifest is present but makes no generation or edit assertion.")
        score = 0.10

    details = {
        "status": "verified" if is_valid else "invalid",
        "present": True,
        "valid": is_valid,
        "validation_state": validation_state,
        "claim_generator": claim_generator or None,
        "title": title or None,
        "declared_format": fmt or None,
        "signature": signature,
        "ai_source_type_declared": ai_source_declared,
        "generator_named": generator_named,
        "edit_actions": edit_actions,
        "create_actions": create_actions,
        "assertion_uris": assertions[:16],
        "summary": summary,
        "deterministic_flag": deterministic_flag,
        "note": "A signed manifest is a cryptographic assertion about provenance, which is "
                "stronger evidence than any pixel-level statistic in this report.",
    }
    try:
        reader.close()
    except Exception:
        pass
    return TechniqueResult(clip01(score), None, details)


def _no_manifest_result() -> TechniqueResult:
    return TechniqueResult(0.0, None, {
        "status": "absent",
        "present": False,
        "valid": None,
        "summary": ["No Content Credentials (C2PA) manifest is embedded in this file."],
        "note": "Absence is NOT evidence of authenticity. Most cameras do not sign their "
                "output, and social platforms, messengers and screenshot tools strip "
                "manifests from genuine photographs.",
        "deterministic_flag": None,
    })


def _validation_state(reader: object) -> str:
    try:
        state = reader.get_validation_state()  # type: ignore[attr-defined]
        name = getattr(state, "name", None)
        return str(name or state).lower()
    except Exception:
        return "unknown"


def _is_valid(reader: object, validation_state: str) -> bool:
    """A manifest counts as valid only if the signature checks out.

    We require both ``is_valid()`` and a non-invalidating ``validation_state``. Trusting either
    alone lets a tampered manifest through: some c2pa versions report ``is_valid() == True`` for
    a structurally-sound store whose signature does not chain.
    """
    try:
        flag = bool(reader.is_valid())  # type: ignore[attr-defined]
    except Exception:
        flag = False
    if validation_state in ("invalid", "invalidated", "untrusted"):
        return False
    return flag


def _signature_info(payload: dict) -> dict:
    sig = payload.get("signature_info") or payload.get("signature") or {}
    if not isinstance(sig, dict):
        return {"raw": _safe(str(sig))}
    return {
        "issuer": sig.get("issuer") or sig.get("cert_issuer"),
        "signer": sig.get("signer") or sig.get("cert_signer"),
        "time": sig.get("time") or sig.get("signing_time"),
        "algorithm": sig.get("alg") or sig.get("algorithm"),
        "trusted": sig.get("trusted"),
    }


def _walk_assertions(payload: dict) -> list[str]:
    """Collect assertion URIs from whichever nesting shape this c2pa version emits."""
    found: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key in ("uri", "url", "assertion"):
                val = node.get(key)
                if isinstance(val, str) and val.startswith("self#jumbf="):
                    found.append(val)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return list(dict.fromkeys(found))


_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)[^\s'\"]+")


def _safe(text: str, limit: int = 240) -> str:
    """Strip filesystem paths from native-library error text before it reaches the API."""
    cleaned = _PATH_RE.sub("<path>", text)
    return " ".join(cleaned.split())[:limit]
