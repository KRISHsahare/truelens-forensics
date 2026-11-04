"""Runtime configuration.

Everything tunable lives here so the same image can run on a 12-core dev box and a
1-vCPU container by changing ``PERF_PROFILE`` alone.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.envbootstrap import POOL_MAX_WORKERS

PerfProfile = Literal["dev-12core", "render-2cpu", "render-1cpu"]

# Per-profile quality/cost knobs. Each entry trades forensic sensitivity for latency;
# the trade-offs are surfaced to the user via reliability penalties so the UI stays honest.
_PROFILE_PRESETS: dict[PerfProfile, dict[str, float | int | bool]] = {
    # `pool_max_workers` is in the presets because leaving it out was a live failure: the pool size
    # otherwise resolves from os.cpu_count(), and a container reports the *host's* cores. Render's
    # free instance showed 12 cores for a 0.1 CPU allocation, so the service spawned seven children
    # that could not finish starting before the request deadline -- every pooled technique came back
    # "skipped" and the analysis had nothing to say. A pool sized to the machine the profile
    # describes is the point of the profile.
    "dev-12core": {
        "ghost_tiles": 3,
        "copy_move_features": 3000,
        "cnn_tta": True,
        "gradcam_min_ai_score": 0.45,
        "overlay_max_dim": 1280,
        "max_concurrent_jobs": 4,
        "job_soft_deadline_s": 6.0,
        "job_hard_deadline_s": 12.0,
    },
    "render-2cpu": {
        "ghost_tiles": 2,
        "copy_move_features": 2000,
        "cnn_tta": True,
        "gradcam_min_ai_score": 0.45,
        "overlay_max_dim": 1280,
        "max_concurrent_jobs": 2,
        "pool_max_workers": 2,
        "job_soft_deadline_s": 8.0,
        "job_hard_deadline_s": 15.0,
    },
    "render-1cpu": {
        "ghost_tiles": 1,
        "copy_move_features": 1200,
        "cnn_tta": False,
        "gradcam_min_ai_score": 0.60,
        "overlay_max_dim": 1024,
        "max_concurrent_jobs": 1,
        "pool_max_workers": 1,
        "job_soft_deadline_s": 10.0,
        "job_hard_deadline_s": 20.0,
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- identity ---
    app_name: str = "TrueLens"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # --- network ---
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_origin: str = "http://localhost:3000"
    # Comma-separated for preview deployments.
    extra_cors_origins: str = ""

    # --- upload limits ---
    max_upload_bytes: int = 10 * 1024 * 1024  # product spec: 10 MB
    max_analysis_pixels: int = 60_000_000
    analysis_max_dim: int = 2048
    native_max_dim: int = 2560
    allowed_formats: str = "image/jpeg,image/png,image/webp,image/bmp"

    # --- performance ---
    perf_profile: PerfProfile = "dev-12core"
    pool_max_workers: int = POOL_MAX_WORKERS
    analyzer_parallel: bool = True
    torch_threads: int = 0  # 0 = derive from profile
    ghost_tiles: int = 3
    copy_move_features: int = 3000
    cnn_tta: bool = True
    gradcam_min_ai_score: float = 0.45
    overlay_max_dim: int = 1280
    max_concurrent_jobs: int = 4
    job_soft_deadline_s: float = 6.0
    job_hard_deadline_s: float = 12.0

    # --- model artifacts ---
    cnn_weights_path: str = "artifacts/model.pt"
    fusion_path: str = "artifacts/fusion.joblib"
    metrics_path: str = "artifacts/metrics.json"
    clip_probe_path: str = "artifacts/clip_probe.joblib"
    enable_clip_probe: bool = False
    yunet_model_path: str = "app/ml/models/face_detection_yunet_2023mar.onnx"

    # --- storage / lifecycle ---
    scratch_root: str = ""  # empty => system temp
    job_ttl_s: int = 600
    overlay_ttl_s: int = 600
    # Strictly less than overlay_ttl_s: a cache entry must never outlive the files it
    # references, otherwise a cache hit hands out URLs that 410.
    result_cache_ttl_s: int = 540
    result_cache_max_entries: int = 256
    sweeper_interval_s: int = 30
    store_backend: Literal["memory", "redis"] = "memory"
    #: Hard ceiling on how many job snapshots the in-memory store holds, independent of TTL. Sized
    #: as a backstop for a burst: at ``rate_limit_analyze``=20/minute and a 600 s TTL the working
    #: set is ~200, so 512 is headroom rather than a limit anyone reaches in normal use.
    job_store_max_entries: int = 512
    redis_url: str = ""
    web_concurrency: int = 1

    # --- security ---
    overlay_signing_key: str = ""
    rate_limit_analyze: str = "20/minute"
    rate_limit_jobs: str = "200/minute"
    rate_limit_batch: str = "5/minute"
    #: Read the client IP from ``X-Forwarded-For`` instead of the socket.
    #: Only true behind a proxy that overwrites the header (Render, nginx with
    #: ``proxy_set_header``). A client can send any header it likes, so trusting an unsanitised
    #: chain lets everyone pick their own rate-limit bucket.
    trust_proxy: bool = False
    redact_pii: bool = True
    batch_max_images: int = 20

    @field_validator("extra_cors_origins", "scratch_root", "redis_url", mode="before")
    @classmethod
    def _empty_str(cls, v: object) -> object:
        return "" if v is None else v

    @property
    def cors_origins(self) -> list[str]:
        origins = [self.frontend_origin]
        origins += self.extra_cors_origins.split(",")
        # Normalised rather than taken literally, because CORS compares exact strings and the
        # failure is total and silent: the server answers 200, the browser refuses to hand the body
        # to the page, and the only symptom is a frontend that says it could not reach a backend
        # that is demonstrably running. `truelens.example.com` and
        # `https://truelens.example.com/` are the two typos that cause it, and neither is a
        # plausible thing to make an operator debug by hand.
        origins = [_normalize_origin(o) for o in origins]
        if self.environment == "dev":
            origins += [_loopback_sibling(o) for o in list(origins)]
        return [o for o in dict.fromkeys(origins) if o]

    def apply_profile(self) -> None:
        """Fill unset perf knobs from the profile preset.

        Only overrides values the user did not set explicitly, so a single env var
        (e.g. ``COPY_MOVE_FEATURES=500``) still wins over the preset.

        The comparison is upper-cased on both sides because ``pydantic-settings`` reads env vars
        case-insensitively: comparing ``key.upper()`` against a lower-cased set of names -- which is
        what this did -- means the membership test can never succeed, every preset value wins, and
        the promise in the paragraph above is quietly false for every knob in the profile.
        """
        preset = _PROFILE_PRESETS[self.perf_profile]
        explicit = {k.upper() for k in os.environ}
        for key, default in preset.items():
            if key.upper() not in explicit:
                setattr(self, key, default)
        if self.torch_threads <= 0:
            cpus = os.cpu_count() or 2
            self.torch_threads = {"dev-12core": min(4, cpus // 3),
                                  "render-2cpu": 1, "render-1cpu": 1}[self.perf_profile]

    def validate_prod(self) -> list[str]:
        """Hard startup checks. Returns a list of fatal problems."""
        problems: list[str] = []
        if self.environment == "prod":
            if not self.overlay_signing_key:
                problems.append(
                    "OVERLAY_SIGNING_KEY must be set in prod; overlay URLs are HMACs over their path "
                    "and expiry, so without a configured key either every worker invents its own (a "
                    "URL minted on one 403s on the next) or the development constant is used, which "
                    "is published in this repository and lets anyone sign any overlay path."
                )
            if not self.cors_origins:
                problems.append(
                    "FRONTEND_ORIGIN must name the origin the site is served from (e.g. "
                    "https://app.example.com). CORS compares exact strings, so an empty or "
                    "unparseable value lets no browser read any response: the API answers 200 and "
                    "the page reports that the engine is unreachable.")
            if self.web_concurrency > 1 and self.store_backend == "memory":
                problems.append(
                    f"WEB_CONCURRENCY={self.web_concurrency} with STORE_BACKEND=memory: jobs "
                    "created on one worker 404 when polled on another. Use redis or set "
                    "WEB_CONCURRENCY=1."
                )
        if self.result_cache_ttl_s >= self.overlay_ttl_s:
            problems.append(
                f"RESULT_CACHE_TTL_S ({self.result_cache_ttl_s}) must be < OVERLAY_TTL_S "
                f"({self.overlay_ttl_s}) or cached results reference deleted overlays."
            )
        if self.job_soft_deadline_s >= self.job_hard_deadline_s:
            problems.append("JOB_SOFT_DEADLINE_S must be < JOB_HARD_DEADLINE_S.")
        return problems


#: Settings fields that name a trained file. All are resolved to absolute paths at startup.
_ARTIFACT_FIELDS = (
    "cnn_weights_path", "fusion_path", "metrics_path", "clip_probe_path", "yunet_model_path")


def _artifact_bases() -> list[Path]:
    """Directories a relative artifact path may be resolved against, in order.

    The current directory first, because that is the contract in Docker and on Render — the
    workdir contains ``artifacts/``. The two following entries are what make a monorepo checkout
    usable without setting anything: a backend started from ``backend/`` finds
    ``backend/artifacts``, and one started from the repository root finds ``artifacts/`` there.
    Without this, the same ``docker compose`` tree works in the container and silently reports
    "no weights" on a laptop, which is the kind of bug that gets papered over with a symlink.
    """
    here = Path(__file__).resolve()
    return [Path.cwd(), here.parents[1], here.parents[2]]


def _resolve_artifacts(settings: Settings) -> None:
    """Rewrite relative artifact paths to whichever base actually contains them.

    A path that exists nowhere is left alone rather than guessed: the health endpoint then reports
    the artifact missing with the path the operator actually configured, which is a better
    diagnostic than a plausible-looking absolute path to a file that was never there.
    """
    for field in _ARTIFACT_FIELDS:
        raw = getattr(settings, field)
        if not raw or Path(raw).is_absolute():
            continue
        for base in _artifact_bases():
            candidate = base / raw
            if candidate.exists():
                setattr(settings, field, str(candidate))
                break


def _normalize_origin(value: str) -> str:
    """A browser `Origin` string, or "" if the input cannot become one.

    Browsers send `https://host` with no scheme-less or trailing-slash variant, so a configured
    origin missing either never matches anything and the whole site stops reading responses while
    every server-side health check still passes. A bare host is completed to `https://` because in
    production that is the only scheme it can legitimately be served over.
    """
    v = (value or "").strip().rstrip("/")
    if not v:
        return ""
    if "://" not in v:
        v = f"https://{v}"
    return v


def _loopback_sibling(origin: str) -> str:
    """The other spelling of the same local address, or the origin unchanged.

    ``localhost`` and ``127.0.0.1`` are one server to a human and two origins to a browser. The
    frontend's own ``.env.example`` points at ``http://127.0.0.1:8010`` while the default here says
    ``http://localhost:3000``, so a developer who followed the repo's instructions got a page whose
    every request was CORS-blocked and whose only symptom was a pill that never stopped spinning.
    Dev only: a deployment names the origin it actually serves, and this widens nothing there.
    """
    for a, b in (("://localhost:", "://127.0.0.1:"), ("://127.0.0.1:", "://localhost:")):
        if a in origin:
            return origin.replace(a, b, 1)
    return origin


#: Stand-in key for local development, where overlay URLs never leave the machine. Named as a
#: constant so the prod path can be tested against the value it must never fall back to.
_DEV_OVERLAY_KEY = "dev-insecure-overlay-signing-key-change-me"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.apply_profile()
    _resolve_artifacts(s)
    if not s.overlay_signing_key and s.environment != "prod":
        # Dev convenience only, and only outside production. Substituting this in prod would make
        # the `validate_prod` check below unsatisfiable-but-passing: it tests for an empty key, the
        # fallback fills one in first, and the service then boots signing with a constant that is
        # published in this repository -- which is the exact "looks like someone else's intermittent
        # bug" failure the check exists to prevent.
        s.overlay_signing_key = _DEV_OVERLAY_KEY
    return s


def reset_settings_cache() -> None:
    """Test helper: drop the cached Settings so env changes take effect."""
    get_settings.cache_clear()
