# TrueLens API. CPU-only inference: the whole point of the image is that it runs on a 1-2 vCPU
# instance, so no CUDA layer belongs in it (~900 MB instead of ~6 GB).
#
#   docker build -t truelens-api backend/
#   docker run -p 8010:8010 -e OVERLAY_SIGNING_KEY=... -e FRONTEND_ORIGIN=http://localhost:3000 truelens-api
#
# Not verified by building on the authoring machine: this repository was assembled on Windows
# without a running Docker daemon, so the compose file is config-validated and the Python side is
# exercised directly (`uv run pytest`). Treat the first `docker build` as a check, not a formality.
FROM python:3.11-slim AS builder

# Pinned to the uv that produced `uv.lock` (lock format revision 3). A materially older uv cannot
# read that lock, and `--frozen` is the guarantee that the image is the lock rather than a fresh
# resolution of the range in pyproject.toml.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
# Dependencies first, source second: this layer survives any change under `app/`.
#
# `--no-install-project` because the service is launched as `uvicorn app.main:app` from the workdir,
# so the project does not need to be *importable as a package* — and skipping it removes a hatchling
# build step, a network fetch for its build requirements, and a second copy of `app/` inside the venv
# that would shadow the one mounted at runtime.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra redis

# The redis extra is in the image so `STORE_BACKEND=redis` works without a rebuild; the memory
# backend remains the default and needs no server.
COPY app ./app
COPY scripts ./scripts

FROM python:3.11-slim AS runtime

# opencv-python-headless still needs the GNU C++/gomp runtime; without libgomp1 the import fails at
# startup with an error naming neither library, because cv2 is imported inside a worker process.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The container is a deployment, so the prod validations run: a missing overlay signing key and
    # a wide web concurrency on the in-memory store both fail at startup instead of producing 403s
    # and 404s for a user who has no way to know which worker answered them.
    ENVIRONMENT=prod \
    PERF_PROFILE=render-2cpu \
    HOST=0.0.0.0 \
    PORT=8010 \
    FRONTEND_ORIGIN=http://localhost:3000 \
    SCRATCH_ROOT=/data/scratch \
    TRUST_PROXY=true

WORKDIR /srv/truelens
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY scripts ./scripts
COPY pyproject.toml ./

# Trained files are mounted, never baked: they are gigabytes, they are licensed per dataset
# agreement, and a rebuilt model should not require a rebuilt image.
#   -v ./artifacts:/srv/truelens/artifacts:ro
RUN mkdir -p /srv/truelens/artifacts /data/scratch \
 && useradd --system --create-home --uid 10001 truelens \
 && chown -R truelens:truelens /data /srv/truelens
USER truelens

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# `exec` form would be faster to start, but the shell is here for one reason: Render publishes the
# app on a port it chooses and injects `PORT`, and `${PORT:-8010}` needs a shell to expand.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8010} --workers 1 --no-server-header --log-level info"]
