# Deployment

Two hosted services and nothing else: a static Next.js site and a Python API. There is no database,
no queue, no object store, and no server-side copy of any image. The browser calls the API directly,
so the two halves are joined by one URL and one CORS allowlist entry.

```
Vercel  ──▶  truelens.vercel.app        (static, edge-cached)
Render  ──▶  truelens-api.onrender.com  (FastAPI, CPU, 1 worker)
             joined by NEXT_PUBLIC_API_BASE_URL  +  FRONTEND_ORIGIN
```

## 0. Where each half goes, and why the API is not a Vercel project

**Frontend → Vercel. Backend → Render, Fly, Railway, or any container host. Not both on Vercel.**

Importing this repository into Vercel without setting the root directory makes Vercel find
`backend/pyproject.toml`, match `backend/app/main.py` as a FastAPI entrypoint, install the whole
dependency tree into a function bundle, and fail with:

```
[LAMBDA_SIZE_EXCEEDED] Total bundle size (1170.89 MB) exceeds the maximum
```

The standard Python bundle limit is 500 MB uncompressed, and torch alone is most of that 1.17 GB.
There is a 5 GB "Large Functions" beta, and `functions.excludeFiles` can trim test and fixture
folders — **do not use either to force this app onto Vercel.** The size limit is the first symptom of
four architectural mismatches, each of which breaks after the bundle problem is solved:

| This app assumes | A Python Vercel Function gives |
|---|---|
| A `spawn` process pool of analysers sized to the machine, started once at boot | A request-scoped process; the pool is built and torn down per cold start |
| `CnnRuntime` loaded exactly once in the parent process, warm forever | A fresh torch import per invocation, seconds before any work starts |
| A background sweeper task and an in-memory job store shared by every poll | Separate instances, so a job created on one 404s when another answers the poll |
| Overlay PNGs written to a scratch directory and served by signed URL for ten minutes | An ephemeral filesystem per instance: the heatmap a viewer just loaded is gone on the next request |

Jobs here also run 6-12 seconds and are polled, which is exactly the shape Function timeouts and cold
starts are worst at. The container image in `backend/Dockerfile` exists precisely so this service can
run where those four assumptions are true.

If you want one host for everything, use Docker Compose on a single VM — `docker compose up --build`
gives you both halves with the same topology the code expects.

## 1. Backend on Render

**New → Web Service**, connect the repository, and set *Root Directory* to `backend`.

Either runtime works:

| Runtime | Build | Start command |
|---|---|---|
| Docker | uses `backend/Dockerfile` | the image's `CMD`, which honours the injected `PORT` |
| Python 3.11 | `uv sync --frozen --extra redis` | `uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

Docker is the better default because the CPU-only torch index is pinned in `pyproject.toml` and the
image is what that pinning was written for. The Python runtime is faster to iterate on and needs the
same environment variables.

Environment:

```
ENVIRONMENT=prod
OVERLAY_SIGNING_KEY=<32+ random bytes>       # python -c "import secrets; print(secrets.token_urlsafe(48))"
FRONTEND_ORIGIN=https://<your-vercel-domain>
PERF_PROFILE=render-1cpu
HEALTH_CHECK_PATH=/api/health
```

- `OVERLAY_SIGNING_KEY` is required: with `ENVIRONMENT=prod`, startup *refuses* to run without one,
  because an auto-generated key makes overlay URLs 403 on whichever worker did not mint them, and that
  is a miserable thing to debug from a log line.
- `PERF_PROFILE=render-1cpu` is not cosmetic. The preset drops CNN test-time augmentation, the JPEG
  ghost tile count and the copy-move feature budget, and lowers `OVERLAY_MAX_DIM`; on 1 vCPU the
  default profile finishes after the request deadline has passed, and the product then honestly
  reports a thin analysis. If you upgrade the instance, use `render-2cpu`.
- Keep `WEB_CONCURRENCY=1` (the default) unless you also add Redis. Two workers with the in-memory job
  store means a job created on one 404s when polled by the other, and `validate_prod` fails the start
  rather than letting you discover it.
- Free instances sleep between requests. The first analysis after a cold start pays for torch import
  and pool spawn, so the health pill reads "Engine offline" for a few seconds and the first job is
  slow. That is the price of the free tier, not a bug in the app.

**Artifacts.** Render has no persistent disk you can write at deploy time, and trained weights do not
belong in a git repository. So a deployed instance runs with `artifacts/` empty unless you add the
files deliberately, and the correct behaviour of that state is: the CNN reports unavailable, fusion
uses the labelled expert prior and says so in `fusion.source`, and the Methodology page shows its
"not trained yet" panel. To ship trained weights, either build them into a custom image
(`COPY artifacts/ /srv/truelens/artifacts/`) or fetch them in an entrypoint from private storage
before uvicorn starts. Do not bake a model trained on synthetic fixtures into a public deployment —
`ml/README.md` explains why that is worse than no model.

## 2. Frontend on Vercel

Import the repository and set *Root Directory* to `frontend`. Framework detection handles Next 16;
the build command is `npm run build` and the output is static plus two dynamic routes
(`/report/[jobId]`, `/history/[jobId]`) that render client-side shells.

Environment — in **Settings → Environment Variables**, for Production *and* Preview:

```
NEXT_PUBLIC_API_BASE_URL=https://truelens-api.onrender.com
```

`NEXT_PUBLIC_*` is inlined into the browser bundle **at build time**. Adding or changing it after a
deploy changes nothing until the next build, and the failure is silent: the site renders, every
request goes to the old origin, and the health pill never turns green. If you use preview
deployments, add the preview origins to the backend's `EXTRA_CORS_ORIGINS` (comma-separated) or the
browser will block them.

Two things this repository deliberately does not do:

- **No `engines.node` pin.** Next 16.4 requires Node `>=20.9.0`, and Vercel's default is its latest
  LTS, which satisfies that. An exact pin is worse than nothing: a version string that does not match
  a runtime Vercel offers fails the build with an engine-mismatch error, which is a common and
  entirely self-inflicted deployment failure.
- **No `output: "standalone"` on Vercel.** That option exists for self-hosting — it emits the traced
  file set `.next/standalone` that `frontend/Dockerfile` copies. Vercel runs its own output
  processing and never reads that directory, so `next.config.ts` enables it only when `VERCEL` is
  unset. Verified both ways: with `VERCEL=1` the build succeeds and produces no `standalone`
  directory; without it the directory is there for the image build.

## 3. Self-hosting with Docker

```bash
cp backend/.env.example backend/.env       # set OVERLAY_SIGNING_KEY
docker compose up --build                  # web :3000, api :8010
docker compose --profile redis up          # + redis, for STORE_BACKEND=redis
```

`docker-compose.yml` mounts `./artifacts` read-only into the API and puts the scratch directory on a
`tmpfs`, so uploads and overlays exist only in memory and are gone when the container stops. The one
build-time value is the API origin the browser will call:

```yaml
services:
  web:
    build:
      args:
        NEXT_PUBLIC_API_BASE_URL: https://api.example.com
```

It must be the address the *browser* can reach — not `http://api:8010`. A container-network name is
unreachable from a laptop, and the resulting page looks healthy until nothing answers.

Bring your own reverse proxy for TLS, and set `TRUST_PROXY=true` on the API only if that proxy
overwrites `X-Forwarded-For`. Otherwise every client can choose its own rate-limit bucket by sending
a header.

## 4. Verify a deployment

```bash
curl -s https://api.example.com/api/health | head -c 400
curl -s -o /dev/null -w "%{http_code}\n" https://api.example.com/api/metrics
curl -s -o /dev/null -w "%{http_code}\n" https://site.example
```

The API origin is not in the served HTML — `NEXT_PUBLIC_*` is inlined into a JS chunk, so grepping the
document finds nothing and a wrong value there is invisible to that check. Verify it where it
actually lands:

```bash
# locally, against the build output
grep -rl "$NEXT_PUBLIC_API_BASE_URL" .next/static/chunks | head -1
# deployed: open DevTools → Network, reload, and read the host the /api/health request went to
```

If that request goes to the old origin or never happens, the variable was set after the build.

Then in the browser: drop `samples/spliced.jpg` on `/analyze` and expect a verdict with a manipulation
sub-score above the synthetic one, an overlay that draws on the image, and a reliability figure with
its penalties itemised. Drop `samples/screenshot.png` and expect `inconclusive` — most checks cannot
run on it, and reporting a confident "Likely authentic" for a file nobody could inspect is the exact
failure the reliability layer exists to prevent.

```bash
curl -s -X POST https://api.example.com/api/analyze -F "file=@samples/spliced.jpg"
# → job_id; then poll:
curl -s https://api.example.com/api/jobs/<job_id> | head -c 300
```

## 5. What to watch

- `GET /api/health` reports `problems`, artifact presence, store backend, cache hit rate and live job
  counts. It is the page to open first, and it is what the Docker healthcheck calls.
- Reliability, not verdict, is the metric that says the service is healthy. A deployment whose average
  reliability drops has a machine too small for its profile, a missing artifact, or an analyser
  failing on real inputs.
- Rate limits are per address: `RATE_LIMIT_ANALYZE=20/minute` by default. Behind a proxy without
  `TRUST_PROXY`, every request shares the proxy's address and the whole site starts answering 429 for
  one busy visitor.
