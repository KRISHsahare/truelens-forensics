/**
 * The only place that talks to the backend.
 *
 * Three rules the rest of the app depends on:
 *
 * - **No proxying.** The frontend calls the API origin directly (`NEXT_PUBLIC_API_BASE_URL`),
 *   because that is the shipped topology — static site on Vercel, Python service on Render — and
 *   because a same-origin proxy would put a 10 MB upload cap behind a proxy buffer limit that
 *   defaults to 10 MB. CORS on the backend is locked to this app's origin for exactly this reason.
 * - **Overlay URLs are relative.** The backend signs the *path and query* only, so a URL minted
 *   behind a proxy stays valid on whatever hostname the browser used. `absoluteOverlay()` adds the
 *   origin back at render time.
 * - **Every failure becomes an `ApiError` with the server's own sentence.** The backend returns one
 *   envelope for all errors; a client that invents its own copy where a server already explained
 *   itself will contradict it.
 */

import {
  ApiError,
  type ApiErrorBody,
  type BatchResult,
  type HealthResponse,
  type JobCreated,
  type JobStatus,
  type MetricsResponse,
} from "./types";

/** Where the API lives. Set this in `.env.local` for local dev and `.env.production` for deploy. */
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000")
  .trim()
  .replace(/\/+$/, "");

/** Mirrors `MAX_UPLOAD_BYTES` on the server (10 MB). Checked before uploading to save the bandwidth. */
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

/** Mirrors `ALLOWED_FORMATS`. The server checks magic bytes, not this list — this is a faster no. */
export const ACCEPTED_MIME = ["image/jpeg", "image/png", "image/webp", "image/bmp"] as const;

export const ACCEPT_ATTRIBUTE = "image/jpeg,image/png,image/webp,image/bmp,.jpg,.jpeg,.png,.webp,.bmp";

/** Mirrors `BATCH_MAX_IMAGES`. */
export const BATCH_MAX_IMAGES = Number(process.env.NEXT_PUBLIC_BATCH_MAX_IMAGES ?? 20);

export interface UploadCheck {
  ok: boolean;
  reason?: string;
}

/** A cheap pre-flight, so a 40 MB phone photo is refused in the browser rather than in transit. */
export function checkFile(file: File | Blob, name?: string): UploadCheck {
  const bytes = file.size;
  if (bytes === 0) return { ok: false, reason: `${name ?? "That file"} is empty.` };
  if (bytes > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      reason: `${name ?? "That file"} is ${(bytes / 1024 / 1024).toFixed(1)} MB. The limit is 10 MB.`,
    };
  }
  const mime = file.type || "";
  if (mime && !mime.startsWith("image/")) {
    return { ok: false, reason: `${name ?? "That file"} is not an image.` };
  }
  return { ok: true };
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init.headers ?? {}) },
  });
  const text = await response.text();
  let parsed: unknown = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = null;
  }
  if (!response.ok) {
    throw new ApiError(
      response.status,
      (parsed ?? null) as ApiErrorBody | null,
      `The server answered ${response.status}.`,
    );
  }
  return parsed as T;
}

export function analyzeFile(file: File | Blob, filename = "upload", signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", file, filename);
  return request<JobCreated>("/api/analyze", { method: "POST", body: form, signal });
}

/** A URL the *server* fetches — the browser never touches the third-party host. */
export function analyzeUrl(url: string, signal?: AbortSignal) {
  const form = new FormData();
  form.append("url", url);
  return request<JobCreated>("/api/analyze", { method: "POST", body: form, signal });
}

export function pollJob(jobId: string, signal?: AbortSignal) {
  return request<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    method: "GET",
    signal,
    cache: "no-store",
  });
}

export function batchAnalyze(files: File[], labels?: string[], signal?: AbortSignal) {
  const form = new FormData();
  files.forEach((file, i) => form.append("files", file, file.name || `image-${i + 1}`));
  if (labels?.length) form.append("labels", labels.join(","));
  return request<BatchResult>("/api/batch", { method: "POST", body: form, signal });
}

export function getHealth(signal?: AbortSignal) {
  return request<HealthResponse>("/api/health", { method: "GET", signal, cache: "no-store" });
}

export function getMetrics(signal?: AbortSignal) {
  return request<MetricsResponse>("/api/metrics", { method: "GET", signal });
}

/**
 * An overlay URL is a *signed, expiring* path.
 *
 * Fetch it as a blob rather than pointing `<img src>` straight at the API: the response is
 * `no-store`, the signature expires with the job, and the viewer needs the decoded pixels anyway
 * for the swipe/zoom compositing. Fall back to the absolute URL when a caller only needs a thumb.
 */
export function absoluteOverlay(overlayUrl: string | null | undefined): string | null {
  if (!overlayUrl) return null;
  if (/^https?:/i.test(overlayUrl)) return overlayUrl;
  return `${API_BASE_URL}${overlayUrl}`;
}

export async function fetchOverlay(
  overlayUrl: string,
  signal?: AbortSignal,
): Promise<Blob | null> {
  const absolute = absoluteOverlay(overlayUrl);
  if (!absolute) return null;
  try {
    const response = await fetch(absolute, { signal, cache: "no-store" });
    if (!response.ok) return null;
    return await response.blob();
  } catch {
    // An expired signature and a deleted file are deliberately indistinguishable on the wire.
    return null;
  }
}
