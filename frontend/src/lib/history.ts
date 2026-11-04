/**
 * History, in the browser and nowhere else.
 *
 * The product promise is that the server stores no images: uploads are decoded in memory, overlays
 * are deleted inside ten minutes, and there is no account. So the history page has to be built from
 * data the *browser* kept. IndexedDB is the only storage that survives a reload, is same-origin, is
 * never sent to a server, and is wipeable by the user in one click.
 *
 * What is stored, and what is not:
 *
 * - The full result envelope, so a past analysis can be re-read offline. The overlay URLs inside it
 *   are signed and expire with the scratch files, so a replayed history entry shows scores, reasons
 *   and metadata but no heatmaps — that is the correct behaviour, not a bug: the pixels belonged to
 *   a ten-minute window.
 * - A 128 px JPEG thumbnail, drawn from the file the user already has. It is the only image byte
 *   retained, it is derived in the browser, and `useThumbnailsEnabled` in `settings.ts` is the setting
 *   that skips it.
 * - Never the filename. `label` is what the user sees in the table; the backend itself refuses to
 *   persist a filename, and re-adding it here would undo that.
 *
 * All of this is client-only. Every entry point is guarded, because these calls would otherwise run
 * during a server render where `indexedDB` does not exist.
 */

import type { AnalysisResult, VerdictLabel } from "./types";

const DB_NAME = "truelens-history";
const DB_VERSION = 1;
const STORE = "analyses";

/** Enough rows to survive a working session; oldest entries are evicted past this. */
export const HISTORY_CAP = 60;

export interface HistoryEntry {
  job_id: string;
  created_at: number;
  sha256: string;
  verdict: VerdictLabel;
  overall: number;
  ai_score: number;
  edited_score: number;
  reliability: number;
  headline: string;
  mime: string;
  size_bytes: number;
  width: number | null;
  height: number | null;
  from_cache: boolean;
  /** Data URL, or null when the user turned thumbnails off or the image could not be decoded. */
  thumbnail: string | null;
  /** The complete envelope, kept so an old analysis still opens without the server. */
  result: AnalysisResult;
}

/** Columns the list view needs; the full record is only fetched when a row is opened. */
export type HistorySummary = Pick<
  HistoryEntry,
  | "job_id"
  | "created_at"
  | "sha256"
  | "verdict"
  | "overall"
  | "ai_score"
  | "edited_score"
  | "reliability"
  | "headline"
  | "mime"
  | "size_bytes"
  | "width"
  | "height"
  | "from_cache"
  | "thumbnail"
>;

export class HistoryUnavailable extends Error {
  constructor() {
    super("This browser will not keep a history (private mode or blocked storage).");
    this.name = "HistoryUnavailable";
  }
}

export function historySupported(): boolean {
  return typeof indexedDB !== "undefined";
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!historySupported()) {
      reject(new HistoryUnavailable());
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "job_id" });
        store.createIndex("created_at", "created_at");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new HistoryUnavailable());
    request.onblocked = () => reject(new HistoryUnavailable());
  });
}

function tx<T>(storeMode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const transaction = db.transaction(STORE, storeMode);
        const request = run(transaction.objectStore(STORE));
        request.onsuccess = () => {
          resolve(request.result);
          db.close();
        };
        request.onerror = () => {
          reject(request.error);
          db.close();
        };
      }),
  );
}

export async function putEntry(entry: HistoryEntry): Promise<void> {
  await tx("readwrite", (store) => store.put(entry) as IDBRequest<IDBValidKey>);
  await enforceCap();
}

async function all(): Promise<HistoryEntry[]> {
  return tx<HistoryEntry[]>("readonly", (store) => store.getAll() as IDBRequest<HistoryEntry[]>).then((rows) => {
    rows.sort((a, b) => b.created_at - a.created_at);
    return rows;
  });
}

export async function listEntries(): Promise<HistorySummary[]> {
  const rows = await all();
  return rows.map(({ result: _result, ...summary }) => summary);
}

export async function getEntry(jobId: string): Promise<HistoryEntry | null> {
  return tx<HistoryEntry | undefined>("readonly", (store) => store.get(jobId) as IDBRequest<HistoryEntry | undefined>)
    .then((row) => row ?? null);
}

export async function deleteEntry(jobId: string): Promise<void> {
  await tx("readwrite", (store) => store.delete(jobId) as unknown as IDBRequest<undefined>);
}

export async function clearEntries(): Promise<void> {
  await tx("readwrite", (store) => store.clear() as unknown as IDBRequest<undefined>);
}

export async function countEntries(): Promise<number> {
  return tx<number>("readonly", (store) => store.count() as IDBRequest<number>);
}

/**
 * Drop the oldest rows past `HISTORY_CAP`.
 *
 * Deliberately *not* a `deleteRange` on an index cursor: the row count is small, the order is
 * already in the data, and a range delete on a keyPath-ordered store would evict by job id — which
 * is time-sorted by accident, not by contract.
 */
async function enforceCap(): Promise<void> {
  const rows = await all();
  if (rows.length <= HISTORY_CAP) return;
  for (const row of rows.slice(HISTORY_CAP)) {
    await tx("readwrite", (store) => store.delete(row.job_id) as unknown as IDBRequest<undefined>);
  }
}

const THUMB_PX = 128;

/**
 * Downscale in a canvas and return a JPEG data URL.
 *
 * 128 px is the size the table renders. Anything larger is bytes kept for a picture the user can
 * re-open from their own disk, and JPEG at 0.72 because the thumbnail is *inside* a browser tab
 * next to a privacy claim.
 */
export async function makeThumbnail(file: File | Blob): Promise<string | null> {
  if (typeof document === "undefined") return null;
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    const scale = Math.min(1, THUMB_PX / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.72);
  } catch {
    return null;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("decode failed"));
    image.src = src;
  });
}

/** The pieces a row needs, assembled from the envelope rather than re-derived from the numbers. */
export function buildEntry(
  jobId: string,
  result: AnalysisResult,
  created: number,
  thumbnail: string | null,
): HistoryEntry {
  const image = result.image ?? {};
  return {
    job_id: jobId,
    created_at: created,
    sha256: result.sha256,
    verdict: result.verdict,
    overall: result.overall,
    ai_score: result.ai_score,
    edited_score: result.edited_score,
    reliability: result.reliability,
    headline: result.headline,
    mime: typeof image.mime === "string" ? image.mime : "image/*",
    size_bytes: typeof image.size_bytes === "number" ? image.size_bytes : 0,
    width: typeof image.width === "number" ? image.width : null,
    height: typeof image.height === "number" ? image.height : null,
    from_cache: result.from_cache,
    thumbnail,
    result,
  };
}
