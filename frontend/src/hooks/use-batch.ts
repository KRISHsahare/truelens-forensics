/**
 * Batch polling.
 *
 * The server hands back *job ids*, not results (`POST /api/batch`), so a batch is twenty independent
 * single-image jobs and this hook is twenty pollers. That has one real constraint: the engine runs
 * `ANALYZER_PARALLEL` analyses at a time, so a client that polls all twenty rows the instant it
 * submits spends its entire budget on rows that are still `queued` and starves the two rows that are
 * actually producing progress. So the scheduler is due-based rather than round-robin:
 *
 * - Each row carries a `nextPollAt` timestamp. A tick only fires polls that are due.
 * - At most `CONCURRENT_POLLS` requests are in flight, so the browser's six-connections-per-host
 *   limit is never the thing that stalls a heatmap fetch.
 * - A row that is not terminal is re-queued at `POLL_MS`; a terminal row is never polled again.
 *
 * Rows rejected at intake (a truncated file in a bundle of twenty) have an empty `job_id` and are
 * never polled — they carry the server's per-item sentence in `error` and stay in the table so the
 * count the user handed over matches the count they see.
 */

import * as React from "react";
import { batchAnalyze, checkFile, pollJob, BATCH_MAX_IMAGES } from "@/lib/api";
import type { AnalysisResult, BatchItem, JobState } from "@/lib/types";
import { ApiError } from "@/lib/types";

export type BatchPhase = "idle" | "submitting" | "running" | "done" | "failed";

export interface BatchRow {
  key: string;
  /** Position in the submitted list — the page uses it to find the original `File` for a thumbnail. */
  index: number;
  label: string;
  jobId: string;
  sha256: string;
  /** `queued` for a row that was accepted but has not been polled yet. */
  status: JobState;
  progress: number;
  result: AnalysisResult | null;
  error: string | null;
  polls: number;
}

export interface BatchSummary {
  accepted: number;
  rejected: number;
  limit: number;
}

const TERMINAL: JobState[] = ["succeeded", "failed", "expired"];
const POLL_MS = 1500;
const FIRST_POLL_MS = 400;
const TICK_MS = 400;
const CONCURRENT_POLLS = 4;
const MAX_POLL_FAILURES = 4;

function isTerminal(status: JobState): boolean {
  return TERMINAL.includes(status);
}

/**
 * A row is done only when its result is in hand — a terminal *status* is not enough.
 *
 * `/api/batch` answers with the per-item status it already knows, and for an image the result cache
 * has seen before that status is `succeeded` on the spot. Polling on status alone therefore skipped
 * cache-hit rows entirely: the table read "succeeded" with no scores, no report link and a CSV of
 * blanks, because the poll is the only place a result is ever read. The `polls > 0` half keeps a
 * genuinely failed or expired row from being re-polled forever when no result will ever arrive.
 */
function isSettled(row: BatchRow): boolean {
  return isTerminal(row.status) && (row.result !== null || row.polls > 0);
}

interface Schedule {
  inFlight: boolean;
  nextPollAt: number;
  failures: number;
}

export interface UseBatch {
  phase: BatchPhase;
  rows: BatchRow[];
  summary: BatchSummary | null;
  error: string | null;
  finished: number;
  total: number;
  submit: (files: File[]) => void;
  reset: () => void;
}

export function useBatch(): UseBatch {
  const [phase, setPhase] = React.useState<BatchPhase>("idle");
  const [rows, setRows] = React.useState<BatchRow[]>([]);
  const [summary, setSummary] = React.useState<BatchSummary | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const rowsRef = React.useRef<BatchRow[]>([]);
  const schedule = React.useRef<Map<string, Schedule>>(new Map());
  const timer = React.useRef<number | null>(null);
  const controller = React.useRef<AbortController | null>(null);
  const mounted = React.useRef(true);

  React.useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stopTimer(timer);
      controller.current?.abort();
    };
  }, []);

  const commit = React.useCallback((next: BatchRow[]) => {
    rowsRef.current = next;
    setRows(next);
  }, []);

  const patch = React.useCallback(
    (jobId: string, changes: Partial<BatchRow>) => {
      commit(
        rowsRef.current.map((row) => (row.jobId === jobId ? { ...row, ...changes } : row)),
      );
    },
    [commit],
  );

  const stop = React.useCallback(() => {
    stopTimer(timer);
    controller.current?.abort();
    controller.current = null;
  }, []);

  const poll = React.useCallback(
    async (jobId: string) => {
      const meta = schedule.current.get(jobId);
      if (!meta) return;
      meta.inFlight = true;
      const signal = (controller.current ??= new AbortController()).signal;
      try {
        const status = await pollJob(jobId, signal);
        if (!mounted.current) return;
        meta.failures = 0;
        meta.nextPollAt = performance.now() + POLL_MS;
        patch(jobId, {
          status: status.status,
          progress: status.progress ?? 0,
          result: status.result ?? null,
          error: status.error ? status.error.message : null,
          polls: (rowsRef.current.find((r) => r.jobId === jobId)?.polls ?? 0) + 1,
        });
      } catch (exc) {
        if (!mounted.current || signal.aborted) return;
        meta.nextPollAt = performance.now() + POLL_MS * 2;
        meta.failures += 1;
        // A 404 is the job being swept, not the network wobbling: retrying it ten times just delays
        // the moment the table admits the result is gone.
        const gone = exc instanceof ApiError && exc.status === 404;
        if (gone || meta.failures >= MAX_POLL_FAILURES) {
          patch(jobId, {
            status: "expired",
            error: gone
              ? "This job aged out of the engine. Run the analysis again."
              : "The engine stopped answering for this row. Re-run the batch.",
          });
        }
      } finally {
        meta.inFlight = false;
      }
    },
    [patch],
  );

  /** One tick: poll every due row, up to the in-flight cap, then decide whether there is work left. */
  const tick = React.useCallback(() => {
    const now = performance.now();
    const current = rowsRef.current;

    let scheduled = 0;
    for (const row of current) {
      if (!row.jobId || isSettled(row)) continue;
      const meta = schedule.current.get(row.jobId);
      if (!meta || meta.inFlight || meta.nextPollAt > now) continue;
      if (scheduled >= CONCURRENT_POLLS) break;
      scheduled += 1;
      void poll(row.jobId);
    }

    const pending = current.filter((row) => row.jobId && !isSettled(row)).length;
    if (pending === 0 && scheduled === 0) {
      stopTimer(timer);
      if (mounted.current) setPhase("done");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = React.useCallback(
    (files: File[]) => {
      const usable = files.slice(0, BATCH_MAX_IMAGES);
      if (!usable.length) {
        setError("Choose at least one image.");
        return;
      }
      const oversize = usable.find((file) => !checkFile(file, file.name).ok);
      if (oversize) {
        setError(checkFile(oversize, oversize.name).reason ?? "That file cannot be analysed.");
        return;
      }

      stop();
      setError(null);
      setSummary(null);
      schedule.current = new Map();
      setPhase("submitting");
      controller.current = new AbortController();

      batchAnalyze(
        usable,
        usable.map((file, i) => file.name || `image-${i + 1}`),
        controller.current.signal,
      )
        .then((result) => {
          if (!mounted.current) return;
          setSummary({ accepted: result.accepted, rejected: result.rejected, limit: result.limit });
          const initial: BatchRow[] = result.items.map((item, index) =>
            rowFromItem(item, index),
          );
          initial.forEach((row) => {
            if (!row.jobId) return;
            // Staggered so the first round of polls is spread across the queue rather than landing
            // on the engine as twenty simultaneous requests for rows 1–4.
            schedule.current.set(row.jobId, {
              inFlight: false,
              nextPollAt: performance.now() + FIRST_POLL_MS + row.index * 120,
              failures: 0,
            });
          });
          commit(initial);
          if (!initial.some((row) => row.jobId)) {
            setPhase("done");
            return;
          }
          setPhase("running");
          startTimer(timer, tick, TICK_MS);
        })
        .catch((exc: unknown) => {
          if (!mounted.current) return;
          setPhase("failed");
          setError(
            exc instanceof ApiError
              ? `${exc.message}${exc.detail ? ` ${exc.detail}` : ""}`
              : "The batch could not be submitted.",
          );
        });
    },
    [commit, stop, tick],
  );

  const reset = React.useCallback(() => {
    stop();
    schedule.current = new Map();
    rowsRef.current = [];
    setRows([]);
    setSummary(null);
    setError(null);
    setPhase("idle");
  }, [stop]);

  const finished = rows.filter((row) => !row.jobId || isTerminal(row.status)).length;

  return {
    phase,
    rows,
    summary,
    error,
    finished,
    total: rows.length,
    submit,
    reset,
  };
}

function rowFromItem(item: BatchItem, index: number): BatchRow {
  return {
    key: item.job_id || `rejected-${index}`,
    index,
    label: item.label || `image-${index + 1}`,
    jobId: item.job_id,
    sha256: item.sha256 ?? "",
    status: item.status,
    progress: 0,
    result: null,
    error: item.error ?? null,
    polls: 0,
  };
}

function startTimer(ref: { current: number | null }, fn: () => void, ms: number) {
  stopTimer(ref);
  ref.current = window.setInterval(fn, ms);
}

function stopTimer(ref: { current: number | null }) {
  if (ref.current != null) {
    window.clearInterval(ref.current);
    ref.current = null;
  }
}
