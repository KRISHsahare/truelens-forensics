"use client";

/**
 * The job lifecycle: submit → poll → settle, and nothing else.
 *
 * Two details are load-bearing:
 *
 * - **Cadence comes from the server.** `POST /api/analyze` answers with `poll_after_ms`, which is
 *   how long the engine thinks the first poll is worth waiting for; the poll itself answers with
 *   how much progress it made. A fixed client interval either fires while the pool is still
 *   decoding or lags a sub-second job by a second.
 * - **One generation per submission.** `generation` is bumped on every `run`/`reset`, and every
 *   timer and `await` continuation checks it before writing. Without that, a superseded poll loop
 *   keeps landing old job stages on top of a new analysis and the stepper shows two runs at once.
 *
 * `expired` ends the loop like any other terminal state. The job record is gone; polling more is a
 * way of producing 404s, not of recovering the result.
 */

import * as React from "react";
import { analyzeFile, analyzeUrl, pollJob } from "@/lib/api";
import {
  ApiError,
  type AnalysisResult,
  type JobState,
  type JobStatus,
  type StageState,
} from "@/lib/types";

export type Phase = "idle" | "submitting" | "running" | "succeeded" | "failed";

export interface AnalysisState {
  phase: Phase;
  jobId: string | null;
  jobStatus: JobState | null;
  /** 0..1 from the server: the fraction of technique stages that reached a terminal state. */
  progress: number;
  stages: Record<string, StageState>;
  stageMs: Record<string, number>;
  result: AnalysisResult | null;
  /** Wall-clock from submit to settle, so the UI can say "took 3.4 s" without trusting `timings`. */
  elapsedMs: number;
  polls: number;
  fromCache: boolean;
  error: { code: string; message: string; retryAfterSeconds: number | null } | null;
}

export interface AnalysisFailure {
  code: string;
  message: string;
  retryAfterSeconds: number | null;
}

const IDLE: AnalysisState = {
  phase: "idle",
  jobId: null,
  jobStatus: null,
  progress: 0,
  stages: {},
  stageMs: {},
  result: null,
  elapsedMs: 0,
  polls: 0,
  fromCache: false,
  error: null,
};

const TERMINAL: readonly JobState[] = ["succeeded", "failed", "expired"];

/** A poll can fail for a reason that fixes itself (a proxy hiccup, a 502). Retry a few times first. */
const MAX_CONSECUTIVE_POLL_FAILURES = 4;
const FIRST_POLL_MS = 300;
const RETRY_POLL_MS = 1000;

export type AnalysisSource = { file: File | Blob; filename?: string } | { url: string };

export interface AnalysisHandle extends AnalysisState {
  run: (source: AnalysisSource) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(onSettled?: (state: AnalysisState) => void): AnalysisHandle {
  const [state, setState] = React.useState<AnalysisState>(IDLE);
  const settled = React.useRef(onSettled);
  settled.current = onSettled;

  const generation = React.useRef(0);
  const controller = React.useRef<AbortController | null>(null);
  const timer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  const stop = React.useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  }, []);

  React.useEffect(() => stop, [stop]);

  const run = React.useCallback(
    async (source: AnalysisSource) => {
      stop();
      const mine = ++generation.current;
      const startedAt = performance.now();
      controller.current = new AbortController();
      const signal = controller.current.signal;

      const alive = () => generation.current === mine && !signal.aborted;
      const fail = (failure: AnalysisFailure) => {
        if (!alive()) return;
        const next: AnalysisState = {
          ...IDLE,
          phase: "failed",
          elapsedMs: performance.now() - startedAt,
          jobStatus: "failed",
          error: failure,
        };
        setState(next);
        settled.current?.(next);
      };
      const schedule = (jobId: string, waitMs: number) => {
        if (!alive()) return;
        timer.current = setTimeout(() => {
          timer.current = null;
          if (alive()) void poll(jobId);
        }, waitMs);
      };

      // Per-submission, not per-hook: a retried poll must not inherit the previous run's tally,
      // and two handles on one page must not share it.
      let failures = 0;
      let polls = 0;

      setState({ ...IDLE, phase: "submitting" });

      let created;
      try {
        created =
          "url" in source
            ? await analyzeUrl(source.url, signal)
            : await analyzeFile(source.file, source.filename ?? "image", signal);
      } catch (error) {
        const api = error instanceof ApiError ? error : null;
        fail({
          code: api?.code ?? "network",
          message:
            api?.detail ??
            api?.message ??
            // A fetch that the browser blocked for CORS and a server that is down are
            // indistinguishable here -- both are a rejected promise with no status -- and the two
            // need opposite fixes. Saying only "check the backend is running" sent an operator to
            // a healthy service while the real problem was its allowlist.
            "The analysis engine did not answer. Either it is not running, or this page's address " +
            "is not in its CORS allowlist: a blocked response looks exactly like a dead server " +
            "from the browser. Check GET /api/health on the engine and read its `network` field.",
          retryAfterSeconds: api?.retryAfterSeconds ?? null,
        });
        return;
      }
      if (!alive()) return;

      const jobId = created.job_id;
      setState({
        ...IDLE,
        phase: "running",
        jobId,
        jobStatus: created.status,
        stages: created.stages,
        fromCache: created.cached,
      });
      schedule(jobId, Math.max(0, created.poll_after_ms ?? FIRST_POLL_MS));

      async function poll(id: string) {
        let status: JobStatus;
        try {
          status = await pollJob(id, signal);
        } catch (error) {
          if (!alive()) return;
          const api = error instanceof ApiError ? error : null;
          // A 404 here means the record aged out mid-poll; that is terminal, not retryable.
          if (api?.status === 404) {
            fail({
              code: "expired",
              message: "This job aged out of the engine. Run the analysis again.",
              retryAfterSeconds: null,
            });
            return;
          }
          failures += 1;
          if (failures >= MAX_CONSECUTIVE_POLL_FAILURES) {
            fail({
              code: api?.code ?? "network",
              message: api?.message ?? "The engine stopped answering for this job.",
              retryAfterSeconds: api?.retryAfterSeconds ?? null,
            });
            return;
          }
          schedule(id, RETRY_POLL_MS);
          return;
        }
        if (!alive()) return;
        failures = 0;
        polls += 1;

        const finished = TERMINAL.includes(status.status);
        const next: AnalysisState = {
          phase: finished ? (status.status === "succeeded" ? "succeeded" : "failed") : "running",
          jobId: id,
          jobStatus: status.status,
          progress: status.progress ?? 0,
          stages: status.stages ?? {},
          stageMs: status.stage_ms ?? {},
          result: status.result ?? null,
          elapsedMs: performance.now() - startedAt,
          polls,
          fromCache: status.from_cache ?? false,
          error: errorOf(status),
        };
        setState(next);
        if (finished) {
          settled.current?.(next);
          return;
        }
        schedule(id, FIRST_POLL_MS);
      }
    },
    [stop],
  );

  const reset = React.useCallback(() => {
    stop();
    setState(IDLE);
  }, [stop]);

  return { ...state, run, reset };
}

/**
 * The poll only carries `error` when the job failed, and `exclude_none` means the key is absent
 * otherwise — so "no error field" and "error: null" are the same answer, and neither is a failure.
 */
function errorOf(status: JobStatus): AnalysisFailure | null {
  if (status.error) {
    return { code: status.error.code, message: status.error.message, retryAfterSeconds: null };
  }
  if (status.status === "expired") {
    return {
      code: "expired",
      message: "This job aged out of the engine. Run the analysis again.",
      retryAfterSeconds: null,
    };
  }
  return null;
}
