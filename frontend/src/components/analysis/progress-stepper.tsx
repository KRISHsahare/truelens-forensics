"use client";

/**
 * The live progress stepper.
 *
 * One row per technique, driven by `stages` from the poll — the same dict the server's `JobRecord`
 * keeps and the same terminal-sticky rule, so a row can never move backwards on screen even if two
 * pool callbacks arrive out of order.
 *
 * Rows are grouped by sub-score rather than listed alphabetically, because while an analysis is
 * running the question is "which half of the answer am I waiting for".
 */

import * as React from "react";
import { AlertTriangle, Check, Clock, Loader2, MinusCircle, SkipForward } from "lucide-react";
import { cn } from "@/lib/utils";
import { GROUP_LABELS, STAGE_LABELS, ms } from "@/lib/format";
import type { CatalogEntry } from "@/lib/catalog";
import type { StageState } from "@/lib/types";

const ICONS: Record<StageState, React.ReactNode> = {
  pending: <Clock className="size-3.5" />,
  running: <Loader2 className="size-3.5 animate-spin" />,
  done: <Check className="size-3.5" />,
  failed: <AlertTriangle className="size-3.5" />,
  skipped: <SkipForward className="size-3.5" />,
  blocked: <MinusCircle className="size-3.5" />,
};

const TONE: Record<StageState, string> = {
  pending: "text-muted-foreground/60",
  running: "text-primary",
  done: "text-risk-low",
  failed: "text-risk-high",
  skipped: "text-muted-foreground",
  blocked: "text-muted-foreground",
};

const GROUP_ORDER: { key: string; match: (e: CatalogEntry) => boolean }[] = [
  { key: "ai", match: (e) => e.group === "ai" },
  { key: "edited", match: (e) => e.group === "edited" },
  { key: "context", match: (e) => e.group === "context" },
];

export interface ProgressStepperProps {
  entries: CatalogEntry[];
  stages: Record<string, StageState>;
  stageMs?: Record<string, number>;
  progress: number;
  /** Terminal rows still render; the header copy changes to say the run is over. */
  finished?: boolean;
  className?: string;
}

export function ProgressStepper({
  entries,
  stages,
  stageMs,
  progress,
  finished = false,
  className,
}: ProgressStepperProps) {
  const done = Object.values(stages).filter(
    (s) => s === "done" || s === "failed" || s === "skipped" || s === "blocked",
  ).length;

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {finished ? "Checks completed" : "Running checks"}
        </p>
        <p className="tabular text-xs text-muted-foreground">
          {done}/{Object.keys(stages).length || entries.length}
        </p>
      </div>

      <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-500 ease-out"
          style={{ width: `${Math.round(Math.min(1, Math.max(0, progress)) * 100)}%` }}
        />
      </div>

      <ul className="flex flex-col gap-3">
        {GROUP_ORDER.map(({ key, match }) => {
          const rows = entries.filter(match);
          if (!rows.length) return null;
          return (
            <li key={key} className="flex flex-col gap-1">
              <p className="text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground/80">
                {GROUP_LABELS[key] ?? key}
              </p>
              <ul className="flex flex-col">
                {rows.map((entry) => {
                  const stage = stages[entry.name] ?? "pending";
                  const elapsed = stageMs?.[entry.name];
                  return (
                    <li
                      key={entry.name}
                      className={cn(
                        "flex items-center gap-2 rounded-md px-1.5 py-1 text-xs transition-colors",
                        stage === "pending" && "opacity-55",
                      )}
                    >
                      <span className={cn("shrink-0", TONE[stage])} aria-hidden>
                        {ICONS[stage]}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-foreground/90">
                        {entry.label}
                      </span>
                      <span className={cn("shrink-0 tabular text-[0.65rem]", TONE[stage])}>
                        {elapsed != null && stage === "done"
                          ? ms(elapsed)
                          : STAGE_LABELS[stage] ?? stage}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </li>
          );
        })}
      </ul>

      {/* One polite announcement per run, not per row: a live region on every row makes a screen
          reader recite the whole list as it ticks over. */}
      <p className="sr-only" role="status" aria-live="polite">
        {finished
          ? "All checks finished."
          : `${done} of ${Object.keys(stages).length || entries.length} checks complete.`}
      </p>
    </div>
  );
}
