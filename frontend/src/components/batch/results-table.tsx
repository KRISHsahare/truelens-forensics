"use client";

import * as React from "react";
import { ArrowDown, ArrowUp, FileText, Loader2 } from "lucide-react";
import { BandBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TBody, Td, Th, THead, Tr } from "@/components/ui/table";
import { BAND_CLASSES, pct, shortSha, VERDICT_LABELS, bandFor } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { BatchRow } from "@/hooks/use-batch";
import type { JobState } from "@/lib/types";

export type SortKey =
  | "label"
  | "status"
  | "verdict"
  | "overall"
  | "ai_score"
  | "edited_score"
  | "reliability";

export interface SortState {
  key: SortKey;
  descending: boolean;
}

const STATUS_ORDER: Record<JobState, number> = {
  queued: 0,
  running: 1,
  succeeded: 2,
  failed: 3,
  expired: 4,
};

const STATUS_COPY: Record<JobState, string> = {
  queued: "Queued",
  running: "Running",
  succeeded: "Done",
  failed: "Failed",
  expired: "Aged out",
};

/**
 * Sort the batch rows.
 *
 * Unmeasured values are pinned to the bottom in *both* directions. A `null` score that sorts as
 * either the highest or the lowest number depending on which arrow the user last clicked turns "no
 * measurement" into a ranking, and the ranking is the thing the table is being read for.
 */
export function sortRows(rows: BatchRow[], sort: SortState): BatchRow[] {
  const value = (row: BatchRow): number | string | null => {
    switch (sort.key) {
      case "label":
        return row.label.toLowerCase();
      case "status":
        return STATUS_ORDER[row.status] ?? 0;
      case "verdict":
        return row.result?.verdict ?? null;
      case "overall":
        return row.result?.overall ?? null;
      case "ai_score":
        return row.result?.ai_score ?? null;
      case "edited_score":
        return row.result?.edited_score ?? null;
      case "reliability":
        return row.result?.reliability ?? null;
    }
  };
  const sign = sort.descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    const left = value(a);
    const right = value(b);
    if (left == null && right == null) return a.key.localeCompare(b.key);
    if (left == null) return 1;
    if (right == null) return -1;
    if (typeof left === "string" || typeof right === "string") {
      return sign * String(left).localeCompare(String(right));
    }
    return sign * (left - right) || a.key.localeCompare(b.key);
  });
}

export interface ResultsTableProps {
  rows: BatchRow[];
  sort: SortState;
  onSort: (key: SortKey) => void;
}

export function ResultsTable({ rows, sort, onSort }: ResultsTableProps) {
  const ordered = React.useMemo(() => sortRows(rows, sort), [rows, sort]);

  return (
    <div className="max-h-[70vh] overflow-auto rounded-lg border border-border">
      <Table>
        <THead>
          <Tr>
            <ColumnHead label="File" sortKey="label" sort={sort} onSort={onSort} />
            <Th className="text-[0.68rem] uppercase tracking-wide">Hash</Th>
            <ColumnHead label="Status" sortKey="status" sort={sort} onSort={onSort} />
            <ColumnHead label="Verdict" sortKey="verdict" sort={sort} onSort={onSort} />
            <ColumnHead label="Overall" sortKey="overall" sort={sort} onSort={onSort} numeric />
            <ColumnHead label="Synthetic" sortKey="ai_score" sort={sort} onSort={onSort} numeric />
            <ColumnHead
              label="Manipulation"
              sortKey="edited_score"
              sort={sort}
              onSort={onSort}
              numeric
            />
            <ColumnHead
              label="Reliability"
              sortKey="reliability"
              sort={sort}
              onSort={onSort}
              numeric
            />
            <Th className="w-10" />
          </Tr>
        </THead>
        <TBody>
          {ordered.map((row) => (
            <BatchRowView key={row.key} row={row} />
          ))}
        </TBody>
      </Table>
    </div>
  );
}

function ColumnHead({
  label,
  sortKey,
  sort,
  onSort,
  numeric = false,
}: {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (key: SortKey) => void;
  numeric?: boolean;
}) {
  const active = sort.key === sortKey;
  return (
    <Th
      aria-sort={active ? (sort.descending ? "descending" : "ascending") : "none"}
      className={cn(numeric && "text-right")}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={cn(
          "inline-flex items-center gap-1 text-[0.68rem] uppercase tracking-wide transition-colors",
          active ? "text-foreground" : "text-muted-foreground hover:text-foreground",
        )}
      >
        {label}
        {active ? (
          sort.descending ? (
            <ArrowDown className="size-3" />
          ) : (
            <ArrowUp className="size-3" />
          )
        ) : (
          <ArrowUp className="size-3 opacity-0" aria-hidden />
        )}
      </button>
    </Th>
  );
}

function BatchRowView({ row }: { row: BatchRow }) {
  const result = row.result;
  const running = row.status === "running" || row.status === "queued";

  return (
    <Tr>
      <Td className="max-w-[14rem]">
        <span className="block truncate text-xs font-medium" title={row.label}>
          {row.label}
        </span>
        {row.error && !result && (
          <span className="mt-0.5 block text-[0.68rem] leading-snug text-risk-high">{row.error}</span>
        )}
      </Td>

      <Td className="tabular text-[0.68rem] text-muted-foreground">
        {row.sha256 ? shortSha(row.sha256, 8, 4) : "—"}
      </Td>

      <Td>
        <span className="flex items-center gap-1.5 text-xs">
          {running && <Loader2 className="size-3 animate-spin text-primary" />}
          {STATUS_COPY[row.status] ?? row.status}
        </span>
        {row.status === "running" && (
          <span className="mt-1 block h-1 w-24 overflow-hidden rounded-full bg-muted">
            <span
              className="block h-full rounded-full bg-primary transition-[width] duration-500"
              style={{ width: `${Math.round(row.progress * 100)}%` }}
            />
          </span>
        )}
      </Td>

      <Td>
        {result ? (
          <BandBadge band={bandFor(result.overall)}>
            {VERDICT_LABELS[result.verdict]}
          </BandBadge>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
      </Td>

      <ScoreCell value={result?.overall ?? null} emphasise />
      <ScoreCell value={result?.ai_score ?? null} />
      <ScoreCell value={result?.edited_score ?? null} />
      <ScoreCell value={result?.reliability ?? null} reliability />

      <Td>
        {result ? (
          <Button
            asChild
            size="icon"
            variant="ghost"
            className="size-8"
            aria-label={`Open the report for ${row.label}`}
          >
            <a href={`/report/${row.jobId}`}>
              <FileText className="size-4" />
            </a>
          </Button>
        ) : null}
      </Td>
    </Tr>
  );
}

/** A null score is a dash in muted graphite, never a zero — same rule as the on-screen bar. */
function ScoreCell({
  value,
  emphasise = false,
  reliability = false,
}: {
  value: number | null | undefined;
  emphasise?: boolean;
  reliability?: boolean;
}) {
  const measured = value != null && !Number.isNaN(value);
  const band = bandFor(value);
  return (
    <Td className="text-right">
      {measured ? (
        <span
          className={cn(
            "tabular inline-flex items-center justify-end gap-1.5",
            emphasise && "font-semibold",
            reliability ? "text-muted-foreground" : BAND_CLASSES[band].text,
          )}
        >
          <span
            aria-hidden
            className={cn("size-1.5 rounded-full", BAND_CLASSES[reliability ? "not_measured" : band].bar)}
          />
          {pct(value)}
        </span>
      ) : (
        <span className="tabular text-muted-foreground">—</span>
      )}
    </Td>
  );
}
