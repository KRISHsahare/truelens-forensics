"use client";

/**
 * Batch mode.
 *
 * This page is the same engine, twenty times, with the per-image ceremony folded into a table. It
 * exists because the question a batch asks is a *triage* question — "which of these 14 press photos
 * should I look at properly?" — and the single analyser answers that badly: it forces a reviewer to
 * open, read, and close twenty reports to find the two with a red bar.
 *
 * What it deliberately does not do:
 *
 * - **It does not average the images.** A mean over twenty unrelated files has no interpretation, and
 *   a "batch risk score" would be the least defensible number on the site.
 * - **It does not re-explain.** Each row links to its own report, which is where the overlays,
 *   technique cards and limitations live. A table cell is a sorting key, not an explanation.
 * - **It keeps nothing after the tab closes**, except the same opt-in browser History every other
 *   analysis writes.
 */

import * as React from "react";
import Link from "next/link";
import { Download, Loader2, RotateCcw, Scan, LayoutGrid } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BatchIntake } from "@/components/batch/intake";
import { ResultsTable, type SortKey, type SortState } from "@/components/batch/results-table";
import { useBatch, type BatchRow } from "@/hooks/use-batch";
import { downloadText, timestampSlug, toCsv } from "@/lib/csv";
import { buildEntry, makeThumbnail, putEntry } from "@/lib/history";
import { thumbnailsEnabled } from "@/lib/settings";
import { VERDICT_LABELS } from "@/lib/format";

const HEADERS = [
  "file",
  "job_id",
  "sha256",
  "status",
  "verdict",
  "overall_risk",
  "ai_generated_risk",
  "edited_risk",
  "reliability",
  "headline",
  "error",
] as const;

export default function BatchPage() {
  const [files, setFiles] = React.useState<File[]>([]);
  const [sort, setSort] = React.useState<SortState>({ key: "overall", descending: true });
  const batch = useBatch();
  const savedRows = React.useRef<Set<string>>(new Set());

  const busy = batch.phase === "submitting" || batch.phase === "running";

  const toggleSort = React.useCallback((key: SortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, descending: !current.descending }
        : { key, descending: key !== "label" },
    );
  }, []);

  /**
   * Write finished rows to the browser history, once each.
   *
   * Thumbnails are generated here rather than in the hook: the hook holds the result, and only the
   * page still holds the `File` the row came from.
   */
  React.useEffect(() => {
    if (batch.phase === "idle") return;
    let cancelled = false;
    (async () => {
      for (const row of batch.rows) {
        if (!row.jobId || !row.result || savedRows.current.has(row.key)) continue;
        savedRows.current.add(row.key);
        const source = files[row.index];
        const thumb = thumbnailsEnabled() && source ? await makeThumbnail(source) : null;
        if (cancelled) return;
        try {
          await putEntry(buildEntry(row.jobId, row.result, Date.now() / 1000, thumb));
        } catch {
          // Private mode or blocked storage: the table still works, only the history page is empty.
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [batch.rows, batch.phase, files]);

  const exportCsv = () => {
    const rows = batch.rows.map((row) => [
      row.label,
      row.jobId || "",
      row.sha256 || "",
      row.status,
      row.result ? VERDICT_LABELS[row.result.verdict] : "",
      // Four decimals, not a rounded percentage: a spreadsheet is where the comparison happens, and
      // 71% versus 71% is the difference between two rows that were never the same number.
      round(row.result?.overall),
      round(row.result?.ai_score),
      round(row.result?.edited_score),
      round(row.result?.reliability),
      row.result?.headline ?? "",
      row.error ?? "",
    ]);
    downloadText(`truelens-batch-${timestampSlug()}.csv`, toCsv(HEADERS, rows));
  };

  const counts = React.useMemo(() => tally(batch.rows), [batch.rows]);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-2">
        <span className="inline-flex items-center gap-2 text-[0.7rem] uppercase tracking-[0.18em] text-primary">
          <LayoutGrid className="size-3.5" /> Batch
        </span>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Screen a set, then open what matters
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Up to {batch.summary?.limit ?? 20} images at once, each analysed as its own job by the same
          engine as the single analyser. Sort the table by the number you care about and open a row
          for the full evidence. Every likelihood is a probability, not proof, and the rows with the
          lowest reliability are usually the ones worth reading in full.
        </p>
      </header>

      {batch.phase === "idle" || batch.phase === "failed" ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Scan className="size-4 text-primary" /> Images to analyse
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {batch.error && (
              <p className="rounded-md border border-risk-high/40 bg-risk-high/10 px-3 py-2 text-xs text-risk-high">
                {batch.error}
              </p>
            )}
            <BatchIntake
              files={files}
              busy={busy}
              onAdd={(incoming) => setFiles((current) => [...current, ...incoming])}
              onRemove={(index) => setFiles((current) => current.filter((_, i) => i !== index))}
              onClear={() => setFiles([])}
              onSubmit={() => batch.submit(files)}
            />
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-sm">Batch results</CardTitle>
              {busy && (
                <Badge variant="muted">
                  <Loader2 className="size-3 animate-spin" /> {batch.finished}/{batch.total} finished
                </Badge>
              )}
              {batch.summary && batch.summary.rejected > 0 && (
                <Badge variant="destructive">{batch.summary.rejected} refused at upload</Badge>
              )}
              <Badge variant="outline">{counts.succeeded} scored</Badge>
              {counts.high > 0 && <Badge variant="destructive">{counts.high} high risk</Badge>}
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={exportCsv}
                disabled={batch.rows.length === 0}
              >
                <Download /> Export CSV
              </Button>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() => {
                  batch.reset();
                  setFiles([]);
                  savedRows.current = new Set();
                }}
              >
                <RotateCcw /> New batch
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {batch.rows.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                Nothing came back from the engine.
              </p>
            ) : (
              <>
                <ResultsTable rows={batch.rows} sort={sort} onSort={toggleSort} />
                <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
                  Blank score cells mean the check did not produce a measurement — not that it passed.
                  The CSV keeps them blank for the same reason. Rows are also written to this
                  browser&apos;s history;{" "}
                  <Link href="/history" className="underline underline-offset-2 hover:text-foreground">
                    manage it here
                  </Link>
                  .
                </p>
              </>
            )}
          </CardContent>
        </Card>
      )}

      <section className="grid gap-3 sm:grid-cols-3">
        {NOTES.map((note) => (
          <Card key={note.title} className="bg-card/50">
            <CardContent className="flex flex-col gap-1.5 py-4">
              <p className="text-xs font-semibold">{note.title}</p>
              <p className="text-[0.72rem] leading-relaxed text-muted-foreground">{note.body}</p>
            </CardContent>
          </Card>
        ))}
      </section>
    </div>
  );
}

const NOTES = [
  {
    title: "One queue, twenty jobs",
    body: "The engine runs a few analyses at a time, so rows finish out of order. That is why the table shows status as well as score.",
  },
  {
    title: "Heatmaps expire",
    body: "Overlay links are signed and die with the job, at most ten minutes after upload. A row's numbers stay on screen; a report opened later may have no pictures.",
  },
  {
    title: "Sort by what you will act on",
    body: "Overall risk is the fused likelihood. Reliability says how much of the evidence was actually measurable — low reliability with a high score is a warning about the file, not the image.",
  },
];

function tally(rows: BatchRow[]): { succeeded: number; high: number } {
  let succeeded = 0;
  let high = 0;
  for (const row of rows) {
    if (!row.result) continue;
    succeeded += 1;
    if (row.result.band === "high") high += 1;
  }
  return { succeeded, high };
}

function round(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "";
  return value.toFixed(4);
}
