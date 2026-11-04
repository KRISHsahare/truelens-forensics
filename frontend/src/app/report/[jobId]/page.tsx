"use client";

/**
 * The forensic report — the same result, arranged to be forwarded.
 *
 * Two ways to arrive and one of two sources for the numbers:
 *
 * - **Fresh job.** The poll still answers, so the report is read from the engine and the heatmaps are
 *   still fetchable. This is the case the "Download PDF" button is for.
 * - **Expired job, stored history.** The scratch files are gone and the poll 404s, but this browser
 *   kept the envelope. The report then renders from IndexedDB and says which pixels are missing.
 *
 * The PDF is *not* this DOM printed through a library. It is rendered server-side by `reportlab` from
 * the stored result (`backend/app/api/routes_report.py`), so the downloaded document lays out the same
 * way on a reviewer's machine as it does here — a forwarded report whose numbers moved because the
 * recipient's fonts differed is the failure mode this avoids. The browser "Print" path still exists,
 * because some people print what is on screen; `globals.css` carries the print rules for it.
 */

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  FileDown,
  FileWarning,
  Printer,
  ShieldAlert,
} from "lucide-react";
import { Badge, BandBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { VerdictPanel } from "@/components/analysis/verdict-panel";
import { TechniqueCard } from "@/components/analysis/technique-card";
import { MetadataPanel } from "@/components/analysis/metadata-panel";
import { TechnicalPanel } from "@/components/analysis/technical-panel";
import { OverlayViewer, type OverlaySource } from "@/components/analysis/overlay-viewer";
import { API_BASE_URL, pollJob } from "@/lib/api";
import { useCatalog } from "@/lib/catalog";
import { getEntry, type HistoryEntry } from "@/lib/history";
import { shortSha, timeAgo, VERDICT_LABELS } from "@/lib/format";
import { ApiError, type AnalysisResult, type JobStatus } from "@/lib/types";

type Source =
  | { kind: "loading" }
  | { kind: "live"; result: AnalysisResult; status: JobStatus }
  | { kind: "stored"; entry: HistoryEntry }
  | { kind: "gone" };

export default function ReportPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params?.jobId ?? "";
  const [source, setSource] = React.useState<Source>({ kind: "loading" });
  const [active, setActive] = React.useState<string | null>(null);
  const catalog = useCatalog();

  React.useEffect(() => {
    if (!jobId) {
      setSource({ kind: "gone" });
      return;
    }
    let cancelled = false;

    // Try the engine first. A job that has aged out is not an error state, it is the ordinary
    // passage of ten minutes, and the stored copy is a legitimate fallback rather than a consolation.
    pollJob(jobId)
      .then((status) => {
        if (cancelled) return;
        if (status.status === "succeeded" && status.result) {
          setSource({ kind: "live", result: status.result, status });
          return;
        }
        void fromHistory();
      })
      .catch(async (exc: unknown) => {
        if (cancelled) return;
        if (exc instanceof ApiError && exc.status === 404) {
          await fromHistory();
          return;
        }
        await fromHistory();
      });

    async function fromHistory() {
      try {
        const entry = await getEntry(jobId);
        if (!cancelled) setSource(entry ? { kind: "stored", entry } : { kind: "gone" });
      } catch {
        if (!cancelled) setSource({ kind: "gone" });
      }
    }

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (source.kind === "loading") {
    return (
      <div className="mx-auto w-full max-w-4xl px-4 py-16 text-center text-sm text-muted-foreground">
        Reading job {shortSha(jobId, 8, 4)}…
      </div>
    );
  }

  if (source.kind === "gone") {
    return (
      <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-4 py-16 text-center">
        <FileWarning className="mx-auto size-7 text-muted-foreground" />
        <h1 className="text-lg font-semibold">This report is no longer available</h1>
        <p className="text-sm leading-relaxed text-muted-foreground">
          The engine has deleted the job, and no copy of it is stored in this browser. Results and
          their overlays exist for ten minutes after upload — that window is the privacy guarantee,
          not a timeout to be raised.
        </p>
        <div className="flex justify-center gap-2">
          <Button asChild size="sm" variant="outline">
            <Link href="/analyze">Analyse again</Link>
          </Button>
          <Button asChild size="sm" variant="ghost">
            <Link href="/history">Browse history</Link>
          </Button>
        </div>
      </div>
    );
  }

  const result = source.kind === "live" ? source.result : source.entry.result;
  const recorded =
    source.kind === "live"
      ? `analysed ${timeAgo(source.status.finished_at ?? source.status.created_at)}`
      : `recorded ${timeAgo(source.entry.created_at)}`;
  // A stored copy's overlay links signed against files that are already deleted, so the viewer is
  // only offered for a live job. Rendering ten expired frames on a forwarded report is worse than
  // saying once that the pictures went with the upload.
  const overlays: OverlaySource[] =
    source.kind === "live"
      ? Object.entries(result.techniques ?? {})
          .filter(([, view]) => view.overlay_url)
          .map(([name, view]) => ({
            name,
            label: view.label,
            kind: view.overlay_kind,
            url: view.overlay_url as string,
          }))
      : [];

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-5 px-4 py-8 sm:px-6 lg:px-8">
      <div className="flex flex-wrap items-center justify-between gap-2 print:hidden">
        <Button asChild size="sm" variant="ghost" className="-ml-2">
          <Link href="/analyze">
            <ArrowLeft /> New analysis
          </Link>
        </Button>
        <div className="flex items-center gap-2">
          <Button type="button" size="sm" variant="outline" onClick={() => window.print()}>
            <Printer /> Print
          </Button>
          {source.kind === "live" ? (
            <Button asChild size="sm">
              <a href={`${API_BASE_URL}/api/report/${jobId}`} rel="noreferrer">
                <FileDown /> Download PDF
              </a>
            </Button>
          ) : (
            <Badge variant="muted">
              <ShieldAlert /> PDF unavailable — the job expired
            </Badge>
          )}
        </div>
      </div>

      <header className="flex flex-col gap-2 border-b border-border pb-4">
        <p className="text-[0.68rem] uppercase tracking-[0.2em] text-muted-foreground">
          TrueLens · image authenticity report
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <BandBadge band={result.band}>{VERDICT_LABELS[result.verdict]}</BandBadge>
          <span className="tabular text-[0.7rem] text-muted-foreground">
            {recorded} · job {shortSha(jobId, 10, 6)}
            {source.kind === "live" && source.status.from_cache ? " · cached" : ""}
          </span>
        </div>
        <h1 className="max-w-3xl text-xl font-semibold leading-snug tracking-tight sm:text-2xl">
          {result.headline}
        </h1>
        {/* `break-all`, not `truncate`: the digest is the report's identity — the string a reader
            would quote when asking whether this document describes the same file they have. A
            64-hex token has no break opportunity, so on a phone it ran past the card edge and off
            the page, and an ellipsis would have hidden exactly the part that cannot be guessed. */}
        <p className="tabular break-all text-xs text-muted-foreground">
          SHA-256 {result.sha256}
        </p>
      </header>

      {source.kind === "stored" && (
        <Card className="border-dashed print:hidden">
          <CardContent className="flex items-start gap-3 py-3.5">
            <FileWarning className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <p className="text-[0.72rem] leading-relaxed text-muted-foreground">
              Rendered from this browser&apos;s stored copy. The heatmaps expired with the upload, so
              the viewer below is the 128 px thumbnail local storage kept, if any — the scores,
              explanations and metadata are exactly what the engine recorded.
            </p>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,20rem)]">
        <VerdictPanel result={result} />
        <MetadataPanel metadata={result.metadata} c2pa={result.c2pa} />
      </div>

      {overlays.length > 0 && (
        <section className="flex flex-col gap-2">
          <h2 className="text-[0.7rem] uppercase tracking-[0.16em] text-muted-foreground">
            Visual evidence
          </h2>
          <OverlayViewer
            baseUrl={null}
            baseAlt={`Analysed image ${shortSha(result.sha256, 8, 4)}`}
            baseSize={
              result.image?.width && result.image?.height
                ? { width: Number(result.image.width), height: Number(result.image.height) }
                : null
            }
            overlays={overlays}
            catalog={catalog.entries}
            active={active ?? overlays[0]?.name ?? null}
            onSelect={setActive}
          />
        </section>
      )}

      <section className="flex flex-col gap-3">
        <h2 className="text-[0.7rem] uppercase tracking-[0.16em] text-muted-foreground">
          Every check, with its reasoning
        </h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {Object.entries(result.techniques ?? {}).map(([name, view]) => (
            <TechniqueCard key={name} technique={view} hasOverlay={Boolean(view.overlay_url)} showing={false} />
          ))}
        </div>
      </section>

      <TechnicalPanel result={result} />

      <footer className="flex flex-col gap-1 border-t border-border pt-4 text-[0.7rem] leading-relaxed text-muted-foreground">
        <p className="font-medium text-foreground">How to read this document</p>
        <p>
          It states a likelihood produced by automated analysis of pixels, container and metadata. No
          human reviewed the image. Detection methods lag the generators they target, re-encoding
          destroys the evidence they rely on, and a low score is not a certificate of authenticity.
          Use it alongside what you know about where the image came from.
        </p>
      </footer>
    </div>
  );
}
