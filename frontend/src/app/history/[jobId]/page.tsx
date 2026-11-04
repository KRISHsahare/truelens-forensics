"use client";

/**
 * A replayed analysis, from local storage.
 *
 * This page reads IndexedDB and makes no request for the result. It exists so a conclusion reached
 * last month can still be read — the numbers, the reasons, the metadata the file carried — after the
 * scratch files, and therefore the heatmaps, were deleted. That boundary is the whole reason the
 * notice at the top is here rather than a broken image tag.
 *
 * `from_cache` is meaningless in this view and the timings are historical, so the technical panel is
 * shown as it was recorded rather than as anything that describes a fresh run.
 */

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, ImageOff, Trash2, History as HistoryIcon } from "lucide-react";
import { BandBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { VerdictPanel } from "@/components/analysis/verdict-panel";
import { TechniqueCard } from "@/components/analysis/technique-card";
import { MetadataPanel } from "@/components/analysis/metadata-panel";
import { TechnicalPanel } from "@/components/analysis/technical-panel";
import { deleteEntry, getEntry, type HistoryEntry } from "@/lib/history";
import { GROUP_LABELS, timeAgo, VERDICT_LABELS } from "@/lib/format";
import { humanize } from "@/lib/catalog";
import type { TechniqueView } from "@/lib/types";

export default function HistoryDetailPage() {
  const params = useParams<{ jobId: string }>();
  const router = useRouter();
  const jobId = params?.jobId ?? "";
  const [entry, setEntry] = React.useState<HistoryEntry | null | "loading">("loading");

  React.useEffect(() => {
    let cancelled = false;
    getEntry(jobId)
      .then((row) => {
        if (!cancelled) setEntry(row);
      })
      .catch(() => {
        if (!cancelled) setEntry(null);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (entry === "loading") {
    return (
      <div className="mx-auto w-full max-w-5xl px-4 py-16 text-center text-sm text-muted-foreground">
        Reading this browser&apos;s storage…
      </div>
    );
  }

  if (!entry) {
    return (
      <div className="mx-auto flex w-full max-w-2xl flex-col gap-4 px-4 py-16 text-center">
        <ImageOff className="mx-auto size-7 text-muted-foreground" />
        <h1 className="text-lg font-semibold">Not in this browser&apos;s history</h1>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Either it was deleted, this is a different browser or device, or the site data was cleared.
          History is stored per browser and is never fetched from a server, so there is nothing here
          to recover.
        </p>
        <Button asChild size="sm" variant="outline" className="mx-auto">
          <Link href="/history">Back to history</Link>
        </Button>
      </div>
    );
  }

  const result = entry.result;
  const techniques = Object.entries(result.techniques ?? {}) as [string, TechniqueView][];
  const groups = ["ai", "edited", "context"] as const;

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-8 sm:px-6 lg:px-8">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Button asChild size="sm" variant="ghost" className="-ml-2">
          <Link href="/history">
            <ArrowLeft /> History
          </Link>
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={async () => {
            await deleteEntry(entry.job_id);
            window.location.href = "/history";
          }}
        >
          <Trash2 /> Delete this entry
        </Button>
      </div>

      <header className="flex flex-col gap-2">
        <span className="inline-flex items-center gap-2 text-[0.7rem] uppercase tracking-[0.18em] text-primary">
          <HistoryIcon className="size-3.5" /> Stored analysis
        </span>
        <div className="flex flex-wrap items-center gap-3">
          <BandBadge band={result.band}>{VERDICT_LABELS[result.verdict]}</BandBadge>
          <span className="tabular text-xs text-muted-foreground">
            recorded {timeAgo(entry.created_at)} · {entry.mime} ·{" "}
            {entry.width && entry.height ? `${entry.width}×${entry.height}` : "size unknown"}
          </span>
        </div>
        <h1 className="max-w-3xl text-xl font-semibold leading-snug tracking-tight sm:text-2xl">
          {result.headline}
        </h1>
      </header>

      <Card className="border-dashed">
        <CardContent className="flex items-start gap-3 py-3.5">
          <ImageOff className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
          <p className="text-[0.72rem] leading-relaxed text-muted-foreground">
            Replayed from this browser. The heatmaps are gone — overlay links are signed against
            scratch files the server deletes within ten minutes, and nothing in local storage holds
            the pixels. Scores, explanations, metadata and timings are exactly what was recorded when
            this ran; the image itself was never stored and cannot be re-analysed from here. To
            inspect it again, drop the file into the{" "}
            <Link href="/analyze" className="underline underline-offset-2 hover:text-foreground">
              analyser
            </Link>
            .
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
        <VerdictPanel result={result} />
        <MetadataPanel metadata={result.metadata} c2pa={result.c2pa} />
      </div>

      <section className="flex flex-col gap-3">
        {groups.map((group) => {
          const inGroup = techniques.filter(([, view]) => view.group === group);
          if (!inGroup.length) return null;
          return (
            <div key={group} className="flex flex-col gap-2">
              <h2 className="text-[0.7rem] uppercase tracking-[0.16em] text-muted-foreground">
                {GROUP_LABELS[group] ?? humanize(group)}
              </h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {inGroup.map(([name, view]) => (
                  <TechniqueCard
                    key={name}
                    technique={view}
                    hasOverlay={false}
                    showing={false}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </section>

      <TechnicalPanel result={result} />
    </div>
  );
}
