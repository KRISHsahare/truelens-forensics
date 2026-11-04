"use client";

/**
 * The Analyzer.
 *
 * The page is a state machine over `useAnalysis`, and the layout follows the order a reviewer reads:
 * what happened to the file (progress), what it means (verdict), where to look (viewer), why we say
 * so (technique cards), what the file claims about itself (metadata), and how it was computed
 * (technical panel). Nothing that qualifies a number is placed after the section it qualifies.
 *
 * One object URL is held for the lifetime of a result — it is the only copy of the photograph the
 * viewer has, it never leaves the tab, and it is revoked on the next submission or on unmount.
 */

import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FileText,
  History as HistoryIcon,
  Loader2,
  RotateCcw,
  Scan,
  Timer,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dropzone } from "@/components/analysis/dropzone";
import { ProgressStepper } from "@/components/analysis/progress-stepper";
import { OverlayViewer, type OverlaySource } from "@/components/analysis/overlay-viewer";
import { TechniqueCard } from "@/components/analysis/technique-card";
import { MetadataPanel } from "@/components/analysis/metadata-panel";
import { TechnicalPanel } from "@/components/analysis/technical-panel";
import { VerdictPanel } from "@/components/analysis/verdict-panel";
import { useAnalysis, type AnalysisSource, type AnalysisState } from "@/hooks/use-analysis";
import { useCatalog, type CatalogEntry } from "@/lib/catalog";
import { buildEntry, makeThumbnail, putEntry } from "@/lib/history";
import { handoff } from "@/lib/handoff";
import { ms, shortSha } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TechniqueView } from "@/lib/types";

const OVERLAY_TTL_S = 600;

export default function AnalyzePage() {
  const catalog = useCatalog();
  const [base, setBase] = React.useState<{ url: string; name: string } | null>(null);
  const [file, setFile] = React.useState<File | Blob | null>(null);
  const [activeOverlay, setActiveOverlay] = React.useState<string | null>(null);
  const [saved, setSaved] = React.useState<"idle" | "saving" | "saved" | "unavailable">("idle");
  const baseRef = React.useRef<string | null>(null);

  const replaceBase = (url: string | null) => {
    if (baseRef.current) URL.revokeObjectURL(baseRef.current);
    baseRef.current = url;
  };

  React.useEffect(() => () => replaceBase(null), []);

  // A file dropped on the landing page arrives through the hand-off, once, and is cleared by
  // `take()` so a reload of this page does not re-run an analysis the user has already seen.
  React.useEffect(() => {
    const pending = handoff.take();
    if (pending) submit({ file: pending.file, filename: pending.filename });
    // `submit` is stable for the lifetime of a mount and this effect must run exactly once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onSettled = React.useCallback(async (state: AnalysisState) => {
    if (state.phase !== "succeeded" || !state.result || !state.jobId) return;
    setSaved("saving");
    try {
      const thumb = file ? await makeThumbnail(file) : null;
      await putEntry(buildEntry(state.jobId, state.result, Date.now() / 1000, thumb));
      setSaved("saved");
    } catch {
      setSaved("unavailable");
    }
  }, [file]);

  const analysis = useAnalysis(onSettled);

  const submit = React.useCallback(
    (source: AnalysisSource) => {
      setActiveOverlay(null);
      setSaved("idle");
      if ("file" in source) {
        replaceBase(URL.createObjectURL(source.file));
        const url = baseRef.current!;
        setFile(source.file);
        setBase({ url, name: source.filename ?? "image" });
      } else {
        replaceBase(null);
        setFile(null);
        setBase(null);
      }
      void analysis.run(source);
    },
    [analysis],
  );

  const reset = () => {
    analysis.reset();
    replaceBase(null);
    setBase(null);
    setFile(null);
    setActiveOverlay(null);
    setSaved("idle");
  };

  const result = analysis.result;
  const overlays = React.useMemo<OverlaySource[]>(
    () =>
      result
        ? Object.values(result.techniques)
            .filter((t) => t.overlay_url)
            .map((t) => ({
              name: t.name,
              url: t.overlay_url as string,
              kind: t.overlay_kind,
              label: t.label,
            }))
        : [],
    [result],
  );

  const busy = analysis.phase === "submitting" || analysis.phase === "running";

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 sm:py-10">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
            <Scan className="size-5 text-primary" /> Analyzer
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Ten independent checks, one probability, and the evidence behind it. Nothing you drop
            here is stored after the job ends.
          </p>
        </div>
        {busy && (
          <Button variant="outline" size="sm" onClick={reset}>
            <X /> Cancel
          </Button>
        )}
      </header>

      {analysis.phase === "idle" && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <DropzoneWrapper>
            <Dropzone onSource={submit} />
          </DropzoneWrapper>
          <HowItWorks entries={catalog.entries} />
        </div>
      )}

      {busy && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
          <Card className="lab-grid">
            <CardContent className="flex flex-col gap-3 pt-4">
              {base && (
                // eslint-disable-next-line @next/next/no-img-element -- local object URL
                <img src={base.url} alt={`Preview of ${base.name}`} className="max-h-52 w-full rounded border border-border object-contain" />
              )}
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" />
                {analysis.phase === "submitting" ? "Uploading…" : "Analysing…"}
              </p>
              <p className="tabular text-[0.7rem] text-muted-foreground">
                {base?.name ?? "fetched from URL"}
                {analysis.jobId && <> · {shortSha(analysis.jobId)}</>}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <ProgressStepper
                entries={catalog.entries}
                stages={analysis.stages}
                stageMs={analysis.stageMs}
                progress={analysis.progress}
              />
            </CardContent>
          </Card>
        </div>
      )}

      {analysis.phase === "failed" && analysis.error && (
        <Card className="border-risk-high/40">
          <CardHeader className="flex-row items-center gap-3">
            <CardTitle className="flex-1 text-risk-high">Analysis did not complete</CardTitle>
            <Button variant="outline" size="sm" onClick={reset}>
              <RotateCcw /> Try again
            </Button>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-xs">
            <p className="text-sm">{analysis.error.message}</p>
            <p className="text-muted-foreground">
              Code <span className="tabular font-mono">{analysis.error.code}</span>
              {analysis.error.retryAfterSeconds != null && (
                <> · retry in {analysis.error.retryAfterSeconds}s</>
              )}
            </p>
            {analysis.error.code === "rate_limited" && (
              <p className="text-muted-foreground">
                The engine limits requests per address so one browser tab cannot exhaust the worker
                pool. Waiting is the whole fix.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {result && analysis.phase === "succeeded" && (
        <div className="flex flex-col gap-6">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="tabular">
              <Timer className="size-3" />
              {ms(analysis.elapsedMs)} · {analysis.polls} polls
            </Badge>
            {result.from_cache && <Badge variant="muted">served from result cache</Badge>}
            <Badge variant="outline" className="tabular">
              links expire in <Expiry/>
            </Badge>
            <div className="ml-auto flex items-center gap-2">
              <span className="text-[0.7rem] text-muted-foreground">
                {saved === "saving" && "Saving to this browser…"}
                {saved === "saved" && "Saved to this browser's history"}
                {saved === "unavailable" && "This browser will not keep history"}
              </span>
              <Button asChild size="sm" variant="outline">
                <Link href="/history">
                  <HistoryIcon /> History
                </Link>
              </Button>
              <Button asChild size="sm">
                <Link href={`/report/${analysis.jobId}`}>
                  <FileText /> Forensic report
                </Link>
              </Button>
            </div>
          </div>

          {base && (
            <p className="text-xs text-muted-foreground">
              Analysed file: <span className="text-foreground">{base.name}</span> — the name is never
              sent to the server or stored by it.
            </p>
          )}

          <VerdictPanel result={result} />

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
            <Card>
              <CardHeader>
                <CardTitle>Evidence viewer</CardTitle>
                <p className="mt-1 text-xs text-muted-foreground">
                  Heatmaps are measurements painted over a measurement. The original is always one
                  slider away.
                </p>
              </CardHeader>
              <CardContent>
                <OverlayViewer
                  baseUrl={base?.url ?? null}
                  baseAlt={base ? `Analysed image ${base.name}` : "Analysed image"}
                  baseSize={
                    typeof result.image.width === "number" && typeof result.image.height === "number"
                      ? { width: result.image.width as number, height: result.image.height as number }
                      : null
                  }
                  overlays={overlays}
                  catalog={catalog.entries}
                  active={activeOverlay}
                  onSelect={setActiveOverlay}
                />
              </CardContent>
            </Card>

            <div className="flex flex-col gap-3">
              {result.metadata && <MetadataPanel metadata={result.metadata} c2pa={result.c2pa} />}
            </div>
          </div>

          <TechniqueSection
            entries={catalog.entries}
            techniques={Object.values(result.techniques)}
            activeOverlay={activeOverlay}
            onToggleOverlay={(name) => setActiveOverlay((current) => (current === name ? null : name))}
          />

          <TechnicalPanel result={result} />
        </div>
      )}
    </div>
  );
}

function DropzoneWrapper({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col gap-3">{children}</div>;
}

/** Counts down the window in which the signed overlay links still resolve. */
function Expiry() {
  const [left, setLeft] = React.useState(OVERLAY_TTL_S);
  React.useEffect(() => {
    const started = Date.now();
    const timer = setInterval(() => {
      const elapsed = (Date.now() - started) / 1000;
      setLeft(Math.max(0, OVERLAY_TTL_S - elapsed));
    }, 1000);
    return () => clearInterval(timer);
  }, []);
  const m = Math.floor(left / 60);
  const s = Math.floor(left % 60);
  return <span className="ml-1">{`${m}:${String(s).padStart(2, "0")}`}</span>;
}

function HowItWorks({ entries }: { entries: CatalogEntry[] }) {
  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle>What runs</CardTitle>
        <p className="mt-1 text-xs text-muted-foreground">
          {entries.length} checks, in parallel, on three resolution tracks.
        </p>
      </CardHeader>
      <CardContent>
        <ol className="flex flex-col gap-2">
          {entries.map((entry, i) => (
            <li key={entry.name} className="flex gap-2 text-[0.72rem] leading-snug">
              <span className="tabular shrink-0 text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
              <span>
                <span className="font-medium">{entry.label}</span>
                <span className="text-muted-foreground"> — {entry.summary}</span>
              </span>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}

function TechniqueSection({
  entries,
  techniques,
  activeOverlay,
  onToggleOverlay,
}: {
  entries: CatalogEntry[];
  techniques: TechniqueView[];
  activeOverlay: string | null;
  onToggleOverlay: (name: string) => void;
}) {
  const [sort, setSort] = React.useState<"registry" | "score">("registry");

  const order = React.useMemo(() => {
    const index = new Map(entries.map((e, i) => [e.name, i]));
    const list = [...techniques];
    list.sort((a, b) => {
      if (sort === "score") {
        const av = a.score ?? -1;
        const bv = b.score ?? -1;
        if (bv !== av) return bv - av;
      }
      return (index.get(a.name) ?? 99) - (index.get(b.name) ?? 99);
    });
    return list;
  }, [techniques, entries, sort]);

  const byGroup = (group: string) => order.filter((t) => t.group === group);

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-sm font-semibold tracking-tight">The ten checks</h2>
        <p className="text-xs text-muted-foreground">
          A greyed card is a check that could not measure this file — it stays visible on purpose.
        </p>
        <div className="ml-auto flex items-center gap-1 text-[0.7rem]">
          <span className="text-muted-foreground">Order</span>
          {(["registry", "score"] as const).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setSort(key)}
              aria-pressed={sort === key}
              className={cn(
                "rounded px-2 py-0.5 transition-colors",
                sort === key ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {key === "registry" ? "by method" : "by strength"}
            </button>
          ))}
        </div>
      </div>

      <Tabs defaultValue="all">
        <TabsList>
          <TabsTrigger value="all">All ({order.length})</TabsTrigger>
          <TabsTrigger value="ai">Synthetic ({byGroup("ai").length})</TabsTrigger>
          <TabsTrigger value="edited">Manipulation ({byGroup("edited").length})</TabsTrigger>
        </TabsList>
        {(["all", "ai", "edited"] as const).map((key) => {
          const rows = key === "all" ? order : byGroup(key);
          return (
            <TabsContent key={key} value={key}>
              <motion.div
                layout
                className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3"
              >
                {rows.map((technique) => (
                  <TechniqueCard
                    key={technique.name}
                    technique={technique}
                    hasOverlay={Boolean(technique.overlay_url)}
                    showing={activeOverlay === technique.name}
                    onToggleOverlay={onToggleOverlay}
                  />
                ))}
                {!rows.length && (
                  <p className="col-span-full text-xs text-muted-foreground">
                    No check feeds that sub-score for this file.
                  </p>
                )}
              </motion.div>
            </TabsContent>
          );
        })}
      </Tabs>
    </section>
  );
}
