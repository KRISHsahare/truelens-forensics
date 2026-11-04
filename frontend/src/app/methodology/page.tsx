"use client";

/**
 * Model & methodology.
 *
 * The page reads `/api/metrics`, which is a pass-through of the artifacts the server actually
 * loaded — so the thresholds and weights shown here are the ones in the running process rather
 * than a copy that can rot. When an artifact is missing the route says so, and this page renders
 * that sentence instead of a chart. A zero substituted for an unmeasured figure would be the most
 * damaging thing this page could do.
 *
 * The technique diagrams describe the *implemented* method, including where the implementation
 * diverges from the textbook version of the check.
 */

import * as React from "react";
import { Activity, Boxes, Database, GitCompare, Scale } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ConfusionMatrix, CurveCharts, DetectorStatus, FusionWeights, MetricTiles } from "@/components/methodology/charts";
import { TechniqueDiagram } from "@/components/methodology/diagrams";
import { getMetrics } from "@/lib/api";
import { GROUP_LABELS } from "@/lib/format";
import { ApiError, type MetricsResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function MethodologyPage() {
  const [data, setData] = React.useState<MetricsResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    getMetrics()
      .then((metrics) => !cancelled && setData(metrics))
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : "The analysis engine is not answering.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
      <header className="flex flex-col gap-2">
        <Badge variant="secondary" className="w-fit">
          <Activity className="size-3" /> how the numbers are made
        </Badge>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Model &amp; methodology</h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Every check, what it measures, what it costs in the fusion, and what the learned detector
          scored on held-out data. Read the last section first if you only read one: it says what
          these numbers do not cover.
        </p>
        {error && (
          <p className="rounded border border-risk-high/40 bg-risk-high/10 px-3 py-2 text-xs text-risk-high">
            {error} — the catalogue below falls back to the built-in list, and no metric is shown
            rather than an assumed one.
          </p>
        )}
      </header>

      {!data && !error && (
        <div className="mt-8 flex flex-col gap-3">
          <Skeleton className="h-20 w-full" />
          <div className="grid gap-3 sm:grid-cols-3">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="h-28 w-full" />
            ))}
          </div>
        </div>
      )}

      {data && (
        <>
          <section className="mt-8 flex flex-col gap-3">
            <h2 className="text-lg font-semibold tracking-tight">Engine state</h2>
            <DetectorStatus detectors={data.detectors} />
            <Card>
              <CardContent className="flex flex-col gap-2 pt-4">
                <p className="text-xs leading-relaxed text-muted-foreground">{data.fusion.note}</p>
                <p className="text-[0.7rem]">
                  fusion artifact: <span className="tabular font-mono">{data.fusion.artifact || "—"}</span>
                </p>
              </CardContent>
            </Card>
          </section>

          <section className="mt-10 flex flex-col gap-3">
            <h2 className="text-lg font-semibold tracking-tight">The checks</h2>
            <p className="max-w-3xl text-xs text-muted-foreground">
              Three resolution tracks, and the split matters: compression checks need the native,
              never-resampled pixels because resampling destroys exactly the artefact they measure,
              while learned and spatial checks need a bounded working size to finish in seconds.
            </p>
            <div className="grid gap-3 lg:grid-cols-2">
              {data.techniques.map((technique) => (
                <Card key={technique.name} className="h-full">
                  <CardHeader className="gap-1">
                    <div className="flex items-center gap-2">
                      <CardTitle className="flex-1 text-sm">{technique.label}</CardTitle>
                      {technique.heavy && <Badge variant="muted" className="shrink-0">torch</Badge>}
                    </div>
                    <p className="flex flex-wrap gap-x-2 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                      <span>{GROUP_LABELS[technique.group] ?? technique.group}</span>
                      <span aria-hidden>·</span>
                      <span>{technique.track} track</span>
                      <span aria-hidden>·</span>
                      <span>{technique.overlay === "none" ? "no overlay" : `${technique.overlay} overlay`}</span>
                      <span aria-hidden>·</span>
                      <span className="tabular">
                        weight{" "}
                        {data.fusion.expert_weights[technique.name] != null
                          ? data.fusion.expert_weights[technique.name].toFixed(2)
                          : "—"}
                      </span>
                    </p>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-2">
                    <TechniqueDiagram name={technique.name} />
                    <p className="text-xs leading-relaxed text-muted-foreground">{technique.summary}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>

          <section className="mt-10 flex flex-col gap-4">
            <h2 className="text-lg font-semibold tracking-tight">Learned detector evaluation</h2>
            {data.evaluation.available ? (
              <>
                <EvaluationMeta evaluation={data.evaluation} />
                <MetricTiles evaluation={data.evaluation} />
                <CurveCharts evaluation={data.evaluation} />
                <Separator />
                <div className="flex flex-col gap-2">
                  <h3 className="text-sm font-semibold">Confusion matrix</h3>
                  <ConfusionMatrix evaluation={data.evaluation} />
                </div>
                {data.evaluation.calibration && (
                  <div className="flex flex-col gap-2">
                    <h3 className="text-sm font-semibold">Calibration</h3>
                    <Card>
                      <CardContent className="pt-4 text-xs leading-relaxed text-muted-foreground">
                        <p>
                          Method: <span className="text-foreground">{data.evaluation.calibration.method ?? "—"}</span>
                        </p>
                        {data.evaluation.calibration.note && <p className="mt-1">{data.evaluation.calibration.note}</p>}
                      </CardContent>
                    </Card>
                  </div>
                )}
              </>
            ) : (
              <MissingEvaluation
                reason={data.evaluation.reason ?? "No evaluation file was found."}
                hint={data.evaluation.hint ?? ""}
                meaning={data.evaluation.meaning ?? ""}
              />
            )}
          </section>

          <section className="mt-10 flex flex-col gap-3">
            <h2 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
              <Scale className="size-4" /> Fusion, and why it is not a hand-tuned sum
            </h2>
            <Card>
              <CardContent className="pt-4">
                <FusionWeights
                  weights={data.fusion.expert_weights}
                  source={data.fusion.source}
                  calibrated={data.fusion.calibrated}
                  note={data.fusion.note}
                />
                <Separator className="my-3" />
                <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
                  The prototype summed weighted scores and stretched the total by a factor of 1.6 to
                  make it look decisive. That multiplier is gone: sub-scores are a noisy-OR over the
                  contributing checks, deterministic facts enter as flags that can floor a score, and
                  a fitted fusion model replaces the expert weights as soon as one is calibrated and
                  present. The weights above are what the running process is using right now.
                </p>
              </CardContent>
            </Card>
          </section>

          <section className="mt-10 flex flex-col gap-3">
            <h2 className="text-lg font-semibold tracking-tight">What these numbers do not cover</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                ["Dataset, not distribution", "Metrics describe the held-out split of the training corpora. A file outside that domain — a document scan, an anime frame, a medical image — is measured against a reference it does not share."],
                ["Still frames only", "No temporal signal means no video deepfake detection. One image is the whole input."],
                ["Re-encoding is destructive", "Platform compression removes the artefacts several checks rely on, which lowers evidence rather than raising it."],
                ["Calibration is a claim about a set", "A probability is only as good as the frequencies it was fitted against; the fusion note above states which kind of weights are in use."],
              ].map(([title, body]) => (
                <Card key={title} className="border-risk-moderate/30">
                  <CardHeader>
                    <CardTitle className="text-sm">{title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xs leading-relaxed text-muted-foreground">{body}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
            <p className="text-xs">
              <a href="/about#limitations" className="text-primary underline-offset-4 hover:underline">
                Full limitations and privacy statement →
              </a>
            </p>
          </section>
        </>
      )}
    </div>
  );
}

function EvaluationMeta({ evaluation }: { evaluation: MetricsResponse["evaluation"] }) {
  const dataset = evaluation.dataset ?? {};
  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <Database className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <CardTitle className="text-sm">Provenance of the evaluation</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            {evaluation.model ?? "model name not recorded"}
            {evaluation.trained_at && <> · trained {new Date(evaluation.trained_at).toLocaleString()}</>}
          </p>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 text-xs">
        <div className="flex flex-wrap gap-x-5 gap-y-1">
          <span>
            <span className="text-muted-foreground">dataset: </span>
            <span className="font-medium">{dataset.name ?? "—"}</span>
          </span>
          {dataset.train != null && <span className="tabular"><span className="text-muted-foreground">train: </span>{dataset.train.toLocaleString()}</span>}
          {dataset.val != null && <span className="tabular"><span className="text-muted-foreground">val: </span>{dataset.val.toLocaleString()}</span>}
          {dataset.test != null && <span className="tabular"><span className="text-muted-foreground">test: </span>{dataset.test.toLocaleString()}</span>}
          {Boolean(dataset.classes?.length) && (
            <span>
              <span className="text-muted-foreground">classes: </span>
              {dataset.classes?.join(", ")}
            </span>
          )}
          {evaluation.metrics?.threshold != null && (
            <span className="tabular">
              <span className="text-muted-foreground">threshold: </span>
              {evaluation.metrics.threshold.toFixed(3)}
            </span>
          )}
        </div>
        {Boolean(dataset.sources?.length) && (
          <ul className="flex flex-wrap gap-1.5">
            {dataset.sources?.map((source) => (
              <li key={source.name}>
                <Badge variant="outline" className="tabular text-[0.65rem]">
                  <Boxes className="size-3" /> {source.name} {source.count.toLocaleString()}
                </Badge>
              </li>
            ))}
          </ul>
        )}
        {Boolean(dataset.augmentations?.length) && (
          <p className="text-[0.7rem] text-muted-foreground">
            Augmentation: {dataset.augmentations?.join(", ")}
          </p>
        )}
        {dataset.note && <p className="text-[0.7rem] leading-snug text-muted-foreground">{dataset.note}</p>}
        {evaluation.tta && Object.keys(evaluation.tta).length > 0 && (
          <p className="flex items-center gap-1.5 text-[0.7rem] text-muted-foreground">
            <GitCompare className="size-3" /> test-time augmentation: {JSON.stringify(evaluation.tta)}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function MissingEvaluation({ reason, hint, meaning }: { reason: string; hint: string; meaning: string }) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col gap-2 py-5">
        <p className="text-sm font-medium">No evaluation is published for this deployment</p>
        <p className="text-xs leading-relaxed text-muted-foreground">{reason}</p>
        {meaning && (
          <p className="rounded border border-risk-moderate/30 bg-risk-moderate/10 px-2.5 py-1.5 text-[0.72rem] leading-snug text-risk-moderate">
            {meaning}
          </p>
        )}
        {hint && (
          <p className="text-[0.7rem] leading-snug text-muted-foreground">
            To produce it: <code className={cn("rounded bg-muted px-1 font-mono text-[0.68rem]")}>{hint}</code>
          </p>
        )}
      </CardContent>
    </Card>
  );
}
