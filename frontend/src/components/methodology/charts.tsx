"use client";

/**
 * The evaluation charts.
 *
 * Everything here renders from `evaluation` — the pass-through of `artifacts/metrics.json` — and
 * nothing here invents a value when a field is missing. A chart with a fabricated point is worse
 * than an empty panel, because a chart is what a reader trusts. The `Missing` component is the
 * default for every figure on this page, and the reason the page can look sparse is the same reason
 * it can be believed.
 *
 * Recharts draws the curves; the confusion matrix is hand-laid HTML, because a matrix wants exact
 * cell borders, per-cell counts and a diagonal that reads as the happy path — none of which a
 * generic heatmap component gives you without a fight.
 */

import * as React from "react";
import {
  Area,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip as ReTooltip,
  XAxis,
  YAxis,
  ComposedChart,
} from "recharts";
import { BAND_COLORS } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Evaluation } from "@/lib/types";

const AXIS = { stroke: "var(--muted-foreground)", fontSize: 11 } as const;

/** Recharts hands the formatter an unknown-ish value union; the curves here are numeric or absent. */
function formatPoint(value: unknown, name: unknown): [string, string] {
  const numeric = typeof value === "number" ? value : Number(value);
  return [Number.isFinite(numeric) ? numeric.toFixed(4) : "—", String(name ?? "")];
}

export function MetricTiles({ evaluation }: { evaluation: Evaluation }) {
  const m = evaluation.metrics ?? {};
  const tiles: [string, number | undefined, string][] = [
    ["ROC AUC", m.roc_auc, "Ranking quality across every threshold. The one number that survives a changed operating point."],
    ["PR AUC", m.pr_auc, "Average precision. Falls harder than ROC AUC when the negative class dominates."],
    ["Accuracy", m.accuracy, "At the reported threshold only — meaningless without it."],
    ["Precision", m.precision, "Of the images flagged, how many were genuinely synthetic."],
    ["Recall", m.recall, "Of the synthetic images, how many were flagged."],
    ["F1", m.f1, "The harmonic mean of the two above, so a lopsided pair cannot hide."],
  ];
  return (
    <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
      {tiles.map(([label, value, hint]) => (
        <div key={label} className="rounded-lg border border-border bg-card p-3" title={hint}>
          <p className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">{label}</p>
          <p className={cn("tabular mt-1 text-xl font-semibold", value == null && "text-muted-foreground")}>
            {value == null ? "—" : value.toFixed(3)}
          </p>
        </div>
      ))}
    </div>
  );
}

/**
 * ROC and precision-recall together.
 *
 * Both are drawn against the *same* threshold sweep so a reader can move between them by eye: a
 * point that looks excellent on ROC and poor on PR is the classic imbalance tell, and having the two
 * curves side by side is what makes that visible without explanation.
 */
export function CurveCharts({ evaluation }: { evaluation: Evaluation }) {
  const roc = evaluation.roc_curve;
  const pr = evaluation.pr_curve;

  const rocData = React.useMemo(
    () =>
      roc?.fpr && roc?.tpr
        ? roc.fpr.map((fpr, i) => ({ fpr, tpr: roc.tpr![i], threshold: roc.thresholds?.[i] }))
        : null,
    [roc],
  );

  const prData = React.useMemo(
    () =>
      pr?.recall && pr?.precision
        ? pr.recall.map((recall, i) => ({ recall, precision: pr.precision![i] }))
        : null,
    [pr],
  );

  if (!rocData && !prData) {
    return (
      <Missing what="No ROC or precision-recall curve" because="metrics.json carries no curve arrays — the evaluation run exported scalar metrics only, or never ran." />
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {rocData ? (
        <Figure title="ROC" sub="True-positive rate against false-positive rate; the diagonal is chance.">
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={rocData}>
              <CartesianGrid stroke="var(--grid-line)" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "false positive rate", position: "insideBottom", offset: -4, ...AXIS }} {...AXIS} />
              <YAxis domain={[0, 1]} {...AXIS} />
              <ReTooltip contentStyle={tooltipStyle} formatter={formatPoint} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line dataKey="tpr" name="model" stroke={BAND_COLORS.low} dot={false} strokeWidth={2} isAnimationActive={false} />
              {/* Recharts has no reference-line-by-equation, so the diagonal is drawn as a second
                  series over the same x values: chance is tpr == fpr, point for point. */}
              <Line dataKey="fpr" name="chance" stroke="var(--muted-foreground)" strokeDasharray="4 4" dot={false} strokeWidth={1} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="mt-1 text-[0.7rem] text-muted-foreground">
            {rocData.length} operating points · AUC{" "}
            <span className="tabular">{evaluation.metrics?.roc_auc?.toFixed(3) ?? "not reported"}</span>
          </p>
        </Figure>
      ) : (
        <Missing compact what="No ROC curve" because="the metrics file has no roc_curve arrays." />
      )}

      {prData ? (
        <Figure title="Precision–recall" sub="Against recall. A curve hugging the top-right is a detector that stays precise as it catches more.">
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={prData}>
              <CartesianGrid stroke="var(--grid-line)" />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} label={{ value: "recall", position: "insideBottom", offset: -4, ...AXIS }} {...AXIS} />
              <YAxis domain={[0, 1]} {...AXIS} />
              <ReTooltip contentStyle={tooltipStyle} formatter={formatPoint} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area dataKey="precision" name="precision" stroke={BAND_COLORS.moderate} fill="var(--chart-2)" fillOpacity={0.12} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <p className="mt-1 text-[0.7rem] text-muted-foreground">
            {prData.length} points · PR AUC{" "}
            <span className="tabular">{evaluation.metrics?.pr_auc?.toFixed(3) ?? "not reported"}</span>
          </p>
        </Figure>
      ) : (
        <Missing compact what="No precision–recall curve" because="the metrics file has no pr_curve arrays." />
      )}
    </div>
  );
}

export function ConfusionMatrix({ evaluation }: { evaluation: Evaluation }) {
  const cm = evaluation.confusion_matrix;
  if (!cm?.matrix?.length) {
    return <Missing what="No confusion matrix" because="metrics.json carries no confusion_matrix — it is written by the evaluation pass of the training script." />;
  }
  const labels = cm.labels?.length === cm.matrix.length ? cm.labels : ["real", "synthetic"];
  const total = cm.matrix.flat().reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...cm.matrix.flat(), 1);

  return (
    <div className="flex flex-col gap-2">
      <div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border" style={{ gridTemplateColumns: `8rem repeat(${labels.length}, 1fr)` }}>
        <CornerCell />
        {labels.map((label) => (
          <div key={`head-${label}`} className="bg-card px-2 py-1.5 text-center text-[0.7rem] font-medium">
            {label}
          </div>
        ))}
        {cm.matrix.map((row, i) => (
          <React.Fragment key={`row-${i}`}>
            <div className="flex items-center justify-end gap-1 bg-card px-2 py-1.5 text-right text-[0.7rem] font-medium">
              {labels[i]}
              <span className="text-[0.6rem] font-normal text-muted-foreground">actual</span>
            </div>
            {row.map((count, j) => {
              const correct = i === j;
              const intensity = count / max;
              return (
                <div
                  key={`cell-${i}-${j}`}
                  className={cn(
                    "relative flex min-h-16 flex-col items-center justify-center gap-0.5 px-2",
                    correct ? "bg-risk-low/10" : "bg-risk-high/10",
                  )}
                >
                  <span
                    aria-hidden
                    className="absolute inset-0"
                    style={{
                      backgroundColor: correct
                        ? `color-mix(in oklch, ${BAND_COLORS.low} ${Math.round(intensity * 55)}%, transparent)`
                        : `color-mix(in oklch, ${BAND_COLORS.high} ${Math.round(intensity * 55)}%, transparent)`,
                    }}
                  />
                  <span className="tabular relative text-base font-semibold">{count.toLocaleString()}</span>
                  <span className="tabular relative text-[0.65rem] text-muted-foreground">
                    {((count / total) * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <p className="text-[0.7rem] text-muted-foreground">
        Rows are the true class, columns the predicted one. The diagonal is agreement; the off-diagonal
        is the two failure modes, and they are not equally bad — a false negative on a disinformation
        image and a false positive on a news photograph are different harms, which is why both halves
        of this matrix get read.
      </p>
    </div>
  );
}

function CornerCell() {
  return (
    <div className="bg-card px-2 py-1.5 text-[0.62rem] text-muted-foreground">
      predicted →
    </div>
  );
}

/**
 * The fusion weights, as they actually ship.
 *
 * `expert` weights are a prior a human wrote down, and the page says so; a fitted model's
 * coefficients are measurements. Those two states must not look the same on a bar chart, because
 * they do not mean the same thing — hence the note under the figure rather than a colour trick.
 */
export function FusionWeights({
  weights,
  source,
  calibrated,
  note,
  measured,
}: {
  weights: Record<string, number>;
  source: string;
  calibrated: boolean;
  note: string;
  measured?: Record<string, number>;
}) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 1);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-[0.7rem]">
        <span className="rounded-full border border-border px-2 py-0.5">
          source: <span className="font-medium">{source}</span>
        </span>
        <span className="rounded-full border border-border px-2 py-0.5">
          calibrated: <span className="font-medium">{calibrated ? "yes" : "no"}</span>
        </span>
      </div>
      <ul className="flex flex-col gap-1">
        {entries.map(([name, value]) => (
          <li key={name} className="flex items-center gap-2">
            <span className="w-32 shrink-0 truncate font-mono text-[0.7rem] text-muted-foreground">{name}</span>
            <span className="relative h-3 flex-1 overflow-hidden rounded bg-muted">
              <span
                className="absolute inset-y-0 left-0 rounded bg-primary/70"
                style={{ width: `${(Math.abs(value) / max) * 100}%` }}
              />
            </span>
            <span className="tabular w-14 shrink-0 text-right text-[0.7rem]">{value.toFixed(3)}</span>
            {measured?.[name] != null && (
              <span className="tabular w-14 shrink-0 text-right text-[0.7rem] text-muted-foreground">
                → {measured[name].toFixed(3)}
              </span>
            )}
          </li>
        ))}
      </ul>
      <p className="text-[0.7rem] leading-snug text-muted-foreground">{note}</p>
    </div>
  );
}

export function DetectorStatus({ detectors }: { detectors: Record<string, { present: boolean; enabled: boolean }> }) {
  const rows = Object.entries(detectors);
  if (!rows.length) return null;
  return (
    <ul className="flex flex-wrap gap-2">
      {rows.map(([name, state]) => (
        <li key={name} className="flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-[0.7rem]">
          <span
            className={cn(
              "size-1.5 rounded-full",
              state.present && state.enabled
                ? "bg-risk-low"
                : state.enabled
                  ? "bg-risk-moderate"
                  : "bg-muted-foreground",
            )}
          />
          <span className="font-medium">{name}</span>
          <span className="text-muted-foreground">
            {state.present ? (state.enabled ? "loaded" : "present, disabled") : state.enabled ? "expected, missing" : "off"}
          </span>
        </li>
      ))}
    </ul>
  );
}

const tooltipStyle = {
  backgroundColor: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 11,
  color: "var(--popover-foreground)",
} as const;

function Figure({ title, sub, children }: { title: string; sub: string; children: React.ReactNode }) {
  return (
    <figure className="flex flex-col gap-1 rounded-lg border border-border bg-card p-3">
      <figcaption className="flex flex-col">
        <span className="text-sm font-medium">{title}</span>
        <span className="text-[0.7rem] text-muted-foreground">{sub}</span>
      </figcaption>
      {children}
    </figure>
  );
}

function Missing({ what, because, compact = false }: { what: string; because: string; compact?: boolean }) {
  return (
    <div className={cn("flex flex-col items-start justify-center gap-1 rounded-lg border border-dashed border-border bg-card/40 p-4", compact ? "min-h-24" : "min-h-40")}>
      <p className="text-xs font-medium text-muted-foreground">{what}</p>
      <p className="max-w-md text-[0.7rem] leading-snug text-muted-foreground/80">{because}</p>
    </div>
  );
}
