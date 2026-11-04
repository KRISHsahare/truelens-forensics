"use client";

/**
 * The answer, and everything that qualifies it.
 *
 * Four things sit next to the number, not under it, because a reader who only sees the gauge sees a
 * different product than the one being shipped:
 *
 * - two sub-scores. `overall` is `max(ai, edited)` — it is *not* an average, and a single blended
 *   figure would hide that a photo can look unedited and ungenerated at the same time.
 * - the evidence flags that moved a score. The engine floors a sub-score when a deterministic fact
 *   (a signed C2PA generative manifest, an identified generator tag) overrides the statistics; the UI
 *   has to say so, or an adjusted number is presented as a model output.
 * - reliability, separately, with the reason for every penalty.
 * - limitations, always expanded enough to be noticed. Not a footnote: the honest answer to "is this
 *   proof" is no, and that answer should not require a click on a page whose subject is proof.
 */

import * as React from "react";
import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, Flag, Sparkles, TriangleAlert, Wrench } from "lucide-react";
import { BandBadge, Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Gauge, ReliabilityMeter } from "./gauge";
import { BAND_CLASSES, VERDICT_LABELS, verdictBand } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AnalysisResult, EvidenceFlagView } from "@/lib/types";

export function VerdictPanel({ result, className }: { result: AnalysisResult; className?: string }) {
  const reduce = useReducedMotion();
  const band = verdictBand(result.verdict, result.overall);
  const classes = BAND_CLASSES[band];

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className={cn("flex flex-col gap-4", className)}
    >
      {/* The banner goes horizontal only when *this card* is wide enough for two gauges and a
          paragraph to coexist. A breakpoint on the viewport was wrong: the report page puts the same
          panel in a narrower column, and the fixed-width gauges squeezed the headline to one word
          per line while the browser still thought it was a desktop. */}
      <Card className="hairline @container overflow-hidden">
        <div className={cn("flex flex-col gap-4 p-4 @2xl:flex-row @2xl:items-center", classes.bg)}>
          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <BandBadge band={band} className="self-start">
              {VERDICT_LABELS[result.verdict]}
            </BandBadge>
            <h2 className="text-lg font-semibold leading-tight tracking-tight">{result.headline}</h2>
            <p className="text-sm leading-relaxed text-foreground/85">{result.detail}</p>
            {Boolean(result.fusion?.adjusted) && (
              <p className="flex items-start gap-1.5 text-xs text-muted-foreground">
                <Flag className="mt-0.5 size-3 shrink-0" />
                A deterministic finding raised one of the scores above what the statistics alone
                produced — see the evidence flags below.
              </p>
            )}
          </div>

          <div className="flex shrink-0 items-end justify-center gap-2 @2xl:justify-end">
            <Gauge
              value={result.ai_score}
              label="Synthetic likelihood"
              caption="Signals that the pixels were produced by a model."
              size="sm"
            />
            <Gauge
              value={result.edited_score}
              label="Manipulation likelihood"
              caption="Signals that a captured photo was altered."
              size="sm"
            />
          </div>
        </div>

        <CardContent className="flex flex-col gap-3 pt-4">
          <ReliabilityMeter
            score={result.reliability}
            note={result.reliability_detail.note}
          />

          {result.reasons.length > 0 && (
            <>
              <Separator />
              <ul className="flex flex-col gap-1.5">
                {result.reasons.map((reason, i) => (
                  <li key={`${i}-${reason.slice(0, 24)}`} className="flex items-start gap-2 text-xs leading-relaxed">
                    <span className={cn("mt-1.5 size-1 shrink-0 rounded-full", classes.bar)} />
                    <span className="text-foreground/85">{reason}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </CardContent>
      </Card>

      {result.flags.length > 0 && <FlagList flags={result.flags} adjusted={Boolean(result.fusion?.adjusted)} />}
      {result.reliability_detail.reasons.length > 0 && (
        <ReliabilityPenalties result={result} />
      )}
      {result.limitations.length > 0 && <Limitations items={result.limitations} />}
    </motion.div>
  );
}

function FlagList({ flags, adjusted }: { flags: EvidenceFlagView[]; adjusted: boolean }) {
  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <Flag className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <CardTitle>Evidence flags</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            {adjusted
              ? "Deterministic findings raised one of the scores above what the statistics alone produced."
              : "Named facts about this file. None of them moved a score."}
          </p>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-1.5">
        {flags.map((flag) => (
          <div
            key={flag.id}
            className="flex items-start gap-2 rounded border border-border/70 bg-muted/30 px-2.5 py-1.5 text-xs"
          >
            {flag.kind === "deterministic" ? (
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-risk-moderate" />
            ) : (
              <Sparkles className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
            )}
            <div className="min-w-0 flex-1">
              <p className="leading-snug">{flag.text}</p>
              {(flag.target || flag.floor != null || flag.forces) && (
                <p className="mt-0.5 text-[0.68rem] text-muted-foreground">
                  {flag.target && <span>on {flag.target} </span>}
                  {flag.floor != null && <span className="tabular">floored to {flag.floor.toFixed(2)} </span>}
                  {flag.forces && <span>· forces “{VERDICT_LABELS[flag.forces]}” </span>}
                  · {flag.kind}
                </p>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function ReliabilityPenalties({ result }: { result: AnalysisResult }) {
  const coverage = result.reliability_detail.coverage ?? {};
  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <TriangleAlert className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <CardTitle>Why reliability is {Math.round(result.reliability * 100)}%</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Penalties multiply; they do not subtract. A check that timed out costs less than a check
            that could not run at all.
          </p>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <ul className="flex flex-col gap-1.5">
          {result.reliability_detail.reasons.map((penalty) => (
            <li key={penalty.id} className="flex items-start gap-2 text-xs leading-snug">
              <Badge variant={penalty.severity === "severe" ? "destructive" : "secondary"} className="mt-0.5 shrink-0">
                ×{penalty.cost.toFixed(2)}
              </Badge>
              <span className="text-foreground/85">{penalty.reason}</span>
            </li>
          ))}
        </ul>
        {(Object.keys(coverage).length > 0) && (
          <p className="text-[0.7rem] text-muted-foreground">
            Measured checks:{" "}
            {Object.entries(coverage)
              .map(([group, count]) => `${group} ${count}`)
              .join(" · ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function Limitations({ items }: { items: { title: string; body: string }[] }) {
  return (
    <Card className="border-border/70 bg-card/60">
      <CardHeader className="flex-row items-center gap-2">
        <Wrench className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <CardTitle>What this analysis cannot tell you</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">
            Fixed properties of the method, not problems with this particular file.
          </p>
        </div>
      </CardHeader>
      <CardContent>
        <ul className="grid gap-2 sm:grid-cols-2">
          {items.map((item) => (
            <li key={item.title} className="rounded border border-border/60 p-2.5">
              <p className="text-xs font-medium">{item.title}</p>
              <p className="mt-1 text-[0.7rem] leading-relaxed text-muted-foreground">{item.body}</p>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
