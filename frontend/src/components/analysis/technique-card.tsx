"use client";

/**
 * One card per technique — the unit the whole product is judged on.
 *
 * A score on its own is a decoration. The card therefore answers, in this order: what was measured,
 * how much it counted, why it counts, and what the raw numbers were. The "why this matters" sentence
 * is the server's (`explanation.why`), never a frontend string, because the reason a check matters
 * changes when its thresholds change and only the analyser knows which ones it used.
 *
 * Unmeasured cards stay on screen. Hiding a check that failed to run is the one presentation choice
 * that would turn an engineering limitation into a misleading result.
 */

import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Activity, ChevronDown, Eye, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScoreBar } from "@/components/ui/progress";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { BAND_CLASSES, GROUP_LABELS, WEIGHT_LABELS, bandFor, ms, pct, prob } from "@/lib/format";
import { cn } from "@/lib/utils";
import { humanize } from "@/lib/catalog";
import type { TechniqueView } from "@/lib/types";

export interface TechniqueCardProps {
  technique: TechniqueView;
  /** Whether an overlay exists, so the "show on image" affordance is honest. */
  hasOverlay: boolean;
  showing: boolean;
  onToggleOverlay?: (name: string) => void;
}

export function TechniqueCard({ technique, hasOverlay, showing, onToggleOverlay }: TechniqueCardProps) {
  const reduce = useReducedMotion();
  const band = technique.band === "not_measured" || technique.score == null ? "not_measured" : technique.band;
  const classes = BAND_CLASSES[band];
  const measured = technique.measured && technique.score != null;

  return (
    <motion.article
      initial={reduce ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={cn(
        "flex flex-col gap-3 rounded-lg border bg-card p-3.5 transition-colors",
        measured ? "border-border" : "border-border/50",
        !technique.ran && "bg-muted/20",
      )}
      aria-label={`${technique.label}: ${measured ? pct(technique.score) : "not measured"}`}
    >
      <header className="flex items-start gap-2">
        <span className={cn("mt-0.5 inline-flex size-6 shrink-0 items-center justify-center rounded", classes.bg, classes.text)}>
          <Activity className="size-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-medium leading-tight">
            {technique.label || humanize(technique.name)}
          </h3>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[0.68rem] text-muted-foreground">
            <span>{GROUP_LABELS[technique.group] ?? technique.group}</span>
            <span aria-hidden>·</span>
            <span className="tabular">{ms(technique.ms)}</span>
            {technique.status !== "ok" && (
              <>
                <span aria-hidden>·</span>
                <span className={cn(technique.status === "skipped" ? "" : "text-risk-moderate")}>
                  {technique.status}
                </span>
              </>
            )}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end">
          <AnimatePresence initial={false}>
            <motion.span
              key={measured ? String(technique.score) : "none"}
              initial={reduce ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={reduce ? undefined : { opacity: 0 }}
              className={cn("tabular text-base font-semibold leading-none", classes.text)}
            >
              {measured ? pct(technique.score) : "—"}
            </motion.span>
          </AnimatePresence>
          <span className="mt-0.5 text-[0.62rem] text-muted-foreground">
            {measured ? prob(technique.score) : "no measurement"}
          </span>
        </div>
      </header>

      <ScoreBar score={measured ? technique.score : null} bandClass={classes.bar} />

      <div className="flex flex-col gap-1.5">
        <p className="text-xs leading-relaxed text-foreground/85">{technique.explanation.found}</p>
        <p className="flex items-start gap-1.5 text-[0.7rem] leading-relaxed text-muted-foreground">
          <Info className="mt-0.5 size-3 shrink-0" />
          <span>{technique.explanation.why}</span>
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant="muted" className={cn(classes.bg, classes.text, "border-transparent")}>
          {WEIGHT_LABELS[technique.explanation.weight] ?? technique.explanation.weight}
        </Badge>
        {technique.explanation.weight_note && (
          <span className="text-[0.65rem] text-muted-foreground">{technique.explanation.weight_note}</span>
        )}
        {hasOverlay && onToggleOverlay && (
          <button
            type="button"
            onClick={() => onToggleOverlay(technique.name)}
            aria-pressed={showing}
            className={cn(
              "ml-auto inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.68rem] transition-colors",
              showing
                ? "border-primary bg-primary/15 text-primary"
                : "border-border text-muted-foreground hover:text-foreground",
            )}
          >
            <Eye className="size-3" />
            {showing ? "Hide on image" : "Show on image"}
          </button>
        )}
      </div>

      {technique.error && (
        <p className="rounded border border-risk-high/30 bg-risk-high/10 px-2 py-1 text-[0.7rem] text-risk-high">
          {technique.error}
        </p>
      )}

      <Accordion type="single" collapsible>
        <AccordionItem value="raw" className="border-b-0">
          <AccordionTrigger className="py-1">Raw measurements</AccordionTrigger>
          <AccordionContent>
            <RawDetails details={technique.details} />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </motion.article>
  );
}

/**
 * The details dict is whatever the analyser returned, so it is rendered structurally rather than
 * field by field: a new key in a technique becomes visible in the UI on the same deploy that
 * introduced it, which is the property a forensic tool needs and an ad-hoc renderer loses.
 */
function RawDetails({ details }: { details: Record<string, unknown> }) {
  const keys = Object.keys(details ?? {});
  if (!keys.length) {
    return <p className="text-[0.7rem] text-muted-foreground">This check reported no numeric detail.</p>;
  }
  return (
    <dl className="flex flex-col divide-y divide-border/60 text-[0.7rem]">
      {keys.map((key) => (
        <div key={key} className="flex items-start justify-between gap-3 py-1">
          <dt className="shrink-0 font-mono text-muted-foreground">{key}</dt>
          <dd className="min-w-0 break-words text-right tabular">{renderValue(details[key])}</dd>
        </div>
      ))}
    </dl>
  );
}

function renderValue(value: unknown): React.ReactNode {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) {
    return value.length > 6 ? `${value.length} items` : value.map((v) => renderValue(v)).join(", ");
  }
  if (typeof value === "object") {
    const json = JSON.stringify(value);
    if (!json) return "—";
    return json.length > 120 ? `${json.slice(0, 117)}…` : json;
  }
  return String(value);
}

/** A compact row for the same data in a table context (batch results). */
export function ScoreChip({ score }: { score: number | null }) {
  const band = bandFor(score);
  return (
    <span className={cn("tabular inline-flex items-center gap-1.5 text-xs", BAND_CLASSES[band].text)}>
      <span className={cn("size-1.5 rounded-full", BAND_CLASSES[band].bar)} />
      {pct(score)}
    </span>
  );
}
