"use client";

/**
 * The technical tail of a result: how it was computed, and from what.
 *
 * This is the part a reviewer checks and a user skips, so it is one collapsible block rather than
 * three panels. Everything in it came from the server's own `timings`, `fusion` and `provenance`
 * blocks — the app does not recompute a number it was given, because a frontend that recalculates
 * fusion would be a second, unfalsifiable copy of the model.
 */

import * as React from "react";
import { Copy, Hash, Timer } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { bytes, ms, shortSha } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AnalysisResult } from "@/lib/types";

export function TechnicalPanel({ result, className }: { result: AnalysisResult; className?: string }) {
  const [copied, setCopied] = React.useState(false);
  const fusion = result.fusion ?? {};
  const contributions = (fusion.contributions ?? {}) as Record<string, number>;
  const image = result.image ?? {};
  const provenance = result.provenance ?? {};
  const timings = result.timings;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(result.sha256);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access is a permission the browser may simply refuse; the hash stays readable.
    }
  };

  return (
    <Card className={cn("border-border/70", className)}>
      <CardHeader className="flex-row items-center gap-2">
        <Hash className="size-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <CardTitle>How this was computed</CardTitle>
          <button
            type="button"
            onClick={copy}
            title="Copy the full SHA-256"
            className="mt-1 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <span className="tabular">{shortSha(result.sha256)}</span>
            <Copy className="size-3" />
            {copied && <span className="text-risk-low">copied</span>}
          </button>
        </div>
        <Badge variant={result.from_cache ? "secondary" : "muted"} className="shrink-0">
          {result.from_cache ? "cached result" : "fresh run"}
        </Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs">
          <Metric label="Total" value={ms(timings.total_ms)} />
          <Metric label="Decode" value={ms(timings.decode_ms)} />
          <Metric label="Queued" value={ms(timings.queue_ms)} />
          <Metric label="Analysis" value={ms(timings.analysis_ms)} />
          <Metric label="Process IPC" value={ms(timings.ipc_ms)} />
          <Metric label="Assembly" value={ms(timings.assembly_ms)} />
        </div>

        <div className="flex flex-col gap-1">
          <p className="text-[0.7rem] font-medium uppercase tracking-wider text-muted-foreground">
            Fusion
          </p>
          <p className="text-xs leading-relaxed text-foreground/85">
            Source: <span className="font-medium">{String(fusion.source ?? "unknown")}</span>
            {fusion.calibrated != null && (
              <>
                {" · calibrated "}
                <span className="font-medium">{fusion.calibrated ? "yes" : "no"}</span>
              </>
            )}
            {" · "}
            {Object.keys(contributions).length} contributor{Object.keys(contributions).length === 1 ? "" : "s"}
            {fusion.support != null && <> · support {Number(fusion.support).toFixed(2)}</>}
          </p>
          {Object.keys(contributions).length > 0 && (
            <ul className="mt-1 flex flex-wrap gap-1">
              {Object.entries(contributions)
                .sort((a, b) => b[1] - a[1])
                .map(([name, value]) => (
                  <li key={name}>
                    <Badge variant="outline" className="tabular text-[0.65rem]">
                      {name} {Number(value).toFixed(3)}
                    </Badge>
                  </li>
                ))}
            </ul>
          )}
        </div>

        <Accordion type="single" collapsible>
          <AccordionItem value="image">
            <AccordionTrigger>File facts</AccordionTrigger>
            <AccordionContent>
              <Facts
                rows={[
                  ["Dimensions", `${image.width ?? "?"} × ${image.height ?? "?"}`],
                  ["Format", String(image.mime ?? "—")],
                  ["Encoded size", bytes(Number(image.size_bytes ?? 0))],
                  ["Est. JPEG quality", image.est_jpeg_quality != null ? String(Math.round(Number(image.est_jpeg_quality))) : "n/a"],
                  ["Bits per pixel", image.bpp != null ? Number(image.bpp).toFixed(3) : "n/a"],
                  ["Analysis track scale", image.analysis_scale != null ? `${(Number(image.analysis_scale) * 100).toFixed(0)}%` : "native"],
                  ["Native crop", image.native_box ? JSON.stringify(image.native_box) : "whole frame"],
                ]}
              />
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="provenance">
            <AccordionTrigger>Build & artifacts</AccordionTrigger>
            <AccordionContent>
              <Facts
                rows={[
                  ["App", String(provenance.app ?? "—")],
                  ["Environment", String(provenance.environment ?? "—")],
                  ["Perf profile", String(provenance.perf_profile ?? "—")],
                  ["Fusion artifact", String(provenance.fusion_artifact ?? "expert weights")],
                  ["CNN weights", provenance.cnn_weights ? "present" : "not loaded"],
                  ["CLIP probe", provenance.clip_probe ? "present" : "not loaded"],
                  ["Overlay TTL", provenance.overlay_ttl_s ? `${provenance.overlay_ttl_s}s` : "—"],
                ]}
              />
            </AccordionContent>
          </AccordionItem>
          <AccordionItem value="timing">
            <AccordionTrigger>Per-technique timings</AccordionTrigger>
            <AccordionContent>
              <ul className="flex flex-col gap-0.5 text-[0.7rem]">
                {Object.entries(timings.per_technique_ms ?? {})
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, value]) => (
                    <li key={name} className="flex items-center justify-between gap-3 border-b border-border/40 py-0.5">
                      <span className="font-mono text-muted-foreground">{name}</span>
                      <span className="tabular">{ms(value)}</span>
                    </li>
                  ))}
              </ul>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {result.notes.length > 0 && (
          <div className="flex flex-col gap-1 border-t border-border/60 pt-3">
            <p className="flex items-center gap-1.5 text-[0.7rem] font-medium uppercase tracking-wider text-muted-foreground">
              <Timer className="size-3" /> Engine notes
            </p>
            <ul className="flex flex-col gap-0.5">
              {result.notes.map((note, i) => (
                <li key={`${i}-${note.slice(0, 20)}`} className="text-[0.7rem] leading-snug text-muted-foreground">
                  {note}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="tabular text-xs">{value}</span>
    </span>
  );
}

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="flex flex-col divide-y divide-border/50 text-[0.7rem]">
      {rows.map(([label, value]) => (
        <div key={label} className="flex items-center justify-between gap-3 py-1">
          <dt className="text-muted-foreground">{label}</dt>
          <dd className="tabular break-words text-right">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
