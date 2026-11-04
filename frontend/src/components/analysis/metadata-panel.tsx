"use client";

/**
 * Metadata and Content Credentials.
 *
 * This panel is where the product's privacy claim has to be visible rather than merely true. The
 * engine redacts GPS and serial numbers from what it *displays* (`exif_pii_redacted`), and the
 * analysed copy is deleted with the job — but a user looking at their own camera model wants to know
 * why it is there, so the header says what was withheld and the footnote says how long the file
 * lives.
 *
 * A generator tag (DALL·E, Midjourney, Stable Diffusion, an Adobe firefly ID) is reported as a fact
 * about the file, not as a verdict. A signed C2PA manifest saying "made with generative AI" is a
 * different class of evidence again and is shown as one.
 */

import * as React from "react";
import { Fingerprint, KeyRound, ShieldAlert, Tag } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { cn } from "@/lib/utils";
import type { C2paBlock, MetadataBlock } from "@/lib/types";

const SEVERITY: Record<string, { box: string; label: string }> = {
  strong: { box: "border-risk-high/30 bg-risk-high/10 text-risk-high", label: "strong" },
  moderate: { box: "border-risk-moderate/30 bg-risk-moderate/10 text-risk-moderate", label: "moderate" },
  weak: { box: "border-border bg-muted/40 text-muted-foreground", label: "weak" },
  ok: { box: "border-risk-low/30 bg-risk-low/10 text-risk-low", label: "consistent" },
  info: { box: "border-border bg-muted/40 text-muted-foreground", label: "note" },
};

export function MetadataPanel({
  metadata,
  c2pa,
  className,
}: {
  metadata: MetadataBlock;
  c2pa: C2paBlock;
  className?: string;
}) {
  const exifEntries = Object.entries(metadata.exif ?? {}).filter(([, v]) => v !== null && v !== "");

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <Card>
        <CardHeader className="flex-row items-center gap-2">
          <Fingerprint className="size-4 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <CardTitle>Embedded metadata</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              {metadata.status === "not_applicable"
                ? "This format carries no EXIF block."
                : metadata.present
                  ? `${exifEntries.length} readable field${exifEntries.length === 1 ? "" : "s"}.`
                  : "No EXIF, XMP or PNG text chunks found."}
            </p>
          </div>
          {metadata.pii_redacted && (
            <Badge variant="muted" className="shrink-0">
              <ShieldAlert className="size-3" /> GPS withheld
            </Badge>
          )}
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {(metadata.camera_make || metadata.camera_model) && (
            <p className="text-xs">
              <span className="text-muted-foreground">Capture device: </span>
              <span className="font-medium">
                {[metadata.camera_make, metadata.camera_model].filter(Boolean).join(" ")}
              </span>
            </p>
          )}
          {metadata.software && (
            <p className="text-xs">
              <span className="text-muted-foreground">Software tag: </span>
              <span className="font-mono">{metadata.software}</span>
            </p>
          )}

          {metadata.findings.length > 0 && (
            <ul className="flex flex-col gap-1">
              {metadata.findings.map((finding, i) => {
                const tone = SEVERITY[finding.severity] ?? SEVERITY.info;
                return (
                  <li
                    key={`${finding.code}-${i}`}
                    className={cn(
                      "flex items-start gap-1.5 rounded border px-2 py-1 text-[0.72rem]",
                      tone.box,
                    )}
                  >
                    <ShieldAlert className="mt-0.5 size-3 shrink-0" />
                    <span className="min-w-0 flex-1 leading-snug">{finding.text}</span>
                    <span className="shrink-0 text-[0.62rem] uppercase tracking-wider opacity-70">
                      {tone.label}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}

          {metadata.generator_hits.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <p className="flex items-center gap-1.5 text-[0.7rem] font-medium uppercase tracking-wider text-muted-foreground">
                <Tag className="size-3" /> Generator signatures
              </p>
              <ul className="flex flex-col gap-1">
                {metadata.generator_hits.map((hit, i) => (
                  <li
                    key={`${hit.chunk}-${hit.keyword}-${i}`}
                    className="rounded border border-primary/30 bg-primary/10 px-2 py-1 text-[0.72rem]"
                  >
                    <span className="font-medium text-primary">{hit.keyword}</span>{" "}
                    <span className="text-muted-foreground">
                      embedded in <span className="font-mono">{hit.chunk}</span>
                    </span>
                  </li>
                ))}
              </ul>
              <p className="text-[0.68rem] leading-snug text-muted-foreground">
                A generator tag is a fact about the file, and it can be added or removed by hand. It
                raises the synthetic-image likelihood; on its own it is not proof of anything.
              </p>
            </div>
          )}

          {exifEntries.length > 0 && (
            <Accordion type="single" collapsible>
              <AccordionItem value="exif">
                <AccordionTrigger>EXIF table ({exifEntries.length})</AccordionTrigger>
                <AccordionContent>
                  <dl className="max-h-64 overflow-y-auto divide-y divide-border/60 text-[0.7rem]">
                    {exifEntries.map(([key, value]) => (
                      <div key={key} className="flex items-start justify-between gap-3 py-1">
                        <dt className="shrink-0 font-mono text-muted-foreground">{key}</dt>
                        <dd className="min-w-0 break-words text-right">{truncate(String(value), 120)}</dd>
                      </div>
                    ))}
                  </dl>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center gap-2">
          <KeyRound className="size-4 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <CardTitle>Content Credentials (C2PA)</CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">{c2pa.note || credentialSentence(c2pa)}</p>
          </div>
          <Badge variant={c2pa.present ? "default" : "muted"} className="shrink-0">
            {c2pa.present ? "Manifest present" : "No manifest"}
          </Badge>
        </CardHeader>
        <CardContent className="flex flex-col gap-2 text-xs">
          {c2pa.present && (
            <>
              <Row label="Credential" value={c2pa.credential} />
              {c2pa.issuer && <Row label="Issuer" value={c2pa.issuer} />}
              {c2pa.claim_generator && <Row label="Claim generator" value={c2pa.claim_generator} mono />}
              {c2pa.assertions.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-[0.7rem] uppercase tracking-wider text-muted-foreground">
                    Assertions
                  </span>
                  <ul className="flex flex-wrap gap-1">
                    {c2pa.assertions.map((assertion, i) => (
                      <li key={i}>
                        <Badge variant="outline" className="font-mono text-[0.65rem]">
                          {String(assertion["label"] ?? assertion["uri"] ?? assertion["type"] ?? "assertion")}
                        </Badge>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
          <p className="text-[0.68rem] leading-snug text-muted-foreground">
            Absence proves nothing: most cameras and phones do not sign images yet, and a signed
            manifest can describe an edited image — the credentials list the actions taken, they do
            not vouch for the subject.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <p className="flex items-baseline justify-between gap-3">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className={cn("min-w-0 break-words text-right", mono && "font-mono text-[0.7rem]")}>{value}</span>
    </p>
  );
}

function credentialSentence(c2pa: C2paBlock): string {
  if (c2pa.status === "not_applicable") return "This format cannot carry a C2PA manifest.";
  if (c2pa.present) return `Status: ${c2pa.status}.`;
  return "No C2PA manifest — the file was not signed by its producer.";
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}
