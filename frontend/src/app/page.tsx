"use client";

/**
 * The landing page.
 *
 * The drop zone is on the hero, not behind a "try it" link, because the first question a visitor has
 * is "will it work on *my* image" and the only answer that counts is to let them drop it. Dropping
 * navigates to the Analyzer with the file in hand, so there is exactly one upload path in the app —
 * this page does not run an analysis of its own.
 *
 * The copy is deliberately hedged. A tool like this earns trust by stating its failure modes before
 * being asked, so limitations get their own section above the FAQ rather than a footnote below it.
 */

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Check,
  Eye,
  GitCompare,
  Lock,
  Scan,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ACCEPT_ATTRIBUTE, checkFile } from "@/lib/api";
import { handoff } from "@/lib/handoff";
import { watchClipboardImage } from "@/lib/paste";
import { useCatalog } from "@/lib/catalog";
import { GROUP_LABELS } from "@/lib/format";
import { cn } from "@/lib/utils";

const STEPS = [
  {
    title: "Drop an image",
    body: "JPEG, PNG, WebP or BMP up to 10 MB — a file, a clipboard paste, or a URL the server fetches. The extension is ignored; the bytes are checked.",
    icon: <Upload className="size-4" />,
  },
  {
    title: "Ten checks run in parallel",
    body: "Three of them look at frequency and learned features for signs of generation. Six look for regions whose compression, noise, grid phase or lighting disagrees with the rest of the frame. One reads what the file says about itself.",
    icon: <Scan className="size-4" />,
  },
  {
    title: "Read the evidence, not the verdict",
    body: "Two likelihoods, a reliability figure, and a heatmap you can fade over the original. Every number opens to the measurement that produced it.",
    icon: <Eye className="size-4" />,
  },
];

const TRUST = [
  {
    title: "Nothing is stored",
    body: "Uploads are decoded in memory, overlays are deleted with the job, and the job ages out in ten minutes. History lives in your browser's IndexedDB, not in a database.",
    icon: <Lock className="size-4" />,
  },
  {
    title: "Probability, not proof",
    body: "The output is a likelihood under a model, calibrated on a specific dataset. It is not evidence about a person, and no score here should be the only thing standing behind a decision.",
    icon: <ShieldCheck className="size-4" />,
  },
  {
    title: "The method is inspectable",
    body: "Every threshold, weight and metric on the Methodology page is the one the server actually used. If a check could not run on your file, its card stays visible and grey.",
    icon: <GitCompare className="size-4" />,
  },
];

const LIMITATIONS = [
  "A sufficiently good generator or a skilled editor with a clean pipeline can beat every check here. Detection is an arms race and the honest answer about any single image is the evidence, not the number.",
  "Screenshots, re-uploads and messenger compressions destroy the artefacts several checks rely on. A WhatsApp forward can look authentic because its history was overwritten, not because it is real.",
  "Small faces and heavy blur carry almost no high-frequency signal, so face-consistency and noise checks measure less than they appear to.",
  "Deepfakes of video are out of scope: this analyses a single still frame, with no temporal signal to work from.",
  "Metrics on the Methodology page come from the datasets the models were trained and evaluated on. A distribution your images do not share is a distribution those numbers do not describe.",
];

const FAQ = [
  {
    q: "Does a high score mean the image is fake?",
    a: "It means the measured signals resemble those the classifier learned from generated images. Text overlays, heavy sharpening, a low-quality phone sensor and a screen photograph can all raise it. Read the technique cards before the headline.",
  },
  {
    q: "Does a low score mean it is authentic?",
    a: "It means no check found a signal worth reporting on this file. Absence of evidence is a property of the method as much as of the image — and re-encoding through a platform can remove the evidence entirely.",
  },
  {
    q: "What is reliability, and why is it not the same as confidence?",
    a: "Reliability answers how many of the checks that could have spoken, did. An 0.8 likelihood measured by one working analyser and an 0.8 measured by nine can look identical on the gauge; the reliability figure is what tells them apart.",
  },
  {
    q: "Why do the overlay images stop loading after a while?",
    a: "Overlay links are signed and expire with the job, and the pixels behind them are deleted with it. Re-run the analysis to regenerate them. A history entry keeps every number and no heatmaps, on purpose.",
  },
  {
    q: "Do you keep the image or my address?",
    a: "Neither is persisted. The engine rate-limits per address in memory and forgets it when the process restarts; there is no analytics layer touching image content.",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const catalog = useCatalog();
  const [error, setError] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const send = React.useCallback((file: File | Blob | undefined, name?: string) => {
    if (!file) return;
    const label = name ?? (file instanceof File ? file.name : "clipboard-image");
    const check = checkFile(file, label);
    if (!check.ok) {
      setError(check.reason ?? "That file cannot be analysed.");
      return;
    }
    setError(null);
    handoff.set({ file, filename: label });
    router.push("/analyze");
  }, [router]);

  React.useEffect(() => watchClipboardImage(send), [send]);

  return (
    <div className="lab-vignette">
      <section className="mx-auto grid w-full max-w-7xl gap-10 px-4 pb-8 pt-12 sm:px-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:pt-20">
        <div className="flex flex-col justify-center gap-5">
          <Badge variant="secondary" className="w-fit">
            <Scan className="size-3" /> ten checks · two likelihoods · one probability
          </Badge>
          <h1 className="text-balance text-3xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
            Is this image what it claims to be?
          </h1>
          <p className="max-w-xl text-pretty text-sm leading-relaxed text-muted-foreground sm:text-base">
            TrueLens runs a set of independent forensic analysers over one file and shows you where
            the evidence sits — frequency artefacts, compression history, noise and block
            alignment, copy-move matches, face consistency, embedded provenance and a learned
            detector. Then it says, in plain language, what it cannot tell.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Button asChild size="lg">
              <Link href="/analyze">
                Open the analyzer <ArrowRight className="size-4" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/methodology">Model &amp; methodology</Link>
            </Button>
          </div>
          <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.7rem] text-muted-foreground">
            <span>No account</span>
            <span aria-hidden>·</span>
            <span>No server-side storage</span>
            <span aria-hidden>·</span>
            <span>Runs locally, CPU only</span>
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
        >
          <HeroDropzone onFile={send} inputRef={inputRef} error={error} />
        </motion.div>
      </section>

      <Section heading="How it works" sub="Three steps, and the second one is where the work actually is.">
        <ol className="grid gap-3 sm:grid-cols-3">
          {STEPS.map((step, i) => (
            <li key={step.title}>
              <Card className="h-full">
                <CardHeader className="flex-row items-start gap-2">
                  <span className="tabular text-xs text-muted-foreground">0{i + 1}</span>
                  <span className="inline-flex size-7 items-center justify-center rounded border border-border bg-secondary text-primary">
                    {step.icon}
                  </span>
                  <CardTitle className="flex-1 text-sm leading-snug">{step.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs leading-relaxed text-muted-foreground">{step.body}</p>
                </CardContent>
              </Card>
            </li>
          ))}
        </ol>
      </Section>

      <Section
        heading="The technique grid"
        sub={`${catalog.entries.length} analysers, grouped by the sub-score they feed. Each one returns a score, a picture, and the numbers behind it.`}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {catalog.entries.map((entry) => (
            <Card key={entry.name} className="h-full">
              <CardHeader className="gap-1">
                <div className="flex items-center gap-2">
                  <CardTitle className="flex-1 text-sm">{entry.label}</CardTitle>
                  {entry.heavy && <Badge variant="muted" className="shrink-0">torch</Badge>}
                </div>
                <p className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                  {GROUP_LABELS[entry.group] ?? entry.group}
                  {entry.overlay !== "none" && <> · {entry.overlay} overlay</>}
                </p>
              </CardHeader>
              <CardContent>
                <p className="text-xs leading-relaxed text-muted-foreground">{entry.summary}</p>
              </CardContent>
            </Card>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          {catalog.source === "api"
            ? "This list came from the running engine."
            : "Showing the built-in list — the engine is not answering, so the checks named here may differ from the ones that run."}
        </p>
      </Section>

      <Section heading="Trust, stated plainly" sub="What the product promises, and what it refuses to promise.">
        <div className="grid gap-3 sm:grid-cols-3">
          {TRUST.map((item) => (
            <Card key={item.title}>
              <CardHeader className="flex-row items-start gap-2">
                <span className="inline-flex size-7 items-center justify-center rounded border border-border bg-secondary text-primary">
                  {item.icon}
                </span>
                <CardTitle className="text-sm">{item.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs leading-relaxed text-muted-foreground">{item.body}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </Section>

      <Section heading="What this cannot do" sub="Read this before you read a score.">
        <Card className="border-risk-moderate/30">
          <CardContent className="pt-4">
            <ul className="flex flex-col gap-2">
              {LIMITATIONS.map((line) => (
                <li key={line.slice(0, 32)} className="flex gap-2 text-xs leading-relaxed">
                  <Check className="mt-0.5 size-3.5 shrink-0 text-risk-moderate" aria-hidden />
                  <span className="text-muted-foreground">{line}</span>
                </li>
              ))}
            </ul>
            <p className="mt-3">
              <Link href="/about#limitations" className="text-xs text-primary underline-offset-4 hover:underline">
                Full limitations and privacy statement
              </Link>
            </p>
          </CardContent>
        </Card>
      </Section>

      <Section heading="Questions">
        <Card>
          <CardContent className="pt-4">
            <Accordion type="single" collapsible className="w-full">
              {FAQ.map((item, i) => (
                <AccordionItem key={item.q} value={`faq-${i}`}>
                  <AccordionTrigger>{item.q}</AccordionTrigger>
                  <AccordionContent>
                    <p className="text-[0.75rem] leading-relaxed text-muted-foreground">{item.a}</p>
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      </Section>

      <section className="mx-auto w-full max-w-7xl px-4 pb-4 sm:px-6">
        <Card className="lab-grid">
          <CardContent className="flex flex-col items-center gap-3 py-8 text-center">
            <h2 className="text-lg font-semibold tracking-tight">Put one in</h2>
            <p className="max-w-md text-xs text-muted-foreground">
              A screenshot, a product photo, something a generator made. The upload path is the same
              as the one above, and it takes about as long to say nothing about the file as it takes
              to say something.
            </p>
            <Button asChild>
              <Link href="/analyze">
                Start an analysis <ArrowRight className="size-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function Section({
  heading,
  sub,
  children,
}: {
  heading: string;
  sub?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">
      <div className="mb-4 flex flex-col gap-1">
        <h2 className="text-lg font-semibold tracking-tight">{heading}</h2>
        {sub && <p className="max-w-3xl text-xs text-muted-foreground sm:text-sm">{sub}</p>}
      </div>
      {children}
    </section>
  );
}

function HeroDropzone({
  onFile,
  inputRef,
  error,
}: {
  onFile: (file: File | Blob | undefined, name?: string) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  error: string | null;
}) {
  const [dragging, setDragging] = React.useState(false);

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={() => setDragging(true)}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        onFile(e.dataTransfer.files?.[0]);
      }}
      className={cn(
        "relative overflow-hidden rounded-xl border border-dashed p-6 transition-colors",
        dragging ? "border-primary bg-primary/10" : "border-border bg-card/60",
      )}
    >
      {dragging && (
        <span aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-px animate-scan bg-primary/70" />
      )}
      <div className="flex flex-col items-center gap-3 text-center">
        <span className="inline-flex size-12 items-center justify-center rounded-full border border-border bg-secondary text-primary">
          <Upload className="size-5" />
        </span>
        <div>
          <p className="text-sm font-medium">{dragging ? "Release to start" : "Drop an image here"}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            or paste with <kbd className="rounded border border-border bg-muted px-1 font-mono text-[0.65rem]">Ctrl+V</kbd>
          </p>
        </div>
        <Button size="sm" variant="secondary" onClick={() => inputRef.current?.click()}>
          Choose file
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTRIBUTE}
          className="sr-only"
          aria-label="Choose an image"
          onChange={(e) => {
            const file = e.target.files?.[0];
            onFile(file, file?.name);
            e.target.value = "";
          }}
        />
        <p
          role="status"
          aria-live="polite"
          className={cn("min-h-4 text-[0.7rem]", error ? "text-risk-high" : "text-muted-foreground")}
        >
          {error ?? "JPEG · PNG · WebP · BMP, up to 10 MB"}
        </p>
      </div>
    </div>
  );
}
