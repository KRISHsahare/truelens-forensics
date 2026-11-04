import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

/**
 * About / limitations / privacy.
 *
 * A server component: nothing here is interactive, and a page whose subject is honesty about
 * limitations should render without JavaScript. The content is written to be read by someone
 * deciding whether to rely on a number from this tool, which is a different audience than the
 * README and a different register than the Analyzer.
 */

const METHOD = [
  [
    "Two likelihoods, not one verdict",
    "The engine separates “produced by a generator” from “a real photograph that was altered”. A single blended score cannot express that a file looks ungenerated and edited at once, and averaging the two would hide exactly that case.",
  ],
  [
    "Ten independent analysers",
    "Each returns a score between zero and one, a picture, and a details dict. They run in parallel across a process pool on three resolution tracks — the native unresampled crop for compression checks, a ≤2048 px working image for learned and spatial checks, and the raw container for metadata.",
  ],
  [
    "Fusion, then rules",
    "Sub-scores come from a fusion step over the measured signals; deterministic facts — an identified generator tag, a signed C2PA generative manifest — act as evidence flags that can floor a score. A flag that moved a number is always named on screen, so an adjusted value is never presented as a raw model output.",
  ],
  [
    "Reliability is a separate figure",
    "It measures how many of the checks that could have spoken on this file actually produced a measurement, and applies multiplicative penalties for timeouts, unavailable models and missing artefacts. A high likelihood measured by one working analyser is not the same claim as one measured by nine.",
  ],
];

const LIMITATIONS = [
  [
    "This is not proof",
    "Every number is a probability under a model trained on particular datasets. It is not evidence about a person, it will not survive cross-examination, and it should never be the only thing standing behind a decision that affects someone.",
  ],
  [
    "Detectors are behind generators",
    "Any sufficiently capable generator, or a human editor working from clean source material with a careful pipeline, can defeat the statistical checks here. Detection is an arms race; the published metrics are a snapshot of one round of it.",
  ],
  [
    "Re-encoding destroys evidence",
    "Social platforms and messengers resize, recompress and strip metadata. Several checks depend on artefacts that a WhatsApp forward erases, so a re-compressed image can read as authentic because its history was overwritten — not because it is real.",
  ],
  [
    "Small and soft subjects measure less",
    "Faces occupying a few dozen pixels, heavy blur, shallow depth of field and low-texture scenes carry little high-frequency signal. Noise, ELA, JPEG-ghost and face-consistency checks all degrade there, and reliability drops to say so.",
  ],
  [
    "Still images only",
    "Video deepfakes, audio clones and generated text are out of scope. A single frame has no temporal signal, and the checks here are built entirely on spatial and frequency evidence.",
  ],
  [
    "Distribution shift",
    "Metrics on the Methodology page come from the training and evaluation corpora. A file from a domain those corpora never covered — a medical scan, an anime frame, a satellite tile, a document scan — is being measured against a reference it does not share.",
  ],
];

const PRIVACY = [
  [
    "What is uploaded",
    "The image bytes, and nothing else. A filename is used only as a transient label inside the job record so a batch row can be matched to a file; it is never written to disk and never returned in a stored result.",
  ],
  [
    "How long anything exists",
    "The decoded arrays live in process memory for the duration of the analysis. Overlay PNGs are written to a scratch directory and deleted with the job; the job record ages out of the store within ten minutes by default. The result cache is keyed by SHA-256 and holds scores, not pixels.",
  ],
  [
    "What is not stored",
    "No image, no filename, no account, no cookies, no analytics on image content. History is written to IndexedDB in your own browser and can be cleared on the History page or by clearing site data.",
  ],
  [
    "What is logged",
    "Structured operational logs: request counts, job timings, rejection reasons and rate-limit events keyed by a hashed address for the lifetime of the process. GPS coordinates and serial numbers found in EXIF are redacted before anything is displayed.",
  ],
  [
    "Links to the evidence",
    "Overlay URLs are HMAC-signed with an expiry matching the job's lifetime, and served with `no-store`. An expired link and a non-existent one return the same 404, so a guesser cannot learn which jobs existed.",
  ],
];

export default function AboutPage() {
  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-10 sm:px-6">
      <header className="flex flex-col gap-3">
        <Badge variant="secondary" className="w-fit">about the method</Badge>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          What TrueLens measures, and what it does not
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          This page exists because a forensic tool that cannot describe its own failure modes is not
          a forensic tool. Read the limitations section before you read a score.
        </p>
      </header>

      <Section id="method" title="How a result is produced">
        <div className="grid gap-3 sm:grid-cols-2">
          {METHOD.map(([title, body]) => (
            <Card key={title}>
              <CardHeader>
                <CardTitle className="text-sm">{title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs leading-relaxed text-muted-foreground">{body}</p>
              </CardContent>
            </Card>
          ))}
        </div>
        <p className="mt-3 text-xs">
          <Link href="/methodology" className="text-primary underline-offset-4 hover:underline">
            Per-technique detail, thresholds and published metrics →
          </Link>
        </p>
      </Section>

      <Section id="limitations" title="Known limitations">
        <ul className="flex flex-col gap-2">
          {LIMITATIONS.map(([title, body]) => (
            <li key={title} className="rounded-lg border border-risk-moderate/30 bg-card p-3">
              <p className="text-sm font-medium">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</p>
            </li>
          ))}
        </ul>
      </Section>

      <Section id="privacy" title="Privacy">
        <div className="flex flex-col gap-2">
          {PRIVACY.map(([title, body]) => (
            <div key={title} className="rounded-lg border border-border bg-card p-3">
              <p className="text-sm font-medium">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</p>
            </div>
          ))}
        </div>
        <p className="mt-4 rounded-lg border border-border bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
          Self-hosting is the version of this promise you can verify. The backend runs in one
          container with no outbound calls, no database and no telemetry;{" "}
          <code className="tabular rounded bg-background px-1">docs/</code> and the README carry the
          compose file.
        </p>
      </Section>

      <Section id="name" title="Why the copy is careful">
        <p className="text-xs leading-relaxed text-muted-foreground">
          The interface never says “fake” or “real”. It says a likelihood was measured, names the
          measurement, and shows the picture it came from. That is not hedging for legal comfort —
          it is the accurate description of what a set of statistical tests over one file can
          support.
        </p>
      </Section>
    </div>
  );
}

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mt-10 scroll-mt-20">
      <h2 className="mb-3 text-lg font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}
