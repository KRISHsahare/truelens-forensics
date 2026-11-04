/**
 * Display arithmetic for a forensic score.
 *
 * The bands come from the server (`band` on every technique and on the result), because the
 * ceilings — 0.35 and 0.60 — are the same constants the verdict engine uses, and a second copy in
 * the frontend would drift. What lives here is only the *rendering* of a band: which colour, which
 * word, and what an unmeasured score looks like.
 *
 * The naming rule is a product rule, not a style rule: the UI never says "fake" or "real". A
 * likelihood is being reported, and the copy has to survive being read aloud to a lawyer.
 */

import type { RiskBand, VerdictLabel } from "./types";

/** Same thresholds as `app.analysis.explainers` — kept in one place here so a chart and a card agree. */
export const LOW_CEILING = 0.35;
export const MODERATE_CEILING = 0.6;

export function bandFor(score: number | null | undefined): RiskBand {
  if (score == null) return "not_measured";
  if (score < LOW_CEILING) return "low";
  if (score < MODERATE_CEILING) return "moderate";
  return "high";
}

/**
 * Tailwind class triples per band.
 *
 * `not_measured` is graphite, not green: a check that could not run must not read as a pass, and
 * it must not read as an accusation either. Grey is the only honest colour for "no signal".
 */
export const BAND_CLASSES: Record<RiskBand, { text: string; bg: string; ring: string; bar: string }> = {
  low: {
    text: "text-risk-low",
    bg: "bg-risk-low/12",
    ring: "ring-risk-low/35",
    bar: "bg-risk-low",
  },
  moderate: {
    text: "text-risk-moderate",
    bg: "bg-risk-moderate/12",
    ring: "ring-risk-moderate/35",
    bar: "bg-risk-moderate",
  },
  high: {
    text: "text-risk-high",
    bg: "bg-risk-high/12",
    ring: "ring-risk-high/35",
    bar: "bg-risk-high",
  },
  not_measured: {
    text: "text-muted-foreground",
    bg: "bg-muted/40",
    ring: "ring-border",
    bar: "bg-muted-foreground/45",
  },
};

/** Raw colours for SVG stroke/fill and Recharts, which cannot take Tailwind classes. */
export const BAND_COLORS: Record<RiskBand, string> = {
  low: "#3ddc97",
  moderate: "#f5b544",
  high: "#f4556a",
  not_measured: "#6b7a8c",
};

export function bandColor(band: RiskBand): string {
  return BAND_COLORS[band] ?? BAND_COLORS.not_measured;
}

export function colorFor(score: number | null | undefined): string {
  return bandColor(bandFor(score));
}

export function pct(score: number | null | undefined, digits = 0): string {
  if (score == null || Number.isNaN(score)) return "—";
  return `${(score * 100).toFixed(digits)}%`;
}

/** "0.712" — the panel shows raw probabilities next to the percentage, because rounding hides margin. */
export function prob(score: number | null | undefined, digits = 3): string {
  if (score == null || Number.isNaN(score)) return "not measured";
  return score.toFixed(digits);
}

export const VERDICT_LABELS: Record<VerdictLabel, string> = {
  authentic: "No manipulation found",
  ai_generated: "Likely AI-generated",
  edited: "Likely manipulated",
  inconclusive: "Inconclusive",
};

export const VERDICT_SHORT: Record<VerdictLabel, string> = {
  authentic: "Authentic",
  ai_generated: "AI-generated",
  edited: "Edited",
  inconclusive: "Inconclusive",
};

/** The verdict band is a *claim*, so the colour is deliberately not red for `ai_generated`. */
export function verdictBand(verdict: VerdictLabel, overall: number): RiskBand {
  if (verdict === "inconclusive") return "not_measured";
  if (verdict === "authentic") return "low";
  return bandFor(overall);
}

export function shortSha(sha: string | null | undefined, head = 10, tail = 6): string {
  if (!sha) return "—";
  if (sha.length <= head + tail + 1) return sha;
  return `${sha.slice(0, head)}…${sha.slice(-tail)}`;
}

export function bytes(size: number | null | undefined): string {
  if (size == null) return "—";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`;
  return `${(size / 1024 / 1024).toFixed(2)} MB`;
}

export function ms(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

export const STAGE_LABELS: Record<string, string> = {
  pending: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
  skipped: "Skipped",
  blocked: "Blocked",
};

export const GROUP_LABELS: Record<string, string> = {
  ai: "Synthetic-image signals",
  edited: "Manipulation signals",
  context: "Provenance & metadata",
};

/** Weight copy for a technique's explanation, straight from the engine's four-level scale. */
export const WEIGHT_LABELS: Record<string, string> = {
  strong: "Strong evidence",
  supporting: "Supporting evidence",
  weak: "Weak signal",
  unavailable: "Nothing to measure",
};

export function timeAgo(epochSeconds: number): string {
  const delta = Date.now() / 1000 - epochSeconds;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`;
  return new Date(epochSeconds * 1000).toLocaleDateString();
}
