/**
 * The technique catalogue the UI renders before a result exists.
 *
 * `/api/metrics` is the authority — `routes_metrics.py` builds that list from the backend's own
 * registry, so a new analyser appears in the stepper without a frontend change. The table below is
 * only the shape used *until* that request answers: it exists so the progress stepper can show ten
 * named rows in the first 300 ms rather than ten spinner dots, and so the page still reads properly
 * when the API is down (which is precisely when a user most wants to know what was supposed to run).
 *
 * Order matches `TECHNIQUES` in `app/analysis/registry.py`: cheap universal checks first, the
 * heavyweight learned detector last.
 */

import { useEffect, useState } from "react";
import type { OverlayKind } from "./types";
import { getMetrics } from "./api";

export type TechniqueGroup = "ai" | "edited" | "context";

export interface CatalogEntry {
  name: string;
  label: string;
  group: TechniqueGroup;
  overlay: OverlayKind;
  summary: string;
  heavy: boolean;
}

export const FALLBACK_CATALOG: CatalogEntry[] = [
  {
    name: "metadata",
    label: "Metadata & Provenance",
    group: "ai",
    overlay: "none",
    heavy: false,
    summary: "EXIF, XMP and PNG text chunks — camera provenance and the parameter blocks generators leave behind.",
  },
  {
    name: "c2pa",
    label: "Content Credentials (C2PA)",
    group: "ai",
    overlay: "none",
    heavy: false,
    summary: "Verifies a cryptographically signed C2PA manifest, if the file carries one.",
  },
  {
    name: "ela",
    label: "Error Level Analysis",
    group: "edited",
    overlay: "heatmap",
    heavy: false,
    summary: "Re-saves at the file's own JPEG quality and looks for regions whose quantisation error disagrees.",
  },
  {
    name: "jpeg_ghost",
    label: "JPEG Ghost",
    group: "edited",
    overlay: "heatmap",
    heavy: false,
    summary: "DCT-coefficient striation showing regions with a second, different compression history.",
  },
  {
    name: "block_grid",
    label: "Block-Grid Alignment",
    group: "edited",
    overlay: "heatmap",
    heavy: false,
    summary: "Whether every part of the image shares one 8×8 JPEG block phase.",
  },
  {
    name: "noise",
    label: "Noise Inconsistency",
    group: "edited",
    overlay: "heatmap",
    heavy: false,
    summary: "Per-block sensor noise floor; a spliced region carries a different camera's noise.",
  },
  {
    name: "copy_move",
    label: "Copy-Move Cloning",
    group: "edited",
    overlay: "vector",
    heavy: false,
    summary: "SIFT keypoint self-matching with displacement voting to find cloned regions.",
  },
  {
    name: "face",
    label: "Face Consistency",
    group: "edited",
    overlay: "vector",
    heavy: false,
    summary: "Faces and landmarks, then blending boundaries, lighting symmetry and specular consistency.",
  },
  {
    name: "frequency",
    label: "Frequency Spectrum",
    group: "ai",
    overlay: "spectrum",
    heavy: false,
    summary: "Amplitude-spectrum decay plus periodicity from checkerboard upsampling.",
  },
  {
    name: "cnn",
    label: "Learned Detector",
    group: "ai",
    overlay: "heatmap",
    heavy: true,
    summary: "Fine-tuned EfficientNet-B0, optionally corroborated by a CLIP linear probe, with Grad-CAM.",
  },
];

export interface CatalogState {
  entries: CatalogEntry[];
  /** Which catalogue the UI is actually showing, for the "engine reports N checks" line. */
  source: "fallback" | "api";
  loading: boolean;
}

let cached: CatalogEntry[] | null = null;

/**
 * One request per page load, shared.
 *
 * The catalogue is static for the life of the server process, so a module-level cache beats every
 * page re-fetching it — and `/api/metrics` is also where Methodology reads its numbers, so the two
 * pages agree on what the ten checks are by construction.
 */
export function useCatalog(): CatalogState {
  const [entries, setEntries] = useState<CatalogEntry[]>(cached ?? FALLBACK_CATALOG);
  const [source, setSource] = useState<CatalogState["source"]>(cached ? "api" : "fallback");
  const [loading, setLoading] = useState(cached === null);

  useEffect(() => {
    if (cached) return;
    let cancelled = false;
    getMetrics()
      .then((metrics) => {
        if (cancelled) return;
        const next: CatalogEntry[] = metrics.techniques.map((t) => ({
          name: t.name,
          label: t.label,
          group: t.group as TechniqueGroup,
          overlay: t.overlay,
          summary: t.summary,
          heavy: t.heavy,
        }));
        if (next.length) {
          cached = next;
          setEntries(next);
          setSource("api");
        }
      })
      .catch(() => {
        // The fallback table stays on screen. A metrics outage is not a reason to show no checks.
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  return { entries, source, loading };
}

export function entryFor(entries: CatalogEntry[], name: string): CatalogEntry | undefined {
  return entries.find((e) => e.name === name);
}

/** `copy_move` → "Copy move". Used where a card has no catalog label to lean on. */
export function humanize(name: string): string {
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}
