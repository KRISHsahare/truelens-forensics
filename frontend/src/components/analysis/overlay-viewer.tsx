"use client";

/**
 * The evidence viewer.
 *
 * **Why the layers are positioned by hand.** An overlay is not a transparent sticker over the
 * photograph: `write_overlay()` stores what the analyser returned, and the analysers return a
 * *composite* — `composite_overlay(native_rgb, heat, alpha)` — painted on the **native track**,
 * which is the largest 8-aligned crop anchored at the top-left (see `core/pipeline.py`). So the
 * overlay PNG is a few pixels narrower than the photo and already contains the photo in it.
 * Two `object-contain` <img> tags stacked in one box would letterbox each by its own aspect ratio
 * and drift the heatmap off the pixels it describes — a misregistered map over a face is an
 * accusation about the wrong region. The rect below is computed from the two images' true sizes,
 * so the evidence lands where the measurement was taken.
 *
 * Three view modes, because "the overlay is a derived image" has three useful readings: **blend**
 * fades it over the original, **swipe** shows both halves of the same frame at the same scale, and
 * **side by side** keeps them as two separate objects. Zoom and pan transform the wrapper that holds
 * *both* layers, which is what keeps them registered.
 *
 * A `spectrum` overlay is the exception the registry keeps visible: the FFT image has frequency on
 * its axes, not position, so it never composites — it renders alone, labelled with what its axes are.
 */

import * as React from "react";
import { Columns2, Eye, MoveHorizontal, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { fetchOverlay } from "@/lib/api";
import { type CatalogEntry, entryFor } from "@/lib/catalog";
import type { OverlayKind } from "@/lib/types";

export type ViewMode = "blend" | "swipe" | "side";

export interface OverlaySource {
  name: string;
  url: string;
  kind: OverlayKind;
  label: string;
}

export interface OverlayViewerProps {
  /**
   * Object URL of the image in the user's browser, or `null` for a URL import — the server fetched
   * those bytes and they were never in this tab. The overlay still displays; it carries its own
   * copy of the photograph.
   */
  baseUrl: string | null;
  baseAlt: string;
  /** Photograph dimensions, from `result.image`, so placement does not wait on the base <img>. */
  baseSize: { width: number; height: number } | null;
  overlays: OverlaySource[];
  catalog: CatalogEntry[];
  active: string | null;
  onSelect: (name: string | null) => void;
  className?: string;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

interface Transform {
  scale: number;
  x: number;
  y: number;
}

const IDENTITY: Transform = { scale: 1, x: 0, y: 0 };

interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The letterbox rect a `contain` layout would give an image of this aspect inside this box. */
function containRect(boxW: number, boxH: number, imgW: number, imgH: number): Rect {
  if (!boxW || !boxH || !imgW || !imgH) return { left: 0, top: 0, width: boxW, height: boxH };
  const scale = Math.min(boxW / imgW, boxH / imgH);
  const width = imgW * scale;
  const height = imgH * scale;
  return { left: (boxW - width) / 2, top: (boxH - height) / 2, width, height };
}

export function OverlayViewer({
  baseUrl,
  baseAlt,
  baseSize,
  overlays,
  catalog,
  active,
  onSelect,
  className,
}: OverlayViewerProps) {
  const [mode, setMode] = React.useState<ViewMode>("blend");
  const [opacity, setOpacity] = React.useState(1);
  const [split, setSplit] = React.useState(0.5);
  const [transform, setTransform] = React.useState<Transform>(IDENTITY);
  const [blobs, setBlobs] = React.useState<Record<string, string | null>>({});
  const [overlaySize, setOverlaySize] = React.useState<{ w: number; h: number } | null>(null);
  const [box, setBox] = React.useState({ w: 0, h: 0 });
  const frameRef = React.useRef<HTMLDivElement>(null);
  const drag = React.useRef<{ x: number; y: number } | null>(null);

  const selected = overlays.find((o) => o.name === active) ?? null;
  const selectedEntry = selected ? entryFor(catalog, selected.name) : undefined;
  const overlayUrl = selected ? blobs[selected.name] : undefined;
  const failed = selected ? blobs[selected.name] === null : false;
  /** Positional overlays are composited over the photo; a spectrum plot is its own figure. */
  const positional = selected != null && selected.kind !== "spectrum";
  const canBlend = Boolean(baseUrl) && positional;

  React.useEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
        const { width, height } = entry.contentRect;
        setBox({ w: width, h: height });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  // Fetch each overlay as a blob, once per result. A signed URL expires with the job and the
  // response is `no-store`, so pointing <img src> at the API would refetch on every remount and
  // break outright the moment the tab is restored from history.
  React.useEffect(() => {
    if (!overlays.length) return;
    const controller = new AbortController();
    const created: string[] = [];
    let cancelled = false;
    overlays.forEach((overlay) => {
      void fetchOverlay(overlay.url, controller.signal).then((blob) => {
        if (cancelled) return; // No URL was ever created, so the blob needs no revoking.
        if (!blob) {
          setBlobs((prev) => ({ ...prev, [overlay.name]: null }));
          return;
        }
        const url = URL.createObjectURL(blob);
        created.push(url);
        setBlobs((prev) => ({ ...prev, [overlay.name]: url }));
      });
    });
    return () => {
      cancelled = true;
      controller.abort();
      created.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [overlays]);

  React.useEffect(() => {
    setOverlaySize(null);
    setTransform(IDENTITY);
  }, [active]);

  const setZoom = (factor: number) =>
    setTransform((t) => {
      const scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, t.scale * factor));
      // Zooming back out to 1× re-centres: a leftover pan at 1× shows empty gutters.
      return scale === 1 ? IDENTITY : { ...t, scale };
    });

  // Registered natively because React attaches `onWheel` passively at the root, and a passive
  // listener cannot preventDefault — scrolling over the frame would zoom *and* move the page.
  React.useEffect(() => {
    const node = frameRef.current;
    if (!node) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      setZoom(event.deltaY < 0 ? 1.15 : 1 / 1.15);
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (event: React.PointerEvent) => {
    if (transform.scale <= 1) return;
    drag.current = { x: event.clientX - transform.x, y: event.clientY - transform.y };
  };
  const onPointerMove = (event: React.PointerEvent) => {
    if (!drag.current) return;
    const start = drag.current;
    setTransform((t) => ({ ...t, x: event.clientX - start.x, y: event.clientY - start.y }));
  };
  const endDrag = () => {
    drag.current = null;
  };

  const photoRect = baseSize ? containRect(box.w, box.h, baseSize.width, baseSize.height) : null;
  // The overlay covers only the native crop, anchored at the photo's top-left, so its rect is the
  // photo rect scaled by the fraction of the frame the analyser actually looked at.
  const overlayRect: Rect | null =
    photoRect && overlaySize && baseSize
      ? {
          left: photoRect.left,
          top: photoRect.top,
          width: photoRect.width * (overlaySize.w / baseSize.width),
          height: photoRect.height * (overlaySize.h / baseSize.height),
        }
      : photoRect;

  const pan = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;

  const photoLayer = (alpha: number) =>
    baseUrl && (
      /* eslint-disable-next-line @next/next/no-img-element -- an object URL of the user's own
         file is exactly the input next/image's optimiser cannot use. */
      <img
        src={baseUrl}
        alt={baseAlt}
        draggable={false}
        className="absolute inset-0 size-full object-contain select-none"
        style={{ opacity: alpha }}
      />
    );

  // The stored overlay is a composite that already contains the photograph at 45–55%, so fading it
  // over the base layer crossfades the *evidence* rather than darkening the picture: at α=0 the
  // reader has the untouched original, at α=1 the analyser's own blend.
  const overlayLayer = (rect: Rect, standalone: boolean, alpha = 1) =>
    overlayUrl && (
      <img
        src={overlayUrl}
        alt={`${selected?.label ?? "Evidence"} map`}
        draggable={false}
        onLoad={(event) => {
          const img = event.currentTarget;
          if (img.naturalWidth && (img.naturalWidth !== overlaySize?.w || img.naturalHeight !== overlaySize?.h)) {
            setOverlaySize({ w: img.naturalWidth, h: img.naturalHeight });
          }
        }}
        className={cn("absolute select-none", standalone ? "size-full object-contain" : "pixels")}
        style={
          standalone
            ? { left: 0, top: 0, width: "100%", height: "100%", opacity: alpha }
            : { left: rect.left, top: rect.top, width: rect.width, height: rect.height, opacity: alpha }
        }
      />
    );

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex flex-wrap items-center gap-1.5">
        <div className="flex items-center gap-0.5 rounded-md border border-border/70 p-0.5">
          {(
            [
              { key: "blend" as const, icon: <Eye className="size-3.5" />, label: "Blend", needsPhoto: true },
              { key: "swipe" as const, icon: <MoveHorizontal className="size-3.5" />, label: "Swipe", needsPhoto: true },
              { key: "side" as const, icon: <Columns2 className="size-3.5" />, label: "Side by side", needsPhoto: false },
            ] as const
          ).map(({ key, icon, label, needsPhoto }) => {
            const disabled = needsPhoto && !baseUrl;
            return (
              <button
                key={key}
                type="button"
                onClick={() => !disabled && setMode(key)}
                aria-pressed={mode === key}
                disabled={disabled}
                title={disabled ? "The original bytes are not in this browser" : label}
                className={cn(
                  "inline-flex items-center gap-1 rounded px-2 py-1 text-[0.7rem] font-medium transition-colors",
                  mode === key ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
                  disabled && "cursor-not-allowed opacity-40",
                )}
              >
                {icon}
                <span className="hidden sm:inline">{label}</span>
              </button>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-0.5">
          <Button variant="ghost" size="icon" className="size-7" onClick={() => setZoom(1 / 1.3)} aria-label="Zoom out" disabled={transform.scale <= MIN_ZOOM}>
            <ZoomOut className="size-3.5" />
          </Button>
          <span className="tabular w-10 text-center text-[0.7rem] text-muted-foreground">
            {transform.scale.toFixed(1)}×
          </span>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => setZoom(1.3)} aria-label="Zoom in" disabled={transform.scale >= MAX_ZOOM}>
            <ZoomIn className="size-3.5" />
          </Button>
          <Button variant="ghost" size="icon" className="size-7" onClick={() => setTransform(IDENTITY)} aria-label="Reset zoom and pan">
            <RotateCcw className="size-3.5" />
          </Button>
        </div>
      </div>

      <div
        ref={frameRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        className={cn(
          "lab-grid relative aspect-[4/3] w-full touch-none overflow-hidden rounded-lg border border-border",
          transform.scale > 1 ? "cursor-grab active:cursor-grabbing" : "cursor-default",
        )}
      >
        {!selected &&
          (baseUrl ? (
            photoLayer(1)
          ) : (
            <Notice>
              The original bytes stayed on the server for a URL import — pick an overlay to see the
              evidence map, which carries its own copy of the frame.
            </Notice>
          ))}

        {selected && mode === "side" && (
          <div className="grid size-full grid-cols-2 gap-px bg-border">
            <div className="relative overflow-hidden">
              <div className="absolute inset-0" style={{ transform: pan, transformOrigin: "center" }}>
                {photoLayer(1)}
              </div>
              {!baseUrl && <Notice small>Original not in this browser</Notice>}
              <Tag className="left-1.5 top-1.5">Original</Tag>
            </div>
            <div className="relative overflow-hidden">
              <div className="absolute inset-0" style={{ transform: pan, transformOrigin: "center" }}>
                {overlayUrl && overlayRect
                  ? overlayLayer(overlayRect, !positional)
                  : failed
                    ? <Notice small>Expired</Notice>
                    : <Notice small>Loading…</Notice>}
              </div>
              <Tag className="left-1.5 top-1.5">{selected.label}</Tag>
            </div>
          </div>
        )}

        {selected && mode === "swipe" && (
          <div className="relative size-full">
            <div className="absolute inset-0" style={{ transform: pan, transformOrigin: "center" }}>
              {photoLayer(1)}
              {overlayRect && overlayLayer(overlayRect, !positional)}
            </div>
            {/* The left half is the original, revealed by clipping a second copy of the base layer
                — same pixels, same transform, so the seam is a straight line through the frame. */}
            {baseUrl && (
              <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - split * 100}% 0 0)` }}>
                <div className="absolute inset-0" style={{ transform: pan, transformOrigin: "center" }}>
                  {photoLayer(1)}
                </div>
              </div>
            )}
            <div className="pointer-events-none absolute inset-y-0 w-px bg-primary/80" style={{ left: `${split * 100}%` }} />
            <input
              type="range"
              min={2}
              max={98}
              step={1}
              value={Math.round(split * 100)}
              onChange={(e) => setSplit(Number(e.target.value) / 100)}
              aria-label="Swipe position between original and evidence"
              className="absolute inset-x-0 bottom-2 mx-auto w-2/3 accent-[var(--primary)]"
            />
          </div>
        )}

        {selected && mode === "blend" && (
          <div className="relative size-full">
            {!positional && (
              <p className="absolute inset-x-0 top-0 z-10 bg-background/85 px-3 py-1.5 text-[0.7rem] text-muted-foreground">
                {selected.kind === "spectrum"
                  ? "This is the frequency spectrum, not a map over the photo: both axes are frequency, and the bright cross is the DC component."
                  : "This check reports numbers, not a picture."}
              </p>
            )}
            <div className="absolute inset-0" style={{ transform: pan, transformOrigin: "center" }}>
              {canBlend ? (
                <>
                  {photoLayer(1)}
                  {overlayRect && overlayLayer(overlayRect, false, opacity)}
                </>
              ) : (
                overlayUrl && overlayLayer(overlayRect ?? { left: 0, top: 0, width: box.w, height: box.h }, true)
              )}
            </div>
            {failed && <Notice small>Link expired</Notice>}
          </div>
        )}
      </div>

      {selected && positional && (
        <div className="flex items-center gap-3">
          <label htmlFor="overlay-opacity" className="text-[0.7rem] font-medium uppercase tracking-wider text-muted-foreground">
            Evidence
          </label>
          <Slider
            id="overlay-opacity"
            min={0}
            max={1}
            step={0.05}
            value={[opacity]}
            onValueChange={([v]) => setOpacity(v)}
            className="flex-1"
            disabled={!canBlend}
          />
          <span className="tabular w-9 text-right text-[0.7rem] text-muted-foreground">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Choose an evidence overlay">
        <Chip on={active === null} onClick={() => onSelect(null)}>
          None
        </Chip>
        {overlays.map((overlay) => (
          <Chip key={overlay.name} on={active === overlay.name} onClick={() => onSelect(overlay.name)}>
            {overlay.label}
            <span className="ml-1 text-[0.6rem] uppercase opacity-70">{overlay.kind}</span>
          </Chip>
        ))}
      </div>

      {selectedEntry?.summary && (
        <p className="text-[0.7rem] leading-snug text-muted-foreground">{selectedEntry.summary}</p>
      )}
    </div>
  );
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={cn(
        "rounded-full border px-2.5 py-1 text-[0.7rem] transition-colors",
        on ? "border-primary bg-primary/15 text-primary" : "border-border text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function Tag({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <span className={cn("absolute rounded bg-background/85 px-1.5 py-0.5 text-[0.65rem] text-muted-foreground", className)}>
      {children}
    </span>
  );
}

function Notice({ children, small = false }: { children: React.ReactNode; small?: boolean }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center p-4 text-center">
      <p className={cn("max-w-xs leading-snug text-muted-foreground", small ? "text-[0.7rem]" : "text-xs")}>
        {children}
      </p>
    </div>
  );
}
