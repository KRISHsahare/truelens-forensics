"use client";

import * as React from "react";
import { ImagePlus, Link2, Loader2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ACCEPT_ATTRIBUTE, checkFile } from "@/lib/api";
import { watchClipboardImage } from "@/lib/paste";
import { cn } from "@/lib/utils";
import type { AnalysisSource } from "@/hooks/use-analysis";

export interface DropzoneProps {
  onSource: (source: AnalysisSource) => void;
  busy?: boolean;
  /** Shown while `busy` — the label has to say what is happening, not just that input is locked. */
  busyLabel?: string;
  compact?: boolean;
}

/**
 * Three ways in, because people have three kinds of image.
 *
 * Paste is the one that matters most on a desktop: the screenshot or the messenger image is
 * already on the clipboard, and a file picker makes the user save it to disk first — which is when
 * a phone screenshot becomes a PNG whose metadata says "screenshot" and the metadata check has
 * something to say. The clipboard read is `items.find(kind === "file")` rather than
 * `clipboardData.files` because Safari only populates the former.
 */
export function Dropzone({ onSource, busy = false, busyLabel, compact = false }: DropzoneProps) {
  const [dragging, setDragging] = React.useState(false);
  const [rejected, setRejected] = React.useState<string | null>(null);
  const [url, setUrl] = React.useState("");
  const [showUrl, setShowUrl] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const dragDepth = React.useRef(0);

  const accept = React.useCallback(
    (file: File | Blob | null | undefined, name?: string) => {
      if (!file) return false;
      const label = name ?? (file instanceof File ? file.name : "clipboard image");
      const check = checkFile(file, label);
      if (!check.ok) {
        setRejected(check.reason ?? "That file cannot be analysed.");
        return false;
      }
      setRejected(null);
      // The name never leaves the browser: the backend labels the job with it so the *response*
      // can be matched to a file, and the scratch tree is deleted with the job.
      onSource({ file, filename: label });
      return true;
    },
    [onSource],
  );

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) accept(file);
  };

  // Window-level paste: the dropzone is not focused when the user presses Ctrl+V after a
  // screenshot, and a handler bound to the zone would simply miss it.
  React.useEffect(() => {
    if (busy) return;
    return watchClipboardImage((file, name) => accept(file, name));
  }, [accept, busy]);

  const submitUrl = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;
    if (!/^https?:\/\//i.test(trimmed)) {
      setRejected("That is not an http(s) address.");
      return;
    }
    setRejected(null);
    setShowUrl(false);
    setUrl("");
    onSource({ url: trimmed });
  };

  return (
    <div className="flex flex-col gap-2">
      <div
        onDragEnter={(e) => {
          e.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragging(false);
        }}
        onDrop={onDrop}
        className={cn(
          "relative overflow-hidden rounded-lg border border-dashed transition-colors",
          dragging
            ? "border-primary bg-primary/10"
            : "border-border bg-card/50 hover:border-primary/50",
          compact ? "px-4 py-5" : "px-6 py-10",
          busy && "pointer-events-none opacity-60",
        )}
      >
        {dragging && !compact && (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 h-px animate-scan bg-primary/70"
          />
        )}

        <div className="flex flex-col items-center gap-3 text-center">
          <span
            className={cn(
              "inline-flex size-11 items-center justify-center rounded-full border transition-colors",
              dragging
                ? "border-primary bg-primary/20 text-primary"
                : "border-border bg-secondary text-muted-foreground",
            )}
          >
            {busy ? (
              <Loader2 className="size-5 animate-spin" />
            ) : (
              <Upload className={cn("size-5", dragging && "scale-110")} />
            )}
          </span>

          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium">
              {busy
                ? (busyLabel ?? "Analysing…")
                : dragging
                  ? "Release to analyse"
                  : "Drop an image, or choose one"}
            </p>
            <p className="text-xs text-muted-foreground">
              JPEG · PNG · WebP · BMP — up to 10 MB. Paste with <kbd className="rounded border border-border bg-muted px-1 font-mono text-[0.65rem]">Ctrl+V</kbd>.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2">
            <Button type="button" size="sm" disabled={busy} onClick={() => inputRef.current?.click()}>
              <ImagePlus /> Choose file
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={busy}
              onClick={() => setShowUrl((v) => !v)}
              aria-expanded={showUrl}
            >
              <Link2 /> From URL
            </Button>
          </div>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT_ATTRIBUTE}
            className="sr-only"
            aria-label="Choose an image to analyse"
            onChange={(event) => {
              accept(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </div>
      </div>

      {showUrl && (
        <form onSubmit={submitUrl} className="flex items-center gap-2">
          <Input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/photo.jpg"
            type="url"
            aria-label="Image URL"
            className="h-9 flex-1 text-xs"
            disabled={busy}
          />
          <Button type="submit" size="sm" disabled={busy || !url.trim()}>
            Fetch
          </Button>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            aria-label="Cancel URL import"
            onClick={() => {
              setShowUrl(false);
              setUrl("");
            }}
          >
            <X />
          </Button>
        </form>
      )}

      <p
        role="status"
        aria-live="polite"
        className={cn("min-h-4 text-xs", rejected ? "text-risk-high" : "text-muted-foreground")}
      >
        {rejected ?? "Nothing leaves your browser except the image bytes; the server keeps them for at most ten minutes."}
      </p>
    </div>
  );
}
