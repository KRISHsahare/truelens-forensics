"use client";

import * as React from "react";
import { ImagePlus, Loader2, Trash2, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ACCEPT_ATTRIBUTE, BATCH_MAX_IMAGES, checkFile } from "@/lib/api";
import { bytes } from "@/lib/format";
import { cn } from "@/lib/utils";

export interface BatchIntakeProps {
  files: File[];
  onAdd: (files: File[]) => void;
  onRemove: (index: number) => void;
  onClear: () => void;
  onSubmit: () => void;
  busy?: boolean;
  notice?: string | null;
}

/**
 * Batch intake.
 *
 * Deliberately *not* the single-image dropzone with `multiple` switched on: batch mode has a queue
 * to show, a count that must not exceed `BATCH_MAX_IMAGES`, and no URL import — twenty remote fetches
 * in one request is a different failure story than one, and this page is not the place to tell it.
 *
 * The clipboard is ignored here on purpose. A pasted screenshot is one image; the single analyser is
 * the page for it, and stealing Ctrl+V on both pages at once would make the two fight over one key.
 */
export function BatchIntake({
  files,
  onAdd,
  onRemove,
  onClear,
  onSubmit,
  busy = false,
  notice = null,
}: BatchIntakeProps) {
  const [dragging, setDragging] = React.useState(false);
  const [rejected, setRejected] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const dragDepth = React.useRef(0);

  const accept = React.useCallback(
    (incoming: File[]) => {
      if (!incoming.length) return;
      const room = Math.max(0, BATCH_MAX_IMAGES - files.length);
      if (room === 0) {
        setRejected(`Batch mode holds ${BATCH_MAX_IMAGES} images. Remove one to add another.`);
        return;
      }
      const kept: File[] = [];
      const refused = new Set<string>();
      for (const file of incoming.slice(0, room)) {
        const check = checkFile(file, file.name);
        if (!check.ok) {
          refused.add(check.reason ?? "That file cannot be analysed.");
          continue;
        }
        kept.push(file);
      }
      if (incoming.length > room) {
        refused.add(`Only the first ${room} could be added — batch mode holds ${BATCH_MAX_IMAGES}.`);
      }
      setRejected(refused.size ? Array.from(refused).join(" ") : null);
      if (kept.length) onAdd(kept);
    },
    [files.length, onAdd],
  );

  const totalBytes = React.useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0),
    [files],
  );

  return (
    <div className="flex flex-col gap-3">
      <div
        onDragEnter={(event) => {
          event.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          dragDepth.current = 0;
          setDragging(false);
          accept(Array.from(event.dataTransfer.files ?? []));
        }}
        className={cn(
          "relative overflow-hidden rounded-lg border border-dashed px-5 py-7 transition-colors",
          dragging
            ? "border-primary bg-primary/10"
            : "border-border bg-card/50 hover:border-primary/50",
          busy && "pointer-events-none opacity-60",
        )}
      >
        {dragging && (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-0 top-0 h-px animate-scan bg-primary/70"
          />
        )}
        <div className="flex flex-col items-center gap-2.5 text-center">
          <span
            className={cn(
              "inline-flex size-10 items-center justify-center rounded-full border",
              dragging
                ? "border-primary bg-primary/20 text-primary"
                : "border-border bg-secondary text-muted-foreground",
            )}
          >
            {busy ? <Loader2 className="size-5 animate-spin" /> : <Upload className="size-5" />}
          </span>
          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium">
              {busy ? "Queueing…" : dragging ? "Release to add these" : "Drop up to 20 images"}
            </p>
            <p className="text-xs text-muted-foreground">
              JPEG · PNG · WebP · BMP — each up to 10 MB. They upload together and analyse as
              separate jobs.
            </p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
          >
            <ImagePlus /> Choose files
          </Button>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT_ATTRIBUTE}
            className="sr-only"
            aria-label="Choose images to analyse"
            onChange={(event) => {
              accept(Array.from(event.target.files ?? []));
              event.target.value = "";
            }}
          />
        </div>
      </div>

      {files.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs">
            <span className="tabular text-muted-foreground">
              {files.length} / {BATCH_MAX_IMAGES} · {bytes(totalBytes)}
            </span>
            <Button type="button" size="sm" variant="ghost" disabled={busy} onClick={onClear}>
              <Trash2 /> Clear
            </Button>
          </div>
          <ul className="max-h-56 overflow-y-auto rounded-md border border-border bg-card">
            {files.map((file, index) => (
              <li
                key={`${file.name}-${index}`}
                className="flex items-center gap-2 border-b border-border/60 px-2.5 py-1.5 text-xs last:border-b-0"
              >
                <span className="tabular w-6 shrink-0 text-right text-muted-foreground">
                  {index + 1}
                </span>
                <span className="min-w-0 flex-1 truncate" title={file.name}>
                  {file.name}
                </span>
                <span className="tabular shrink-0 text-muted-foreground">{bytes(file.size)}</span>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-6 shrink-0"
                  aria-label={`Remove ${file.name}`}
                  disabled={busy}
                  onClick={() => onRemove(index)}
                >
                  <X className="size-3.5" />
                </Button>
              </li>
            ))}
          </ul>
          <Button type="button" onClick={onSubmit} disabled={busy || files.length === 0}>
            {busy ? <Loader2 className="animate-spin" /> : <ImagePlus />}
            Analyse {files.length} image{files.length === 1 ? "" : "s"}
          </Button>
        </div>
      )}

      <p
        role="status"
        aria-live="polite"
        className={cn("min-h-4 text-xs", rejected ? "text-risk-high" : "text-muted-foreground")}
      >
        {rejected ?? notice ?? "Nothing leaves your browser except the image bytes; the server keeps them for at most ten minutes."}
      </p>
    </div>
  );
}
