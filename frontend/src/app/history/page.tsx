"use client";

/**
 * History — read out of the browser, not the server.
 *
 * There is no `/api/history`. The product promises the server keeps no images and has no accounts, so
 * a history page can only be built from what *this* browser stored in IndexedDB (`lib/history.ts`).
 * Three consequences the UI makes visible rather than hiding:
 *
 * - **It is per-device and per-browser.** Clearing site data, a private window, or another laptop
 *   produces an empty page. That is the same mechanism that keeps the images private, so the page says
 *   so instead of apologising.
 * - **Old rows have no heatmaps.** Overlay links are signed against scratch files that are deleted
 *   within ten minutes. A replayed analysis shows scores, reasons and metadata, and says plainly that
 *   the pixels are gone.
 * - **Deleting is local and immediate.** One row or all of them, no server call, no tombstone.
 */

import * as React from "react";
import Link from "next/link";
import {
  Database,
  FileClock,
  HardDriveDownload,
  History as HistoryIcon,
  ImageOff,
  Search,
  Trash2,
} from "lucide-react";
import { Badge, BandBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  clearEntries,
  deleteEntry,
  historySupported,
  listEntries,
  type HistorySummary,
} from "@/lib/history";
import { useThumbnailsEnabled } from "@/lib/settings";
import { bandFor, bytes, pct, shortSha, timeAgo, VERDICT_SHORT } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { VerdictLabel } from "@/lib/types";

type Filter = VerdictLabel | "all";

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "ai_generated", label: "AI-generated" },
  { value: "edited", label: "Edited" },
  { value: "authentic", label: "No finding" },
  { value: "inconclusive", label: "Inconclusive" },
];

export default function HistoryPage() {
  // `null` until the browser has answered. `indexedDB` does not exist during the server render, so
  // deciding this during render makes the markup disagree with the client's and React throws the
  // whole tree away — the hydration error this line used to cause. `null` renders the same thing the
  // server did, and the unsupported card replaces it a frame later in the one browser that needs it.
  const [supported, setSupported] = React.useState<boolean | null>(null);
  const [entries, setEntries] = React.useState<HistorySummary[] | null>(null);
  const [filter, setFilter] = React.useState<Filter>("all");
  const [query, setQuery] = React.useState("");
  const [confirmClear, setConfirmClear] = React.useState(false);
  const [notice, setNotice] = React.useState<string | null>(null);
  const [thumbnails, setThumbnails] = useThumbnailsEnabled();

  React.useEffect(() => {
    setSupported(historySupported());
  }, []);

  const refresh = React.useCallback(() => {
    if (!supported) return;
    listEntries()
      .then(setEntries)
      .catch(() => setEntries([]));
  }, [supported]);

  React.useEffect(refresh, [refresh]);

  const shown = React.useMemo(() => {
    const list = entries ?? [];
    const needle = query.trim().toLowerCase();
    return list.filter((entry) => {
      if (filter !== "all" && entry.verdict !== filter) return false;
      if (!needle) return true;
      return (
        entry.sha256.toLowerCase().includes(needle) ||
        entry.headline.toLowerCase().includes(needle) ||
        VERDICT_SHORT[entry.verdict].toLowerCase().includes(needle)
      );
    });
  }, [entries, filter, query]);

  async function remove(jobId: string) {
    await deleteEntry(jobId);
    setNotice("Removed from this browser.");
    refresh();
  }

  async function wipe() {
    await clearEntries();
    setConfirmClear(false);
    setNotice("History cleared. Nothing was sent anywhere to be deleted.");
    refresh();
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
      <header className="flex flex-col gap-2">
        <span className="inline-flex items-center gap-2 text-[0.7rem] uppercase tracking-[0.18em] text-primary">
          <FileClock className="size-3.5" /> History
        </span>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Kept by your browser alone
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Every row below lives in this browser&apos;s IndexedDB. There is no account to sign into and
          no server-side log to request: clear the site data and this page is empty on every future
          visit, from every device.
        </p>
      </header>

      {supported === false && (
        <Card className="border-risk-moderate/40 bg-risk-moderate/5">
          <CardContent className="flex items-start gap-3 py-4">
            <ImageOff className="mt-0.5 size-4 shrink-0 text-risk-moderate" />
            <div className="flex flex-col gap-1">
              <p className="text-sm font-medium">This browser will not keep a history</p>
              <p className="text-xs leading-relaxed text-muted-foreground">
                Private/incognito mode or blocked site storage prevents IndexedDB. Analyses still run;
                nothing is remembered once the tab closes — which is the same guarantee the rest of
                this page describes, just enforced by the browser instead of by a button.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {supported === true && (
        <Card grid>
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-sm">Analyses in this browser</CardTitle>
              <Badge variant="outline">{entries?.length ?? 0} stored</Badge>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <Switch
                  id="history-thumbnails"
                  checked={thumbnails}
                  onCheckedChange={setThumbnails}
                  aria-label="Keep a 128 pixel thumbnail for each stored analysis"
                />
                <span className="text-[0.7rem] text-muted-foreground">Thumbnails</span>
              </div>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => setConfirmClear(true)}
                disabled={!entries?.length}
              >
                <Trash2 /> Clear history
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
                <TabsList>
                  {FILTERS.map((item) => (
                    <TabsTrigger key={item.value} value={item.value} className="text-xs">
                      {item.label}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
              <div className="relative sm:w-64">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="hash or headline"
                  aria-label="Search history"
                  className="h-9 pl-8 text-xs"
                />
              </div>
            </div>

            <p className="flex items-start gap-2 text-[0.7rem] leading-relaxed text-muted-foreground">
              <HardDriveDownload className="mt-0.5 size-3.5 shrink-0" />
              {thumbnails
                ? "Each row keeps a 128 px JPEG drawn in this tab, plus the numbers. The photograph itself is never stored — re-open it from wherever you saved it."
                : "Thumbnails are off, so rows store scores and metadata only. Turn them back on to keep a 128 px preview per analysis."}
            </p>

            {notice && (
              <p role="status" aria-live="polite" className="text-xs text-primary">
                {notice}
              </p>
            )}

            {entries === null ? (
              <p className="py-8 text-center text-sm text-muted-foreground">Reading local storage…</p>
            ) : shown.length === 0 ? (
              <div className="flex flex-col items-center gap-2 py-10 text-center">
                <Database className="size-6 text-muted-foreground" />
                <p className="text-sm">
                  {entries.length === 0
                    ? "Nothing here yet."
                    : "No stored analysis matches this filter."}
                </p>
                <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
                  {entries.length === 0
                    ? "Run one from the analyser — the result is written to this browser when it finishes, and it stays only as long as you do not clear it."
                    : "Clear the search or switch the filter back to All."}
                </p>
              </div>
            ) : (
              <ul className="flex flex-col gap-2">
                {shown.map((entry) => (
                  <HistoryRow key={entry.job_id} entry={entry} onRemove={remove} />
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      )}

      <Dialog open={confirmClear} onOpenChange={setConfirmClear}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-sm">Delete every stored analysis?</DialogTitle>
            <DialogDescription className="text-xs leading-relaxed">
              This removes all rows, thumbnails and cached results from this browser. It cannot be
              undone, and there is no server copy to restore from — that is the point.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <DialogClose asChild>
              <Button type="button" size="sm" variant="ghost">
                Keep them
              </Button>
            </DialogClose>
            <Button type="button" size="sm" variant="destructive" onClick={wipe}>
              <Trash2 /> Delete all
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Card className="border-dashed">
        <CardContent className="flex flex-col gap-2 py-4 text-xs leading-relaxed text-muted-foreground">
          <p className="font-medium text-foreground">What a stored row is</p>
          <p>
            The full result envelope — verdict, sub-scores, every technique&apos;s explanation, the
            metadata and C2PA blocks — plus a 128 px thumbnail and the image hash. Around the size of
            a couple of photographs, capped at 60 rows, oldest evicted first.
          </p>
          <p>
            It is enough to re-read a conclusion months later without a server. It is never enough to
            re-run a check: the pixels went away with the job, which is why reopened rows carry the
            &quot;heatmaps expired&quot; notice.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function HistoryRow({
  entry,
  onRemove,
}: {
  entry: HistorySummary;
  onRemove: (jobId: string) => void;
}) {
  const band = bandFor(entry.overall);
  return (
    <li
      className={cn(
        "flex items-center gap-3 rounded-lg border border-border bg-card px-3 py-2.5",
        "transition-colors hover:border-primary/40",
      )}
    >
      <span className="grid size-14 shrink-0 place-items-center overflow-hidden rounded border border-border bg-muted">
        {entry.thumbnail ? (
          <img src={entry.thumbnail} alt="" className="size-full object-cover" />
        ) : (
          <ImageOff className="size-4 text-muted-foreground" />
        )}
      </span>

      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <BandBadge band={band}>{VERDICT_SHORT[entry.verdict]}</BandBadge>
          <span className="truncate text-xs">{entry.headline}</span>
          {entry.from_cache && <Badge variant="muted">cached</Badge>}
        </div>
        <p className="tabular flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[0.68rem] text-muted-foreground">
          <span>{timeAgo(entry.created_at)}</span>
          <span>{shortSha(entry.sha256, 8, 4)}</span>
          <span>
            {entry.width && entry.height ? `${entry.width}×${entry.height}` : "size unknown"}
          </span>
          <span>{bytes(entry.size_bytes)}</span>
        </p>
      </div>

      <div className="tabular hidden shrink-0 flex-col items-end gap-0.5 text-[0.7rem] sm:flex">
        <span>overall {pct(entry.overall)}</span>
        <span className="text-muted-foreground">
          synthetic {pct(entry.ai_score)} · edited {pct(entry.edited_score)}
        </span>
        <span className="text-muted-foreground">reliability {pct(entry.reliability)}</span>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <Button asChild size="sm" variant="ghost" className="px-2">
          <Link href={`/history/${entry.job_id}`} aria-label={`Open the stored analysis ${shortSha(entry.sha256, 8, 4)}`}>
            Open
          </Link>
        </Button>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          className="size-8"
          aria-label="Remove this entry from history"
          onClick={() => onRemove(entry.job_id)}
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>
    </li>
  );
}
