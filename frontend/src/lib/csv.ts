/**
 * CSV export for the batch table.
 *
 * Two rules, both about what a spreadsheet does to a forensic number:
 *
 * - **An unmeasured score is an empty cell, never `0`.** Excel renders an empty cell as blank and a
 *   zero as `0`, and a zero in a risk column reads as "this check passed". The frontend already
 *   refuses to paint that lie on screen; a download that quietly accepts it is worse, because the
 *   file leaves the room.
 * - **Every field is quoted** when it contains a comma, a quote, a newline, or leading/trailing
 *   whitespace, and quotes double. Filenames and the server's own sentences contain all of those.
 *
 * A UTF-8 BOM goes first so Excel on Windows opens a report with é in a caption instead of mojibake.
 */

export type CsvValue = string | number | null | undefined;

const BOM = "﻿";

export function csvEscape(value: CsvValue): string {
  if (value == null) return "";
  const text = String(value);
  if (text === "") return "";
  if (/[",\n\r]|^\s|\s$/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

export function toCsv(headers: readonly string[], rows: readonly CsvValue[][]): string {
  const lines = [headers.map(csvEscape).join(",")];
  for (const row of rows) lines.push(row.map(csvEscape).join(","));
  return BOM + lines.join("\r\n") + "\r\n";
}

/**
 * Trigger a download of generated text.
 *
 * The anchor is appended to the document because a detached one is ignored by some browsers, and it
 * is revoked on a timeout rather than immediately — revoking before the navigation starts loses the
 * file on Safari.
 */
export function downloadText(filename: string, text: string, mime = "text/csv;charset=utf-8"): void {
  if (typeof document === "undefined") return;
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function timestampSlug(date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    pad(date.getHours()),
    pad(date.getMinutes()),
  ].join("-");
}
