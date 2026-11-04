/**
 * Clipboard image watching, shared by the landing hero and the Analyzer dropzone.
 *
 * `items.find(kind === "file")` rather than `clipboardData.files`, because Safari only populates the
 * former — and a screenshot pasted from a phone-mirror or a capture tool is precisely the case where
 * the file list is empty while the item list is not.
 */

export function watchClipboardImage(onImage: (file: File, name: string) => void): () => void {
  const handler = (event: ClipboardEvent) => {
    const target = event.target as HTMLElement | null;
    // A paste into a text field is the user typing, not the user dropping a file.
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
      return;
    }
    const item = Array.from(event.clipboardData?.items ?? []).find(
      (entry) => entry.kind === "file" && entry.type.startsWith("image/"),
    );
    const file = item?.getAsFile();
    if (!file) return;
    event.preventDefault();
    onImage(file, file.name || `pasted-${new Date().toISOString().slice(11, 19).replace(/:/g, "")}.png`);
  };
  window.addEventListener("paste", handler);
  return () => window.removeEventListener("paste", handler);
}
