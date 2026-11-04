/**
 * Passing a dropped file from the landing page to the Analyzer.
 *
 * A module-level hand-off rather than a URL, a store, or an IndexedDB staging row: the file must
 * survive a client-side navigation and nothing else. Anything persistent would mean the image sits
 * somewhere after the tab closes, which is the property the privacy page spends a section denying.
 * `take()` clears it, so a reload of `/analyze` starts from an empty dropzone.
 */

export interface PendingUpload {
  file: File | Blob;
  filename: string;
}

let pending: PendingUpload | null = null;

export const handoff = {
  set(upload: PendingUpload) {
    pending = upload;
  },
  take(): PendingUpload | null {
    const value = pending;
    pending = null;
    return value;
  },
};
