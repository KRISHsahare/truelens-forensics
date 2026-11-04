/**
 * The handful of things the browser remembers about *this user's preferences*.
 *
 * These are not application state — they are the two switches the privacy promise needs a place to
 * live, and they have to survive a reload without a server round-trip (there is no account, and the
 * server is not supposed to know who you are). `localStorage` is the right store: same-origin, never
 * sent in a request, and cleared with the rest of the site's data.
 *
 * Every read is guarded and defaults to the more useful behaviour, because these run during a server
 * render where `localStorage` does not exist and because a preference that throws takes the page down
 * with it.
 */

import * as React from "react";

const THUMBNAILS_KEY = "truelens-history-thumbnails";

function read(key: string, fallback: boolean): boolean {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : raw === "on";
  } catch {
    // A browser that will not hand over storage (private mode, blocked site data) still gets a page.
    return fallback;
  }
}

function write(key: string, value: boolean): void {
  try {
    localStorage.setItem(key, value ? "on" : "off");
  } catch {
    // Ignored: the preference is a nicety, and failing to keep it is not worth a user-visible error.
  }
}

export function thumbnailsEnabled(): boolean {
  return read(THUMBNAILS_KEY, true);
}

export function setThumbnailsEnabled(enabled: boolean): void {
  write(THUMBNAILS_KEY, enabled);
}

/**
 * Hydration-safe toggle.
 *
 * The first render has to match the server's, so the state starts at the default and the stored value
 * is applied in an effect. That means the switch can visibly settle a frame after mount — which is
 * correct, because the alternative is reading `localStorage` during render and shipping markup that
 * disagrees with the client's.
 */
export function useThumbnailsEnabled(): [boolean, (enabled: boolean) => void] {
  const [enabled, setEnabled] = React.useState(true);

  React.useEffect(() => {
    setEnabled(thumbnailsEnabled());
  }, []);

  const update = React.useCallback((next: boolean) => {
    setEnabled(next);
    setThumbnailsEnabled(next);
  }, []);

  return [enabled, update];
}
