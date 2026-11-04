"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BookOpen,
  FileClock,
  LayoutGrid,
  Moon,
  Scan,
  ShieldAlert,
  Sun,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getHealth } from "@/lib/api";
import { clock } from "@/lib/format";
import { Tooltip } from "@/components/ui/tooltip";
import type { HealthResponse } from "@/lib/types";

const NAV = [
  { href: "/analyze", label: "Analyze", icon: Scan },
  { href: "/methodology", label: "Methodology", icon: Activity },
  { href: "/batch", label: "Batch", icon: LayoutGrid },
  { href: "/history", label: "History", icon: FileClock },
  { href: "/about", label: "About", icon: ShieldAlert },
];

/**
 * The engine does not grow a second copy of this list.
 *
 * `/api/health` answers `blocked` when the job store, the scratch root or the signing key is wrong,
 * and an upload against that backend fails with a 503 the user cannot explain. Saying "offline"
 * next to the logo is the cheapest possible version of the same sentence.
 */
type Health =
  | { state: "checking"; detail?: undefined }
  | { state: "ok"; detail?: undefined; uptime: number }
  | { state: "degraded"; detail: string }
  | { state: "down"; detail: string };

const HEALTH_POLL_MS = 60_000;

function useBackendHealth() {
  const [health, setHealth] = React.useState<Health>({ state: "checking" });

  React.useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const response: HealthResponse = await getHealth();
        if (cancelled) return;
        if (response.status === "ok") {
          setHealth({ state: "ok", uptime: response.uptime_s ?? 0 });
        } else {
          setHealth({
            state: "degraded",
            detail: (response.problems ?? []).join("; ") || response.status,
          });
        }
      } catch (error) {
        if (!cancelled) {
          setHealth({
            state: "down",
            detail: error instanceof Error ? error.message : "unreachable",
          });
        }
      }
    };
    void check();
    const timer = setInterval(() => void check(), HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return health;
}

function HealthPill({ className }: { className?: string }) {
  const health = useBackendHealth();
  const copy: Record<Health["state"], string> = {
    checking: "Checking engine",
    ok: "Engine ready",
    degraded: "Engine degraded",
    down: "Engine offline",
  };
  const tone: Record<Health["state"], string> = {
    checking: "text-muted-foreground",
    ok: "text-risk-low",
    degraded: "text-risk-moderate",
    down: "text-risk-high",
  };
  const dot: Record<Health["state"], string> = {
    checking: "bg-muted-foreground animate-pulse-soft",
    ok: "bg-risk-low",
    degraded: "bg-risk-moderate",
    down: "bg-risk-high",
  };

  return (
    <Tooltip
      content={
        health.state === "ok"
          ? `Engine has been up for ${clock(health.uptime)}.`
          : (health.detail ?? "Checking the analysis engine.")
      }
      side="bottom"
    >
      <Link
        href="/about"
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-card/60 px-2.5 py-1 text-[0.7rem] font-medium",
          tone[health.state],
          className,
        )}
      >
        <span aria-hidden className={cn("size-1.5 rounded-full", dot[health.state])} />
        <span className="hidden sm:inline">{copy[health.state]}</span>
        <span className="sr-only">{copy[health.state]}</span>
      </Link>
    </Tooltip>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = React.useState<"dark" | "light">("dark");

  // The bootstrap script in the layout has already written the attribute; read it rather than
  // assuming, or the toggle's first press flips to the value the user is already looking at.
  React.useEffect(() => {
    const current = document.documentElement.getAttribute("data-theme");
    if (current === "light" || current === "dark") setTheme(current);
  }, []);

  const flip = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("truelens-theme", next);
    } catch {
      // A private-mode write failure changes nothing about the current view.
    }
  };

  return (
    <button
      type="button"
      onClick={flip}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
      className="inline-flex size-8 items-center justify-center rounded-md border border-border/70 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
    >
      {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => setOpen(false), [pathname]);

  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70 print:hidden">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-3 px-4 sm:px-6">
        <Link href="/" className="group flex items-center gap-2">
          <span className="relative inline-flex size-7 items-center justify-center rounded-md border border-primary/40 bg-primary/10 text-primary">
            <Scan className="size-4" />
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-sm font-semibold tracking-tight">TrueLens</span>
            <span className="text-[0.65rem] text-muted-foreground">image authenticity lab</span>
          </span>
        </Link>

        <nav aria-label="Primary" className="ml-4 hidden items-center gap-0.5 md:flex">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
                  active
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <Icon className="size-3.5" />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <HealthPill className="hidden lg:inline-flex" />
          <ThemeToggle />
          <Link
            href="/analyze"
            className="hidden rounded-md bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground transition-colors hover:bg-primary/90 sm:inline-flex"
          >
            Run analysis
          </Link>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="mobile-nav"
            className="inline-flex size-8 items-center justify-center rounded-md border border-border/70 text-muted-foreground md:hidden"
          >
            <span className="sr-only">Menu</span>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
              <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </button>
        </div>
      </div>

      {open && (
        <nav
          id="mobile-nav"
          aria-label="Primary"
          className="border-t border-border/70 px-4 py-2 md:hidden"
        >
          <ul className="flex flex-col gap-0.5">
            {NAV.map(({ href, label }) => (
              <li key={href}>
                <Link
                  href={href}
                  className="block rounded-md px-2 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
                >
                  {label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border/70 print:hidden">
      <div className="mx-auto grid w-full max-w-7xl gap-6 px-4 py-8 sm:grid-cols-3 sm:px-6">
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-2 text-sm font-semibold">
            <Scan className="size-4 text-primary" /> TrueLens
          </span>
          <p className="max-w-xs text-xs leading-relaxed text-muted-foreground">
            A probability, not proof. Every score is one measurement among several, and the
            evidence behind it is always shown.
          </p>
        </div>

        <nav aria-label="Product" className="flex flex-col gap-1.5 text-xs">
          {NAV.map(({ href, label }) => (
            <Link key={href} href={href} className="text-muted-foreground hover:text-foreground">
              {label}
            </Link>
          ))}
        </nav>

        <div className="flex flex-col gap-1.5 text-xs">
          <Link href="/about#privacy" className="text-muted-foreground hover:text-foreground">
            Privacy: uploads are deleted
          </Link>
          <Link href="/about#limitations" className="text-muted-foreground hover:text-foreground">
            Known limitations
          </Link>
          <span className="text-muted-foreground">
            Images stay in your browser history only — the server keeps nothing after the job ends.
          </span>
          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <BookOpen className="size-3.5" /> Source &amp; methodology published
          </span>
        </div>
      </div>
      <div className="border-t border-border/60 py-3 text-center text-[0.7rem] text-muted-foreground">
        © {new Date().getFullYear()} TrueLens. Not legal or evidentiary advice.
      </div>
    </footer>
  );
}
