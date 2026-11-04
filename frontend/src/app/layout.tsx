import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { SiteFooter, SiteHeader } from "@/components/site/chrome";
import { TooltipProvider } from "@/components/ui/tooltip";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "TrueLens — image authenticity analysis",
    template: "%s · TrueLens",
  },
  description:
    "Ten independent forensic checks on any JPEG, PNG, WebP or BMP: frequency artefacts, a " +
    "learned classifier, error-level analysis, copy-move, noise, JPEG ghosts, block alignment, "
    + "faces, EXIF and C2PA credentials — fused into a probability with the evidence behind it.",
  applicationName: "TrueLens",
  metadataBase: new URL("https://truelens.app"),
  openGraph: {
    title: "TrueLens — image authenticity analysis",
    description:
      "Why an image looks synthetic or edited, shown as evidence rather than as a verdict.",
    type: "website",
  },
  twitter: { card: "summary_large_image", title: "TrueLens" },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0a1017" },
    { media: "(prefers-color-scheme: light)", color: "#f7f9fc" },
  ],
  colorScheme: "dark light",
};

/**
 * Runs before first paint.
 *
 * The palette is a product decision, not a decoration, so the choice has to be in the DOM the first
 * time the browser paints a red bar — a theme applied after hydration would show a flash of the
 * wrong colours on every reload. `data-theme` is what the CSS variables key off; localStorage is
 * what the toggle writes, and absence of a stored value means dark.
 */
const THEME_BOOTSTRAP = `
(function () {
  try {
    var stored = localStorage.getItem("truelens-theme");
    var theme = stored === "light" || stored === "dark" ? stored : "dark";
    document.documentElement.setAttribute("data-theme", theme);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={geistSans.variable}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body className={`${geistMono.variable} flex min-h-dvh flex-col antialiased`}>
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground"
        >
          Skip to analysis
        </a>
        <TooltipProvider>
          <SiteHeader />
          <main id="main" className="flex-1">
            {children}
          </main>
          <SiteFooter />
        </TooltipProvider>
      </body>
    </html>
  );
}
