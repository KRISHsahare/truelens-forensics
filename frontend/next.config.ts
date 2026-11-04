import type { NextConfig } from "next";

/**
 * Static export target, no server-side data fetching.
 *
 * `cacheComponents` is off on purpose. It optimises *server* renders that fetch data per request, and
 * this app has none: every page is a shell plus a client component that polls the API, so the cache
 * layer would only add rules to work around (uncached client UI under `Suspense`, `connection()`
 * where a page needs a request-time value). The same is true of `proxy.ts`/middleware — there is no
 * per-request server decision to make, and the API is called directly from the browser rather than
 * proxied, because a same-origin proxy would buffer a 10 MB upload against a 10 MB default limit.
 *
 * If this ever grows server-rendered pages that hit the backend, re-enable it with
 * `use cache` + `cacheLife` rather than the removed `dynamic`/`revalidate` segment exports.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // The dev server binds `localhost` and refuses cross-origin requests to `/_next/*` unless the host
  // is named here — so opening `http://127.0.0.1:3000`, which `frontend/.env.local` and the backend's
  // default port both point at, served a page whose HMR and dev routes were blocked. The page looked
  // rendered and was inert. Production builds ignore this key.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  // `next start` from a copied `node_modules` is a 400 MB image of packages the running server never
  // reads. Standalone emits the traced subset — the server, its dependencies, and nothing else — which
  // is what `frontend/Dockerfile` copies. Vercel does not need this and is unaffected by it.
  output: "standalone",
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
