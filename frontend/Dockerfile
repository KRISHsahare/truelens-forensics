# TrueLens web. Two stages, because `NEXT_PUBLIC_*` values are inlined into the browser bundle at
# build time: a runtime env var cannot supply them, so the API origin is a build argument.
#
#   docker build -t truelens-web --build-arg NEXT_PUBLIC_API_BASE_URL=https://api.example.com frontend/
#
# Not verified by building on the authoring machine — see backend/Dockerfile for why. The same
# `next build` that this image runs is what `npm run build` does locally, and that is exercised.
FROM node:24-slim AS builder

WORKDIR /app
# Only the lockfile first, so a source change does not re-install the dependency tree.
COPY package.json package-lock.json ./
# `npm ci` rather than `install`: the lockfile is the input, and a build that quietly resolves a
# newer minor is a build that cannot be reproduced six months later.
RUN npm ci --no-audit --no-fund

COPY . .
# The origin the browser will call. Empty means the app's own default (http://localhost:8000), which
# is right for the compose stack where the API is reachable at `api:8010` from the *browser* only
# through a published port — see docker-compose.yml for why this is not the container's network name.
ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build

FROM node:24-slim AS runtime

# `next start` needs no dev dependencies, and `output: "standalone"` has already traced exactly what
# it does need into `.next/standalone`.
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

# The standalone server runs as node, and the copied tree is owned by root by default. It only reads.
USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+process.env.PORT+'/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
