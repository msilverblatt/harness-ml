#!/usr/bin/env bash
# Build frontend and output to Python package for static serving.
# Vite outputs directly to src/harnessml/studio/static/ via outDir config.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
bun install --frozen-lockfile

# Clear TypeScript build cache to prevent stale output
rm -f tsconfig.tsbuildinfo tsconfig.app.tsbuildinfo tsconfig.node.tsbuildinfo
rm -rf node_modules/.vite

bun run build
