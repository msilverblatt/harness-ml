#!/usr/bin/env bash
# Build frontend and copy to Python package for static serving
set -euo pipefail
cd "$(dirname "$0")/../frontend"
bun install --frozen-lockfile
bun run build
rm -rf ../src/harnessml/studio/static
cp -r dist ../src/harnessml/studio/static
