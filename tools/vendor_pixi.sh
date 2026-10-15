#!/usr/bin/env bash
# Vendor PixiJS locally so the dashboard works with no CDN and no network.
#
# PixiJS v8 is MIT licensed. The build ships an ESM bundle, which is the one the
# page imports. Re-run this if you upgrade PixiJS.
set -euo pipefail

cd "$(dirname "$0")/.."
npm install pixi.js@8 --no-audit --no-fund
mkdir -p static/vendor
cp node_modules/pixi.js/dist/pixi.min.mjs static/vendor/pixi.min.mjs
ls -lh static/vendor/pixi.min.mjs
echo "vendored. version: $(node -p "require('./node_modules/pixi.js/package.json').version")"
