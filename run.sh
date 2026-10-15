#!/usr/bin/env bash
# Start the Grand Prix dashboard.
#
# Nothing races until you press Start race: pick your models and your map in the
# UI, then start it.
#
#   ./run.sh                 # port 8078, no race running, you choose in the UI
#   ./run.sh --replay        # autostart two synthetic racers, needs no models
#   ./run.sh --replay 9000   # pick a port
#
# Nothing to install: stdlib Python plus the PixiJS bundle vendored in
# static/vendor/. No CDN, no npm, no network.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODE=""
if [ "${1:-}" = "--replay" ] || [ "${1:-}" = "--autostart" ]; then
  MODE="$1"
  shift
fi
PORT="${1:-8078}"

python3 - <<'PY' || exit 1
import sys
if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required (found %s). Try: brew install python@3.12" % sys.version.split()[0])
PY

cd "$ROOT"
echo "dashboard  http://127.0.0.1:$PORT/"
echo "nothing is racing yet: choose your models and map on the page, then press Start race"
[ -n "$MODE" ] && echo "mode       $MODE (starts a race immediately)"
exec python3 server.py --port "$PORT" $MODE
