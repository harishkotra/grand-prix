#!/usr/bin/env bash
# Stop the engines this project started, using the pid files engines_up.sh wrote.
# It does not touch a llama-server it did not start.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/logs"

stopped=0
for pidfile in "$LOGS"/*.pid; do
  [ -e "$pidfile" ] || continue
  name="$(basename "$pidfile" .pid)"
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null && echo "stopped       $name  pid $pid"
    stopped=$((stopped + 1))
  else
    echo "not running   $name"
  fi
  rm -f "$pidfile"
done

[ "$stopped" -eq 0 ] && echo "nothing of ours was running"

echo
echo "Ports still listening (anything here is not ours):"
for port in 8081 8082 8083 11434 8078; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  :$port  still busy"
  fi
done
