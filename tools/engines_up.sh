#!/usr/bin/env bash
# Bring up every decision-model engine the board can race.
# Safe to re-run: an engine that is already answering is left alone.
#
#   ./tools/engines_up.sh              # three llama.cpp engines + Tev1 + Clef-flash
#   SKIP_CLEF=1 ./tools/engines_up.sh  # skip the 11 GB Clef-flash download
#   ./tools/engines_down.sh            # stop the ones this script started
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

command -v llama-server >/dev/null 2>&1 || {
  echo "llama-server not found. Install it first:"
  echo "  brew install llama.cpp"
  exit 1
}

# An engine counts as up only if it answers /v1/models. A port in LISTEN with a
# dead process behind it is the failure mode this avoids.
up() { curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$1/v1/models"; }

launch() { # name port repo [extra llama-server args...]
  local name="$1" port="$2" repo="$3"; shift 3
  if up "$port"; then
    echo "already up    $name  :$port"
    return 0
  fi
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "PORT BUSY     $name  :$port  something else is listening there"
    return 1
  fi
  echo "starting      $name  :$port  $repo"
  nohup llama-server -hf "$repo" --port "$port" --host 127.0.0.1 -c 4096 "$@" \
    >"$LOGS/$name.log" 2>&1 &
  echo $! >"$LOGS/$name.pid"
  return 0
}

# First run downloads the weights into the Hugging Face cache, so this is the
# slow part. Later runs load from disk.
launch julia-1-144m 8081 ggml-org/Julia-1-GGUF:Q8_0
launch laya-421m    8082 ggml-org/Laya-GGUF:Q8_0
launch kev-0.8b     8083 ggml-org/Kev-0.8B-GGUF:Q8_0

# Ollama owns port 11434 and serves two of the racers itself: Tev1 (0.8B) and
# Clef-flash (9B, 11 GB). Decision models need Ollama 0.35.1 or newer.
if command -v ollama >/dev/null 2>&1; then
  if ! curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:11434/api/version"; then
    echo "starting      ollama serve"
    nohup ollama serve >"$LOGS/ollama.log" 2>&1 &
    echo $! >"$LOGS/ollama.pid"
    sleep 3
  fi
  echo "pulling       tev1:0.8b  (752 MB, first run only)"
  ollama pull tev1:0.8b || echo "  tev1 pull failed, the board will show it unavailable"
  if [ "${SKIP_CLEF:-0}" = "1" ]; then
    echo "skipping      clef-flash:9b  (SKIP_CLEF=1)"
  else
    echo "pulling       clef-flash:9b  (11 GB, first run only, SKIP_CLEF=1 to skip)"
    ollama pull clef-flash:9b || echo "  clef-flash pull failed, the board will show it unavailable"
  fi
else
  echo "skipping      Tev1 and Clef-flash (ollama not installed: brew install ollama)"
fi

echo
echo "waiting for engines to answer /v1/models ..."
failed_ports=""
for port in 8081 8082 8083; do
  answered=0
  for _ in $(seq 1 90); do
    if up "$port"; then answered=1; break; fi
    sleep 1
  done
  [ "$answered" = "1" ] || failed_ports="$failed_ports $port"
done

# A port that never answered has a reason and it is in that engine's log. Print it
# here, so the failure explains itself instead of leaving a bare connection refused.
for port in $failed_ports; do
  case "$port" in
    8081) name="julia-1-144m" ;;
    8082) name="laya-421m" ;;
    8083) name="kev-0.8b" ;;
    *)    name="" ;;
  esac
  echo
  echo "!! nothing answered on :$port. Last lines of logs/$name.log:"
  tail -n 15 "$LOGS/$name.log" 2>/dev/null | sed 's/^/     /'
done

echo
echo "Live check. This is the only result that counts:"
python3 "$ROOT/tools/probe_engines.py"
