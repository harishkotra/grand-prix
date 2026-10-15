# Setup: five decision models on one 8 GB Mac

Everything here ends at the same place: a `POST /v1/systemone` the dashboard can
call. The dashboard itself needs nothing installed beyond Python 3.11+ and the
vendored PixiJS bundle.

## 1. Check what is already up

```bash
python3 tools/probe_engines.py
```

It live-invokes every configured engine against a real race state and prints the
distribution each one returned. Anything that did not answer is reported as not
answering. This is the only check that counts; a model can be downloaded, listed,
and still not serve the endpoint.

## 2. Start the llama.cpp engines

One server per model, because a decision model needs its option logits read at a
single position and swapping models inside one server would add a load per
decision.

```bash
llama-server -hf ggml-org/Julia-1-GGUF:Q8_0      --port 8081 --host 127.0.0.1 -c 4096
llama-server -hf ggml-org/Laya-GGUF:Q8_0         --port 8082 --host 127.0.0.1 -c 4096
llama-server -hf ggml-org/Kev-0.8B-GGUF:Q8_0     --port 8083 --host 127.0.0.1 -c 4096
```

Notes that cost time when you learn them the hard way:

- **A quant tag that does not exist fails the load.** If a repo does not publish
  `Q8_0`, drop the tag (`-hf ggml-org/Laya-GGUF`) and llama.cpp picks the repo's
  default, then read what actually loaded from `GET /v1/models`.
- **A GGUF without `.decision.` metadata answers `501 This model is not a
  decision model`.** llama.cpp will not fall back to reading a chat model's
  letter logits. `python3 tools/gguf_meta.py <file>` tells you before you blame
  the build. All five models above are from the decision collection and carry the
  metadata.
- **Pick free ports.** A busy port fails at startup with `couldn't bind HTTP
  server socket` and the process exits; `lsof` can show nothing if the port is in
  TIME_WAIT.
- **A chat window shows nonsense for these models.** There is no `ollama run`
  path and no chat path. They are driven through `/v1/systemone` only.

## 3. Start Tev1 through Ollama

Needs Ollama 0.35 or newer, which added `/v1/systemone`.

```bash
ollama serve
ollama pull tev1:0.8b
```

Confirm the endpoint exists rather than guessing:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:11434/v1/systemone \
  -H 'Content-Type: application/json' -d '{}'
# 400 {"error":"model is required"} -> endpoint present
# 404                              -> Ollama too old
```

## 4. Clef-flash (optional, needs more than this laptop has)

Clef-flash is 9B and Ollama serves it, so there is no cloud account to set up:

```bash
ollama pull clef-flash:9b     # 11 GB
```

At 11 GB it will not co-reside with the other five here, and this is the 8 GB
machine, so on this host it is a one-at-a-time curiosity. If you want the six-racer
board on this laptop, run the Q4_K_M GGUF instead, which is about 5.4 GB:

```bash
llama-server -hf ggml-org/Clef-Flash-GGUF:Q4_K_M --port 8085 --host 127.0.0.1 -c 4096
```

and set the `clef-flash` racer's `"type"` to `llama_cpp` with `"port": 8085`.

If neither is available, the panel reads `unavailable` with the reason. That is
the intended behaviour: the board never invents a decision.

## 5. Memory budget on 8 GB

Roughly 2.2 GB of weights for the four llama.cpp models plus Tev1 (144M + 421M +
812M + 752M) plus a small context per server. That fits, but it does not fit
alongside a browser, an IDE and Docker. Close what you are not using, and start the
engines one at a time so you can see which one fails.

If it swaps, drop to four racers by editing `config.json`: delete the racer entry
and it disappears from the board. Nothing else needs to change.

## 6. Run the dashboard

```bash
python3 server.py --port 8078            # then press Start race
python3 server.py --port 8078 --autostart --replay   # synthetic, no models needed
```

Measure before you film:

```bash
python3 tools/run_headless.py --laps 1
python3 tools/run_headless.py --laps 1 --engines julia-1-144m,kev-0.8b
```

Headless writes `runs/<race_id>.jsonl` (one row per decision, with the full
distribution) and `runs/<race_id>.summary.json`, and prints the table.
