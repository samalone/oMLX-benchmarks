# omlxbench

A background benchmark recorder for [oMLX](https://omlx.ai), the MLX inference
server for Apple Silicon.

oMLX has good built-in benchmarks, but it keeps their results only in memory,
its web UI has no history or comparison view, and its upload to omlx.ai is
blocked by Cloudflare. omlxbench fills that gap:

- **Declare what you want measured** in `targets.toml`: models × benchmarks.
- **It fills in whatever is missing** whenever the server is idle, running
  oMLX's own benchmarks plus two of its own suites.
- **It steps aside the moment you use the server**, without breaking your
  requests.
- **Everything goes into SQLite**, both extracted columns and the verbatim oMLX
  JSON, so that you, or an AI agent, can compare models and make
  recommendations for *your* machine.

It is deliberately rough-and-ready: a Python CLI, a launchd agent and a
database. There is no UI.

## What it measures

| Suite | Source | What it tells you |
|---|---|---|
| `perf` | oMLX built-in | Prefill and generation speed, time to first token and peak memory at a given prompt length; aggregate throughput for 2–8 concurrent requests. |
| `context` | oMLX built-in | The largest prompt the machine can actually prefill. |
| `accuracy` | oMLX built-in | mmlu, mmlu_pro, gsm8k, humaneval, mbpp, livecodebench and others, with thinking on or off. |
| `tools` | omlxbench | Tool calling: [BFCL v3](https://gorilla.cs.berkeley.edu/leaderboard.html) cases sent through oMLX's chat API with real `tools`, so the model's chat template and oMLX's tool-call parser are tested together, scored like BFCL. |
| `agent_turns` | omlxbench | A replay of a fixed agent session: a ~20k-token system prompt with 40 tools, then tool results growing it to ~64k tokens, prefix-cached. It records time to first token at every step, which is the closest single number to how an agent like [Hermes](https://github.com/NousResearch/hermes-agent) feels. |

Each result is stored against a snapshot of the model (quantization, context
length, the acceleration features enabled) and of the environment (chip, RAM,
macOS, oMLX and mlx-lm versions, perf-relevant server settings). Changing a
model's settings therefore creates new gaps without discarding old data.

## Requirements

- macOS on Apple Silicon, with oMLX running locally. Developed against oMLX
  0.7.0; it talks to oMLX's admin API, which may change between versions.
- [uv](https://docs.astral.sh/uv/).
- oMLX's main API key. Sub-keys can't log in to the admin API.

## Setup

```sh
git clone https://github.com/samalone/oMLX-benchmarks.git
cd oMLX-benchmarks
echo 'OMLX_API_KEY=sk-omlx-...' > .env       # never committed
uv run omlxbench snapshot                     # check it can see the server and your models
uv run omlxbench plan                         # what targets.toml would run
```

Edit `targets.toml` to choose models and benchmarks. Lower tiers are completed
for every model before a higher tier starts, so cheap comparisons arrive
first. Removing an entry only stops future runs; stored results are kept.

To run it in the background, starting at login and restarting after a crash:

```sh
launchd/install.sh             # install and start the launchd agent
launchd/install.sh uninstall   # remove it
```

Optional environment variables (in `.env` or the environment): `OMLX_URL`
(default `http://127.0.0.1:8000`), `OMLXBENCH_DB` and `OMLXBENCH_TARGETS`.

## Everyday use

```sh
uv run omlxbench status        # runner state, current run, what's missing, recent runs
uv run omlxbench report        # latest results per model (add --json for agents)
uv run omlxbench html          # sortable HTML report in data/reports/, opened in the browser
uv run omlxbench pause 2h      # keep the server free; `pause` alone = until resumed
uv run omlxbench resume
uv run omlxbench requeue --model 'Qwen*' --spec 'perf.single:*' --yes   # measure again
uv run omlxbench import-ui     # keep results of runs you started in the oMLX web UI
```

The runner also imports web-UI runs automatically every 10 minutes, since oMLX
forgets them on restart.

## How it stays out of your way

Benchmarks take over the whole server: oMLX's built-in ones unload every model
first. So the runner starts a run only after `quiet_minutes` (default 10) with
no API traffic, and watches for real use while it runs:

- **API requests completing.** oMLX's request counter only counts HTTP API
  requests, never its own benchmarks.
- **Requests in flight** beyond the benchmark's own.
- **Another model being loaded.**
- **Requests oMLX rejects outright**, read from its server log. For example,
  HTTP 507 when a benchmark holds the memory a model needs. These are
  otherwise invisible: they are never in flight and never counted.
- **A pause** from `omlxbench pause`.

One oMLX 0.7.0 quirk shapes the design: cancelling a built-in benchmark unloads
its model with an immediate abort, which leaves any *other* request on that
model hanging forever. So when you are talking to the model under test, the
runner lets your request finish first and cancels right after. omlxbench's own
suites own their requests, so they just drop the connection and yield
instantly. Results recorded before a cancel are kept; only the test in progress
is lost.

## Data and analysis

The database is `data/omlxbench.sqlite3`; its schema is versioned and migrated
automatically. [`AGENTS.md`](AGENTS.md) documents the tables and views, the
caveats worth knowing before drawing conclusions (single trials, sampling, BFCL
grading, prompt-length effects) and example queries. It is written so you can
point an AI agent at this repository and ask it which model to use.

## Things to know

- **Uploads to omlx.ai.** oMLX uploads every completed built-in run to omlx.ai
  automatically, with your hardware details and a hardware-derived ID. There is
  no setting to turn it off. The upload is currently blocked by Cloudflare;
  each run records whether it succeeded.
- **Code execution.** humaneval, mbpp and livecodebench run model-generated
  code on your Mac (in oMLX, with resource limits).
- **The `context` benchmark** writes a new `max_context_window` into the
  model's settings. The runner restores the previous value afterwards.
- **Sampling.** Thinking models loop under greedy decoding, so the accuracy
  and tool suites use each model's saved sampling settings. Save each model's
  recommended temperature, top_p and top_k (from its `generation_config.json`)
  in oMLX first. Agents that send no sampling parameters, like Hermes, then get
  the same values.

## License

MIT; see [LICENSE](LICENSE). Bundled test data keeps its own license:

- BFCL v3: Apache 2.0.
- The CPython standard-library excerpts: PSF.
