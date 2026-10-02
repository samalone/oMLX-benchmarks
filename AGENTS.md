# omlxbench — notes for agents

This project records benchmark results from a local [oMLX](https://omlx.ai) server
(an MLX inference server for Apple Silicon) so they can be used to compare models
and make recommendations for this specific machine and its owner.

## Quick start

```sh
uv run omlxbench report --json   # compact per-model digest; start here
uv run omlxbench report          # same, as markdown tables
uv run omlxbench status          # is the runner up, what is it doing, what's missing
uv run omlxbench plan            # every run still needed to satisfy targets.toml
```

For anything the digest doesn't answer, query the database directly:
`sqlite3 data/omlxbench.sqlite3` (read-only use is always safe; the runner uses WAL).

## Operating the runner

- The runner is installed as a launchd agent (`launchd/install.sh`; `launchd/install.sh
  uninstall` removes it), so it is normally already running. Its log is `data/runner.log`.
  Restart it after code changes with `launchctl kickstart -k gui/$(id -u)/com.samalone.omlxbench`.
- `uv run omlxbench run`: long-running loop. It starts a benchmark only after
  `quiet_minutes` (in `targets.toml`) with no API traffic, and cancels it when the
  user starts using the server.
- `uv run omlxbench pause [30m|2h|1d]` and `uv run omlxbench resume`.
- `targets.toml` defines what *should* be measured. Removing an entry never
  deletes data; it only stops future runs.
- `uv run omlxbench requeue --model GLOB --spec GLOB [--failed] [--before-version X] --yes`
  forgets "done" status so tests are measured again. Stored results are kept.
- Benchmarks take over the whole server: every loaded model is unloaded first.
  Never start one while the user may be using the server. Use the runner, which
  waits for idle, rather than calling oMLX's benchmark endpoints directly.

## Data model

| Table / view | What it holds |
|---|---|
| `v_latest_perf` | Newest speed result per model × test × settings fingerprint. **Use this for comparisons.** |
| `v_perf` | Every speed result ever recorded (history, repeats). |
| `v_latest_accuracy`, `v_accuracy` | Same for intelligence (accuracy) suites. |
| `v_context` | Largest prompt the machine could actually prefill, per model. |
| `v_tools`, `tool_cases` | Tool calling (BFCL v3, run by omlxbench through the chat API, thinking on): accuracy per category, and each case with the parsed calls, the reason for a failure, and the raw response. |
| `accuracy_questions` | Per-question results: expected, predicted, raw response, tokens, time. |
| `model_snapshots` | Model metadata: architecture (`arch`), `quant_bits`, `size_bytes`, `native_context`, full `settings_json` and `config_json`. |
| `environments` | Hardware, macOS, oMLX and mlx-lm versions, perf-relevant global settings. |
| `runs`, `run_events` | Every oMLX benchmark run (including cancelled ones) and its full event stream. |
| `usage_snapshots` | Daily copies of oMLX's 30-day usage history: which models the user actually uses, with token counts and cache hit rates. |
| `spec_status` | Planner bookkeeping: which tests are done or have been given up on. |

Every result table has `raw_json`, the verbatim oMLX object. Fields without a
column (for example `system_metrics`: CPU/GPU utilization, thermal state, memory
footprint) are reachable with `json_extract(raw_json, '$.system_metrics.gpu.util_avg')`.

### Perf columns

- **Single-request tests** (`test_type = 'single'`):
  - `processing_tps`: prompt processing (prefill) speed in tokens/s.
  - `gen_tps`: decode speed in tokens/s. It is NULL if fewer than 16 tokens were generated.
  - `ttft_ms`: time to first token.
  - `e2e_latency_s`: total time for the request.
  - `peak_memory_bytes`: MLX peak memory.
  - Prompt length is `pp`; generation length is `tg`, always 128.
- **Batch tests** (`test_type = 'batch'`):
  - `batch_size` concurrent requests at pp1024.
  - `gen_tps` and `processing_tps` are *aggregate* throughput across all requests.
  - `avg_ttft_ms` is the mean time to first token.
- `context_profile` is the corpus the prompt was built from. It only matters when
  speculative features (MTP, DFlash) are enabled.

### Accuracy columns

- `accuracy`: fraction correct, from 0 to 1.
- `total`: number of questions.
- `sample_size`: requested size; 0 = full dataset.
- `thinking_used`: whether thinking/reasoning output was used.
- `truncated_count`: answers cut off at the token limit. A high count means the score understates the model.
- Samples are deterministic (seed 42): the same suite and n always uses the same questions, so scores at the same n are directly comparable across models.

### Settings fingerprint

`settings_fingerprint` lists the active acceleration features. Possible values include
`baseline`, `turboquant_kv_4bit` and `dflash,lightning_mtp`. Results with different
fingerprints are different configurations of the same model; compare like with like,
or treat each fingerprint as a separate option. Full settings are in
`model_snapshots.settings_json`.

## Caveats when drawing conclusions

- **Sampling.** Intelligence (tier 4) and tool-calling results use each model's own
  recommended sampling (saved in oMLX's per-model settings, which is also what Hermes
  gets), not greedy decoding. Scores therefore vary a little between runs; treat gaps of
  a few points as ties. Older rows with `sampling_profile = 'deterministic'` are greedy.
- **Tool calling** is graded like BFCL: exact function, no unknown arguments, values
  matching an accepted answer. A failure `reason` containing "parser did not extract
  it" means the model produced a tool call as text that oMLX's parser missed. That is
  a model/template/parser compatibility problem worth knowing about separately from
  model quality.

- **Each speed figure is a single trial.** Differences under ~5% are noise. If repeated
  runs exist in `v_perf`, use the median.
- **Speed vs. prompt length.** Prefill speed drops and TTFT grows roughly linearly with
  prompt length. Agentic and coding use typically means 8k–32k-token prompts, so weight
  those tests over pp1024 if they exist.
- **Memory.** `size_bytes` is the weights on disk. `peak_memory_bytes` at the relevant
  prompt length is a better guide to what fits alongside other work. Check
  `environments.memory_gb`.
- **VLM entries.** Vision-language models are benchmarked through their text path.
- **Coverage.** Accuracy numbers come from small samples (e.g. mmlu n=300 is about
  ±3 points), so close scores are ties.
- **Truncation.** An intelligence run with thinking enabled but a low token budget may
  truncate. Check `truncated_count` before trusting a low score.
- **Cancelled runs.** They still contribute the results they finished, since each
  result is recorded as it arrives.

## Example queries

Decode speed at 4k context, fastest first, with size and quantization:

```sql
SELECT model_id, quant_bits, round(size_bytes/1e9,1) AS gb, settings_fingerprint,
       round(gen_tps,1) AS tg, round(processing_tps) AS pp, round(ttft_ms) AS ttft
FROM v_latest_perf WHERE test_type='single' AND pp=4096 ORDER BY gen_tps DESC;
```

Speed against accuracy (mmlu), one row per model:

```sql
SELECT p.model_id, round(p.gen_tps,1) AS tg_4k, round(100*a.accuracy,1) AS mmlu
FROM v_latest_perf p JOIN v_latest_accuracy a
  ON a.model_id = p.model_id AND a.settings_fingerprint = p.settings_fingerprint
WHERE p.test_type='single' AND p.pp=4096 AND a.suite='mmlu'
ORDER BY mmlu DESC;
```

Which models the user actually used over the last 30 days (latest snapshot):

```sql
SELECT json_extract(m.value,'$.model_id') AS model,
       json_extract(m.value,'$.requests') AS requests,
       json_extract(m.value,'$.prompt_tokens') / json_extract(m.value,'$.requests') AS avg_prompt_tokens,
       round(json_extract(m.value,'$.cache_efficiency'),2) AS cache_hit
FROM usage_snapshots u, json_each(u.json,'$.models') m
WHERE u.id = (SELECT max(id) FROM usage_snapshots)
ORDER BY requests DESC;
```

The typical prompt size from that query tells you which `pp` row matters most.

## Recommending

Ask what the user wants to optimize for: speed, quality, memory headroom or long
context. Answer from the data, not from general knowledge about the model families:
results on this machine and oMLX version are the point of this project.

If the data is missing for a fair comparison, say so and suggest adding a test to
`targets.toml`. The runner will fill it in when the server is idle.
