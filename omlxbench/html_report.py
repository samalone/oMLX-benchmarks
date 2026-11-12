"""A self-contained HTML report: sortable comparison tables built from the digest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from html import escape

HIGH, LOW = "high", "low"


@dataclass
class Col:
    label: str
    better: str | None = None  # HIGH, LOW, or None for columns with no "best"
    nd: int = 1                # decimal places
    unit: str = ""             # appended to the displayed value, e.g. "%"
    title: str = ""            # header tooltip


@dataclass
class Cell:
    value: float | None
    title: str = ""


def _k(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


_SECTIONS = ("perf", "accuracy", "tools", "agent_turns", "pulse_memory", "context")
ENGINE_NAMES = {"omlx": "oMLX", "tensorfold": "TensorFold"}


def _rows(d: dict) -> list[dict]:
    """One row per model × engine × settings fingerprint, with its results."""
    rows = []
    for m in d["models"]:
        keys = []
        for section in _SECTIONS:
            for r in m[section]:
                k = (r.get("engine", "omlx"), r["settings"])
                if k not in keys:
                    keys.append(k)
        keys.sort(key=lambda k: (k[0] != "omlx", k[0]))  # oMLX first, order kept within
        for engine, fp in keys:
            rows.append({
                "model": m, "engine": engine, "settings": fp,
                **{s: [r for r in m[s] if r["settings"] == fp
                       and r.get("engine", "omlx") == engine] for s in _SECTIONS},
            })
    return rows


def _single(r: dict, pp: int) -> dict | None:
    return next((p for p in r["perf"] if p["test"] == "single" and p["pp"] == pp), None)


def _stream(r: dict, pp: int) -> dict | None:
    return next((p for p in r["perf"] if p["test"] == "stream" and p["pp"] == pp), None)


def _test(r: dict, test: str) -> dict | None:
    return next((p for p in r["perf"] if p["test"] == test), None)


def _tools_overall(r: dict) -> Cell:
    n = sum(t["n"] for t in r["tools"])
    if not n:
        return Cell(None)
    correct = sum(t["n"] * (t["accuracy"] or 0) for t in r["tools"])
    return Cell(100 * correct / n, f"{correct:.0f} of {n} cases")


def _pct(a: dict | None) -> Cell:
    if a is None or a["accuracy"] is None:
        return Cell(None)
    note = f"n={a['n']}"
    if a.get("truncated"):
        note += f", {a['truncated']} truncated"
    return Cell(100 * a["accuracy"], note)


def _rate(p: dict | None, key: str) -> Cell:
    if p is None or p.get(key) is None:
        return Cell(None)
    return Cell(100 * p[key], f"{round(p[key] * p['n'])} of {p['n']} requests")


def _get(p: dict | None, key: str, scale: float = 1) -> Cell:
    v = None if p is None else p.get(key)
    if key == "peak_mem_gb" and not v:
        v = None  # the digest reports a missing peak memory as 0
    return Cell(None if v is None else v * scale)


def _tables(d: dict) -> list[tuple[str, str, list[Col], list[tuple[dict, list[Cell]]]]]:
    rows = _rows(d)
    pps = sorted({p["pp"] for r in rows for p in r["perf"] if p["test"] == "single"})
    batches = sorted({p["test"] for r in rows for p in r["perf"]
                      if p["test"].startswith("batch")}, key=lambda t: int(t[5:]))
    stream_pps = sorted({p["pp"] for r in rows for p in r["perf"] if p["test"] == "stream"})
    suites = sorted({a["suite"] for r in rows for a in r["accuracy"]})
    cats = sorted({t["category"] for r in rows for t in r["tools"]})
    tables = []

    def first(xs):
        return xs[0] if xs else None

    def suite(r, s):
        return next((a for a in r["accuracy"] if a["suite"] == s), None)

    def cat(r, c):
        return next((t for t in r["tools"] if t["category"] == c), None)

    # Pick representative prompt lengths for the overview: 4k for decode, the
    # longest agentic-sized prompt (≤32k) that exists for prefill and memory.
    decode_pp = 4096 if 4096 in pps else first(pps)
    long_pp = max((p for p in pps if p <= 32768), default=None)

    cols = [Col("Bits", nd=0), Col("Size GB", LOW, 1, title="Weights on disk"),
            Col("Warm step s", LOW, 1, title="Agent session: mean time to first token per "
                "step after the first, with prefix caching (the best guide to how Hermes feels)"),
            Col("Cold start s", LOW, 1, title="Agent session: time to first token for the "
                "~20k-token first prompt")]
    if decode_pp:
        cols.append(Col(f"Decode tok/s @{_k(decode_pp)}", HIGH, 1))
    if long_pp:
        cols += [Col(f"Prefill tok/s @{_k(long_pp)}", HIGH, 0),
                 Col(f"Peak GB @{_k(long_pp)}", LOW, 1, title="MLX peak memory")]
    cols += [Col("Max context", HIGH, 0, "k", title="Largest prompt actually prefilled "
                 "(thousands of tokens)"),
             Col("Tool calls", HIGH, 1, "%", title="BFCL v3, all categories pooled"),
             Col("Pulse memory", HIGH, 0, "%", title="Share of Pulse memory requests whose "
                 "JSON reply Pulse accepts")]
    cols += [Col(s, HIGH, 1, "%") for s in suites]
    body = []
    for r in rows:
        # v_context keeps every measurement, oldest first: use the newest.
        m, at = r["model"], first(r["agent_turns"])
        ctx = r["context"][-1] if r["context"] else None
        cells = [Cell(m.get("quant_bits")), Cell(m.get("size_gb")),
                 _get(at, "warm_ttft_s"), _get(at, "cold_ttft_s")]
        if decode_pp:
            cells.append(_get(_single(r, decode_pp), "tg_tps"))
        if long_pp:
            p = _single(r, long_pp)
            cells += [_get(p, "pp_tps"), _get(p, "peak_mem_gb")]
        cells += [_get(ctx, "verified_tokens", 1 / 1024), _tools_overall(r),
                  _rate(first(r["pulse_memory"]), "pulse_ok")]
        cells += [_pct(suite(r, s)) for s in suites]
        body.append((r, cells))
    tables.append(("Overview", "One row per model, engine and acceleration settings. "
                   "Hover a header for what it measures.", cols, body))

    engines_of: dict[str, set] = {}
    for r in rows:
        engines_of.setdefault(r["model"]["model_id"], set()).add(r["engine"])
    multi = {m for m, es in engines_of.items() if len(es) > 1}
    if multi:
        s_decode = 4096 if 4096 in stream_pps else first(stream_pps)
        s_long = max((p for p in stream_pps if p <= 32768), default=None)
        cols = []
        if s_decode:
            cols.append(Col(f"Decode tok/s @{_k(s_decode)}", HIGH, 1,
                            title="Streamed test, timed by omlxbench (same on every engine)"))
        if s_long:
            cols += [Col(f"Decode tok/s @{_k(s_long)}", HIGH, 1),
                     Col(f"Prefill tok/s @{_k(s_long)}", HIGH, 0),
                     Col(f"TTFT s @{_k(s_long)}", LOW, 1, title="Nothing cached")]
        cols += [Col("Warm step s", LOW, 1, title="Agent session: mean time to first token per "
                     "step after the first, with prefix caching"),
                 Col("Cold start s", LOW, 1, title="Agent session: ~20k-token first prompt"),
                 Col("Tool calls", HIGH, 1, "%", title="BFCL v3, all categories pooled"),
                 Col("Pulse memory", HIGH, 0, "%")]
        body = []
        for r in rows:
            if r["model"]["model_id"] not in multi:
                continue
            at = first(r["agent_turns"])
            cells = []
            if s_decode:
                cells.append(_get(_stream(r, s_decode), "tg_tps"))
            if s_long:
                p = _stream(r, s_long)
                cells += [_get(p, "tg_tps"), _get(p, "pp_tps"), _get(p, "ttft_ms", 1 / 1000)]
            cells += [_get(at, "warm_ttft_s"), _get(at, "cold_ttft_s"), _tools_overall(r),
                      _rate(first(r["pulse_memory"]), "pulse_ok")]
            body.append((r, cells))
        body.sort(key=lambda rc: rc[0]["model"]["model_id"])
        tables.append(("Engine comparison", "Models measured on more than one engine, one row "
                       "per engine. Speed columns come from the streamed test, which every "
                       "engine runs the same way; each engine runs with its own defaults "
                       "(TensorFold's include speculative drafts) and the same sampling.",
                       cols, body))

    if stream_pps:
        cols = []
        for pp in stream_pps:
            cols += [Col(f"Decode @{_k(pp)}", HIGH, 1, title="tokens/s"),
                     Col(f"Prefill @{_k(pp)}", HIGH, 0, title="tokens/s"),
                     Col(f"TTFT s @{_k(pp)}", LOW, 1)]
        body = []
        for r in rows:
            cells = []
            for pp in stream_pps:
                p = _stream(r, pp)
                cells += [_get(p, "tg_tps"), _get(p, "pp_tps"), _get(p, "ttft_ms", 1 / 1000)]
            body.append((r, cells))
        tables.append(("Streamed speed", "One streamed request per prompt length, nothing "
                       "cached, timed by omlxbench from the client: comparable across engines, "
                       "not with oMLX's own figures below.", cols, body))

    cols = [Col("Cold TTFT s", LOW, 1, title="Session start: first prompt, nothing cached"),
            Col("Cold prefill tok/s", HIGH, 0),
            Col("Warm TTFT s", LOW, 1, title="Mean over later steps, prefix cache in use"),
            Col("Warm new tokens", None, 0, title="Mean uncached tokens per later step"),
            Col("Warm prefill tok/s", HIGH, 0),
            Col("Last step TTFT s", LOW, 1, title="Final step, at the full session length")]
    if 65536 in pps:
        cols.append(Col("Decode tok/s @64k", HIGH, 1, title="Single-request decode at pp65536"))
    body = []
    for r in rows:
        at = first(r["agent_turns"])
        if at is None:
            continue
        cells = [_get(at, "cold_ttft_s"), _get(at, "cold_prefill_tps"), _get(at, "warm_ttft_s"),
                 _get(at, "warm_new_tokens"), _get(at, "warm_prefill_tps"), _get(at, "last_ttft_s")]
        if 65536 in pps:
            cells.append(_get(_single(r, 65536), "tg_tps"))
        body.append((r, cells))
    tables.append(("Agent sessions", "A fixed Hermes-shaped session (~20k-token system prompt "
                   "with 40 tools, growing to ~64k) replayed per model.", cols, body))

    for title, note, key, better, nd in [
        ("Prefill speed", "Prompt processing, tokens/s, single request.", "pp_tps", HIGH, 0),
        ("Decode speed", "Generation, tokens/s, single request, 128 tokens.", "tg_tps", HIGH, 1),
        ("Time to first token", "Seconds, single request, nothing cached.", "ttft_ms", LOW, 1),
        ("Peak memory", "MLX peak memory in GB, single request.", "peak_mem_gb", LOW, 1),
    ]:
        scale = 1 / 1000 if key == "ttft_ms" else 1
        cols = [Col(f"pp {_k(pp)}", better, nd) for pp in pps]
        body = [(r, [_get(_single(r, pp), key, scale) for pp in pps]) for r in rows]
        tables.append((title, note + " Columns are prompt lengths.", cols, body))

    if batches:
        cols = [Col("1 req tok/s", HIGH, 1, title="Single request decode at pp1024")]
        cols += [Col(f"{t[5:]} req tok/s", HIGH, 1, title="Aggregate decode across all "
                     "concurrent requests at pp1024") for t in batches]
        cols += [Col(f"{batches[-1][5:]} req speedup", HIGH, 2, "×"),
                 Col(f"{batches[-1][5:]} req TTFT s", LOW, 1, title="Mean time to first token")]
        body = []
        for r in rows:
            one, top = _single(r, 1024), _test(r, batches[-1])
            cells = [_get(one, "tg_tps")] + [_get(_test(r, t), "tg_tps") for t in batches]
            speedup = (top["tg_tps"] / one["tg_tps"]
                       if one and top and one["tg_tps"] and top["tg_tps"] else None)
            cells += [Cell(speedup), _get(top, "ttft_ms", 1 / 1000)]
            body.append((r, cells))
        tables.append(("Concurrent throughput", "Overall decode throughput with several "
                       "requests in flight at once (pp1024).", cols, body))

    if cats:
        cols = [Col("Overall", HIGH, 1, "%")] + [Col(c, HIGH, 1, "%") for c in cats]
        cols += [Col("Avg s/case", LOW, 1), Col("Avg tokens/case", LOW, 0,
                                                 title="Completion tokens, including thinking")]
        def mean(ts, key):
            # Weighted by cases, over categories that report it: failed requests
            # (e.g. a model that never loaded) have no timing or token counts.
            ts = [t for t in ts if t[key] is not None]
            n = sum(t["n"] for t in ts)
            return Cell(sum(t["n"] * t[key] for t in ts) / n if n else None)

        body = []
        for r in rows:
            cells = [_tools_overall(r)] + [_pct(cat(r, c)) for c in cats]
            cells += [mean(r["tools"], "avg_time_s"), mean(r["tools"], "avg_completion_tokens")]
            body.append((r, cells))
        tables.append(("Tool calling", "BFCL v3 through the chat API with thinking on. "
                       "Hover a score for the sample size.", cols, body))

    cols = [Col("Pulse accepts", HIGH, 0, "%", title="Reply is a JSON object with an items "
                "list, in time: what Pulse needs to keep anything"),
            Col("Sensible", HIGH, 0, "%", title="Also well-formed items that fit the chat "
                "(loose check)"),
            Col("Cases failing", LOW, 0, title="Distinct chats with at least one rejected reply"),
            Col("Avg s", LOW, 1), Col("Max s", LOW, 1, title="Pulse gives up after 120 s")]
    body = []
    for r in rows:
        pm = first(r["pulse_memory"])
        if pm is None:
            continue
        body.append((r, [_rate(pm, "pulse_ok"), _rate(pm, "content_ok"),
                         _get(pm, "cases_with_failures"), _get(pm, "avg_time_s"),
                         _get(pm, "max_time_s")]))
    tables.append(("Pulse memory", "Hermes's Pulse plugin asking for memories as JSON "
                   "(response_format json_object), synthetic chats, each sent several times. "
                   "Hover a score for the number of requests.", cols, body))

    if suites:
        cols = [Col(s, HIGH, 1, "%") for s in suites]
        body = [(r, [_pct(suite(r, s)) for s in suites]) for r in rows]
        tables.append(("Intelligence", "Accuracy with each model's own sampling. Gaps of a "
                       "few points are ties. Hover a score for n and truncations.", cols, body))

    return tables


def _fmt(v: float, col: Col) -> str:
    return f"{v:,.{col.nd}f}{col.unit}"


def _table_html(cols: list[Col], body: list[tuple[dict, list[Cell]]]) -> str:
    body = [(r, cells) for r, cells in body if any(c.value is not None for c in cells)]
    if not body:
        return "<p class=empty>No data yet.</p>"
    # The best cells are highlighted by the page script, over the visible rows only.
    out = ["<table><thead><tr><th data-type=text>Model</th>"]
    for col in cols:
        arrow = {HIGH: " ↑", LOW: " ↓"}.get(col.better, "")
        tip = col.title or ""
        if col.better:
            tip = (tip + " — " if tip else "") + ("higher" if col.better == HIGH else "lower") \
                + " is better"
        out.append(f'<th data-better="{col.better or ""}" data-nd={col.nd} title="{escape(tip)}">'
                   f"{escape(col.label)}<span class=dir>{arrow}</span></th>")
    out.append("</tr></thead><tbody>")
    for r, cells in body:
        m = r["model"]
        tag = "" if r["settings"] == "baseline" else f' <span class=fp>{escape(r["settings"])}</span>'
        engine = r.get("engine", "omlx")
        if engine != "omlx":
            tag = (f' <span class=engine>{escape(ENGINE_NAMES.get(engine, engine))}</span>'
                   + tag)
        gone = not m.get("installed", True)
        if gone:
            tag += " <span class=gone-tag>removed</span>"
        out.append(f'<tr{" class=gone" if gone else ""}><td class=model '
                   f'data-v="{escape(m["model_id"])} {escape(engine)}">'
                   f'{escape(m["model_id"])}{tag}</td>')
        for col, c in zip(cols, cells):
            if c.value is None:
                out.append("<td class=na>—</td>")
                continue
            title = f' title="{escape(c.title)}"' if c.title else ""
            out.append(f'<td data-v="{c.value}"{title}>{_fmt(c.value, col)}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render(d: dict, generated: datetime, version: str | None = None) -> str:
    """With a `version`, the page is served live: it polls /version and reloads
    itself when the results change, keeping its sort order, toggle and scroll."""
    hw = d.get("hardware")
    engines = " · ".join(f"{ENGINE_NAMES.get(e, e)} {escape(str(v))}" for e, v in
                         ((hw or {}).get("engines") or {"omlx": (hw or {}).get("omlx")}).items())
    sub = (f"{escape(hw['chip'])}, {hw['memory_gb']} GB, {hw['gpu_cores']} GPU cores · "
           f"macOS {escape(hw['macos'])} · {engines}" if hw else "")
    sections = []
    for title, note, cols, body in _tables(d):
        sections.append(f"<section><h2>{escape(title)}</h2><p class=note>{escape(note)}</p>"
                        f"<div class=scroll>{_table_html(cols, body)}</div></section>")
    gone = sum(not m.get("installed", True) for m in d["models"])
    toggle = (f'<label class=toggle><input type=checkbox id=show-gone> Show {gone} removed '
              f'model{"" if gone == 1 else "s"} (no longer on disk)</label>' if gone else "")
    live = (" · <span class=live>updates when new results arrive</span>" if version else "")
    script = _LIVE_SCRIPT.replace("__VERSION__", json.dumps(version)) if version else ""
    return _PAGE.format(generated=generated.strftime("%Y-%m-%d %H:%M"), sub=sub, live=live,
                        toggle=toggle, sections="\n".join(sections), live_script=script)


_PAGE = """<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>oMLX Benchmarks</title>
<style>
:root {{
  --bg: #fbfbfa; --fg: #1d1d1f; --muted: #6e6e73; --line: #e3e3e0; --head: #f2f2ef;
  --best-bg: #d9f2df; --best-fg: #0f5c25; --hover: #f0f4fa; --tag: #e8e3f7; --tag-fg: #4b3a8c;
  --eng: #fdecd2; --eng-fg: #7a4a00;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #161618; --fg: #ececee; --muted: #9a9aa0; --line: #2e2e33; --head: #202024;
    --best-bg: #17432a; --best-fg: #9fe8b6; --hover: #1e2430; --tag: #2e2747; --tag-fg: #c9bdf5;
    --eng: #46320f; --eng-fg: #f5cf8e;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif; }}
main {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 36px 0 2px; }}
.sub, .note {{ color: var(--muted); margin: 0 0 10px; }}
.legend {{ color: var(--muted); font-size: 13px; }}
.legend .best {{ padding: 1px 6px; border-radius: 4px; }}
.scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th {{ background: var(--head); font-weight: 600; text-align: right; cursor: pointer;
  user-select: none; }}
th:first-child, td.model {{ text-align: left; }}
th:hover {{ color: #0a64d8; }}
th .dir {{ color: var(--muted); font-weight: 400; }}
th[aria-sort=ascending]::after {{ content: " ▲"; font-size: 10px; }}
th[aria-sort=descending]::after {{ content: " ▼"; font-size: 10px; }}
td {{ text-align: right; }}
td.model {{ font-weight: 500; white-space: normal; min-width: 240px; }}
th:first-child, td.model {{ position: sticky; left: 0; z-index: 1; background: var(--bg);
  box-shadow: 1px 0 0 var(--line); }}
th:first-child {{ background: var(--head); }}
td.na {{ color: var(--muted); }}
tbody tr:hover td {{ background: var(--hover); }}
tbody tr:last-child td {{ border-bottom: 0; }}
.best, tbody tr:hover td.best {{ background: var(--best-bg); color: var(--best-fg); font-weight: 650; }}
.engine {{ font-size: 11px; background: var(--eng); color: var(--eng-fg); padding: 1px 6px;
  border-radius: 4px; font-weight: 500; }}
.fp {{ font-size: 11px; background: var(--tag); color: var(--tag-fg); padding: 1px 6px;
  border-radius: 4px; font-weight: 500; }}
.empty {{ padding: 12px; margin: 0; color: var(--muted); }}
.toggle {{ display: inline-flex; gap: 6px; align-items: center; font-size: 13px; cursor: pointer; }}
body:not(.show-gone) tr.gone {{ display: none; }}
tr.gone td {{ color: var(--muted); }}
.live::before {{ content: "● "; color: #2f9e5a; }}
.gone-tag {{ font-size: 11px; border: 1px solid var(--line); color: var(--muted);
  padding: 0 6px; border-radius: 4px; font-weight: 500; }}
</style></head>
<body><main>
<h1>oMLX benchmarks</h1>
<p class=sub>{sub}<br>Generated {generated}{live}</p>
<p class=legend><span class=best>Highlighted</span> cells are the best in their column among the models shown.
Click a column head to sort; click again to reverse. Speed figures are single trials,
so differences under ~5% are noise.</p>
{toggle}
{sections}
</main>
<script>
// Highlight the best value in each column among the rows currently shown,
// rounded as displayed so values that look identical tie.
function highlightBest() {{
  const showGone = document.body.classList.contains("show-gone");
  document.querySelectorAll("table").forEach(table => {{
    const rows = [...table.tBodies[0].rows].filter(r => showGone || !r.classList.contains("gone"));
    [...table.tHead.rows[0].cells].forEach((th, i) => {{
      const better = th.dataset.better, nd = Number(th.dataset.nd);
      const cells = [...table.tBodies[0].rows].map(r => r.cells[i]);
      cells.forEach(c => c.classList.remove("best"));
      if (!better) return;
      const val = c => Number(Number(c.dataset.v).toFixed(nd));
      const shown = rows.map(r => r.cells[i]).filter(c => c.dataset.v !== undefined);
      const vals = shown.map(val);
      if (new Set(vals).size < 2) return;
      const best = better === "high" ? Math.max(...vals) : Math.min(...vals);
      shown.forEach(c => {{ if (val(c) === best) c.classList.add("best"); }});
    }});
  }});
}}
const showGone = document.getElementById("show-gone");
if (showGone) showGone.addEventListener("change", () => {{
  document.body.classList.toggle("show-gone", showGone.checked);
  highlightBest();
}});
highlightBest();

function sortTable(table, i, dir) {{
  const heads = [...table.tHead.rows[0].cells], th = heads[i];
  heads.forEach(h => h.removeAttribute("aria-sort"));
  th.setAttribute("aria-sort", dir);
  const sign = dir === "ascending" ? 1 : -1;
  const tbody = table.tBodies[0];
  const rows = [...tbody.rows];
  rows.sort((a, b) => {{
    const x = a.cells[i].dataset.v, y = b.cells[i].dataset.v;
    if (x === undefined || y === undefined)  // missing values always sink
      return (x === undefined) - (y === undefined);
    const nx = Number(x), ny = Number(y);
    const c = th.dataset.type === "text" || isNaN(nx) || isNaN(ny) ? x.localeCompare(y) : nx - ny;
    return sign * c;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

document.querySelectorAll("table").forEach(table => {{
  const heads = [...table.tHead.rows[0].cells];
  heads.forEach((th, i) => th.addEventListener("click", () => {{
    // First click puts the best values on top (or A→Z for text).
    const first = th.dataset.better === "low" || th.dataset.type === "text" ? "ascending" : "descending";
    const dir = th.getAttribute("aria-sort") === first
      ? (first === "ascending" ? "descending" : "ascending") : first;
    sortTable(table, i, dir);
  }}));
}});
</script>
{live_script}
</body></html>
"""

# Served pages only. The view state lives in sessionStorage, so a reload (ours
# or the user's) comes back sorted, toggled and scrolled as it was.
_LIVE_SCRIPT = """<script>
const VERSION_SHOWN = __VERSION__, STATE_KEY = "omlxbench-view";

function saveView() {
  const sorts = [...document.querySelectorAll("table")].map(t => {
    const i = [...t.tHead.rows[0].cells].findIndex(h => h.hasAttribute("aria-sort"));
    return i < 0 ? null : [i, t.tHead.rows[0].cells[i].getAttribute("aria-sort")];
  });
  const gone = document.getElementById("show-gone");
  try {
    sessionStorage.setItem(STATE_KEY, JSON.stringify(
      {sorts, showGone: !!(gone && gone.checked), scroll: [scrollX, scrollY]}));
  } catch (e) {}
}

function restoreView() {
  let v;
  try { v = JSON.parse(sessionStorage.getItem(STATE_KEY)); } catch (e) {}
  if (!v) return;
  const tables = document.querySelectorAll("table");
  (v.sorts || []).forEach((s, n) => { if (s && tables[n] && tables[n].tHead.rows[0].cells[s[0]])
    sortTable(tables[n], s[0], s[1]); });
  const gone = document.getElementById("show-gone");
  if (gone && v.showGone) { gone.checked = true; document.body.classList.add("show-gone"); highlightBest(); }
  if (v.scroll) scrollTo(v.scroll[0], v.scroll[1]);
}

async function checkForNewResults() {
  if (document.hidden) return;
  try {
    const r = await fetch("version", {cache: "no-store"});
    if (r.ok && (await r.text()).trim() !== VERSION_SHOWN) { saveView(); location.reload(); }
  } catch (e) {}  // runner restarting: try again next time
}

restoreView();
addEventListener("pagehide", saveView);
document.addEventListener("visibilitychange", checkForNewResults);
setInterval(checkForNewResults, 15000);
</script>"""
