"""A self-contained HTML report: sortable comparison tables built from the digest."""

from __future__ import annotations

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


def _rows(d: dict) -> list[dict]:
    """One row per model × settings fingerprint, with that fingerprint's results."""
    rows = []
    for m in d["models"]:
        fps = []
        for section in ("perf", "accuracy", "tools", "agent_turns", "context"):
            for r in m[section]:
                if r["settings"] not in fps:
                    fps.append(r["settings"])
        for fp in fps:
            rows.append({
                "model": m, "settings": fp,
                **{s: [r for r in m[s] if r["settings"] == fp]
                   for s in ("perf", "accuracy", "tools", "agent_turns", "context")},
            })
    return rows


def _single(r: dict, pp: int) -> dict | None:
    return next((p for p in r["perf"] if p["test"] == "single" and p["pp"] == pp), None)


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


def _get(p: dict | None, key: str, scale: float = 1) -> Cell:
    v = None if p is None else p.get(key)
    return Cell(None if v is None else v * scale)


def _tables(d: dict) -> list[tuple[str, str, list[Col], list[tuple[dict, list[Cell]]]]]:
    rows = _rows(d)
    pps = sorted({p["pp"] for r in rows for p in r["perf"] if p["test"] == "single"})
    batches = sorted({p["test"] for r in rows for p in r["perf"] if p["test"] != "single"},
                     key=lambda t: int(t[5:]))
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
             Col("Tool calls", HIGH, 1, "%", title="BFCL v3, all categories pooled")]
    cols += [Col(s, HIGH, 1, "%") for s in suites]
    body = []
    for r in rows:
        m, at, ctx = r["model"], first(r["agent_turns"]), first(r["context"])
        cells = [Cell(m.get("quant_bits")), Cell(m.get("size_gb")),
                 _get(at, "warm_ttft_s"), _get(at, "cold_ttft_s")]
        if decode_pp:
            cells.append(_get(_single(r, decode_pp), "tg_tps"))
        if long_pp:
            p = _single(r, long_pp)
            cells += [_get(p, "pp_tps"), _get(p, "peak_mem_gb")]
        cells += [_get(ctx, "verified_tokens", 1 / 1024), _tools_overall(r)]
        cells += [_pct(suite(r, s)) for s in suites]
        body.append((r, cells))
    tables.append(("Overview", "One row per model and acceleration settings. "
                   "Hover a header for what it measures.", cols, body))

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
        body = []
        for r in rows:
            ts = r["tools"]
            n = sum(t["n"] for t in ts) or None
            cells = [_tools_overall(r)] + [_pct(cat(r, c)) for c in cats]
            cells += [Cell(n and sum(t["n"] * (t["avg_time_s"] or 0) for t in ts) / n),
                      Cell(n and sum(t["n"] * (t["avg_completion_tokens"] or 0) for t in ts) / n)]
            body.append((r, cells))
        tables.append(("Tool calling", "BFCL v3 through the chat API with thinking on. "
                       "Hover a score for the sample size.", cols, body))

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
    best = []
    for i, col in enumerate(cols):
        vals = [cells[i].value for _, cells in body if cells[i].value is not None]
        # Rounded, so values that display identically tie for best.
        vals = [round(v, col.nd) for v in vals]
        best.append(None if col.better is None or len(vals) < 2 or len(set(vals)) < 2
                    else (max if col.better == HIGH else min)(vals))
    out = ["<table><thead><tr><th data-type=text>Model</th>"]
    for col in cols:
        arrow = {HIGH: " ↑", LOW: " ↓"}.get(col.better, "")
        tip = col.title or ""
        if col.better:
            tip = (tip + " — " if tip else "") + ("higher" if col.better == HIGH else "lower") \
                + " is better"
        out.append(f'<th data-better="{col.better or ""}" title="{escape(tip)}">'
                   f"{escape(col.label)}<span class=dir>{arrow}</span></th>")
    out.append("</tr></thead><tbody>")
    for r, cells in body:
        m = r["model"]
        tag = "" if r["settings"] == "baseline" else f' <span class=fp>{escape(r["settings"])}</span>'
        out.append(f'<tr><td class=model data-v="{escape(m["model_id"])}">'
                   f'{escape(m["model_id"])}{tag}</td>')
        for i, (col, c) in enumerate(zip(cols, cells)):
            if c.value is None:
                out.append("<td class=na>—</td>")
                continue
            cls = ' class=best' if best[i] is not None and round(c.value, col.nd) == best[i] else ""
            title = f' title="{escape(c.title)}"' if c.title else ""
            out.append(f'<td data-v="{c.value}"{cls}{title}>{_fmt(c.value, col)}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render(d: dict, generated: datetime) -> str:
    hw = d.get("hardware")
    sub = (f"{escape(hw['chip'])}, {hw['memory_gb']} GB, {hw['gpu_cores']} GPU cores · "
           f"macOS {escape(hw['macos'])} · oMLX {escape(hw['omlx'])}" if hw else "")
    sections = []
    for title, note, cols, body in _tables(d):
        sections.append(f"<section><h2>{escape(title)}</h2><p class=note>{escape(note)}</p>"
                        f"<div class=scroll>{_table_html(cols, body)}</div></section>")
    return _PAGE.format(
        generated=generated.strftime("%Y-%m-%d %H:%M"), sub=sub, sections="\n".join(sections))


_PAGE = """<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>oMLX Benchmarks</title>
<style>
:root {{
  --bg: #fbfbfa; --fg: #1d1d1f; --muted: #6e6e73; --line: #e3e3e0; --head: #f2f2ef;
  --best-bg: #d9f2df; --best-fg: #0f5c25; --hover: #f0f4fa; --tag: #e8e3f7; --tag-fg: #4b3a8c;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #161618; --fg: #ececee; --muted: #9a9aa0; --line: #2e2e33; --head: #202024;
    --best-bg: #17432a; --best-fg: #9fe8b6; --hover: #1e2430; --tag: #2e2747; --tag-fg: #c9bdf5;
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
.fp {{ font-size: 11px; background: var(--tag); color: var(--tag-fg); padding: 1px 6px;
  border-radius: 4px; font-weight: 500; }}
.empty {{ padding: 12px; margin: 0; color: var(--muted); }}
</style></head>
<body><main>
<h1>oMLX benchmarks</h1>
<p class=sub>{sub}<br>Generated {generated}</p>
<p class=legend><span class=best>Highlighted</span> cells are the best in their column.
Click a column head to sort; click again to reverse. Speed figures are single trials,
so differences under ~5% are noise.</p>
{sections}
</main>
<script>
document.querySelectorAll("table").forEach(table => {{
  const heads = [...table.tHead.rows[0].cells];
  heads.forEach((th, i) => th.addEventListener("click", () => {{
    const better = th.dataset.better;
    // First click puts the best values on top (or A→Z for text).
    const first = better === "low" || th.dataset.type === "text" ? "ascending" : "descending";
    const dir = th.getAttribute("aria-sort") === first
      ? (first === "ascending" ? "descending" : "ascending") : first;
    heads.forEach(h => h.removeAttribute("aria-sort"));
    th.setAttribute("aria-sort", dir);
    const sign = dir === "ascending" ? 1 : -1;
    const tbody = table.tBodies[0];
    const rows = [...tbody.rows];
    rows.sort((a, b) => {{
      const x = a.cells[i].dataset.v, y = b.cells[i].dataset.v;
      if (x === undefined || y === undefined)  // missing values always sink
        return (x === undefined) - (y === undefined);
      const nx = parseFloat(x), ny = parseFloat(y);
      const c = isNaN(nx) || isNaN(ny) ? x.localeCompare(y) : nx - ny;
      return sign * c;
    }});
    rows.forEach(r => tbody.appendChild(r));
  }}));
}});
</script>
</body></html>
"""
