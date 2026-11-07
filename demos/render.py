#!/usr/bin/env python3
"""Build the dependency-free cmachine demo showcase.

The page is deliberately generated from checked-in JSON records.  It does not
run a model, fetch fonts, or depend on a JavaScript package.  A future
``demos/results/pretrained/summary.json`` is picked up automatically when it
exists; its absence is shown as a scope boundary rather than an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "showcase.html"


def load_json(path: Path, *, optional: bool = False) -> Any:
    if not path.exists():
        if optional:
            return None
        raise FileNotFoundError(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def embedded_json(value: Any) -> str:
    """Serialize JSON so it cannot terminate an application/json script tag."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        encoded.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>cmachine / execution gallery</title>
  <style>
    :root {
      --bg: #071316;
      --panel: rgba(12, 32, 38, .82);
      --panel-strong: #102a31;
      --ink: #e9f7f4;
      --muted: #91b3b1;
      --faint: #527875;
      --line: rgba(143, 219, 210, .16);
      --teal: #72e2d4;
      --teal-deep: #168c8b;
      --purple: #b9a0ff;
      --purple-deep: #7957c8;
      --amber: #f5c776;
      --red: #ff8b8b;
      --green: #82e2a3;
      --shadow: 0 22px 65px rgba(0, 0, 0, .25);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% -5%, rgba(27, 144, 143, .27), transparent 34rem),
        radial-gradient(circle at 90% 8%, rgba(121, 87, 200, .24), transparent 32rem),
        var(--bg);
      font: 15px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: var(--teal); text-decoration: none; border-bottom: 1px solid rgba(114, 226, 212, .35); }
    a:hover { color: #c7fff5; border-bottom-color: currentColor; }
    .shell { width: min(1180px, calc(100% - 36px)); margin: 0 auto; padding-bottom: 90px; }
    .hero { min-height: 610px; padding: 72px 0 48px; display: grid; align-content: center; }
    .eyebrow, .tag {
      color: var(--teal); text-transform: uppercase; letter-spacing: .16em; font-size: 11px; font-weight: 750;
    }
    .eyebrow { display: inline-flex; align-items: center; gap: 9px; }
    .eyebrow::before { content: ""; width: 24px; height: 1px; background: var(--teal); }
    h1, h2, h3, p { margin-top: 0; }
    h1 { max-width: 820px; margin: 22px 0 20px; font-size: clamp(42px, 7vw, 86px); line-height: .98; letter-spacing: -.06em; font-weight: 720; }
    h1 em { color: var(--teal); font-style: normal; }
    h2 { margin-bottom: 12px; font-size: clamp(26px, 3.3vw, 42px); line-height: 1.08; letter-spacing: -.04em; }
    h3 { margin-bottom: 8px; font-size: 18px; letter-spacing: -.015em; }
    .lede { max-width: 680px; color: var(--muted); font-size: 19px; line-height: 1.55; }
    .hero-grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(290px, .85fr); gap: 28px; align-items: end; margin-top: 30px; }
    .hero-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 24px; }
    .hero-badge { padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); background: rgba(255,255,255,.035); font-size: 11px; }
    .quickstart { display: flex; flex-wrap: wrap; gap: 11px 15px; align-items: center; margin-top: 25px; padding: 14px 16px; border: 1px solid rgba(114,226,212,.2); border-radius: 13px; background: rgba(114,226,212,.06); }
    .quickstart code { color: var(--ink); font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .quickstart span:last-child { color: var(--muted); font-size: 13px; }
    .quote-card { position: relative; padding: 26px 28px; border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(145deg, rgba(24, 64, 68, .8), rgba(45, 29, 73, .8)); box-shadow: var(--shadow); }
    .quote-card::before { content: "“"; position: absolute; top: 8px; right: 20px; color: rgba(114, 226, 212, .25); font-size: 80px; line-height: 1; }
    .quote-label { color: var(--amber); text-transform: uppercase; letter-spacing: .14em; font-size: 10px; font-weight: 750; }
    blockquote { margin: 15px 0 18px; max-width: 520px; font-size: 21px; line-height: 1.4; }
    .quote-meta { color: var(--muted); font-size: 12px; }
    .section { padding: 72px 0 10px; }
    .section-heading { max-width: 760px; margin-bottom: 28px; }
    .section-heading p { color: var(--muted); font-size: 16px; }
    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .metric, .panel, .step, .memory-card, .link-card { border: 1px solid var(--line); background: var(--panel); box-shadow: 0 12px 34px rgba(0,0,0,.11); }
    .metric { min-height: 132px; padding: 19px; border-radius: 16px; }
    .metric-value { display: block; margin: 8px 0 3px; color: var(--teal); font-size: 38px; line-height: 1; letter-spacing: -.055em; font-weight: 720; }
    .metric-label { color: var(--muted); font-size: 12px; }
    .two-col { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(280px, .75fr); gap: 18px; margin-top: 18px; }
    .panel { border-radius: 20px; padding: 25px; }
    .panel p { color: var(--muted); }
    .panel strong { color: var(--ink); }
    .kicker-line { display: flex; justify-content: space-between; gap: 14px; align-items: baseline; color: var(--muted); font-size: 12px; }
    .kicker-line b { color: var(--teal); font-size: 20px; }
    .trend-list { display: grid; gap: 15px; margin-top: 22px; }
    .trend-row { display: grid; grid-template-columns: 126px 1fr; gap: 13px; align-items: center; }
    .trend-label { color: var(--muted); font-size: 12px; }
    .trend-label b { display: block; color: var(--ink); font-size: 13px; }
    .bar-stack { display: grid; gap: 4px; }
    .bar-line { display: grid; grid-template-columns: 66px 1fr 68px; gap: 8px; align-items: center; font-size: 10px; color: var(--faint); }
    .bar-track { height: 7px; overflow: hidden; border-radius: 99px; background: rgba(255,255,255,.07); }
    .bar-fill { height: 100%; min-width: 2px; border-radius: inherit; }
    .bar-fill.baseline { background: #668b8a; }
    .bar-fill.updated { background: var(--teal); }
    .bar-fill.later { background: var(--purple); }
    .countertrend { margin-top: 17px; padding: 12px 14px; border-left: 2px solid var(--purple); color: #d5c9ff; background: rgba(121, 87, 200, .1); font-size: 13px; }
    .callout { padding: 17px; border-radius: 13px; background: rgba(114, 226, 212, .075); border: 1px solid rgba(114, 226, 212, .18); }
    .callout p:last-child { margin-bottom: 0; }
    .program-grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(290px, .85fr); gap: 18px; margin-top: 18px; }
    .pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin: 20px 0 9px; }
    .pipeline-node { padding: 10px 12px; border: 1px solid rgba(114,226,212,.25); border-radius: 10px; color: var(--ink); background: rgba(114,226,212,.08); font-size: 12px; }
    .pipeline-node.purple { border-color: rgba(185,160,255,.28); background: rgba(121,87,200,.12); }
    .pipeline-arrow { color: var(--faint); font: 17px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .program-note { margin-bottom: 0; color: var(--muted); font-size: 13px; }
    .program-stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 16px; }
    .program-stat { padding: 13px; border-radius: 12px; background: rgba(255,255,255,.035); }
    .program-stat b { display: block; color: var(--teal); font-size: 20px; }
    .program-stat span { color: var(--muted); font-size: 11px; }
    .steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
    .step { position: relative; min-height: 170px; padding: 17px; border-radius: 16px; }
    .step-num { color: var(--purple); font: 700 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .step h3 { margin-top: 22px; font-size: 16px; }
    .step p { margin-bottom: 0; color: var(--muted); font-size: 13px; }
    .code-panel { margin-top: 16px; padding: 20px; overflow: auto; border: 1px solid var(--line); border-radius: 16px; background: #061012; color: #b6e9df; }
    pre { margin: 0; font: 13px/1.75 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
    .memory-card { overflow: hidden; margin-top: 18px; border-radius: 20px; }
    .memory-head { display: flex; justify-content: space-between; gap: 16px; align-items: end; padding: 23px 25px 15px; }
    .memory-head p { max-width: 620px; margin-bottom: 0; color: var(--muted); }
    .memory-saving { color: var(--amber); font-size: 25px; font-weight: 720; white-space: nowrap; }
    .memory-table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 720px; }
    th, td { padding: 12px 14px; text-align: left; border-top: 1px solid var(--line); vertical-align: top; }
    th { color: var(--faint); font-size: 10px; letter-spacing: .12em; text-transform: uppercase; }
    td { color: var(--muted); font-size: 13px; }
    td.policy { color: var(--ink); font-weight: 700; }
    .answer-chip { display: inline-block; margin: 2px 4px 2px 0; padding: 3px 7px; border-radius: 999px; border: 1px solid rgba(255,255,255,.13); font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .answer-chip.ok { color: var(--green); border-color: rgba(130,226,163,.35); background: rgba(130,226,163,.08); }
    .answer-chip.miss { color: var(--red); border-color: rgba(255,139,139,.35); background: rgba(255,139,139,.08); }
    .memory-foot { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 18px 25px 23px; }
    .mini-stat { padding: 13px; border-radius: 12px; background: rgba(255,255,255,.035); }
    .mini-stat b { display: block; color: var(--ink); font-size: 15px; }
    .mini-stat span { color: var(--muted); font-size: 11px; }
    .plot-shell { overflow: hidden; margin-top: 18px; border: 1px solid var(--line); border-radius: 20px; background: var(--panel); }
    .plot-toolbar { display: flex; flex-wrap: wrap; gap: 15px 20px; align-items: center; padding: 18px 22px; border-bottom: 1px solid var(--line); }
    .control { display: inline-flex; gap: 8px; align-items: center; color: var(--muted); font-size: 12px; }
    select, input[type=range] { accent-color: var(--teal); }
    select { padding: 7px 10px; border: 1px solid var(--line); border-radius: 8px; color: var(--ink); background: #0a2024; }
    input[type=range] { width: min(300px, 44vw); }
    .plot-time { margin-left: auto; color: var(--teal); font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .svg-wrap { padding: 10px 17px 0; }
    #statePlot { display: block; width: 100%; height: auto; min-height: 300px; }
    .plot-axis { fill: var(--faint); font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .plot-grid { stroke: rgba(143,219,210,.12); stroke-width: 1; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 15px; padding: 10px 25px 21px; color: var(--muted); font-size: 12px; }
    .legend-item { display: inline-flex; gap: 7px; align-items: center; }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; }
    .plot-note { padding: 0 25px 21px; color: var(--muted); font-size: 13px; }
    .observations { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }
    .observation { padding: 17px; border-left: 2px solid var(--teal); border-radius: 0 13px 13px 0; background: rgba(114,226,212,.065); }
    .observation b { display: block; margin-bottom: 5px; color: var(--ink); }
    .observation span { color: var(--muted); font-size: 13px; }
    .links { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .link-card { padding: 19px; border-radius: 15px; }
    .link-card p { margin: 7px 0 0; color: var(--muted); font-size: 13px; }
    .scope-strip { margin-top: 19px; padding: 18px 20px; border: 1px solid rgba(245,199,118,.25); border-radius: 14px; color: #e7d3a9; background: rgba(245,199,118,.075); }
    .footer { padding-top: 58px; color: var(--faint); font-size: 12px; }
    @media (max-width: 880px) {
      .hero-grid, .two-col { grid-template-columns: 1fr; }
      .program-grid { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, 1fr); }
      .steps { grid-template-columns: repeat(2, 1fr); }
      .links { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 560px) {
      .shell { width: min(100% - 24px, 1180px); }
      .hero { min-height: auto; padding-top: 46px; }
      .metric-grid, .steps, .memory-foot, .observations, .links { grid-template-columns: 1fr; }
      .memory-head { display: block; }
      .memory-saving { margin-top: 12px; }
      .trend-row { grid-template-columns: 92px 1fr; }
      .bar-line { grid-template-columns: 52px 1fr 58px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="hero">
      <div class="eyebrow">cmachine / evidence gallery</div>
      <h1>Execute. Inspect.<br><em>Branch. Replace.</em></h1>
      <p class="lede">Generate text, save a conversation, rewind a branch, bind an alternative operand program, and inspect both the state and the result. This gallery connects a native Qwen session, a recorded first-layer replacement, a memory policy, and a finite nonlinear state lab.</p>
      <div class="hero-badges"><span class="hero-badge">Pinned Qwen results</span><span class="hero-badge">Runnable C examples</span><span class="hero-badge">Weight-free CPU lab</span></div>
      <div class="quickstart"><span class="tag">Start here</span><code>python3 demos/run.py tour</code><span>Run the local tour, then open this page to inspect the recorded boundaries.</span></div>
      <div class="hero-grid">
        <div>
          <div class="tag">Scope at a glance</div>
          <p class="lede" style="font-size:15px;max-width:560px;margin-top:10px">Historical pretrained records are pinned and verified; the closed-state experiment is freshly reproducible as a CPU fixture. The page keeps those scopes visibly separate.</p>
        </div>
        <div class="quote-card">
          <div class="quote-label">Recorded native answer</div>
          <blockquote id="nativeAnswer">Loading recorded response…</blockquote>
          <div class="quote-meta" id="answerMeta"></div>
        </div>
      </div>
    </header>

    <main>
      <section class="section" id="fresh-panel" style="display:none">
        <div class="section-heading">
          <div class="eyebrow">01 / fresh pretrained lease</div>
          <h2>Save a conversation. Explore both futures.</h2>
          <p>The public C/CUDA examples ran on a pretrained Qwen model. Branch A changes a badge to green; rollback restores the parent; regeneration reproduces A exactly; branch B changes it to yellow and commits.</p>
        </div>
        <div class="metric-grid">
          <div class="metric"><span class="tag">job</span><strong class="metric-value" id="freshJob">—</strong><span class="metric-label">fresh lease identity</span></div>
          <div class="metric"><span class="tag">chat replay</span><strong class="metric-value" id="freshReplay">—</strong><span class="metric-label">same-parent replay</span></div>
          <div class="metric"><span class="tag">lexical</span><strong class="metric-value" id="freshLexical">—</strong><span class="metric-label">full-logit rows</span></div>
          <div class="metric"><span class="tag">coverage</span><strong class="metric-value" id="freshIDs">—</strong><span class="metric-label">candidate IDs covered</span></div>
        </div>
        <div class="two-col">
          <div class="panel"><h3>Model-generated answers</h3><div class="callout"><p><strong>Initial</strong></p><p class="fresh-answer" id="freshInitial">—</p><p><strong>Branch A</strong></p><p class="fresh-answer" id="freshBranchA">—</p><p><strong>Rollback → regenerate A</strong></p><p class="fresh-answer" id="freshReplayText">—</p><p><strong>Rollback → branch B → commit</strong></p><p class="fresh-answer" id="freshBranchB">—</p></div></div>
          <div class="panel"><h3>Exact replacement, accountable cost</h3><p id="freshMeta">—</p><div class="scope-strip" id="freshLimits">—</div><p><a href="results/pretrained/README.md">Inspect the fresh results and source →</a></p><details><summary>Full claims and limits</summary><p id="freshDetails"></p></details></div>
        </div>
      </section>

      <section class="section" id="signal">
        <div class="section-heading">
          <div class="eyebrow">02 / pinned readout</div>
          <h2>The useful signal is a chain of exact boundaries.</h2>
          <p>The gallery starts with what the records can certify mechanically, then lets you inspect where a compact state or learned replacement becomes conditional at the same full-logit consumer.</p>
        </div>
        <div class="metric-grid">
          <div class="metric"><span class="tag">native trace</span><strong class="metric-value" id="protocolCount">—</strong><span class="metric-label">exact protocol entries</span></div>
          <div class="metric"><span class="tag">operand readout</span><strong class="metric-value" id="lexicalCount">—</strong><span class="metric-label">exact full-logit steps</span></div>
          <div class="metric"><span class="tag">conditional KL</span><strong class="metric-value" id="klMetric">—</strong><span class="metric-label">mean reduction, four cases</span></div>
          <div class="metric"><span class="tag">memory</span><strong class="metric-value" id="memoryMetric">—</strong><span class="metric-label" id="memoryMetricLabel">tracked session state vs full-byte</span></div>
        </div>
        <div class="two-col">
          <div class="panel">
            <div class="kicker-line"><b>Context update → future readout</b><span id="klRange">—</span></div>
            <div class="trend-list" id="trendList"></div>
            <div class="countertrend" id="countertrend">Loading countertrend…</div>
          </div>
          <div class="panel">
            <h3>Read the number with its boundary.</h3>
            <p id="klExplanation">Loading…</p>
            <div class="callout">
              <p><strong>13.59% is a trend in a finite conditional panel.</strong></p>
              <p>The same records show a countertrend when a separately compiled layer-18 Q/K program is added after the updated layer-17 query: KL rises in every row. This is evidence about composition at a named consumer boundary, not a universal accuracy or speedup claim.</p>
            </div>
            <details><summary>Read the generated answers</summary><div id="answerCases"></div></details>
          </div>
        </div>
      </section>

      <section class="section" id="protocol">
        <div class="section-heading">
          <div class="eyebrow">03 / protocol</div>
          <h2>Conversation is a sequence of owned transitions.</h2>
          <p>The server consumes token IDs, advances its cache, emits its own answer IDs, and exposes a state boundary. Branching is a reversible lifecycle operation around that boundary.</p>
        </div>
        <div class="steps">
          <div class="step"><span class="step-num">01</span><h3>Tokenize</h3><p>The local chat template turns messages into the exact processed prefix.</p></div>
          <div class="step"><span class="step-num">02</span><h3>Append</h3><p><code>A</code> transfers only the suffix the server has not consumed.</p></div>
          <div class="step"><span class="step-num">03</span><h3>Generate</h3><p><code>G</code> returns model-owned IDs and leaves a pending token.</p></div>
          <div class="step"><span class="step-num">04</span><h3>Branch</h3><p><code>P</code>, <code>B</code>, and <code>K</code> save, roll back, or adopt a continuation.</p></div>
          <div class="step"><span class="step-num">05</span><h3>Inspect</h3><p><code>S</code> reports cache, pending token, position, and readout hashes.</p></div>
        </div>
        <div class="code-panel"><pre><code>A &lt;unconsumed token IDs&gt;
G &lt;budget&gt;       # server-generated IDs; final one remains pending
P                 # push a token-boundary frame
B &lt;frame&gt;         # restore cache + pending state
K &lt;frame&gt;         # commit the current branch
S                 # inspect state and full-vocabulary hashes</code></pre></div>
      </section>

      <section class="section" id="program">
        <div class="section-heading">
          <div class="eyebrow">04 / operand program</div>
          <h2>Replace one boundary, keep the consumer visible.</h2>
          <p>The compiled lexical demo uses the public Qwen program API. It looks up first-layer Q/K/V atoms for consumed IDs, then leaves the remaining native computation in place so the replacement can be judged at the full-vocabulary consumer.</p>
        </div>
        <div class="program-grid">
          <div class="panel">
            <h3>Native path and replacement boundary</h3>
            <div class="pipeline"><span class="pipeline-node">token IDs</span><span class="pipeline-arrow">→</span><span class="pipeline-node">first-layer Q/K/V lookup</span><span class="pipeline-arrow">→</span><span class="pipeline-node purple">RoPE + attention / MLP</span><span class="pipeline-arrow">→</span><span class="pipeline-node">27 native layers + head</span></div>
            <p class="program-note">The table is compiled offline from a source-covered trajectory. Original weights remain resident; table construction, transfers, and the native reference pass are paid. This is an exact historical readout, not held-out generalization or a broad speedup claim.</p>
            <div class="program-stats">
              <div class="program-stat"><b id="lexicalExact">—</b><span>historical full-logit readouts</span></div>
              <div class="program-stat"><b id="lexicalIDs">—</b><span>covered lexical IDs</span></div>
              <div class="program-stat"><b id="lexicalCompile">—</b><span>offline compile MACs</span></div>
              <div class="program-stat"><b id="lexicalHost">—</b><span>compiled table host bytes</span></div>
              <div class="program-stat"><b id="lexicalTransfer">—</b><span>table transfer bytes</span></div>
            </div>
          </div>
          <div class="panel">
            <h3>What the record binds</h3>
            <div class="callout"><p><strong id="lexicalExactText">Loading…</strong></p><p>The historical integration compares the entire vocabulary vector at 16 steps. The fresh 44-step demo above also hashes every valid K/V row across all 28 layers.</p></div>
            <p style="margin-top:16px">The compiled table is model-qualified and source-bound. It changes a named operand formation boundary; it does not erase the dense attention, MLP, normalization, or final head that consume that operand.</p>
            <a href="compiled_operands/README.md">Open the compiled-operand README →</a>
          </div>
        </div>
      </section>

      <section class="section" id="memory">
        <div class="section-heading">
          <div class="eyebrow">05 / memory policy</div>
          <h2>A smaller memory can replay and still answer differently.</h2>
          <p>The recorded policies preserve exact replay mechanics while their delayed answers expose the consumer-facing limit. The expected answer sequence remains visible beside every actual answer.</p>
        </div>
        <div class="memory-card">
          <div class="memory-head">
            <div><h3>Delayed memory answers</h3><p>Green means the recorded answer matches the expected sequence; red is retained as evidence of a policy-specific miss.</p></div>
            <div class="memory-saving" id="memorySaving">—</div>
          </div>
          <div class="memory-table-wrap"><table><thead><tr><th>Policy</th><th>Actual answer sequence</th><th>Expected sequence</th><th>Exact replay</th><th>Tracked state</th></tr></thead><tbody id="memoryRows"></tbody></table></div>
          <div class="memory-foot">
            <div class="mini-stat"><b id="fullByteState">—</b><span>full-byte tracked state</span></div>
            <div class="mini-stat"><b id="boundedState">—</b><span>bounded tracked state</span></div>
            <div class="mini-stat"><b id="memoryRecovery">—</b><span>bounded recovery mechanics</span></div>
          </div>
        </div>
      </section>

      <section class="section" id="closed-state">
        <div class="section-heading">
          <div class="eyebrow">06 / closed-state laboratory</div>
          <h2>Closure is a consumer contract, not a compression slogan.</h2>
          <p>Scrub the supplied eight-dimensional organism through time. The closed compact path agrees with both hidden orientations; the escape writes an omitted coordinate and breaks the stale code; the repair adds that coordinate back.</p>
        </div>
        <div class="plot-shell">
          <div class="plot-toolbar">
            <label class="control">mode <select id="plotMode"><option value="closed">closed / 3 paths</option><option value="escape">escape / 3 paths</option><option value="repair">repair / 4 paths</option><option value="all">all / 8 paths</option></select></label>
            <label class="control">coordinate <select id="plotCoordinate"><option value="0">state[0]</option><option value="1">state[1]</option><option value="2">state[2]</option></select></label>
            <label class="control">time <input id="timeSlider" type="range" min="0" max="16" step="1" value="0"></label>
            <span class="plot-time" id="plotTime">t = 0 / 16</span>
          </div>
          <div class="svg-wrap"><svg id="statePlot" viewBox="0 0 960 355" role="img" aria-label="Closed-state trajectories"></svg></div>
          <div class="legend" id="closedLegend"></div>
          <div class="plot-note" id="plotNote">Loading trajectory…</div>
        </div>
        <div class="observations">
          <div class="observation"><b>Closed control</b><span id="closedObservation">Loading…</span></div>
          <div class="observation"><b>Escape control</b><span id="escapeObservation">Loading…</span></div>
          <div class="observation"><b>Repair</b><span id="repairObservation">Loading…</span></div>
        </div>
      </section>

      <section class="section" id="evidence">
        <div class="section-heading">
          <div class="eyebrow">07 / evidence trail</div>
          <h2>Inspect the source, then choose the next cut.</h2>
          <p>The links preserve the distinction between a runnable fixture, a pinned historical record, and the evidence bundle that explains its custody.</p>
        </div>
        <div class="links">
          <div class="link-card"><a href="closed_state/README.md">Closed-state demo README</a><p>Fresh-clone CPU fixture, recurrence, exactness scope, and escape/repair interpretation.</p></div>
          <div class="link-card"><a href="chat_branch/README.md">Reversible chat README</a><p>Public token-server client with explicit model, tokenizer, branch, rollback, and replay arguments.</p></div>
          <div class="link-card"><a href="../README.md">cmachine README</a><p>Canonical runtime protocol, model boundary, build instructions, and numerical contract.</p></div>
          <div class="link-card"><a href="../docs/evidence/integration-7a6824d/README.md">Integration evidence</a><p>Historical pretrained custody, native/reference joins, and memory-policy records.</p></div>
          <div class="link-card"><a href="../docs/evidence/context-query-20261106/README.md">Context-query evidence</a><p>Conditional operand replacement, future readouts, and the KL panel represented above.</p></div>
          <div class="link-card"><a href="recorded/build.py">Recorded gallery builder</a><p>Verifies pinned records and regenerates the compact historical summary without running a model.</p></div>
        </div>
        <div class="scope-strip" id="scopeStrip">Loading scope…</div>
        <div class="scope-strip" id="pretrainedStrip" style="display:none"></div>
      </section>
    </main>
    <footer class="footer">Generated from checked-in JSON records by <code>demos/render.py</code>. No network assets, fonts, or model execution are used to render this page.</footer>
  </div>

  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
  (() => {
    "use strict";
    const data = JSON.parse(document.getElementById("payload").textContent);
    const recorded = data.recorded;
    const closed = data.closed;
    const $ = (id) => document.getElementById(id);
    const setText = (id, value) => { $(id).textContent = String(value); };
    const number = (value, digits = 3) => Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
    const bytes = (value) => {
      const n = Number(value);
      if (!Number.isFinite(n)) return "—";
      if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
      if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
      if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
      return `${n} B`;
    };
    const pct = (value, digits = 2) => `${(Number(value) * 100).toFixed(digits)}%`;
    const el = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = String(text);
      return node;
    };

    const lexical = recorded.operands.lexical;
    const memoryRows = recorded.memory.rows || [];
    setText("nativeAnswer", recorded.chat.answer);
    setText("answerMeta", `Qwen native record · ${recorded.model.profile_version} · ${recorded.chat.native_reference_protocol_entries_exact} protocol entries`);
    setText("protocolCount", recorded.chat.native_reference_protocol_entries_exact);
    setText("lexicalCount", lexical.lexical_replayed_steps);
    setText("klMetric", pct(recorded.operands.relative_kl_reduction));
    setText("memoryMetric", pct(recorded.memory.state_saving_vs_full_byte));
    setText("memoryMetricLabel", `tracked session state vs full-byte · logical capacity ${recorded.memory.logical_capacity}`);
    setText("klRange", `${number(recorded.operands.fresh_baseline_mean_kl, 4)} → ${number(recorded.operands.fresh_updated_mean_kl, 4)}`);
    setText("klExplanation", `Across ${recorded.operands.fresh_case_count} recorded context cases, the mean conditional KL moves from ${number(recorded.operands.fresh_baseline_mean_kl, 5)} to ${number(recorded.operands.fresh_updated_mean_kl, 5)}. The lexical panel contains ${lexical.lexical_full_logits_exact ? "exact" : "non-exact"} full-logit readouts, while this page treats answer text as an observation.`);

    const laterCounter = recorded.operands.cases.filter((row) => Number(row.later_layer_kl) > Number(row.updated_kl)).length;
    setText("countertrend", `${laterCounter} / ${recorded.operands.cases.length} cases rise again when a separately compiled layer-18 Q/K program is added after the updated layer-17 query (${number(recorded.operands.cases[0].updated_kl, 4)} → ${number(recorded.operands.cases[0].later_layer_kl, 4)} in the first row). The improvement is therefore conditional on the composed program and its consumer boundary.`);
    recorded.operands.cases.forEach((row) => {
      $("answerCases").appendChild(el("h4", "", `${row.kind} · ${row.context_tokens} context tokens`));
      [["Updated layer-17 query", row.updated_answers], ["Add layer-18 Q/K", row.later_layer_answers]].forEach(([label, answers]) => {
        $("answerCases").appendChild(el("p", "", label));
        const text = el("pre", "", answers.join("\n\n"));
        text.style.whiteSpace = "pre-wrap";
        text.style.overflowWrap = "anywhere";
        $("answerCases").appendChild(text);
      });
    });

    const trendList = $("trendList");
    const trendMax = Math.max(...recorded.operands.cases.flatMap((row) => [row.baseline_kl, row.updated_kl, row.later_layer_kl]));
    recorded.operands.cases.forEach((row, index) => {
      const wrap = el("div", "trend-row");
      const label = el("div", "trend-label");
      const labelName = el("b", "", row.kind);
      label.appendChild(labelName);
      label.appendChild(document.createTextNode(`${row.context_tokens} context tokens`));
      wrap.appendChild(label);
      const stack = el("div", "bar-stack");
      [["baseline", "baseline", row.baseline_kl], ["updated", "updated", row.updated_kl], ["later", "layer18 Q/K", row.later_layer_kl]].forEach(([className, labelText, value]) => {
        const line = el("div", "bar-line");
        line.appendChild(el("span", "", labelText));
        const track = el("div", "bar-track");
        const fill = el("div", `bar-fill ${className}`);
        fill.style.width = `${Math.max(2, Number(value) / trendMax * 100)}%`;
        track.appendChild(fill);
        line.appendChild(track);
        line.appendChild(el("span", "", number(value, 4)));
        stack.appendChild(line);
      });
      wrap.appendChild(stack);
      trendList.appendChild(wrap);
    });

    setText("memorySaving", `${pct(recorded.memory.state_saving_vs_full_byte)} less tracked state`);
    const memoryBody = $("memoryRows");
    memoryRows.forEach((row) => {
      const tr = document.createElement("tr");
      const policy = el("td", "policy", row.policy);
      tr.appendChild(policy);
      const actual = el("td", "");
      row.answers.forEach((answer, index) => {
        const ok = Boolean(row.correct[index]);
        actual.appendChild(el("span", `answer-chip ${ok ? "ok" : "miss"}`, answer));
      });
      tr.appendChild(actual);
      const expected = el("td", "", row.expected.join(" · "));
      tr.appendChild(expected);
      tr.appendChild(el("td", "", row.exact_replay ? "exact replay" : "replay differs"));
      tr.appendChild(el("td", "", bytes(row.tracked_state_bytes)));
      memoryBody.appendChild(tr);
    });
    setText("fullByteState", bytes((memoryRows.find((row) => row.policy === "full-byte") || {}).tracked_state_bytes));
    setText("boundedState", bytes((memoryRows.find((row) => row.policy === "bounded") || {}).tracked_state_bytes));
    setText("memoryRecovery", recorded.memory.bounded_recovery && recorded.memory.bounded_recovery.exact_child_replay ? "exact child replay" : "see record");

    setText("lexicalExact", lexical.lexical_replayed_steps);
    setText("lexicalIDs", lexical.lexical_covered_ids);
    setText("lexicalCompile", number(lexical.lexical_compile_linear_macs, 0));
    setText("lexicalHost", bytes(lexical.lexical_host_bytes));
    setText("lexicalTransfer", bytes(lexical.lexical_transfer_bytes));
    setText("lexicalExactText", lexical.lexical_full_logits_exact ? "Full-vocabulary comparisons are exact" : "Full-vocabulary comparisons are not exact");

    const allPaths = closed.paths || [];
    const groups = {
      closed: allPaths.filter((path) => path.name.startsWith("closed-")),
      escape: allPaths.filter((path) => path.name.startsWith("escape-")),
      repair: allPaths.filter((path) => path.name.startsWith("repair-") || ["escape-plus", "escape-minus"].includes(path.name)),
      all: allPaths,
    };
    const palette = ["#72e2d4", "#b9a0ff", "#f5c776", "#ff8b8b", "#82e2a3", "#8fc4ff", "#e58cff", "#f6a87e"];
    const pathColor = (path, index) => palette[allPaths.indexOf(path) % palette.length] || palette[index % palette.length];
    const svg = $("statePlot");
    const plotMode = $("plotMode");
    const plotCoordinate = $("plotCoordinate");
    const timeSlider = $("timeSlider");
    const NS = "http://www.w3.org/2000/svg";
    const svgNode = (tag, attrs = {}, text) => {
      const node = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (text !== undefined) node.textContent = String(text);
      return node;
    };
    function drawPlot() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const selected = groups[plotMode.value] || groups.closed;
      const coordinate = Number(plotCoordinate.value);
      const time = Number(timeSlider.value);
      const width = 960, height = 355, left = 58, right = 18, top = 20, bottom = 42;
      const available = selected.filter((path) => path.states.every((state) => Number.isFinite(Number(state[coordinate]))));
      const omitted = selected.filter((path) => !available.includes(path));
      if (!available.length) {
        setText("plotTime", "no coordinate in selected paths");
        setText("plotNote", `No selected path contains coordinate ${coordinate}.`);
        return;
      }
      const values = available.flatMap((path) => path.states.map((state) => Number(state[coordinate])));
      let min = Math.min(...values), max = Math.max(...values);
      if (min === max) { min -= 1; max += 1; }
      const pad = (max - min) * .12;
      min -= pad; max += pad;
      const steps = Math.max(...available.map((path) => path.states.length)) - 1;
      const x = (t) => left + (steps ? t / steps : 0) * (width - left - right);
      const y = (value) => top + (max - value) / (max - min) * (height - top - bottom);
      [0, .25, .5, .75, 1].forEach((fraction) => {
        const value = max - fraction * (max - min);
        svg.appendChild(svgNode("line", {x1: left, x2: width - right, y1: y(value), y2: y(value), class: "plot-grid"}));
        svg.appendChild(svgNode("text", {x: 5, y: y(value) + 4, class: "plot-axis"}, number(value, 2)));
      });
      svg.appendChild(svgNode("line", {x1: left, x2: width - right, y1: height - bottom, y2: height - bottom, class: "plot-grid"}));
      svg.appendChild(svgNode("text", {x: left, y: height - 12, class: "plot-axis"}, "t=0"));
      svg.appendChild(svgNode("text", {x: width - right - 30, y: height - 12, class: "plot-axis"}, `t=${steps}`));
      available.forEach((path, index) => {
        const points = path.states.map((state, t) => `${x(t)},${y(Number(state[coordinate]))}`).join(" ");
        const color = pathColor(path, index);
        svg.appendChild(svgNode("polyline", {points, fill: "none", stroke: color, "stroke-width": 2.4, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: path.role === "reference" ? .62 : 1}));
        const state = path.states[Math.min(time, path.states.length - 1)];
        svg.appendChild(svgNode("circle", {cx: x(Math.min(time, path.states.length - 1)), cy: y(Number(state[coordinate])), r: 4.5, fill: color, stroke: "#071316", "stroke-width": 2}));
      });
      svg.appendChild(svgNode("line", {x1: x(time), x2: x(time), y1: top, y2: height - bottom, stroke: "#e9f7f4", "stroke-width": 1, "stroke-dasharray": "3 5", opacity: .45}));
      setText("plotTime", `t = ${time} / ${steps}`);
      const legend = $("closedLegend");
      while (legend.firstChild) legend.removeChild(legend.firstChild);
      selected.forEach((path, index) => {
        const item = el("span", "legend-item");
        const dot = el("span", "legend-dot");
        dot.style.backgroundColor = pathColor(path, index);
        item.appendChild(dot);
        item.appendChild(document.createTextNode(path.name));
        legend.appendChild(item);
      });
      const selectedValues = available.map((path) => Number(path.states[Math.min(time, path.states.length - 1)][coordinate]));
      const omittedNote = omitted.length ? ` Omitted for this coordinate: ${omitted.map((path) => path.name).join(", ")}; their compact states do not carry it.` : "";
      setText("plotNote", `${plotMode.options[plotMode.selectedIndex].text} · coordinate ${coordinate} · at t=${time}, retained values span ${number(Math.min(...selectedValues), 4)} to ${number(Math.max(...selectedValues), 4)}. Reference paths are dimmed; all values come from the checked-in FP32 fixture.${omittedNote}`);
    }
    timeSlider.max = String(Math.max(...allPaths.map((path) => path.states.length)) - 1);
    [plotMode, plotCoordinate, timeSlider].forEach((control) => {
      control.addEventListener("input", drawPlot);
      control.addEventListener("change", drawPlot);
    });
    drawPlot();

    const mechanics = closed.mechanics;
    setText("closedObservation", mechanics.closed_vs_plus_byte_exact && mechanics.closed_vs_minus_byte_exact ? "compact retained trajectories match both hidden full references byte-for-byte." : "see mechanics record");
    setText("escapeObservation", `omitted energy after the first update is ${number(closed.escape_observation.one_step_omitted_energy_plus, 3)} / ${number(closed.escape_observation.one_step_omitted_energy_minus, 3)}; visible retained state first differs at update ${mechanics.stale_vs_escape_plus_first_state_difference}; final declared output separation is ${number(closed.escape_observation.full_final_declared_output_l2, 3)} L2.`);
    setText("repairObservation", mechanics.repair_vs_escape_plus_byte_exact && mechanics.repair_vs_escape_minus_byte_exact ? "adding the written coordinate restores both reference trajectories exactly." : "see mechanics record");

    setText("scopeStrip", `Historical pretrained record: ${recorded.scope} The closed-state panel is ${closed.scientific_scope.finite_profile}; it is not a learned-model compression or serving-speed claim.`);
    if (data.pretrained) {
      const fresh = data.pretrained;
      const freshChat = fresh.fresh_chat || {};
      const freshLexical = fresh.lexical || {};
      const freshBytes = freshLexical.bytes || {};
      $("fresh-panel").style.display = "block";
      setText("freshJob", fresh.job_id || "present");
      setText("freshReplay", freshChat.replay_exact ? "exact" : "observe");
      setText("freshLexical", freshLexical.steps ?? "—");
      setText("freshIDs", freshLexical.covered_ids ?? "—");
      setText("freshInitial", freshChat.initial && freshChat.initial.text || "—");
      setText("freshBranchA", freshChat.branch_a && freshChat.branch_a.text || "—");
      setText("freshReplayText", freshChat.replay_exact ? freshChat.branch_a.text : "See report");
      setText("freshBranchB", freshChat.branch_b && freshChat.branch_b.text || "—");
      const compileMacs = freshLexical.linear_macs && (freshLexical.linear_macs.compile ?? freshLexical.linear_macs) || "—";
      const candidateTokens = freshLexical.candidate_generated_tokens_consumed ?? freshLexical.generated_tokens_consumed ?? "—";
      const omittedMacs = freshLexical.omitted_linear_macs ?? freshLexical.omitted_first_layer_qkv_macs ?? "—";
      setText("freshMeta", `The frozen lexical program advanced ${freshLexical.steps} tokens, consuming ${candidateTokens} of its own generated tokens. Every full-vocabulary vector, valid all-layer cache hash and selected token matched native execution. Offline compilation: ${number(compileMacs, 0)} linear MACs. Host table: ${bytes(freshBytes.table_host)}. Candidate transfers: ${bytes(freshBytes.candidate_transfer)}. Original Q/K/V work omitted online: ${number(omittedMacs, 0)} linear MACs.`);
      const claims = Array.isArray(fresh.claims) ? fresh.claims : [];
      const limits = Array.isArray(fresh.limits) ? fresh.limits : [];
      setText("freshLimits", "Source-covered reuse: native discovery, compilation, transfers, original weights and remaining dense computation are paid. No held-out generalization or serving speedup is established.");
      setText("freshDetails", `Claims: ${claims.length ? claims.join(" · ") : "see summary"} Limits: ${limits.length ? limits.join(" · ") : "see summary"}`);
      const pretrained = $("pretrainedStrip");
      pretrained.style.display = "block";
      const status = data.pretrained.status || data.pretrained.schema || "summary loaded";
      pretrained.textContent = `Optional fresh pretrained summary loaded: ${status}. It is displayed as a separate record and does not rewrite the pinned historical panels.`;
    }
  })();
  </script>
</body>
</html>
'''


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    recorded_path = HERE / "recorded" / "results.json"
    closed_path = HERE / "closed_state" / "results.json"
    pretrained_path = HERE / "results" / "pretrained" / "summary.json"
    payload = {
        "recorded": load_json(recorded_path),
        "closed": load_json(closed_path),
        "pretrained": load_json(pretrained_path, optional=True),
        "sources": {
            "recorded": "recorded/results.json",
            "closed_state": "closed_state/results.json",
            "pretrained": "results/pretrained/summary.json",
        },
    }
    rendered = HTML.replace("__PAYLOAD_JSON__", embedded_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build(args.output.resolve())
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
