"""Beast worker: cmachine per-token ledger (research/ledger, docs/LEDGER.md). Run from the payload root:
    python worker.py OUTPUT_DIR PLAN_JSON

Order: sampler first (one synchronous probe aborts the job if nvidia-smi/psutil fail) -> idle baseline ->
CUDA build (make cuda, make ledger-run) -> tokenize -> C process (FP32 smoke gates everything; exit 3 aborts
before the scored arms) -> analysis (timing, G1 bit-identity, G2 lean vs fast, G3 float64 recomputation,
G4 margin closure, ledger read) -> Part 4 screened-head feasibility (NumPy float64, after the C process exits)
-> sampler stopped last. Samples go to the mrun outbox (kind=sample rows) and samples.csv."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAND = ROOT / "candidate"
sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from mrun import outbox  # noqa: E402

MODEL = Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
HF_DIR = Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
PROMPTS = [  # the speedups/overhead prompt set
    "The history of the printing press begins in the fifteenth century, when a goldsmith from Mainz",
    "In a quiet village at the edge of the forest, an old clockmaker spent every evening repairing",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number using iteration.\"\"\"\n    a, b = 0, 1\n",
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into",
    "The committee met on Tuesday to discuss the budget for the coming year, and the first item on the agenda",
    "Once upon a time, in a kingdom beyond the seven mountains, there lived a young princess who",
    "To install the package on a fresh Linux machine, first update the package index and then",
    "The theory of general relativity describes gravity not as a force but as the curvature of",
    "She opened the old wooden chest and found, beneath a layer of yellowed letters, a small brass key that",
    "Mount Everest, the highest mountain above sea level, sits on the border between Nepal and",
    "A good cup of coffee starts with fresh beans, water just off the boil, and patience; the first step is to",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon, and millions of people watched as",
    "The river wound slowly through the valley, past orchards and stone bridges, until it reached the sea where",
    "Machine learning models are trained by adjusting their parameters to minimize a loss function, which measures",
    "My grandmother's kitchen always smelled of cinnamon and warm bread, and on Sunday afternoons she would",
    "The central bank raised interest rates again this quarter, citing persistent inflation and a tight labor market, which",
]
# Off-arm reference medians from job-1d03b313def2 (docs/SPEEDUPS.md), measured there.
BASELINE = {"job": "job-1d03b313def2", "decode_ms": {"C0": 10.544, "F1": 10.177, "G1": 9.973, "G8": 11.263, "G64": 13.036, "bf16_G1": 5.364},
            "prefill_ms": {"F1": 11.56, "C0": 210.5}}
_PHASE = {"file": None}

HEAD_F = ["token", "position", "sources", "has_readout", "top1", "top2", "margin", "rho_final", "c_norm", "margin_embed", "margin_sum"]
LAYER_F = ["rho_in", "rho_mid", "attn_dot", "attn_norm2", "mlp_dot", "mlp_norm2", "margin_attn", "margin_mlp",
           "gate_small", "gate_off", "gate_on", "even2", "odd2", "even_odd", "prod2"]
HEAD_ARR_F = ["h_logz", "h_pfirst", "h_entropy", "h_pmax", "h_argmax"]
DISCRETE = {"token", "position", "sources", "has_readout", "top1", "top2", "gate_small", "gate_off", "gate_on", "h_argmax"}
READOUT = {"has_readout", "top1", "top2", "margin", "rho_final", "c_norm", "margin_embed", "margin_sum", "margin_attn", "margin_mlp"}


def set_phase(label):
    path = _PHASE["file"]
    if path:
        tmp = str(path) + ".tmp"
        Path(tmp).write_text(label)
        os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def run(command, cwd=None, timeout=900, log=None):
    t = time.monotonic()
    r = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if log:
        Path(log).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode:
        raise RuntimeError(f"{command} failed ({r.returncode}): {r.stderr[-3000:]} {r.stdout[-1500:]}")
    return r, time.monotonic() - t


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Sampler(threading.Thread):
    COLS = ["t", "t_rel", "phase", "vram_used_mib", "gpu_util_pct", "power_w", "sm_clock_mhz", "mem_clock_mhz",
            "own_gpu_mib", "foreign", "tree_rss_mb", "tree_cpu_pct"]

    def __init__(self, phase_file, csv_path, box, interval=0.5):
        super().__init__(daemon=True)
        self.phase_file, self.csv_path, self.box, self.interval = phase_file, csv_path, box, interval
        self.stop_event, self.rows, self.procs, self.n, self.t0 = threading.Event(), [], {}, 0, time.time()
        self.last_phase, self.errors = "init", 0
        self.csv = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(self.COLS)

    def smi(self, args):
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True).stdout

    def sample(self):
        t = time.time()
        g = [x.strip() for x in self.smi(["--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem"]).splitlines()[0].split(",")]
        apps = []
        for line in self.smi(["--query-compute-apps=pid,used_memory"]).splitlines():
            if line.strip():
                p, m = [x.strip() for x in line.split(",")[:2]]
                apps.append((int(p), num(m) or 0.0))
        root = psutil.Process(os.getpid())
        live = [root] + root.children(recursive=True)
        rss, cpu, tree = 0, 0.0, set()
        for p in live:
            try:
                if p.name() == "nvidia-smi":
                    continue
                tree.add(p.pid)
                if p.pid not in self.procs:
                    self.procs[p.pid] = p
                    p.cpu_percent(None)
                else:
                    cpu += self.procs[p.pid].cpu_percent(None)
                rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        own = sum(m for p, m in apps if p in tree)
        foreign = ";".join(f"{p}:{m:g}" for p, m in apps if p not in tree)
        try:
            label = Path(self.phase_file).read_text().strip() or self.last_phase
        except OSError:
            label = self.last_phase
        self.last_phase = label
        row = {"t": t, "t_rel": t - self.t0, "phase": label, "vram_used_mib": num(g[0]), "gpu_util_pct": num(g[1]), "power_w": num(g[2]),
               "sm_clock_mhz": num(g[3]), "mem_clock_mhz": num(g[4]), "own_gpu_mib": own, "foreign": foreign,
               "tree_rss_mb": rss / 2**20, "tree_cpu_pct": cpu}
        self.rows.append(row)
        self.writer.writerow([row[c] for c in self.COLS])
        self.box.record(f"sample/{self.n:05d}", row, kind="sample")
        self.n += 1

    def run(self):
        nxt = time.time() + self.interval
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception:  # noqa: BLE001
                self.errors += 1
            nxt += self.interval
            self.stop_event.wait(max(0.0, nxt - time.time()))

    def finish(self):
        self.stop_event.set()
        self.join(timeout=15)
        self.csv.flush()
        self.csv.close()


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def summarize_phases(rows):
    out = {}
    for ph in dict.fromkeys(r["phase"] for r in rows):
        rs = [r for r in rows if r["phase"] == ph]
        d = {"n": len(rs), "t_rel_start": rs[0]["t_rel"], "t_rel_end": rs[-1]["t_rel"]}
        for key in ("vram_used_mib", "own_gpu_mib", "tree_rss_mb", "tree_cpu_pct", "gpu_util_pct", "power_w", "sm_clock_mhz"):
            v = [r[key] for r in rs if r[key] is not None]
            if v:
                d[key] = {"min": min(v), "median": float(np.median(v)), "max": max(v)}
        d["foreign_present_samples"] = sum(1 for r in rs if r["foreign"])
        out[ph] = d
    return out


def region_contaminated(rows, t0, t1, slack=0.5):
    return any(r["foreign"] for r in rows if t0 - slack <= r["t"] <= t1 + slack)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def dist(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    return {"n": int(v.size), "median": float(np.median(v)), "p10": pct(v, 10), "p90": pct(v, 90), "p99": pct(v, 99), "max": float(v.max()), "min": float(v.min()), "mean": float(v.mean())}


# ---------------------------------------------------------------- ledger layout
class Layout:
    def __init__(self, h):
        self.L, self.H, self.RF = h["layers"], h["qheads"], h["row_floats"]
        self.d, self.inter, self.kv, self.hd, self.eps = h["hidden"], h["intermediate"], h["kvheads"], h["head"], float(h["epsilon"])
        self.blk = h["ql_layer"] + h["ql_per_head"] * self.H
        assert h["ql_head"] + self.L * self.blk == self.RF, "row layout mismatch"
        self.cols = {n: np.array([i]) for i, n in enumerate(HEAD_F)}
        for k, n in enumerate(LAYER_F):
            self.cols[n] = np.array([h["ql_head"] + l * self.blk + k for l in range(self.L)])
        for f, n in enumerate(HEAD_ARR_F):
            self.cols[n] = np.array([h["ql_head"] + l * self.blk + h["ql_layer"] + f * self.H + j for l in range(self.L) for j in range(self.H)])

    def layer(self, rows, name):  # (n, L) or (n, L, H)
        v = rows[:, self.cols[name]]
        return v.reshape(len(rows), self.L, self.H) if name in HEAD_ARR_F else v


def field_diff(lay, A, B, only=None):
    """Per-field differences of aligned row arrays A (reference) and B. Readout fields only where both rows have one."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    both = (A[:, 3] == 1) & (B[:, 3] == 1)
    out = {}
    for name, cols in lay.cols.items():
        if only and name not in only:
            continue
        a, b = A[:, cols], B[:, cols]
        if name in READOUT and name != "has_readout":
            a, b = a[both], b[both]
        if not a.size:
            continue
        d = np.abs(a - b)
        scale = float(np.max(np.abs(a))) or 1.0
        rel = d / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
        rec = {"n": int(a.size), "max_abs": float(d.max()), "max_abs_over_field_scale": float(d.max() / scale), "max_rel": float(rel.max()),
               "p99_rel": pct(rel.ravel(), 99), "median_rel": pct(rel.ravel(), 50), "bit_equal": int(np.sum(a == b))}
        if name in DISCRETE:
            rec["mismatches"] = int(np.sum(a != b))
        out[name] = rec
    return out


CONT_F = ["rho_in", "rho_mid", "rho_final", "attn_norm2", "mlp_norm2", "even2", "odd2", "prod2", "c_norm", "attn_dot", "mlp_dot", "even_odd",
          "margin", "margin_sum", "margin_embed", "margin_attn", "margin_mlp", "h_logz", "h_entropy", "h_pfirst", "h_pmax"]
EXACT_F = ["token", "position", "sources", "top1", "top2"]
COUNT_F = ["gate_small", "gate_off", "gate_on"]
ARGMAX_TIE = 1e-5  # overwritten from the plan's predeclared criteria (argmax_tie_prob)
_GROUPS = {}


def norm_err(lay, A, B):
    """|a - b| / s per field, s = a scale fixed by the field's meaning (computed from the reference rows A):
    positive magnitudes (rho, norm2, even2, odd2, prod2, c_norm): |a|; dots: Cauchy-Schwarz bound
    (attn_dot: ||x|| ||w_attn||, mlp_dot: ||r|| ||w_mlp||, even_odd: sqrt(even2 odd2)); margin fields:
    the row's sum of |terms| (|embed| + sum |attn| + sum |mlp|); logZ, entropy: max(|a|, 1); probabilities: 1.
    Readout fields only on rows where both have a readout. Returns {field: dist of normalized errors}."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    both = (A[:, 3] == 1) & (B[:, 3] == 1)
    d = lay.d
    x2_in = d * (1 / lay.layer(A, "rho_in") ** 2 - lay.eps)
    x2_mid = d * (1 / lay.layer(A, "rho_mid") ** 2 - lay.eps)
    terms = np.abs(A[:, 9]) + np.abs(lay.layer(A, "margin_attn")).sum(1) + np.abs(lay.layer(A, "margin_mlp")).sum(1)
    out = {}
    for name in CONT_F:
        a, b = lay.layer(A, name), lay.layer(B, name)
        if name in ("attn_dot",):
            s = np.sqrt(np.maximum(x2_in * lay.layer(A, "attn_norm2"), 0))
        elif name == "mlp_dot":
            s = np.sqrt(np.maximum(x2_mid * lay.layer(A, "mlp_norm2"), 0))
        elif name == "even_odd":
            s = np.sqrt(np.maximum(lay.layer(A, "even2") * lay.layer(A, "odd2"), 0))
        elif name in ("margin", "margin_sum", "margin_embed", "margin_attn", "margin_mlp"):
            s = terms.reshape(-1, *([1] * (a.ndim - 1)))
        elif name in ("h_logz", "h_entropy"):
            s = np.maximum(np.abs(a), 1.0)
        elif name in ("h_pfirst", "h_pmax"):
            s = np.ones_like(a)
        else:
            s = np.abs(a)
        e = np.abs(a - b) / np.maximum(s, 1e-30)
        if name in READOUT:
            e = e[both]
        if e.size:
            out[name] = dist(e.ravel())
    return out


def judge(report, crit):
    """Pass/fail of G1-G4 and determinism against the plan's predeclared criteria (written before the run)."""
    v = {"criteria": crit}
    g1 = report.get("G1_bit_identity") or {}
    v["G1"] = {"pass": bool(g1.get("pass")), "failed": g1.get("failed")}
    det = report.get("ledger_determinism") or {}
    v["determinism"] = {"pass": bool(det) and all(x.split("/")[0] == x.split("/")[1] for x in det.values()), "checks": det}
    def tier(fd, nd, c, label, extra_fail):
        fails = list(extra_fail)
        for f in c.get("exact", []):
            if f in fd and fd[f].get("mismatches", 0):
                fails.append(f"{f}: {fd[f]['mismatches']} mismatches (must be exact)")
        for f in COUNT_F:
            if f in fd and fd[f]["max_abs"] > c["count_abs_max"]:
                fails.append(f"{f}: max |diff| {fd[f]['max_abs']:.0f} > {c['count_abs_max']}")
        for f, dd in nd.items():
            if dd["p99"] > c["continuous_p99_norm"]:
                fails.append(f"{f}: p99 normalized {dd['p99']:.3g} > {c['continuous_p99_norm']}")
            if dd["max"] > c["continuous_max_norm"]:
                fails.append(f"{f}: max normalized {dd['max']:.3g} > {c['continuous_max_norm']}")
        return {"pass": not fails, "failed": fails, "label": label}
    g2 = report.get("G2_lean_vs_fast") or {}
    if "all_rows" in g2:
        c = crit["G2"]
        ex = []
        fd = g2["all_rows"]
        n_am = fd["h_argmax"]["n"]
        bad_am = [m for m in g2.get("argmax_mismatches", []) if abs(m["pmax_lean"] - m["pmax_fast"]) > c["argmax_pmax_agree_abs"]]
        if fd["h_argmax"]["mismatches"] > c["argmax_mismatch_frac_max"] * n_am:
            ex.append(f"h_argmax: {fd['h_argmax']['mismatches']} mismatches > {c['argmax_mismatch_frac_max']} of {n_am}")
        if bad_am:
            ex.append(f"h_argmax: {len(bad_am)} mismatches whose pmax differ by > {c['argmax_pmax_agree_abs']} (not a tie)")
        v["G2"] = tier(fd, g2["all_rows_norm"], c, "lean vs fast, teacher-forced", ex)
    g3 = report.get("G3_independent_truth") or {}
    if "lean_vs_truth" in g3:
        c = crit["G3_lean"]
        cons = g3["consistency"]
        ex = []
        for k in ("raw_minus_input_plus_aw_bits", "port_consistency_output_vs_next_input_bits"):
            if cons.get(k):
                ex.append(f"{k} = {cons[k]}")
        if cons.get("native_margin_vs_ledger_max_abs", 0) > 0:
            ex.append(f"native margin vs ledger margin {cons['native_margin_vs_ledger_max_abs']}")
        if cons.get("argmax_vs_p64_mismatch_not_near_tie", 0):
            ex.append(f"h_argmax vs float64 softmax: {cons['argmax_vs_p64_mismatch_not_near_tie']} mismatches with top-2 probability gap >= {crit['argmax_tie_prob']}")
        if g3.get("errors"):
            ex.append(f"g3 errors {g3['errors']}")
        if fd_am := g3["lean_vs_truth"].get("h_argmax", {}).get("mismatches"):
            ex.append(f"h_argmax lean vs native-probability argmax: {fd_am} mismatches")
        v["G3_lean"] = tier(g3["lean_vs_truth"], g3["lean_vs_truth_norm"], c, "lean ledger vs float64 recomputation from its own ports", ex)
        if "fast_vs_truth" in g3:
            v["G3_fast"] = tier(g3["fast_vs_truth"], g3["fast_vs_truth_norm"], crit["G2"], "fast ledger vs float64 truth of the lean ports (executor tier)", [])
    t12 = report.get("G3_top12_float64")
    if t12:
        ex = []
        if t12.get("error"):
            ex.append(t12["error"])
        else:
            if t12["top1_mismatch_not_near_tie"]:
                ex.append(f"top1: {t12['top1_mismatch_not_near_tie']} mismatches with float64 gap >= {crit['top12_tie_logit']}")
            if t12["top2_mismatch_not_near_tie"]:
                ex.append(f"top2: {t12['top2_mismatch_not_near_tie']} mismatches with float64 gap >= {crit['top12_tie_logit']}")
        v["G3_top12"] = {"pass": not ex, "failed": ex}
    g4 = report.get("G4_margin_closure") or {}
    if g4:
        c = crit["G4"]
        ex = [f"{k}: max |rem| {x['abs_remainder']['max']:.3g} > {c['abs_remainder_max']}" for k, x in g4.items() if x["abs_remainder"]["max"] > c["abs_remainder_max"]]
        ms = (((g3.get("lean_vs_truth_norm") or {}).get("margin_sum")) or {}).get("max")
        if ms is not None and ms > c["split_vs_float64_norm_max"]:
            ex.append(f"ledger split vs float64 split: {ms:.3g} > {c['split_vs_float64_norm_max']} (of sum |terms|)")
        v["G4"] = {"pass": not ex, "failed": ex, "split_vs_float64_norm_max": ms}
    v["all_pass"] = all(x.get("pass") for k, x in v.items() if isinstance(x, dict) and "pass" in x)
    return v


def load_ledger_rows(out, rows, lay):
    """{(run_type, arm, ledger): {"rows": array, "prompt":..., "step":..., "seq":..., "B":..., "lane":...}} concatenated in file order."""
    groups, mm = {}, {}
    for r in rows:
        if r["type"] != "ledger_rows":
            continue
        f = r["file"]
        if f not in mm:
            mm[f] = np.memmap(out / f, dtype=np.float32, mode="r").reshape(-1, lay.RF)
        g = groups.setdefault((r["run_type"], r["arm"], r["ledger"]), {"rows": [], "prompt": [], "step": [], "seq": [], "run": [], "B": r["B"], "lane": r["lane"], "forced": r["forced"]})
        g["rows"].append(np.array(mm[f][r["row_start"]: r["row_start"] + r["count"]]))
        g["prompt"] += r["prompt"]
        g["step"] += r["step"]
        g["seq"] += r["seq"]
        g["run"] += [len(g["run"])] * r["count"]
    for g in groups.values():
        g["rows"] = np.concatenate(g["rows"]) if g["rows"] else np.zeros((0, lay.RF), np.float32)
        for k in ("prompt", "step", "seq", "run"):
            g[k] = np.array(g[k])
    return groups


# ---------------------------------------------------------------- timing + G1
def timing(rows, samples):
    runs = [r for r in rows if r["type"] == "run"]
    arms = {}
    for r in runs:
        a = arms.setdefault((r["arm"], r["ledger"]), {"lane": r["lane"], "B": r["B"], "graph": r["graph"], "forced": r["forced"], "ms": [], "prefill": [], "regions": 0, "contaminated": 0, "capture_ms": []})
        a["ms"].extend(r["ms"])
        a["prefill"].extend(r["prefill_ms"])
        a["regions"] += 1
        a["contaminated"] += region_contaminated(samples, r["t_start"], r["t_end"]) if samples else 0
        if r["graph"]:
            a["capture_ms"].append(r["extra_ms"])
    table = {}
    for (arm, mode), a in arms.items():
        ms = np.array(a["ms"])
        pf = np.array(a["prefill"])
        table[f"{arm}/{mode}"] = {"arm": arm, "ledger": mode, "lane": a["lane"], "B": a["B"], "graph": a["graph"], "forced": a["forced"], "n_steps": int(ms.size),
                                  "ms_median": float(np.median(ms)), "ms_p10": pct(ms, 10), "ms_p90": pct(ms, 90), "ms_mean": float(ms.mean()),
                                  "tok_per_s": a["B"] * 1e3 / float(np.median(ms)),
                                  "prefill_n": int(pf.size), "prefill_ms_median": float(np.median(pf)), "prefill_ms_p10": pct(pf, 10), "prefill_ms_p90": pct(pf, 90),
                                  "contaminated_regions": f"{a['contaminated']}/{a['regions']}", "contaminated": a["contaminated"] > 0,
                                  "graph_capture_ms_median": float(np.median(a["capture_ms"])) if a["capture_ms"] else None}
    ratios = {}
    for key, t in table.items():
        if t["ledger"] == "off" or t["forced"]:
            continue
        off = table.get(f"{t['arm']}/off")
        if not off:
            continue
        rec = {"decode_on_over_off": t["ms_median"] / off["ms_median"], "decode_delta_ms": t["ms_median"] - off["ms_median"],
               "decode_overhead_pct": 100 * (t["ms_median"] / off["ms_median"] - 1),
               "prefill_on_over_off": t["prefill_ms_median"] / off["prefill_ms_median"], "prefill_delta_ms": t["prefill_ms_median"] - off["prefill_ms_median"]}
        if t["ledger"] == "on" and f"{t['arm']}/onnr" in table:
            nr = table[f"{t['arm']}/onnr"]
            rec["readback_ms_per_step"] = t["ms_median"] - nr["ms_median"]
            rec["kernel_ms_per_step"] = nr["ms_median"] - off["ms_median"]
        ratios[key] = rec
    base = {}
    for arm, ref in BASELINE["decode_ms"].items():
        t = table.get(f"{arm}/off")
        if t:
            base[f"{arm}_decode"] = {"this_job_ms": t["ms_median"], "speedups_job_ms": ref, "ratio": t["ms_median"] / ref}
    for arm, ref in BASELINE["prefill_ms"].items():
        t = table.get(f"{arm}/off")
        if t:
            base[f"{arm}_prefill"] = {"this_job_ms": t["prefill_ms_median"], "speedups_job_ms": ref, "ratio": t["prefill_ms_median"] / ref}
    return table, ratios, {"reference_job": BASELINE["job"], "compare": base}


def g1_gate(out, rows, header):
    NP, V = header["prompts"], header["vocab"]
    banks = {}
    for r in rows:
        if r["type"] == "bank":
            banks[r["name"]] = (np.fromfile(out / f"logits-{r['name']}.f32", dtype=np.float32).reshape(NP, V), r["filled"])
    res = {}
    for name in banks:
        if not name.endswith("_off"):
            continue
        stem = name[: -len("_off")]
        for mode in ("on", "onnr"):
            other = f"{stem}_{mode}"
            if other not in banks:
                continue
            (A, fa), (B, fb) = banks[name], banks[other]
            ps = [p for p in range(NP) if fa[p] and fb[p]]
            same = sum(bool(np.array_equal(A[p].view(np.uint32), B[p].view(np.uint32))) for p in ps)
            res[f"logits_{stem}_{mode}_vs_off"] = f"{same}/{len(ps)}"
    runs = [r for r in rows if r["type"] == "run"]
    arms = sorted({r["arm"] for r in runs})
    for arm in arms:
        off = {(tuple(r["prompts"]), b): ids for r in runs if r["arm"] == arm and r["ledger"] == "off" for b, ids in enumerate(r["ids"])}
        for mode in ("on", "onnr"):
            oth = {(tuple(r["prompts"]), b): ids for r in runs if r["arm"] == arm and r["ledger"] == mode for b, ids in enumerate(r["ids"])}
            ks = sorted(set(off) & set(oth))
            if ks:
                res[f"ids_{arm}_{mode}_vs_off"] = f"{sum(off[k] == oth[k] for k in ks)}/{len(ks)}"
    fails = [k for k, v in res.items() if v.split("/")[0] != v.split("/")[1] or v.endswith("/0")]
    # C0 vs fast token agreement (informative; FP32 fast differs from lean by summation order)
    c0 = {r["prompts"][0]: r["ids"][0] for r in runs if r["arm"] == "C0" and r["ledger"] == "off"}
    agree = {}
    for arm in arms:
        if arm == "C0" or arm.startswith("FORCE"):
            continue
        tot = same = 0
        for r in runs:
            if r["arm"] == arm and r["ledger"] == "off":
                for p, ids in zip(r["prompts"], r["ids"]):
                    if p in c0:
                        tot += 1
                        same += ids == c0[p]
        if tot:
            agree[arm] = f"{same}/{tot}"
    return {"pass": not fails and bool(res), "failed": fails, "checks": res, "ids_identical_to_C0_off": agree}


# ---------------------------------------------------------------- G2, G4, determinism, read
def keyed(g, lay):
    pos = g["rows"][:, 1].astype(int)
    return {(int(p), int(q)): i for i, (p, q) in enumerate(zip(g["prompt"], pos))}


def g2_gate(groups, lay):
    lean = groups.get(("run", "C0", "on"))
    fast = groups.get(("run", "FORCE_F1", "on"))
    if lean is None or fast is None:
        return {"error": "missing C0/on or FORCE_F1/on rows"}
    kl, kf = keyed(lean, lay), keyed(fast, lay)
    common = sorted(set(kl) & set(kf))
    A = lean["rows"][[kl[k] for k in common]]
    B = fast["rows"][[kf[k] for k in common]]
    dec = np.array([fast["step"][kf[k]] >= 0 for k in common])
    res = {"rows_compared": len(common), "decode_rows": int(dec.sum()), "readout_rows_both": int(np.sum((A[:, 3] == 1) & (B[:, 3] == 1))),
           "all_rows": field_diff(lay, A, B), "decode_only": field_diff(lay, A[dec], B[dec]), "all_rows_norm": norm_err(lay, A, B)}
    am_a, am_b = lay.layer(A, "h_argmax"), lay.layer(B, "h_argmax")
    pa, pb = lay.layer(A, "h_pmax"), lay.layer(B, "h_pmax")
    mis = np.argwhere(am_a != am_b)
    res["argmax_mismatches"] = [{"row": int(i), "prompt": int(common[i][0]), "position": int(common[i][1]), "layer": int(l), "head": int(h),
                                 "lean": int(am_a[i, l, h]), "fast": int(am_b[i, l, h]), "pmax_lean": float(pa[i, l, h]), "pmax_fast": float(pb[i, l, h])} for i, l, h in mis[:50]]
    return res


def g4_closure(groups, lay):
    out = {}
    for (rt, arm, mode), g in groups.items():
        if rt != "run":
            continue
        R = g["rows"].astype(np.float64)
        R = R[R[:, 3] == 1]
        if not len(R):
            continue
        rem = R[:, 6] - R[:, 10]
        terms = np.abs(R[:, 9]) + np.abs(lay.layer(R, "margin_attn")).sum(1) + np.abs(lay.layer(R, "margin_mlp")).sum(1)
        out[f"{arm}/{mode}"] = {"rows": int(len(R)), "remainder": dist(rem), "abs_remainder": dist(np.abs(rem)),
                                "abs_remainder_over_abs_margin": dist(np.abs(rem) / np.maximum(np.abs(R[:, 6]), 1e-30)),
                                "abs_remainder_over_sum_abs_terms": dist(np.abs(rem) / np.maximum(terms, 1e-30)),
                                "margin": dist(R[:, 6])}
    return out


def rms_identity(R, lay):
    """d*(1/rho_mid^2 - 1/rho_in^2) vs 2 attn_dot + attn_norm2, and the MLP analogue (next rho_in or rho_final)."""
    R = R.astype(np.float64)
    d = lay.d
    ri, rm = lay.layer(R, "rho_in"), lay.layer(R, "rho_mid")
    lhs_a = d * (1 / rm**2 - 1 / ri**2)
    rhs_a = 2 * lay.layer(R, "attn_dot") + lay.layer(R, "attn_norm2")
    nxt = np.concatenate([ri[:, 1:], np.where(R[:, 3:4] == 1, R[:, 7:8], np.nan)], axis=1)
    lhs_m = d * (1 / nxt**2 - 1 / rm**2)
    rhs_m = 2 * lay.layer(R, "mlp_dot") + lay.layer(R, "mlp_norm2")
    def rel(l, r):
        return np.abs(l - r) / np.maximum(np.abs(r), 1e-30)
    return {"attention": dist(rel(lhs_a, rhs_a).ravel()), "mlp": dist(rel(lhs_m, rhs_m).ravel()),
            "note": "relative residual; limited by float32 storage of rho (1/rho^2 cancellation)"}


def determinism(groups):
    """Ledger rows of F1/on (non-graph) vs G1/on (graph) and G1/onnr last-step rows vs G1/on: same kernels -> expect bit-equal.
    Key = (prompt, step, seq, position), so every prefill row (step -1) is compared, not only the last."""
    res = {}
    def idx(g):
        pos = g["rows"][:, 1].astype(int)
        return {(int(p), int(s), int(q), int(x)): i for i, (p, s, q, x) in enumerate(zip(g["prompt"], g["step"], g["seq"], pos))}
    for a, b in ((("run", "F1", "on"), ("run", "G1", "on")), (("run", "G1", "on"), ("run", "G1", "onnr")), (("run", "bf16_G1", "on"), ("run", "bf16_G1", "onnr"))):
        if a in groups and b in groups:
            ia, ib = idx(groups[a]), idx(groups[b])
            ks = sorted(set(ia) & set(ib))
            same = sum(bool(np.array_equal(groups[a]["rows"][ia[k]].view(np.uint32), groups[b]["rows"][ib[k]].view(np.uint32))) for k in ks)
            res[f"{a[1]}/{a[2]}_vs_{b[1]}/{b[2]}"] = f"{same}/{len(ks)}"
    return res


def ledger_read(groups, lay):
    """What the ledger shows on C0/on decode rows (one checkpoint, the prompt set): per-layer means."""
    g = groups.get(("run", "C0", "on"))
    if g is None:
        return None
    R = g["rows"][g["step"] >= 0].astype(np.float64)
    if not len(R):
        return None
    d = lay.d
    x2_in = d * (1 / lay.layer(R, "rho_in") ** 2 - lay.eps)    # ||x||^2 before the attention write
    x2_mid = d * (1 / lay.layer(R, "rho_mid") ** 2 - lay.eps)  # ||r||^2 before the MLP write
    cos_a = lay.layer(R, "attn_dot") / np.sqrt(np.maximum(x2_in * lay.layer(R, "attn_norm2"), 1e-300))
    cos_m = lay.layer(R, "mlp_dot") / np.sqrt(np.maximum(x2_mid * lay.layer(R, "mlp_norm2"), 1e-300))
    ma, mm_ = lay.layer(R, "margin_attn"), lay.layer(R, "margin_mlp")
    marg = R[:, 6]
    ent = lay.layer(R, "h_entropy")
    prm = g["prompt"][g["step"] >= 0]
    per_prompt = {int(q): {"rows": int(np.sum(prm == q)), "margin_mean": float(marg[prm == q].mean()), "margin_median": float(np.median(marg[prm == q])),
                           "sum_attn_mean": float(ma[prm == q].sum(1).mean()), "last_mlp_mean": float(mm_[prm == q, -1].mean())} for q in sorted(set(prm.tolist()))}
    out = {"rows": int(len(R)), "prompts": int(len(set(prm.tolist()))),
           "note": "per-layer statistics over decode rows of 16 prompts; rows of one prompt are autocorrelated, so n_eff is nearer 16 than the row count",
           "margin": dist(marg), "margin_embed_mean": float(R[:, 9].mean()), "margin_embed_median": float(np.median(R[:, 9])),
           "sum_attn": dist(ma.sum(1)), "sum_mlp": dist(mm_.sum(1)), "per_prompt": per_prompt,
           "per_layer": []}
    e2, o2, eo, p2 = (lay.layer(R, n_) for n_ in ("even2", "odd2", "even_odd", "prod2"))
    corr = eo / np.sqrt(np.maximum(e2 * o2, 1e-300))
    def mm(x):
        return float(np.mean(x)), float(np.median(x))
    for l in range(lay.L):
        rec = {"layer": l}
        def put(name, x):
            rec[name + "_mean"], rec[name + "_median"] = mm(x)
        put("margin_attn", ma[:, l]); put("margin_mlp", mm_[:, l])
        put("margin_attn_abs", np.abs(ma[:, l])); put("margin_mlp_abs", np.abs(mm_[:, l]))
        put("pfirst", lay.layer(R, "h_pfirst")[:, l]); put("pfirst_max_head", lay.layer(R, "h_pfirst")[:, l].max(1))
        put("eff_rows", np.exp(ent[:, l])); put("eff_rows_over_sources", np.exp(ent[:, l]) / R[:, 2:3])
        rec["argmax_is_row0_frac"] = float((lay.layer(R, "h_argmax")[:, l] == 0).mean())
        put("gate_small", lay.layer(R, "gate_small")[:, l]); put("gate_off", lay.layer(R, "gate_off")[:, l]); put("gate_on", lay.layer(R, "gate_on")[:, l])
        rec["gate_off_frac_mean"] = rec["gate_off_mean"] / lay.inter
        rec["gate_small_frac_mean"] = rec["gate_small_mean"] / lay.inter
        # SwiGLU census: even = g u / 2, odd = even tanh(g/2), even + odd = SiLU(g) u = product.
        # even2 + odd2 + 2 even_odd = prod2; even and odd cancel where tanh(g/2) ~ -1 (hard-off neurons).
        put("prod2", p2[:, l]); put("even2", e2[:, l]); put("odd2", o2[:, l])
        put("even_odd_corr", corr[:, l])
        put("even2_plus_odd2_over_prod2", (e2[:, l] + o2[:, l]) / np.maximum(p2[:, l], 1e-300))
        put("cos_x_attn_write", cos_a[:, l]); put("cos_r_mlp_write", cos_m[:, l])
        put("attn_write_over_x_norm", np.sqrt(lay.layer(R, "attn_norm2")[:, l] / np.maximum(x2_in[:, l], 1e-300)))
        put("mlp_write_over_r_norm", np.sqrt(lay.layer(R, "mlp_norm2")[:, l] / np.maximum(x2_mid[:, l], 1e-300)))
        out["per_layer"].append(rec)
    return out


# ---------------------------------------------------------------- G3: float64 recomputation from observer ports
def g3_gate(out, rows, lay, groups):
    recs = [r for r in rows if r["type"] == "g3"]
    errs = [r for r in rows if r["type"] == "g3_error"]
    if not recs:
        return {"records": 0, "errors": errs}
    G = np.memmap(out / "g3.f32", dtype=np.float32, mode="r")
    gamma = np.fromfile(out / "g3_static.f32", dtype=np.float32).astype(np.float64)
    L, H, KV, hd, d, h, eps = lay.L, lay.H, lay.kv, lay.hd, lay.d, lay.inter, lay.eps
    group = H // KV
    truths, ledgers, keys, extra = [], [], [], {"probs_vs_softmax64_max_abs": 0.0, "raw_minus_input_plus_aw_bits": 0, "native_margin_vs_ledger_max_abs": 0.0,
                                               "port_consistency_output_vs_next_input_bits": 0}
    for r in recs:
        P = r["position"] + 1
        v = np.array(G[r["offset"]: r["offset"] + r["floats"]])
        sizes = [d, d, d, d, h, h, h, d, d, H * P]
        per = sum(sizes)
        assert r["floats"] == L * per + L * KV * P * hd + 2 * d + 2 + lay.RF, "g3 record size"
        ports = []
        o = 0
        for l in range(L):
            parts = []
            for s in sizes:
                parts.append(v[o: o + s])
                o += s
            ports.append(parts)
        K = v[o: o + L * KV * P * hd].reshape(L, KV, P, hd).astype(np.float64)
        o += L * KV * P * hd
        Wa, Wb = v[o: o + d].astype(np.float64), v[o + d: o + 2 * d].astype(np.float64)
        o += 2 * d
        la, lb = float(v[o]), float(v[o + 1])
        o += 2
        led = v[o: o + lay.RF].astype(np.float64)
        T = np.full(lay.RF, np.nan)
        T[0], T[1], T[2], T[3] = r["token"], r["position"], P, 1
        T[4], T[5] = led[4], led[5]  # NOT a check: top1/top2 are verified independently in G3_top12_float64 (Part 4 logits)
        x0 = ports[0][0].astype(np.float64)
        for l in range(L):
            inp, q, aw, raw, g, u, prod, mw, outp, pr = [p.astype(np.float64) for p in ports[l]]
            extra["raw_minus_input_plus_aw_bits"] += int(np.sum((ports[l][0] + ports[l][2]).view(np.uint32) != ports[l][3].view(np.uint32)))
            if l + 1 < L:
                extra["port_consistency_output_vs_next_input_bits"] += int(np.sum(ports[l][8].view(np.uint32) != ports[l + 1][0].view(np.uint32)))
            b = lay.cols
            T[b["rho_in"][l]] = 1 / np.sqrt(np.mean(inp**2) + eps)
            T[b["rho_mid"][l]] = 1 / np.sqrt(np.mean(raw**2) + eps)
            T[b["attn_dot"][l]], T[b["attn_norm2"][l]] = inp @ aw, aw @ aw
            T[b["mlp_dot"][l]], T[b["mlp_norm2"][l]] = raw @ mw, mw @ mw
            T[b["gate_small"][l]], T[b["gate_off"][l]], T[b["gate_on"][l]] = np.sum(np.abs(g) < 1), np.sum(g <= -5), np.sum(g >= 5)
            even = g * u * 0.5
            odd = even * np.tanh(g * 0.5)
            T[b["even2"][l]], T[b["odd2"][l]], T[b["even_odd"][l]], T[b["prod2"][l]] = even @ even, odd @ odd, even @ odd, prod @ prod
            qh = q.reshape(H, hd)
            s = np.stack([K[l, j // group] @ qh[j] for j in range(H)]) / np.sqrt(hd)  # (H, P)
            m = s.max(1, keepdims=True)
            logz = (m[:, 0] + np.log(np.exp(s - m).sum(1)))
            p = pr.reshape(H, P)
            sm = np.exp(s - m)
            sm /= sm.sum(1, keepdims=True)
            extra["probs_vs_softmax64_max_abs"] = max(extra["probs_vs_softmax64_max_abs"], float(np.abs(sm - p).max()))
            srt = -np.sort(-sm, axis=1)
            gap = srt[:, 0] - srt[:, 1]
            lam = led[b["h_argmax"][l * H: (l + 1) * H]].astype(int)
            mis = lam != sm.argmax(1)
            extra.setdefault("argmax_vs_p64_mismatch", 0)
            extra.setdefault("argmax_vs_p64_mismatch_not_near_tie", 0)
            extra.setdefault("argmax_vs_p64_min_gap_at_mismatch", None)
            extra["argmax_vs_p64_mismatch"] += int(mis.sum())
            extra["argmax_vs_p64_mismatch_not_near_tie"] += int(np.sum(mis & (gap >= ARGMAX_TIE)))
            if mis.any():
                g_ = float(gap[mis].max())
                extra["argmax_vs_p64_min_gap_at_mismatch"] = g_ if extra["argmax_vs_p64_min_gap_at_mismatch"] is None else max(extra["argmax_vs_p64_min_gap_at_mismatch"], g_)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0).sum(1)
            vals = {"h_logz": logz, "h_pfirst": p[:, 0], "h_entropy": ent, "h_pmax": p.max(1), "h_argmax": p.argmax(1).astype(np.float64)}
            for f, name in enumerate(HEAD_ARR_F):
                T[b[name][l * H: (l + 1) * H]] = vals[name]
        xF = ports[L - 1][8].astype(np.float64)
        rhoF = 1 / np.sqrt(np.mean(xF**2) + eps)
        c = gamma * (Wa - Wb)
        T[7], T[8] = rhoF, np.sqrt(c @ c)
        T[9] = rhoF * (c @ x0)
        tot = T[9]
        for l in range(L):
            T[lay.cols["margin_attn"][l]] = rhoF * (c @ ports[l][2].astype(np.float64))
            T[lay.cols["margin_mlp"][l]] = rhoF * (c @ ports[l][7].astype(np.float64))
            tot += T[lay.cols["margin_attn"][l]] + T[lay.cols["margin_mlp"][l]]
        T[10] = tot
        T[6] = rhoF * (c @ xF)  # float64 margin of the native final state
        extra["native_margin_vs_ledger_max_abs"] = max(extra["native_margin_vs_ledger_max_abs"], abs((la - lb) - led[6]))
        truths.append(T)
        ledgers.append(led)
        keys.append((r["prompt"], r["position"]))
    Tm, Lm = np.array(truths), np.array(ledgers)
    res = {"records": len(recs), "errors": errs, "consistency": extra, "lean_vs_truth": field_diff(lay, Tm, Lm),
           "lean_vs_truth_norm": norm_err(lay, Tm, Lm), "note": "top1/top2 are not checked here (same source as the ledger); see G3_top12_float64"}
    fast = groups.get(("run", "FORCE_F1", "on"))
    if fast is not None:
        kf = keyed(fast, lay)
        sel = [i for i, k in enumerate(keys) if k in kf]
        if sel:
            res["fast_vs_truth"] = field_diff(lay, Tm[sel], fast["rows"][[kf[keys[i]] for i in sel]])
            res["fast_vs_truth_norm"] = norm_err(lay, Tm[sel], fast["rows"][[kf[keys[i]] for i in sel]])
            res["fast_rows_matched"] = len(sel)
    return res


# ---------------------------------------------------------------- Part 4: screened exact head (feasibility arithmetic)
def safetensor(name):
    for f in sorted(HF_DIR.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            if name in hdr:
                e = hdr[name]
                a, b = e["data_offsets"]
                fh.seek(8 + n + a)
                raw = np.frombuffer(fh.read(b - a), dtype=np.uint16 if e["dtype"] == "BF16" else np.float32)
                if e["dtype"] == "BF16":
                    raw = (raw.astype(np.uint32) << 16).view(np.float32)
                elif e["dtype"] != "F32":
                    raise ValueError(f"dtype {e['dtype']}")
                return raw.reshape(e["shape"])
    raise KeyError(name)


def topk2(chunk_vals, best):
    """Running top-2 (values) per column: best = (b1, b2) arrays (S,)."""
    if chunk_vals.shape[0] >= 2:
        part = -np.partition(-chunk_vals, 1, axis=0)[:2]
    else:
        part = np.vstack([chunk_vals, np.full_like(chunk_vals, -np.inf)])
    allv = np.vstack([best[0][None], best[1][None], part])
    srt = -np.sort(-allv, axis=0)
    return srt[0], srt[1]


def part4(out, rows, plan, header):
    t0 = time.monotonic()
    st = next((r for r in rows if r["type"] == "states"), None)
    if not st or not st["count"]:
        return {"error": "no states"}
    d, V = header["hidden"], header["vocab"]
    N = np.fromfile(out / "states.f32", dtype=np.float32).reshape(st["count"], d)
    cap = plan.get("part4_max_states")
    if cap and len(N) > cap:
        N = N[:cap]
    S = len(N)
    W = safetensor("model.embed_tokens.weight").astype(np.float32)
    gamma = safetensor("model.norm.weight").astype(np.float64)
    assert W.shape == (V, d), W.shape
    res = {"label": "feasibility arithmetic on dumped lean decode states; not a measured speedup", "states": S, "vocab": V, "hidden": d}
    gs = out / "g3_static.f32"
    if gs.exists():
        res["gamma_equals_tcm_bits"] = bool(np.array_equal(np.fromfile(gs, dtype=np.float32), gamma.astype(np.float32)))
    nz = gamma != 0
    n = np.zeros((S, d))
    n[:, nz] = N[:, nz].astype(np.float64) / gamma[nz]  # final normalized state without gain: logits = M n, M = W diag(gamma)
    CH = 8192
    Gm = np.zeros((d, d))
    rown2 = np.zeros(V)
    for i in range(0, V, CH):
        Mc = W[i: i + CH].astype(np.float64) * gamma
        Gm += Mc.T @ Mc
        rown2[i: i + CH] = np.einsum("ij,ij->i", Mc, Mc)
    ev, U = np.linalg.eigh(Gm)
    order = np.argsort(ev)[::-1]
    ev, U = np.clip(ev[order], 0, None), U[:, order]
    energy = np.cumsum(ev) / ev.sum()
    ks = sorted(int(k) for k in plan.get("part4_ks", [32, 64, 128, 256, 512]))
    kmax = ks[-1]
    P = U[:, :kmax]
    A = np.zeros((V, kmax))
    for i in range(0, V, CH):
        A[i: i + CH] = (W[i: i + CH].astype(np.float64) * gamma) @ P
    cumA = np.cumsum(A**2, axis=1)
    PAD = 1e-10
    tails = {k: np.sqrt(np.maximum(rown2 - cumA[:, k - 1], 0) + PAD * rown2) for k in ks}  # row tail ||M_v (I - P_k P_k^T)||, padded
    del cumA
    proj = n @ P
    nn2 = np.einsum("ij,ij->i", n, n)
    cumP = np.cumsum(proj**2, axis=1)
    rt = {k: np.sqrt(np.maximum(nn2 - cumP[:, k - 1], 0) + PAD * nn2) for k in ks}  # state tail ||(I - P_k P_k^T) n||
    # pass 1: top-2 of lower bounds and of exact logits, max of approx (uniform-bound variant)
    lo = {k: (np.full(S, -np.inf), np.full(S, -np.inf)) for k in ks}
    amax = {k: np.full(S, -np.inf) for k in ks}
    ex_best = (np.full(S, -np.inf), np.full(S, -np.inf))
    ex3 = np.full((3, S), -np.inf)  # exact top-3 values (for near-tie classification of top1/top2)
    ex_id = np.zeros(S, dtype=np.int64)
    ex_id2 = np.zeros(S, dtype=np.int64)
    exact_v = np.full(S, -np.inf)
    for i in range(0, V, CH):
        Mc = W[i: i + CH].astype(np.float64) * gamma
        E = Mc @ n.T  # (Vc, S) exact float64 logits
        j = E.argmax(0)
        better = E[j, np.arange(S)] > exact_v
        ex_id[better] = i + j[better]
        exact_v[better] = E[j, np.arange(S)][better]
        ex_best = topk2(E, ex_best)
        k3 = -np.partition(-E, min(2, E.shape[0] - 1), axis=0)[:3] if E.shape[0] >= 3 else E
        ex3 = -np.sort(-np.vstack([ex3, k3]), axis=0)[:3]
        for k in ks:
            ap = A[i: i + CH, :k] @ proj[:, :k].T
            lo[k] = topk2(ap - tails[k][i: i + CH, None] * rt[k][None], lo[k])
            amax[k] = np.maximum(amax[k], ap.max(0))
    # exact runner-up id: the column entry equal to the second value (ties resolved to any matching id)
    for i in range(0, V, CH):
        E = (W[i: i + CH].astype(np.float64) * gamma) @ n.T
        hit = (E == ex_best[1][None]) & ((i + np.arange(E.shape[0]))[:, None] != ex_id[None])
        rr, cc = np.nonzero(hit)
        ex_id2[cc] = i + rr
    lean_t1 = np.array(st["top1"][:S])
    res["exact64_top1_equals_lean_top1"] = f"{int(np.sum(ex_id == lean_t1))}/{S}"
    res["_exact"] = {"prompt": np.array(st["prompt"][:S]), "position": np.array(st["position"][:S]), "top1": ex_id, "top2": ex_id2,
                     "v1": ex3[0], "v2": ex3[1], "v3": ex3[2]}
    res["singular_spectrum"] = {"sigma_max": float(np.sqrt(ev[0])), "energy_at_k": {str(k): float(energy[k - 1]) for k in ks},
                                "sigma_at_k": {str(k): float(np.sqrt(ev[k - 1])) for k in ks}, "energy_at_1024": float(energy[1023]) if d >= 1024 else None}
    res["row_norm"] = dist(np.sqrt(rown2))
    Tmax = {k: float(tails[k].max()) for k in ks}
    # pass 2: count candidates and check that the exact winners are inside
    c1 = {k: np.zeros(S, np.int64) for k in ks}
    c2 = {k: np.zeros(S, np.int64) for k in ks}
    cu = {k: np.zeros(S, np.int64) for k in ks}
    miss1 = {k: 0 for k in ks}
    miss2 = {k: 0 for k in ks}
    for i in range(0, V, CH):
        ids = i + np.arange(min(CH, V - i))
        for k in ks:
            ap = A[i: i + CH, :k] @ proj[:, :k].T
            up = ap + tails[k][i: i + CH, None] * rt[k][None]
            c1[k] += (up >= lo[k][0][None]).sum(0)
            c2[k] += (up >= lo[k][1][None]).sum(0)
            cu[k] += (ap >= (amax[k] - 2 * Tmax[k] * rt[k])[None]).sum(0)
            for s_ in range(S):
                for tid, thr, miss in ((ex_id[s_], lo[k][0][s_], miss1), (ex_id2[s_], lo[k][1][s_], miss2)):
                    if i <= tid < i + len(ids) and up[tid - i, s_] < thr:
                        miss[k] += 1
    tab = {}
    full = V * d
    for k in ks:
        def cand(c):
            c = c.astype(np.float64)
            return {"median": float(np.median(c)), "p90": pct(c, 90), "max": float(c.max()), "frac_le_1pct_V": float(np.mean(c <= 0.01 * V)), "frac_le_10pct_V": float(np.mean(c <= 0.10 * V))}
        macs1 = (k * d + k * V + c1[k] * d) / full
        macsu = (k * d + k * V + cu[k] * d) / full
        tab[str(k)] = {"energy": float(energy[k - 1]), "row_tail": dist(tails[k]), "row_tail_over_row_norm": dist(tails[k] / np.sqrt(rown2)),
                       "state_tail_over_norm": dist(rt[k] / np.sqrt(nn2)),
                       "top1_candidates": cand(c1[k]), "top2_candidates": cand(c2[k]), "uniform_bound_top1_candidates": cand(cu[k]),
                       "macs_ratio_top1": dist(macs1), "macs_ratio_uniform": dist(macsu), "frac_states_saving_top1": float(np.mean(macs1 < 1)),
                       "screen_only_macs_ratio": (k * d + k * V) / full, "exact_top1_outside_candidates": miss1[k], "exact_top2_outside_candidates": miss2[k],
                       "max_row_tail": Tmax[k], "bound_pad": f"sqrt(t^2 + {PAD}*||.||^2)"}
    res["per_k"] = tab
    best = min(ks, key=lambda k: tab[str(k)]["macs_ratio_top1"]["median"])
    bm = tab[str(best)]["macs_ratio_top1"]["median"]
    res["verdict"] = (f"saves work: best median MAC ratio {bm:.3f} at k={best}" if bm < 1 else
                      f"no k saves work: best median MAC ratio {bm:.3f} at k={best} (>= 1 means scoring all rows is cheaper)")
    res["seconds"] = time.monotonic() - t0
    return res


def g3_top12(groups, lay, ex, tie):
    """Ledger top1/top2 of C0/on decode rows vs float64 top-1/top-2 ids from the dumped final normalized
    states and the readout matrix (Part 4 arithmetic, independent of the native logits and of S.top)."""
    g = groups.get(("run", "C0", "on"))
    if g is None or ex is None:
        return {"error": "missing C0/on rows or Part 4 exact ids"}
    kl = keyed(g, lay)
    rows_t1, rows_t2, gap12, gap23, found = [], [], [], [], []
    for i, (p, q) in enumerate(zip(ex["prompt"], ex["position"])):
        j = kl.get((int(p), int(q)))
        if j is None:
            continue
        found.append(i)
        rows_t1.append(g["rows"][j, 4])
        rows_t2.append(g["rows"][j, 5])
    f = np.array(found)
    t1, t2 = np.array(rows_t1).astype(np.int64), np.array(rows_t2).astype(np.int64)
    e1, e2 = ex["top1"][f], ex["top2"][f]
    g12, g23 = (ex["v1"] - ex["v2"])[f], (ex["v2"] - ex["v3"])[f]
    m1, m2 = t1 != e1, t2 != e2
    near = [{"prompt": int(ex["prompt"][f[i]]), "position": int(ex["position"][f[i]]), "ledger": [int(t1[i]), int(t2[i])], "float64": [int(e1[i]), int(e2[i])],
             "gap12": float(g12[i]), "gap23": float(g23[i])} for i in np.nonzero((g12 < tie) | (g23 < tie) | m1 | m2)[0][:50]]
    return {"states_matched": int(len(f)), "top1_equal": f"{int(np.sum(~m1))}/{len(f)}", "top2_equal": f"{int(np.sum(~m2))}/{len(f)}",
            "top1_mismatch_not_near_tie": int(np.sum(m1 & (g12 >= tie))), "top2_mismatch_not_near_tie": int(np.sum(m2 & (g23 >= tie) & (g12 >= tie))),
            "near_tie_states_top1": int(np.sum(g12 < tie)), "near_tie_states_top2": int(np.sum(g23 < tie)),
            "gap12": dist(g12), "gap23": dist(g23), "near_tie_or_mismatch_cases": near, "tie_threshold_logit": tie}


# ---------------------------------------------------------------- job body
def body(out, plan, box):
    meta = {"job_id": os.environ.get("MRUN_JOB_ID"), "plan": plan, "sources": json.loads((ROOT / "sources.json").read_text()),
            "worker_sha256": sha(Path(__file__))}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    nv = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    meta.update({"gpu": smi, "nvcc": nv, "backend": "c-cublas (cmachine lean + qwen_fast) with per-token ledger", "device": "cuda", "dtype": "fp32 (+bf16 exact-in-weights lane)",
                 "hostname": os.uname().nodename, "python": sys.version.split()[0], "model_tcm": str(MODEL)})
    box.record("meta", outbox.jsonable(meta), kind="meta")
    set_phase("baseline")
    time.sleep(3.0)
    set_phase("build")
    _, dt = run(["make", "-C", CAND, "cuda"], timeout=900, log=out / "build_cuda.log")
    _, dt2 = run(["make", "-C", CAND, "ledger-run"], timeout=600, log=out / "build_ledger.log")
    meta["build_s"] = dt + dt2
    meta["binary_sha256"] = sha(CAND / "build/ledger-run")
    set_phase("tokenize")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR, local_files_only=True)
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS[: plan["prompts"]]]
    (out / "prompts.txt").write_text("".join(f"{len(v)} {' '.join(map(str, v))}\n" for v in ids))
    meta["prompt_tokens"] = [len(v) for v in ids]
    t = time.monotonic()
    set_phase("c_launch")
    cmd = [str(CAND / "build/ledger-run"), str(MODEL), str(out / "prompts.txt"), str(out), "--steps", str(plan["steps"]), "--warmup-steps", str(plan["warmup_steps"]),
           "--smoke-steps", str(plan["smoke_steps"]), "--g3-prompts", str(plan["g3_prompts"]), "--g3-tokens", str(plan["g3_tokens"]),
           "--batches", ",".join(map(str, plan["batches"])), "--bf16", str(int(plan.get("bf16", 1))), "--profile", str(int(plan.get("profile", 0))),
           "--phase-file", str(_PHASE["file"])]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=plan["c_timeout_s"])
    (out / "c_run.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
    meta["c_process_s"] = time.monotonic() - t
    meta["c_returncode"] = r.returncode
    set_phase("analysis")
    rows = load_jsonl(out / "runs.jsonl") if (out / "runs.jsonl").exists() else []
    box.put_file(out / "runs.jsonl", "runs.jsonl")
    meta["smoke"] = [x for x in rows if x["type"] == "smoke_result"]
    meta["convert"] = next((x for x in rows if x["type"] == "convert"), None)
    meta["footer"] = next((x for x in rows if x["type"] == "footer"), None)
    meta["int8_refusal"] = next((x for x in rows if x["type"] == "int8_refusal"), None)
    if r.returncode:
        meta["c_error_tail"] = r.stderr[-3000:]
    if not any(x["type"] == "smoke_result" and x["phase"] == "fp32" and x["ok"] for x in rows):
        raise RuntimeError(f"FP32 smoke failed or C process died (rc={r.returncode}): {r.stderr[-2000:]}")
    return meta, rows


def profile_table(table):
    """Profile arms P<B>_off / P<B>_s<mask>: cost of each piece = full ledger (s0) minus the piece skipped."""
    names = {1: "rms_dots", 2: "attention_softmax_fields", 4: "swiglu_gate_census", 8: "readout_top2_margin", 15: "all_pieces_skipped (row init only)"}
    out = {}
    for B in sorted({t["B"] for k, t in table.items() if k.startswith("P")}):
        off, full = table.get(f"P{B}_off/off"), table.get(f"P{B}_s0/onnr")
        if not off or not full:
            continue
        rec = {"off_ms": off["ms_median"], "full_ms": full["ms_median"], "ledger_ms": full["ms_median"] - off["ms_median"],
               "ledger_pct": 100 * (full["ms_median"] / off["ms_median"] - 1), "pieces": {}}
        for m, n in names.items():
            t = table.get(f"P{B}_s{m}/onnr")
            if t:
                rec["pieces"][n] = {"ms_with_piece_skipped": t["ms_median"], "piece_cost_ms": full["ms_median"] - t["ms_median"]}
        out[f"B{B}"] = rec
    return out


def analyze(out, rows, samples, plan, report):
    header = next(r for r in rows if r["type"] == "header")
    lay = Layout(header)
    report["header"] = header
    report["timing"], report["ledger_overhead"], report["baseline_check"] = timing(rows, samples)
    if plan.get("profile"):
        report["profile"] = profile_table(report["timing"])
        return
    report["G1_bit_identity"] = g1_gate(out, rows, header)
    groups = load_ledger_rows(out, rows, lay)
    _GROUPS["groups"] = groups
    report["ledger_row_counts"] = {f"{k[0]}:{k[1]}/{k[2]}": int(len(v["rows"])) for k, v in groups.items()}
    report["G2_lean_vs_fast"] = g2_gate(groups, lay)
    report["G4_margin_closure"] = g4_closure(groups, lay)
    report["ledger_determinism"] = determinism(groups)
    lean = groups.get(("run", "C0", "on"))
    if lean is not None:
        report["rms_identity_C0_on"] = rms_identity(lean["rows"], lay)
    report["ledger_read"] = ledger_read(groups, lay)
    set_phase("g3")
    report["G3_independent_truth"] = g3_gate(out, rows, lay, groups)


def main():
    global ARGMAX_TIE
    out = Path(sys.argv[1]).resolve()
    plan = json.loads(Path(sys.argv[2]).read_text())
    ARGMAX_TIE = float((plan.get("criteria") or {}).get("argmax_tie_prob", ARGMAX_TIE))
    out.mkdir(parents=True, exist_ok=False)
    _PHASE["file"] = out / "phase.txt"
    set_phase("init")
    box = outbox.default()
    sampler = Sampler(_PHASE["file"], out / "samples.csv", box, plan.get("sample_interval_s", 0.5))
    sampler.sample()
    sampler.start()
    report = {"status": "failed"}
    t0 = time.monotonic()
    rows = []
    try:
        meta, rows = body(out, plan, box)
        report = {"status": "complete" if meta["c_returncode"] == 0 else "partial", "meta": meta}
    except Exception:  # noqa: BLE001
        report = {"status": "failed", "error": traceback.format_exc(), "worker_sha256": sha(Path(__file__))}
        print(report["error"], flush=True)
        try:
            if (out / "runs.jsonl").exists():
                part = load_jsonl(out / "runs.jsonl")
                report["smoke"] = [x for x in part if x["type"] == "smoke_result"]
                box.put_file(out / "runs.jsonl", "runs.jsonl")
            for name in ("c_run.log", "build_cuda.log", "build_ledger.log"):
                if (out / name).exists():
                    box.put_file(out / name, name)
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            if rows and any(r["type"] == "header" for r in rows):
                analyze(out, rows, sampler.rows, plan, report)
        except Exception:  # noqa: BLE001
            report["analysis_error"] = traceback.format_exc()
            print(report["analysis_error"], flush=True)
        if plan.get("part4", True) and rows and any(r["type"] == "states" for r in rows):
            set_phase("part4")
            try:
                report["part4_screened_head"] = part4(out, rows, plan, next(r for r in rows if r["type"] == "header"))
            except Exception:  # noqa: BLE001
                report["part4_error"] = traceback.format_exc()
                print(report["part4_error"], flush=True)
        crit = plan.get("criteria")
        if crit and rows and any(r["type"] == "header" for r in rows) and not plan.get("profile"):
            try:
                hdr = next(r for r in rows if r["type"] == "header")
                lay = Layout(hdr)
                ex = (report.get("part4_screened_head") or {}).pop("_exact", None)
                groups = _GROUPS.get("groups") or load_ledger_rows(out, rows, lay)
                report["G3_top12_float64"] = g3_top12(groups, lay, ex, crit["top12_tie_logit"])
                report["verdicts"] = judge(report, crit)
            except Exception:  # noqa: BLE001
                report["judge_error"] = traceback.format_exc()
                print(report["judge_error"], flush=True)
        (report.get("part4_screened_head") or {}).pop("_exact", None)
        set_phase("done")
        time.sleep(1.0)
        sampler.finish()
        report["worker_seconds"] = time.monotonic() - t0
        report["sampler"] = {"samples": len(sampler.rows), "errors": sampler.errors, "interval_s": plan.get("sample_interval_s", 0.5)}
        if sampler.rows:
            report["resource_by_phase"] = summarize_phases(sampler.rows)
            foreign = {}
            for r in sampler.rows:
                for item in filter(None, r["foreign"].split(";")):
                    k, v = item.split(":")[0], float(item.split(":")[1])
                    foreign.setdefault(k, [0, 0.0])
                    foreign[k][0] += 1
                    foreign[k][1] = max(foreign[k][1], v)
            report["foreign_compute_pids"] = {k: {"samples": v[0], "max_mib": v[1]} for k, v in foreign.items()}
            report["own_gpu_mib_max"] = max((r["own_gpu_mib"] or 0) for r in sampler.rows)
            report["tree_rss_mb_max"] = max((r["tree_rss_mb"] or 0) for r in sampler.rows)
            report["rss_source"] = "tree_rss_mb_max: worker sampler, RSS summed over the worker process tree, every sample_interval_s"
        (out / "results.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
        try:
            box.put_file(out / "samples.csv", "samples.csv")
            box.put_file(out / "results.json", "results.json")
            for name in ("c_run.log", "prompts.txt", "build_cuda.log", "build_ledger.log"):
                if (out / name).exists():
                    box.put_file(out / name, name)
            box.report(outbox.jsonable(report))
        except Exception:  # noqa: BLE001
            print("outbox report failed:\n" + traceback.format_exc(), flush=True)
        slim = {k: report.get(k) for k in ("status", "profile", "ledger_overhead", "baseline_check", "G1_bit_identity", "ledger_determinism", "own_gpu_mib_max", "tree_rss_mb_max",
                                           "worker_seconds", "error", "analysis_error", "part4_error", "judge_error", "foreign_compute_pids")}
        slim["verdicts"] = {k: (v.get("pass"), v.get("failed")) if isinstance(v, dict) else v for k, v in (report.get("verdicts") or {}).items() if k != "criteria"}
        slim["smoke"] = (report.get("meta") or {}).get("smoke")
        slim["part4_verdict"] = (report.get("part4_screened_head") or {}).get("verdict")
        print("LEDGER_REPORT " + json.dumps(slim, default=str), flush=True)
        if report["status"] == "failed":
            sys.exit(1)


if __name__ == "__main__":
    main()
