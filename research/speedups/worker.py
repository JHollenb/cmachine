"""Beast worker: cmachine speedup arms (research/speedups, docs/SPEEDUPS.md). Run from the payload root:
    python worker.py OUTPUT_DIR PLAN_JSON

Order: sampler first (one synchronous probe aborts the job if nvidia-smi/psutil fail) -> idle baseline ->
CUDA build (make cuda, make speedups-run) -> tokenize -> C process (FP32 smoke -> arms -> hashing ->
BF16/INT8 conversion -> per-lane smoke -> lane arms) -> analysis -> sampler stopped last.
Samples go to the mrun outbox (kind=sample rows) and samples.csv."""
from __future__ import annotations

import csv
import hashlib
import json
import os
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
PEAK_GBPS = 669.9          # measured copy bandwidth (overhead job)
PEAK_FP32_TFLOPS = 48.7    # vendor spec, non-tensor FP32 (asserted, not measured)
PROMPTS = [
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
_PHASE = {"file": None}


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


# ---------------------------------------------------------------- analysis
def softmax64(x):
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def logit_stats(ref, arm):
    d = np.abs(ref.astype(np.float64) - arm.astype(np.float64))
    p, q = softmax64(ref), softmax64(arm)
    kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "kl_ref_arm": kl, "top1_same": int(ref.argmax() == arm.argmax()),
            "bit_identical": bool(np.array_equal(ref.view(np.uint32), arm.view(np.uint32)))}


def analyze(out, rows, samples):
    header = next(r for r in rows if r["type"] == "header")
    V, steps, NP = header["vocab"], header["steps"], header["prompts"]
    lane_bytes = header["lane_weight_bytes"]
    params = lane_bytes["fp32"] / 4  # every weight element, matrices + vectors
    runs = [r for r in rows if r["type"] == "run"]
    c0 = {r["prompts"][0]: r["ids"][0] for r in runs if r["arm"] == "C0"}
    arms = {}
    for r in runs:
        a = arms.setdefault(r["arm"], {"lane": r["lane"], "B": r["B"], "graph": r["graph"], "tf32": r["tf32"], "forced": r["forced"], "prefill_mode": r["prefill_mode"],
                                       "ms": [], "prefill": [], "regions": 0, "contaminated_regions": 0, "seqs": 0, "seqs_identical": 0, "first_div": [],
                                       "token_agree": [], "extra_ms": [], "hash": []})
        a["ms"].extend(r["ms"])
        a["prefill"].extend(r["prefill_ms"])
        a["regions"] += 1
        a["contaminated_regions"] += region_contaminated(samples, r["t_start"], r["t_end"])
        a["extra_ms"].append(r.get("extra_ms", 0.0))
        if r.get("hash_mode"):
            a["hash"].append({"prompt": r["prompts"][0], "chain": r["chain"], "verify_ok": r["verify_ok"], "ms": r["ms"], "p0": r["prompt_tokens"][0]})
        if not r["forced"]:
            for p, ids in zip(r["prompts"], r["ids"]):
                ref = c0.get(p)
                if ref is None:
                    continue
                a["seqs"] += 1
                same = ids == ref
                a["seqs_identical"] += same
                d = next((i for i in range(min(len(ids), len(ref))) if ids[i] != ref[i]), None)
                if d is not None:
                    a["first_div"].append(d)
                a["token_agree"].append(float(np.mean([u == v for u, v in zip(ids, ref)])))
    base = float(np.median(arms["C0"]["ms"])) if "C0" in arms else None
    table = {}
    for name, a in arms.items():
        ms = np.array(a["ms"])
        med = float(np.median(ms))
        wb = lane_bytes[a["lane"]]
        row = {"lane": a["lane"], "B": a["B"], "graph": a["graph"], "tf32": a["tf32"], "forced": a["forced"], "prefill_mode": a["prefill_mode"], "n_steps": int(ms.size),
               "ms_median": med, "ms_p10": float(np.percentile(ms, 10)), "ms_p90": float(np.percentile(ms, 90)), "ms_mean": float(ms.mean()),
               "tok_per_s": a["B"] * 1e3 / med, "eff_GBps": wb / (med / 1e3) / 1e9, "frac_of_669.9": wb / (med / 1e3) / 1e9 / PEAK_GBPS,
               "flops_per_step": 2 * params * a["B"], "achieved_TFLOPs": 2 * params * a["B"] / (med / 1e3) / 1e12,
               "frac_of_48.7_vendor": 2 * params * a["B"] / (med / 1e3) / 1e12 / PEAK_FP32_TFLOPS,
               "prefill_ms_median": float(np.median(a["prefill"])) if a["prefill"] else None, "prefill_ms_p10": float(np.percentile(a["prefill"], 10)) if a["prefill"] else None,
               "prefill_ms_p90": float(np.percentile(a["prefill"], 90)) if a["prefill"] else None,
               "contaminated": a["contaminated_regions"] > 0, "contaminated_regions": f"{a['contaminated_regions']}/{a['regions']}",
               "graph_capture_ms_median": float(np.median(a["extra_ms"])) if a["graph"] else None}
        if base:
            row["speedup_vs_C0_per_step"] = base / med
            row["throughput_vs_C0"] = (a["B"] * 1e3 / med) / (1e3 / base)
        if a["seqs"]:
            row["ids_identical_to_C0"] = f"{a['seqs_identical']}/{a['seqs']}"
            row["first_divergence_index"] = sorted(a["first_div"])
            row["mean_token_agreement"] = float(np.mean(a["token_agree"]))
        table[name] = row
    # ---- logit gates ----
    banks = {}
    for r in rows:
        if r["type"] == "bank":
            path = out / f"logits-{r['name']}.f32"
            banks[r["name"]] = (np.fromfile(path, dtype=np.float32).reshape(NP, V), r["filled"])
    def ctx_same(bank):
        # final logits share C0's context iff the arm fed C0's ids[0..steps-1]; forced banks always do
        if "FORCE" in bank:
            return {p: True for p in range(NP)}
        arm = bank.replace("final_", "")
        res = {}
        for r in runs:
            if r["arm"] == arm and not r["forced"]:
                for p, ids in zip(r["prompts"], r["ids"]):
                    if p in c0 and p not in res:
                        res[p] = ids[:steps] == c0[p][:steps]
        return res
    gates = {}
    for name, (data, filled) in banks.items():
        if name in ("first_C0", "final_C0"):
            continue
        refname = "first_C0" if name.startswith("first_") else "final_C0"
        if refname not in banks:
            continue
        ref, rf = banks[refname]
        same = ctx_same(name) if name.startswith("final_") else {p: True for p in range(NP)}
        per = []
        for p in range(NP):
            if filled[p] and rf[p]:
                s = logit_stats(ref[p], data[p])
                s["prompt"], s["context_same_as_C0"] = p, bool(same.get(p, False))
                per.append(s)
        valid = [s for s in per if s["context_same_as_C0"]]
        gates[f"{name}_vs_{refname}"] = {
            "prompts": len(per), "context_same": len(valid),
            "max_abs": max((s["max_abs"] for s in valid), default=None), "mean_abs": float(np.mean([s["mean_abs"] for s in valid])) if valid else None,
            "kl_mean": float(np.mean([s["kl_ref_arm"] for s in valid])) if valid else None, "kl_max": max((s["kl_ref_arm"] for s in valid), default=None),
            "top1_same": f"{sum(s['top1_same'] for s in valid)}/{len(valid)}", "bit_identical": f"{sum(s['bit_identical'] for s in valid)}/{len(valid)}",
            "per_prompt": per}
    # graph vs non-graph bitwise
    pairs = [("final_G1", "final_F1"), ("final_G8", "final_B8"), ("final_G32", "final_B32"), ("final_G64", "final_B64"), ("final_TF32_G64", "final_TF32_B64"),
             ("final_bf16_G1", "final_bf16_F1"), ("final_int8_G1", "final_int8_F1")]
    graph_bitwise = {}
    for g, n in pairs:
        if g in banks and n in banks:
            (A, fa), (B, fb) = banks[g], banks[n]
            ps = [p for p in range(NP) if fa[p] and fb[p]]
            graph_bitwise[f"{g}_vs_{n}"] = f"{sum(np.array_equal(A[p].view(np.uint32), B[p].view(np.uint32)) for p in ps)}/{len(ps)}"
    for g, n in [("G1", "F1"), ("G8", "B8"), ("G32", "B32"), ("G64", "B64"), ("TF32_G64", "TF32_B64"), ("bf16_G1", "bf16_F1"), ("bf16_G8", "bf16_B8"), ("int8_G1", "int8_F1"), ("int8_G8", "int8_B8")]:
        ga = {(tuple(r["prompts"]), b): ids for r in runs if r["arm"] == g for b, ids in enumerate(r["ids"])}
        na = {(tuple(r["prompts"]), b): ids for r in runs if r["arm"] == n for b, ids in enumerate(r["ids"])}
        ks = sorted(set(ga) & set(na))
        if ks:
            graph_bitwise[f"ids_{g}_vs_{n}"] = f"{sum(ga[k] == na[k] for k in ks)}/{len(ks)}"
    # ---- hashing ----
    hashing = {}
    h0 = arms.get("H0")
    if h0:
        h0_med = float(np.median(h0["ms"]))
        for mode in ("H0", "HFULL", "HCPU", "HASYNC", "HGPU"):
            a = arms.get(mode)
            if not a:
                continue
            pos, over = [], []
            for h in a["hash"]:
                for i, m in enumerate(h["ms"]):
                    pos.append(h["p0"] + i)
                    over.append(m - h0_med)
            slope, icpt = np.polyfit(pos, over, 1) if len(pos) > 2 else (None, None)
            med = float(np.median(a["ms"]))
            hashing[mode] = {"ms_median": med, "overhead_ms": med - h0_med, "overhead_pct": 100 * (med - h0_med) / h0_med,
                             "fit_overhead_ms": {"intercept": float(icpt), "per_position": float(slope)} if slope is not None else None,
                             "verify_ok": f"{sum(1 for h in a['hash'] if h['verify_ok'] == 1)}/{len(a['hash'])}" if mode in ("HCPU", "HASYNC", "HGPU") else None,
                             "drain_ms_median": float(np.median(a["extra_ms"])) if mode == "HASYNC" else None}
        chains = {}
        for mode in ("HCPU", "HASYNC", "HGPU"):
            for h in arms.get(mode, {}).get("hash", []):
                chains.setdefault(h["prompt"], {})[mode] = h["chain"]
        hashing["chains_equal_across_modes"] = f"{sum(len(set(v.values())) == 1 and len(v) == 3 for v in chains.values())}/{len(chains)}"
    return header, table, gates, graph_bitwise, hashing


def body(out, plan, box):
    meta = {"job_id": os.environ.get("MRUN_JOB_ID"), "plan": plan, "sources": json.loads((ROOT / "sources.json").read_text())}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    nv = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    meta.update({"gpu": smi, "nvcc": nv, "backend": "c-cublas (cmachine lean + qwen_fast)", "device": "cuda", "dtype": "fp32 (+tf32/bf16/int8 opt-in lanes)",
                 "hostname": os.uname().nodename, "python": sys.version.split()[0], "model_tcm": str(MODEL)})
    box.record("meta", outbox.jsonable(meta), kind="meta")
    set_phase("baseline")
    time.sleep(3.0)
    set_phase("build")
    _, dt = run(["make", "-C", CAND, "cuda"], timeout=900, log=out / "build_cuda.log")
    _, dt2 = run(["make", "-C", CAND, "speedups-run"], timeout=600, log=out / "build_speedups.log")
    meta["build_s"] = dt + dt2
    meta["binary_sha256"] = sha(CAND / "build/speedups-run")
    set_phase("tokenize")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR, local_files_only=True)
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS[: plan["prompts"]]]
    (out / "prompts.txt").write_text("".join(f"{len(v)} {' '.join(map(str, v))}\n" for v in ids))
    meta["prompt_tokens"] = [len(v) for v in ids]
    t = time.monotonic()
    set_phase("c_launch")
    r = subprocess.run([str(CAND / "build/speedups-run"), str(MODEL), str(out / "prompts.txt"), str(out), "--steps", str(plan["steps"]), "--warmup-steps", str(plan["warmup_steps"]),
                        "--smoke-steps", str(plan["smoke_steps"]), "--hash-prompts", str(plan["hash_prompts"]), "--phase-file", str(_PHASE["file"])],
                       capture_output=True, text=True, timeout=plan["c_timeout_s"])
    (out / "c_run.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-20000:])
    meta["c_process_s"] = time.monotonic() - t
    meta["c_returncode"] = r.returncode
    set_phase("analysis")
    rows = load_jsonl(out / "runs.jsonl") if (out / "runs.jsonl").exists() else []
    box.put_file(out / "runs.jsonl", "runs.jsonl")
    meta["smoke"] = [x for x in rows if x["type"] == "smoke_result"]
    meta["convert"] = next((x for x in rows if x["type"] == "convert"), None)
    meta["footer"] = next((x for x in rows if x["type"] == "footer"), None)
    if r.returncode:
        meta["c_error_tail"] = r.stderr[-3000:]
    if not any(x["type"] == "smoke_result" and x["phase"] == "fp32" and x["ok"] for x in rows):
        raise RuntimeError(f"FP32 smoke failed or C process died (rc={r.returncode}): {r.stderr[-2000:]}")
    return meta, rows


def main():
    out = Path(sys.argv[1]).resolve()
    plan = json.loads(Path(sys.argv[2]).read_text())
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
        report = {"status": "failed", "error": traceback.format_exc()}
        print(report["error"], flush=True)
    finally:
        set_phase("done")
        time.sleep(1.0)
        sampler.finish()
        try:
            if rows and any(r["type"] == "header" for r in rows):
                header, table, gates, graph_bitwise, hashing = analyze(out, rows, sampler.rows)
                report.update({"arms": table, "logit_gates": gates, "graph_bitwise": graph_bitwise, "hashing": hashing, "header": header})
        except Exception:  # noqa: BLE001
            report["analysis_error"] = traceback.format_exc()
            print(report["analysis_error"], flush=True)
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
        (out / "results.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
        try:
            box.put_file(out / "samples.csv", "samples.csv")
            box.put_file(out / "results.json", "results.json")
            for name in ("c_run.log", "prompts.txt", "build_cuda.log", "build_speedups.log"):
                if (out / name).exists():
                    box.put_file(out / name, name)
            box.report(outbox.jsonable({k: v for k, v in report.items() if k != "logit_gates"}))
        except Exception:  # noqa: BLE001
            print("outbox report failed:\n" + traceback.format_exc(), flush=True)
        slim = {k: report.get(k) for k in ("status", "arms", "graph_bitwise", "hashing", "foreign_compute_pids", "own_gpu_mib_max", "tree_rss_mb_max", "worker_seconds", "error", "analysis_error")}
        slim["gates"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"} for k, v in report.get("logit_gates", {}).items()}
        slim["smoke"] = (report.get("meta") or {}).get("smoke")
        print("SPEEDUPS_REPORT " + json.dumps(slim, default=str), flush=True)
        if report["status"] == "failed":
            sys.exit(1)


if __name__ == "__main__":
    main()
