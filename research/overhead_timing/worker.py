"""Beast worker: cmachine analysis-overhead timing (research/overhead_timing). Run from the payload root:
    python worker.py OUTPUT_DIR PLAN_JSON

Order: sampler first (one synchronous probe aborts the job if nvidia-smi/psutil fail) -> idle baseline ->
CUDA build -> tokenize -> C process (smoke, warmup, arms C0..C4 + C2b, rotated per prompt) -> exits ->
torch process (smoke, warmup, arms T0, T1, T1b, T2, T3) -> analysis -> sampler stopped last.
Samples go to the mrun outbox (kind=sample rows) and to samples.csv, which is stored as an outbox file."""
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
WEIGHT_BYTES = 6_174_857_216
PEAK_GBPS = 669.9
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
                key = p.pid
                if key not in self.procs:
                    self.procs[key] = p
                    p.cpu_percent(None)
                else:
                    cpu += self.procs[key].cpu_percent(None)
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


def pct(a, q):
    return float(np.percentile(a, q))


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


def gpu_segments(rows, threshold=200.0):
    segs, cur = [], None
    for r in rows:
        on = (r["own_gpu_mib"] or 0) > threshold
        if on and cur is None:
            cur = {"rise_t_rel": r["t_rel"], "phase_at_rise": r["phase"], "vals": []}
        if on:
            cur["vals"].append(r["own_gpu_mib"])
            cur["last_on"] = r["t_rel"]
            cur["phase_at_last"] = r["phase"]
        if not on and cur is not None:
            segs.append(cur)
            cur = None
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        v = s.pop("vals")
        s.update({"max_mib": max(v), "plateau_median_mib": float(np.median(v)), "fall_t_rel": s["last_on"]})
        out.append(s)
    return out


def region_contaminated(rows, t0, t1, slack=0.5):
    return any(r["foreign"] for r in rows if t0 - slack <= r["t"] <= t1 + slack)


def analyze(c_rows, t_rows, samples, c_footer, t_footer):
    arms = {}
    def add(rows, family):
        for r in rows:
            if r["type"] != "run":
                continue
            a = arms.setdefault(r["arm"], {"family": family, "ms": [], "prefill": [], "ids": {}, "regions": 0, "contaminated_regions": 0, "capture": []})
            a["ms"].extend(r["ms"])
            a["prefill"].append(r["prefill_ms"])
            a["ids"][r["prompt"]] = r["ids"]
            a["regions"] += 1
            a["contaminated_regions"] += region_contaminated(samples, r["t_start"], r["t_end"])
            a["capture"].append(r.get("obs_bytes_per_token", r.get("capture_bytes_per_token", 0)))
    add(c_rows, "C")
    add(t_rows, "T")
    base = float(np.median(arms["C0"]["ms"]))
    table = {}
    for name, a in arms.items():
        ms = np.array(a["ms"])
        med = float(np.median(ms))
        row = {"family": a["family"], "n_tokens": int(ms.size), "prompts": len(a["ids"]), "ms_median": med, "ms_p10": pct(ms, 10), "ms_p90": pct(ms, 90),
               "ms_mean": float(ms.mean()), "prefill_ms_median": float(np.median(a["prefill"])), "capture_bytes_per_token_median": float(np.median(a["capture"])),
               "contaminated": a["contaminated_regions"] > 0, "contaminated_regions": f"{a['contaminated_regions']}/{a['regions']}"}
        if a["family"] == "C":
            row.update({"overhead_ms": med - base, "overhead_pct": 100 * (med - base) / base, "eff_GBps": WEIGHT_BYTES / (med / 1e3) / 1e9,
                        "frac_of_669.9": WEIGHT_BYTES / (med / 1e3) / 1e9 / PEAK_GBPS})
        table[name] = row
    def agree(a_name, b_name):
        A, B = arms.get(a_name), arms.get(b_name)
        if not A or not B:
            return None
        common = sorted(set(A["ids"]) & set(B["ids"]))
        same, first_div, tokfrac = 0, [], []
        for p in common:
            x, y = A["ids"][p], B["ids"][p]
            ok = x == y
            same += ok
            d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
            if d is not None:
                first_div.append(d)
            tokfrac.append(float(np.mean([u == v for u, v in zip(x, y)])))
        return {"prompts_identical": f"{same}/{len(common)}", "first_divergence_index": first_div, "mean_token_agreement": float(np.mean(tokfrac)) if tokfrac else None}
    agreement = {f"{n}_vs_C0": agree(n, "C0") for n in arms if arms[n]["family"] == "C" and n != "C0"}
    agreement.update({f"{n}_vs_T0": agree(n, "T0") for n in arms if arms[n]["family"] == "T" and n != "T0"})
    agreement["C0_vs_T0"] = agree("C0", "T0")
    return table, agreement


def body(out, plan, box, sampler):
    meta = {"job_id": os.environ.get("MRUN_JOB_ID"), "plan": plan, "sources": json.loads((ROOT / "sources.json").read_text())}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    meta.update({"gpu": smi, "backend": "c-cublas-fp32 (cmachine) + hf-pytorch-fp32-sdpa", "device": "cuda", "dtype": "fp32", "hostname": os.uname().nodename,
                 "python": sys.version.split()[0], "model_tcm": str(MODEL), "model_hf": str(HF_DIR)})
    box.record("meta", outbox.jsonable(meta), kind="meta")
    set_phase("baseline")
    time.sleep(4.0)
    # ---- build ----
    set_phase("build")
    _, dt = run(["make", "-C", CAND, "cuda"], timeout=900)
    run(["make", "-C", CAND, "overhead-run"], timeout=300)
    meta["build_s"] = dt
    meta["binary_sha256"] = sha(CAND / "build/overhead-run")
    # ---- tokenize ----
    set_phase("tokenize")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR, local_files_only=True)
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    (out / "prompts.txt").write_text("".join(f"{len(v)} {' '.join(map(str, v))}\n" for v in ids))
    meta["prompt_tokens"] = [len(v) for v in ids]
    # ---- C process ----
    t = time.monotonic()
    set_phase("c_launch")
    run([CAND / "build/overhead-run", MODEL, out / "prompts.txt", out / "c_runs.jsonl", "--steps", plan["steps"], "--warmup-steps", plan["warmup_steps"],
         "--smoke-steps", plan["smoke_steps"], "--deadline", plan["c_deadline_s"], "--phase-file", _PHASE["file"]], timeout=900, log=out / "c_run.log")
    meta["c_process_s"] = time.monotonic() - t
    set_phase("between")
    c_rows = load_jsonl(out / "c_runs.jsonl")
    box.put_file(out / "c_runs.jsonl", "c_runs.jsonl")
    for r in c_rows:
        if r["type"] == "run":
            box.record(f"run/{r['arm']}/{r['prompt']:02d}", outbox.jsonable(r), kind="run")
    if not any(r["type"] == "smoke_result" and r["ok"] for r in c_rows):
        raise RuntimeError("C smoke failed")
    time.sleep(2.0)
    # ---- torch process ----
    t = time.monotonic()
    set_phase("torch_launch")
    env = dict(os.environ, CM_PHASE_FILE=str(_PHASE["file"]))
    r = subprocess.run([sys.executable, "-u", str(ROOT / "torch_arms.py"), str(HF_DIR), str(out / "prompts.txt"), str(out / "torch_runs.jsonl"), "--steps", str(plan["steps"]),
                        "--warmup-steps", str(plan["warmup_steps"]), "--smoke-steps", str(plan["smoke_steps"]), "--deadline", str(plan["torch_deadline_s"])],
                       capture_output=True, text=True, env=env, timeout=900)
    (out / "torch_run.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-6000:])
    meta["torch_process_s"] = time.monotonic() - t
    set_phase("between2")
    t_rows = load_jsonl(out / "torch_runs.jsonl") if (out / "torch_runs.jsonl").exists() else []
    if t_rows:
        box.put_file(out / "torch_runs.jsonl", "torch_runs.jsonl")
        for x in t_rows:
            if x["type"] == "run":
                box.record(f"run/{x['arm']}/{x['prompt']:02d}", outbox.jsonable(x), kind="run")
    meta["torch_returncode"] = r.returncode
    if r.returncode:
        meta["torch_error_tail"] = r.stderr[-2000:]
    time.sleep(2.0)
    set_phase("analysis")
    c_header = next(x for x in c_rows if x["type"] == "header")
    c_footer = next((x for x in c_rows if x["type"] == "footer"), None)
    t_footer = next((x for x in t_rows if x["type"] == "footer"), None)
    t_header = next((x for x in t_rows if x["type"] == "header"), None)
    meta["c_header"], meta["c_footer"], meta["torch_header"], meta["torch_footer"] = c_header, c_footer, t_header, t_footer
    return meta, c_rows, t_rows, c_footer, t_footer


def main():
    out = Path(sys.argv[1]).resolve()
    plan = json.loads(Path(sys.argv[2]).read_text())
    out.mkdir(parents=True, exist_ok=False)
    _PHASE["file"] = out / "phase.txt"
    set_phase("init")
    box = outbox.default()
    sampler = Sampler(_PHASE["file"], out / "samples.csv", box, plan.get("sample_interval_s", 0.5))
    sampler.sample()  # synchronous probe: abort now if nvidia-smi or psutil cannot be read
    sampler.start()
    report = {"status": "failed"}
    t0 = time.monotonic()
    try:
        meta, c_rows, t_rows, c_footer, t_footer = body(out, plan, box, sampler)
        table, agreement = analyze(c_rows, t_rows, sampler.rows, c_footer, t_footer)
        report = {"status": "complete", "meta": meta, "arms": table, "agreement": agreement}
    except Exception:  # noqa: BLE001
        report = {"status": "failed", "error": traceback.format_exc()}
        print(report["error"], flush=True)
        raise
    finally:
        set_phase("done")
        time.sleep(1.0)
        sampler.finish()
        report["worker_seconds"] = time.monotonic() - t0
        report["sampler"] = {"samples": len(sampler.rows), "errors": sampler.errors, "interval_s": plan.get("sample_interval_s", 0.5)}
        if sampler.rows:
            report["resource_by_phase"] = summarize_phases(sampler.rows)
            report["gpu_memory_segments"] = gpu_segments(sampler.rows)
            foreign = {}
            for r in sampler.rows:
                for item in filter(None, r["foreign"].split(";")):
                    foreign.setdefault(item.split(":")[0], [0, 0.0])
                    foreign[item.split(":")[0]][0] += 1
                    foreign[item.split(":")[0]][1] = max(foreign[item.split(":")[0]][1], float(item.split(":")[1]))
            report["foreign_compute_pids"] = {k: {"samples": v[0], "max_mib": v[1]} for k, v in foreign.items()}
        (out / "results.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
        try:
            box.put_file(out / "samples.csv", "samples.csv")
            box.put_file(out / "results.json", "results.json")
            for name in ("c_run.log", "torch_run.log", "prompts.txt"):
                if (out / name).exists():
                    box.put_file(out / name, name)
            box.report(outbox.jsonable(report))
        except Exception:  # noqa: BLE001
            print("outbox report failed:\n" + traceback.format_exc(), flush=True)
        slim = {k: report.get(k) for k in ("status", "arms", "agreement", "gpu_memory_segments", "foreign_compute_pids", "worker_seconds", "error")}
        print("OVERHEAD_REPORT " + json.dumps(slim, default=str), flush=True)


if __name__ == "__main__":
    main()
