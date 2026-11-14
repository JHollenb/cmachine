"""Recompute timing medians / overheads from raw runs.jsonl (copied read-only from Beast job dir).
usage: python timing_recompute.py RUNS_JSONL RESULTS_JSON [SPEEDUPS_RESULTS_JSON]"""
import json, sys, statistics as st
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
res = json.load(open(sys.argv[2]))
runs = [r for r in rows if r["type"] == "run"]
arms = {}
order = []
for r in runs:
    k = (r["arm"], r["ledger"])
    a = arms.setdefault(k, {"ms": [], "pf": [], "n": 0})
    a["ms"] += r["ms"]; a["pf"] += r["prefill_ms"]; a["n"] += 1
    order.append(k)
med = {k: float(np.median(v["ms"])) for k, v in arms.items()}
pmed = {k: float(np.median(v["pf"])) for k, v in arms.items()}
print("arm/ledger  n_steps  median  p10  p90  | results.json median  match")
for k, v in sorted(arms.items()):
    m = np.array(v["ms"]); key = f"{k[0]}/{k[1]}"
    rj = res["timing"][key]["ms_median"]
    print(f"{key:18s} {m.size:5d} {med[k]:.5f} {np.percentile(m,10):.3f} {np.percentile(m,90):.3f} | {rj:.5f} {'OK' if abs(rj-med[k])<1e-9 else 'DIFF'}  prefill {pmed[k]:.4f} vs {res['timing'][key]['prefill_ms_median']:.4f}")
print("\noverheads (on/off, onnr/off), kernel, readback:")
for arm in sorted({k[0] for k in arms}):
    off = med.get((arm, "off"))
    if off is None: continue
    on, nr = med.get((arm, "on")), med.get((arm, "onnr"))
    s = f"{arm:8s} off {off:.4f} on {on:.4f} d {on-off:+.4f} ({100*(on/off-1):+.2f}%)"
    if nr: s += f" onnr {nr:.4f} kernel {nr-off:.4f} readback {on-nr:.4f}"
    s += f" | prefill off {pmed[(arm,'off')]:.3f} on {pmed[(arm,'on')]:.3f} ({100*(pmed[(arm,'on')]/pmed[(arm,'off')]-1):+.2f}%)"
    print(s)
# Paired per-region overhead (robustness): for B=1 arms, each region = one prompt; pair on vs off by prompt
print("\npaired per-prompt median deltas (B=1 arms):")
for arm in sorted({k[0] for k in arms}):
    regs = {}
    for r in runs:
        if r["arm"] == arm: regs.setdefault(r["ledger"], {})[tuple(r["prompts"])] = float(np.median(r["ms"]))
    if "off" in regs and "on" in regs:
        ks = sorted(set(regs["off"]) & set(regs["on"]))
        d = [regs["on"][k] - regs["off"][k] for k in ks]
        print(f"{arm:8s} n={len(ks)} median paired delta {np.median(d):.4f} min {min(d):.4f} max {max(d):.4f}")
# Arm order
print("\nrun order (arm,ledger) first 60:")
print(" ".join(f"{a}:{l}" for a, l in order[:60]))
print("... order of ledger modes per arm:")
for arm in sorted({k[0] for k in arms}):
    seq = [(r["ledger"], tuple(r["prompts"])[:1]) for r in runs if r["arm"] == arm]
    print(arm, " ".join(f"{l}{p[0]}" for l, p in seq[:24]))
# baseline
bc = res["baseline_check"]["compare"]
print("\nbaseline ratios (recomputed from raw medians vs hardcoded refs):")
ref = {"C0": 10.544, "F1": 10.177, "G1": 9.973, "G8": 11.263, "G64": 13.036, "bf16_G1": 5.364}
for a, rv in ref.items():
    print(a, f"{med[(a,'off')]/rv:.4f}", f"results {bc[a+'_decode']['ratio']:.4f}")
print("C0 prefill", f"{pmed[('C0','off')]/210.5:.4f}", "F1 prefill", f"{pmed[('F1','off')]/11.56:.4f}")
if len(sys.argv) > 3:
    sp = json.load(open(sys.argv[3]))
    print("\nspeedups job refs (looked up):")
    def find(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items(): yield from find(v, path + "/" + str(k))
        else:
            yield path, o
    for p, v in find(sp.get("timing", sp)):
        if p.endswith("ms_median") or p.endswith("prefill_ms_median"):
            print(p, v)
