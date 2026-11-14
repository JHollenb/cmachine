"""Independent re-check of one G3 observer record (own parser and own float64 math, not worker.g3_gate).
Adds checks the worker does not make: output == raw_receiver + mlp_write (bitwise), entropy/pmax
from a float64 softmax of recomputed scores (not the native probabilities), top-2 consistency
of logit_a/logit_b with the ledger margin, and the margin split vs exact logit difference.
usage: python g3_recheck.py RUNS_JSONL RECORD_F32 RECORD_INDEX G3_STATIC_F32"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
h = next(r for r in rows if r["type"] == "header")
L, H, KV, hd, d, I, eps = int(h["layers"]), int(h["qheads"]), int(h["kvheads"]), int(h["head"]), int(h["hidden"]), int(h["intermediate"]), float(h["epsilon"])
rec = [r for r in rows if r["type"] == "g3"][int(sys.argv[3])]
v = np.fromfile(sys.argv[2], dtype=np.float32)
assert v.size == rec["floats"], (v.size, rec["floats"])
gamma = np.fromfile(sys.argv[4], dtype=np.float32).astype(np.float64)
P = rec["position"] + 1; BLK = 16 + 5 * H; RF = 16 + L * BLK
sizes = [d, d, d, d, I, I, I, d, d, H * P]
o = 0; ports = []
for l in range(L):
    pl = []
    for s in sizes: pl.append(v[o:o + s]); o += s
    ports.append(pl)
K = v[o:o + L * KV * P * hd].reshape(L, KV, P, hd).astype(np.float64); o += L * KV * P * hd
Wa, Wb = v[o:o + d].astype(np.float64), v[o + d:o + 2 * d].astype(np.float64); o += 2 * d
la, lb = float(v[o]), float(v[o + 1]); o += 2
row = v[o:o + RF].astype(np.float64); o += RF
assert o == v.size
print("record", sys.argv[3], "prompt", rec["prompt"], "position", rec["position"], "top1/top2", rec["top1"], rec["top2"])
bits = lambda a, b: int(np.sum(a.view(np.uint32) != b.view(np.uint32)))
print("bits(input+attn_write != raw)", sum(bits(p[0] + p[2], p[3]) for p in ports))
print("bits(raw+mlp_write != output)  [not checked by worker]", sum(bits(p[3] + p[7], p[8]) for p in ports))
print("bits(output[l] != input[l+1])", sum(bits(ports[l][8], ports[l + 1][0]) for l in range(L - 1)))
def lf(l, f): return row[16 + l * BLK + f]
def hf(l, f): return row[16 + l * BLK + 16 + f * H: 16 + l * BLK + 16 + (f + 1) * H]
worst = {}
def upd(name, a, b):
    r = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-30)); worst[name] = max(worst.get(name, 0), float(r))
for l in range(L):
    inp, q, aw, raw, g, u, prod, mw, out, pr = [x.astype(np.float64) for x in ports[l]]
    upd("rho_in", 1 / np.sqrt(np.mean(inp**2) + eps), lf(l, 0)); upd("rho_mid", 1 / np.sqrt(np.mean(raw**2) + eps), lf(l, 1))
    upd("attn_dot", inp @ aw, lf(l, 2)); upd("attn_norm2", aw @ aw, lf(l, 3)); upd("mlp_dot", raw @ mw, lf(l, 4)); upd("mlp_norm2", mw @ mw, lf(l, 5))
    sil = g / (1 + np.exp(-g)) * u  # float64 SiLU(g)u, vs native product
    upd("prod2_vs_float64_silu", sil @ sil, lf(l, 14))
    s = np.stack([K[l, j // (H // KV)] @ q.reshape(H, hd)[j] for j in range(H)]) / np.sqrt(hd)
    m = s.max(1, keepdims=True); e = np.exp(s - m); p64 = e / e.sum(1, keepdims=True)
    upd("logz", (m[:, 0] + np.log(e.sum(1))), hf(l, 0))
    upd("entropy_vs_p64", -(p64 * np.log(np.where(p64 > 0, p64, 1))).sum(1), hf(l, 2))
    upd("pmax_vs_p64", p64.max(1), hf(l, 3))
    worst["argmax_vs_p64_mismatch"] = worst.get("argmax_vs_p64_mismatch", 0) + int(np.sum(p64.argmax(1) != hf(l, 4)))
xF = ports[L - 1][8].astype(np.float64); rhoF = 1 / np.sqrt(np.mean(xF**2) + eps); c = gamma * (Wa - Wb)
split = rhoF * (c @ ports[0][0].astype(np.float64)) + sum(rhoF * (c @ ports[l][2].astype(np.float64)) + rhoF * (c @ ports[l][7].astype(np.float64)) for l in range(L))
print({k: (f"{x:.2e}" if isinstance(x, float) else x) for k, x in worst.items()})
print(f"margin: ledger {row[6]:.7g}  native la-lb {la-lb:.7g}  float64 rho*c.xF {rhoF*(c@xF):.7g}  float64 split {split:.7g}  ledger margin_sum {row[10]:.7g}")
print("ledger top1/top2 fields", row[4], row[5], "== record", rec["top1"], rec["top2"], "(same source S.top: not an independent check)")
