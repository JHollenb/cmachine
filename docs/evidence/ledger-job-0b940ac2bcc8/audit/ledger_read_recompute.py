"""Independent re-parse of ledger-C0-on.f32 (raw rows from the Beast job dir) to recheck the
'What the ledger shows' numbers. Own layout code from src/qwen_ledger.h (not worker.Layout).
usage: python ledger_read_recompute.py RUNS_JSONL LEDGER_C0_ON_F32 RESULTS_JSON"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hdr = next(r for r in rows if r["type"] == "header")
L, H, d, eps = int(hdr["layers"]), int(hdr["qheads"]), int(hdr["hidden"]), float(hdr["epsilon"])
QH, QLY, QPH = 16, 16, 5
BLK = QLY + QPH * H
RF = QH + L * BLK
ALL = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, RF).astype(np.float64)
idx = [r for r in rows if r["type"] == "ledger_rows" and r["run_type"] == "run" and r["arm"] == "C0" and r["ledger"] == "on"]
# the dump file also holds smoke/warmup rows of arm C0: select by row_start/count
R = np.concatenate([ALL[r["row_start"]: r["row_start"] + r["count"]] for r in idx])
step = np.concatenate([np.array(r["step"]) for r in idx]); prompt = np.concatenate([np.array(r["prompt"]) for r in idx])
assert len(step) == len(R) and all(len(r["step"]) == r["count"] for r in idx)
print("file rows", len(ALL), "scored C0/on rows", len(R))
D = R[step >= 0]
print("decode rows", len(D), "prompts", len(set(prompt[step >= 0])))
def lay(f): return D[:, [QH + l * BLK + f for l in range(L)]]
def head(f): return D[:, [QH + l * BLK + QLY + f * H + j for l in range(L) for j in range(H)]].reshape(len(D), L, H)
RHO_IN, RHO_MID, ADOT, AN2, MDOT, MN2, MA, MM, GS, GOFF, GON, E2, O2, EO, P2 = range(15)
marg = D[:, 6]
print(f"margin mean {marg.mean():.4f} median {np.median(marg):.4f} p10 {np.percentile(marg,10):.4f} p90 {np.percentile(marg,90):.4f}")
ma, mm = lay(MA), lay(MM)
print(f"embed mean {D[:,9].mean():.4f}  sum attn {ma.sum(1).mean():.4f}  sum mlp {mm.sum(1).mean():.4f}  total {D[:,9].mean()+ma.sum(1).mean()+mm.sum(1).mean():.4f}")
print(f"mlp layers 0..26 sum {mm[:, :27].sum(1).mean():.4f}; mlp layer 27 mean {mm[:,27].mean():.4f} |.| {np.abs(mm[:,27]).mean():.4f}")
print(f"attn layers 19..27 sum {ma[:, 19:].sum(1).mean():.4f}")
absw = (np.abs(ma) + np.abs(mm)) / 2
print("mean |contribution| per write by layer (attn,mlp):", " ".join(f"{l}:{np.abs(ma[:,l]).mean():.2f}/{np.abs(mm[:,l]).mean():.2f}" for l in range(L)))
pf, am, ent = head(1), head(4), head(2)
src = D[:, 2:3]
print("pfirst mean by layer:", " ".join(f"{l}:{pf[:,l].mean():.3f}" for l in range(L)))
print("argmax==0 frac:", " ".join(f"{l}:{(am[:,l]==0).mean():.3f}" for l in range(L)))
eff = np.exp(ent)
print("eff rows:", " ".join(f"{l}:{eff[:,l].mean():.1f}({(eff[:,l]/src).mean():.2f})" for l in range(L)))
print("gate small:", " ".join(f"{l}:{lay(GS)[:,l].mean():.0f}" for l in range(L)))
print("gate off:", " ".join(f"{l}:{lay(GOFF)[:,l].mean():.1f}" for l in range(L)))
print("gate on:", " ".join(f"{l}:{lay(GON)[:,l].mean():.1f}" for l in range(L)))
odd_share = lay(O2) / (lay(E2) + lay(O2))
print("odd share:", " ".join(f"{l}:{odd_share[:,l].mean():.2f}" for l in range(L)))
x2i = d * (1 / lay(RHO_IN) ** 2 - eps); x2m = d * (1 / lay(RHO_MID) ** 2 - eps)
ca = lay(ADOT) / np.sqrt(x2i * lay(AN2)); cm = lay(MDOT) / np.sqrt(x2m * lay(MN2))
print("cos attn:", " ".join(f"{l}:{ca[:,l].mean():.2f}" for l in range(L)))
print("cos mlp:", " ".join(f"{l}:{cm[:,l].mean():.2f}" for l in range(L)))
print("attn ||w||/||x||:", " ".join(f"{l}:{np.sqrt(lay(AN2)[:,l]/x2i[:,l]).mean():.2f}" for l in range(L)))
print("mlp ||w||/||r||:", " ".join(f"{l}:{np.sqrt(lay(MN2)[:,l]/x2m[:,l]).mean():.2f}" for l in range(L)))
# per-prompt spread of the headline split (how stable across the 16 prompts)
pp = prompt[step >= 0]
print("\nper-prompt: mean margin / sum attn / sum mlp / mlp27")
for p in sorted(set(pp)):
    s = pp == p
    print(p, f"{marg[s].mean():.2f} {ma[s].sum(1).mean():.2f} {mm[s].sum(1).mean():.2f} {mm[s,27].mean():.2f}")
res = json.load(open(sys.argv[3]))["ledger_read"]
pl = res["per_layer"]
mx = max(abs(pl[l]["margin_mlp_mean"] - mm[:, l].mean()) for l in range(L))
print("\nmax |worker per-layer margin_mlp_mean - mine|:", mx)
