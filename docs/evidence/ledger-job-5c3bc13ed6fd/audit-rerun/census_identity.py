"""Header-level identity on raw ledger rows: even + odd = SiLU(g)u  =>  even2 + odd2 + 2*even_odd = prod2
(prod2 uses the native product; even/odd are side terms). Also gate counts <= intermediate,
small+off+on <= intermediate. Tests the census semantics against the header, not lean-vs-fast.
usage: python census_identity.py RUNS_JSONL LEDGER_F32 ARM"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hdr = next(r for r in rows if r["type"] == "header")
L, H, inter = int(hdr["layers"]), int(hdr["qheads"]), int(hdr["intermediate"])
BLK = 16 + 5 * H; RF = 16 + L * BLK
ALL = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, RF).astype(np.float64)
idx = [r for r in rows if r["type"] == "ledger_rows" and r["run_type"] == "run" and r["arm"] == sys.argv[3] and r["ledger"] == "on"]
R = np.concatenate([ALL[r["row_start"]: r["row_start"] + r["count"]] for r in idx])
def lay(f): return R[:, [16 + l * BLK + f for l in range(L)]]
E2, O2, EO, P2 = lay(11), lay(12), lay(13), lay(14)
rel = np.abs(E2 + O2 + 2 * EO - P2) / P2
print(sys.argv[3], "rows", len(R), f"|even2+odd2+2eo-prod2|/prod2: median {np.median(rel):.2e} p99 {np.percentile(rel,99):.2e} max {rel.max():.2e}")
cnt = lay(8) + lay(9) + lay(10)
print("max small+off+on", cnt.max(), "<= intermediate", inter, bool(cnt.max() <= inter))
# Cauchy-Schwarz on census: even_odd^2 <= even2*odd2
print("CS violations even_odd^2 > even2*odd2*(1+1e-6):", int(np.sum(EO**2 > E2 * O2 * (1 + 1e-6))))
# Cancellation diagnostic: hard-off neurons give even ~ -odd (tanh -> -1), so even2+odd2+2eo
# cancels large terms; judge the residual against the scale of the cancelling terms.
rel2 = np.abs(E2 + O2 + 2 * EO - P2) / (E2 + O2)
print(f"  same residual / (even2+odd2): median {np.median(rel2):.2e} p99 {np.percentile(rel2,99):.2e} max {rel2.max():.2e}")
w = np.unravel_index(np.argmax(rel), rel.shape)
print("  worst cell (row, layer)", w, "prod2", P2[w], "even2", E2[w], "gate_off", lay(9)[w])
# Is "odd2/(even2+odd2)" a share of the product energy? Only if even_odd ~ 0. Per layer:
ratio = (E2 + O2) / P2
corr = EO / np.sqrt(E2 * O2)
print("  layer: median (even2+odd2)/prod2 | median even_odd/sqrt(even2*odd2) | mean odd2/(even2+odd2)")
print("  " + " ".join(f"{l}:{np.median(ratio[:,l]):.3g}|{np.median(corr[:,l]):+.2f}|{(O2/(E2+O2))[:,l].mean():.2f}" for l in range(L)))
