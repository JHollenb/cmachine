"""Part 4 sanity arithmetic: ||n|| of the dumped lean states (n = N / gamma), the Cauchy-Schwarz
half-width rt*row_tail implied by the reported tails, vs the reported logit gaps.
usage: python part4_arithmetic.py STATES_F32 G3_STATIC_F32 RESULTS_JSON"""
import json, sys
import numpy as np
N = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 1536).astype(np.float64)
g = np.fromfile(sys.argv[2], dtype=np.float32).astype(np.float64)
res = json.load(open(sys.argv[3]))
p4 = res["part4_screened_head"]
print("states", N.shape, "gamma zeros", int((g == 0).sum()), "gamma min/max", g.min(), g.max())
n = N / g
nn = np.linalg.norm(n, axis=1)
print(f"||n|| median {np.median(nn):.4f} min {nn.min():.4f} max {nn.max():.4f}  sqrt(d)={np.sqrt(1536):.4f}")
rn = p4["row_norm"]
T_est = p4["vocab"] * (rn["mean"] ** 2)  # lower bound on ||M||_F^2 (Jensen)
s1 = p4["singular_spectrum"]["sigma_max"] ** 2
print(f"||M||_F^2 >= {T_est:.4g}; sigma_max^2 = {s1:.4g} -> top direction <= {s1/T_est:.3f} of energy")
for k, v in sorted(p4["per_k"].items(), key=lambda kv: int(kv[0])):
    st = v["state_tail_over_norm"]["median"] * np.median(nn)
    hw_med = st * v["row_tail"]["median"]
    hw_min = v["state_tail_over_norm"]["min"] * nn.min() * v["row_tail"]["min"]
    # head-logit magnitude bound: ||A_v[:k]|| * ||proj[:k]||
    head_row = np.sqrt(max(rn["max"] ** 2 - v["row_tail"]["min"] ** 2, 0))
    head_state = np.sqrt(max(nn.max() ** 2 - (v["state_tail_over_norm"]["min"] * nn.min()) ** 2, 0))
    print(f"k={k:>3}: half-width t_v*||n_tail|| median ~{hw_med:.1f}, smallest possible ~{hw_min:.2f}; "
          f"|head logit| <= {head_row*head_state:.1f}; candidates median {v['top1_candidates']['median']:.0f}")
m = np.array(next(json.loads(l) for l in open(sys.argv[4]) if '"type":"states"' in l)["margin"]) if len(sys.argv) > 4 else None
if m is not None:
    print(f"top1-top2 gaps of these states: median {np.median(m):.3f} max {m.max():.3f}")
