"""Plot the retained receiver-plane observations; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path);args=parser.parse_args();root=args.directory
report=json.loads((root/"report.json").read_text());frontier=json.loads((root/"continuation-frontier.json").read_text())
comparisons=report["comparisons"]
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
labels=["+.25 read","−.25 read","−1 read","+.25 carry","×1.5 receiver"]
colors=["#20806b","#3479b9","#ab5b25","#8355a3"]
for j,(port,label) in enumerate((("mlp_return","MLP write"),("output","layer output"),("current-kv","current KV"),("next-kv","common next KV"))):
    axes[0].bar(np.arange(5)+(j-1.5)*.19,[100*x["dense_vs_native"][port]["relative_l2"] for x in comparisons[1:]],width=.18,label=label,color=colors[j])
axes[0].set(title="Native edit effects",ylabel="Relative L2 change (%)",xticks=np.arange(5),xticklabels=labels)
axes[0].tick_params(axis="x",rotation=22);axes[0].legend(fontsize=8,frameon=False)
axes[0].annotate("MLP write: 0.000072%",xy=(3.7,0),xytext=(2.5,34),arrowprops={"arrowstyle":"->","color":"#444"},fontsize=8,color="#333")
for j,(port,label) in enumerate((("mlp_return","MLP write"),("current-kv","current KV"),("next-kv","common next KV"))):
    axes[1].plot(np.arange(6),[x["compiled_vs_dense"][port]["relative_l2"] for x in comparisons],"o-",label=label,color=colors[j])
axes[1].set(title="Compiled program versus dense",ylabel="Relative L2 error",yscale="log",xticks=np.arange(6),xticklabels=["zero"]+labels,ylim=(1e-7,2e-6))
axes[1].tick_params(axis="x",rotation=22);axes[1].legend(fontsize=8,frameon=False)
scale=next(r for r in frontier["rows"] if r["name"]=="dense-5")
for j,(clock,kind,label) in enumerate((("current","key","current K"),("current","value","current V"),("next","key","common next K"),("next","value","common next V"))):
    axes[2].plot(range(17,28),[100*r[kind]["relative_l2"] for r in scale["clocks"][clock]],"o",label=label,color=colors[j],linestyle="--" if clock=="next" else "-")
axes[2].set(title="×1.5 receiver: the cache frontier",xlabel="Layer (zero-based)",ylabel="Change versus native KV (%)",xticks=[17,19,21,23,25,27])
axes[2].legend(fontsize=8,frameon=False)
for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
fig.suptitle("One Qwen receiver plane: nonlinear computation, residual state and later memory",fontsize=13)
fig.supxlabel("One supplied-fact context · MLP16 on final prompt token · own-state continuations · original weights retained",fontsize=9)
path=root/"receiver-plane.png";fig.savefig(path,dpi=160);plt.close(fig)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(root/"figure-provenance.json").write_text(json.dumps({"report_sha256":sha(root/"report.json"),"frontier_sha256":sha(root/"continuation-frontier.json"),"script_sha256":sha(Path(__file__)),"figure_sha256":sha(path)},indent=2)+"\n")
print(path)
