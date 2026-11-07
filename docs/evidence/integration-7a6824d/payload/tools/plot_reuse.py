"""Standalone figure for the prospective B2/B3 computation comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    report=json.loads((root/"report.json").read_text());by={(r["group"],r["mode"]):r for r in report["comparisons"]}
    groups=("Niko-A","Niko-B");specs=(("Receiver omitted norm",None,"omitted_relative_norm",100,"% of live receiver"),("MLP-write error","mlp_return","relative_l2",100,"% of native write"),("Current upper-KV error","current-kv","relative_l2",100,"% of native upper KV"),("Largest first-logit error","first","max_abs",1,"logits"))
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained");colors=("#64748b","#0f766e")
    for ax,(title,port,metric,scale,ylabel) in zip(axes.flat,specs):
        for k,mode in enumerate(("project2","project")):
            values=[(by[g,mode]["ports"][port][metric] if port else by[g,mode][metric])*scale for g in groups];x=np.arange(2)+(k-.5)*.32
            ax.bar(x,values,width=.3,color=colors[k],label=("Frozen B2","Calibrated B3")[k])
            for xx,y in zip(x,values):ax.annotate(f"{y:.1f}" if scale==100 else f"{y:.2f}",(xx,y),ha="center",xytext=(0,4),textcoords="offset points",fontsize=10)
        ax.set_xticks(np.arange(2),["Fresh query A","Fresh query B"]);ax.set_title(title,loc="left",weight="bold",fontsize=12);ax.set_ylabel(ylabel);ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True);ax.spines[["right","top"]].set_visible(False)
    axes[0,0].set_ylim(0,60);axes[0,0].legend(frameon=False,loc="upper right");fig.suptitle("One calibrated operand transfers partially\nLocal write improves; full-logit effects remain mixed",fontsize=16,weight="bold")
    fig.supxlabel("Qwen2.5-1.5B, C FP32, layer 16 · two predeclared new wordings\nResidual-corrected write error ≤ 6.4e−7 · all short answers still match dense",fontsize=10)
    path=root/"receiver-reuse.png";fig.savefig(path,dpi=180,facecolor="white");plt.close(fig)
    (root/"receiver-reuse-figure.json").write_text(json.dumps({"job_id":root.name,"report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),"figure_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"scope":"prospective two-query B2/B3 observations; no accuracy or speed certification"},indent=2)+"\n");return path


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);print(plot(p.parse_args().root))
