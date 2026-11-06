"""Export the measured component-correction panel as a standalone figure."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    report=json.loads((root/"report.json").read_text());by={r["mode"]:r for r in report["comparisons"] if r["group"]=="Niko-C"};detail=report["summary"]["component_diagnostics"]
    modes=("project","carry","read","correct");labels=("Frozen B3","+ carry","+ read\ncomplement","+ full\nresidual");colors=("#64748b","#0f766e","#b45309","#3b82f6")
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for ax,(title,port,metric,scale,ylabel) in zip(axes.flat,(("MLP-write error","mlp_return","relative_l2",100,"% of native write"),("Largest first-logit error","first","max_abs",1,"logits"),("Current upper-KV error","current-kv","relative_l2",100,"% of native upper KV"),("Common-next upper-KV error","next-kv","relative_l2",100,"% of native upper KV"))):
        values=[by[m]["ports"][port][metric]*scale for m in modes];ax.bar(np.arange(4),values,color=colors,width=.65)
        for x,y in enumerate(values):ax.annotate(f"{y:.2g}" if y<.01 else f"{y:.2f}",(x,y),ha="center",xytext=(0,5),textcoords="offset points",fontsize=10)
        ax.set_xticks(np.arange(4),labels,fontsize=10);ax.set_ylim(0,max(values)*1.25);ax.set_title(title,loc="left",weight="bold",fontsize=12);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True);ax.spines[["right","top"]].set_visible(False)
    interaction=detail["interaction_l2_over_native_write_norm"]*100;rounding=detail["split_joint_vs_full_corrected_write"]["relative_l2"]
    fig.suptitle("Correcting omitted carry and read separately\nOne frozen program · one predeclared fresh question",fontsize=16,weight="bold")
    fig.supxlabel(f"Qwen2.5-1.5B · C FP32 · layer 16 · actual residual and RMS scale retained\nNative write interaction: {interaction:.2f}% of write norm; split/full write mismatch: {rounding:.2g}",fontsize=10)
    target=root/"carry-read-corrections.png";fig.savefig(target,dpi=180,facecolor="white");plt.close(fig)
    (root/"carry-read-figure.json").write_text(json.dumps({"report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),"figure_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"scope":"component corrections through native continuation; full correction is a dense diagnostic, not a speed improvement"},indent=2)+"\n");return target


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);print(plot(p.parse_args().root))
