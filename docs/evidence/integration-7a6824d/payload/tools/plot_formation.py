"""Plot measured native errors from the sealed formation panel."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root):
    report=json.loads((root/"report.json").read_text())
    rows={row["mode"]:row for row in report["comparisons"] if row["group"]=="Niko-D"}
    modes=("project","inherited","new","carry");labels=("Frozen B3","Inherited","New write","Both: carry")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
    figure,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(port,title) in zip(axes.flat,(("mlp_return","Layer-16 MLP write"),("first","Full first logits"),("current-kv","Current upper-layer K/V"),("next-kv","Common-next upper-layer K/V"))):
        values=[100*rows[m]["ports"][port]["relative_l2"] for m in modes]
        bars=ax.bar(labels,values,color=("#747b87","#bd6a42","#cf913f","#247d82"),width=.62)
        ax.set_title(title,loc="left",weight="bold");ax.set_ylabel("Relative L2 error vs native (%)")
        ax.set_ylim(0,max(values)*1.23);ax.spines[["top","right"]].set_visible(False)
        for bar,value in zip(bars,values):ax.annotate(f"{value:.2f}%",(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),textcoords="offset points",ha="center",size=11)
        ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    figure.suptitle("A coupled carry: isolated corrections worsen fidelity",weight="bold",size=18,y=.98)
    figure.text(.5,.035,"Both restores the carried omission; the current read remains omitted. All five arms answer blue.\nOne fresh Qwen question, frozen B3, C FP32. Separate port denominators; no chat-accuracy or speed claim.",ha="center",size=10,color="#444444")
    figure.tight_layout(rect=(0,.1,1,.94));path=root/"formation-consumers.png";figure.savefig(path,dpi=150);plt.close(figure)
    source={"schema":"tcm-formation-figure-v1","report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"png_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/"formation-figure.json").write_text(json.dumps(source,indent=2)+"\n")
    (root/"analysis-source"/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);print(plot(parser.parse_args().root))
