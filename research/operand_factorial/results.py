"""Independent hash-checked saved-array result projection and scientific plots."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
JOB = "job-eb06becf984f"
RUN = HERE / "runs" / JOB
ARMS = ["y00", "y10", "y01", "y11", "zero"]
LABELS = ["Approximate", "Restore gate", "Restore up", "Restore both", "Zero"]
COLORS = ["#2864b7", "#bb733a", "#26877b", "#8263a5", "#74808a"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    report = json.loads((RUN / "report.json").read_text())
    report_sha = sha(RUN / "report.json")
    assert json.loads((RUN / "verification.json").read_text())["report_sha256"] == report_sha
    raw = RUN / "raw"
    def arr(name):
        p = raw / "arrays" / name
        assert sha(p) == report["raw_files"]["arrays/"+name]["sha256"]
        a = np.fromfile(p, dtype="<f4")
        assert np.isfinite(a).all()
        return a.astype(np.float64)
    def port(state, name):
        return arr(state["stem"]+"-"+name+".f32")
    def norm(a):
        return float(np.linalg.norm(a))
    cuts, endpoints, memory, identity = [], [], [], []
    power = {a: [0.0,0.0] for a in ARMS}
    for cut in report["driver"]["cuts"]:
        branches = {b["current_mode"]: b for b in cut["branches"]}
        h = cut["references"]["H"]
        writes = {a: port(b["current"], "mlp_return") for a,b in branches.items()}
        native = writes["y11"]
        write_err = {a: norm(w-native)/norm(native) for a,w in writes.items()}
        inter = writes["y11"]-writes["y10"]-writes["y01"]+writes["y00"]
        projected = arr(cut["interaction"]["write_file"])
        four_gpu = arr(cut["interaction"]["four_write_file"])
        identity.append({"cut":cut["index"],"interaction_over_native_write":norm(inter)/norm(native),
                         "projected_remainder_over_interaction":norm(projected-inter)/norm(inter),
                         "four_gpu_remainder_over_native":norm(four_gpu-inter)/norm(native)})
        row = {k:cut[k] for k in ("index","case","clock","position","retrospective")}
        row["write_errors"] = write_err
        row["Y11_H_layer_exit_byte_exact"] = np.array_equal(port(branches["y11"]["current"],"output"),port(h["current"],"output"))
        cuts.append(row)
        if not cut["retrospective"]:
            for a,w in writes.items():
                power[a][0] += norm(w-native)**2
                power[a][1] += norm(native)**2
        for horizon in range(1+len(cut["continuation_ids"])):
            states = {a:b["current"] if horizon==0 else b["future"][horizon-1] for a,b in branches.items()}
            hs = h["current"] if horizon==0 else h["future"][horizon-1]
            logits = {a:arr(s["readout"]["file"]) for a,s in states.items()}
            H = arr(hs["readout"]["file"])
            C,L = logits["y00"],logits["y11"]
            current,history,total = C-L,L-H,C-H
            errors = {a:norm(v-H)/norm(H) for a,v in logits.items()}
            e = {**row,"horizon":horizon,"logit_errors":errors,
                 "H_logit_norm":norm(H),
                 "current_effect_over_H":norm(current)/norm(H),"remaining_state_effect_over_H":norm(history)/norm(H),
                 "signed_cross_over_H_squared":float(2*np.dot(current,history)/np.dot(H,H)),
                 "decomposition_remainder":norm(total-current-history),
                 "decisions":{a:{"top":int(s["readout"]["top"])} for a,s in states.items()},
                 "native_same_id_top":int(hs["readout"]["top"])}
            if cut["retrospective"] and horizon==0:
                e["truth_digit_margin"] = {a:float(v[21]-v[20]) for a,v in {**logits,"H":H}.items()}
            endpoints.append(e)
            kv = {a:arr(s["current_kv_row_file"]).reshape(2,28,2,128) for a,s in states.items()}
            kh = arr(hs["current_kv_row_file"]).reshape(2,28,2,128)
            by_layer = {}
            for label,left,right in (("current",kv["y00"],kv["y11"]),("state",kv["y11"],kh)):
                ds = [norm(left[:,l]-right[:,l]) for l in range(28)]
                by_layer[label] = {"first_changed_layer":next((l for l,d in enumerate(ds) if d!=0),None),
                                   "layer_norms":ds,"new_row_17_exact":np.array_equal(left[:,17],right[:,17])}
            memory.append({"cut":cut["index"],"case":cut["case"],"clock":cut["clock"],"horizon":horizon,
                           "C_L_full_cache_equal":states["y00"]["full_kv_sha256"]==states["y11"]["full_kv_sha256"],
                           "L_H_full_cache_equal":states["y11"]["full_kv_sha256"]==hs["full_kv_sha256"],**by_layer})
    fixed = [r for r in endpoints if not r["retrospective"] and r["horizon"]==0]
    summary = {"schema":"tcm-operand-factorial-independent-results-v1","job":JOB,"report_sha256":report_sha,
               "fixed_cuts":len(fixed),"retrospective_cuts":1,"endpoints":len(endpoints),
               "fixed_pooled_write_error":{a:(x/y)**.5 for a,(x,y) in power.items()},
               "fixed_pooled_logit_error":{a:(sum(r["logit_errors"][a]**2*r["H_logit_norm"]**2 for r in fixed)/sum(r["H_logit_norm"]**2 for r in fixed))**.5 for a in ARMS},
               "fixed_write_improves_vs_approx":{a:sum(r["write_errors"][a]<r["write_errors"]["y00"] for r in fixed) for a in ARMS},
               "fixed_logits_improve_vs_approx":{a:sum(r["logit_errors"][a]<r["logit_errors"]["y00"] for r in fixed) for a in ARMS},
               "fixed_write_improves_logits_worsen":{a:sum(r["write_errors"][a]<r["write_errors"]["y00"] and r["logit_errors"][a]>r["logit_errors"]["y00"] for r in fixed) for a in ARMS},
               "current_rows":cuts,"endpoint_rows":endpoints,"memory_rows":memory,"identity_rows":identity,
               "models_evaluated":0,"terminal_status":"not-requested"}
    (RUN/"independent-results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:summary[k] for k in ("job","report_sha256","fixed_pooled_write_error","fixed_write_improves_vs_approx","fixed_logits_improve_vs_approx","fixed_write_improves_logits_worsen")},indent=2))
    figures = HERE/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={"width_ratios":[1,1.65]})
    for i,a in enumerate(ARMS):
        ys=[r["write_errors"][a] for r in fixed]
        axes[0].scatter(i+np.linspace(-.12,.12,len(ys)),ys,s=30,color=COLORS[i],alpha=.8)
        axes[0].plot([i-.22,i+.22],[np.median(ys)]*2,color=COLORS[i],lw=3)
    axes[0].axhline(1,ls="--",color="#74808a",lw=1)
    axes[0].set(title="Current native write reconstruction",ylabel="Relative L2 error",xticks=range(5),
                xticklabels=["Approx.","Gate","Up","Both","Zero"])
    xs=np.arange(12)
    for i,a in enumerate(ARMS):
        axes[1].plot(xs,[100*r["logit_errors"][a] for r in fixed],marker="o",ms=4,lw=1.5,
                     color=COLORS[i],label=LABELS[i],ls="--" if a=="zero" else "-")
    axes[1].set(title="Immediate readout against native same-ID history",ylabel="Full-logit relative L2 error (%)",
                xticks=xs,xticklabels=[str(i+1) for i in range(4)]*3)
    axes[1].axvline(3.5,color="#c7cdd3",lw=1);axes[1].axvline(7.5,color="#c7cdd3",lw=1)
    axes[1].legend(fontsize=8,ncol=2,loc="upper left",frameon=False)
    for x,label in ((1.5,"Recall"),(5.5,"Arithmetic"),(9.5,"Prose")):
        axes[1].text(x,-.16,label,transform=axes[1].get_xaxis_transform(),ha="center")
    fig.suptitle("Restoring the write and restoring the readout are different interventions",x=.05,ha="left",weight="bold",fontsize=14)
    fig.text(.05,.025,"Twelve fixed cuts; dots are correlated observations. Clock 1: first, 2: after answer 0, 3: after followup, 4: final. Native operands are paid controls.",fontsize=8.6)
    fig.tight_layout(rect=(0,.10,1,.92));f1=figures/(JOB+"-restoration.png");fig.savefig(f1,dpi=160);plt.close(fig)
    selected=[(0,"first"),(1,"wrong_digit"),(2,"first")]
    rows=[];labels=[]
    for case,clock in selected:
        for kind in ("current","state"):
            rr=sorted([r for r in memory if r["case"]==case and r["clock"]==clock],key=lambda r:r["horizon"])
            rows.append([r[kind]["first_changed_layer"] for r in rr]);labels.append(f"{['Recall','Arithmetic digit','Prose'][case]}  {'C versus L' if kind=='current' else 'L versus H'}")
    fig,ax=plt.subplots(figsize=(10,4.4));matrix=np.asarray(rows,dtype=float)
    ax.imshow(matrix,vmin=16,vmax=19,cmap="Blues",aspect="auto")
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(int(v)),ha="center",va="center",fontsize=13)
    ax.set(xticks=range(3),xticklabels=["Intervention token","Next supplied token","Second supplied token"],yticks=range(6),yticklabels=labels)
    ax.set_title("First changed layer in newly written K/V",loc="left",weight="bold",pad=14)
    fig.text(.04,.03,"Shown: three representative cuts. Matching a new layer-17 row does not restore the different historical cache. All futures use native transitions on fixed IDs.",fontsize=8.7)
    fig.tight_layout(rect=(0,.10,1,.98));f2=figures/(JOB+"-memory-clock.png");fig.savefig(f2,dpi=160);plt.close(fig)
    blog_images=HERE.parents[2]/"obsidian/blog/images";blog_images.mkdir(exist_ok=True)
    for p in (f1,f2):shutil.copy2(p,blog_images/p.name)
    (figures/"manifest.json").write_text(json.dumps({"report_sha256":report_sha,"results_source_sha256":sha(__file__),
       "summary_sha256":sha(RUN/"independent-results.json"),"figures":[{"file":p.name,"sha256":sha(p)} for p in (f1,f2)]},indent=2)+"\n")


if __name__=="__main__":main()
