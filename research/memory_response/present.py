"""Saved-only tables and publication figures. No model or fitting calls."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
POLICIES = ("native", "byte_full", "byte_top4", "byte_recent4", "history_deleted")
LABELS = ("Native", "Byte full", "Byte top four", "Byte recent four", "History deleted")


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()), "median": float(np.median(values))}


def main(root):
    root = Path(root).resolve()
    report = json.loads((root / "report.json").read_text())
    analysis = report["analysis"]
    paths = report["driver"]["cases"]
    data = {"schema": "memory-response-saved-presentation-v1", "models_evaluated": 0,
            "report_sha256": hashlib.sha256((root / "report.json").read_bytes()).hexdigest(), "policies": {}}
    per_token_macs = 28 * (2*1536*1536 + 2*1536*256 + 3*1536*8960) + 1536*151936
    data["learned_linear_macs_per_complete_token"] = per_token_macs
    data["learned_linear_calls_per_complete_token"] = 197
    for policy in POLICIES:
        outcomes = [x for x in analysis["cases"] if x["policy"] == policy]
        local = [x for x in analysis["attention_response_rows"] if x["policy"] == policy]
        path_rows = [x for x in paths if x["policy"] == policy]
        first = [x for x in analysis["readout_comparisons"] if x["policy"] == policy and str(x["clock"]).endswith("_input")]
        matched = [x for x in first if x["comparison_domain"] == "same_id"]
        for path in path_rows:
            work = path["work"]
            extra_calls = 126 if policy == "native" else 129
            extra_macs = 18*(2*1536*1536 + 2*1536*256 + 3*1536*8960) + (0 if policy == "native" else 1536*1536 + 2*1536*256)
            aborted = int(path["mechanics"]["tested"])
            assert work["learned_linear_calls"] == work["physical_token_commits"]*197 + aborted*extra_calls
            assert work["learned_linear_macs"] == work["physical_token_commits"]*per_token_macs + aborted*extra_macs
        packet = {
            "correct": sum(x["truth"]["matched"] is True for x in outcomes), "questions": len(outcomes),
            "answers": [{"case": x["input_case_id"], "question": x["query_id"], "text": x["truth"]["answer_text"], "expected": x["truth"]["expected_color"], "matched": x["truth"]["matched"], "format": x["truth"]["format_status"]} for x in outcomes],
            "conditional_response_relative_l2": {key: stats([x["metrics"]["conditional_effects"][key]["output"]["relative_l2"] for x in local]) for key in ("codec_byte_full_minus_native", "selection_top4_minus_byte_full", "selection_recent4_minus_byte_full")},
            "actual_vs_fp64_replay_relative_l2": stats([x["metrics"]["actual_vs_conditional"]["context"]["relative_l2"] for x in local]),
            "same_id_first_readout_relative_l2": stats([x["metric"]["relative_l2"] for x in matched]) if matched else None,
            "first_readout_comparison_domains": {domain: sum(x["comparison_domain"] == domain for x in first) for domain in sorted({x["comparison_domain"] for x in first})},
            "state_bytes": path_rows[0]["state_bytes"],
            "work_including_controls_and_mechanics": {key: sum(x["work"][key] for x in path_rows) for key in ("physical_token_attempts", "physical_token_commits", "learned_linear_calls", "learned_linear_macs", "score_macs", "value_macs_nominal", "decoded_scalars", "value_probability_scans")},
            "parent_positions": [x["parent_position"] for x in path_rows],
        }
        data["policies"][policy] = packet
    data["all_observed_learned_work_matches_complete_tokens_plus_partial_attempts"] = True
    data["tracked_gpu_peak_bytes"] = report["driver"]["tracked_gpu_peak_bytes"]
    data["gpu_weight_bytes"] = report["driver"]["gpu_weight_bytes"]
    native_bytes = data["policies"]["native"]["state_bytes"]["tracked_gpu_total"]
    for policy in POLICIES[:4]:
        state = data["policies"][policy]["state_bytes"]["tracked_gpu_total"]
        data["policies"][policy]["state_savings_fraction_vs_native"] = 1 - state/native_bytes
    data["first_readout_color_contrasts"] = [x for x in analysis["four_corner_contrasts"] if str(x["clock"]).endswith("_input")]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    for query, ax in enumerate(axes):
        matrix = np.zeros((5, 4)); texts = {}
        for row in analysis["readout_declarations"]:
            if row["question_index"] != query or not str(row["clock"]).endswith("_input"): continue
            policy = row["policy"]; case = int(str(row["task"]).replace("case", ""))
            matrix[POLICIES.index(policy), case] = row["color_margin"]["primary"]["red_minus_blue"]
        limit = float(np.max(np.abs(matrix)))
        picture = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), aspect="auto")
        for i, policy in enumerate(POLICIES):
            for j in range(4):
                outcome = next(x for x in data["policies"][policy]["answers"] if x["case"] == j and x["question"] == query)
                text = outcome["text"].strip() if outcome["format"] == "one_color" else "format miss"
                ax.text(j, i, f"{text}\n{matrix[i,j]:+.2f}", ha="center", va="center", color="white" if abs(matrix[i,j]) > limit*.55 else "black", fontsize=9)
        ax.set_xticks(range(4), ["red / retain", "red / exchange", "blue / retain", "blue / exchange"], rotation=20, ha="right")
        ax.set_yticks(range(5), LABELS)
        ax.set_title(f"{'Mira' if query == 0 else 'Niko'}: emitted answer and red − blue logit")
        ax.set_xlabel("Mira's initial badge / update instruction")
        fig.colorbar(picture, ax=ax, label="Bare-color logit margin")
    fig.suptitle("Answers can preserve a record while missing its update", fontsize=16)
    fig.savefig(figures / "memory-response-margins.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    keys = ("codec_byte_full_minus_native", "selection_top4_minus_byte_full", "selection_recent4_minus_byte_full")
    for offset, policy, color, marker in ((-.08,"native","#254f85","o"),(.08,"byte_full","#d77432","s")):
        local = [x for x in analysis["attention_response_rows"] if x["policy"] == policy]
        for i,key in enumerate(keys):
            values = np.array([x["metrics"]["conditional_effects"][key]["output"]["relative_l2"]*100 for x in local])
            axes[0].scatter(i+offset+np.linspace(-.025,.025,len(values)),values,s=18,alpha=.65,color=color,marker=marker,label=("Native queries" if policy=="native" else "Byte-full own queries") if i==0 else None)
            axes[0].plot([i+offset-.05,i+offset+.05],[values.mean()]*2,color=color,lw=3)
    axes[0].set_yscale("log");axes[0].set_ylabel("Conditional context error (%)")
    axes[0].set_xticks(range(3),["Byte codec\nvs original full", "Top four\nvs byte full", "Recent four\nvs byte full"])
    axes[0].set_title("Paired precision loss is smaller than four-row deletion")
    axes[0].legend(frameon=False,fontsize=9);axes[0].grid(axis="y",alpha=.2)
    state = [data["policies"][p]["state_bytes"]["tracked_gpu_total"]/1e6 for p in POLICIES[:4]]
    bars=axes[1].bar(range(4),state,color=["#6c7784","#254f85","#6289b8","#d77432"])
    axes[1].set_xticks(range(4),["Native", "Byte full", "Byte top four", "Byte recent four"],rotation=15,ha="right")
    axes[1].set_ylabel("Tracked GPU state allocation (MB)");axes[1].set_ylim(0,40)
    axes[1].bar_label(bars,labels=[f"{x:.3f}" for x in state],padding=4)
    axes[1].set_title("Capacity 512; one primary session")
    axes[1].text(.02,.97,"Ring backup: 59.1 KB host\nNative learned weights remain paid: 6.175 GB",transform=axes[1].transAxes,va="top",fontsize=9)
    fig.suptitle("Local response fidelity and retained state are separate budgets",fontsize=15)
    fig.savefig(figures / "memory-response-state-and-response.png",dpi=180);plt.close(fig)

    destination = HERE.parents[2] / "obsidian/blog/images"
    destination.mkdir(exist_ok=True)
    data["figures"] = []
    for figure in sorted(figures.glob("*.png")):
        copied=destination/(root.name+"-"+figure.name);shutil.copy2(figure,copied)
        data["figures"].append({"source":str(figure.relative_to(root)),"obsidian":str(copied),"sha256":hashlib.sha256(figure.read_bytes()).hexdigest()})
    (root/"presentation.json").write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    shutil.copy2(__file__,root/"presentation-source.py")
    print(json.dumps({"report_sha256":data["report_sha256"],"outcomes":{p:f"{data['policies'][p]['correct']}/8" for p in POLICIES},"learned_macs_per_token":per_token_macs,"figures":data["figures"]},indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
