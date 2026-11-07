"""Add a source-bound continuation decomposition without modifying the sealed run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(raw, name):
    v = np.fromfile(raw / name, dtype="<f4")
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite saved array")
    return v.astype(np.float64)

def full(path):
    return dict(zip(("initial", "first-end", "followup", "second-end"), [s for s in path["steps"] if "readout" in s]))

def analyze(root):
    raw = root / "raw"
    report = json.loads((root / "report.json").read_text())
    verified = json.loads((root / "verification.json").read_text())
    if verified["report_sha256"] != sha(root / "report.json"):
        raise ValueError("Prerequisite verification differs")
    comparisons, paths = [], []
    for case, refs in zip(report["driver"]["primary_cases"], report["driver"]["references"]):
        native = refs["native"]
        for arm, same in zip(case["arms"], refs["same_ids"]):
            checkpoints = full(arm)
            local_probes = sorted(raw.glob(f"local-replay-{arm['name']}-p*.json"))
            for label, probe in zip(("initial", "followup"), local_probes):
                row, rebuild, own_native = checkpoints[label], full(same)[label], full(native)[label]
                q = json.loads(probe.read_text())
                if q["candidate_step"] != row["step"]:
                    raise ValueError("Native local probe clock differs")
                c = load(raw, f"{arm['name']}-s{row['step']:03}-logits.f32")
                l = load(raw, probe.stem+"-s000-logits.f32")
                h = load(raw, f"{same['name']}-s{rebuild['step']:03}-logits.f32")
                n = load(raw, f"{native['name']}-s{own_native['step']:03}-logits.f32")
                terms = (c-l, l-h, h-n)
                error = c-n
                denominator = float(n@n)
                names = ("current_law_defect", "accumulated_state_effect", "history_conditioning")
                powers = {name: float(v@v) for name, v in zip(names, terms)}
                cross = {names[i]+"__"+names[j]: float(2*(terms[i]@terms[j])) for i in range(3) for j in range(i+1, 3)}
                combined = terms[0] + terms[1] + terms[2]
                local_norm, carried_norm = (float(np.linalg.norm(v)) for v in terms[:2])
                comparisons.append({"case": case["case"], "policy": arm["name"], "boundary": label, "candidate_position": row["position"], "native_position": own_native["position"], "total_l2": float(np.linalg.norm(error)), "relative_to_native_norm": float(np.linalg.norm(error))/np.sqrt(denominator), "term_l2": {name: float(np.linalg.norm(v)) for name, v in zip(names, terms)}, "term_relative_to_native_norm": {name: float(np.linalg.norm(v))/np.sqrt(denominator) for name, v in zip(names, terms)}, "local_carried_cosine": float(terms[0]@terms[1])/(local_norm*carried_norm) if local_norm*carried_norm else None, "term_squared_norms": powers, "signed_cross_terms": cross, "sum_vector_closure_l2": float(np.linalg.norm(error-combined)), "sum_energy_closure": float(error@error)-sum(powers.values())-sum(cross.values()), "normalization_native_squared_norm": denominator})
            native_answers = [a["ids"] for a in native["answers"]]
            first_divergence = []
            for candidate_answer, target_answer in zip(arm["answers"], native_answers):
                ids = candidate_answer["ids"]
                different = next((i for i, (a, b) in enumerate(zip(ids, target_answer)) if a != b), None)
                if different is None and len(ids) != len(target_answer):
                    different = min(len(ids), len(target_answer))
                first_divergence.append(different)
            captured_kv16_exact = True
            kv17_differences = []
            for j in range(len(arm["steps"])):
                c = load(raw, f"{arm['name']}-s{j:03}-kv16-27.f32").reshape(2, 12, 256)
                h = load(raw, f"{same['name']}-s{j:03}-kv16-27.f32").reshape(2, 12, 256)
                captured_kv16_exact &= c[:, 0].tobytes() == h[:, 0].tobytes()
                kv17_differences.append(float(np.linalg.norm(c[:, 1]-h[:, 1])))
            paths.append({"policy": arm["name"], "answers": report["texts"][arm["name"]], "native_answers": report["texts"][native["name"]], "answer_ids_equal_native": [a["ids"] == b for a, b in zip(arm["answers"], native_answers)], "first_divergent_answer_index": first_divergence, "captured_current_kv16_same_ids_byte_exact_all_rows": bool(captured_kv16_exact), "kv17_difference_l2_max": max(kv17_differences), "instrumented_seconds": arm["instrumented_seconds"], "consumed_tokens": len(arm["steps"]), "termination": arm["answers"]})
    write_rows = [x for x in report["analysis"]["observations"] if x["consumer"] == "same-operand-write"]
    linear = 1543569408
    zero_mac = 41287680
    sources = report["driver"]["work"]
    total_tokens = sum(sources[k] for k in ("source_native_tokens", "primary_tokens", "reference_tokens", "replay_tokens", "local_native_tokens"))
    total_matvecs = 197*(sources["source_native_tokens"]+sources["local_native_tokens"]) + sum(report["analysis"]["work"][k]["all_dense_matvec_calls"] for k in ("primary", "native_own_history", "postseal_replays")) + sources["saved_exact_swiglu_down_diagnostics"]
    result = {"schema": "tcm-gate-law-continuation-v1", "report_sha256": sha(root / "report.json"), "verification_sha256": sha(root / "verification.json"), "analysis_source_sha256": sha(Path(__file__)), "scope": "posthoc saved-array computation; zero new model calls; original report unchanged", "equation": "C-N=(C-L)+(L-H)+(H-N), L=native step on actual candidate cache; H=native rebuild on candidate consumed IDs", "comparisons": comparisons, "paths": paths,
              "local_law": {"evaluated_polynomial_invocations": len(write_rows), "closer_to_same_input_native_write_than_zero": sum(x["write"]["relative_l2"] < 1 for x in write_rows), "same_input_write_relative_l2_max": max(x["write"]["relative_l2"] for x in write_rows), "independent_contexts": 2, "duplicate_initial_once_recurring_pairs": 2},
              "cost": {"native_dense_matvecs_per_token": 197, "polynomial_dense_matvecs_per_token": 197, "native_learned_linear_macs_per_token": linear, "polynomial_learned_linear_macs_per_token": linear, "zero_learned_linear_macs_per_token": linear-zero_mac, "zero_saves_fraction": zero_mac/linear, "prior_B3_avoids_macs_per_projected_token": 27525120, "prior_B3_nominal_fraction": 27525120/linear, "selected_silu_channels": 8960, "native_silu_sites_per_token": 28, "recurring_native_silu_sites_per_token": 27, "law_gpu_bytes_per_anchor": 179200, "all_experiment_c_token_transitions": total_tokens, "all_experiment_dense_matvec_calls_including_saved_diagnostics": total_matvecs, "source_coefficient_compilations": 6},
              "truth_observations": {"expected_initial_math": "391", "expected_followup_math_from_correct_initial_result": "367", "native_initial_math": report["texts"]["native-c1"][0], "native_followup_math": report["texts"]["native-c1"][1], "native_expected_subtraction_from_own_prior_answer": "377", "zero_initial_math": report["texts"]["zero-c1"][0], "zero_followup_math": report["texts"]["zero-c1"][1], "zero_expected_subtraction_from_own_prior_answer": "385", "scope": "wrong multiplication under every policy; zero improves subtraction consistency with its own wrong prior answer; follow-up text is budget truncated; fidelity and task truth differ"}}
    (root / "continuation-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result

def plot(root, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads((root / "report.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = ["#2563eb", "#bd4b27"]
    for case in (0, 1):
        rows = [r for r in report["analysis"]["observations"] if r["consumer"] == "same-operand-write" and r["policy"] == f"recurring-c{case}"]
        axes[0].plot([r["step"] for r in rows], [100*r["write"]["relative_l2"] for r in rows], color=colors[case], label=("Recall" if not case else "Arithmetic"))
    axes[0].axhline(100, color="#7a7a7a", linestyle="--", label="Zero-write distance")
    axes[0].set(title="Law error on its own receiving state", xlabel="Consumed event after cut", ylabel="Same-input MLP write error (%)")
    axes[0].legend(frameon=False)
    x = np.arange(2)
    for offset, kind, label, color in ((-.25, "total", "Total", "#2563eb"), (0, "current_law_defect", "Current law", "#d87b36"), (.25, "accumulated_state_effect", "Carried state", "#718096")):
        values = []
        for case in (0, 1):
            row = next(a for a in result["comparisons"] if a["policy"] == f"recurring-c{case}" and a["boundary"] == "followup")
            values.append(100*(row["relative_to_native_norm"] if kind == "total" else row["term_relative_to_native_norm"][kind]))
        bars = axes[1].bar(x+offset, values, .23, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(title="Follow-up full-vocabulary discrepancy", xticks=x, xticklabels=["Recall", "Arithmetic"], ylabel="Norm / same native-logit norm (%)", ylim=(0, 13))
    axes[1].legend(frameon=False, fontsize=8)
    c = result["cost"]
    values = np.array([c["native_learned_linear_macs_per_token"], c["polynomial_learned_linear_macs_per_token"], c["zero_learned_linear_macs_per_token"]])/1e9
    bars = axes[2].bar(["Native", "Gate law", "Zero MLP16"], values, color=["#64748b", "#2563eb", "#a3a3a3"])
    axes[2].bar_label(bars, fmt="%.3f", padding=3)
    axes[2].set(title="Learned-linear work remains", ylabel="Billion MACs / consumed token", ylim=(0, 1.75))
    fig.suptitle("Frozen source gate law: better than deletion locally, accumulated state error remains", fontsize=13)
    fig.savefig(root / "gate-law-results.png", dpi=170, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    r = analyze(a.root)
    plot(a.root, r)
    print(json.dumps({"local_law": r["local_law"], "cost": r["cost"], "maximum_vector_identity_remainder": max(x["sum_vector_closure_l2"] for x in r["comparisons"])}))
