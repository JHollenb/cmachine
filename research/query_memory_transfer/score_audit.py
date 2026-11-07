"""Post-hoc Q/K score decomposition on authenticated common initial inputs.

No model invocation, CUDA replay, fitting, or mutation of executed evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return math.sqrt(math.fsum(float(v) * float(v) for v in x))


def ratio(value, denominator):
    return value / denominator if denominator else (0.0 if value == 0 else None)


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def softmax(score):
    w = np.exp(score - np.max(score))
    return w / math.fsum(float(x) for x in w)


def decompose(reference, changed, query, key, interaction):
    reference, changed, query, key, interaction = [
        np.asarray(x, dtype=float).reshape(-1)
        for x in (reference, changed, query, key, interaction)]
    delta = changed - reference
    terms = {"query": query, "key": key, "interaction": interaction}
    rn, dn = norm(reference), norm(delta)
    result = {"reference_norm": rn, "delta_norm": dn,
              "delta_relative_to_reference": ratio(dn, rn),
              "terms": {name: {"norm": norm(x),
                               "relative_to_reference": ratio(norm(x), rn)}
                        for name, x in terms.items()}}
    result["signed_cross_products"] = {
        f"{a}_{b}": 2 * math.fsum(float(x) * float(y) for x, y in zip(terms[a], terms[b]))
        for a, b in (("query", "key"), ("query", "interaction"), ("key", "interaction"))}
    closure = delta - query - key - interaction
    result["closure_norm"] = norm(closure)
    result["closure_relative_to_reference"] = ratio(norm(closure), rn)
    result["closure_relative_to_delta"] = ratio(norm(closure), dn)
    reconstructed_energy = sum(norm(x) ** 2 for x in terms.values()) + sum(result["signed_cross_products"].values())
    result["signed_energy_closure"] = dn ** 2 - reconstructed_energy
    return result


def audit(root, expected):
    root = Path(root)
    if sha(root / "report.json") != expected:
        raise ValueError("Report identity differs")
    report = json.loads((root / "report.json").read_text())
    paths = {p["name"]: p for p in report["driver"]["paths"]}
    used = {}

    def load(name, shape):
        file = root / "raw" / name
        record = report["raw_files"][name]
        if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
            raise ValueError("Raw evidence differs: " + name)
        data = np.fromfile(file, dtype="<f4")
        if data.size != math.prod(shape) or not np.isfinite(data).all():
            raise ValueError("Raw extent/finite contract differs: " + name)
        used[name] = record
        return data.reshape(shape)

    cases = []
    for case in range(3):
        names = [f"case{case}-native", f"case{case}-values-native"]
        desc = [paths[name] for name in names]
        count = desc[0]["turns"][0]["prefill"]["position"]
        if desc[1]["turns"][0]["prefill"]["position"] != count or desc[0]["consumed_ids"][:count] != desc[1]["consumed_ids"][:count]:
            raise ValueError("Initial common history differs")
        arrays = []
        for name, d in zip(names, desc):
            a = {port: load(f"{name}-{port}.f32", (d["count"], *shape))[:count].astype(float)
                 for port, shape in (("norm", (1536,)), ("q", (12, 128)),
                                     ("k", (2, 128)), ("v", (2, 128)),
                                     ("attention", (12, 128)))}
            a["probabilities"] = load(f"{name}-probabilities.f32", (6*d["count"]*(d["count"]+1),))
            arrays.append(a)
        native, compiled = arrays
        if not np.array_equal(native["norm"], compiled["norm"]) or not np.array_equal(native["v"], compiled["v"]):
            raise ValueError("Receiver or native-value pair differs")
        pooled = {kind: [[] for _ in range(5)] for kind in ("raw", "row_centered")}
        responses = {arm: [] for arm in ("native", "query_only", "key_only", "both")}
        probability_errors = {arm: [] for arm in responses}
        profile = {arm: {"probability_l1": [], "probability_max_abs": [], "mixture_error_norm": [],
                         "captured_probability_mixture_error_norm": []} for arm in ("native", "both")}
        row_metrics = []
        for t in range(count):
            captured = [a["probabilities"][6*t*(t+1):6*(t+1)*(t+2)].reshape(12, t+1).astype(float)
                        for a in arrays]
            for head in range(12):
                group = head // 6
                qn, qc = native["q"][t, head], compiled["q"][t, head]
                kn, kc = native["k"][:t+1, group], compiled["k"][:t+1, group]
                vn = native["v"][:t+1, group]
                dq, dk = qc-qn, kc-kn
                sn, sc = kn@qn/math.sqrt(128), kc@qc/math.sqrt(128)
                terms = [kn@dq/math.sqrt(128), dk@qn/math.sqrt(128), dk@dq/math.sqrt(128)]
                vectors = [sn, sc, *terms]
                row = {"position": t, "head": head, "history_rows": t+1}
                for kind in pooled:
                    v = vectors if kind == "raw" else [x - math.fsum(float(z) for z in x)/len(x) for x in vectors]
                    for target, x in zip(pooled[kind], v):
                        target.append(x)
                    row[kind] = decompose(*v)
                scores = {"native": sn, "query_only": kn@qc/math.sqrt(128),
                          "key_only": kc@qn/math.sqrt(128), "both": sc}
                probs = {arm: softmax(score) for arm, score in scores.items()}
                for arm, prob in probs.items():
                    responses[arm].append(prob@vn)
                    probability_errors[arm].append(float(np.sum(np.abs(prob-probs["native"]))))
                for arm, idx in (("native", 0), ("both", 1)):
                    cp = captured[idx][head]
                    saved = arrays[idx]["attention"][t, head]
                    profile[arm]["probability_l1"].append(float(np.sum(np.abs(probs[arm]-cp))))
                    profile[arm]["probability_max_abs"].append(float(np.max(np.abs(probs[arm]-cp))))
                    profile[arm]["mixture_error_norm"].append(norm(responses[arm][-1]-saved))
                    profile[arm]["captured_probability_mixture_error_norm"].append(norm(cp@vn-saved))
                row_metrics.append(row)
        aggregates = {kind: decompose(*(np.concatenate(items) for items in vectors)) for kind, vectors in pooled.items()}
        response = {arm: np.asarray(x) for arm, x in responses.items()}
        response_den = norm(response["native"])
        hybrid = {arm: {"attention_error_norm": norm(y-response["native"]),
                        "attention_error_relative_to_fp64_native": ratio(norm(y-response["native"]), response_den),
                        "probability_row_l1_difference": stats(probability_errors[arm])}
                  for arm, y in response.items()}
        interaction = response["both"]-response["query_only"]-response["key_only"]+response["native"]
        hybrid["nonlinear_attention_interaction_relative_to_fp64_native"] = ratio(norm(interaction), response_den)
        for arm, idx in (("native", 0), ("both", 1)):
            saved = arrays[idx]["attention"].reshape(-1, 128)
            profile[arm]["fp64_response_relative_to_saved_cuda"] = ratio(norm(response[arm]-saved), norm(saved))
            for key in ("probability_l1", "probability_max_abs", "mixture_error_norm", "captured_probability_mixture_error_norm"):
                profile[arm][key] = stats(profile[arm][key])
        cases.append({"case": case, "name": report["inputs"]["cases"][case]["name"],
                      "initial_prompt_positions": count, "query_rows": 12*count,
                      "score_entries": 6*count*(count+1), "shared_receivers_and_native_values_exact": True,
                      "score_decomposition": aggregates,
                      "query_term_exceeds_key_term_rows": {kind: sum(row[kind]["terms"]["query"]["norm"] > row[kind]["terms"]["key"]["norm"] for row in row_metrics) for kind in pooled},
                      "row_term_norms": {kind: {term: stats([row[kind]["terms"][term]["norm"] for row in row_metrics])
                                                for term in ("query", "key", "interaction")} for kind in pooled},
                      "hybrid_fp64_responses": hybrid, "native_profile_qualification": profile})
    return {"schema": "tcm-saved-qk-score-audit-v1", "report_sha256": expected,
            "helper_sha256": sha(__file__), "model_identity": report["model"]["identity"],
            "raw_files": used, "arithmetic": "FP64 NumPy dot products; fsum scalar norms and normalization; scale1/sqrt128; supplied saved post-RoPE operands",
            "scope": "Post-hoc initial common prompts only. Hybrid Q/K responses are CPU64 diagnostics with native V, not executed CUDA mixed policies or autonomous continuations. Row centering removes each head/clock mean over causal positions. Aggregates pool score entries, not independent samples.",
            "models_evaluated": 0, "target_fitting": False, "causal_attribution_percentages": False,
            "cases": cases, "terminal_status": "not-requested"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.result_directory / "score-audit.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit: " + str(output))
    result = audit(args.result_directory, args.expected_report_sha256)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "cases": len(result["cases"]), "models_evaluated": 0}))
