"""Additive cross-policy interpretation of authenticated composition evidence."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import shutil
import numpy as np
from analysis import distance, history_domains, same
from zero_analysis import analyze as analyze_zero


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticated(root):
    report = json.loads((root/"report.json").read_text())
    verification = json.loads((root/"verification.json").read_text())
    if verification["report_sha256"] != sha(root/"report.json") or not verification["analysis_recomputed"]:
        raise ValueError("Unverified source report")
    for name, info in report["raw_files"].items():
        p = root/"raw"/name
        if sha(p) != info["sha256"] or p.stat().st_size != info["bytes"]:
            raise ValueError("Source array identity differs")
    return report


def write(primary, zero, out):
    first, control = authenticated(primary), authenticated(zero)
    # Includes the strengthened exact-arm-set audit without changing the
    # original producer or its executed analyzer snapshot.
    analyze_zero(zero/"raw", zero/"payload/bank", control["driver"], control["texts"], first)
    if first["model"] != control["model"] or first["baseline"]["bank_concat_sha256"] != control["baseline"]["bank_concat_sha256"]:
        raise ValueError("Cross-run model/bank differs")
    arms = {a["name"]: (a, primary) for a in first["driver"]["arms"]}
    arms.update({a["name"]: (a, zero) for a in control["driver"]["arms"]})
    def load(name, step, port, n=1536):
        _, root = arms[name]
        value = np.fromfile(root/"raw"/f"{name}-s{step:03d}-{port}.f32", dtype="<f4")
        if value.shape != (n,) or not np.isfinite(value).all():raise ValueError("Invalid comparison array")
        return value
    def first_difference(a, b):
        x, y = a["own"]["ids"], b["own"]["ids"]
        index = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), None)
        return min(len(x), len(y)) if index is None and len(x) != len(y) else index
    def pair(left, right):
        a, _ = arms[left];b, _ = arms[right]
        if (a["parent_position"], a["parent_kv_sha256"]) != (b["parent_position"], b["parent_kv_sha256"]):raise ValueError("Different token parents")
        domains = history_domains(a, b)
        rows = []
        for j, domain in enumerate(domains):
            va, vb = load(left, j, "raw_receiver"), load(right, j, "raw_receiver")
            if domain:
                for port in ("input", "raw_receiver", "mlp_input"):
                    if not same(load(left, j, port), load(right, j, port)):raise ValueError("Shared lower-layer dependency changed")
                ka, kb = (load(name, j, "kv16-17", 1024).reshape(2, 2, 256) for name in (left, right))
                if ka[:, 0].tobytes() != kb[:, 0].tobytes():raise ValueError("KV16 changed before its cut")
            wa, wb = load(left, j, "mlp_return"), load(right, j, "mlp_return")
            n0, n1 = np.linalg.norm(wa.astype(float)), np.linalg.norm(wb.astype(float))
            rows.append({"step": j, "same_token_history": domain, "actual_receiver_bytes_equal": same(va, vb),
                         "write_difference": distance(wa, wb),
                         "left_write_norm_over_right": float(n0/n1) if n1 else None,
                         "write_cosine": float(np.dot(wa.astype(float), wb.astype(float))/(n0*n1)) if n0*n1 else None,
                         "comparison_kind": "same-input-local-write" if domain else "different-history-trajectory"})
        eligible = [r["write_difference"]["relative_l2"] for r in rows if r["same_token_history"] and r["step"] > 0]
        return {"left": left, "right": right, "first_different_emitted_token_index": first_difference(a, b),
                "same_history_steps": sum(domains), "later_same_input_write_difference_range": [min(eligible), max(eligible)] if eligible else None,
                "rows": rows}
    recurring_vs_once = [pair(f"{t}-recurring", f"{t}-once") for t in ("recall", "math", "prose")]
    corrected_vs_native = [pair(f"{t}-corrected", f"{t}-native") for t in ("recall", "math", "prose")]
    zero_vs_native = [pair(f"{t}-zero", f"{t}-native") for t in ("recall", "math", "prose")]
    zero_vs_recurring = [pair(f"{t}-zero", f"{t}-recurring") for t in ("recall", "math", "prose")]
    groups = defaultdict(list)
    for record in first["analysis"]["own_state_diagnostics"]:
        groups[record["normalized_input_sha256"]].append({k: record[k] for k in ("arm", "step", "position")})
    multiplicity = dict(sorted(Counter(map(len, groups.values())).items()))
    per_path_repeats = {}
    for name in arms:
        if name.endswith("-zero"):continue
        rows = [r for r in first["analysis"]["own_state_diagnostics"] if r["arm"] == name]
        per_path_repeats[name] = len(rows)-len({r["normalized_input_sha256"] for r in rows})
    first_fields = []
    for task in ("recall", "math", "prose"):
        for label in ("once", "recurring", "corrected", "zero"):
            name, native = f"{task}-{label}", f"{task}-native"
            first_fields.append({"task": task, "policy": label,
                                 "write_relative_difference": distance(load(name, 0, "mlp_return"), load(native, 0, "mlp_return"))["relative_l2"],
                                 "first_logits_relative_difference": distance(load(name, 0, "first", 151936), load(native, 0, "first", 151936))["relative_l2"]})
    correction_errors = [r["write_difference"]["relative_l2"] for p in corrected_vs_native for r in p["rows"]]
    answer_rows = first["analysis"]["summary"]["answers"]+control["analysis"]["summary"]["answers"]
    analysis_sources = {name: sha(Path(__file__).parent/name) for name in ("derive.py", "analysis.py", "zero_analysis.py")}
    data = {"schema": "tcm-composition-study-v1", "sources": {"primary_report_sha256": sha(primary/"report.json"), "zero_report_sha256": sha(zero/"report.json"), "analysis_sources": analysis_sources},
            "mechanics_status": "complete", "instrument_status": "executed", "trend_status": "bounded-observations", "terminal_status": "not-requested",
            "recurring_vs_once": recurring_vs_once, "corrected_vs_native": corrected_vs_native,
            "zero_vs_native": zero_vs_native, "zero_vs_recurring": zero_vs_recurring,
            "corrected_same_input_write_difference_range": [min(correction_errors), max(correction_errors)],
            "first_fields": first_fields, "answers": answer_rows,
            "operand_reuse": {"scope": "primary panel; equality of native MLP normalized operands under the bound weights/profile; prospective sharing only; PROJECT additionally reads coefficients/rho", "occurrences": sum(map(len, groups.values())), "unique_inputs": len(groups), "group_multiplicities": multiplicity, "within_path_repeats": per_path_repeats, "groups": dict(groups)},
            "cost": {"primary": json.loads((primary/"job.json").read_text())["result"], "zero": json.loads((zero/"job.json").read_text())["result"]}}
    out.mkdir(parents=True, exist_ok=True)
    (out/"analysis-source").mkdir(exist_ok=True)
    for name in analysis_sources:shutil.copy2(Path(__file__).parent/name, out/"analysis-source"/name)
    (out/"study.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"corrected_range": data["corrected_same_input_write_difference_range"], "recurring_vs_once": [{k: p[k] for k in ("left", "first_different_emitted_token_index", "same_history_steps", "later_same_input_write_difference_range")} for p in recurring_vs_once], "operand_reuse": {k: data["operand_reuse"][k] for k in ("occurrences", "unique_inputs", "group_multiplicities")}, "answers": answer_rows}, indent=2))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("primary", type=Path);parser.add_argument("zero", type=Path);parser.add_argument("out", type=Path);args = parser.parse_args()
    write(args.primary, args.zero, args.out)
