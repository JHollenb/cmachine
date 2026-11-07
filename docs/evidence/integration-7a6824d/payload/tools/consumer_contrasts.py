"""Consumer-bound finite contrasts on sealed C-assay arrays; no model execution.

The full endpoint is the executed full-correction arm. A split-joint write
diagnostic has no downstream continuation and is never substituted for it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("project", "carry", "read", "correct", "dense")
COUNTS = {"mlp_return": 1536, "output": 1536, "first": 151936,
          "current-kv": 5632, "next-kv": 5632}
MODES = {"project": 2, "carry": 7, "read": 8, "correct": 3, "dense": 1}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ratio(numerator, denominator):
    # A zero effect has no defined relative size or direction. Do not turn it
    # into an enormous number by inserting an arbitrary denominator floor.
    return float(numerator / denominator) if denominator else None


def contrast(arrays):
    """A five-arm accounting identity at ONE common observable and clock."""
    if set(arrays) != set(ARMS):
        raise ValueError("contrast needs all four executed arms and native reference")
    v = {k: np.asarray(value, dtype=np.float64).reshape(-1)
         for k, value in arrays.items()}
    if not len(v["dense"]) or any(value.shape != v["dense"].shape or
                                   not np.isfinite(value).all() for value in v.values()):
        raise ValueError("consumer arrays need equal nonempty finite geometry")
    b, c, r, f, native = (v[k] for k in ARMS)
    terms = {"carry": c-b, "read": r-b, "interaction": f-c-r+b}
    joint = f-b
    conditional = {"read_given_carry": f-c, "carry_given_read": f-r}
    dot = lambda a, b: float(np.dot(a, b))
    norm = lambda a: float(np.linalg.norm(a))
    jnorm = norm(joint)

    def geometry(value):
        length = norm(value)
        return {"l2": length, "rms": length / np.sqrt(len(value)),
                "max_abs": float(np.max(np.abs(value))),
                "norm_over_joint": ratio(length, jnorm),
                "signed_projection_onto_joint": ratio(dot(value, joint), dot(joint, joint)),
                "cosine_with_joint": ratio(dot(value, joint), length * jnorm)}

    pairs = {}
    names = list(terms)
    for index, name in enumerate(names):
        for other in names[index+1:]:
            pairs[name+" / "+other] = 2 * dot(terms[name], terms[other])
    powers = {k: dot(value, value) for k, value in terms.items()}
    power_sum = sum(powers.values()) + sum(pairs.values())
    errors = {k: value-native for k, value in v.items()}
    error_norm = {k: norm(value) for k, value in errors.items()}
    transitions = {}
    for name, before, after in (("carry_alone", "project", "carry"),
                                ("read_alone", "project", "read"),
                                ("read_given_carry", "carry", "correct"),
                                ("carry_given_read", "read", "correct")):
        e0, e1 = errors[before], errors[after]
        delta = v[after]-v[before]
        # Fixed native reference at both endpoints. This is paired error
        # accounting, not a measured native response or a probabilistic model.
        p0, p1, cross = dot(e0, e0), dot(e1, e1), dot(e0, e1)
        transitions[name] = {
            "before_error_l2": error_norm[before], "after_error_l2": error_norm[after],
            "error_norm_reduction": (1-ratio(error_norm[after], error_norm[before]))
                                    if error_norm[before] else None,
            "signed_error_direction_removed": ratio(-dot(delta, e0), p0),
            "paired_error_identity": {"before_squared": p0, "after_squared": p1,
                "uncentered_cross_inner_product": cross,
                "error_change_squared": dot(e1-e0, e1-e0),
                "identity_roundoff": dot(e1-e0, e1-e0)-(p0+p1-2*cross)}}
    return {"count": len(native), "native_l2": norm(native),
            "joint": geometry(joint),
            "terms": {k: geometry(value) for k, value in terms.items()},
            "conditional_effects": {k: geometry(value) for k, value in conditional.items()},
            "read_alone_vs_read_given_carry_cosine": ratio(dot(r-b, f-c), norm(r-b)*norm(f-c)),
            "native_error_l2": error_norm, "native_error_transitions": transitions,
            "energy_identity": {"term_squared_norms": powers, "twice_pair_inner_products": pairs,
                "joint_squared_norm": dot(joint, joint), "expanded_squared_norm": power_sum,
                "roundoff": dot(joint, joint)-power_sum},
            "tensor_identity_roundoff_max_abs": float(np.max(np.abs(joint-sum(terms.values()))))}


def common_parent(rows):
    """Integrity checks, independent of whether any scientific effect is large."""
    if set(rows) != set(ARMS):
        raise ValueError("missing executed component arm")
    native = rows["dense"]
    for arm, row in rows.items():
        if row["mode"] != MODES[arm] or row["rank"] != 3:
            raise ValueError("consumer arm role or rank differs")
        for key in ("position", "parent_kv_prefix_sha256", "rho", "rank"):
            if row[key] != native[key]:
                raise ValueError("consumer quartet has incompatible "+key)
        if not row["prefix_intact"]:
            raise ValueError("consumer quartet changed retained prefix")


def analyze(root, verified_context=None):
    root = Path(root)
    report_bytes = (root/"report.json").read_bytes()
    report = json.loads(report_bytes)
    if report["schema"] != "tcm-components-experiment-v1":
        raise ValueError("consumer adapter requires the sealed component assay schema")
    # verify.py establishes scheduler-log/report/source/receipt joins. Standalone
    # use requires its prior receipt for this exact report; the adapter does not
    # independently attest execution merely because report-listed arrays hash.
    if verified_context is None:
        verified_context = json.loads((root/"verification.json").read_text())
    if (verified_context["report_sha256"] != sha(report_bytes) or
            not verified_context["scheduler_log_seal"] or not verified_context["report_log_join"]):
        raise ValueError("run verify.py first: consumer adapter needs verified report custody")
    submission = json.loads((root/"submission.json").read_text())
    if submission["job_id"] != verified_context["job_id"]:
        raise ValueError("consumer verification job identity differs")
    source_bindings = {}
    for name in ("tests/reuse_experiment.c", "tests/assay_io.h"):
        digest = sha((root/"payload"/name).read_bytes())
        if digest != submission["config"]["sources"][name]:
            raise ValueError("common-token/array-layout source identity differs")
        source_bindings[name] = digest
    raw_driver = (root/"raw/reuse-driver.json").read_bytes()
    if sha(raw_driver) != report["driver_raw_sha256"]:
        raise ValueError("consumer raw driver identity differs")
    driver = report["driver"]
    unannotated = json.loads(json.dumps(driver))
    for row in unannotated["rows"]+unannotated["calibration_rows"]+unannotated["target_rows"]:
        row["own"].pop("text")
        for question in row["questions"]:
            question["output"].pop("text")
    unannotated["rebuilt_source_own"].pop("text")
    if unannotated != json.loads(raw_driver):
        raise ValueError("consumer raw driver/report join differs")
    if not driver["program_unchanged"] or driver["layer"] != 16:
        raise ValueError("frozen program or declared clock differs")
    rows = {}
    for arm in ARMS:
        matches = [r for r in driver["target_rows"] if r["name"] == "Niko-C-"+arm]
        if len(matches) != 1:
            raise ValueError("missing or duplicate executed component arm")
        rows[arm] = matches[0]
    common_parent(rows)
    consumed = {}

    def load(name, count):
        raw = (root/"raw"/name).read_bytes()
        record = report["raw_files"][name]
        if len(raw) != count*4 or record["bytes"] != len(raw) or sha(raw) != record["sha256"]:
            raise ValueError("consumer raw identity/extent differs: "+name)
        value = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite consumer array: "+name)
        consumed[name] = record
        return value

    def read(arm, port, count):
        return load(rows[arm]["name"]+"-"+port+".f32", count)

    # Actual receiving operands, not just a matching tensor shape or prefix hash.
    for port in ("entry", "attention-write", "receiver", "reconstruction", "omitted", "mlp_input"):
        expected = read("dense", port, 1536)
        for arm in ARMS[:-1]:
            if expected.tobytes() != read(arm, port, 1536).tobytes():
                raise ValueError("consumer arms have different receiving operand: "+port)

    # The original driver does not serialize a common-token field. Its sealed
    # source sets common to the first (dense) pending ID before visiting siblings.
    native = rows["dense"]
    common_token = native["first"]["top"]
    if native["own"]["ids"][0] != common_token:
        raise ValueError("native pending/own token join differs")
    position = native["position"]
    records = []
    for port, count in COUNTS.items():
        arrays = {arm: read(arm, port, count) for arm in ARMS}
        if port == "first":
            for arm, array in arrays.items():
                if int(np.argmax(array)) != rows[arm]["first"]["top"]:
                    raise ValueError("consumer first-token readout join differs")
        horizon = {"mlp_return": "layer16-mlp-write", "output": "layer16-residual-exit",
                   "first": "current-token-final-readout", "current-kv": "current-token-upper-memory",
                   "next-kv": "common-next-token-upper-memory"}[port]
        binding = {"port": port, "consumer": horizon,
                   "position": position+int(port == "next-kv"),
                   "common_next_input_token_id": common_token if port == "next-kv" else None}
        records.append({**binding, "selection": "full-retained-array", "metrics": contrast(arrays)})
        if port.endswith("kv"):
            # assay_io.h writes KIND first, then layer, then combined KV head.
            shaped = {arm: a.reshape(2, 11, 256) for arm, a in arrays.items()}
            for kind, label in enumerate(("key", "value")):
                for index in range(11):
                    records.append({**binding, "selection": label, "layer": 17+index,
                        "metrics": contrast({arm: a[kind, index] for arm, a in shaped.items()})})

    if driver["split_joint"]["state_advanced"] or not driver["split_joint"]["session_metadata_unchanged"]:
        raise ValueError("split-joint diagnostic unexpectedly advanced state")
    joint_write = load("split-joint-mlp_return.f32", 1536)
    full_write = read("correct", "mlp_return", 1536)
    return {"schema": "tcm-consumer-contrasts-v1", "kind": "posthoc-observation",
        "mechanics_status": "valid", "instrument_status": "applicable",
        "trend_status": "observations-retained", "terminal_status": "unassessed",
        "report_sha256": sha(report_bytes), "analyzer_sha256": sha(Path(__file__).read_bytes()),
        "custody": {"job_id": verified_context["job_id"],
            "prerequisite": "verify.py scheduler-log/report/source/receipt verification for this exact report",
            "verifier_sha256": verified_context["verifier_sha256"],
            "driver_raw_sha256": sha(raw_driver), "source_bindings": source_bindings},
        "model": {k: report["model"][k] for k in ("identity", "content_sha256", "profile_version", "numerical_contract")},
        "execution": {"binary_sha256": report["candidate_binary_sha256"],
            "driver_binary_sha256": report["driver_binary_sha256"],
            "layer": driver["layer"], "position": position,
            "parent_kv_prefix_sha256": native["parent_kv_prefix_sha256"],
            "parent_scope": "KV-prefix seal; not an independently serialized complete-state identity",
            "program_sha256": driver["program_sha256"],
            "extension_program_sha256": driver["extension_program_sha256"],
            "rho": native["rho"], "target_input": report["input"]["targets"][0],
            "common_next_input_token_id": common_token,
            "common_next_token_binding": "sealed reuse_experiment.c row(): first dense pending ID, reused for every sibling",
            "own_token_histories_equal": all(r["own"]["ids"] == native["own"]["ids"] for r in rows.values()),
            "own_output_ids": {arm: row["own"]["ids"] for arm, row in rows.items()}},
        "definitions": {"00": "executed frozen-B3 project arm", "10": "executed carry-correction arm",
            "01": "executed read-complement correction arm", "11": "executed full-omission correction arm",
            "interaction": "Y11-Y10-Y01+Y00; operational full-correction endpoint",
            "conditional_effects": "Y11-Y10 and Y11-Y01; matched sibling endpoint contrasts, not sequential interventions",
            "error_reference": "same dense tensor for all endpoints; paired numerical identity, not a native edited-minus-baseline forecast",
            "cross_port_comparison": "separate port geometries and response denominators; ratios do not establish amplification",
            "behavior": "same-token-history numerical effects; no task-necessity, semantic-mediation or accuracy claim"},
        "local_split_joint_diagnostic": {
            "full_vs_joint_write_l2": float(np.linalg.norm(full_write-joint_write)),
            "relative_to_full_write": ratio(np.linalg.norm(full_write-joint_write), np.linalg.norm(full_write)),
            "state_advanced": driver["split_joint"]["state_advanced"],
            "scope": "pure local write only; no split-joint continuation exists and no downstream error bound follows"},
        "raw_files": consumed, "records": records}


def write(root, verified_context=None):
    root = Path(root)
    result = analyze(root, verified_context)
    snapshot = root/"analysis-source"/"consumer_contrasts.py"
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    (root/"consumer-contrasts.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = write(args.root)
    print(json.dumps({"schema": result["schema"], "records": len(result["records"]),
                      "raw_files": len(result["raw_files"]), "report_sha256": result["report_sha256"]}))
