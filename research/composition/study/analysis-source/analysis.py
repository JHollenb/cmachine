"""Saved-array analysis; trajectory comparisons retain their history domain."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

D, H, VOCAB = 1536, 8960, 151936
TASKS = ("recall", "math", "prose")
POLICIES = ("native", "once", "recurring", "corrected")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seqdot(a, b):
    result = 0.0
    for x, y in zip(a, b):
        result += float(x) * float(y)
    return result


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def distance(a, b):
    a, b = a.astype(float), b.astype(float)
    delta = a-b
    denominator = float(np.linalg.norm(b))
    return {"l2": float(np.linalg.norm(delta)),
            "relative_l2": float(np.linalg.norm(delta)/denominator) if denominator else None,
            "max_abs": float(np.max(np.abs(delta)))}


def validate_token_chain(arm):
    steps, own = arm["steps"], arm["own"]
    if not own["all_delivered_consumed"] or len(steps) != len(own["ids"])+1:
        raise ValueError("Consumed token closure differs")
    if len(own["ids"]) > arm["budget"] or not own["ids"]:
        raise ValueError("Generation budget differs")
    if own["stopped"] == own["truncated"] or own["stopped"] != (own["ids"][-1] in (151645, 151643)):
        raise ValueError("Termination metadata differs")
    if own["truncated"] and len(own["ids"]) != arm["budget"]:
        raise ValueError("Early truncation")
    if any(token in (151645, 151643) for token in own["ids"][:-1]):
        raise ValueError("Generation continued after EOS")
    for j, token in enumerate(own["ids"]):
        if token != steps[j]["top"] or token != steps[j+1]["input_id"]:
            raise ValueError("A generated token was replaced")


def history_domains(arm, reference):
    own_inputs, native_inputs, result = [], [], []
    for a, b in zip(arm["steps"], reference["steps"]):
        own_inputs.append(a["input_id"]);native_inputs.append(b["input_id"])
        result.append(own_inputs == native_inputs)
    return result


def analyze(raw, bank, driver, texts):
    raw, bank = Path(raw), Path(bank)
    if driver["schema"] != "tcm-composition-driver-v1" or driver["layer"] != 16 or driver["capacity"] != 512:
        raise ValueError("Composition contract differs")
    def load(stem, port, n=D):
        a = np.fromfile(raw/f"{stem}-{port}.f32", dtype="<f4")
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid retained array: {stem}-{port}")
        return a
    base = np.fromfile(bank/"reuse-program.f32", dtype="<f4")
    extra = np.fromfile(bank/"enrich-program.f32", dtype="<f4")
    qr = np.fromfile(bank/"reuse-qr.f64", dtype="<f8")
    qe = np.fromfile(bank/"enrich-qr.f64", dtype="<f8")
    if tuple(map(len, (base, extra, qr, qe))) != (40448, 19456, 3076, 1539):
        raise ValueError("Frozen bank extent differs")
    if not all(np.isfinite(a).all() for a in (base, extra, qr, qe)):
        raise ValueError("Nonfinite bank")
    basis = (base[:D], base[D:2*D], extra[:D])
    q = (qr[:D], qr[D:2*D], qe[:D])
    r00, r01, r11, _ = qr[2*D:]
    r02, r12, r22 = qe[D:]
    arms = {a["name"]: a for a in driver["arms"]}
    if set(arms) != {f"{t}-{p}" for t in TASKS for p in POLICIES}:
        raise ValueError("Incomplete composition panel")
    comparisons, own_diagnostics, observations = [], [], []
    for task in TASKS:
        reference = arms[f"{task}-native"]
        initial = reference["steps"][0]
        initial_receiver = load(f"{reference['name']}-s000", "raw_receiver")
        for policy, label in enumerate(POLICIES):
            arm = arms[f"{task}-{label}"]
            if arm["policy"] != policy or not arm["prefix_intact"] or not arm["own"]["all_delivered_consumed"]:
                raise ValueError("Policy/cursor contract differs")
            validate_token_chain(arm)
            if (arm["parent_position"], arm["parent_kv_sha256"]) != (reference["parent_position"], reference["parent_kv_sha256"]):
                raise ValueError("Token parents differ")
            if len(arm["steps"]) != len(arm["own"]["ids"])+1 or arm["steps"][0]["input_id"] != initial["input_id"]:
                raise ValueError("Consumed token closure differs")
            if len(arm["own"]["ids"]) > arm["budget"] or arm["final_position"] != arm["parent_position"]+len(arm["steps"]):
                raise ValueError("Clock/budget differs")
            if arm["own"]["stopped"] == arm["own"]["truncated"]:
                raise ValueError("Termination metadata differs")
            # Every emitted token is exactly the pending token of the preceding
            # transition and exactly the input of the next transition.
            for j, token in enumerate(arm["own"]["ids"]):
                if token != arm["steps"][j]["top"] or token != arm["steps"][j+1]["input_id"]:
                    raise ValueError("A generated token was replaced")
            for j, step in enumerate(arm["steps"]):
                if step["step"] != j or step["position"] != arm["parent_position"]+j:
                    raise ValueError("Step ordering differs")
                active = policy in (2, 3) or (policy == 1 and j == 0)
                if step["substitution"] != active or step["gate_up_calls"] != (0 if active and policy != 3 else 2) or step["down_calls"] != 1 or step["compile_calls"] != 0:
                    raise ValueError("Physical work differs")
                stem = f"{arm['name']}-s{j:03d}"
                receiver, write, out = (load(stem, p) for p in ("raw_receiver", "mlp_return", "output"))
                load(stem, "input"); normalized = load(stem, "mlp_input"); load(stem, "kv16-17", 1024)
                if not same(out, np.add(receiver, write)):
                    raise ValueError("Actual skip was replaced")
                projected = np.zeros(D, dtype=np.float64)
                for column in q:
                    projected += column*seqdot(column, receiver)
                geometric_omission = float(np.linalg.norm(receiver.astype(float)-projected))
                detail = {"arm": arm["name"], "step": j, "position": step["position"],
                          "substitution": active, "own_receiver_l2": float(np.linalg.norm(receiver.astype(float))),
                          "own_Q3_geometric_omission_l2": geometric_omission,
                          "normalized_input_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
                          "scope": "own-state geometry; no same-input missing-write estimate"}
                if active:
                    v0, v1, v2 = (seqdot(column, receiver) for column in q)
                    c2 = v2/r22; c1 = (v1-r12*c2)/r11; c0 = (v0-r01*c1-r02*c2)/r00
                    coefficients = np.asarray([c0, c1, c2])
                    if not np.allclose(coefficients, step["coefficients"], rtol=1e-12, atol=1e-14):
                        raise ValueError("Coordinate solve differs")
                    c = np.asarray(step["executed"], dtype=np.float32)
                    if not same(c, coefficients.astype(np.float32)):
                        raise ValueError("Executed coordinates differ")
                    reconstruction = np.add(np.add(c[0]*basis[0], c[1]*basis[1]), c[2]*basis[2])
                    if not same(reconstruction, load(stem, "reconstruction")):
                        raise ValueError("Reconstruction arithmetic differs")
                    omitted = np.add(np.add(receiver, -np.float32(1)*reconstruction), np.float32(0)*reconstruction)
                    if not same(omitted, load(stem, "omitted")):
                        raise ValueError("Executed omission differs")
                    if not np.isclose(np.linalg.norm(omitted.astype(float)), step["omitted_norm"], rtol=1e-12, atol=1e-14):
                        raise ValueError("Omission norm differs")
                    detail["executed_omission_l2"] = step["omitted_norm"]
                own_diagnostics.append(detail)
            first_stem = f"{arm['name']}-s000"
            first = load(first_stem, "first", VOCAB)
            final = load(arm["name"], "final", VOCAB)
            if digest(raw/f"{first_stem}-first.f32") != arm["steps"][0]["first"]["sha256"] or digest(raw/f"{arm['name']}-final.f32") != arm["final"]["sha256"]:
                raise ValueError("Readout identity differs")
            if int(np.argmax(first)) != arm["steps"][0]["top"] or int(np.argmax(final)) != arm["final"]["top"]:
                raise ValueError("Raw greedy selection differs")
            if not same(load(first_stem, "raw_receiver"), initial_receiver):
                raise ValueError("First live receivers differ")
            if policy:
                # First-token projected G/U operand assembly is authenticated
                # independently of its downstream consumer.
                c = np.asarray(arm["steps"][0]["executed"], dtype=np.float32)
                for channel, offset, extra_offset in (("gate", 2*D, D), ("up", 2*D+2*H, D+H)):
                    value = np.add(c[0]*base[offset:offset+H], c[1]*base[offset+H:offset+2*H])
                    value = np.multiply(np.float32(arm["steps"][0]["rho"]), np.add(value, c[2]*extra[extra_offset:extra_offset+H]))
                    if policy == 3:
                        value = np.add(value, load(first_stem, "correction-"+channel, H))
                    if not same(value, load(first_stem, "mlp_"+channel, H)):
                        raise ValueError("First projected operands differ")
        once, recurring = (arms[f"{task}-{p}"] for p in ("once", "recurring"))
        if once["steps"][0] != recurring["steps"][0]:
            raise ValueError("Once/recurring first complete token differs")
        for port in ("input", "raw_receiver", "mlp_input", "mlp_gate", "mlp_up", "mlp_return", "output", "kv16-17", "first", "reconstruction", "omitted"):
            if (raw/f"{once['name']}-s000-{port}.f32").read_bytes() != (raw/f"{recurring['name']}-s000-{port}.f32").read_bytes():
                raise ValueError("Once/recurring first arrays differ")
        for label in POLICIES[1:]:
            arm = arms[f"{task}-{label}"]
            first_token_difference = next((j for j, (a, b) in enumerate(zip(arm["own"]["ids"], reference["own"]["ids"])) if a != b), None)
            if first_token_difference is None and len(arm["own"]["ids"]) != len(reference["own"]["ids"]):
                first_token_difference = min(len(arm["own"]["ids"]), len(reference["own"]["ids"]))
            record = {"task": task, "policy": label, "name": arm["name"],
                      "own_text": texts[arm["name"]], "native_text": texts[reference["name"]],
                      "own_ids_equal": arm["own"]["ids"] == reference["own"]["ids"],
                      "first_different_emitted_token_index": first_token_difference,
                      "first_input_id": initial["input_id"], "same_input_first": {}, "trajectory": []}
            for port, extent in (("mlp_return", D), ("output", D), ("first", VOCAB), ("kv16-17", 1024)):
                record["same_input_first"][port] = distance(load(f"{arm['name']}-s000", port, extent), load(f"{reference['name']}-s000", port, extent))
            domains = history_domains(arm, reference)
            for j, (a, b) in enumerate(zip(arm["steps"], reference["steps"])):
                same_history = domains[j]
                if same_history:
                    for port in ("input", "raw_receiver", "mlp_input"):
                        if not same(load(f"{arm['name']}-s{j:03d}", port), load(f"{reference['name']}-s{j:03d}", port)):
                            raise ValueError("Below-cut input changed on identical token history")
                    ka = load(f"{arm['name']}-s{j:03d}", "kv16-17", 1024).reshape(2, 2, 256)
                    kb = load(f"{reference['name']}-s{j:03d}", "kv16-17", 1024).reshape(2, 2, 256)
                    if ka[:, 0].tobytes() != kb[:, 0].tobytes():
                        raise ValueError("KV16 changed before the MLP16 intervention")
                metric = {"step": j, "same_complete_token_history": same_history,
                          "comparison_kind": "same-token-history-fidelity" if same_history else "different-history-trajectory",
                          "same_local_receiver_bytes": same_history,
                          "receiving_input": distance(load(f"{arm['name']}-s{j:03d}", "raw_receiver"), load(f"{reference['name']}-s{j:03d}", "raw_receiver")),
                          "write": distance(load(f"{arm['name']}-s{j:03d}", "mlp_return"), load(f"{reference['name']}-s{j:03d}", "mlp_return"))}
                record["trajectory"].append(metric)
                observations.append({"kind": "observation", "case": arm["name"], "step": j, "metrics": metric})
            comparisons.append(record)
    answers = []
    for arm in driver["arms"]:
        text = texts[arm["name"]].strip()
        answers.append({"name": arm["name"], "task": arm["task"], "text": texts[arm["name"]],
                        "task_correct": text.lower() == "blue" if arm["task"] == "recall" else text == "391" if arm["task"] == "math" else None,
                        "emitted_tokens": len(arm["own"]["ids"]), "stopped": arm["own"]["stopped"], "truncated": arm["own"]["truncated"]})
    substitutions = {a["name"]: sum(s["substitution"] for s in a["steps"]) for a in driver["arms"]}
    unique_norm_inputs = {a["name"]: len({x["normalized_input_sha256"] for x in own_diagnostics if x["arm"] == a["name"]}) for a in driver["arms"]}
    summary = {"arms": len(arms), "all_once_recurring_first_states_exact": True,
               "selected_layer_substitutions": substitutions, "unique_normalized_inputs": unique_norm_inputs,
               "selected_gate_up_calls": sum(s["gate_up_calls"] for a in driver["arms"] for s in a["steps"]),
               "selected_down_calls": sum(len(a["steps"]) for a in driver["arms"]),
               "original_model_projection_compile_calls": 0,
               "source_text": texts["source"], "answers": answers}
    return {"summary": summary, "comparisons": comparisons, "own_state_diagnostics": own_diagnostics, "observations": observations}
