"""Model-free recomputation of sealed gate-law and conversation evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

HIDDEN, INTERMEDIATE, VOCAB = 1536, 8960, 151936

def archive_bound(followup_lengths):
    if len(followup_lengths) != 2 or any(n < 2 for n in followup_lengths):
        raise ValueError("Two nonempty follow-up templates required")
    # Two primary+replay copies of each policy, native+three same-ID references,
    # twelve local probes, saved exact products/writes, 92 full readouts, anchors.
    return 963584*(66+sum(followup_lengths)) + 2*982016 + 55912448 + 716800 + 16*1024**2

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array(raw, name, count=None):
    v = np.fromfile(Path(raw) / name, dtype="<f4")
    if not np.isfinite(v).all() or (count is not None and v.size != count):
        raise ValueError("Invalid numerical extent: " + name)
    return v

def difference(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Unaligned numerical shapes")
    delta = a - b
    norm = float(np.linalg.norm(b))
    return {"l2": float(np.linalg.norm(delta)), "relative_l2": float(np.linalg.norm(delta)) / norm if norm else None, "max_abs": float(np.max(np.abs(delta))), "byte_exact": a.tobytes() == b.tobytes()}

def polynomial_product(g, u, law):
    source, p, first, second, s = np.asarray(law, dtype=np.float32)
    g, u = np.asarray(g, dtype=np.float32), np.asarray(u, dtype=np.float32)
    d = g - source
    sigma = (p + first * d) + (second * d) * d
    return (s + (g * sigma - source * p)) * u

def validate_path(path, source_position, generated):
    steps = path["steps"]
    if [s["input_id"] for s in steps] != path["consumed_ids"]:
        raise ValueError("Consumed history ledger differs")
    if len(steps) != len(path["consumed_ids"]) or not path["all_delivered_consumed"]:
        raise ValueError("Incomplete consumption contract")
    for j, row in enumerate(steps):
        if row["step"] != j or row["position"] != source_position + j:
            raise ValueError("Token clock differs")
        if generated and row["role"] == "assistant" and (not j or row["input_id"] != steps[j-1]["top"]):
            raise ValueError("Teacher token entered own-output trajectory")
        mode = row["mode"]
        expected = (0, 0, 0, 0) if mode == 2 else (2, 1, int(mode == 0), int(mode == 1))
        actual = tuple(row[k] for k in ("gate_up_calls", "down_calls", "native_swiglu", "polynomial_swiglu"))
        if actual != expected:
            raise ValueError("Executed gate-law work differs")
    expected_mode = lambda j: 2 if path["policy"] == 3 else int(path["policy"] == 2 or (path["policy"] == 1 and j == 0))
    if any(s["mode"] != expected_mode(j) for j, s in enumerate(steps)):
        raise ValueError("Once/recurring scope differs")
    assistant_ids = [s["input_id"] for s in steps if s["role"] == "assistant"]
    if assistant_ids != [token for answer in path["answers"] for token in answer["ids"]]:
        raise ValueError("Answer delivery ledger differs")
    return steps

def full_boundaries(path):
    rows = [s for s in path["steps"] if "readout" in s]
    if len(rows) != 4:
        raise ValueError("Expected four complete readout boundaries")
    return dict(zip(("initial", "first-end", "followup", "second-end"), rows))

def totals(paths):
    rows = [s for p in paths for s in p["steps"]]
    n = len(rows)
    result = {k: sum(s[k] for s in rows) for k in ("gate_up_calls", "down_calls", "native_swiglu", "polynomial_swiglu")}
    result.update(tokens=n, other_layer_native_swiglu=27*n, all_dense_matvec_calls=194*n + result["gate_up_calls"] + result["down_calls"])
    dense_mac = 28 * (2*HIDDEN**2 + 2*256*HIDDEN + 3*HIDDEN*INTERMEDIATE) + VOCAB*HIDDEN
    zero = sum(s["mode"] == 2 for s in rows)
    result["learned_linear_macs"] = dense_mac*n - 3*HIDDEN*INTERMEDIATE*zero
    result["dense_linear_macs_per_native_token"] = dense_mac
    result["attention_score_value_macs"] = sum(2*12*128*(s["position"]+1)*28 for s in rows)
    return result

def analyze(raw, unused_bank, driver, texts):
    raw = Path(raw)
    if driver.get("schema") != "tcm-gate-law-driver-v1":
        a = array(raw, "ordinary-historical.f32", VOCAB)
        b = array(raw, "ordinary-current.f32", VOCAB)
        if a.tobytes() != b.tobytes():
            raise ValueError("Native binary parity differs")
        return {"summary": {"lane": "mechanics", "native_logits_byte_exact": True, "logits": VOCAB, "gate_lifecycle_source_assertions_executed": True}, "observations": [{"case": "gate-mechanics", "native_logits_byte_exact": True}]}
    seal = json.loads((raw / "primary-seal.json").read_text())
    if seal["driver_prefix_bytes"] != driver["primary_seal_bytes"] or hashlib.sha256((raw / "driver.json").read_bytes()[:driver["primary_seal_bytes"]]).hexdigest() != driver["primary_seal_sha256"] or seal["driver_prefix_sha256"] != driver["primary_seal_sha256"]:
        raise ValueError("Primary prefix seal differs")
    for row in seal["files"]:
        if (raw / row["name"]).stat().st_size != row["bytes"] or sha(raw / row["name"]) != row["sha256"]:
            raise ValueError("A primary file changed after sealing")
    inputs = json.loads((raw / "input-manifest.json").read_text())
    if len(driver["primary_cases"]) != 2 or len(driver["references"]) != 2 or len(inputs["cases"]) != 2 or [c["case"] for c in driver["primary_cases"]] != [0, 1] or [c["case"] for c in driver["references"]] != [0, 1]:
        raise ValueError("Incomplete case geometry")
    observations, summaries, all_primary, all_native, all_replays = [], [], [], [], []
    replay_arrays = polynomial_arrays = readouts = 0
    for case, refs, declared in zip(driver["primary_cases"], driver["references"], inputs["cases"]):
        c = case["case"]
        position = len(declared["prompt_ids"]) - 1
        if position != case["source_position"] or case["source_anchor_position"] != position-1:
            raise ValueError("Source anchor clock differs")
        law = array(raw, f"case{c}-law.f32", 5*INTERMEDIATE).reshape(5, INTERMEDIATE)
        if sha(raw / f"case{c}-law.f32") != case["law_sha256"]:
            raise ValueError("Frozen law bytes differ")
        ones = np.ones(INTERMEDIATE, dtype=np.float32)
        anchor = polynomial_product(law[0], ones, law)
        if anchor.tobytes() != law[4].tobytes():
            raise ValueError("Finite anchored expression does not reconstruct source")
        native = refs["native"]
        validate_path(native, position, True)
        all_native.append(native)
        native_full = full_boundaries(native)
        if len(case["arms"]) != 3 or len(refs["same_ids"]) != 3 or len(refs["candidate_replays"]) != 3 or [a["policy"] for a in case["arms"]] != [1, 2, 3] or [a["policy"] for a in refs["candidate_replays"]] != [1, 2, 3] or any(a["policy"] for a in refs["same_ids"]):
            raise ValueError("Incomplete arm geometry")
        # Once and recurring must really begin with the same intervention.
        for suffix in ("raw_receiver", "mlp_input", "mlp_gate", "mlp_up", "product", "mlp_return", "output", "kv16-27", "logits"):
            a, b = (raw/f"{policy}-c{c}-s000-{suffix}.f32" for policy in ("once", "recurring"))
            if a.read_bytes() != b.read_bytes():
                raise ValueError("Once/recurring initial states differ")
        for arm, same, replay in zip(case["arms"], refs["same_ids"], refs["candidate_replays"]):
            validate_path(arm, position, True)
            validate_path(same, position, False)
            validate_path(replay, position, False)
            if arm["consumed_ids"] != same["consumed_ids"] or arm["consumed_ids"] != replay["consumed_ids"]:
                raise ValueError("Post-seal replay history differs")
            if arm["steps"][0]["input_id"] != declared["prompt_ids"][-1]:
                raise ValueError("Native first answer was supplied instead of pending prompt token")
            follow = [s["input_id"] for s in arm["steps"] if s["role"] in ("followup-template", "host-close")]
            if follow != declared["followup_ids"][arm["template_skip"]:]:
                raise ValueError("External user/template history differs")
            all_primary.append(arm)
            all_replays.extend((same, replay))
            errors = []
            for j, (row, replay_row) in enumerate(zip(arm["steps"], replay["steps"])):
                stem, other = f"{arm['name']}-s{j:03}", f"{replay['name']}-s{j:03}"
                suffixes = ["raw_receiver", "mlp_input", "mlp_return", "output", "kv16-27"]
                if row["mode"] != 2:
                    suffixes += ["mlp_gate", "mlp_up", "product"]
                if "readout" in row:
                    suffixes += ["logits"]
                    if row["valid_kv_sha256"] != replay_row["valid_kv_sha256"]:
                        raise ValueError("Candidate full cache replay differs")
                for suffix in suffixes:
                    if (raw / f"{stem}-{suffix}.f32").read_bytes() != (raw / f"{other}-{suffix}.f32").read_bytes():
                        raise ValueError("Candidate replay changed: " + stem + "-" + suffix)
                    replay_arrays += 1
                if row["mode"] == 1:
                    g, u = array(raw, stem+"-mlp_gate.f32", INTERMEDIATE), array(raw, stem+"-mlp_up.f32", INTERMEDIATE)
                    product = array(raw, stem+"-product.f32", INTERMEDIATE)
                    if polynomial_product(g, u, law).tobytes() != product.tobytes():
                        raise ValueError("Saved polynomial primitive differs")
                    polynomial_arrays += 1
                    exact_write = array(raw, stem+"-exact-write.f32", HIDDEN)
                    write_error = difference(array(raw, stem+"-mlp_return.f32", HIDDEN), exact_write)
                    gate_error = difference(product, array(raw, stem+"-exact-product.f32", INTERMEDIATE))
                    displacement = float(np.linalg.norm(g.astype(np.float64)-law[0]))
                    errors.append(write_error["relative_l2"])
                    observations.append({"case": c, "policy": arm["name"], "consumer": "same-operand-write", "step": j, "role": row["role"], "gate_displacement_l2": displacement, "write": write_error, "product": gate_error})
            arm_full, same_full = full_boundaries(arm), full_boundaries(same)
            comparisons = {}
            for label, row in arm_full.items():
                own = array(raw, f"{arm['name']}-s{row['step']:03}-logits.f32", VOCAB)
                same_row, native_row = same_full[label], native_full[label]
                common = array(raw, f"{same['name']}-s{same_row['step']:03}-logits.f32", VOCAB)
                native_logits = array(raw, f"{native['name']}-s{native_row['step']:03}-logits.f32", VOCAB)
                if sha(raw / f"{arm['name']}-s{row['step']:03}-logits.f32") != row["readout"]["sha256"]:
                    raise ValueError("Full vocabulary readout digest differs")
                comparisons[label] = {"same_consumed_ids": difference(own, common), "own_history": difference(own, native_logits), "history_conditioning": difference(common, native_logits), "candidate_position": row["position"], "native_position": native_row["position"]}
                observations.append({"case": c, "policy": arm["name"], "consumer": "logits-"+label, **comparisons[label]})
                readouts += 1
            local = []
            for probe in sorted(raw.glob(f"local-replay-{arm['name']}-p*.json")):
                q = json.loads(probe.read_text())
                j = q["candidate_step"]
                own = array(raw, f"{arm['name']}-s{j:03}-logits.f32", VOCAB)
                actual_cache_native = array(raw, probe.stem+"-s000-logits.f32", VOCAB)
                row = {"step": j, "position": arm["steps"][j]["position"], "same_cache_local_logits": difference(own, actual_cache_native)}
                observations.append({"case": c, "policy": arm["name"], "consumer": "same-cache-native-step", **row})
                local.append(row)
            if len(local) != 2:
                raise ValueError("Missing same-cache first/follow-up probes")
            summaries.append({"case": c, "policy": arm["name"], "answers": texts[arm["name"]], "native_answers": texts[native["name"]], "termination": arm["answers"], "tokens": len(arm["steps"]), "write_error_relative_l2_min": min(errors) if errors else None, "write_error_relative_l2_max": max(errors) if errors else None, "write_error_relative_l2_mean": float(np.mean(errors)) if errors else None, "boundaries": comparisons, "local_probes": local})
    for path in all_primary + all_native + all_replays:
        for row in full_boundaries(path).values():
            file = f"{path['name']}-s{row['step']:03}-logits.f32"
            values = array(raw, file, VOCAB)
            if sha(raw/file) != row["readout"]["sha256"] or int(np.argmax(values)) != row["top"] or row["top"] != row["readout"]["top"]:
                raise ValueError("Full readout identity/argmax differs")
    probes = list(raw.glob("local-replay-*.json"))
    if len(probes) != 12:
        raise ValueError("Incomplete same-cache probe geometry")
    for file in probes:
        q = json.loads(file.read_text())
        if len(q["steps"]) != 1 or q["steps"][0]["mode"] != 0:
            raise ValueError("Native local probe did not execute")
        row = q["steps"][0]
        logits = file.stem+"-s000-logits.f32"
        if sha(raw/logits) != row["readout"]["sha256"] or int(np.argmax(array(raw, logits, VOCAB))) != row["top"]:
            raise ValueError("Local readout differs")
    if sum(len(p["steps"]) for p in all_primary) != driver["work"]["primary_tokens"] or sum(len(p["steps"]) for p in all_native) != driver["work"]["reference_tokens"] or sum(len(p["steps"]) for p in all_replays) != driver["work"]["replay_tokens"] or driver["work"]["local_native_tokens"] != len(probes) or driver["work"]["saved_exact_swiglu_down_diagnostics"] != polynomial_arrays:
        raise ValueError("Executed work ledger differs")
    work = {"primary": totals(all_primary), "native_own_history": totals(all_native), "postseal_replays": totals(all_replays), "driver": driver["work"], "notes": "Each transition also pays native attention, norm, RoPE, all remaining dense weights and full readout. MAC counts omit elementwise work and source coefficient compilation; diagnostic native steps and saved SiLU/down calls are separate."}
    return {"summary": {"primary_paths": len(all_primary), "native_paths": len(all_native), "two_turn_answers": len(all_primary)*2, "primary_files_sealed": len(seal["files"]), "candidate_replay_arrays_byte_exact": replay_arrays, "polynomial_products_independently_recomputed": polynomial_arrays, "full_candidate_readouts": readouts, "scope": "two development contexts, source-only gate law; original dense state/weights retained"}, "arms": summaries, "work": work, "observations": observations}
