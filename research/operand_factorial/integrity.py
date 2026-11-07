"""Model-free mechanics joins for the executed gate/up causal instrument."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(raw, payload, driver):
    raw, payload = Path(raw), Path(payload)
    inputs = json.loads((payload / "inputs/input-manifest.json").read_text())
    assert sha(payload / "predecessor/report.json") == inputs["predecessor_report_sha256"]
    assert sha(raw / "input-manifest.json") == driver["input_manifest_sha256"]
    assert (raw / "input-manifest.json").read_bytes() == (payload / "inputs/input-manifest.json").read_bytes()
    assert driver["model_digest"] == inputs["model"]["content_sha256"]
    assert driver["layer"] == 16 and driver["rank"] == 64 and driver["capacity"] == 512
    assert driver["teacher_forced"] is True
    assert driver["future_policy"] == "native-up-to-two-supplied-history-IDs"
    assert sha(payload / "inputs/cuts.tsv") == driver["cuts_sha256"]
    for cid in range(3):
        assert sha(payload / "inputs" / f"history-case{cid}.txt") == driver["history_sha256"][cid]
    for record in inputs["basis_files"].values():
        assert sha(payload / "bank" / record["file"]) == record["sha256"]
    for driver_key, input_key in (("source_qt", "q_t"), ("source_gq", "gate"), ("source_uq", "up")):
        assert driver["basis_files"][driver_key]["sha256"] == inputs["basis_files"][input_key]["sha256"]
    assert sha(payload / "bank/bank_manifest.json") == driver["bank_manifest_sha256"]
    for name, record in inputs["imported_raw_files"].items():
        p = payload / "predecessor/raw" / name
        assert sha(p) == record["sha256"] and p.stat().st_size == record["bytes"]

    comparisons = []
    arrays_verified = set()
    def array(name, count):
        p = raw / "arrays" / name
        b = p.read_bytes()
        assert len(b) == 4*count, (name, len(b), count)
        assert np.isfinite(np.frombuffer(b, dtype="<f4")).all(), name
        arrays_verified.add(name)
        return b

    def equal_bytes(actual, expected, label):
        assert actual == expected, label
        comparisons.append({"kind": label, "byte_exact": True})

    def work(step, mode):
        w = step["work"]
        expected = {
            "y00": (0, 0, 1, 1, 1, 1),
            "y10": (1, 0, 1, 1, 0, 1),
            "y01": (0, 1, 1, 1, 1, 0),
            "y11": (1, 1, 1, 0, 0, 0),
            "zero": (0, 0, 0, 0, 0, 0),
            "native": (1, 1, 1, 0, 0, 0),
        }[mode]
        keys = ("native_gate_calls", "native_up_calls", "native_down_calls", "basis_q_calls", "basis_g_calls", "basis_u_calls")
        assert tuple(w[k] for k in keys) == expected, (mode, w)
        assert w["selected_gate_up_calls"] == expected[0]+expected[1]
        assert w["selected_down_calls"] == expected[2]

    cuts = driver["cuts"]
    assert len(cuts) == len(inputs["cuts"])
    refs = {(c["case"], c["cut_index"]): c["h_native_same_id"] for c in driver["references"]}
    assert len(refs) == len(cuts)
    old_boundaries, all_parent_checks, horizon_checks = 0, 0, 0
    for expected, cut in zip(inputs["cuts"], cuts):
        for key in ("case", "position", "clock", "input_id", "retrospective", "correct_id", "wrong_id", "index"):
            assert cut[key] == expected[key], (key, cut, expected)
        assert cut["parent_history_ids"] == expected["history_ids"][:-1]
        assert cut["continuation_ids"] == expected["continuation_ids"]
        branches = {b["current_mode"]: b for b in cut["branches"]}
        assert set(branches) == {"y00", "y10", "y01", "y11", "zero"}
        assert all(b["parent_mode"] == "y00" for b in branches.values())
        parents = {b["parent_prefix_kv_sha256"] for b in branches.values()}
        assert len(parents) == 1 and len(next(iter(parents))) == 64
        all_parent_checks += 1
        h = refs[(cut["case"], cut["position"])]
        assert h["parent_mode"] == "native" and h["current_mode"] == "native"
        worlds = {**branches, "native": h}
        for mode, branch in worlds.items():
            assert branch["current_position"] == cut["position"] and branch["current_token"] == cut["input_id"]
            current = branch["current"]
            assert current["position"] == cut["position"] and current["token"] == cut["input_id"]
            work(current, mode)
            assert len(branch["future"]) == len(expected["continuation_ids"])
            previous = current
            for index, step in enumerate(branch["future"]):
                assert step["position"] == cut["position"]+index+1
                assert step["token"] == expected["continuation_ids"][index]
                work(step, "native")
                assert step["prefix_kv_sha256"] == previous["full_kv_sha256"]
                previous = step
                horizon_checks += 1
            for step in [current, *branch["future"]]:
                logits = array(step["readout"]["file"], 151936)
                assert hashlib.sha256(logits).hexdigest() == step["readout"]["sha256"]
                kv = array(step["current_kv_row_file"], 2*28*2*128)
                assert hashlib.sha256(kv).hexdigest() == step["current_kv_row_sha256"]
                for suffix in ("raw_receiver", "mlp_input", "mlp_return", "output"):
                    array(step["stem"]+"-"+suffix+".f32", 1536)
                array(step["stem"]+"-mlp_product.f32", 8960)

        # This model's first changed operation is MLP16: same text must retain
        # its raw receiver/norm even when the upper cache was formed differently.
        for horizon in range(1+len(expected["continuation_ids"])):
            states = {mode: b["current"] if horizon == 0 else b["future"][horizon-1] for mode,b in worlds.items()}
            for suffix in ("raw_receiver", "mlp_input"):
                values = [array(s["stem"]+"-"+suffix+".f32", 1536) for s in states.values()]
                assert len(set(values)) == 1, (cut["case"],cut["position"],horizon,suffix)
                comparisons.append({"kind": "same-ID-"+suffix, "cut": expected["id"], "horizon": horizon, "byte_exact": True})
        y00, y11 = branches["y00"]["current"], branches["y11"]["current"]
        for suffix, old_name in expected["expected_source_files"].items():
            equal_bytes(array(y00["stem"]+"-"+suffix+".f32",1536),
                        (payload/"predecessor/raw"/old_name).read_bytes(), "historical-Y00-"+suffix)
        equal_bytes(array(y11["stem"]+"-mlp_return.f32",1536),
                    (payload/"predecessor/raw"/expected["expected_native_write_file"]).read_bytes(), "historical-Y11-native-write")
        equal_bytes(array(y11["stem"]+"-mlp_return.f32",1536),
                    array(h["current"]["stem"]+"-mlp_return.f32",1536), "Y11-H-current-write")
        if not expected["retrospective"]:
            for state, key in ((y00, "historical_source_boundary"), (h["current"], "historical_native_sameid_boundary")):
                boundary = expected[key]
                equal_bytes(array(state["readout"]["file"],151936),
                            (payload/"predecessor/raw"/boundary["logits"]["file"]).read_bytes(), key+"-logits")
                assert state["full_kv_sha256"] == boundary["valid_kv_sha256"], key
                old_boundaries += 1
        else:
            assert y00["readout"]["top"] == 20
            assert expected["correct_id"] == 21 and expected["wrong_id"] == 20
        for mode in ("y00", "y10", "y01", "y11"):
            state = branches[mode]["current"]
            for suffix in ("mlp_gate", "mlp_up"):
                array(state["stem"]+"-"+suffix+".f32",8960)
        interaction = cut["interaction"]
        array(interaction["product_file"],8960)
        array(interaction["write_file"],1536)
        array(interaction["four_write_file"],1536)

    # Hash the primary array seal before the native H reference was appended.
    seal = json.loads((raw/"arrays/primary-seal.json").read_text())
    for record in seal["files"]:
        p = raw/"arrays"/record["name"]
        assert sha(p) == record["sha256"] and p.stat().st_size == record["bytes"]
    driver_bytes = (raw/"driver.json").read_bytes()
    prefix = driver_bytes[:driver["primary_seal"]["bytes"]]
    assert hashlib.sha256(prefix).hexdigest() == driver["primary_seal"]["sha256"]
    assert driver["primary_seal"]["array_seal_timing"] == "before-H"
    assert driver["primary_seal"]["metadata_digest_timing"] == "after-H"
    prefix_steps = sum(c["position"] for c in inputs["cuts"])
    future_steps = sum(len(c["continuation_ids"]) for c in inputs["cuts"])
    n = len(cuts)
    expected_work = {
        "primary_prefix_steps": prefix_steps,
        "primary_replay_steps": prefix_steps+5*(n+future_steps),
        "native_reference_replay_steps": prefix_steps+n+future_steps,
        "checkpoint_restores": 5*n,
        "diagnostic_down_calls": n,
        "aggregate_native_gate_calls": prefix_steps+3*n+6*future_steps,
        "aggregate_native_up_calls": prefix_steps+3*n+6*future_steps,
        "aggregate_native_down_calls": 2*prefix_steps+5*n+6*future_steps,
        "aggregate_basis_q_calls": prefix_steps+3*n,
        "aggregate_basis_g_calls": prefix_steps+2*n,
        "aggregate_basis_u_calls": prefix_steps+2*n,
    }
    for key, value in expected_work.items():
        assert driver["work"][key] == value, (key, driver["work"].get(key), value)
    return {"schema": "tcm-operand-factorial-integrity-v1", "mechanics_status": "complete",
            "instrument_status": "complete", "cuts": len(cuts), "same_parent_groups": all_parent_checks,
            "historical_full_boundaries_exact": old_boundaries,
            "literal_native_future_steps": horizon_checks, "arrays_shape_finite_hash_checked": len(arrays_verified),
            "primary_files_sealed": len(seal["files"]), "comparisons": comparisons,
            "physical_work_verified": expected_work,
            "scientific_quality_gate": "none", "models_evaluated": 0}
