"""Authenticate the executed zero-write paths against presealed token parents."""
from pathlib import Path
import hashlib
import numpy as np
from analysis import validate_token_chain


def analyze(raw, bank, driver, texts, reference):
    raw = Path(raw)
    previous = {a["name"]: a for a in reference["driver"]["arms"]}
    if driver["model_digest"] != reference["driver"]["model_digest"] or len(driver["arms"]) != 3:
        raise ValueError("Null-control specimen differs")
    if {a["name"] for a in driver["arms"]} != {"recall-zero", "math-zero", "prose-zero"}:
        raise ValueError("Null-control task set differs")
    answers = []
    observations = []
    for arm in driver["arms"]:
        if arm["policy"] != 4 or arm["name"] != arm["task"]+"-zero" or not arm["prefix_intact"]:
            raise ValueError("Zero policy differs")
        validate_token_chain(arm)
        native = previous[arm["task"]+"-native"]
        if (arm["parent_position"], arm["parent_kv_sha256"], arm["steps"][0]["input_id"], arm["budget"]) != (native["parent_position"], native["parent_kv_sha256"], native["steps"][0]["input_id"], native["budget"]):
            raise ValueError("Target token parent differs")
        for j, step in enumerate(arm["steps"]):
            if step["step"] != j or step["position"] != arm["parent_position"]+j or not step["substitution"] or any(step[k] for k in ("gate_up_calls", "down_calls", "compile_calls")):
                raise ValueError("Zero execution cost/clock differs")
            stem = f"{arm['name']}-s{j:03d}"
            values = {}
            for port, n in (("input", 1536), ("raw_receiver", 1536), ("mlp_input", 1536), ("mlp_return", 1536), ("output", 1536), ("kv16-17", 1024)):
                a = np.fromfile(raw/f"{stem}-{port}.f32", dtype="<f4")
                if a.shape != (n,) or not np.isfinite(a).all():raise ValueError("Invalid zero-control array")
                values[port] = a
            if values["mlp_return"].tobytes() != np.zeros(1536, dtype="<f4").tobytes():
                raise ValueError("Nonzero write consumed")
            if values["output"].tobytes() != np.add(values["raw_receiver"], values["mlp_return"]).tobytes():
                raise ValueError("Zero control changed the skip")
            if any((raw/f"{stem}-{p}.f32").exists() for p in ("mlp_gate", "mlp_up")):
                raise ValueError("Fabricated held-path projection ports")
            if not j:
                for port in ("input", "raw_receiver", "mlp_input"):
                    expected = reference["raw_files"][f"{native['name']}-s000-{port}.f32"]["sha256"]
                    if hashlib.sha256(values[port].tobytes()).hexdigest() != expected:
                        raise ValueError("Native first operand differs")
            observations.append({"kind": "observation", "case": arm["name"], "step": j, "metrics": {"zero_write_exact": True, "gate_up_calls": 0, "down_calls": 0}})
        for stem, port, record in ((f"{arm['name']}-s000", "first", arm["steps"][0]["first"]), (arm["name"], "final", arm["final"])):
            data = (raw/f"{stem}-{port}.f32").read_bytes();values = np.frombuffer(data, dtype="<f4")
            if values.shape != (151936,) or not np.isfinite(values).all() or hashlib.sha256(data).hexdigest() != record["sha256"] or int(np.argmax(values)) != record["top"]:
                raise ValueError("Zero readout seal differs")
        text = texts[arm["name"]].strip()
        answers.append({"name": arm["name"], "text": texts[arm["name"]], "task_correct": text.lower() == "blue" if arm["task"] == "recall" else text == "391" if arm["task"] == "math" else None, "emitted_tokens": len(arm["own"]["ids"]), "stopped": arm["own"]["stopped"], "truncated": arm["own"]["truncated"]})
    return {"summary": {"arms": 3, "answers": answers, "selected_gate_up_calls": 0, "selected_down_calls": 0, "zero_writes": sum(len(a["steps"]) for a in driver["arms"])}, "observations": observations}
