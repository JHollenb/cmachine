"""Verify pinned public records and build a small, model-free result gallery."""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    pins = json.loads((HERE / "sources.json").read_text())
    for name, expected in pins["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Recorded source changed: " + name)

    def load(name):
        if name not in pins["files"]:
            raise ValueError("Unpinned source: " + name)
        return json.loads((ROOT / name).read_text())

    base = "docs/evidence/integration-7a6824d/"
    report = load(base + "report.json")
    native = load(base + "raw/native-protocol.json")
    reference = load(base + "raw/reference-native-protocol.json")
    if native != reference:
        raise ValueError("Historical native/reference protocol differs")
    for name in pins["files"]:
        if name.startswith(base + "raw/"):
            row = report["raw_files"][name.split("/raw/", 1)[1]]
            if row["sha256"] != pins["files"][name] or row["bytes"] != (ROOT / name).stat().st_size:
                raise ValueError("Report/raw join differs: " + name)
    summary = report["analysis"]["summary"]
    clients = {row["name"]: row for row in summary["clients"]}
    protocols = {row["name"]: row for row in summary["protocol"]}
    memory = []
    truth = ["17", "29", "amber", "46"]
    for policy in ("full-byte", "bounded"):
        answers = [a.strip() for a in clients[policy + "-delayed"]["answers"][-4:]]
        memory.append({"policy": policy, "answers": answers, "expected": truth,
                       "correct": [a == b for a, b in zip(answers, truth)],
                       "tracked_state_bytes": protocols[policy]["state_gpu_bytes"],
                       "tracked_model_and_state_bytes": protocols[policy]["model_and_state_gpu_bytes"],
                       "exact_replay": protocols[policy]["exact_replay"]})

    context = "docs/evidence/context-query-20261106/"
    integration = load(context + "integration.json")
    query = load(context + "analysis.json")
    case_rows = []
    for case in query["fresh"]:
        arms = {row["arm"]: row for row in case["arms"]}
        case_rows.append({"context_tokens": case["context_tokens"], "kind": case["kind"],
                          "baseline_kl": arms[1]["future_kl"],
                          "updated_kl": arms[2]["future_kl"],
                          "later_layer_kl": arms[3]["future_kl"],
                          "updated_answers": arms[2]["texts"],
                          "later_layer_answers": arms[3]["texts"]})
    means = query["fresh_arm_mean_kl"]
    trace = []
    for entry in native:
        reply = entry["reply"]
        if reply.get("event") in {"state", "pushed", "rolled_back", "committed", "generated"}:
            trace.append({"command": entry["command"], "reply": reply})
    return {
        "schema": "cmachine-recorded-demo-v1",
        "execution_kind": "verified-historical-pretrained-records",
        "models_executed_by_this_command": 0,
        "source_files_verified": len(pins["files"]),
        "source_manifest_sha256": digest(HERE / "sources.json"),
        "jobs": ["job-f645fcd76a75", "job-fe51e7f4f0ee"],
        "model": {key: report["model"][key] for key in ("identity", "content_sha256", "profile_version")},
        "chat": {"prompt": "Explain why bread rises when yeast is added. Use two short sentences.",
                 "answer": clients["native-demo"]["answers"][0].strip(),
                 "native_reference_protocol_entries_exact": len(native),
                 "native_mechanics": summary["real_mechanics"]["native"], "trace": trace,
                 "dense_weight_bytes": native[0]["reply"]["gpu_weight_bytes"]},
        "operands": {"lexical": integration, "fresh_case_count": len(case_rows),
                     "fresh_baseline_mean_kl": means[1], "fresh_updated_mean_kl": means[2],
                     "relative_kl_reduction": 1 - means[2] / means[1], "cases": case_rows,
                     "scope": "Two task families at two lengths; policies visit their own histories. Most original work remains."},
        "memory": {"rows": memory, "bounded_physical_rows": 112, "logical_capacity": 768,
                   "state_saving_vs_full_byte": 1 - memory[1]["tracked_state_bytes"] / memory[0]["tracked_state_bytes"],
                   "bounded_recovery": summary["real_mechanics"]["bounded"],
                   "scope": "Tracked GPU allocations, including recovery; host transcript and CUDA overhead additional. Both policies retain all learned maps."},
        "provenance": pins,
        "scope": "Pinned records checked and report/raw joins verified. This command reruns no pretrained model and is not a new experimental reproduction."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Record verified: {result['source_files_verified']} pinned sources; {result['chat']['native_reference_protocol_entries_exact']} exact protocol entries.")
    print(f"Recorded lexical program: {result['operands']['lexical']['lexical_replayed_steps']} exact full-logit steps.")
    print(f"Recorded own-history query improvement: {100 * result['operands']['relative_kl_reduction']:.2f}% mean conditional KL.")
    print("Pretrained models executed by this command: 0.")
    print("Saved " + str(args.output))


if __name__ == "__main__":
    main()
