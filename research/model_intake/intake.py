"""Offline source/IR intake using canonical mrun and manalysis identities.

This research command performs no forward, tensor decoding, C lowering or
semantic certification. Numerical serving has no dependency on this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


SCHEMA = "cmachine-model-intake-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sealed(value):
    return {**value, "fingerprint": hashlib.sha256(canonical(value)).hexdigest()}


def write_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def inventory(bundle):
    """Describe executable declarations; do not infer a C target from names."""
    from manalysis.decompiler_bridge import build_exact_graph

    graph = build_exact_graph(bundle)
    operation_refs = {
        node["operation_id"]: node["ref"]
        for node in graph["nodes"]
        if node["kind"] == "operation"
    }
    kinds = Counter(operation.kind for operation in bundle.model.operations)
    return {
        "architecture_id": bundle.model.architecture_id,
        "operation_kinds": dict(sorted(kinds.items())),
        "operation_count": len(bundle.model.operations),
        "parameter_count": len(bundle.model.parameters),
        "state_slot_count": len(bundle.state.slots),
        "numerical_semantics": bundle.model.numerical_semantics.as_dict(),
        "operations": [
            {"operation_id": operation.operation_id,
             "exact_ref": operation_refs[operation.operation_id],
             "kind": operation.kind,
             "inputs": list(operation.inputs), "outputs": list(operation.outputs),
             "parameters": list(operation.parameters),
             "attributes": operation.attributes,
             "c_target_binding": "not-qualified-by-this-intake"}
            for operation in bundle.model.operations
        ],
        "state_contract": bundle.state.as_dict(),
        "io_contract": bundle.io.as_dict(),
    }


def _markdown(report):
    lines = ["# Model intake", "", f"Status: {report['status']}.", "",
             "This is an offline structural import. No model forward, C execution, numerical parity or semantic discovery was performed.", ""]
    if report.get("inventory") is not None:
        data = report["inventory"]
        lines += [f"Architecture: `{data['architecture_id']}`.", "",
                  f"Declared operations: {data['operation_count']}; parameters: {data['parameter_count']}; state slots: {data['state_slot_count']}.", "",
                  "| Declared operation kind | Count |", "| --- | ---: |"]
        lines += [f"| `{kind}` | {count} |" for kind, count in data["operation_kinds"].items()]
        lines += ["", "The exact graph, coordinated state/IO contract and separate hypothesis-only role priors are retained beside this report. A source map must join future observations to these identities before a C target can be qualified."]
    else:
        lines += ["The canonical decompiler did not emit coordinated IR. Its refusal and source evidence are retained; no substitute graph was guessed."]
    lines += ["", "Next work: bind a supported C target; compare actual source/C boundaries; qualify stateful continuation; then run consumer-based discovery. Unknown semantic roles do not block an otherwise qualified executor.", ""]
    return "\n".join(lines)


def intake(*, output, source=None, ir=None, source_id=None, revision=None):
    """Create a new evidence directory, preserving the canonical input boundary."""
    if (source is None) == (ir is None):
        raise ValueError("provide exactly one of source or ir")
    from manalysis.decompiler_bridge import import_ir_bundle, load_decompiler_artifact

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite intake output: {destination}")
    frozen_source = None
    result = None
    loaded = None
    if source is not None:
        source_path = Path(source).resolve()
        if destination == source_path or source_path in destination.parents:
            raise ValueError("intake output must be outside the frozen source")
        from mrun.decompiler import decompile_source

        result = decompile_source(source_path, source_id=source_id, resolved_revision=revision)
        payload = result.as_dict()
        bundle = result.ir_bundle
        frozen_source = result.source
        validation_scope = "decompile-result-source-index-report"
    else:
        loaded = load_decompiler_artifact(ir)
        bundle = loaded.bundle
        payload = bundle.as_dict()
        # Only normalized IR is retained on this path. Do not claim retained
        # source custody merely because an incoming wrapper was validated.
        validation_scope = "coordinated-ir-bundle"

    # Validate/project before creating output, so malformed IR cannot leave a report.
    data = inventory(bundle) if bundle is not None else None
    destination.mkdir(parents=True, exist_ok=False)
    try:
        input_path = destination / "input.json"
        write_json(input_path, payload)
        artifacts = {"input": "input.json"}
        if bundle is not None:
            imported = import_ir_bundle(input_path, destination / "workspace")
            artifacts.update({name: str(Path(path).relative_to(destination))
                              for name, path in imported["artifacts"].items()})
            imported["artifacts"] = {name: artifacts[name] for name in imported["artifacts"]}
        else:
            imported = None
        if frozen_source is not None:
            frozen_source.assert_unchanged()
        report = sealed({
            "schema_version": SCHEMA,
            "status": "structurally-imported" if bundle is not None else result.report.status,
            "source_validation_scope": validation_scope,
            "input_fingerprint": hashlib.sha256(canonical(payload)).hexdigest(),
            "bundle_fingerprint": bundle.fingerprint if bundle is not None else None,
            "inventory": data,
            "workspace": imported,
            "source_report": result.report.as_dict() if result is not None else None,
            "authority": {
                "mechanics_status": "canonical-ir-validated" if bundle is not None else "source-not-decoded",
                "instrument_status": "no-numerical-instrument-run",
                "trend_status": "not-measured",
                "terminal_status": "no-execution-or-semantic-claim",
                "c_execution_qualified": False,
                "semantic_claim": False,
                "role_priors": "hypothesis-only",
                "framework_fallback_executed": False,
            },
            "artifacts": artifacts,
        })
        with (destination / "REPORT.md").open("x") as handle:
            handle.write(_markdown(report))
        # Published last: its presence is the complete-intake marker.
        write_json(destination / "intake.json", report)
        return report
    except BaseException:
        shutil.rmtree(destination)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--source", type=Path, help="local declarative checkpoint directory")
    origin.add_argument("--ir", type=Path, help="canonical mrun IR or source report JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--revision")
    args = parser.parse_args(argv)
    try:
        report = intake(output=args.output, source=args.source, ir=args.ir,
                        source_id=args.source_id, revision=args.revision)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"intake refused: {exc}\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "bundle_fingerprint": report["bundle_fingerprint"],
                      "c_execution_qualified": False, "semantic_claim": False}))
    return 0 if report["bundle_fingerprint"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
