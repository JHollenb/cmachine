"""Project the sealed operand-basis panel onto Saturn's evidence plane.

The native panel and its raw arrays remain the authority.  This adapter only
builds a target-local, scalar JSONL stream with the report and custody hashes
bound to every row, plus a separate expanded stream for the standalone
same-ID receiver/norm probes.  It does not load a model, recompute a forward,
copy vectors, or turn any observed error into a scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]
DEFAULT_JOB = "job-51df6b792d28"
OBSERVATION_SCHEMA = "tcm-operand-basis-evidence-observation-v1"
SOURCE_SCHEMA = "tcm-operand-basis-evidence-source-v1"
PROBE_SCHEMA = "tcm-operand-basis-standalone-probe-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _artifact(root: Path, relative: str, role: str) -> dict[str, Any]:
    path = root / relative
    _require(path.is_file(), f"missing custody artifact: {relative}")
    return {
        "role": role,
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _scalar_tree(value: Any) -> Any:
    """Retain scalar metadata while refusing accidental vector projection."""

    if isinstance(value, Mapping):
        return {str(key): _scalar_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # The projected rows should remain scalar.  Small integer ID lists are
        # useful authority metadata, but full token/vector payloads belong in
        # the immutable report and are deliberately omitted here.
        if len(value) <= 16 and all(isinstance(item, (bool, int, str)) for item in value):
            return list(value)
        return {"omitted": True, "count": len(value)}
    return value


def _base_row(
    report_sha: str,
    model: Mapping[str, Any],
    observable: str,
    stream: str,
    case: Any,
) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "kind": "observation",
        "observable": observable,
        "stream": stream,
        "case": case,
        "model_id": model.get("identity"),
        "model_content_sha256": model.get("content_sha256"),
        "report_sha256": report_sha,
    }


def _identity_joins(job: str, root: Path, report: Mapping[str, Any], verification: Mapping[str, Any], submission: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck byte/hash joins needed to interpret the scalar projection.

    The job's existing verifier remains authoritative for the full execution
    contract.  These joins pin the projection to that verified report and make
    any relocation or stale-array problem visible without rerunning the
    verifier or invoking a model.
    """

    report_path = root / "report.json"
    verification_path = root / "verification.json"
    job_path = root / "job.json"
    submission_path = root / "submission.json"
    report_sha = _sha256(report_path)
    verification_sha = _sha256(verification_path)
    _require(_load(job_path).get("job_id") == job, "scheduler job identity mismatch")
    _require(_load(submission_path).get("job_id") == job, "submission job identity mismatch")
    _require(verification.get("report_sha256") == report_sha, "verification/report hash mismatch")
    _require(report.get("schema") == "tcm-operand-basis-experiment-v1", "unexpected report schema")
    _require(verification.get("analysis_recomputed") is True, "analysis was not recomputed")
    _require(verification.get("whole_invocation_receipt") is True, "whole invocation receipt missing")
    _require(verification.get("models_evaluated") == 0, "evidence adapter cannot evaluate models")

    raw = root / "raw"
    raw_manifest = report.get("raw_files", {})
    _require(isinstance(raw_manifest, Mapping), "report raw file manifest missing")
    actual_raw = {path.relative_to(raw).as_posix() for path in raw.rglob("*") if path.is_file()}
    expected_raw = set(str(name) for name in raw_manifest)
    _require(expected_raw <= actual_raw, "report references a missing raw artifact")
    # The worker keeps both an initial partial report and the final report in
    # raw/; neither is included in the sealed raw_files map.
    _require(actual_raw - expected_raw <= {"report.json", "partial-report.json"}, "unexpected raw artifact")
    raw_checked = 0
    for relative, expected in raw_manifest.items():
        path = raw / str(relative)
        _require(path.stat().st_size == expected.get("bytes"), f"raw extent mismatch: {relative}")
        _require(_sha256(path) == expected.get("sha256"), f"raw hash mismatch: {relative}")
        raw_checked += 1
    _require(_load(raw / "report.json") == report, "raw/report.json differs from collected report")
    _require(_load(raw / "driver.json") == report.get("driver"), "raw/driver.json differs from report driver")

    payload = root / "payload"
    payload_manifest_path = payload / "source-manifest.json"
    payload_manifest = _load(payload_manifest_path)
    config = _load(submission_path).get("config", {})
    source_map = payload_manifest.get("sources", {})
    _require(source_map == config.get("sources"), "payload/submission source map mismatch")
    source_manifest_sha = _sha256(payload_manifest_path)
    _require(config.get("source_manifest_sha256") == source_manifest_sha, "source manifest hash mismatch")
    source_checked = 0
    for relative, expected in source_map.items():
        path = payload / str(relative)
        _require(path.is_file(), f"missing payload source: {relative}")
        _require(_sha256(path) == expected, f"payload source hash mismatch: {relative}")
        source_checked += 1

    model = report.get("model", {})
    driver = report.get("driver", {})
    receipt_wrapper = report.get("saturn_receipt", {})
    receipt = dict(receipt_wrapper.get("receipt", {}))
    receipt_fingerprint = receipt.pop("fingerprint", None)
    _require(receipt_fingerprint is not None, "receipt fingerprint missing")
    receipt_fingerprint_observed = _canonical_sha256(receipt)
    _require(receipt_fingerprint_observed == receipt_fingerprint, "receipt fingerprint mismatch")
    request = receipt_wrapper.get("request", {})
    _require(request.get("model_digest") == model.get("content_sha256") == driver.get("model_digest"), "model digest join mismatch")
    _require(request.get("binary_sha256") == report.get("binary_sha256"), "binary receipt join mismatch")
    input_manifest_path = raw / "input-manifest.json"
    input_manifest_sha = _sha256(input_manifest_path)
    _require(input_manifest_sha == report.get("input_manifest_sha256") == request.get("input_manifest_sha256"), "input manifest hash mismatch")
    _require(request.get("inputs") == report.get("inputs"), "receipt/request input identity mismatch")
    _require(request.get("source_manifest_sha256") == source_manifest_sha, "receipt/source manifest join mismatch")

    binary_path = raw / "binary" / "operand-chat"
    _require(_sha256(binary_path) == report.get("binary_sha256"), "executed binary hash mismatch")
    compiled = raw / "compiled"
    bank_manifest_path = compiled / "bank_manifest.json"
    _require(_load(bank_manifest_path) == report.get("compiler"), "compiled bank manifest differs from report")
    basis_expected = driver.get("basis_file_sha256", driver.get("basis_sha256", {}))
    basis_files = {
        "source_qt": "source-QT.f32",
        "source_gq": "source-GQ.f32",
        "source_uq": "source-UQ.f32",
        "random_qt": "random-QT.f32",
        "random_gq": "random-GQ.f32",
        "random_uq": "random-UQ.f32",
    }
    basis_joins = {}
    for key, filename in basis_files.items():
        observed = _sha256(compiled / filename)
        expected = basis_expected.get(key)
        basis_joins[key] = {"file": f"compiled/{filename}", "expected_sha256": expected, "observed_sha256": observed, "match": expected == observed}
        _require(expected == observed, f"compiled basis hash mismatch: {filename}")
    source_export = report.get("source_export", {})
    source_input_manifest = _load(compiled / "source-input-manifest.json")
    source_joins = {}
    for key, compiled_name in (("G", "gate"), ("U", "up"), ("gamma", "gamma")):
        exported = source_export.get(key, {})
        source_record = source_input_manifest.get(key, {})
        compiler_record = report.get("compiler", {}).get("source", {}).get("weights", {}).get(compiled_name, {})
        source_joins[key] = {
            "input_sha256": source_record.get("sha256"),
            "report_sha256": exported.get("sha256"),
            "compiler_sha256": compiler_record.get("sha256"),
            "input_report_match": source_record.get("sha256") == exported.get("sha256"),
            "report_compiler_match": exported.get("sha256") == compiler_record.get("sha256"),
        }
        _require(source_joins[key]["input_report_match"], f"source export/input mismatch: {key}")
        _require(source_joins[key]["report_compiler_match"], f"source export/compiler mismatch: {key}")

    raw_observations_path = root / "observations.jsonl"
    native_observation_rows = []
    with raw_observations_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid native observation JSONL at line {line_number}") from exc
            native_observation_rows.append(row)
    _require(native_observation_rows, "native observation stream is empty")
    _require(all(row.get("report_sha256") == report_sha for row in native_observation_rows), "native observation/report hash mismatch")

    return {
        "report_sha256": report_sha,
        "raw_report_sha256": _sha256(raw / "report.json"),
        "verification_sha256": verification_sha,
        "submission_sha256": _sha256(submission_path),
        "source_manifest_sha256": source_manifest_sha,
        "input_manifest_sha256": input_manifest_sha,
        "binary_sha256": _sha256(binary_path),
        "compiled_manifest_sha256": _sha256(bank_manifest_path),
        "raw_files_manifest_sha256": _canonical_sha256(raw_manifest),
        "raw_files_verified": raw_checked,
        "source_files_verified": source_checked,
        "native_observations_sha256": _sha256(raw_observations_path),
        "native_observation_rows": len(native_observation_rows),
        "receipt_fingerprint": receipt_fingerprint,
        "receipt_fingerprint_verified": True,
        "model": {
            "report_identity": model.get("identity"),
            "report_content_sha256": model.get("content_sha256"),
            "driver_model_digest": driver.get("model_digest"),
            "receipt_model_identity": receipt.get("model_identity"),
        },
        "basis_file_joins": basis_joins,
        "source_weight_joins": source_joins,
        "analysis_integrity": report.get("analysis", {}).get("integrity", {}),
    }


def _readout_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = analysis.get("readout_comparisons", {})
    for relation in sorted(comparisons):
        for item in comparisons.get(relation, []):
            row = _base_row(report_sha, model, "full_vocab_readout", "readout-comparisons", item.get("case"))
            row.update(
                {
                    "relation": relation,
                    "boundary": item.get("boundary"),
                    "left_path": item.get("left_path"),
                    "right_path": item.get("right_path"),
                    "left_top": item.get("left_top"),
                    "right_top": item.get("right_top"),
                    "relative_l2_denominator": item.get("relative_l2_denominator"),
                    "left_readout": _scalar_tree(item.get("left_readout", {})),
                    "right_readout": _scalar_tree(item.get("right_readout", {})),
                    "difference": _scalar_tree(item.get("difference", {})),
                }
            )
            rows.append(row)
    return rows


def _local_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in analysis.get("local_writes", []):
        row = _base_row(report_sha, model, "mlp_local_write", "primary-local-writes", item.get("case"))
        row.update(
            {
                "arm": item.get("arm"),
                "path": item.get("path"),
                "step": item.get("step"),
                "position": item.get("position"),
                "prefix_sha256": item.get("prefix_sha256"),
                "stem": item.get("stem"),
                "candidate_write_file": item.get("candidate_write_file"),
                "native_write_file": item.get("native_write_file"),
                "native_write_reference_available": item.get("native_write_reference_available"),
                "raw_receiver": _scalar_tree(item.get("raw_receiver", {})),
                "normalized_input": _scalar_tree(item.get("normalized_input", {})),
                "candidate_write": _scalar_tree(item.get("candidate_write", {})),
                "native_write": _scalar_tree(item.get("native_write", {})),
                "write_error": _scalar_tree(item.get("write_error", {})),
                "receiver_norm_l2": item.get("receiver_norm_l2"),
                "normalized_input_norm_l2": item.get("normalized_input_norm_l2"),
            }
        )
        rows.append(row)
    return rows


def _call_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.get("observations", []):
        if item.get("kind") != "call_ledger":
            continue
        row = _base_row(report_sha, model, "executed_call_ledger", "call-ledger", item.get("case"))
        row.update({key: _scalar_tree(value) for key, value in item.items() if key != "kind"})
        rows.append(row)
    return rows


def _token_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.get("token_authority", []):
        row = _base_row(report_sha, model, "token_authority", "token-authority", item.get("case"))
        row.update(_scalar_tree(item))
        rows.append(row)
    return rows


def _common_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.get("common_history_raw_receiver", []):
        row = _base_row(report_sha, model, "common_history_receiver", "common-history", item.get("case"))
        comparisons = item.get("comparisons", [])
        row.update(
            {
                "status": item.get("status"),
                "groups_compared": item.get("groups_compared"),
                "pair_comparisons": len(comparisons),
                "byte_mismatch_count": item.get("byte_mismatch_count"),
                "missing_count": item.get("missing_count"),
                "grouping": item.get("grouping"),
            }
        )
        rows.append(row)
    return rows


def _same_id_summary_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.get("same_id_receiver_norm", []):
        comparisons = item.get("comparisons", [])
        receiver_mismatches = sum(not bool(entry.get("raw_receiver", {}).get("byte_exact", False)) for entry in comparisons)
        norm_mismatches = sum(not bool(entry.get("mlp_input", {}).get("byte_exact", False)) for entry in comparisons)
        row = _base_row(report_sha, model, "same_id_receiver_norm_summary", "same-id-summary", item.get("case"))
        row.update(
            {
                "candidate": item.get("candidate"),
                "native_same_id": item.get("native_same_id"),
                "status": item.get("status"),
                "comparison_count": len(comparisons),
                "receiver_byte_mismatch_count": receiver_mismatches,
                "norm_byte_mismatch_count": norm_mismatches,
                "grouping": item.get("grouping"),
            }
        )
        rows.append(row)
    return rows


def _standalone_probe_rows(report_sha: str, model: Mapping[str, Any], analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the 12 same-ID paths into their 1,128 scalar probe rows."""

    rows = []
    for item in analysis.get("same_id_receiver_norm", []):
        for comparison in item.get("comparisons", []):
            row = {
                "schema": PROBE_SCHEMA,
                "kind": "observation",
                "observable": "standalone_local_probe",
                "stream": "standalone-local-probes",
                "case": item.get("case"),
                "model_id": model.get("identity"),
                "model_content_sha256": model.get("content_sha256"),
                "report_sha256": report_sha,
                "candidate": item.get("candidate"),
                "native_same_id": item.get("native_same_id"),
                "candidate_step": comparison.get("candidate_step"),
                "native_same_id_step": comparison.get("native_same_id_step"),
                "prefix_length": comparison.get("prefix_length"),
                "prefix_sha256": comparison.get("prefix_sha256"),
                "raw_receiver": _scalar_tree(comparison.get("raw_receiver", {})),
                "mlp_input": _scalar_tree(comparison.get("mlp_input", {})),
                "raw_receiver_files": _scalar_tree(comparison.get("raw_receiver_files", {})),
                "mlp_input_files": _scalar_tree(comparison.get("mlp_input_files", {})),
            }
            rows.append(row)
    return rows


def _signed_trends(local_rows: list[Mapping[str, Any]], readout_rows: list[Mapping[str, Any]], call_rows: list[Mapping[str, Any]], probes: list[Mapping[str, Any]]) -> dict[str, Any]:
    write_errors = [row.get("write_error", {}) for row in local_rows]
    readout_differences = [row.get("difference", {}) for row in readout_rows]
    finite_write = [float(item["relative_l2"]) for item in write_errors if isinstance(item.get("relative_l2"), (int, float))]
    finite_readout = [float(item["relative_l2"]) for item in readout_differences if isinstance(item.get("relative_l2"), (int, float))]
    return {
        "primary_write_rows_with_status_not_ok": sum(item.get("status") != "ok" for item in write_errors),
        "primary_write_relative_l2_gt_one": sum(value > 1.0 for value in finite_write),
        "primary_write_relative_l2_min": min(finite_write) if finite_write else None,
        "primary_write_relative_l2_max": max(finite_write) if finite_write else None,
        "readout_rows_with_status_not_ok": sum(item.get("status") != "ok" for item in readout_differences),
        "readout_relative_l2_gt_zero": sum(value > 0.0 for value in finite_readout),
        "call_ledger_mismatch_rows": sum(item.get("status") == "mismatch" for item in call_rows),
        "standalone_probe_receiver_mismatch_rows": sum(not bool(row.get("raw_receiver", {}).get("byte_exact", False)) for row in probes),
        "standalone_probe_norm_mismatch_rows": sum(not bool(row.get("mlp_input", {}).get("byte_exact", False)) for row in probes),
        "interpretation": "signed observations and bad rows are retained; none of these counts is a scientific gate",
    }


def _run_evidence_pipeline(index: Path, target: Path, action: str) -> dict[str, Any]:
    args = [
        str(DOMAINS / "common/bin/common"),
        "exec",
        "saturn-evidence",
        action,
        str(target),
        "--watch-root",
        str(index / "watch"),
        "--registry",
        str(index / "claims.jsonl"),
        "--results-root",
        str(target),
        "--json",
    ]
    if action == "run":
        args.extend(("--xref-db", str(index / "xref.sqlite"), "--xref-root", str(target)))
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    (index / f"evidence-{action}.json").write_text(completed.stdout, encoding="utf-8")
    (index / f"evidence-{action}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    try:
        summary = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"saturn-evidence {action} emitted invalid JSON: {completed.stderr[-3000:]}") from exc
    if completed.returncode not in (0, 3):
        raise RuntimeError(f"saturn-evidence {action} refused input (exit {completed.returncode})")
    return {"returncode": completed.returncode, "summary": summary}


def project(job: str = DEFAULT_JOB) -> dict[str, Any]:
    root = HERE / "runs" / job
    index = HERE / "index" / job
    target = index / "target-streams"
    standalone = index / "standalone"
    target.mkdir(parents=True, exist_ok=True)
    standalone.mkdir(parents=True, exist_ok=True)

    report = _load(root / "report.json")
    verification = _load(root / "verification.json")
    submission = _load(root / "submission.json")
    custody = _identity_joins(job, root, report, verification, submission)
    model = report.get("model", {})
    analysis = report.get("analysis", {})
    report_sha = custody["report_sha256"]

    local_rows = _local_rows(report_sha, model, analysis)
    readout_rows = _readout_rows(report_sha, model, analysis)
    call_rows = _call_rows(report_sha, model, analysis)
    token_rows = _token_rows(report_sha, model, analysis)
    common_rows = _common_rows(report_sha, model, analysis)
    same_id_rows = _same_id_summary_rows(report_sha, model, analysis)
    probe_rows = _standalone_probe_rows(report_sha, model, analysis)
    main_rows = local_rows + readout_rows + call_rows + token_rows + common_rows + same_id_rows

    expected = {
        "primary_local_write": 1128,
        "readout_comparison": 144,
        "call_ledger": 12,
        "token_authority": 12,
        "common_history": 3,
        "same_id_summary": 12,
        "standalone_local_probe": 1128,
        "native_observations": 1311,
    }
    observed = {
        "primary_local_write": len(local_rows),
        "readout_comparison": len(readout_rows),
        "call_ledger": len(call_rows),
        "token_authority": len(token_rows),
        "common_history": len(common_rows),
        "same_id_summary": len(same_id_rows),
        "standalone_local_probe": len(probe_rows),
        "native_observations": custody["native_observation_rows"],
    }
    _require(observed == expected, f"evidence coverage mismatch: {observed} != {expected}")
    _require(len(main_rows) == expected["native_observations"], "main scalar coverage mismatch")

    observations_path = target / "observations.jsonl"
    observations_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in main_rows),
        encoding="utf-8",
    )
    probes_path = standalone / "local-probes.jsonl"
    probes_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in probe_rows),
        encoding="utf-8",
    )
    observations_sha = _sha256(observations_path)
    probes_sha = _sha256(probes_path)

    trends = _signed_trends(local_rows, readout_rows, call_rows, probe_rows)
    coverage = {
        "schema": "tcm-operand-basis-evidence-coverage-v1",
        "job": job,
        "report_sha256": report_sha,
        "verification_sha256": custody["verification_sha256"],
        "observations_sha256": observations_sha,
        "standalone_probes_sha256": probes_sha,
        "rows": len(main_rows),
        "counts": observed,
        "main_formula": "1128 primary local writes + 144 full-vocabulary readout comparisons + 12 call ledgers + 12 token-authority rows + 3 common-history summaries + 12 same-ID summaries",
        "standalone_formula": "12 same-ID native replays expanded into 1,128 complete-prefix receiver/norm probe rows",
        "trends": trends,
        "limits": {
            "models_evaluated": 0,
            "raw_vectors_copied": False,
            "full_raw_report_remains_authority": True,
            "scientific_quality_gate": "none",
            "readout_metric": "signed full-vocabulary L2 and relative L2 using the declared right-array denominator",
        },
    }
    (index / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    source_manifest = _load(root / "payload" / "source-manifest.json")
    source = {
        "schema": SOURCE_SCHEMA,
        "job": job,
        "scope": "target-local scalar projection of the verified operand-basis panel; no model calls",
        "instrument_status": report.get("instrument_status", analysis.get("summary", {}).get("instrument_status")),
        "mechanics_status": report.get("mechanics_status", analysis.get("summary", {}).get("mechanics_status")),
        "trend_status": report.get("trend_status", analysis.get("summary", {}).get("trend_status", "observations")),
        "terminal_status": report.get("terminal_status", "not-requested"),
        "report_sha256": report_sha,
        "raw_report_sha256": custody["raw_report_sha256"],
        "verification_sha256": custody["verification_sha256"],
        "submission_sha256": custody["submission_sha256"],
        "observations_sha256": observations_sha,
        "standalone_probes_sha256": probes_sha,
        "coverage": coverage,
        "model": report.get("model"),
        "input": {
            "input_manifest_sha256": custody["input_manifest_sha256"],
            "inputs": report.get("inputs"),
            "driver_schema": report.get("driver", {}).get("schema"),
            "primary_paths": analysis.get("summary", {}).get("primary_paths"),
            "physical_token_totals": analysis.get("summary", {}).get("physical_token_totals"),
        },
        "program": {
            "binary_sha256": custody["binary_sha256"],
            "compiled_manifest_sha256": custody["compiled_manifest_sha256"],
            "basis_file_joins": custody["basis_file_joins"],
            "source_weight_joins": custody["source_weight_joins"],
            "driver_basis_sha256": report.get("driver", {}).get("basis_file_sha256", report.get("driver", {}).get("basis_sha256", {})),
        },
        "identity_joins": custody,
        "source_files": source_manifest.get("sources", {}),
        "artifacts": [
            _artifact(root, "report.json", "collected-report"),
            _artifact(root, "verification.json", "existing-verification"),
            _artifact(root, "job.json", "scheduler-record"),
            _artifact(root, "submission.json", "submission-record"),
            _artifact(root, "observations.jsonl", "native-observation-record"),
            _artifact(root, "raw/report.json", "raw-report-join"),
            _artifact(root, "raw/driver.json", "raw-driver-join"),
            _artifact(root, "raw/primary-seal.json", "primary-seal"),
            _artifact(root, "raw/input-manifest.json", "executed-input-manifest"),
            _artifact(root, "raw/compiled/bank_manifest.json", "compiled-bank-manifest"),
            _artifact(root, "raw/compiled/source-input-manifest.json", "compiled-source-input-manifest"),
            _artifact(root, "payload/source-manifest.json", "executed-source-manifest"),
        ],
        "limitations": [
            "Only scalar array metadata, signed differences, hashes, and coverage are projected; raw vectors and full native trajectories remain in the immutable run tree.",
            "The primary local-write rows compare each candidate return with a post-seal dense native diagnostic on the same saved receiver; they do not establish serving cost or semantic fidelity.",
            "The standalone probe stream expands same-ID native receiver/norm comparisons and is kept outside the standard main stream to avoid double counting.",
            "Watch fires and trend counts are instrument observations; no row, aggregate, or terminal promotion gate is defined here.",
        ],
        "custody": custody,
    }
    (target / "source.json").write_text(json.dumps(source, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    pipeline = {
        "run": _run_evidence_pipeline(index, target, "run"),
        "verify": _run_evidence_pipeline(index, target, "verify"),
    }
    summary = {
        "job": job,
        "target": str(target.relative_to(HERE)),
        "standalone": str(probes_path.relative_to(HERE)),
        "coverage": coverage,
        "pipeline": pipeline,
    }
    (index / "sweep-results.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "job": job,
                "rows": len(main_rows),
                "standalone_probe_rows": len(probe_rows),
                "report_sha256": report_sha,
                "pipeline": {
                    key: {
                        "returncode": value["returncode"],
                        "findings": value["summary"].get("findings"),
                        "broken": value["summary"].get("broken"),
                    }
                    for key, value in pipeline.items()
                },
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default=DEFAULT_JOB)
    args = parser.parse_args()
    project(args.job)


if __name__ == "__main__":  # pragma: no cover
    main()
