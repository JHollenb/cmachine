"""Project the verified state-closure run onto the Saturn evidence plane.

This adapter is deliberately source-bound.  It does not copy the native raw
arrays and it does not turn any effect size into a scientific gate.  Instead
it emits a small, typed JSONL stream containing the code-equality panel, the
three primary write contrasts, one query-step row for each of the 24 policy
arms, and the eight fixed attention calibration rows.  The run and all of
the larger raw artifacts remain the authority in ``runs/<job>``.

The output is confined to ``index/<job>/``.  In particular, this file never
rewrites a native report, verification record, scheduler log, or raw array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]
DEFAULT_JOB = "job-7b8a5933f1dd"
OBSERVATION_SCHEMA = "tcm-closure-observation-v1"
SOURCE_SCHEMA = "tcm-closure-evidence-source-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _artifact(root: Path, relative: str, role: str) -> dict[str, Any]:
    path = root / relative
    _require(path.is_file(), f"missing custody artifact: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finite_scalar(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return value == value and value not in (float("inf"), float("-inf"))


def _flatten_metrics(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Flatten the small numeric comparison dictionaries one level at a time."""

    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(item, dict):
                _flatten_metrics(name, item, out)
            elif _finite_scalar(item):
                out[name] = item


def _validate_run(
    job: str,
    root: Path,
    report: dict[str, Any],
    verification: dict[str, Any],
    job_record: dict[str, Any],
    submission: dict[str, Any],
) -> dict[str, Any]:
    """Verify custody and shape contracts before projecting observations."""

    report_sha = _sha256(root / "report.json")
    verification_sha = _sha256(root / "verification.json")
    _require(job_record.get("job_id") == job, "job record identity mismatch")
    _require(submission.get("job_id") == job, "submission identity mismatch")
    _require(verification.get("report_sha256") == report_sha, "verification/report hash mismatch")
    _require(report.get("schema") == "tcm-state-closure-experiment-v1", "unexpected report schema")
    _require(verification.get("analysis_recomputed") is True, "analysis was not recomputed")
    _require(verification.get("actual_loaded_model_verified") is True, "actual model identity was not verified")
    _require(verification.get("bank_and_geometry_verified") is True, "bank/geometry custody was not verified")
    _require(verification.get("models_evaluated") == 0, "unexpected model count in verification")

    driver = report.get("driver", {})
    model = report.get("model", {})
    _require(driver.get("schema") == "tcm-state-closure-driver-v1", "unexpected driver schema")
    _require(driver.get("model_digest") == model.get("content_sha256"), "driver/model digest mismatch")
    _require(driver.get("source_replay_checked") is True, "source replay was not checked")
    _require(
        driver.get("frozen_bank_before_sha256") == driver.get("frozen_bank_after_sha256"),
        "frozen bank changed during native execution",
    )

    manifest_path = root / "payload" / "source-manifest.json"
    source_reference_path = root / "payload" / "source-reference.json"
    manifest = _load(manifest_path)
    _require(report.get("source_manifest_sha256") == _sha256(manifest_path), "source manifest hash mismatch")
    _require(report.get("source_report_sha256") == _sha256(source_reference_path), "source report hash mismatch")
    _require(manifest.get("sources") == submission.get("config", {}).get("sources"), "payload/submission source map mismatch")

    checked_sources = 0
    for relative, expected in manifest.get("sources", {}).items():
        source_path = root / "payload" / relative
        _require(source_path.is_file(), f"missing payload source: {relative}")
        _require(_sha256(source_path) == expected, f"payload source hash mismatch: {relative}")
        checked_sources += 1

    raw_manifest = report.get("raw_files", {})
    raw_root = root / "raw"
    actual_raw = {path.relative_to(raw_root).as_posix() for path in raw_root.rglob("*") if path.is_file()}
    expected_raw = set(raw_manifest)
    _require(expected_raw <= actual_raw, "report references a missing raw artifact")
    # The collector retains a local copy of report.json inside raw/; it is not
    # part of the worker's raw manifest and is allowed as a custody extra.
    _require(actual_raw - expected_raw <= {"report.json"}, "unexpected unbound raw artifact")
    checked_raw = 0
    for relative, expected in raw_manifest.items():
        raw_path = raw_root / relative
        _require(raw_path.stat().st_size == expected.get("bytes"), f"raw byte count mismatch: {relative}")
        _require(_sha256(raw_path) == expected.get("sha256"), f"raw hash mismatch: {relative}")
        checked_raw += 1

    binary_receipt_path = root / "binary" / "receipt.json"
    binary_path = root / "binary" / "closure"
    _require(binary_path.is_file() and binary_receipt_path.is_file(), "executed binary custody is absent")
    binary_receipt = _load(binary_receipt_path)
    _require(_sha256(binary_path) == report.get("binary_sha256"), "executed binary/report hash mismatch")
    _require(binary_receipt.get("sha256") == report.get("binary_sha256"), "binary receipt hash mismatch")
    _require(binary_receipt.get("report_sha256") == report_sha, "binary receipt/report mismatch")
    _require(binary_receipt.get("matches_recorded_executed_binary") is True, "binary receipt is not authoritative")

    job_log = root / "logs.txt"
    expected_log = job_record.get("result", {}).get("artifacts", {}).get("job_log", {})
    _require(job_log.is_file(), "collected scheduler log is absent")
    _require(_sha256(job_log) == expected_log.get("sha256"), "scheduler log hash mismatch")
    _require(job_log.stat().st_size == expected_log.get("bytes"), "scheduler log byte count mismatch")

    receipt_wrapper = report.get("saturn_receipt", {})
    receipt = dict(receipt_wrapper.get("receipt", {}))
    fingerprint = receipt.pop("fingerprint", None)
    _require(fingerprint is not None, "receipt fingerprint missing")
    _require(_canonical_sha256(receipt) == fingerprint, "receipt fingerprint mismatch")
    receipt_request = receipt_wrapper.get("request", {})
    for key in ("binary_sha256", "model_digest", "source_manifest_sha256", "source_report_sha256"):
        if key in receipt_request:
            expected = report.get("binary_sha256") if key == "binary_sha256" else report.get(
                {"model_digest": "driver", "source_manifest_sha256": "source_manifest_sha256", "source_report_sha256": "source_report_sha256"}[key],
                {},
            )
            if key == "model_digest":
                expected = driver.get("model_digest")
            _require(receipt_request[key] == expected, f"receipt request mismatch: {key}")

    return {
        "report_sha256": report_sha,
        "verification_sha256": verification_sha,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_report_sha256": _sha256(source_reference_path),
        "binary_sha256": _sha256(binary_path),
        "binary_receipt_sha256": _sha256(binary_receipt_path),
        "scheduler_log_sha256": _sha256(job_log),
        "source_files_verified": checked_sources,
        "raw_files_verified": checked_raw,
        "raw_files_manifest_sha256": _canonical_sha256(raw_manifest),
        "receipt_fingerprint": fingerprint,
        "receipt_fingerprint_verified": True,
    }


def _geometry_lookup(report: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    lookup: dict[str, dict[int, dict[str, Any]]] = {}
    for task in report.get("geometry", {}).get("tasks", []):
        candidates = task.get("candidates", [])
        lookup[task.get("task")] = {int(row["index"]): row for row in candidates}
    return lookup


def _base_row(report_sha: str, model: dict[str, Any], observable: str, case: str) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "kind": "observation",
        "observable": observable,
        "case": case,
        "model_id": model.get("identity"),
        "model_content_sha256": model.get("content_sha256"),
        "report_sha256": report_sha,
    }


def _build_observations(
    job: str,
    report: dict[str, Any],
    custody: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    report_sha = custody["report_sha256"]
    model = report.get("model", {})
    driver = report["driver"]
    analysis = report["analysis"]
    geometry = _geometry_lookup(report)
    exact_by_task = {item["task"]: item for item in analysis["exact_fiber_codes"]}
    driver_by_task = {item["task"]: item for item in driver["cases"]}

    rows: list[dict[str, Any]] = []
    counts = {"candidate_code": 0, "primary_write": 0, "arm_actual_step": 0, "attention": 0}

    # 192 source-only code equality observations.  Candidate eligibility is
    # computed before any candidate MLP is evaluated; native outcomes do not
    # enter this stream.
    for task in sorted(exact_by_task):
        exact = exact_by_task[task]
        native_case = driver_by_task[task]
        _require(exact.get("actual_runtime_encoder_verified") is True, f"encoder not verified: {task}")
        _require(exact.get("selection_used_outcomes") is False, f"outcome leakage in selection: {task}")
        _require(exact.get("selection_order_checked") is True, f"selection order not checked: {task}")
        _require(native_case.get("selection_before_any_candidate_mlp") is True, f"candidate selection order invalid: {task}")
        _require(exact.get("selected_id") == exact.get("first_eligible_exact_fiber_id"), f"selection is not first eligible candidate: {task}")
        _require(native_case.get("selected_index") == exact.get("selected_id"), f"driver/analysis selected candidate mismatch: {task}")
        source_hex = exact["source_code_hex"]
        source_sha = hashlib.sha256(bytes.fromhex(source_hex)).hexdigest()
        _require(source_sha == exact["source_code_sha256"], f"source code hash mismatch: {task}")
        driver_candidates = {int(c["index"]): c for c in native_case.get("candidates", [])}
        _require(len(exact.get("candidates", [])) == 64, f"candidate count mismatch: {task}")
        for candidate in sorted(exact["candidates"], key=lambda item: int(item["index"])):
            index = int(candidate["index"])
            code_hex = candidate["code_hex"]
            code_sha = hashlib.sha256(bytes.fromhex(code_hex)).hexdigest()
            _require(len(bytes.fromhex(code_hex)) == 16, f"candidate code width mismatch: {task}/{index}")
            _require(code_sha == candidate["code_sha256"], f"candidate code hash mismatch: {task}/{index}")
            _require(candidate["same_code"] == (code_hex == source_hex), f"candidate equality mismatch: {task}/{index}")
            _require(index in driver_candidates, f"driver candidate missing: {task}/{index}")
            _require(driver_candidates[index]["code_hex"] == code_hex, f"analysis/driver code mismatch: {task}/{index}")
            geom = geometry.get(task, {}).get(index, {})
            native_candidate = driver_candidates[index]
            row = _base_row(report_sha, model, "source_code_equality", task)
            row.update(
                {
                    "stream": "candidate-code",
                    "candidate_index": index,
                    "candidate_id": int(candidate["id"]),
                    "source_code_sha256": source_sha,
                    "candidate_code_sha256": code_sha,
                    "candidate_code_hex": code_hex,
                    "source_code_hex": source_hex,
                    "receiver_identity": candidate.get("receiver_identity"),
                    "selection_order_checked": bool(exact["selection_order_checked"]),
                    "selected": index == int(exact["selected_id"]),
                    "metrics": {
                        "same_code": bool(candidate["same_code"]),
                        "different_receiver": bool(candidate["different_receiver"]),
                        "eligible_exact_fiber": bool(candidate["eligible_exact_fiber"]),
                        "candidate_index": index,
                        "candidate_id": int(candidate["id"]),
                        "native_rho": native_candidate.get("rho"),
                        "angle_rad": geom.get("angle_rad"),
                        "direction": geom.get("direction"),
                    },
                }
            )
            rows.append(row)
            counts["candidate_code"] += 1

    # Three source-vs-selected-fiber write contrasts.  The complete branch
    # dictionaries remain in report.json; this stream emits scalar metrics and
    # points back to the report rather than copying any hidden vectors.
    write_by_task = {item["task"]: item for item in analysis["per_case_write_differences"]}
    _require(set(write_by_task) == set(exact_by_task), "write panel/task mismatch")
    for task in sorted(write_by_task):
        panel = write_by_task[task]
        pair = panel.get("pairs", {}).get("source_vs_fiber")
        _require(isinstance(pair, dict), f"missing primary write pair: {task}")
        row = _base_row(report_sha, model, "mlp_write", task)
        row.update(
            {
                "stream": "primary-write",
                "pair": "source_vs_selected_fiber",
                "step": 0,
                "metrics": {},
                "selected_candidate_id": exact_by_task[task]["selected_id"],
                "source_code_sha256": exact_by_task[task]["source_code_sha256"],
            }
        )
        _flatten_metrics("", pair, row["metrics"])
        row["bytes_equal"] = bool(pair.get("write", {}).get("bytes_equal"))
        rows.append(row)
        counts["primary_write"] += 1

    # One source-bound query-step observation per arm.  The row is anchored at
    # the intervention/query step (step 0) and carries the complete own-token
    # ID list as metadata, so a downstream reader cannot mistake it for a
    # teacher-forced whole continuation.
    for case in sorted(driver["cases"], key=lambda item: item["task"]):
        for arm in sorted(case["arms"], key=lambda item: int(item["policy"])):
            _require(arm.get("steps"), f"arm has no actual steps: {arm.get('name')}")
            step = arm["steps"][0]
            own = arm.get("own", {})
            row = _base_row(report_sha, model, "native_policy_step", case["task"])
            row.update(
                {
                    "stream": "arm-actual-step",
                    "arm": arm["name"],
                    "policy": int(arm["policy"]),
                    "step_anchor": "query-transition-step-0",
                    "position": int(step["position"]),
                    "input_id": int(step["input_id"]),
                    "top": int(step["top"]),
                    "runner_up": int(step["runner_up"]),
                    "logits_sha256": step["logits_sha256"],
                    "valid_kv_sha256": step["valid_kv_sha256"],
                    "own_token_ids": list(own.get("ids", [])),
                    "prefix_intact": bool(arm.get("prefix_intact")),
                    "metrics": {
                        "policy": int(arm["policy"]),
                        "position": int(step["position"]),
                        "input_id": int(step["input_id"]),
                        "top": int(step["top"]),
                        "runner_up": int(step["runner_up"]),
                        "margin": step.get("margin"),
                        "down_calls": int(step.get("down_calls", 0)),
                        "gate_up_calls": int(step.get("gate_up_calls", 0)),
                        "substitution": bool(step.get("substitution", False)),
                        "rho": step.get("rho"),
                        "own_token_count": len(own.get("ids", [])),
                        "own_stopped": bool(own.get("stopped", False)),
                        "own_truncated": bool(own.get("truncated", False)),
                        "actual_step_count": len(arm["steps"]),
                        "parent_position": int(arm["parent_position"]),
                        "final_position": int(arm["final_position"]),
                    },
                }
            )
            rows.append(row)
            counts["arm_actual_step"] += 1

    # Eight fixed attention calibration observations are structural checks,
    # not chat-performance measurements.
    for index, calibration in enumerate(driver.get("attention", [])):
        prefix = calibration["prefix"]
        row = _base_row(report_sha, model, "attention_calibration", prefix)
        row.update(
            {
                "stream": "attention-calibration",
                "calibration_index": index,
                "prefix": prefix,
                "input_kind": "fixed-analytic-calibration",
                "metrics": {
                    "query": int(calibration["query"]),
                    "length": int(calibration["length"]),
                    "world": int(calibration["world"]),
                    "output_first": calibration["output_first"],
                },
            }
        )
        rows.append(row)
        counts["attention"] += 1

    expected = {"candidate_code": 192, "primary_write": 3, "arm_actual_step": 24, "attention": 8}
    _require(counts == expected, f"observation coverage mismatch: got {counts}, expected {expected}")
    return rows, counts


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
        raise RuntimeError(f"saturn-evidence {action} emitted invalid JSON: {completed.stderr[-2000:]}") from exc
    if completed.returncode not in (0, 3):
        raise RuntimeError(f"saturn-evidence {action} refused input (exit {completed.returncode})")
    return {"returncode": completed.returncode, "summary": summary}


def project(job: str) -> dict[str, Any]:
    root = HERE / "runs" / job
    index = HERE / "index" / job
    target = index / "target-streams"
    target.mkdir(parents=True, exist_ok=True)

    report = _load(root / "report.json")
    verification = _load(root / "verification.json")
    job_record = _load(root / "job.json")
    submission = _load(root / "submission.json")
    custody = _validate_run(job, root, report, verification, job_record, submission)
    rows, counts = _build_observations(job, report, custody)

    observations_path = target / "observations.jsonl"
    observations_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    observations_sha = _sha256(observations_path)

    manifest = _load(root / "payload" / "source-manifest.json")
    exact = report["analysis"]["exact_fiber_codes"]
    raw_manifest = report["raw_files"]
    source = {
        "schema": SOURCE_SCHEMA,
        "job": job,
        "scope": "source-bound closure observations; standard watch/xref only; no semantic admission",
        "instrument_status": "applicable",
        "terminal_status": "not-authorized",
        "report_sha256": custody["report_sha256"],
        "verification_sha256": custody["verification_sha256"],
        "observations_sha256": observations_sha,
        "coverage": {**counts, "total": len(rows), "expected_total": 227},
        "model": report["model"],
        "input": {
            "source_position": report["driver"].get("source_position"),
            "source_kv_sha256": report["driver"].get("source_kv_sha256"),
            "source_replay_checked": report["driver"].get("source_replay_checked"),
            "raw_files_manifest_sha256": custody["raw_files_manifest_sha256"],
            "raw_files_count": len(raw_manifest),
        },
        "code": {
            "exact_code_bytes": report["analysis"]["summary"].get("exact_code_bytes"),
            "binary_sha256": custody["binary_sha256"],
            "binary_receipt_sha256": custody["binary_receipt_sha256"],
            "model_digest": report["driver"].get("model_digest"),
            "source_manifest_sha256": custody["source_manifest_sha256"],
            "source_report_sha256": custody["source_report_sha256"],
            "selected_exact_fiber": [
                {
                    "task": item["task"],
                    "selected_id": item["selected_id"],
                    "selected_code_sha256": hashlib.sha256(bytes.fromhex(item["selected_code_hex"])).hexdigest(),
                    "selection_used_outcomes": item["selection_used_outcomes"],
                }
                for item in exact
            ],
        },
        "source_files": manifest.get("sources", {}),
        "artifacts": [
            _artifact(root, "report.json", "verified-native-report"),
            _artifact(root, "verification.json", "postcollection-verification"),
            _artifact(root, "job.json", "scheduler-record"),
            _artifact(root, "submission.json", "submission-record"),
            _artifact(root, "logs.txt", "scheduler-log"),
            _artifact(root, "observations.jsonl", "native-observation-record"),
            _artifact(root, "payload/source-manifest.json", "executed-source-manifest"),
            _artifact(root, "payload/source-reference.json", "executed-source-reference"),
            _artifact(root, "payload/driver.c", "native-driver-source"),
            _artifact(root, "payload/run.py", "launcher-source"),
            _artifact(root, "payload/worker.py", "worker-source"),
            _artifact(root, "payload/geometry/geometry.json", "geometry-manifest"),
            _artifact(root, "binary/closure", "executed-native-binary"),
            _artifact(root, "binary/receipt.json", "executed-binary-receipt"),
        ],
        "custody": custody,
        "limitations": [
            "Candidate eligibility is source-only exact 16-byte code equality plus the declared receiver identity; native outcomes are not used for selection.",
            "The arm stream contains the actual query transition (step 0) and own-token ID list, not a copied teacher-forced continuation.",
            "The eight attention rows are fixed analytic calibration inputs and are structural runtime checks, not learned chat evidence.",
            "Only scalar summaries are projected here; full vectors and native trajectories remain in the immutable run/raw tree.",
            "The evidence sweep is offline and target-local; watch fires are review signals, not scientific nulls or promotions.",
        ],
    }
    (target / "source.json").write_text(json.dumps(source, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    coverage = {
        "schema": "tcm-closure-evidence-coverage-v1",
        "job": job,
        "report_sha256": custody["report_sha256"],
        "verification_sha256": custody["verification_sha256"],
        "observations_sha256": observations_sha,
        "rows": len(rows),
        "counts": counts,
        "coverage_formula": "3 tasks x 64 candidates + 3 primary writes + 3 tasks x 8 arms + 8 attention calibrations",
        "limitations": source["limitations"],
    }
    (target / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pipeline = {
        "run": _run_evidence_pipeline(index, target, "run"),
        "verify": _run_evidence_pipeline(index, target, "verify"),
    }
    summary = {"job": job, "target": str(target), "coverage": coverage, "pipeline": pipeline}
    (index / "coverage.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"job": job, "target": str(target), "rows": len(rows), "counts": counts, "pipeline": {k: {"returncode": v["returncode"], "findings": v["summary"].get("findings"), "broken": v["summary"].get("broken")} for k, v in pipeline.items()}}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default=DEFAULT_JOB)
    args = parser.parse_args()
    project(args.job)


if __name__ == "__main__":
    main()
