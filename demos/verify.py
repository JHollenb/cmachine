#!/usr/bin/env python3
"""Verify a collected pretrained demo run without loading a model.

The verifier authenticates custody, source and raw-file identity, the guarded
CUDA execution envelope, the Saturn whole-invocation receipt, and the two
demo-specific mechanical reports.  It deliberately does not judge whether a
generated answer is semantically correct.

Usage from the cmachine repository root::

    python3 demos/verify.py demos/fleet-runs/JOB
    python3 demos/verify.py demos/fleet-runs/JOB --publish demos/results/pretrained
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable


PUBLIC_PAYLOAD_LIMIT = 12 * 1024 * 1024


class VerificationError(RuntimeError):
    """A required provenance or mechanics contract did not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VerificationError(f"missing JSON evidence: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON evidence {path}: {exc}") from exc


def safe_relative(name: str) -> Path:
    path = Path(str(name))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise VerificationError(f"unsafe relative evidence path: {name!r}")
    return path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def bool_map(values: dict[str, Any]) -> bool:
    """Return a named aggregate only for already-required mechanics fields."""

    return all(value is True for value in values.values())


def locate_result_files(target: Path) -> tuple[Path, Path, Path, Path, Path]:
    payload = target / "payload"
    raw = target / "raw"
    return (
        target / "report.json",
        target / "job.json",
        target / "guarded-receipt.json",
        target / "submission.json",
        payload / "source-manifest.json",
    )


def verify_source_payload(
    target: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    payload: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    payload = payload or (target / "payload")
    manifest_path = manifest_path or (payload / "source-manifest.json")
    require(payload.is_dir(), f"missing executed payload directory: {payload}")
    sources = manifest.get("sources")
    require(isinstance(sources, dict) and sources, "source manifest has no sources")
    verified = 0
    payload_bytes = 0
    source_rows: list[dict[str, Any]] = []
    for name, declared in sorted(sources.items()):
        relative = safe_relative(name)
        path = payload / relative
        require(path.is_file() and not path.is_symlink(), f"missing or non-regular payload source: {name}")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        require(isinstance(declared, dict), f"source manifest row is not an object: {name}")
        require(actual_sha == declared.get("sha256"), f"payload source hash mismatch: {name}")
        if declared.get("bytes") is not None:
            require(actual_bytes == declared.get("bytes"), f"payload source size mismatch: {name}")
        verified += 1
        payload_bytes += actual_bytes
        source_rows.append({"path": name, "sha256": actual_sha, "bytes": actual_bytes})
    manifest_sha = sha256_file(manifest_path)
    require(report.get("source_manifest_sha256") == manifest_sha,
            "report source-manifest digest does not match the collected payload")
    request = report.get("request")
    if isinstance(request, dict):
        require(request.get("source_manifest_sha256") == manifest_sha,
                "invocation request source-manifest digest differs")
    return {
        "source_files_verified": verified,
        "payload_source_bytes": payload_bytes,
        "source_manifest_sha256": manifest_sha,
        "sources": source_rows,
    }


def verify_raw_files(target: Path, report: dict[str, Any]) -> dict[str, Any]:
    raw = target / "raw"
    require(raw.is_dir(), f"missing raw evidence directory: {raw}")
    declared = report.get("raw_files")
    require(isinstance(declared, dict), "report has no raw_files manifest")
    verified = 0
    raw_bytes = 0
    for name, row in sorted(declared.items()):
        relative = safe_relative(name)
        path = raw / relative
        require(path.is_file() and not path.is_symlink(), f"missing or non-regular raw file: {name}")
        require(isinstance(row, dict), f"raw manifest row is not an object: {name}")
        actual_sha = sha256_file(path)
        actual_bytes = path.stat().st_size
        require(actual_sha == row.get("sha256"), f"raw file hash mismatch: {name}")
        require(actual_bytes == row.get("bytes"), f"raw file size mismatch: {name}")
        verified += 1
        raw_bytes += actual_bytes

    # The worker intentionally omits .bin files from both report.raw_files and
    # its evidence archive. If a future worker ships one anyway, keep its hash
    # visible instead of silently ignoring an unindexed artifact.
    unlisted: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(raw).as_posix()
        if name in declared:
            continue
        if name == "report.json":
            continue
        unlisted.append({"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    raw_report = read_json(raw / "report.json")
    report_bytes = read_json(target / "report.json")
    require(raw_report == report_bytes, "raw/report.json does not match collected report.json")
    results = report.get("results") if isinstance(report.get("results"), dict) else {}
    nested_reports = {
        "chat/report.json": results.get("chat"),
        "operands/report.json": results.get("operands"),
    }
    for relative, expected in nested_reports.items():
        require(isinstance(expected, dict), f"worker result missing nested report: {relative}")
        nested_path = raw / relative
        require(nested_path.is_file() and not nested_path.is_symlink(),
                f"missing nested raw report: {relative}")
        require(read_json(nested_path) == expected,
                f"nested raw report does not match collected report: {relative}")
    return {
        "raw_files_verified": verified,
        "raw_bytes_verified": raw_bytes,
        "unlisted_raw_files": unlisted,
        "raw_report_matches_collected_report": True,
    }


def verify_scheduler(
    target: Path,
    report: dict[str, Any],
    job: dict[str, Any],
    guard: dict[str, Any],
    submission: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    job_id = job.get("job_id")
    require(isinstance(job_id, str) and job_id, "scheduler job has no job_id")
    require(job_id == submission.get("job_id"), "submission/job identity differs")
    require(job_id == guard.get("job_id"), "guarded receipt/job identity differs")
    require(report.get("job_id") == job_id, "worker report/job identity differs")
    require(job.get("state") == "succeeded", f"scheduler terminal state is {job.get('state')!r}")
    result = job.get("result")
    require(isinstance(result, dict), "scheduler result is missing")
    require(result.get("status") == "ok" and result.get("returncode") == 0,
            "scheduler terminal result is not a successful zero-exit execution")
    require(guard.get("state") == "succeeded", "guarded receipt is not terminal succeeded")
    guard_result = guard.get("result")
    require(isinstance(guard_result, dict) and guard_result.get("status") == "ok"
            and guard_result.get("returncode") == 0,
            "guarded receipt terminal result is not successful")

    custody = guard.get("payload_custody")
    require(isinstance(custody, dict), "guarded receipt has no payload custody")
    declared = custody.get("declared")
    sealed = custody.get("sealed")
    executed = custody.get("executed")
    require(isinstance(declared, dict) and isinstance(sealed, dict) and isinstance(executed, dict),
            "payload custody is incomplete")
    custody_tuple = lambda row: (row.get("sha256"), row.get("size_bytes"))
    require(custody_tuple(declared) == custody_tuple(sealed) == custody_tuple(executed),
            "declared, sealed, and executed payloads differ")
    executed_payload = result.get("executed_payload")
    require(isinstance(executed_payload, dict), "scheduler result has no executed payload")
    require(custody_tuple(executed_payload) == custody_tuple(declared),
            "scheduler executed payload differs from guarded custody")

    submitted = guard.get("submitted_request")
    require(isinstance(submitted, dict), "guarded receipt has no submitted request")
    submitted_command = submitted.get("cmd")
    actual_command = job.get("cmd")
    require(isinstance(submitted_command, list) and isinstance(actual_command, list),
            "scheduler command records are missing")
    require(submitted_command == actual_command, "submitted and executed scheduler commands differ")
    require(any("worker.py" in str(part) for part in actual_command),
            "executed scheduler command is not the demo worker")

    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    submitted_config = submitted.get("config") if isinstance(submitted.get("config"), dict) else {}
    require(config.get("device") == "cuda", "job configuration did not request CUDA")
    require(config.get("backend") == "hf", "job configuration backend is not the declared HF path")
    require(job.get("needs", {}).get("cuda") is True, "scheduler CUDA guard was not requested")
    require(submitted_config.get("device") == "cuda" and submitted_config.get("backend") == "hf",
            "guarded submitted request does not bind CUDA/HF")
    require(config.get("source_manifest_sha256") == sha256_file(target / "payload/source-manifest.json"),
            "job config source-manifest digest differs")
    require(config.get("sources") == manifest.get("sources"), "job config source rows differ")
    require(submitted_config.get("sources") == manifest.get("sources"),
            "guarded submitted source rows differ")
    source_manifest_sha = sha256_file(target / "payload/source-manifest.json")
    require(config.get("model_digest") == manifest.get("model", {}).get("content_sha256"),
            "job config model digest differs from source manifest")
    require(submitted_config.get("source_manifest_sha256") == source_manifest_sha,
            "guarded submitted source-manifest digest differs")
    require(submitted_config.get("model_digest") == manifest.get("model", {}).get("content_sha256"),
            "guarded submitted model digest differs")

    plan = job.get("plan") if isinstance(job.get("plan"), dict) else {}
    resolved = guard.get("resolved_execution") if isinstance(guard.get("resolved_execution"), dict) else {}
    plans = resolved.get("plans") if isinstance(resolved.get("plans"), dict) else {}
    planned_cuda = plan.get("device") == "cuda" or any(
        isinstance(row, dict) and row.get("device") == "cuda" for row in plans.values()
    )
    require(planned_cuda, "guarded execution has no CUDA plan")
    require(str(job.get("assigned_host", "")), "scheduler did not record an assigned host")

    artifact_log = (((result.get("artifacts") or {}).get("job_log"))
                    if isinstance(result.get("artifacts"), dict) else None)
    log_path = target / "logs.txt"
    require(log_path.is_file(), "scheduler log is missing")
    if isinstance(artifact_log, dict) and artifact_log.get("sha256"):
        require(sha256_file(log_path) == artifact_log["sha256"],
                "collected scheduler log differs from sealed job artifact")

    return {
        "job_id": job_id,
        "terminal_state": job["state"],
        "returncode": result["returncode"],
        "assigned_host": job.get("assigned_host"),
        "cuda_requested": True,
        "cuda_plan_observed": True,
        "command": actual_command,
        "payload_custody_exact": True,
        "scheduler_log_sha256": sha256_file(log_path),
        "executed_payload": executed_payload,
    }


def _nested_digest_candidates(chat: dict[str, Any]) -> tuple[list[str], list[str]]:
    digests: list[str] = []
    identities: list[str] = []
    inputs = chat.get("inputs") if isinstance(chat.get("inputs"), dict) else {}
    model_input = inputs.get("model") if isinstance(inputs.get("model"), dict) else {}
    server = chat.get("server") if isinstance(chat.get("server"), dict) else {}
    ready = server.get("ready") if isinstance(server.get("ready"), dict) else {}
    for value in (
        model_input.get("server_digest"),
        server.get("model_digest"),
        ready.get("model_digest"),
    ):
        if isinstance(value, str):
            digests.append(value)
    for value in (
        model_input.get("server_identity"),
        server.get("model_identity"),
        ready.get("model_identity"),
    ):
        if isinstance(value, str):
            identities.append(value)
    return digests, identities


def verify_model_and_binaries(
    target: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    model = manifest.get("model")
    require(isinstance(model, dict), "source manifest has no model identity")
    expected_digest = model.get("content_sha256")
    expected_identity = model.get("identity")
    require(isinstance(expected_digest, str) and expected_digest, "missing expected model digest")
    require(isinstance(expected_identity, str) and expected_identity, "missing expected model identity")
    request = report.get("request")
    require(isinstance(request, dict), "worker report has no invocation request")
    require(request.get("model_digest") == expected_digest, "invocation model digest differs")

    results = report.get("results")
    require(isinstance(results, dict), "worker report has no results")
    chat = results.get("chat")
    operands = results.get("operands")
    require(isinstance(chat, dict) and isinstance(operands, dict), "chat/operand result pair is incomplete")
    server = chat.get("server") if isinstance(chat.get("server"), dict) else {}
    server_command = server.get("command")
    require(isinstance(server_command, list) and "serve" in server_command and "--context" in server_command,
            "chat did not record the expected native C/CUDA serve command")
    arithmetic_contract = str((chat.get("profile") or {}).get("arithmetic_contract", ""))
    require("FP32" in arithmetic_contract and "cuBLAS" in arithmetic_contract,
            "chat report does not bind its FP32/cuBLAS arithmetic profile")
    chat_digests, chat_identities = _nested_digest_candidates(chat)
    require(chat_digests and all(value == expected_digest for value in chat_digests),
            "chat server/model digest does not match the manifest")
    require(chat_identities and all(value == expected_identity for value in chat_identities),
            "chat server/model identity does not match the manifest")
    require(operands.get("model_digest") == expected_digest,
            "compiled operand report model digest differs")

    binary_sha = request.get("binary_sha256")
    operand_binary_sha = request.get("operand_binary_sha256")
    chat_binary = ((chat.get("inputs") or {}).get("binary")
                   if isinstance(chat.get("inputs"), dict) else None)
    require(isinstance(binary_sha, str) and isinstance(chat_binary, dict)
            and chat_binary.get("sha256") == binary_sha,
            "chat binary digest does not match invocation request")
    invocation = read_json(target / "raw/operands/invocation.json")
    require(isinstance(operand_binary_sha, str)
            and invocation.get("binary_sha256") == operand_binary_sha,
            "compiled operand binary digest does not match invocation request")
    return {
        "model_digest": expected_digest,
        "model_identity": expected_identity,
        "chat_binary_sha256": binary_sha,
        "operand_binary_sha256": operand_binary_sha,
        "identity_join_exact": True,
    }


def verify_saturn_receipt(target: Path, report: dict[str, Any]) -> dict[str, Any]:
    wrapper = report.get("saturn_receipt")
    require(isinstance(wrapper, dict), "worker report has no Saturn receipt")
    receipt = wrapper.get("receipt")
    require(isinstance(receipt, dict), "Saturn receipt body is missing")
    sealed = dict(receipt)
    fingerprint = sealed.pop("fingerprint", None)
    require(isinstance(fingerprint, str), "Saturn receipt has no fingerprint")
    require(canonical_sha(sealed) == fingerprint, "Saturn receipt fingerprint mismatch")
    require(sealed.get("parent") == wrapper.get("before"), "Saturn receipt parent closure mismatch")
    require(sealed.get("result") == wrapper.get("after"), "Saturn receipt result closure mismatch")
    require(sealed.get("boundary") == "complete", "Saturn receipt is not complete")
    require(wrapper.get("request") == report.get("request"),
            "Saturn receipt request differs from worker invocation request")
    raw_request = target / "raw/request.json"
    if raw_request.exists():
        require(read_json(raw_request) == wrapper.get("request"),
                "raw Saturn request differs from receipt request")
    execution = sealed.get("execution")
    require(isinstance(execution, dict) and execution.get("scope") == "whole C invocation",
            "Saturn receipt does not bind the whole C invocation")
    return {
        "fingerprint": fingerprint,
        "parent": wrapper.get("before"),
        "result": wrapper.get("after"),
        "boundary": sealed.get("boundary"),
        "whole_invocation_scope": execution.get("scope"),
    }


def verify_chat(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    chat = report["results"]["chat"]
    require(chat.get("schema") == "tcm-chat-branch-demo-v1", "unexpected chat report schema")
    require(chat.get("status") == "ok", "chat demo did not complete")
    require(chat.get("mechanics_status") == "valid", "chat mechanics status is not valid")
    comparisons = chat.get("comparisons")
    require(isinstance(comparisons, dict), "chat comparisons are missing")
    rollback = comparisons.get("rollback_to_root")
    replay_rollback = comparisons.get("replay_rollback_to_root")
    replay = comparisons.get("replay_exact")
    required = {
        "rollback_to_root": isinstance(rollback, dict) and rollback.get("equal") is True,
        "replay_rollback_to_root": isinstance(replay_rollback, dict) and replay_rollback.get("equal") is True,
        "replay_exact": isinstance(replay, dict) and bool_map(replay),
        "branch_updates_different": comparisons.get("branch_updates_different") is True,
    }
    require(bool_map(required), f"chat mechanics comparison failed: {required}")
    profile = chat.get("profile") if isinstance(chat.get("profile"), dict) else {}
    if profile.get("cache_policy") == "native":
        require(comparisons.get("native_rollback_logits_invalid_expected") is True,
                "native rollback expectation missing")
        require(comparisons.get("rollback_logits_valid_observed") is False,
                "native rollback unexpectedly retained valid logits")

    turns = chat.get("turns")
    require(isinstance(turns, dict), "chat turns are missing")
    turn_rows: dict[str, dict[str, Any]] = {}
    for label in ("initial", "branch_a", "branch_a_replay", "branch_b"):
        row = turns.get(label)
        require(isinstance(row, dict), f"chat turn missing: {label}")
        require(isinstance(row.get("generated_ids"), list) and row["generated_ids"],
                f"chat turn has no generated IDs: {label}")
        require(isinstance(row.get("generated_text"), str), f"chat turn has no text: {label}")
        turn_rows[label] = row

    replay_exact = bool_map(replay)
    summary = {
        "initial": {"text": turn_rows["initial"]["generated_text"],
                    "generated_tokens": len(turn_rows["initial"]["generated_ids"])},
        "branch_a": {"text": turn_rows["branch_a"]["generated_text"],
                     "generated_tokens": len(turn_rows["branch_a"]["generated_ids"])},
        "branch_b": {"text": turn_rows["branch_b"]["generated_text"],
                     "generated_tokens": len(turn_rows["branch_b"]["generated_ids"])},
        "replay_exact": replay_exact,
    }
    return {"checks": required, "comparisons": comparisons}, summary


def verify_lexical(target: Path, report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    operands = report["results"]["operands"]
    require(operands.get("schema") == "compiled-operands-demo-v1", "unexpected lexical report schema")
    rows = operands.get("rows")
    require(isinstance(rows, list) and rows, "lexical report has no rows")
    row_checks = {
        "full_logits_exact": all(row.get("full_logits_exact") is True for row in rows),
        "valid_cache_hash_exact": all(row.get("valid_cache_hash_exact") is True for row in rows),
        "token_exact": all(row.get("token_exact") is True for row in rows),
    }
    require(bool_map(row_checks), f"lexical exactness row failed: {row_checks}")
    required_scalars = (
        "source_steps", "candidate_steps", "covered_ids", "compile_linear_macs",
        "table_host_bytes", "candidate_transfer_bytes", "native_model_gpu_bytes",
        "session_gpu_bytes",
    )
    for name in required_scalars:
        require(isinstance(operands.get(name), int) and operands[name] >= 0,
                f"lexical scalar missing or invalid: {name}")
    require(operands["source_steps"] == len(rows) == operands["candidate_steps"],
            "lexical source/candidate/row counts differ")
    require(operands.get("all_steps_exact") is True, "lexical all_steps_exact is false")
    # This flag concerns the covered candidate trajectory.  The missing-ID
    # control has its own field below; a successful trajectory must not refuse
    # one of its own source-covered token transitions.
    require(operands.get("candidate_transition_refused") is False,
            "lexical candidate refused during its covered trajectory")
    require(operands.get("missing_refusal_no_mutation") is True,
            "lexical missing-token refusal mutated state")
    require(operands.get("save_load_identity") is True, "lexical table save/load identity failed")

    seal = read_json(target / "raw/operands/source-seal.json")
    require(seal.get("schema") == "compiled-operands-source-v1", "unexpected lexical source-seal schema")
    require(seal.get("model_digest") == report["request"]["model_digest"],
            "lexical source seal model digest differs")
    require(seal.get("program_fingerprint") == operands.get("program_fingerprint"),
            "lexical source seal program fingerprint differs")
    steps = seal.get("steps")
    require(isinstance(steps, list) and len(steps) == len(rows), "lexical source seal row count differs")
    for index, (row, source) in enumerate(zip(rows, steps)):
        for field in ("position", "consumed", "pending", "logits_sha256", "valid_cache_sha256"):
            if field in row and field in source:
                require(row[field] == source[field], f"lexical source/candidate row differs at {index}: {field}")

    invocation = read_json(target / "raw/operands/invocation.json")
    ids_path = target / "raw/operand-prompt.ids"
    require(ids_path.is_file(), "lexical prompt IDs are missing")
    require(invocation.get("tokens_sha256") == sha256_file(ids_path),
            "lexical prompt token file digest differs")
    table_ids = seal.get("table_ids")
    require(isinstance(table_ids, list) and len(table_ids) == operands["covered_ids"],
            "lexical covered-ID count differs from source seal")
    require(len(set(table_ids)) == len(table_ids), "lexical source table IDs are not unique")
    require(operands.get("candidate_generated_tokens_consumed") == seal.get("generated_tokens_consumed"),
            "lexical source/candidate generated-token counts differ")

    summary = {
        "steps": len(rows),
        "covered_ids": operands["covered_ids"],
        "generated_tokens_consumed": operands["candidate_generated_tokens_consumed"],
        "linear_macs": operands["compile_linear_macs"],
        "source_nominal_linear_macs": operands["source_native_nominal_linear_macs"],
        "candidate_nominal_linear_macs": operands["candidate_nominal_linear_macs"],
        "omitted_linear_macs": operands["candidate_original_first_layer_qkv_macs_omitted"],
        "omitted_first_layer_qkv_macs": operands["candidate_original_first_layer_qkv_macs_omitted"],
        "bytes": {
            "table_host": operands["table_host_bytes"],
            "candidate_transfer": operands["candidate_transfer_bytes"],
            "native_model_gpu": operands["native_model_gpu_bytes"],
            "session": operands["session_gpu_bytes"],
        },
        "rows_full_logits_cache_token_exact": True,
    }
    return {"row_checks": row_checks, "rows": len(rows), "source_seal_steps": len(steps),
            "missing_refusal_no_mutation": True}, summary


def redact_command(command: Any) -> list[str] | None:
    if not isinstance(command, list):
        return None
    # The submitted command already uses {env}; remove any accidental absolute
    # scheduler workspace path while retaining the executable contract.
    return ["{env}" if "/mrun-work/" in str(part) else str(part) for part in command]


def public_guard(job: dict[str, Any], guard: dict[str, Any], report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    submitted = guard.get("submitted_request") if isinstance(guard.get("submitted_request"), dict) else {}
    submitted_config = submitted.get("config") if isinstance(submitted.get("config"), dict) else {}
    custody = guard.get("payload_custody", {})
    result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
    plan = job.get("plan", {}) if isinstance(job.get("plan"), dict) else {}
    request = report.get("request", {}) if isinstance(report.get("request"), dict) else {}
    config = {
        "schema": submitted_config.get("schema"),
        "backend": submitted_config.get("backend"),
        "device": submitted_config.get("device"),
        "dtype": submitted_config.get("dtype"),
        "source_manifest_sha256": report.get("source_manifest_sha256"),
        "model_digest": request.get("model_digest"),
        "binary_sha256": request.get("binary_sha256"),
        "operand_binary_sha256": request.get("operand_binary_sha256"),
        "max_retries": submitted_config.get("max_retries"),
    }
    return {
        "schema": "cmachine-pretrained-demo-guard-public-v1",
        "job_id": job.get("job_id"),
        "state": guard.get("state"),
        "assigned_host": guard.get("assigned_host") or job.get("assigned_host"),
        "submitted_command": redact_command(submitted.get("cmd")),
        "executed_command": redact_command(job.get("cmd")),
        "config": config,
        "plan": {key: plan.get(key) for key in ("backend", "device", "dtype", "max_batch", "threads")},
        "payload_custody": custody,
        "result": {
            "status": result.get("status"),
            "returncode": result.get("returncode"),
            "executed_payload": result.get("executed_payload"),
        },
        "redactions": ["scheduler logs and opaque base64 exports", "scheduler private admission/configuration fields"],
        "source_manifest_sha256": report.get("source_manifest_sha256"),
    }


def verify_public_manifest(target: Path) -> dict[str, Any]:
    """Verify the self-contained file index of a published bundle."""

    manifest = read_json(target / "public-manifest.json")
    require(isinstance(manifest, dict)
            and manifest.get("schema") == "cmachine-pretrained-demo-public-manifest-v1",
            "unexpected public evidence manifest schema")
    rows = manifest.get("files")
    require(isinstance(rows, list) and rows, "public evidence manifest has no files")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "public evidence manifest row is not an object")
        name = row.get("path")
        require(isinstance(name, str), "public evidence manifest row has no path")
        relative = safe_relative(name)
        require(relative.as_posix() == name, f"public evidence path is not normalized: {name!r}")
        require(name != "public-manifest.json", "public evidence manifest self-entry is not allowed")
        require(name not in indexed, f"duplicate public evidence path: {name}")
        path = target / relative
        require(path.is_file() and not path.is_symlink(),
                f"missing or non-regular public evidence file: {name}")
        require(isinstance(row.get("sha256"), str) and isinstance(row.get("bytes"), int),
                f"public evidence hash/size missing: {name}")
        require(sha256_file(path) == row["sha256"], f"public evidence hash mismatch: {name}")
        require(path.stat().st_size == row["bytes"], f"public evidence size mismatch: {name}")
        indexed[name] = row

    required = {
        "summary.json", "verification.json", "report.json", "source-manifest.json",
        "payload/source-manifest.json", "guarded-receipt.json", "raw/report.json",
        "raw/chat/report.json", "raw/operands/report.json",
    }
    require(required <= set(indexed), "public evidence bundle is missing required files")
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    require(actual == set(indexed) | {"public-manifest.json"},
            "public evidence contains an unindexed file or omits an indexed file")
    return {
        "schema": manifest["schema"],
        "files_verified": len(indexed),
        "required_files_present": True,
        "indexed_bytes": sum(int(row["bytes"]) for row in indexed.values()),
    }


def verify_public_guard(target: Path, report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Check the redacted receipt without treating it as the private proof."""

    guard = read_json(target / "guarded-receipt.json")
    require(isinstance(guard, dict)
            and guard.get("schema") == "cmachine-pretrained-demo-guard-public-v1",
            "unexpected public guarded-receipt schema")
    require(guard.get("state") == "succeeded", "public guarded receipt is not succeeded")
    require(guard.get("job_id") == report.get("job_id"),
            "public guarded receipt/report job identity differs")
    result = guard.get("result")
    require(isinstance(result, dict) and result.get("status") == "ok"
            and result.get("returncode") == 0,
            "public guarded receipt does not attest zero-exit success")
    custody = guard.get("payload_custody")
    require(isinstance(custody, dict), "public guarded receipt has no payload custody")
    rows = [custody.get(key) for key in ("declared", "sealed", "executed")]
    require(all(isinstance(row, dict) for row in rows),
            "public guarded receipt payload custody is incomplete")
    tuples = [(row.get("sha256"), row.get("size_bytes")) for row in rows]
    require(tuples[0] == tuples[1] == tuples[2],
            "public guarded receipt payload custody differs")
    executed = result.get("executed_payload")
    require(isinstance(executed, dict)
            and (executed.get("sha256"), executed.get("size_bytes")) == tuples[0],
            "public guarded receipt executed payload differs")
    config = guard.get("config")
    require(isinstance(config, dict), "public guarded receipt has no execution config")
    source_sha = sha256_file(target / "source-manifest.json")
    expected_model = manifest.get("model", {}).get("content_sha256")
    request = report.get("request")
    require(isinstance(request, dict), "public report has no invocation request")
    require(config.get("source_manifest_sha256") == source_sha == report.get("source_manifest_sha256"),
            "public receipt/source-manifest join differs")
    require(config.get("model_digest") == expected_model == request.get("model_digest"),
            "public receipt/model digest join differs")
    require(config.get("binary_sha256") == request.get("binary_sha256"),
            "public receipt/native binary digest differs")
    require(config.get("operand_binary_sha256") == request.get("operand_binary_sha256"),
            "public receipt/operand binary digest differs")
    require(config.get("backend") == "hf" and config.get("device") == "cuda",
            "public receipt does not bind the declared HF/CUDA execution")
    submitted = guard.get("submitted_command")
    executed_command = guard.get("executed_command")
    require(isinstance(submitted, list) and isinstance(executed_command, list)
            and submitted == executed_command
            and any("worker.py" in str(part) for part in executed_command),
            "public receipt command attestation is incomplete")
    redactions = guard.get("redactions")
    require(isinstance(redactions, list) and redactions,
            "public receipt does not state its redactions")
    return {
        "status": "redacted_attestation",
        "job_id": guard["job_id"],
        "state": guard["state"],
        "returncode": result["returncode"],
        "payload_custody_exact": True,
        "cuda_hf_config_bound": True,
        "command_attested": True,
        "redactions": redactions,
        "scope": "public bundle consistency; full guarded/log evidence was verified before publication",
    }


def build_summary(
    target: Path,
    report: dict[str, Any],
    scheduler: dict[str, Any],
    model: dict[str, Any],
    saturn: dict[str, Any],
    chat_summary: dict[str, Any],
    lexical_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "cmachine-pretrained-demo-summary-v1",
        "job_id": scheduler["job_id"],
        "source_manifest_sha256": report.get("source_manifest_sha256"),
        "model": {"content_sha256": model["model_digest"], "identity": model["model_identity"]},
        "fresh_chat": chat_summary,
        "lexical": lexical_summary,
        "claims": [
            "The guarded C/CUDA demo completed with exact source and custody joins.",
            "The chat branch/replay protocol checks passed for the recorded native run.",
            "The lexical program reproduced its source-covered own-token rows with exact full-logit, cache, and token checks.",
        ],
        "limits": [
            "Generated text is recorded; no semantic color, arithmetic, or prose truth claim is made.",
            "Lexical coverage is source-derived own-token reuse, not held-out generalization.",
            "Original dense weights and the remaining model computation remain paid.",
            "No speedup, universal compact state, or forward-pass-free chat claim is certified.",
            "The Saturn receipt covers the whole C invocation; internal CUDA operations are covered by the demo reports and raw hashes.",
        ],
        "terminal_status": "not_requested",
        "verification_scope": "provenance and mechanics only; semantic outcomes remain observations",
    }


def verify_public(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck a published bundle without private scheduler files."""

    target = target.resolve()
    public_manifest = verify_public_manifest(target)
    report = read_json(target / "report.json")
    manifest = read_json(target / "source-manifest.json")
    payload_manifest = read_json(target / "payload/source-manifest.json")
    require(isinstance(report, dict) and report.get("schema") == "cmachine-fresh-demos-v1",
            "unexpected public worker report schema")
    require(isinstance(manifest, dict) and manifest.get("schema") == "cmachine-demo-sources-v1",
            "unexpected public source-manifest schema")
    require(payload_manifest == manifest, "public source-manifest copies differ")

    source = verify_source_payload(target, manifest, report,
                                   payload=target / "payload",
                                   manifest_path=target / "source-manifest.json")
    raw = verify_raw_files(target, report)
    receipt = verify_public_guard(target, report, manifest)
    model = verify_model_and_binaries(target, report, manifest)
    saturn = verify_saturn_receipt(target, report)
    chat_checks, chat_summary = verify_chat(report)
    lexical_checks, lexical_summary = verify_lexical(target, report)
    pseudo_scheduler = {"job_id": report.get("job_id")}
    recomputed_summary = build_summary(target, report, pseudo_scheduler, model, saturn,
                                       chat_summary, lexical_summary)
    saved_summary = read_json(target / "summary.json")
    require(saved_summary == recomputed_summary,
            "published summary does not match recomputed public evidence")
    saved_verification = read_json(target / "verification.json")
    for key, expected in {
        "schema": "cmachine-pretrained-demo-verification-v1",
        "job_id": report.get("job_id"),
        "report_sha256": sha256_file(target / "report.json"),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source": source,
        "raw": raw,
        "model_and_binaries": model,
        "saturn": saturn,
        "chat": chat_checks,
        "lexical": lexical_checks,
    }.items():
        require(saved_verification.get(key) == expected,
                f"published verification field differs: {key}")

    verification = {
        "schema": "cmachine-pretrained-demo-public-consistency-v1",
        "status": "verified-public-bundle",
        "mechanics_status": "valid",
        "instrument_status": "valid",
        "trend_status": "observational_only",
        "terminal_status": "not_requested",
        "job_id": report.get("job_id"),
        "report_sha256": sha256_file(target / "report.json"),
        "public_manifest": public_manifest,
        "source": source,
        "raw": raw,
        "scheduler": receipt,
        "model_and_binaries": model,
        "saturn": saturn,
        "chat": chat_checks,
        "lexical": lexical_checks,
        "scope": "public bundle consistency; scheduler receipt is a redacted attestation; full guarded/log evidence was verified before publication",
    }
    return verification, recomputed_summary


def copy_public(target: Path, destination: Path, summary: dict[str, Any], verification: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    target = target.resolve()
    destination = destination.resolve()
    require(destination != target and target not in destination.parents,
            "publish destination cannot be the collected run or one of its children")
    require(not destination.exists() or not any(destination.iterdir()),
            f"publish destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []

    def copy_file(source: Path, relative: str) -> None:
        relative_path = safe_relative(relative)
        output = destination / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        require(source.is_file() and not source.is_symlink(), f"cannot publish missing file: {source}")
        shutil.copyfile(source, output)
        copied.append({"path": relative_path.as_posix(), "sha256": sha256_file(output), "bytes": output.stat().st_size})

    copy_file(target / "summary.json", "summary.json")
    copy_file(target / "verification.json", "verification.json")
    copy_file(target / "report.json", "report.json")
    copy_file(target / "payload/source-manifest.json", "source-manifest.json")
    copy_file(target / "payload/source-manifest.json", "payload/source-manifest.json")
    (destination / "raw").mkdir(exist_ok=True)
    for path in sorted((target / "raw").rglob("*")):
        if path.is_file() and not path.is_symlink():
            copy_file(path, "raw/" + path.relative_to(target / "raw").as_posix())

    manifest = read_json(target / "payload/source-manifest.json")
    payload_bytes = 0
    for name in sorted(manifest["sources"]):
        source = target / "payload" / safe_relative(name)
        payload_bytes += source.stat().st_size
        require(payload_bytes <= PUBLIC_PAYLOAD_LIMIT,
                f"public frozen source payload exceeds {PUBLIC_PAYLOAD_LIMIT} bytes")
        copy_file(source, "payload/" + safe_relative(name).as_posix())

    (destination / "guarded-receipt.json").write_text(json.dumps(guard, indent=2, sort_keys=True) + "\n")
    copied.append({"path": "guarded-receipt.json", "sha256": sha256_file(destination / "guarded-receipt.json"),
                   "bytes": (destination / "guarded-receipt.json").stat().st_size})
    public_manifest = {
        "schema": "cmachine-pretrained-demo-public-manifest-v1",
        "files": copied,
        "excluded": ["logs.txt", "job.json", "submission.json", "full guarded-receipt.json"],
        "source_payload_limit_bytes": PUBLIC_PAYLOAD_LIMIT,
    }
    (destination / "public-manifest.json").write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n")
    return {"destination": str(destination), "files": len(copied), "source_payload_bytes": payload_bytes}


def verify(target: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = target.resolve()
    report_path, job_path, guard_path, submission_path, manifest_path = locate_result_files(target)
    report = read_json(report_path)
    job = read_json(job_path)
    guard = read_json(guard_path)
    submission = read_json(submission_path)
    manifest = read_json(manifest_path)
    require(isinstance(report, dict) and report.get("schema") == "cmachine-fresh-demos-v1",
            "unexpected worker report schema")
    require(isinstance(manifest, dict) and manifest.get("schema") == "cmachine-demo-sources-v1",
            "unexpected source-manifest schema")

    source = verify_source_payload(target, manifest, report)
    raw = verify_raw_files(target, report)
    scheduler = verify_scheduler(target, report, job, guard, submission, manifest)
    model = verify_model_and_binaries(target, report, manifest)
    saturn = verify_saturn_receipt(target, report)
    chat_checks, chat_summary = verify_chat(report)
    lexical_checks, lexical_summary = verify_lexical(target, report)

    verification = {
        "schema": "cmachine-pretrained-demo-verification-v1",
        "status": "verified",
        "mechanics_status": "valid",
        "instrument_status": "valid",
        "trend_status": "observational_only",
        "terminal_status": "not_requested",
        "job_id": scheduler["job_id"],
        "report_sha256": sha256_file(report_path),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "source": source,
        "raw": raw,
        "scheduler": scheduler,
        "model_and_binaries": model,
        "saturn": saturn,
        "chat": chat_checks,
        "lexical": lexical_checks,
        "scope": "provenance and mechanics verification without model evaluation",
    }
    summary = build_summary(target, report, scheduler, model, saturn, chat_summary, lexical_summary)
    (target / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    (target / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return verification, summary, public_guard(job, guard, report, manifest)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true",
                        help="verify a published public bundle without private scheduler files")
    parser.add_argument("fleetdir", type=Path, help="collected demos/fleet-runs/JOB directory")
    parser.add_argument("--publish", type=Path, help="optional public evidence destination")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.public:
            require(args.publish is None, "--publish cannot be combined with --public")
            verification, summary = verify_public(args.fleetdir)
            output = {
                "status": verification["status"],
                "job_id": verification["job_id"],
                "scope": verification["scope"],
                "bundle": str(args.fleetdir.resolve()),
            }
            print(json.dumps(output, sort_keys=True))
            return 0

        verification, summary, guard = verify(args.fleetdir)
        publication = None
        if args.publish:
            publication = copy_public(args.fleetdir.resolve(), args.publish.resolve(), summary, verification, guard)
        output = {"status": verification["status"], "job_id": verification["job_id"],
                  "summary": str(args.fleetdir.resolve() / "summary.json"),
                  "verification": str(args.fleetdir.resolve() / "verification.json")}
        if publication is not None:
            output["published"] = publication
        print(json.dumps(output, sort_keys=True))
        return 0
    except (OSError, VerificationError, json.JSONDecodeError) as error:
        print(f"demo verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
