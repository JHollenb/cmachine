"""Independent saved-only verification for transition-laws results.

This module checks custody, typed shapes, token/clock authority, factor-file
lineage, and replay array geometry.  It does not execute the model and does
not turn KL, truth, or token agreement into a scientific gate.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # payload execution imports this module beside analysis.py
    from analysis import PROFILE, _as_ints, _basis_summary, _merge_run, _path_records, _resolve, _turn_answers, sha256_file, token_authority
except ImportError:  # pragma: no cover - package-style invocation
    from .analysis import PROFILE, _as_ints, _basis_summary, _merge_run, _path_records, _resolve, _turn_answers, sha256_file, token_authority


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _record_check(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    expected_hash = record.get("sha256", record.get("hash"))
    expected_bytes = record.get("bytes", record.get("size"))
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        result["status"] = "missing"
        return result
    actual_hash = sha256_file(path)
    actual_bytes = path.stat().st_size
    result.update({"sha256": actual_hash, "bytes": actual_bytes, "hash_matches": expected_hash in (None, actual_hash), "bytes_matches": expected_bytes in (None, actual_bytes), "status": "ok" if expected_hash in (None, actual_hash) and expected_bytes in (None, actual_bytes) else "mismatch"})
    return result


def _manifest_records(value: Any, prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        if ("sha256" in value or "hash" in value) and ("bytes" in value or "size" in value):
            found.append((prefix, value))
        for key, item in value.items():
            if key in {"raw_files", "files", "sources", "artifacts", "factor_files"} or isinstance(item, (Mapping, list)):
                found.extend(_manifest_records(item, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_manifest_records(item, f"{prefix}/{index}" if prefix else str(index)))
    return found


def _ref_path(root: Path, logical: str, record: Mapping[str, Any] | None = None) -> Path | None:
    if record:
        for key in ("path", "file", "filename", "relative_path"):
            if key in record:
                path = _resolve(root, record[key])
                if path:
                    return path
    path = _resolve(root, logical)
    if path:
        return path
    # Raw manifests use names relative to raw/; source manifests use names
    # relative to the payload root.  _resolve already checks both, but this
    # explicit fallback helps old collected archives with a leading prefix.
    for base in (root / "raw", root / "payload", root):
        candidate = base / logical.lstrip("/")
        if candidate.is_file():
            return candidate.resolve()
    return None


def _finite_file(path: Path, dtype: str = "<f4", chunk_values: int = 1 << 20) -> tuple[bool, int]:
    try:
        values = np.memmap(path, dtype=dtype, mode="r")
    except (OSError, ValueError):
        return False, 0
    count = int(values.size)
    for start in range(0, count, chunk_values):
        if not np.isfinite(values[start : min(start + chunk_values, count)]).all():
            return False, count
    return True, count


def _verify_logits(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    answers, _, positions = _turn_answers(record)
    expected_rows = len(answers)
    vocab = int(record.get("vocab", PROFILE["vocab"]))
    result: dict[str, Any] = {"expected_rows": expected_rows, "vocab": vocab, "files": {}}
    for kind, names in (("candidate", ("candidate_logits_file", "candidate_decision_logits_file", "candidate_logits", "candidate_logit_file")), ("shadow", ("shadow_logits_file", "native_shadow_logits_file", "shadow_logits", "native_logits_file"))):
        reference = None
        for name in names:
            if name in record:
                reference = record[name]
                break
        if reference is None and isinstance(record.get("files"), Mapping):
            reference = next((record["files"].get(name) for name in names if name in record["files"]), None)
        path = _resolve(root, reference)
        entry: dict[str, Any] = {"reference": reference, "path": str(path.relative_to(root)) if path else None, "status": "missing"}
        if path is not None:
            expected_bytes = expected_rows * vocab * 4
            finite, values = _finite_file(path)
            entry.update({"status": "ok" if values == expected_rows * vocab and finite else "mismatch", "bytes": path.stat().st_size, "expected_bytes": expected_bytes, "values": values, "finite": finite, "shape": [expected_rows, vocab], "sha256": sha256_file(path)})
        result["files"][kind] = entry
    result["decision_positions"] = positions
    result["decision_positions_expected"] = [int(x) for x in positions] if positions else []
    result["position_count_matches"] = len(positions) in (0, expected_rows)
    return result


def _input_budget(root: Path, case_id: Any, turn_index: int) -> int | None:
    case = str(case_id)
    candidates = [root / "raw" / "source-inputs.json", root / "raw" / "monitor-inputs.json", root / "raw" / "heldout-inputs.json", root / "payload" / "inputs.json", root / "inputs.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            groups = value.values()
        elif isinstance(value, list):
            groups = (value,)
        else:
            continue
        for group in groups:
            if not isinstance(group, list):
                continue
            for item in group:
                if isinstance(item, Mapping) and str(item.get("id")) == case:
                    try:
                        return int(item.get("budget"))
                    except (TypeError, ValueError):
                        pass
    return None


def _verify_turns(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    answers, _, positions = _turn_answers(record)
    turns = record.get("turns", [])
    if isinstance(turns, Mapping):
        turns = list(turns.values())
    errors: list[str] = []
    clock_count = 0
    for turn_index, turn in enumerate(turns if isinstance(turns, list) else []):
        if not isinstance(turn, Mapping):
            errors.append(f"turn {turn_index} is not an object")
            continue
        ids = [int(x) for x in turn.get("answer_ids", turn.get("generated_ids", []))]
        prefill = turn.get("prefill")
        if isinstance(prefill, Mapping):
            prefill = prefill.get("position", prefill.get("count"))
        try:
            prefill_int = int(prefill)
        except (TypeError, ValueError):
            prefill_int = None
        explicit = turn.get("decision_positions")
        expected_minus_one = list(range(prefill_int - 1, prefill_int + len(ids) - 1)) if prefill_int is not None else []
        expected_boundary = list(range(prefill_int, prefill_int + len(ids))) if prefill_int is not None else []
        if explicit is not None and [int(x) for x in explicit] not in (expected_minus_one, expected_boundary):
            errors.append(f"turn {turn_index} decision positions are not contiguous at the prefill boundary")
        if prefill_int is None:
            errors.append(f"turn {turn_index} has no prefill position")
        budget = turn.get("budget", record.get("budget"))
        # Older C labels left TurnInfo.budget at zero even though the RUN
        # command carried the correct case budget.  Derive that declared
        # input budget for verification; never rewrite the raw label.
        if budget in (None, 0):
            budget = _input_budget(root, record.get("case_id", record.get("case")), turn_index)
        if budget is not None and len(ids) > int(budget):
            errors.append(f"turn {turn_index} exceeds budget")
        if turn.get("stopped") is True and turn.get("truncated") is True:
            errors.append(f"turn {turn_index} both stopped and truncated")
        clock_count += len(ids)
    if clock_count != len(answers):
        errors.append("flattened answer count differs")
    return {"turn_count": len(turns) if isinstance(turns, list) else 0, "answer_count": len(answers), "decision_count": len(positions), "errors": errors, "status": "valid" if not errors else "invalid", "budget_source": "turn-label-or-input-manifest"}


def _verify_token_files(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Join consumed/tops arrays to turns and the candidate's greedy IDs."""
    consumed = _as_ints(record.get("consumed_ids", record.get("token_ids")))
    if not consumed:
        ref = record.get("consumed_ids_file", record.get("token_ids_file"))
        path = _resolve(root, ref)
        if path is not None:
            try:
                consumed = [int(x) for x in np.fromfile(path, dtype="<i4").tolist()]
            except (OSError, ValueError):
                consumed = []
    tops = _as_ints(record.get("tops"))
    if not tops:
        path = _resolve(root, record.get("tops_file"))
        if path is not None:
            try:
                tops = [int(x) for x in np.fromfile(path, dtype="<i4").tolist()]
            except (OSError, ValueError):
                tops = []
    answers, _, _ = _turn_answers(record)
    errors: list[str] = []
    if not consumed:
        errors.append("consumed token array unavailable")
    if not tops:
        errors.append("greedy top-token array unavailable")
    if consumed and int(record.get("count", len(consumed))) != len(consumed):
        errors.append("consumed array length differs from count")
    if tops and len(tops) != len(consumed):
        errors.append("tops array length differs from consumed array")
    turns = record.get("turns", [])
    if isinstance(turns, Mapping):
        turns = list(turns.values())
    for turn_index, turn in enumerate(turns if isinstance(turns, list) else []):
        if not isinstance(turn, Mapping):
            continue
        ids = _as_ints(turn.get("answer_ids", turn.get("generated_ids")))
        try:
            prefill = int(turn.get("prefill"))
            final = int(turn.get("final"))
        except (TypeError, ValueError):
            continue
        if consumed and not (0 <= prefill <= final <= len(consumed)):
            errors.append(f"turn {turn_index} boundary outside consumed history")
            continue
        if consumed and consumed[prefill:final] != ids:
            errors.append(f"turn {turn_index} answer IDs differ from consumed suffix")
        # tops[i] is the pending next ID after consuming token i.  The first
        # decision is therefore at the last prompt token, prefill-1; later
        # decisions follow each consumed answer token.  This check is
        # independent of whether the public boundary clock is written as
        # prefill-1+j or prefill+j.
        if tops and ids:
            source = [tops[prefill - 1 + j] for j in range(len(ids)) if 0 <= prefill - 1 + j < len(tops)]
            if source != ids:
                errors.append(f"turn {turn_index} tops do not reproduce answer IDs")
    return {"consumed_count": len(consumed), "tops_count": len(tops), "answer_count": len(answers), "consumed_sha256": hashlib.sha256(np.asarray(consumed, dtype="<i4").tobytes()).hexdigest() if consumed else None, "tops_sha256": hashlib.sha256(np.asarray(tops, dtype="<i4").tobytes()).hexdigest() if tops else None, "errors": errors, "status": "valid" if not errors else "invalid"}


def _verify_boundaries(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    boundaries = record.get("boundaries", {})
    if not isinstance(boundaries, Mapping):
        return {"status": "unavailable", "files": []}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for stage, value in boundaries.items():
        if not isinstance(value, Mapping):
            continue
        for field, reference in value.items():
            if not isinstance(reference, (str, Mapping)) or not reference:
                continue
            path = _resolve(root, reference)
            item: dict[str, Any] = {"stage": stage, "field": field, "path": str(path.relative_to(root)) if path else None}
            if path is None:
                item["status"] = "missing"
                errors.append(f"{stage}/{field} missing")
                rows.append(item)
                continue
            finite, values = _finite_file(path)
            expected = PROFILE["vocab"] if "logit" in field else PROFILE["layers"] * PROFILE["hidden"]
            item.update({"bytes": path.stat().st_size, "values": values, "expected_values": expected, "finite": finite, "sha256": sha256_file(path), "status": "ok" if finite and values == expected else "invalid"})
            if item["status"] != "ok":
                errors.append(f"{stage}/{field} shape or finiteness")
            rows.append(item)
    return {"status": "valid" if not errors else "invalid", "files": rows, "errors": errors}


def _verify_counters(record: Mapping[str, Any]) -> dict[str, Any]:
    """Check the declared primary physical-work identity per consumed token."""
    try:
        count = int(record.get("count", 0))
        layers = int(record.get("layers", PROFILE["layers"]))
        hidden = int(record.get("hidden", PROFILE["hidden"]))
        intermediate = int(record.get("intermediate", PROFILE["intermediate"]))
        rank = int(record.get("rank", PROFILE["rank"]))
        mask = int(record.get("mask", 0))
    except (TypeError, ValueError):
        return {"status": "unavailable", "errors": ["missing counter geometry"]}
    selected = (mask & ((1 << layers) - 1)).bit_count()
    expected = {
        "linear_calls": count * (197 + selected),
        "linear_macs": count * (1543569408 - selected * (2 * hidden * intermediate - rank * (hidden + 2 * intermediate))),
        "native_gate_up_calls": count * 2 * (layers - selected),
        "compiled_gate_up_calls": count * 2 * selected,
        "mlp_coordinate_calls": count * selected,
    }
    actual = {key: record.get(key) for key in expected}
    errors: list[str] = []
    for key, wanted in expected.items():
        if actual[key] is not None:
            try:
                if int(actual[key]) != wanted:
                    errors.append(f"{key}: expected {wanted}, got {actual[key]}")
            except (TypeError, ValueError):
                errors.append(f"{key}: noninteger {actual[key]!r}")
    return {"status": "valid" if not errors else "invalid", "count": count, "selected_layers": selected, "expected": expected, "actual": actual, "errors": errors}


def _receipt_status(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        return {"status": "missing"}
    body = receipt.get("receipt") if isinstance(receipt.get("receipt"), Mapping) else receipt
    if not isinstance(body, Mapping) or "fingerprint" not in body:
        return {"status": "present", "fingerprint_checked": False}
    copy_value = dict(body)
    wanted = copy_value.pop("fingerprint")
    encoded = json.dumps(copy_value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    boundary_ok = True
    if isinstance(receipt.get("before"), str) or isinstance(receipt.get("after"), str):
        boundary_ok = body.get("parent") == receipt.get("before") and body.get("result") == receipt.get("after")
    return {"status": "valid" if actual == wanted and boundary_ok else "invalid", "fingerprint_checked": True, "boundary_matches": boundary_ok, "expected": wanted, "actual": actual}


def _verify_source_manifest(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    expected = report.get("source_manifest_sha256")
    path = _resolve(root, "source-manifest.json")
    if path is None:
        return {"status": "missing", "expected_sha256": expected}
    actual = sha256_file(path)
    result: dict[str, Any] = {"path": str(path.relative_to(root)), "expected_sha256": expected, "actual_sha256": actual, "status": "valid" if expected in (None, actual) else "invalid"}
    try:
        manifest = _load(path)
    except (OSError, json.JSONDecodeError):
        result["manifest_json"] = "invalid"
        return result
    result["manifest_json"] = "valid"
    checks: list[dict[str, Any]] = []
    sources = manifest.get("sources", {}) if isinstance(manifest, Mapping) else {}
    if isinstance(sources, Mapping):
        for logical, wanted in sources.items():
            p = _resolve(root, f"payload/{logical}") or _resolve(root, logical)
            checks.append({"logical": logical, **(_record_check(p, {"sha256": wanted}) if p else {"status": "missing", "path": None})})
    result["source_entries"] = checks
    result["source_entries_valid"] = all(row.get("status") == "ok" for row in checks)
    return result


def verify(target: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(target).resolve()
    report_path = root / "report.json"
    failures: list[str] = []
    if not report_path.is_file():
        result = {"schema": "transition-laws-verification-v1", "mechanics_status": "invalid", "failures": ["missing report.json"]}
        (root / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    try:
        report = _load(report_path)
    except (OSError, json.JSONDecodeError) as error:
        result = {"schema": "transition-laws-verification-v1", "mechanics_status": "invalid", "failures": [str(error)]}
        (root / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    raw_checks: list[dict[str, Any]] = []
    raw_records = report.get("raw_files", {})
    if not isinstance(raw_records, Mapping):
        failures.append("report.raw_files is absent or not a mapping")
    else:
        for logical, record in raw_records.items():
            if not isinstance(record, Mapping):
                failures.append(f"raw record {logical} is not an object")
                continue
            path = _ref_path(root, str(logical), record)
            check = _record_check(path, record) if path else {"path": None, "status": "missing"}
            check["logical"] = logical
            raw_checks.append(check)
            if check.get("status") != "ok":
                failures.append(f"raw artifact {logical}: {check.get('status')}")

    source_manifest = _verify_source_manifest(root, report)
    if source_manifest.get("status") != "valid" or source_manifest.get("manifest_json") != "valid" or source_manifest.get("source_entries_valid") is False:
        failures.append("source manifest is missing, malformed, or differs")
    receipt = _receipt_status(report.get("saturn_receipt"))
    if receipt.get("status") != "valid":
        failures.append("Saturn receipt is missing or fingerprint differs")

    run_checks: dict[str, Any] = {}
    runs = report.get("runs", {})
    if not isinstance(runs, Mapping):
        failures.append("report.runs is absent or not a mapping")
        runs = {}
    for group_label, value in runs.items():
        group_driver = value.get("driver") if isinstance(value, Mapping) and isinstance(value.get("driver"), Mapping) else value if isinstance(value, Mapping) else {}
        profile_errors = []
        for key, expected in (("layers", PROFILE["layers"]), ("hidden", PROFILE["hidden"]), ("intermediate", PROFILE["intermediate"]), ("vocab", PROFILE["vocab"]), ("rank", PROFILE["rank"])):
            if key in group_driver and int(group_driver[key]) != expected:
                profile_errors.append(f"{key}={group_driver[key]} expected {expected}")
        if profile_errors:
            failures.append(f"run group {group_label} profile: {'; '.join(profile_errors)}")
        for label, record, _group_metrics in _path_records(str(group_label), value):
            turn_check = _verify_turns(root, record)
            logits_check = _verify_logits(root, record)
            authority = token_authority(record, root)
            token_files = _verify_token_files(root, record)
            counters = _verify_counters(record)
            boundaries = _verify_boundaries(root, record)
            run_check = {"case_id": record.get("case_id", record.get("case")), "group_label": str(group_label), "policy": record.get("policy", record.get("program", group_label)), "turns": turn_check, "logits": logits_check, "token_authority": authority, "token_files": token_files, "boundaries": boundaries, "counters": counters, "status": "valid" if turn_check["status"] == "valid" and token_files["status"] == "valid" and boundaries["status"] in ("valid", "unavailable") and authority.get("consumed_count", 0) > 0 and all(item.get("status") == "ok" for item in logits_check["files"].values()) and counters["status"] in ("valid", "unavailable") else "invalid"}
            run_checks[str(label)] = run_check
            if run_check["status"] == "invalid":
                failures.append(f"run {label} typed geometry")
            if counters.get("status") == "invalid":
                failures.append(f"run {label} physical counter identity")

    basis = _basis_summary(root, report)
    for program, item in basis.get("programs", {}).items():
        if item.get("status") != "ok" or not item.get("all_layer_files_present", False):
            failures.append(f"factor files incomplete for {program}")
    for name, item in basis.get("sampling", {}).items():
        if item.get("status") != "ok" or not item.get("lineage_rows_64_each", False) or item.get("shape_matches") is False or item.get("selected_hash_matches") is False:
            failures.append(f"sampling lineage incomplete for {name}")

    # The source/monitor/heldout split is a custody observation unless the
    # producer supplied explicit memberships.  Report overlap rather than
    # silently deciding that an absent split is a failure.
    split_sets: dict[str, set[str]] = {}
    for split in ("source", "monitor", "heldout"):
        values = report.get(f"{split}_cases", report.get("inputs", {}).get(split, []) if isinstance(report.get("inputs"), Mapping) else [])
        if isinstance(values, list):
            split_sets[split] = {str(item.get("id", item)) if isinstance(item, Mapping) else str(item) for item in values}
    overlaps = {f"{a}:{b}": sorted(split_sets.get(a, set()) & split_sets.get(b, set())) for a in split_sets for b in split_sets if a < b}
    if any(overlaps.values()):
        failures.append("source/monitor/heldout case memberships overlap")

    result = {
        "schema": "transition-laws-verification-v1",
        "report_sha256": sha256_file(report_path),
        "raw_files_verified": len(raw_checks),
        "raw_checks": raw_checks,
        "source_manifest": source_manifest,
        "saturn_receipt": receipt,
        "runs": run_checks,
        "basis_verification": basis,
        "split_overlaps": overlaps,
        "failures": failures,
        "mechanics_status": "valid" if not failures else "invalid",
        "instrument_status": report.get("instrument_status", "unknown"),
        "trend_status": report.get("trend_status", "observations"),
        "terminal_status": "not-requested",
        "scope": "Saved-only report/raw/source custody, factor rank and hash lineage, token/clock authority, and independent logits array geometry; semantic metrics remain observations.",
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def exportverify(targetdir: str | os.PathLike[str]) -> dict[str, Any]:
    """Public API used by the worker/collector and by offline reviewers."""
    return verify(targetdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(exportverify(args.target), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
