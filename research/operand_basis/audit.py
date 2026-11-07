"""Independent, saved-only audit for an operand-basis C run.

This file intentionally does not import the producer's ``analysis`` module.
It reads the immutable driver ledger and the archived arrays, then recomputes
the custody, call, token, parity, and capture-geometry checks.  It never
changes ``report.json`` or any executed artifact.  The audit is useful for a
small smoke and for the full run because all dimensions are read from the
driver's recorded profile.

Usage::

    python audit.py --run runs/job-23dd9b4d0de9

The output is ``audit.json`` and ``AUDIT.md`` beside the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


HIDDEN = 1536
INTERMEDIATE = 8960
VOCAB = 151936
COUNTERS = (
    "basis_q_calls",
    "basis_g_calls",
    "basis_u_calls",
    "selected_gate_up_calls",
    "selected_down_calls",
)
FACTOR_FILES = {
    "source_qt": "source-QT.f32",
    "source_gq": "source-GQ.f32",
    "source_uq": "source-UQ.f32",
    "random_qt": "random-QT.f32",
    "random_gq": "random-GQ.f32",
    "random_uq": "random-UQ.f32",
}
FACTOR_COUNTS = {
    "source_qt": 64 * HIDDEN,
    "source_gq": INTERMEDIATE * 64,
    "source_uq": INTERMEDIATE * 64,
    "random_qt": 64 * HIDDEN,
    "random_gq": INTERMEDIATE * 64,
    "random_uq": INTERMEDIATE * 64,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _path_file(raw: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else raw / path


def array_meta(path: Path, expected_count: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"file": path.name, "status": "missing"}
    data = path.read_bytes()
    result: dict[str, Any] = {
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "status": "ok",
    }
    if len(data) % 4:
        result.update(status="invalid_bytes", count=None)
        return result
    values = np.frombuffer(data, dtype="<f4")
    result["count"] = int(values.size)
    result["finite"] = bool(np.isfinite(values).all())
    if expected_count is not None and values.size != expected_count:
        result["status"] = "shape_mismatch"
        result["expected_count"] = int(expected_count)
    elif not result["finite"]:
        result["status"] = "nonfinite"
    return result


def byte_difference(left: Path, right: Path) -> dict[str, Any]:
    if not left.is_file() or not right.is_file():
        return {
            "status": "missing",
            "left": left.name,
            "right": right.name,
            "left_present": left.is_file(),
            "right_present": right.is_file(),
        }
    lb = left.read_bytes()
    rb = right.read_bytes()
    result: dict[str, Any] = {
        "status": "ok",
        "left": left.name,
        "right": right.name,
        "left_sha256": hashlib.sha256(lb).hexdigest(),
        "right_sha256": hashlib.sha256(rb).hexdigest(),
        "byte_exact": lb == rb,
        "left_bytes": len(lb),
        "right_bytes": len(rb),
    }
    if len(lb) != len(rb) or len(lb) % 4:
        result["status"] = "shape_mismatch"
        return result
    if lb == rb:
        result.update(max_abs=0.0, l2=0.0, count=len(lb) // 4)
        return result
    left_a = np.frombuffer(lb, dtype="<f4").astype(np.float64)
    right_a = np.frombuffer(rb, dtype="<f4").astype(np.float64)
    delta = left_a - right_a
    result.update(
        count=int(left_a.size),
        max_abs=float(np.max(np.abs(delta))) if delta.size else 0.0,
        l2=float(np.linalg.norm(delta)),
        finite=bool(np.isfinite(left_a).all() and np.isfinite(right_a).all()),
    )
    result["status"] = "divergent"
    return result


def primary_paths(driver: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result = []
    for case in driver.get("primary_cases", []):
        if not isinstance(case, Mapping):
            continue
        for path in case.get("arms", []):
            if isinstance(path, Mapping):
                result.append((case, path))
    return result


def reference_map(driver: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for record in driver.get("references", []):
        if isinstance(record, Mapping):
            result[_int(record.get("case"))] = record
    return result


def steps(path: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in path.get("steps", []) if isinstance(row, Mapping)]


def step_token(row: Mapping[str, Any]) -> int | None:
    value = row.get("input_id", row.get("token", row.get("id")))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def path_ops(path: Mapping[str, Any]) -> dict[str, int]:
    rows = steps(path)
    result = {key: sum(_int(row.get(key)) for row in rows) for key in COUNTERS}
    result["steps"] = len(rows)
    result["derived_projection_calls"] = result["basis_q_calls"] + result["basis_g_calls"] + result["basis_u_calls"]
    result["known_selected_matvec_calls"] = result["selected_gate_up_calls"] + result["selected_down_calls"]
    return result


def path_expected_tokens(case: Mapping[str, Any], path: Mapping[str, Any]) -> list[int]:
    answer0 = path.get("answer0", {})
    answer1 = path.get("answer1", {})
    a0 = list(answer0.get("ids", [])) if isinstance(answer0, Mapping) else []
    a1 = list(answer1.get("ids", [])) if isinstance(answer1, Mapping) else []
    follow = list(case.get("followup_tokens", case.get("followup_ids", [])))
    skip = _int(path.get("followup_skip"))
    return [int(x) for x in list(case.get("prompt_tokens", case.get("prompt_ids", []))) + a0 + follow[skip:] + a1]


def token_record(case: Mapping[str, Any], path: Mapping[str, Any], role: str) -> dict[str, Any]:
    actual = [step_token(row) for row in steps(path)]
    expected = path_expected_tokens(case, path)
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    complete = len(actual) == len(expected) and mismatch is None and all(x is not None for x in actual)
    positions = [_int(row.get("position"), i) for i, row in enumerate(steps(path))]
    position_ok = positions == list(range(len(positions)))
    consumed = path.get("consumed_ids")
    consumed_match = consumed is None or [int(x) for x in consumed] == [int(x) for x in actual if x is not None]
    return {
        "role": role,
        "path": path.get("name"),
        "case": case.get("case"),
        "status": "ok" if complete and position_ok and consumed_match else "mismatch",
        "actual_count": len(actual),
        "expected_count": len(expected),
        "mismatch_index": mismatch,
        "position_ok": position_ok,
        "consumed_ids_match_steps": consumed_match,
        "all_delivered_consumed": bool(path.get("all_delivered_consumed", False)),
        "actual_prefix": actual[:8],
        "expected_prefix": expected[:8],
    }


def expected_calls(arm: str, count: int) -> dict[str, int] | None:
    if arm in ("source64", "random64"):
        return {"basis_q_calls": count, "basis_g_calls": count, "basis_u_calls": count, "selected_gate_up_calls": 0, "selected_down_calls": count}
    if arm == "b3":
        return {"basis_q_calls": 0, "basis_g_calls": 0, "basis_u_calls": 0, "selected_gate_up_calls": 0, "selected_down_calls": count}
    if arm in ("zero", "deletion"):
        return {"basis_q_calls": 0, "basis_g_calls": 0, "basis_u_calls": 0, "selected_gate_up_calls": 0, "selected_down_calls": 0}
    if arm in ("native", "native_own", "native_replay"):
        return {"basis_q_calls": 0, "basis_g_calls": 0, "basis_u_calls": 0, "selected_gate_up_calls": 2 * count, "selected_down_calls": count}
    return None


def call_record(path: Mapping[str, Any], role: str) -> dict[str, Any]:
    observed = path_ops(path)
    count = observed["steps"]
    arm = str(path.get("arm", "native_replay" if role == "same_id" else ""))
    expected = expected_calls(arm, count)
    mismatches = {}
    if expected is not None:
        mismatches = {key: {"expected": value, "observed": observed[key]} for key, value in expected.items() if observed[key] != value}
    return {
        "role": role,
        "path": path.get("name"),
        "arm": arm,
        "steps": count,
        "observed": observed,
        "expected": expected,
        "status": "ok" if expected is None or not mismatches else "mismatch",
        "mismatches": mismatches,
    }


def factor_audit(raw: Path, driver: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    compiled = raw / "compiled"
    expected = driver.get("basis_file_sha256", {})
    old_expected = driver.get("basis_sha256", {})
    bank_path = compiled / "bank_manifest.json"
    source_input_path = compiled / "source-input-manifest.json"
    bank = load_json(bank_path) if bank_path.is_file() else {}
    source_input = load_json(source_input_path) if source_input_path.is_file() else {}
    bank_names = {
        "source_qt": ("source", "q_t"),
        "source_gq": ("source", "gate"),
        "source_uq": ("source", "up"),
        "random_qt": ("matched_random", "q_t"),
        "random_gq": ("matched_random", "gate"),
        "random_uq": ("matched_random", "up"),
    }
    checks = []
    for key, filename in FACTOR_FILES.items():
        path = compiled / filename
        meta = array_meta(path, FACTOR_COUNTS[key])
        actual = meta.get("sha256")
        record = bank.get("banks", {}).get(bank_names[key][0], {}).get(bank_names[key][1], {}) if isinstance(bank, Mapping) else {}
        manifest_hash = record.get("sha256") if isinstance(record, Mapping) else None
        checks.append({
            "key": key,
            "file": filename,
            "observed_sha256": actual,
            "driver_basis_file_sha256": expected.get(key) if isinstance(expected, Mapping) else None,
            "driver_basis_sha256": old_expected.get(key) if isinstance(old_expected, Mapping) else None,
            "bank_manifest_sha256": manifest_hash,
            "byte_count": meta.get("bytes"),
            "count": meta.get("count"),
            "finite": meta.get("finite"),
            "all_hashes_match": bool(actual and actual == (expected.get(key) if isinstance(expected, Mapping) else None) and actual == (old_expected.get(key) if isinstance(old_expected, Mapping) else None) and actual == manifest_hash),
        })
    source_export = report.get("source_export", {}) if isinstance(report, Mapping) else {}
    source_joins = []
    for source_key, input_key in (("G", "G"), ("U", "U"), ("gamma", "gamma")):
        exp = source_export.get(source_key, {}) if isinstance(source_export, Mapping) else {}
        inp = source_input.get(input_key, {}) if isinstance(source_input, Mapping) else {}
        source_joins.append({
            "source_export": source_key,
            "source_input": input_key,
            "source_export_sha256": exp.get("sha256") if isinstance(exp, Mapping) else None,
            "source_input_sha256": inp.get("sha256") if isinstance(inp, Mapping) else None,
            "match": bool(isinstance(exp, Mapping) and isinstance(inp, Mapping) and exp.get("sha256") and exp.get("sha256") == inp.get("sha256")),
        })
    source_model = source_input.get("model_digest") if isinstance(source_input, Mapping) else None
    report_model = report.get("model", {}) if isinstance(report, Mapping) else {}
    model_claims = {
        "driver_model_digest": driver.get("model_digest"),
        "source_input_model_digest": source_model,
        "report_model_content_sha256": report_model.get("content_sha256") if isinstance(report_model, Mapping) else None,
        "all_match": bool(driver.get("model_digest") and driver.get("model_digest") == source_model == (report_model.get("content_sha256") if isinstance(report_model, Mapping) else None)),
    }
    compiled_hashes = {
        "bank_manifest": {"expected": driver.get("bank_manifest_sha256"), "observed": sha256(bank_path) if bank_path.is_file() else None},
        "source_input_manifest": {"expected": driver.get("input_manifest_sha256"), "observed": sha256(source_input_path) if source_input_path.is_file() else None},
    }
    return {
        "compiled_directory": "compiled",
        "checks": checks,
        "all_factor_hashes_match": bool(checks) and all(item["all_hashes_match"] for item in checks),
        "compiled_manifest_hashes": compiled_hashes,
        "compiled_manifest_hashes_match": all(item["expected"] and item["expected"] == item["observed"] for item in compiled_hashes.values()),
        "source_export_input_joins": source_joins,
        "source_export_input_joins_match": bool(source_joins) and all(item["match"] for item in source_joins),
        "source_operands_bytes_retained": {name: (compiled / name).is_file() for name in ("G.f32", "U.f32", "gamma.f32")},
        "model_claims": model_claims,
    }


def seal_audit(raw: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    driver_path = raw / "driver.json"
    seal_path = raw / "primary-seal.json"
    if not driver_path.is_file() or not seal_path.is_file():
        return {"status": "missing", "file_count": 0}
    seal = load_json(seal_path)
    data = driver_path.read_bytes()
    prefix_bytes = _int(driver.get("primary_seal_bytes"), -1)
    observed_prefix = hashlib.sha256(data[:prefix_bytes]).hexdigest() if 0 <= prefix_bytes <= len(data) else None
    records = seal.get("files", []) if isinstance(seal, Mapping) else []
    checks = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping):
            continue
        path = raw / str(record.get("name", ""))
        observed = sha256(path) if path.is_file() else None
        checks.append({
            "name": record.get("name"),
            "expected_sha256": record.get("sha256"),
            "observed_sha256": observed,
            "bytes_expected": _int(record.get("bytes"), -1),
            "bytes_observed": path.stat().st_size if path.is_file() else None,
            "match": bool(path.is_file() and observed == record.get("sha256") and path.stat().st_size == _int(record.get("bytes"), -1)),
        })
    file_match = bool(checks) and all(item["match"] for item in checks)
    prefix_match = bool(observed_prefix and observed_prefix == driver.get("primary_seal_sha256") and observed_prefix == seal.get("driver_prefix_sha256"))
    return {
        "status": "ok" if prefix_match and file_match else "mismatch",
        "prefix": {"bytes": prefix_bytes, "driver_bytes": len(data), "observed_sha256": observed_prefix, "driver_sha256": driver.get("primary_seal_sha256"), "seal_sha256": seal.get("driver_prefix_sha256"), "match": prefix_match},
        "file_count": len(checks),
        "file_mismatch_count": sum(not item["match"] for item in checks),
        "sealed_names": [item["name"] for item in checks],
    }


def same_id_audit(raw: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    pairs = []
    total = {"raw_receiver": 0, "mlp_input": 0}
    mismatches = {"raw_receiver": 0, "mlp_input": 0}
    max_abs = {"raw_receiver": 0.0, "mlp_input": 0.0}
    l2_max = {"raw_receiver": 0.0, "mlp_input": 0.0}
    refmap = reference_map(driver)
    prim = {str(path.get("name")): (case, path) for case, path in primary_paths(driver)}
    for case_id, record in sorted(refmap.items()):
        for same in record.get("same_ids", []):
            if not isinstance(same, Mapping):
                continue
            pname = str(same.get("primary_name", ""))
            pair = {"case": case_id, "primary": pname, "same_id": same.get("name"), "status": "ok", "steps": 0, "raw_receiver_mismatch_count": 0, "mlp_input_mismatch_count": 0, "prefix_match": True}
            primary = prim.get(pname, (None, None))[1]
            if not isinstance(primary, Mapping):
                pair.update(status="missing_primary")
                pairs.append(pair)
                continue
            p_rows, s_rows = steps(primary), steps(same)
            pair["steps"] = min(len(p_rows), len(s_rows))
            if len(p_rows) != len(s_rows):
                pair["status"] = "mismatch"
            p_tokens = [step_token(row) for row in p_rows]
            s_tokens = [step_token(row) for row in s_rows]
            pair["prefix_match"] = p_tokens == s_tokens and list(same.get("consumed_ids", [])) == [int(x) for x in p_tokens if x is not None]
            if not pair["prefix_match"]:
                pair["status"] = "mismatch"
            for i, (prow, srow) in enumerate(zip(p_rows, s_rows)):
                pstem = str(prow.get("stem", f"{pname}-s{i:03d}"))
                sstem = str(srow.get("stem", f"{same.get('name')}-s{i:03d}"))
                for suffix in ("raw_receiver", "mlp_input"):
                    total[suffix] += 1
                    diff = byte_difference(raw / f"{pstem}-{suffix}.f32", raw / f"{sstem}-{suffix}.f32")
                    if not diff.get("byte_exact", False):
                        mismatches[suffix] += 1
                        pair[f"{suffix}_mismatch_count"] += 1
                        pair["status"] = "mismatch"
                    max_abs[suffix] = max(max_abs[suffix], float(diff.get("max_abs") or 0.0))
                    l2_max[suffix] = max(l2_max[suffix], float(diff.get("l2") or 0.0))
            pairs.append(pair)
    return {
        "status": "byte_exact" if not any(mismatches.values()) and pairs and all(x.get("status") == "ok" for x in pairs) else "divergent_or_incomplete",
        "pair_count": len(pairs),
        "steps_compared": total,
        "byte_mismatch_count": mismatches,
        "max_abs": max_abs,
        "l2_max": l2_max,
        "pairs": pairs,
        "comparison": "candidate primary versus native same-ID replay, matched by complete consumed token history and position",
    }


def common_history_audit(raw: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the shared-history receiver invariant independently.

    Candidate arms are keyed by their complete consumed-token prefix rather
    than by step number.  This is deliberately separate from same-ID native
    replay: it checks the common attention/receiver state before the policies
    have had a chance to write different MLP updates.
    """
    case_records = []
    total_groups = 0
    total_pairs = 0
    receiver_mismatches = 0
    norm_mismatches = 0
    max_abs = {"raw_receiver": 0.0, "mlp_input": 0.0}
    pairwise_cases = []
    seen_cases: set[int] = set()
    for case, _ in primary_paths(driver):
        case_id = _int(case.get("case"))
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        arms = [path for c, path in primary_paths(driver) if _int(c.get("case")) == case_id]
        if not arms:
            continue
        by_arm: dict[str, dict[tuple[int, ...], tuple[int, Mapping[str, Any]]]] = {}
        for path in arms:
            prefix_map: dict[tuple[int, ...], tuple[int, Mapping[str, Any]]] = {}
            prefix: list[int] = []
            for i, row in enumerate(steps(path)):
                token = step_token(row)
                if token is None:
                    continue
                prefix.append(token)
                prefix_map[tuple(prefix)] = (i, row)
            by_arm[str(path.get("name"))] = prefix_map
        common = set.intersection(*(set(value) for value in by_arm.values())) if by_arm else set()
        case_receiver_mismatches = 0
        case_norm_mismatches = 0
        for prefix in sorted(common, key=lambda value: (len(value), value)):
            total_groups += 1
            names = sorted(by_arm)
            ref_name = names[0]
            ref_i, ref_row = by_arm[ref_name][prefix]
            ref_stem = str(ref_row.get("stem", f"{ref_name}-s{ref_i:03d}"))
            for name in names[1:]:
                total_pairs += 1
                i, row = by_arm[name][prefix]
                stem = str(row.get("stem", f"{name}-s{i:03d}"))
                for suffix in ("raw_receiver", "mlp_input"):
                    diff = byte_difference(raw / f"{ref_stem}-{suffix}.f32", raw / f"{stem}-{suffix}.f32")
                    if not diff.get("byte_exact", False):
                        if suffix == "raw_receiver":
                            receiver_mismatches += 1
                            case_receiver_mismatches += 1
                        else:
                            norm_mismatches += 1
                            case_norm_mismatches += 1
                    max_abs[suffix] = max(max_abs[suffix], float(diff.get("max_abs") or 0.0))
        case_records.append({
            "case": case_id,
            "arms": [path.get("name") for path in arms],
            "common_prefix_groups": len(common),
            "raw_receiver_mismatch_count": case_receiver_mismatches,
            "mlp_input_mismatch_count": case_norm_mismatches,
        })
        # The producer groups the union of all arms by complete prefix and
        # compares every entry after the first.  This differs from the
        # all-arm intersection above when one policy has already emitted a
        # different token.  Recompute that exact grouping independently so
        # its group/pair counts can be compared without trusting its booleans.
        groups: dict[tuple[int, ...], list[tuple[str, int, Mapping[str, Any]]]] = {}
        for name, prefix_map in by_arm.items():
            for prefix, (index, row) in prefix_map.items():
                groups.setdefault(prefix, []).append((name, index, row))
        union_groups = [entries for entries in groups.values() if len(entries) >= 2]
        union_pairs = 0
        union_raw_mismatches = 0
        union_norm_mismatches = 0
        for entries in union_groups:
            ref_name, ref_index, ref_row = entries[0]
            ref_stem = str(ref_row.get("stem", f"{ref_name}-s{ref_index:03d}"))
            for name, index, row in entries[1:]:
                union_pairs += 1
                stem = str(row.get("stem", f"{name}-s{index:03d}"))
                raw_diff = byte_difference(raw / f"{ref_stem}-raw_receiver.f32", raw / f"{stem}-raw_receiver.f32")
                norm_diff = byte_difference(raw / f"{ref_stem}-mlp_input.f32", raw / f"{stem}-mlp_input.f32")
                union_raw_mismatches += not raw_diff.get("byte_exact", False)
                union_norm_mismatches += not norm_diff.get("byte_exact", False)
        pairwise_cases.append({
            "case": case_id,
            "groups_compared": len(union_groups),
            "pair_comparisons": union_pairs,
            "raw_receiver_mismatch_count": union_raw_mismatches,
            "mlp_input_mismatch_count": union_norm_mismatches,
        })
    return {
        "status": "byte_exact" if receiver_mismatches == 0 and norm_mismatches == 0 else "divergent_observed",
        "common_prefix_groups": total_groups,
        "pair_comparisons": total_pairs,
        "raw_receiver_mismatch_count": receiver_mismatches,
        "mlp_input_mismatch_count": norm_mismatches,
        "max_abs": max_abs,
        "cases": case_records,
        "producer_style_pairwise": pairwise_cases,
        "comparison": "all-arm intersection plus independent producer-style union grouping by complete consumed-token prefix",
    }


def capture_geometry(raw: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    prim = primary_paths(driver)
    refmap = reference_map(driver)
    primary_expected: set[str] = set()
    native_expected: set[str] = set()
    same_expected: set[str] = set()
    readout_expected: set[str] = set()

    def add_path_arrays(target: set[str], path: Mapping[str, Any], suffixes: Iterable[str]) -> None:
        for i, row in enumerate(steps(path)):
            stem = str(row.get("stem", f"{path.get('name', 'path')}-s{i:03d}"))
            for suffix in suffixes:
                target.add(f"{stem}-{suffix}.f32")

    def add_readouts(path: Mapping[str, Any]) -> None:
        for block in path.values():
            if not isinstance(block, Mapping):
                continue
            logits = block.get("logits", block)
            if isinstance(logits, Mapping) and isinstance(logits.get("file"), str) and logits["file"].endswith(".f32"):
                readout_expected.add(logits["file"])

    for case, path in prim:
        add_path_arrays(primary_expected, path, ("raw_receiver", "mlp_input", "mlp_return", "output", "native_write"))
        add_readouts(path)
    for record in refmap.values():
        native = record.get("native")
        if isinstance(native, Mapping):
            add_path_arrays(native_expected, native, ("raw_receiver", "mlp_input", "mlp_return", "output"))
            add_readouts(native)
        for same in record.get("same_ids", []):
            if isinstance(same, Mapping):
                add_path_arrays(same_expected, same, ("raw_receiver", "mlp_input"))
                add_readouts(same)

    root_f32 = {path.name for path in raw.glob("*.f32")}
    compiled_f32 = {path.name for path in (raw / "compiled").glob("*.f32")}
    gate_up = {name for name in root_f32 if name.endswith("-mlp_gate.f32") or name.endswith("-mlp_up.f32")}
    expected_core = primary_expected | native_expected | same_expected | readout_expected
    missing_core = sorted(expected_core - root_f32)
    core_present = sorted(expected_core & root_f32)
    # All files under these categories are checked independently; optional
    # gate/up diagnostics are counted but are not required at every step.
    finite_failures = []
    count_expectations: list[tuple[str, int]] = []
    for name in primary_expected | native_expected | same_expected:
        count_expectations.append((name, HIDDEN))
    for name in gate_up:
        count_expectations.append((name, INTERMEDIATE))
    for name in readout_expected:
        count_expectations.append((name, VOCAB))
    for name, count in count_expectations:
        meta = array_meta(raw / name, count)
        if meta.get("status") != "ok":
            finite_failures.append({"name": name, **meta})

    suffix_counts = {
        suffix: sum(1 for name in root_f32 if name.endswith(suffix))
        for suffix in ("-raw_receiver.f32", "-mlp_input.f32", "-mlp_return.f32", "-native_write.f32", "-output.f32", "-mlp_gate.f32", "-mlp_up.f32")
    }
    all_root_bytes = sum(path.stat().st_size for path in raw.glob("*" ) if path.is_file())
    f32_bytes = sum(path.stat().st_size for path in raw.glob("*.f32"))
    return {
        "status": "complete" if not missing_core and not finite_failures else "incomplete",
        "primary_path_count": len(prim),
        "primary_step_count": sum(len(steps(path)) for _, path in prim),
        "native_own_path_count": sum(1 for record in refmap.values() if isinstance(record.get("native"), Mapping)),
        "native_own_step_count": sum(len(steps(record["native"])) for record in refmap.values() if isinstance(record.get("native"), Mapping)),
        "same_id_path_count": sum(len(record.get("same_ids", [])) for record in refmap.values()),
        "same_id_step_count": sum(len(steps(same)) for record in refmap.values() for same in record.get("same_ids", []) if isinstance(same, Mapping)),
        "expected_core_files": len(expected_core),
        "present_core_files": len(core_present),
        "missing_core_files": missing_core,
        "optional_gate_up_file_count": len(gate_up),
        "readout_file_count": len(readout_expected),
        "root_f32_file_count": len(root_f32),
        "compiled_f32_file_count": len(compiled_f32),
        "suffix_counts": suffix_counts,
        "finite_or_shape_failures": finite_failures,
        "root_bytes": all_root_bytes,
        "root_f32_bytes": f32_bytes,
        "compiled_f32_bytes": sum(path.stat().st_size for path in (raw / "compiled").glob("*.f32")),
    }


def call_audit(driver: Mapping[str, Any]) -> dict[str, Any]:
    primary_records = [call_record(path, "primary") for _, path in primary_paths(driver)]
    native_records = []
    same_records = []
    for record in reference_map(driver).values():
        native = record.get("native")
        if isinstance(native, Mapping):
            native_records.append(call_record(native, "native_own"))
        for same in record.get("same_ids", []):
            if isinstance(same, Mapping):
                same_records.append(call_record(same, "same_id"))

    def aggregate(records: list[Mapping[str, Any]]) -> dict[str, int]:
        return {key: sum(_int(record.get("observed", {}).get(key)) for record in records) for key in (*COUNTERS, "steps")}

    primary_aggregate = aggregate(primary_records)
    native_aggregate = aggregate(native_records)
    same_aggregate = aggregate(same_records)
    primary_zero_g_u = sum(record["observed"]["selected_gate_up_calls"] for record in primary_records if record["arm"] in ("source64", "random64", "b3", "zero"))
    return {
        "primary": {"records": primary_records, "aggregate": primary_aggregate},
        "native_own": {"records": native_records, "aggregate": native_aggregate},
        "same_id_replay": {"records": same_records, "aggregate": same_aggregate},
        "primary_selected_gate_up_calls_on_non_native_arms": primary_zero_g_u,
        "primary_basis_q_g_u_calls_on_zero_arm": sum(record["observed"]["basis_q_calls"] + record["observed"]["basis_g_calls"] + record["observed"]["basis_u_calls"] for record in primary_records if record["arm"] == "zero"),
        "primary_records_with_contract_mismatch": sum(record["status"] == "mismatch" for record in primary_records),
        "native_records_with_contract_mismatch": sum(record["status"] == "mismatch" for record in native_records),
        "same_id_records_with_contract_mismatch": sum(record["status"] == "mismatch" for record in same_records),
    }


def audit(run: Path) -> dict[str, Any]:
    raw = run / "raw" if (run / "raw").is_dir() else run
    driver = load_json(raw / "driver.json")
    report = load_json(raw / "report.json") if (raw / "report.json").is_file() else {}
    raw_report_hash = sha256(raw / "report.json") if (raw / "report.json").is_file() else None
    outer_report_hash = sha256(run / "report.json") if (run / "report.json").is_file() else None
    prim = primary_paths(driver)
    refmap = reference_map(driver)
    token_records = [token_record(case, path, "primary") for case, path in prim]
    for case_id, record in refmap.items():
        primary_case = next((case for case, _ in prim if _int(case.get("case")) == case_id), None)
        if primary_case is None:
            continue
        native = record.get("native")
        if isinstance(native, Mapping):
            token_records.append(token_record(primary_case, native, "native_own"))
        primary_by_name = {str(path.get("name")): path for case, path in prim if _int(case.get("case")) == case_id}
        for same in record.get("same_ids", []):
            if not isinstance(same, Mapping):
                continue
            primary = primary_by_name.get(str(same.get("primary_name")))
            if isinstance(primary, Mapping):
                p_tokens = [step_token(row) for row in steps(primary)]
                s_tokens = [step_token(row) for row in steps(same)]
                token_records.append({
                    "role": "same_id",
                    "path": same.get("name"),
                    "case": case_id,
                    "status": "ok" if p_tokens == s_tokens and list(same.get("consumed_ids", [])) == [int(x) for x in p_tokens if x is not None] else "mismatch",
                    "actual_count": len(s_tokens),
                    "expected_count": len(p_tokens),
                    "mismatch_index": next((i for i, (x, y) in enumerate(zip(s_tokens, p_tokens)) if x != y), None),
                    "prefix_match_primary": p_tokens == s_tokens,
                    "consumed_ids_match_primary": list(same.get("consumed_ids", [])) == [int(x) for x in p_tokens if x is not None],
                })
    factor = factor_audit(raw, driver, report)
    seal = seal_audit(raw, driver)
    parity = same_id_audit(raw, driver)
    common_history = common_history_audit(raw, driver)
    producer_common = report.get("analysis", {}).get("common_history_raw_receiver", []) if isinstance(report.get("analysis", {}), Mapping) else []
    if isinstance(producer_common, list):
        producer_by_case = {str(item.get("case")): item for item in producer_common if isinstance(item, Mapping)}
        common_history["producer_report_comparison"] = []
        for item in common_history.get("producer_style_pairwise", []):
            producer = producer_by_case.get(str(item.get("case")), {})
            common_history["producer_report_comparison"].append({
                "case": item.get("case"),
                "producer_groups_compared": producer.get("groups_compared"),
                "independent_producer_style_groups_compared": item.get("groups_compared"),
                "producer_pair_comparisons": producer.get("pair_comparisons"),
                "independent_producer_style_pair_comparisons": item.get("pair_comparisons"),
                "group_count_matches": producer.get("groups_compared") == item.get("groups_compared"),
                "pair_count_matches": producer.get("pair_comparisons") == item.get("pair_comparisons"),
            })
    geometry = capture_geometry(raw, driver)
    calls = call_audit(driver)
    token_mismatches = sum(item.get("status") != "ok" for item in token_records)
    source_input_match = factor["compiled_manifest_hashes_match"] and factor["source_export_input_joins_match"] and factor["model_claims"]["all_match"]
    mechanics = {
        "primary_seal": seal["status"],
        "factor_and_source_hashes": "ok" if factor["all_factor_hashes_match"] and source_input_match else "mismatch",
        "token_authority": "ok" if token_mismatches == 0 else "mismatch",
        "call_ledger": "ok" if not calls["primary_records_with_contract_mismatch"] and not calls["native_records_with_contract_mismatch"] and not calls["same_id_records_with_contract_mismatch"] else "mismatch",
        "capture_geometry": geometry["status"],
    }
    mechanics_status = "complete" if all(value == "ok" or value == "complete" for value in mechanics.values()) else "evidence_incomplete"
    audit_result = {
        "schema": "tcm-operand-basis-independent-audit-v1",
        "run": str(run),
        "raw_directory": str(raw),
        "source": {
            "driver_sha256": sha256(raw / "driver.json"),
            "raw_report_sha256": raw_report_hash,
            "outer_report_sha256": outer_report_hash,
            "primary_seal_json_sha256": sha256(raw / "primary-seal.json") if (raw / "primary-seal.json").is_file() else None,
            "raw_input_manifest_sha256": sha256(raw / "input-manifest.json") if (raw / "input-manifest.json").is_file() else None,
        },
        "declared": {
            "driver_schema": driver.get("schema"),
            "model_digest": driver.get("model_digest"),
            "layer": driver.get("layer"),
            "rank": driver.get("rank"),
            "budget": driver.get("budget"),
            "mode": driver.get("mode"),
            "primary_seal_bytes": driver.get("primary_seal_bytes"),
        },
        "factor_and_source_hashes": factor,
        "primary_seal": seal,
        "physical_calls": calls,
        "token_authority": {
            "records": token_records,
            "mismatch_count": token_mismatches,
            "primary_count": len(prim),
            "native_own_count": sum(1 for record in refmap.values() if isinstance(record.get("native"), Mapping)),
            "same_id_count": sum(len(record.get("same_ids", [])) for record in refmap.values()),
        },
        "same_id_receiver_norm_parity": parity,
        "common_history_receiver_norm": common_history,
        "capture_geometry": geometry,
        "mechanics_checks": mechanics,
        "mechanics_status": mechanics_status,
        "instrument_status": "complete" if geometry["status"] == "complete" and parity["status"] == "byte_exact" else "observed_divergence_or_incomplete",
        "scientific_status": "observations_only; no quality or promotion gate applied",
        "limitations": [
            "Source G/U/gamma bytes are not retained in the result archive; their pre-target hashes and source-input manifest are retained and checked.",
            "Same-ID parity is a consumer/replay mechanics check, not a claim that candidate future token outcomes match native outcomes.",
            "The audit reads the recorded native same-ID arrays and does not rerun the model or reconstruct omitted KV tensors.",
        ],
    }
    return audit_result


def markdown(result: Mapping[str, Any]) -> str:
    calls = result["physical_calls"]
    geo = result["capture_geometry"]
    parity = result["same_id_receiver_norm_parity"]
    common = result["common_history_receiver_norm"]
    factor = result["factor_and_source_hashes"]
    seal = result["primary_seal"]
    tok = result["token_authority"]
    return "\n".join([
        "# Independent saved-only operand-basis audit",
        "",
        f"Run: `{result['run']}`",
        f"Audit schema: `{result['schema']}`",
        "",
        f"Mechanics status: **{result['mechanics_status']}**. Instrument status: **{result['instrument_status']}**.",
        "The audit reads `raw/driver.json`, archived arrays, compiled factor files, and the primary seal; it does not import producer analysis or alter the run.",
        "",
        "## Physical call ledger",
        "",
        f"Primary paths: {calls['primary']['aggregate']['steps']} steps; native-own references: {calls['native_own']['aggregate']['steps']}; native same-ID replays: {calls['same_id_replay']['aggregate']['steps']}.",
        f"Selected G/U calls on source/random/B3/zero primary arms: **{calls['primary_selected_gate_up_calls_on_non_native_arms']}**; zero-arm derived Q/G/U calls: **{calls['primary_basis_q_g_u_calls_on_zero_arm']}**.",
        f"Native-own selected G/U calls: {calls['native_own']['aggregate']['selected_gate_up_calls']}; same-ID selected G/U calls: {calls['same_id_replay']['aggregate']['selected_gate_up_calls']}.",
        "",
        "## Same-ID native consumer parity",
        "",
        f"{parity['pair_count']} pairs and {sum(parity['steps_compared'].values()) // 2 if parity['steps_compared'] else 0} candidate steps were compared by complete consumed-token history and position.",
        f"Raw receiver byte mismatches: {parity['byte_mismatch_count']['raw_receiver']}; normalized-input byte mismatches: {parity['byte_mismatch_count']['mlp_input']}. Status: **{parity['status']}**.",
        f"Shared-primary-history groups: {common['common_prefix_groups']}; receiver mismatches: {common['raw_receiver_mismatch_count']}; normalized-input mismatches: {common['mlp_input_mismatch_count']}. Status: **{common['status']}**.",
        "The audit records both the strict all-arm intersection and the producer-style union grouping, so early arm divergence is not mistaken for a mechanics failure.",
        "",
        "## Token authority and custody",
        "",
        f"Token records: {len(tok['records'])}; mismatches: **{tok['mismatch_count']}**.",
        f"Primary seal: `{seal['status']}` ({seal['file_count']} sealed files, {seal['file_mismatch_count']} mismatches).",
        f"Compiled factors: all six factor hashes match driver and bank manifest = **{factor['all_factor_hashes_match']}**; source input joins = **{factor['source_export_input_joins_match']}**.",
        "",
        "## Capture geometry",
        "",
        f"{geo['primary_path_count']} primary paths / {geo['primary_step_count']} primary steps; {geo['native_own_path_count']} native-own paths / {geo['native_own_step_count']} steps; {geo['same_id_path_count']} same-ID paths / {geo['same_id_step_count']} steps.",
        f"Expected core files: {geo['expected_core_files']}; present: {geo['present_core_files']}; optional gate/up files: {geo['optional_gate_up_file_count']}; readout files: {geo['readout_file_count']}; root FP32 bytes: {geo['root_f32_bytes']:,}.",
        f"Geometry status: **{geo['status']}**.",
        "",
        "Scientific observations remain separate from these mechanics checks; no quality or promotion gate was applied.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    output = args.output or (args.run / "audit.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown_path = output.with_name("AUDIT.md")
    markdown_path.write_text(markdown(result))
    print(json.dumps({"audit": str(output), "markdown": str(markdown_path), "mechanics_status": result["mechanics_status"], "instrument_status": result["instrument_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
