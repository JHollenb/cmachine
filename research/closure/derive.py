"""Additive derivation for the authenticated state-closure run.

The producer report and its raw arrays are immutable evidence.  This module
only reads them, recomputes a compact set of consumer-facing comparisons, and
writes a separate ``study.json``.  In particular, a selected ``kv16-17`` file
contains the newly appended rows at the current position; it is not the full
valid-prefix cache.  Full-prefix identity is therefore reported from the
driver's ``valid_kv_sha256`` fields beside the selected-row measurements.

Usage::

    python derive.py saturn/cmachine/closure/runs/job-... \
        saturn/cmachine/closure/study/study.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA = "tcm-state-closure-study-v1"
DERIVE_VERSION = "closure-derive-v1"
TASKS = ("recall", "math", "prose")
ARM_LABELS = (
    "native",
    "sham",
    "fiber",
    "fiber-repair",
    "compact-once",
    "compact-recurring",
    "corrected-recurring",
    "zero",
)
HIDDEN = 1536
INTERMEDIATE = 8960
VOCAB = 151936
KV_DIM = 256
KV_EXTENT = 1024
F32 = np.dtype("<f4")


class DeriveError(ValueError):
    """Raised when the retained evidence cannot support a derived row."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeriveError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, Mapping):
        raise DeriveError(f"expected JSON object: {path}")
    return value


def load_f32(path: Path, extent: int | None = None) -> np.ndarray:
    if not path.exists():
        raise DeriveError(f"missing retained array: {path}")
    payload = path.read_bytes()
    if len(payload) % 4:
        raise DeriveError(f"unaligned FP32 array: {path}")
    value = np.frombuffer(payload, dtype=F32)
    if extent is not None and value.shape != (extent,):
        raise DeriveError(f"invalid array extent {path}: {value.shape}, expected {(extent,)}")
    if not np.isfinite(value).all():
        raise DeriveError(f"non-finite retained array: {path}")
    return np.ascontiguousarray(value, dtype=F32)


def exact(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def metric(left: np.ndarray, right: np.ndarray, *, denominator: str = "right") -> dict[str, Any]:
    """Return an array metric with an explicit relative-error denominator.

    Historical ``array_metrics`` rows use the right operand's norm.  Derived
    rows retain that convention under ``legacy_relative_l2`` while also
    exposing source-normalized values where the scientific question asks for
    error relative to the native/source endpoint.
    """

    if left.shape != right.shape:
        raise DeriveError(f"array shape mismatch: {left.shape} vs {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    left_norm = float(np.linalg.norm(left.astype(np.float64)))
    right_norm = float(np.linalg.norm(right.astype(np.float64)))
    l2 = float(np.linalg.norm(delta))
    if denominator == "left":
        denom, denom_label = left_norm, "left"
    elif denominator == "right":
        denom, denom_label = right_norm, "right"
    elif denominator == "source":
        denom, denom_label = right_norm, "source/right endpoint"
    elif denominator == "fiber":
        denom, denom_label = right_norm, "fiber/right endpoint"
    else:
        raise DeriveError(f"unknown metric denominator: {denominator}")
    return {
        "bytes_equal": exact(left, right),
        "l2": l2,
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "left_norm_l2": left_norm,
        "right_norm_l2": right_norm,
        "legacy_relative_l2": float(l2 / right_norm) if right_norm else None,
        "relative_l2": float(l2 / denom) if denom else None,
        "relative_denominator": denom_label,
    }


def path_from_report(root: Path, path_value: str, *, sibling: str | None = None) -> Path:
    """Resolve stale absolute worker paths using the collected run tree."""

    direct = Path(path_value)
    if direct.exists():
        return direct
    options = [root / direct.name]
    if sibling:
        options.extend([root / sibling / direct.name, root / "payload" / sibling / direct.name])
    options.extend([root / "payload" / direct.name, root / "raw" / direct.name])
    for option in options:
        if option.exists():
            return option
    raise DeriveError(f"cannot resolve retained reference: {path_value}")


def authenticate(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    report_path = root / "report.json"
    verification_path = root / "verification.json"
    report = read_json(report_path)
    verification = read_json(verification_path)
    report_hash = sha(report_path)
    if verification.get("report_sha256") != report_hash:
        raise DeriveError("verification report hash does not match report.json")
    if verification.get("analysis_recomputed") is not True:
        raise DeriveError("source analysis was not recomputed")
    if report.get("schema") != "tcm-state-closure-experiment-v1":
        raise DeriveError("unexpected closure report schema")
    raw_manifest = report.get("raw_files")
    if not isinstance(raw_manifest, Mapping):
        raise DeriveError("closure report has no raw_files manifest")
    # Verify the immutable raw manifest independently.  This is read-only and
    # deliberately precedes all scientific comparisons.
    for name, info in raw_manifest.items():
        if not isinstance(info, Mapping):
            raise DeriveError(f"invalid raw manifest row: {name}")
        path = root / "raw" / str(name)
        if not path.exists() or sha(path) != info.get("sha256") or path.stat().st_size != int(info.get("bytes", -1)):
            raise DeriveError(f"raw identity differs: {name}")
    driver_path = root / "raw" / "driver.json"
    geometry_path = root / "payload" / "geometry" / "geometry.json"
    source_hashes = {
        "report_sha256": report_hash,
        "verification_sha256": sha(verification_path),
        "raw_driver_sha256": sha(driver_path),
        "geometry_json_sha256": sha(geometry_path),
        "source_manifest_sha256": str(report.get("source_manifest_sha256")),
        "binary_sha256": str(report.get("binary_sha256")),
        "model_content_sha256": str(report.get("model", {}).get("content_sha256")),
    }
    return report, verification, source_hashes


def task_geometry(root: Path, geometry: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    rows = geometry.get("tasks", [])
    if not isinstance(rows, list):
        raise DeriveError("geometry tasks is not a list")
    for row in rows:
        if isinstance(row, Mapping) and row.get("task") == task:
            return row
    raise DeriveError(f"geometry task missing: {task}")


def branch_prefix(case: Mapping[str, Any], branch: str) -> str:
    spec = case.get("writes", {}).get(branch)
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping) and isinstance(spec.get("prefix"), str):
        return str(spec["prefix"])
    raise DeriveError(f"{case.get('task')} has no branch prefix: {branch}")


def load_branch(root: Path, case: Mapping[str, Any], branch: str) -> dict[str, np.ndarray]:
    prefix = branch_prefix(case, branch)
    extents = {"receiver": HIDDEN, "normalized": HIDDEN, "gate": INTERMEDIATE, "up": INTERMEDIATE, "write": HIDDEN}
    result: dict[str, np.ndarray] = {}
    for port, extent in extents.items():
        if branch == "project" and port == "receiver":
            continue
        result[port] = load_f32(root / "raw" / f"{prefix}-{port}.f32", extent)
    return result


def arm_map(case: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    arms = case.get("arms")
    if not isinstance(arms, list):
        raise DeriveError(f"{case.get('task')} has no arms")
    result = {}
    for arm in arms:
        if not isinstance(arm, Mapping) or not isinstance(arm.get("name"), str):
            raise DeriveError("invalid closure arm")
        result[str(arm["name"])] = arm
    return result


def arm_port(root: Path, arm: Mapping[str, Any], step: int, port: str, extent: int | None = None) -> np.ndarray:
    name = str(arm["name"])
    return load_f32(root / "raw" / f"{name}-s{step:03d}-{port}.f32", extent)


def first_logit(root: Path, arm: Mapping[str, Any]) -> np.ndarray:
    return load_f32(root / "raw" / f"{arm['name']}-s000-first.f32", VOCAB)


def history_same(arm: Mapping[str, Any], reference: Mapping[str, Any], step: int) -> bool:
    a = arm.get("steps", [])[0 : step + 1]
    b = reference.get("steps", [])[0 : step + 1]
    return [row.get("input_id") for row in a] == [row.get("input_id") for row in b]


def first_different_ids(arm: Mapping[str, Any], reference: Mapping[str, Any]) -> int | None:
    left = arm.get("own", {}).get("ids", [])
    right = reference.get("own", {}).get("ids", [])
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def selected_rows(array: np.ndarray) -> dict[str, np.ndarray]:
    if array.shape != (KV_EXTENT,):
        raise DeriveError(f"selected KV extent differs: {array.shape}")
    rows = array.reshape(4, KV_DIM)
    return {"k16": rows[0], "k17": rows[1], "v16": rows[2], "v17": rows[3]}


def selected_kv_compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    l, r = selected_rows(left), selected_rows(right)
    fields = {key: metric(l[key], r[key], denominator="right") for key in ("k16", "k17", "v16", "v17")}
    return {
        "scope": "newly-appended current-position rows only; not full valid-prefix memory",
        "layer16_bytes_equal": bool(fields["k16"]["bytes_equal"] and fields["v16"]["bytes_equal"]),
        "layer17_bytes_equal": bool(fields["k17"]["bytes_equal"] and fields["v17"]["bytes_equal"]),
        "rows": fields,
    }


def prefix_hash_equal(left_step: Mapping[str, Any], right_step: Mapping[str, Any]) -> bool | None:
    a, b = left_step.get("valid_kv_sha256"), right_step.get("valid_kv_sha256")
    return None if a is None or b is None else bool(a == b)


def operation_counts(arm: Mapping[str, Any]) -> dict[str, Any]:
    steps = arm.get("steps", [])
    fields = ("gate_up_calls", "down_calls", "compile_calls")
    counts = {field: int(sum(int(row.get(field, 0)) for row in steps)) for field in fields}
    counts["steps"] = len(steps)
    counts["own_tokens"] = len(arm.get("own", {}).get("ids", []))
    counts["substitution_steps"] = int(sum(bool(row.get("substitution", False)) for row in steps))
    counts["all_step_fields_present"] = all(field in row for row in steps for field in ("gate_up_calls", "down_calls"))
    counts["compile_calls_declared"] = all("compile_calls" in row for row in steps)
    return counts


def fixed_skip_identity(root: Path, arm: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for step in range(len(arm.get("steps", []))):
        receiver = arm_port(root, arm, step, "raw_receiver", HIDDEN)
        write = arm_port(root, arm, step, "mlp_return", HIDDEN)
        output = arm_port(root, arm, step, "output", HIDDEN)
        expected = np.add(receiver, write)
        rows.append({
            "step": step,
            "bytes_equal": exact(expected, output),
            "output_minus_receiver_plus_write": metric(output, expected, denominator="right"),
        })
    return {
        "identity": bool(rows) and all(row["bytes_equal"] for row in rows),
        "rows": rows,
        "formula": "output = FP32_add(raw_receiver, mlp_return)",
    }


def native_sham_profile(root: Path, arms: Mapping[str, Mapping[str, Any]], task: str) -> dict[str, Any]:
    native, sham = arms[f"{task}-native"], arms[f"{task}-sham"]
    rows = []
    for step in range(min(len(native["steps"]), len(sham["steps"]))):
        ports = {}
        for port, extent in (("input", HIDDEN), ("raw_receiver", HIDDEN), ("mlp_input", HIDDEN), ("mlp_return", HIDDEN), ("output", HIDDEN), ("kv16-17", KV_EXTENT)):
            ports[port] = metric(arm_port(root, native, step, port, extent), arm_port(root, sham, step, port, extent), denominator="right")
        rows.append({
            "step": step,
            "ports": ports,
            "logits_sha256_equal": native["steps"][step].get("logits_sha256") == sham["steps"][step].get("logits_sha256"),
            "valid_kv_sha256_equal": native["steps"][step].get("valid_kv_sha256") == sham["steps"][step].get("valid_kv_sha256"),
        })
    first = metric(first_logit(root, sham), first_logit(root, native), denominator="right")
    all_ports = all(row["ports"][port]["bytes_equal"] for row in rows for port in row["ports"])
    all_hashes = all(row["logits_sha256_equal"] and row["valid_kv_sha256_equal"] for row in rows)
    return {
        "task": task,
        "steps": len(rows),
        "all_step_ports_byte_equal": all_ports,
        "all_step_logit_and_prefix_hash_equal": all_hashes,
        "first_full_logit": first,
        "rows": rows,
    }


def own_history_profile(root: Path, arms: Mapping[str, Mapping[str, Any]], task: str) -> list[dict[str, Any]]:
    native = arms[f"{task}-native"]
    rows = []
    for label in ARM_LABELS:
        if label == "native":
            continue
        name = f"{task}-{label}"
        if name not in arms:
            continue
        arm = arms[name]
        trajectory = []
        for step in range(min(len(arm["steps"]), len(native["steps"]))):
            same_history = history_same(arm, native, step)
            selected = selected_kv_compare(
                arm_port(root, arm, step, "kv16-17", KV_EXTENT),
                arm_port(root, native, step, "kv16-17", KV_EXTENT),
            )
            trajectory.append({
                "step": step,
                "same_complete_token_history": same_history,
                "full_valid_prefix_hash_equal": prefix_hash_equal(arm["steps"][step], native["steps"][step]),
                "selected_kv_rows": selected,
                "selected_kv_scope_warning": "selected rows are not the full valid-prefix cache",
            })
        rows.append({
            "task": task,
            "arm": name,
            "own_ids": list(arm.get("own", {}).get("ids", [])),
            "native_ids": list(native.get("own", {}).get("ids", [])),
            "first_different_emitted_token_index": first_different_ids(arm, native),
            "same_history_steps": sum(row["same_complete_token_history"] for row in trajectory),
            "trajectory": trajectory,
        })
    return rows


def once_recurring_profile(root: Path, arms: Mapping[str, Mapping[str, Any]], task: str) -> dict[str, Any]:
    once, recurring = arms[f"{task}-compact-once"], arms[f"{task}-compact-recurring"]
    rows = []
    for step in range(min(len(once["steps"]), len(recurring["steps"]))):
        same_history = history_same(recurring, once, step)
        rows.append({
            "step": step,
            "same_complete_token_history": same_history,
            "input_id_equal": once["steps"][step].get("input_id") == recurring["steps"][step].get("input_id"),
            "full_valid_prefix_hash_equal": prefix_hash_equal(recurring["steps"][step], once["steps"][step]),
            "raw_receiver": metric(arm_port(root, recurring, step, "raw_receiver", HIDDEN), arm_port(root, once, step, "raw_receiver", HIDDEN), denominator="right"),
            "mlp_return": metric(arm_port(root, recurring, step, "mlp_return", HIDDEN), arm_port(root, once, step, "mlp_return", HIDDEN), denominator="right"),
            "selected_kv_rows": selected_kv_compare(arm_port(root, recurring, step, "kv16-17", KV_EXTENT), arm_port(root, once, step, "kv16-17", KV_EXTENT)),
        })
    return {
        "task": task,
        "left": "compact-recurring",
        "right": "compact-once",
        "first_different_emitted_token_index": first_different_ids(recurring, once),
        "same_history_steps": sum(row["same_complete_token_history"] for row in rows),
        "rows": rows,
    }


def selected_geometry_profile(root: Path, report: Mapping[str, Any], geometry: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    task = str(case["task"])
    gt = task_geometry(root, geometry, task)
    selected_index = int(case.get("selected_index", -1))
    candidates = gt.get("candidates", [])
    if selected_index < 0 or selected_index >= len(candidates):
        raise DeriveError(f"{task} selected geometry index is unavailable")
    candidate = candidates[selected_index]
    candidate_file = path_from_report(root, str(gt["candidate_file"]), sibling="payload/geometry")
    candidate_blob = candidate_file.read_bytes()
    if sha(candidate_file) != gt.get("candidate_file_sha256"):
        raise DeriveError(f"{task} candidate geometry file hash differs")
    offset = int(candidate["offset_bytes"])
    extent_bytes = int(candidate["bytes"])
    candidate_bytes = candidate_blob[offset : offset + extent_bytes]
    if len(candidate_bytes) != extent_bytes or hashlib.sha256(candidate_bytes).hexdigest() != candidate.get("sha256"):
        raise DeriveError(f"{task} selected candidate slice hash differs")
    source_file = path_from_report(root, str(gt["source_file"]), sibling="payload/geometry")
    source_geometry = load_f32(source_file, HIDDEN)
    source_branch = load_branch(root, case, "source")["receiver"]
    fiber_branch = load_branch(root, case, "fiber")["receiver"]
    candidate_array = np.frombuffer(candidate_bytes, dtype=F32)
    source_norm = float(np.linalg.norm(source_branch.astype(np.float64)))
    fiber_norm = float(np.linalg.norm(fiber_branch.astype(np.float64)))
    source_to_geometry = metric(source_branch, source_geometry, denominator="right")
    fiber_to_geometry = metric(fiber_branch, candidate_array, denominator="right")
    return {
        "task": task,
        "selected_index": selected_index,
        "selected_id": candidate.get("id"),
        "angle_rad": candidate.get("angle_rad", candidate.get("angle")),
        "direction": candidate.get("direction"),
        "offset_bytes": offset,
        "candidate_slice_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_slice_sha256_expected": candidate.get("sha256"),
        "candidate_file_sha256": sha(candidate_file),
        "candidate_file_sha256_expected": gt.get("candidate_file_sha256"),
        "source_code_hex": case.get("source_code_hex"),
        "candidate_code_hex": next((row.get("code_hex") for row in case.get("candidates", []) if int(row.get("index", -1)) == selected_index), None),
        "exact_code_equal": bool(case.get("candidates", [])[selected_index].get("same_code")),
        "different_receiver_declared": bool(case.get("candidates", [])[selected_index].get("different_receiver")),
        "source_receiver_norm_l2": source_norm,
        "fiber_receiver_norm_l2": fiber_norm,
        "fiber_norm_over_source": fiber_norm / source_norm if source_norm else None,
        "fiber_norm_delta_l2": fiber_norm - source_norm,
        "fiber_norm_delta_relative_to_source": (fiber_norm - source_norm) / source_norm if source_norm else None,
        "geometry_source_norm_l2_f64": gt.get("source_geometry", {}).get("source_l2_f64"),
        "geometry_candidate_norm_l2_f64": candidate.get("geometry", {}).get("candidate_l2_f64"),
        "geometry_candidate_norm_l2_f32": candidate.get("geometry", {}).get("candidate_l2_f32"),
        "geometry_candidate_minus_source_l2_f32": candidate.get("geometry", {}).get("candidate_minus_source_l2_f32"),
        "source_branch_vs_geometry_source": source_to_geometry,
        "fiber_branch_vs_geometry_candidate": fiber_to_geometry,
    }


def branch_write_profile(root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    task = str(case["task"])
    branches = {name: load_branch(root, case, name) for name in ("source", "fiber", "project", "repair", "fiber_project") if name in case.get("writes", {})}
    pairs = {}
    for left, right in (("source", "fiber"), ("source", "project"), ("fiber", "repair"), ("fiber", "project")):
        if left not in branches or right not in branches:
            continue
        ports = {}
        for port in ("normalized", "gate", "up", "write"):
            ports[port] = metric(branches[left][port], branches[right][port], denominator="right")
        pairs[f"{left}_vs_{right}"] = ports
    project_identity = None
    if "project" in branches and "fiber_project" in branches:
        ports = {port: metric(branches["project"][port], branches["fiber_project"][port], denominator="right") for port in ("gate", "up", "write")}
        project_identity = {"ports": ports, "exact": all(row["bytes_equal"] for row in ports.values()), "scope": "code-derived gate/up/write; normalized RMS diagnostic excluded"}
    fiber_vs_repair = pairs.get("fiber_vs_repair", {})
    normalized_to_source = None
    if "source" in branches and "fiber" in branches:
        normalized_to_source = {
            "fiber_normalized_vs_source": metric(branches["fiber"]["normalized"], branches["source"]["normalized"], denominator="source"),
            "scope": "fiber normalized operand relative to native/source normalized operand",
        }
    return {
        "task": task,
        "branches": {name: {"ports": sorted(value), "prefix": branch_prefix(case, name)} for name, value in branches.items()},
        "pairs": pairs,
        "normalized_to_source": normalized_to_source,
        "project_equal_code_identity": project_identity,
        "fiber_vs_repair_targeted": {"write": fiber_vs_repair.get("write")},
    }


def attention_profile(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for query in (0, 1):
        for length in (2, 3):
            for world in (0, 1):
                prefix = f"attention-q{query}-n{length}-w{world}"
                actual_output = load_f32(root / "raw" / f"{prefix}-output.f32", 128)
                actual_weights = load_f32(root / "raw" / f"{prefix}-weights.f32", length)
                if query == 0:
                    scores = np.zeros(length, dtype=np.float64)
                elif length == 2:
                    sign = 1.0 if world == 0 else -1.0
                    scores = np.asarray([sign, -sign], dtype=np.float64)
                else:
                    sign = 1.0 if world == 0 else -1.0
                    scores = np.asarray([sign, -sign, 0.0], dtype=np.float64)
                weights = np.exp(scores - np.max(scores))
                weights /= weights.sum()
                values = np.asarray([1.0, -1.0, 1.0][:length], dtype=np.float64)
                analytic_output = float(np.dot(weights, values))
                expected_weights = weights.astype(F32)
                expected_output = np.float32(analytic_output)
                uniform_expected = np.full(length, 1.0 / length, dtype=F32)
                row = {
                    "query": query,
                    "length": length,
                    "world": world,
                    "actual_output_first": float(actual_output[0]),
                    "analytic_output_first": analytic_output,
                    "analytic_output_fp32": float(expected_output),
                    "output_abs_error_vs_analytic": abs(float(actual_output[0]) - analytic_output),
                    "output_abs_error_vs_analytic_fp32": abs(float(actual_output[0]) - float(expected_output)),
                    "weights_l2_vs_analytic": float(np.linalg.norm(actual_weights.astype(np.float64) - weights)),
                    "weights_bytes_equal_analytic_fp32": exact(actual_weights, expected_weights),
                    "uniform_query_identity": query == 0 and exact(actual_weights, uniform_expected),
                    "analytic": {
                        "scores": scores.tolist(),
                        "weights": weights.tolist(),
                        "values_first_coordinate": values.tolist(),
                    },
                }
                rows.append(row)
    by_key = {(r["query"], r["length"], r["world"]): r for r in rows}
    append = []
    for query in (0, 1):
        for world in (0, 1):
            short, extended = by_key[(query, 2, world)], by_key[(query, 3, world)]
            append.append({
                "query": query,
                "world": world,
                "actual_output_delta_n3_minus_n2": extended["actual_output_first"] - short["actual_output_first"],
                "analytic_output_delta_n3_minus_n2": extended["analytic_output_first"] - short["analytic_output_first"],
                "actual_minus_analytic_delta": (extended["actual_output_first"] - short["actual_output_first"]) - (extended["analytic_output_first"] - short["analytic_output_first"]),
                "fixed_append_identity_scope": "same q/world, append third key/value row",
            })
    return {
        "rows": rows,
        "fixed_append": append,
        "exact_uniform_identities": [
            {"query": row["query"], "length": row["length"], "world": row["world"], "exact": row["uniform_query_identity"]}
            for row in rows
            if row["query"] == 0
        ],
        "analytic_formula": "softmax(scores) dot [1,-1,1]; q1,n2 world0= tanh(1), world1=-tanh(1); n3 uses the fixed appended zero-score row",
        "gpu_rounding_note": "nonuniform CUDA expf/BLAS rows are compared to analytic FP64 and FP32 casts; only uniform identities are exact-byte checks",
    }


def derive(root: Path, output: Path) -> dict[str, Any]:
    report, verification, source_hashes = authenticate(root)
    driver = read_json(root / "raw" / "driver.json")
    geometry = read_json(root / "payload" / "geometry" / "geometry.json")
    if driver.get("schema") != "tcm-state-closure-driver-v1":
        raise DeriveError("unexpected driver schema")
    cases = driver.get("cases")
    if not isinstance(cases, list) or len(cases) != len(TASKS):
        raise DeriveError("closure case panel differs")
    source_hashes["driver_schema"] = str(driver.get("schema"))
    source_hashes["geometry_schema"] = str(geometry.get("schema"))
    source_hashes["source_report_sha256"] = str(report.get("source_report_sha256"))
    source_hashes["verification_report_sha256"] = str(verification.get("report_sha256"))

    selected_geometry = []
    writes = []
    operation_profiles = []
    fixed_skips = []
    native_sham = []
    own_history = []
    once_recurring = []
    source_fiber_logits = []
    fiber_repair = []
    all_steps = 0
    all_ops = Counter()
    for case in cases:
        task = str(case.get("task"))
        if task not in TASKS:
            raise DeriveError(f"unexpected task: {task}")
        if int(case.get("selected_index", -1)) < 0:
            raise DeriveError(f"{task} lacks selected exact collision")
        selected_geometry.append(selected_geometry_profile(root, report, geometry, case))
        writes.append(branch_write_profile(root, case))
        arms = arm_map(case)
        for label, arm in arms.items():
            profile = operation_counts(arm)
            profile.update({"task": task, "arm": label, "policy": arm.get("policy")})
            operation_profiles.append(profile)
            all_steps += profile["steps"]
            for key in ("gate_up_calls", "down_calls", "compile_calls", "substitution_steps", "own_tokens"):
                all_ops[key] += profile[key]
            fixed_skips.append({"task": task, "arm": label, **fixed_skip_identity(root, arm)})
        native_sham.append(native_sham_profile(root, arms, task))
        own_history.extend(own_history_profile(root, arms, task))
        once_recurring.append(once_recurring_profile(root, arms, task))
        native, fiber = arms[f"{task}-native"], arms[f"{task}-fiber"]
        source_fiber_logits.append({
            "task": task,
            "source_arm": native["name"],
            "fiber_arm": fiber["name"],
            "fiber_first_logit_vs_source": metric(first_logit(root, fiber), first_logit(root, native), denominator="source"),
            "source_first_logit_sha256": sha(root / "raw" / f"{native['name']}-s000-first.f32"),
            "fiber_first_logit_sha256": sha(root / "raw" / f"{fiber['name']}-s000-first.f32"),
        })
        repair = arms[f"{task}-fiber-repair"]
        fiber_repair.append({
            "task": task,
            "write": metric(arm_port(root, repair, 0, "mlp_return", HIDDEN), arm_port(root, fiber, 0, "mlp_return", HIDDEN), denominator="fiber"),
            "first_full_logit": metric(first_logit(root, repair), first_logit(root, fiber), denominator="fiber"),
            "logits_sha256_equal": repair["steps"][0].get("logits_sha256") == fiber["steps"][0].get("logits_sha256"),
            "valid_kv_sha256_equal": repair["steps"][0].get("valid_kv_sha256") == fiber["steps"][0].get("valid_kv_sha256"),
        })

    source_hashes["derive_version"] = DERIVE_VERSION
    source_hashes["derive_script_sha256"] = sha(Path(__file__))
    attention = attention_profile(root, report)
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "version": DERIVE_VERSION,
        "sources": source_hashes,
        "mechanics_status": "complete",
        "instrument_status": "executed",
        "trend_status": "bounded-observations",
        "terminal_status": "not-requested",
        "contract": {
            "layer": 16,
            "capacity": 512,
            "code_bytes": 16,
            "primary_observable": "dense MLP write",
            "selected_kv_scope": "current-position K/V rows at layers 16/17; full-prefix cache is represented only by valid_kv_sha256",
            "relative_error_conventions": {
                "legacy_array_metrics": "right endpoint norm",
                "source_normalized_rows": "source/native endpoint norm, explicitly labeled",
                "fiber_repair_rows": "fiber endpoint norm, explicitly labeled",
            },
        },
        "exact_code_collisions_and_geometry": selected_geometry,
        "write_profiles": writes,
        "code_only_minimax_half": [
            {
                "task": row["task"],
                "endpoint_distance_l2": row["pairs"]["source_vs_fiber"]["write"]["l2"],
                "deterministic_code_only_minimax_half_l2": row["pairs"]["source_vs_fiber"]["write"]["l2"] / 2.0,
                "relative_endpoint_distance_legacy_right": row["pairs"]["source_vs_fiber"]["write"]["legacy_relative_l2"],
                "bound_scope": "this exact source/fiber endpoint pair and dense MLP-write observable",
            }
            for row in writes
        ],
        "source_vs_fiber_first_full_vocab_logits": source_fiber_logits,
        "fixed_skip_identity": fixed_skips,
        "native_sham_all_steps": native_sham,
        "fiber_vs_repair_targeted": fiber_repair,
        "own_history_survival": own_history,
        "compact_once_vs_recurring": once_recurring,
        "operation_profiles": operation_profiles,
        "operation_totals": {"steps": all_steps, **dict(all_ops)},
        "attention_toy": attention,
        "summary": {
            "tasks": list(TASKS),
            "selected_exact_code_collisions": len(selected_geometry),
            "selected_indices": {row["task"]: row["selected_index"] for row in selected_geometry},
            "native_sham_all_steps_exact": all(row["all_step_ports_byte_equal"] and row["all_step_logit_and_prefix_hash_equal"] for row in native_sham),
            "fixed_skip_all_arm_steps_exact": all(row["identity"] for row in fixed_skips),
            "project_equal_code_gate_up_write_exact": all(bool(row.get("project_equal_code_identity", {}).get("exact")) for row in writes),
            "attention_uniform_identities_exact": all(row["exact"] for row in attention["exact_uniform_identities"]),
            "full_prefix_and_selected_row_scopes_separate": True,
            "total_arm_steps": all_steps,
            "total_gate_up_calls": all_ops["gate_up_calls"],
            "total_down_calls": all_ops["down_calls"],
            "total_compile_calls": all_ops["compile_calls"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="completed closure run directory")
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).parent / "study" / "study.json")
    args = parser.parse_args()
    data = derive(args.root, args.output)
    print(json.dumps(data["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
