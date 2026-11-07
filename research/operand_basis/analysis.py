"""Evidence-only analysis for the source-derived operand-basis assay.

The C driver deliberately emits raw arrays and a small JSON ledger instead of
declaring a scientific winner.  This module rechecks that ledger and reports
the signed, finite observations that can be reconstructed from it:

* candidate selected-layer writes versus the dense write reconstructed from
  the same saved raw receiver;
* raw receiver and normalized-input byte identity across policies that really
  consumed the same complete token prefix;
* native own-history versus native same-ID full-vocabulary readouts;
* emitted-token authority and exact consumption order; and
* the executed projection/image/down call ledger.

No metric below is a quality gate.  A missing, non-finite, or divergent
measurement is retained as an observation with an explicit status.  The only
strict checks are custody/shape checks needed to interpret the raw evidence,
and even those are returned in the report rather than converted into a
scientific null.  The adapter accepts both the first operand-basis driver
schema and the additive same-ID reference fields being added to it.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_HIDDEN = 1536
DEFAULT_INTERMEDIATE = 8960
DEFAULT_VOCAB = 151936
VECTOR_SUFFIXES = ("raw_receiver", "mlp_input", "mlp_gate", "mlp_up", "mlp_return", "output")


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _profile(driver: Mapping[str, Any]) -> dict[str, int]:
    profile = driver.get("profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    model = driver.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    return {
        "hidden": _safe_int(profile.get("hidden", model.get("hidden", driver.get("hidden", DEFAULT_HIDDEN))), DEFAULT_HIDDEN),
        "intermediate": _safe_int(profile.get("intermediate", model.get("intermediate", driver.get("intermediate", DEFAULT_INTERMEDIATE))), DEFAULT_INTERMEDIATE),
        "vocab": _safe_int(profile.get("vocab", model.get("vocab", driver.get("vocab", DEFAULT_VOCAB))), DEFAULT_VOCAB),
    }


def _as_path(raw: Path, value: Any) -> Path | None:
    if value is None:
        return None
    try:
        path = Path(str(value)).expanduser()
    except (TypeError, ValueError):
        return None
    candidates = [path]
    if not path.is_absolute():
        candidates += [raw / path, raw.parent / path, raw.parent / "compiled" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _logical_file(raw: Path, path: Path) -> str:
    """Return a relocation-stable path for derived JSON reports.

    Raw result archives are copied between the Mac and the worker host.  An
    absolute producer path is therefore evidence noise and makes a derived
    report non-replayable.  Files outside the archive are represented by their
    basename; the archive-local copy is preferred by callers whenever one is
    available.
    """

    try:
        return path.relative_to(raw).as_posix()
    except ValueError:
        return path.name


def _array(raw: Path, filename: str | Path, count: int | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    path = _as_path(raw, filename)
    if path is None or not path.is_file():
        name = Path(str(filename)).name if Path(str(filename)).is_absolute() else str(filename)
        return None, {"status": "missing", "file": name}
    try:
        data = path.read_bytes()
        if len(data) % 4:
            return None, {"status": "invalid_bytes", "file": _logical_file(raw, path), "bytes": len(data)}
        values = np.frombuffer(data, dtype="<f4").copy()
    except (OSError, ValueError) as exc:
        return None, {"status": "read_error", "file": _logical_file(raw, path), "error": str(exc)}
    result = {
        "status": "ok",
        "file": _logical_file(raw, path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "count": int(values.size),
        "finite": bool(np.isfinite(values).all()),
    }
    if count is not None and values.size != int(count):
        result.update(status="shape_mismatch", expected_count=int(count))
    if not result["finite"]:
        result["status"] = "nonfinite"
    return np.ascontiguousarray(values, dtype="<f4"), result


def _difference(left: np.ndarray | None, right: np.ndarray | None) -> dict[str, Any]:
    """Compare two arrays while retaining non-finite and missing states."""

    if left is None or right is None:
        return {"status": "missing", "left_present": left is not None, "right_present": right is not None}
    left = np.asarray(left)
    right = np.asarray(right)
    result: dict[str, Any] = {
        "status": "ok",
        "shape_equal": bool(left.shape == right.shape),
        "byte_exact": bool(left.tobytes(order="C") == right.tobytes(order="C")) if left.shape == right.shape else False,
        "left_finite": bool(np.isfinite(left).all()),
        "right_finite": bool(np.isfinite(right).all()),
    }
    if left.shape != right.shape:
        result["status"] = "shape_mismatch"
        return result
    if not result["left_finite"] or not result["right_finite"]:
        result["status"] = "nonfinite"
        result.update(l2=None, relative_l2=None, max_abs=None)
        return result
    delta = left.astype(np.float64) - right.astype(np.float64)
    reference = float(np.linalg.norm(right.astype(np.float64)))
    delta_l2 = float(np.linalg.norm(delta))
    result.update(
        l2=delta_l2,
        relative_l2=delta_l2 / reference if reference else (0.0 if delta_l2 == 0.0 else None),
        max_abs=float(np.max(np.abs(delta))) if delta.size else 0.0,
    )
    return result


def _array_metrics(array: np.ndarray | None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = dict(metadata or {})
    if array is None:
        result.setdefault("status", "missing")
        return result
    finite = bool(np.isfinite(array).all())
    result["finite"] = finite
    result["count"] = int(array.size)
    if finite:
        result["l2"] = float(np.linalg.norm(array.astype(np.float64)))
        result["min"] = float(np.min(array)) if array.size else None
        result["max"] = float(np.max(array)) if array.size else None
    else:
        result["status"] = "nonfinite"
        result.update(l2=None, min=None, max=None)
    return result


def _step_value(row: Mapping[str, Any], *keys: str, default: Any = 0) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    nested = row.get("ops", row.get("operations", {}))
    if isinstance(nested, Mapping):
        for key in keys:
            if key in nested:
                return nested[key]
    return default


def _step_number(row: Mapping[str, Any], *keys: str) -> int:
    return _safe_int(_step_value(row, *keys), 0)


def _step_index(row: Mapping[str, Any], fallback: int) -> int:
    return _safe_int(row.get("step", row.get("index", fallback)), fallback)


def _step_token(row: Mapping[str, Any]) -> int | None:
    for key in ("input_id", "token", "id", "input_token"):
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _step_stem(path: Mapping[str, Any], row: Mapping[str, Any], fallback: int) -> str:
    for key in ("stem", "file_stem", "prefix"):
        value = row.get(key)
        if value:
            return str(value)
    name = str(path.get("name", path.get("id", "path")))
    return f"{name}-s{_step_index(row, fallback):03d}"


def _file_name(path: Mapping[str, Any], row: Mapping[str, Any], suffix: str, fallback: int) -> str:
    aliases = {
        "raw_receiver": ("raw_receiver_file", "raw_receiver", "receiver_file"),
        "mlp_input": ("mlp_input_file", "mlp_input", "norm_file"),
        "mlp_gate": ("mlp_gate_file", "mlp_gate", "gate_file"),
        "mlp_up": ("mlp_up_file", "mlp_up", "up_file"),
        "mlp_return": ("mlp_return_file", "mlp_return", "write_file"),
        "output": ("output_file", "output"),
        "native_write": ("native_write_file", "native_write", "exact_write_file"),
    }
    for key in aliases.get(suffix, ()):
        value = row.get(key)
        if value:
            return str(value)
    return f"{_step_stem(path, row, fallback)}-{suffix}.f32"


def _read_step_array(raw: Path, path: Mapping[str, Any], row: Mapping[str, Any], suffix: str, count: int, fallback: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    return _array(raw, _file_name(path, row, suffix, fallback), count)


def _prefix(path: Mapping[str, Any], through: int) -> tuple[int, ...]:
    rows = path.get("steps", [])
    if isinstance(rows, list):
        values = [_step_token(row) for row in rows[: through + 1] if isinstance(row, Mapping)]
        return tuple(value for value in values if value is not None)
    consumed = path.get("consumed_ids", [])
    return tuple(int(value) for value in consumed[: through + 1])


def _prefix_digest(prefix: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(prefix), separators=(",", ":")).encode()).hexdigest()


def _readout_info(path: Mapping[str, Any], boundary: str) -> Mapping[str, Any] | None:
    block = path.get(boundary)
    if not isinstance(block, Mapping):
        return None
    info = block.get("logits", block.get("readout", block))
    return info if isinstance(info, Mapping) else None


def _readout_array(raw: Path, path: Mapping[str, Any], boundary: str, vocab: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    info = _readout_info(path, boundary)
    if info is None:
        return None, {"status": "missing_metadata", "boundary": boundary}
    filename = info.get("file", info.get("path"))
    if filename is None:
        # The current C driver always emits a file field; this fallback keeps
        # the analyzer useful for compact hand-authored mechanics fixtures.
        filename = f"{path.get('name', 'path')}-{boundary}-logits.f32"
    return _array(raw, filename, vocab)


def _boundary_labels(path: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(label for label in ("first", "after_answer0", "after_followup", "final", "initial", "first_end", "followup", "second_end") if _readout_info(path, label) is not None)


def _path_ops(path: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in path.get("steps", []) if isinstance(row, Mapping)]
    sums = {
        "steps": len(rows),
        "basis_q_calls": sum(_step_number(row, "basis_q_calls", "q_calls", "derived_q_calls") for row in rows),
        "basis_g_calls": sum(_step_number(row, "basis_g_calls", "g_calls", "derived_g_calls") for row in rows),
        "basis_u_calls": sum(_step_number(row, "basis_u_calls", "u_calls", "derived_u_calls") for row in rows),
        "selected_gate_up_calls": sum(_step_number(row, "selected_gate_up_calls", "gate_up_calls", "native_gate_up_calls") for row in rows),
        "selected_down_calls": sum(_step_number(row, "selected_down_calls", "down_calls", "native_down_calls") for row in rows),
    }
    sums["derived_projection_calls"] = sums["basis_q_calls"] + sums["basis_g_calls"] + sums["basis_u_calls"]
    sums["known_selected_matvec_calls"] = sums["selected_gate_up_calls"] + sums["selected_down_calls"]
    return sums


OP_KEYS = (
    "steps",
    "basis_q_calls",
    "basis_g_calls",
    "basis_u_calls",
    "selected_gate_up_calls",
    "selected_down_calls",
    "derived_projection_calls",
    "known_selected_matvec_calls",
)


def _aggregate_op_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Sum executed row counters without conflating path classes.

    The C driver's ``reference_steps`` is the native-own trajectory only;
    same-ID native replays are a separate physical pass.  Keeping this
    helper generic lets the report expose both classes and an explicit total
    instead of silently treating a replay count as a token count.
    """

    return {key: sum(_safe_int(row.get(key), 0) for row in rows) for key in OP_KEYS}


def _path_op_rows(paths: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    per_path: dict[str, dict[str, int]] = {}
    for index, path in enumerate(paths):
        if not isinstance(path, Mapping):
            continue
        name = str(path.get("name", path.get("arm", f"path-{index}")))
        # Names are intended to be unique in the C ledger.  Preserve a
        # duplicate instead of overwriting a prior trajectory in a malformed
        # or hand-authored fixture.
        if name in per_path:
            name = f"{name}#{index}"
        per_path[name] = _path_ops(path)
    return per_path, _aggregate_op_rows(per_path.values())


def _call_ledger_check(path: Mapping[str, Any], ops: Mapping[str, Any]) -> dict[str, Any]:
    """Compare observable per-step calls with the declared branch contract.

    This is a mechanics observation.  It does not decide whether a branch is
    useful, and B3 remains intentionally unconstrained because its historical
    subspace implementation has a different internal call decomposition.
    """

    arm = str(path.get("arm", ""))
    steps = _safe_int(ops.get("steps"), 0)
    expected: dict[str, int] | None = None
    if arm in ("source64", "random64"):
        expected = {"basis_q_calls": steps, "basis_g_calls": steps, "basis_u_calls": steps, "selected_gate_up_calls": 0, "selected_down_calls": steps}
    elif arm in ("zero", "deletion"):
        expected = {"basis_q_calls": 0, "basis_g_calls": 0, "basis_u_calls": 0, "selected_gate_up_calls": 0, "selected_down_calls": 0}
    elif arm in ("native", "native_own"):
        expected = {"basis_q_calls": 0, "basis_g_calls": 0, "basis_u_calls": 0, "selected_gate_up_calls": 2 * steps, "selected_down_calls": steps}
    if expected is None:
        return {"status": "not_applicable", "arm": arm, "observed": dict(ops)}
    mismatches = {
        key: {"expected": value, "observed": _safe_int(ops.get(key), 0)}
        for key, value in expected.items()
        if _safe_int(ops.get(key), 0) != value
    }
    return {"status": "ok" if not mismatches else "mismatch", "arm": arm, "expected": expected, "observed": dict(ops), "mismatches": mismatches}


def _primary_array_check(local: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("raw_receiver", "normalized_input", "candidate_write")
    missing: list[dict[str, Any]] = []
    nonfinite: list[dict[str, Any]] = []
    for row in local:
        for key in required:
            metadata = row.get(key, {})
            status = metadata.get("status") if isinstance(metadata, Mapping) else None
            if status in ("missing", "read_error", "invalid_bytes", "shape_mismatch"):
                missing.append({"path": row.get("path"), "step": row.get("step"), "array": key, "status": status})
            if status == "nonfinite" or (isinstance(metadata, Mapping) and metadata.get("finite") is False):
                nonfinite.append({"path": row.get("path"), "step": row.get("step"), "array": key})
    status = "ok" if not missing and not nonfinite else "evidence_incomplete"
    return {"status": status, "required_arrays": list(required), "observations": len(local), "missing": missing, "nonfinite": nonfinite}


def _primary_readout_check(raw: Path, paths: Sequence[Mapping[str, Any]], profile: Mapping[str, int]) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    nonfinite: list[dict[str, Any]] = []
    checked = 0
    for path in paths:
        for boundary in ("first", "after_answer0", "after_followup", "final"):
            # Older finite smokes only had first/final.  Newer drivers expose
            # the intermediate clocks; require every boundary they declare.
            if boundary not in path and boundary not in ("first", "final"):
                continue
            checked += 1
            values, metadata = _readout_array(raw, path, boundary, int(profile["vocab"]))
            if metadata.get("status") in ("missing", "missing_metadata", "read_error", "shape_mismatch", "invalid_bytes"):
                missing.append({"path": path.get("name"), "boundary": boundary, "status": metadata.get("status")})
            if metadata.get("status") == "nonfinite" or (values is not None and not np.isfinite(values).all()):
                nonfinite.append({"path": path.get("name"), "boundary": boundary})
    status = "ok" if not missing and not nonfinite else "evidence_incomplete"
    return {"status": status, "checked": checked, "missing": missing, "nonfinite": nonfinite}


def _token_authority(path: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in path.get("steps", []) if isinstance(row, Mapping)]
    actual = [_step_token(row) for row in rows]
    prompt = list(case.get("prompt_tokens", case.get("prompt_ids", [])))
    follow = list(case.get("followup_tokens", case.get("followup_ids", [])))
    answer0 = path.get("answer0", {})
    answer1 = path.get("answer1", {})
    answer0_ids = list(answer0.get("ids", [])) if isinstance(answer0, Mapping) else []
    answer1_ids = list(answer1.get("ids", [])) if isinstance(answer1, Mapping) else []
    skip = _safe_int(path.get("followup_skip", 0), 0)
    expected = prompt + answer0_ids + follow[skip:] + answer1_ids
    mismatch = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]), None)
    complete = len(actual) == len(expected) and mismatch is None and all(value is not None for value in actual)
    return {
        "status": "ok" if complete else "mismatch",
        "autonomous_declared": bool(path.get("autonomous", True)),
        "actual_count": len(actual),
        "expected_count": len(expected),
        "mismatch_index": mismatch,
        "actual_prefix": actual[:8],
        "expected_prefix": expected[:8],
        "all_delivered_consumed": bool(path.get("all_delivered_consumed", False)),
        "answer0_stopped": bool(answer0.get("stopped", False)) if isinstance(answer0, Mapping) else None,
        "answer0_truncated": bool(answer0.get("truncated", False)) if isinstance(answer0, Mapping) else None,
        "answer1_stopped": bool(answer1.get("stopped", False)) if isinstance(answer1, Mapping) else None,
        "answer1_truncated": bool(answer1.get("truncated", False)) if isinstance(answer1, Mapping) else None,
    }


def _local_write_observations(raw: Path, path: Mapping[str, Any], profile: Mapping[str, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    hidden = int(profile["hidden"])
    for fallback, row in enumerate(path.get("steps", [])):
        if not isinstance(row, Mapping):
            continue
        stem = _step_stem(path, row, fallback)
        receiver, receiver_meta = _read_step_array(raw, path, row, "raw_receiver", hidden, fallback)
        norm, norm_meta = _read_step_array(raw, path, row, "mlp_input", hidden, fallback)
        candidate_write, candidate_meta = _read_step_array(raw, path, row, "mlp_return", hidden, fallback)
        native_name = _file_name(path, row, "native_write", fallback)
        native_write, native_meta = _array(raw, native_name, hidden)
        result.append(
            {
                "case": path.get("case"),
                "arm": path.get("arm", path.get("name")),
                "path": path.get("name"),
                "step": _step_index(row, fallback),
                "position": row.get("position"),
                "prefix_sha256": _prefix_digest(_prefix(path, fallback)),
                "raw_receiver": _array_metrics(receiver, receiver_meta),
                "normalized_input": _array_metrics(norm, norm_meta),
                "candidate_write_file": candidate_meta.get("file"),
                "native_write_file": native_meta.get("file"),
                "candidate_write": _array_metrics(candidate_write, candidate_meta),
                "native_write": _array_metrics(native_write, native_meta),
                "write_error": _difference(candidate_write, native_write),
                "native_write_reference_available": native_write is not None,
                "receiver_norm_l2": float(np.linalg.norm(receiver.astype(np.float64))) if receiver is not None and np.isfinite(receiver).all() else None,
                "normalized_input_norm_l2": float(np.linalg.norm(norm.astype(np.float64))) if norm is not None and np.isfinite(norm).all() else None,
                "stem": stem,
            }
        )
    return result


def _primary_paths(driver: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result = []
    for case in driver.get("primary_cases", []):
        if not isinstance(case, Mapping):
            continue
        for path in case.get("arms", []):
            if isinstance(path, Mapping):
                result.append((case, path))
    return result


def _reference_record(driver: Mapping[str, Any], case_id: Any) -> Mapping[str, Any] | None:
    for record in driver.get("references", []):
        if isinstance(record, Mapping) and record.get("case") == case_id:
            return record
    return None


def _same_id_paths(record: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(record, Mapping):
        return []
    for key in ("same_ids", "same_id", "native_same_ids", "same_id_native", "candidate_replays"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _match_same_id(primary: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    primary_name = str(primary.get("name", ""))
    primary_arm = str(primary.get("arm", ""))
    for candidate in candidates:
        hints = [candidate.get(key) for key in ("parent", "parent_name", "candidate", "candidate_name", "primary", "primary_name", "source_primary")]
        if primary_name and primary_name in [str(value) for value in hints if value is not None]:
            return candidate
        name = str(candidate.get("name", ""))
        if primary_name and primary_name in name:
            return candidate
        if primary_arm and primary_arm in name and ("same" in name.lower() or "native" in name.lower()):
            return candidate
    return None


def _common_history_audit(raw: Path, case: Mapping[str, Any], profile: Mapping[str, int]) -> dict[str, Any]:
    groups: dict[tuple[int, ...], list[tuple[str, Mapping[str, Any], str]]] = defaultdict(list)
    for path in case.get("arms", []):
        if not isinstance(path, Mapping):
            continue
        for fallback, row in enumerate(path.get("steps", [])):
            if not isinstance(row, Mapping):
                continue
            prefix = _prefix(path, fallback)
            filename = _file_name(path, row, "raw_receiver", fallback)
            groups[prefix].append((str(path.get("name", path.get("arm", "path"))), row, filename))
    comparisons: list[dict[str, Any]] = []
    missing = 0
    mismatches = 0
    for prefix, entries in groups.items():
        if len(entries) < 2:
            continue
        arrays: list[tuple[str, np.ndarray | None, dict[str, Any]]] = []
        for name, _row, filename in entries:
            values, metadata = _array(raw, filename, int(profile["hidden"]))
            arrays.append((name, values, metadata))
            if values is None:
                missing += 1
        reference_name, reference, _ = arrays[0]
        for name, values, metadata in arrays[1:]:
            difference = _difference(values, reference)
            if difference.get("status") != "ok" or not difference.get("byte_exact", False):
                mismatches += 1
            comparisons.append(
                {
                    "prefix_sha256": _prefix_digest(prefix),
                    "prefix_length": len(prefix),
                    "position": entries[0][1].get("position"),
                    "reference_arm": reference_name,
                    "arm": name,
                    "receiver": difference,
                    "reference_file": arrays[0][2].get("file"),
                    "file": metadata.get("file"),
                }
            )
    return {
        "status": "observed",
        "groups_compared": sum(1 for entries in groups.values() if len(entries) >= 2),
        "pair_comparisons": len(comparisons),
        "byte_mismatch_count": mismatches,
        "missing_count": missing,
        "comparisons": comparisons,
        "grouping": "complete_consumed_token_prefix_and_position; token ID alone is never used",
    }


def _same_id_audit(raw: Path, primary: Mapping[str, Any], native_same: Mapping[str, Any], profile: Mapping[str, int]) -> dict[str, Any]:
    comparisons = []
    primary_by_prefix: dict[tuple[int, ...], tuple[int, Mapping[str, Any]]] = {}
    same_by_prefix: dict[tuple[int, ...], tuple[int, Mapping[str, Any]]] = {}
    for fallback, row in enumerate(primary.get("steps", [])):
        if isinstance(row, Mapping):
            primary_by_prefix[_prefix(primary, fallback)] = (fallback, row)
    for fallback, row in enumerate(native_same.get("steps", [])):
        if isinstance(row, Mapping):
            same_by_prefix[_prefix(native_same, fallback)] = (fallback, row)
    for prefix in sorted(set(primary_by_prefix) & set(same_by_prefix), key=lambda value: (len(value), value)):
        p_index, p_row = primary_by_prefix[prefix]
        n_index, n_row = same_by_prefix[prefix]
        item: dict[str, Any] = {
            "prefix_sha256": _prefix_digest(prefix),
            "prefix_length": len(prefix),
            "candidate_step": p_index,
            "native_same_id_step": n_index,
        }
        for suffix, count in (("raw_receiver", profile["hidden"]), ("mlp_input", profile["hidden"])):
            candidate, candidate_meta = _read_step_array(raw, primary, p_row, suffix, count, p_index)
            native, native_meta = _read_step_array(raw, native_same, n_row, suffix, count, n_index)
            item[suffix] = _difference(candidate, native)
            item[f"{suffix}_files"] = {"candidate": candidate_meta.get("file"), "native_same_id": native_meta.get("file")}
        comparisons.append(item)
    return {
        "status": "observed" if comparisons else "no_common_prefixes",
        "candidate": primary.get("name"),
        "native_same_id": native_same.get("name"),
        "comparisons": comparisons,
        "receiver_byte_mismatch_count": sum(not row["raw_receiver"].get("byte_exact", False) for row in comparisons),
        "norm_byte_mismatch_count": sum(not row["mlp_input"].get("byte_exact", False) for row in comparisons),
        "grouping": "complete_consumed_token_prefix_and_position",
    }


def _readout_audit(raw: Path, own: Mapping[str, Any], other: Mapping[str, Any], profile: Mapping[str, int], relation: str) -> list[dict[str, Any]]:
    result = []
    for boundary in ("first", "after_answer0", "after_followup", "final", "initial", "first_end", "followup", "second_end"):
        if _readout_info(own, boundary) is None or _readout_info(other, boundary) is None:
            continue
        left, left_meta = _readout_array(raw, own, boundary, int(profile["vocab"]))
        right, right_meta = _readout_array(raw, other, boundary, int(profile["vocab"]))
        entry = {
            "relation": relation,
            "boundary": boundary,
            "left_path": own.get("name"),
            "right_path": other.get("name"),
            "difference": _difference(left, right),
            "left_readout": left_meta,
            "right_readout": right_meta,
            "relative_l2_denominator": f"right_array_l2:{other.get('name', 'path')}:{boundary}",
        }
        if left is not None and np.isfinite(left).all():
            entry["left_top"] = int(np.argmax(left))
        if right is not None and np.isfinite(right).all():
            entry["right_top"] = int(np.argmax(right))
        result.append(entry)
    return result


def _locate_compiled(raw: Path, driver: Mapping[str, Any]) -> Path | None:
    # The worker copies these files into the result archive before calling
    # analyze.  Prefer that immutable local copy over a host-specific source
    # path retained in the driver ledger.
    candidates = [raw / "compiled", raw.parent / "compiled", raw.parent / "payload" / "compiled"]
    values = [driver.get("compiled_dir"), driver.get("compiled_bank"), driver.get("compiled")]
    for value in values:
        path = _as_path(raw, value)
        if path is not None:
            candidates.append(path)
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / name).is_file() for name in ("source-QT.f32", "source-GQ.f32", "source-UQ.f32")):
            return candidate
    return None


def _compiled_audit(raw: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    directory = _locate_compiled(raw, driver)
    # The C driver calls this field ``basis_file_sha256``; the first Python
    # smoke schema called it ``basis_sha256``.  Accept both while keeping the
    # report's keys stable.  A missing expected hash is an integrity
    # observation, never a quality verdict.
    expected = driver.get("basis_file_sha256", driver.get("basis_sha256", {}))
    if not isinstance(expected, Mapping):
        expected = {}
    names = {
        "source_qt": "source-QT.f32",
        "source_gq": "source-GQ.f32",
        "source_uq": "source-UQ.f32",
        "random_qt": "random-QT.f32",
        "random_gq": "random-GQ.f32",
        "random_uq": "random-UQ.f32",
    }
    checks = []
    for key, filename in names.items():
        path = directory / filename if directory is not None else None
        observed = sha(path) if path is not None and path.is_file() else None
        checks.append({"key": key, "file": filename, "expected_sha256": expected.get(key), "observed_sha256": observed, "match": bool(observed and expected.get(key) == observed) if expected.get(key) else None})
    manifest = None
    manifest_path = directory / "bank_manifest.json" if directory is not None else None
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
    manifest_checks = []
    manifest_names = {
        "source_qt": ("source", "q_t", "source-QT.f32"),
        "source_gq": ("source", "gate", "source-GQ.f32"),
        "source_uq": ("source", "up", "source-UQ.f32"),
        "random_qt": ("matched_random", "q_t", "random-QT.f32"),
        "random_gq": ("matched_random", "gate", "random-GQ.f32"),
        "random_uq": ("matched_random", "up", "random-UQ.f32"),
    }
    for key, (bank_name, component, filename) in manifest_names.items():
        record = None
        if isinstance(manifest, Mapping):
            bank_record = manifest.get("banks", {}).get(bank_name, {}) if isinstance(manifest.get("banks", {}), Mapping) else {}
            record = bank_record.get(component) if isinstance(bank_record, Mapping) else None
        manifest_hash = record.get("sha256") if isinstance(record, Mapping) else None
        observed_hash = next((item.get("observed_sha256") for item in checks if item.get("key") == key), None)
        manifest_checks.append({"key": key, "file": filename, "manifest_sha256": manifest_hash, "observed_sha256": observed_hash, "match": bool(manifest_hash and observed_hash and manifest_hash == observed_hash)})
    source_join = {"status": "unavailable"}
    input_manifest_path = None
    if directory is not None:
        for name in ("source-input-manifest.json", "input-manifest.json", "compile-input/input-manifest.json"):
            candidate = directory / name
            if candidate.is_file():
                input_manifest_path = candidate
                break
    if input_manifest_path is None:
        input_manifest_path = _as_path(raw, driver.get("compiler_input_dir"))
    if input_manifest_path is not None and input_manifest_path.is_dir():
        input_manifest_path = input_manifest_path / "input-manifest.json"
    if input_manifest_path is not None and input_manifest_path.is_file() and isinstance(manifest, Mapping):
        try:
            input_manifest = json.loads(input_manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            input_manifest = None
        weights = manifest.get("source", {}).get("weights", {}) if isinstance(manifest.get("source"), Mapping) else {}
        input_names = {"G": "gate", "U": "up", "gamma": "gamma"}
        joins = []
        if isinstance(input_manifest, Mapping) and isinstance(weights, Mapping):
            for input_key, manifest_key in input_names.items():
                input_record = input_manifest.get(input_key, {})
                input_hash = input_record.get("sha256") if isinstance(input_record, Mapping) else None
                compiled_record = weights.get(manifest_key, {})
                compiled_hash = compiled_record.get("sha256") if isinstance(compiled_record, Mapping) else None
                joins.append({"input": input_key, "compiled": manifest_key, "input_sha256": input_hash, "compiled_sha256": compiled_hash, "match": bool(input_hash and compiled_hash and input_hash == compiled_hash)})
        source_join = {"status": "observed", "input_manifest": _logical_file(raw, input_manifest_path), "joins": joins}
    return {
        "status": "observed" if directory is not None else "unavailable",
        "directory": _logical_file(raw, directory) if directory is not None else None,
        "checks": checks,
        "manifest_checks": manifest_checks,
        "manifest_present": bool(manifest_path and manifest_path.is_file()),
        "source_hash_join": source_join,
        "all_expected_files_match": bool(checks) and all(item["match"] is True for item in checks if item["expected_sha256"]),
    }


def _seal_audit(raw: Path, driver: Mapping[str, Any]) -> dict[str, Any]:
    driver_path = raw / "driver.json"
    seal_path = raw / "primary-seal.json"
    issues: list[str] = []
    checks: dict[str, Any] = {}
    if not driver_path.is_file() or not seal_path.is_file():
        return {"status": "unavailable", "issues": ["driver.json or primary-seal.json missing"], "checks": checks}
    try:
        seal = json.loads(seal_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "issues": [f"primary seal JSON: {exc}"], "checks": checks}
    raw_driver = driver_path.read_bytes()
    prefix_bytes = _safe_int(driver.get("primary_seal_bytes"), -1)
    expected_hash = driver.get("primary_seal_sha256")
    observed_hash = hashlib.sha256(raw_driver[:prefix_bytes]).hexdigest() if 0 <= prefix_bytes <= len(raw_driver) else None
    checks["driver_prefix_bytes"] = {"expected": prefix_bytes, "observed_available": len(raw_driver), "valid_extent": observed_hash is not None}
    checks["driver_prefix_hash"] = {"expected": expected_hash, "observed": observed_hash, "match": bool(expected_hash and observed_hash == expected_hash)}
    checks["seal_prefix_hash"] = {"expected": seal.get("driver_prefix_sha256"), "observed": expected_hash, "match": bool(seal.get("driver_prefix_sha256") and seal.get("driver_prefix_sha256") == expected_hash)}
    if not checks["driver_prefix_hash"]["match"]:
        issues.append("driver prefix hash mismatch")
    if not checks["seal_prefix_hash"]["match"]:
        issues.append("primary-seal prefix hash mismatch")
    files = seal.get("files", [])
    file_checks = []
    for record in files if isinstance(files, list) else []:
        if not isinstance(record, Mapping):
            continue
        path = raw / str(record.get("name", ""))
        observed = sha(path) if path.is_file() else None
        match = bool(observed and observed == record.get("sha256") and path.stat().st_size == _safe_int(record.get("bytes"), -1))
        file_checks.append({"name": record.get("name"), "expected_sha256": record.get("sha256"), "observed_sha256": observed, "match": match})
        if not match:
            issues.append(f"sealed file mismatch: {record.get('name')}")
    checks["files"] = file_checks
    return {"status": "ok" if not issues else "mismatch", "issues": issues, "checks": checks}


def analyze(raw: str | Path, unused_bank: str | Path | None, driver: Mapping[str, Any], texts: Mapping[str, Any] | None) -> dict[str, Any]:
    """Analyze one completed driver result without declaring a quality winner."""

    del unused_bank  # Historical B3 files are not the source-derived bank.
    raw = Path(raw)
    profile = _profile(driver)
    observations: list[dict[str, Any]] = []
    arm_summaries: list[dict[str, Any]] = []
    all_local: list[dict[str, Any]] = []
    all_ops: dict[str, Any] = {}
    primary_paths = [path for _case, path in _primary_paths(driver)]
    token_authority: list[dict[str, Any]] = []
    common_history: list[dict[str, Any]] = []
    readout_observations: list[dict[str, Any]] = []
    readout_by_relation: dict[str, list[dict[str, Any]]] = {
        "candidate_vs_native_same_id": [],
        "candidate_vs_native_own_history": [],
        "native_own_vs_native_same_id": [],
    }
    call_checks: list[dict[str, Any]] = []
    same_id_observations: list[dict[str, Any]] = []
    native_own_paths: list[Mapping[str, Any]] = []
    same_id_paths: list[Mapping[str, Any]] = []

    for case, path in _primary_paths(driver):
        case_id = case.get("case")
        local = _local_write_observations(raw, path, profile)
        all_local.extend(local)
        ops = _path_ops(path)
        name = str(path.get("name", path.get("arm", "path")))
        all_ops[name] = ops
        call_check = _call_ledger_check(path, ops)
        call_check["path"] = name
        call_checks.append(call_check)
        authority = _token_authority(path, case)
        authority["case"] = case_id
        authority["path"] = name
        token_authority.append(authority)
        observations.extend({"kind": "local_write", **item} for item in local)
        observations.append({"kind": "token_authority", **authority})
        observations.append({"kind": "call_ledger", **call_check})
        arm_summaries.append({"case": case_id, "name": name, "arm": path.get("arm"), "tokens": ops["steps"], "ops": ops, "text": (texts or {}).get(name) if isinstance(texts, Mapping) else None})
        if str(path.get("arm", "")) not in ("native", "native_own"):
            # Local errors are defined relative to the dense write reconstructed
            # from this exact receiver; they are not task-quality scores.
            pass

    for case in driver.get("primary_cases", []):
        if isinstance(case, Mapping):
            common_history.append({"case": case.get("case"), **_common_history_audit(raw, case, profile)})
            observations.append({"kind": "common_history_raw_receiver", "case": case.get("case"), **common_history[-1]})

    for case in driver.get("primary_cases", []):
        if not isinstance(case, Mapping):
            continue
        record = _reference_record(driver, case.get("case"))
        native = record.get("native") if isinstance(record, Mapping) else None
        if not isinstance(native, Mapping):
            continue
        native_own_paths.append(native)
        same_id_paths.extend(_same_id_paths(record))
        for primary in case.get("arms", []):
            if not isinstance(primary, Mapping):
                continue
            same = _match_same_id(primary, _same_id_paths(record))
            if same is not None:
                item = _same_id_audit(raw, primary, same, profile)
                item["case"] = case.get("case")
                same_id_observations.append(item)
                observations.append({"kind": "same_id_receiver_norm", **item})
                # Keep the three causal comparisons separate.  C-N measures
                # replacement fidelity on the candidate's own path, C-H
                # measures it on a native replay of the same consumed IDs,
                # and N-H isolates the effect of history/conditioning.  Each
                # relative error names its right-array L2 denominator.
                candidate_vs_same = _readout_audit(raw, primary, same, profile, "candidate_vs_native_same_id")
                candidate_vs_own = _readout_audit(raw, primary, native, profile, "candidate_vs_native_own_history")
                readouts = _readout_audit(raw, native, same, profile, "native_own_vs_native_same_id")
                for relation, rows in (
                    ("candidate_vs_native_same_id", candidate_vs_same),
                    ("candidate_vs_native_own_history", candidate_vs_own),
                    ("native_own_vs_native_same_id", readouts),
                ):
                    tagged = [{"case": case.get("case"), **row} for row in rows]
                    readout_by_relation[relation].extend(tagged)
                    readout_observations.extend(tagged)
                    observations.extend({"kind": "readout_comparison", "case": case.get("case"), **row} for row in rows)
            else:
                same_id_observations.append({"case": case.get("case"), "candidate": primary.get("name"), "status": "not_available"})

    seal = _seal_audit(raw, driver)
    compiled = _compiled_audit(raw, driver)
    primary_arrays = _primary_array_check(all_local)
    primary_readouts = _primary_readout_check(raw, primary_paths, profile)
    aggregate_ops = _aggregate_op_rows(all_ops.values())
    native_own_ops, native_own_aggregate = _path_op_rows(native_own_paths)
    same_id_ops, same_id_aggregate = _path_op_rows(same_id_paths)
    all_physical_aggregate = _aggregate_op_rows(
        [aggregate_ops, native_own_aggregate, same_id_aggregate]
    )
    driver_work = driver.get("work", {}) if isinstance(driver.get("work", {}), Mapping) else {}
    common_mismatch_count = sum(item.get("byte_mismatch_count", 0) for item in common_history)
    source_join_status = compiled.get("source_hash_join", {}).get("status")
    compiled_status = compiled.get("status")
    compiled_expected_ok = compiled.get("all_expected_files_match")
    mechanics_checks = {
        "primary_seal": seal.get("status"),
        "compiled_program": "ok" if compiled_status == "observed" and compiled_expected_ok else compiled_status,
        "compiled_source_hash_join": "observed" if source_join_status == "observed" else source_join_status,
        "primary_arrays": primary_arrays.get("status"),
        "primary_readouts": primary_readouts.get("status"),
        "call_ledger": "ok" if all(item.get("status") in ("ok", "not_applicable") for item in call_checks) else "mismatch",
        "token_authority": "ok" if all(item.get("status") == "ok" for item in token_authority) else "mismatch",
        "common_history_receiver": "ok" if common_mismatch_count == 0 else "mismatch_observed",
    }
    mechanics_status = "complete" if all(value in ("ok", "observed") for value in mechanics_checks.values()) else "evidence_incomplete"
    instrument_status = "complete" if primary_arrays.get("status") == "ok" and primary_readouts.get("status") == "ok" else "evidence_incomplete"
    physical_token_totals = {
        "primary_path_steps": aggregate_ops["steps"],
        "native_own_steps": native_own_aggregate["steps"],
        "same_id_replay_steps": same_id_aggregate["steps"],
        "all_path_steps": all_physical_aggregate["steps"],
        "primary_path_count": len(primary_paths),
        "native_own_path_count": len(native_own_paths),
        "same_id_replay_path_count": len(same_id_paths),
        "driver_primary_steps": _safe_int(driver_work.get("primary_steps"), 0),
        # Explicitly name this as native-own: the C driver does not include
        # same-ID replay steps in reference_steps.
        "driver_reference_steps_native_own_only": _safe_int(driver_work.get("reference_steps"), 0),
        "driver_same_id_replay_count": _safe_int(driver_work.get("same_id_replays"), 0),
    }
    summary = {
        "schema": driver.get("schema"),
        "scope": "source/random/B3/zero primary paths with native references when present",
        "profile": profile,
        "primary_paths": len(_primary_paths(driver)),
        "primary_files_sealed": len(_seal_file_records(raw)),
        "local_write_observations": len(all_local),
        "finite_local_write_observations": sum(item.get("write_error", {}).get("status") == "ok" for item in all_local),
        "common_history_groups": sum(item.get("groups_compared", 0) for item in common_history),
        "common_history_receiver_mismatches": sum(item.get("byte_mismatch_count", 0) for item in common_history),
        "same_id_pairs": sum(item.get("status") != "not_available" for item in same_id_observations),
        "full_vocab_native_same_id_readouts": len(readout_observations),
        "token_authority_records": len(token_authority),
        "token_authority_mismatches": sum(item.get("status") != "ok" for item in token_authority),
        "op_counts": aggregate_ops,
        "physical_token_totals": physical_token_totals,
        "driver_work": dict(driver_work),
        "primary_seal_status": seal.get("status"),
        "compiled_program_status": compiled.get("status"),
        "mechanics_status": mechanics_status,
        "instrument_status": instrument_status,
        "trend_status": "observations",
        "terminal_status": "not_requested",
        "scientific_quality_gate": "none; signed finite/divergent observations retained",
    }
    return _jsonable(
        {
            "summary": summary,
            "arms": arm_summaries,
            "observations": observations,
            "local_writes": all_local,
            "common_history_raw_receiver": common_history,
            "same_id_receiver_norm": same_id_observations,
            "readout_native_own_vs_same_id": readout_observations,
            "readout_comparisons": readout_by_relation,
            "token_authority": token_authority,
            "op_counts": {
                "per_path": all_ops,
                "aggregate": aggregate_ops,
                "native_own_per_path": native_own_ops,
                "native_own_aggregate": native_own_aggregate,
                "same_id_per_path": same_id_ops,
                "same_id_aggregate": same_id_aggregate,
                "all_physical_aggregate": all_physical_aggregate,
                "driver": dict(driver_work),
            },
            "integrity": {
                "primary_seal": seal,
                "compiled_program": compiled,
                "primary_arrays": primary_arrays,
                "primary_readouts": primary_readouts,
                "call_ledger": call_checks,
                "mechanics_checks": mechanics_checks,
            },
        }
    )


def _seal_file_records(raw: Path) -> list[Mapping[str, Any]]:
    path = raw / "primary-seal.json"
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    files = document.get("files", [])
    return [item for item in files if isinstance(item, Mapping)] if isinstance(files, list) else []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("analysis.py is an importable worker module; use worker.py")
