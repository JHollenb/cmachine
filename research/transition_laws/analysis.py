"""Saved-only analysis for the transition-laws panel.

The C worker owns execution, token selection, and the native shadow.  This
module recomputes readout distributions from the saved FP32 decision arrays,
keeps native-shadow KL separate from closed-loop token behavior, and emits
small observation rows.  It never loads a model and never calls CUDA.

The producer schema is intentionally read by name rather than by a single
positional layout.  That lets the audit preserve a useful result when an
optional factor or boundary artifact is absent while still reporting the
missing instrument field explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROFILE = {
    "layers": 28,
    "hidden": 1536,
    "intermediate": 8960,
    "vocab": 151936,
    "rank": 256,
}
POLICIES = ("native", "p0", "p1", "p0-layer16", "p1-layer16")
_FILE_KEYS = {
    "path", "file", "filename", "label_file", "driver_file", "array_file",
    "candidate_logits_file", "shadow_logits_file", "logits_file",
    "source_file", "factor_file", "manifest_file",
}


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value)).all())
    except (TypeError, ValueError):
        return False


def _as_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _resolve(root: Path, reference: Any) -> Path | None:
    """Resolve a producer reference without allowing it to escape the run."""
    if isinstance(reference, Mapping):
        for key in ("path", "file", "filename", "relative_path"):
            if key in reference:
                return _resolve(root, reference[key])
        return None
    if not isinstance(reference, (str, os.PathLike)):
        return None
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [root / path, root / "raw" / path, root / "payload" / path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            # Absolute references in a report are accepted only when they are
            # inside the collected run.  This prevents a saved report from
            # silently reading a mutable worker checkout.
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _load_json_ref(root: Path, reference: Any) -> tuple[dict[str, Any] | None, Path | None]:
    path = _resolve(root, reference)
    if path is None:
        return None, None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, path
    return value if isinstance(value, dict) else None, path


def _find_named(value: Any, names: Sequence[str]) -> Any:
    wanted = set(names)
    if isinstance(value, Mapping):
        for key in names:
            if key in value and value[key] is not None:
                return value[key]
        for key, item in value.items():
            if str(key) in wanted and item is not None:
                return item
        for item in value.values():
            found = _find_named(item, names)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_named(item, names)
            if found is not None:
                return found
    return None


def _merge_run(root: Path, label: str, record: Any) -> tuple[dict[str, Any], Path | None]:
    if not isinstance(record, Mapping):
        return {"label": label}, None
    merged: dict[str, Any] = dict(record)
    # The worker keeps the C label under ``driver`` and keeps its producer
    # metrics beside it.  Flatten only the driver fields for the offline
    # readers; retain the nested object so the report remains auditable.
    driver = record.get("driver")
    if isinstance(driver, Mapping):
        merged = dict(driver)
        merged.update(record)
        merged["_driver"] = dict(driver)
    label_value = record.get("label")
    label_data: dict[str, Any] | None = None
    label_path: Path | None = None
    if isinstance(label_value, Mapping):
        label_data = dict(label_value)
    elif label_value is not None:
        label_data, label_path = _load_json_ref(root, label_value)
    if label_data is None:
        for key in ("label_file", "driver_file", "path", "json"):
            if key in record:
                label_data, label_path = _load_json_ref(root, record[key])
                if label_data is not None:
                    break
    if label_data:
        # Driver metadata wins for paths and timing, while the external label
        # supplies the authoritative token/decision sequence.
        base = dict(label_data)
        base.update(merged)
        merged = base
        merged["_label"] = label_data
    merged.setdefault("label", label)
    return merged, label_path


def _load_array(root: Path, reference: Any, rows: int | None, width: int | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    path = _resolve(root, reference)
    info: dict[str, Any] = {"reference": _json(reference), "path": str(path) if path else None}
    if path is None:
        info["status"] = "missing"
        return None, info
    try:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            key = archive.files[0]
            array = archive[key]
        else:
            array = np.fromfile(path, dtype="<f4")
            if rows is not None and width is not None:
                array = array.reshape(rows, width)
    except (OSError, ValueError, EOFError) as error:
        info.update({"status": "unreadable", "error": str(error)})
        return None, info
    array = np.asarray(array)
    info.update({"status": "ok", "shape": list(array.shape), "dtype": str(array.dtype), "sha256": sha256_file(path), "bytes": path.stat().st_size, "finite": _finite(array)})
    return array, info


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("softmax expects one logit row")
    pivot = np.max(values)
    weights = np.exp(values - pivot)
    total = float(np.sum(weights))
    return weights / total


def kl_divergence(reference: np.ndarray, candidate: np.ndarray, eps: float = 1e-30) -> float:
    """Compute KL(reference || candidate) from saved logit rows."""
    p = np.maximum(_softmax(reference), eps)
    q = np.maximum(_softmax(candidate), eps)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _logit_metrics(candidate: np.ndarray, shadow: np.ndarray, row: int, decision_position: int | None, answer_id: int | None) -> dict[str, Any]:
    c = np.asarray(candidate[row], dtype=np.float64)
    s = np.asarray(shadow[row], dtype=np.float64)
    c_order = np.argsort(c)[-2:][::-1]
    s_order = np.argsort(s)[-2:][::-1]
    c_top = int(c_order[0])
    s_top = int(s_order[0])
    c_margin = float(c[c_order[0]] - c[c_order[1]]) if c.size > 1 else 0.0
    s_margin = float(s[s_order[0]] - s[s_order[1]]) if s.size > 1 else 0.0
    delta = c - s
    shadow_norm = float(np.linalg.norm(s))
    return {
        "row": int(row),
        "decision_position": None if decision_position is None else int(decision_position),
        "answer_id": None if answer_id is None else int(answer_id),
        "candidate_argmax": c_top,
        "shadow_argmax": s_top,
        "candidate_margin": c_margin,
        "shadow_margin": s_margin,
        "margin_delta": c_margin - s_margin,
        "argmax_match": bool(c_top == s_top),
        "answer_matches_candidate": None if answer_id is None else bool(c_top == int(answer_id)),
        "answer_matches_shadow": None if answer_id is None else bool(s_top == int(answer_id)),
        "kl_shadow_to_candidate": kl_divergence(s, c),
        "kl_candidate_to_shadow": kl_divergence(c, s),
        "symmetric_kl": 0.5 * (kl_divergence(s, c) + kl_divergence(c, s)),
        "logit_l2": float(np.linalg.norm(delta)),
        "logit_linf": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "shadow_logit_l2": shadow_norm,
        "relative_logit_l2": float(np.linalg.norm(delta) / shadow_norm) if shadow_norm else (0.0 if not delta.any() else None),
        "candidate_finite": bool(np.isfinite(c).all()),
        "shadow_finite": bool(np.isfinite(s).all()),
    }


def _turn_answers(record: Mapping[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Return flattened answer ids, turn ids, and decision positions."""
    answers: list[int] = []
    turns: list[int] = []
    positions: list[int] = []
    raw_turns = record.get("turns", [])
    if isinstance(raw_turns, Mapping):
        raw_turns = list(raw_turns.values())
    for ti, turn in enumerate(raw_turns if isinstance(raw_turns, list) else []):
        if not isinstance(turn, Mapping):
            continue
        ids = _as_ints(turn.get("answer_ids", turn.get("generated_ids", turn.get("output_ids"))))
        p = _as_ints(turn.get("decision_positions", turn.get("positions")))
        if not p:
            # A label may record the first decision position and an ordered
            # answer.  The C worker's explicit positions remain authoritative
            # whenever available.
            first = turn.get("answer_start_position", turn.get("start_position"))
            if first is not None:
                try:
                    p = list(range(int(first), int(first) + len(ids)))
                except (TypeError, ValueError):
                    p = []
        answers.extend(ids)
        turns.extend([ti] * len(ids))
        positions.extend(p if len(p) == len(ids) else [None] * len(ids))
    if not answers:
        answers = _as_ints(record.get("answer_ids", record.get("generated_ids")))
        positions = _as_ints(record.get("decision_positions"))
        turns = [0] * len(answers)
    return answers, turns, positions


def _read_i32(root: Path | None, reference: Any) -> list[int]:
    if root is None:
        return []
    path = _resolve(root, reference)
    if path is None:
        return []
    try:
        values = np.fromfile(path, dtype="<i4")
    except (OSError, ValueError):
        return []
    return [int(x) for x in values.tolist()]


def token_authority(record: Mapping[str, Any], root: Path | None = None) -> dict[str, Any]:
    consumed = _as_ints(record.get("consumed_ids", record.get("token_ids")))
    if not consumed:
        consumed = _read_i32(root, record.get("consumed_ids_file", record.get("token_ids_file")))
    answers, _, _ = _turn_answers(record)
    turns = record.get("turns", [])
    if isinstance(turns, Mapping):
        turns = list(turns.values())
    turn_ids: list[int] = []
    complete = True
    for turn in turns if isinstance(turns, list) else []:
        if not isinstance(turn, Mapping):
            continue
        for key in ("prefill_ids", "prompt_ids", "ingested_ids", "input_ids"):
            if key in turn:
                values = turn[key]
                if isinstance(values, (list, tuple, np.ndarray)):
                    turn_ids.extend(_as_ints(values))
                else:
                    complete = False
                break
        turn_ids.extend(_as_ints(turn.get("answer_ids", turn.get("generated_ids"))))
    status = "unavailable"
    if consumed and turn_ids:
        status = "exact" if consumed == turn_ids else "mismatch"
    elif consumed:
        status = "declared_only"
    return {
        "consumed_count": len(consumed),
        "answer_count": len(answers),
        "turn_recoverable_count": len(turn_ids),
        "turn_segments_complete": complete,
        "consumed_sha256": sha256_bytes(np.asarray(consumed, dtype="<i4").tobytes()) if consumed else None,
        "status": status,
    }


def _path_records(group_label: str, value: Any) -> list[tuple[str, dict[str, Any], Mapping[str, Any] | None]]:
    """Expand a report run group into one typed record per C path."""
    if not isinstance(value, Mapping):
        return [(group_label, {"label": group_label}, None)]
    driver = value.get("driver") if isinstance(value.get("driver"), Mapping) else value
    paths = driver.get("paths", []) if isinstance(driver, Mapping) else []
    group_metrics = value.get("metrics") if isinstance(value.get("metrics"), Mapping) else None
    if not isinstance(paths, list) or not paths:
        record = dict(driver) if isinstance(driver, Mapping) else dict(value)
        record.setdefault("label", group_label)
        return [(group_label, record, group_metrics)]
    result: list[tuple[str, dict[str, Any], Mapping[str, Any] | None]] = []
    for index, path in enumerate(paths):
        if not isinstance(path, Mapping):
            continue
        record = dict(path)
        record.setdefault("label", group_label)
        record["_group_label"] = group_label
        record["_group_driver"] = driver
        key = f"{group_label}/{record.get('case_id', index)}"
        result.append((key, record, group_metrics))
    return result


def _task_truth(root: Path, record: Mapping[str, Any], answers: Sequence[int]) -> dict[str, Any]:
    """Apply only descriptive, strict task checks when expected answers exist."""
    inputs: dict[str, Any] | None = None
    for candidate in (root / "inputs.json", root / "payload" / "inputs.json"):
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text())
                if isinstance(value, dict):
                    inputs = value
                    break
            except json.JSONDecodeError:
                pass
    case_id = str(record.get("case_id", record.get("case", record.get("id", ""))))
    expected: list[Any] | None = None
    fmt: str | None = None
    if inputs:
        for group in ("source", "monitor", "heldout"):
            for item in inputs.get(group, []):
                if isinstance(item, Mapping) and str(item.get("id")) == case_id:
                    expected = list(item.get("expected", [])) if isinstance(item.get("expected"), list) else None
                    fmt = str(item.get("format", "")) or None
                    break
    texts = record.get("decoded_answers", record.get("answer_text", record.get("decoded")))
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, (list, tuple)):
        texts = []
    text_values = [str(x).strip() for x in texts]
    rows: list[dict[str, Any]] = []
    if expected is not None:
        for index, wanted in enumerate(expected):
            text = text_values[index] if index < len(text_values) else None
            if fmt == "integer":
                valid = bool(text is not None and re.fullmatch(r"[+-]?\d+", text))
                observed: Any = int(text) if valid and text is not None else None
                match = bool(valid and int(wanted) == observed)
            elif fmt == "word":
                valid = bool(text is not None and re.fullmatch(r"[A-Za-z]+", text))
                observed = text.lower() if valid and text is not None else None
                match = bool(valid and str(wanted).lower() == observed)
            else:
                valid = None
                observed = text
                match = None
            rows.append({"index": index, "expected": wanted, "observed": observed, "format": fmt, "strict_valid": valid, "match": match})
    return {"case_id": case_id, "format": fmt, "expected_available": expected is not None, "rows": rows, "scope": "observation-only"}


def _array_reference(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    direct = _find_named(record, names)
    return direct


def _factor_summary(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    found: dict[str, dict[str, Any]] = {}
    records: list[tuple[str, Any]] = []

    def walk(value: Any, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                low = name.lower()
                if any(token in low for token in ("factor", "svd", "components", "basis", "egamma", "lg", "lu")) and isinstance(item, (str, Mapping)):
                    records.append((".".join(key_path + (name,)), item))
                walk(item, key_path + (name,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, key_path + (str(index),))
    walk(report)
    for logical, reference in records:
        path = _resolve(root, reference)
        if path is None:
            found[logical] = {"reference": _json(reference), "status": "missing"}
            continue
        item: dict[str, Any] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size, "status": "present"}
        if isinstance(reference, Mapping):
            for key in ("shape", "dtype", "rank", "rows", "columns"):
                if key in reference:
                    item[key] = _json(reference[key])
            shape = reference.get("shape")
            if isinstance(shape, (list, tuple)) and reference.get("dtype", "float32") in ("float32", "<f4", "f4"):
                item["expected_bytes"] = int(np.prod(shape)) * 4
                item["size_matches_shape"] = item["expected_bytes"] == item["bytes"]
        found[logical] = item
    return {"files": found, "rank": report.get("rank", report.get("compiler", {}).get("rank")), "maps_per_layer": len(found)}


def _basis_summary(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the producer's per-layer factor hashes and sample lineage.

    The producer stores the actual E bytes under ``raw/p{0,1}/E-<layer>.f32``
    and a compact ``basis.json``.  Reading those arrays is cheap compared with
    the model run and catches a report that merely copied the producer's
    metadata.  Recomputing the randomized SVD itself would not be an
    independent test unless the exact torch version and CUDA reduction order
    were also reproduced, so SVD residuals remain producer observations while
    rank/shape/hash/lineage are independently checked here.
    """
    compiler = report.get("compiler", {})
    programs: dict[str, Any] = {}
    for program in ("p0", "p1"):
        basis_ref = None
        if isinstance(compiler, Mapping) and isinstance(compiler.get(program), Mapping):
            basis_ref = compiler[program].get("basis_file", f"{program}/basis.json")
        basis, basis_path = _load_json_ref(root, basis_ref or f"{program}/basis.json")
        item: dict[str, Any] = {"program": program, "basis_file": str(basis_path.relative_to(root)) if basis_path else None, "status": "missing"}
        if basis is None:
            programs[program] = item
            continue
        layer_rows = basis.get("layers", [])
        layer_checks: list[dict[str, Any]] = []
        for layer in range(int(report.get("layers", PROFILE["layers"]))):
            row = layer_rows[layer] if isinstance(layer_rows, list) and layer < len(layer_rows) and isinstance(layer_rows[layer], Mapping) else {}
            ref = row.get("E", {}) if isinstance(row, Mapping) else {}
            e_path = _resolve(root, f"{program}/E-{layer}.f32")
            expected_bytes = 4 * PROFILE["rank"] * PROFILE["hidden"]
            check: dict[str, Any] = {"layer": layer, "rank": row.get("rank"), "samples": row.get("samples"), "path": str(e_path.relative_to(root)) if e_path else None}
            if e_path is None:
                check["status"] = "missing"
            else:
                actual_hash = sha256_file(e_path)
                actual_bytes = e_path.stat().st_size
                check.update({"status": "ok", "sha256": actual_hash, "bytes": actual_bytes, "shape": [PROFILE["rank"], PROFILE["hidden"]], "size_matches_shape": actual_bytes == expected_bytes, "hash_matches_basis": not isinstance(ref, Mapping) or ref.get("sha256") in (None, actual_hash), "basis_bytes_matches": not isinstance(ref, Mapping) or ref.get("bytes") in (None, actual_bytes)})
            layer_checks.append(check)
        item.update({"status": "ok", "method": basis.get("method"), "native_sampling_sha256": basis.get("native_sampling_sha256"), "own_sampling_sha256": basis.get("own_sampling_sha256"), "layers": layer_checks, "all_layer_files_present": all(row.get("status") == "ok" for row in layer_checks)})
        programs[program] = item

    sampling: dict[str, Any] = {}
    for name in ("source-native", "source-compact"):
        value, path = _load_json_ref(root, f"{name}-sampling.json")
        entry: dict[str, Any] = {"path": str(path.relative_to(root)) if path else None, "status": "missing"}
        if value is not None:
            shape = value.get("shape")
            lineage = value.get("lineage", [])
            selected_ref = value.get("samples", {})
            selected_path = _resolve(root, f"{name}-selected.f32")
            selected_bytes = selected_path.stat().st_size if selected_path else None
            lineage_checks = []
            for row in lineage if isinstance(lineage, list) else []:
                if not isinstance(row, Mapping):
                    lineage_checks.append({"status": "invalid"})
                    continue
                source_path = _resolve(root, row.get("source_file"))
                source_record = row.get("source", {}) if isinstance(row.get("source"), Mapping) else {}
                indices = [int(x) for x in row.get("indices", [])]
                source_ok = source_path is not None and source_record.get("sha256", sha256_file(source_path) if source_path else None) == (sha256_file(source_path) if source_path else None)
                rows_count = int(row.get("rows", 0))
                lineage_checks.append({"case_id": row.get("case_id"), "indices_count": len(indices), "indices_unique": len(set(indices)) == len(indices), "indices_in_range": bool(indices) and min(indices) >= 0 and max(indices) < rows_count, "source_hash_matches": source_ok, "status": "ok" if len(indices) == 64 and len(set(indices)) == 64 and source_ok else "invalid"})
            entry.update({"status": "ok", "shape": shape, "lineage_count": len(lineage) if isinstance(lineage, list) else None, "selected_file": str(selected_path.relative_to(root)) if selected_path else None, "selected_bytes": selected_bytes, "expected_bytes": 384 * PROFILE["layers"] * PROFILE["hidden"] * 4, "shape_matches": shape == [384, PROFILE["layers"], PROFILE["hidden"]], "lineage_checks": lineage_checks, "lineage_rows_64_each": bool(len(lineage_checks) == 6 and all(row.get("status") == "ok" for row in lineage_checks)), "selected_hash_matches": bool(isinstance(selected_ref, Mapping) and selected_path is not None and selected_ref.get("sha256") in (None, sha256_file(selected_path)))})
        sampling[name] = entry
    return {"programs": programs, "sampling": sampling, "scope": "hash/shape/lineage recomputation; randomized-SVD residuals remain observations"}


def _cost_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        low = str(key).lower()
        if any(token in low for token in ("mac", "call", "byte", "gpu", "timing", "seconds", "work")):
            if isinstance(value, (int, float, str, bool)):
                result[str(key)] = _json(value)
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean(name: str, subset: Sequence[Mapping[str, Any]]) -> float | None:
        values = [float(row[name]) for row in subset if isinstance(row.get(name), (int, float)) and math.isfinite(float(row[name]))]
        return float(np.mean(values)) if values else None
    turn_groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        turn_groups[(row.get("label"), row.get("turn"))].append(row)
    turn_rows = [{"kl_shadow_to_candidate": mean("kl_shadow_to_candidate", group), "relative_logit_l2": mean("relative_logit_l2", group)} for group in turn_groups.values()]
    finite_turn_kl = [row["kl_shadow_to_candidate"] for row in turn_rows if row["kl_shadow_to_candidate"] is not None]
    finite_turn_l2 = [row["relative_logit_l2"] for row in turn_rows if row["relative_logit_l2"] is not None]
    return {
        "n": len(rows),
        "mean_kl_shadow_to_candidate": mean("kl_shadow_to_candidate", rows),
        "mean_symmetric_kl": mean("symmetric_kl", rows),
        "mean_logit_l2": mean("logit_l2", rows),
        "mean_relative_logit_l2": mean("relative_logit_l2", rows),
        "equal_turn_count": len(turn_rows),
        "equal_turn_mean_kl_shadow_to_candidate": float(np.mean(finite_turn_kl)) if finite_turn_kl else None,
        "equal_turn_mean_relative_logit_l2": float(np.mean(finite_turn_l2)) if finite_turn_l2 else None,
        "argmax_match_rate": mean("argmax_match", rows),
        "answer_matches_candidate_rate": mean("answer_matches_candidate", [r for r in rows if r.get("answer_matches_candidate") is not None]),
        "answer_matches_shadow_rate": mean("answer_matches_shadow", [r for r in rows if r.get("answer_matches_shadow") is not None]),
    }


def analyze(report_or_root: Mapping[str, Any] | str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Analyze a collected report, returning a JSON-serializable dictionary."""
    if isinstance(report_or_root, Mapping):
        report = dict(report_or_root)
        root_path = Path(root or ".").resolve()
    else:
        root_path = Path(report_or_root).resolve()
        if root_path.is_file():
            root_path = root_path.parent
        report = json.loads((root_path / "report.json").read_text())
    runs = report.get("runs", {})
    if not isinstance(runs, Mapping):
        runs = {}
    output_runs: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for group_label, group_value in runs.items():
      for label, record, group_metrics in _path_records(str(group_label), group_value):
        answers, turn_ids, positions = _turn_answers(record)
        candidate_ref = _array_reference(record, ("candidate_logits_file", "candidate_logits", "candidate_logit_file"))
        shadow_ref = _array_reference(record, ("shadow_logits_file", "native_shadow_logits_file", "shadow_logits", "native_logits_file"))
        decision_count = len(positions) or len(answers) or (len(record.get("decision_positions", [])) if isinstance(record.get("decision_positions"), list) else None)
        model_spec = report.get("model_spec", report.get("model", {}))
        if not isinstance(model_spec, Mapping):
            model_spec = {}
        vocab = int(record.get("vocab", model_spec.get("vocab", PROFILE["vocab"])))
        candidate, candidate_info = _load_array(root_path, candidate_ref, decision_count, vocab)
        shadow, shadow_info = _load_array(root_path, shadow_ref, decision_count, vocab)
        if candidate is not None and candidate.ndim == 1 and decision_count:
            candidate = candidate.reshape(decision_count, -1)
        if shadow is not None and shadow.ndim == 1 and decision_count:
            shadow = shadow.reshape(decision_count, -1)
        n = min(candidate.shape[0], shadow.shape[0]) if candidate is not None and shadow is not None and candidate.ndim == 2 and shadow.ndim == 2 else 0
        metrics: list[dict[str, Any]] = []
        for index in range(n):
            pos = positions[index] if index < len(positions) else None
            ans = answers[index] if index < len(answers) else None
            metric = _logit_metrics(candidate, shadow, index, pos, ans)
            metric["turn"] = turn_ids[index] if index < len(turn_ids) else None
            metrics.append(metric)
            flat_rows.append({"label": str(label), "group_label": str(group_label), "case_id": record.get("case_id", record.get("case")), "policy": record.get("policy", record.get("program", group_label)), **metric})
        truth = _task_truth(root_path, record, answers)
        authority = token_authority(record, root_path)
        producer_metrics = group_metrics
        if producer_metrics and isinstance(producer_metrics.get("turns"), list):
            producer_metrics = next((row for row in producer_metrics["turns"] if isinstance(row, Mapping) and str(row.get("case_id")) == str(record.get("case_id"))), producer_metrics)
        recomputed_summary = _aggregate(metrics)
        metric_delta: dict[str, Any] = {}
        if producer_metrics:
            for ours, theirs in (("equal_turn_mean_kl_shadow_to_candidate", "mean_KL"), ("equal_turn_mean_relative_logit_l2", "mean_logit_relative_L2"), ("equal_turn_mean_kl_shadow_to_candidate", "equal_case_turn_KL"), ("equal_turn_mean_relative_logit_l2", "equal_case_turn_logit_relative_L2")):
                if isinstance(producer_metrics.get(theirs), (int, float)) and isinstance(recomputed_summary.get(ours), (int, float)):
                    metric_delta[ours] = float(recomputed_summary[ours]) - float(producer_metrics[theirs])
        output_runs[str(label)] = {
            "case_id": record.get("case_id", record.get("case", record.get("id"))),
            "policy": record.get("policy", record.get("program", record.get("mode"))),
            "phase": record.get("phase", record.get("split")),
            "group_label": str(group_label),
            "token_authority": authority,
            "decision_count_declared": decision_count,
            "decision_count_compared": n,
            "candidate_logits": candidate_info,
            "shadow_logits": shadow_info,
            "decision_metrics": metrics,
            "summary": recomputed_summary,
            "producer_metrics": _json(producer_metrics) if producer_metrics is not None else None,
            "producer_metric_deltas": metric_delta,
            "task_truth": truth,
            "cost": _cost_summary(record),
            "source_lineage": record.get("source_lineage", record.get("fit_lineage", {})),
        }
        observations.append({"schema": "transition-laws-observation-v1", "kind": "run", "label": str(label), "group_label": str(group_label), "case_id": output_runs[str(label)]["case_id"], "policy": output_runs[str(label)]["policy"], "metrics": output_runs[str(label)]["summary"], "producer_metric_deltas": metric_delta, "token_authority": authority["status"]})
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case_turn_policy: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in flat_rows:
        by_policy[str(row.get("policy"))].append(row)
        by_case[str(row.get("case_id"))].append(row)
        by_case_turn_policy[(row.get("case_id"), row.get("turn"), row.get("policy"))].append(row)
    case_turn_rows = [_aggregate(group) for group in by_case_turn_policy.values()]
    equal_case_turn_kl = [row["mean_kl_shadow_to_candidate"] for row in case_turn_rows if row.get("mean_kl_shadow_to_candidate") is not None]
    equal_case_turn_l2 = [row["mean_relative_logit_l2"] for row in case_turn_rows if row.get("mean_relative_logit_l2") is not None]
    summary = {
        "all_decision_rows": len(flat_rows),
        "by_policy": {key: _aggregate(rows) for key, rows in sorted(by_policy.items())},
        "by_case": {key: _aggregate(rows) for key, rows in sorted(by_case.items())},
        "equal_case_turn_mean_kl_shadow_to_candidate": float(np.mean(equal_case_turn_kl)) if equal_case_turn_kl else None,
        "equal_case_turn_mean_relative_logit_l2": float(np.mean(equal_case_turn_l2)) if equal_case_turn_l2 else None,
        "source_rows_declared": report.get("source_rows", report.get("compiler", {}).get("source_rows")),
        "own_rows_declared": report.get("own_rows", report.get("compiler", {}).get("own_rows")),
        "heldout_rows_declared": report.get("heldout_rows", report.get("compiler", {}).get("heldout_rows")),
        "native_mac_per_token": report.get("native_mac_per_token", 1543569408),
        "factor_mac_per_layer": 18743296,
        "dense_mlp_mac_per_layer": 41287680,
        "quality_scope": "observations only; no semantic terminal gate",
    }
    return {
        "schema": "transition-laws-analysis-v1",
        "model_spec": _json(report.get("model_spec", report.get("model", {}))),
        "compiler": _json(report.get("compiler", {})),
        "selection": _json(report.get("selection", {})),
        "source_manifest": _json(report.get("source_manifest", {})),
        "factor_summary": _factor_summary(root_path, report),
        "basis_verification": _basis_summary(root_path, report),
        "summary": summary,
        "runs": output_runs,
        "observations": observations,
        "limitations": [
            "Native-shadow KL is conditional on candidate token IDs.",
            "Candidate-owned continuation and native-owned continuation are separate trajectories.",
            "Native D, attention, K/V, readout, and original weights remain paid in this panel.",
            "Task truth and token agreement are observations, not a general-chat certification.",
        ],
    }


def write_outputs(target: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(target).resolve()
    result = analyze(root)
    (root / "analysis.json").write_text(json.dumps(_json(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows: list[dict[str, Any]] = []
    for label, run in result.get("runs", {}).items():
        for metric in run.get("decision_metrics", []):
            rows.append({"schema": "transition-laws-observation-v1", "label": label, "case_id": run.get("case_id"), "policy": run.get("policy"), **metric})
    (root / "observations.jsonl").write_text("".join(json.dumps(_json(row), sort_keys=True, allow_nan=False) + "\n" for row in rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = write_outputs(args.target)
    print(json.dumps({"analysis": str(args.target / 'analysis.json'), "decision_rows": result["summary"]["all_decision_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
