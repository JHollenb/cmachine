"""Independent saved-artifact audit for ``operand_composition``.

This audit deliberately does not rerun the Qwen model and does not turn a
missing optional arm into a null scientific result.  It checks the things that
make the saved observations interpretable: typed profile, path identities,
token declarations, finite FP32 arrays, factor/source custody, and any
producer-owned array seal.  Effect sizes are left to :mod:`analysis`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # package invocation
    from .analysis import (
        ARRAY_FIELDS,
        PROFILE,
        _candidate_files,
        _factor_path,
        _path_mode,
        _path_name,
        _profile,
        _read_f32,
        _rows,
        _safe_relative,
        _turns,
        _json,
        boundary_records,
        factor_layout,
        sha256_file,
        token_authority,
    )
except ImportError:  # direct ``python audit.py`` invocation
    from analysis import (  # type: ignore
        ARRAY_FIELDS,
        PROFILE,
        _candidate_files,
        _factor_path,
        _path_mode,
        _path_name,
        _profile,
        _read_f32,
        _rows,
        _safe_relative,
        _turns,
        _json,
        boundary_records,
        factor_layout,
        sha256_file,
        token_authority,
    )


def _load_json(value: Mapping[str, Any] | os.PathLike[str] | str | None, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if value is None:
        return default or {}
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text())


def _sha_from_record(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("sha256", "sha", "digest", "hash"):
            if value.get(key):
                return str(value[key])
        return None
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
        return value
    return None


def _file_value(record: Any) -> str | None:
    if isinstance(record, Mapping):
        for key in ("path", "file", "filename", "relative_path", "name"):
            if record.get(key):
                return str(record[key])
        return None
    if isinstance(record, (str, os.PathLike)):
        return str(record)
    return None


def _iter_file_records(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if _file_value(value) is not None:
            yield prefix, value
        else:
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from _iter_file_records(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from _iter_file_records(child, f"{prefix}[{i}]")
    elif isinstance(value, (str, os.PathLike)):
        yield prefix, value


def _expected_file_map(driver: Mapping[str, Any], seal: Mapping[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for source, obj in (("driver", driver), ("seal", seal)):
        if not isinstance(obj, Mapping):
            continue
        for key in ("factor_files", "factors", "basis_files", "compiled_factors", "files", "arrays", "entries", "hashes"):
            if key in obj:
                out.extend((f"{source}.{key}.{name}", record) for name, record in _iter_file_records(obj[key]))
    return out


def _resolve_record(raw: Path, record: Any) -> Path | None:
    value = _file_value(record)
    if value is None:
        return None
    return _safe_relative(raw, value)


def _path_shape_contract(field: str, p: Mapping[str, int]) -> tuple[int, ...] | None:
    return {
        "mlp-norm": (p["hidden"],),
        "mlp-write": (p["hidden"],),
        "mlp-output": (p["hidden"],),
        "receiver": (p["hidden"],),
        "q": (p["hidden"],),
        "k": (p["kv"],),
        "v": (p["kv"],),
        "attention": (p["hidden"],),
        "output": (p["hidden"],),
        "mlp-code": (p["rank"],),
        "query-code": (p["query_rank"],),
        "kv16": (p["kv"] * 2,),
        "kv18": (p["kv"] * 2,),
    }.get(field)


def _check_f32_file(path: Path, shape: Sequence[int] | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    try:
        a = _read_f32(path, shape)
    except Exception as exc:
        rec.update({"status": "invalid", "error": str(exc)})
        return rec
    rec.update({"status": "valid", "shape": list(a.shape), "finite": bool(np.isfinite(a).all()), "dtype": "<f4"})
    if not rec["finite"]:
        rec["status"] = "nonfinite"
    return rec


def audit(raw: os.PathLike[str] | str, driver: Mapping[str, Any] | os.PathLike[str] | str | None = None, inputs: Mapping[str, Any] | os.PathLike[str] | str | None = None) -> dict[str, Any]:
    raw_path = Path(raw)
    if raw_path.is_file():
        raw_path = raw_path.parent
    driver_obj = _load_json(driver or (raw_path / "driver.json" if (raw_path / "driver.json").exists() else None))
    input_obj = _load_json(inputs) if inputs is not None else {}
    profile = _profile(driver_obj)
    paths = [p for p in driver_obj.get("paths", []) if isinstance(p, Mapping)] if isinstance(driver_obj.get("paths", []), list) else []
    own = [p for p in paths if _path_mode(p) == "own"]
    replay = [p for p in paths if _path_mode(p) != "own"]
    fatal: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    # Identity/profile fields are checks only when supplied.  An older driver
    # without a field is incomplete metadata rather than a failed effect.
    identity = {}
    for key in ("schema", "model_digest", "model_sha256", "execution", "model_identity", "profile", "record_bytes", "weight_gpu_bytes", "program_gpu_bytes", "epsilon", "theta"):
        if key in driver_obj:
            identity[key] = _json(driver_obj[key])
    if not driver_obj:
        fatal.append({"kind": "missing_driver", "status": "fatal"})
    if driver_obj.get("schema") and not str(driver_obj["schema"]).startswith("tcm-operand-composition-driver"):
        diagnostics.append({"kind": "unexpected_schema", "value": driver_obj.get("schema")})

    authorities = [token_authority(p) for p in paths]
    for rec in authorities:
        if rec["status"] == "mismatch":
            fatal.append({"kind": "token_authority_mismatch", **rec})

    # Validate every explicitly declared factor/source/array hash without
    # trusting the producer's boolean summary.  Files outside raw are rejected
    # as custody violations rather than silently followed.
    seal_obj: Mapping[str, Any] = {}
    for name in ("own-seal.json", "array-seal.json", "seal.json"):
        candidate = raw_path / name
        if candidate.exists():
            try:
                seal_obj = json.loads(candidate.read_text())
                diagnostics.append({"kind": "seal_loaded", "path": str(candidate), "sha256": sha256_file(candidate)})
            except Exception as exc:
                fatal.append({"kind": "invalid_seal_json", "path": str(candidate), "error": str(exc)})
            break
    expected_files = _expected_file_map(driver_obj, seal_obj)
    checked_refs: set[str] = set()
    for label, record in expected_files:
        p = _resolve_record(raw_path, record)
        expected_sha = _sha_from_record(record)
        if p is None:
            diagnostics.append({"kind": "unresolved_declared_file", "label": label, "record": _json(record)})
            continue
        key = str(p.resolve())
        if key in checked_refs:
            continue
        checked_refs.add(key)
        if not p.exists():
            fatal.append({"kind": "missing_declared_file", "label": label, "path": str(p)})
            continue
        got = sha256_file(p)
        rec = {"kind": "declared_file", "label": label, "path": str(p), "sha256": got, "expected_sha256": expected_sha}
        if expected_sha is not None:
            rec["hash_match"] = got.lower() == expected_sha.lower()
            if not rec["hash_match"]:
                fatal.append({"kind": "declared_hash_mismatch", **rec})
        diagnostics.append(rec)

    # The producer seals an own-only driver before it appends replay paths.
    # Compare that immutable prefix to the final driver by path identity and
    # policy, then leave the replay suffix visible for separate checks.
    own_driver_path = raw_path / "own-driver.json"
    if own_driver_path.exists():
        try:
            own_driver = json.loads(own_driver_path.read_text())
            sealed_paths = [p for p in own_driver.get("paths", []) if isinstance(p, Mapping)]
            final_own = [p for p in paths if _path_mode(p) == "own"]
            def path_identity(p: Mapping[str, Any]) -> tuple[Any, ...]:
                return (
                    str(p.get("name")), str(p.get("task")), str(p.get("policy")), str(p.get("mode", "own")),
                    bool(p.get("writer", False)), bool(p.get("reader", False)), int(p.get("count", 0) or 0),
                    tuple(_json(p.get("consumed_ids", []))), _json(p.get("turns", [])),
                )
            sealed_identity = [path_identity(p) for p in sealed_paths]
            final_identity = [path_identity(p) for p in final_own]
            match = sealed_identity == final_identity
            diagnostics.append({"kind": "own_driver_join", "status": "exact" if match else "mismatch", "sealed_count": len(sealed_paths), "final_own_count": len(final_own), "sealed_sha256": sha256_file(own_driver_path)})
            if not match:
                fatal.append({"kind": "own_driver_join_mismatch", "sealed_count": len(sealed_paths), "final_own_count": len(final_own), "sealed_prefix": sealed_identity[:3], "final_prefix": final_identity[:3]})
        except Exception as exc:
            fatal.append({"kind": "invalid_own_driver", "path": str(own_driver_path), "error": str(exc)})
    elif seal_obj:
        diagnostics.append({"kind": "own_driver_missing", "status": "incomplete"})

    # Factor shape and finite checks are independent from the producer report.
    factor_report = factor_layout(raw_path, driver_obj, profile)
    for name, rec in factor_report.get("factors", {}).items():
        if rec.get("status") in ("invalid", "nonfinite"):
            fatal.append({"kind": "factor_invalid", "factor": name, **rec})
    for p in paths:
        # The frozen producer writes each field as one path-contiguous array;
        # inspect each file once rather than rereading it once per token.
        for field in ARRAY_FIELDS:
            shape = _path_shape_contract(field, profile)
            candidates = _candidate_files(raw_path, p, None, field)
            if not candidates:
                continue
            file_path = candidates[0]
            check = _check_f32_file(file_path, None)
            expected_per = int(np.prod(shape)) if shape is not None else None
            count = int(p.get("count", 0) or 0)
            if check.get("status") == "valid" and expected_per is not None:
                values = check["bytes"] // 4
                expected = {expected_per}
                if count:
                    expected.add(count * expected_per)
                if values not in expected and values % expected_per:
                    check.update({"status": "invalid", "error": f"{values} FP32 values not compatible with row width {expected_per} and count {count}"})
                else:
                    check["row_width"] = expected_per
                    check["row_count"] = values // expected_per
            check.update({"path_name": _path_name(p), "field": field})
            diagnostics.append({"kind": "row_array", **check})
            if check["status"] in ("invalid", "nonfinite"):
                fatal.append({"kind": "row_array_invalid", **check})
        for b in boundary_records(raw_path, p, profile["vocab"]):
            meta = b.get("meta", {})
            if meta.get("status") == "invalid":
                fatal.append({"kind": "boundary_invalid", "path_name": _path_name(p), **meta})
            elif meta.get("status") == "valid":
                diagnostics.append({"kind": "boundary", "path_name": _path_name(p), **meta})

    # Counts are descriptive unless the producer explicitly declares an exact
    # geometry.  Missing optional paths stay visible rather than pruning rows.
    declared_counts = {"all": len(paths), "own": len(own), "replay": len(replay), "expected_own": 24, "expected_replay": 18}
    if len(own) != 24 or len(replay) != 18:
        diagnostics.append({"kind": "geometry_count_observation", **declared_counts})

    mechanics = "valid" if not fatal else "invalid"
    instrument = "available" if any(d.get("kind") in ("row_array", "boundary") for d in diagnostics) else "incomplete"
    result = {
        "schema": "operand-composition-audit-v1",
        "raw": str(raw_path),
        "identity": identity,
        "profile": profile,
        "geometry": declared_counts,
        "token_authority": authorities,
        "factor_layout": factor_report,
        "mechanics_status": mechanics,
        "instrument_status": instrument,
        "terminal_status": "not_assessed",
        "fatal": fatal,
        "diagnostics": diagnostics,
        "input_manifest_keys": sorted(input_obj) if isinstance(input_obj, Mapping) else [],
        "notes": [
            "All numeric effect measures remain observations; this audit gates only custody, typed array integrity, and declared token identity.",
            "Expected 24/18 path counts are reported as geometry observations unless an execution integrity failure is separately present.",
            "Native/candidate MAC counts are copied only as metadata; they are not interpreted as a speedup claim here.",
        ],
    }
    return _json(result)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--driver", type=Path)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--output", type=Path)
    ns = ap.parse_args(argv)
    result = audit(ns.raw, ns.driver, ns.inputs)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if ns.output:
        ns.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
