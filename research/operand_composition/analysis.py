"""Offline analysis for the rank-64 MLP / rank-128 attention composition run.

The producer owns the native execution and the raw arrays.  This module only
reads those arrays and the sealed driver description.  In particular, it does
not infer an effect from a path name: paths are joined by task, token history,
clock, and (for replays) the producer's explicit parent identity.

The implementation intentionally keeps three questions separate:

* mechanics: are the declared arrays, shapes, hashes, and token histories
  usable;
* instrument observations: what do the component and factorial contrasts say;
* terminal claims: none are made here.

The dimensions below are the Qwen2.5-1.5B profile, but all arithmetic helpers
accept the dimensions found in a synthetic test fixture as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROFILE = {
    "hidden": 1536,
    "intermediate": 8960,
    "vocab": 151936,
    "rank": 64,
    "query_rank": 128,
    "kv": 256,
}

POLICIES = {0: "00", 1: "10", 2: "01", 3: "11"}
POLICY_ORDER = ("00", "10", "01", "11")
BOUNDARY_KINDS = ("prefill", "final")
ARRAY_FIELDS = (
    "mlp-norm",
    "mlp-write",
    "mlp-output",
    "receiver",
    "q",
    "k",
    "v",
    "attention",
    "output",
    "mlp-code",
    "query-code",
    "kv16",
    "kv18",
)


def _json(value: Any) -> Any:
    """Convert numpy scalars/arrays to values suitable for JSON reports."""

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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return default if value is None else value


def _finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Signed, symmetric diagnostics for two arrays.

    The byte flag is intentionally exact.  The numeric metrics are diagnostics
    and are not converted into a scientific pass/fail gate.
    """

    aa = np.asarray(a)
    bb = np.asarray(b)
    out: dict[str, Any] = {
        "shape_a": list(aa.shape),
        "shape_b": list(bb.shape),
        "shape_equal": bool(aa.shape == bb.shape),
    }
    if aa.shape != bb.shape:
        out.update({"byte_equal": False, "l2": None, "max_abs": None, "relative_l2": None})
        return out
    # Use the supplied storage dtype for the byte authority.  FP32 raw arrays
    # are normally little-endian; retaining the original bytes exposes signed
    # zero and BLAS-order differences rather than erasing them.
    out["byte_equal"] = bool(np.asarray(aa).tobytes(order="C") == np.asarray(bb).tobytes(order="C"))
    d = np.asarray(aa, dtype=np.float64) - np.asarray(bb, dtype=np.float64)
    out["l2"] = float(np.linalg.norm(d.ravel()))
    out["max_abs"] = float(np.max(np.abs(d))) if d.size else 0.0
    denom = float(np.linalg.norm(np.asarray(bb, dtype=np.float64).ravel()))
    out["relative_l2"] = float(out["l2"] / denom) if denom else (0.0 if out["l2"] == 0 else None)
    out["finite_a"] = _finite(np.asarray(aa))
    out["finite_b"] = _finite(np.asarray(bb))
    return out


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    p = np.exp(x - m)
    return p / np.sum(p)


def symmetric_kl(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    """Symmetric KL after stable softmax, for readout trend diagnostics."""

    pa = np.maximum(stable_softmax(a), eps)
    pb = np.maximum(stable_softmax(b), eps)
    return float(0.5 * (np.sum(pa * np.log(pa / pb)) + np.sum(pb * np.log(pb / pa))))


def _profile(driver: Mapping[str, Any]) -> dict[str, int]:
    p = dict(PROFILE)
    for key in p:
        value = driver.get(key)
        if value is None and isinstance(driver.get("profile"), Mapping):
            value = driver["profile"].get(key)
        if value is not None:
            try:
                p[key] = int(value)
            except (TypeError, ValueError):
                pass
    return p


def _path_policy(path: Mapping[str, Any]) -> str:
    policy = path.get("policy", path.get("arm", path.get("mode_policy")))
    if isinstance(policy, str):
        s = policy
        for candidate in POLICY_ORDER:
            if candidate == s or candidate in s:
                return candidate
    try:
        return POLICIES.get(int(policy), str(policy))
    except (TypeError, ValueError):
        name = str(path.get("name", ""))
        for candidate in POLICY_ORDER:
            if name.endswith(candidate) or f"-{candidate}" in name or f"_{candidate}" in name:
                return candidate
    return "unknown"


def _path_task(path: Mapping[str, Any]) -> str:
    for key in ("task", "case", "case_id", "context"):
        if key in path:
            return str(path[key])
    name = str(path.get("name", ""))
    return name.split("-")[0] if name else "unknown"


def _path_mode(path: Mapping[str, Any]) -> str:
    return str(path.get("mode", "own"))


def _path_name(path: Mapping[str, Any], fallback: str = "path") -> str:
    return str(path.get("name", fallback))


def _rows(path: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = path.get("steps", path.get("tokens", path.get("events", [])))
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    records = [r for r in rows if isinstance(r, Mapping)]
    if records:
        return records
    # The frozen C driver stores one contiguous array per field and exposes
    # only count/consumed_ids in driver.json.  Materialize lightweight row
    # views so all field readers use the same positional contract.
    try:
        count = int(path.get("count", 0))
    except (TypeError, ValueError):
        count = 0
    return [{"index": i, "position": i} for i in range(max(0, count))]


def _row_index(row: Mapping[str, Any], fallback: int) -> int:
    for key in ("index", "step", "token_index", "i", "position"):
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                pass
    return fallback


def _turns(path: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    turns = path.get("turns", path.get("boundaries", []))
    if isinstance(turns, Mapping):
        turns = list(turns.values())
    return [t for t in turns if isinstance(t, Mapping)]


def _ids_from_turn(turn: Mapping[str, Any]) -> list[int]:
    ids: list[int] = []
    for key in ("ingested_ids", "ingested", "input_ids", "prompt_ids"):
        value = turn.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            for x in value:
                try:
                    ids.append(int(x))
                except (TypeError, ValueError):
                    pass
            break
    for key in ("answer_ids", "generated_ids", "output_ids"):
        value = turn.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            for x in value:
                try:
                    ids.append(int(x))
                except (TypeError, ValueError):
                    pass
            break
    return ids


def token_authority(path: Mapping[str, Any]) -> dict[str, Any]:
    """Compare declared consumed IDs with IDs recoverable from turn records."""

    declared = []
    for x in _as_list(path.get("consumed_ids", path.get("token_ids"))):
        try:
            declared.append(int(x))
        except (TypeError, ValueError):
            pass
    from_turns: list[int] = []
    complete_segments = True
    for turn in _turns(path):
        # The composition producer stores ``ingested`` as a count and keeps
        # the authoritative complete sequence in consumed_ids.  Counts are
        # useful metadata, but must not be mistaken for token IDs.
        if any(key in turn and not isinstance(turn.get(key), (list, tuple, np.ndarray)) for key in ("ingested_ids", "ingested", "input_ids", "prompt_ids")):
            complete_segments = False
        from_turns.extend(_ids_from_turn(turn))
    exact = bool(complete_segments and declared and from_turns and declared == from_turns)
    # A turn may record counts instead of token arrays.  That is not a false
    # token mismatch; report it as unavailable authority.
    status = "exact" if exact else ("declared_only" if declared and (not from_turns or not complete_segments) else ("unavailable" if not declared else "mismatch"))
    return {
        "name": _path_name(path),
        "declared_count": len(declared),
        "turn_recoverable_count": len(from_turns),
        "turn_segments_complete": complete_segments,
        "declared_sha256": sha256_bytes(np.asarray(declared, dtype="<i4").tobytes()) if declared else None,
        "turn_sha256": sha256_bytes(np.asarray(from_turns, dtype="<i4").tobytes()) if from_turns else None,
        "status": status,
    }


def _safe_relative(raw: Path, value: Any) -> Path | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    p = Path(value)
    if not p.is_absolute():
        p = raw / p
    try:
        p.resolve().relative_to(raw.resolve())
    except ValueError:
        return None
    return p


def _explicit_file(container: Mapping[str, Any], field: str) -> list[Any]:
    values: list[Any] = []
    for key in (field, f"{field}_file", f"{field}-file"):
        if key in container:
            values.append(container[key])
    for map_key in ("files", "arrays", "artifacts", "raw_files"):
        mapping = container.get(map_key)
        if isinstance(mapping, Mapping) and field in mapping:
            values.append(mapping[field])
    return values


def _candidate_files(raw: Path, path: Mapping[str, Any], row: Mapping[str, Any] | None, field: str) -> list[Path]:
    row = row or {}
    vals: list[Any] = []
    vals.extend(_explicit_file(row, field))
    vals.extend(_explicit_file(path, field))
    out: list[Path] = []
    for value in vals:
        if isinstance(value, Mapping):
            value = value.get("path", value.get("file"))
        p = _safe_relative(raw, value)
        if p is not None and p.exists():
            out.append(p)
    name = _path_name(path)
    indices = []
    if row is not None:
        idx = _row_index(row, 0)
        indices.extend([str(idx), f"{idx:03d}", f"{idx:04d}"])
        if "position" in row:
            try:
                pos = int(row["position"])
                indices.extend([str(pos), f"{pos:03d}", f"{pos:04d}"])
            except (TypeError, ValueError):
                pass
    for idx in dict.fromkeys(indices):
        for stem in (f"{name}-t{idx}", f"{name}-step{idx}", f"{name}_{idx}"):
            for suffix in (f"-{field}.f32", f"_{field}.f32", f".{field}.f32"):
                p = raw / (stem + suffix)
                if p.exists():
                    out.append(p)
    # The composition driver writes one contiguous array per path, with rows
    # in consumed-token order.  Keep this after row-specific candidates so a
    # future producer may still publish individual packets with the same
    # readable naming convention.
    for stem in (f"{name}-{field}.f32", f"{name}_{field}.f32"):
        p = raw / stem
        if p.exists():
            out.append(p)
    # Some producers place all event files in a directory named after path.
    for directory in (raw / name, raw / "arrays" / name, raw / "paths" / name):
        if directory.exists() and row is not None:
            idx = _row_index(row, 0)
            for suffix in (f"{idx}-{field}.f32", f"{idx:03d}-{field}.f32", f"t{idx}-{field}.f32"):
                p = directory / suffix
                if p.exists():
                    out.append(p)
    return list(dict.fromkeys(out))


def _read_f32(path: Path, shape: Sequence[int] | None = None) -> np.ndarray:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path} is not a whole FP32 array")
    a = np.frombuffer(data, dtype="<f4").copy()
    if shape is not None:
        expected = int(np.prod(shape))
        if a.size != expected:
            raise ValueError(f"{path}: expected {expected} values, got {a.size}")
        a = a.reshape(tuple(shape))
    return a


def _read_path_field(raw: Path, path: Mapping[str, Any], field: str, shape: Sequence[int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read a path-contiguous field once and reshape it by token."""

    candidates = _candidate_files(raw, path, None, field)
    if not candidates:
        return None, {"field": field, "status": "missing"}
    p = candidates[0]
    try:
        flat = _read_f32(p)
    except Exception as exc:
        return None, {"field": field, "status": "invalid", "path": str(p), "error": str(exc)}
    per = int(np.prod(shape))
    count = int(path.get("count", 0) or 0)
    if flat.size == per:
        arr = flat.reshape(tuple(shape))
    elif count and flat.size == count * per:
        arr = flat.reshape((count,) + tuple(shape))
    elif flat.size % per == 0:
        arr = flat.reshape((flat.size // per,) + tuple(shape))
    else:
        return None, {"field": field, "status": "invalid", "path": str(p), "error": f"{flat.size} values are not a multiple of {per}"}
    return arr, {"field": field, "status": "valid", "path": str(p), "sha256": sha256_file(p), "shape": list(arr.shape), "finite": _finite(arr)}


def read_field(raw: Path, path: Mapping[str, Any], row: Mapping[str, Any], field: str, shape: Sequence[int] | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates = _candidate_files(raw, path, row, field)
    if not candidates:
        return None, {"field": field, "status": "missing"}
    p = candidates[0]
    try:
        if shape is None:
            arr = _read_f32(p)
        else:
            flat = _read_f32(p)
            per = int(np.prod(shape))
            row_index = _row_index(row, 0)
            if flat.size == per:
                arr = flat.reshape(tuple(shape))
            elif flat.size % per == 0 and row_index < flat.size // per:
                arr = flat.reshape((-1,) + tuple(shape))[row_index]
            else:
                raise ValueError(f"expected one row of {per} values or a contiguous path array, got {flat.size}")
    except Exception as exc:  # keep row-level evidence and continue other fields
        return None, {"field": field, "status": "invalid", "path": str(p), "error": str(exc)}
    return arr, {
        "field": field,
        "status": "valid",
        "path": str(p),
        "sha256": sha256_file(p),
        "shape": list(arr.shape),
        "finite": _finite(arr),
    }


def _boundary_file_candidates(raw: Path, path: Mapping[str, Any], turn: Mapping[str, Any], kind: str) -> list[Path]:
    vals: list[Any] = []
    vals.extend(_explicit_file(turn.get(kind, {}) if isinstance(turn.get(kind), Mapping) else {}, "logits"))
    if isinstance(turn.get(kind), (str, os.PathLike)):
        vals.append(turn[kind])
    vals.extend(_explicit_file(path, f"{kind}_logits"))
    vals.extend(_explicit_file(path, f"{kind}-logits"))
    out: list[Path] = []
    for value in vals:
        if isinstance(value, Mapping):
            value = value.get("path", value.get("file"))
        p = _safe_relative(raw, value)
        if p is not None and p.exists():
            out.append(p)
    name = _path_name(path)
    tval = turn.get("turn", turn.get("index", turn.get("id", 0)))
    for token in (str(tval), f"{int(tval):03d}" if str(tval).lstrip("-").isdigit() else str(tval)):
        for stem in (f"{name}-t{token}-{kind}-logits.f32", f"{name}-{token}-{kind}-logits.f32", f"{name}-{kind}-{token}-logits.f32"):
            p = raw / stem
            if p.exists():
                out.append(p)
    return list(dict.fromkeys(out))


def read_boundary(raw: Path, path: Mapping[str, Any], turn: Mapping[str, Any], kind: str, vocab: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    candidates = _boundary_file_candidates(raw, path, turn, kind)
    if not candidates:
        return None, {"status": "missing", "kind": kind}
    p = candidates[0]
    try:
        a = _read_f32(p, (vocab,))
    except Exception as exc:
        return None, {"status": "invalid", "kind": kind, "path": str(p), "error": str(exc)}
    return a, {"status": "valid", "kind": kind, "path": str(p), "sha256": sha256_file(p), "finite": _finite(a)}


def boundary_records(raw: Path, path: Mapping[str, Any], vocab: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ti, turn in enumerate(_turns(path)):
        turn_id = turn.get("turn", turn.get("index", ti))
        for kind in BOUNDARY_KINDS:
            arr, meta = read_boundary(raw, path, turn, kind, vocab)
            records.append({"turn": turn_id, "kind": kind, "array": arr, "meta": meta})
    # A compact producer may put readouts directly in a map.
    if not records and isinstance(path.get("readouts"), Mapping):
        for key, value in path["readouts"].items():
            if isinstance(value, Mapping):
                kind = str(value.get("kind", key))
                p = _safe_relative(raw, value.get("path", value.get("file")))
            else:
                kind, p = str(key), _safe_relative(raw, value)
            if p is None or not p.exists():
                records.append({"turn": None, "kind": kind, "array": None, "meta": {"status": "missing"}})
            else:
                try:
                    arr = _read_f32(p, (vocab,))
                    records.append({"turn": None, "kind": kind, "array": arr, "meta": {"status": "valid", "path": str(p), "sha256": sha256_file(p)}})
                except Exception as exc:
                    records.append({"turn": None, "kind": kind, "array": None, "meta": {"status": "invalid", "path": str(p), "error": str(exc)}})
    return records


def _boundary_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("turn")), str(record.get("kind"))


def _vector_summary(v: np.ndarray) -> dict[str, Any]:
    a = np.asarray(v)
    return {
        "shape": list(a.shape),
        "finite": _finite(a),
        "l2": l2(a),
        "mean": float(np.mean(a)) if a.size else 0.0,
        "max_abs": float(np.max(np.abs(a))) if a.size else 0.0,
        "argmax_abs": int(np.argmax(np.abs(a))) if a.size else None,
        "sha256_fp32": sha256_bytes(np.asarray(a, dtype="<f4").tobytes(order="C")),
    }


def _compact_vector_summary(v: np.ndarray) -> dict[str, Any]:
    """Per-token signed trend summary.

    The raw path-contiguous FP32 arrays and the audit hashes remain the exact
    vector authority.  Repeating shape/finite/hash metadata for every one of
    thousands of token rows only inflates the analysis report, so dynamics
    retain signed moments and norms here.
    """

    a = np.asarray(v, dtype=np.float64)
    return {
        "l2": float(np.linalg.norm(a.ravel())),
        "mean": float(np.mean(a)) if a.size else 0.0,
        "sum": float(np.sum(a)) if a.size else 0.0,
        "min": float(np.min(a)) if a.size else 0.0,
        "max": float(np.max(a)) if a.size else 0.0,
        "max_abs": float(np.max(np.abs(a))) if a.size else 0.0,
    }


def factorial_contrast(vectors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return exact signed factorial components and a compact Gram summary."""

    missing = [p for p in POLICY_ORDER if p not in vectors]
    if missing:
        return {"status": "incomplete", "missing": missing}
    y00 = np.asarray(vectors["00"], dtype=np.float32)
    y10 = np.asarray(vectors["10"], dtype=np.float32)
    y01 = np.asarray(vectors["01"], dtype=np.float32)
    y11 = np.asarray(vectors["11"], dtype=np.float32)
    interaction = y11.astype(np.float64) - y10.astype(np.float64) - y01.astype(np.float64) + y00.astype(np.float64)
    # Keep the arithmetic identity in FP64 and a replayable FP32 signed
    # vector hash.  The latter is not a tolerance gate.
    # Policy bits are writer/query bits in the C driver:
    # 00 = native MLP + native query, 10 = compiled MLP + native query,
    # 01 = native MLP + compiled query, 11 = compiled MLP + compiled query.
    d_mlp_native_reader = y10.astype(np.float64) - y00.astype(np.float64)
    d_mlp_compiled_reader = y11.astype(np.float64) - y01.astype(np.float64)
    d_reader_native_mlp = y01.astype(np.float64) - y00.astype(np.float64)
    d_reader_compiled_mlp = y11.astype(np.float64) - y10.astype(np.float64)
    components = {
        "Y00": y00,
        "Y10": y10,
        "Y01": y01,
        "Y11": y11,
        "interaction": interaction,
        "mlp_at_reader_native": d_mlp_native_reader,
        "mlp_at_reader_compiled": d_mlp_compiled_reader,
        "reader_at_mlp_native": d_reader_native_mlp,
        "reader_at_mlp_compiled": d_reader_compiled_mlp,
    }
    # Gram matrix of contrasts gives signed cross-term information without
    # putting a giant vocab-sized vector into JSON.
    contrast_names = ["mlp_at_reader_native", "reader_at_mlp_native", "interaction"]
    mat = np.stack([np.asarray(components[n], dtype=np.float64).ravel() for n in contrast_names]).astype(np.float64)
    gram = mat @ mat.T
    out: dict[str, Any] = {"status": "valid", "shape": list(y00.shape), "components": {}, "gram_names": contrast_names, "gram": gram.tolist()}
    for name, value in components.items():
        out["components"][name] = _vector_summary(value)
    out["softmax_symmetric_kl"] = {
        "Y00_Y10": symmetric_kl(y00, y10),
        "Y00_Y01": symmetric_kl(y00, y01),
        "Y00_Y11": symmetric_kl(y00, y11),
        "Y10_Y01": symmetric_kl(y10, y01),
    }
    return out


def _factor_candidates(raw: Path, name: str) -> list[Path]:
    roots = [raw, raw / "factors", raw / "compiled", raw / "bank"]
    out: list[Path] = []
    for root in roots:
        for stem in (name, name.replace("_", "-")):
            for suffix in (".f32", ".bin"):
                p = root / (stem + suffix)
                if p.exists():
                    out.append(p)
    return list(dict.fromkeys(out))


def _factor_path(raw: Path, names: Sequence[str], driver: Mapping[str, Any] | None = None) -> Path | None:
    vals: list[Any] = []
    if isinstance(driver, Mapping):
        for section in ("factor_files", "factors", "basis_files", "compiled_factors"):
            m = driver.get(section)
            if isinstance(m, Mapping):
                for n in names:
                    if n in m:
                        vals.append(m[n])
    for value in vals:
        if isinstance(value, Mapping):
            value = value.get("path", value.get("file"))
        p = _safe_relative(raw, value)
        if p is not None and p.exists():
            return p
    for n in names:
        c = _factor_candidates(raw, n)
        if c:
            return c[0]
    return None


def factor_layout(raw: Path, driver: Mapping[str, Any], profile: Mapping[str, int]) -> dict[str, Any]:
    """Inspect the rank factors and run a CPU check when producer arrays exist."""

    d, h, r = profile["hidden"], profile["intermediate"], profile["rank"]
    specs = {
        "mlp-E": ((r, d), ("mlp-E", "E", "union64-E")),
        "mlp-Lg": ((h, r), ("mlp-Lg", "Lg", "coupled64-Lg")),
        "mlp-Lu": ((h, r), ("mlp-Lu", "Lu", "coupled64-Lu")),
        "query-B": ((profile["query_rank"], d), ("query-B", "B")),
        "query-F": ((d, profile["query_rank"]), ("query-F", "F")),
        "query-K": ((profile["kv"], profile["query_rank"]), ("query-K", "K")),
    }
    out: dict[str, Any] = {"status": "valid", "factors": {}, "cpu_replays": []}
    loaded: dict[str, np.ndarray] = {}
    for label, (shape, names) in specs.items():
        p = _factor_path(raw, names, driver)
        rec: dict[str, Any] = {"expected_shape": list(shape), "status": "missing"}
        if p is not None:
            try:
                a = _read_f32(p, shape)
                loaded[label] = a
                rec.update({"status": "valid", "path": str(p), "sha256": sha256_file(p), "shape": list(a.shape), "dtype": "<f4", "c_contiguous": bool(a.flags.c_contiguous), "finite": _finite(a)})
            except Exception as exc:
                rec.update({"status": "invalid", "path": str(p), "error": str(exc)})
                out["status"] = "incomplete"
        else:
            out["status"] = "incomplete"
        out["factors"][label] = rec
    # CPU map check for the first recorded full gate/up pair in each path.  It
    # explicitly verifies row-major E=Q^T, Lg=GQ, Lu=UQ rather than merely
    # checking matrix shapes.
    if all(k in loaded for k in ("mlp-E", "mlp-Lg", "mlp-Lu")):
        for path in driver.get("paths", []) if isinstance(driver.get("paths"), list) else []:
            if not isinstance(path, Mapping) or _path_mode(path) != "own":
                continue
            # The producer's first gate/up observer is physical native data on
            # native-writer arms and the compiled factor output on writer arms.
            # Only the latter is a meaningful CPU reconstruction of E/Lg/Lu;
            # retain native arms for the separate native baseline diagnostics.
            writer = path.get("writer")
            if writer is None:
                writer = _path_policy(path) in ("10", "11")
            if not bool(writer):
                continue
            gate_names = [f"{_path_name(path)}-first-gate.f32", f"{_path_name(path)}-gate-first.f32"]
            up_names = [f"{_path_name(path)}-first-up.f32", f"{_path_name(path)}-up-first.f32"]
            gp = next((raw / n for n in gate_names if (raw / n).exists()), None)
            up = next((raw / n for n in up_names if (raw / n).exists()), None)
            rows = _rows(path)
            if gp is None or up is None or not rows:
                continue
            x, xm = read_field(raw, path, rows[0], "mlp-norm", (d,))
            if x is None:
                continue
            try:
                gcap = _read_f32(gp, (h,))
                ucap = _read_f32(up, (h,))
            except Exception as exc:
                out["cpu_replays"].append({"path": _path_name(path), "status": "invalid", "error": str(exc)})
                continue
            z = loaded["mlp-E"].astype(np.float64) @ x.astype(np.float64)
            g = loaded["mlp-Lg"].astype(np.float64) @ z
            u = loaded["mlp-Lu"].astype(np.float64) @ z
            out["cpu_replays"].append({"path": _path_name(path), "status": "valid", "input_meta": xm, "gate": compare_arrays(g.astype(np.float32), gcap), "up": compare_arrays(u.astype(np.float32), ucap)})
    return out


def _path_boundary_map(raw: Path, path: Mapping[str, Any], vocab: int) -> dict[tuple[str, str], tuple[np.ndarray, dict[str, Any]]]:
    return {
        _boundary_key(rec): (rec["array"], rec["meta"])
        for rec in boundary_records(raw, path, vocab)
        if rec.get("array") is not None
    }


def _own_groups(paths: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for path in paths:
        if _path_mode(path) == "own":
            groups.setdefault(_path_task(path), []).append(path)
    return groups


def _factorial_boundaries(raw: Path, paths: Sequence[Mapping[str, Any]], vocab: int) -> list[dict[str, Any]]:
    # The four own trajectories are deliberately allowed to generate different
    # answer histories.  The factorial cell is the *same-parent* panel: own
    # 11 supplies the parent history and the three replay paths supply 00/10/01
    # on exactly those consumed IDs.  Own-arm contrasts remain separately
    # observable in ``own_history_factorial`` below.
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for p in paths:
        grouped.setdefault(_path_task(p), []).append(p)
    out: list[dict[str, Any]] = []
    for task, members in grouped.items():
        by_policy: dict[str, Mapping[str, Any]] = {}
        for p in members:
            pol = _path_policy(p)
            if _path_mode(p) == "own" and pol == "11":
                by_policy[pol] = p
            elif _path_mode(p) != "own" and pol in ("00", "10", "01"):
                by_policy[pol] = p
        if any(k not in by_policy for k in POLICY_ORDER):
            out.append({"task": task, "panel": "same_parent_replay", "status": "incomplete", "available_policies": sorted(by_policy)})
            continue
        ids = {pol: tuple(int(x) for x in _as_list(by_policy[pol].get("consumed_ids"))) for pol in POLICY_ORDER}
        same_history = len(set(ids.values())) == 1
        maps = {k: _path_boundary_map(raw, by_policy[k], vocab) for k in POLICY_ORDER}
        keys = set.intersection(*(set(m) for m in maps.values())) if maps else set()
        if not keys:
            out.append({"task": task, "panel": "same_parent_replay", "status": "no_common_boundaries", "same_history": same_history, "available_policies": sorted(by_policy)})
            continue
        for clock in sorted(keys):
            vectors = {pol: maps[pol][clock][0] for pol in POLICY_ORDER}
            rec = factorial_contrast(vectors)
            rec.update({"task": task, "panel": "same_parent_replay", "turn": clock[0], "clock": clock[1], "same_history": same_history, "paths": {pol: _path_name(by_policy[pol]) for pol in POLICY_ORDER}})
            out.append(rec)
    return out


def _own_history_factorial(raw: Path, paths: Sequence[Mapping[str, Any]], vocab: int) -> list[dict[str, Any]]:
    """Descriptive own-trajectory contrasts, never used as same-parent cells."""

    out: list[dict[str, Any]] = []
    for task, members in _own_groups(paths).items():
        by_policy = {_path_policy(p): p for p in members}
        if len(by_policy) != 4:
            continue
        maps = {k: _path_boundary_map(raw, by_policy[k], vocab) for k in POLICY_ORDER}
        keys = set.intersection(*(set(m) for m in maps.values())) if maps else set()
        for clock in sorted(keys):
            rec = factorial_contrast({pol: maps[pol][clock][0] for pol in POLICY_ORDER})
            rec.update({"task": task, "panel": "own_history", "turn": clock[0], "clock": clock[1], "history_ids_sha256": {pol: sha256_bytes(np.asarray(_as_list(by_policy[pol].get("consumed_ids")), dtype="<i4").tobytes()) for pol in POLICY_ORDER}, "paths": {pol: _path_name(by_policy[pol]) for pol in POLICY_ORDER}})
            out.append(rec)
    return out


def _replay_matches(paths: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    own = {str(p.get("name")): p for p in paths if _path_mode(p) == "own"}
    result: list[dict[str, Any]] = []
    for replay in paths:
        if _path_mode(replay) == "own":
            continue
        target = replay.get("replay_of", replay.get("parent_name", replay.get("source_path")))
        if isinstance(target, Mapping):
            target = target.get("name")
        if target is None:
            # The producer commonly names these as replay-<own-name>-<policy>.
            candidates = [name for name in own if name in _path_name(replay) or _path_name(replay) in name]
            target = candidates[0] if len(candidates) == 1 else None
        result.append({"replay": _path_name(replay), "target": str(target) if target is not None else None, "match_status": "matched" if target in own else "unresolved"})
    return result


def _same_id_readout(raw: Path, paths: Sequence[Mapping[str, Any]], vocab: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    own = {str(p.get("name")): p for p in paths if _path_mode(p) == "own"}
    for replay in paths:
        if _path_mode(replay) == "own":
            continue
        target_name = replay.get("replay_of", replay.get("parent_name"))
        if isinstance(target_name, Mapping):
            target_name = target_name.get("name")
        target = own.get(str(target_name)) if target_name is not None else None
        if target is None:
            continue
        ra = _path_boundary_map(raw, replay, vocab)
        oa = _path_boundary_map(raw, target, vocab)
        for key in sorted(set(ra) & set(oa)):
            out.append({"replay": _path_name(replay), "own": _path_name(target), "turn": key[0], "clock": key[1], "comparison": compare_arrays(ra[key][0], oa[key][0]), "kl": symmetric_kl(ra[key][0], oa[key][0])})
    return out


def _producer_rows(raw: Path, paths: Sequence[Mapping[str, Any]], profile: Mapping[str, int]) -> list[dict[str, Any]]:
    shapes = {
        "mlp-norm": (profile["hidden"],),
        "mlp-write": (profile["hidden"],),
        "mlp-output": (profile["hidden"],),
        "receiver": (profile["hidden"],),
        "q": (profile["hidden"],),
        "k": (profile["kv"],),
        "v": (profile["kv"],),
        "attention": (profile["hidden"],),
        "output": (profile["hidden"],),
        "mlp-code": (profile["rank"],),
        "query-code": (profile["query_rank"],),
        "kv16": (profile["kv"] * 2,),
        "kv18": (profile["kv"] * 2,),
    }
    records: list[dict[str, Any]] = []
    for path in paths:
        rows = _rows(path)
        loaded = {field: _read_path_field(raw, path, field, shape) for field, shape in shapes.items()}
        path_rows: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            fields: dict[str, Any] = {}
            metas: dict[str, Any] = {}
            for field, shape in shapes.items():
                whole, meta = loaded[field]
                arr = None
                if whole is not None:
                    if whole.ndim == len(shape):
                        arr = whole if i == 0 else None
                    elif i < whole.shape[0]:
                        arr = whole[i]
                metas[field] = meta
                if arr is not None:
                    fields[field] = _compact_vector_summary(arr)
            path_rows.append({"row": _row_index(row, i), "fields": fields})
        # Keep path identity once.  Field availability/shape belongs in a
        # compact path-level map; exact bytes and hashes remain in audit.py.
        field_status = {
            field: {k: meta.get(k) for k in ("status", "shape", "finite")}
            for field, (_, meta) in loaded.items()
        }
        records.append({
            "path": _path_name(path),
            "task": _path_task(path),
            "policy": _path_policy(path),
            "mode": _path_mode(path),
            "count": len(path_rows),
            "field_status": field_status,
            "rows": path_rows,
        })
    return records


def _compact_compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Per-row comparison without repeating vector shape/hash metadata."""

    metric = compare_arrays(a, b)
    return {key: metric.get(key) for key in ("byte_equal", "l2", "relative_l2", "max_abs", "finite_a", "finite_b")}


def _matched_component_observations(raw: Path, paths: Sequence[Mapping[str, Any]], profile: Mapping[str, int]) -> list[dict[str, Any]]:
    """Compare receiver/attention/KV fields on the same-parent replay panel.

    Own arms can have different generated histories.  Comparing their rows as
    if they were paired would conflate policy effects with token-history
    effects, so this routine selects own ``11`` and replay ``00/10/01`` only.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for p in paths:
        grouped.setdefault(_path_task(p), []).append(p)
    shape_map = {
        "mlp-write": (profile["hidden"],),
        "receiver": (profile["hidden"],),
        "attention": (profile["hidden"],),
        "output": (profile["hidden"],),
        "kv16": (profile["kv"] * 2,),
        "kv18": (profile["kv"] * 2,),
    }
    out: list[dict[str, Any]] = []
    pair_names = (("00", "01"), ("10", "11"), ("00", "10"), ("01", "11"))
    for task, members in grouped.items():
        rows_by_policy: dict[str, dict[int, Mapping[str, Any]]] = {}
        path_by_policy: dict[str, Mapping[str, Any]] = {}
        for p in members:
            policy = _path_policy(p)
            if _path_mode(p) == "own" and policy == "11":
                path_by_policy[policy] = p
            elif _path_mode(p) != "own" and policy in ("00", "10", "01"):
                path_by_policy[policy] = p
        if any(policy not in path_by_policy for policy in POLICY_ORDER):
            out.append({"task": task, "panel": "same_parent_replay", "status": "incomplete", "paths": {policy: _path_name(path_by_policy[policy]) for policy in sorted(path_by_policy)}})
            continue
        loaded_by_policy = {
            policy: {field: _read_path_field(raw, path_by_policy[policy], field, shape)[0] for field, shape in shape_map.items()}
            for policy, p in path_by_policy.items()
        }
        for policy, p in path_by_policy.items():
            rows_by_policy[_path_policy(p)] = {_row_index(r, i): r for i, r in enumerate(_rows(p))}
        common_indices = set.intersection(*(set(v) for v in rows_by_policy.values())) if rows_by_policy else set()
        ids = {policy: tuple(int(x) for x in _as_list(path_by_policy[policy].get("consumed_ids"))) for policy in POLICY_ORDER}
        group_rows: list[dict[str, Any]] = []
        for idx in sorted(common_indices):
            row_fields: dict[str, Any] = {}
            for field, shape in shape_map.items():
                arrays: dict[str, np.ndarray] = {}
                for policy, rowmap in rows_by_policy.items():
                    whole = loaded_by_policy.get(policy, {}).get(field)
                    arr = None
                    if whole is not None:
                        arr = whole if whole.ndim == len(shape) and idx == 0 else (whole[idx] if whole.ndim > len(shape) and idx < whole.shape[0] else None)
                    if arr is not None:
                        arrays[policy] = arr
                if all(a in arrays and b in arrays for a, b in pair_names):
                    row_fields[field] = {f"{a}:{b}": _compact_compare(arrays[a], arrays[b]) for a, b in pair_names}
            group_rows.append({"row": idx, "fields": row_fields})
        out.append({
            "task": task,
            "panel": "same_parent_replay",
            "status": "valid",
            "same_history": len(set(ids.values())) == 1,
            "history_ids_sha256": {policy: sha256_bytes(np.asarray(ids[policy], dtype="<i4").tobytes()) for policy in POLICY_ORDER},
            "paths": {policy: _path_name(path_by_policy[policy]) for policy in POLICY_ORDER},
            "rows": group_rows,
        })
    return out


def _counter_checks(paths: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Check typed per-path work counters against the declared arm policy.

    These are execution diagnostics, not effect gates.  A producer may expose
    extra bookkeeping calls, so the report retains both the observed value and
    the narrow component expectation.
    """

    keys = (
        "linear_calls", "linear_macs", "native_gate_up_calls", "compiled_gate_up_calls",
        "mlp_coordinate_calls", "native_qk_calls", "compiled_qk_calls",
        "query_coordinate_calls", "native_attention_calls", "reader_products",
    )
    out: list[dict[str, Any]] = []
    for path in paths:
        try:
            count = int(path.get("count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        policy = _path_policy(path)
        writer = bool(path.get("writer", policy in ("10", "11")))
        reader = bool(path.get("reader", policy in ("01", "11")))
        expected = {
            "native_gate_up_calls": 0 if writer else 2 * count,
            "compiled_gate_up_calls": 2 * count if writer else 0,
            "mlp_coordinate_calls": count if writer else 0,
            "native_qk_calls": 0 if reader else 2 * count,
            "compiled_qk_calls": 2 * count if reader else 0,
            "query_coordinate_calls": count if reader else 0,
            "native_attention_calls": count,
        }
        observed = {key: path.get(key) for key in keys if key in path}
        mismatches = {
            key: {"observed": observed.get(key), "expected": value}
            for key, value in expected.items()
            if key in observed and observed[key] != value
        }
        out.append({"name": _path_name(path), "policy": policy, "count": count, "writer": writer, "reader": reader, "observed": observed, "component_expected": expected, "component_mismatches": mismatches, "status": "exact" if not mismatches else "diagnostic_mismatch"})
    return out


def analyze(raw: os.PathLike[str] | str, driver: Mapping[str, Any] | os.PathLike[str] | str | None = None, inputs: Mapping[str, Any] | os.PathLike[str] | str | None = None, texts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze a collected composition run without executing a model."""

    raw_path = Path(raw)
    if raw_path.is_file():
        raw_path = raw_path.parent
    if driver is None:
        candidate = raw_path / "driver.json"
        driver_obj: Mapping[str, Any] = json.loads(candidate.read_text()) if candidate.exists() else {}
    elif isinstance(driver, (str, os.PathLike)):
        driver_obj = json.loads(Path(driver).read_text())
    else:
        driver_obj = driver
    profile = _profile(driver_obj)
    paths = [p for p in driver_obj.get("paths", []) if isinstance(p, Mapping)] if isinstance(driver_obj.get("paths", []), list) else []
    authorities = [token_authority(p) for p in paths]
    boundary_count = sum(len(boundary_records(raw_path, p, profile["vocab"])) for p in paths)
    factor = factor_layout(raw_path, driver_obj, profile)
    report = {
        "schema": "operand-composition-analysis-v1",
        "raw": str(raw_path),
        "profile": profile,
        "path_counts": {"all": len(paths), "own": sum(_path_mode(p) == "own" for p in paths), "replay": sum(_path_mode(p) != "own" for p in paths)},
        "token_authority": authorities,
        "token_authority_summary": {s: sum(a["status"] == s for a in authorities) for s in ("exact", "declared_only", "unavailable", "mismatch")},
        "boundary_file_count": boundary_count,
        "factor_layout": factor,
        "factorial": _factorial_boundaries(raw_path, paths, profile["vocab"]),
        "own_history_factorial": _own_history_factorial(raw_path, paths, profile["vocab"]),
        "replay_matches": _replay_matches(paths),
        "same_id_readouts": _same_id_readout(raw_path, paths, profile["vocab"]),
        "producer_rows": _producer_rows(raw_path, paths, profile),
        "component_observations": _matched_component_observations(raw_path, paths, profile),
        "counter_checks": _counter_checks(paths),
        "cost_ledger": [dict(p) for p in paths if any(k in p for k in ("linear_macs", "linear_calls", "native_attention_calls"))],
        "notes": [
            "Factorial vectors are signed observations; their hashes preserve the exact FP32 reconstruction without embedding full-vocabulary vectors in JSON.",
            "Readout comparisons are only paired when the producer declares a common task/history or an explicit replay parent.",
            "Numeric closeness, cosine, KL, activation rank, and token equality are not scientific hard gates in this exploratory report.",
        ],
    }
    return _json(report)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--driver", type=Path)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--output", type=Path)
    ns = ap.parse_args(argv)
    result = analyze(ns.raw, ns.driver, ns.inputs)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if ns.output:
        ns.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
