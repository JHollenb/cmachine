"""Exact-code state-closure helpers for the C Qwen subspace assay.

This module deliberately does no model execution.  It constructs geometric
candidate receivers and filters them by the raw 16-byte local code before any
consumer arrays are loaded.  The code is the little-endian FP32 bytes of
``executed_coefficients[3]`` followed by native FP32 ``rho``.  JSON decimal
fields, tolerances and rounded norms never establish code equality.

The eventual worker may add a report adapter around these helpers.  Keeping
the geometry and selection logic independent of that report schema makes the
ordering auditable: source-only candidate construction -> exact code filter
-> MLP/write/consumer outcomes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


HIDDEN = 1536
RANK = 3
CODE_BYTES = (RANK + 1) * 4
F32_LE = np.dtype("<f4")
F64_LE = np.dtype("<f8")

# These names are rejected only when they occur in candidate/source metadata.
# Consumer outcomes are loaded after the exact-code filter and may of course
# contain measured logits and writes.
TARGET_METADATA_KEYS = frozenset(
    {
        "target",
        "answer",
        "answer_ids",
        "target_ids",
        "target_logits",
        "target_hidden",
        "target_kv",
        "teacher",
        "teacher_forced",
        "oracle",
    }
)


class ClosureInputError(ValueError):
    """Raised when a source candidate violates the declared geometry contract."""


@dataclass(frozen=True)
class Projection:
    """A source receiver split into its basis projection and omitted part."""

    projection: np.ndarray
    omitted: np.ndarray
    coefficients: np.ndarray
    gram: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One source-only spherical receiver candidate.

    ``receiver`` is the exact FP32 array that a later native worker would
    consume.  ``projection`` and ``omitted`` are FP64 construction diagnostics;
    they are not used to claim exact code equality.
    """

    name: str
    receiver: np.ndarray
    projection: np.ndarray
    omitted: np.ndarray
    source_omitted_norm: float
    candidate_omitted_norm: float
    construction_seed: int


def f32_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return a finite contiguous little-endian FP32 array without rounding comparisons."""

    array = np.asarray(value, dtype=F32_LE)
    if shape is not None and array.shape != shape:
        raise ClosureInputError(f"expected FP32 shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ClosureInputError("non-finite FP32 input")
    return np.ascontiguousarray(array, dtype=F32_LE)


def f64_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return a finite contiguous little-endian FP64 array for construction math."""

    array = np.asarray(value, dtype=F64_LE)
    if shape is not None and array.shape != shape:
        raise ClosureInputError(f"expected FP64 shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ClosureInputError("non-finite FP64 input")
    return np.ascontiguousarray(array, dtype=F64_LE)


def exact_code(coefficients: Any, rho: Any) -> bytes:
    """Encode the declared local code as exactly 16 raw FP32 bytes.

    ``coefficients`` must be the three *executed* FP32 values produced after
    the QR solve.  ``rho`` must be the native FP32 RMS scale returned by the
    runtime.  Do not pass printed JSON doubles here: the worker should retain
    the original FP32 values or their raw bytes.
    """

    c = f32_array(coefficients, shape=(RANK,))
    r = f32_array(np.asarray(rho, dtype=F32_LE).reshape(1), shape=(1,))
    return c.tobytes(order="C") + r.tobytes(order="C")


def exact_code_from_bytes(coefficient_bytes: bytes, rho_bytes: bytes) -> bytes:
    """Validate and concatenate retained FP32 code fields without decoding."""

    if len(coefficient_bytes) != RANK * 4 or len(rho_bytes) != 4:
        raise ClosureInputError("code fields must be 12 and 4 bytes")
    # Validate finiteness while preserving every original bit, including signed
    # zero.  ``frombuffer`` is read-only and does not normalize NaNs or zeros.
    c = np.frombuffer(coefficient_bytes, dtype=F32_LE)
    r = np.frombuffer(rho_bytes, dtype=F32_LE)
    if not np.isfinite(c).all() or not np.isfinite(r).all() or not (r[0] > 0):
        raise ClosureInputError("invalid retained code bytes")
    return bytes(coefficient_bytes) + bytes(rho_bytes)


def code_digest(code: bytes) -> str:
    """Digest a validated raw code for report indexing."""

    if len(code) != CODE_BYTES:
        raise ClosureInputError(f"code must contain {CODE_BYTES} bytes")
    return hashlib.sha256(code).hexdigest()


def exact_code_equal(left: bytes, right: bytes) -> bool:
    """Compare code bits exactly, never with numeric tolerances."""

    return len(left) == CODE_BYTES and len(right) == CODE_BYTES and left == right


def sequential_dot_f64(left: np.ndarray, right: np.ndarray) -> float:
    """Mirror the C sequential FP64 dot used for QR coordinates."""

    if left.shape != right.shape:
        raise ClosureInputError("dot operands have different shapes")
    result = 0.0
    for a, b in zip(left, right):
        result += float(a) * float(b)
    return result


def gram_projection(basis: Any, vector: Any) -> Projection:
    """Project ``vector`` onto arbitrary full-rank basis columns using its Gram matrix.

    Inputs have shapes ``(D, 3)`` and ``(D,)``.  This is construction geometry
    only.  The runtime's exact local code is still the retained FP32 QR code,
    which is checked separately after the candidate is materialized.
    """

    b = f64_array(basis)
    x = f64_array(vector)
    if b.ndim != 2 or b.shape[1] != RANK or x.ndim != 1 or b.shape[0] != x.shape[0]:
        raise ClosureInputError("basis/vector geometry must be (D,3)/(D,)")
    gram = b.T @ b
    if np.linalg.matrix_rank(gram) != RANK:
        raise ClosureInputError("basis Gram matrix is rank deficient")
    coefficients = np.linalg.solve(gram, b.T @ x)
    projection = b @ coefficients
    omitted = x - projection
    return Projection(
        projection=np.ascontiguousarray(projection, dtype=F64_LE),
        omitted=np.ascontiguousarray(omitted, dtype=F64_LE),
        coefficients=np.ascontiguousarray(coefficients, dtype=F64_LE),
        gram=np.ascontiguousarray(gram, dtype=F64_LE),
    )


def materialized_geometry(source: Any, candidate: Any, basis: Any) -> dict[str, float | bool]:
    """Report post-materialization projection/norm differences as diagnostics."""

    source_f = f32_array(source)
    candidate_f = f32_array(candidate, shape=source_f.shape)
    src = gram_projection(basis, source_f.astype(F64_LE))
    cand = gram_projection(basis, candidate_f.astype(F64_LE))
    return {
        "source_norm": float(np.linalg.norm(source_f.astype(F64_LE))),
        "candidate_norm": float(np.linalg.norm(candidate_f.astype(F64_LE))),
        "norm_delta": float(np.linalg.norm(candidate_f.astype(F64_LE)) - np.linalg.norm(source_f.astype(F64_LE))),
        "projection_delta_l2": float(np.linalg.norm(cand.projection - src.projection)),
        "omitted_norm_source": float(np.linalg.norm(src.omitted)),
        "omitted_norm_candidate": float(np.linalg.norm(cand.omitted)),
        "receiver_bytes_equal": source_f.tobytes() == candidate_f.tobytes(),
    }


def spherical_candidate(
    source: Any,
    basis: Any,
    direction: Any,
    *,
    name: str,
    construction_seed: int,
) -> Candidate:
    """Create a FP32 candidate with the source projection and omitted norm.

    ``direction`` is projected with the general Gram projector, then normalized
    to the source omitted norm.  The returned receiver is materialized in FP32;
    exact code equality is intentionally *not* inferred from this construction.
    """

    source_f = f32_array(source)
    direction_f = f64_array(direction, shape=source_f.shape)
    source_split = gram_projection(basis, source_f.astype(F64_LE))
    direction_split = gram_projection(basis, direction_f)
    d = direction_split.omitted
    d_norm = float(np.linalg.norm(d))
    source_e_norm = float(np.linalg.norm(source_split.omitted))
    if not np.isfinite(d_norm) or d_norm <= 0 or not np.isfinite(source_e_norm):
        raise ClosureInputError("spherical direction has no nonzero complement")
    candidate64 = source_split.projection + d * (source_e_norm / d_norm)
    candidate_f = np.ascontiguousarray(candidate64, dtype=F32_LE)
    candidate_split = gram_projection(basis, candidate_f.astype(F64_LE))
    return Candidate(
        name=name,
        receiver=candidate_f,
        projection=np.ascontiguousarray(candidate_split.projection, dtype=F64_LE),
        omitted=np.ascontiguousarray(candidate_split.omitted, dtype=F64_LE),
        source_omitted_norm=source_e_norm,
        candidate_omitted_norm=float(np.linalg.norm(candidate_split.omitted)),
        construction_seed=int(construction_seed),
    )


def random_spherical_candidates(
    source: Any,
    basis: Any,
    *,
    count: int,
    seed: int,
    prefix: str = "candidate",
) -> list[Candidate]:
    """Generate deterministic source-only directions for a finite geometry panel."""

    if count < 1:
        raise ClosureInputError("candidate count must be positive")
    source_f = f32_array(source)
    rng = np.random.default_rng(seed)
    result: list[Candidate] = []
    for index in range(count):
        direction = rng.standard_normal(source_f.shape[0]).astype(F64_LE)
        result.append(
            spherical_candidate(
                source_f,
                basis,
                direction,
                name=f"{prefix}-{index:03d}",
                construction_seed=seed + index,
            )
        )
    return result


def reject_target_metadata(metadata: Mapping[str, Any] | None) -> None:
    """Reject target/teacher fields from source candidate metadata."""

    if not metadata:
        return
    bad = sorted(TARGET_METADATA_KEYS.intersection(metadata))
    if bad:
        raise ClosureInputError("source candidate metadata contains target fields: " + ", ".join(bad))


def code_key(entry: Mapping[str, Any]) -> bytes:
    """Extract raw code bytes from a schema-neutral candidate entry."""

    # The closure driver retains the ABI-level code as a hex string.  Accept a
    # second spelling used by small in-memory tests, but never reconstruct code
    # equality from printed decimal fields when the raw field is available.
    for field in ("code_hex", "code"):
        if field in entry:
            value = entry[field]
            if isinstance(value, str):
                try:
                    value = bytes.fromhex(value)
                except ValueError as exc:
                    raise ClosureInputError(f"{field} is not hexadecimal") from exc
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise ClosureInputError(f"{field} must be raw bytes or hex")
            value = bytes(value)
            if len(value) != CODE_BYTES:
                raise ClosureInputError("invalid raw code extent")
            return exact_code_from_bytes(value[:12], value[12:])
    if "code_bytes" in entry:
        code = entry["code_bytes"]
        if isinstance(code, str):
            code = bytes.fromhex(code)
        if not isinstance(code, (bytes, bytearray, memoryview)):
            raise ClosureInputError("code_bytes must be raw bytes or hex")
        code = bytes(code)
        if len(code) != CODE_BYTES:
            raise ClosureInputError("invalid raw code extent")
        # Validate without normalizing the retained representation.
        return exact_code_from_bytes(code[:12], code[12:])
    if "coefficients_f32" not in entry or "rho_f32" not in entry:
        raise ClosureInputError("candidate entry has no retained exact code")
    return exact_code(entry["coefficients_f32"], entry["rho_f32"])


def group_exact_codes(entries: Iterable[Mapping[str, Any]]) -> dict[bytes, list[Mapping[str, Any]]]:
    """Group source candidates by raw code, before reading any outcomes."""

    groups: dict[bytes, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        reject_target_metadata(entry.get("metadata"))
        groups[code_key(entry)].append(entry)
    return {key: value for key, value in groups.items() if len(value) > 1}


def evaluate_exact_code_pairs(
    entries: Iterable[Mapping[str, Any]],
    outcome_loader: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Load outcomes only for exact-code groups.

    The callback is intentionally the only place that can read MLP/write or
    consumer outcomes.  Entries with unique codes never reach it, making the
    source-only filtering order testable and explicit.
    """

    groups = group_exact_codes(entries)
    results: list[dict[str, Any]] = []
    for code, group in groups.items():
        loaded = [outcome_loader(entry) for entry in group]
        results.append({
            "code_bytes": code,
            "code_sha256": code_digest(code),
            "entries": group,
            "outcomes": loaded,
        })
    return results


def array_metrics(left: Any, right: Any) -> dict[str, float | bool]:
    """Compare retained FP32 outcome arrays after code selection."""

    a = f32_array(left)
    b = f32_array(right, shape=a.shape)
    delta = a.astype(np.float64) - b.astype(np.float64)
    denominator = float(np.linalg.norm(b.astype(np.float64)))
    return {
        "bytes_equal": a.tobytes() == b.tobytes(),
        "l2": float(np.linalg.norm(delta)),
        "relative_l2": float(np.linalg.norm(delta) / denominator) if denominator else None,
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Saved-report adapter
# ---------------------------------------------------------------------------

def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Load a mapping from an already-loaded object or a JSON path."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        try:
            loaded = json.loads(Path(value).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ClosureInputError(f"cannot load {label}: {value}") from exc
        if not isinstance(loaded, Mapping):
            raise ClosureInputError(f"{label} must contain a JSON object")
        return loaded
    raise ClosureInputError(f"{label} must be a mapping or JSON path")


def _raw_directory(raw: Any) -> Path:
    if isinstance(raw, (str, Path)):
        return Path(raw)
    if isinstance(raw, Mapping):
        directory = raw.get("directory", raw.get("path"))
        if directory is not None:
            return Path(directory)
    raise ClosureInputError("raw must be a directory or a mapping with directory/path")


def _read_f32(path: Path, *, extent: int | None = None, required: bool = True) -> np.ndarray | None:
    """Read an immutable saved FP32 port with strict shape/finiteness checks."""

    if not path.exists():
        if required:
            raise ClosureInputError(f"missing retained array: {path}")
        return None
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ClosureInputError(f"cannot read retained array: {path}") from exc
    if len(payload) % 4:
        raise ClosureInputError(f"unaligned FP32 array: {path}")
    array = np.frombuffer(payload, dtype=F32_LE)
    if extent is not None and array.shape != (extent,):
        raise ClosureInputError(f"invalid extent for {path}: expected {extent}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ClosureInputError(f"non-finite retained array: {path}")
    # frombuffer aliases the bytes read above.  A copy prevents an accidental
    # caller mutation from changing the object used by a later comparison.
    return np.ascontiguousarray(array, dtype=F32_LE)


def _port_extent(port: str, hidden: int, intermediate: int) -> int | None:
    if port in {"receiver", "normalized", "write", "input", "raw_receiver", "mlp_input", "mlp_return", "output"}:
        return hidden
    if port in {"gate", "up", "mlp_gate", "mlp_up", "correction-gate", "correction-up"}:
        return intermediate
    if port == "kv16-17":
        return None  # the runtime's selected K/V extent is ABI metadata
    return None


def _port_path(raw_dir: Path, prefix: str, port: str) -> Path:
    return raw_dir / f"{prefix}-{port}.f32"


def _load_port(
    raw_dir: Path,
    prefix: str,
    port: str,
    *,
    hidden: int,
    intermediate: int,
    required: bool = True,
) -> np.ndarray | None:
    return _read_f32(
        _port_path(raw_dir, prefix, port),
        extent=_port_extent(port, hidden, intermediate),
        required=required,
    )


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ClosureInputError(f"cannot hash retained file: {path}") from exc


def _source_reject(value: Any, path: str = "source") -> None:
    """Reject target/teacher metadata in source candidate/geometry objects.

    This is intentionally applied before loading any outcome arrays.  It does
    not scan the post-selection arm records, whose logits and token outcomes
    are the declared observations rather than candidate-selection inputs.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in TARGET_METADATA_KEYS or normalized.startswith("target_"):
                raise ClosureInputError(f"{path} contains target metadata: {key}")
            _source_reject(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _source_reject(child, f"{path}[{index}]")


def _hex_bytes(value: Any, *, extent: int | None = None, label: str = "bytes") -> bytes:
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError as exc:
            raise ClosureInputError(f"{label} is not hexadecimal") from exc
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ClosureInputError(f"{label} must be bytes or hex")
    result = bytes(value)
    if extent is not None and len(result) != extent:
        raise ClosureInputError(f"{label} has extent {len(result)}, expected {extent}")
    return result


def _geometry_cases(geometry: Any) -> Mapping[str, Any]:
    if geometry is None:
        return {}
    if isinstance(geometry, (str, Path)):
        geometry = _mapping(geometry, "geometry")
    if not isinstance(geometry, Mapping):
        raise ClosureInputError("geometry must be a mapping or JSON path")
    cases = geometry.get("cases", geometry.get("tasks", {}))
    if isinstance(cases, Mapping):
        return cases
    if isinstance(cases, list):
        out = {}
        for item in cases:
            if isinstance(item, Mapping) and "task" in item:
                out[str(item["task"])] = item
        return out
    return {}


def _geometry_case(geometry: Any, task: str) -> Mapping[str, Any]:
    cases = _geometry_cases(geometry)
    value = cases.get(task, {})
    return value if isinstance(value, Mapping) else {}


def _optional_receiver(
    value: Any,
    *,
    raw_dir: Path,
    label: str,
) -> bytes | None:
    """Resolve an optional source/candidate receiver reference for identity checks."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("receiver_bytes_hex", "bytes_hex", "hex"):
            if key in value:
                return _hex_bytes(value[key], label=label)
        for key in ("receiver_file", "receiver_path", "file", "path"):
            if key in value:
                return _optional_receiver(value[key], raw_dir=raw_dir, label=label)
        if "values" in value:
            value = value["values"]
    if isinstance(value, str):
        path = _resolve_geometry_path(value, raw_dir)
        return path.read_bytes()
    if isinstance(value, (list, tuple, np.ndarray)):
        return f32_array(value).tobytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ClosureInputError(f"unsupported {label} reference")


def _resolve_geometry_path(value: str | Path, raw_dir: Path) -> Path:
    """Resolve a geometry reference across local and collected result layouts.

    Geometry JSON can contain an absolute path from the Mac worker.  Collection
    intentionally moves the immutable files, so an absent absolute path is
    retried by basename under the result's ``geometry`` and ``payload/geometry``
    siblings.  No output directory is searched by content; the reference stays
    bound to its declared filename.
    """

    path = Path(value)
    if path.exists():
        return path
    candidates: list[Path] = []
    if not path.is_absolute():
        candidates.append(raw_dir / path)
        candidates.append(raw_dir.parent / path)
    candidates.extend(
        [
            raw_dir.parent / "geometry" / path.name,
            raw_dir.parent / "payload" / "geometry" / path.name,
            raw_dir.parent.parent / "geometry" / path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ClosureInputError(f"geometry reference is unavailable: {value}")


def _candidate_receiver(
    candidate: Mapping[str, Any],
    case_geometry: Mapping[str, Any],
    *,
    raw_dir: Path,
    candidate_index: int,
) -> bytes | None:
    for key in ("receiver_bytes_hex", "receiver_hex", "receiver_file", "receiver_path", "receiver"):
        if key in candidate:
            return _optional_receiver(candidate[key], raw_dir=raw_dir, label=f"candidate {candidate_index} receiver")
    table = case_geometry.get("candidate_receivers", case_geometry.get("receivers"))
    if isinstance(table, Mapping):
        identifier = candidate.get("id", candidate.get("index", candidate_index))
        value = table.get(str(identifier), table.get(identifier))
        if value is not None:
            return _optional_receiver(value, raw_dir=raw_dir, label=f"candidate {candidate_index} receiver")
    if isinstance(table, list) and candidate_index < len(table):
        return _optional_receiver(table[candidate_index], raw_dir=raw_dir, label=f"candidate {candidate_index} receiver")
    return None


def _source_receiver(case_geometry: Mapping[str, Any], *, raw_dir: Path) -> bytes | None:
    for key in ("source_receiver_bytes_hex", "source_receiver_hex", "source_receiver_file", "source_receiver_path", "source_receiver"):
        if key in case_geometry:
            return _optional_receiver(case_geometry[key], raw_dir=raw_dir, label="source receiver")
    return None


def _declared_bool(entry: Mapping[str, Any], key: str, actual: bool, label: str) -> bool:
    if key in entry and bool(entry[key]) != actual:
        raise ClosureInputError(f"{label} declaration disagrees with exact bytes")
    return actual


def _candidate_records(
    case: Mapping[str, Any],
    *,
    raw_dir: Path,
    geometry: Mapping[str, Any],
) -> tuple[bytes, list[dict[str, Any]], Any]:
    if "source_code_hex" not in case:
        raise ClosureInputError("case has no source_code_hex")
    source_code = _hex_bytes(case["source_code_hex"], extent=CODE_BYTES, label="source_code_hex")
    # Validate source code fields without changing the original bytes.  A
    # source rho of zero is not a valid RMS encoder output.
    source_code = exact_code_from_bytes(source_code[:12], source_code[12:])
    candidates = case.get("candidates", [])
    if not isinstance(candidates, list):
        raise ClosureInputError("case candidates must be a list")
    source_receiver = _source_receiver(geometry, raw_dir=raw_dir)
    records: list[dict[str, Any]] = []
    eligible: list[Any] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ClosureInputError(f"candidate {index} is not an object")
        _source_reject(candidate, f"candidate[{index}]")
        code = code_key(candidate)
        same_code = exact_code_equal(source_code, code)
        declared_same = _declared_bool(candidate, "same_code", same_code, f"candidate {index} same_code")
        candidate_receiver = _candidate_receiver(candidate, geometry, raw_dir=raw_dir, candidate_index=index)
        different_verified = None
        if source_receiver is not None and candidate_receiver is not None:
            different_verified = source_receiver != candidate_receiver
        declared_different = candidate.get("different_receiver")
        if declared_different is not None:
            if different_verified is not None and bool(declared_different) != different_verified:
                raise ClosureInputError(f"candidate {index} different_receiver disagrees with exact bytes")
            different_receiver = bool(declared_different)
            receiver_identity = "exact-bytes" if different_verified is not None else "declared"
        elif different_verified is not None:
            different_receiver = different_verified
            receiver_identity = "exact-bytes"
        else:
            different_receiver = False
            receiver_identity = "unavailable"
        row = {
            "index": index,
            "id": candidate.get("id", candidate.get("index", index)),
            "code_hex": code.hex(),
            "code_sha256": code_digest(code),
            "same_code": same_code,
            "different_receiver": different_receiver,
            "receiver_identity": receiver_identity,
            "eligible_exact_fiber": bool(same_code and different_receiver),
        }
        # Preserve the geometry ordering/offset provenance when the candidate
        # generator supplied it.  An offset is descriptive until an exact
        # receiver reference is available; it never participates in code
        # equality or selection.
        for key in ("array_offset", "offset", "array_index"):
            if key in candidate:
                row["array_offset"] = candidate[key]
                break
        declared_array_hash = candidate.get("array_sha256", candidate.get("receiver_sha256"))
        if declared_array_hash is not None:
            row["declared_receiver_sha256"] = str(declared_array_hash)
        if candidate_receiver is not None:
            receiver_hash = hashlib.sha256(candidate_receiver).hexdigest()
            row["receiver_sha256"] = receiver_hash
            if declared_array_hash is not None and str(declared_array_hash) != receiver_hash:
                raise ClosureInputError(f"candidate {index} receiver hash disagrees with exact bytes")
        records.append(row)
        if row["eligible_exact_fiber"]:
            eligible.append(row["id"])
    eligible_ids = [row["id"] for row in records if row["eligible_exact_fiber"]]
    first_eligible_id = eligible_ids[0] if eligible_ids else None
    selected_id = case.get("selected_id", case.get("selected_index"))
    # The original C emitter uses selected_index=-1 for an absent collision;
    # the revised JSON contract uses selected_id=null.  They have the same
    # source-only meaning and must not be treated as a malformed ID.
    if selected_id == -1:
        selected_id = None
    if selected_id is not None:
        # Old closure workers called the field selected_index.  Match either
        # candidate ID or its integer index while retaining the declared value.
        selected_row = next((row for row in records if row["id"] == selected_id), None)
        if selected_row is None and isinstance(selected_id, int) and 0 <= selected_id < len(records):
            selected_row = records[selected_id]
        if selected_row is None:
            raise ClosureInputError("selected candidate is absent")
        if selected_row["id"] not in eligible:
            raise ClosureInputError("selected candidate is not an exact-code/different-receiver pair")
        if first_eligible_id != selected_row["id"]:
            raise ClosureInputError("selected candidate is not the first source-only exact-fiber collision")
        selected_id = selected_row["id"]
    return source_code, records, selected_id


def _prefix(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("prefix"), str):
        return value["prefix"]
    raise ClosureInputError(f"{label} has no prefix")


def _branch_ports(
    raw_dir: Path,
    branches: Mapping[str, Any],
    *,
    hidden: int,
    intermediate: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any] = {}
    for branch, spec in branches.items():
        prefix = _prefix(spec, f"writes.{branch}")
        ports: dict[str, np.ndarray] = {}
        # A project branch intentionally has no receiver in the saved ABI.
        names = ("receiver", "normalized", "gate", "up", "write")
        for port in names:
            if branch == "project" and port == "receiver":
                continue
            array = _load_port(raw_dir, prefix, port, hidden=hidden, intermediate=intermediate, required=False)
            if array is not None:
                ports[port] = array
        loaded[branch] = ports
        metadata[branch] = {
            "prefix": prefix,
            "ports": sorted(ports),
            "files": {port: _file_sha256(_port_path(raw_dir, prefix, port)) for port in ports},
        }
    return loaded, metadata


def _compare_ports(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    *,
    ports: Iterable[str] = ("receiver", "normalized", "gate", "up", "write"),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for port in ports:
        if port not in left or port not in right:
            result[port] = {"available": False}
        else:
            result[port] = {"available": True, **array_metrics(left[port], right[port])}
    return result


def _minimax_half(metric: Mapping[str, Any], *, observable: str, pair: str) -> dict[str, Any] | None:
    if not metric.get("available") or metric.get("l2") is None:
        return None
    distance_l2 = float(metric["l2"])
    return {
        "observable": observable,
        "pair": pair,
        "endpoint_distance_l2": distance_l2,
        "deterministic_code_only_minimax_half_l2": distance_l2 / 2.0,
        "relative_endpoint_distance": metric.get("relative_l2"),
        "numeric_floor_fp32_epsilon": float(np.finfo(np.float32).eps),
        "interpretation": "lower bound for predicting this observable from the retained code alone on this exact pair",
    }


def _arm_label(arm: Mapping[str, Any]) -> str:
    policy = arm.get("policy")
    if isinstance(policy, str):
        return policy.lower().replace("_", "-")
    labels = ("native", "sham", "fiber", "fiber-repair", "compact-once", "compact-recurring", "corrected-recurring", "zero")
    if isinstance(policy, int) and 0 <= policy < len(labels):
        return labels[policy]
    name = str(arm.get("name", "")).lower()
    for label in labels:
        if name.endswith("-" + label) or name == label:
            return label
    return str(policy) if policy is not None else "unknown"


def _validate_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    steps = arm.get("steps")
    own = arm.get("own")
    if not isinstance(steps, list) or not isinstance(own, Mapping):
        raise ClosureInputError(f"arm {arm.get('name')} lacks steps/own")
    ids = own.get("ids", [])
    if not isinstance(ids, list) or len(steps) != len(ids) + 1:
        raise ClosureInputError(f"arm {arm.get('name')} token closure differs")
    for j, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ClosureInputError(f"arm {arm.get('name')} step {j} is not an object")
        if "step" in step and int(step["step"]) != j:
            raise ClosureInputError(f"arm {arm.get('name')} step order differs")
        if "position" in step and "parent_position" in arm:
            if int(step["position"]) != int(arm["parent_position"]) + j:
                raise ClosureInputError(f"arm {arm.get('name')} position clock differs")
    for j, token in enumerate(ids):
        if "top" in steps[j] and int(token) != int(steps[j]["top"]):
            raise ClosureInputError(f"arm {arm.get('name')} replaced an emitted token")
        if "input_id" in steps[j + 1] and int(token) != int(steps[j + 1]["input_id"]):
            raise ClosureInputError(f"arm {arm.get('name')} replaced a consumed token")
    if own.get("stopped") is not None and own.get("truncated") is not None and bool(own["stopped"]) == bool(own["truncated"]):
        raise ClosureInputError(f"arm {arm.get('name')} termination metadata differs")
    if own.get("all_delivered_consumed") is False:
        raise ClosureInputError(f"arm {arm.get('name')} did not consume delivered tokens")
    budget = arm.get("budget")
    if budget is not None and len(ids) > int(budget):
        raise ClosureInputError(f"arm {arm.get('name')} exceeded generation budget")
    if "final_position" in arm and "parent_position" in arm:
        expected = int(arm["parent_position"]) + len(steps)
        if int(arm["final_position"]) != expected:
            raise ClosureInputError(f"arm {arm.get('name')} final position differs")
    return {"name": arm.get("name"), "label": _arm_label(arm), "steps": len(steps), "ids": list(ids)}


def _arm_port(
    raw_dir: Path,
    arm: Mapping[str, Any],
    step_index: int,
    port: str,
    *,
    hidden: int,
    intermediate: int,
    required: bool = False,
) -> np.ndarray | None:
    name = arm.get("name")
    if not isinstance(name, str):
        return None
    stem = f"{name}-s{step_index:03d}"
    return _load_port(raw_dir, stem, port, hidden=hidden, intermediate=intermediate, required=required)


def _first_logit(
    raw_dir: Path,
    arm: Mapping[str, Any],
    *,
    vocab: int | None,
) -> tuple[np.ndarray | None, str | None]:
    name = arm.get("name")
    if not isinstance(name, str):
        return None, None
    path = raw_dir / f"{name}-s000-first.f32"
    if not path.exists():
        return None, None
    return _read_f32(path, extent=vocab, required=True), _file_sha256(path)


def _history_same(arm: Mapping[str, Any], reference: Mapping[str, Any], j: int) -> bool:
    a_steps, b_steps = arm.get("steps", []), reference.get("steps", [])
    if j >= len(a_steps) or j >= len(b_steps):
        return False
    a_prefix = [s.get("input_id") for s in a_steps[: j + 1]]
    b_prefix = [s.get("input_id") for s in b_steps[: j + 1]]
    return a_prefix == b_prefix


def _native_sham_check(
    raw_dir: Path,
    native: Mapping[str, Any] | None,
    sham: Mapping[str, Any] | None,
    *,
    hidden: int,
    intermediate: int,
    vocab: int | None,
) -> dict[str, Any]:
    if native is None or sham is None:
        return {"status": "not-found", "exact": None}
    native_first, native_hash = _first_logit(raw_dir, native, vocab=vocab)
    sham_first, sham_hash = _first_logit(raw_dir, sham, vocab=vocab)
    ports: dict[str, Any] = {}
    for port in ("input", "raw_receiver", "mlp_input", "mlp_return", "output", "kv16-17"):
        left = _arm_port(raw_dir, native, 0, port, hidden=hidden, intermediate=intermediate)
        right = _arm_port(raw_dir, sham, 0, port, hidden=hidden, intermediate=intermediate)
        if left is None or right is None:
            ports[port] = {"available": False}
        else:
            ports[port] = {"available": True, **array_metrics(left, right)}
    step_left = (native.get("steps") or [{}])[0]
    step_right = (sham.get("steps") or [{}])[0]
    metadata = {
        "logits_sha256_equal": bool(step_left.get("logits_sha256") == step_right.get("logits_sha256")) if step_left.get("logits_sha256") is not None and step_right.get("logits_sha256") is not None else None,
        "valid_kv_sha256_equal": bool(step_left.get("valid_kv_sha256") == step_right.get("valid_kv_sha256")) if step_left.get("valid_kv_sha256") is not None and step_right.get("valid_kv_sha256") is not None else None,
    }
    first_logit = {"available": native_first is not None and sham_first is not None}
    if native_first is not None and sham_first is not None:
        first_logit.update(array_metrics(native_first, sham_first))
    elif native_hash is not None or sham_hash is not None:
        first_logit.update({"native_sha256": native_hash, "sham_sha256": sham_hash, "bytes_equal": native_hash == sham_hash})
    checks = list(ports.values()) + [first_logit]
    available = [x for x in checks if x.get("available")]
    exact = bool(available) and all(bool(x.get("bytes_equal")) for x in available)
    return {
        "status": "observed" if available else "metadata-only" if any(metadata.values()) else "no-retained-ports",
        "exact": exact if available else None,
        "native": native.get("name"),
        "sham": sham.get("name"),
        "ports": ports,
        "first_logit": first_logit,
        "metadata": metadata,
    }


def _recursion_rows(
    raw_dir: Path,
    arms: list[Mapping[str, Any]],
    *,
    hidden: int,
    intermediate: int,
) -> list[dict[str, Any]]:
    native = next((a for a in arms if _arm_label(a) == "native"), None)
    if native is None:
        return []
    rows: list[dict[str, Any]] = []
    for arm in arms:
        if arm is native or _arm_label(arm) in {"native", "sham"}:
            continue
        _validate_arm(arm)
        steps = arm.get("steps", [])
        nsteps = native.get("steps", [])
        trajectory: list[dict[str, Any]] = []
        for j in range(min(len(steps), len(nsteps))):
            same_history = _history_same(arm, native, j)
            entry: dict[str, Any] = {
                "step": j,
                "same_complete_token_history": same_history,
                "comparison_kind": "same-token-history-fidelity" if same_history else "different-history-trajectory",
                "ports": {},
            }
            for port in ("input", "raw_receiver", "mlp_input", "mlp_return", "output", "kv16-17"):
                left = _arm_port(raw_dir, arm, j, port, hidden=hidden, intermediate=intermediate)
                right = _arm_port(raw_dir, native, j, port, hidden=hidden, intermediate=intermediate)
                if left is None or right is None:
                    entry["ports"][port] = {"available": False}
                else:
                    entry["ports"][port] = {"available": True, **array_metrics(left, right)}
            trajectory.append(entry)
        rows.append({"arm": arm.get("name"), "label": _arm_label(arm), "trajectory": trajectory})
    return rows


def analyze(raw: Any, bank: Any, driver: Any, texts: Mapping[str, Any] | None = None, geometry: Any = None) -> dict[str, Any]:
    """Analyze a saved state-closure report without executing a model.

    Candidate code and receiver identity are inspected first.  Retained MLP,
    readout and continuation arrays are loaded only for a driver-declared exact
    code/different-receiver selection.  This ordering is part of the analysis
    contract and prevents a downstream outcome from selecting a favorable
    member of the fiber.

    ``bank`` is accepted and recorded for caller symmetry and custody joins;
    the report adapter does not reinterpret basis/weights from a mutable bank.
    Optional geometry JSON can supply exact receiver bytes and dimensions for
    independent identity checks.
    """

    del bank  # Static bank custody is owned by the worker/driver; no re-solving here.
    raw_dir = _raw_directory(raw)
    report = _mapping(driver, "driver")
    if report.get("schema") != "tcm-state-closure-driver-v1":
        raise ClosureInputError("closure driver schema differs")
    if int(report.get("layer", -1)) != 16 or int(report.get("capacity", -1)) != 512:
        raise ClosureInputError("closure layer/capacity contract differs")
    if report.get("source_replay_checked") is not True:
        raise ClosureInputError("source replay was not checked")
    geom_obj = _mapping(geometry, "geometry") if isinstance(geometry, (str, Path)) else (geometry or {})
    _source_reject(geom_obj, "geometry")
    hidden = int(geom_obj.get("hidden", HIDDEN)) if isinstance(geom_obj, Mapping) else HIDDEN
    intermediate = int(geom_obj.get("intermediate", 8960)) if isinstance(geom_obj, Mapping) else 8960
    vocab_value = geom_obj.get("vocab") if isinstance(geom_obj, Mapping) else None
    vocab = int(vocab_value) if vocab_value is not None else None
    if hidden <= 0 or intermediate <= 0:
        raise ClosureInputError("invalid closure dimensions")
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ClosureInputError("closure driver cases must be a list")

    exact_fiber_codes: list[dict[str, Any]] = []
    per_case_writes: list[dict[str, Any]] = []
    minimax: list[dict[str, Any]] = []
    native_sham: list[dict[str, Any]] = []
    recursions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    selected_count = 0

    for case in cases:
        if not isinstance(case, Mapping):
            raise ClosureInputError("closure case is not an object")
        task = str(case.get("task", f"case-{len(exact_fiber_codes)}"))
        case_geometry = _geometry_case(geometry, task)
        source_code, candidates, selected_id = _candidate_records(case, raw_dir=raw_dir, geometry=case_geometry)
        if case.get("actual_runtime_encoder_verified") is False:
            raise ClosureInputError(f"{task} candidates were not verified by the runtime encoder")
        selected = next((row for row in candidates if row["id"] == selected_id), None) if selected_id is not None else None
        first_eligible_id = next((row["id"] for row in candidates if row["eligible_exact_fiber"]), None)
        case_record = {
            "task": task,
            "source_code_hex": source_code.hex(),
            "source_code_sha256": code_digest(source_code),
            "candidates": candidates,
            "eligible_exact_fiber_ids": [row["id"] for row in candidates if row["eligible_exact_fiber"]],
            "first_eligible_exact_fiber_id": first_eligible_id,
            "selected_id": selected_id,
            "selected_code_hex": selected["code_hex"] if selected else None,
            "selection_order_checked": selected_id == first_eligible_id,
            "actual_runtime_encoder_verified": case.get("actual_runtime_encoder_verified"),
            "selection_used_outcomes": False,
            "outcome_scope": "not-loaded-before-exact-code-selection",
        }
        exact_fiber_codes.append(case_record)
        if selected is None:
            case_record["status"] = "evidence-incomplete-no-exact-fiber-selection"
            per_case_writes.append({"task": task, "status": case_record["status"], "branches": {}, "pairs": {}})
            continue
        selected_count += 1
        # From this point onward the candidate is fixed by raw source code and
        # receiver identity.  Outcome loading is intentionally below the gate.
        branches = case.get("writes", {})
        if not isinstance(branches, Mapping):
            raise ClosureInputError(f"{task} writes must be an object")
        loaded, branch_meta = _branch_ports(raw_dir, branches, hidden=hidden, intermediate=intermediate)
        pairs: dict[str, Any] = {}
        for left_name, right_name in (("source", "fiber"), ("source", "repair"), ("fiber", "repair"), ("source", "project"), ("fiber", "project")):
            if left_name in loaded and right_name in loaded:
                pairs[f"{left_name}_vs_{right_name}"] = _compare_ports(loaded[left_name], loaded[right_name])
        project_identity: dict[str, Any]
        project_alias = next((name for name in ("fiber_project", "project_fiber", "project-alt") if name in loaded), None)
        if project_alias is None:
            project_identity = {"status": "unavailable", "ports": {}}
        else:
            project_identity = {
                "status": "observed",
                "alias": project_alias,
                # PROJECT's normalized port is an actual RMS diagnostic for
                # the receiver, while its gate/up/write operands are formed
                # from the retained code.  Compare only the code-derived
                # consumer ports for the equal-code identity check.
                "ports": _compare_ports(loaded.get("project", {}), loaded[project_alias], ports=("gate", "up", "write")),
            }
            project_identity["exact"] = all(
                value.get("available") and value.get("bytes_equal")
                for value in project_identity["ports"].values()
            )
        case_writes = {
            "task": task,
            "status": "observed",
            "branches": branch_meta,
            "pairs": pairs,
            "project_equal_code_identity": project_identity,
        }
        per_case_writes.append(case_writes)
        write_metric = pairs.get("source_vs_fiber", {}).get("write")
        lower_bound = _minimax_half(write_metric, observable="mlp_write", pair="source_vs_fiber")
        if lower_bound is not None:
            lower_bound["task"] = task
            minimax.append(lower_bound)
            observations.append({"kind": "observation", "task": task, "observable": "mlp_write", "metric": write_metric})

        arms = case.get("arms", [])
        if not isinstance(arms, list):
            raise ClosureInputError(f"{task} arms must be a list")
        validated_arms = []
        for arm in arms:
            if not isinstance(arm, Mapping):
                raise ClosureInputError(f"{task} arm is not an object")
            validated_arms.append(_validate_arm(arm))
        native_arm = next((arm for arm in arms if _arm_label(arm) == "native"), None)
        sham_arm = next((arm for arm in arms if _arm_label(arm) == "sham"), None)
        native_sham.append({"task": task, **_native_sham_check(raw_dir, native_arm, sham_arm, hidden=hidden, intermediate=intermediate, vocab=vocab)})
        recursions.extend({"task": task, **row} for row in _recursion_rows(raw_dir, arms, hidden=hidden, intermediate=intermediate))

        if texts:
            # Text is deliberately attached after selection and never enters
            # candidate choice.  Keep it opt-in because it can be held-out
            # answer material in some callers.
            case_writes["texts_observed_after_selection"] = {
                str(arm.get("name")): texts[str(arm.get("name"))]
                for arm in arms
                if isinstance(arm.get("name"), str) and arm.get("name") in texts
            }

    no_collision = selected_count == 0
    summary = {
        "cases": len(cases),
        "selected_exact_fiber_cases": selected_count,
        "exact_code_bytes": CODE_BYTES,
        "hidden": hidden,
        "intermediate": intermediate,
        "mechanics_status": "valid",
        "instrument_status": "evidence-incomplete" if no_collision else "applicable",
        "trend_status": "no-selected-exact-pair" if no_collision else "observed",
        "terminal_status": "not-authorized",
        "selection_rule": "raw 16-byte code equality plus source-only receiver identity; no outcome search",
        "numeric_floor_fp32_epsilon": float(np.finfo(np.float32).eps),
    }
    return {
        "schema": "tcm-state-closure-analysis-v1",
        "summary": summary,
        "exact_fiber_codes": exact_fiber_codes,
        "per_case_write_differences": per_case_writes,
        "code_only_minimax_half": minimax,
        "native_sham_exact_checks": native_sham,
        "recursions_same_history": recursions,
        "observations": observations,
    }


__all__ = [
    "Candidate",
    "ClosureInputError",
    "CODE_BYTES",
    "HIDDEN",
    "Projection",
    "RANK",
    "array_metrics",
    "code_digest",
    "code_key",
    "evaluate_exact_code_pairs",
    "exact_code",
    "exact_code_equal",
    "exact_code_from_bytes",
    "f32_array",
    "gram_projection",
    "group_exact_codes",
    "analyze",
    "materialized_geometry",
    "random_spherical_candidates",
    "reject_target_metadata",
    "sequential_dot_f64",
    "spherical_candidate",
]
