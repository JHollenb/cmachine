"""Deterministic CPU geometry for omitted-complement receiver fibers.

This module does not load a model or invoke CUDA.  It consumes authenticated
composition receiver captures and the frozen B3 QR files, then writes one
source FP32 vector and one contiguous FP32 candidate file per task.  The native
worker remains responsible for replaying candidates through the C consumer and
for selecting candidates whose GPU RMS scale collides with the source scale.

The public entry point is generate(raw, bank, out).  raw is the composition raw
directory, bank is the composition payload/bank directory, and out is an output
directory.  The returned manifest has a tasks list suitable for a C reader.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SEED = 20261106
TASKS = ("recall", "math", "prose")
ANGLE_MAGNITUDES = (1.0, 0.6, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
ANGLE_PAIRS = tuple(((-value, value)) for value in ANGLE_MAGNITUDES)
ANGLES = tuple(angle for pair in ANGLE_PAIRS for angle in pair)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_f32(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype="<f4")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"non-finite or non-vector array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes(order="C"))


def _read_f32(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) == 0 or len(data) % 4:
        raise ValueError(f"invalid FP32 extent: {path}")
    values = np.frombuffer(data, dtype="<f4").copy()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite FP32 source: {path}")
    return values


def _read_f64(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) == 0 or len(data) % 8:
        raise ValueError(f"invalid FP64 extent: {path}")
    values = np.frombuffer(data, dtype="<f8").copy()
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite FP64 source: {path}")
    return values


def _file_record(path: Path, shape: Sequence[int] | None = None) -> dict:
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if shape is not None:
        record["shape"] = list(shape)
    return record


def _resolve_raw(raw: str | Path | Mapping[str, str | Path]) -> dict[str, Path]:
    if isinstance(raw, Mapping):
        paths = {str(task): Path(value) for task, value in raw.items()}
    else:
        root = Path(raw)
        paths = {
            task: root / f"{task}-native-s000-raw_receiver.f32"
            for task in TASKS
        }
        if not all(path.is_file() for path in paths.values()):
            nested = root / "raw"
            paths = {
                task: nested / f"{task}-native-s000-raw_receiver.f32"
                for task in TASKS
            }
    if set(paths) != set(TASKS):
        raise ValueError(f"raw inputs must contain exactly {TASKS}")
    for task, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing receiver for {task}: {path}")
    return paths


def _resolve_bank(
    bank: str | Path | Mapping[str, str | Path] | Sequence[str | Path],
) -> tuple[Path, Path]:
    if isinstance(bank, Mapping):
        reuse = bank.get("reuse_qr") or bank.get("reuse-qr.f64")
        enrich = bank.get("enrich_qr") or bank.get("enrich-qr.f64")
        if reuse is None or enrich is None:
            raise ValueError("bank mapping needs reuse_qr and enrich_qr")
        return Path(reuse), Path(enrich)
    if isinstance(bank, Sequence) and not isinstance(bank, (str, bytes, Path)):
        if len(bank) != 2:
            raise ValueError("bank sequence must contain reuse and enrich QR paths")
        return Path(bank[0]), Path(bank[1])
    root = Path(bank)
    if root.is_file():
        raise ValueError("bank must be a directory or two QR paths")
    return root / "reuse-qr.f64", root / "enrich-qr.f64"


def _load_bank(
    bank: str | Path | Mapping[str, str | Path] | Sequence[str | Path],
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], dict]:
    reuse_path, enrich_path = _resolve_bank(bank)
    reuse = _read_f64(reuse_path)
    enrich = _read_f64(enrich_path)
    if reuse.size < 4 or (reuse.size - 4) % 2:
        raise ValueError("reuse QR has no valid two-column layout")
    dimension = (reuse.size - 4) // 2
    if enrich.size != dimension + 3:
        raise ValueError("enrich QR dimension does not match reuse QR")
    columns = np.column_stack(
        (reuse[:dimension], reuse[dimension : 2 * dimension], enrich[:dimension])
    ).astype(np.float64, copy=False)
    factors = (
        float(reuse[2 * dimension]),
        float(reuse[2 * dimension + 1]),
        float(reuse[2 * dimension + 2]),
        float(enrich[dimension]),
        float(enrich[dimension + 1]),
        float(enrich[dimension + 2]),
    )
    if not all(math.isfinite(value) for value in factors):
        raise ValueError("non-finite QR factors")
    if any(abs(value) <= 0.0 for value in (factors[0], factors[2], factors[5])):
        raise ValueError("singular QR diagonal")
    records = {
        "reuse_qr": _file_record(reuse_path),
        "enrich_qr": _file_record(enrich_path),
        "dimension": dimension,
        "column_sha256": [
            _sha256_bytes(np.asarray(columns[:, index], dtype="<f8").tobytes())
            for index in range(3)
        ],
        "gram": (columns.T @ columns).tolist(),
        "factors": list(factors),
    }
    return columns, factors, records


def _project(columns: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return b=Q(QTQ)^-1QTr and e=r-b in float64."""
    gram = columns.T @ columns
    rhs = columns.T @ vector
    coordinates = np.linalg.solve(gram, rhs)
    base = columns @ coordinates
    return base, vector - base


def _orthogonal_direction(
    columns: np.ndarray,
    omitted: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    target_norm = float(np.linalg.norm(omitted))
    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError("source receiver has no nonzero omitted complement")
    constraints = np.column_stack((columns, omitted))
    gram = constraints.T @ constraints
    for _ in range(8):
        proposal = np.asarray(rng.standard_normal(columns.shape[0]), dtype=np.float64)
        correction = constraints @ np.linalg.solve(gram, constraints.T @ proposal)
        direction = proposal - correction
        norm = float(np.linalg.norm(direction))
        if math.isfinite(norm) and norm > target_norm * 1e-12:
            return direction * (target_norm / norm)
    raise ValueError("seeded random direction was numerically dependent")


def _sequential_dots(columns: np.ndarray, vector: np.ndarray) -> tuple[float, float, float]:
    dots: list[float] = []
    for column in range(3):
        value = 0.0
        for index in range(vector.size):
            value += float(columns[index, column]) * float(vector[index])
        dots.append(value)
    return dots[0], dots[1], dots[2]


def historical_coefficients(
    vector: np.ndarray,
    columns: np.ndarray,
    factors: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Match the C sequential-double QR solve and FP32 coefficient cast."""
    value = np.asarray(vector)
    if value.ndim != 1 or value.size != columns.shape[0]:
        raise ValueError("vector/QR dimensions differ")
    r00, r01, r11, r02, r12, r22 = (float(item) for item in factors)
    v0, v1, v2 = _sequential_dots(columns, value)
    c2 = v2 / r22
    c1 = (v1 - r12 * c2) / r11
    c0 = (v0 - r01 * c1 - r02 * c2) / r00
    fp64 = np.asarray((c0, c1, c2), dtype=np.float64)
    fp32 = np.asarray(fp64, dtype="<f4")
    if not np.isfinite(fp64).all() or not np.isfinite(fp32).all():
        raise ValueError("non-finite coefficient solve")
    return fp64, fp32


def _angle_label(angle: float) -> str:
    sign = "p" if angle > 0 else "m"
    magnitude = f"{abs(angle):.6f}".replace(".", "d")
    return f"{sign}{magnitude}"


def _candidate_record(
    *,
    index: int,
    task: str,
    direction: int,
    angle: float,
    source: np.ndarray,
    base: np.ndarray,
    omitted: np.ndarray,
    vector: np.ndarray,
    candidate: np.ndarray,
    materialized: np.ndarray,
    columns: np.ndarray,
    factors: Sequence[float],
    candidate_file: Path,
) -> dict:
    fp64, fp32 = historical_coefficients(materialized, columns, factors)
    qdot = columns.T @ materialized.astype(np.float64)
    source_norm = float(np.linalg.norm(source.astype(np.float64)))
    candidate_f64_norm = float(np.linalg.norm(candidate))
    candidate_f32_norm = float(np.linalg.norm(materialized.astype(np.float64)))
    return {
        "index": index,
        "id": f"{task}-d{direction:02d}-a{_angle_label(angle)}",
        "task": task,
        "direction": direction,
        # Keep the short spelling alongside the unit-explicit spelling so a
        # small native-side manifest reader can consume the record without
        # having to know the Python implementation's naming convention.
        "angle": float(angle),
        "angle_rad": float(angle),
        "path": str(candidate_file),
        "offset_bytes": index * materialized.size * 4,
        "bytes": materialized.size * 4,
        "dtype": "<f4",
        "shape": [int(materialized.size)],
        "sha256": _sha256_bytes(materialized.tobytes()),
        "finite": True,
        "geometry": {
            "source_l2_f64": source_norm,
            "base_l2_f64": float(np.linalg.norm(base)),
            "omitted_l2_f64": float(np.linalg.norm(omitted)),
            "direction_l2_f64": float(np.linalg.norm(vector)),
            "candidate_l2_f64": candidate_f64_norm,
            "candidate_l2_f32": candidate_f32_norm,
            "candidate_minus_source_l2_f32": float(
                np.linalg.norm(materialized.astype(np.float64) - source.astype(np.float64))
            ),
            "fiber_norm_error_f64": candidate_f64_norm - source_norm,
            "q_dot_omitted_max_abs": float(np.max(np.abs(columns.T @ omitted))),
            "q_dot_direction_max_abs": float(np.max(np.abs(columns.T @ vector))),
            "q_dot_candidate": qdot.tolist(),
            "raw_sum_squares_f64": float(np.sum(materialized.astype(np.float64) ** 2)),
        },
        "coefficients_fp64": [float(item) for item in fp64],
        "coefficients_fp32": [float(item) for item in fp32],
        "coefficient_bytes_hex": fp32.tobytes().hex(),
        "coefficient_sha256": _sha256_bytes(fp32.tobytes()),
        "coefficient_algorithm": "sequential-double-dot-then-triangular-backsolve",
    }


def generate(
    raw: str | Path | Mapping[str, str | Path],
    bank: str | Path | Mapping[str, str | Path] | Sequence[str | Path],
    out: str | Path,
) -> dict:
    """Write source and 64-candidate spherical fibers for recall, math, prose."""
    raw_paths = _resolve_raw(raw)
    columns, factors, bank_record = _load_bank(bank)
    dimension = columns.shape[0]
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(SEED))
    tasks: list[dict] = []

    for task in TASKS:
        source_path = raw_paths[task]
        source = _read_f32(source_path)
        if source.size != dimension:
            raise ValueError(f"{task} receiver dimension {source.size} != bank {dimension}")
        source = np.asarray(source, dtype="<f4")
        base, omitted = _project(columns, source.astype(np.float64))
        source_file = output / f"{task}-source.f32"
        candidate_file = output / f"{task}-candidates.f32"
        ids_file = output / f"{task}-candidate-ids.txt"
        base_file = output / f"{task}-base.f32"
        omitted_file = output / f"{task}-omitted.f32"
        _write_f32(source_file, source)
        _write_f32(base_file, base)
        _write_f32(omitted_file, omitted)

        directions: list[np.ndarray] = [
            _orthogonal_direction(columns, omitted, rng) for _ in range(4)
        ]
        for direction_index, vector in enumerate(directions):
            if np.max(np.abs(columns.T @ vector)) > 1e-10:
                raise ValueError("direction is not orthogonal to B3")
        candidate_records: list[dict] = []
        candidate_bytes: list[bytes] = []
        candidate_ids: list[str] = []
        index = 0
        for magnitude, pair in zip(ANGLE_MAGNITUDES, ANGLE_PAIRS):
            del magnitude
            for direction_index, vector in enumerate(directions):
                for angle in pair:
                    candidate = (
                        base
                        + omitted * math.cos(angle)
                        + vector * math.sin(angle)
                    )
                    materialized = np.asarray(candidate, dtype="<f4")
                    if not np.isfinite(materialized).all():
                        raise ValueError(f"non-finite candidate {task} d{direction_index} {angle}")
                    record = _candidate_record(
                        index=index,
                        task=task,
                        direction=direction_index,
                        angle=angle,
                        source=source,
                        base=base,
                        omitted=omitted,
                        vector=vector,
                        candidate=candidate,
                        materialized=materialized,
                        columns=columns,
                        factors=factors,
                        candidate_file=candidate_file,
                    )
                    candidate_records.append(record)
                    candidate_ids.append(record["id"])
                    candidate_bytes.append(materialized.tobytes())
                    index += 1
        if index != 64:
            raise AssertionError(f"expected 64 candidates, got {index}")
        candidate_file.write_bytes(b"".join(candidate_bytes))
        ids_file.write_text("\n".join(candidate_ids) + "\n")

        source_fp64, source_fp32 = historical_coefficients(source, columns, factors)
        tasks.append(
            {
                "task": task,
                "dimension": dimension,
                "source": {
                    "input_path": str(source_path),
                    "input_sha256": _sha256(source_path),
                    "file": str(source_file),
                    "sha256": _sha256(source_file),
                    "bytes": source_file.stat().st_size,
                    "dtype": "<f4",
                    "shape": [dimension],
                },
                "source_sha256": _sha256(source_file),
                # These aliases are deliberately direct task-level fields:
                # the C driver already knows the output directory and only
                # needs the stable per-task file names/count.  The nested
                # source record retains the input and byte-level details.
                "source_file": str(source_file),
                "baseline_file": str(source_file),
                "base_file": str(base_file),
                "omitted_file": str(omitted_file),
                "candidate_file": str(candidate_file),
                "candidate_ids_file": str(ids_file),
                "candidate_count": 64,
                "candidate_vector_bytes": dimension * 4,
                "candidate_file_sha256": _sha256(candidate_file),
                "candidate_ids_file_sha256": _sha256(ids_file),
                "source_coefficients_fp64": [float(item) for item in source_fp64],
                "source_coefficients_fp32": [float(item) for item in source_fp32],
                "source_coefficient_bytes_hex": source_fp32.tobytes().hex(),
                "source_geometry": {
                    "source_l2_f64": float(np.linalg.norm(source.astype(np.float64))),
                    "base_l2_f64": float(np.linalg.norm(base)),
                    "omitted_l2_f64": float(np.linalg.norm(omitted)),
                    "q_dot_omitted_max_abs": float(
                        np.max(np.abs(columns.T @ omitted))
                    ),
                    "source_minus_base_minus_omitted_l2_f64": float(
                        np.linalg.norm(source.astype(np.float64) - base - omitted)
                    ),
                },
                "directions": [
                    {
                        "direction": index,
                        "norm_l2_f64": float(np.linalg.norm(vector)),
                        "q_dot_max_abs": float(np.max(np.abs(columns.T @ vector))),
                        "omitted_dot": float(np.dot(omitted, vector)),
                    }
                    for index, vector in enumerate(directions)
                ],
                "candidates": candidate_records,
            }
        )

    manifest = {
        "schema": "tcm-closure-geometry-v2",
        "seed": SEED,
        "tasks": tasks,
        "angles_rad": [float(angle) for angle in ANGLES],
        "angle_order": "descending_abs_angle_then_direction_then_negative_positive",
        "directions_per_task": 4,
        "candidates_per_task": 64,
        "dimension": dimension,
        "dtype": "<f4",
        "bank": bank_record,
        "selection": {
            "gpu_rho_collision": "deferred to native worker",
            "candidate_materialization": "FP64 geometry then little-endian FP32 cast",
            "baseline": "source file is separate; no zero-angle candidate is included",
        },
    }
    manifest_path = output / "geometry.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = [
    "ANGLE_MAGNITUDES",
    "ANGLES",
    "SEED",
    "TASKS",
    "generate",
    "historical_coefficients",
]
