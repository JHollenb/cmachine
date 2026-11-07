"""Compile source-only low-rank gate/up operand banks.

The C Qwen loader exposes the selected projection weights as row-major
``float32`` arrays, with shape ``(intermediate, hidden)``.  This module keeps
that ABI explicit and compiles a common *raw receiver* basis without reading
any receiver, token, target, or activation data.

For a raw receiver ``r`` and learned post-attention scale ``gamma`` we use

    A = G diag(gamma),  B = U diag(gamma)
    z = Q.T r
    g_hat = rho(r) (A Q) z
    u_hat = rho(r) (B Q) z

where the runtime files containing ``Q.T`` have shape ``(rank, hidden)``.
The source basis is a deterministic randomized right-singular subspace of the
joint map ``J = [A; B]``.  The matched control is an independent deterministic
Gaussian orthonormal basis with the same rank.  Compilation is performed in
float64 for stable CPU QR/SVD operations and the runtime factors are emitted
as little-endian contiguous float32, exactly as the C matvec ABI consumes
them.  The compiler never constructs a basis from future activations.

This is a small, model-free compiler.  It deliberately does not import torch,
the C runtime, a tokenizer, or a model loader.  The caller is responsible for
exporting the exact selected C tensors and for binding this source file in its
experiment manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "tcm-operand-basis-bank-v1"
DEFAULT_HIDDEN = 1536
DEFAULT_INTERMEDIATE = 8960
DEFAULT_RANK = 64
DEFAULT_OVERSAMPLE = 16
DEFAULT_POWER_ITERATIONS = 2
DEFAULT_SEED = 20261106
DEFAULT_RANDOM_SEED = DEFAULT_SEED + 1


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of an exact byte sequence."""

    return hashlib.sha256(data).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/containers to ordinary JSON values."""

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


@dataclass(frozen=True)
class SourceArray:
    """One exact exported little-endian FP32 tensor and its custody metadata."""

    name: str
    path: str
    resolved_path: str
    shape: tuple[int, ...]
    dtype: str
    order: str
    byte_count: int
    sha256: str
    finite: bool
    min_value: float
    max_value: float
    l2_norm: float

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class SourceWeights:
    """Exact C-exported tensors needed by the source-only compiler."""

    gate: np.ndarray
    up: np.ndarray
    gamma: np.ndarray
    records: dict[str, SourceArray]
    manifest_path: str | None = None

    @property
    def hidden(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def intermediate(self) -> int:
        return int(self.gate.shape[0])


@dataclass(frozen=True)
class CompilerConfig:
    layer: int
    rank: int
    oversample: int
    power_iterations: int
    seed: int
    random_seed: int
    algorithm_dtype: str = "float64"
    materialized_dtype: str = "<f4"
    input_basis: str = "raw_receiver"
    source_method: str = "randomized_right_singular_subspace"
    random_method: str = "gaussian_qr"


@dataclass
class CompiledBanks:
    """Compiled source/random arrays before they are serialized."""

    config: CompilerConfig
    source: SourceWeights
    banks: dict[str, dict[str, np.ndarray]]
    metrics: dict[str, Any]


def _as_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None


def _manifest_entry(manifest: Mapping[str, Any], key: str) -> tuple[Any, tuple[int, ...] | None]:
    """Find a tensor path in a few intentionally small exporter schemas."""

    containers: list[Mapping[str, Any]] = []
    for container_name in ("weights", "source_weights", "source"):
        container = manifest.get(container_name)
        if isinstance(container, Mapping):
            containers.append(container)
    containers.append(manifest)

    # The native exporter uses the C tensor names ``G`` and ``U`` while the
    # Python API calls the same arrays ``gate`` and ``up``.  Accept both, but
    # keep the normalized names in SourceWeights and in the output manifest.
    aliases_by_key = {
        "gate": ("gate", "G", "g", "gate_path", "G_path", "gate_file", "G_file"),
        "up": ("up", "U", "u", "up_path", "U_path", "up_file", "U_file"),
        "gamma": ("gamma", "Gamma", "gamma_path", "gamma_file"),
    }
    aliases = aliases_by_key.get(key, (key, f"{key}_path", f"{key}_file"))
    for container in containers:
        for alias in aliases:
            if alias not in container:
                continue
            value = container[alias]
            declared_shape = None
            if isinstance(value, Mapping):
                declared_shape = _as_shape(value.get("shape"))
                if declared_shape is None and "rows" in value and "cols" in value:
                    declared_shape = (int(value["rows"]), int(value["cols"]))
                if declared_shape is None and "count" in value:
                    declared_shape = (int(value["count"]),)
                value = value.get("path", value.get("file"))
            if value is not None:
                return value, declared_shape
        shapes = container.get("shapes")
        if isinstance(shapes, Mapping) and key in shapes:
            # A path can live at the parent while shapes are kept in a table.
            for alias in aliases:
                if alias in container:
                    return container[alias], _as_shape(shapes[key])
    raise ValueError(f"manifest does not contain a path for {key!r}")


def _resolve_path(path_value: Any, *, manifest_path: Path | None) -> Path:
    path = Path(os.fspath(path_value))
    if not path.is_absolute() and manifest_path is not None:
        path = manifest_path.parent / path
    return path.expanduser().resolve()


def _read_f32(path: Path, name: str, shape: tuple[int, ...], declared_shape: tuple[int, ...] | None) -> tuple[np.ndarray, SourceArray]:
    if declared_shape is not None and declared_shape != shape:
        raise ValueError(f"{name} declared shape {declared_shape} != expected {shape}")
    raw = path.read_bytes()
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"{name} has {len(raw)} bytes; expected {expected_bytes} for {shape}")
    # Parsing as <f4 is intentional: the C export is an exact little-endian
    # FP32 byte stream, and the source hash below is over the unmodified bytes.
    arr = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    finite = bool(np.isfinite(arr).all())
    if not finite:
        raise ValueError(f"{name} contains non-finite source weights")
    arr = np.asarray(arr, dtype="<f4", order="C")
    record = SourceArray(
        name=name,
        path=str(path),
        resolved_path=str(path),
        shape=shape,
        dtype="<f4",
        order="C",
        byte_count=len(raw),
        sha256=sha256_bytes(raw),
        finite=finite,
        min_value=float(arr.min()),
        max_value=float(arr.max()),
        l2_norm=float(np.linalg.norm(arr.astype(np.float64))),
    )
    return arr, record


def load_source_weights(
    *,
    manifest: str | os.PathLike[str] | None = None,
    gate: str | os.PathLike[str] | None = None,
    up: str | os.PathLike[str] | None = None,
    gamma: str | os.PathLike[str] | None = None,
    hidden: int = DEFAULT_HIDDEN,
    intermediate: int = DEFAULT_INTERMEDIATE,
) -> SourceWeights:
    """Load exact little-endian FP32 C exports from a manifest or three paths.

    Manifest paths are resolved relative to the manifest.  Direct paths are
    resolved by the normal filesystem rules.  The returned records retain the
    exact source-byte SHA-256 needed to bind a compiled bank to its exporter.
    """

    manifest_path: Path | None = None
    document: Mapping[str, Any] | None = None
    if manifest is not None:
        manifest_path = Path(manifest).expanduser().resolve()
        document = json.loads(manifest_path.read_text())
        if not isinstance(document, Mapping):
            raise ValueError("source manifest must contain a JSON object")

    provided = {"gate": gate, "up": up, "gamma": gamma}
    expected = {
        "gate": (int(intermediate), int(hidden)),
        "up": (int(intermediate), int(hidden)),
        "gamma": (int(hidden),),
    }
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, SourceArray] = {}
    for name, direct_path in provided.items():
        declared_shape = None
        path_value: Any = direct_path
        if path_value is None:
            if document is None:
                raise ValueError(f"missing direct path for {name}")
            path_value, declared_shape = _manifest_entry(document, name)
        path = _resolve_path(path_value, manifest_path=manifest_path)
        array, record = _read_f32(path, name, expected[name], declared_shape)
        # Keep the path as supplied where possible, while resolved_path remains
        # authoritative for audit/debugging.
        if direct_path is not None:
            record = SourceArray(
                name=record.name,
                path=os.fspath(direct_path),
                resolved_path=record.resolved_path,
                shape=record.shape,
                dtype=record.dtype,
                order=record.order,
                byte_count=record.byte_count,
                sha256=record.sha256,
                finite=record.finite,
                min_value=record.min_value,
                max_value=record.max_value,
                l2_norm=record.l2_norm,
            )
        arrays[name] = array
        records[name] = record
    return SourceWeights(
        gate=arrays["gate"],
        up=arrays["up"],
        gamma=arrays["gamma"],
        records=records,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
    )


def _validate_arrays(gate: np.ndarray, up: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = np.asarray(gate, dtype="<f4", order="C")
    up = np.asarray(up, dtype="<f4", order="C")
    gamma = np.asarray(gamma, dtype="<f4", order="C").reshape(-1)
    if gate.ndim != 2 or up.ndim != 2 or gamma.ndim != 1:
        raise ValueError("gate/up must be rank-2 and gamma rank-1")
    if gate.shape != up.shape or gate.shape[1] != gamma.shape[0]:
        raise ValueError(f"incompatible source shapes gate={gate.shape}, up={up.shape}, gamma={gamma.shape}")
    if not (np.isfinite(gate).all() and np.isfinite(up).all() and np.isfinite(gamma).all()):
        raise ValueError("source arrays must be finite")
    return gate, up, gamma


def _canonicalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64, order="C").copy()
    for i in range(rows.shape[0]):
        pivot = int(np.argmax(np.abs(rows[i])))
        if rows[i, pivot] < 0.0:
            rows[i] *= -1.0
    return rows


def _qr_sign_stable(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix, mode="reduced")
    # QR signs do not matter for the final subspace, but fixing them makes
    # intermediate and repeated compiler manifests reproducible across runs.
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q *= signs[None, :]
    return q


def _randomized_source_basis(
    folded_gate: np.ndarray,
    folded_up: np.ndarray,
    *,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``Q.T`` for the joint map using source-only randomized iteration."""

    h, d = folded_gate.shape
    width = int(rank) + int(oversample)
    if rank < 1 or width > d:
        raise ValueError(f"rank+oversample must be in [1, hidden], got {width} and {d}")
    if power_iterations < 0:
        raise ValueError("power_iterations must be non-negative")

    rng = np.random.default_rng(int(seed))
    omega = rng.standard_normal((d, width)).astype(np.float64)

    def apply(x: np.ndarray) -> np.ndarray:
        return np.vstack((folded_gate @ x, folded_up @ x))

    def transpose(y: np.ndarray) -> np.ndarray:
        return folded_gate.T @ y[:h] + folded_up.T @ y[h:]

    left = _qr_sign_stable(apply(omega))
    for _ in range(int(power_iterations)):
        right = _qr_sign_stable(transpose(left))
        left = _qr_sign_stable(apply(right))

    # Avoid materializing the 2H-by-D joint map a second time merely to form
    # the thin sketch.  The two terms are algebraically the same as
    # ``left.T @ J`` and keep the CPU peak lower while preserving the frozen
    # source-only algorithm.
    sketch = left[:h].T @ folded_gate + left[h:].T @ folded_up
    _, singular_values, vt = np.linalg.svd(sketch, full_matrices=False)
    q_t = _canonicalize_rows(vt[:rank])
    total_energy = float(np.sum(folded_gate * folded_gate) + np.sum(folded_up * folded_up))
    captured = float(np.sum((folded_gate @ q_t.T) ** 2) + np.sum((folded_up @ q_t.T) ** 2))
    metrics = {
        "randomized_width": width,
        "power_iterations": int(power_iterations),
        "seed": int(seed),
        "sketch_singular_values": singular_values[:rank].tolist(),
        # These are algorithm-space diagnostics.  The primary bank metrics
        # below are recomputed after the emitted Q^T is rounded to FP32,
        # because that is the basis the C consumer actually multiplies.
        "algorithm_q64_joint_frobenius_energy": total_energy,
        "algorithm_q64_joint_basis_energy": captured,
        "algorithm_q64_joint_captured_energy_fraction": captured / total_energy if total_energy > 0.0 else None,
    }
    return q_t, metrics


def _random_basis(hidden: int, rank: int, seed: int) -> np.ndarray:
    if rank < 1 or rank > hidden:
        raise ValueError(f"rank must be in [1, hidden], got {rank} and {hidden}")
    rng = np.random.default_rng(int(seed))
    return _canonicalize_rows(_qr_sign_stable(rng.standard_normal((hidden, rank))).T)


def _basis_metrics(q_t: np.ndarray) -> dict[str, float]:
    gram_error = np.asarray(q_t @ q_t.T - np.eye(q_t.shape[0]), dtype=np.float64)
    return {
        "orthonormal_frobenius_error": float(np.linalg.norm(gram_error)),
        "orthonormal_max_abs_error": float(np.max(np.abs(gram_error))),
        "row_l2_min": float(np.min(np.linalg.norm(q_t, axis=1))),
        "row_l2_max": float(np.max(np.linalg.norm(q_t, axis=1))),
    }


def _image_metrics(image64: np.ndarray, image32: np.ndarray) -> dict[str, float]:
    error = np.asarray(image32.astype(np.float64) - image64, dtype=np.float64)
    denominator = float(np.linalg.norm(image64))
    return {
        "float64_l2": denominator,
        "cast_abs_max": float(np.max(np.abs(error))),
        "cast_l2": float(np.linalg.norm(error)),
        "cast_relative_l2": float(np.linalg.norm(error) / denominator) if denominator > 0.0 else 0.0,
        "finite": bool(np.isfinite(image32).all()),
    }


def _basis_cast_metrics(q64: np.ndarray, q32: np.ndarray) -> dict[str, float]:
    """Measure the actual Q64-to-emitted-Q32 basis perturbation."""

    q32_wide = np.asarray(q32, dtype=np.float64)
    error = np.asarray(q32_wide - q64, dtype=np.float64)
    denominator = float(np.linalg.norm(q32_wide))
    return {
        "q64_l2": float(np.linalg.norm(q64)),
        "q32_l2": denominator,
        "basis_cast_abs_max": float(np.max(np.abs(error))),
        "basis_cast_l2": float(np.linalg.norm(error)),
        "basis_cast_relative_l2": float(np.linalg.norm(error) / denominator) if denominator > 0.0 else 0.0,
    }


def _image_basis_cast_metrics(image_q64: np.ndarray, image_q32: np.ndarray) -> dict[str, float]:
    """Measure image movement caused only by emitting Q^T as FP32."""

    q32_wide = np.asarray(image_q32, dtype=np.float64)
    error = np.asarray(q32_wide - image_q64, dtype=np.float64)
    denominator = float(np.linalg.norm(q32_wide))
    return {
        "q64_image_l2": float(np.linalg.norm(image_q64)),
        "q32_image_l2": denominator,
        "basis_cast_abs_max": float(np.max(np.abs(error))),
        "basis_cast_l2": float(np.linalg.norm(error)),
        "basis_cast_relative_l2": float(np.linalg.norm(error) / denominator) if denominator > 0.0 else 0.0,
    }


def _records_for_arrays(gate: np.ndarray, up: np.ndarray, gamma: np.ndarray) -> dict[str, SourceArray]:
    records: dict[str, SourceArray] = {}
    for name, array in (("gate", gate), ("up", up), ("gamma", gamma)):
        arr = np.asarray(array, dtype="<f4", order="C")
        raw = arr.tobytes(order="C")
        records[name] = SourceArray(
            name=name,
            path="<in-memory>",
            resolved_path="<in-memory>",
            shape=tuple(int(v) for v in arr.shape),
            dtype="<f4",
            order="C",
            byte_count=len(raw),
            sha256=sha256_bytes(raw),
            finite=bool(np.isfinite(arr).all()),
            min_value=float(arr.min()),
            max_value=float(arr.max()),
            l2_norm=float(np.linalg.norm(arr.astype(np.float64))),
        )
    return records


def compile_banks(
    source: SourceWeights | None = None,
    *,
    gate: np.ndarray | None = None,
    up: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    layer: int = 16,
    rank: int = DEFAULT_RANK,
    oversample: int = DEFAULT_OVERSAMPLE,
    power_iterations: int = DEFAULT_POWER_ITERATIONS,
    seed: int = DEFAULT_SEED,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> CompiledBanks:
    """Compile source and rank-matched random banks from exact source arrays."""

    if source is not None and any(v is not None for v in (gate, up, gamma)):
        raise ValueError("provide either source or gate/up/gamma, not both")
    if source is None:
        if gate is None or up is None or gamma is None:
            raise ValueError("gate, up and gamma are all required without SourceWeights")
        gate, up, gamma = _validate_arrays(gate, up, gamma)
        source = SourceWeights(gate, up, gamma, _records_for_arrays(gate, up, gamma))
    gate, up, gamma = _validate_arrays(source.gate, source.up, source.gamma)
    h, d = gate.shape
    rank = int(rank)
    oversample = int(oversample)
    power_iterations = int(power_iterations)
    config = CompilerConfig(
        layer=int(layer),
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=int(seed),
        random_seed=int(random_seed),
    )

    # Widening the exact exported FP32 values before folding gamma makes the
    # source objective explicit. Runtime files are cast back to <f4 only after
    # the basis and derived images are complete.
    gamma64 = gamma.astype(np.float64)
    folded_gate = gate.astype(np.float64) * gamma64[None, :]
    folded_up = up.astype(np.float64) * gamma64[None, :]
    source_q_t, source_metrics = _randomized_source_basis(
        folded_gate,
        folded_up,
        rank=rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=int(seed),
    )
    random_q_t = _random_basis(d, rank, int(random_seed))

    # The C consumer receives the emitted FP32 Q^T.  Recompute GΓQ and UΓQ
    # from that exact emitted basis widened to float64; using Q64 here would
    # make the stored images describe a slightly different program than the
    # one the native driver executes.
    source_q_t32 = np.asarray(source_q_t, dtype="<f4", order="C")
    random_q_t32 = np.asarray(random_q_t, dtype="<f4", order="C")
    source_q_t32_wide = source_q_t32.astype(np.float64)
    random_q_t32_wide = random_q_t32.astype(np.float64)
    source_gate_q64 = folded_gate @ source_q_t.T
    source_up_q64 = folded_up @ source_q_t.T
    random_gate_q64 = folded_gate @ random_q_t.T
    random_up_q64 = folded_up @ random_q_t.T
    source_gate64 = folded_gate @ source_q_t32_wide.T
    source_up64 = folded_up @ source_q_t32_wide.T
    random_gate64 = folded_gate @ random_q_t32_wide.T
    random_up64 = folded_up @ random_q_t32_wide.T

    arrays = {
        "source": {
            "q_t": source_q_t32,
            "gate": np.asarray(source_gate64, dtype="<f4", order="C"),
            "up": np.asarray(source_up64, dtype="<f4", order="C"),
        },
        "matched_random": {
            "q_t": random_q_t32,
            "gate": np.asarray(random_gate64, dtype="<f4", order="C"),
            "up": np.asarray(random_up64, dtype="<f4", order="C"),
        },
    }
    source_joint_energy_q32 = float(
        np.sum((folded_gate @ source_q_t32_wide.T) ** 2)
        + np.sum((folded_up @ source_q_t32_wide.T) ** 2)
    )
    random_joint_energy_q32 = float(
        np.sum((folded_gate @ random_q_t32_wide.T) ** 2)
        + np.sum((folded_up @ random_q_t32_wide.T) ** 2)
    )
    total_joint_energy = float(np.sum(folded_gate * folded_gate) + np.sum(folded_up * folded_up))
    metrics: dict[str, Any] = {
        # Primary basis units are the actual emitted Q32 values.  The Q64
        # entries remain explicit algorithm diagnostics rather than silently
        # defining a different physical program.
        "source_basis": {
            **_basis_metrics(source_q_t32_wide),
            **source_metrics,
            "basis_dtype": "<f4",
            "joint_frobenius_energy": total_joint_energy,
            "joint_basis_energy": source_joint_energy_q32,
            "joint_captured_energy_fraction": source_joint_energy_q32 / total_joint_energy if total_joint_energy > 0.0 else None,
            "q64_basis": _basis_metrics(source_q_t),
            "basis_cast": _basis_cast_metrics(source_q_t, source_q_t32_wide),
        },
        "matched_random_basis": {
            **_basis_metrics(random_q_t32_wide),
            "basis_dtype": "<f4",
            "joint_frobenius_energy": total_joint_energy,
            "joint_basis_energy": random_joint_energy_q32,
            "joint_captured_energy_fraction": random_joint_energy_q32 / total_joint_energy if total_joint_energy > 0.0 else None,
            "q64_basis": _basis_metrics(random_q_t),
            "basis_cast": _basis_cast_metrics(random_q_t, random_q_t32_wide),
        },
        "source_folded_norms": {
            "gate_l2": float(np.linalg.norm(folded_gate)),
            "up_l2": float(np.linalg.norm(folded_up)),
            "gamma_l2": float(np.linalg.norm(gamma64)),
            "gamma_min": float(np.min(gamma64)),
            "gamma_max": float(np.max(gamma64)),
        },
        "images": {
            "source_gate": {
                **_image_metrics(source_gate64, arrays["source"]["gate"]),
                "basis_cast_difference": _image_basis_cast_metrics(source_gate_q64, source_gate64),
            },
            "source_up": {
                **_image_metrics(source_up64, arrays["source"]["up"]),
                "basis_cast_difference": _image_basis_cast_metrics(source_up_q64, source_up64),
            },
            "matched_random_gate": {
                **_image_metrics(random_gate64, arrays["matched_random"]["gate"]),
                "basis_cast_difference": _image_basis_cast_metrics(random_gate_q64, random_gate64),
            },
            "matched_random_up": {
                **_image_metrics(random_up64, arrays["matched_random"]["up"]),
                "basis_cast_difference": _image_basis_cast_metrics(random_up_q64, random_up64),
            },
        },
    }
    return CompiledBanks(config=config, source=source, banks=arrays, metrics=metrics)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_entry(path: Path, array: np.ndarray) -> dict[str, Any]:
    canonical = np.asarray(array, dtype="<f4", order="C")
    if not np.isfinite(canonical).all():
        raise ValueError(f"compiled bank {path.name} contains non-finite values")
    raw = canonical.tobytes(order="C")
    _atomic_write_bytes(path, raw)
    return {
        "file": path.name,
        "shape": [int(v) for v in canonical.shape],
        "dtype": "<f4",
        "order": "C",
        "byte_count": len(raw),
        "sha256": sha256_bytes(raw),
        "finite": bool(np.isfinite(canonical).all()),
    }


def write_banks(
    compiled: CompiledBanks,
    output_dir: str | os.PathLike[str],
    *,
    manifest_name: str = "bank_manifest.json",
) -> dict[str, Any]:
    """Write immutable factor bytes and a self-describing custody manifest."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # These names are the C driver's frozen ABI.  The shape/rank are carried
    # in the manifest rather than baked into the filename, so the same writer
    # remains useful for the small model-free tests while the real 64x1536
    # export still has stable names for the native consumer.
    names = {
        "source": {"q_t": "source-QT.f32", "gate": "source-GQ.f32", "up": "source-UQ.f32"},
        "matched_random": {"q_t": "random-QT.f32", "gate": "random-GQ.f32", "up": "random-UQ.f32"},
    }
    bank_entries: dict[str, dict[str, Any]] = {}
    for bank_name, entries in compiled.banks.items():
        bank_entries[bank_name] = {}
        for component, array in entries.items():
            if bank_name not in names or component not in names[bank_name]:
                raise ValueError(f"unknown bank/component {bank_name}/{component}")
            bank_entries[bank_name][component] = _array_entry(output / names[bank_name][component], array)

    compiler_source = Path(__file__).read_bytes()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "profile": {
            "architecture": "Qwen2.5-1.5B-Instruct",
            "layer": compiled.config.layer,
            "hidden": int(compiled.source.hidden),
            "intermediate": int(compiled.source.intermediate),
            "rank": compiled.config.rank,
            "basis_coordinate": "raw_receiver_r",
            "c_weight_layout": "row-major out_by_in; little-endian FP32",
        },
        "compiler": _jsonable(asdict(compiled.config)) | {
            "module": Path(__file__).name,
            "module_sha256": sha256_bytes(compiler_source),
            "runtime_formula": {
                "z": "Q_T @ r",
                "gate": "rho(r) * (G @ diag(gamma) @ Q) @ z",
                "up": "rho(r) * (U @ diag(gamma) @ Q) @ z",
            },
        },
        "source": {
            "manifest_path": compiled.source.manifest_path,
            "weights": {name: record.as_dict() for name, record in compiled.source.records.items()},
            "future_activation_inputs": False,
            "target_outcome_inputs": False,
            "source_weights_only": True,
            "folding": "A=G@diag(gamma), B=U@diag(gamma), computed in float64 from exact FP32 source values",
        },
        "banks": bank_entries,
        "metrics": _jsonable(compiled.metrics),
        "provenance": {
            "source_weights_only": True,
            "future_activation_inputs": False,
            "target_outcome_inputs": False,
            "compiled_before_primary_trajectory": True,
        },
    }
    manifest_path = output / manifest_name
    _atomic_write_bytes(manifest_path, (json.dumps(_jsonable(manifest), sort_keys=True, indent=2) + "\n").encode("utf-8"))
    manifest["manifest_file"] = manifest_path.name
    manifest["manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())
    # The returned object includes the digest of the bytes that were actually
    # written.  The digest is intentionally not inserted into the file itself,
    # avoiding a self-referential manifest.
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="JSON manifest containing gate/up/gamma paths")
    parser.add_argument("--input-dir", "--input", dest="input_dir", help="directory containing input-manifest.json and G/U/gamma.f32")
    parser.add_argument("--gate", help="raw little-endian FP32 gate matrix")
    parser.add_argument("--up", help="raw little-endian FP32 up matrix")
    parser.add_argument("--gamma", help="raw little-endian FP32 post-attention scale")
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--power-iterations", type=int, default=DEFAULT_POWER_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--manifest-name", default="bank_manifest.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = args.manifest
    gate_path, up_path, gamma_path = args.gate, args.up, args.gamma
    if args.input_dir is not None:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if any(value is not None for value in (manifest_path, gate_path, up_path, gamma_path)):
            raise ValueError("--input-dir cannot be combined with --manifest or direct tensor paths")
        manifest_path = str(input_dir / "input-manifest.json")
    source = load_source_weights(
        manifest=manifest_path,
        gate=gate_path,
        up=up_path,
        gamma=gamma_path,
        hidden=args.hidden,
        intermediate=args.intermediate,
    )
    compiled = compile_banks(
        source,
        layer=args.layer,
        rank=args.rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        seed=args.seed,
        random_seed=args.random_seed,
    )
    manifest = write_banks(compiled, args.output_dir, manifest_name=args.manifest_name)
    summary = {
        "schema": SCHEMA,
        "manifest": str(Path(args.output_dir).expanduser().resolve() / args.manifest_name),
        "manifest_sha256": manifest["manifest_sha256"],
        "profile": manifest["profile"],
        "source_sha256": {name: record.sha256 for name, record in source.records.items()},
        "bank_sha256": {
            bank: {component: entry["sha256"] for component, entry in entries.items()}
            for bank, entries in manifest["banks"].items()
        },
        "provenance": manifest["provenance"],
    }
    print(json.dumps(_jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
