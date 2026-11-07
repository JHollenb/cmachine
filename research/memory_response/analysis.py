"""Offline analysis for the bounded-memory response experiment.

The runner owns model execution.  This module consumes only the runner's sealed
manifest and raw arrays and returns descriptive observations.  It deliberately
does not decide whether a policy is good: native fidelity, truth, own-history
behaviour, and mechanics/work are separate records.

The small input contract is intentionally explicit.  ``driver`` may contain
``cases`` or ``arms``.  Each case has ``task``, ``policy`` and optional
``binding``, ``corner``, ``question`` and ``phase`` fields, plus:

* ``consumed_ids`` and ``answer_ids`` (or their ``*_file`` descriptors);
* ``queries``: nested independent query forks, each with its own input and
  generated IDs and ``readouts``; a path-level ``parent_history_ids`` and
  ``prefix_boundaries`` remain separate from those query outcomes;
* ``readouts``: ``[{clock, position, logits, top}]``; and
* ``attention``/``attention_rows``: rows with ``layer``, ``position``,
  ``native_weights``, ``policy_weights``, ``native_values`` and optionally
  ``policy_values`` and ``selected_indices``.

Array descriptors are either inline arrays, ``{"file": ..., "shape": ...}``,
or a relative filename.  The implementation also accepts the equivalent
``arrays``/``files`` aliases used by earlier C drivers.  The output retains
logical filenames only, so the derived report is relocatable.

The important domain distinction is ``same_id`` versus ``own_history``.  A
candidate's logits may be compared with native at the same position only while
the consumed token histories agree.  After the first token difference the row
is retained as an own-history trajectory observation and is never presented as
a native-fidelity comparison.  A runner-supplied forced-native-history replay
may be labelled separately as ``forced_same_id``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POLICIES = (
    "native",
    "byte_full",
    "byte_top4",
    "byte_recent4",
    "history_deleted",
)

POLICY_ALIASES = {
    "native_full": "native",
    "bytefull": "byte_full",
    "top4": "byte_top4",
    "byte_top4": "byte_top4",
    "recent4": "byte_recent4",
    "byte_recent4": "byte_recent4",
    "history_deleted": "history_deleted",
    "current_only": "history_deleted",
    "deletion": "history_deleted",
}
POLICY_INDEX = {
    0: "native",
    1: "byte_full",
    2: "byte_top4",
    3: "byte_recent4",
    4: "history_deleted",
}


def canonical_policy(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        if int(value) not in POLICY_INDEX:
            raise ValueError(f"unknown memory-response policy index: {value}")
        return POLICY_INDEX[int(value)]
    value = str(value)
    return POLICY_ALIASES.get(value, value)


def _finite(a: Any, *, dtype: Any = float) -> np.ndarray:
    out = np.asarray(a, dtype=dtype)
    if not np.isfinite(out).all():
        raise ValueError("non-finite array")
    return out


def relative_l2(a: Any, b: Any, *, floor: float = 1e-30) -> float:
    """Relative L2 distance, with a finite zero-denominator convention."""

    aa = _finite(a, dtype=float)
    bb = _finite(b, dtype=float)
    den = max(float(np.linalg.norm(bb.ravel())), floor)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def norm_ratio(a: Any, reference: Any, *, floor: float = 1e-30) -> float:
    """Norm of ``a`` divided by the norm of ``reference``.

    This is intentionally separate from :func:`relative_l2`, whose numerator
    is the distance ``a-b``.  Factor terms and signed attention contributions
    need the former quantity; fidelity comparisons need the latter.
    """

    aa = _finite(a, dtype=float)
    rr = _finite(reference, dtype=float)
    return float(np.linalg.norm(aa.ravel()) / max(float(np.linalg.norm(rr.ravel())), floor))


def signed_stats(x: Any) -> dict[str, Any]:
    """Finite signed distribution summary; signs are never discarded."""

    a = _finite(x, dtype=float).ravel()
    if not a.size:
        return {
            "n": 0,
            "sum": 0.0,
            "l1": 0.0,
            "mean": 0.0,
            "min": None,
            "max": None,
        }
    return {
        "n": int(a.size),
        "sum": float(a.sum()),
        "l1": float(np.abs(a).sum()),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = _finite(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_output(weights: Any, values: Any) -> np.ndarray:
    """Apply attention weights to values for one or many heads.

    Accepted values are ``[N,D]`` or ``[H,N,D]`` and weights are ``[N]`` or
    ``[H,N]``.  The return is ``[D]`` or ``[H,D]`` respectively.
    """

    p = _finite(weights, dtype=float)
    v = _finite(values, dtype=float)
    if p.ndim == 1:
        if v.ndim != 2 or p.shape[0] != v.shape[0]:
            raise ValueError("weight/value shape mismatch")
        return p @ v
    if p.ndim != 2:
        raise ValueError("weights must be [N] or [H,N]")
    if v.ndim == 2:
        v = np.broadcast_to(v[None, ...], (p.shape[0],) + v.shape)
    if v.ndim != 3 or v.shape[:2] != p.shape:
        raise ValueError("head weight/value shape mismatch")
    return np.einsum("hn,hnd->hd", p, v, optimize=False)


def _canonical_kv(values: Any, groups: int | None = None) -> np.ndarray:
    """Return grouped K/V rows as ``[G,N,D]``.

    The C traces have used both ``[G,N,D]`` and ``[N,G,D]`` at different
    boundaries.  A declared ``groups`` removes ambiguity; otherwise the
    smaller of the first two axes is treated as the group axis.
    """

    a = _finite(values, dtype=float)
    if a.ndim != 3:
        raise ValueError("key/value rows must be three dimensional")
    if groups is not None:
        if a.shape[0] == groups:
            return a
        if a.shape[1] == groups:
            return np.moveaxis(a, 1, 0)
    if a.shape[0] <= a.shape[1]:
        return a
    return np.moveaxis(a, 1, 0)


def attention_read(query: Any, keys: Any, values: Any, support: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute a GQA attention read and return ``(output, full_weights)``.

    ``query`` is ``[H,D]``.  Keys/values may be grouped ``[G,N,D]`` or
    position-major ``[N,G,D]``.  ``support`` is either a global row index list
    or one index list per query head.  The returned weights are always expanded
    over all ``N`` rows, which makes signed tail analysis unambiguous.
    """

    q = _finite(query, dtype=float)
    if q.ndim != 2:
        raise ValueError("query must be [H,D]")
    k = _canonical_kv(keys)
    v = _canonical_kv(values, groups=k.shape[0])
    if k.shape != v.shape or k.shape[2] != q.shape[1]:
        raise ValueError("query/key/value dimensions differ")
    groups, rows, _ = k.shape
    if q.shape[0] % groups:
        raise ValueError("query heads are not divisible by KV groups")
    group_for_head = np.arange(q.shape[0]) * groups // q.shape[0]
    scores = np.einsum("hd,hnd->hn", q, k[group_for_head], optimize=False) / np.sqrt(float(q.shape[1]))
    expanded = np.zeros_like(scores)
    outputs = np.zeros((q.shape[0], q.shape[1]), dtype=float)
    if support is None:
        support_rows = [np.arange(rows, dtype=int)] * q.shape[0]
    elif isinstance(support, (list, tuple)) and support and isinstance(support[0], (list, tuple, np.ndarray)):
        support_rows = [np.asarray(x, dtype=int) for x in support]
    else:
        support_rows = [np.asarray(support, dtype=int)] * q.shape[0]
    for h, indices in enumerate(support_rows):
        if np.any(indices < 0) or np.any(indices >= rows) or len(np.unique(indices)) != len(indices):
            raise ValueError("attention support contains an invalid row")
        if not len(indices):
            raise ValueError("attention support is empty")
        p = _stable_softmax(scores[h, indices])
        expanded[h, indices] = p
        outputs[h] = p @ v[group_for_head[h], indices]
    return outputs, expanded


def _top_support(scores: np.ndarray, size: int) -> list[np.ndarray]:
    if scores.ndim != 2 or size <= 0:
        raise ValueError("top support expects [H,N] and positive size")
    return [np.argsort(-scores[h], kind="stable")[: min(size, scores.shape[1])] for h in range(scores.shape[0])]


def response_family(
    query: Any,
    native_keys: Any,
    native_values: Any,
    byte_keys: Any,
    byte_values: Any,
    *,
    top_size: int = 4,
    recent_size: int = 4,
    active_start: int | None = None,
) -> dict[str, Any]:
    """Evaluate the audit-only native/byte/full/top4/recent4 response family.

    This is conditional on the recorded query and row history.  It does not
    stand in for an own-history model transition.  Top-four selection uses the
    byte-full score, while recent-four retains the final rows; both return full
    expanded weight vectors for signed tail accounting.
    """

    q = _finite(query, dtype=float)
    nk = _canonical_kv(native_keys)
    nv = _canonical_kv(native_values, groups=nk.shape[0])
    bk = _canonical_kv(byte_keys, groups=nk.shape[0])
    bv = _canonical_kv(byte_values, groups=nk.shape[0])
    native_y, native_p = attention_read(q, nk, nv)
    byte_y, byte_p = attention_read(q, bk, bv)
    # Recover deterministic top support from byte scores.  We intentionally
    # use the same GQA grouping as attention_read rather than token-ID groups.
    groups, rows, dim = bk.shape
    heads = q.shape[0]
    group_for_head = np.arange(heads) * groups // heads
    byte_scores = np.einsum("hd,hnd->hn", q, bk[group_for_head], optimize=False) / np.sqrt(float(dim))
    top_support = _top_support(byte_scores, top_size)
    recent = [np.arange(max(0, rows - recent_size), rows, dtype=int)] * heads
    top_y, top_p = attention_read(q, bk, bv, top_support)
    recent_y, recent_p = attention_read(q, bk, bv, recent)
    outputs = {
        "native_full": (native_y, native_p, nk, nv),
        "byte_full": (byte_y, byte_p, bk, bv),
        "byte_top4": (top_y, top_p, bk, bv),
        "byte_recent4": (recent_y, recent_p, bk, bv),
    }
    result: dict[str, Any] = {
        "top_support": [x.tolist() for x in top_support],
        "recent_support": [x.tolist() for x in recent],
        "responses": {},
    }
    native_head_values = nv[group_for_head]
    byte_head_values = bv[group_for_head]
    for name, (out, weights, keys, values) in outputs.items():
        result["responses"][name] = {
            "output": out,
            "weights": weights,
            "output_relative_to_native": relative_l2(out, native_y),
        }
        if name != "native_full":
            result["responses"][name]["tail"] = local_tail_identity(
                native_p, weights, native_head_values,
                byte_head_values if name != "native_full" else native_head_values,
                result["top_support"] if name == "byte_top4" else result["recent_support"] if name == "byte_recent4" else None,
            )
            if name in ("byte_top4", "byte_recent4"):
                # Selection must be decomposed on the same decoded bank.  The
                # native-reference tail above includes codec and selection
                # together; this second identity isolates support choice.
                support = result["top_support"] if name == "byte_top4" else result["recent_support"]
                result["responses"][name]["same_byte_bank_tail"] = local_tail_identity(
                    byte_p,
                    weights,
                    byte_head_values,
                    byte_head_values,
                    support,
                )
    if active_start is not None:
        start = int(active_start)
        if start < 0 or start >= rows:
            raise ValueError("active_start must select a nonempty native suffix")
        suffix = [np.arange(start, rows, dtype=int)] * heads
        suffix_y, suffix_p = attention_read(q, nk, nv, suffix)
        result["responses"]["history_deleted"] = {
            "output": suffix_y,
            "weights": suffix_p,
            "output_relative_to_native": relative_l2(suffix_y, native_y),
            "tail": local_tail_identity(
                native_p,
                suffix_p,
                native_head_values,
                native_head_values,
                suffix,
            ),
        }
    return result


def _expand_weights(weights: Any, selected: Sequence[int] | None, n: int) -> np.ndarray:
    """Expand a compact support vector to the native key-row domain."""

    p = _finite(weights, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
        squeeze = True
    elif p.ndim == 2:
        squeeze = False
    else:
        raise ValueError("weights must be one or two dimensional")
    if p.shape[1] == n:
        out = p
    else:
        out = np.zeros((p.shape[0], n), dtype=float)
        nested = isinstance(selected, (list, tuple)) and selected and isinstance(selected[0], (list, tuple, np.ndarray))
        if nested:
            if len(selected) != p.shape[0]:
                raise ValueError("per-head selected_indices count differs")
            for h, raw_idx in enumerate(selected):
                idx = np.asarray(raw_idx, dtype=int)
                if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx) or len(idx) != p.shape[1]:
                    raise ValueError("invalid per-head selected_indices")
                out[h, idx] = p[h]
        else:
            if selected is None or len(selected) != p.shape[1]:
                raise ValueError("compact weights need selected_indices")
            idx = np.asarray(selected, dtype=int)
            if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx):
                raise ValueError("invalid selected_indices")
            out[:, idx] = p
    return out[0] if squeeze else out


def _expand_values(values: Any, selected: Sequence[int] | None, n: int, heads: int) -> np.ndarray:
    """Expand compact ``[K,D]``/``[H,K,D]`` values to ``[H,N,D]``."""

    v = _finite(values, dtype=float)
    if v.ndim == 2:
        v = v[None, ...]
    if v.ndim != 3:
        raise ValueError("values must be [N,D] or [H,N,D]")
    if v.shape[0] == 1 and heads != 1:
        v = np.broadcast_to(v, (heads,) + v.shape[1:])
    if v.shape[0] != heads:
        raise ValueError("head value count differs")
    if v.shape[1] == n:
        return v
    out = np.zeros((heads, n, v.shape[2]), dtype=float)
    nested = isinstance(selected, (list, tuple)) and selected and isinstance(selected[0], (list, tuple, np.ndarray))
    if nested:
        if len(selected) != heads:
            raise ValueError("per-head selected_indices count differs")
        for h, raw_idx in enumerate(selected):
            idx = np.asarray(raw_idx, dtype=int)
            if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx) or len(idx) != v.shape[1]:
                raise ValueError("invalid per-head selected_indices")
            out[h, idx] = v[h]
    else:
        if selected is None or len(selected) != v.shape[1]:
            raise ValueError("compact values need selected_indices")
        idx = np.asarray(selected, dtype=int)
        if np.any(idx < 0) or np.any(idx >= n) or len(np.unique(idx)) != len(idx):
            raise ValueError("invalid selected_indices")
        out[:, idx] = v
    return out


def local_tail_identity(
    native_weights: Any,
    policy_weights: Any,
    native_values: Any,
    policy_values: Any | None = None,
    selected_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Signed attention-tail identity for one local read.

    With native ``p_N,V_N`` and policy ``p_P,V_P`` this reports

    ``Y_P-Y_N = sum((p_P-p_N)V_N) + sum(p_P(V_P-V_N))``.

    The first term is the address/weight term and the second is the value term.
    The tail fields use the complement of ``selected_indices`` and are signed;
    they are not absolute-mass gates.  For a candidate that retains only a
    support, ``policy_weights`` can be given on the compact support.
    """

    pn = _finite(native_weights, dtype=float)
    pp0 = _finite(policy_weights, dtype=float)
    vn = _finite(native_values, dtype=float)
    vp = vn if policy_values is None else _finite(policy_values, dtype=float)
    if pn.ndim == 1:
        pn = pn[None, :]
        squeeze = True
    elif pn.ndim == 2:
        squeeze = False
    else:
        raise ValueError("native weights must be [N] or [H,N]")
    n = pn.shape[1]
    pp = _expand_weights(pp0, selected_indices, n)
    if pp.ndim == 1:
        pp = pp[None, :]
    vn = _expand_values(vn, None, n, pn.shape[0])
    vp = _expand_values(vp, selected_indices, n, pn.shape[0])
    if vn.shape[:2] != pn.shape or vp.shape[:2] != pn.shape:
        raise ValueError("attention value/weight shape mismatch")
    if vn.shape != vp.shape:
        raise ValueError("native/policy value shape mismatch")

    delta_p = pp - pn
    address = np.einsum("hn,hnd->hd", delta_p, vn, optimize=False)
    value = np.einsum("hn,hnd->hd", pp, vp - vn, optimize=False)
    native_y = np.einsum("hn,hnd->hd", pn, vn, optimize=False)
    policy_y = np.einsum("hn,hnd->hd", pp, vp, optimize=False)
    delta_y = policy_y - native_y
    residual = delta_y - address - value
    if selected_indices is None:
        support = np.abs(pp) > 0
    else:
        nested = isinstance(selected_indices, (list, tuple)) and selected_indices and isinstance(selected_indices[0], (list, tuple, np.ndarray))
        support = np.zeros((pn.shape[0], n), dtype=bool)
        if nested:
            if len(selected_indices) != pn.shape[0]:
                raise ValueError("per-head selected_indices count differs")
            for h, raw_idx in enumerate(selected_indices):
                support[h, np.asarray(raw_idx, dtype=int)] = True
        else:
            support[:, np.asarray(selected_indices, dtype=int)] = True
    tail = ~support
    out = {
        "heads": int(pn.shape[0]),
        "rows": int(n),
        "selected_rows": int(support.sum()),
        "tail_rows": int(tail.sum()),
        "native_weight_sum": signed_stats(pn.sum(axis=1) - 1.0),
        "policy_weight_sum": signed_stats(pp.sum(axis=1) - 1.0),
        "signed_tail_mass_per_head": np.where(tail, delta_p, 0.0).sum(axis=1).tolist(),
        "native_tail_mass_per_head": np.where(tail, pn, 0.0).sum(axis=1).tolist(),
        "policy_tail_mass_per_head": np.where(tail, pp, 0.0).sum(axis=1).tolist(),
        "signed_tail_mass": signed_stats(delta_p[tail]),
        "signed_tail_value_contribution": signed_stats(
            np.einsum("hn,hnd->hd", np.where(tail, delta_p, 0.0), vn, optimize=False)
        ),
        "address_term": {
            "l2": float(np.linalg.norm(address)),
            "relative_to_native": norm_ratio(address, native_y),
            "signed": signed_stats(address),
        },
        "value_term": {
            "l2": float(np.linalg.norm(value)),
            "relative_to_native": norm_ratio(value, native_y),
            "signed": signed_stats(value),
        },
        "output_delta": {
            "l2": float(np.linalg.norm(delta_y)),
            "relative_to_native": norm_ratio(delta_y, native_y),
            "signed": signed_stats(delta_y),
        },
        "identity_residual": {
            "l2": float(np.linalg.norm(residual)),
            "relative_to_native": norm_ratio(residual, native_y),
        },
    }
    if squeeze:
        out["single_head_input"] = True
    return out


def policy_factorial(native: Any, address_only: Any, value_only: Any, both: Any) -> dict[str, Any]:
    """Signed four-arm factorial, retaining the address/value interaction."""

    n, a, v, b = (_finite(x, dtype=float) for x in (native, address_only, value_only, both))
    da = a - n
    dv = v - n
    interaction = b - a - v + n
    joint = b - n
    den = max(float(np.linalg.norm(n.ravel())), 1e-30)
    return {
        "address_only_relative_l2": float(np.linalg.norm(da.ravel()) / den),
        "value_only_relative_l2": float(np.linalg.norm(dv.ravel()) / den),
        "both_relative_l2": float(np.linalg.norm(joint.ravel()) / den),
        "interaction_relative_l2": float(np.linalg.norm(interaction.ravel()) / den),
        "address_value_cross": float(2.0 * np.dot(da.ravel(), dv.ravel()) / den**2),
        "signed_interaction_sum": float(interaction.sum()),
        "closure_relative_l2": float(np.linalg.norm((joint - da - dv - interaction).ravel()) / den),
    }


def _relative_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return None
    if isinstance(value, Mapping):
        value = value.get("file", value.get("path", value.get("name")))
    return None if value is None else str(value)


def _logical_path(raw: Path, value: Any) -> str | None:
    """Return a relocation-safe name for a raw array descriptor.

    The worker is allowed to record a host path in an executed driver, but a
    saved analyzer result must compare identically after the capsule is moved
    to another checkout.  Prefer a path relative to the raw root and fall
    back to its basename when an external absolute path was supplied.
    """

    name = _relative_path(value)
    if name is None:
        return None
    path = Path(name)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(raw).as_posix()
    except ValueError:
        return path.name


def _load_array(raw: Path, value: Any, *, dtype: str = "<f4", shape: Sequence[int] | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return _finite(value, dtype=float)
    if isinstance(value, (list, tuple)):
        return _finite(value, dtype=float)
    if isinstance(value, Mapping):
        if "values" in value:
            return _finite(value["values"], dtype=float)
        dtype = value.get("dtype", dtype)
        shape = value.get("shape", shape)
        value = value.get("file", value.get("path", value.get("name")))
    if value is None:
        raise ValueError("missing array descriptor")
    path = raw / str(value)
    data = np.fromfile(path, dtype=dtype)
    if shape is not None:
        expected = int(np.prod(tuple(shape)))
        if data.size != expected:
            raise ValueError(f"array size differs for {value}: {data.size} != {expected}")
        data = data.reshape(tuple(shape))
    return _finite(data, dtype=float)


def _scalar_or_array(raw: Path, case: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in case:
            value = case[name]
            if isinstance(value, (str, Mapping)) and (name.endswith("_file") or name.endswith("_path")):
                return _load_array(raw, value)
            return value
    arrays = case.get("arrays", {})
    for name in names:
        if name in arrays:
            return _load_array(raw, arrays[name])
    return default


def _case_list(driver: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = driver.get("cases", driver.get("arms", driver.get("paths", driver.get("records", []))))

    def expand(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        row = dict(item)
        for key in ("questions", "queries", "question_forks", "forks", "branches"):
            children = row.get(key)
            if isinstance(children, list) and children and all(isinstance(x, Mapping) for x in children):
                parent = {k: v for k, v in row.items() if k != key}
                result: list[dict[str, Any]] = []
                for child in children:
                    merged = dict(parent)
                    merged.update(child)
                    result.extend(expand(merged))
                return result
        return [row]

    if isinstance(source, Mapping):
        result: list[dict[str, Any]] = []
        for key, value in source.items():
            if isinstance(value, Mapping) and "policy" not in value:
                for policy, item in value.items():
                    if isinstance(item, Mapping):
                        row = dict(item)
                        row.setdefault("task", key)
                        row.setdefault("policy", policy)
                        result.extend(expand(row))
            elif isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("task", key)
                result.extend(expand(row))
        return result
    result: list[dict[str, Any]] = []
    for item in source:
        if isinstance(item, Mapping):
            result.extend(expand(item))
    return result


def _field(case: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in case:
            return case[name]
    return default


def _readouts(raw: Path, case: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = case.get("readouts", case.get("clocks", []))
    if isinstance(source, Mapping):
        source = [dict(v, clock=k) if isinstance(v, Mapping) else {"clock": k, "logits": v} for k, v in source.items()]
    result = [dict(x) for x in source]
    if not result and isinstance(case.get("turns"), list):
        # C drivers commonly put boundary metadata under turns while saving
        # the full vectors using ``<name>-t<turn>-<stage>-logits.f32``.
        name = case.get("name", case.get("case_id"))
        for turn in case["turns"]:
            if not isinstance(turn, Mapping):
                continue
            turn_id = turn.get("turn", turn.get("index", 0))
            for stage, boundary in turn.items():
                if stage in ("turn", "index", "answer_ids", "steps", "generated_ids") or not isinstance(boundary, Mapping):
                    continue
                row = dict(boundary)
                row.setdefault("clock", stage)
                row.setdefault("position", boundary.get("position"))
                if "logits" not in row and "logits_file" not in row and name is not None:
                    inferred = f"{name}-t{turn_id}-{stage}-logits.f32"
                    if (raw / inferred).exists():
                        row["logits"] = inferred
                result.append(row)
    if not result:
        logits = _field(case, "logits", "logits_file", "readout_logits")
        if logits is not None:
            result = [{"clock": _field(case, "clock", default="final"), "position": _field(case, "position"), "logits": logits}]
    for row in result:
        if "logits" not in row:
            row["logits"] = row.get("logits_file", row.get("file"))
        if row.get("logits") is not None:
            descriptor = row["logits"]
            # A boundary may carry a digest rather than a filename.  Such a
            # row remains in the driver but cannot be used for full-vocabulary
            # comparison until its declared vector is present.
            path_name = _relative_path(descriptor)
            inline = isinstance(descriptor, (list, tuple, np.ndarray)) or isinstance(descriptor, Mapping) and "values" in descriptor
            if inline or path_name is not None and (raw / path_name).exists():
                row["_logits_array"] = _load_array(raw, descriptor)
        if "top" not in row and row.get("_logits_array") is not None:
            row["top"] = int(np.argmax(row["_logits_array"]))
    return result


def _parent_history(case: Mapping[str, Any], raw: Path) -> list[int] | None:
    """Load the path-level history that precedes an independent query fork."""

    value = case.get("parent_history_ids")
    if value is None:
        value = case.get("parent_history_ids_file")
    if value is None:
        return None
    if isinstance(value, (str, Mapping)):
        value = _load_array(raw, value, dtype="<i4").astype(int).tolist()
    return [int(x) for x in value]


def _declared_digest(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def _validate_history_digest(
    ids: Sequence[int],
    declared: Any,
    *,
    position: int | None,
    context: str,
) -> dict[str, Any] | None:
    """Validate an executed prefix digest and retain the authority record.

    A declared hash is an execution/mechanics contract.  It is therefore
    strict when present, while absent hashes remain an explicit limitation of
    older fixtures.  The digest covers little-endian int32 token IDs, which
    is the C driver's ``IdHistory`` representation.
    """

    expected = _declared_digest(declared)
    if expected is None:
        return None
    if position is None:
        raise ValueError(f"{context}: history hash has no position")
    position = int(position)
    if position < 0 or position > len(ids):
        raise ValueError(f"{context}: history position {position} exceeds {len(ids)} IDs")
    actual = _history_digest(ids[:position])
    if actual != expected:
        raise ValueError(f"{context}: history digest differs ({actual} != {expected})")
    return {"sha256": expected, "position": position, "verified": True}


def _prefix_boundaries(
    raw: Path,
    case: Mapping[str, Any],
    *,
    case_id: str,
    task: str,
    policy: str,
    parent_ids: Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Project C prefix checkpoints without turning them into query answers."""

    source = case.get("prefix_boundaries", case.get("prefixes", []))
    if isinstance(source, Mapping):
        source = [dict(v, kind=k) if isinstance(v, Mapping) else {"kind": k, "logits": v} for k, v in source.items()]
    if not isinstance(source, list):
        return []
    rows: list[dict[str, Any]] = []
    for boundary in source:
        if not isinstance(boundary, Mapping):
            continue
        item = dict(boundary)
        position = item.get("position")
        position = None if position is None else int(position)
        history_check = None
        if item.get("history_sha256") is not None:
            if parent_ids is None:
                raise ValueError(f"{case_id}: prefix history hash has no parent_history_ids")
            history_check = _validate_history_digest(
                parent_ids,
                item.get("history_sha256"),
                position=position,
                context=f"{case_id}/{item.get('kind', 'prefix')}",
            )
        logits = item.get("logits", item.get("logits_file", item.get("file")))
        row = {
            "case_id": case_id,
            "task": task,
            "policy": policy,
            "kind": item.get("kind", item.get("clock", "prefix")),
            "position": position,
            "pending": item.get("pending"),
            "top": item.get("top"),
            "runner_up": item.get("runner_up"),
            "top_margin": item.get("top_margin"),
            "history_sha256": _declared_digest(item.get("history_sha256")),
            "history_check": history_check,
            "cache_sha256": _declared_digest(item.get("cache_sha256")),
            "logits_file": _logical_path(raw, logits),
            "logits_sha256": _declared_digest(
                item.get("logits_sha256")
                or (logits.get("sha256") if isinstance(logits, Mapping) else None)
            ),
        }
        rows.append(row)
    return rows


def _actual_weights_global(actual_weights: Any, rows: int, active_start: int) -> np.ndarray:
    """Place an active-window weight tensor back in the original row domain."""

    weights = _finite(actual_weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("actual_weights must have shape [heads, active_length]")
    if active_start < 0 or active_start > rows or active_start + weights.shape[1] > rows:
        raise ValueError("actual_weights window exceeds original key rows")
    out = np.zeros((weights.shape[0], rows), dtype=float)
    out[:, active_start : active_start + weights.shape[1]] = weights
    return out


def _pair_effect(left: Any, right: Any) -> dict[str, Any]:
    """Signed conditional effect with an explicit reference denominator."""

    return _metric(_finite(left, dtype=float), _finite(right, dtype=float))


def _flag_is_false(value: Any) -> bool:
    if value is False or (isinstance(value, (int, np.integer)) and int(value) == 0):
        return True
    return isinstance(value, str) and value.strip().lower() in {"false", "0", "no"}


def _structured_attention_packet(raw: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Analyze an own-query audit row without treating it as serving state.

    The primary runner supplies the actual context and weights.  The original
    and decoded banks are then replayed conditionally on that same captured
    query.  Differences among those conditional readers are codec/selection
    effects; actual-versus-conditional residuals are kept in a separate field
    because they include finite/runtime arithmetic drift.
    """

    def required(*names: str) -> Any:
        for name in names:
            if row.get(name) is not None:
                return row[name]
        raise ValueError("structured attention audit is missing " + "/".join(names))

    query = _load_array(raw, required("query", "query_file"))
    original_keys = _load_array(raw, required("original_keys", "native_keys", "keys"))
    original_values = _load_array(raw, required("original_values", "native_values", "values"))
    byte_keys = _load_array(raw, required("byte_keys", "decoded_byte_keys"))
    byte_values = _load_array(raw, required("byte_values", "decoded_byte_values"))
    actual_context = _load_array(raw, required("actual_context", "policy_context"))
    actual_weights = _load_array(raw, required("actual_weights", "policy_weights"))

    q = _finite(query, dtype=float)
    if q.ndim != 2:
        raise ValueError("structured attention query must have shape [heads, dim]")
    nk = _canonical_kv(original_keys)
    nv = _canonical_kv(original_values, groups=nk.shape[0])
    bk = _canonical_kv(byte_keys, groups=nk.shape[0])
    bv = _canonical_kv(byte_values, groups=nk.shape[0])
    if nk.shape != nv.shape or bk.shape != bv.shape:
        raise ValueError("structured attention key/value shapes differ")
    if nk.shape[2] != q.shape[1] or bk.shape[2] != q.shape[1]:
        raise ValueError("structured attention query/key dimensions differ")
    rows = int(nk.shape[1])
    declared_rows = row.get("key_count")
    if declared_rows is not None and int(declared_rows) != rows:
        raise ValueError(f"structured attention key_count differs: {declared_rows} != {rows}")
    active_start = int(row.get("active_start", 0))
    actual_global_weights = _actual_weights_global(actual_weights, rows, active_start)
    actual_context = _finite(actual_context, dtype=float)
    if actual_context.shape != (q.shape[0], q.shape[1]):
        raise ValueError("actual_context shape differs from query heads/dim")
    if actual_global_weights.shape[0] != q.shape[0]:
        raise ValueError("actual_weights head count differs from query")

    policy = canonical_policy(row.get("policy", "native"))
    # ``active_start`` on recent4 describes its physical ring window.  It
    # must never be reinterpreted as the history-deletion intervention.  Only
    # a history_deleted packet receives the deletion conditional reader.
    family = response_family(
        q,
        nk,
        nv,
        bk,
        bv,
        top_size=int(row.get("top_size", 4)),
        recent_size=int(row.get("recent_size", 4)),
        active_start=active_start if policy == "history_deleted" else None,
    )
    responses = family["responses"]
    expected_name = {
        "native": "native_full",
        "byte_full": "byte_full",
        "byte_top4": "byte_top4",
        "byte_recent4": "byte_recent4",
        "history_deleted": "history_deleted",
    }.get(policy)
    if expected_name is None:
        raise ValueError(f"structured attention has unknown policy: {policy}")
    expected = responses[expected_name]

    def response_effect(name: str, reference: str) -> dict[str, Any]:
        left = responses[name]
        right = responses[reference]
        return {
            "output": _pair_effect(left["output"], right["output"]),
            "weights": _pair_effect(left["weights"], right["weights"]),
            "tail_decomposition": left.get("same_byte_bank_tail", left.get("tail")),
            "left": name,
            "right": reference,
        }

    conditional_effects = {
        "codec_byte_full_minus_native": response_effect("byte_full", "native_full"),
        "selection_top4_minus_byte_full": response_effect("byte_top4", "byte_full"),
        "selection_recent4_minus_byte_full": response_effect("byte_recent4", "byte_full"),
    }
    if policy == "history_deleted" and "history_deleted" in responses:
        conditional_effects["history_deleted_minus_native"] = response_effect("history_deleted", "native_full")
    actual_vs_conditional = {
        "policy": policy,
        "expected_reader": expected_name,
        "context": _pair_effect(actual_context, expected["output"]),
        "weights": _pair_effect(actual_global_weights, expected["weights"]),
        "interpretation": "actual own-query context minus conditional replay; includes finite/runtime arithmetic drift",
    }
    codec_mismatch = policy in {"byte_full", "byte_top4", "byte_recent4"} and _flag_is_false(row.get("codec_audit_exact"))
    if codec_mismatch:
        # Keep the finite numbers as an instrument observation, but prevent a
        # byte-policy reconstruction mismatch from being reported as an
        # actual-policy error.  The primary runtime outcome remains valid.
        actual_vs_conditional.update(
            {
                "status": "not_applicable_codec_mismatch",
                "usable_for_actual_policy_claim": False,
                "diagnostic_limit": "captured byte rows do not match the runtime cache; conditional metrics are nonactual",
            }
        )
        codec_status = "instrument-broken"
    else:
        actual_vs_conditional.update({"status": "applicable", "usable_for_actual_policy_claim": True})
        codec_status = "applicable"
    return {
        "layer": row.get("layer"),
        "position": row.get("position"),
        "policy": policy,
        "key_count": rows,
        "active_start": active_start,
        "active_length": int(actual_weights.shape[1]),
        "top_support": family["top_support"],
        "recent_support": family["recent_support"],
        "codec_audit_exact": row.get("codec_audit_exact"),
        "codec_reconstruction_status": codec_status,
        "conditional_effects": conditional_effects,
        "actual_vs_conditional": actual_vs_conditional,
    }


def _attention_rows(raw: Path, case: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = None
    for key in ("attention", "attention_rows", "audit_attention", "attention_audit", "audit_rows"):
        value = case.get(key)
        if value:
            source = value
            break
    if source is None:
        source = []
    if isinstance(source, Mapping):
        if any(key in source for key in ("query", "query_file", "native_weights", "policy_weights", "native_keys", "keys")):
            source = [source]
        else:
            source = [dict(v, layer=k) if isinstance(v, Mapping) else {"layer": k, "native_weights": v} for k, v in source.items()]
    rows: list[dict[str, Any]] = []
    for item in source:
        row = dict(item)
        structured_fields = {
            "query", "query_file", "original_keys", "original_values",
            "byte_keys", "byte_values", "actual_context", "actual_weights",
        }
        if {"actual_context", "actual_weights"}.issubset(row) and structured_fields.intersection(row):
            # This is an own-query audit packet.  Its arrays are intentionally
            # consumed here and reduced to signed, relocatable metrics; they
            # never become serving state or a future-history claim.
            structured_row = dict(row)
            structured_row.setdefault("policy", case.get("policy", "native"))
            packet = _structured_attention_packet(raw, structured_row)
            rows.append(
                {
                    "audit_kind": "response_family",
                    "layer": row.get("layer"),
                    "position": row.get("position"),
                    "policy": canonical_policy(structured_row.get("policy", "native")),
                    "support_kind": "response_family",
                    "response_family": packet,
                }
            )
            continue
        # Audit-only conditional response packet.  The C driver may expose
        # original own-history rows and decoded byte rows rather than weights;
        # reconstruct all requested readers here after primary sealing.
        query_desc = row.get("query", row.get("query_file"))
        native_keys = row.get("native_keys", row.get("own_keys", row.get("keys")))
        native_values = row.get("native_values", row.get("own_values", row.get("values")))
        byte_keys = row.get("byte_keys", row.get("decoded_byte_keys"))
        byte_values = row.get("byte_values", row.get("decoded_byte_values"))
        if query_desc is not None and native_keys is not None and native_values is not None and byte_keys is not None and byte_values is not None:
            try:
                family = response_family(
                    _load_array(raw, query_desc),
                    _load_array(raw, native_keys),
                    _load_array(raw, native_values),
                    _load_array(raw, byte_keys),
                    _load_array(raw, byte_values),
                    top_size=int(row.get("top_size", 4)),
                    recent_size=int(row.get("recent_size", 4)),
                )
            except (OSError, ValueError):
                family = None
            if family is not None:
                for policy, payload in family["responses"].items():
                    if policy == "native_full":
                        continue
                    tail = payload.get("tail")
                    if tail is None:
                        continue
                    rows.append(
                        {
                            "layer": row.get("layer"),
                            "position": row.get("position"),
                            "support_kind": policy,
                            "policy": policy,
                            "precomputed_tail": tail,
                            "output_relative_to_native": payload["output_relative_to_native"],
                        }
                    )
                continue
        for key in (
            "native_weights",
            "policy_weights",
            "native_values",
            "policy_values",
            "selected_indices",
            "native_output",
            "policy_output",
        ):
            if key in row:
                value = row[key]
                if key == "selected_indices":
                    row["_" + key] = np.asarray(value, dtype=int)
                else:
                    row["_" + key] = _load_array(raw, value)
        if row.get("precomputed_tail") is not None:
            continue
        if "_native_weights" not in row or "_policy_weights" not in row or "_native_values" not in row:
            continue
        rows.append(row)
    return rows


def _history(case: Mapping[str, Any], raw: Path) -> list[int]:
    value = _field(case, "consumed_ids", "history_ids")
    if isinstance(value, (str, Mapping)):
        value = _load_array(raw, value, dtype="<i4").astype(int).tolist()
    if value is None:
        value = _field(case, "consumed_ids_file", "history_ids_file")
        if value is not None:
            value = _load_array(raw, value, dtype="<i4").astype(int).tolist()
    if value is None:
        parent = _field(case, "parent_history_ids", default=[])
        query = _field(case, "input_ids", "query_ids", default=[])
        answer = _field(case, "answer_ids", "generated_ids", default=[])
        if parent is not None or query is not None or answer is not None:
            value = list(parent or []) + list(query or []) + list(answer or [])
    return [] if value is None else [int(x) for x in value]


def _answers(case: Mapping[str, Any], raw: Path) -> list[int]:
    value = _field(case, "answer_ids", "generated_ids")
    if isinstance(value, (str, Mapping)):
        value = _load_array(raw, value, dtype="<i4").astype(int).tolist()
    if value is None:
        value = _field(case, "answer_ids_file", "generated_ids_file")
        if value is not None:
            value = _load_array(raw, value, dtype="<i4").astype(int).tolist()
    return [] if value is None else [int(x) for x in value]


def _case_id_value(case: Mapping[str, Any]) -> Any:
    """Return the frozen input case identity when one is declared."""

    value = _field(case, "input_case_id", "case_id", "case", "task_id", "task")
    if isinstance(value, str):
        match = re.search(r"(?:case|task)[_-]?(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        if value.isdigit():
            return int(value)
    return value


def _question_index(case: Mapping[str, Any]) -> int | None:
    value = _field(case, "question_index", "question_id", "query_id", "question", "question_name")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("mira", "q0", "question0", "question_0", "0"):
            return 0
        if lowered in ("niko", "q1", "question1", "question_1", "1"):
            return 1
        match = re.search(r"(?:question|query|q)[_-]?(\d+)", lowered)
        if match:
            return int(match.group(1))
    for key in ("name", "case_id", "id", "branch"):
        value = case.get(key)
        if value is not None:
            match = re.search(r"(?:question|query|q)[_-]?(\d+)", str(value).lower())
            if match:
                return int(match.group(1))
    return None


def _answer_text(case: Mapping[str, Any], texts: Mapping[str, Any]) -> str | None:
    """Find the worker's decoded answer without guessing a missing name."""

    if not isinstance(texts, Mapping):
        return None
    names = [case.get(key) for key in ("answer_name", "name", "case_id", "id", "readout_name")]
    for name in names:
        if name is None:
            continue
        value = texts.get(str(name), texts.get(name))
        if isinstance(value, str):
            return value
    return None


def _color_token(text: str | None) -> tuple[str | None, str]:
    """Parse exactly one red/blue reply, retaining format failures."""

    if text is None:
        return None, "missing_text"
    value = str(text).strip()
    # Tokenizer decoding may leave a terminal newline or punctuation.  Permit
    # punctuation explicitly, but reject extra words so a verbose answer is a
    # format miss rather than a lucky substring match.
    match = re.fullmatch(r"([A-Za-z]+)[\s.!?,;:]*", value)
    if not match:
        return None, "format_miss"
    color = match.group(1).lower()
    if color not in ("red", "blue"):
        return None, "format_miss"
    return color, "one_color"


def _manifest_truth(raw: Path) -> dict[Any, list[Any]]:
    """Read expected colors from the immutable runtime input manifest."""

    candidates = (raw / "inputs" / "runtime-inputs.json", raw / "runtime-inputs.json")
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        result = {}
        for item in data.get("cases", []):
            if "id" in item and "expected" in item:
                result[item["id"]] = list(item["expected"])
        return result
    return {}


def _manifest_cases(raw: Path) -> dict[Any, dict[str, Any]]:
    for path in (raw / "inputs" / "runtime-inputs.json", raw / "runtime-inputs.json"):
        if path.exists():
            data = json.loads(path.read_text())
            return {item["id"]: dict(item) for item in data.get("cases", []) if "id" in item}
    return {}


def _manifest_case(cases: Mapping[Any, Mapping[str, Any]], case_id: Any) -> dict[str, Any]:
    """Look up a frozen input row across JSON's string/int key boundary."""

    if case_id in cases:
        return dict(cases[case_id])
    for key, value in cases.items():
        if str(key) == str(case_id):
            return dict(value)
    return {}


def _manifest_color_variants(raw: Path) -> dict[str, dict[str, list[int]]]:
    """Read the frozen tokenizer surfaces without inventing token IDs."""

    for path in (raw / "inputs" / "runtime-inputs.json", raw / "runtime-inputs.json"):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        source = data.get("color_token_variants", {})
        result: dict[str, dict[str, list[int]]] = {}
        if not isinstance(source, Mapping):
            return result
        for color in ("red", "blue"):
            entries = source.get(color, {})
            if not isinstance(entries, Mapping):
                continue
            parsed: dict[str, list[int]] = {}
            for surface, value in entries.items():
                if isinstance(value, Mapping):
                    value = value.get("token_ids", value.get("ids", value.get("values")))
                if not isinstance(value, (list, tuple)):
                    continue
                try:
                    ids = [int(token) for token in value]
                except (TypeError, ValueError):
                    continue
                parsed[str(surface)] = ids
            result[color] = parsed
        return result
    return {}


def _surface_style(surface: str, color: str) -> str:
    """Name a token-surface class while retaining its literal surface below."""

    text = str(surface)
    core = text.strip()
    leading = text[:1].isspace()
    trailing = text[-1:].isspace()
    capitalized = bool(core) and core[0].isupper()
    # Keep punctuation/suffix distinctions explicit rather than silently
    # matching a tokenized phrase to a bare color.
    suffix = core[len(color) :] if core.lower().startswith(color) else core
    suffix_tag = "_suffix" if suffix else ""
    return f"{'leading' if leading else 'bare'}_{'capitalized' if capitalized else 'lower'}{'_trailing' if trailing else ''}{suffix_tag}"


def _color_margin(logits: Any, variants: Mapping[str, Mapping[str, Sequence[int]]]) -> dict[str, Any]:
    """Compute exact single-token red-minus-blue margins and retain surfaces.

    A readout is a vector at one clock.  Multi-token color surfaces are
    reported as sequences but deliberately receive no fabricated scalar
    margin; this keeps the primary bare-token measurement separate from
    leading-space/capitalized tokenizer variants.
    """

    arrays: list[dict[str, Any]] = []
    style_entries: dict[str, dict[str, dict[str, Any]]] = {}
    for color in ("red", "blue"):
        for surface, ids in (variants.get(color, {}) or {}).items():
            style = _surface_style(surface, color)
            item = {
                "color": color,
                "surface": surface,
                "token_ids": list(map(int, ids)),
                "style": style,
                "single_token": len(ids) == 1,
            }
            arrays.append(item)
            style_entries.setdefault(style, {})[color] = item

    def one_token_margin(red: Mapping[str, Any] | None, blue: Mapping[str, Any] | None) -> tuple[float | None, str]:
        if red is None or blue is None:
            return None, "missing_surface_pair"
        rids, bids = red["token_ids"], blue["token_ids"]
        if len(rids) != 1 or len(bids) != 1:
            return None, "multi_token_sequence"
        if logits is None:
            return None, "logits_unavailable"
        values = _finite(logits, dtype=float).ravel()
        if rids[0] < 0 or bids[0] < 0 or rids[0] >= values.size or bids[0] >= values.size:
            return None, "token_id_out_of_range"
        return float(values[rids[0]] - values[bids[0]]), "single_token_exact"

    red_bare = (variants.get("red", {}) or {}).get("red")
    blue_bare = (variants.get("blue", {}) or {}).get("blue")
    primary_value, primary_status = one_token_margin(
        {"token_ids": list(map(int, red_bare))} if red_bare is not None else None,
        {"token_ids": list(map(int, blue_bare))} if blue_bare is not None else None,
    )
    primary = {
        "red_surface": "red",
        "blue_surface": "blue",
        "red_token_ids": None if red_bare is None else list(map(int, red_bare)),
        "blue_token_ids": None if blue_bare is None else list(map(int, blue_bare)),
        "red_minus_blue": primary_value,
        "status": primary_status if variants else "manifest_unavailable",
    }
    pooled_ids = {
        color: sorted(
            {
                int(ids[0])
                for surface, ids in (variants.get(color, {}) or {}).items()
                if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)) and len(ids) == 1
            }
        )
        for color in ("red", "blue")
    }
    pooled_margin = None
    pooled_status = "manifest_unavailable" if not variants else "logits_unavailable"
    if logits is not None and pooled_ids["red"] and pooled_ids["blue"]:
        values = _finite(logits, dtype=float).ravel()
        if all(0 <= token < values.size for color in pooled_ids for token in pooled_ids[color]):
            red_values = values[pooled_ids["red"]]
            blue_values = values[pooled_ids["blue"]]
            red_max, blue_max = np.max(red_values), np.max(blue_values)
            red_lse = red_max + np.log(np.exp(red_values - red_max).sum())
            blue_lse = blue_max + np.log(np.exp(blue_values - blue_max).sum())
            pooled_margin = float(red_lse - blue_lse)
            pooled_status = "diagnostic_variant_pooled_single_token"
        else:
            pooled_status = "token_id_out_of_range"
    elif variants and (not pooled_ids["red"] or not pooled_ids["blue"]):
        pooled_status = "no_single_token_variant_pair"
    variant_margins: dict[str, Any] = {}
    for style, entries in sorted(style_entries.items()):
        margin, status = one_token_margin(entries.get("red"), entries.get("blue"))
        variant_margins[style] = {
            "red_surface": None if entries.get("red") is None else entries["red"]["surface"],
            "blue_surface": None if entries.get("blue") is None else entries["blue"]["surface"],
            "red_token_ids": None if entries.get("red") is None else entries["red"]["token_ids"],
            "blue_token_ids": None if entries.get("blue") is None else entries["blue"]["token_ids"],
            "red_minus_blue": margin,
            "status": status,
        }
    return {
        "primary": primary,
        "variant_pooled_single_token": {
            "red_token_ids": pooled_ids["red"],
            "blue_token_ids": pooled_ids["blue"],
            "red_minus_blue_logsumexp": pooled_margin,
            "status": pooled_status,
            "interpretation": "diagnostic semantic color mass; not an evaluator probability",
        },
        "surfaces": arrays,
        "variant_margins": variant_margins,
    }


def _history_digest(ids: Sequence[int]) -> str:
    values = np.asarray(list(map(int, ids)), dtype="<i4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _same_history(a: Sequence[int], b: Sequence[int], position: int | None = None) -> tuple[bool, int]:
    n = min(len(a), len(b)) if position is None else min(position, len(a), len(b))
    first = n
    for i in range(n):
        if int(a[i]) != int(b[i]):
            first = i
            break
    if position is None:
        return first == n and len(a) == len(b), first
    return first == n and len(a) >= position and len(b) >= position, first


def _metric(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = a - b
    return {
        "relative_l2": relative_l2(a, b),
        "l2": float(np.linalg.norm(d.ravel())),
        "signed": signed_stats(d),
        "shape": list(a.shape),
    }


def _corner_contrasts(readouts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute signed initial/update/interaction contrasts of color margins.

    The four corner values are retained verbatim.  Initial and update effects
    are balanced two-corner mean differences; the mixed term is the signed
    ``I=1,U=1 - I=1,U=0 - I=0,U=1 + I=0,U=0`` interaction.  These are
    descriptive policy-trajectory contrasts, not native-fidelity claims.
    """

    groups: dict[tuple[Any, Any, Any, Any], dict[tuple[int, int], dict[str, Any]]] = {}
    for row in readouts:
        margin = row.get("color_margin", {}).get("primary", {}).get("red_minus_blue")
        # Absolute positions may differ when an own policy emits a different
        # ACK/EOS length.  Keep that as a diagnostic while comparing the same
        # named boundary across the four frozen corners.
        key = (row.get("policy"), row.get("question_index"), row.get("clock"), row.get("phase"))
        initial, update = row.get("initial_bit"), row.get("update_bit")
        if initial not in (0, 1) or update not in (0, 1):
            continue
        groups.setdefault(key, {})[(int(initial), int(update))] = {
            "margin": margin,
            "case_id": row.get("case_id"),
            "position": row.get("position"),
        }
    result: list[dict[str, Any]] = []
    for key, corners in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        policy, question, clock, phase = key
        ordered = {f"i{i}_u{u}": corners.get((i, u), {}).get("margin") for i in (0, 1) for u in (0, 1)}
        positions = {f"i{i}_u{u}": corners.get((i, u), {}).get("position") for i in (0, 1) for u in (0, 1)}
        complete = all(value is not None for value in ordered.values())
        entry: dict[str, Any] = {
            "policy": policy,
            "question_index": question,
            "clock": clock,
            "phase": phase,
            "comparison_domain": "own_policy_trajectory",
            "corners": ordered,
            "corner_positions": positions,
            "same_absolute_position": len({value for value in positions.values() if value is not None}) <= 1,
            "status": "complete" if complete else "incomplete_color_margin_coverage",
        }
        if complete:
            i0 = (float(ordered["i0_u0"]) + float(ordered["i0_u1"])) / 2.0
            i1 = (float(ordered["i1_u0"]) + float(ordered["i1_u1"])) / 2.0
            u0 = (float(ordered["i0_u0"]) + float(ordered["i1_u0"])) / 2.0
            u1 = (float(ordered["i0_u1"]) + float(ordered["i1_u1"])) / 2.0
            entry["initial_edge_update0"] = float(ordered["i1_u0"]) - float(ordered["i0_u0"])
            entry["initial_edge_update1"] = float(ordered["i1_u1"]) - float(ordered["i0_u1"])
            entry["update_edge_initial0"] = float(ordered["i0_u1"]) - float(ordered["i0_u0"])
            entry["update_edge_initial1"] = float(ordered["i1_u1"]) - float(ordered["i1_u0"])
            entry["initial_signed_contrast"] = i1 - i0
            entry["update_signed_contrast"] = u1 - u0
            entry["mixed_signed_interaction"] = (
                float(ordered["i1_u1"])
                - float(ordered["i1_u0"])
                - float(ordered["i0_u1"])
                + float(ordered["i0_u0"])
            )
        result.append(entry)
    return result


def _truth_result(
    case: Mapping[str, Any],
    answers: Sequence[int],
    texts: Mapping[str, Any],
    raw: Path,
    manifest_truth: Mapping[Any, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    answer_text = _answer_text(case, texts)
    normalized, format_status = _color_token(answer_text)
    truth = _field(case, "truth", "task_truth", default=None)
    if truth is None:
        task = _case_id_value(case)
        truth = (manifest_truth or _manifest_truth(raw)).get(task)
    # A driver may carry the frozen expected color directly.  This is useful
    # for a saved-only verifier after runtime-inputs.json has been relocated.
    if truth is None:
        truth = _field(case, "expected_color", "expected", default=None)
    if truth is None:
        return {
            "status": "unknown",
            "matched": None,
            "answer_text": answer_text,
            "normalized_answer": normalized,
            "format_status": format_status,
            "answer_ids": list(map(int, answers)),
        }
    if isinstance(truth, Mapping):
        color_key = "color" in truth or "expected_color" in truth
        expected = truth.get("color", truth.get("expected_color", truth.get("answer_ids", truth.get("ids", truth.get("value")))))
    else:
        color_key = isinstance(truth, str) or (isinstance(truth, (list, tuple)) and any(isinstance(x, str) for x in truth))
        expected = truth
    # Retain an exact numeric-ID check for low-level model-free fixtures and
    # drivers that carry token truth.  The frozen panel normally uses colors.
    numeric_expected = (
        isinstance(expected, (int, np.integer))
        or isinstance(expected, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in expected)
    ) and not color_key
    if numeric_expected:
        expected_ids = [int(expected)] if isinstance(expected, (int, np.integer)) else list(map(int, expected))
        matched = list(map(int, answers)) == expected_ids
        return {
            "status": "observed_ids",
            "matched": bool(matched),
            "expected_ids": expected_ids,
            "answer_ids": list(map(int, answers)),
            "answer_text": answer_text,
            "normalized_answer": normalized,
            "format_status": format_status,
        }
    if isinstance(expected, (list, tuple)):
        q = _question_index(case)
        if q is None or q >= len(expected):
            return {
                "status": "ambiguous_question",
                "matched": None,
                "expected_colors": [str(x).strip().lower() for x in expected],
                "answer_ids": list(map(int, answers)),
                "answer_text": answer_text,
                "normalized_answer": normalized,
                "format_status": format_status,
            }
        expected_color = expected[q]
    else:
        expected_color = expected
    expected_color = str(expected_color).strip().lower() if expected_color is not None else None
    matched = format_status == "one_color" and normalized == expected_color
    return {
        "status": "observed",
        "matched": bool(matched),
        "expected_color": expected_color,
        "answer_text": answer_text,
        "normalized_answer": normalized,
        "format_status": format_status,
        "answer_ids": list(map(int, answers)),
    }


def _work(case: Mapping[str, Any]) -> dict[str, Any] | None:
    value = case.get("work", case.get("resources", case.get("cost")))
    if value is None:
        # The C driver keeps mechanics counters beside the path record rather
        # than under a synthetic ``work`` object.  Preserve those declarations
        # as observations so work/state accounting does not disappear merely
        # because the producer used the flatter schema.
        direct: dict[str, Any] = {}
        for name in (
            "state_bytes",
            "memory_bytes",
            "cache_bytes",
            "dense_cache_bytes",
            "gpu_weight_bytes",
            "gpu_peak_bytes",
            "physical_work",
        ):
            if name in case:
                direct[name] = case[name]
        for name in ("callbacks", "mechanics"):
            if name in case:
                direct[name] = case[name]
        value = direct or None
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return {"declared": value}
    # Preserve declared counters verbatim, but make the names that matter to
    # the comparison easy to find.  These are observations, not gates.
    out = dict(value)
    aliases = {
        "linear_calls": ("linear_calls", "calls"),
        "linear_macs": ("linear_macs", "macs"),
        "state_bytes": ("state_bytes", "memory_bytes", "owned_bytes"),
        "dense_cache_bytes": ("dense_cache_bytes", "cache_bytes"),
    }
    for target, names in aliases.items():
        for name in names:
            if name in value:
                out.setdefault(target, value[name])
                break
    return out


def _coverage_authority(raw: Path, driver: Mapping[str, Any]) -> tuple[set[tuple[Any, str, int]], str] | None:
    """Build expected ``(input-case, policy, question)`` rows.

    The runtime input manifest is authoritative for a real run.  A driver may
    provide the same authority explicitly for saved fixtures via
    ``expected_coverage``.  Returning ``None`` is reserved for model-free
    fixtures with neither authority; a real worker always has the manifest.
    """

    manifest = None
    for path in (raw / "inputs" / "runtime-inputs.json", raw / "runtime-inputs.json"):
        if path.exists():
            manifest = json.loads(path.read_text())
            break
    expected = driver.get("expected_coverage")
    if expected is not None:
        rows: set[tuple[Any, str, int]] = set()
        for item in expected:
            if isinstance(item, Mapping):
                rows.add((_case_id_value(item), canonical_policy(item.get("policy")), int(_question_index(item))))
            else:
                case_id, policy, question = item
                rows.add((case_id, canonical_policy(policy), int(question)))
        return rows, "driver.expected_coverage"
    if not isinstance(manifest, Mapping) or not manifest.get("policies") or not manifest.get("questions"):
        return None
    case_ids = [item["id"] for item in manifest.get("cases", []) if "id" in item]
    policies = [canonical_policy(policy) for policy in manifest["policies"]]
    questions = range(len(manifest["questions"]))
    return {(case_id, policy, int(question)) for case_id in case_ids for policy in policies for question in questions}, "runtime-inputs.json"


def _observed_coverage(rows: Sequence[Mapping[str, Any]]) -> tuple[set[tuple[Any, str, int]], list[dict[str, Any]]]:
    observed: set[tuple[Any, str, int]] = set()
    malformed: list[dict[str, Any]] = []
    for row in rows:
        case_id = row.get("input_case_id")
        policy = row.get("policy")
        question = row.get("question_index")
        if case_id is None or question is None:
            malformed.append({"case_id": row.get("case_id"), "input_case_id": case_id, "policy": policy, "question_index": question})
            continue
        observed.add((case_id, canonical_policy(policy), int(question)))
    return observed, malformed


def analyze(raw: str | Path, bank: Any, driver: Mapping[str, Any], texts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze one sealed memory-response report.

    ``bank`` is accepted for Saturn's common analyzer ABI; this experiment's
    raw report is authoritative and the bank is intentionally not inspected.
    """

    del bank
    raw_path = Path(raw)
    texts = {} if texts is None else texts
    schema = driver.get("schema")
    if schema not in (None, "tcm-memory-response-driver-v1"):
        raise ValueError(f"unexpected memory-response schema: {schema}")
    cases = _case_list(driver)
    if not cases:
        raise ValueError("memory-response report has no cases")
    manifest_truth = _manifest_truth(raw_path)
    manifest_cases = _manifest_cases(raw_path)
    color_variants = _manifest_color_variants(raw_path)
    policy_names = {canonical_policy(_field(c, "policy", default="")) for c in cases}
    observations: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    readout_declarations: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    prefix_seen: set[tuple[Any, str, str, Any, Any]] = set()
    attention_rows: list[dict[str, Any]] = []
    attention_response_rows: list[dict[str, Any]] = []
    work_rows: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        policy_value = _field(case, "policy", default="unknown")
        policy_raw = str(policy_value)
        policy = canonical_policy(policy_value)
        input_case_id = _case_id_value(case)
        explicit_task = _field(case, "task", "task_id", default=None)
        task = (
            str(explicit_task)
            if explicit_task is not None
            else f"case{input_case_id}" if input_case_id is not None else "unknown"
        )
        question_index = _question_index(case)
        # ``name`` is the preferred unique identity for the nested query
        # records emitted by the C driver.  Older model-free fixtures used an
        # arbitrary string ``id`` instead; retain that identity while avoiding
        # turning a numeric frozen input case id into a duplicate path name.
        declared_name = _field(case, "name", "arm_name", "path_name", "case_name")
        if declared_name is None and isinstance(case.get("id"), str):
            declared_name = case["id"]
        if declared_name is not None:
            case_id = str(declared_name)
        else:
            case_id = f"case{input_case_id}-{policy}"
            if question_index is not None:
                case_id += f"-q{question_index}"
            if case_id == "caseNone-unknown":
                case_id = f"case-{index}"
        query_key = f"{task}/query{question_index}" if question_index is not None else task
        history = _history(case, raw_path)
        parent_ids = _parent_history(case, raw_path)
        if case.get("parent_history_sha256") is not None:
            if parent_ids is None:
                raise ValueError(f"{case_id}: parent_history_sha256 has no parent_history_ids")
            _validate_history_digest(
                parent_ids,
                case.get("parent_history_sha256"),
                position=len(parent_ids),
                context=f"{case_id}/parent",
            )
        frozen = _manifest_case(manifest_cases, input_case_id)
        frozen_summary: dict[str, Any] = {}
        for name in ("id", "initial_bit", "update_bit", "expected", "token_bound"):
            if name in frozen:
                frozen_summary["expected_colors" if name == "expected" else name] = frozen[name]
        if "files" in frozen and isinstance(frozen["files"], Mapping):
            frozen_summary["input_files"] = {
                str(key): _logical_path(raw_path, value)
                for key, value in frozen["files"].items()
            }
        corner = _field(case, "corner")
        if corner is None and "initial_bit" in frozen and "update_bit" in frozen:
            corner = f"initial{frozen['initial_bit']}_update{frozen['update_bit']}"
        answers = _answers(case, raw_path)
        if parent_ids is not None and not history:
            history = list(parent_ids)
        truth = _truth_result(case, answers, texts, raw_path, manifest_truth)
        row = {
            "case_id": case_id,
            "task": task,
            "query_key": query_key,
            "policy": policy,
            "policy_declared": policy_raw,
            "binding": _field(case, "binding", "binding_id"),
            "corner": corner,
            "question": _field(case, "question", "question_id", "query_id"),
            "question_index": question_index,
            "query_id": _field(case, "query_id", "question_id"),
            "input_case_id": input_case_id,
            "phase": _field(case, "phase", "stage"),
            "parent": _field(case, "parent", "parent_id", "candidate_parent"),
            "history_length": len(history),
            "history": history,
            "parent_history_length": None if parent_ids is None else len(parent_ids),
            "parent_history_sha256": _declared_digest(case.get("parent_history_sha256")),
            "parent_cache_sha256": _declared_digest(case.get("parent_cache_sha256")),
            "after_rollback_cache_sha256": _declared_digest(case.get("after_rollback_cache_sha256")),
            "answer_ids": answers,
            "stopped": case.get("stopped"),
            "truncated": case.get("truncated"),
            "final_position": case.get("final_position"),
            "final_pending": case.get("final_pending"),
            "truth": truth,
            "state_intervention": _field(case, "state_intervention", "history_intervention"),
            "callbacks": case.get("callbacks"),
            "mechanics": case.get("mechanics"),
            "state_bytes": case.get("state_bytes"),
            "frozen_case": frozen_summary or None,
            "work": _work(case),
        }
        normalized.append(row)
        if row["work"] is not None:
            work_rows.append({"case_id": case_id, "task": task, "query_key": query_key, "policy": policy, **row["work"]})
        observations.append({"kind": "own-history-outcome", **row})

        # Prefix boundaries belong to the path-level parent, not to either
        # independent question.  Emit them once per (case, policy, corner)
        # while retaining both the full parent history authority and the
        # question rows below.
        prefix_key = (input_case_id, policy, task, row["binding"], row["corner"])
        if prefix_key not in prefix_seen:
            prefix_seen.add(prefix_key)
            for prefix in _prefix_boundaries(
                raw_path,
                case,
                case_id=f"case{input_case_id}-{policy}" if input_case_id is not None else f"path-{index}",
                task=task,
                policy=policy,
                parent_ids=parent_ids,
            ):
                prefix_rows.append(prefix)
                observations.append({"kind": "prefix-boundary", **prefix})

        for readout in _readouts(raw_path, case):
            logits = readout.get("_logits_array")
            position = readout.get("position")
            position = None if position is None else int(position)
            declared_history = _declared_digest(readout.get("history_sha256"))
            history_check = None
            if declared_history is not None:
                history_check = _validate_history_digest(
                    history,
                    declared_history,
                    position=position,
                    context=f"{case_id}/{readout.get('clock', readout.get('kind', 'readout'))}",
                )
            logits_descriptor = readout.get("logits")
            color_margin = _color_margin(logits, color_variants)
            declaration = {
                "case_id": case_id,
                "task": task,
                "query_key": query_key,
                "policy": policy,
                "question_index": question_index,
                "clock": readout.get("clock", readout.get("kind", "unknown")),
                "kind": readout.get("kind", readout.get("clock", "unknown")),
                "position": position,
                "pending": readout.get("pending"),
                "top": readout.get("top"),
                "runner_up": readout.get("runner_up"),
                "top_margin": readout.get("top_margin"),
                "history_sha256": declared_history,
                "history_check": history_check,
                "cache_sha256": _declared_digest(readout.get("cache_sha256")),
                "logits_file": _logical_path(raw_path, logits_descriptor),
                "logits_sha256": _declared_digest(
                    readout.get("logits_sha256")
                    or (logits_descriptor.get("sha256") if isinstance(logits_descriptor, Mapping) else None)
                ),
                "color_margin": color_margin,
                "available": logits is not None,
            }
            readout_declarations.append(declaration)
            if logits is None:
                continue
            readout_rows.append(
                {
                    "case_id": case_id,
                    "task": task,
                    "query_key": query_key,
                    "policy": policy,
                    "binding": row["binding"],
                    "corner": row["corner"],
                    "question": row["question"],
                    "question_index": question_index,
                    "phase": row["phase"],
                    "initial_bit": frozen_summary.get("initial_bit"),
                    "update_bit": frozen_summary.get("update_bit"),
                    "clock": readout.get("clock", readout.get("kind", "unknown")),
                    "kind": readout.get("kind", readout.get("clock", "unknown")),
                    "position": position,
                    "top": int(readout.get("top", int(np.argmax(logits)))),
                    "logits": logits,
                    "history": history,
                    "answer_ids": answers,
                    "history_sha256": declared_history,
                    "history_check": history_check,
                    "cache_sha256": _declared_digest(readout.get("cache_sha256")),
                    "logits_file": _logical_path(raw_path, logits_descriptor),
                    "logits_sha256": declaration["logits_sha256"],
                    "color_margin": color_margin,
                }
            )
        for att in _attention_rows(raw_path, case):
            if att.get("audit_kind") == "response_family":
                response_row = {
                    "case_id": case_id,
                    "task": task,
                    "query_key": query_key,
                    "policy": canonical_policy(att.get("policy", policy)),
                    "binding": row["binding"],
                    "corner": row["corner"],
                    "question": row["question"],
                    "phase": row["phase"],
                    "layer": att.get("layer"),
                    "position": att.get("position"),
                    "support_kind": "response_family",
                    "metrics": att["response_family"],
                }
                attention_response_rows.append(response_row)
                observations.append({"kind": "conditional-response-family", **response_row})
                continue
            metrics = att.get("precomputed_tail")
            if metrics is None:
                metrics = local_tail_identity(
                    att["_native_weights"],
                    att["_policy_weights"],
                    att["_native_values"],
                    att.get("_policy_values"),
                    att.get("_selected_indices"),
                )
            metric_row = {
                "case_id": case_id,
                "task": task,
                "query_key": query_key,
                "policy": canonical_policy(att.get("policy", policy)),
                "binding": row["binding"],
                "corner": row["corner"],
                "question": row["question"],
                "phase": row["phase"],
                "layer": att.get("layer"),
                "position": att.get("position"),
                "support_kind": att.get("support_kind", policy),
                "metrics": metrics,
            }
            if "output_relative_to_native" in att:
                metric_row["output_relative_to_native"] = att["output_relative_to_native"]
            attention_rows.append(metric_row)
            observations.append({"kind": "local-attention-tail", **metric_row})

    corner_contrasts = _corner_contrasts(readout_rows)
    for contrast in corner_contrasts:
        observations.append({"kind": "four-corner-color-contrast", **contrast})

    coverage_authority = _coverage_authority(raw_path, driver)
    coverage = None
    if coverage_authority is not None:
        expected, authority = coverage_authority
        observed, malformed = _observed_coverage(normalized)
        missing = sorted(expected - observed, key=lambda x: (str(x[0]), x[1], x[2]))
        unexpected = sorted(observed - expected, key=lambda x: (str(x[0]), x[1], x[2]))
        coverage = {
            "authority": authority,
            "expected_rows": len(expected),
            "observed_rows": len(observed),
            "missing": [list(x) for x in missing],
            "unexpected": [list(x) for x in unexpected],
            "malformed_rows": malformed,
            "status": "complete" if not missing and not unexpected and not malformed else "incomplete",
        }
        # A real experiment's lane is frozen by runtime-inputs.json.  Missing
        # case/policy/question rows are a mechanics/evidence defect, not a
        # scientific null, and must not be silently dropped.
        if coverage["status"] != "complete":
            raise ValueError("memory-response coverage incomplete: " + json.dumps(coverage, sort_keys=True))

    # Pair candidate readouts with the native case at the same semantic key.
    # If a candidate has diverged history, retain the row with a post-divergence
    # label rather than silently treating it as a native-fidelity comparison.
    native_cases = [c for c in normalized if c["policy"] == "native"]
    native_readouts = [r for r in readout_rows if r["policy"] == "native"]
    comparisons: list[dict[str, Any]] = []
    for candidate in (c for c in normalized if c["policy"] != "native"):
        key = (candidate["task"], candidate["binding"], candidate["corner"], candidate["question"], candidate["phase"])
        native = next((c for c in native_cases if (c["task"], c["binding"], c["corner"], c["question"], c["phase"]) == key), None)
        if native is None:
            native = next((c for c in native_cases if c["task"] == candidate["task"] and c["question"] == candidate["question"]), None)
        first = None
        same_prefix = False
        if native is not None:
            same_prefix, first = _same_history(candidate["answer_ids"] if not candidate["history"] else candidate["history"], native["answer_ids"] if not native["history"] else native["history"])
        cand_rs = [r for r in readout_rows if r["case_id"] == candidate["case_id"]]
        nat_rs = [r for r in native_readouts if native is not None and r["case_id"] == native["case_id"]]
        for cr in cand_rs:
            nr = next((x for x in nat_rs if x["clock"] == cr["clock"] and x["position"] == cr["position"]), None)
            if nr is None:
                continue
            position = cr["position"]
            same_at_position, first_at_position = _same_history(candidate["history"], native["history"], position) if native is not None else (False, 0)
            forced = bool(candidate.get("forced_native_history") or cr.get("forced_native_history"))
            if same_at_position:
                domain = "same_id"
            elif forced:
                domain = "forced_same_id"
            else:
                domain = "post_divergence" if first_at_position < (position or 0) else "different_clock"
            cmp = {
                "kind": "readout-comparison",
                "candidate_case": candidate["case_id"],
                "native_case": native["case_id"] if native else None,
                "task": candidate["task"],
                "query_key": candidate.get("query_key"),
                "policy": candidate["policy"],
                "binding": candidate["binding"],
                "corner": candidate["corner"],
                "question": candidate["question"],
                "phase": candidate["phase"],
                "clock": cr["clock"],
                "position": position,
                "comparison_domain": domain,
                "first_history_difference": first_at_position,
                "metric": _metric(cr["logits"], nr["logits"]),
                "candidate_top": cr["top"],
                "native_top": nr["top"],
                "candidate_red_minus_blue": cr.get("color_margin", {}).get("primary", {}).get("red_minus_blue"),
                "native_red_minus_blue": nr.get("color_margin", {}).get("primary", {}).get("red_minus_blue"),
            }
            comparisons.append(cmp)
            observations.append(cmp)

    # A compact summary is deliberately descriptive.  ``native_fidelity`` is
    # restricted to same-ID rows; ``own_history`` includes all candidate rows.
    same_id = [x for x in comparisons if x["comparison_domain"] in ("same_id", "forced_same_id")]
    post = [x for x in comparisons if x["comparison_domain"] == "post_divergence"]
    summary = {
        "schema": "tcm-memory-response-analysis-v1",
        "driver_identity": {
            key: driver.get(key)
            for key in (
                "schema",
                "model_identity",
                "model_digest",
                "source_bundle_bytes",
                "capacity",
                "ack_budget",
                "answer_budget",
                "callback_layers",
                "lane",
            )
            if key in driver
        },
        "cases": len(normalized),
        "tasks": sorted({x["task"] for x in normalized}),
        "policies": sorted(policy_names),
        "policy_case_counts": {p: sum(x["policy"] == p for x in normalized) for p in sorted(policy_names)},
        "readouts": len(readout_rows),
        "readout_declarations": len(readout_declarations),
        "readout_arrays_available": sum(x["available"] for x in readout_declarations),
        "color_margin_readouts": sum(
            row.get("color_margin", {}).get("primary", {}).get("red_minus_blue") is not None
            for row in readout_declarations
        ),
        "four_corner_contrasts": len(corner_contrasts),
        "four_corner_complete_contrasts": sum(x["status"] == "complete" for x in corner_contrasts),
        "prefix_boundaries": len(prefix_rows),
        "attention_rows": len(attention_rows),
        "attention_response_rows": len(attention_response_rows),
        "attention_response_instrument_status": {
            status: sum(
                row["metrics"].get("codec_reconstruction_status") == status
                for row in attention_response_rows
            )
            for status in sorted(
                {
                    row["metrics"].get("codec_reconstruction_status")
                    for row in attention_response_rows
                    if row["metrics"].get("codec_reconstruction_status") is not None
                }
            )
        },
        "comparison_rows": len(comparisons),
        "native_fidelity_same_id_rows": len(same_id),
        "post_divergence_rows": len(post),
        "truth_matches": {"matched": sum(x["truth"]["matched"] is True for x in normalized), "known": sum(x["truth"]["matched"] is not None for x in normalized)},
        "status": "observational",
        "limits": [
            "same-ID readout metrics apply only before consumed histories diverge",
            "post-divergence rows are own-history policy outcomes, not native-fidelity evidence",
            "truth and native correctness are separate observables",
            "history_deleted is interpreted from the declared native-parent intervention; no recurrence is inferred",
        ],
    }
    seal_path = raw_path / "primary-seal.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        summary["primary_seal"] = {
            "present": True,
            "schema": seal.get("schema"),
            "file_count": len(seal.get("files", {})) if isinstance(seal.get("files"), Mapping) else None,
            "model_calls_after_seal": seal.get("model_calls_after_seal"),
            "timing": seal.get("timing"),
        }
    else:
        summary["primary_seal"] = {"present": False}
    if coverage is not None:
        summary["coverage"] = coverage
    return {
        "summary": summary,
        "cases": normalized,
        "readout_declarations": readout_declarations,
        "prefix_boundaries": prefix_rows,
        "four_corner_contrasts": corner_contrasts,
        "readout_comparisons": comparisons,
        "attention_tail_rows": attention_rows,
        "attention_response_rows": attention_response_rows,
        "work_rows": work_rows,
        "observations": observations,
        "terminal_status": "not-requested",
    }


__all__ = [
    "POLICIES",
    "attention_output",
    "local_tail_identity",
    "norm_ratio",
    "policy_factorial",
    "relative_l2",
    "signed_stats",
    "analyze",
]
