"""Saved-only custody and analyzer verification for operand composition.

The verifier never loads the model and never calls CUDA.  It joins the mrun
receipt, immutable raw archive, executed payload, driver, and the strict saved
mechanics/numerics instruments.  The large factorial analyzer is deliberately
deferred to post-collection. Numeric metrics remain observations; only source,
schema, finite-array, replay, and custody defects can make this verifier fail.
"""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Any) -> Any:
    """Normalize relocation-dependent raw paths while preserving all values."""
    if isinstance(value, dict):
        return {str(key): _canonical_path(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def _canonical_path(key: Any, value: Any) -> Any:
    if isinstance(value, str) and (str(key) in {"raw", "path", "file"} or str(key).endswith("_path")):
        path = Path(value)
        if path.is_absolute():
            parts = path.parts
            # Worker roots are intentionally different from collected verifier
            # roots.  Preserve the logical raw-relative suffix only.
            for marker in ("results", "raw"):
                if marker in parts:
                    index = max(i for i, item in enumerate(parts) if item == marker)
                    suffix = parts[index + 1 :]
                    return "<RAW>" + ("/" + "/".join(suffix) if suffix else "")
            return "<ABS>" + ("/" + path.name if path.name else "")
    return _canonical(value)


def _compare(a: Any, b: Any, path: tuple[Any, ...] = ()) -> None:
    aa, bb = _canonical(a), _canonical(b)
    if isinstance(aa, dict):
        if not isinstance(bb, dict) or set(aa) != set(bb):
            raise ValueError(f"Saved analysis schema differs at {path}")
        for key in aa:
            _compare(aa[key], bb[key], path + (key,))
        return
    if isinstance(aa, list):
        if not isinstance(bb, list) or len(aa) != len(bb):
            raise ValueError(f"Saved analysis row count differs at {path}")
        for index, (left, right) in enumerate(zip(aa, bb)):
            _compare(left, right, path + (index,))
        return
    if isinstance(aa, float) or isinstance(bb, float):
        if not isinstance(aa, (int, float)) or not isinstance(bb, (int, float)):
            raise ValueError(f"Saved scalar type differs at {path}")
        if not math.isfinite(float(aa)) or not math.isfinite(float(bb)):
            raise ValueError(f"Nonfinite saved scalar at {path}")
        if not math.isclose(float(aa), float(bb), rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError(f"Saved analysis arithmetic differs at {path}: {aa} != {bb}")
        return
    if aa != bb:
        raise ValueError(f"Saved analysis value differs at {path}: {aa!r} != {bb!r}")


def _module(payload: Path, name: str):
    spec = importlib.util.spec_from_file_location("operand_composition_saved_" + name, payload / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load payload/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(payload))
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _receipt_check(wrapper: dict) -> None:
    receipt = copy.deepcopy(wrapper["receipt"])
    fingerprint = receipt.pop("fingerprint")
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != fingerprint:
        raise ValueError("Saturn receipt fingerprint differs")
    if receipt.get("parent") != wrapper.get("before") or receipt.get("result") != wrapper.get("after"):
        raise ValueError("Saturn receipt boundary differs")


def _observations(analysis: dict, report_sha: str, model_id: str) -> list[dict]:
    rows: list[dict] = []
    for key in ("factorial", "own_history_factorial", "same_id_readouts", "component_observations"):
        value = analysis.get(key, [])
        if not isinstance(value, list):
            continue
        for index, row in enumerate(value):
            if isinstance(row, dict):
                rows.append({"schema": "operand-composition-observation-v1", "kind": key, "row": index, "model_id": model_id, "report_sha256": report_sha, **row})
    return rows


def verify(target: str | Path) -> dict:
    root = Path(target).resolve()
    report = _load_json(root / "report.json")
    submission = _load_json(root / "submission.json")
    job = _load_json(root / "job.json")
    if submission.get("job_id") != job.get("job_id"):
        raise ValueError("Scheduler job identity differs")
    log_path = root / "logs.txt"
    job_log = job.get("result", {}).get("artifacts", {}).get("job_log", {}).get("sha256")
    if job_log and sha(log_path) != job_log:
        raise ValueError("Scheduler log identity differs")
    logged = [json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1]))) for line in log_path.read_text().splitlines() if line.startswith("OPERAND_COMPOSITION_RESULT_GZIP_B64=")]
    if logged != [report]:
        raise ValueError("Report log join differs")
    if _load_json(root / "raw" / "report.json") != report:
        raise ValueError("Raw report join differs")

    config = submission["config"]
    for name, expected in config["sources"].items():
        path = root / "payload" / name
        if not path.exists() or sha(path) != expected:
            raise ValueError(f"Frozen payload source differs: {name}")
    manifest = _load_json(root / "payload" / "source-manifest.json")
    if sha(root / "payload" / "source-manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Payload manifest differs")
    if manifest.get("execution_paths") != {"own": 24, "replay": 18, "total": 42}:
        raise ValueError("Declared execution geometry differs")

    raw_records = report.get("raw_files", {})
    for name, record in raw_records.items():
        path = root / "raw" / name
        if not path.exists() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Raw artifact differs: {name}")
    driver = report.get("driver")
    if not isinstance(driver, dict) or _load_json(root / "raw" / "driver.json") != driver:
        raise ValueError("Driver join differs")
    if driver.get("schema") != "tcm-operand-composition-driver-v1" or len(driver.get("paths", [])) != 42:
        raise ValueError("Driver geometry/schema differs")
    if driver.get("model_digest") != report["model"]["content_sha256"] or driver.get("model_identity") != report["model"]["identity"]:
        raise ValueError("Driver model identity differs")
    if driver.get("native_sham_logits_and_kv_byte_exact") is not True:
        raise ValueError("Native sham mechanics did not close")
    if report.get("mechanics_status") != "complete":
        raise ValueError("Run mechanics are incomplete")
    if "saturn_receipt" not in report:
        raise ValueError("Missing Saturn receipt")
    _receipt_check(report["saturn_receipt"])
    program_seal = _load_json(root / "raw" / "program-seal.json")
    binary = root / "raw" / "operand-composition.bin"
    if program_seal.get("binary_sha256") != sha(binary) or program_seal.get("binary_bytes") != binary.stat().st_size:
        raise ValueError("Sealed executable custody differs")

    payload = root / "payload"
    mechanics_module = _module(payload, "mechanics")
    numerics_module = _module(payload, "numerics")
    recomputed_mechanics = mechanics_module.verify_mechanics(root / "raw", driver, report.get("inputs", {}))
    recomputed_numerics = numerics_module.audit_numerics(root / "raw", driver)
    _compare(recomputed_mechanics, report.get("mechanics", {}), ("mechanics",))
    expected_deferred = {"stage": "deferred-saved-only", "raw_arrays_authoritative": True, "analysis_module": "analysis.py", "audit_module": "audit.py"}
    _compare(expected_deferred, report.get("analysis", {}), ("analysis",))
    _compare(recomputed_numerics, report.get("numerics", {}), ("numerics",))
    report_sha = sha(root / "report.json")
    observations = [{"schema": "operand-composition-observation-v1", "kind": "numerics", "row": index, "model_id": report["model"]["identity"], "report_sha256": report_sha, **row} for index, row in enumerate(recomputed_numerics.get("rows", []))]
    (root / "observations.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in observations))
    result = {
        "schema": "tcm-operand-composition-verification-v1",
        "report_sha256": report_sha,
        "scheduler_log_bound": True,
        "payload_sources_verified": len(config["sources"]),
        "raw_files_verified": len(raw_records),
        "driver_geometry_verified": True,
        "mechanics_recomputed": True,
        "native_sham_verified": True,
        "whole_invocation_receipt": True,
        "analysis_deferred": True,
        "audit_deferred": True,
        "numerics_recomputed": True,
        "observation_rows": len(observations),
        "models_evaluated": 0,
        "mechanics_status": "valid",
        "instrument_status": report.get("instrument_status", "unknown"),
        "trend_status": report.get("trend_status", "observations"),
        "terminal_status": "not-requested",
        "scope": "Saved raw report, source payload, typed driver, native sham, strict mechanics and CPU numerical equations; factorial analyzer/audit remain post-collection; no new model calls.",
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.target), indent=2, sort_keys=True))
