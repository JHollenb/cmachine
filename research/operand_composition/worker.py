"""Run the sealed MLP16/query17 composition panel on one resident C model."""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results"
LIMIT = 512 * 1024**2
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args: list[str], timeout: float = 720) -> subprocess.CompletedProcess[str]:
    print("[run]", " ".join(map(str, args)), flush=True)
    process = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"Process {process.returncode}: {process.stderr[-12000:]}")
    return process


def _manifest() -> dict:
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        actual = sha(ROOT / name)
        if actual != expected:
            raise ValueError(f"Payload source differs: {name}")
    return manifest


def _model() -> dict:
    path = ROOT / "model.json"
    if path.exists():
        return json.loads(path.read_text())
    baseline = json.loads((ROOT / "baseline.json").read_text())
    return baseline["model"]


def _check_factors() -> dict:
    """Check the six C-facing arrays before any target tokenization."""
    expected = {
        "mlp-E.f32": (64 * 1536, "<f4"),
        "mlp-Lg.f32": (8960 * 64, "<f4"),
        "mlp-Lu.f32": (8960 * 64, "<f4"),
        "query-B.f32": (128 * 1536, "<f4"),
        "query-F.f32": (1536 * 128, "<f4"),
        "query-K.f32": (256 * 128, "<f4"),
    }
    import numpy as np

    rows = {}
    for name, (count, dtype) in expected.items():
        path = ROOT / name
        if path.stat().st_size != count * 4:
            raise ValueError(f"Factor extent differs: {name}")
        values = np.fromfile(path, dtype=dtype)
        if values.size != count or not np.isfinite(values).all():
            raise ValueError(f"Factor values invalid: {name}")
        rows[name] = {"bytes": path.stat().st_size, "sha256": sha(path), "elements": count, "dtype": dtype}
    return rows


def _seal_program(manifest: dict, model: dict, factors: dict, binary: Path) -> dict:
    provenance_path = ROOT / "factor-provenance.json"
    source_manifest = ROOT / "source-manifest.json"
    seal = {
        "schema": "tcm-operand-composition-program-seal-v1",
        "phase": "before-target-tokenization",
        "source_manifest_sha256": sha(source_manifest),
        "factor_provenance_sha256": sha(provenance_path),
        "factor_images": factors,
        "binary_sha256": sha(binary),
        "binary_bytes": binary.stat().st_size,
        "model_identity": model["identity"],
        "model_digest": model["content_sha256"],
        "path_contract": {"own": 24, "replay": 18, "total": 42},
        "memory_contract": "full native FP32 K/V for all four arms",
    }
    (RAW / "program-seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    return seal


def _tokenizer(model: dict):
    from transformers import AutoTokenizer

    source = Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name, expected in model["tokenizer_files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Tokenizer differs: {name}")
    return AutoTokenizer.from_pretrained(source, local_files_only=True), source


def _tokenize_inputs(tokenizer, model: dict) -> tuple[dict, list[int]]:
    specification = json.loads((ROOT / "inputs.json").read_text())
    if specification.get("schema") != "tcm-operand-composition-input-v1":
        raise ValueError("Input schema differs")
    cases = []
    for row in specification["cases"]:
        case_id = int(row["id"])
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        extended = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": ""}, {"role": "user", "content": row["followup"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not extended.startswith(prompt_text):
            raise ValueError(f"Case {case_id} template prefix differs")
        suffix_ids = tokenizer.encode(extended[len(prompt_text):], add_special_tokens=False)
        if not suffix_ids or suffix_ids[0] != 151645:
            raise ValueError(f"Case {case_id} missing consumed assistant boundary")
        if tokenizer.encode(extended, add_special_tokens=False) != prompt_ids + suffix_ids:
            raise ValueError(f"Case {case_id} token concatenation differs")
        (RAW / f"case{case_id}-prompt.txt").write_text(" ".join(map(str, prompt_ids)) + "\n")
        (RAW / f"case{case_id}-followup.txt").write_text(" ".join(map(str, suffix_ids)) + "\n")
        cases.append({
            "id": case_id,
            "name": row["name"],
            "family": row["family"],
            "format": row["format"],
            "prompt_text": row["prompt"],
            "followup_text": row["followup"],
            "prompt_ids": prompt_ids,
            "followup_ids": suffix_ids,
            "budget": int(row["budget"]),
            "expected": row.get("expected"),
            "split": "fresh",
        })
    if [row["budget"] for row in cases] != specification["budgets"]:
        raise ValueError("Budget list differs")
    budgets = [int(row["budget"]) for row in cases]
    (RAW / "budgets.txt").write_text("".join(f"{budget}\n" for budget in budgets))
    # C consumes this compact manifest for task IDs/budgets; it still receives
    # the two token files as the authoritative token bytes.
    case_manifest = {
        "schema": "tcm-operand-composition-cases-v1",
        "cases": [{"id": row["id"], "name": row["name"], "prompt_file": f"case{row['id']}-prompt.txt", "followup_file": f"case{row['id']}-followup.txt", "budget": row["budget"], "family": row["family"], "format": row["format"]} for row in cases],
        "budgets_file": "budgets.txt",
        "prompt_and_followup_ids_are_sealed": True,
    }
    (RAW / "cases.json").write_text(json.dumps(case_manifest, indent=2, sort_keys=True) + "\n")
    return {"schema": "tcm-operand-composition-input-manifest-v1", "source_inputs_sha256": sha(ROOT / "inputs.json"), "cases": cases, "tokenizer": {"files": model["tokenizer_files"], "source": str(Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))}, "budgets": budgets}


def _capacity_bound(inputs: dict) -> tuple[int, list[int]]:
    maxima = [len(row["prompt_ids"]) + len(row["followup_ids"]) + 2 * int(row["budget"]) for row in inputs["cases"]]
    # Seven paths are emitted per case (four own arms plus three same-ID
    # replays).  Count the actual Record extent rather than a guessed trace
    # stride, then add the four full-vocabulary boundaries and the independent
    # gate/up audit vectors.  The fixed allowance covers JSON, hashes, factor
    # images and the small runtime archive; it is deliberately bounded.
    records = 7 * sum(maxima) * 49920
    readouts = 42 * 4 * 151936 * 4
    gate_up = 42 * 4 * 2 * 8960 * 4
    first_gate_up = 42 * 2 * 8960 * 4
    fixed = 32 * 1024**2
    bound = records + readouts + gate_up + first_gate_up + fixed
    return bound, maxima


def _invoke(binary: Path, model: dict, input_manifest: dict, program_seal: dict, source_manifest_sha: str):
    sys.path.insert(0, str(ROOT / "public"))
    from saturn_bridge import invoke

    args = [binary, model["path"], RAW]

    def controller(_request):
        process = command(args)
        (RAW / "driver.stdout.txt").write_text(process.stdout)
        (RAW / "driver.stderr.txt").write_text(process.stderr)
        return json.loads((RAW / "driver.json").read_text())

    request = {
        "model_digest": model["content_sha256"],
        "model_identity": model["identity"],
        "binary_sha256": sha(binary),
        "input_manifest_sha256": sha(RAW / "input-manifest.json"),
        "program_seal_sha256": sha(RAW / "program-seal.json"),
        "source_manifest_sha256": source_manifest_sha,
        "scope": "Frozen coupled64 MLP16 x frozen response-fit query17; full native FP32 K/V; six two-turn cases; 24 own + 18 replay paths",
    }
    return invoke(args, None, request, controller=controller)


def _driver_mechanics(driver: dict) -> None:
    if driver.get("schema") != "tcm-operand-composition-driver-v1":
        raise ValueError("Driver schema differs")
    paths = driver.get("paths")
    if not isinstance(paths, list) or len(paths) != 42:
        raise ValueError("Driver path count differs")
    own = [row for row in paths if row.get("mode") == "own"]
    replay = [row for row in paths if row.get("mode") == "replay"]
    if len(own) != 24 or len(replay) != 18:
        raise ValueError("Driver own/replay path coverage differs")
    if any(row.get("task") not in range(6) for row in paths):
        raise ValueError("Driver task ID differs")


def _driver_model_identity(driver: dict, model: dict) -> None:
    if driver.get("model_digest") != model["content_sha256"] or driver.get("model_identity") != model["identity"]:
        raise ValueError("Executed driver model identity differs")


def experiment() -> dict:
    from mechanics import verify_mechanics

    manifest = _manifest()
    model = _model()
    factors = _check_factors()
    # The seal must be written only after the compiled executable exists and all
    # factor bytes are checked, but before inputs are parsed or tokenized.
    RAW.mkdir(exist_ok=True)
    command(["make"])
    binary = ROOT / "build" / "operand-composition"
    if not binary.exists():
        raise FileNotFoundError(binary)
    program_seal = _seal_program(manifest, model, factors, binary)
    shutil.copy2(binary, RAW / "operand-composition.bin")
    # The C ABI reads its immutable images from RAW, while the payload copies
    # remain the source-of-truth for the seal.  Copy only after checking the
    # bytes and before tokenization; no factor can be replaced between seal and
    # execution without a source-manifest failure.
    for name in factors:
        shutil.copy2(ROOT / name, RAW / name)
    shutil.copy2(ROOT / "factor-provenance.json", RAW / "factor-provenance.json")
    print("sealed executable and six frozen factor images before target tokenization", flush=True)

    tokenizer, tokenizer_source = _tokenizer(model)
    input_manifest = _tokenize_inputs(tokenizer, model)
    bound, maxima = _capacity_bound(input_manifest)
    if bound > LIMIT:
        raise ValueError(f"Archive geometry exceeds declared bound: {bound} > {LIMIT}")
    (RAW / "input-manifest.json").write_text(json.dumps(input_manifest, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "tcm-operand-composition-experiment-v1",
        "model": model,
        "input_manifest_sha256": sha(RAW / "input-manifest.json"),
        "input_source_sha256": input_manifest["source_inputs_sha256"],
        "program_seal": program_seal,
        "program_seal_sha256": sha(RAW / "program-seal.json"),
        "evidence_capacity_preflight": {"worst_case_bytes": bound, "limit_bytes": LIMIT, "maxima": maxima, "paths": 42, "record_bytes": 49920, "full_readout_boundaries_per_path": 4, "gate_up_vectors_per_boundary": 2, "gate_up_width": 8960, "first_boundary_gate_up": True, "fixed_allowance_bytes": 32 * 1024**2},
        "inputs": input_manifest,
        "texts": {},
        "environment": {"platform": platform.platform(), "python": sys.version, "tokenizer_source": str(tokenizer_source), "job_id": os.environ.get("MRUN_JOB_ID")},
        "mechanics_status": "in-progress",
        "instrument_status": "in-progress",
        "trend_status": "observations",
        "terminal_status": "not-requested",
    }
    (RAW / "partial-report.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    driver, receipt = _invoke(binary, model, input_manifest, program_seal, sha(ROOT / "source-manifest.json"))
    _driver_mechanics(driver)
    _driver_model_identity(driver, model)
    mechanics = verify_mechanics(RAW, driver, input_manifest)
    report["driver"] = driver
    report["mechanics"] = mechanics
    report["saturn_receipt"] = receipt
    for path in driver["paths"]:
        report["texts"][path["name"]] = [tokenizer.decode(turn.get("answer_ids", []), skip_special_tokens=True) for turn in path.get("turns", [])]
    (RAW / "partial-report.json").write_text(json.dumps(report, sort_keys=True) + "\n")

    # The producer does not materialize the large factorial/readout JSON.  The
    # saved analysis and audit adapters run after collection from immutable raw
    # arrays, so a diagnostics bug cannot turn a completed native execution
    # into a missing primary result or inflate the live worker archive.
    report["analysis"] = {"stage": "deferred-saved-only", "raw_arrays_authoritative": True, "analysis_module": "analysis.py", "audit_module": "audit.py"}
    from numerics import audit_numerics
    report["numerics"] = audit_numerics(RAW, driver)
    report["raw_files"] = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(RAW.iterdir()) if path.is_file() and path.name not in {"partial-report.json", "report.json"}}
    report["mechanics_status"] = "complete" if mechanics.get("mechanics_status") == "complete" else "incomplete"
    report["instrument_status"] = "complete"
    return report


def emit() -> None:
    files = []
    for directory, prefix in ((RAW, "raw"), (ROOT / "runtime", "payload/runtime"), (ROOT / "public", "payload/public")):
        if not directory.exists():
            continue
        files.extend((path, prefix + "/" + str(path.relative_to(directory))) for path in sorted(directory.rglob("*")) if path.is_file() and "__pycache__" not in path.parts)
    ignored = {"build", "results", "staged", "runs", ".git"}
    files.extend((path, "payload/" + path.name) for path in sorted(ROOT.iterdir()) if path.is_file() and path.name not in ignored and path.name not in {"partial-report.json", "report.json"})
    total = sum(path.stat().st_size for path, _ in files)
    if total > LIMIT:
        raise ValueError(f"Archive bound exceeded: {total} > {LIMIT}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print("OPERAND_COMPOSITION_FILES_TGZ_B64=" + base64.b64encode(buffer.getvalue()).decode(), flush=True)


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        result = experiment()
        code = 0
    except Exception:
        result = json.loads((RAW / "partial-report.json").read_text()) if (RAW / "partial-report.json").exists() else {}
        result.update(error=traceback.format_exc(), mechanics_status="incomplete", instrument_status="incomplete")
        code = 1
    result["worker_seconds"] = time.monotonic() - started
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    (RAW / "report.json").write_bytes(data + b"\n")
    print("OPERAND_COMPOSITION_RESULT_GZIP_B64=" + base64.b64encode(gzip.compress(data, mtime=0)).decode(), flush=True)
    emit()
    raise SystemExit(code)
