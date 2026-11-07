"""Stage and submit the frozen MLP16/query composition panel.

The executable source is intentionally kept separate from this launcher.  The
driver is supplied by the experiment owner and consumes the six MLP/query
factor images staged here.  This file only assembles authenticated predecessors,
seals the payload, and uses mrun's intent-first scheduler; it never fits a new
program and never tokenizes a target during staging.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
QUERY_RUN = HERE.parent / "query_response" / "runs" / "job-b141eadfe952"
QUERY_REPORT_SHA256 = "2ff3b934b26b496c054be44feef4767bedda779866bb33372a6498f4b1addc53"
COUPLED_RUN = WORKSPACE / "saturn" / "experiments" / "2026-11-06-contextual-union" / "runs" / "job-7a1a2715be48"
COUPLED_REPORT_SHA256 = "92f2f527fb3c0e9fafdb4d2cbb2b04b79ba29544fd9ab6068006bb6f39dbfbe0"
COUPLED_SOURCE = COUPLED_RUN / "raw" / "qwen25-15b" / "programs" / "coupled64.npz"
COUPLED_SELECTION = COUPLED_RUN / "raw" / "qwen25-15b" / "selection-seal.json"
LIMIT = 512 * 1024**2


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path: Path, expected: str) -> Path:
    path = Path(path)
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Source identity differs for {path}: {actual} != {expected}")
    return path


def record(path: Path) -> dict:
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def _copy_checked(source: Path, destination: Path, expected: str | None = None) -> str:
    if expected is not None:
        checked(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return sha(destination)


def _factor_images(payload: Path, source_npz: Path) -> dict:
    """Emit the exact row-major FP32 factors expected by driver.c.

    The contextual-union source is an authenticated NPZ container.  The C
    driver never opens NPZ or invokes Python: extraction is a one-time payload
    construction step, and every emitted image is sealed by its own hash.
    """
    import numpy as np

    expected = {
        "E": ((64, 1536), "mlp-E.f32"),
        "Lg": ((8960, 64), "mlp-Lg.f32"),
        "Lu": ((8960, 64), "mlp-Lu.f32"),
    }
    emitted = {}
    with np.load(source_npz, allow_pickle=False) as arrays:
        for key, (shape, filename) in expected.items():
            if key not in arrays:
                raise ValueError(f"coupled64 source misses {key}")
            value = np.asarray(arrays[key])
            if value.shape != shape:
                raise ValueError(f"{key} shape {value.shape} != {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains non-finite values")
            value = np.asarray(value, dtype="<f4", order="C")
            destination = payload / filename
            destination.write_bytes(value.tobytes(order="C"))
            emitted[filename] = {**record(destination), "shape": list(shape), "dtype": "<f4", "source_key": key}
    return emitted


def stage() -> Path:
    """Create a source-bound payload without loading a model or parsing targets."""
    import numpy as np  # noqa: F401  (documents the stage-time dependency)

    query_report = json.loads(checked(QUERY_RUN / "report.json", QUERY_REPORT_SHA256).read_text())
    query_submission = json.loads((QUERY_RUN / "submission.json").read_text())
    if query_report["model"]["profile_version"] != "qwen25-fp32-execution-v2":
        raise ValueError("Unexpected predecessor model profile")
    query_sources = query_submission["config"]["sources"]

    coupled_report = json.loads(checked(COUPLED_RUN / "report.json", COUPLED_REPORT_SHA256).read_text())
    coupled_selection = json.loads(COUPLED_SELECTION.read_text())
    coupled_model = next(row["model"] for row in coupled_report["models"] if row["model"]["key"] == "qwen25-15b")
    model = query_report["model"]
    coupled_source_files = coupled_model["source"]["files"]
    if coupled_source_files["model.safetensors"]["sha256"] != model["source_weights_sha256"]:
        raise ValueError("MLP source checkpoint differs from serving model weights")
    if coupled_source_files["config.json"]["sha256"] != model["tokenizer_files"]["config.json"]:
        raise ValueError("MLP source checkpoint config differs from serving model config")
    if (coupled_model["selected_layer"], coupled_model["hidden"], coupled_model["intermediate"], coupled_model["kv_width"]) != (16, 1536, 8960, 256):
        raise ValueError("MLP source model geometry differs")
    selection_record = coupled_report["raw_files"]["qwen25-15b/selection-seal.json"]
    checked(COUPLED_SELECTION, selection_record["sha256"])
    if COUPLED_SELECTION.stat().st_size != selection_record["bytes"]:
        raise ValueError("coupled64 selection seal extent differs")
    coupled_record = coupled_report["raw_files"]["qwen25-15b/programs/coupled64.npz"]
    checked(COUPLED_SOURCE, coupled_record["sha256"])
    if coupled_record["bytes"] != COUPLED_SOURCE.stat().st_size:
        raise ValueError("coupled64 source extent differs")
    if coupled_selection["programs"]["coupled64"]["factor_sha256"] != coupled_record["sha256"]:
        raise ValueError("coupled64 selection seal does not bind source bytes")
    if coupled_selection["programs"]["coupled64"]["rank"] != 64:
        raise ValueError("coupled64 rank differs")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    payload = HERE / "staged" / stamp / "payload"
    (payload / "runtime").mkdir(parents=True)
    (payload / "public").mkdir(parents=True)

    # Copy only the authenticated runtime/public boundary from query-response.
    # No predecessor worker, compiler, target cases, or old driver is reused.
    predecessor_payload = QUERY_RUN / "payload"
    for relative in sorted(query_sources):
        if not (relative.startswith("runtime/") or relative.startswith("public/")):
            continue
        source = checked(predecessor_payload / relative, query_sources[relative])
        _copy_checked(source, payload / relative, query_sources[relative])
    bridge_name = "saturn_bridge.py"
    bridge = checked(predecessor_payload / bridge_name, query_sources[bridge_name])
    _copy_checked(bridge, payload / bridge_name, query_sources[bridge_name])

    # Local executable sources are copied only after the owner has supplied the
    # driver and saved analyzer.  Keeping the requirement explicit prevents a
    # launcher from producing a payload that cannot close its declared ABI.
    local_sources = ("driver.c", "analysis.py", "audit.py", "mechanics.py", "numerics.py", "Makefile", "worker.py", "run.py", "verification.py", "inputs.json")
    missing = [name for name in local_sources if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError("Composition source is not ready: " + ", ".join(missing))
    for name in local_sources:
        shutil.copy2(HERE / name, payload / name)
    # Explanatory material and model-free tests are included when present.
    # verification.py is required above because the collected report cannot
    # close its saved-only custody contract without the verifier.
    for candidate in ("PLAN.md", "README.md", "MATH.md", "test_analysis.py", "test_mechanics.py", "test_numerics.py"):
        if (HERE / candidate).is_file():
            shutil.copy2(HERE / candidate, payload / candidate)

    # Preserve the predecessor arrays under source names and emit the exact
    # fixed names consumed by the new C driver.
    query_files = {
        "query-B.f32": "program128-basis.f32",
        "query-F.f32": "query-fit.f32",
        "query-K.f32": "program128-k.f32",
    }
    query_records = {}
    for emitted_name, source_name in query_files.items():
        source_record = query_report["raw_files"][source_name]
        source = checked(QUERY_RUN / "raw" / source_name, source_record["sha256"])
        if source.stat().st_size != source_record["bytes"]:
            raise ValueError(f"Query predecessor extent differs for {source_name}")
        _copy_checked(source, payload / emitted_name, source_record["sha256"])
        # Keep an immutable copy for a future saved-only provenance audit.
        _copy_checked(source, payload / ("source-" + source_name), source_record["sha256"])
        query_records[emitted_name] = {"source_name": source_name, **record(payload / emitted_name), "source_record": source_record}

    _copy_checked(COUPLED_SOURCE, payload / "source-coupled64.npz", coupled_record["sha256"])
    _copy_checked(COUPLED_SELECTION, payload / "source-coupled64-selection-seal.json", selection_record["sha256"])
    coupled_records = _factor_images(payload, COUPLED_SOURCE)

    (payload / "model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    (payload / "baseline.json").write_text(json.dumps({"model": model, "predecessor_report_sha256": QUERY_REPORT_SHA256}, indent=2, sort_keys=True) + "\n")

    provenance = {
        "schema": "tcm-operand-composition-provenance-v1",
        "model": {"identity": model["identity"], "content_sha256": model["content_sha256"], "profile_version": model["profile_version"]},
        "mlp16": {
            "source_job_id": "job-7a1a2715be48",
            "source_report_sha256": COUPLED_REPORT_SHA256,
            "selection_seal_sha256": sha(COUPLED_SELECTION),
            "source_npz_sha256": coupled_record["sha256"],
            "source_npz_bytes": coupled_record["bytes"],
            "source_checkpoint_sha256": coupled_source_files["model.safetensors"]["sha256"],
            "source_config_sha256": coupled_source_files["config.json"]["sha256"],
            "program": coupled_selection["programs"]["coupled64"],
            "images": coupled_records,
            "formula": "post-RMSNorm/gamma x -> z=E x; m=D[SiLU(Lg z) * (Lu z)] with native D and all other layers",
            "source_model_profile": "Qwen2.5-1.5B PyTorch FP32 contextual-union selected coupled64",
        },
        "query17": {
            "source_job_id": "job-b141eadfe952",
            "source_report_sha256": QUERY_REPORT_SHA256,
            "selected_image": "query-fit.f32",
            "selected_image_sha256": query_records["query-F.f32"]["sha256"],
            "basis_and_key_image_source": "same query-response predecessor; program128-basis.f32 and program128-k.f32",
            "images": query_records,
            "formula": "q=F B r; compiled K=K B r; native V/O/D and native full FP32 K/V cache",
            "fit_profile": "source-only CPU64 response fit against 16-recent/16-prototype memory",
            "serving_profile": "full native FP32 K/V cache",
            "transfer_status": "diagnostic_transfer_outside_fit_memory_profile",
            "target_fitting": False,
            "new_fit_in_this_experiment": False,
        },
        "execution_boundary": {
            "factor_arrays_sealed_before_target_tokenization": True,
            "model_weights_frozen": True,
            "native_dense_V_O_D_and_other_layers": True,
            "memory_policy": "full_fp32_native_kv_for_all_four_arms",
            "policy_bits": {"00": "native MLP16 + native query17", "10": "compiled MLP16 + native query17", "01": "native MLP16 + compiled query17", "11": "compiled MLP16 + compiled query17"},
        },
    }
    (payload / "factor-provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    sources = {str(path.relative_to(payload)): sha(path) for path in sorted(payload.rglob("*")) if path.is_file()}
    manifest = {
        "schema": "tcm-operand-composition-payload-v1",
        "sources": sources,
        "created_utc": stamp,
        "predecessor_reports": {"query_response": QUERY_REPORT_SHA256, "contextual_union": COUPLED_REPORT_SHA256},
        "raw_archive_limit_bytes": LIMIT,
        "program_seal_phase": "before-target-tokenization",
        "execution_paths": {"own": 24, "replay": 18, "total": 42},
    }
    (payload / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"payload": str(payload), "source_files": len(sources), "payload_bytes": sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}))
    return payload


def submit(payload: str | Path) -> None:
    from mrun.client.submit import launch

    payload = Path(payload).resolve()
    manifest = json.loads((payload / "source-manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        checked(payload / name, expected)
    config = {
        "schema": "tcm-operand-composition-execution-v1",
        "backend": "hf",
        "candidate_backend": "c-cublas-fp32",
        "device": "cuda",
        "dtype": "fp32",
        "max_seq_len": 512,
        "backend_options": {"physical_batch_size": 1},
        "max_retries": 0,
        "sources": manifest["sources"],
        "source_manifest_sha256": sha(payload / "source-manifest.json"),
        "raw_archive_limit_bytes": LIMIT,
        "predecessor_reports": manifest["predecessor_reports"],
        "physical_work": "one resident Qwen2.5-1.5B C model; six fresh two-turn cases; four own MLP/query factorial arms per case; three replay arms on the compiled-both IDs; 24 own + 18 replay paths; full FP32 native K/V; four readout boundaries and selected per-token traces",
        "scheduling_contract": {
            "model_loads": 1,
            "logical_paths": 42,
            "own_paths": 24,
            "replay_paths": 18,
            "branch_execution": "in_process",
            "collection_policy": "direct_job_only",
            "mrun_retry_policy": "none_for_finite_lease",
        },
    }
    job = launch(
        "uv run --offline --project {env} python worker.py",
        experiment="c-state-machine-operand-composition",
        payload=payload,
        model="qwen2.5-1.5b-instruct",
        prefer_host="beast",
        needs={"cuda": True},
        config=config,
        env_alias="mrun",
        timeout_s=840,
        detach=True,
        retry_on_kill=False,
        preflight=True,
        note="Operand composition: frozen coupled64 MLP16 x frozen response-fit query17 on full native FP32 KV; six fresh two-turn cases; 24 own + 18 replay paths; all factors sealed before target tokenization; stop after finite 42-path panel",
    )
    target = HERE / "runs" / str(job)
    target.mkdir(parents=True, exist_ok=True)
    (target / "submission.json").write_text(json.dumps({"job_id": job, "payload": str(payload), "config": config}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"job_id": job, "result_directory": str(target)}))


def collect(job: str | Path) -> None:
    from mrun.client.api import Api

    job = str(job)
    _, record_body, _ = Api().request("GET", f"/api/jobs/{job}", timeout_s=20)
    _, logs, _ = Api().request("GET", f"/api/jobs/{job}/logs?offset=0&wait_s=1", timeout_s=30)
    target = HERE / "runs" / job
    target.mkdir(parents=True, exist_ok=True)
    (target / "job.json").write_text(json.dumps(json.loads(record_body), indent=2, sort_keys=True) + "\n")
    (target / "logs.txt").write_bytes(logs)
    report = None
    for line in logs.decode("utf-8", "replace").splitlines():
        if line.startswith("OPERAND_COMPOSITION_RESULT_GZIP_B64="):
            report = json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1])))
            (target / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if line.startswith("OPERAND_COMPOSITION_FILES_TGZ_B64="):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=", 1)[1])), mode="r:gz") as archive:
                members = archive.getmembers()
                if sum(member.size for member in members) > LIMIT:
                    raise ValueError("Archive exceeds declared limit")
                for member in members:
                    relative = Path(member.name)
                    if not member.isfile() or relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"raw", "payload"}:
                        raise ValueError("Invalid evidence member")
                    data = archive.extractfile(member).read()
                    destination = target / relative
                    if destination.exists() and destination.read_bytes() != data:
                        raise ValueError("Collected immutable evidence changed")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
    print(json.dumps({"job_id": job, "state": json.loads(record_body).get("state"), "has_report": report is not None, "error": report.get("error") if report else None}))


def verify(target: str | Path) -> None:
    """Delegate saved-only verification to the owner-supplied verifier."""
    target = Path(target).resolve()
    verifier = target / "payload" / "verification.py"
    if not verifier.exists():
        verifier = target / "payload" / "audit.py"
    if not verifier.exists():
        raise FileNotFoundError(verifier)
    import importlib.util
    spec = importlib.util.spec_from_file_location("operand_composition_saved_verifier", verifier)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "verify"):
        raise AttributeError("payload/verification.py must expose verify(target)")
    result = module.verify(target)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "submit", "collect", "verify"))
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    if args.action == "stage":
        stage()
    elif args.action == "submit":
        if not args.target:
            raise SystemExit("submit requires a staged payload")
        submit(args.target)
    elif args.action == "collect":
        if not args.target:
            raise SystemExit("collect requires a job id")
        collect(args.target)
    else:
        if not args.target:
            raise SystemExit("verify requires a collected run")
        verify(args.target)
