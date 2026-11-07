"""Freeze, dispatch and authenticate the memory-response development panel."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]
PRIOR = HERE.parent / "composition/compact_append/runs/job-2a28feeb7e5b"
PRIOR_SHA = "5d046f5b56574dafadf4422af9cd855089da1c2df99a8039efb2573db042ab4a"
BLOG = DOMAINS / "obsidian/blog/2026-11-06-154253-a-hypothesis-for-cheaper-model-state-transitions.md"
LIMIT = 256 * 1024**2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError("Identity differs: " + str(path))
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stage(lane):
    previous = json.loads(checked(PRIOR / "report.json", PRIOR_SHA).read_text())
    old_sources = json.loads((PRIOR / "submission.json").read_text())["config"]["sources"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    payload = HERE / "staged" / stamp / "payload"
    payload.mkdir(parents=True)
    for name, digest in old_sources.items():
        if name.startswith("public/") or name == "saturn_bridge.py":
            target = payload / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checked(PRIOR / "payload" / name, digest), target)
        if name.startswith("runtime/"):
            target = payload / "ancestor_runtime" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checked(PRIOR / "payload" / name, digest), target)
    for path in HERE.iterdir():
        if path.is_file() and (path.suffix in (".py", ".c", ".h", ".cu", ".md", ".json") or path.name == "Makefile"):
            shutil.copy2(path, payload / path.name)
    for path in (HERE / "runtime").iterdir():
        if path.is_file():
            target = payload / "runtime" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    required = ("driver.c", "cache.h", "audit.h", "cache_cuda.cu", "Makefile", "worker.py", "analysis.py", "PLAN.md", "inputs.json")
    if not all((payload / name).is_file() for name in required):
        raise ValueError("Implementation is incomplete")
    shutil.copy2(BLOG, payload / "HYPOTHESIS.md")
    write_json(payload / "model.json", previous["model"])
    write_json(payload / "lane.json", {"lane": lane, "cases": [0] if lane == "smoke" else [0, 1, 2, 3]})
    sources = {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest = {"schema": "tcm-memory-response-payload-v1", "created_utc": stamp, "lane": lane,
                "sources": sources, "plan_sha256": sha(payload / "PLAN.md"),
                "hypothesis_sha256": sha(payload / "HYPOTHESIS.md"), "inputs_sha256": sha(payload / "inputs.json"),
                "predecessor_report_sha256": PRIOR_SHA, "raw_archive_limit_bytes": LIMIT}
    write_json(payload / "source-manifest.json", manifest)
    print(json.dumps({"payload": str(payload), "lane": lane, "source_files": len(sources),
                      "payload_bytes": sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}))
    return payload


def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload / "source-manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        checked(payload / name, expected)
    count = 1 if manifest["lane"] == "smoke" else 4
    config = {"schema": "tcm-memory-response-execution-v1", "backend": "hf", "candidate_backend": "c-cublas-fp32",
              "device": "cuda", "dtype": "fp32", "cpu_threads": 1, "max_seq_len": 512,
              "backend_options": {"physical_batch_size": 1}, "max_retries": 0,
              "sources": manifest["sources"], "source_manifest_sha256": sha(payload / "source-manifest.json"),
              "predecessor_report_sha256": PRIOR_SHA, "raw_archive_limit_bytes": LIMIT,
              "lane": manifest["lane"], "physical_work": f"one resident C model;{count} binding/update corners x5policies x2independent query forks; empty bootstrap; native maps; bytefull/top4/recent4 plus native-history deletion; bounded own IDs; no fitting",
              "reservation_source": "same C FP32 CUDA compact_append job-2a28feeb7e5b RSS4060.1MB/VRAM6378MB; B1 capacity512; callback parity temporarily allocates two states against one weight bank (31.15MB beyond prior state); audit arrays host only;6144MB gives6758.4MB guard ceiling",
              "scheduling_contract": {"model_loads": 1, "physical_batch_size": 1, "case_count": count,
                  "policies": 5, "question_forks": 2, "branch_execution": "in_process",
                  "max_mrun_submissions": 1, "collection_policy": "direct_job_only", "mrun_retry_policy": "none_for_finite_lease"}}
    job = launch("uv run --offline --project {env} python worker.py", experiment="c-state-machine-memory-response",
                 payload=payload, model="qwen2.5-1.5b-instruct", prefer_host="beast", pin="beast",
                 needs={"cuda": True}, config=config, env_alias="mrun", ram_mb=6144, vram_mb=6144,
                 cpu_threads=1, est_wall_s=180 if count == 1 else 480, timeout_s=840,
                 detach=True, retry_on_kill=False, preflight=True,
                 note=f"Memory response {manifest['lane']}: {count} binding/update corners; 5 all-layer policies; 2 independent queries each; native maps B1 CPU1; own ack/answer tokens; byte append/top4/recent4 and history deletion; stop frozen panel; no retries")
    root = HERE / "runs" / job
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "submission.json", {"job_id": job, "payload": str(payload), "config": config})
    print(json.dumps({"job_id": job, "result_directory": str(root)}))


def collect(job):
    from mrun.client.api import Api
    api = Api()
    _, data, _ = api.request("GET", f"/api/jobs/{job}", timeout_s=20)
    _, logs, _ = api.request("GET", f"/api/jobs/{job}/logs?offset=0&wait_s=1", timeout_s=30)
    record = json.loads(data)
    root = HERE / "runs" / job
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "job.json", record)
    (root / "logs.txt").write_bytes(logs)
    report = None
    for line in logs.decode("utf-8", "replace").splitlines():
        if line.startswith("COMPOSITION_RESULT_GZIP_B64="):
            report = json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1])))
            write_json(root / "report.json", report)
        if line.startswith("COMPOSITION_FILES_TGZ_B64="):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=", 1)[1])), mode="r:gz") as archive:
                members = archive.getmembers()
                if sum(m.size for m in members) > LIMIT:
                    raise ValueError("Archive exceeds declared bound")
                for member in members:
                    rel = Path(member.name)
                    if not member.isfile() or rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in ("raw", "payload"):
                        raise ValueError("Unsafe evidence member")
                    target = root / rel
                    value = archive.extractfile(member).read()
                    if target.exists() and target.read_bytes() != value:
                        raise ValueError("Collected immutable evidence differs")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(value)
    print(json.dumps({"job_id": job, "state": record.get("state"), "has_report": report is not None,
                      "summary": report.get("analysis", {}).get("summary") if report else None,
                      "error": report.get("error") if report else None}))


def verify(root):
    root = Path(root).resolve()
    report = json.loads((root / "report.json").read_text())
    job = json.loads((root / "job.json").read_text())
    submitted = json.loads((root / "submission.json").read_text())
    if job["job_id"] != submitted["job_id"] or sha(root / "logs.txt") != job["result"]["artifacts"]["job_log"]["sha256"]:
        raise ValueError("Scheduler/log identity differs")
    logged = [json.loads(gzip.decompress(base64.b64decode(s.split("=", 1)[1]))) for s in (root / "logs.txt").read_text().splitlines() if s.startswith("COMPOSITION_RESULT_GZIP_B64=")]
    if logged != [report] or json.loads((root / "raw/report.json").read_text()) != report:
        raise ValueError("Report/log/raw join differs")
    for field in ("sources", "source_manifest_sha256", "lane", "predecessor_report_sha256"):
        if job["config"][field] != submitted["config"][field]:
            raise ValueError("Scheduler request differs: " + field)
    for name, digest in submitted["config"]["sources"].items():
        checked(root / "payload" / name, digest)
    checked(root / "payload/source-manifest.json", submitted["config"]["source_manifest_sha256"])
    for name, record in report["raw_files"].items():
        path = checked(root / "raw" / name, record["sha256"])
        if path.stat().st_size != record["bytes"]:
            raise ValueError("Raw extent differs")
    if json.loads((root / "raw/driver.json").read_text()) != report["driver"]:
        raise ValueError("Driver join differs")
    seal = json.loads((root / "raw/primary-seal.json").read_text())
    if seal["model_calls_after_seal"] != 0:
        raise ValueError("Analysis escaped its saved-only boundary")
    for name, sealed in seal["files"].items():
        if report["raw_files"].get(name) != sealed:
            raise ValueError("Primary evidence seal differs: " + name)
    for path in report["driver"]["cases"]:
        if not path["rollback_parent_exact"] or path["after_rollback_cache_sha256"] != path["query_parent_cache_sha256"]:
            raise ValueError("Query parent rollback differs")
        if path["policy"] == "history_deleted":
            if not path["native_prefix_control"]["exact_logits_each_boundary"] or path["native_reference_cache_sha256"] != path["callback_reference_cache_sha256"]:
                raise ValueError("Native callback reference differs")
        if path["case_id"] == 0:
            for field in ("tested", "sibling_changed_id", "replay_logits_exact", "replay_cache_exact", "partial_abort_parent_exact"):
                if path["mechanics"][field] is not True:
                    raise ValueError("Declared replay/abort mechanics differs")
    sys.path.insert(0, str(root / "payload"))
    spec = importlib.util.spec_from_file_location("memory_response_saved_analysis", root / "payload/analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    again = module.analyze(root / "raw", root / "payload", report["driver"], report["texts"])
    arithmetic_differences = []
    original_tolerance_refusals = []
    def compare(a, b, path="analysis"):
        if isinstance(a, dict):
            if a.keys() != b.keys(): raise ValueError("Schema differs at " + path)
            for key in a: compare(a[key], b[key], path + "." + str(key))
        elif isinstance(a, list):
            if len(a) != len(b): raise ValueError("Count differs at " + path)
            for i, (left, right) in enumerate(zip(a, b)): compare(left, right, path + "." + str(i))
        elif isinstance(a, float):
            if not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite or nonnumeric arithmetic at " + path)
            if a != b:
                item = {"path": path, "recomputed": a, "recorded": b, "absolute_difference": abs(a-b)}
                arithmetic_differences.append(item)
                if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14):
                    original_tolerance_refusals.append(item)
        elif a != b: raise ValueError("Observation differs at " + path)
    compare(again, report["analysis"])
    wrapper = report["saturn_receipt"]
    receipt = dict(wrapper["receipt"])
    fingerprint = receipt.pop("fingerprint")
    if hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != fingerprint or receipt["parent"] != wrapper["before"] or receipt["result"] != wrapper["after"]:
        raise ValueError("Saturn receipt differs")
    verification = {"schema": "tcm-memory-response-verification-v1", "report_sha256": sha(root / "report.json"),
        "source_files_verified": len(submitted["config"]["sources"]), "raw_files_verified": len(report["raw_files"]),
        "scheduler_log_seal": True, "analysis_recomputed": True, "whole_invocation_receipt": True,
        "analysis_schema_and_discrete_outputs_exact": True,
        "analysis_float_exact": not arithmetic_differences,
        "analysis_float_fields_different": len(arithmetic_differences),
        "analysis_maximum_absolute_difference": max((x["absolute_difference"] for x in arithmetic_differences), default=0),
        "analysis_original_tolerance_passed": not original_tolerance_refusals,
        "analysis_original_tolerance_refusals": original_tolerance_refusals,
        "analysis_numeric_policy": "FP64 cross-platform differences are retained diagnostics; exact schemas, discrete observations, raw arrays, source and native C replay remain integrity checks. No claim of cross-platform bit-identical derived arithmetic.",
        "models_evaluated": 0, "terminal_status": "not-requested"}
    write_json(root / "verification.json", verification)
    (root / "verification-source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(verification))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage-smoke", "stage-panel", "submit", "collect", "verify"))
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    if args.action.startswith("stage-"): stage(args.action.split("-", 1)[1])
    elif args.action == "submit": submit(args.target)
    elif args.action == "collect": collect(args.target)
    else: verify(args.target)
