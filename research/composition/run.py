"""One finite mrun lease and source-bound collection for the composition panel."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
MACHINE = HERE.parent
DOMAINS = MACHINE.parents[1]
PRIOR = MACHINE/"results/job-0e69e4a9b62e"
SOURCE_REPORT_SHA = "9dca57343af2638ea5579d9b0593b31301e18985125b4f51999cc1529b44ed4d"
BANK = ("reuse-program.f32", "reuse-qr.f64", "enrich-program.f32", "enrich-qr.f64")
RUNTIME = ("src/model.c", "src/qwen.c", "src/qwen.h", "src/qwen_cuda.cu", "src/gpu.h", "src/machine.h", "tests/assay_io.h")
LIMIT = 32*1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f"Source identity differs: {path}")
    return path


def stage(panel="primary"):
    report = json.loads(checked(PRIOR/"report.json", SOURCE_REPORT_SHA).read_text())
    previous = json.loads((PRIOR/"submission.json").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = HERE/"staged"/stamp
    payload = target/"payload"
    (payload/"runtime").mkdir(parents=True)
    (payload/"bank").mkdir()
    source_records = {}
    for name in RUNTIME:
        source = checked(PRIOR/"payload"/name, previous["config"]["sources"][name])
        destination = payload/"runtime"/Path(name).name
        shutil.copy2(source, destination)
        source_records[name] = sha(source)
    for name in BANK:
        record = report["raw_files"][name]
        source = checked(PRIOR/"raw"/name, record["sha256"])
        if source.stat().st_size != record["bytes"]:
            raise ValueError("Bank extent differs")
        shutil.copy2(source, payload/"bank"/name)
    for name in ("driver.c", "Makefile", "worker.py", "analysis.py", "zero_analysis.py", "PLAN.md", "ZERO_PLAN.md", "run.py"):
        shutil.copy2(HERE/name, payload/name)
    if panel == "zero":
        reference = checked(HERE/"runs/job-86cce06d33b1/report.json", "760e14432e8de8158185296eb59055d98af1a2f926aa768df81f74ca3633a62b")
        shutil.copy2(reference, payload/"zero-reference.json")
    bridge = checked(PRIOR/"payload/tools/saturn_bridge.py", previous["config"]["sources"]["tools/saturn_bridge.py"])
    shutil.copy2(bridge, payload/"saturn_bridge.py")
    shutil.copytree(PRIOR/"payload/public", payload/"public", ignore=shutil.ignore_patterns("__pycache__"))
    provenance = {"schema": "tcm-composition-source-v1", "predecessor_report_sha256": SOURCE_REPORT_SHA,
                  "predecessor_job": "job-0e69e4a9b62e", "model": report["model"],
                  "original_program_sha256": report["driver"]["program_sha256"],
                  "extension_program_sha256": report["driver"]["extension_program_sha256"],
                  "bank_files": {name: report["raw_files"][name] for name in BANK},
                  "bank_concat_sha256": hashlib.sha256(b"".join((payload/"bank"/name).read_bytes() for name in BANK)).hexdigest(),
                  "runtime_sources": source_records,
                  "source_template_text": report["input"]["text"],
                  "source_template_ids": report["input"]["ids"],
                  "historical_source_own_ids": report["driver"]["rebuilt_source_own"]["ids"],
                  "historical_source_position": len(report["input"]["ids"])+len(report["driver"]["rebuilt_source_own"]["ids"]),
                  "historical_source_kv_sha256": report["driver"]["source_answer_kv_prefix_sha256"],
                  "projector_origin": "supplied historical source/Mira calibration at layer16; no target refresh"}
    (payload/"baseline.json").write_text(json.dumps(provenance, indent=2)+"\n")
    files = {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest = {"schema": "tcm-composition-payload-v1", "sources": files, "created_utc": stamp, "panel_kind": panel,
                "plan_sha256": sha(payload/("ZERO_PLAN.md" if panel == "zero" else "PLAN.md")), "raw_archive_limit_bytes": LIMIT}
    (payload/"source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"payload": str(payload), "source_files": len(files), "payload_bytes": sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}))
    return payload


def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload/"source-manifest.json").read_text())
    zero = manifest.get("panel_kind") == "zero"
    for name, expected in manifest["sources"].items():
        checked(payload/name, expected)
    config = {"schema": "tcm-composition-execution-v1", "backend": "hf", "candidate_backend": "c-cublas-fp32",
              "device": "cuda", "dtype": "fp32", "cpu_threads": 1, "max_seq_len": 512,
              "backend_options": {"physical_batch_size": 1}, "max_retries": 0,
              "sources": manifest["sources"], "source_manifest_sha256": sha(payload/"source-manifest.json"),
              "physical_work": "one C model resident; native source own answer;3 independent new questions x4 policies; budgets8/8/48; every emitted token consumed; no training or new calibration",
              "reservation_source": "measured same C FP32 backend: formation job-0e69e4a9b62e RSS4005.1MB/VRAM6378MB;512context; existing360452-byte subspace workspace;6144MB reservation gives6758.4MB VRAM ceiling",
              "predecessor_report_sha256": SOURCE_REPORT_SHA, "raw_archive_limit_bytes": LIMIT,
              "scheduling_contract": {"model_loads": 1, "logical_branches": 12, "branch_execution": "in_process", "max_mrun_submissions": 1, "collection_policy": "direct_job_only", "mrun_retry_policy": "none_for_finite_lease"}}
    if zero:
        config["physical_work"] = "one C model resident;3 known-question recurring-zero paths,8/8/48 own-token budgets; no repeated primary paths; posthoc deletion control"
        config["scheduling_contract"]["logical_branches"] = 3
    job = launch("uv run --offline --project {env} python worker.py", experiment="c-state-machine-composition-zero" if zero else "c-state-machine-composition",
                 payload=payload, model="qwen2.5-1.5b-instruct", prefer_host="beast", pin="beast", needs={"cuda": True},
                 config=config, env_alias="mrun", ram_mb=6144, vram_mb=6144, cpu_threads=1, est_wall_s=180,
                 timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
                 note="C composition zero:3 known-question deletion controls atMLP16;8/8/48 own-token budgets inclEOS;0G/U0down atsite; originalbankimmutable;one model/batch1/CPU1;stop after3paths" if zero else "C composition: imported frozen B3,3question parents x native/once/recurring/corrected;8/8/48 own-token budgets including EOS consumption; one model/batch1/CPU1; full first/final logits and selected ports; stop after12paths")
    target = HERE/"runs"/job
    target.mkdir(parents=True, exist_ok=True)
    (target/"submission.json").write_text(json.dumps({"job_id": job, "payload": str(payload), "config": config}, indent=2)+"\n")
    print(json.dumps({"job_id": job, "result_directory": str(target)}))


def collect(job, result_root=None):
    from mrun.client.api import Api
    api = Api()
    _, record, _ = api.request("GET", f"/api/jobs/{job}", timeout_s=20)
    _, logs, _ = api.request("GET", f"/api/jobs/{job}/logs?offset=0&wait_s=1", timeout_s=30)
    target = (Path(result_root) if result_root is not None else HERE/"runs")/job
    target.mkdir(parents=True, exist_ok=True)
    (target/"job.json").write_text(json.dumps(json.loads(record), indent=2)+"\n")
    (target/"logs.txt").write_bytes(logs)
    report = None
    for line in logs.decode("utf-8", "replace").splitlines():
        if line.startswith("COMPOSITION_RESULT_GZIP_B64="):
            report = json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1])))
            (target/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        if line.startswith("COMPOSITION_FILES_TGZ_B64="):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=", 1)[1])), mode="r:gz") as archive:
                members = archive.getmembers()
                if sum(m.size for m in members) > LIMIT:
                    raise ValueError("Archive exceeds declared limit")
                for member in members:
                    relative = Path(member.name)
                    if not member.isfile() or relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("raw", "payload"):
                        raise ValueError("Invalid evidence member")
                    data = archive.extractfile(member).read()
                    path = target/relative
                    if path.exists() and path.read_bytes() != data:
                        raise ValueError("Collected immutable evidence changed")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
    print(json.dumps({"job_id": job, "state": json.loads(record).get("state"), "has_report": report is not None,
                      "summary": report.get("analysis", {}).get("summary") if report else None,
                      "error": report.get("error") if report else None}))


def verify(root, numerical_policy=None):
    root = Path(root)
    report = json.loads((root/"report.json").read_text())
    job = json.loads((root/"job.json").read_text())
    submission = json.loads((root/"submission.json").read_text())
    if job["job_id"] != submission["job_id"] or sha(root/"logs.txt") != job["result"]["artifacts"]["job_log"]["sha256"]:
        raise ValueError("Scheduler job/log identity differs")
    logged = [json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1]))) for line in (root/"logs.txt").read_text().splitlines() if line.startswith("COMPOSITION_RESULT_GZIP_B64=")]
    if logged != [report] or json.loads((root/"raw/report.json").read_text()) != report:
        raise ValueError("Report log/raw join differs")
    for name, expected in submission["config"]["sources"].items():
        checked(root/"payload"/name, expected)
    checked(root/"payload/source-manifest.json", submission["config"]["source_manifest_sha256"])
    for name, record in report["raw_files"].items():
        checked(root/"raw"/name, record["sha256"])
        if (root/"raw"/name).stat().st_size != record["bytes"]:
            raise ValueError("Raw extent differs")
    if json.loads((root/"raw/driver.json").read_text()) != report["driver"]:
        raise ValueError("Driver join differs")
    # Recompute using the exact executed analyzer, without a model or tokenizer.
    import importlib.util
    zero = report.get("panel_kind") == "zero"
    sys.path.insert(0, str((root/"payload").resolve()))
    spec = importlib.util.spec_from_file_location("composition_saved_analyzer", root/"payload"/("zero_analysis.py" if zero else "analysis.py"))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if zero:
        reference = json.loads((root/"payload/zero-reference.json").read_text())
        recalculated = module.analyze(root/"raw", root/"payload/bank", report["driver"], report["texts"], reference)
    else:recalculated = module.analyze(root/"raw", root/"payload/bank", report["driver"], report["texts"])
    def compare(a, b, path=()):
        if isinstance(a, dict):
            if a.keys() != b.keys():raise ValueError("Analysis schema differs")
            for k in a:compare(a[k], b[k], path+(k,))
        elif isinstance(a, list):
            if len(a) != len(b):raise ValueError("Analysis row count differs")
            for index, (x, y) in enumerate(zip(a, b)):compare(x, y, path+(index,))
        elif isinstance(a, float):
            import math
            if numerical_policy is not None and numerical_policy(path, a, b):return
            if not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14):raise ValueError("Analysis arithmetic differs")
        elif a != b:raise ValueError("Analysis observation differs")
    compare(recalculated, report["analysis"])
    wrapper = report["saturn_receipt"];receipt = dict(wrapper["receipt"]);fingerprint = receipt.pop("fingerprint")
    if hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != fingerprint or receipt["parent"] != wrapper["before"] or receipt["result"] != wrapper["after"]:
        raise ValueError("Saturn receipt differs")
    verification = {"schema": "tcm-composition-verification-v1", "report_sha256": sha(root/"report.json"),
                    "scheduler_log_seal": True, "source_files_verified": len(submission["config"]["sources"]),
                    "raw_files_verified": len(report["raw_files"]), "whole_invocation_receipt": True,
                    "analysis_recomputed": True, "models_evaluated": 0, "terminal_status": "not-requested"}
    (root/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    observations = []
    for row in recalculated["observations"]:
        observations.append({**row, "schema": "tcm-composition-observation-v1", "model_id": report["model"]["identity"], "report_sha256": verification["report_sha256"]})
    (root/"observations.jsonl").write_text("".join(json.dumps(x, sort_keys=True, allow_nan=False)+"\n" for x in observations))
    print(json.dumps(verification))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("action", choices=("stage", "stage-zero", "submit", "collect", "verify"));parser.add_argument("target", nargs="?");args = parser.parse_args()
    if args.action == "stage":stage()
    elif args.action == "stage-zero":stage("zero")
    elif args.action == "submit":submit(args.target)
    elif args.action == "collect":collect(args.target)
    else:verify(args.target)
