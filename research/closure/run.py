"""Stage, execute once through mrun, collect and verify a bounded closure cut."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "composition/runs/job-86cce06d33b1"
SOURCE_SHA = "760e14432e8de8158185296eb59055d98af1a2f926aa768df81f74ca3633a62b"
LIMIT = 32 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f"source identity differs: {path}")
    return path


def stage():
    from geometry import generate
    report = json.loads(checked(SOURCE / "report.json", SOURCE_SHA).read_text())
    submission = json.loads((SOURCE / "submission.json").read_text())
    for name, expected in submission["config"]["sources"].items():
        checked(SOURCE / "payload" / name, expected)
    for task in ("recall", "math", "prose"):
        name = f"{task}-native-s000-raw_receiver.f32"
        checked(SOURCE / "raw" / name, report["raw_files"][name]["sha256"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    payload = HERE / "staged" / stamp / "payload"
    payload.mkdir(parents=True)
    for name in ("runtime", "bank", "public"):
        shutil.copytree(SOURCE / "payload" / name, payload / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(SOURCE / "payload/driver.c", payload / "composition_driver.c")
    shutil.copy2(SOURCE / "payload/saturn_bridge.py", payload / "saturn_bridge.py")
    shutil.copy2(SOURCE / "report.json", payload / "source-reference.json")
    for name in ("driver.c", "Makefile", "worker.py", "run.py", "analyze.py", "geometry.py", "PLAN.md"):
        shutil.copy2(HERE / name, payload / name)
    geometry = generate(SOURCE / "raw", SOURCE / "payload/bank", payload / "geometry")
    (payload / "geometry/geometry.json").write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    files = {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest = {"schema": "tcm-state-closure-payload-v1", "created_utc": stamp, "sources": files,
                "source_report_sha256": SOURCE_SHA, "raw_archive_limit_bytes": LIMIT}
    (payload / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"payload": str(payload), "files": len(files), "bytes": sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}))
    return payload


def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload / "source-manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        checked(payload / name, expected)
    config = {"schema": "tcm-state-closure-execution-v1", "backend": "hf", "candidate_backend": "c-cublas-fp32",
              "device": "cuda", "dtype": "fp32", "cpu_threads": 1, "max_seq_len": 512,
              "backend_options": {"physical_batch_size": 1}, "max_retries": 0,
              "sources": manifest["sources"], "source_manifest_sha256": sha(payload / "source-manifest.json"),
              "physical_work": "one resident C model;192 source-only encoder candidates;3 question parents x up to8policies x up to9token forwards; no calibration; fixed-skip held-write consumers",
              "reservation_source": "measured identical C FP32 model/context backend: composition job-86cce06d33b1 RSS3973.3MB VRAM6378MB; additions under2MB;6144MB reservation/6758.4MB VRAM guard",
              "scheduling_contract": {"model_loads": 1, "branch_execution": "in_process", "max_mrun_submissions": 1,
                                      "collection_policy": "direct_job_only", "mrun_retry_policy": "none"}}
    job = launch("uv run --offline --project {env} python worker.py", experiment="c-state-closure",
                 payload=payload, model="qwen2.5-1.5b-instruct", prefer_host="beast", pin="beast", needs={"cuda": True},
                 config=config, env_alias="mrun", ram_mb=6144, vram_mb=6144, cpu_threads=1,
                 est_wall_s=180, timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
                 note="Local-state closure: exact16byte B3 encoder collision search before outcomes;3known receivers, up to24 own-token paths/8tokens+question step; one original C FP32 model/batch1/CPU1; fixed residual skip; stop after declared branches; no training")
    target = HERE / "runs" / job
    target.mkdir(parents=True, exist_ok=True)
    (target / "submission.json").write_text(json.dumps({"job_id": job, "payload": str(payload), "config": config}, indent=2) + "\n")
    print(json.dumps({"job_id": job, "result_directory": str(target)}))


def collect(job):
    from mrun.client.api import Api
    api = Api()
    _, raw_job, _ = api.request("GET", f"/api/jobs/{job}", timeout_s=20)
    _, logs, _ = api.request("GET", f"/api/jobs/{job}/logs?offset=0&wait_s=1", timeout_s=30)
    record = json.loads(raw_job)
    target = HERE / "runs" / job
    target.mkdir(parents=True, exist_ok=True)
    (target / "job.json").write_text(json.dumps(record, indent=2) + "\n")
    (target / "logs.txt").write_bytes(logs)
    report = None
    for line in logs.decode("utf-8", "replace").splitlines():
        if line.startswith("CLOSURE_RESULT_GZIP_B64="):
            report = json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1])))
            (target / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        elif line.startswith("CLOSURE_FILES_TGZ_B64="):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=", 1)[1])), mode="r:gz") as archive:
                members = archive.getmembers()
                if sum(m.size for m in members) > LIMIT:
                    raise ValueError("archive exceeds declared limit")
                for member in members:
                    relative = Path(member.name)
                    if not member.isfile() or relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("raw", "payload"):
                        raise ValueError("invalid member")
                    data = archive.extractfile(member).read()
                    path = target / relative
                    if path.exists() and path.read_bytes() != data:
                        raise ValueError("immutable collected bytes differ")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
    print(json.dumps({"job_id": job, "state": record.get("state"), "has_report": report is not None,
                      "summary": report.get("analysis", {}).get("summary") if report else None}))


def verify(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    job = json.loads((root / "job.json").read_text())
    submission = json.loads((root / "submission.json").read_text())
    if job["job_id"] != submission["job_id"] or sha(root / "logs.txt") != job["result"]["artifacts"]["job_log"]["sha256"]:
        raise ValueError("scheduler/log join differs")
    logged = [json.loads(gzip.decompress(base64.b64decode(line.split("=", 1)[1])))
              for line in (root / "logs.txt").read_text().splitlines() if line.startswith("CLOSURE_RESULT_GZIP_B64=")]
    if logged != [report] or json.loads((root / "raw/report.json").read_text()) != report:
        raise ValueError("report/log/raw join differs")
    for name, expected in submission["config"]["sources"].items():
        checked(root / "payload" / name, expected)
    checked(root / "payload/source-manifest.json", submission["config"]["source_manifest_sha256"])
    for name, item in report["raw_files"].items():
        checked(root / "raw" / name, item["sha256"])
        if (root / "raw" / name).stat().st_size != item["bytes"]:
            raise ValueError("raw extent differs")
    if json.loads((root / "raw/driver.json").read_text()) != report["driver"]:
        raise ValueError("driver join differs")
    source = json.loads(checked(root / "payload/source-reference.json", SOURCE_SHA).read_text())
    if report["model"] != source["model"] or report["driver"]["model_digest"] != source["model"]["content_sha256"]:
        raise ValueError("actual loaded model/source identity differs")
    expected_bank = source["baseline"]["bank_concat_sha256"]
    if report["driver"]["frozen_bank_before_sha256"] != expected_bank or report["driver"]["frozen_bank_after_sha256"] != expected_bank:
        raise ValueError("actual frozen bank/source identity differs")
    if report["geometry"] != json.loads((root / "payload/geometry/geometry.json").read_text()):
        raise ValueError("report geometry/payload join differs")
    sys.path.insert(0, str((root / "payload").resolve()))
    spec = importlib.util.spec_from_file_location("saved_closure_analyzer", root / "payload/analyze.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    recalculated = module.analyze(root / "raw", root / "payload/bank", report["driver"], report["texts"], report["geometry"])
    def compare(a, b):
        if isinstance(a, dict):
            if a.keys() != b.keys(): raise ValueError("analysis fields differ")
            for key in a: compare(a[key], b[key])
        elif isinstance(a, list):
            if len(a) != len(b): raise ValueError("analysis length differs")
            for x, y in zip(a, b): compare(x, y)
        elif isinstance(a, float):
            import math
            if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14): raise ValueError("analysis calculation differs")
        elif a != b: raise ValueError("analysis observation differs")
    compare(recalculated, report["analysis"])
    wrapper = report["saturn_receipt"]
    receipt = dict(wrapper["receipt"])
    fingerprint = receipt.pop("fingerprint")
    if hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != fingerprint or receipt["parent"] != wrapper["before"] or receipt["result"] != wrapper["after"]:
        raise ValueError("Saturn receipt differs")
    result = {"schema": "tcm-state-closure-verification-v1", "report_sha256": sha(root / "report.json"),
              "scheduler_log_seal": True, "source_files_verified": len(submission["config"]["sources"]),
              "raw_files_verified": len(report["raw_files"]), "analysis_recomputed": True,
              "actual_loaded_model_verified": True, "bank_and_geometry_verified": True,
              "models_evaluated": 0, "terminal_status": "not-requested"}
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    observations = [{**row, "model_id": report["model"]["identity"], "report_sha256": result["report_sha256"]}
                    for row in recalculated["observations"]]
    (root / "observations.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in observations))
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "submit", "collect", "verify"))
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    if args.action == "stage": stage()
    elif args.action == "submit": submit(args.target)
    elif args.action == "collect": collect(args.target)
    else: verify(args.target)
