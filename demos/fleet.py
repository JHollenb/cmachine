"""Domains-only mrun launcher for the public pretrained demo programs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import gzip
import hashlib
import io
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "demos"
LIMIT = 12 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_exact(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Collected evidence changed: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def stage(model, tokenizer):
    payload = HERE / "staged" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") / "payload"
    payload.mkdir(parents=True)
    names = ["Makefile", "demos/fleet_worker.py", "validation/saturn_bridge.py"]
    for folder in ("src", "demos/chat_branch", "demos/compiled_operands", "validation/vendor/saturn_pub"):
        names += [str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*")
                  if p.is_file() and p.suffix in {".c", ".h", ".cu", ".inc", ".py"} and "__pycache__" not in p.parts]
    sources = {}
    for name in sorted(set(names)):
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / name).read_bytes()
        path.write_bytes(data)
        sources[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    model_record = json.loads((ROOT / "docs/evidence/integration-7a6824d/report.json").read_text())["model"]
    model_record["path"] = str(model)
    record = {"schema": "cmachine-demo-sources-v1", "sources": sources,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "model": model_record, "tokenizer_path": str(tokenizer), "archive_limit_bytes": LIMIT,
              "scope": "Frozen canonical source plus demonstration programs; no model bytes in payload."}
    (payload / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(payload)
    return payload


def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload / "source-manifest.json").read_text())
    for name, row in manifest["sources"].items():
        if sha(payload / name) != row["sha256"]:
            raise ValueError("Staged source changed: " + name)
    config = {"schema": "cmachine-public-demos-v1", "backend": "hf", "device": "cuda", "dtype": "fp32",
              "execution_backend": "canonical-c-cuda-cuBLAS", "cpu_threads": 1, "max_seq_len": 768,
              "backend_options": {"physical_batch_size": 1}, "max_retries": 0,
              "source_manifest_sha256": sha(payload / "source-manifest.json"),
              "sources": manifest["sources"], "model_digest": manifest["model"]["content_sha256"],
              "physical_work": "One native chat branch/replay process then one lexical source/candidate process; B1; model loads sequential; no sweep or retries.",
              "prior_resource_evidence": "Native-inclusive integration job-f645fcd76a75 peaked6396MBVRAM/4206.6MBRSS; demos retain one native model at a time and smaller state. Intent sizing retained."}
    job = launch("uv run --offline --project {env} python demos/fleet_worker.py", experiment="cmachine-public-demos",
                 model="qwen2.5-1.5b-instruct", payload=payload, prefer_host="beast", pin="beast", needs={"cuda": True},
                 config=config, timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note="Public cmachine demos: build current C/CUDA; native branch/replay then frozen first-layer lexical reuse, sequential B1/CPU1. Host-local authenticated TCM/tokenizer and sm89. Stop two demos; no retries/installation/speed claim.")
    result = HERE / "fleet-runs" / job
    result.mkdir(parents=True, exist_ok=True)
    (result / "submission.json").write_text(json.dumps({"job_id": job, "payload": str(payload), "config": config}, indent=2) + "\n")
    print(json.dumps({"job_id": job, "result": str(result)}))


def collect(job):
    from mrun.client.api import Api
    api = Api()
    _, data, _ = api.request("GET", f"/api/jobs/{job}", timeout_s=20)
    state = json.loads(data)
    result = HERE / "fleet-runs" / job
    result.mkdir(parents=True, exist_ok=True)
    (result / "job.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["state"] not in {"succeeded", "failed", "killed", "killed_vram", "killed_ram", "cancelled"}:
        print(json.dumps({"job_id": job, "state": state["state"]}))
        return
    _, log, _ = api.request("GET", f"/api/jobs/{job}/logs?offset=0&wait_s=1", timeout_s=30)
    _, receipt, _ = api.request("GET", f"/api/jobs/{job}/receipt", timeout_s=20)
    (result / "logs.txt").write_bytes(log)
    (result / "guarded-receipt.json").write_bytes(receipt)
    for line in log.decode().splitlines():
        if line.startswith("CMACHINE_DEMO_REPORT="):
            save_exact(result / "report.json", gzip.decompress(base64.b64decode(line.split("=", 1)[1])) + b"\n")
        elif line.startswith("CMACHINE_DEMO_FILES="):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=", 1)[1])), mode="r:gz") as archive:
                members = archive.getmembers()
                if sum(m.size for m in members) > LIMIT:
                    raise ValueError("Evidence exceeds declared limit")
                for member in members:
                    path = Path(member.name)
                    if (not member.isfile() or path.is_absolute() or ".." in path.parts
                            or not path.parts or path.parts[0] not in {"raw", "payload"}):
                        raise ValueError("Unsafe evidence member")
                    target = result / path
                    save_exact(target, archive.extractfile(member).read())
    print(json.dumps({"job_id": job, "state": state["state"], "has_report": (result / "report.json").exists()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit", "collect"))
    parser.add_argument("target", nargs="?")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    if args.action == "stage":
        if not args.model_path or not args.tokenizer_path:
            parser.error("stage requires host-local --model-path and --tokenizer-path")
        stage(args.model_path, args.tokenizer_path)
    elif not args.target:
        parser.error("target payload directory or job ID required")
    elif args.action == "submit":
        submit(args.target)
    else:
        collect(args.target)


if __name__ == "__main__":
    main()
