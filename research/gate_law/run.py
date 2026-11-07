"""Bounded, source-frozen gate-law mrun lanes and authenticated collection."""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
MACHINE = HERE.parent
LIMIT = 256 * 1024**2
spec = importlib.util.spec_from_file_location("composition_collection", MACHINE / "composition/run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def stage(lane):
    previous = MACHINE / "results/job-11cdb86ad29b"
    baseline = json.loads((previous / "report.json").read_text())
    submission = json.loads((previous / "submission.json").read_text())
    payload = HERE / "staged" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") / "payload"
    (payload / "runtime/src").mkdir(parents=True)
    (payload / "runtime/tests").mkdir()
    for source in sorted((MACHINE / "src").iterdir()):
        if source.suffix in (".c", ".h", ".cu"):
            shutil.copy2(source, payload / "runtime/src" / source.name)
    for name in ("state_mechanics.c", "writer_mechanics.h", "gate_mechanics.h", "assay_io.h"):
        shutil.copy2(MACHINE / "tests" / name, payload / "runtime/tests" / name)
    shutil.copy2(MACHINE / "Makefile", payload / "runtime/Makefile")
    for name in ("run.py", "worker.py", "analysis.py", "driver.c", "Makefile", "PLAN.md", "test_analysis.py"):
        shutil.copy2(HERE / name, payload / name)
    shutil.copy2(MACHINE / "composition/run.py", payload / "support_run.py")
    shutil.copy2(MACHINE / "tools/saturn_bridge.py", payload / "saturn_bridge.py")
    for source in sorted((previous / "payload/public").rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts:
            relative = source.relative_to(previous / "payload")
            base.checked(source, submission["config"]["sources"][str(relative)])
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (payload / "baseline.json").write_text(json.dumps({"model": baseline["model"], "engineering_predecessor_report_sha256": base.sha(previous / "report.json")}, indent=2) + "\n")
    sources = {str(p.relative_to(payload)): base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    (payload / "source-manifest.json").write_text(json.dumps({"schema": "tcm-gate-law-payload-v1", "lane": lane, "sources": sources, "raw_archive_limit_bytes": LIMIT}, indent=2) + "\n")
    print(json.dumps({"payload": str(payload), "lane": lane, "source_files": len(sources)}))
    return payload

def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload / "source-manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        base.checked(payload / name, digest)
    smoke = manifest["lane"] == "smoke"
    config = {"schema": "tcm-gate-law-execution-v1", "backend": "hf", "candidate_backend": "c-cublas-fp32", "device": "cuda", "dtype": "fp32", "cpu_threads": 1, "max_seq_len": 512, "backend_options": {"physical_batch_size": 1}, "max_retries": 0,
              "sources": manifest["sources"], "source_manifest_sha256": base.sha(payload / "source-manifest.json"), "raw_archive_limit_bytes": LIMIT,
              "reservation_source": "same C FP32 capacity512 job-11cdb86ad29b: RSS4121.1MB, VRAM6378MB; source law179200 bytes per anchor;6144MB reservation x1.1 VRAM ceiling6758.4MB",
              "physical_work": "pretrained lifecycle plus one historical/current ordinary-output parity case" if smoke else "two development conversations;6 primary own-token paths sealed before2native paths,6same-ID native replays,6candidate replays,12same-cache native steps and saved-operand SiLU/down diagnostics;16tokens/answer,2answers/path;no target reanchoring",
              "scheduling_contract": {"model_loads_scientific_driver": 0 if smoke else 1, "branch_execution": "in_process", "collection_policy": "direct_job_only", "max_retries": 0}}
    job = launch("uv run --offline --project {env} python worker.py", experiment="c-state-machine-gate-law-" + manifest["lane"], payload=payload, model="qwen2.5-1.5b-instruct", prefer_host="beast", pin="beast", needs={"cuda": True}, config=config, env_alias="mrun", ram_mb=6144, vram_mb=6144, cpu_threads=1, est_wall_s=60 if smoke else 240, timeout_s=600, detach=True, retry_on_kill=False, preflight=True,
                 note="Gate-law smoke: compile explicit C FP32 hook, real pretrained lifecycle/anchor identity/native no-op/lineage, old-new ordinary output; stop after mechanics+2ordinary paths" if smoke else "Gate-law:2conversations x once/recurring/zero,2answers16tokens;all consumed headers/turns;seal6primary paths before native references and exact diagnostics;one C FP32 model B1 CPU1 context512; finite discovery,no retries")
    target = HERE / "runs" / job
    target.mkdir(parents=True, exist_ok=True)
    (target / "submission.json").write_text(json.dumps({"job_id": job, "payload": str(payload), "config": config}, indent=2) + "\n")
    print(json.dumps({"job_id": job, "result_directory": str(target)}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("stage-smoke", "stage-panel", "submit", "collect", "verify"))
    p.add_argument("target", nargs="?")
    a = p.parse_args()
    if a.action.startswith("stage-"):
        stage(a.action.split("-", 1)[1])
    elif a.action == "submit":
        submit(a.target)
    elif a.action == "collect":
        base.LIMIT = LIMIT
        base.collect(a.target, HERE / "runs")
    else:
        base.verify(a.target)
