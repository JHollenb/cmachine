"""Source-frozen, finite C operand-basis mrun submission and collection."""
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
    predecessor = MACHINE / "results/job-0e69e4a9b62e"
    historical = json.loads(base.checked(predecessor / "report.json", base.SOURCE_REPORT_SHA).read_text())
    submission = json.loads((predecessor / "submission.json").read_text())
    payload = HERE / "staged" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") / "payload"
    (payload / "runtime/src").mkdir(parents=True)
    (payload / "runtime/tests").mkdir()
    (payload / "bank").mkdir()
    for source in sorted((MACHINE / "src").iterdir()):
        if source.suffix in (".c", ".h", ".cu"):
            shutil.copy2(source, payload / "runtime/src" / source.name)
    for source in sorted((MACHINE / "tests").iterdir()):
        if source.suffix in (".c", ".h"):
            shutil.copy2(source, payload / "runtime/tests" / source.name)
    shutil.copy2(MACHINE / "Makefile", payload / "runtime/Makefile")
    for source in sorted(HERE.iterdir()):
        if source.is_file() and (source.suffix in (".py", ".c", ".h", ".cu") or source.name in ("Makefile", "PLAN.md", "ADVISOR.md")):
            shutil.copy2(source, payload / source.name)
    shutil.copy2(MACHINE / "composition/run.py", payload / "support_run.py")
    bridge = base.checked(predecessor / "payload/tools/saturn_bridge.py", submission["config"]["sources"]["tools/saturn_bridge.py"])
    shutil.copy2(bridge, payload / "saturn_bridge.py")
    for source in sorted((predecessor / "payload/public").rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts:
            relative = source.relative_to(predecessor / "payload")
            base.checked(source, submission["config"]["sources"][str(relative)])
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in base.BANK:
        source = base.checked(predecessor / "raw" / name, historical["raw_files"][name]["sha256"])
        shutil.copy2(source, payload / "bank" / name)
    baseline = {"schema": "tcm-operand-basis-baseline-v1", "model": historical["model"],
                "bank_files": {name: historical["raw_files"][name] for name in base.BANK},
                "bank_concat_sha256": hashlib.sha256(b"".join((payload / "bank" / name).read_bytes() for name in base.BANK)).hexdigest(),
                "bank_predecessor_job": "job-0e69e4a9b62e", "bank_predecessor_report_sha256": base.SOURCE_REPORT_SHA,
                "bank_origin": "historical supplied source-activation calibration; unchanged comparator, no fresh calibration"}
    (payload / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    sources = {str(p.relative_to(payload)): base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest = {"schema": "tcm-operand-basis-payload-v1", "lane": lane, "sources": sources,
                "raw_archive_limit_bytes": LIMIT, "rank": 64, "oversampling": 16, "power_iterations": 2,
                "coordinate": "raw-receiver-with-learned-RMS-gamma-folded-into-source-maps",
                "primary_from_first_prompt_token": True}
    (payload / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"payload": str(payload), "lane": lane, "sources": len(sources)}))
    return payload


def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve()
    manifest = json.loads((payload / "source-manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        base.checked(payload / name, digest)
    smoke = manifest["lane"] == "smoke"
    config = {"schema": "tcm-operand-basis-execution-v1", "backend": "hf", "candidate_backend": "c-cublas-fp32",
              "device": "cuda", "dtype": "float32", "cpu_threads": 1, "max_seq_len": 512,
              "backend_options": {"physical_batch_size": 1}, "max_retries": 0,
              "sources": manifest["sources"], "source_manifest_sha256": base.sha(payload / "source-manifest.json"),
              "raw_archive_limit_bytes": LIMIT, "rank": 64, "oversampling": 16, "power_iterations": 2,
              "reservation_source": "Measured operand-basis smoke job-23dd9b4d0de9 B1/context512 CPU1, one C model plus CPU compiler and two banks: RSS4148.6MB/VRAM6394MB. Same allocated context and bank geometry for panel;6144MB reservation gives6758.4MB VRAM ceiling",
              "physical_work": "isolated native adapter mechanics and reduced-budget panel" if smoke else "3 two-turn conversations x4 primary policies from first input;12 paths sealed before3 native and postseal same-ID/local diagnostics;16 tokens per answer; original G/U forbidden in source/random primary",
              "scheduling_contract": {"model_loads_scientific_driver": 1, "branch_execution": "in_process", "max_mrun_submissions": 1,
                                      "collection_policy": "direct_job_only", "mrun_retry_policy": "none_for_finite_lease"}}
    job = launch("uv run --offline --project {env} python worker.py", experiment="c-state-machine-operand-basis-" + manifest["lane"],
                 payload=payload, model="qwen2.5-1.5b-instruct", prefer_host="beast", pin="beast", needs={"cuda": True},
                 config=config, env_alias="mrun", ram_mb=6144, vram_mb=6144, cpu_threads=1,
                 est_wall_s=120 if smoke else 300, timeout_s=840, detach=True, retry_on_kill=False, preflight=True,
                 note="Operand basis smoke: isolated C adapter, one resident FP32 model, CPU source-only rank64 compilation, bounded mechanics; B1 CPU1 context512, no retries" if smoke else "Operand basis:3 two-turn cases xsource64/random64/B3/zero from first token,16 tokens/answer; seal12 primary worlds before native diagnostics; one C FP32 model, CPU compiler, B1 CPU1 context512, no retries")
    target = HERE / "runs" / job
    target.mkdir(parents=True, exist_ok=True)
    (target / "submission.json").write_text(json.dumps({"job_id": job, "payload": str(payload), "config": config}, indent=2) + "\n")
    print(json.dumps({"job_id": job, "result_directory": str(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage-smoke", "stage-panel", "submit", "collect", "verify"))
    parser.add_argument("target", nargs="?")
    args = parser.parse_args()
    if args.action.startswith("stage-"):
        stage(args.action.split("-", 1)[1])
    elif args.action == "submit":
        submit(args.target)
    elif args.action == "collect":
        base.LIMIT = LIMIT
        base.collect(args.target, HERE / "runs")
    else:
        base.verify(args.target)
