"""Stage a bounded inherited runtime; submit once and reuse verified collection."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location("prior_composition_run",PARENT/"run.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMIT=96*1024**2

def stage():
    report=json.loads(base.checked(base.PRIOR/"report.json",base.SOURCE_REPORT_SHA).read_text())
    previous=json.loads((base.PRIOR/"submission.json").read_text())
    payload=HERE/"staged"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")/"payload"
    (payload/"runtime").mkdir(parents=True);(payload/"bank").mkdir()
    for name in base.RUNTIME:
        shutil.copy2(base.checked(base.PRIOR/"payload"/name,previous["config"]["sources"][name]),payload/"runtime"/Path(name).name)
    for name in base.BANK:
        row=report["raw_files"][name];source=base.checked(base.PRIOR/"raw"/name,row["sha256"])
        if source.stat().st_size!=row["bytes"]:raise ValueError("Bank extent differs")
        shutil.copy2(source,payload/"bank"/name)
    # Keep the already qualified import/evidence helpers, not the concurrently edited runtime.
    prior=base.checked(PARENT/"runs/job-86cce06d33b1/report.json","760e14432e8de8158185296eb59055d98af1a2f926aa768df81f74ca3633a62b")
    prior_dir=prior.parent
    for original,destination in (("driver.c","legacy_driver.c"),("worker.py","support.py"),("analysis.py","legacy_analysis.py")):
        frozen=prior_dir/"payload"/original
        expected=json.loads((prior_dir/"submission.json").read_text())["config"]["sources"][original]
        shutil.copy2(base.checked(frozen,expected),payload/destination)
    provenance=json.loads(base.checked(prior_dir/"payload/baseline.json",json.loads((prior_dir/"submission.json").read_text())["config"]["sources"]["baseline.json"]).read_text())
    (payload/"baseline.json").write_text(json.dumps(provenance,indent=2)+"\n")
    for name in ("driver.c","worker.py","analysis.py","run.py","PLAN.md","test_analysis.py"):
        shutil.copy2(HERE/name,payload/name)
    shutil.copy2(PARENT/"Makefile",payload/"Makefile")
    bridge=base.checked(base.PRIOR/"payload/tools/saturn_bridge.py",previous["config"]["sources"]["tools/saturn_bridge.py"])
    shutil.copy2(bridge,payload/"saturn_bridge.py")
    shutil.copytree(base.PRIOR/"payload/public",payload/"public",ignore=shutil.ignore_patterns("__pycache__"))
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest={"schema":"tcm-whole-turn-payload-v1","sources":sources,"plan_sha256":base.sha(payload/"PLAN.md"),"raw_archive_limit_bytes":LIMIT}
    (payload/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"payload":str(payload),"files":len(sources),"bytes":sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}))
    return payload

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/"source-manifest.json").read_text())
    for name,expected in manifest["sources"].items():base.checked(payload/name,expected)
    config={"schema":"tcm-whole-turn-execution-v1","backend":"hf","candidate_backend":"c-cublas-fp32","device":"cuda","dtype":"fp32","cpu_threads":1,"max_seq_len":512,"backend_options":{"physical_batch_size":1},"max_retries":0,"sources":manifest["sources"],"source_manifest_sha256":base.sha(payload/"source-manifest.json"),"raw_archive_limit_bytes":LIMIT,
            "physical_work":"one native source replay;4 policies x correction and2 independent readers;then9 sealed dense same-token diagnostic segments;8 own-token budget/turn;no calibration/training",
            "reservation_source":"same512-context C FP32 job-86cce06d33b1 RSS3973.3MB/VRAM6378MB;job-9337489b61af RSS4058.1MB/VRAM6378MB;6144MB reservation gives6758.4MB VRAM ceiling",
            "scheduling_contract":{"model_loads":1,"primary_turns":12,"diagnostic_segments":9,"branch_execution":"in_process","max_mrun_submissions":1,"collection_policy":"direct_job_only","mrun_retry_policy":"none_for_finite_lease"}}
    job=launch("uv run --offline --project {env} python worker.py",experiment="c-state-machine-whole-turn",payload=payload,model="qwen2.5-1.5b-instruct",prefer_host="beast",needs={"cuda":True},config=config,env_alias="mrun",ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=180,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
               note="C whole-turn: frozen B3,4policies all template/user/generated tokens;correction then independent Niko/Mira readers,8tokens/turn;seal12primaryturns before9dense same-token diagnostics;one model B1 CPU1 capacity512;stop after21segments;no retries")
    target=HERE/"runs"/job;target.mkdir(parents=True,exist_ok=True)
    (target/"submission.json").write_text(json.dumps({"job_id":job,"payload":str(payload),"config":config},indent=2)+"\n")
    print(json.dumps({"job_id":job,"result_directory":str(target)}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("stage","submit","collect","verify"));p.add_argument("target",nargs="?");a=p.parse_args()
    if a.action=="stage":stage()
    elif a.action=="submit":submit(a.target)
    elif a.action=="collect":base.LIMIT=LIMIT;base.collect(a.target,HERE/"runs")
    else:base.verify(a.target)
