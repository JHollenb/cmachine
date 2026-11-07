"""One finite resident span-response cut, with pinned runtime and parent."""
from __future__ import annotations
import argparse,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location("composition_base",PARENT/"run.py");base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PRIOR=PARENT/"attention_read/runs/job-812a0486995d"
PRIOR_SHA="34b611215557b2d430cfc1122b4c6f6b3da35b4fe7f65d14b4f565fb55696dc2"
LIMIT=48*1024**2
QUESTIONS=[{"name":"recall","text":"Whose badge became green? Reply with just the name.","budget":8,"file":"recall.txt"},{"name":"prose","text":"In one sentence, summarize the current badge assignments for both people.","budget":32,"file":"prose.txt"}]

def origin():
    r=json.loads(base.checked(PRIOR/"report.json",PRIOR_SHA).read_text())
    return {"schema":"tcm-span-response-origin-v1","predecessor_report_sha256":PRIOR_SHA,"model":r["model"],"source_history_ids":r["driver"]["source_history_ids"],"source_kv_sha256":r["driver"]["source_kv_sha256"],"correction":r["driver"]["formations"][0]["correction"],"correction_input":r["inputs"]["questions"][0]}

def stage():
    ref=origin();previous=json.loads((PRIOR/"submission.json").read_text());payload=HERE/"staged"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")/"payload";payload.mkdir(parents=True)
    keep={"support.py","saturn_bridge.py","baseline.json","legacy_driver.c","legacy_analysis.py","turn_helpers.h","whole_turn_analysis.py","whole_turn_driver.c","test_token_authority.py"}
    for name,expected in previous["config"]["sources"].items():
        if name not in keep and not name.startswith(("runtime/","public/")):continue
        dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/"payload"/name,expected),dest)
    for name in ("driver.c","span_cuda.cu","span_bank.h","Makefile","worker.py","analysis.py","run.py","PLAN.md","test_span.py","bank_smoke.c"):
        shutil.copy2(HERE/name,payload/name)
    (payload/"origin.json").write_text(json.dumps(ref,indent=2)+"\n");(payload/"questions.json").write_text(json.dumps(QUESTIONS,indent=2)+"\n")
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()};manifest={"schema":"tcm-span-response-payload-v1","sources":sources,"plan_sha256":base.sha(payload/"PLAN.md"),"predecessor_report_sha256":PRIOR_SHA,"raw_archive_limit_bytes":LIMIT}
    (payload/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"payload":str(payload),"files":len(sources),"bytes":sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}));return payload

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/"source-manifest.json").read_text())
    for name,expected in manifest["sources"].items():base.checked(payload/name,expected)
    config={"schema":"tcm-span-response-execution-v1","backend":"hf","candidate_backend":"c-cublas-fp32","device":"cuda","dtype":"fp32","cpu_threads":1,"max_seq_len":512,"backend_options":{"physical_batch_size":1},"max_retries":0,"sources":manifest["sources"],"source_manifest_sha256":base.sha(payload/"source-manifest.json"),"predecessor_report_sha256":PRIOR_SHA,"raw_archive_limit_bytes":LIMIT,
            "physical_work":"one native source+correction;memory-onlyWard1/8/exact34freeze before2questionfilesopened;2native sharedquestionprefixes;5singleL17readarmsperquestion;8/32own-tokenbudgets;laternative;no training/calibration/fallback",
            "reservation_source":"same512context C FP32 job-812a0486995d RSS3914.7MB/VRAM6378MB;1MiBprivatereadworkspaceplus<0.1MiBprograms;6144MBVRAM gives6758.4MBceiling",
            "scheduling_contract":{"model_loads":1,"logical_branches":10,"selected_read_applications":10,"branch_execution":"in_process","max_mrun_submissions":1,"collection_policy":"direct_job_only","mrun_retry_policy":"none_for_finite_lease"}}
    job=launch("uv run --offline --project {env} python worker.py",experiment="c-state-machine-span-response",payload=payload,model="qwen2.5-1.5b-instruct",prefer_host="beast",needs={"cuda":True},config=config,env_alias="mrun",ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
               note="Frozen span response:Ward1/8/exact34 fromnative34rowKV17 beforefuturequeries;2questions x5single-readarms(native/exact/1/8/delete),8/32own-tokenbudgets;densecomplement+laternativeexplicit;one residentFP32 B1 CPU1;stop10branches,no retries")
    target=HERE/"runs"/job;target.mkdir(parents=True,exist_ok=True);(target/"submission.json").write_text(json.dumps({"job_id":job,"payload":str(payload),"config":config},indent=2)+"\n");print(json.dumps({"job_id":job,"result_directory":str(target)}))

def verify(root):
    root=Path(root)
    if json.loads((root/"payload/origin.json").read_text())!=origin() or json.loads((root/"payload/questions.json").read_text())!=QUESTIONS:raise ValueError("Authenticated parent/questions differ")
    prior=json.loads((PRIOR/"submission.json").read_text());sub=json.loads((root/"submission.json").read_text());job=json.loads((root/"job.json").read_text())
    for name,digest in prior["config"]["sources"].items():
        if name.startswith("runtime/") and sub["config"]["sources"].get(name)!=digest:raise ValueError("Pinned runtime differs")
    for field in ("sources","source_manifest_sha256","predecessor_report_sha256"):
        if job["config"][field]!=sub["config"][field]:raise ValueError("Scheduler request differs")
    base.verify(root)
    (root/"verification-source.py").write_bytes(Path(__file__).read_bytes());(root/"verification-base-source.py").write_bytes((PARENT/"run.py").read_bytes());print(json.dumps({"authenticated_parent":True,"questions_bound":True,"pinned_runtime":True,"scheduler_request_join":True}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("stage","submit","collect","verify"));p.add_argument("target",nargs="?");a=p.parse_args()
    if a.action=="stage":stage()
    elif a.action=="submit":submit(a.target)
    elif a.action=="collect":base.LIMIT=LIMIT;base.collect(a.target,HERE/"runs")
    else:verify(a.target)
