"""One finite resident span-recurrent cut, with pinned runtime and parent."""
from __future__ import annotations
import argparse,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location("composition_base",PARENT/"run.py");base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PRIOR=PARENT/"span_response/runs/job-524febce43e5"
PRIOR_SHA="1bc0692bc3a2243dc4031387ac152bd9b8d98713ef0931251fb6c4ddcc7f93fa"
LIMIT=48*1024**2
QUESTIONS=[{"name":"recall","text":"Whose badge became green? Reply with just the name.","budget":8,"file":"recall.txt"},{"name":"prose","text":"In one sentence, summarize the current badge assignments for both people.","budget":32,"file":"prose.txt"},{"name":"count","text":"How many people currently have a green badge? Reply with only the integer.","budget":8,"file":"count.txt"}]

def origin():
    r=json.loads(base.checked(PRIOR/"report.json",PRIOR_SHA).read_text())
    return {"schema":"tcm-span-recurrent-origin-v1","predecessor_report_sha256":PRIOR_SHA,"model":r["model"],"source_history_ids":r["driver"]["source_history_ids"],"source_kv_sha256":r["driver"]["source_kv_sha256"],"correction":r["driver"]["correction"],"correction_input":r["inputs"]["correction"]}

def stage():
    ref=origin();previous=json.loads((PRIOR/"submission.json").read_text());payload=HERE/"staged"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")/"payload";payload.mkdir(parents=True)
    keep={"support.py","saturn_bridge.py","baseline.json","legacy_driver.c","legacy_analysis.py","turn_helpers.h","whole_turn_analysis.py","whole_turn_driver.c","test_token_authority.py"}
    for name,expected in previous["config"]["sources"].items():
        if name not in keep and not name.startswith(("runtime/","public/")):continue
        dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/"payload"/name,expected),dest)
    for name in ("driver.c","span_cuda.cu","byte_bank.h","Makefile","worker.py","analysis.py","run.py","PLAN.md","MATH.md","test_recurrent.py","byte_smoke.c"):
        shutil.copy2(HERE/name,payload/name)
    for name,dest in (("span_bank.h","span_bank.h"),("analysis.py","span_parent_analysis.py")):
        shutil.copy2(base.checked(PRIOR/"payload"/name,previous["config"]["sources"][name]),payload/dest)
    parent_driver=base.checked(PRIOR/"payload/driver.c",previous["config"]["sources"]["driver.c"]).read_text()
    (payload/"span_helpers.h").write_text(parent_driver.split("\nint main(",1)[0]+"\n")
    (payload/"origin.json").write_text(json.dumps(ref,indent=2)+"\n");(payload/"questions.json").write_text(json.dumps(QUESTIONS,indent=2)+"\n")
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()};manifest={"schema":"tcm-span-recurrent-payload-v1","sources":sources,"plan_sha256":base.sha(payload/"PLAN.md"),"predecessor_report_sha256":PRIOR_SHA,"raw_archive_limit_bytes":LIMIT}
    (payload/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"payload":str(payload),"files":len(sources),"bytes":sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}));return payload

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/"source-manifest.json").read_text())
    for name,expected in manifest["sources"].items():base.checked(payload/name,expected)
    config={"schema":"tcm-span-recurrent-execution-v1","backend":"hf","candidate_backend":"c-cublas-fp32","device":"cuda","dtype":"fp32","cpu_threads":1,"max_seq_len":512,"backend_options":{"physical_batch_size":1},"max_retries":0,"sources":manifest["sources"],"source_manifest_sha256":base.sha(payload/"source-manifest.json"),"predecessor_report_sha256":PRIOR_SHA,"raw_archive_limit_bytes":LIMIT,
            "physical_work":"one native source+correction; memory-only Ward8/exact34/byte freeze before3questionfiles; native then poison C17 and four candidate chains x3linked turns; every future L17 read;8/32/8 own budgets; no training/repair/fallback",
            "reservation_source":"same512context C FP32 job-524febce43e5 RSS4083.3MB/VRAM6378MB; same1MiBworkspaceplus<0.2MiBprograms;6144MBVRAM gives6758.4MBceiling",
            "scheduling_contract":{"model_loads":1,"logical_branches":5,"linked_turns_per_branch":3,"branch_execution":"in_process","max_mrun_submissions":1,"collection_policy":"direct_job_only","mrun_retry_policy":"none_for_finite_lease"}}
    job=launch("uv run --offline --project {env} python worker.py",experiment="c-state-machine-span-recurrent",payload=payload,model="qwen2.5-1.5b-instruct",prefer_host="beast",needs={"cuda":True},config=config,env_alias="mrun",ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
               note="Recurring fixed span: source-only int8 paired34rows vs Ward8/exact/native/delete; native then poisoned C17 root, no dense L17 fallback;5 own-state chains x3turns, every future L17 read;8/32/8 budgets;one residentFP32 B1 CPU1;stop15turns,no retries")
    target=HERE/"runs"/job;target.mkdir(parents=True,exist_ok=True);(target/"submission.json").write_text(json.dumps({"job_id":job,"payload":str(payload),"config":config},indent=2)+"\n");print(json.dumps({"job_id":job,"result_directory":str(target)}))

def verify(root):
    root=Path(root)
    if json.loads((root/"payload/origin.json").read_text())!=origin() or json.loads((root/"payload/questions.json").read_text())!=QUESTIONS:raise ValueError("Authenticated parent/questions differ")
    prior=json.loads((PRIOR/"submission.json").read_text());sub=json.loads((root/"submission.json").read_text());job=json.loads((root/"job.json").read_text())
    for name,digest in prior["config"]["sources"].items():
        if name.startswith("runtime/") and sub["config"]["sources"].get(name)!=digest:raise ValueError("Pinned runtime differs")
    for field in ("sources","source_manifest_sha256","predecessor_report_sha256"):
        if job["config"][field]!=sub["config"][field]:raise ValueError("Scheduler request differs")
    if (root/"audit-replay.json").exists():
        from verify_replay import verify as verify_same_environment
        return verify_same_environment(root)
    base.verify(root)
    (root/"verification-source.py").write_bytes(Path(__file__).read_bytes());(root/"verification-base-source.py").write_bytes((PARENT/"run.py").read_bytes());print(json.dumps({"authenticated_parent":True,"questions_bound":True,"pinned_runtime":True,"scheduler_request_join":True}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("stage","submit","collect","verify"));p.add_argument("target",nargs="?");a=p.parse_args()
    if a.action=="stage":stage()
    elif a.action=="submit":submit(a.target)
    elif a.action=="collect":base.LIMIT=LIMIT;base.collect(a.target,HERE/"runs")
    else:verify(a.target)
