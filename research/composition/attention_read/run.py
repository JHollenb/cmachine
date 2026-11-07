"""One finite resident trace, with authenticated uninstrumented endpoints."""
from __future__ import annotations
import argparse,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location("composition_base",PARENT/"run.py");base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PRIOR=PARENT/"formation_read/runs/job-c593b2c6e4a6"
PRIOR_SHA="3e960abf835542fabe9cc62cd4d874ef998b7987738ff2e018a29a784c296299"
WHOLE=PARENT/"whole_turn/runs/job-a071c7a43a38"
WHOLE_SHA="cda1042c01e53762643245f13573431e2f17983e46934a69cb6bf7a8e88ed444"
LIMIT=48*1024**2

def origin():
    hybrid=json.loads(base.checked(PRIOR/"report.json",PRIOR_SHA).read_text());whole=json.loads(base.checked(WHOLE/"report.json",WHOLE_SHA).read_text())
    old={a["name"]:a for a in whole["driver"]["arms"]};new={a["name"]:a for a in hybrid["driver"]["arms"]}
    arms={"NN":old["native"],"PN":new["PN"],"NP":new["NP"],"PP":old["project"]}
    if hybrid["model"]!=whole["model"] or hybrid["inputs"]!=whole["inputs"]:raise ValueError("Predecessors differ")
    return {"schema":"tcm-attention-read-origin-v1","predecessors":{"whole_turn":WHOLE_SHA,"formation_read":PRIOR_SHA},"model":whole["model"],"inputs":whole["inputs"],"source_history_ids":whole["driver"]["source_history_ids"],"source_kv_sha256":whole["driver"]["source_kv_sha256"],"formations":{"N":old["native"]["turns"][0],"P":old["project"]["turns"][0]},"readers":{w:{task:a["turns"][i] for i,task in ((1,"target"),(2,"collateral"))} for w,a in arms.items()}}

def stage():
    ref=origin();previous=json.loads((PRIOR/"submission.json").read_text());payload=HERE/"staged"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")/"payload";payload.mkdir(parents=True)
    keep={"support.py","saturn_bridge.py","baseline.json","legacy_driver.c","legacy_analysis.py","turn_helpers.h","whole_turn_analysis.py","whole_turn_driver.c","test_token_authority.py"}
    for name,expected in previous["config"]["sources"].items():
        if name not in keep and not name.startswith(("runtime/","bank/","public/")):continue
        dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/"payload"/name,expected),dest)
    for name in ("driver.c","attention_cuda.cu","Makefile","worker.py","analysis.py","run.py","PLAN.md","test_attention.py"):
        shutil.copy2(HERE/name,payload/name)
    (payload/"origin.json").write_text(json.dumps(ref,indent=2)+"\n")
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest={"schema":"tcm-attention-read-payload-v1","sources":sources,"plan_sha256":base.sha(payload/"PLAN.md"),"predecessors":ref["predecessors"],"raw_archive_limit_bytes":LIMIT}
    (payload/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"payload":str(payload),"files":len(sources),"bytes":sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}));return payload

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/"source-manifest.json").read_text())
    for name,expected in manifest["sources"].items():base.checked(payload/name,expected)
    config={"schema":"tcm-attention-read-execution-v1","backend":"hf","candidate_backend":"c-cublas-fp32","device":"cuda","dtype":"fp32","cpu_threads":1,"max_seq_len":512,"backend_options":{"physical_batch_size":1},"max_retries":0,"sources":manifest["sources"],"source_manifest_sha256":base.sha(payload/"source-manifest.json"),"predecessors":manifest["predecessors"],"raw_archive_limit_bytes":LIMIT,
            "physical_work":"one source replay;2corrections shared across8independentreaderchildren;4worlds x2questions x2selectedclocks=16L17traces;8own-tokenbudget;identical historical endpoints required;no training/calibration/diagnostic replay",
            "reservation_source":"same512context C FP32 job-c593b2c6e4a6 RSS4028.3MB/VRAM6378MB;noaddedGPUallocation;6144MBVRAM gives6758.4MBceiling",
            "scheduling_contract":{"model_loads":1,"primary_turns":10,"selected_traces":16,"branch_execution":"in_process","max_mrun_submissions":1,"collection_policy":"direct_job_only","mrun_retry_policy":"none_for_finite_lease"}}
    job=launch("uv run --offline --project {env} python worker.py",experiment="c-state-machine-attention-read",payload=payload,model="qwen2.5-1.5b-instruct",prefer_host="beast",needs={"cuda":True},config=config,env_alias="mrun",ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
               note="C attention read:L17 native QK/probability/KV/context traces;4worlds x2questions x2clocks=16;2sharedcorrections+8readers;one residentFP32 B1 CPU1 capacity512;historical full-logit/KV equality;stop10turns,no retries")
    target=HERE/"runs"/job;target.mkdir(parents=True,exist_ok=True);(target/"submission.json").write_text(json.dumps({"job_id":job,"payload":str(payload),"config":config},indent=2)+"\n");print(json.dumps({"job_id":job,"result_directory":str(target)}))

def verify(root):
    root=Path(root)
    if json.loads((root/"payload/origin.json").read_text())!=origin():raise ValueError("Authenticated predecessor projection differs")
    previous=json.loads((PRIOR/"submission.json").read_text());submission=json.loads((root/"submission.json").read_text());job=json.loads((root/"job.json").read_text())
    for name in previous["config"]["sources"]:
        if name.startswith(("runtime/","bank/")) and submission["config"]["sources"].get(name)!=previous["config"]["sources"][name]:raise ValueError("Pinned original runtime/bank differs")
    for field in ("sources","source_manifest_sha256","predecessors"):
        if job["config"][field]!=submission["config"][field]:raise ValueError("Scheduler request differs")
    base.verify(root)
    (root/"verification-source.py").write_bytes(Path(__file__).read_bytes());(root/"verification-base-source.py").write_bytes((PARENT/"run.py").read_bytes())
    print(json.dumps({"authenticated_predecessors":True,"pinned_original_runtime_bank":True,"scheduler_request_join":True}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("stage","submit","collect","verify"));p.add_argument("target",nargs="?");a=p.parse_args()
    if a.action=="stage":stage()
    elif a.action=="submit":submit(a.target)
    elif a.action=="collect":base.LIMIT=LIMIT;base.collect(a.target,HERE/"runs")
    else:verify(a.target)
