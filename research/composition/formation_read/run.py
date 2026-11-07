"""Only the two missing hybrid policies, joined to authenticated prior endpoints."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
PRIOR=PARENT/"whole_turn/runs/job-a071c7a43a38"
PRIOR_SHA="cda1042c01e53762643245f13573431e2f17983e46934a69cb6bf7a8e88ed444"
spec=importlib.util.spec_from_file_location("composition_base",PARENT/"run.py");base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMIT=48*1024**2

def origin():
    report=json.loads(base.checked(PRIOR/"report.json",PRIOR_SHA).read_text())
    arms={a["name"]:a for a in report["driver"]["arms"]}
    formations={label:arms[label]["turns"][0] for label in ("native","project")}
    readers={name:{task:arms[label]["turns"][i] for i,task in ((1,"target"),(2,"collateral"))} for name,label in (("NN","native"),("PP","project"))}
    relevant={t["name"] for t in formations.values()}|{t["name"] for row in readers.values() for t in row.values()}
    records={name:record for name,record in report["raw_files"].items() if any(name.startswith(prefix+"-") for prefix in relevant)}
    return {"schema":"tcm-formation-read-origin-v1","report_sha256":PRIOR_SHA,"model":report["model"],"inputs":report["inputs"],"colors":report["colors"],"source_history_ids":report["driver"]["source_history_ids"],"source_kv_sha256":report["driver"]["source_kv_sha256"],"formations":formations,"readers":readers,"raw_records":records,"texts":{k:v for k,v in report["texts"].items() if any(k==prefix for prefix in relevant)}}

def stage():
    ref=origin();previous=json.loads((PRIOR/"submission.json").read_text())
    payload=HERE/"staged"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")/"payload";payload.mkdir(parents=True)
    omit={"driver.c","worker.py","analysis.py","run.py","PLAN.md","test_analysis.py"}
    for name,expected in previous["config"]["sources"].items():
        if name in omit:continue
        path=payload/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/"payload"/name,expected),path)
    for source,destination in (("driver.c","whole_turn_driver.c"),("analysis.py","whole_turn_analysis.py"),("test_analysis.py","test_token_authority.py")):
        shutil.copy2(base.checked(PRIOR/"payload"/source,previous["config"]["sources"][source]),payload/destination)
    driver=(payload/"whole_turn_driver.c").read_text();marker="\nint main(int argc,char **argv){"
    if driver.count(marker)!=1:raise ValueError("Historical helper boundary differs")
    (payload/"turn_helpers.h").write_text(driver.split(marker)[0]+"\n")
    (payload/"origin.json").write_text(json.dumps(ref,indent=2)+"\n")
    (payload/"historical").mkdir()
    for row in ref["readers"].values():
        for turn in row.values():
            names=[f"{turn['name']}-{boundary}.f32" for boundary in ("prefill","final")]
            names += [f"{turn['name']}-s{s['step']:03d}-kv16-17.f32" for s in turn["steps"]+turn["generation_steps"]]
            for name in names:
                record=ref["raw_records"][name]
                source=base.checked(PRIOR/"raw"/name,record["sha256"])
                if source.stat().st_size!=record["bytes"]:raise ValueError("Historical endpoint extent differs")
                shutil.copy2(source,payload/"historical"/name)
    for name in ("driver.c","worker.py","analysis.py","run.py","PLAN.md","test_factorial.py"):
        shutil.copy2(HERE/name,payload/name)
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest={"schema":"tcm-formation-read-payload-v1","sources":sources,"plan_sha256":base.sha(payload/"PLAN.md"),"predecessor_report_sha256":PRIOR_SHA,"raw_archive_limit_bytes":LIMIT}
    (payload/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"payload":str(payload),"files":len(sources),"bytes":sum(p.stat().st_size for p in payload.rglob("*") if p.is_file())}));return payload

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/"source-manifest.json").read_text())
    for name,expected in manifest["sources"].items():base.checked(payload/name,expected)
    config={"schema":"tcm-formation-read-execution-v1","backend":"hf","candidate_backend":"c-cublas-fp32","device":"cuda","dtype":"fp32","cpu_threads":1,"max_seq_len":512,"backend_options":{"physical_batch_size":1},"max_retries":0,"sources":manifest["sources"],"source_manifest_sha256":base.sha(payload/"source-manifest.json"),"predecessor_report_sha256":PRIOR_SHA,"raw_archive_limit_bytes":LIMIT,
            "physical_work":"one source replay;only PN and NP hybrid correction/question episode policies;2corrections+4independentreaders;8own-tokenbudget;historicalNN/PPanalysis-only;no diagnostic replay/calibration/training",
            "reservation_source":"same512-context C FP32 job-a071c7a43a38 RSS3993.1MB/VRAM6378MB;6144MBVRAM reservation ceiling6758.4MB",
            "scheduling_contract":{"model_loads":1,"primary_turns":6,"new_hybrid_policies":2,"historical_endpoints":2,"branch_execution":"in_process","max_mrun_submissions":1,"collection_policy":"direct_job_only","mrun_retry_policy":"none_for_finite_lease"}}
    job=launch("uv run --offline --project {env} python worker.py",experiment="c-state-machine-formation-read",payload=payload,model="qwen2.5-1.5b-instruct",prefer_host="beast",needs={"cuda":True},config=config,env_alias="mrun",ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
               note="C formation/read: only two missing PN/NP policies,6turns,8own-tokenbudgets; fullcorrectionhistory/KV replayjoin beforeeachreader; historicalNN/PPanalysis-only;one residentFP32Cmodel B1 CPU1 capacity512;stop after6turns,no retries")
    target=HERE/"runs"/job;target.mkdir(parents=True,exist_ok=True);(target/"submission.json").write_text(json.dumps({"job_id":job,"payload":str(payload),"config":config},indent=2)+"\n");print(json.dumps({"job_id":job,"result_directory":str(target)}))

def verify(root):
    import math
    root=Path(root)
    if json.loads((root/"payload/origin.json").read_text())!=origin():raise ValueError("Authenticated predecessor projection differs")
    report=json.loads((root/"report.json").read_text());numerical_rows=[]
    def numerical_policy(path,local,producer):
        if not path or path[-1]!="power_closure_residual":return False
        parent=report["analysis"]
        for key in path[:-1]:parent=parent[key]
        if "signed_energy" not in parent or "joint_power" not in parent:return False
        terms=parent["signed_energy"];powers=[terms[k] for k in ("formation_power","question_power","interaction_power")];d2=parent["joint_power"]
        n=151936;u=2.**-53;m=2*n+16;gamma=m*u/(1-m*u);defect=math.sqrt(n)*parent["pointwise_closure_max_abs"]
        scale=d2+sum(math.sqrt(p) for p in powers)**2;bound=gamma*scale+2*math.sqrt(d2)*defect+defect**2
        if not all(math.isfinite(v) for v in (local,producer,bound)) or abs(local)>bound or abs(producer)>bound:raise ValueError("FP64 power identity exceeds its absolute-product roundoff bound")
        numerical_rows.append({"path":list(path),"producer_remainder":producer,"local_remainder":local,"roundoff_bound":bound,"gamma":gamma,"absolute_product_scale":scale,"pointwise_defect_l2_bound":defect});return True
    base.verify(root,numerical_policy=numerical_policy)
    job=json.loads((root/"job.json").read_text());submission=json.loads((root/"submission.json").read_text())
    for field in ("sources","source_manifest_sha256","predecessor_report_sha256"):
        if job["config"][field]!=submission["config"][field]:raise ValueError("Scheduler source/request join differs")
    source=(root/"payload/whole_turn_driver.c").read_text().split("\nint main(int argc,char **argv){")[0]+"\n"
    if (root/"payload/turn_helpers.h").read_text()!=source:raise ValueError("Helper extraction differs")
    policy=HERE/"NUMERICAL_AUDIT_POLICY.md"
    (root/"verification-source.py").write_bytes(Path(__file__).read_bytes());(root/"verification-base-source.py").write_bytes((PARENT/"run.py").read_bytes());(root/"numerical-audit-policy.md").write_bytes(policy.read_bytes())
    (root/"numerical-audit.json").write_text(json.dumps({"report_sha256":base.sha(root/"report.json"),"policy_sha256":base.sha(policy),"verifier_source_sha256":base.sha(root/"verification-source.py"),"base_verifier_source_sha256":base.sha(root/"verification-base-source.py"),"scope":"derived FP64 power remainder only; original report unchanged","models_evaluated":0,"rows":numerical_rows},indent=2)+"\n")
    print(json.dumps({"authenticated_predecessor_projection":True,"helper_extraction_exact":True,"scheduler_source_request_join":True}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("stage","submit","collect","verify"));p.add_argument("target",nargs="?");a=p.parse_args()
    if a.action=="stage":stage()
    elif a.action=="submit":submit(a.target)
    elif a.action=="collect":base.LIMIT=LIMIT;base.collect(a.target,HERE/"runs")
    else:verify(a.target)
