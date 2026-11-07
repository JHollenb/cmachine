"""Submit/collect the saved-only exact-environment verification transaction."""
from pathlib import Path
import argparse,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def submit(root):
    from mrun.client.submit import launch
    root=Path(root).resolve();r=json.loads((root/"report.json").read_text());sub=json.loads((root/"submission.json").read_text());job=sub["job_id"];payload=root/"audit-replay-payload";payload.mkdir(exist_ok=True)
    shutil.copy2(HERE/"audit_worker.py",payload/"audit_worker.py")
    request={"original_job_id":job,"original_worker_root":"/mnt/hdd8tb/mrun-work/"+job,"source_manifest_sha256":sha(root/"payload/source-manifest.json"),"raw_report_sha256":sha(root/"raw/report.json"),"analysis_sha256":hashlib.sha256(canonical(r["analysis"])).hexdigest()}
    (payload/"request.json").write_text(json.dumps(request,indent=2)+"\n");sources={p.name:sha(p) for p in payload.iterdir() if p.is_file()}
    config={"schema":"tcm-exact-analysis-replay-request-v1","device":"cpu","dtype":"fp64","max_retries":0,"sources":sources,"physical_work":"one saved-only analyzer replay; original1107raw+76source hashes; zero model loads or forwards; require exact canonical original analysis digest; original worker environment","scheduling_contract":{"model_loads":0,"logical_branches":1,"max_mrun_submissions":1,"mrun_retry_policy":"none_for_finite_lease","collection_policy":"direct_job_only"}}
    audit_job=launch("uv run --offline --project {env} python audit_worker.py",experiment="c-state-machine-span-recurrent-audit",payload=payload,prefer_host="beast",needs={"cpu":True},config=config,env_alias="mrun",timeout_s=180,detach=True,retry_on_kill=False,preflight=True,note="Saved-only exact frozen analyzer replay in original Linux/NumPy profile; hash1107raw+76sources,zero model/zeroGPU; require canonical analysis SHA exact; stop after1replay")
    out={"job_id":audit_job,"config":config,"request":request};(root/"audit-replay-submission.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"audit_job_id":audit_job}))
def collect(root):
    from mrun.client.api import Api
    root=Path(root);s=json.loads((root/"audit-replay-submission.json").read_text());a=Api();_,raw,_=a.request("GET",f"/api/jobs/{s['job_id']}",timeout_s=20);job=json.loads(raw);_,logs,_=a.request("GET",f"/api/jobs/{s['job_id']}/logs?offset=0&wait_s=1",timeout_s=30)
    if job["state"]!="succeeded":raise ValueError("Audit not complete: "+job["state"])
    if hashlib.sha256(logs).hexdigest()!=job["result"]["artifacts"]["job_log"]["sha256"] or job["config"]["sources"]!=s["config"]["sources"]:raise ValueError("Audit scheduler/source seal differs")
    matches=[json.loads(line.split("=",1)[1]) for line in logs.decode().splitlines() if line.startswith("AUDIT_RESULT=")]
    if len(matches)!=1:raise ValueError("Audit result missing/duplicate")
    result=matches[0];payload=root/"audit-replay-payload"
    for name,digest in s["config"]["sources"].items():
        if sha(payload/name)!=digest:raise ValueError("Local audit payload differs")
    if result["audit_job_id"]!=s["job_id"] or result["request_sha256"]!=sha(payload/"request.json") or result["audit_worker_sha256"]!=sha(payload/"audit_worker.py") or result["analysis_sha256"]!=s["request"]["analysis_sha256"] or not result["analysis_canonical_bytes_exact"] or result["models_evaluated"]:raise ValueError("Exact replay result differs")
    (root/"audit-replay-job.json").write_text(json.dumps(job,indent=2)+"\n");(root/"audit-replay-logs.txt").write_bytes(logs);(root/"audit-replay.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("submit","collect"));p.add_argument("root");a=p.parse_args();globals()[a.action](a.root)
