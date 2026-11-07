"""One bounded mask-ablation followup; no repeated full-policy suite."""
import argparse,hashlib,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'context_query/runs/job-fe51e7f4f0ee';LIMIT=128*1024**2
spec=importlib.util.spec_from_file_location('collect',ROOT.parent/'composition/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stage():
    r=read(PRIOR/'report.json')
    if sha(PRIOR/'report.json')!='d97a8dedb3a3e187e958c56ecf81bf59d8f2b45f3b69c4c27a76925507d9e9e2':raise ValueError('Prior differs')
    out=ROOT/'policy-staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';out.mkdir(parents=True)
    def copy(p,n,h=None):
        if h and sha(p)!=h:raise ValueError('Custody differs')
        q=out/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    m=read(PRIOR/'payload/source-manifest.json')
    for p,h in m['sources'].items():
        if p.startswith(('runtime/','public/')) or p in ('distribution.h','saturn_bridge.py'):copy(PRIOR/'payload'/p,p,h)
    copy(PRIOR/'payload/driver.c','engine.c',m['sources']['driver.c']);copy(PRIOR/'payload/worker.py','engine_worker.py',m['sources']['worker.py'])
    for p,n in [('program128-basis.f32','basis17.f32'),('program128-k.f32','key17.f32')]:copy(PRIOR/'payload'/p,n,m['sources'][p])
    for p,n in [('query-proposal-3.f32','query17.f32'),('basis18.f32','basis18.f32'),('image18-q.f32','query18.f32'),('image18-k.f32','key18.f32')]:copy(PRIOR/'raw'/p,n,r['raw_files'][p]['sha256'])
    copy(PRIOR/'raw/input-manifest.json','input-manifest.json',r['raw_files']['input-manifest.json']['sha256']);(out/'baseline.json').write_text(json.dumps({'model':r['model'],'previous_report_sha256':sha(PRIOR/'report.json')},indent=2)+'\n')
    for p,n in [('policy_driver.c','driver.c'),('policy_worker.py','worker.py'),('PolicyMakefile','Makefile'),('ON_POLICY_PLAN.md','PLAN.md'),('policy_run.py','run.py')]:copy(ROOT/p,n)
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};(out/'source-manifest.json').write_text(json.dumps({'schema':'qk18-policy-payload-v1','sources':files,'raw_archive_limit_bytes':LIMIT},indent=2)+'\n');print(json.dumps({'payload':str(out),'sources':len(files)}));return out
def submit(payload):
    from mdb.job_support import guarded_mrun_launch,one_lease_scheduling_contract
    payload=Path(payload).resolve();m=read(payload/'source-manifest.json')
    for p,h in m['sources'].items():base.checked(payload/p,h)
    request={'schema':'qk18-policy-request-v1','model':'qwen2.5-1.5b-instruct','backend':'hf','candidate_backend':'C CUDA FP32 + CPU64 prototype17','device':'cuda','dtype':'fp32','needs':{'cuda':True},'max_seq_len':2048,'cpu_threads':1,'sources':m['sources'],'source_manifest_sha256':sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'physical_work':'8new paths: frozen Q-only18/K-only18 x4known128/1024conversations, own32+32 inclEOS and independentnative shadows. No fit, old full policies not repeated.','scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_branches=8,logical_operations=8,max_local_work_items=8,checkpoint_policy='Each policy starts empty; candidate session destroyed before independent same-ID native shadow. No mutable cache rollback or original history refresh.'),'resource_exception':{'ram_mb':8192,'vram_mb':8192,'calibration_job':'job-fe51e7f4f0ee','reason':'Identical C executor/model/capacity; measured6472MBVRAM/4033.6MBRSS. One session at a time, one frozen bank;8GiB measured envelope.'}}
    job=guarded_mrun_launch(['uv','run','--offline','--project','{env}','python','worker.py'],request=request,model=request['model'],task='analysis',experiment='c-state-machine-query-key-on-policy',payload=payload,prefer_host='beast',needs=request['needs'],env_alias='mrun',ram_mb=8192,vram_mb=8192,disk_gb=2,cpu_threads=1,est_wall_s=180,timeout_s=480,retry_on_kill=False,preflight=True,note='Constructive followup toQKcausalcut:8new Q-only/K-only18 own32+follow32 paths at128/1024 on4knowncases. FixedB/H,compact17 unchanged; native same-ID shadows after candidate destruction. No fit/new heldout/old full-policy repeats.1CmodelCPU1B1,strict8GiB measured,480s/no retry.')
    d=ROOT/'runs'/job;d.mkdir(parents=True,exist_ok=False);(d/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':request},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(d)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    else:base.LIMIT=LIMIT;base.collect(a.target,ROOT/'runs')
