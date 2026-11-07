"""Frozen operands, exact predecessor identities and one finite diagnostic lease."""
import argparse,hashlib,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'context_query/runs/job-fe51e7f4f0ee';LIMIT=192*1024**2
spec=importlib.util.spec_from_file_location('collection',ROOT.parent/'composition/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stage():
    report=read(PRIOR/'report.json')
    if sha(PRIOR/'report.json')!='d97a8dedb3a3e187e958c56ecf81bf59d8f2b45f3b69c4c27a76925507d9e9e2' or read(PRIOR/'verification.json')['report_sha256']!=sha(PRIOR/'report.json'):raise ValueError('Source report differs')
    out=ROOT/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';out.mkdir(parents=True)
    def copy(p,name,h=None):
        if h and sha(p)!=h:raise ValueError('Source custody differs '+str(p))
        q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    manifest=read(PRIOR/'payload/source-manifest.json')
    for name,h in manifest['sources'].items():
        if name.startswith(('runtime/','public/')) or name=='saturn_bridge.py':copy(PRIOR/'payload'/name,name,h)
    for src,name in [('program128-basis.f32','basis17.f32'),('program128-k.f32','key17.f32')]:copy(PRIOR/'payload'/src,name,manifest['sources'][src])
    for src,name in [('query-proposal-3.f32','query17.f32'),('basis18.f32','basis18.f32'),('image18-q.f32','query18.f32'),('image18-k.f32','key18.f32')]:copy(PRIOR/'raw'/src,name,report['raw_files'][src]['sha256'])
    inputs=read(PRIOR/'raw/input-manifest.json');cases=[]
    for context in (128,1024):
        stem=f'c0-n{context}';trace_name=stem+'-arm3.json';copy(PRIOR/'raw'/trace_name,f'case{context}-previous.json',report['raw_files'][trace_name]['sha256']);case=next(x for x in inputs if x['stem']==stem);cases.append(case)
        for history,arm,fields in [(0,3,[(t,b) for t in (0,1) for b in (1,2)]),(1,2,[(0,1)])]:
            for turn,boundary in fields:
                src=f'{stem}-arm{arm}-turn{turn}-b{boundary}-candidate.f32';copy(PRIOR/'raw'/src,f'case{context}-h{history}-turn{turn}-b{boundary}.f32',report['raw_files'][src]['sha256'])
    (out/'baseline.json').write_text(json.dumps({'model':report['model'],'previous_report_sha256':sha(PRIOR/'report.json'),'cases':cases,'scope':'Known failures; no new fit or heldout evaluation.'},indent=2)+'\n')
    for name in ('driver.c','worker.py','analyze.py','Makefile','PLAN.md','run.py'):copy(ROOT/name,name)
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};(out/'source-manifest.json').write_text(json.dumps({'schema':'qk18-cut-payload-v1','sources':files,'raw_archive_limit_bytes':LIMIT},indent=2,sort_keys=True)+'\n');print(json.dumps({'payload':str(out),'sources':len(files),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}));return out
def submit(payload):
    from mdb.job_support import guarded_mrun_launch,one_lease_scheduling_contract
    payload=Path(payload).resolve();m=read(payload/'source-manifest.json')
    for p,h in m['sources'].items():base.checked(payload/p,h)
    request={'schema':'qk18-cut-request-v1','model':'qwen2.5-1.5b-instruct','backend':'hf','candidate_backend':'C CUDA pedantic FP32 + CPU64 prototype17','device':'cuda','dtype':'fp32','needs':{'cuda':True},'max_seq_len':2048,'cpu_threads':1,'sources':m['sources'],'source_manifest_sha256':sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'physical_work':'Known128/1024inventory failures.2same-ID histories x9logical single-read branches,one alias perparent:36logical/32physical. Fullcurrent QxK history cross,sham,wrongtime,zero,reversedV. Own24suffix+followup32.8000advance/360Csecondguard. Native Q/K diagnostics paid at everytoken.','scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_branches=36,logical_operations=32,max_local_work_items=36,checkpoint_policy='Same model/session/epoch/lineage token parent; full prototype17 snapshot and append-only full18cursors; explicit parentlogit restore; fullvalidprefix+memory+logit hash before every physical branch.'),'resource_exception':{'ram_mb':8192,'vram_mb':8192,'calibration_job':'job-fe51e7f4f0ee','reason':'Same C FP32 capacity2048 measured6472MBVRAM/4033.6MBRSS; one session at a time,~9MBextraGPU key/value buffers and<100MBhosthash/operand buffers. Measured8GiB exception,2GiB scratch.'}}
    job=guarded_mrun_launch(['uv','run','--offline','--project','{env}','python','worker.py'],request=request,model=request['model'],task='analysis',experiment='c-state-machine-query-key-cut',payload=payload,prefer_host='beast',needs=request['needs'],env_alias='mrun',ram_mb=8192,vram_mb=8192,disk_gb=2,cpu_threads=1,est_wall_s=120,timeout_s=480,retry_on_kill=False,preflight=True,note='Firstwrongbinding QK18 causalcut onknown128/1024inventorycases. Sameactualbadprefix, native/compiled18histories;36logical32physicalone-readbranches fromexactparent:QxK,untouched/sham/wrongtime/zero/Vreverse. Own24+follow32,nocorrecttokenssupplied.1CmodelCPU1B1,fullproto17snapshot andvalidprefix/logit restorehash;8000adv/360Csec/480lease.8GiB measured,no fit/retries.')
    d=ROOT/'runs'/job;d.mkdir(parents=True,exist_ok=False);(d/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':request},indent=2,sort_keys=True)+'\n');print(json.dumps({'job_id':job,'result_directory':str(d)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    else:base.LIMIT=LIMIT;base.collect(a.target,ROOT/'runs')
