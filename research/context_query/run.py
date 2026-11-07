"""Source-bound payload, one finite guarded lease, direct collection."""
import argparse,hashlib,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;MACHINE=ROOT.parents[1];WORK=MACHINE.parent
PRIOR=ROOT.parent/'query_own_history/runs/job-47cc27324445';LIMIT=384*1024**2
spec=importlib.util.spec_from_file_location('collection',ROOT.parent/'composition/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stage():
    report=read(PRIOR/'report.json')
    if sha(PRIOR/'report.json')!='d043d1490e7bf10bc8d688e20e178a07ad36be221518690a78d1ff11b974ba39':raise ValueError('Source report differs')
    out=ROOT/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';out.mkdir(parents=True)
    def copy(p,name,h=None):
        if h and sha(p)!=h:raise ValueError('Source custody differs: '+str(p))
        q=out/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for name in ('model.c','qwen.c','qwen.h','qwen_operands.inc','prototype_attention.h','qwen_cuda.cu','qwen_cache.inc','qwen_weights.inc','machine.h','gpu.h','byte_codec.h'):copy(MACHINE/'src'/name,'runtime/'+name)
    for name in ('driver.c','worker.py','fit.py','run.py','Makefile','PLAN.md','fresh.json'):copy(ROOT/name,name)
    copy(ROOT.parent/'query_own_history/distribution.h','distribution.h');copy(ROOT.parent/'query_own_history/reader.py','reader.py');copy(ROOT.parent/'query_response/compiler.py','image_transactions.py')
    prior_manifest=read(PRIOR/'payload/source-manifest.json')
    for name,h in prior_manifest['sources'].items():
        if name.startswith('public/') or name=='saturn_bridge.py':copy(PRIOR/'payload'/name,name,h)
    for name in ('program128-basis.f32','program128-k.f32','query-fit.f32'):copy(PRIOR/'raw'/name,name,report['raw_files'][name]['sha256'])
    for c in range(3):
        pair=c+3
        for field in ('q.f32','k.f32','v.f32','attention.f32','coeff.f32','tops.i32','trace.json'):
            name=f'pair{pair}-compact-{field}';copy(PRIOR/'raw'/name,f'parity{c}-{field}',report['raw_files'][name]['sha256'])
        trace=read(PRIOR/'raw'/f'pair{pair}-compact-trace.json');(out/f'parity{c}.txt').write_text(' '.join(map(str,trace['consumed_ids']))+'\n')
    targets=ROOT.parent/'drawer_suffix/source-targets-v1';manifest=read(targets/'manifest.json');copy(targets/'manifest.json','targets/manifest.json')
    for name,row in manifest['files'].items():copy(targets/name,'targets/'+name,row['sha256'])
    copy(PRIOR/'raw/input-manifest.json','source-inputs.json',report['raw_files']['input-manifest.json']['sha256']);(out/'baseline.json').write_text(json.dumps({'model':report['model'],'source_report_sha256':sha(PRIOR/'report.json')},indent=2)+'\n')
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};(out/'source-manifest.json').write_text(json.dumps({'schema':'context-query-payload-v1','sources':files,'raw_archive_limit_bytes':LIMIT},indent=2,sort_keys=True)+'\n');print(json.dumps({'payload':str(out),'files':len(files),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}));return out
def submit(payload):
    from mdb.job_support import guarded_mrun_launch,one_lease_scheduling_contract
    payload=Path(payload).resolve();m=read(payload/'source-manifest.json')
    for p,h in m['sources'].items():base.checked(payload/p,h)
    request={'schema':'context-query-request-v1','model':'qwen2.5-1.5b-instruct','backend':'hf','candidate_backend':'C CUDA pedantic FP32 + CPU64 prototype reader','device':'cuda','dtype':'fp32','needs':{'cuda':True},'max_seq_len':2048,'cpu_threads':1,'sources':m['sources'],'source_manifest_sha256':sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'physical_work':'420 exact source clocks+32lexical parity;4signed source images x3own32+32 histories+native shadows; source-only rank128 layer18 compilation;4fresh128/1024contexts x5policies+shadows; one resident model; no target fitting.36000advance/780Csecondguard.','scheduling_contract':one_lease_scheduling_contract(model_loads=1,logical_branches=32,logical_operations=32,max_local_work_items=40,checkpoint_policy='Image-only Saturn transactions restore exact parent. C candidate sessions start empty and are destroyed before fresh native shadows. Prototype restore refused; no false cursor-only rollback.'),'resource_exception':{'ram_mb':8192,'vram_mb':8192,'calibration_job':'job-935667de6e2f','reason':'Same C FP32/capacity2048 measured6470MBVRAM and~4.3GiBRSS. One session at a time, source norms<16MB and sequential sampled-logit files<150MB.8GiB measured exception;4GiB scratch.'}}
    job=guarded_mrun_launch(['uv','run','--offline','--project','{env}','python','worker.py'],request=request,model=request['model'],task='analysis',experiment='c-state-machine-context-query',payload=payload,prefer_host='beast',needs=request['needs'],env_alias='mrun',ram_mb=8192,vram_mb=8192,disk_gb=4,cpu_threads=1,est_wall_s=650,timeout_s=900,retry_on_kill=False,preflight=True,note='Integrate public QKV/atom/prototype APIs then source-only query response direction.4signed candidates x3own32+32 source futures; native same-ID shadows select before fresh prompts. Compile QK18 from actual selected source receivers.4new128/1024 two-turn conversations xnative/base17/fit17/+QK18full/+QK18compact. One C model CPU1B1;36000advances/780Cseconds/900lease;no retries/targetfit.8GiB measured,strictpreflight.')
    d=ROOT/'runs'/job;d.mkdir(parents=True,exist_ok=False);(d/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':request},indent=2,sort_keys=True)+'\n');print(json.dumps({'job_id':job,'result_directory':str(d)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    else:base.LIMIT=LIMIT;base.collect(a.target,ROOT/'runs')
