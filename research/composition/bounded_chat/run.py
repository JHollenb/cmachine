"""Finite persistent-interface integration, direct custody and saved verification."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.util,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sp=importlib.util.spec_from_file_location('composition_base',PARENT/'run.py');base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
PRIOR=PARENT/'bounded_retention/runs/job-1d8106dee187';PRIOR_SHA='eb72cf9b4111d5e3f9310ee3886add6d86eff7d0bfe643ea5848b64c70726df6';LIMIT=24*1024**2
def origin():
    r=json.loads(base.checked(PRIOR/'report.json',PRIOR_SHA).read_text());return {'predecessor_report_sha256':PRIOR_SHA,'predecessor_verification_sha256':base.sha(PRIOR/'verification.json'),'model':r['model'],'inputs':r['inputs'],'historical':{policy:r['driver']['arms'][i] for policy,i in (('full-byte',1),('bounded',4))}}
def stage():
    ref=origin();sub=json.loads((PRIOR/'submission.json').read_text());payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True)
    keep=('weight_cuda.cu','cache_cuda.cu','ring_cuda.cu','ring.h','codec.h','support.py','saturn_bridge.py','gpu_stub.c','weight_stub.c','mechanics.c','test_mechanics.py')
    for n,h in sub['config']['sources'].items():
        if not (n in keep or n.startswith('runtime/') or n.startswith('public/')):continue
        p=payload/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/'payload'/n,h),p)
    # Keep the predecessor's independent mechanics test separate from the new analyzer.
    shutil.copy2(base.checked(PRIOR/'payload/analysis.py',sub['config']['sources']['analysis.py']),payload/'retention_analysis.py')
    text=(payload/'test_mechanics.py').read_text().replace('from analysis import','from retention_analysis import');(payload/'test_mechanics.py').write_text(text)
    for n in ('PLAN.md','run.py','worker.py','analysis.py','controller.py','install.py','bridge.py','server.c','test_client.py'):shutil.copy2(HERE/n,payload/n)
    shutil.copy2(HERE.parents[1]/'tools/chat.py',payload/'chat.py')
    make=base.checked(PRIOR/'payload/Makefile',sub['config']['sources']['Makefile']).read_text().replace('build/composition','build/tcm-bounded-chat').replace('driver.c','server.c');(payload/'Makefile').write_text(make)
    (payload/'origin.json').write_text(json.dumps(ref,indent=2)+'\n');sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()};manifest={'sources':sources,'plan_sha256':base.sha(payload/'PLAN.md'),'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT};(payload/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'payload':str(payload),'files':len(sources),'bytes':sum(p.stat().st_size for p in payload.rglob('*') if p.is_file())}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();m=json.loads((payload/'source-manifest.json').read_text())
    for n,h in m['sources'].items():base.checked(payload/n,h)
    config={'schema':'tcm-bounded-chat-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32-tile1024-persistent','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':768,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':m['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT,'physical_work':'7sequential persistent processes:2protocol historical+replay,4actual text-client demos/delayed ledgers,1unchanged native client; sourceBF16 tiled/fullbyte vsbounded112, opt-in versioned install; no service','reservation_source':'job1d8106dee187 measuredRSS4227.7MB/VRAM3526MB; native d4ba default measured6378MB VRAM fits retained6144MB guard6758.4MB; models sequential','scheduling_contract':{'model_loads':7,'logical_branches':7,'branch_execution':'sequential_processes','max_mrun_submissions':1,'collection_policy':'direct_job_only','mrun_retry_policy':'none_for_finite_lease'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-bounded-chat',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=240,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note='Opt-in persistent bounded chat:7sequential processes,B1 CPU1;2historical protocol/replay,4real client demos/ledgers,1native fallback; host-local install bounded-current after mechanics; stable current untouched; stop7,no retries')
    out=HERE/'runs'/job;out.mkdir(parents=True,exist_ok=True);(out/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(out)}))
def verify(root):
    root=Path(root);sub=json.loads((root/'submission.json').read_text());job=json.loads((root/'job.json').read_text());prior=json.loads((PRIOR/'submission.json').read_text())
    if json.loads((root/'payload/origin.json').read_text())!=origin():raise ValueError('Historical protocol reference differs')
    for n,h in prior['config']['sources'].items():
        if n.startswith('runtime/') or n in ('ring.h','ring_cuda.cu','cache_cuda.cu','weight_cuda.cu','codec.h'):
            if sub['config']['sources'].get(n)!=h:raise ValueError('Frozen memory/arithmetic source differs')
    for k in ('sources','source_manifest_sha256','predecessor_report_sha256'):
        if job['config'][k]!=sub['config'][k]:raise ValueError('Scheduler request differs')
    r=json.loads((root/'report.json').read_text())
    if json.loads((root/'raw/request.json').read_text())!=r['saturn_receipt']['request']:raise ValueError('Persistent request receipt differs')
    base.verify(root);(root/'verification-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'historical_parent_authenticated':True,'memory_and_weight_source_unchanged':True,'installation_receipt_saved':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:verify(a.target)
