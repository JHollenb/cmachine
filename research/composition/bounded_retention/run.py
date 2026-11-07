"""Stage, launch, collect and verify one bounded-retention development assay."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.util,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sp=importlib.util.spec_from_file_location('composition_base',PARENT/'run.py');base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
PRIOR=PARENT/'weight_tiles/runs/job-02559c65b72e';PRIOR_SHA='1b09e1ebb205c5dac1dc5b08c3684f3e5f08082606c00774e90f6f942d97d85d';LIMIT=192*1024**2
PANEL=[
{'name':'source','text':"We are keeping a small ledger for later questions. Please remember the following independent facts carefully. Eno's number is 17. Tavi's number is 23. Luma's color is amber. Reply only OK.",'budget':8},
{'name':'correction','text':"Update the ledger: Tavi's number is now 29, replacing 23. Reply only OK.",'budget':8},
{'name':'delay','text':"For an unrelated reading exercise, read the following passage and reply only OK. "+"A path crosses a quiet field beside a stream. The morning light reaches the trees, and a wooden bridge leads toward a distant hill. A traveler walks slowly, listening to the wind and watching clouds move across the sky. Near the bridge, smooth stones line the bank and tall grass moves in gentle waves. The path continues past a garden, a gate and a small house. ".__mul__(3),'budget':8},
{'name':'early','text':"According to our ledger, what is Eno's number? Reply with only the number.",'budget':8,'expected':'17'},
{'name':'late0','text':"According to our ledger, what is Eno's number? Reply with only the number.",'budget':8,'expected':'17'},
{'name':'late1','text':"According to our ledger, what is Tavi's current number? Reply with only the number.",'budget':8,'expected':'29'},
{'name':'late2','text':"According to our ledger, what is Luma's color? Reply with only the color.",'budget':8,'expected':'amber'},
{'name':'late3','text':"What is Eno's number plus Tavi's current number in our ledger? Reply with only the sum.",'budget':16,'expected':'46'}]
def origin():
    r=json.loads(base.checked(PRIOR/'report.json',PRIOR_SHA).read_text());return {'predecessor_report_sha256':PRIOR_SHA,'predecessor_verification_sha256':base.sha(PRIOR/'verification.json'),'model':r['model'],'contract':'source BF16 matrix residence and matched FP32 tile consumer; new retention policy'}
def stage():
    ref=origin();sub=json.loads((PRIOR/'submission.json').read_text());payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True)
    keep=('weight_cuda.cu','cache_cuda.cu','codec.h','support.py','saturn_bridge.py','gpu_stub.c','weight_stub.c')
    for n,h in sub['config']['sources'].items():
        if not (n in keep or n.startswith('runtime/') or n.startswith('public/')):continue
        p=payload/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/'payload'/n,h),p)
    (payload/'original_runtime').mkdir()
    for n in ('qwen.c','qwen.h'):shutil.copy2(payload/'runtime'/n,payload/'original_runtime'/n)
    for n in ('PLAN.md','run.py','worker.py','analysis.py','runtime_patch.py','ring.h','ring_cuda.cu','driver.c','mechanics.c','test_mechanics.py'):shutil.copy2(HERE/n,payload/n)
    make=base.checked(PRIOR/'payload/Makefile',sub['config']['sources']['Makefile']).read_text().replace('weight_cuda.cu cache_cuda.cu cache.h codec.h','ring_cuda.cu ring.h weight_cuda.cu cache_cuda.cu codec.h').replace('-c weight_cuda.cu -o build/cuda.o','-c ring_cuda.cu -o build/cuda.o');(payload/'Makefile').write_text(make)
    sp=importlib.util.spec_from_file_location('retention_patch',HERE/'runtime_patch.py');patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(patch);patch.apply(payload)
    (payload/'origin.json').write_text(json.dumps(ref,indent=2)+'\n');(payload/'panel.json').write_text(json.dumps(PANEL,indent=2)+'\n');sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()};manifest={'sources':sources,'plan_sha256':base.sha(payload/'PLAN.md'),'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT};(payload/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'payload':str(payload),'files':len(sources),'bytes':sum(p.stat().st_size for p in payload.rglob('*') if p.is_file())}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for n,h in manifest['sources'].items():base.checked(payload/n,h)
    config={'schema':'tcm-bounded-retention-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32-tile1024','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':768,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT,'physical_work':'one BF16 source weight load, five sequential sessions: native/fullbyte/recent64/anchors16recent64/reservoir32;8own-token turns each, delayed independent forks; bounded3snapshot recovery','reservation_source':'measured job02559c65 RSS4242.9MB/VRAM6386MB includes full FP32 controls; this run source BF16 plus at most50MB recovery state;6144MB upper bound retained','scheduling_contract':{'model_loads':1,'logical_branches':5,'branch_execution':'in_process_sequential','max_mrun_submissions':1,'collection_policy':'direct_job_only','mrun_retry_policy':'none_for_finite_lease'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-bounded-retention',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=240,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note='Bounded C memory: one BF16 model load,5sequential own-token arms x8turns; recent64/anchors16/reservoir32 vsfull/native; logical768, B1 CPU1; snapshot/abort and delayed recall, no retries; stop40turns')
    out=HERE/'runs'/job;out.mkdir(parents=True,exist_ok=True);(out/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(out)}))
def verify(root):
    root=Path(root);sub=json.loads((root/'submission.json').read_text());job=json.loads((root/'job.json').read_text());prior=json.loads((PRIOR/'submission.json').read_text())
    if json.loads((root/'payload/origin.json').read_text())!=origin() or json.loads((root/'payload/panel.json').read_text())!=PANEL:raise ValueError('Parent or planned inputs differ')
    sp=importlib.util.spec_from_file_location('frozen_retention_patch',root/'payload/runtime_patch.py');patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(patch)
    for n in ('qwen.c','qwen.h'):
        if base.sha(root/'payload/original_runtime'/n)!=prior['config']['sources']['runtime/'+n] or patch.transform(n,(root/'payload/original_runtime'/n).read_text())!=(root/'payload/runtime'/n).read_text():raise ValueError('Recovery transform differs')
    for k in ('sources','source_manifest_sha256','predecessor_report_sha256'):
        if job['config'][k]!=sub['config'][k]:raise ValueError('Scheduler request differs')
    if json.loads((root/'raw/inputs.json').read_text())!=json.loads((root/'report.json').read_text())['inputs']:raise ValueError('Report inputs differ from actual token files')
    base.verify(root);(root/'verification-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'authenticated_parent':True,'recovery_transform':True,'planned_questions':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:verify(a.target)
