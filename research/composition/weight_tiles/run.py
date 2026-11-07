"""One finite matched weight-storage assay over the authenticated all-byte trajectory."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.util,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sp=importlib.util.spec_from_file_location('composition_base',PARENT/'run.py');base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
PRIOR=PARENT/'compact_all/runs/job-bbd58171a613';PRIOR_SHA='368e835512f7f04cc61ad00cdb1580774d1764aa950ec28840d3e868c4791ecb';REANALYSIS_SHA='11b5eaf4d5d7b0c36b93e4d8d3a49958e758e3c53afb5457cfac1a8c110833dc';LIMIT=128*1024**2
def origin():
    r=json.loads(base.checked(PRIOR/'report.json',PRIOR_SHA).read_text());base.checked(PRIOR/'reanalysis.json',REANALYSIS_SHA);inventory=json.loads((PRIOR/'raw-inventory.json').read_text())
    return {'predecessor_report_sha256':PRIOR_SHA,'predecessor_reanalysis_sha256':REANALYSIS_SHA,'predecessor_verification_sha256':base.sha(PRIOR/'verification.json'),'model':r['model'],'correction_input':json.loads((PRIOR/'payload/origin.json').read_text())['correction_input'],'historical_byte_chain':r['driver']['chains'][2],'historical_byte_raw':{n:v for n,v in inventory.items() if n.startswith('byte_all-')}}
def stage():
    ref=origin();sub=json.loads((PRIOR/'submission.json').read_text());payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True)
    excluded={'run.py','worker.py','analysis.py','runtime_patch.py','PLAN.md','origin.json','questions.json','Makefile','test_append.py'}
    for n,h in sub['config']['sources'].items():
        if n in excluded or n.startswith('original_runtime/'):continue
        dest=payload/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/'payload'/n,h),dest)
    (payload/'original_runtime').mkdir();shutil.copy2(payload/'runtime/qwen.c',payload/'original_runtime/qwen.c');shutil.copy2(payload/'driver.c',payload/'original_driver.c')
    for n in ('PLAN.md','runtime_patch.py','weight_cuda.cu','weight_stub.c','test_storage.py','panel_bridge.py','worker.py','analysis.py','run.py'):shutil.copy2(HERE/n,payload/n)
    make=base.checked(PRIOR/'payload/Makefile',sub['config']['sources']['Makefile']).read_text();make=make.replace('driver.c cache_cuda.cu cache.h codec.h','driver.c weight_cuda.cu cache_cuda.cu cache.h codec.h').replace('-c cache_cuda.cu -o build/cuda.o','-c weight_cuda.cu -o build/cuda.o');(payload/'Makefile').write_text(make)
    sp=importlib.util.spec_from_file_location('weight_patch',HERE/'runtime_patch.py');patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(patch);patch.apply(payload)
    (payload/'origin.json').write_text(json.dumps(ref,indent=2)+'\n');shutil.copy2(PRIOR/'payload/questions.json',payload/'questions.json');sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()};manifest={'sources':sources,'plan_sha256':base.sha(payload/'PLAN.md'),'predecessor_report_sha256':PRIOR_SHA,'predecessor_reanalysis_sha256':REANALYSIS_SHA,'raw_archive_limit_bytes':LIMIT};(payload/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'payload':str(payload),'files':len(sources),'bytes':sum(p.stat().st_size for p in payload.rglob('*') if p.is_file())}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for n,h in manifest['sources'].items():base.checked(payload/n,h)
    config={'schema':'tcm-weight-tiles-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32-tile1024','weight_storage':'source BF16 matrices/rank1 FP32, matched FP32 tiled/ordinary controls','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'predecessor_report_sha256':PRIOR_SHA,'predecessor_reanalysis_sha256':REANALYSIS_SHA,'raw_archive_limit_bytes':LIMIT,'physical_work':'3sequential processes/model loads; ordinaryFP32/tiledFP32/tiledsourceBF16; all28byte caches fromtoken0;5linkedturns each; rollback/sibling/last27abort','reservation_source':'same512 C FP32 jobbbd581 RSS4205.4MB/VRAM6378MB plus36.7MBshared weight tile;6144MBVRAM guard6758.4MB','scheduling_contract':{'model_loads':3,'logical_branches':3,'linked_turns_per_branch':5,'branch_execution':'sequential_subprocesses','max_mrun_submissions':1,'collection_policy':'direct_job_only','mrun_retry_policy':'none_for_finite_lease'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-weight-tiles',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=240,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note='Lossless BF16 matrix residence: ordinaryFP32/tiledFP32/tiledBF16,3sequential model processes x5own turns; all28byte KV;1024row shared tile;compare same arithmetic,not quantization;B1 CPU1;stop15turns,no retries')
    out=HERE/'runs'/job;out.mkdir(parents=True,exist_ok=True);(out/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(out)}))
def verify(root):
    root=Path(root);sub=json.loads((root/'submission.json').read_text());job=json.loads((root/'job.json').read_text());prior=json.loads((PRIOR/'submission.json').read_text())
    if json.loads((root/'payload/origin.json').read_text())!=origin() or (root/'payload/questions.json').read_bytes()!=(PRIOR/'payload/questions.json').read_bytes():raise ValueError('Authenticated parent differs')
    sp=importlib.util.spec_from_file_location('frozen_weight_patch',root/'payload/runtime_patch.py');patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(patch)
    if base.sha(root/'payload/original_runtime/qwen.c')!=prior['config']['sources']['runtime/qwen.c'] or patch.transform('qwen.c',(root/'payload/original_runtime/qwen.c').read_text())!=(root/'payload/runtime/qwen.c').read_text() or base.sha(root/'payload/original_driver.c')!=prior['config']['sources']['driver.c'] or patch.driver_transform((root/'payload/original_driver.c').read_text())!=(root/'payload/driver.c').read_text():raise ValueError('Weight transform differs')
    for n,h in prior['config']['sources'].items():
        if n.startswith('runtime/') and n!='runtime/qwen.c' and sub['config']['sources'].get(n)!=h:raise ValueError('Pinned runtime differs')
    for k in ('sources','source_manifest_sha256','predecessor_report_sha256','predecessor_reanalysis_sha256'):
        if job['config'][k]!=sub['config'][k]:raise ValueError('Scheduler request differs')
    base.verify(root);(root/'verification-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'authenticated_parent':True,'runtime_and_driver_transforms':True,'questions_bound':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:verify(a.target)
