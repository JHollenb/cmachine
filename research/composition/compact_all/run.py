"""Finite all-layer construction cut, joined to authenticated append history."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.util,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
spec=importlib.util.spec_from_file_location('composition_base',PARENT/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PRIOR=PARENT/'compact_append/runs/job-2a28feeb7e5b';PRIOR_SHA='5d046f5b56574dafadf4422af9cd855089da1c2df99a8039efb2573db042ab4a';LIMIT=128*1024**2
def origin():
    r=json.loads(base.checked(PRIOR/'report.json',PRIOR_SHA).read_text());return {'predecessor_report_sha256':PRIOR_SHA,'model':r['model'],'correction':r['driver']['correction'],'correction_input':r['inputs']['correction'],'historical_native_turns':r['driver']['chains'][0]['turns']}
def questions():
    r=json.loads((PRIOR/'payload/baseline.json').read_text());o=origin();return [{'name':'source','text':r['source_template_text'],'budget':8,'file':'source.txt'},{'name':'correction','text':o['correction_input']['text'],'budget':8,'file':'correction.txt'},*[{'name':q['name'],'text':q['text'],'budget':q['budget'],'file':q['file']} for q in json.loads((PRIOR/'payload/questions.json').read_text())]]
def stage():
    ref=origin();previous=json.loads((PRIOR/'submission.json').read_text());payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True)
    keep={'support.py','saturn_bridge.py','baseline.json','legacy_driver.c','legacy_analysis.py','turn_helpers.h','whole_turn_analysis.py','whole_turn_driver.c','test_token_authority.py','test_append.py','codec.h','cache_cuda.cu','span_parent_analysis.py','toy_fixture.h','gpu_stub.c'}
    for name,digest in previous['config']['sources'].items():
        if name not in keep and not name.startswith(('runtime/','public/')):continue
        dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base.checked(PRIOR/'payload'/name,digest),dest)
    shutil.copy2(base.checked(PRIOR/'payload/analysis.py',previous['config']['sources']['analysis.py']),payload/'append_analysis.py')
    for name in ('driver.c','cache.h','runtime_patch.py','worker.py','analysis.py','run.py','PLAN.md','cache_mechanics.c'):shutil.copy2(HERE/name,payload/name)
    shutil.copy2(base.checked(PRIOR/'payload/Makefile',previous['config']['sources']['Makefile']),payload/'Makefile')
    (payload/'original_runtime').mkdir();shutil.copy2(payload/'runtime/qwen.c',payload/'original_runtime/qwen.c')
    spec_patch=importlib.util.spec_from_file_location('all_runtime_patch',HERE/'runtime_patch.py');patch=importlib.util.module_from_spec(spec_patch);spec_patch.loader.exec_module(patch);patch.apply(payload/'runtime')
    (payload/'origin.json').write_text(json.dumps(ref,indent=2)+'\n');(payload/'questions.json').write_text(json.dumps(questions(),indent=2)+'\n');sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()};manifest={'sources':sources,'plan_sha256':base.sha(payload/'PLAN.md'),'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT};(payload/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'payload':str(payload),'files':len(sources),'bytes':sum(p.stat().st_size for p in payload.rglob('*') if p.is_file())}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for n,h in manifest['sources'].items():base.checked(payload/n,h)
    config={'schema':'tcm-compact-all-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT,'physical_work':'one model; native/exactall/byteall from token0;5 linked turns each; every current row; rollback/sibling/last-layer abort; full retention, original weights','reservation_source':'same512 C FP32 append job2a28 RSS4060.1MB/VRAM6378MB; all-layer states no larger than native','scheduling_contract':{'model_loads':1,'logical_branches':3,'linked_turns_per_branch':5,'branch_execution':'in_process','max_mrun_submissions':1,'collection_policy':'direct_job_only','mrun_retry_policy':'none_for_finite_lease'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-compact-all',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note='All28cache append from token0: native/exact-FP32/byte x5linkedturns; no selected dense slots; one model, sequential B1 CPU1; rollback/sibling/last27abort after27publications; full retention; stop15turns,no retries')
    target=HERE/'runs'/job;target.mkdir(parents=True,exist_ok=True);(target/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(target)}))
def verify(root):
    root=Path(root);previous=json.loads((PRIOR/'submission.json').read_text());sub=json.loads((root/'submission.json').read_text());job=json.loads((root/'job.json').read_text())
    if json.loads((root/'payload/origin.json').read_text())!=origin() or json.loads((root/'payload/questions.json').read_text())!=questions():raise ValueError('Authenticated parent/questions differ')
    sp=importlib.util.spec_from_file_location('frozen_all_runtime_patch',root/'payload/runtime_patch.py');patch=importlib.util.module_from_spec(sp);sp.loader.exec_module(patch)
    for n,h in previous['config']['sources'].items():
        if n=='runtime/qwen.c':
            if base.sha(root/'payload/original_runtime/qwen.c')!=h or patch.transform('qwen.c',(root/'payload/original_runtime/qwen.c').read_text())!=(root/'payload/runtime/qwen.c').read_text():raise ValueError('Runtime transform differs')
        elif n.startswith('runtime/') and sub['config']['sources'].get(n)!=h:raise ValueError('Pinned runtime differs')
    for k in ('sources','source_manifest_sha256','predecessor_report_sha256'):
        if job['config'][k]!=sub['config'][k]:raise ValueError('Scheduler request differs')
    if (root/'analysis-repair-source.py').exists():
        sp=importlib.util.spec_from_file_location('saved_all_repair',HERE/'repair_analysis.py');repair=importlib.util.module_from_spec(sp);sp.loader.exec_module(repair);repair.repair(root)
    else:base.verify(root)
    (root/'verification-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'authenticated_parent':True,'runtime_transform':True,'questions_bound':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:verify(a.target)
