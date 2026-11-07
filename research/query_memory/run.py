"""Source-frozen finite execution and authenticated collection."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
MACHINE=HERE.parent
spec=importlib.util.spec_from_file_location('composition_collection',MACHINE/'composition/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMIT=128*1024**2
def stage():
    previous=MACHINE/'results/job-11cdb86ad29b';baseline=json.loads((previous/'report.json').read_text());submission=json.loads((previous/'submission.json').read_text())
    payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';(payload/'runtime').mkdir(parents=True)
    for name in ('model.c','machine.h','gpu.h','qwen.h','qwen.c','qwen_cuda.cu'):shutil.copy2(MACHINE/'src'/name,payload/'runtime'/name)
    for name in ('run.py','worker.py','analysis.py','test_analysis.py','weight_basis.py','source_basis.py','driver.c','memory.h','test_memory.c','Makefile','PLAN.md'):shutil.copy2(HERE/name,payload/name)
    source_run=HERE/'runs/job-2230480e1c18';source_report=json.loads((source_run/'report.json').read_text())
    provenance={'job_id':'job-2230480e1c18','report_sha256':base.sha(source_run/'report.json'),'model_digest':source_report['driver']['model_digest'],'model_identity':source_report['driver']['model_identity'],'files':{}}
    for old,new in (('source-inputs.f32','source-inputs.f32'),('program-basis.f32','source-prefix32.f32')):
        base.checked(source_run/'raw'/old,source_report['raw_files'][old]['sha256']);shutil.copy2(source_run/'raw'/old,payload/new);provenance['files'][new]=base.sha(payload/new)
    (payload/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    shutil.copy2(MACHINE/'composition/run.py',payload/'support_run.py');shutil.copy2(MACHINE/'tools/saturn_bridge.py',payload/'saturn_bridge.py')
    for source in sorted((previous/'payload/public').rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            relative=source.relative_to(previous/'payload');base.checked(source,submission['config']['sources'][str(relative)]);target=payload/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (payload/'baseline.json').write_text(json.dumps({'model':baseline['model'],'predecessor_report_sha256':base.sha(previous/'report.json')},indent=2)+'\n')
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()};(payload/'source-manifest.json').write_text(json.dumps({'schema':'tcm-query-memory-payload-v1','sources':sources,'raw_archive_limit_bytes':LIMIT},indent=2)+'\n');print(json.dumps({'payload':str(payload),'sources':len(sources)}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():base.checked(payload/name,digest)
    config={'schema':'tcm-query-memory-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32+cpu-fp64-centroid','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'reservation_source':'measured same full C model/capacity512:gate-lawjob91ba52c5fe9aRSS3981.4MB/VRAM6378MB; newGPUprogram458880B,paritysecondsession~30MB;6144MB*1.1VRAMceiling6758MB fits','physical_work':'oneCmodel;32nativeparitytokensteps;3sourceprompts8owntokens,96sourceprojectioncompiles;3twoturncasesx5policies8/16/32answerbudgets;12candidatessealbefore3native;allheadersgeneratedEOSconsumed','scheduling_contract':{'gpu_model_loads':1,'branch_execution':'in_process','physical_batch_size':1,'max_retries':0,'collection_policy':'direct_job_only'}}
    config['physical_work']='oneCmodel;32nativeparitysteps;authenticated193historicalsource-activationrows,preserve32basisbytesandexpandto128,480GPUimagecompilecalls;3twoturncasesx7policies8/16/32answers;18candidatessealbefore3native;no newcalibrationforwards'
    config['compiler_kind']='authenticated-source-activation-prefix32-plus96'
    config['program_ranks']=[32,128]
    config['reservation_source']='measured full C model/capacity512:job2230480e1c18RSS3994.6MB/VRAM6406MB inclparitysecondsession;new2programs2294400GPUbytes;6144MB*1.1VRAMceiling6758MBfits'
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-query-memory',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=240,timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note='Own-query/memory source-span refinement:authenticated193historicalsourceinputs;preserveoriginal32basisbytesandexpand128;16recent16prototypes;7armsx3twoturntasks;18candidatessealbefore3native;oneCmodelB1CPU1;no newcalibrationforwards;qualify9controlsand6factor32pathsagainstjob2230480e1c18;stopafterpanel')
    target=HERE/'runs'/job;target.mkdir(parents=True,exist_ok=True);(target/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(target)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:base.verify(a.target)
