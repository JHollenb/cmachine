"""Stage, launch once, and reuse the authenticated composition collector."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
MACHINE = HERE.parent
spec = importlib.util.spec_from_file_location('composition_collection',MACHINE/'composition/run.py')
base = importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def stage():
    previous = MACHINE/'results/job-11cdb86ad29b'
    baseline = json.loads((previous/'report.json').read_text())
    submission = json.loads((previous/'submission.json').read_text())
    payload = HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload'
    (payload/'runtime').mkdir(parents=True)
    for name in ('model.c','machine.h','gpu.h','qwen_cuda.cu'):
        shutil.copy2(MACHINE/'src'/name,payload/'runtime'/name)
    for name in ('run.py','worker.py','analysis.py','test_analysis.py','audit.py','evidence.py','driver.c','Makefile','PLAN.md'):
        shutil.copy2(HERE/name,payload/name)
    shutil.copy2(MACHINE/'composition/run.py',payload/'support_run.py')
    shutil.copy2(MACHINE/'tools/saturn_bridge.py',payload/'saturn_bridge.py')
    for source in sorted((previous/'payload/public').rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            relative = source.relative_to(previous/'payload')
            base.checked(source,submission['config']['sources'][str(relative)])
            target = payload/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (payload/'baseline.json').write_text(json.dumps({'model':baseline['model'],'predecessor_report_sha256':base.sha(previous/'report.json')},indent=2)+'\n')
    sources = {str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    (payload/'source-manifest.json').write_text(json.dumps({'schema':'tcm-state-quotient-payload-v1','sources':sources,'raw_archive_limit_bytes':base.LIMIT},indent=2)+'\n')
    print(json.dumps({'payload':str(payload),'source_files':len(sources)}));return payload

def submit(payload):
    from mrun.client.submit import launch
    payload = Path(payload).resolve();manifest = json.loads((payload/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():base.checked(payload/name,digest)
    config = {'schema':'tcm-state-quotient-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32-component','device':'cuda','dtype':'fp32','cpu_threads':1,'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':base.LIMIT,'physical_work':'128 supplied d8/h4 recurrent component updates,384 tiny matvecs;256x256 exact modular minor;zero Qwen forwards;CPU mmap identity check only','scheduling_contract':{'model_loads_gpu':0,'logical_paths':8,'physical_batch_size':1,'branch_execution':'in_process','max_retries':0,'collection_policy':'direct_job_only'}}
    job = launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-nonlinear-quotient',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',timeout_s=300,detach=True,retry_on_kill=False,preflight=True,note='Nonlinear quotient:4compact paths sealed before4full refs;16 own updates each,d8/h4,384tiny cuBLAS matvecs;no Qwen GPU bank/forwards;CPU checksum mmap+256square original down minor exact modular rank;stop after one condition/escape/repair panel;intent-first sizing')
    target = HERE/'runs'/job;target.mkdir(parents=True,exist_ok=True)
    (target/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n')
    print(json.dumps({'job_id':job,'result_directory':str(target)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.collect(a.target,HERE/'runs')
    else:base.verify(a.target)
