"""One frozen programme x memory panel, with exact-job collection."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
MACHINE=HERE.parent
PREVIOUS=MACHINE/'query_memory_transfer/runs/job-b94ac2f8a6c9'
EXPECTED='2ad6b50ea1a88fdd73c2d99978bf2ccce826f13c1eb042362e5cfbe2acca17a3'
LIMIT=256*1024**2
spec=importlib.util.spec_from_file_location('composition_collection',MACHINE/'composition/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def stage():
    base.checked(PREVIOUS/'report.json',EXPECTED);report=json.loads((PREVIOUS/'report.json').read_text());submission=json.loads((PREVIOUS/'submission.json').read_text());verification=json.loads((PREVIOUS/'verification.json').read_text())
    if verification['report_sha256']!=EXPECTED:raise ValueError('Predecessor verification differs')
    payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True)
    for source in sorted((PREVIOUS/'payload').rglob('*')):
        relative=str(source.relative_to(PREVIOUS/'payload'))
        if source.is_file() and '__pycache__' not in source.parts and (relative.startswith('runtime/') or relative.startswith('public/')):
            base.checked(source,submission['config']['sources'][relative]);target=payload/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in ('driver.c','memory.h','Makefile','analysis.py','audit.py','compiler.py','test_compiler.py','fresh.json','test_analysis.py','test_memory.c','worker.py','run.py','PLAN.md','ADVICE.md'):shutil.copy2(HERE/name,payload/name)
    source_run=MACHINE/'query_memory/runs/job-2230480e1c18';source_sha='c4897a9c69bd8c644e026ac92dcccc959cd65c9e25f885873de459ce53ac51e2'
    base.checked(source_run/'report.json',source_sha);source_report=json.loads((source_run/'report.json').read_text());record=source_report['raw_files']['source-inputs.f32'];base.checked(source_run/'raw/source-inputs.f32',record['sha256']);shutil.copy2(source_run/'raw/source-inputs.f32',payload/'source-inputs.f32')
    if source_report['driver']['model_digest']!=report['driver']['model_digest'] or source_report['driver']['model_identity']!=report['driver']['model_identity']:raise ValueError('Historical calibration model differs')
    (payload/'source-provenance.json').write_text(json.dumps({'job_id':'job-2230480e1c18','report_sha256':source_sha,'model_digest':report['driver']['model_digest'],'model_identity':report['driver']['model_identity'],'files':{'source-inputs.f32':record},'source_trajectories':source_report['driver']['sources'],'basis_already_used_all_three_source_trajectories':True},indent=2)+'\n')
    shutil.copy2(MACHINE/'composition/run.py',payload/'support_run.py');base.checked(PREVIOUS/'payload/saturn_bridge.py',submission['config']['sources']['saturn_bridge.py']);shutil.copy2(PREVIOUS/'payload/saturn_bridge.py',payload/'saturn_bridge.py')
    provenance={'source_job_id':'job-b94ac2f8a6c9','source_report_sha256':EXPECTED,'model_digest':report['driver']['model_digest'],'model_identity':report['driver']['model_identity'],'source_rows':193,'rank':128,'new_compilation_forwards':0,'new_calibration_forwards':0,'historical_acquisition_and_compilation_paid':True,'files':{}}
    for kind in ('basis','q','k','v'):
        name=f'program128-{kind}.f32';description=report['raw_files'][name];base.checked(PREVIOUS/'raw'/name,description['sha256']);shutil.copy2(PREVIOUS/'raw'/name,payload/name);provenance['files'][name]=description
    (payload/'program-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');(payload/'baseline.json').write_text(json.dumps({'model':report['model'],'predecessor_report_sha256':EXPECTED},indent=2)+'\n')
    sources={str(p.relative_to(payload)):base.sha(p) for p in sorted(payload.rglob('*')) if p.is_file()};(payload/'source-manifest.json').write_text(json.dumps({'schema':'tcm-query-program-payload-v1','sources':sources,'raw_archive_limit_bytes':LIMIT},indent=2)+'\n');print(json.dumps({'payload':str(payload),'sources':len(sources)}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():base.checked(payload/name,digest)
    config={'schema':'tcm-query-program-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32+cpu-fp64-centroid','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'reservation_source':'measured same original C model/capacity512:jobb94ac2f8a6c9RSS4040.3MB/VRAM6406MB;one1.835MBfrozenprogram+0.786MBqueryimage;6.3MBtemporaryhostprobabilityrecorder;6144MB*1.1ceiling6758fits','physical_work':'oneCmodelB1CPU1;32nativeparityadvances;30ownpaths;3devx7policies+3freshx3;24candidatessealbefore6native;oneCPUsource-only128queryfit;no newsourceadvances;packedprobabilities;join9historicalpaths','scheduling_contract':{'gpu_model_loads':1,'branch_execution':'in_process','physical_batch_size':1,'max_retries':0,'collection_policy':'direct_job_only'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-query-program',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=180,timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note='Samecostsource-onlyqueryimage;LSon193historicalrowsB128;3devx7+3freshx3=30ownpaths;24candidatessealbefore6native;oneCmodelCPU1B1;9historicaljoins;nativeQpaidcontrols;stopafterpanel')
    result=HERE/'runs'/job;result.mkdir(parents=True,exist_ok=True);(result/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(result)}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('stage','submit','collect','verify'));parser.add_argument('target',nargs='?');args=parser.parse_args()
    if args.action=='stage':stage()
    elif args.action=='submit':submit(args.target)
    elif args.action=='collect':base.LIMIT=LIMIT;base.collect(args.target,HERE/'runs')
    else:
        if (Path(args.target)/'audit-replay.json').exists():
            from verify_replay import verify
            verify(args.target)
        else:base.verify(args.target)
