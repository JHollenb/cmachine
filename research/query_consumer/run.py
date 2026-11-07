"""Source-bound one-lease submission and exact-job collection."""
import argparse,importlib.util,json,shutil
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;MACHINE=HERE.parent;PREVIOUS=MACHINE/'query_response/runs/job-b141eadfe952';EXPECTED='2ff3b934b26b496c054be44feef4767bedda779866bb33372a6498f4b1addc53';LIMIT=256*1024**2
spec=importlib.util.spec_from_file_location('composition_collection',MACHINE/'composition/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def stage():
    base.checked(PREVIOUS/'report.json',EXPECTED);r=json.loads((PREVIOUS/'report.json').read_text());submission=json.loads((PREVIOUS/'submission.json').read_text())
    if json.loads((PREVIOUS/'verification.json').read_text())['report_sha256']!=EXPECTED:raise ValueError('Predecessor verification differs')
    out=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';out.mkdir(parents=True)
    for p in sorted((PREVIOUS/'payload').rglob('*')):
        relative=str(p.relative_to(PREVIOUS/'payload'))
        if p.is_file() and '__pycache__' not in p.parts and (relative.startswith('runtime/') or relative.startswith('public/') or relative=='saturn_bridge.py'):
            base.checked(p,submission['config']['sources'][relative]);target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ('driver.c','distribution.h','memory.h','Makefile','reader.py','metrics.py','analysis.py','worker.py','run.py','fresh.json','PLAN.md','test_metrics.py','test_distribution.c','test_memory.c'):shutil.copy2(HERE/name,out/name)
    def raw(name):
        row=r['raw_files'][name];p=PREVIOUS/'raw'/name;base.checked(p,row['sha256']);shutil.copy2(p,out/name);return row
    for name in ('source-provenance.json','source-inputs.f32'):raw(name)
    provenance={'source_job_id':'job-b141eadfe952','source_report_sha256':EXPECTED,'model_digest':r['driver']['model_digest'],'model_identity':r['driver']['model_identity'],'source_rows':193,'rank':128,'new_compilation_forwards':0,'new_calibration_forwards':1351,'historical_acquisition_and_compilation_paid':True,'files':{}}
    for kind in ('basis','q','k','v'):
        name=f'program128-{kind}.f32';provenance['files'][name]=raw(name)
    family={'schema':'tcm-frozen-query-dose-family-v1','source_report_sha256':EXPECTED,'source_query_compiler_sha256':r['raw_files']['query-compiler.json']['sha256'],'source_response_selected_index':4,'doses':[0,-1,.25,.5,1,2],'source_response_proposals':r['query_compiler']['proposals'],'files':{}}
    for i in range(6):
        name=f'query-proposal-{i}.f32';family['files'][name]=raw(name)
    for name,value in [('program-provenance.json',provenance),('family-provenance.json',family),('baseline.json',{'model':r['model'],'predecessor_report_sha256':EXPECTED})]:(out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    files={str(p.relative_to(out)):base.sha(p) for p in sorted(out.rglob('*')) if p.is_file()};(out/'source-manifest.json').write_text(json.dumps({'schema':'tcm-query-consumer-payload-v1','sources':files,'raw_archive_limit_bytes':LIMIT},indent=2)+'\n');print(json.dumps({'payload':str(out),'sources':len(files),'payload_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}));return out
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():base.checked(payload/name,digest)
    config={'schema':'tcm-query-consumer-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32+cpu-fp64-centroid','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'reservation_source':'sameCcapacity512 job-b141eadfe952 peakRSS4224.3MB/VRAM6380MB; add117.3MB nativehostcache+stream audit,one0.786MBreferenceGPUimage;6144MB*1.1ceiling6758fits','physical_work':'one resident C modelCPU1B1;32sham;1351fixedsource advances(native193+6x193),6fullclockC/Pythonstream audits;source-selection seal then3dev+3fresh x4logicalpolicies,aliases dedup=>18or24physicaltargetpaths;no targetselection','scheduling_contract':{'gpu_model_loads':1,'branch_execution':'in_process','physical_batch_size':1,'max_retries':0,'collection_policy':'direct_job_only'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-query-consumer',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=180,timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note='Select6frozenquerydoses by nativeCsourceKL;1351forcedsource+32sham;all193sourceclocks x6 independentlystreamaudited;sealbeforefresh;6conversations x4logicalpolicies withaliasdedup;oneCmodelCPU1B1;stopfixedpanel')
    result=HERE/'runs'/job;result.mkdir(parents=True,exist_ok=True);(result/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(result)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:
        if (Path(a.target)/'analysis-recovery.json').exists():
            from recover import verify
            verify(a.target)
        else:base.verify(a.target)
