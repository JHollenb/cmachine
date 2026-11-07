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
    provenance={'source_job_id':'job-b141eadfe952','source_report_sha256':EXPECTED,'model_digest':r['driver']['model_digest'],'model_identity':r['driver']['model_identity'],'source_rows':193,'rank':128,'new_compilation_forwards':0,'new_calibration_forwards_bound':7*3*512,'target_control_query_index':5,'historical_acquisition_and_compilation_paid':True,'files':{}}
    for kind in ('basis','q','k','v'):
        name=f'program128-{kind}.f32';description=raw(name)
        if kind=='q':
            description=r['raw_files']['query-proposal-5.f32'];base.checked(PREVIOUS/'raw/query-proposal-5.f32',description['sha256']);shutil.copy2(PREVIOUS/'raw/query-proposal-5.f32',out/name)
        provenance['files'][name]=description
    family={'schema':'tcm-frozen-query-dose-family-v1','source_report_sha256':EXPECTED,'source_query_compiler_sha256':r['raw_files']['query-compiler.json']['sha256'],'source_response_selected_index':4,'doses':[0,-1,.25,.5,1,2],'source_response_proposals':r['query_compiler']['proposals'],'files':{}}
    for i in range(6):
        name=f'query-proposal-{i}.f32';family['files'][name]=raw(name)
    source_prompts={'prompts':[s['text'] for s in r['inputs']['sources']],'followup':'Now turn your answer into a short practical note for someone who missed the conversation. Use two sentences.','answer_budget':32}
    (out/'source-prompts.json').write_text(json.dumps(source_prompts,sort_keys=True,indent=2)+'\n')
    consumer=MACHINE/'query_consumer/runs/job-97f4f4ac5191';consumer_report='0b8f475f60411080434491a0c4010eb496f4da27b3cf1e721e530d337fbeb75c';base.checked(consumer/'report.json',consumer_report);verification=json.loads((consumer/'verification.json').read_text());base.checked(consumer/'analysis-recovery.json',verification['analysis_recovery_sha256']);recovery=json.loads((consumer/'analysis-recovery.json').read_text());base.checked(consumer/'raw/source-selection.json',recovery['archive_files']['raw/source-selection.json']['sha256']);historical=json.loads((consumer/'raw/source-selection.json').read_text())
    if historical['selected_index']!=5 or historical['selected_image_sha256']!=family['files']['query-proposal-5.f32']['sha256']:raise ValueError('Historical all-clock control differs')
    (out/'historical-control.json').write_text(json.dumps({'job_id':'job-97f4f4ac5191','report_sha256':consumer_report,'analysis_recovery_sha256':verification['analysis_recovery_sha256'],'selection':historical},sort_keys=True,indent=2)+'\n')
    prefixes={'report_sha256':consumer_report,'analysis_recovery_sha256':verification['analysis_recovery_sha256'],'files':{},'source_lengths':[66,63,64]}
    for c in range(3):
        for label in ['native']+[f'proposal{i}' for i in range(6)]:
            for suffix in ('q.f32','k.f32','v.f32','attention.f32','tops.i32'):
                name=f'source{c}-{label}-{suffix}';record=recovery['archive_files']['raw/'+name];base.checked(consumer/'raw'/name,record['sha256']);prefixes['files'][name]=record
    for i in range(6):
        name=f'source-metrics-{i}.json';base.checked(consumer/'raw'/name,recovery['archive_files']['raw/'+name]['sha256']);shutil.copy2(consumer/'raw'/name,out/f'historical-{name}')
    (out/'prefix-provenance.json').write_text(json.dumps(prefixes,sort_keys=True,indent=2)+'\n')
    for name,value in [('program-provenance.json',provenance),('family-provenance.json',family),('baseline.json',{'model':r['model'],'predecessor_report_sha256':EXPECTED})]:(out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    files={str(p.relative_to(out)):base.sha(p) for p in sorted(out.rglob('*')) if p.is_file()};(out/'source-manifest.json').write_text(json.dumps({'schema':'tcm-query-horizon-payload-v1','sources':files,'raw_archive_limit_bytes':LIMIT},indent=2)+'\n');print(json.dumps({'payload':str(out),'sources':len(files),'payload_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}));return out
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():base.checked(payload/name,digest)
    config={'schema':'tcm-query-horizon-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32+cpu-fp64-centroid','device':'cuda','dtype':'fp32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':base.sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,'reservation_source':'sameCcapacity512 job-b141eadfe952 peakRSS4224.3MB/VRAM6380MB; bounded~300MBnativehostcache+streamaudit,one0.786MBreferenceGPUimage;6144MB*1.1ceiling6758fits','physical_work':'one resident C modelCPU1B1;32sham;3native source conversations each32+sharedfollowup+32,6forcedcandidateimages;sourcephysicaladvances7xruntime source length;allclockC/Pythonstream audits;futureprediction selection seal before3fresh x4logicalpolicies aliasdedup=>9or12physicaltargetpaths','scheduling_contract':{'gpu_model_loads':1,'branch_execution':'in_process','physical_batch_size':1,'max_retries':0,'collection_policy':'direct_job_only'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-query-horizon',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=240,timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note='Futurefocusedsourcequeryselection:3samecontexts32answer+sharedfollowup+32;nativecreatesIDs+6forcedimagehistories;pretokenpredictionclocksEOSincluded,fullC/Pythonaudit;3freshconversationsx4logicalpolicieswithaliasdedup;oneCmodelCPU1B1;stopfixedpanel')
    result=HERE/'runs'/job;result.mkdir(parents=True,exist_ok=True);(result/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(result)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','submit','collect','verify'));p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':base.LIMIT=LIMIT;base.collect(a.target,HERE/'runs')
    else:base.verify(a.target)
