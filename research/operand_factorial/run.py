"""Freeze predecessor evidence, submit one finite lease, collect and verify."""
import argparse,base64,copy,gzip,hashlib,importlib.util,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;MACHINE=HERE.parent
PRIOR=MACHINE/'operand_basis/runs/job-51df6b792d28'
PRIOR_SHA='e5d23802467bd705ef6e1d9ea7e3dd8925f52589d0dd2942c325404e171a59b7'
LIMIT=256*1024**2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checked(p,digest):
    if sha(p)!=digest:raise ValueError('identity differs: '+str(p))
    return p

def stage():
    prior=json.loads(checked(PRIOR/'report.json',PRIOR_SHA).read_text());submission=json.loads((PRIOR/'submission.json').read_text())
    payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True)
    def source(name,destination=None):
        p=checked(PRIOR/'payload'/name,submission['config']['sources'][name]);out=payload/(destination or name);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    for name in submission['config']['sources']:
        if name.startswith(('runtime/','public/')) or name=='saturn_bridge.py':source(name)
    source('driver.c','predecessor.c')
    for name in ['driver.c','Makefile','worker.py','analysis.py','run.py','test_analysis.py','PLAN.md']:shutil.copy2(HERE/name,payload/name)
    def raw(name,destination=None):
        record=prior['raw_files'][name];p=checked(PRIOR/'raw'/name,record['sha256']);out=payload/(destination or ('saved/'+name));out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
    for name in prior['raw_files']:
        if name.startswith('compiled/') and name.endswith('.f32'):raw(name,name)
    all_clocks=[];selected=[];lines=[]
    same_ids={a['primary_name']:a for c in prior['driver']['references'] for a in c['same_ids']}
    for case in prior['driver']['primary_cases']:
        for arm in case['arms']:
            for step in arm['steps']:
                stem=step['stem']
                if stem+'-mlp_gate.f32' not in prior['raw_files']:continue
                for suffix in ['native_gate','native_up','mlp_gate','mlp_up','native_write','mlp_return','raw_receiver']:raw(stem+'-'+suffix+'.f32')
                ids=[s['input_id'] for s in arm['steps'][:step['step']+1]]
                all_clocks.append({'case':case['case'],'policy':arm['arm'],'step':step['step'],'stem':stem,'consumed_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
            if arm['arm'] not in ['source64','random64']:continue
            boundaries=['first'] if case['case']==0 else ['first','after_followup'] if case['case']==1 else []
            for boundary in boundaries:
                expected=arm[boundary]['logits']['file'];native_expected=same_ids[arm['name']][boundary]['logits']['file']
                if boundary=='first':index=len(case['prompt_tokens'])-1
                else:index=len(case['prompt_tokens'])+len(arm['answer0']['ids'])+arm['followup_steps']-1
                step=arm['steps'][index];key=f"case{case['case']}-{arm['arm']}-{boundary}"
                ids=[s['input_id'] for s in arm['steps'][:index+1]]
                (payload/'saved'/(key+'-ids.txt')).write_text(' '.join(map(str,ids))+'\n')
                raw(expected);raw(native_expected)
                selected.append({'key':key,'case':case['case'],'policy':arm['arm'],'boundary':boundary,'saved_stem':step['stem'],'ids':ids,'expected_carried_logits':expected,'expected_native_logits':native_expected})
                lines.append(f"{key} {1 if arm['arm']=='source64' else 2} saved/{key}-ids.txt")
    (payload/'saved/clocks.json').write_text(json.dumps(all_clocks,indent=2)+'\n')
    plan={'schema':'tcm-operand-factorial-clock-plan-v1','predecessor_report_sha256':PRIOR_SHA,'predecessor_job':'job-51df6b792d28','model':prior['model'],'tokenizer_path':'/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306','clocks':selected,'selection':'fixed recall and arithmetic prompt endpoints plus inspected arithmetic followup endpoint; development evidence','horizon':4,'archive_bound_bytes':190*1024**2}
    (payload/'clock-plan.json').write_text(json.dumps(plan,indent=2)+'\n');(payload/'clock-list.txt').write_text('\n'.join(lines)+'\n')
    sources={str(p.relative_to(payload)):sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    (payload/'source-manifest.json').write_text(json.dumps({'schema':'tcm-operand-factorial-payload-v1','sources':sources,'predecessor_report_sha256':PRIOR_SHA,'raw_archive_limit_bytes':LIMIT},indent=2)+'\n')
    print(json.dumps({'payload':str(payload),'sources':len(sources),'decomposition_clocks':len(all_clocks),'factorial_clocks':len(selected)}));return payload

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():checked(payload/name,digest)
    config={'schema':'tcm-operand-factorial-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32','device':'cuda','dtype':'float32','cpu_threads':1,'max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':sha(payload/'source-manifest.json'),'raw_archive_limit_bytes':LIMIT,
        'physical_work':'One model load; six imported clocks x2 prefix policies x7 branches; each branch one local operand restoration, native suffix and4 native own tokens;36 saved operand decompositions after C exits. No source compiler or training.',
        'reservation_source':'Measured exact predecessor C B1/context512 CPU1 job-51df6b792d28 RSS4031.1MB/VRAM6394MB; same original model/banks, fewer auxiliary allocations, no compiler;6144MB reservation with6758.4MB VRAM ceiling.',
        'scheduling_contract':{'model_loads_scientific_driver':1,'branch_execution':'in_process','max_mrun_submissions':1,'collection_policy':'direct_job_only','mrun_retry_policy':'none_for_finite_lease'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-operand-factorial',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=180,timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note='Operand factorial:6 development clocks xcarried/native-same-ID parents x7 current-write arms; common-parent restore, full logits,4 native future tokens;36 saved signed decompositions;B1 FP32 CPU1 context512;stop84branches;no training/retries')
    out=HERE/'runs'/job;out.mkdir(parents=True,exist_ok=True);(out/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'result_directory':str(out)}))

def support():
    spec=importlib.util.spec_from_file_location('factorial_collection',MACHINE/'composition/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.LIMIT=LIMIT;return m

def verify(root):
    root=Path(root);r=json.loads((root/'report.json').read_text());s=json.loads((root/'submission.json').read_text());j=json.loads((root/'job.json').read_text())
    assert j['job_id']==s['job_id'];assert sha(root/'logs.txt')==j['result']['artifacts']['job_log']['sha256']
    logged=[json.loads(gzip.decompress(base64.b64decode(x.split('=',1)[1]))) for x in (root/'logs.txt').read_text().splitlines() if x.startswith('COMPOSITION_RESULT_GZIP_B64=')]
    assert logged==[r];assert json.loads((root/'raw/report.json').read_text())==r
    for name,digest in s['config']['sources'].items():checked(root/'payload'/name,digest)
    checked(root/'payload/source-manifest.json',s['config']['source_manifest_sha256'])
    receipt=copy.deepcopy(r['saturn_receipt']['receipt']);fingerprint=receipt.pop('fingerprint')
    assert hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==fingerprint
    request=r['saturn_receipt']['request']
    assert request['model_digest']==r['model']['content_sha256']==r['driver']['model_digest']
    assert request['source_manifest_sha256']==s['config']['source_manifest_sha256']
    assert request['binary_sha256']==r['binary_sha256']==sha(root/'raw/binary/operand-factorial')
    assert request['clock_plan_sha256']==sha(root/'payload/clock-plan.json')
    for name,record in r['raw_files'].items():checked(root/'raw'/name,record['sha256']);assert (root/'raw'/name).stat().st_size==record['bytes']
    assert json.loads((root/'raw/driver.json').read_text())==r['driver']
    spec=importlib.util.spec_from_file_location('factorial_saved_analysis',root/'payload/analysis.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=copy.deepcopy(r['analysis'])
    for row in expected['observations']:row.pop('continuation_text',None)
    # Numerical reduction differences across BLAS implementations are reported,
    # while all C byte-replay mechanics and imported hashes remain exact.
    rebuilt=module.analyze(root/'raw',root/'payload/saved',r['driver'],r['plan'])
    import numpy as np
    def compare(a,b):
        if isinstance(a,dict):assert a.keys()==b.keys();return all(compare(a[k],b[k]) for k in a)
        if isinstance(a,list):assert len(a)==len(b);return all(compare(x,y) for x,y in zip(a,b))
        if isinstance(a,float):return bool(np.isclose(a,b,rtol=2e-10,atol=2e-12))
        return a==b
    valid=compare(expected,rebuilt)
    v={'schema':'tcm-operand-factorial-verification-v1','report_sha256':sha(root/'report.json'),'scheduler_log_bound':True,'whole_invocation_receipt':True,'frozen_sources':len(s['config']['sources']),'raw_files':len(r['raw_files']),'saved_analysis_reproduced':valid,'numerical_policy':'FP64 derived scalar replay rtol2e-10 atol2e-12 accommodates BLAS reduction order; C replay mechanics require byte equality','mechanics':rebuilt['summary']}
    (root/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));assert valid

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','collect','verify']);p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':support().collect(a.target,HERE/'runs')
    else:verify(a.target)
