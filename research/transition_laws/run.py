"""Freeze, schedule and collect the finite all-layer receiver-law experiment."""
from __future__ import annotations
import argparse,base64,gzip,hashlib,io,json,shutil,tarfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'query_response/runs/job-b141eadfe952'
LIMIT=5*1024**3

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def stage():
    predecessor=json.loads((PRE/'submission.json').read_text())['config']['sources']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out=HERE/'staged'/stamp/'payload';out.mkdir(parents=True)
    for name,expected in predecessor.items():
        if not (name.startswith(('runtime/','public/')) or name=='saturn_bridge.py'):continue
        src=PRE/'payload'/name
        if sha(src)!=expected:raise ValueError('Predecessor source changed: '+name)
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    names=['driver.c','worker.py','run.py','analysis.py','verification.py','Makefile','inputs.json','PLAN.md','MATH.md']
    for name in names:
        if not (HERE/name).is_file():raise FileNotFoundError(name)
        shutil.copy2(HERE/name,out/name)
    specification=json.loads((HERE/'inputs.json').read_text())
    for split in ['source','monitor','heldout']:
        (out/(split+'.json')).write_text(json.dumps(specification[split],indent=2,sort_keys=True)+'\n')
    for name in ['test_analysis.py','test_compiler.py','REVIEW.md','PROTOCOL.md']:
        if (HERE/name).is_file():shutil.copy2(HERE/name,out/name)
    model=json.loads((PRE/'report.json').read_text())['model']
    (out/'model.json').write_text(json.dumps(model,indent=2,sort_keys=True)+'\n')
    manifest={'schema':'transition-laws-payload-v1','sources':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'created_utc':stamp,'predecessor_report_sha256':sha(PRE/'report.json'),'artifact_limit_bytes':LIMIT}
    (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'payload':str(out),'source_files':len(manifest['sources'])}));return out

def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();manifest=json.loads((payload/'source-manifest.json').read_text())
    for name,wanted in manifest['sources'].items():
        if sha(payload/name)!=wanted:raise ValueError(name)
    config={'schema':'transition-laws-execution-v1','backend':'hf','candidate_backend':'c-cublas-fp32','device':'cuda','dtype':'fp32','max_seq_len':512,'backend_options':{'physical_batch_size':1},'max_retries':0,'sources':manifest['sources'],'source_manifest_sha256':sha(payload/'source-manifest.json'),'physical_work':'One resident C Qwen1.5B; all28 rank256 G/U programs; native source harvest, candidate own-state harvest, one equal-size source PCA refresh; monitor3 and heldout6 two-turn conversations; independent native same-ID shadows; no nested scheduler jobs','scheduling_contract':{'model_loads':1,'branch_execution':'in_process','collection_policy':'direct_job_only','mrun_retry_policy':'none_for_finite_lease'}}
    config['reservation_exception']={'source_job':'job-1df6797e0f96','observed_peak_vram_mb':6936,'native_plan_ceiling_mb':6851.68,'two_factor_banks_bytes':1115684864,'reason':'Native-only server plan killed the measured C plus Torch compiler before the two full factor banks. Add measured compiler context, both exact-size banks and bounded workspace. RAM retains prior planned envelope; disk includes raw plus custody tar.'}
    reservation={'ram_mb':8630.4,'vram_mb':9216,'cpu_threads':4,'disk_gb':12,'est_wall_s':360}
    job=launch('uv run --offline --project {env} python worker.py',experiment='c-state-machine-transition-laws',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',needs={'cuda':True},config=config,**reservation,env_alias='mrun',timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note='All28 MLP gate/up rank256: ~40.9% nominal linear saving; native-only vs one own-state refresh; source6/monitor3/fresh6; one resident C model. Measured VRAM exception after job-1df6797e0f96 native-only planner kill: 6936MB observed plus two 557842432B factor banks and compiler workspace. Stop at finite panel.')
    dst=HERE/'runs'/str(job);dst.mkdir(parents=True,exist_ok=True)
    (dst/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'job_id':str(job),'result_directory':str(dst)}))

def collect(job):
    from mrun.client.api import Api
    api=Api();_,body,_=api.request('GET',f'/api/jobs/{job}',timeout_s=20)
    _,logs,_=api.request('GET',f'/api/jobs/{job}/logs?offset=0&wait_s=1',timeout_s=30)
    dst=HERE/'runs'/job;dst.mkdir(parents=True,exist_ok=True)
    (dst/'job.json').write_bytes(body);(dst/'logs.txt').write_bytes(logs)
    result=None;asset=None
    for line in logs.decode('utf-8','replace').splitlines():
        if line.startswith('TRANSITION_LAWS_RESULT_GZIP_B64='):
            result=json.loads(gzip.decompress(base64.b64decode(line.split('=',1)[1])))
            path=dst/'report.json';encoded=json.dumps(result,indent=2,sort_keys=True)+'\n'
            if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Immutable report changed')
            path.write_text(encoded)
        if line.startswith('TRANSITION_LAWS_CUSTODY='):asset=json.loads(line.split('=',1)[1])
    if asset and not (dst/'collection-complete.json').exists():
        from mrun.mlops.publish import make_s3_client
        client=make_s3_client(endpoint_url=asset.get('endpoint_url'))
        try:client.head_object(Bucket=asset['bucket'],Key=asset['key'])
        except Exception as error:
            if asset.get('endpoint_url') or getattr(error,'response',{}).get('Error',{}).get('Code')!='404':raise
            # The first worker used Saturn's documented legacy checkpoint
            # endpoint but omitted it from the ref. Resolve that historical
            # ref explicitly, then authenticate the full bytes below.
            client=make_s3_client(endpoint_url='http://192.168.68.202:9010')
            client.head_object(Bucket=asset['bucket'],Key=asset['key'])
        asset={**asset,'endpoint_url':client.meta.endpoint_url}
        archive=dst/'evidence.tar';client.download_file(asset['bucket'],asset['key'],str(archive))
        if archive.stat().st_size!=asset['bytes'] or sha(archive)!=asset['sha256']:raise ValueError('Artifact custody mismatch')
        with tarfile.open(archive,'r:') as tar:
            members=tar.getmembers()
            if sum(m.size for m in members)>LIMIT:raise ValueError('Artifact expansion limit')
            for member in members:
                p=Path(member.name)
                if not member.isfile() or p.is_absolute() or '..' in p.parts or p.parts[0] not in ('raw','payload'):raise ValueError('Invalid artifact member')
                target=dst/p;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
        (dst/'collection-complete.json').write_text(json.dumps(asset,indent=2,sort_keys=True)+'\n');archive.unlink()
    print(json.dumps({'job':job,'state':json.loads(body).get('state'),'report':bool(result),'custody':bool(asset),'error':result.get('error') if result else None}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','collect','verify']);p.add_argument('target',nargs='?');a=p.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':collect(a.target)
    else:
        from verification import verify
        print(json.dumps(verify(HERE/'runs'/a.target),sort_keys=True))
