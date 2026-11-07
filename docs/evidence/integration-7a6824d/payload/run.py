"""Commit-only staging, guarded fleet execution and offline verification."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,gzip,hashlib,importlib.util,io,json,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'validation'
LIMIT=24*1024**2
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*map(str,args)],cwd=ROOT)
def stage(commit):
    commit=git('rev-parse',commit).decode().strip();tree=git('rev-parse',commit+'^{tree}').decode().strip();baseline=git('rev-parse','b81e928').decode().strip()
    payload=HERE/'staged'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')/'payload';payload.mkdir(parents=True);sources={}
    def add(path,dest,rev=commit):
        data=git('show',rev+':'+path);out=payload/dest;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);sources[dest]={'git_commit':rev,'git_path':path,'sha256':hashlib.sha256(data).hexdigest()}
    names=git('ls-tree','-r','--name-only',commit,'src','tests','tools','Makefile','validation').decode().splitlines()
    for path in names:
        if path.startswith('validation/vendor/'):
            add(path,'public/'+path[len('validation/vendor/'):])
        elif path.startswith('validation/'):
            name=Path(path).name
            if '/' in path[len('validation/'):]:continue
            add(path,'origin.json' if name=='reference.json' else name)
        else:add(path,path)
    for path in git('ls-tree','-r','--name-only',baseline,'src').decode().splitlines():add(path,'baseline/'+Path(path).name,baseline)
    add('docs/evidence/local-checks.json','local-checks.json')
    manifest={'schema':'cmachine-commit-source-manifest-v1','git_commit':commit,'git_tree':tree,'native_reference_commit':baseline,'sources':sources,'raw_archive_limit_bytes':LIMIT,'source_rule':'every shipped source is exact git-show bytes from its recorded commit/path'}
    (payload/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'payload':str(payload),'git_commit':commit,'source_files':len(sources),'bytes':sum(p.stat().st_size for p in payload.rglob('*') if p.is_file())}));return payload
def submit(payload):
    from mrun.client.submit import launch
    payload=Path(payload).resolve();m=json.loads((payload/'source-manifest.json').read_text())
    for name,row in m['sources'].items():
        if sha(payload/name)!=row['sha256'] or git('show',row['git_commit']+':'+row['git_path'])!=(payload/name).read_bytes():raise ValueError('Payload differs from committed source')
    config={'schema':'cmachine-commit-integration-execution-v1','backend':'hf','device':'cuda','dtype':'fp32','candidate_backend':'committed-c-cuda-native-and-tile1024','cpu_threads':1,'max_seq_len':768,'backend_options':{'physical_batch_size':1},'max_retries':0,'source_commit':m['git_commit'],'native_reference_commit':m['native_reference_commit'],'sources':m['sources'],'source_manifest_sha256':sha(payload/'source-manifest.json'),'physical_work':'12 sequential model processes:original/integrated native protocols,3encoded-cache profiles,2real lifecycle assays,5text clients; no simultaneous models','reservation_source':'bounded-chat job-f53e7d9c952f RSS4202.5MB/VRAM6394MB including native reference; same geometry and sequential processes','scheduling_contract':{'model_loads':12,'branch_execution':'sequential_processes','max_mrun_submissions':1,'mrun_retry_policy':'none_for_finite_lease'}}
    job=launch('uv run --offline --project {env} python worker.py',experiment='cmachine-commit-integration',payload=payload,model='qwen2.5-1.5b-instruct',prefer_host='beast',pin='beast',needs={'cuda':True},config=config,env_alias='mrun',ram_mb=6144,vram_mb=6144,cpu_threads=1,est_wall_s=300,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,note='Verify committed cmachine integration:12sequential B1 CPU1 model processes; native versus original Git core, tiled/byte/bounded historical parity, real lineage+overwrite abort, actual clients; publish canonical/legacy aliases after checks; stop12,no retries. Beast pin for host-local install.')
    out=HERE/'runs'/job;out.mkdir(parents=True,exist_ok=True);(out/'submission.json').write_text(json.dumps({'job_id':job,'payload':str(payload),'config':config},indent=2)+'\n');print(json.dumps({'job_id':job,'git_commit':m['git_commit'],'result_directory':str(out)}))
def collect(job):
    from mrun.client.api import Api
    api=Api();_,record,_=api.request('GET',f'/api/jobs/{job}',timeout_s=20);_,logs,_=api.request('GET',f'/api/jobs/{job}/logs?offset=0&wait_s=1',timeout_s=30);_,receipt,_=api.request('GET',f'/api/jobs/{job}/receipt',timeout_s=20);target=HERE/'runs'/job;target.mkdir(parents=True,exist_ok=True);j=json.loads(record);(target/'job.json').write_text(json.dumps(j,indent=2)+'\n');(target/'guarded-receipt.json').write_bytes(receipt);(target/'logs.txt').write_bytes(logs)
    report=None
    for line in logs.decode().splitlines():
        if line.startswith('COMPOSITION_RESULT_GZIP_B64='):
            report=json.loads(gzip.decompress(base64.b64decode(line.split('=',1)[1])));(target/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        elif line.startswith('COMPOSITION_FILES_TGZ_B64='):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split('=',1)[1])),mode='r:gz') as tar:
                members=tar.getmembers()
                if sum(x.size for x in members)>LIMIT:raise ValueError('Archive exceeds declared bound')
                for member in members:
                    p=Path(member.name)
                    if not member.isfile() or p.is_absolute() or '..' in p.parts or p.parts[0] not in ('raw','payload'):raise ValueError('Invalid custody member')
                    data=tar.extractfile(member).read();out=target/p
                    if out.exists() and out.read_bytes()!=data:raise ValueError('Collected evidence changed')
                    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    print(json.dumps({'job_id':job,'state':j['state'],'has_report':report is not None,'error':report.get('error') if report else None,'summary':report.get('analysis',{}).get('summary') if report else None}))
def verify(target):
    target=Path(target).resolve();report=json.loads((target/'report.json').read_text());job=json.loads((target/'job.json').read_text());receipt=json.loads((target/'guarded-receipt.json').read_bytes());manifest=json.loads((target/'payload/source-manifest.json').read_text());sub=json.loads((target/'submission.json').read_text())
    if job['job_id']!=sub['job_id'] or receipt['job_id']!=job['job_id'] or job['config']['sources']!=manifest['sources'] or job['config']['source_commit']!=manifest['git_commit']:raise ValueError('Job/submission/source identity differs')
    if job['state']!='succeeded' or sha(target/'logs.txt')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Scheduler success/log seal differs')
    if manifest['git_commit']!=report['git_commit'] or git('rev-parse',manifest['git_commit']+'^{tree}').decode().strip()!=manifest['git_tree'] or sub['config']['source_manifest_sha256']!=sha(target/'payload/source-manifest.json'):raise ValueError('Commit/tree/submission join differs')
    for name,row in manifest['sources'].items():
        if sha(target/'payload'/name)!=row['sha256'] or git('show',row['git_commit']+':'+row['git_path'])!=(target/'payload'/name).read_bytes():raise ValueError('Executed source differs from Git object: '+name)
    local=json.loads((target/'payload/local-checks.json').read_text());candidate={r['git_path']:r['sha256'] for r in manifest['sources'].values() if r['git_commit']==manifest['git_commit']}
    if any(candidate.get(n)!=h for n,h in local['source_sha256'].items()) or any(r['returncode'] for r in local['checks']):raise ValueError('Local checks do not bind the committed source')
    submitted=receipt['submitted_request']
    if submitted['config']['source_commit']!=manifest['git_commit'] or submitted['config']['sources']!=manifest['sources']:raise ValueError('Guarded submitted request differs')
    custody=receipt['payload_custody'];declared=custody['declared'];sealed=custody['sealed'];executed=custody['executed']
    for row in (sealed,executed):
        if (row['sha256'],row['size_bytes'])!=(declared['sha256'],declared['size_bytes']):raise ValueError('Guarded declared/sealed/executed payload differs')
    for name,row in report['raw_files'].items():
        if sha(target/'raw'/name)!=row['sha256'] or (target/'raw'/name).stat().st_size!=row['bytes']:raise ValueError('Raw artifact differs')
    logged=[json.loads(gzip.decompress(base64.b64decode(x.split('=',1)[1]))) for x in (target/'logs.txt').read_text().splitlines() if x.startswith('COMPOSITION_RESULT_GZIP_B64=')]
    if logged!=[report] or json.loads((target/'raw/report.json').read_text())!=report or json.loads((target/'raw/driver.json').read_text())!=report['driver']:raise ValueError('Report log/raw joins differ')
    sys.path.insert(0,str(target/'payload'));spec=importlib.util.spec_from_file_location('commit_saved_analysis',target/'payload/analysis.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.analyze(target/'raw',report['driver'],manifest)!=report['analysis']:raise ValueError('Saved-only analysis differs')
    wrapper=report['saturn_receipt'];r=dict(wrapper['receipt']);fingerprint=r.pop('fingerprint')
    if hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=fingerprint or r['parent']!=wrapper['before'] or r['result']!=wrapper['after'] or wrapper['request']!=json.loads((target/'raw/request.json').read_text()):raise ValueError('Saturn invocation receipt differs')
    out={'schema':'cmachine-commit-verification-v1','source_commit':manifest['git_commit'],'source_tree':manifest['git_tree'],'native_reference_commit':manifest['native_reference_commit'],'report_sha256':sha(target/'report.json'),'source_manifest_sha256':sha(target/'payload/source-manifest.json'),'source_files_verified':len(manifest['sources']),'raw_files_verified':len(report['raw_files']),'raw_bytes_verified':sum(x['bytes'] for x in report['raw_files'].values()),'every_source_matches_git_object':True,'guarded_payload_declared_sealed_executed_exact':True,'payload_custody':custody,'scheduler_log_sha256':sha(target/'logs.txt'),'saturn_receipt_fingerprint':fingerprint,'analysis_recomputed_without_model':True,'models_evaluated_by_verifier':0,'scope':'bounded integration engineering parity; historical bounded-memory quality failures remain'}
    (target/'verification.json').write_text(json.dumps(out,indent=2)+'\n');(target/'observations.jsonl').write_text(''.join(json.dumps({**x,'model_id':report['model']['identity'],'source_commit':manifest['git_commit'],'report_sha256':out['report_sha256']},sort_keys=True)+'\n' for x in report['analysis']['observations']));print(json.dumps(out))
    return out
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('stage','submit','collect','verify'));parser.add_argument('target');a=parser.parse_args()
    if a.action=='stage':stage(a.target)
    elif a.action=='submit':submit(a.target)
    elif a.action=='collect':collect(a.target)
    else:verify(a.target)
