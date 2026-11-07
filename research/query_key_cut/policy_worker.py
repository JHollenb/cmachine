"""Frozen operand masks on known own-token conversations, not a new fit."""
import base64,gzip,hashlib,io,json,os,selectors,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'results';LIMIT=128*1024**2
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def experiment():
    sys.path.insert(0,str(ROOT/'public'))
    from saturn_bridge import invoke
    from engine_worker import audit
    from transformers import AutoTokenizer
    manifest=read(ROOT/'source-manifest.json')
    for p,h in manifest['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Payload differs '+p)
    baseline=read(ROOT/'baseline.json');model=baseline['model'];inputs=read(ROOT/'input-manifest.json');write(RAW/'policy-seal.json',{'phase':'before new model futures','policies':['compiled-Q18/native-KV18','compiled-K18/native-QV18'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'scope':'Known development inputs; no fit or selection'})
    for name,width in [('basis17.f32',1536),('query17.f32',1536),('key17.f32',256),('basis18.f32',1536),('query18.f32',1536),('key18.f32',256)]:
        shutil.copy2(ROOT/name,RAW/name)
        if (RAW/name).stat().st_size!=128*width*4 or sha(RAW/name)!=manifest['sources'][name]:raise ValueError('Executor operand preflight differs '+name)
    for case in inputs:
        stem=case['stem']
        for name,ids in ((stem,case['prompt_ids']),(stem+'-follow',case['followup_ids'])):(RAW/(name+'.txt')).write_text(' '.join(map(str,ids))+'\n')
    build=subprocess.run(['make'],cwd=ROOT,capture_output=True,text=True,timeout=120);(RAW/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:raise RuntimeError('Build failed '+build.stderr[-5000:])
    report={'schema':'qk18-on-policy-v1','model':model,'baseline_report_sha256':baseline['previous_report_sha256'],'inputs':inputs,'binary_sha256':sha(ROOT/'build/query-key-policy'),'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'development-observations','terminal_status':'not-requested'};write(RAW/'partial-report.json',report)
    args=[str(ROOT/'build/query-key-policy'),model['path'],str(RAW)]
    def controller(_):
        trials=[];lines=[]
        with (RAW/'driver.stderr.txt').open('w') as err:
            process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1);poll=selectors.DefaultSelector();poll.register(process.stdout,selectors.EVENT_READ);deadline=time.monotonic()+400
            try:
                for policy in range(2):
                    for context in (128,1024):
                        for task in range(2):
                            name=f'p{policy}-c{task}-n{context}'
                            while not poll.select(1):
                                if time.monotonic()>deadline or process.poll() is not None:raise RuntimeError('Driver timeout/end '+(RAW/'driver.stderr.txt').read_text()[-5000:])
                            line=process.stdout.readline();lines.append(line)
                            if line.strip()!='DONE '+name:raise RuntimeError('Protocol '+repr(line)+': '+(RAW/'driver.stderr.txt').read_text()[-5000:])
                            a=audit(name);trials.append({'name':name,'policy':policy,'task':task,'context_tokens':context,**a});print(json.dumps({'completed':name,'future_kl':a['future_mean_kl']}),flush=True);process.stdin.write('\n');process.stdin.flush()
                code=process.wait(timeout=10)
                if code:raise RuntimeError('Driver failure')
            finally:
                poll.close()
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                process.stdin.close();process.stdout.close();(RAW/'driver.stdout.txt').write_text(''.join(lines))
        return {'trials':trials,'execution':read(RAW/'execution.json')}
    result,report['saturn_receipt']=invoke(args,None,{'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'policy_seal_sha256':sha(RAW/'policy-seal.json'),'scope':'One finite C invocation, frozen query-only/key-only programs from empty state, known own-token conversations with native shadows'},controller=controller);report.update(result)
    if result['execution']['model_digest']!=model['content_sha256']:raise ValueError('Model identity differs')
    tokroot=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for p,h in model['tokenizer_files'].items():
        if sha(tokroot/p)!=h:raise ValueError('Tokenizer differs')
    tok=AutoTokenizer.from_pretrained(tokroot,local_files_only=True);report['texts']={x['name']:[tok.decode(ids,skip_special_tokens=True) for ids in read(RAW/(x['name']+'.json'))['answers']] for x in report['trials']};report.update(mechanics_status='complete',instrument_status='complete');return report
def emit():
    files=[(p,'raw/'+str(p.relative_to(RAW))) for p in sorted(RAW.rglob('*')) if p.is_file() and p.name!='partial-report.json' and p.suffix!='.tmp']+[(p,'payload/'+str(p.relative_to(ROOT))) for p in sorted(ROOT.rglob('*')) if p.is_file() and not set(p.relative_to(ROOT).parts)&{'results','build','__pycache__'}]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('Archive cap')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for p,name in files:archive.add(p,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)
if __name__=='__main__':
    RAW.mkdir(exist_ok=True);start=time.monotonic()
    try:report=experiment();code=0
    except Exception:report=read(RAW/'partial-report.json') if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-start;report['raw_files']={str(p.relative_to(RAW)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.rglob('*')) if p.is_file() and p.name not in ('report.json','partial-report.json') and p.suffix!='.tmp'};write(RAW/'report.json',report);data=json.dumps(report,sort_keys=True,allow_nan=False).encode();print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit();raise SystemExit(code)
