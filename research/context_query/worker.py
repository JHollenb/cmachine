"""One resident C model; source futures select an image before fresh tokenization."""
import base64,gzip,hashlib,io,json,os,selectors,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'results';LIMIT=384*1024**2
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def audit(name):
    metrics=read(RAW/f'{name}-metrics.json');trace=read(RAW/f'{name}.json');native=np.memmap(RAW/'native-logits.tmp',mode='r',dtype='<f4').reshape(-1,151936);candidate=np.memmap(RAW/'candidate-logits.tmp',mode='r',dtype='<f4').reshape(native.shape);errors=[]
    for i,row in enumerate(metrics['samples']):
        n=native[i].astype(float);c=candidate[i].astype(float);n-=n.max();c-=c.max();ln=n-np.log(np.exp(n).sum());lc=c-np.log(np.exp(c).sum());kl=float(np.exp(ln)@(ln-lc));error=abs(kl-row['kl']);errors.append(error)
        if error>1e-10*max(1.,abs(kl)) or sha_bytes(native[i])!=row['native_sha256'] or sha_bytes(candidate[i])!=row['candidate_sha256']:raise ValueError('C/Python distribution or stream identity differs')
        if row['future'] and (row['candidate_top']!=trace['consumed_ids'][row['clock']+1] or trace['prediction_turn'][row['clock']]!=row['turn']):raise ValueError('Own-token decision accounting differs')
    del native,candidate
    for f in ('native-logits.tmp','candidate-logits.tmp'):(RAW/f).unlink()
    per=[float(np.mean([r['kl'] for r in metrics['samples'] if r['future'] and r['turn']==t])) for t in (0,1)]
    result={'name':name,'sample_rows':len(metrics['samples']),'max_distribution_discrepancy':max(errors),'per_turn_future_kl':per,'future_mean_kl':sum(per)/2,'sampled_prefill_kl':float(np.mean([r['kl'] for r in metrics['samples'] if not r['future']])),'native_top_agreement_future':float(np.mean([r['native_top']==r['candidate_top'] for r in metrics['samples'] if r['future']]))}
    write(RAW/f'{name}-audit.json',result);return result
def sha_bytes(a):return hashlib.sha256(a.tobytes()).hexdigest()

def experiment():
    sys.path.insert(0,str(ROOT/'public'))
    from saturn_bridge import invoke
    from fit import compile_source
    from transformers import AutoTokenizer
    manifest=read(ROOT/'source-manifest.json')
    for p,h in manifest['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Source hash differs: '+p)
    model=read(ROOT/'baseline.json')['model'];source=read(ROOT/'source-inputs.json')
    for p in ROOT.glob('*.f32'):shutil.copy2(p,RAW/p.name)
    fit=compile_source(ROOT,RAW)
    for c,row in enumerate(source['sources']):
        for stem,ids in ((f'source{c}',row['prompt_ids']),(f'source{c}-follow',row['followup_ids'])):(RAW/f'{stem}.txt').write_text(' '.join(map(str,ids))+'\n')
    for p in ROOT.glob('parity*'):shutil.copy2(p,RAW/p.name)
    build=subprocess.run(['make'],cwd=ROOT,text=True,capture_output=True,timeout=120);(RAW/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:raise RuntimeError('Build failed: '+build.stderr[-8000:])
    report={'schema':'context-query-experiment-v1','model':model,'fit':fit,'binary_sha256':sha(ROOT/'build/context-query'),'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested','job_id':os.environ.get('MRUN_JOB_ID')};write(RAW/'partial-report.json',report)
    args=[str(ROOT/'build/context-query'),model['path'],str(RAW)]
    def controller(_):
        with (RAW/'driver.stderr.txt').open('w') as err:
            process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1);poll=selectors.DefaultSelector();poll.register(process.stdout,selectors.EVENT_READ);lines=[];deadline=time.monotonic()+800
            def expect(expected):
                while time.monotonic()<deadline:
                    if not poll.select(1):
                        if process.poll() is not None:break
                        continue
                    line=process.stdout.readline();lines.append(line)
                    if line.strip()!=expected:raise RuntimeError('Driver protocol '+repr(line)+': '+(RAW/'driver.stderr.txt').read_text()[-5000:])
                    return
                raise RuntimeError('Finite driver deadline/process end: '+(RAW/'driver.stderr.txt').read_text()[-5000:])
            def command(text,expected):process.stdin.write(text+'\n');process.stdin.flush();expect(expected)
            trials=[]
            def trial(name,policy,prompt,follow,is_source):
                command(f'RUN {name} {policy} {prompt} {follow} {int(is_source)}','DONE '+name);a=audit(name);trials.append({'name':name,'policy':policy,'source':is_source,**a});print(json.dumps({'completed':name,'future_kl':a['future_mean_kl']}),flush=True);return a
            try:
                expect('READY');report['integration']=read(RAW/'integration.json')
                if report['integration']['model_digest']!=model['content_sha256']:raise ValueError('Integrated model identity differs')
                feedback=[]
                for i in range(4):
                    rows=[trial(f's{c}-p{i}',i,f'source{c}.txt',f'source{c}-follow.txt',True) for c in range(3)];feedback.append({'index':i,'dose':fit['proposals'][i]['dose'],'future_kl':float(np.mean([r['future_mean_kl'] for r in rows])),'per_source':rows})
                chosen=min(range(4),key=lambda i:(feedback[i]['future_kl'],abs(feedback[i]['dose']),i));command('SELECT '+str(chosen),'SELECTED');report['feedback']=feedback;report['selected_index']=chosen
                selection={'phase':'source-only suffix selection before fresh input parsing','selected_index':chosen,'feedback':feedback,'fit_sha256':sha(RAW/'fit.json'),'selected_image_sha256':sha(RAW/f'query-proposal-{chosen}.f32')};write(RAW/'selection-seal.json',selection)
                x=np.concatenate([np.fromfile(RAW/f's{c}-p{chosen}-norm18.f32',dtype='<f4').reshape(-1,1536) for c in range(3)]).astype(float);_,singular,basis=np.linalg.svd(x,full_matrices=False);b=np.asarray(basis[:128],dtype='<f4');b.tofile(RAW/'basis18.f32');write(RAW/'basis18.json',{'source_rows':len(x),'source_files':{f's{c}-p{chosen}-norm18.f32':sha(RAW/f's{c}-p{chosen}-norm18.f32') for c in range(3)},'rank':128,'captured_source_energy':float(np.sum((x@b.astype(float).T)**2)/np.sum(x*x)),'singular_values':singular.tolist(),'basis_sha256':sha(RAW/'basis18.f32')});command('COMPILE18','COMPILED18')
                seal={'phase':'all source choices and later-layer program sealed before parsing fresh.json','selection_sha256':sha(RAW/'selection-seal.json'),'basis18_sha256':sha(RAW/'basis18.f32'),'query18_sha256':sha(RAW/'image18-q.f32'),'key18_sha256':sha(RAW/'image18-k.f32'),'fresh_spec_sha256':sha(ROOT/'fresh.json')};write(RAW/'fresh-seal.json',seal)
                # Fresh text is deliberately first parsed here, after all source choices.
                fresh=read(ROOT/'fresh.json');tokroot=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
                for p,h in model['tokenizer_files'].items():
                    if sha(tokroot/p)!=h:raise ValueError('Tokenizer identity differs')
                tokenizer=AutoTokenizer.from_pretrained(tokroot,local_files_only=True);inputs=[]
                for context in fresh['context_lengths']:
                    for c,case in enumerate(fresh['cases']):
                        stem=f'c{c}-n{context}'
                        # Preserve complete chat-template token boundaries and task suffix.
                        before=tokenizer.apply_chat_template([{'role':'user','content':'MARKER'}],tokenize=False,add_generation_prompt=True);left,right=before.split('MARKER');a=tokenizer.encode(left,add_special_tokens=False);z=tokenizer.encode(case['text']+right,add_special_tokens=False);padding=tokenizer.encode(fresh['padding']*100,add_special_tokens=False);need=context-len(a)-len(z)
                        if need<0:raise ValueError('Fresh context shorter than complete task')
                        ids=a+padding[:need]+z;extended=tokenizer.apply_chat_template([{'role':'user','content':'MARKER'},{'role':'assistant','content':''},{'role':'user','content':case['followup']}],tokenize=False,add_generation_prompt=True)
                        base=before;follow=tokenizer.encode(extended[len(base):],add_special_tokens=False)
                        if not extended.startswith(base) or len(ids)!=context or not follow or follow[0]!=151645:raise ValueError('Fresh chat boundary differs')
                        for label,seq in ((stem,ids),(stem+'-follow',follow)):(RAW/f'{label}.txt').write_text(' '.join(map(str,seq))+'\n')
                        inputs.append({'stem':stem,'kind':case['kind'],'context_tokens':context,'prompt_ids':ids,'followup_ids':follow,'decoded_prompt':tokenizer.decode(ids),'followup':case['followup']})
                        policies=[-1,0,chosen,4,5];seen={}
                        for logical,policy in enumerate(policies):
                            name=stem+f'-arm{logical}'
                            if policy in seen:trials.append({'name':name,'alias_of':seen[policy],'policy':policy,'source':False});continue
                            seen[policy]=name;trial(name,policy,stem+'.txt',stem+'-follow.txt',False)
                write(RAW/'input-manifest.json',inputs);texts={}
                for row in trials:
                    if 'alias_of' not in row:texts[row['name']]=[tokenizer.decode(a,skip_special_tokens=True) for a in read(RAW/(row['name']+'.json'))['answers']]
                process.stdin.write('QUIT\n');process.stdin.flush();code=process.wait(timeout=10)
                if code:raise RuntimeError('C process failed')
                report.update(trials=trials,texts=texts,execution=read(RAW/'execution.json'),compile18=read(RAW/'compile18.json'),basis18=read(RAW/'basis18.json'),mechanics_status='complete',instrument_status='complete');write(RAW/'partial-report.json',report)
                return {'execution':report['execution'],'selection_seal_sha256':sha(RAW/'selection-seal.json'),'fresh_seal_sha256':sha(RAW/'fresh-seal.json'),'trials':len(trials)}
            finally:
                poll.close()
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                process.stdin.close();process.stdout.close();(RAW/'driver.stdout.txt').write_text(''.join(lines))
    _,report['saturn_receipt']=invoke(args,None,{'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'source_manifest_sha256':report['source_manifest_sha256'],'scope':'One resident model, public operand programs, source-only response proposal and native suffix selection, fresh own-state two-layer continuations'},controller=controller)
    return report
def emit():
    files=[(p,'raw/'+str(p.relative_to(RAW))) for p in sorted(RAW.rglob('*')) if p.is_file() and p.suffix!='.tmp' and p.name!='partial-report.json']
    files += [(p,'payload/'+str(p.relative_to(ROOT))) for p in sorted(ROOT.rglob('*')) if p.is_file() and not set(p.relative_to(ROOT).parts)&{'results','build','__pycache__'}]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('Archive cap exceeded')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for p,name in files:archive.add(p,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)
if __name__=='__main__':
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:
        report=read(RAW/'partial-report.json') if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-started;report['raw_files']={str(p.relative_to(RAW)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.rglob('*')) if p.is_file() and p.name not in ('partial-report.json','report.json') and p.suffix!='.tmp'};write(RAW/'report.json',report);data=json.dumps(report,sort_keys=True,allow_nan=False).encode();print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit();raise SystemExit(code)
