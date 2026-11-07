"""One resident C model, one source-state refresh and sealed own-history futures."""
from __future__ import annotations
import base64,gzip,hashlib,json,os,platform,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'results';D=1536;H=8960;L=28;R=256;V=151936;ALL=(1<<28)-1
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(name,value):
    p=RAW/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');return value

def record(p):return {'sha256':sha(p),'bytes':Path(p).stat().st_size}

class Server:
    def __init__(self,binary,model):
        self.model=model
        self.err=(RAW/'driver.stderr.txt').open('w');self.log=(RAW/'commands.jsonl').open('w')
        self.proc=subprocess.Popen([str(binary),model['path'],str(RAW)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
        self.wait('READY')
    def wait(self,prefix):
        while True:
            line=self.proc.stdout.readline()
            if not line:raise RuntimeError('C worker ended: '+str(self.proc.poll()))
            self.log.write(json.dumps({'received':line.strip(),'time_ns':time.time_ns()})+'\n');self.log.flush()
            if line.startswith(prefix):return line.strip()
            print('C',line.strip(),flush=True)
    def send(self,text,done):
        self.log.write(json.dumps({'sent':text,'time_ns':time.time_ns()})+'\n');self.log.flush()
        self.proc.stdin.write(text+'\n');self.proc.stdin.flush();return self.wait(done)
    def run(self,label,program,mask,capture,cases):
        text=f'RUN {label} {program} {mask} {int(capture)} {len(cases)}\n'+'\n'.join(f"{c['id']} {c['budget']}" for c in cases)
        print('RUN',label,flush=True);self.send(text,'DONE '+label)
        value=json.loads((RAW/(label+'.json')).read_text())
        if value.get('model_identity')!=self.model['identity'] or value.get('model_digest')!=self.model['content_sha256']:
            raise ValueError('Executed C model identity differs')
        # Public C ABI uses explicit file references. Preserve them while adding
        # checked array/key views for the Python compiler and metric consumer.
        import numpy as np
        for path in value['paths']:
            path['consumed_ids']=np.fromfile(RAW/path['consumed_ids_file'],dtype='<i4').tolist()
            path['candidate_logits_file']=path['candidate_decision_logits_file']
            path['shadow_logits_file']=path['native_shadow_logits_file']
            path['receiver_file']=path.get('x_file','')
            primary=RAW/f"{path['case_id']}-{label}-primary.json"
            if primary.exists():
                closed=json.loads(primary.read_text())
                for turn,sealed in zip(path['turns'],closed['turns'],strict=True):
                    if 'decision_positions' in sealed:turn['decision_positions']=sealed['decision_positions']
                    if 'boundaries' in sealed and 'boundaries' not in turn:turn['boundaries']=sealed['boundaries']
        return value
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('QUIT\n');self.proc.stdin.flush();self.proc.wait(timeout=30)
        self.log.close();self.err.close()
        if self.proc.returncode!=0 or not (RAW/'execution-totals.json').is_file():
            raise RuntimeError('C did not finish with a saved physical-work ledger: '+str(self.proc.returncode))

def tokenize(cases,tokenizer,split):
    result=[]
    for case in cases:
        messages=[{'role':'user','content':case['prompt']}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':''},{'role':'user','content':case['followup']}],tokenize=False,add_generation_prompt=True)
        if not full.startswith(text):raise ValueError('Template prefix')
        prompt=tokenizer.encode(text,add_special_tokens=False);follow=tokenizer.encode(full[len(text):],add_special_tokens=False)
        if not follow or follow[0]!=151645 or tokenizer.encode(full,add_special_tokens=False)!=prompt+follow:raise ValueError('Template token closure')
        if len(prompt)+len(follow)+2*case['budget']>=512:raise ValueError('Declared capacity')
        for suffix,ids in [('prompt',prompt),('followup',follow)]:
            (RAW/(case['id']+'-'+suffix+'.txt')).write_text(' '.join(map(str,ids))+'\n')
        result.append({**case,'prompt_ids':prompt,'followup_ids':follow,'split':split})
    write(split+'-inputs.json',result);return result

def capture_samples(run,label):
    import numpy as np
    parts=[];lineage=[]
    for path in run['paths']:
        name=path.get('receiver_file',f"{label}-{path['case_id']}-x.f32")
        values=np.fromfile(RAW/name,dtype='<f4').reshape(-1,L,D)
        if len(values)!=len(path['consumed_ids']) or len(values)<64 or not np.isfinite(values).all():raise ValueError('Capture shape/finite/minimum clock count')
        indices=np.linspace(0,len(values)-1,64,dtype=np.int64)
        if len(set(indices.tolist()))!=64:raise ValueError('Distinct sampling')
        parts.append(values[indices].copy());lineage.append({'case_id':path['case_id'],'source_file':name,'source':record(RAW/name),'rows':len(values),'indices':indices.tolist(),'consumed_ids_sha256':hashlib.sha256(np.asarray(path['consumed_ids'],dtype='<i4').tobytes()).hexdigest()})
    samples=np.concatenate(parts,axis=0);samples.tofile(RAW/(label+'-selected.f32'))
    write(label+'-sampling.json',{'shape':list(samples.shape),'samples':record(RAW/(label+'-selected.f32')),'lineage':lineage})
    return samples

def compile_basis(native,own,name):
    import numpy as np,torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    target=RAW/name;target.mkdir();rows=[];start=time.monotonic()
    # Same uncentered randomized-SVD budget and seed per layer in both arms.
    # An extra sixteen directions are computed as a fixed oversampling budget.
    for layer in range(L):
        torch.manual_seed(20261106+layer);torch.cuda.manual_seed_all(20261106+layer)
        a=torch.from_numpy(np.concatenate([native[:,layer],own[:,layer]],axis=0)).to('cuda')/(2.0**0.5)
        _,s,v=torch.svd_lowrank(a,q=R+16,niter=2)
        e=v[:,:R].T.contiguous()
        pivot=e.abs().argmax(dim=1);sign=e[torch.arange(R,device='cuda'),pivot].sign();e=e*torch.where(sign==0,1,sign)[:,None]
        orth=float((e@e.T-torch.eye(R,device='cuda')).abs().max());residual=float((a-(a@e.T)@e).square().sum()/a.square().sum())
        values=e.cpu().numpy().astype('<f4');path=target/f'E-{layer}.f32';values.tofile(path)
        rows.append({'layer':layer,'rank':R,'samples':len(a),'E':record(path),'orthogonality_max_abs':orth,'relative_receiver_residual_energy':residual,'singular_values':s.cpu().tolist()})
        del a,s,v,e
        print('BASIS',name,layer,'residual-energy',residual,flush=True)
    torch.cuda.synchronize();torch.cuda.empty_cache()
    return write(name+'/basis.json',{'schema':'transition-laws-basis-v1','method':'uncentered randomized SVD q272 niter2; rank256; fixed per-layer seed and max-pivot sign; equal source-conversation samples','native_sampling_sha256':sha(RAW/'source-native-sampling.json'),'own_sampling_sha256':sha(RAW/'source-compact-sampling.json') if name=='p1' else sha(RAW/'source-native-sampling.json'),'layers':rows,'seconds':time.monotonic()-start})

def metrics(run,label,tokenizer):
    import numpy as np
    rows=[];decisions=[]
    for path in run['paths']:
        candidate=np.memmap(RAW/path['candidate_logits_file'],dtype='<f4',mode='r').reshape(-1,V)
        shadow=np.memmap(RAW/path['shadow_logits_file'],dtype='<f4',mode='r').reshape(-1,V)
        if candidate.shape!=shadow.shape:raise ValueError('Shadow decision shape')
        offset=0
        for turn,t in enumerate(path['turns']):
            ids=t['answer_ids'];kl=[];l2=[];tops=[]
            expected=list(range(t['prefill']-1,t['prefill']+len(ids)-1))
            if t.get('decision_positions',expected)!=expected:raise ValueError('Decision clocks')
            for j,token in enumerate(ids):
                c=np.asarray(candidate[offset+j],dtype=np.float64);n=np.asarray(shadow[offset+j],dtype=np.float64)
                if not np.isfinite(c).all() or not np.isfinite(n).all() or int(c.argmax())!=token:raise ValueError('Finite distribution/token authority')
                cn=c-c.max();nn=n-n.max();clog=cn-np.log(np.exp(cn).sum());nlog=nn-np.log(np.exp(nn).sum());p=np.exp(nlog)
                loss=float(np.dot(p,nlog-clog));distance=float(np.linalg.norm(c-n)/max(np.linalg.norm(n),1e-30))
                kl.append(loss);l2.append(distance);tops.append(int(n.argmax())==token)
                decisions.append({'label':label,'case_id':path['case_id'],'turn':turn,'position':expected[j],'decision_index':offset+j,'emitted':token,'native_shadow_top':int(n.argmax()),'KL':loss,'logit_relative_L2':distance})
            rows.append({'case_id':path['case_id'],'turn':turn,'tokens':len(ids),'answer':tokenizer.decode(ids,skip_special_tokens=True),'answer_ids':ids,'stopped':t['stopped'],'truncated':not t['stopped'],'mean_KL':float(np.mean(kl)),'mean_logit_relative_L2':float(np.mean(l2)),'conditional_top_matches':sum(tops),'decision_positions':expected})
            offset+=len(ids)
        if offset!=len(candidate):raise ValueError('Extra/missing decision rows')
    result={'schema':'transition-laws-metrics-v1','label':label,'equal_case_turn_KL':float(np.mean([r['mean_KL'] for r in rows])),'equal_case_turn_logit_relative_L2':float(np.mean([r['mean_logit_relative_L2'] for r in rows])),'turns':rows,'decisions':decisions}
    write(label+'-metrics.json',result);return result

def experiment():
    from transformers import AutoTokenizer
    from mrun.mlops.publish import make_s3_client
    manifest=json.loads((ROOT/'source-manifest.json').read_text());model=json.loads((ROOT/'model.json').read_text())
    for name,wanted in manifest['sources'].items():
        if sha(ROOT/name)!=wanted:raise ValueError('Payload identity '+name)
    bucket=os.environ.get('MRUN_EXPERIMENTS_BUCKET') or os.environ.get('SATURN_CHECKPOINT_BUCKET') or 'experiments'
    make_s3_client().head_bucket(Bucket=bucket)
    write('custody-preflight.json',{'bucket':bucket,'readable':True,'artifact_limit_bytes':5*1024**3})
    source=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name,wanted in model['tokenizer_files'].items():
        if sha(source/name)!=wanted:raise ValueError('Tokenizer identity '+name)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    process=subprocess.run(['make'],cwd=ROOT,text=True,capture_output=True);(RAW/'build.txt').write_text(process.stdout+process.stderr)
    if process.returncode:raise RuntimeError('C build: '+process.stderr[-6000:])
    binary=ROOT/'build/transition-laws';shutil.copy2(binary,RAW/'transition-laws.bin')
    report={'schema':'transition-laws-report-v1','model':model,'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'backend':'C FP32 pedantic cuBLAS; Torch randomized-SVD compiler only','compiler':{},'runs':{},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested','environment':{'python':sys.version,'platform':platform.platform(),'job_id':os.environ.get('MRUN_JOB_ID')}}
    def partial():write('partial-report.json',report)
    server=Server(binary,model)
    try:
        source_cases=tokenize(json.loads((ROOT/'source.json').read_text()),tokenizer,'source')
        def run(label,program,mask,capture,cases):
            data=server.run(label,program,mask,capture,cases);summary=metrics(data,label,tokenizer)
            report['runs'][label]={'driver':data,'metrics':summary};partial();return data
        native=run('source-native','none',0,True,source_cases);xnative=capture_samples(native,'source-native')
        report['compiler']['p0']=compile_basis(xnative,xnative,'p0');server.send('COMPILE p0','DONE COMPILE p0')
        compact=run('source-compact','p0',ALL,True,source_cases);xown=capture_samples(compact,'source-compact')
        report['compiler']['p1']=compile_basis(xnative,xown,'p1');server.send('COMPILE p1','DONE COMPILE p1')
        monitor=tokenize(json.loads((ROOT/'monitor.json').read_text()),tokenizer,'monitor')
        run('monitor-p0','p0',ALL,False,monitor);run('monitor-p1','p1',ALL,False,monitor)
        objectives={key:report['runs']['monitor-'+key]['metrics']['equal_case_turn_KL'] for key in ['p0','p1']}
        selected=min(objectives,key=lambda key:(objectives[key],key))
        seal={'schema':'transition-laws-selection-v1','selected':selected,'monitor_objectives':objectives,'policy':'minimum equal-conversation/equal-answer-turn native-shadow conditional KL; retain both candidates and all trends','heldout_opened':False,'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'source_inputs_sha256':sha(RAW/'source-inputs.json'),'monitor_inputs_sha256':sha(RAW/'monitor-inputs.json'),'factor_files':{str(p.relative_to(RAW)):record(p) for key in ['p0','p1'] for p in sorted((RAW/key).rglob('*')) if p.is_file()},'time_ns':time.time_ns()}
        write('selection-seal.json',seal);report['selection']=seal;partial();print('SEALED',selected,objectives,flush=True)
        heldout=tokenize(json.loads((ROOT/'heldout.json').read_text()),tokenizer,'heldout')
        for label,program,mask in [('heldout-native','none',0),('heldout-p0-layer16','p0',1<<16),('heldout-p0','p0',ALL),('heldout-p1','p1',ALL)]:run(label,program,mask,False,heldout)
        for name,wanted in seal['factor_files'].items():
            if record(RAW/name)!=wanted:raise ValueError('Evaluation mutated factors')
        report.update(mechanics_status='complete',instrument_status='complete')
    finally:server.close()
    report['execution_totals']=json.loads((RAW/'execution-totals.json').read_text())
    return report

def emit(report):
    report['raw_files']={str(p.relative_to(RAW)):record(p) for p in sorted(RAW.rglob('*')) if p.is_file() and p.name not in ['report.json','partial-report.json']}
    write('report.json',report)
    print('TRANSITION_LAWS_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(json.dumps(report,sort_keys=True,allow_nan=False).encode(),mtime=0)).decode(),flush=True)
    from mrun.mlops.publish import make_s3_client
    archive=ROOT/'evidence.tar'
    with tarfile.open(archive,'w:') as tar:
        for folder,prefix in [(RAW,'raw'),(ROOT,'payload')]:
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts:continue
                if folder==ROOT and (p.is_relative_to(RAW) or p==archive or p.is_relative_to(ROOT/'build')):continue
                tar.add(p,arcname=prefix+'/'+str(p.relative_to(folder)),recursive=False)
    if archive.stat().st_size>5*1024**3:raise ValueError('Custody archive exceeds 5GiB limit')
    client=make_s3_client();bucket=os.environ.get('MRUN_EXPERIMENTS_BUCKET') or os.environ.get('SATURN_CHECKPOINT_BUCKET') or 'experiments'
    key='saturn/transition-laws/'+os.environ.get('MRUN_JOB_ID','unknown')+'/evidence.tar'
    client.upload_file(str(archive),bucket,key)
    asset={'bucket':bucket,'key':key,'endpoint_url':client.meta.endpoint_url,**record(archive)}
    print('TRANSITION_LAWS_CUSTODY='+json.dumps(asset,sort_keys=True),flush=True)

if __name__=='__main__':
    RAW.mkdir(exist_ok=True);started=time.monotonic();report={};code=0
    try:
        sys.path.insert(0,str(ROOT/'public'));from saturn_bridge import invoke
        request={'schema':'transition-laws-request-v1','source_manifest_sha256':sha(ROOT/'source-manifest.json'),'heldout_selection_forbidden':True,'max_nested_submissions':0,'scope':'One fixed all28 rank256 source-state refresh, source/monitor/heldout; own-token primary before independent same-ID native shadow'}
        report,receipt=invoke([sys.executable,str(ROOT/'worker.py')],None,request,controller=lambda _:experiment());report['saturn_receipt']=receipt
    except Exception:
        report=json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {}
        report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete',trend_status='evidence-incomplete',terminal_status='not-requested');code=1
    report['worker_seconds']=time.monotonic()-started
    try:emit(report)
    except Exception:print('CUSTODY_ERROR='+traceback.format_exc(),flush=True);code=1
    raise SystemExit(code)
