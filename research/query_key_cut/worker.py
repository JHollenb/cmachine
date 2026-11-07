"""One model-bound native-consumer cut, with no training or target selection."""
import base64,gzip,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'results';LIMIT=192*1024**2
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def experiment():
    sys.path.insert(0,str(ROOT/'public'))
    from saturn_bridge import invoke
    from transformers import AutoTokenizer
    from analyze import analyze
    m=read(ROOT/'source-manifest.json')
    for p,h in m['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Payload hash differs '+p)
    baseline=read(ROOT/'baseline.json');model=baseline['model'];tokroot=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for p,h in model['tokenizer_files'].items():
        if sha(tokroot/p)!=h:raise ValueError('Tokenizer identity differs')
    tokenizer=AutoTokenizer.from_pretrained(tokroot,local_files_only=True);inputs=[]
    for p in ROOT.glob('*.f32'):shutil.copy2(p,RAW/p.name)
    for case in baseline['cases']:
        context=case['context_tokens'];previous=read(ROOT/f'case{context}-previous.json');answer=previous['answers'][0];text=tokenizer.decode(answer,skip_special_tokens=True);location=text.rfind('amber')
        if location<0 or 'whistle' not in text[:location]:raise ValueError('Declared wrong binding is absent')
        index=next(i for i in range(len(answer)) if len(tokenizer.decode(answer[:i+1],skip_special_tokens=True))>location);prefix=case['prompt_ids']+answer[:index];parent,cut=prefix[:-1],prefix[-1]
        if answer[index]!=tokenizer.encode(' amber',add_special_tokens=False)[0]:raise ValueError('Binding token alignment differs')
        for suffix,ids in [('parent',parent),('token',[cut]),('follow',case['followup_ids']),('original',previous['consumed_ids'])]:(RAW/f'case{context}-{suffix}.txt').write_text(' '.join(map(str,ids))+'\n')
        inputs.append({**case,'source_trace_sha256':sha(ROOT/f'case{context}-previous.json'),'bad_answer_ids':answer,'bad_answer_text':text,'binding_decision_index':index,'binding_token_id':answer[index],'answer_prefix_ids':answer[:index],'parent_ids':parent,'cut_token':cut,'cut_position':len(parent),'parent_prefix_sha256':hashlib.sha256(__import__('struct').pack('<'+'i'*len(parent),*parent)).hexdigest(),'labels':{'correct':{'text':' silver','ids':tokenizer.encode(' silver',add_special_tokens=False)},'incorrect':{'text':' amber','ids':tokenizer.encode(' amber',add_special_tokens=False)}}})
    write(RAW/'inputs.json',inputs);build=subprocess.run(['make'],cwd=ROOT,capture_output=True,text=True,timeout=120);(RAW/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:raise RuntimeError('Build failed '+build.stderr[-8000:])
    report={'schema':'qk18-cut-experiment-v1','model':model,'baseline_report_sha256':baseline['previous_report_sha256'],'binary_sha256':sha(ROOT/'build/query-key-cut'),'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'inputs':inputs,'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested','job_id':os.environ.get('MRUN_JOB_ID')};write(RAW/'partial-report.json',report)
    args=[str(ROOT/'build/query-key-cut'),model['path'],str(RAW)]
    def controller(_):
        with (RAW/'driver.stderr.txt').open('w') as err,(RAW/'driver.stdout.txt').open('w') as out:process=subprocess.run(args,stdout=out,stderr=err,text=True,timeout=375)
        if process.returncode:raise RuntimeError('C executor failed: '+(RAW/'driver.stderr.txt').read_text()[-6000:])
        return read(RAW/'driver.json')
    report['driver'],report['saturn_receipt']=invoke(args,None,{'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'source_manifest_sha256':report['source_manifest_sha256'],'input_sha256':sha(RAW/'inputs.json'),'scope':'Whole finite C invocation; inner model/session-bound single-attention cuts and complete owned cache/parent recovery are recorded and checked by C'},controller=controller)
    if report['driver']['model_digest']!=model['content_sha256'] or report['driver']['model_identity']!=model['identity']:raise ValueError('Model identity differs')
    texts={}
    for row in report['driver']['rows']:
        if 'alias_of' in row:continue
        case=next(x for x in inputs if x['context_tokens']==row['context']);texts[row['name']]=[tokenizer.decode(case['answer_prefix_ids']+row['answers'][0],skip_special_tokens=True),tokenizer.decode(row['answers'][1],skip_special_tokens=True)]
    report['texts']=texts;report['analysis']=analyze(RAW,report);report.update(mechanics_status='complete',instrument_status='complete');return report
def emit():
    files=[(p,'raw/'+str(p.relative_to(RAW))) for p in sorted(RAW.rglob('*')) if p.is_file() and p.name!='partial-report.json']+[(p,'payload/'+str(p.relative_to(ROOT))) for p in sorted(ROOT.rglob('*')) if p.is_file() and not set(p.relative_to(ROOT).parts)&{'results','build','__pycache__'}]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('Archive cap exceeded')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for p,name in files:archive.add(p,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)
if __name__=='__main__':
    RAW.mkdir(exist_ok=True);start=time.monotonic()
    try:report=experiment();code=0
    except Exception:report=read(RAW/'partial-report.json') if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-start;report['raw_files']={str(p.relative_to(RAW)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.rglob('*')) if p.is_file() and p.name not in ('partial-report.json','report.json')};write(RAW/'report.json',report);data=json.dumps(report,sort_keys=True,allow_nan=False).encode();print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit();raise SystemExit(code)
