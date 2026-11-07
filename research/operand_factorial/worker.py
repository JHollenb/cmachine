"""One finite mrun lease: source-frozen C factorial, then saved-array analysis."""
import base64,gzip,hashlib,io,json,os,platform,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'results';LIMIT=256*1024**2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def command(args,timeout=720):
    print('[run] '+' '.join(map(str,args)),flush=True)
    r=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'{r.returncode}: {r.stderr[-10000:]}')
    return r

def experiment():
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source differs: '+name)
    plan=json.loads((ROOT/'clock-plan.json').read_text());model=plan['model']
    if plan['archive_bound_bytes']>LIMIT:raise ValueError('declared archive bound')
    RAW.mkdir(exist_ok=True);build=command(['make']);(RAW/'build.stderr.txt').write_text(build.stderr)
    args=[ROOT/'build/operand-factorial',model['path'],RAW,ROOT/'compiled',ROOT/'clock-list.txt',model['content_sha256']]
    sys.path.insert(0,str(ROOT/'public'));from saturn_bridge import invoke
    def control(_):
        run=command(args);(RAW/'driver.stdout.txt').write_text(run.stdout);(RAW/'driver.stderr.txt').write_text(run.stderr)
        return json.loads((RAW/'driver.json').read_text())
    request={'schema':'tcm-operand-factorial-request-v1','model_digest':model['content_sha256'],'clock_plan_sha256':sha(ROOT/'clock-plan.json'),'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'binary_sha256':sha(ROOT/'build/operand-factorial'),'horizon':4,'diagnostic_original_operands_paid':True}
    driver,receipt=invoke(args,None,request,controller=control)
    from analysis import analyze
    analysis=analyze(RAW,ROOT/'saved',driver,plan,save_projected=True)
    from transformers import AutoTokenizer
    token_path=Path(plan['tokenizer_path'])
    for name,digest in model['tokenizer_files'].items():
        if sha(token_path/name)!=digest:raise ValueError('tokenizer differs')
    tokenizer=AutoTokenizer.from_pretrained(token_path,local_files_only=True)
    for r in analysis['observations']:r['continuation_text']=tokenizer.decode(r['continuation']['ids'],skip_special_tokens=True)
    (RAW/'binary').mkdir();shutil.copy2(ROOT/'build/operand-factorial',RAW/'binary/operand-factorial')
    return {'schema':'tcm-operand-factorial-report-v1','model':model,'plan':plan,'driver':driver,'analysis':analysis,'saturn_receipt':receipt,'binary_sha256':request['binary_sha256'],
        'mechanics_status':'complete' if analysis['summary']['mechanics_mismatches']==0 else 'instrument-discrepancy','instrument_status':'complete' if analysis['summary']['mechanics_mismatches']==0 else 'instrument-discrepancy','trend_status':'observations','terminal_status':'not-requested','environment':{'platform':platform.platform(),'python':sys.version,'job_id':os.environ.get('MRUN_JOB_ID')},
        'raw_files':{str(p.relative_to(RAW)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.rglob('*')) if p.is_file()}}

def emit(report):
    encoded=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/'report.json').write_bytes(encoded+b'\n')
    print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(encoded,mtime=0)).decode(),flush=True)
    files=[(p,'raw/'+str(p.relative_to(RAW))) for p in RAW.rglob('*') if p.is_file()]
    files += [(p,'payload/'+str(p.relative_to(ROOT))) for p in ROOT.rglob('*') if p.is_file() and not p.is_relative_to(RAW) and '__pycache__' not in p.parts and 'build' not in p.parts]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('archive bound')
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as t:
        for p,name in files:t.add(p,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buf.getvalue()).decode(),flush=True)

if __name__=='__main__':
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:report={'error':traceback.format_exc(),'mechanics_status':'incomplete','instrument_status':'incomplete','trend_status':'evidence-incomplete','terminal_status':'not-requested'};code=1
    report['worker_seconds']=time.monotonic()-started;emit(report);raise SystemExit(code)
