"""Three matched weight-storage profiles; sequential model processes and sealed evidence."""
from pathlib import Path
import base64,gzip,io,json,os,platform,sys,tarfile,time,traceback,subprocess
import support
from support import ROOT,RAW,sha,command,preserve
support.LIMIT=128*1024**2
def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze
    sys.path.insert(0,str(ROOT/'public'));from panel_bridge import invoke
    baseline=json.loads((ROOT/'baseline.json').read_text());manifest=json.loads((ROOT/'source-manifest.json').read_text());ref=json.loads((ROOT/'origin.json').read_text());questions=json.loads((ROOT/'questions.json').read_text())
    for name,expected in manifest['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Shipped source differs')
    model=baseline['model']
    if ref['model']!=model or ref['predecessor_report_sha256']!=manifest['predecessor_report_sha256']:raise ValueError('Historical model differs')
    source=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name,expected in model['tokenizer_files'].items():
        if sha(source/name)!=expected:raise ValueError('Tokenizer differs')
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);messages=[{'role':'user','content':baseline['source_template_text']}];text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(text,add_special_tokens=False)
    if ids!=baseline['source_template_ids']:raise ValueError('Source template differs')
    inputs=[]
    for ti,q in enumerate(questions):
        if ti==0:tokens=ids
        else:
            full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':''},{'role':'user','content':q['text']}],tokenize=False,add_generation_prompt=True);tokens=tokenizer.encode(full[len(text):],add_special_tokens=False)
            if not full.startswith(text) or tokenizer.encode(full,add_special_tokens=False)!=ids+tokens or not tokens or tokens[0]!=151645:raise ValueError('Append grammar differs')
            if ti==1 and tokens!=ref['correction_input']['ids']:raise ValueError('Correction template differs')
        (RAW/q['file']).write_text(' '.join(map(str,tokens))+'\n');inputs.append({**q,'ids':tokens})
    nmax=sum(len(q['ids'])+q['budget'] for q in inputs)
    if nmax>512:raise ValueError('Context bound exceeded')
    payload_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'results' not in p.relative_to(ROOT).parts)
    lengths=[];p=0
    for q in inputs:p+=len(q['ids'])+q['budget'];lengths.append(p)
    bound=3*nmax*(28*2*2*128*4+4*12*128*4)+30*151936*4+30*4*6*1536*4+3*28*sum(lengths)*528+3*151936*4+8*1024**2+payload_bytes
    if bound>support.LIMIT:raise ValueError('Custody bound exceeded')
    build=command(['make']);report={'schema':'tcm-weight-tiles-experiment-v1','model':model,'inputs':inputs,'predecessor_report_sha256':ref['predecessor_report_sha256'],'environment':{'platform':platform.platform(),'python':sys.version,'job_id':os.environ.get('MRUN_JOB_ID')},'binary_sha256':sha(ROOT/'build/composition'),'build_stderr':build.stderr,'budget':{'sessions':3,'turns_per_session':5,'main_token_bound':3*nmax,'selected_cache_step_bound':84*(nmax+7),'model_loads':3,'raw_and_payload_byte_bound':bound,'raw_archive_limit_bytes':support.LIMIT},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested'};preserve(report)
    args=[ROOT/'build/composition',model['path'],*[RAW/q['file'] for q in inputs]];commands=[[str(x) for x in [*args,RAW/f'profile-{mode}',mode]] for mode in range(3)]
    request={'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'inputs':inputs,'plan_sha256':manifest['plan_sha256'],'origin_sha256':sha(ROOT/'origin.json'),'predecessor_report_sha256':ref['predecessor_report_sha256'],'construction':'three sequential weight profiles, all-byte cache from token0, own tokens, no native refresh','execution_commands':commands,'budget':report['budget']}
    def controller(_):
        profiles=[]
        for mode in range(3):
            out=RAW/f'profile-{mode}';out.mkdir();print(f'[profile {mode}] start',flush=True)
            r=subprocess.run(commands[mode],capture_output=True,text=True)
            (out/'driver.stdout.txt').write_text(r.stdout);(out/'driver.stderr.txt').write_text(r.stderr)
            for p in list(out.iterdir()):p.rename(RAW/f'w{mode}-{p.name}')
            out.rmdir()
            if r.returncode:raise RuntimeError(f'Weight profile {mode} exit{r.returncode}: '+r.stderr[-5000:])
            d=json.loads((RAW/f'w{mode}-driver.json').read_text())
            if d['model_digest']!=model['content_sha256']:raise ValueError('Actual model differs')
            profiles.append({'profile':mode,'driver':d});report['driver']={'schema':'tcm-weight-tiles-driver-v1','profiles':profiles};preserve(report);print(f'[profile {mode}] complete {d["seconds"]:.3f}s',flush=True)
        return report['driver']
    driver,report['saturn_receipt']=invoke(commands,None,request,controller=controller);report['driver']=driver;(RAW/'driver.json').write_text(json.dumps(driver,sort_keys=True)+'\n')
    report['texts']={f'w{p["profile"]}-'+a['name']:tokenizer.decode(a['answer_ids'],skip_special_tokens=True) for p in driver['profiles'] for c in p['driver']['chains'] for a in c['turns']+c['later_turns']};preserve(report)
    report['analysis']=analyze(RAW,ROOT,driver,report['texts']);report['raw_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in ('.f32','.f64','.bin','.txt') or p.name.endswith('driver.json')};report.update(mechanics_status='complete',instrument_status='complete');return report

def emit_files():
    files=[(p,'raw/'+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_file() and '__pycache__' not in rel.parts and (len(rel.parts)==1 or rel.parts[0] in ('runtime','public','original_runtime')):files.append((p,'payload/'+str(rel)))
    if sum(p.stat().st_size for p,_ in files)>support.LIMIT:raise ValueError('Archive exceeds bound')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path,name in files:archive.add(path,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)
if __name__=='__main__':
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:
        report=json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-started;data=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/'report.json').write_bytes(data+b'\n');print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit_files();raise SystemExit(code)
