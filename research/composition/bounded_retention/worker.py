"""One native-consumer C invocation, five sequential own-state retention policies."""
from pathlib import Path
import base64,gzip,io,json,os,platform,sys,tarfile,time,traceback,subprocess
import support
from support import ROOT,RAW,sha,command,preserve
support.LIMIT=192*1024**2
def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze
    sys.path.insert(0,str(ROOT/'public'));from saturn_bridge import invoke
    origin=json.loads((ROOT/'origin.json').read_text());manifest=json.loads((ROOT/'source-manifest.json').read_text());panel=json.loads((ROOT/'panel.json').read_text());model=origin['model']
    for name,h in manifest['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Shipped source differs')
    source=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name,h in model['tokenizer_files'].items():
        if sha(source/name)!=h:raise ValueError('Tokenizer differs')
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);source_text=tokenizer.apply_chat_template([{'role':'user','content':panel[0]['text']}],tokenize=False,add_generation_prompt=True);source_ids=tokenizer.encode(source_text,add_special_tokens=False);inputs=[]
    for i,q in enumerate(panel):
        if i==0:rendered=source_text;ids=source_ids
        else:
            full=tokenizer.apply_chat_template([{'role':'user','content':panel[0]['text']},{'role':'assistant','content':''},{'role':'user','content':q['text']}],tokenize=False,add_generation_prompt=True)
            if not full.startswith(source_text):raise ValueError('Chat append prefix differs')
            rendered=full[len(source_text):];ids=tokenizer.encode(rendered,add_special_tokens=False)
            if tokenizer.encode(full,add_special_tokens=False)!=source_ids+ids or ids[0]!=151645:raise ValueError('Append token grammar differs')
        if len(ids)>512:raise ValueError('Input reader bound exceeded')
        clauses={0:["Eno's number is 17.","Tavi's number is 23.","Luma's color is amber."],1:["Tavi's number is now 29, replacing 23."]}.get(i,[]);spans=[]
        if clauses:
            encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
            if encoded['input_ids']!=ids:raise ValueError('Offset tokenizer differs')
            for clause in clauses:
                start=rendered.index(clause);end=start+len(clause);slots=[j for j,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>start];spans.append({'clause':clause,'token_indices':slots})
        (RAW/f'input-{i}.txt').write_text(' '.join(map(str,ids))+'\n');inputs.append({**q,'ids':ids,'file':f'input-{i}.txt','fact_spans':spans})
    (RAW/'inputs.json').write_text(json.dumps(inputs,sort_keys=True)+'\n')
    facts_start=len(tokenizer.encode(source_text[:source_text.index("Eno's number")],add_special_tokens=False))
    if facts_start<=16 or len(inputs[2]['ids'])<160:raise ValueError('Eviction geometry not discriminating')
    root_bound=sum(len(inputs[i]['ids'])+inputs[i]['budget'] for i in (0,1,2));child=max(len(inputs[i]['ids'])+inputs[i]['budget'] for i in (3,4,5,6,7))
    if root_bound+max(child,132)>768:raise ValueError('Logical context bound exceeded')
    build=command(['make']);report={'schema':'tcm-bounded-retention-v1','model':model,'inputs':inputs,'fact_source_start':facts_start,'predecessor_report_sha256':origin['predecessor_report_sha256'],'environment':{'platform':platform.platform(),'python':sys.version,'job_id':os.environ.get('MRUN_JOB_ID')},'binary_sha256':sha(ROOT/'build/composition'),'build_stderr':build.stderr,'budget':{'model_loads':1,'sessions':5,'turns_per_session':8,'logical_capacity':768,'root_bound':root_bound,'child_bound':child,'raw_archive_limit_bytes':support.LIMIT},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested'};preserve(report)
    args=[str(ROOT/'build/composition'),model['path'],str(RAW),str(RAW)];request={'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'inputs':inputs,'plan_sha256':manifest['plan_sha256'],'origin_sha256':sha(ROOT/'origin.json'),'execution_command':args,'construction':'five sequential own-state sessions, one source BF16 weight load, all byte policies from token zero','budget':report['budget']}
    def controller(_):
        r=subprocess.run(args,capture_output=True,text=True);(RAW/'driver.stdout.txt').write_text(r.stdout);(RAW/'driver.stderr.txt').write_text(r.stderr)
        if r.returncode:raise RuntimeError(f'C process exit {r.returncode}: '+r.stderr[-6000:])
        d=json.loads((RAW/'driver.json').read_text())
        if d['model_digest']!=model['content_sha256']:raise ValueError('Actual model differs')
        return d
    report['driver'],report['saturn_receipt']=invoke(args,RAW/'driver.json',request,controller=controller)
    report['texts']={t['name']:tokenizer.decode(t['answer_ids'],skip_special_tokens=True) for a in report['driver']['arms'] for t in a['turns']};preserve(report)
    report['analysis']=analyze(RAW,ROOT,report['driver'],report['texts']);report['raw_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name not in ('partial-report.json','report.json')};report.update(mechanics_status='complete',instrument_status='complete');return report
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
    except Exception:report=json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-started;data=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/'report.json').write_bytes(data+b'\n');print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit_files();raise SystemExit(code)
