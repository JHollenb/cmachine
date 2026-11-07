"""One resident original C model; frozen program x memory transfer panel."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'results'
LIMIT=128*1024**2
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):
    print('[run]',' '.join(map(str,args)),flush=True);p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=720)
    if p.returncode:raise RuntimeError(f'Process{p.returncode}: {p.stderr[-10000:]}')
    return p
def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze
    sys.path.insert(0,str(ROOT/'public'))
    from saturn_bridge import invoke
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source differs '+name)
    model=json.loads((ROOT/'baseline.json').read_text())['model']
    source=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name,digest in model['tokenizer_files'].items():
        if sha(source/name)!=digest:raise ValueError('Tokenizer differs')
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    source_texts=(
        'A notebook records a person, a place, a color, and a number. Explain briefly why keeping the records separate helps someone read them later.',
        'Iris keeps a yellow card and Pavel keeps a purple card. Pavel changes his card to orange. Describe the final cards briefly.',
        'Explain briefly how to multiply two small integers, subtract a number from the result, and check the calculation. Use a simple example.')
    sources=[]
    for i,prompt in enumerate(source_texts):
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_dict=False)
        if not isinstance(ids,list) or not all(type(x)is int for x in ids):raise ValueError('Tokenizer IDs differ')
        (RAW/f'source{i}.txt').write_text(' '.join(map(str,ids))+'\n');sources.append({'text':prompt,'ids':ids})
    cases=(
        ('recall','Tessa keeps a red badge and Rowan keeps a blue badge. Which color belongs to Rowan? Reply with only the color.','Tessa trades her red badge for a green badge. What color does Tessa have now? Reply with only the color.',8),
        ('math','Compute 14 * 19. Reply with only the integer.','Add 13 to your previous answer. Reply with the integer, then one short sentence explaining the calculation.',16),
        ('prose','Explain in one short sentence why a person should label a notebook.','Now give one practical example of a useful label, in one short sentence.',32))
    inputs={'sources':sources,'cases':[],'source_answer_budget':0,'ranks':[128],'compiler':'authenticated-source128-bank-reuse','policies':['native','compiled','memory','compiled-memory','values-native','values-native-memory'],'recent':16,'prototypes':16,'targets_start_from_token_zero':True}
    for i,(name,prompt,follow,budget) in enumerate(cases):
        messages=[{'role':'user','content':prompt}];text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(text,add_special_tokens=False)
        extended=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':''},{'role':'user','content':follow}],tokenize=False,add_generation_prompt=True)
        if not extended.startswith(text):raise ValueError('Template prefix differs')
        suffix=tokenizer.encode(extended[len(text):],add_special_tokens=False)
        if tokenizer.encode(extended,add_special_tokens=False)!=ids+suffix or suffix[0]!=151645:raise ValueError('Token concatenation differs')
        for label,tokens in (('prompt',ids),('followup',suffix)):(RAW/f'case{i}-{label}.txt').write_text(' '.join(map(str,tokens))+'\n')
        inputs['cases'].append({'name':name,'prompt_text':prompt,'followup_text':follow,'prompt_ids':ids,'followup_ids':suffix,'budget':budget})
    maxima=[len(c['prompt_ids'])+len(c['followup_ids'])+2*c['budget'] for c in inputs['cases']]
    bound=sum(6*n*27140+3*6*n*(n+1)*4 for n in maxima)+18*4*151936*4+9*4*98464+1835008+8*1024**2
    if bound>LIMIT:raise ValueError('Archive geometry exceeds declared bound')
    (RAW/'input-manifest.json').write_text(json.dumps(inputs,sort_keys=True)+'\n')
    provenance=json.loads((ROOT/'program-provenance.json').read_text())
    if provenance['model_digest']!=model['content_sha256'] or provenance['model_identity']!=model['identity']:raise ValueError('Compiled program model differs')
    for name,description in provenance['files'].items():
        if sha(ROOT/name)!=description['sha256'] or (ROOT/name).stat().st_size!=description['bytes']:raise ValueError('Compiled program bytes differ')
        shutil.copy2(ROOT/name,RAW/name)
    (RAW/'program-provenance.json').write_text(json.dumps(provenance,sort_keys=True)+'\n')
    report={'schema':'tcm-query-memory-transfer-experiment-v1','model':model,'inputs':inputs,'input_manifest_sha256':sha(RAW/'input-manifest.json'),'evidence_capacity_preflight':{'worst_case_bytes':bound,'limit_bytes':LIMIT,'maxima':maxima},'program_provenance':provenance,'texts':{},'environment':{'platform':platform.platform(),'python':sys.version,'job_id':os.environ.get('MRUN_JOB_ID')},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested'}
    (RAW/'partial-report.json').write_text(json.dumps(report)+'\n');command(['make']);report['binary_sha256']=sha(ROOT/'build/query-memory-transfer')
    args=[ROOT/'build/query-memory-transfer',model['path'],RAW]
    def controller(_):
        p=command(args);(RAW/'driver.stdout.txt').write_text(p.stdout);(RAW/'driver.stderr.txt').write_text(p.stderr);print(p.stderr,end='',flush=True);return json.loads((RAW/'driver.json').read_text())
    report['driver'],report['saturn_receipt']=invoke(args,None,{'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'input_manifest_sha256':report['input_manifest_sha256'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'scope':'NN/CC/CNxfull/compact memory;15candidate paths sealed before3native;reusedauthenticated128program;ownrawprompts/IDs/memory;probabilitycapture'},controller=controller)
    if report['driver']['model_digest']!=model['content_sha256'] or report['driver']['model_identity']!=model['identity']:raise ValueError('Executed model differs')
    for path in report['driver']['paths']:report['texts'][path['name']]=[tokenizer.decode(t['answer_ids'],skip_special_tokens=True) for t in path['turns']]
    (RAW/'partial-report.json').write_text(json.dumps(report)+'\n')
    report['analysis']=analyze(RAW,ROOT/'bank',report['driver'],report['texts'])
    report['raw_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name!='partial-report.json'}
    report.update(mechanics_status='complete',instrument_status='complete');return report
def emit():
    files=[]
    for directory,prefix in ((RAW,'raw'),(ROOT/'runtime','payload/runtime'),(ROOT/'public','payload/public')):
        files += [(p,prefix+'/'+str(p.relative_to(directory))) for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    files += [(p,'payload/'+p.name) for p in sorted(ROOT.iterdir()) if p.is_file()]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('Archive bound exceeded')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz')as archive:
        for path,name in files:archive.add(path,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)
if __name__=='__main__':
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:
        report=json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-started;data=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/'report.json').write_bytes(data+b'\n');print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit();raise SystemExit(code)
