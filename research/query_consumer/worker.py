"""One resident C model; source-consumer selection then quarantined targets."""
import base64,gzip,hashlib,io,json,os,platform,selectors,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'results';LIMIT=256*1024**2
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args):
    print('[run]',' '.join(map(str,args)),flush=True);p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError(f'Process{p.returncode}: {p.stderr[-12000:]}')
    return p
def targets(model,selection):
    from transformers import AutoTokenizer
    source=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name,digest in model['tokenizer_files'].items():
        if sha(source/name)!=digest:raise ValueError('Tokenizer differs')
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    cases=(('recall','Tessa keeps a red badge and Rowan keeps a blue badge. Which color belongs to Rowan? Reply with only the color.','Tessa trades her red badge for a green badge. What color does Tessa have now? Reply with only the color.',8),('math','Compute 14 * 19. Reply with only the integer.','Add 13 to your previous answer. Reply with the integer, then one short sentence explaining the calculation.',16),('prose','Explain in one short sentence why a person should label a notebook.','Now give one practical example of a useful label, in one short sentence.',32))
    fresh=json.loads((ROOT/'fresh.json').read_text());cases+=tuple(tuple(row) for row in fresh['cases']);inputs={'cases':[],'fresh_input_sha256':sha(ROOT/'fresh.json'),'selection_seal':selection,'targets_start_from_token_zero':True,'scope':'Targets parsed/tokenized only after verified source selection; no target fitting or image selection.'}
    for i,(name,prompt,follow,budget) in enumerate(cases):
        messages=[{'role':'user','content':prompt}];text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(text,add_special_tokens=False);extended=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':''},{'role':'user','content':follow}],tokenize=False,add_generation_prompt=True)
        if not extended.startswith(text):raise ValueError('Template prefix differs')
        suffix=tokenizer.encode(extended[len(text):],add_special_tokens=False)
        if tokenizer.encode(extended,add_special_tokens=False)!=ids+suffix or suffix[0]!=151645 or budget!=(8,16,32)[i%3]:raise ValueError('Token authority/budget differs')
        for label,tokens in (('prompt',ids),('followup',suffix)):(RAW/f'case{i}-{label}.txt').write_text(' '.join(map(str,tokens))+'\n')
        inputs['cases'].append({'name':name,'prompt_text':prompt,'followup_text':follow,'prompt_ids':ids,'followup_ids':suffix,'budget':budget,'split':'development' if i<3 else 'prospective-transfer'})
    maxima=[len(c['prompt_ids'])+len(c['followup_ids'])+2*c['budget'] for c in inputs['cases']]
    # Worst case: three distinct compact candidates and native per target.
    target_arrays=sum(4*n*27140+6*n*(n+1)*4+16*151936*4+12*98464 for n in maxima)
    preexisting_raw=sum(p.stat().st_size for p in RAW.iterdir() if p.is_file())
    payload_bytes=sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())+sum(p.stat().st_size for d in ('runtime','public') for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    metadata_allowance=16*1024**2
    bound=target_arrays+preexisting_raw+payload_bytes+metadata_allowance
    if bound>LIMIT:raise ValueError('Worst-case source+target archive exceeds declared bound')
    (RAW/'input-manifest.json').write_text(json.dumps(inputs,sort_keys=True)+'\n');return inputs,tokenizer,{'maxima':maxima,'worst_case_bytes':bound,'target_arrays_bound_bytes':target_arrays,'preexisting_raw_bytes':preexisting_raw,'payload_bytes':payload_bytes,'metadata_allowance_bytes':metadata_allowance,'limit_bytes':LIMIT}
def experiment():
    sys.path.insert(0,str(ROOT/'public'));from saturn_bridge import invoke
    from metrics import audit_source,choose
    from analysis import analyze
    manifest=json.loads((ROOT/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source differs '+name)
    model=json.loads((ROOT/'baseline.json').read_text())['model'];provenance=json.loads((ROOT/'program-provenance.json').read_text());family=json.loads((ROOT/'family-provenance.json').read_text());source=json.loads((ROOT/'source-provenance.json').read_text())
    for name in ('program-provenance.json','family-provenance.json','source-provenance.json','source-inputs.f32'):shutil.copy2(ROOT/name,RAW/name)
    files={**provenance['files'],**family['files']}
    for name,record in files.items():
        if sha(ROOT/name)!=record['sha256'] or (ROOT/name).stat().st_size!=record['bytes']:raise ValueError('Bank/image identity differs')
        shutil.copy2(ROOT/name,RAW/name)
    shutil.copy2(ROOT/'query-proposal-4.f32',RAW/'query-fit.f32')
    for i,row in enumerate(source['source_trajectories']):
        ids=row['prompt_ids']+row['own_ids']
        if row['task']!=i or len(ids)!=row['advances']:raise ValueError('Source IDs differ')
        (RAW/f'source{i}.txt').write_text(' '.join(map(str,ids))+'\n')
    (RAW/'source-prompt-lengths.txt').write_text(' '.join(str(len(r['prompt_ids'])) for r in source['source_trajectories'])+'\n')
    report={'schema':'tcm-query-consumer-experiment-v1','model':model,'program_provenance':provenance,'family_provenance':family,'source_provenance':source,'texts':{},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested','environment':{'platform':platform.platform(),'python':sys.version,'job_id':os.environ.get('MRUN_JOB_ID')}}
    (RAW/'partial-report.json').write_text(json.dumps(report)+'\n');command(['make']);report['binary_sha256']=sha(ROOT/'build/query-program');args=[ROOT/'build/query-program',model['path'],RAW]
    def controller(_):
        audits=[];selected=False;stdout=[];deadline=time.monotonic()+600
        with (RAW/'driver.stderr.txt').open('w') as error:
            process=subprocess.Popen(list(map(str,args)),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1);poll=selectors.DefaultSelector();poll.register(process.stdout,selectors.EVENT_READ)
            try:
                while True:
                    if time.monotonic()>deadline:raise RuntimeError('Bounded source/target process timeout')
                    if not poll.select(1):
                        if process.poll() is not None:break
                        continue
                    line=process.stdout.readline()
                    if not line:
                        if process.poll() is not None:break
                        continue
                    if len(line)>4096:raise ValueError('Unexpected driver protocol output')
                    stdout.append(line)
                    if line.startswith('SOURCE_CANDIDATE_READY '):
                        index=int(line.split()[1])
                        if index!=len(audits) or selected:raise ValueError('Source phase order differs')
                        audits.append(audit_source(RAW,index));print(f'source proposal{index}: C/Python full-clock KL agrees; mean={audits[-1]["equal_trajectory_mean_kl_nats"]:.9g}',flush=True)
                    elif line.startswith('SOURCE_SELECTION_READY '):
                        _,index,selection_sha,image_sha=line.split();index=int(index)
                        if selected or len(audits)!=6 or sha(RAW/'source-selection.json')!=selection_sha or sha(RAW/'selected-query.f32')!=image_sha:raise ValueError('Source selection seal differs')
                        selection=json.loads((RAW/'source-selection.json').read_text())
                        if selection['selected_index']!=index or choose([a['equal_trajectory_mean_kl_nats'] for a in audits])!=index or image_sha!=family['files'][f'query-proposal-{index}.f32']['sha256']:raise ValueError('Independent source selection differs')
                        seal={'phase':'before-parsing-fresh-prompts-or-tokenizing-targets','selected_index':index,'selection_sha256':selection_sha,'image_sha256':image_sha,'source_audit_sha256':[sha(RAW/f'source-audit-{i}.json') for i in range(6)]};(RAW/'selection-seal.json').write_text(json.dumps(seal,sort_keys=True)+'\n')
                        for name in ('source-native-logits.tmp','source-candidate-logits.tmp'):(RAW/name).unlink()
                        report['inputs'],tok,report['evidence_capacity_preflight']=targets(model,seal);report['input_manifest_sha256']=sha(RAW/'input-manifest.json');report['source_selection']=selection;report['selection_seal']=seal;report['_tokenizer']=tok;selected=True;print(f'sealed source-KL image{index}; parsed new targets afterward',flush=True)
                    else:raise ValueError('Unexpected driver protocol message')
                    process.stdin.write('\n');process.stdin.flush()
                code=process.wait(timeout=5)
                if code or not selected:raise RuntimeError(f'C process{code}; stderr: {(RAW/"driver.stderr.txt").read_text()[-12000:]}')
            finally:
                poll.close()
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                process.stdin.close();process.stdout.close();(RAW/'driver.stdout.txt').write_text(''.join(stdout))
                for name in ('source-native-logits.tmp','source-candidate-logits.tmp'):
                    if (RAW/name).exists():(RAW/name).unlink()
        return json.loads((RAW/'driver.json').read_text())
    report['driver'],report['saturn_receipt']=invoke(args,None,{'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'source_provenance_sha256':sha(ROOT/'source-provenance.json'),'family_provenance_sha256':sha(ROOT/'family-provenance.json'),'scope':'Source-native-KL selection on six frozen images, then targets after source seal; final result binds later selection and target manifest; one resident C invocation'},controller=controller)
    tokenizer=report.pop('_tokenizer')
    if report['driver']['model_digest']!=model['content_sha256'] or report['driver']['model_identity']!=model['identity'] or report['driver']['source_selection_sha256']!=report['selection_seal']['selection_sha256'] or report['driver']['target_input_manifest_sha256']!=report['input_manifest_sha256']:raise ValueError('Final C model/phase bindings differ')
    for path in report['driver']['paths']:report['texts'][path['name']]=[tokenizer.decode(t['answer_ids'],skip_special_tokens=True) for t in path['turns']]
    (RAW/'partial-report.json').write_text(json.dumps(report)+'\n');report['analysis']=analyze(RAW,ROOT/'bank',report['driver'],report['texts']);report['raw_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name!='partial-report.json'};report.update(mechanics_status='complete',instrument_status='complete');return report
def emit():
    files=[]
    for directory,prefix in ((RAW,'raw'),(ROOT/'runtime','payload/runtime'),(ROOT/'public','payload/public')):files += [(p,prefix+'/'+str(p.relative_to(directory))) for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
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
