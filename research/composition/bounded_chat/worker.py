"""Build, exercise and install the opt-in persistent chat interface in one lease."""
from pathlib import Path
import base64,gzip,io,json,os,platform,sys,tarfile,time,traceback,subprocess
import support
from support import ROOT,RAW,sha,command,preserve
support.LIMIT=24*1024**2
def experiment():
    from analysis import analyze
    from controller import protocol_case
    from install import prepare,publish
    sys.path.insert(0,str(ROOT/'public'));from bridge import invoke
    manifest=json.loads((ROOT/'source-manifest.json').read_text());origin=json.loads((ROOT/'origin.json').read_text());model=origin['model'];tokenizer_path=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for n,h in manifest['sources'].items():
        if sha(ROOT/n)!=h:raise ValueError('Payload differs')
    for n,h in model['tokenizer_files'].items():
        if sha(tokenizer_path/n)!=h:raise ValueError('Tokenizer differs')
    build=command(['make']);installation=prepare(ROOT);target=Path(installation['path']);binary=target/'build/tcm-bounded-chat';client=target/'tools/chat.py';native=target/'build/tcm-cuda'
    (RAW/'installed-files.json').write_bytes((target/'install-manifest.json').read_bytes())
    policies=('full-byte','bounded');protocol_commands=[[str(binary),'serve',model['path'],'--context','768','--cache-policy',p] for p in policies]
    demo=['/save','Explain why bread rises when yeast is added. Use two short sentences.','/state','/push','Compute 17 + 29. Reply with only the integer.','/state','/rollback','/push',"Remember: Eno's number is 17. Reply only OK.",'/push','/push',"What is Eno's number? Reply with only the number.",'/state','/rollback',"What is Eno's number? Reply with only the number.",'/state','/commit','/state','/quit']
    q=origin['inputs'];delayed=[q[0]['text'],'/push',q[3]['text'],'/rollback',q[1]['text'],q[2]['text'],'/state']
    for i in range(4,8):delayed+=['/push',q[i]['text'],'/state','/rollback']
    delayed+=['/quit'];cases=[]
    for p in policies:
        for label,lines in (('demo',demo),('delayed',delayed)):
            cases.append({'name':p+'-'+label,'policy':p,'stdin':'\n'.join(lines)+'\n','command':[sys.executable,str(client),'--model',model['path'],'--tokenizer',str(tokenizer_path),'--context','768','--new-tokens','32','--cache-policy',p]})
    cases.append({'name':'native-demo','policy':'native','stdin':'Explain why bread rises when yeast is added. Use two short sentences.\n/state\n/quit\n','command':[sys.executable,str(client),'--binary',str(native),'--model',model['path'],'--tokenizer',str(tokenizer_path),'--context','768','--new-tokens','32']})
    commands=protocol_commands+[c['command'] for c in cases];request={'model_digest':model['content_sha256'],'binary_sha256':sha(binary),'client_sha256':sha(client),'execution_commands':commands,'inputs':origin['inputs'],'client_transcripts':[{k:v for k,v in c.items() if k!='command'} for c in cases],'plan_sha256':manifest['plan_sha256'],'installation_path':str(target),'horizon':'seven sequential persistent process invocations'};(RAW/'request.json').write_text(json.dumps(request,sort_keys=True)+'\n')
    report={'schema':'tcm-bounded-chat-v1','model':model,'environment':{'platform':platform.platform(),'job_id':os.environ.get('MRUN_JOB_ID')},'binary_sha256':sha(binary),'build_stderr':build.stderr,'installation_prepared':dict(installation),'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'opt-in-execution-only'};preserve(report)
    def controller(_):
        protocols=[];clients=[]
        for p,args in zip(policies,protocol_commands):
            print('[protocol] '+p,flush=True);result=protocol_case(args,RAW,p,origin);protocols.append({'policy':p,'command':args,'result':result});report['protocol_partial']=protocols;preserve(report)
        for c in cases:
            print('[client] '+c['name'],flush=True);(RAW/(c['name']+'.stdin.txt')).write_text(c['stdin']);r=subprocess.run(c['command'],input=c['stdin'],capture_output=True,text=True,timeout=180);(RAW/(c['name']+'.stdout.txt')).write_text(r.stdout);(RAW/(c['name']+'.stderr.txt')).write_text(r.stderr)
            if r.returncode:raise RuntimeError('Client '+c['name']+' failed: '+r.stderr[-4000:])
            clients.append({k:v for k,v in c.items() if k!='stdin'}|{'returncode':r.returncode,'stdin_sha256':sha(RAW/(c['name']+'.stdin.txt')),'stdout_sha256':sha(RAW/(c['name']+'.stdout.txt'))});report['clients_partial']=clients;preserve(report)
        # Validate all saved mechanics before making the opt-in link discoverable.
        provisional={'schema':'tcm-bounded-chat-driver-v1','binary_sha256':sha(binary),'protocol':protocols,'clients':clients,'installation':dict(installation)}
        analyze(RAW,ROOT,provisional,{},installation_stage='prepared')
        installation_receipt=publish(installation);return {**provisional,'installation':installation_receipt}
    report['driver'],report['saturn_receipt']=invoke(commands,request,controller);(RAW/'driver.json').write_text(json.dumps(report['driver'],sort_keys=True)+'\n');(RAW/'installation.json').write_text(json.dumps(report['driver']['installation'],indent=2)+'\n');report['texts']={};report['analysis']=analyze(RAW,ROOT,report['driver'],{});report['raw_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name not in ('partial-report.json','report.json')};report.update(mechanics_status='complete',instrument_status='complete');return report
def emit_files():
    files=[(p,'raw/'+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_file() and '__pycache__' not in rel.parts and (len(rel.parts)==1 or rel.parts[0] in ('runtime','public')):files.append((p,'payload/'+str(rel)))
    if sum(p.stat().st_size for p,_ in files)>support.LIMIT:raise ValueError('Custody bound exceeded')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path,name in files:archive.add(path,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)
if __name__=='__main__':
    RAW.mkdir(exist_ok=True);start=time.monotonic()
    try:report=experiment();code=0
    except Exception:report=json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-start;data=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/'report.json').write_bytes(data+b'\n');print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit_files();raise SystemExit(code)
