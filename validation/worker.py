"""Build a committed integration and test it in one finite CUDA lease."""
from pathlib import Path
import base64,gzip,io,json,os,platform,subprocess,sys,tarfile,time,traceback
from support import ROOT,RAW,LIMIT,sha,command,preserve

def build_reference():
    flags=['-O3','-std=c11','-ffp-contract=off','-DTCM_WITH_CUDA','-Ibaseline']
    objects=[];logs=[]
    for name in ('model','isa','main','qwen'):
        obj='build/reference-'+name+'.o';r=command(['cc',*flags,'-c','baseline/'+name+'.c','-o',obj]);objects.append(obj);logs.append(r.stdout+r.stderr)
    # Same reporting-only server view, forced to the reference's original ABI.
    r=command(['cc',*flags,'-include','baseline/qwen.h','-include','baseline_serve_stub.h','-c','src/qwen_serve.c','-o','build/reference-serve.o']);logs.append(r.stdout+r.stderr);objects.append('build/reference-serve.o')
    command(['cc',*flags,'-c','baseline_serve_stub.c','-o','build/reference-stub.o']);objects.append('build/reference-stub.o')
    command(['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','--fmad=false','-arch=sm_89','-Ibaseline','-c','baseline/qwen_cuda.cu','-o','build/reference-cuda.o']);objects.append('build/reference-cuda.o')
    command(['/usr/local/cuda/bin/nvcc',*objects,'-lcublas','-lm','-o','build/tcm-reference'])
    (RAW/'reference-build.txt').write_text(''.join(logs))

def experiment():
    from analysis import analyze
    from controller import protocol_case,native_case
    from install import prepare,publish
    sys.path.insert(0,str(ROOT/'public'));from bridge import invoke
    manifest=json.loads((ROOT/'source-manifest.json').read_text());origin=json.loads((ROOT/'origin.json').read_text());model=origin['model'];tokenizer=Path('/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    for name,row in manifest['sources'].items():
        if sha(ROOT/name)!=row['sha256']:raise ValueError('Commit source byte differs: '+name)
    for name,h in model['tokenizer_files'].items():
        if sha(tokenizer/name)!=h:raise ValueError('Tokenizer identity differs')
    build=command(['make','cuda','stack-mechanics','bounded-gpu-mechanics']);(RAW/'build.stdout.txt').write_text(build.stdout);(RAW/'build.stderr.txt').write_text(build.stderr);build_reference()
    installation=prepare(ROOT,manifest);target=Path(installation['path']);binary=target/'build/tcm-cuda';client=target/'tools/chat.py';(RAW/'installed-files.json').write_bytes((target/'install-manifest.json').read_bytes())
    entries=[]
    for name,policy,exe,extra in (('reference-native','native',ROOT/'build/tcm-reference',[]),('native','native',binary,[]),('full-byte','full-byte',binary,[]),('bounded','bounded',binary,[]),('bounded-fp32','bounded',binary,['--weight-profile','tiled-fp32'])):
        if name=='bounded-fp32':exe=target/'build/tcm-bounded-chat'
        policy_args=[] if name=='bounded-fp32' else ['--cache-policy',policy]
        entries.append({'name':name,'policy':policy,'command':[str(exe),'serve',model['path'],'--context','768',*policy_args,*extra]})
    mechanics=[{'name':'native','command':[str(ROOT/'build/stack-mechanics'),model['path']]},{'name':'bounded','command':[str(ROOT/'build/bounded-gpu-mechanics'),model['path']]}]
    demo=['/save','Explain why bread rises when yeast is added. Use two short sentences.','/state','/push','Compute 17 + 29. Reply with only the integer.','/state','/rollback','/push',"Remember: Eno's number is 17. Reply only OK.",'/push','/push',"What is Eno's number? Reply with only the number.",'/state','/rollback',"What is Eno's number? Reply with only the number.",'/state','/commit','/state','/quit']
    q=origin['inputs'];delayed=[q[0]['text'],'/push',q[3]['text'],'/rollback',q[1]['text'],q[2]['text'],'/state']
    for i in range(4,8):delayed+=['/push',q[i]['text'],'/state','/rollback']
    delayed+=['/quit'];clients=[]
    for policy in ('full-byte','bounded'):
        for label,lines in (('demo',demo),('delayed',delayed)):
            clients.append({'name':policy+'-'+label,'policy':policy,'stdin':'\n'.join(lines)+'\n','command':[sys.executable,str(client),'--model',model['path'],'--tokenizer',str(tokenizer),'--context','768','--new-tokens','32','--cache-policy',policy]})
    clients.append({'name':'native-demo','policy':'native','stdin':'Explain why bread rises when yeast is added. Use two short sentences.\n/state\n/quit\n','command':[sys.executable,str(client),'--model',model['path'],'--tokenizer',str(tokenizer),'--context','768','--new-tokens','32']})
    commands=[x['command'] for x in entries+mechanics+clients]
    request={'git_commit':manifest['git_commit'],'git_tree':manifest['git_tree'],'native_reference_commit':manifest['native_reference_commit'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'binary_sha256':sha(binary),'reference_binary_sha256':sha(ROOT/'build/tcm-reference'),'model_digest':model['content_sha256'],'execution_commands':commands,'inputs':origin['inputs'],'client_transcripts':[{k:v for k,v in x.items() if k!='command'} for x in clients],'scope':'twelve sequential process invocations; committed unified native/full-byte/bounded profiles'}
    (RAW/'request.json').write_text(json.dumps(request,sort_keys=True)+'\n');report={'schema':'cmachine-commit-integration-v1','git_commit':manifest['git_commit'],'git_tree':manifest['git_tree'],'model':model,'environment':{'platform':platform.platform(),'job_id':os.environ.get('MRUN_JOB_ID')},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'historical-behavior-preserved','terminal_status':'bounded-engineering-parity'};preserve(report)
    def controller(_):
        protocols=[];checks=[];texts=[]
        for x in entries:
            print('[protocol] '+x['name'],flush=True);r=native_case(x['command'],RAW,x['name'],origin) if x['policy']=='native' else protocol_case(x['command'],RAW,x['policy'],origin,x['name']);protocols.append({**x,'result':r});report['protocol_partial']=protocols;preserve(report)
        for x in mechanics:
            print('[mechanics] '+x['name'],flush=True);r=command(x['command']);(RAW/(x['name']+'-mechanics.stdout.txt')).write_text(r.stdout);(RAW/(x['name']+'-mechanics.stderr.txt')).write_text(r.stderr);checks.append({**x,'result':json.loads(r.stdout)})
        for x in clients:
            print('[client] '+x['name'],flush=True);(RAW/(x['name']+'.stdin.txt')).write_text(x['stdin']);r=subprocess.run(x['command'],input=x['stdin'],text=True,capture_output=True,timeout=180);(RAW/(x['name']+'.stdout.txt')).write_text(r.stdout);(RAW/(x['name']+'.stderr.txt')).write_text(r.stderr)
            if r.returncode:raise RuntimeError('Text client failed: '+r.stderr[-4000:])
            texts.append({k:v for k,v in x.items() if k!='stdin'}|{'returncode':r.returncode,'stdin_sha256':sha(RAW/(x['name']+'.stdin.txt')),'stdout_sha256':sha(RAW/(x['name']+'.stdout.txt'))});report['clients_partial']=texts;preserve(report)
        driver={'git_commit':manifest['git_commit'],'binary_sha256':sha(binary),'protocol':protocols,'mechanics':checks,'clients':texts,'installation':dict(installation)}
        analyze(RAW,driver,manifest,published=False);driver['installation']=publish(installation);return driver
    report['driver'],report['saturn_receipt']=invoke(commands,request,controller);(RAW/'driver.json').write_text(json.dumps(report['driver'],sort_keys=True)+'\n');(RAW/'installation.json').write_text(json.dumps(report['driver']['installation'],indent=2)+'\n');report['analysis']=analyze(RAW,report['driver'],manifest);report['raw_files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name not in ('partial-report.json','report.json')};report.update(mechanics_status='complete',instrument_status='complete');return report

def emit_files():
    files=[(p,'raw/'+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    manifest=json.loads((ROOT/'source-manifest.json').read_text());names=list(manifest['sources'])+['source-manifest.json']
    files += [(ROOT/n,'payload/'+n) for n in names]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('Evidence archive exceeds declared bound')
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as tar:
        for p,n in files:tar.add(p,arcname=n,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(out.getvalue()).decode(),flush=True)

if __name__=='__main__':
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:report=json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {};report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete');code=1
    report['worker_seconds']=time.monotonic()-started;data=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/'report.json').write_bytes(data+b'\n');print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True);emit_files();raise SystemExit(code)
