"""Record the model-free lane against exact source bytes before committing."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,tempfile
ROOT=Path(__file__).resolve().parents[1]
def main():
    commands=[[sys.executable,'-m','unittest','discover','-s','tests','-v'],['make','sanitize','state-mechanics','bounded-mechanics'],['cc','-std=c11','-Wall','-Wextra','-Wpedantic','-Werror','-Isrc','-fsyntax-only','src/qwen_bounded_serve.c','src/qwen_serve.c']]
    rows=[]
    for args in commands:
        t=time.monotonic();r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);rows.append({'command':args,'returncode':r.returncode,'seconds':time.monotonic()-t,'stdout':r.stdout,'stderr':r.stderr});print(json.dumps({'command':args,'returncode':r.returncode}),flush=True)
        if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
    reference_commit=subprocess.check_output(['git','rev-parse','b81e928'],cwd=ROOT).decode().strip();headers={}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ('qwen.h','gpu.h','machine.h'):
            data=subprocess.check_output(['git','show',reference_commit+':src/'+name],cwd=ROOT);(Path(tmp)/name).write_bytes(data);headers[name]=hashlib.sha256(data).hexdigest()
        args=['cc','-std=c11','-Wall','-Wextra','-Wpedantic','-Werror','-I'+tmp,'-include',str(Path(tmp)/'qwen.h'),'-include','validation/baseline_serve_stub.h','-fsyntax-only','src/qwen_serve.c'];r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);rows.append({'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(json.dumps({'reference_abi_syntax':True,'returncode':r.returncode}),flush=True)
        if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
    paths=[ROOT/'Makefile']
    for directory in ('src','tools','tests'):
        paths.extend(p for p in (ROOT/directory).iterdir() if p.is_file())
    paths.extend(p for p in (ROOT/'validation').iterdir() if p.is_file())
    paths.extend(p for p in (ROOT/'validation/vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    out={'schema':'cmachine-local-checks-v1','models_loaded':0,'source_sha256':sources,'native_reference_commit':reference_commit,'reference_header_sha256':headers,'checks':rows,'scope':'model-free mechanics and C syntax; CUDA/numerical preservation requires the subsequent commit-bound fleet run'}
    path=ROOT/'docs/evidence/local-checks.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'receipt':str(path),'source_files':len(sources),'all_checks_passed':True}))
if __name__=='__main__':main()
