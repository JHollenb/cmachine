"""One finite native-primitives lease, no Qwen forward or trained candidate."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'results'
LIMIT = 32*1024**2
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(args):
    print('[run]', ' '.join(map(str,args)),flush=True)
    p = subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(f'Process {p.returncode}: {p.stderr[-10000:]}')
    return p

def experiment():
    from analysis import analyze
    sys.path.insert(0,str(ROOT/'public'))
    from saturn_bridge import invoke
    manifest = json.loads((ROOT/'source-manifest.json').read_text())
    for name,digest in manifest['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source differs: '+name)
    model = json.loads((ROOT/'baseline.json').read_text())['model']
    report = {'schema':'tcm-state-quotient-experiment-v1','model':model,'texts':{},'environment':{'platform':platform.platform(),'python':sys.version,'job_id':os.environ.get('MRUN_JOB_ID')},'mechanics_status':'in-progress','instrument_status':'in-progress','trend_status':'observations','terminal_status':'not-requested'}
    (RAW/'partial-report.json').write_text(json.dumps(report)+'\n')
    command(['make'])
    report['binary_sha256'] = sha(ROOT/'build/quotient')
    args = [ROOT/'build/quotient',model['path'],RAW]
    def controller(_):
        p = command(args)
        (RAW/'driver.stdout.txt').write_text(p.stdout)
        (RAW/'driver.stderr.txt').write_text(p.stderr)
        print(p.stderr,end='',flush=True)
        return json.loads((RAW/'driver.json').read_text())
    report['driver'],report['saturn_receipt'] = invoke(args,None,{'model_digest':model['content_sha256'],'binary_sha256':report['binary_sha256'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),'scope':'supplied nonlinear organism,8 own-state paths16 steps; exact original down minor; zero Qwen forwards'},controller=controller)
    minor = report['driver']['qwen_minor']
    if minor['model_digest']!=model['content_sha256'] or minor['model_identity']!=model['identity']:
        raise ValueError('Executed model identity differs')
    report['analysis'] = analyze(RAW,ROOT/'bank',report['driver'],{})
    report['raw_files'] = {p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(RAW.iterdir()) if p.is_file() and p.name!='partial-report.json'}
    report.update(mechanics_status='complete',instrument_status='complete')
    return report

def emit_files():
    files = []
    for directory,prefix in ((RAW,'raw'),(ROOT/'runtime','payload/runtime'),(ROOT/'public','payload/public')):
        files += [(p,prefix+'/'+str(p.relative_to(directory))) for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    files += [(p,'payload/'+p.name) for p in sorted(ROOT.iterdir()) if p.is_file()]
    if sum(p.stat().st_size for p,_ in files)>LIMIT:raise ValueError('Evidence archive bound exceeded')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path,name in files:archive.add(path,arcname=name,recursive=False)
    print('COMPOSITION_FILES_TGZ_B64='+base64.b64encode(buffer.getvalue()).decode(),flush=True)

if __name__=='__main__':
    RAW.mkdir(exist_ok=True)
    start = time.monotonic()
    try:
        report = experiment();code = 0
    except Exception:
        report = json.loads((RAW/'partial-report.json').read_text()) if (RAW/'partial-report.json').exists() else {}
        report.update(error=traceback.format_exc(),mechanics_status='incomplete',instrument_status='incomplete');code = 1
    report['worker_seconds'] = time.monotonic()-start
    data = json.dumps(report,sort_keys=True,allow_nan=False).encode()
    (RAW/'report.json').write_bytes(data+b'\n')
    print('COMPOSITION_RESULT_GZIP_B64='+base64.b64encode(gzip.compress(data,mtime=0)).decode(),flush=True)
    emit_files()
    raise SystemExit(code)
