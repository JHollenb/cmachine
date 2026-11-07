"""Small process and byte-custody helpers; no model framework import."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'
LIMIT=24*1024**2
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args,timeout=300):
    args=list(map(str,args));print('[run] '+' '.join(args),flush=True)
    r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError('Process failed: '+r.stderr[-8000:])
    return r
def preserve(report):(RAW/'partial-report.json').write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n')
