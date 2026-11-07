"""Exact frozen-analyzer replay in its original CPU environment; no model execution."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def main():
    started=time.monotonic();request=json.loads(Path("request.json").read_text());root=Path(request["original_worker_root"]);raw=root/"results";manifest=json.loads((root/"source-manifest.json").read_text());report=json.loads((raw/"report.json").read_text())
    if sha(root/"source-manifest.json")!=request["source_manifest_sha256"] or sha(raw/"report.json")!=request["raw_report_sha256"]:raise ValueError("Original closure identity differs")
    for name,digest in manifest["sources"].items():
        if sha(root/name)!=digest:raise ValueError("Original source differs")
    for name,record in report["raw_files"].items():
        if sha(raw/name)!=record["sha256"] or (raw/name).stat().st_size!=record["bytes"]:raise ValueError("Original raw differs")
    if hashlib.sha256(canonical(report["analysis"])).hexdigest()!=request["analysis_sha256"]:raise ValueError("Original analysis identity differs")
    sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location("original_executed_analysis",root/"analysis.py");module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calculated=module.analyze(raw,root,report["driver"],report["texts"]);digest=hashlib.sha256(canonical(calculated)).hexdigest()
    if digest!=request["analysis_sha256"]:raise ValueError("Zero-tolerance same-environment analysis differs")
    import numpy as np
    result={"schema":"tcm-same-environment-analysis-replay-v1","audit_job_id":os.environ.get("MRUN_JOB_ID"),"request_sha256":sha("request.json"),"audit_worker_sha256":sha(__file__),"original_job_id":request["original_job_id"],"original_source_manifest_sha256":request["source_manifest_sha256"],"original_raw_report_sha256":request["raw_report_sha256"],"analysis_sha256":digest,"analysis_canonical_bytes_exact":True,"source_files_verified":len(manifest["sources"]),"raw_files_verified":len(report["raw_files"]),"models_evaluated":0,"numpy_version":np.__version__,"python":sys.version,"seconds":time.monotonic()-started,"terminal_status":"not-requested"}
    print("AUDIT_RESULT="+json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
if __name__=="__main__":main()
