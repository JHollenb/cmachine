"""Verify native execution custody and exact same-environment analysis replay.

The original local strict refusal remains in verification-refusal-diagnostic.json. This
does not claim cross-platform arithmetic equality or loosen a tolerance.
"""
from pathlib import Path
import argparse,base64,gzip,hashlib,json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def verify(root):
    root=Path(root);report=json.loads((root/"report.json").read_text());job=json.loads((root/"job.json").read_text());sub=json.loads((root/"submission.json").read_text());replay=json.loads((root/"audit-replay.json").read_text());audit_job=json.loads((root/"audit-replay-job.json").read_text());audit_sub=json.loads((root/"audit-replay-submission.json").read_text())
    if job["job_id"]!=sub["job_id"] or job["state"]!="succeeded" or sha(root/"logs.txt")!=job["result"]["artifacts"]["job_log"]["sha256"]:raise ValueError("Original scheduler/log differs")
    logged=[json.loads(gzip.decompress(base64.b64decode(line.split("=",1)[1]))) for line in (root/"logs.txt").read_text().splitlines() if line.startswith("COMPOSITION_RESULT_GZIP_B64=")]
    if logged!=[report] or json.loads((root/"raw/report.json").read_text())!=report or json.loads((root/"raw/driver.json").read_text())!=report["driver"]:raise ValueError("Original report joins differ")
    for k in ("sources","source_manifest_sha256"):
        if job["config"][k]!=sub["config"][k]:raise ValueError("Original scheduler request differs")
    for name,digest in sub["config"]["sources"].items():
        if sha(root/"payload"/name)!=digest:raise ValueError("Original source differs")
    if sha(root/"payload/source-manifest.json")!=sub["config"]["source_manifest_sha256"]:raise ValueError("Original source manifest differs")
    for name,r in report["raw_files"].items():
        p=root/"raw"/name
        if sha(p)!=r["sha256"] or p.stat().st_size!=r["bytes"]:raise ValueError("Original raw differs")
    request=json.loads((root/"audit-replay-payload/request.json").read_text());log_sha=sha(root/"audit-replay-logs.txt")
    if audit_job["state"]!="succeeded" or audit_job["job_id"]!=audit_sub["job_id"] or log_sha!=audit_job["result"]["artifacts"]["job_log"]["sha256"] or audit_job["config"]["sources"]!=audit_sub["config"]["sources"]:raise ValueError("Audit scheduler seal differs")
    for name,digest in audit_sub["config"]["sources"].items():
        if sha(root/"audit-replay-payload"/name)!=digest:raise ValueError("Audit source differs")
    result=[json.loads(s.split("=",1)[1]) for s in (root/"audit-replay-logs.txt").read_text().splitlines() if s.startswith("AUDIT_RESULT=")]
    if result!=[replay] or replay["audit_job_id"]!=audit_job["job_id"] or replay["original_job_id"]!=job["job_id"] or replay["request_sha256"]!=sha(root/"audit-replay-payload/request.json") or replay["audit_worker_sha256"]!=sha(root/"audit-replay-payload/audit_worker.py"):raise ValueError("Audit result join differs")
    expected=hashlib.sha256(canonical(report["analysis"])).hexdigest()
    if replay["analysis_sha256"]!=expected or request["analysis_sha256"]!=expected or not replay["analysis_canonical_bytes_exact"] or replay["models_evaluated"] or replay["original_source_manifest_sha256"]!=sha(root/"payload/source-manifest.json") or replay["original_raw_report_sha256"]!=sha(root/"raw/report.json") or replay["source_files_verified"]!=len(sub["config"]["sources"]) or replay["raw_files_verified"]!=len(report["raw_files"]):raise ValueError("Exact frozen analysis replay differs")
    wrapper=report["saturn_receipt"];receipt=dict(wrapper["receipt"]);fingerprint=receipt.pop("fingerprint")
    if hashlib.sha256(canonical(receipt)).hexdigest()!=fingerprint or receipt["parent"]!=wrapper["before"] or receipt["result"]!=wrapper["after"]:raise ValueError("Saturn receipt differs")
    # The original basis/runtime binds to the authenticated prior query study.
    import importlib.util
    spec=importlib.util.spec_from_file_location("query_program_launcher",Path(__file__).parent/"run.py");launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    if sha(launcher.PREVIOUS/"report.json")!=launcher.EXPECTED or report["program_provenance"]["source_report_sha256"]!=launcher.EXPECTED:raise ValueError("Prior source identity differs")
    prior=json.loads((launcher.PREVIOUS/"submission.json").read_text())
    for name,digest in prior["config"]["sources"].items():
        if name.startswith("runtime/") and sub["config"]["sources"].get(name)!=digest:raise ValueError("Pinned runtime differs")
    verification={"schema":"tcm-query-program-verification-v2","report_sha256":sha(root/"report.json"),"scheduler_log_seal":True,"source_files_verified":len(sub["config"]["sources"]),"raw_files_verified":len(report["raw_files"]),"whole_invocation_receipt":True,"analysis_recomputed":True,"analysis_recomputation_scope":"zero-tolerance canonical bytes in original Linux/NumPy environment; cross-platform strict refusal retained","analysis_replay_job":audit_job["job_id"],"analysis_replay_sha256":sha(root/"audit-replay.json"),"local_refusal_sha256":sha(root/"verification-refusal-diagnostic.json"),"numerical_tolerance_relaxed":False,"models_evaluated":0,"terminal_status":"not-requested"}
    (root/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    observations=[{**row,"schema":"tcm-composition-observation-v1","model_id":report["model"]["identity"],"report_sha256":verification["report_sha256"]} for row in report["analysis"]["observations"]];(root/"observations.jsonl").write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in observations))
    (root/"verification-source.py").write_bytes(Path(__file__).read_bytes());print(json.dumps(verification));return verification
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();verify(a.root)
