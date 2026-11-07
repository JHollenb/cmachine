"""Run the standard evidence plane over only this experiment's verified rows."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
HERE=Path(__file__).resolve().parent
DOMAINS=HERE.parents[3]

def evidence(job,here=HERE):
    here=Path(here)
    root=here/"runs"/job;index=here/"index"/job;target=index/"target-streams";target.mkdir(parents=True,exist_ok=True)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    verification=json.loads((root/"verification.json").read_text())
    if verification["report_sha256"]!=sha(root/"report.json"):raise ValueError("Verification reference differs")
    shutil.copy2(root/"observations.jsonl",target/"observations.jsonl")
    (target/"source.json").write_text(json.dumps({"job":job,"report_sha256":sha(root/"report.json"),"verification_sha256":sha(root/"verification.json"),"observations_sha256":sha(target/"observations.jsonl"),"scope":"whole-turn source-bound observations; stream coverage only; no automatic semantic admission"},indent=2)+"\n")
    for action in ("run","verify"):
        args=[str(DOMAINS/"common/bin/common"),"exec","saturn-evidence",action,str(target),"--watch-root",str(index/"watch"),"--registry",str(index/"claims.jsonl"),"--results-root",str(target),"--json"]
        if action=="run":args.extend(("--xref-db",str(index/"xref.sqlite"),"--xref-root",str(target)))
        result=subprocess.run(args,capture_output=True,text=True);(index/f"evidence-{action}.json").write_text(result.stdout);(index/f"evidence-{action}.stderr.txt").write_text(result.stderr)
        summary=json.loads(result.stdout);print(json.dumps({"job":job,"action":action,"returncode":result.returncode,"steps":summary.get("steps")}))
        if result.returncode not in (0,3):raise RuntimeError("Evidence instrument refused input")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("job");a=p.parse_args();evidence(a.job)
