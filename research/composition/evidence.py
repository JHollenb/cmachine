"""Narrow evidence-plane projection; exclude copied historical package fixtures."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]
for job in ("job-86cce06d33b1", "job-9337489b61af"):
    root = HERE/"runs"/job
    index = HERE/"index"/job
    target = index/"target-streams"
    target.mkdir(parents=True, exist_ok=True)
    report_sha = hashlib.sha256((root/"report.json").read_bytes()).hexdigest()
    verification = json.loads((root/"verification.json").read_text())
    if report_sha != verification["report_sha256"]:raise ValueError("Verification reference differs")
    shutil.copy2(root/"observations.jsonl", target/"observations.jsonl")
    source = {"job": job, "report_sha256": report_sha,
              "verification_sha256": hashlib.sha256((root/"verification.json").read_bytes()).hexdigest(),
              "observations_sha256": hashlib.sha256((target/"observations.jsonl").read_bytes()).hexdigest(),
              "scope": "source-bound saved composition observations; target-stream coverage; no automatic semantic claims or internal C address admission"}
    (target/"source.json").write_text(json.dumps(source, indent=2)+"\n")
    for action in ("run", "verify"):
        args = [str(DOMAINS/"common/bin/common"), "exec", "saturn-evidence", action, str(target),
                "--watch-root", str(index/"watch"), "--registry", str(index/"claims.jsonl"), "--results-root", str(target), "--json"]
        if action == "run":args += ["--xref-db", str(index/"xref.sqlite"), "--xref-root", str(target)]
        result = subprocess.run(args, capture_output=True, text=True)
        (index/f"evidence-{action}.json").write_text(result.stdout)
        (index/f"evidence-{action}.stderr.txt").write_text(result.stderr)
        try:
            summary = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(result.stderr[-2000:])
        print(json.dumps({"job": job, "action": action, "returncode": result.returncode, "steps": summary.get("steps")}))
        if result.returncode not in (0, 3):raise RuntimeError("Evidence instrument refused input")
