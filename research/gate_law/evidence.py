"""Standard offline sweep, scoped only to this job's verified observation stream."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import types

HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]

def evidence(job):
    result = HERE / "runs" / job
    index = HERE / "index" / job
    target = index / "target-streams"
    target.mkdir(parents=True, exist_ok=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    verification = json.loads((result / "verification.json").read_text())
    if verification["report_sha256"] != sha(result / "report.json"):
        raise ValueError("Verification reference differs")
    shutil.copy2(result / "observations.jsonl", target / "observations.jsonl")
    (target / "source.json").write_text(json.dumps({"job": job, "report_sha256": sha(result / "report.json"), "verification_sha256": sha(result / "verification.json"), "observations_sha256": sha(target / "observations.jsonl"), "scope": "source-bound gate-law observations; standard stream coverage, no semantic certification"}, indent=2) + "\n")
    package = types.ModuleType("saturn")
    package.__path__ = [str(DOMAINS / "saturn/src/saturn")]
    sys.modules["saturn"] = package
    from saturn.evidence_pipeline import main
    for action in ("run", "verify"):
        args = [action, str(target), "--watch-root", str(index / "watch"), "--registry", str(index / "claims.jsonl"), "--results-root", str(target), "--json"]
        if action == "run":
            args += ["--xref-root", str(target), "--xref-db", str(index / "xref.sqlite")]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(args)
        (index / f"evidence-{action}.json").write_text(output.getvalue())
        summary = json.loads(output.getvalue())
        print(json.dumps({"job": job, "action": action, "exit_code": code, "steps": summary.get("steps")}))
        if code not in (0, 3):
            raise RuntimeError("Evidence instrument refused input")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("job")
    evidence(p.parse_args().job)
