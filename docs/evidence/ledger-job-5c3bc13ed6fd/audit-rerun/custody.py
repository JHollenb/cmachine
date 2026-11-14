"""Custody check: manifest sha256 vs staged payload and vs current worktree files."""
import hashlib, json, sys
from pathlib import Path
WT = Path(__file__).resolve().parents[4]
LED = WT / "research/ledger"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for name in sys.argv[1:] or ["full1"]:
    man = json.loads((LED / f"sources-{name}.json").read_text())
    pay = LED / f"payload-{name}"
    bad_pay, bad_tree, missing_tree = [], [], []
    for rel, dg in man["sources"].items():
        if rel == "sources.json": continue
        p = pay / rel
        if not p.exists() or sha(p) != dg: bad_pay.append(rel)
        if rel.startswith("candidate/"):
            t = WT / rel[len("candidate/"):]
        elif rel in ("worker.py", "run.sh"):
            t = LED / rel
        elif rel == "plan.json":
            t = LED / f"plan-{name}.json"
        else:
            continue
        if not t.exists(): missing_tree.append(rel)
        elif sha(t) != dg: bad_tree.append(rel)
    # files in tree not in manifest (src/tests/tools + Makefile)
    extra = []
    for d in ("src", "tests", "tools"):
        for p in sorted((WT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                rel = "candidate/" + p.relative_to(WT).as_posix()
                if rel not in man["sources"]: extra.append(rel)
    print(json.dumps({"plan": name, "n_manifest": len(man["sources"]), "payload_mismatch": bad_pay,
                      "tree_differs_from_manifest": bad_tree, "manifest_file_missing_in_tree": missing_tree,
                      "tree_files_not_in_manifest": extra, "worker_sha256": man["sources"].get("worker.py"),
                      "provenance_head": man["provenance"]["cmachine_head"]}, indent=1))
