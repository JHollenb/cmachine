"""Stage and submit the analysis-overhead timing job (one mrun job, Beast).

  MRUN_URL=http://127.0.0.1:19025 PYTHONPATH=<mrun-quant-gpu>/src <mrun venv python> launch.py stage NAME
  ... launch.py submit NAME   (prints the job id; do not edit payload-NAME before that)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
DOMAINS = Path("/Users/jakeholl/domains")
MRUN_SRC = DOMAINS / ".worktrees/mrun-quant-gpu/src"
MDB_SRC = DOMAINS / "saturn-pub/packages/mdb/src/mdb"
SATURN_VALUES = DOMAINS / "saturn-pub/src/saturn_pub/values.py"


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def stage(name):
    payload = HERE / f"payload-{name}"
    if payload.exists():
        raise ValueError("payload exists")
    cand = payload / "candidate"
    for d in ("src", "tests", "tools"):
        shutil.copytree(WORKTREE / d, cand / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(WORKTREE / "Makefile", cand / "Makefile")
    for f in ("worker.py", "torch_arms.py", "run.sh"):
        shutil.copy2(HERE / f, payload / f)
    shutil.copy2(HERE / f"plan-{name}.json", payload / "plan.json")
    v = payload / "vendor"
    for pkg in ("mrun", "mdb", "saturn_pub"):
        (v / pkg).mkdir(parents=True)
        (v / pkg / "__init__.py").write_text("")
    shutil.copy2(MRUN_SRC / "mrun/outbox.py", v / "mrun/outbox.py")
    for f in ("compiled_state.py", "compiled_state_runner.py"):
        shutil.copy2(MDB_SRC / f, v / "mdb" / f)
    shutil.copy2(SATURN_VALUES, v / "saturn_pub/values.py")
    prov = {"cmachine_branch": git(WORKTREE, "branch", "--show-current"), "cmachine_head": git(WORKTREE, "rev-parse", "HEAD"),
            "cmachine_worktree_status": git(WORKTREE, "status", "--short"),
            "cmachine_diff_sha256": hashlib.sha256(git(WORKTREE, "diff", "HEAD").encode()).hexdigest(),
            "mrun_main_head": git(DOMAINS / "mrun", "rev-parse", "HEAD"), "mrun_client_head": git(DOMAINS / ".worktrees/mrun-quant-gpu", "rev-parse", "HEAD"),
            "mdb_head": git(DOMAINS / "saturn-pub", "rev-parse", "HEAD"), "vendored": ["mrun/outbox.py", "mdb/compiled_state.py", "mdb/compiled_state_runner.py", "saturn_pub/values.py"]}
    (payload / "sources.json").write_text(json.dumps({"provenance": prov}, indent=1) + "\n")
    sources = {p.relative_to(payload).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(payload.rglob("*")) if p.is_file()}
    manifest = {"provenance": prov, "sources": sources}
    (payload / "sources.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (HERE / f"sources-{name}.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"payload": str(payload), "files": len(sources), "head": prov["cmachine_head"], "worker_sha256": sources["worker.py"]}))


def submit(name, vram_mb, ram_mb, timeout_s):
    from mrun.client.api import Api
    from mrun.client.submit import submit as mrun_submit
    payload = HERE / f"payload-{name}"
    manifest = json.loads((HERE / f"sources-{name}.json").read_text())
    for rel, digest in manifest["sources"].items():
        if rel == "sources.json":
            continue
        if hashlib.sha256((payload / rel).read_bytes()).hexdigest() != digest:
            raise ValueError(f"payload changed after staging: {rel}")
    prov = manifest["provenance"]
    req = dict(experiment=f"cmachine-overhead-{name}", cmd=["bash", "run.sh", name], payload=payload, needs={"cuda": True}, host="beast",
               reservation={"ram_mb": ram_mb, "vram_mb": vram_mb, "cpu_threads": 4}, timeout_s=timeout_s, priority=70,
               model="qwen2.5-1.5b-instruct",
               config={"task_family": "c-executor-analysis-overhead", "backend": "c-cublas-fp32 + hf-pytorch-fp32-sdpa", "device": "cuda", "dtype": "fp32",
                       "cmachine_head": prov["cmachine_head"], "cmachine_branch": prov["cmachine_branch"], "mrun_main_head": prov["mrun_main_head"],
                       "mrun_client_head": prov["mrun_client_head"], "worker_sha256": manifest["sources"]["worker.py"], "gpu": "NVIDIA GeForce RTX 4080, driver 580.159.03 (Beast, queried 2026-11-13)"},
               note="cmachine analysis-overhead timing; stop after arms C0-C4,T0-T3 complete")
    job = mrun_submit(**req, retry_on_kill=False, api=Api(timeout_s=60), preflight=True)
    (HERE / f"submission-{name}.json").write_text(json.dumps({"job": job, "reservation": req["reservation"]}, indent=1, default=str) + "\n")
    print(job)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("stage", "submit"))
    p.add_argument("name")
    p.add_argument("--vram-mb", type=int, default=8300)
    p.add_argument("--ram-mb", type=int, default=8000)
    p.add_argument("--timeout-s", type=int, default=900)
    a = p.parse_args()
    stage(a.name) if a.action == "stage" else submit(a.name, a.vram_mb, a.ram_mb, a.timeout_s)
