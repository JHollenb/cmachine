"""Finite guarded CUDA execution of the public demo programs; no installation."""
from pathlib import Path
import base64
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path.cwd()
RAW = ROOT / "demo-results"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(name, argv):
    print("[demo] " + name, flush=True)
    started = time.monotonic()
    process = subprocess.run(list(map(str, argv)), cwd=ROOT, text=True, capture_output=True, timeout=300)
    (RAW / (name + ".stdout.txt")).write_text(process.stdout)
    (RAW / (name + ".stderr.txt")).write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(name + " failed: " + process.stderr[-5000:])
    return {"name": name, "seconds_including_instrumentation": time.monotonic() - started, "returncode": 0}


def experiment():
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, row in manifest["sources"].items():
        if sha(ROOT / name) != row["sha256"]:
            raise ValueError("Source changed: " + name)
    tokenizer_path = Path(manifest["tokenizer_path"])
    for name, expected in manifest["model"]["tokenizer_files"].items():
        if sha(tokenizer_path / name) != expected:
            raise ValueError("Tokenizer differs: " + name)
    model = manifest["model"]["path"]
    commands = [command("build", [sys.executable, "demos/compiled_operands/run.py", "--build-only"])]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    text = "Write one short sentence about bread."
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False)
    if not ids or len(ids) + 8 > 64:
        raise ValueError("Compiled demo prompt exceeds finite geometry")
    (RAW / "operand-prompt.ids").write_text(" ".join(map(str, ids)) + "\n")
    (RAW / "operand-prompt.json").write_text(json.dumps({"text": text, "ids": ids}) + "\n")
    sys.path.insert(0, str(ROOT / "validation/vendor"))
    sys.path.insert(0, str(ROOT / "validation"))
    from saturn_bridge import invoke
    request = {"source_manifest_sha256": sha(ROOT / "source-manifest.json"),
               "model_digest": manifest["model"]["content_sha256"],
               "binary_sha256": sha(ROOT / "build/tcm-cuda"),
               "operand_binary_sha256": sha(ROOT / "build/demo-compiled-operands"),
               "scope": "Two sequential public demo invocations; C/driver checks qualify internal state."}

    def controller(_):
        commands.append(command("chat", [sys.executable, "demos/chat_branch/chat_branch.py", "--binary", ROOT / "build/tcm-cuda",
                                          "--model", model, "--tokenizer", tokenizer_path, "--output", RAW / "chat/report.json",
                                          "--context", "768", "--new-tokens", "16"]))
        commands.append(command("operands", [sys.executable, "demos/compiled_operands/run.py", "--binary", ROOT / "build/demo-compiled-operands",
                                              "--model", model, "--tokens-file", RAW / "operand-prompt.ids", "--output", RAW / "operands", "--steps", "8"]))
        chat = json.loads((RAW / "chat/report.json").read_text())
        operands = json.loads((RAW / "operands/report.json").read_text())
        if (chat["server"]["model_digest"] != manifest["model"]["content_sha256"]
                or chat["server"]["model_identity"] != manifest["model"]["identity"]):
            raise ValueError("Executed chat model identity differs")
        if operands["model_digest"] != manifest["model"]["content_sha256"]:
            raise ValueError("Executed operand model identity differs")
        return {"chat": chat, "operands": operands}

    result, receipt = invoke([], None, request, controller=controller)
    return {"schema": "cmachine-fresh-demos-v1", "execution_kind": "fresh-pretrained-C-CUDA",
            "job_id": os.environ.get("MRUN_JOB_ID"), "base_commit": manifest["base_commit"],
            "source_manifest_sha256": sha(ROOT / "source-manifest.json"), "request": request,
            "environment": {"platform": platform.platform(), "arithmetic": "FP32/CUDA/cuBLAS pedantic; sm89; contraction disabled"},
            "commands": commands, "results": result, "saturn_receipt": receipt,
            "scope": "Native branch/replay and source-covered lexical reuse; original model work retained. No speedup or generalization claim."}


if __name__ == "__main__":
    RAW.mkdir()
    started = time.monotonic()
    try:
        report = experiment()
        code = 0
    except Exception:
        report = {"schema": "cmachine-fresh-demos-v1", "error": traceback.format_exc(), "status": "execution-incomplete"}
        code = 1
    report["worker_seconds"] = time.monotonic() - started
    report["raw_files"] = {str(p.relative_to(RAW)): {"sha256": sha(p), "bytes": p.stat().st_size}
                           for p in RAW.rglob("*") if p.is_file() and p.suffix != ".bin"}
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    (RAW / "report.json").write_bytes(raw + b"\n")
    print("CMACHINE_DEMO_REPORT=" + base64.b64encode(gzip.compress(raw, mtime=0)).decode(), flush=True)
    files = [(p, "raw/" + str(p.relative_to(RAW))) for p in RAW.rglob("*") if p.is_file() and p.suffix != ".bin"]
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    files += [(ROOT / name, "payload/" + name) for name in [*manifest["sources"], "source-manifest.json"]]
    if sum(p.stat().st_size for p, _ in files) > manifest["archive_limit_bytes"]:
        raise ValueError("Demo evidence exceeds declared bound")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path, name in files:
            tar.add(path, arcname=name, recursive=False)
    print("CMACHINE_DEMO_FILES=" + base64.b64encode(archive.getvalue()).decode(), flush=True)
    raise SystemExit(code)
