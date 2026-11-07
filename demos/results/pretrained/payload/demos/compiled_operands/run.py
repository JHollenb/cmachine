#!/usr/bin/env python3
"""Build/run the public lexical API demo. CUDA execution belongs in mrun in domains."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def build():
    subprocess.run(["make", "cuda"], cwd=ROOT, check=True)
    obj = ROOT / "build" / "demo-compiled-operands.o"
    binary = ROOT / "build" / "demo-compiled-operands"
    subprocess.run([os.environ.get("CC", "cc"), "-O3", "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-ffp-contract=off", "-I", str(ROOT / "src"), "-c", str(HERE / "driver.c"), "-o", str(obj)], check=True)
    nvcc = os.environ.get("NVCC", str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"))
    subprocess.run([nvcc, str(ROOT / "build/model.o"), str(ROOT / "build/qwen.o"), str(ROOT / "build/qwen_cuda.o"), str(obj), "-lcublas", "-lm", "-o", str(binary)], check=True)
    return binary


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--tokens-file", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--binary", type=Path, help="Use an already built canonical-linked demo")
    p.add_argument("--build-only", action="store_true")
    a = p.parse_args()
    binary = a.binary.resolve() if a.binary else build()
    if a.build_only:
        print(binary)
        return
    if not (a.model and a.tokens_file and a.output):
        p.error("--model, --tokens-file and --output are required to run")
    ids = [int(x) for x in a.tokens_file.read_text().split()]
    if not ids or a.steps < 1 or len(ids) + a.steps > 64:
        p.error("nonempty prompt + 1..64 generated steps must fit 64 total advances")
    a.output.mkdir(parents=True, exist_ok=False)
    invocation = {"schema": "compiled-operands-invocation-v1", "driver_sha256": sha(HERE / "driver.c"), "binary_sha256": sha(binary), "tokens_sha256": sha(a.tokens_file), "steps": a.steps, "prompt_tokens": len(ids)}
    (a.output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    result = subprocess.run([str(binary), str(a.model.resolve()), str(a.tokens_file.resolve()), str(a.output.resolve()), str(a.steps)], capture_output=True, text=True)
    (a.output / "stdout.txt").write_text(result.stdout)
    (a.output / "stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((a.output / "report.json").read_text())
    print(json.dumps({k: report[k] for k in ("source_steps", "candidate_steps", "all_steps_exact", "missing_refusal_no_mutation", "covered_ids", "compile_linear_macs", "table_host_bytes", "candidate_transfer_bytes", "candidate_original_first_layer_qkv_macs_omitted")}, indent=2))


if __name__ == "__main__":
    main()
