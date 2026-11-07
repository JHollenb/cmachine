"""A short tour of executable C model state, plus authenticated recorded results."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(command, cwd=ROOT):
    result = subprocess.run(list(map(str, command)), cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        raise RuntimeError(result.stderr or "Command failed: " + str(command))
    return result


def mechanics(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("state", "bounded", "operand"):
        started = time.monotonic()
        result = run(["make", name + "-mechanics"])
        values = []
        for line in result.stdout.splitlines():
            if line.startswith("{"):
                values.append(json.loads(line))
        if not values:
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("Mechanics fixture emitted no result: " + name)
            values = [{"format": "fixture-text", "message": lines[-1]}]
        binary = ROOT / "build" / (name + "-mechanics")
        rows.append({"name": name, "returncode": result.returncode,
                     "elapsed_seconds_including_build": time.monotonic() - started,
                     "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                     "results": values})
    report = {"schema": "cmachine-demo-mechanics-v1", "execution_kind": "fresh-model-free-C-fixtures",
              "pretrained_models_evaluated": 0, "fixtures": rows,
              "scope": "Canonical C state/operand/retention code with CPU GPU stubs under sanitizers; not pretrained inference or CUDA parity."}
    path = output / "mechanics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", nargs="?", default="tour", choices=("tour", "recorded", "mechanics", "closed-state"))
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.demo in ("tour", "recorded"):
        print("\n1. Inspect recorded pretrained execution")
        run([sys.executable, HERE / "recorded/build.py", "--output", output / "recorded.json"])
        public_result = HERE / "results/pretrained"
        if (public_result / "public-manifest.json").exists():
            run([sys.executable, HERE / "verify.py", "--public", public_result])
    if args.demo in ("tour", "mechanics"):
        print("\n2. Run the canonical C state, memory and operand fixtures")
        mechanics(output)
    if args.demo in ("tour", "closed-state"):
        print("\n3. Advance a small nonlinear state, break its interface, repair it")
        run([sys.executable, HERE / "closed_state/run.py", "--output", output / "closed-state.json"])
    print("\nFor pretrained runs: see demos/chat_branch/ and demos/compiled_operands/.")
    print("Open demos/showcase.html for the interactive visual tour.")


if __name__ == "__main__":
    main()
