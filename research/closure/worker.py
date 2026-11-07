"""One resident native C closure assay with source-bound output transport."""
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

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results"
LIMIT = 32 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, timeout=300):
    print("[run] " + " ".join(map(str, args)), flush=True)
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"process failed {result.returncode}: {result.stderr[-8000:]}")
    return result


def experiment():
    from transformers import AutoTokenizer
    from analyze import analyze
    sys.path.insert(0, str(ROOT / "public"))
    from saturn_bridge import invoke
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"shipped identity differs: {name}")
    source = json.loads((ROOT / "source-reference.json").read_text())
    geometry = json.loads((ROOT / "geometry/geometry.json").read_text())
    model = source["model"]
    build = command(["make"])
    inputs = source["inputs"]
    (RAW / "source.txt").write_text(" ".join(map(str, inputs["source_ids"])) + "\n")
    (RAW / "source-own.txt").write_text(" ".join(map(str, source["driver"]["source_own"]["ids"])) + "\n")
    for row in inputs["questions"]:
        (RAW / f"{row['name']}.txt").write_text(" ".join(map(str, row["ids"])) + "\n")
    args = [ROOT / "build/closure", model["path"], RAW / "source.txt",
            *[RAW / f"{task}.txt" for task in ("recall", "math", "prose")],
            RAW, ROOT / "bank", RAW / "source-own.txt",
            source["driver"]["source_kv_sha256"], source["baseline"]["bank_concat_sha256"], ROOT / "geometry"]
    request = {"model_digest": model["content_sha256"], "binary_sha256": sha(ROOT / "build/closure"),
               "source_report_sha256": manifest["source_report_sha256"], "plan_sha256": manifest["sources"]["PLAN.md"],
               "geometry_sha256": manifest["sources"]["geometry/geometry.json"], "source_manifest_sha256": sha(ROOT / "source-manifest.json")}
    def controller(_):
        result = command(args)
        (RAW / "driver.stdout.txt").write_text(result.stdout)
        (RAW / "driver.stderr.txt").write_text(result.stderr)
        return json.loads((RAW / "driver.json").read_text())
    driver, receipt = invoke(args, None, request, controller=controller)
    tokenizer_path = Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name, expected in model["tokenizer_files"].items():
        if sha(tokenizer_path / name) != expected:
            raise ValueError("tokenizer identity differs")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    texts = {arm["name"]: tokenizer.decode(arm["own"]["ids"], skip_special_tokens=True)
             for case in driver["cases"] for arm in case["arms"]}
    analysis = analyze(RAW, ROOT / "bank", driver, texts, geometry)
    report = {"schema": "tcm-state-closure-experiment-v1", "model": model, "source_report_sha256": manifest["source_report_sha256"],
              "source_manifest_sha256": sha(ROOT / "source-manifest.json"), "driver": driver, "geometry": geometry,
              "texts": texts, "analysis": analysis, "saturn_receipt": receipt, "binary_sha256": request["binary_sha256"],
              "environment": {"platform": platform.platform(), "python": sys.version, "job_id": os.environ.get("MRUN_JOB_ID")},
              "build_stderr": build.stderr, "mechanics_status": "complete", "instrument_status": "executed",
              "trend_status": "bounded-observations", "terminal_status": "not-requested"}
    report["raw_files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                           for p in sorted(RAW.iterdir()) if p.is_file() and p.suffix in (".f32", ".txt", ".json")}
    return report


def emit_files():
    files = []
    for directory, prefix in ((RAW, "raw"), (ROOT / "runtime", "payload/runtime"),
                              (ROOT / "bank", "payload/bank"), (ROOT / "geometry", "payload/geometry"),
                              (ROOT / "public", "payload/public")):
        files += [(p, prefix + "/" + str(p.relative_to(directory))) for p in sorted(directory.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts]
    files += [(p, "payload/" + p.name) for p in sorted(ROOT.iterdir()) if p.is_file()]
    if sum(p.stat().st_size for p, _ in files) > LIMIT:
        raise ValueError("closure evidence exceeds declared32MiB transport")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print("CLOSURE_FILES_TGZ_B64=" + base64.b64encode(stream.getvalue()).decode(), flush=True)


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        result = experiment()
        result["worker_seconds"] = time.monotonic() - started
        (RAW / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("CLOSURE_RESULT_GZIP_B64=" + base64.b64encode(gzip.compress(json.dumps(result, sort_keys=True, allow_nan=False).encode())).decode(), flush=True)
        emit_files()
    except Exception as error:
        traceback.print_exc()
        (RAW / "failure.json").write_text(json.dumps({"error": repr(error), "elapsed_s": time.monotonic() - started}) + "\n")
        emit_files()
        raise
