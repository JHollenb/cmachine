"""Native C execution and saved-data analysis inside one scheduler lease."""
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
RAW = ROOT/"results"
LIMIT = 32*1024**2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1<<20), b""):h.update(block)
    return h.hexdigest()


def command(args, timeout=300):
    print("[run]", " ".join(map(str, args)), flush=True)
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Process failed {result.returncode}: {result.stderr[-8000:]}")
    return result


def preserve(report):
    path = RAW/"partial-report.json"
    path.write_text(json.dumps(report, sort_keys=True, allow_nan=False)+"\n")


def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze
    sys.path.insert(0, str(ROOT/"public"))
    from saturn_bridge import invoke
    baseline = json.loads((ROOT/"baseline.json").read_text())
    manifest = json.loads((ROOT/"source-manifest.json").read_text())
    zero = manifest.get("panel_kind") == "zero"
    for name, expected in manifest["sources"].items():
        if sha(ROOT/name) != expected:raise ValueError("Shipped source identity differs")
    for name, record in baseline["bank_files"].items():
        if sha(ROOT/"bank"/name) != record["sha256"] or (ROOT/"bank"/name).stat().st_size != record["bytes"]:
            raise ValueError("Frozen bank identity differs")
    model = baseline["model"]
    source = Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name, expected in model["tokenizer_files"].items():
        if sha(source/name) != expected:raise ValueError("Tokenizer identity differs")
    build = command(["make"])
    report = {"schema": "tcm-composition-zero-experiment-v1" if zero else "tcm-composition-experiment-v1", "panel_kind": "zero" if zero else "primary", "model": model, "baseline": baseline,
              "environment": {"platform": platform.platform(), "python": sys.version, "job_id": os.environ.get("MRUN_JOB_ID")},
              "binary_sha256": sha(ROOT/"build/composition"), "build_stderr": build.stderr,
              "mechanics_status": "in-progress", "instrument_status": "in-progress", "trend_status": "observations", "terminal_status": "not-requested"}
    preserve(report)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    messages = [{"role": "user", "content": baseline["source_template_text"]}]
    source_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(source_text, add_special_tokens=False)
    if ids != baseline["source_template_ids"] or tokenizer.encode("red", add_special_tokens=False) != [1151] or tokenizer.encode("blue", add_special_tokens=False) != [12203]:
        raise ValueError("Source template/color IDs differ")
    prompt = RAW/"source.txt";prompt.write_text(" ".join(map(str, ids))+"\n")
    source_own = RAW/"historical-source-own.txt";source_own.write_text(" ".join(map(str, baseline["historical_source_own_ids"]))+"\n")
    questions = (("recall", "What badge color was assigned to Niko? Reply with just the color.", 8),
                 ("math", "Compute 17 * 23. Reply with only the integer.", 8),
                 ("prose", "Write two short sentences explaining why a reader needs context to interpret a message.", 48))
    inputs = []
    for name, text, budget in questions:
        extended = tokenizer.apply_chat_template(messages+[{"role": "assistant", "content": ""}, {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        if not extended.startswith(source_text):raise ValueError("Question template prefix differs")
        suffix = tokenizer.encode(extended[len(source_text):], add_special_tokens=False)
        if tokenizer.encode(extended, add_special_tokens=False) != ids+suffix or suffix[0] != 151645:
            raise ValueError("Question template concatenation differs")
        path = RAW/f"{name}.txt";path.write_text(" ".join(map(str, suffix))+"\n")
        inputs.append({"name": name, "text": text, "budget": budget, "ids": suffix, "file": path.name})
    report["inputs"] = {"source_ids": ids, "questions": inputs}
    request = {"model_digest": model["content_sha256"], "binary_sha256": report["binary_sha256"], "inputs": report["inputs"],
               "bank_files": baseline["bank_files"], "runtime_sources": baseline["runtime_sources"], "plan_sha256": manifest["plan_sha256"]}
    preserve(report)
    args = [ROOT/"build/composition", model["path"], prompt, *[RAW/q["file"] for q in inputs], RAW, ROOT/"bank", source_own, baseline["historical_source_kv_sha256"], baseline["bank_concat_sha256"]]
    if zero:args.append("zero")
    def controller(_):
        result = command(args)
        (RAW/"driver.stdout.txt").write_text(result.stdout);(RAW/"driver.stderr.txt").write_text(result.stderr)
        return json.loads((RAW/"driver.json").read_text())
    driver, report["saturn_receipt"] = invoke(args, None, request, controller=controller)
    report["driver"] = driver
    if driver["model_digest"] != model["content_sha256"]:raise ValueError("Executed model differs")
    if driver["frozen_bank_before_sha256"] != baseline["bank_concat_sha256"] or driver["frozen_bank_after_sha256"] != baseline["bank_concat_sha256"]:
        raise ValueError("Owned frozen program changed")
    texts = {a["name"]: tokenizer.decode(a["own"]["ids"], skip_special_tokens=True) for a in driver["arms"]}
    texts["source"] = tokenizer.decode(driver["source_own"]["ids"], skip_special_tokens=True)
    report["texts"] = texts
    report["source_replay_ids_equal"] = driver["source_own"]["ids"] == baseline["historical_source_own_ids"]
    if not driver["source_replay_checked_before_candidates"] or not report["source_replay_ids_equal"] or driver["source_position"] != baseline["historical_source_position"] or driver["source_kv_sha256"] != baseline["historical_source_kv_sha256"]:
        raise ValueError("Historical source replay differs")
    preserve(report)
    if zero:
        from zero_analysis import analyze as analyze_zero
        reference = json.loads((ROOT/"zero-reference.json").read_text())
        report["zero_reference_sha256"] = sha(ROOT/"zero-reference.json")
        report["analysis"] = analyze_zero(RAW, ROOT/"bank", driver, texts, reference)
    else:report["analysis"] = analyze(RAW, ROOT/"bank", driver, texts)
    report["raw_files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in (".f32", ".txt") or p.name == "driver.json"}
    report.update(mechanics_status="complete", instrument_status="complete")
    return report


def emit_files():
    files = []
    for directory, name in ((RAW, "raw"), (ROOT/"runtime", "payload/runtime"), (ROOT/"bank", "payload/bank"), (ROOT/"public", "payload/public")):
        files.extend((p, name+"/"+str(p.relative_to(directory))) for p in sorted(directory.rglob("*")) if p.is_file() and "__pycache__" not in p.parts)
    files.extend((p, "payload/"+p.name) for p in sorted(ROOT.iterdir()) if p.is_file())
    if sum(p.stat().st_size for p, _ in files) > LIMIT:raise ValueError("Evidence archive exceeds declared limit")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, name in files:archive.add(path, arcname=name, recursive=False)
    print("COMPOSITION_FILES_TGZ_B64="+base64.b64encode(buffer.getvalue()).decode(), flush=True)


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    start = time.monotonic()
    try:
        report = experiment();code = 0
    except Exception:
        report = json.loads((RAW/"partial-report.json").read_text()) if (RAW/"partial-report.json").exists() else {}
        report.update(error=traceback.format_exc(), mechanics_status="incomplete", instrument_status="incomplete");code = 1
    report["worker_seconds"] = time.monotonic()-start
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    (RAW/"report.json").write_bytes(raw+b"\n")
    print("COMPOSITION_RESULT_GZIP_B64="+base64.b64encode(gzip.compress(raw, mtime=0)).decode(), flush=True)
    emit_files()
    raise SystemExit(code)
