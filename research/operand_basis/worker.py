"""Finite C operand-basis experiment inside one public Saturn invocation."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results"
LIMIT = 256 * 1024**2
CASES = (
    ("recall", "Mira keeps a red badge and Niko keeps a blue badge. Which badge color belongs to Niko? Answer only the color.", "Now Mira trades her red badge for a green badge. What color does Mira have now? Answer only the color."),
    ("math", "Compute 17 * 23. Reply with only the integer.", "Subtract 24 from that result. Reply with the integer, then one short sentence explaining the calculation."),
    ("prose", "Write two short sentences explaining why a reader needs context to interpret a message.", "Now give one short example using the word bank."),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, timeout=720):
    print("[run]", " ".join(map(str, args)), flush=True)
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Process failed {result.returncode}: {result.stderr[-12000:]}")
    return result


def preserve(report):
    (RAW / "partial-report.json").write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")


def make_inputs(tokenizer, smoke):
    inputs = []
    for index, (name, prompt, followup) in enumerate(CASES):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        extended = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": ""}, {"role": "user", "content": followup}], tokenize=False, add_generation_prompt=True)
        if not extended.startswith(text):
            raise ValueError("Template prefix differs")
        suffix = tokenizer.encode(extended[len(text):], add_special_tokens=False)
        if tokenizer.encode(extended, add_special_tokens=False) != ids + suffix or suffix[0] != 151645:
            raise ValueError("Template concatenation differs")
        for label, values in (("prompt", ids), ("followup", suffix)):
            (RAW / f"case{index}-{label}.txt").write_text(" ".join(map(str, values)) + "\n")
        inputs.append({"case": index, "name": name, "prompt_text": prompt, "followup_text": followup, "prompt_ids": ids, "followup_ids": suffix})
    return {"cases": inputs, "budget": 2 if smoke else 16,
            "candidate_order": ["source64", "random64", "b3", "zero"],
            "primary_from_first_prompt_token": True, "native_references_after_primary_seal": True,
            "basis_calibration": "original selected weights only; B3 comparator retains historical activation calibration"}


def archive_bound(inputs):
    """Upper bound for the declared driver capture geometry, before execution."""
    d, h, vocab = 1536, 8960, 151936
    count = len(inputs["cases"])
    events = sum(len(c["prompt_ids"]) + len(c["followup_ids"]) + 2 * inputs["budget"] for c in inputs["cases"])
    # Four primary ports + local native write + two same-ID native ports.
    trajectories = 4 * events * (7 * d * 4 + 1024)
    native_own = events * (4 * d * 4 + 1024)
    readouts = (4 + 1 + 4) * count * 4 * vocab * 4
    gate_up_clocks = (4 + 1) * count * 8 * 2 * h * 4
    banks = 2 * 4_980_736
    return trajectories + native_own + readouts + gate_up_clocks + banks + 16 * 1024**2


def experiment():
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from transformers import AutoTokenizer
    from analysis import analyze
    sys.path.insert(0, str(ROOT / "public"))
    from saturn_bridge import invoke
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Payload source differs: " + name)
    baseline = json.loads((ROOT / "baseline.json").read_text())
    for name, record in baseline["bank_files"].items():
        if sha(ROOT / "bank" / name) != record["sha256"]:
            raise ValueError("Historical B3 bank differs")
    model = baseline["model"]
    source = Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name, expected in model["tokenizer_files"].items():
        if sha(source / name) != expected:
            raise ValueError("Tokenizer identity differs")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    report = {"schema": "tcm-operand-basis-experiment-v1", "lane": manifest["lane"], "model": model, "baseline": baseline,
              "environment": {"platform": platform.platform(), "python": sys.version, "job_id": os.environ.get("MRUN_JOB_ID")},
              "mechanics_status": "in-progress", "instrument_status": "in-progress", "trend_status": "observations", "terminal_status": "not-requested"}
    preserve(report)
    build = command(["make"])
    report["build_stderr"] = build.stderr
    report["inputs"] = make_inputs(tokenizer, manifest["lane"] == "smoke")
    bound = archive_bound(report["inputs"])
    report["evidence_capacity_preflight"] = {"worst_case_bytes": bound, "limit_bytes": LIMIT,
                                             "excludes_source_weight_exports_retained_near_compute": True}
    if bound > LIMIT:
        raise ValueError("Declared capture geometry exceeds evidence capacity")
    (RAW / "input-manifest.json").write_text(json.dumps(report["inputs"], sort_keys=True, allow_nan=False) + "\n")
    report["input_manifest_sha256"] = sha(RAW / "input-manifest.json")
    report["binary_sha256"] = sha(ROOT / "build/operand-chat")
    (RAW / "binary").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "build/operand-chat", RAW / "binary/operand-chat")
    preserve(report)
    # The isolated driver's CLI is intentionally kept in one adapter here.
    args = driver_arguments(model, baseline, report["inputs"])
    def controller(_):
        result = command(args)
        (RAW / "driver.stdout.txt").write_text(result.stdout)
        (RAW / "driver.stderr.txt").write_text(result.stderr)
        print(result.stderr, end="", flush=True)
        return json.loads((RAW / "driver.json").read_text())
    request = {"model_digest": model["content_sha256"], "binary_sha256": report["binary_sha256"],
               "inputs": report["inputs"], "input_manifest_sha256": report["input_manifest_sha256"],
               "source_manifest_sha256": sha(ROOT / "source-manifest.json"), "bank_concat_sha256": baseline["bank_concat_sha256"]}
    report["driver"], report["saturn_receipt"] = invoke(args, None, request, controller=controller)
    if report["driver"]["model_digest"] != model["content_sha256"]:
        raise ValueError("Executed model differs")
    if report["driver"].get("budget") != report["inputs"]["budget"] or report["driver"].get("rank") != 64 or report["driver"].get("capacity") != 512:
        raise ValueError("Executed budget or geometry differs")
    compiled = ROOT / "compiled"
    (RAW / "compiled").mkdir(exist_ok=True)
    for path in sorted(compiled.iterdir()):
        if path.is_file() and path.suffix in (".f32", ".json"):
            shutil.copy2(path, RAW / "compiled" / path.name)
    shutil.copy2(compiled / "compile-input/input-manifest.json", RAW / "compiled/source-input-manifest.json")
    report["compiler"] = json.loads((compiled / "bank_manifest.json").read_text())
    report["source_export"] = json.loads((compiled / "compile-input/input-manifest.json").read_text())
    report["source_weight_exports_retained_near_compute"] = str(compiled / "compile-input")
    report["texts"] = decode_answers(tokenizer, report["driver"])
    report["analysis"] = analyze(RAW, ROOT / "bank", report["driver"], report["texts"])
    preserve(report)
    report["raw_files"] = {str(p.relative_to(RAW)): {"sha256": sha(p), "bytes": p.stat().st_size}
                           for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "partial-report.json"}
    report.update(mechanics_status=report["analysis"]["summary"].get("mechanics_status", "evidence_incomplete"),
                  instrument_status=report["analysis"]["summary"].get("instrument_status", "evidence_incomplete"))
    return report


def driver_arguments(model, baseline, inputs):
    args = [ROOT / "build/operand-chat", model["path"], RAW, ROOT / "bank", ROOT / "compiled", sys.executable, ROOT / "compiler.py"]
    for case in inputs["cases"]:
        args += [RAW / f"case{case['case']}-prompt.txt", RAW / f"case{case['case']}-followup.txt"]
    args.append(str(inputs["budget"]))
    if inputs["budget"] == 2:
        args.append("smoke")
    return args


def decode_answers(tokenizer, driver):
    texts = {}
    for case in driver.get("primary_cases", []):
        for arm in case["arms"]:
            answers = arm.get("answers", [arm.get("answer0", {}), arm.get("answer1", {})])
            texts[arm["name"]] = [tokenizer.decode(turn.get("ids", []), skip_special_tokens=True) for turn in answers]
    for case in driver.get("references", []):
        arm = case["native"]
        answers = arm.get("answers", [arm.get("answer0", {}), arm.get("answer1", {})])
        texts[arm["name"]] = [tokenizer.decode(turn.get("ids", []), skip_special_tokens=True) for turn in answers]
    return texts


def emit_files():
    files = []
    for directory, prefix in ((RAW, "raw"), (ROOT / "runtime", "payload/runtime"), (ROOT / "public", "payload/public"), (ROOT / "bank", "payload/bank")):
        files += [(p, prefix + "/" + str(p.relative_to(directory))) for p in sorted(directory.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts and "build" not in p.parts]
    files += [(p, "payload/" + p.name) for p in sorted(ROOT.iterdir()) if p.is_file()]
    total = sum(p.stat().st_size for p, _ in files)
    if total > LIMIT:
        raise ValueError(f"Evidence exceeds declared archive limit: {total}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print("COMPOSITION_FILES_TGZ_B64=" + base64.b64encode(buffer.getvalue()).decode(), flush=True)


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    started = time.monotonic()
    try:
        report = experiment()
        code = 0
    except Exception:
        report = json.loads((RAW / "partial-report.json").read_text()) if (RAW / "partial-report.json").exists() else {}
        report.update(error=traceback.format_exc(), mechanics_status="incomplete", instrument_status="incomplete")
        code = 1
    report["worker_seconds"] = time.monotonic() - started
    encoded = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    (RAW / "report.json").write_bytes(encoded + b"\n")
    print("COMPOSITION_RESULT_GZIP_B64=" + base64.b64encode(gzip.compress(encoded, mtime=0)).decode(), flush=True)
    emit_files()
    raise SystemExit(code)
