"""One resident C experiment under a public Saturn invocation receipt."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results"
LIMIT = 256 * 1024**2

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(args, timeout=480):
    print("[run]", " ".join(map(str, args)), flush=True)
    p = subprocess.run(list(map(str, args)), text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"process {p.returncode}: {p.stderr[-10000:]}")
    return p

def experiment():
    import numpy as np
    from transformers import AutoTokenizer
    from analysis import analyze, archive_bound
    sys.path.insert(0, str(ROOT / "public"))
    from saturn_bridge import invoke
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Payload source differs: " + name)
    model = json.loads((ROOT / "baseline.json").read_text())["model"]
    source = Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name, digest in model["tokenizer_files"].items():
        if sha(source / name) != digest:
            raise ValueError("Tokenizer identity differs")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    report = {"schema": "tcm-gate-law-experiment-v1", "lane": manifest["lane"], "model": model, "environment": {"platform": platform.platform(), "python": sys.version, "job_id": os.environ.get("MRUN_JOB_ID")}, "mechanics_status": "in-progress", "instrument_status": "in-progress", "trend_status": "observations", "terminal_status": "not-requested", "texts": {}}
    def preserve():
        (RAW / "partial-report.json").write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    preserve()
    command(["make"])
    report["binary_sha256"] = sha(ROOT / "build/gate-chat")
    if manifest["lane"] == "smoke":
        args = [ROOT / "runtime/build/stack-mechanics", model["path"]]
        def lifecycle(_):
            p = command(args)
            (RAW / "lifecycle.stdout.txt").write_text(p.stdout)
            return json.loads(p.stdout)
        report["driver"], report["saturn_receipt"] = invoke(args, None, {"model_digest": model["content_sha256"], "binary_sha256": sha(args[0]), "scope": "pretrained lifecycle with explicit gate-law checks; synthetic tokens"}, controller=lifecycle)
        (RAW / "driver.json").write_text(json.dumps(report["driver"]) + "\n")
        ids = tokenizer.apply_chat_template([{"role": "user", "content": "Mira has a red badge. What color is Mira's badge? Reply with only the color."}], tokenize=True, add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or not ids or not all(type(token) is int for token in ids):
            raise ValueError("Tokenizer did not return flat token IDs")
        prompt = RAW / "ordinary-input.txt"
        prompt.write_text(" ".join(map(str, ids)) + "\n")
        historical = Path("/mnt/big/saturn-cmachine/builds/d4ba05c0161a737d/build/tcm-cuda")
        if sha(historical) != "d4ba05c0161a737d9d8c227a679ecb72ea6673773045bf638ca65df888995613":
            raise ValueError("Historical binary differs")
        report["ordinary"] = {}
        for label, binary in (("historical", historical), ("current", ROOT / "runtime/build/tcm-cuda")):
            path = RAW / f"ordinary-{label}.json"
            command([binary, "qwen", model["path"], "--tokens-file", prompt, "--new-tokens", "8", "--context", "512", "--checkpoint-test", "--output", path, "--first-logits", path.with_suffix(".f32")])
            report["ordinary"][label] = json.loads(path.read_text())
        a, b = (np.fromfile(RAW / f"ordinary-{label}.f32", dtype="<f4") for label in ("historical", "current"))
        if a.shape != (151936,) or not np.isfinite(a).all() or a.tobytes() != b.tobytes() or report["ordinary"]["historical"]["tokens"] != report["ordinary"]["current"]["tokens"]:
            raise ValueError("Ordinary native parity differs")
        report["inputs"] = {"ordinary_ids": ids}
    else:
        cases = (("recall", "Mira keeps a red badge and Niko keeps a blue badge. Which badge color belongs to Niko? Answer only the color.", "Now Mira trades her red badge for a green badge. What color does Mira have now? Answer only the color."),
                 ("math", "Compute 17 * 23. Reply with only the integer.", "Subtract 24 from that result. Reply with the integer, then one short sentence explaining the calculation."))
        inputs = []
        for i, (name, prompt_text, follow_text) in enumerate(cases):
            messages = [{"role": "user", "content": prompt_text}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            extended = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": ""}, {"role": "user", "content": follow_text}], tokenize=False, add_generation_prompt=True)
            if not extended.startswith(text):
                raise ValueError("Template prefix differs")
            suffix = tokenizer.encode(extended[len(text):], add_special_tokens=False)
            if tokenizer.encode(extended, add_special_tokens=False) != ids + suffix or suffix[0] != 151645:
                raise ValueError("Template token concatenation differs")
            for label, tokens in (("prompt", ids), ("followup", suffix)):
                (RAW / f"case{i}-{label}.txt").write_text(" ".join(map(str, tokens)) + "\n")
            inputs.append({"case": i, "name": name, "prompt_text": prompt_text, "followup_text": follow_text, "prompt_ids": ids, "followup_ids": suffix})
        report["inputs"] = {"cases": inputs, "budget": 16, "source_only": True, "candidate_order": ["once", "recurring", "zero"], "native_references_after_primary_seal": True}
        bound = archive_bound([len(case["followup_ids"]) for case in inputs])
        report["evidence_capacity_preflight"] = {"worst_case_bytes": bound, "limit_bytes": LIMIT, "followup_lengths": [len(case["followup_ids"]) for case in inputs], "includes_16mib_json_source_margin": True}
        if bound > LIMIT:
            raise ValueError("Declared evidence capacity is smaller than worst-case geometry")
        input_bytes = json.dumps(report["inputs"], sort_keys=True, allow_nan=False).encode()
        (RAW / "input-manifest.json").write_bytes(input_bytes + b"\n")
        report["input_manifest_sha256"] = sha(RAW / "input-manifest.json")
        preserve()
        args = [ROOT / "build/gate-chat", model["path"], RAW / "case0-prompt.txt", RAW / "case0-followup.txt", RAW / "case1-prompt.txt", RAW / "case1-followup.txt", RAW]
        def controller(_):
            p = command(args)
            (RAW / "driver.stdout.txt").write_text(p.stdout)
            (RAW / "driver.stderr.txt").write_text(p.stderr)
            print(p.stderr, end="", flush=True)
            return json.loads((RAW / "driver.json").read_text())
        report["driver"], report["saturn_receipt"] = invoke(args, None, {"model_digest": model["content_sha256"], "binary_sha256": report["binary_sha256"], "inputs": report["inputs"], "input_manifest_sha256": report["input_manifest_sha256"], "source_manifest_sha256": sha(ROOT / "source-manifest.json")}, controller=controller)
        if report["driver"]["model_digest"] != model["content_sha256"]:
            raise ValueError("Executed model differs")
        for case in report["driver"]["primary_cases"]:
            for arm in case["arms"]:
                report["texts"][arm["name"]] = [tokenizer.decode(turn["ids"], skip_special_tokens=True) for turn in arm["answers"]]
        for case in report["driver"]["references"]:
            arm = case["native"]
            report["texts"][arm["name"]] = [tokenizer.decode(turn["ids"], skip_special_tokens=True) for turn in arm["answers"]]
    preserve()
    report["analysis"] = analyze(RAW, ROOT / "bank", report["driver"], report["texts"])
    report["raw_files"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in (".f32", ".txt", ".json") and p.name != "partial-report.json"}
    report.update(mechanics_status="complete", instrument_status="complete")
    return report

def emit_files():
    files = []
    for directory, prefix in ((RAW, "raw"), (ROOT / "runtime", "payload/runtime"), (ROOT / "public", "payload/public")):
        files += [(p, prefix + "/" + str(p.relative_to(directory))) for p in sorted(directory.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and "build" not in p.parts]
    files += [(p, "payload/" + p.name) for p in sorted(ROOT.iterdir()) if p.is_file()]
    if sum(p.stat().st_size for p, _ in files) > LIMIT:
        raise ValueError("Evidence exceeds declared archive limit")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print("COMPOSITION_FILES_TGZ_B64=" + base64.b64encode(buffer.getvalue()).decode(), flush=True)

if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    start = time.monotonic()
    try:
        report = experiment()
        code = 0
    except Exception:
        report = json.loads((RAW / "partial-report.json").read_text()) if (RAW / "partial-report.json").exists() else {}
        report.update(error=traceback.format_exc(), mechanics_status="incomplete", instrument_status="incomplete")
        code = 1
    report["worker_seconds"] = time.monotonic() - start
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    (RAW / "report.json").write_bytes(raw + b"\n")
    print("COMPOSITION_RESULT_GZIP_B64=" + base64.b64encode(gzip.compress(raw, mtime=0)).decode(), flush=True)
    emit_files()
    raise SystemExit(code)
