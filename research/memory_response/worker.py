"""Execute one frozen memory-response panel in one resident CUDA process."""
from pathlib import Path
import base64
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results"
LIMIT = 256 * 1024**2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, timeout=780, live=False):
    print("[run]", " ".join(map(str, args)), flush=True)
    if live:
        process = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, errors = [], []
        def drain(stream, retained):
            for line in stream:
                retained.append(line)
                print(line, end="", flush=True)
        readers = [threading.Thread(target=drain, args=(process.stdout, output)), threading.Thread(target=drain, args=(process.stderr, errors))]
        for reader in readers: reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            for reader in readers: reader.join()
            (RAW / "last-failed.stdout.txt").write_text("".join(output))
            (RAW / "last-failed.stderr.txt").write_text("".join(errors))
            raise
        for reader in readers: reader.join()
        result = subprocess.CompletedProcess(args, process.returncode, "".join(output), "".join(errors))
    else:
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        (RAW / "last-failed.stdout.txt").write_text(result.stdout)
        (RAW / "last-failed.stderr.txt").write_text(result.stderr)
        raise RuntimeError(f"Process failed {result.returncode}: {result.stderr[-12000:]}")
    return result


def preserve(report):
    (RAW / "partial-report.json").write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")


def tokenize_inputs(spec, tokenizer, lane):
    destination = RAW / "inputs"
    destination.mkdir()
    cases = []
    sample = [{"role": "user", "content": "placeholder"}]
    prefix = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    def suffix(text):
        complete = tokenizer.apply_chat_template(sample + [{"role": "assistant", "content": ""}, {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        if not complete.startswith(prefix): raise ValueError("Append template differs")
        ids = tokenizer.encode(complete[len(prefix):], add_special_tokens=False)
        if tokenizer.encode(complete, add_special_tokens=False) != prefix_ids + ids or not ids or ids[0] != 151645:
            raise ValueError("Token append boundary differs")
        return ids
    for row in spec["cases"]:
        if row["id"] not in lane["cases"]: continue
        initial = spec["initial_template"].format(mira=row["mira"], niko=row["niko"])
        messages = [{"role": "user", "content": initial}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        segments = {"initial": tokenizer.encode(text, add_special_tokens=False),
                    "update": suffix(spec["updates"][row["update_bit"]]),
                    "distractor": suffix(spec["distractor"]),
                    "query0": suffix(spec["questions"][0]), "query1": suffix(spec["questions"][1])}
        bound = sum(len(segments[s]) for s in ("initial", "update", "distractor")) + 3 * spec["ack_budget"] + max(len(segments["query0"]), len(segments["query1"])) + spec["answer_budget"]
        if bound > spec["capacity"]: raise ValueError("Context capacity exceeded")
        files = {}
        for name, ids in segments.items():
            filename = f"case{row['id']}-{name}.txt"
            (destination / filename).write_text(" ".join(map(str, ids)) + "\n")
            files[name] = filename
        cases.append({**row, "initial_text": initial, "segments": segments, "files": files, "token_bound": bound})
    token_variants = {color: {surface: tokenizer.encode(surface, add_special_tokens=False) for surface in (color, " " + color, color.capitalize(), " " + color.capitalize())} for color in ("red", "blue")}
    manifest = {**spec, "cases": cases, "lane": lane["lane"], "color_token_variants": token_variants,
                "input_authority": "literal supplied user segments; branch owns all generated acknowledgement and answer IDs"}
    (destination / "runtime-inputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def decode_answers(driver, tokenizer):
    texts = {}
    def visit(item):
        if isinstance(item, dict):
            if "answer_ids" in item:
                if "name" not in item: raise ValueError("Answer missing unique name")
                texts[item["name"]] = tokenizer.decode(item["answer_ids"], skip_special_tokens=True)
            for value in item.values(): visit(value)
        elif isinstance(item, list):
            for value in item: visit(value)
    visit(driver)
    return texts


def experiment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from transformers import AutoTokenizer
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        if sha(ROOT / name) != digest: raise ValueError("Payload source differs: " + name)
    spec = json.loads((ROOT / "inputs.json").read_text())
    model = json.loads((ROOT / "model.json").read_text())
    lane = json.loads((ROOT / "lane.json").read_text())
    source = Path(spec["tokenizer_path"])
    for name, digest in model["tokenizer_files"].items():
        if sha(source / name) != digest: raise ValueError("Tokenizer differs")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    inputs = tokenize_inputs(spec, tokenizer, lane)
    # Ten full vocabulary snapshots per policy/case, plus two question audits
    # at two layers with original and decoded paired K/V, bounded by the actual
    # token geometry. Header/source/binary/metadata receive a separate allowance.
    capture_bound = 16 * 1024**2 + 5 * sum(10 * 151936 * 4 + 2 * 8192 * row["token_bound"] for row in inputs["cases"])
    if capture_bound > LIMIT:
        raise ValueError("Declared capture geometry exceeds collection capacity")
    build = command(["make"])
    binary = ROOT / "build/memory-response"
    (RAW / "binary").mkdir()
    shutil.copy2(binary, RAW / "binary/memory-response")
    report = {"schema": "tcm-memory-response-experiment-v1", "model": model, "inputs": inputs,
              "lane": lane["lane"], "hypothesis_sha256": manifest["hypothesis_sha256"],
              "capture_capacity": {"declared_upper_bound_bytes": capture_bound, "limit_bytes": LIMIT,
                                   "scope": "ten full readouts per path, own-query audit at17/18, sources/binary/metadata"},
              "environment": {"platform": platform.platform(), "python": sys.version, "job_id": os.environ.get("MRUN_JOB_ID")},
              "binary_sha256": sha(binary), "build_stderr": build.stderr,
              "mechanics_status": "in-progress", "instrument_status": "in-progress",
              "trend_status": "observations", "terminal_status": "not-requested"}
    preserve(report)
    sys.path.insert(0, str(ROOT / "public"))
    from saturn_bridge import invoke
    args = [binary, model["path"], RAW / "inputs", RAW, lane["lane"]]
    request = {"model_digest": model["content_sha256"], "binary_sha256": report["binary_sha256"],
               "source_manifest_sha256": sha(ROOT / "source-manifest.json"), "plan_sha256": manifest["plan_sha256"],
               "hypothesis_sha256": manifest["hypothesis_sha256"], "runtime_inputs_sha256": sha(RAW / "inputs/runtime-inputs.json"),
               "inputs": inputs, "state_authority": "empty bootstrap; actual branch-owned inputs and encoded rows; native operands remain paid"}
    def control(_):
        result = command(args, live=True)
        (RAW / "driver.stdout.txt").write_text(result.stdout)
        (RAW / "driver.stderr.txt").write_text(result.stderr)
        return json.loads((RAW / "driver.json").read_text())
    report["driver"], report["saturn_receipt"] = invoke(args, None, request, controller=control)
    if report["driver"]["model_digest"] != model["content_sha256"]: raise ValueError("Executed model differs")
    primary = {str(p.relative_to(RAW)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "partial-report.json"}
    (RAW / "primary-seal.json").write_text(json.dumps({"schema": "tcm-memory-response-primary-seal-v1", "files": primary,
        "timing": "after all C primary/mechanics execution, before saved-only response analysis", "model_calls_after_seal": 0}, indent=2, sort_keys=True) + "\n")
    report["texts"] = decode_answers(report["driver"], tokenizer)
    preserve(report)
    from analysis import analyze
    report["analysis"] = analyze(RAW, ROOT, report["driver"], report["texts"])
    report["raw_files"] = {str(p.relative_to(RAW)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "partial-report.json"}
    report.update(mechanics_status="complete", instrument_status="complete")
    return report


def emit_files():
    files = [(p, "raw/" + str(p.relative_to(RAW))) for p in sorted(RAW.rglob("*")) if p.is_file()]
    files += [(p, "payload/" + str(p.relative_to(ROOT))) for p in sorted(ROOT.rglob("*")) if p.is_file() and not p.is_relative_to(RAW) and "__pycache__" not in p.parts and "build" not in p.relative_to(ROOT).parts]
    if sum(p.stat().st_size for p, _ in files) > LIMIT: raise ValueError("Raw custody exceeds declared bound")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, name in files: archive.add(path, arcname=name, recursive=False)
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
