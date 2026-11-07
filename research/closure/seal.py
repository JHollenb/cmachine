"""Seal the completed native cut and its explanatory delivery as a local capsule."""
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]
JOB = "job-7b8a5933f1dd"
RUN = HERE / "runs" / JOB
OUT = HERE / "custody" / JOB
REPORT_SHA = "b693fc8b12dbf0e79f93a9337136517aae1d020f267467a74357ce84ed58208b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_file(path):
    return path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def seal():
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha(RUN / "report.json") == REPORT_SHA
    verification = json.loads((RUN / "verification.json").read_text())
    assert verification["actual_loaded_model_verified"] and verification["bank_and_geometry_verified"]
    extra = [
        DOMAINS / "obsidian/blog/2026-11-06-125119-what-the-compact-mlp-state-forgot.md",
        DOMAINS / "obsidian/blog/2026-11-06-115221-how-we-get-from-state-prediction-to-autonomous-chat.md",
        DOMAINS / "obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md",
        DOMAINS / "obsidian/blog/2026-08-14-what-saturn-can-actually-do.md",
        DOMAINS / "obsidian/blog/2026-10-08-183803-from-time-formed-circuits-to-editable-executable-state.md",
        DOMAINS / "obsidian/blog/images/job-7b8a5933f1dd-code-collisions.png",
        DOMAINS / "obsidian/blog/images/job-7b8a5933f1dd-memory-clock.png",
        DOMAINS / "saturn/cmachine/composition/whole_turn/FINDINGS.md",
        DOMAINS / "saturn/cmachine/composition/whole_turn/runs/job-a071c7a43a38/report.json",
        DOMAINS / "saturn/cmachine/composition/whole_turn/runs/job-a071c7a43a38/verification.json",
        DOMAINS / "model-experiments/experiments/2026-10-08-future-qualified-state/FINDINGS.md",
        DOMAINS / "model-experiments/experiments/2026-10-08-future-qualified-state/core.py",
        DOMAINS / "saturn/experiments/2026-11-05-operational-variable-closure/ADVISOR-STATE-CLOSURE.md",
        DOMAINS / "saturn/experiments/2026-11-05-operational-variable-closure/PLAN-AUTONOMOUS-CHAT.md",
    ]
    files = [p for p in RUN.rglob("*") if valid_file(p)]
    files += [p for p in HERE.iterdir() if valid_file(p)]
    for directory in (HERE / "study", HERE / "index" / JOB):
        files += [p for p in directory.rglob("*") if valid_file(p)]
    files += extra
    files = sorted(set(files))
    assert all(valid_file(p) for p in files)
    records = [{"path": str(p.relative_to(DOMAINS)), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]
    selected_request = next(json.loads(line) for line in (DOMAINS / "saturn/request.jsonl").read_text().splitlines()
                            if json.loads(line).get("handle") == "future-qualified-state-cache-certificate")
    manifest = {"schema": "tcm-state-closure-capsule-manifest-v1", "job_id": JOB,
                "report_sha256": REPORT_SHA, "files": records, "selected_feature_request": selected_request,
                "mechanics_status": "verified", "trend_status": "bounded-counterexamples-and-consumer-observations",
                "terminal_status": "not-requested", "native_models_evaluated_by_sealer": 0}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (OUT / "capsule-manifest.json").write_bytes(manifest_bytes)
    archive_path = OUT / "state-closure-capsule.tgz"
    def entry(archive, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = 0
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                entry(archive, "capsule-manifest.json", manifest_bytes)
                for path, record in zip(files, records):
                    data = path.read_bytes()
                    assert hashlib.sha256(data).hexdigest() == record["sha256"] and len(data) == record["bytes"]
                    entry(archive, record["path"], data)
    # Independently check all archive members against the manifest just written.
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.extractfile("capsule-manifest.json").read() == manifest_bytes
        assert len(archive.getmembers()) == len(records) + 1
        for record in records:
            data = archive.extractfile(record["path"]).read()
            assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    spec = importlib.util.spec_from_file_location("closure_artifact_custody", DOMAINS / "saturn/src/saturn/artifact_custody.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    custody = [asdict(module.take_custody(p, OUT)) for p in (
        archive_path, OUT / "capsule-manifest.json", HERE / "FINDINGS.md", HERE / "MATH.md",
        HERE / "STATE-INTERFACE.md", HERE / "ADVISOR-RESULTS.md", HERE / "study/study.json",
        extra[0], RUN / "binary/receipt.json")]
    result = module.verify_custody(OUT)
    assert result["chain_ok"] and all(row["ok"] for row in result["files"])
    (OUT / "custody-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    chain = (OUT / "artifacts/custody-manifest.jsonl").read_text().splitlines()
    receipt = {"schema": "tcm-state-closure-delivery-v1", "job_id": JOB,
               "created_utc": datetime.now(timezone.utc).isoformat(), "report_sha256": REPORT_SHA,
               "capsule_sha256": sha(archive_path), "capsule_bytes": archive_path.stat().st_size,
               "capsule_files_verified": len(records), "custody": custody,
               "custody_chain_head": json.loads(chain[-1])["sha256"], "custody_manifest_rows": len(chain),
               "source_model_verified": True, "native_report_preserved": True,
               "whole_turn_evidence_is_separate_job": "job-a071c7a43a38",
               "native_models_evaluated_by_sealer": 0, "terminal_status": "not-requested"}
    (OUT / "delivery.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("job_id", "capsule_files_verified", "capsule_bytes", "capsule_sha256", "custody_manifest_rows")}))
    return receipt


if __name__ == "__main__":
    seal()
