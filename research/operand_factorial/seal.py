"""Take local Saturn custody of the completed operand-factorial cut and delivery."""
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
JOB = "job-eb06becf984f"
RUN = HERE / "runs" / JOB
OUT = HERE / "custody" / JOB
REPORT_SHA = "5ceab12e2ab62d20978269df55702bb4a26376d0db79752270df9ac5b74eaa0f"
BLOG = DOMAINS / "obsidian/blog/2026-11-06-151646-restoring-the-current-write-leaves-the-conversation-state-different.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_file(path):
    return path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def seal():
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha(RUN / "report.json") == REPORT_SHA
    verification = json.loads((RUN / "verification.json").read_text())
    assert verification["report_sha256"] == REPORT_SHA
    assert verification["scheduler_log_seal"] and verification["analysis_recomputed"]
    assert verification["whole_invocation_receipt"]
    audit = json.loads((RUN / "audit.json").read_text())
    assert audit["mechanics_status"] == "complete" and audit["instrument_status"] == "complete"
    sweep = json.loads((HERE / "index" / JOB / "sweep-results.json").read_text())
    for stage in ("run", "verify"):
        assert sweep["pipeline"][stage]["returncode"] == 0
        assert not sweep["pipeline"][stage]["summary"]["broken"]
    files = [p for p in RUN.rglob("*") if valid_file(p)]
    # This checkout is shared with ongoing work. Executed program sources are
    # already retained in RUN/payload; do not relabel later root edits as them.
    files += [HERE / name for name in (
        "README.md", "FINDINGS.md", "DELIVERY-NOTE.md", "ADVISOR.md",
        "CODE-REVIEW.md", "SMOKE-ADVISOR.md", "RESULTS-ADVISOR.md",
        "CLAIM-REVIEW.md", "VERIFICATION-NOTE.md", "results.py", "audit.py",
        "advisor_analyze.py", "seal.py")]
    for directory in (HERE / "index" / JOB, HERE / "figures"):
        files += [p for p in directory.rglob("*") if valid_file(p)]
    # The main run is complete here. The smoke is represented by metadata;
    # its independent full raw archive stays in its original run directory.
    smoke = HERE / "runs/job-13b692356e63"
    files += [smoke / name for name in ("report.json", "verification.json", "AUDIT.md", "audit.json", "job.json", "submission.json")]
    files += [BLOG, HERE.parent / "README.md",
              DOMAINS / "obsidian/blog/2026-08-14-what-saturn-can-actually-do.md",
              DOMAINS / "obsidian/blog/2026-11-06-143449-why-more-weight-energy-did-not-preserve-the-mlp-write.md",
              DOMAINS / "obsidian/blog/images/job-eb06becf984f-restoration.png",
              DOMAINS / "obsidian/blog/images/job-eb06becf984f-memory-clock.png"]
    files = sorted(set(files))
    assert all(valid_file(p) for p in files)
    snapshot = OUT / "snapshot"
    records = []
    for path in files:
        name = str(path.relative_to(DOMAINS))
        data = path.read_bytes()
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    manifest = {"schema": "tcm-operand-factorial-capsule-v1", "job_id": JOB, "report_sha256": REPORT_SHA,
                "files": records, "main_run": "complete collected payload, raw arrays, binary, logs, reports and additive audits",
                "smoke_scope": "metadata and audit only; complete raw evidence remains in original run directory",
                "source_factor_provenance": "Frozen source64 Q/GQ/UQ imported byte-exactly from predecessor job-51df6b792d28; no recompilation",
                "mechanics_status": "verified", "instrument_status": "complete",
                "trend_status": "coupled restoration; exact current write leaves persistent historical-state differences",
                "terminal_status": "not-requested", "models_evaluated_by_sealer": 0}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    manifest_path = OUT / "capsule-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    archive_path = OUT / "operand-factorial-capsule.tgz"
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
                for record in records:
                    data = (snapshot / record["path"]).read_bytes()
                    assert hashlib.sha256(data).hexdigest() == record["sha256"] and len(data) == record["bytes"]
                    entry(archive, record["path"], data)
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.extractfile("capsule-manifest.json").read() == manifest_bytes
        assert len(archive.getmembers()) == len(records) + 1
        for record in records:
            data = archive.extractfile(record["path"]).read()
            assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    spec = importlib.util.spec_from_file_location("operand_artifact_custody", DOMAINS / "saturn/src/saturn/artifact_custody.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    custody = [asdict(module.take_custody(p, OUT)) for p in (
        archive_path, manifest_path, HERE / "FINDINGS.md", HERE / "RESULTS-ADVISOR.md", BLOG,
        RUN / "verification.json", RUN / "AUDIT.md", HERE / "index" / JOB / "sweep-results.json")]
    result = module.verify_custody(OUT)
    assert result["chain_ok"] and all(row["ok"] for row in result["files"])
    (OUT / "custody-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    chain = (OUT / "artifacts/custody-manifest.jsonl").read_text().splitlines()
    receipt = {"schema": "tcm-operand-factorial-delivery-v1", "job_id": JOB,
               "created_utc": datetime.now(timezone.utc).isoformat(), "report_sha256": REPORT_SHA,
               "capsule_sha256": sha(archive_path), "capsule_bytes": archive_path.stat().st_size,
               "capsule_files_verified": len(records), "custody": custody,
               "custody_chain_head": json.loads(chain[-1])["sha256"], "custody_manifest_rows": len(chain),
               "source_model_verified": True, "native_report_preserved": True,
               "models_evaluated_by_sealer": 0, "terminal_status": "not-requested"}
    (OUT / "delivery.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("job_id", "capsule_files_verified", "capsule_bytes", "capsule_sha256", "custody_manifest_rows")}))
    return receipt


if __name__ == "__main__":
    seal()
