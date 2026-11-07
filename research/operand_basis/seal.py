"""Take local Saturn custody of the completed operand-basis cut and delivery."""
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
JOB = "job-51df6b792d28"
RUN = HERE / "runs" / JOB
OUT = HERE / "custody" / JOB
REPORT_SHA = "e5d23802467bd705ef6e1d9ea7e3dd8925f52589d0dd2942c325404e171a59b7"
BLOG = DOMAINS / "obsidian/blog/2026-11-06-143449-why-more-weight-energy-did-not-preserve-the-mlp-write.md"


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
    files += [p for p in HERE.iterdir() if valid_file(p)]
    for directory in (HERE / "index" / JOB, HERE / "figures"):
        files += [p for p in directory.rglob("*") if valid_file(p)]
    # The main run is complete here. The smoke is represented by metadata;
    # its independent full raw archive stays in its original run directory.
    smoke = HERE / "runs/job-23dd9b4d0de9"
    files += [smoke / name for name in ("report.json", "verification.json", "AUDIT.md", "audit.json", "job.json", "submission.json")]
    files += [BLOG, HERE.parent / "README.md",
              DOMAINS / "obsidian/blog/2026-08-14-what-saturn-can-actually-do.md",
              DOMAINS / "obsidian/blog/2026-11-06-134300-from-a-state-machine-to-a-forward-pass-free-chat-emulator.md",
              DOMAINS / "obsidian/blog/images/job-51df6b792d28-energy-and-write.png",
              DOMAINS / "obsidian/blog/images/job-51df6b792d28-readout-clocks.png"]
    files = sorted(set(files))
    assert all(valid_file(p) for p in files)
    records = [{"path": str(p.relative_to(DOMAINS)), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]
    manifest = {"schema": "tcm-operand-basis-capsule-v1", "job_id": JOB, "report_sha256": REPORT_SHA,
                "files": records, "main_run": "complete collected payload, raw arrays, binary, logs, reports and additive audits",
                "smoke_scope": "metadata and audit only; complete raw evidence remains in original run directory",
                "source_weight_exports": "G/U/gamma bytes retained near compute; hashes/manifests and source model identity included",
                "mechanics_status": "verified", "instrument_status": "complete",
                "trend_status": "mixed; source64 has no aggregate write advantage over deletion",
                "terminal_status": "not-requested", "models_evaluated_by_sealer": 0}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    manifest_path = OUT / "capsule-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    archive_path = OUT / "operand-basis-capsule.tgz"
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
    receipt = {"schema": "tcm-operand-basis-delivery-v1", "job_id": JOB,
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
