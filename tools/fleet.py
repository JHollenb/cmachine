"""Ship only this program, derived ISA bundles and the frozen native source.

One finite mrun lease, no nested submissions or retries. Raw model outputs are
collected from the exact job log and retained alongside its scheduler record.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOMAINS=ROOT.parent
RESEARCH=ROOT/'research'
CUDA_MODES=("qwen","continuation","client","stack","plane","reuse","enrich","components","formation","writer","lineage")
ASSAY_MODES=("plane","reuse","enrich","components","formation","writer","lineage")


def submit(mode):
    from mrun.client.submit import launch
    with tempfile.TemporaryDirectory(prefix="tcm-payload-") as temporary:
        staging=Path(temporary)
        for directory in ("src","tools","tests"):
            shutil.copytree(ROOT/directory,staging/directory,ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT/"Makefile",staging/"Makefile")
        if mode in ("isa","continuation"):
            (staging/"models").mkdir()
            for name in ("isa-r3.tcm","post2-quadratic.tcm"):
                shutil.copy2(ROOT/"models"/name,staging/"models"/name)
                shutil.copy2(ROOT/"models"/Path(name).with_suffix(".json"),staging/"models"/Path(name).with_suffix(".json"))
            for package,name in (("direct_effect","model.py"),("isa_lab","ssm.py")):
                target=staging/"native"/package;target.mkdir(parents=True)
                (target/"__init__.py").write_text("")
                shutil.copy2(DOMAINS/"isa/outputs/direct-effect-v2/r3-mamba/source"/package/name,target/name)
        if mode in ("continuation","stack")+ASSAY_MODES:
            shutil.copytree(DOMAINS/"saturn-pub/src/saturn_pub",staging/"public/saturn_pub",ignore=shutil.ignore_patterns("__pycache__",*( ["trial_data"] if mode in ASSAY_MODES else [])))
        if mode=="plane":shutil.copy2(RESEARCH/"PLANE_EXPERIMENT.md",staging/"PLANE_EXPERIMENT.md")
        if mode in ("reuse","enrich","components","formation","writer"):shutil.copy2(RESEARCH/"REUSE_EXPERIMENT.md",staging/"REUSE_EXPERIMENT.md")
        if mode in ("enrich","components","formation","writer"):shutil.copy2(RESEARCH/"ENRICH_EXPERIMENT.md",staging/"ENRICH_EXPERIMENT.md")
        if mode in ("components","formation"):shutil.copy2(RESEARCH/"CARRY_READ_EXPERIMENT.md",staging/"CARRY_READ_EXPERIMENT.md")
        if mode=="formation":shutil.copy2(RESEARCH/"CARRY_FORMATION_EXPERIMENT.md",staging/"CARRY_FORMATION_EXPERIMENT.md")
        if mode=="writer":shutil.copy2(RESEARCH/"WRITER_INTERVENTION_EXPERIMENT.md",staging/"WRITER_INTERVENTION_EXPERIMENT.md")
        if mode=="lineage":shutil.copy2(RESEARCH/"PARENT_LINEAGE.md",staging/"PARENT_LINEAGE.md")
        sources={str(p.relative_to(staging)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(staging.rglob("*")) if p.is_file()}
        config={"schema":"tcm-fleet-v1","mode":mode,"sources":sources,"backend":"hf" if mode=="qwen" else "c-fp32-cpu","candidate_backend":"c-cublas-fp32" if mode=="qwen" else "c-fp32-cpu","dtype":"fp32","device":"cuda" if mode=="qwen" else "cpu",
                "physical_work":"26 native calls; 52 C full instruction calls plus 16 row-31 checkpoint suffix replays" if mode=="isa" else "environment inventory only",
                "cpu_threads":1,"max_retries":0}
        if mode=="qwen":config["physical_work"]="3 native FP32 and 3 native BF16 greedy paths, then 3 independent C prefill/generation paths; budgets64/24/80, one model resident at a time"
        if mode=="continuation":
            config.update(backend="hf",candidate_backend="c-cublas-fp32",device="cuda",
                physical_work="one persistent C Qwen session: split/whole generation, own-output appends, full-prefix replay, optional explicit memory; 4 native FP32 paths, one typed 2-instruction ISA call",
                reservation_source="measured backend exception: job-52f2231facaf peak RSS5076.3MB/VRAM6718MB, same1024 context, one model resident at a time")
        if mode=="client":
            config.update(backend="hf",candidate_backend="c-cublas-fp32",device="cuda",
                physical_work="one persistent C Qwen process plus tokenizer-only client, supplied-fact recall and rewind/capacity controls;512context/16reply budget",
                reservation_source="measured backend exception: job-52f2231facaf peak RSS5076.3MB/VRAM6718MB; client has smaller512context, no native model resident")
        if mode=="stack":
            config.update(backend="hf",candidate_backend="c-cublas-fp32",device="cuda",
                physical_work="one pretrained lifecycle test,3historical+3candidate32token paths with14selectedports per candidate,1nested text-client session; models sequential,512maxcontext; no HF model or training",
                reservation_source="measured backend exception: job-52f2231facaf RSS5076.3MB/VRAM6718MB at1024context; current512maxcontext, no concurrent native model",
                max_seq_len=512,backend_options={"physical_batch_size":1})
        if mode=="plane":
            config.update(backend="hf",candidate_backend="c-cublas-fp32",device="cuda",
                physical_work="one real-model lifecycle check; historical/current native8-token binding paths; one resident C plane driver,19branches,6fixed doses,8-token own replies and2own-state followups each;512context; no HF model or training",
                reservation_source="measured C backend exception: job-40b06fda7656 RSS5149.5MB/VRAM6378MB at512context; current extra234KiB program;6144MB VRAM reservation has6758.4MB enforced ceiling,380.4MB above prior peak; one weight bank at a time",
                max_seq_len=512,backend_options={"physical_batch_size":1},evidence_archive_limit_mib=32)
        if mode in ("reuse","enrich","components","formation","writer"):
            config.update(backend="hf",candidate_backend="c-cublas-fp32",device="cuda",
                physical_work="one real-model lifecycle check;2ordinary8-token source paths; one resident C reuse driver,4source controls+2later questions x5arms,8-token own answers+2followups each later arm;512context; no HF model/training",
                reservation_source="measured C backend exception: job-f8409b2c0606 RSS4242.7MB/VRAM6378MB at512context; new program/workspace below1MiB;6144MB reservation enforces6758.4MB ceiling; one weight bank at a time",
                max_seq_len=512,backend_options={"physical_batch_size":1},evidence_archive_limit_mib=32)
            if mode=="enrich":config["physical_work"]="one real-model lifecycle check;2ordinary8-token source paths; one resident C driver,4source+1Mira calibration+2fresh Niko x4arms; one explicit extra operand (2compile GEMVs);8-token own answers+16followups;512context; no HF model/training"
            if mode=="components":config.update(physical_work="one real-model lifecycle check;2ordinary8-token source paths; one resident C driver,4source+1Mira calibration+1fresh Niko x5arms; fixed predecessor B3;8-token own answers+10followups; pure split-joint diagnostic0G/U+1down;512context; no HF model/training",reservation_source="measured C backend exception: job-c034107916a2 RSS4031.4MB/VRAM6378MB at512context; extra12KiB GPU split scratch;6144MB reservation enforces6758.4MB ceiling; one model bank at a time")
            if mode=="formation":config.update(physical_work="one real-model lifecycle check;2ordinary8-token source paths; one resident C driver,4source+1Mira calibration+1fresh Niko x5arms; frozen B3;25actual layer15 ports; inherited/new/fullcarried correction with8-token own answers+10followups;1pure joint down;512context; no training",reservation_source="measured C backend exception: job-00038c17cfcd RSS4001.1MB/VRAM6378MB at512context; extra24KiB GPU operands, same one resident model;6144MB VRAM reservation has6758.4MB enforced ceiling")
            if mode=="writer":config.update(physical_work="one real-model lifecycle check;2ordinary source paths;4source+1Mira calibration;4sequential actual-writer policies+1ordinary target control; full first/common-next logits,KV15-27,own8tokens+10followups;512context; no training",reservation_source="measured C backend exception: formation job-0e69e4a9b62e RSS4005.1MB/VRAM6378MB; writer adds6144GPU+43008host bytes, same one resident bank/512context;6144MB VRAM reservation enforces6758.4MB ceiling")
        if mode=="lineage":config.update(backend="hf",candidate_backend="c-cublas-fp32",device="cuda",physical_work="one pretrained lifecycle with hostile same-position/different-parent writer refusal and valid rollback reuse; one historical/current source pair,8token budget/full first logits/checkpoint replay; one C model at a time; no science rerun",reservation_source="measured C exception: job-87eed6acd554 RSS4180.2MB/VRAM6378MB at512context; lineage adds512host bytes to markers and zero GPU bytes;6144MB reservation/6758.4MB enforced ceiling",max_seq_len=512,backend_options={"physical_batch_size":1},evidence_archive_limit_mib=32)
        (staging/"source-manifest.json").write_text(json.dumps(config,indent=2)+"\n")
        job=launch("uv run --offline --project {env} python tools/worker.py "+mode,experiment="c-state-machine-"+mode,payload=staging,
                   model="qwen2.5-1.5b-instruct" if mode in CUDA_MODES else None,
                   prefer_host="beast",pin="beast",needs={"cuda":True} if mode in CUDA_MODES else {},config=config,env_alias="mrun",timeout_s=600 if mode in CUDA_MODES else 180,
                   ram_mb=6144 if mode in CUDA_MODES and mode!="qwen" else None,vram_mb=6144 if mode in ASSAY_MODES else 8192 if mode in ("continuation","client","stack") else None,
                   cpu_threads=1,est_wall_s=60 if mode=="lineage" else 120 if mode in ("formation","writer") else None,
                   detach=True,retry_on_kill=False,preflight=True,
                   note=("C parent lineage:1pretrained lifecycle,wrong-parent sameclock refusal/actualparent rollback reuse;1historical+1current source8tokens/fullfirstlogits/checkpointreplay;512context,one model/CPU;no scientific rerun;stop after mechanics" if mode=="lineage" else "C actual writers:1lifecycle,2ordinary source paths,4source+1calibration;4writer policies+1ordinary target control; own8tokens+10followups;full first/next logits,KV15-27;512context,one model/CPU;stop after panel" if mode=="writer" else "C Qwen: 3 FP32 + 3 BF16 native greedy paths, then 3 C paths; budgets64/24/80; own prefill/cache; one model resident; stop after panel" if mode=="qwen" else "C nested stack: one lifecycle,3old+3new32token paths,42ports,1client;512context; one weight bank at a time; stop after panel" if mode=="stack" else "C receiver plane: one lifecycle,2ordinary8token paths,19branches/6doses with own8tokens+2followups;512context,one model,one CPU thread; measured6378MB VRAM vs6758MB hard ceiling; stop after panel" if mode=="plane" else "C frozen receiver reuse:1lifecycle,2ordinary paths,4source+10later branches,8-token own answers+20followups;512context,one model,one CPU;6378MB measured VRAM vs6758MB ceiling; stop after panel" if mode=="reuse" else "C B3:1lifecycle,2ordinary paths,4source+1calibration+8fresh-target branches,8-token own answers+16followups;512context,one model,one CPU;6378MB measured VRAM vs6758MB ceiling; stop after panel" if mode=="enrich" else "C carry/read:1lifecycle,2ordinary paths,4source+1calibration+5fresh-target branches,8-token own answers+10followups;1pure joint down diagnostic;512context,one model,one CPU;6378MB measured vs6758MB ceiling; stop after panel" if mode=="components" else "C carry formation:1lifecycle,2ordinary paths,4source+1calibration+5fresh arms;25layer15 ports;inherited/new/fullcarried;10followups;1joint down;512context,one model;stop after panel" if mode=="formation" else "C state-machine "+mode+": one CPU thread, finite declared work; stop after panel"))
    target=ROOT/"results"/job;target.mkdir(parents=True,exist_ok=True)
    (target/"submission.json").write_text(json.dumps({"job_id":job,"config":config},indent=2)+"\n")
    print(json.dumps({"job_id":job,"result_directory":str(target)}))


def collect(job):
    from mrun.client.api import Api
    api=Api()
    _,raw,_=api.request("GET",f"/api/jobs/{job}",timeout_s=20)
    record=json.loads(raw)
    _,logs,_=api.request("GET",f"/api/jobs/{job}/logs?offset=0&wait_s=1",timeout_s=30)
    target=ROOT/"results"/job;target.mkdir(parents=True,exist_ok=True)
    (target/"job.json").write_text(json.dumps(record,indent=2)+"\n")
    (target/"logs.txt").write_bytes(logs)
    report=None
    for line in logs.decode("utf-8","replace").splitlines():
        if line.startswith("TCM_RESULT_GZIP_B64="):
            report=json.loads(gzip.decompress(base64.b64decode(line.split("=",1)[1])))
    if report is not None:
        (target/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        for line in logs.decode("utf-8","replace").splitlines():
            if line.startswith("TCM_FILES_TGZ_B64="):
                import io,tarfile
                with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=",1)[1])),mode="r:gz") as archive:
                    members=archive.getmembers()
                    limit=32 if report.get("schema") in ("tcm-plane-experiment-v1","tcm-reuse-experiment-v1","tcm-enrichment-experiment-v1","tcm-components-experiment-v1","tcm-formation-experiment-v1","tcm-writer-experiment-v1","tcm-lineage-mechanics-v1") else 16
                    if sum(m.size for m in members)>limit*1024**2:raise ValueError("evidence archive exceeds declared limit")
                    for member in members:
                        relative=Path(member.name)
                        if not member.isfile() or relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("raw","payload"):
                            raise ValueError("invalid evidence member")
                        destination=target/relative;data=archive.extractfile(member).read()
                        if destination.exists() and destination.read_bytes()!=data:raise ValueError("collected evidence changed")
                        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        print(json.dumps({"job_id":job,"state":record.get("state"),"summary":report.get("summary"),"error":report.get("error"),"environment":report.get("environment",report)}))
    else:print(json.dumps({"job_id":job,"state":record.get("state"),"report":None}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("submit","collect"));parser.add_argument("target");args=parser.parse_args()
    (submit if args.action=="submit" else collect)(args.target)
