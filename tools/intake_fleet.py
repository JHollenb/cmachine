"""Finite source-custodied mrun launches for model intake, and safe collection."""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
ROOT=Path(__file__).resolve().parents[1]

def chat_model():
    # Correct repository identity with a sizing label until the normalized
    # artifact is inventory-qualified. No model weights are loaded on submit.
    from mrun.models import REGISTRY,ModelSpec
    from mamba_chat_source import PROFILES,DEFAULT_PROFILE
    profile=PROFILES[DEFAULT_PROFILE]
    name="cmachine-"+DEFAULT_PROFILE
    REGISTRY[name]=ModelSpec(name,profile["repository"],"mamba",profile["sizing_label"])
    return name

def submit(mode):
    from mrun.client.submit import launch
    with tempfile.TemporaryDirectory(prefix="tcm-intake-") as temporary:
        stage=Path(temporary)
        shutil.copytree(ROOT/"src",stage/"src")
        shutil.copy2(ROOT/"Makefile",stage/"Makefile")
        (stage/"tools").mkdir()
        for name in ("bundle.py","program.py","program_chat.py","inspect_program.py","intake_worker.py","mamba_chat_source.py"):
            shutil.copy2(ROOT/"tools"/name,stage/"tools"/name)
        if mode=="pan":
            pan=ROOT.parent/"phase-accumulator-network-research"
            shutil.copytree(pan/"src/pan",stage/"pan",ignore=shutil.ignore_patterns("__pycache__"))
            (stage/"inputs").mkdir()
            shutil.copy2(pan/"data/20260418_results/primes_primary_k/model_P113-s0-e68f0b94a1.pt",stage/"inputs/pan.pt")
        elif mode not in ("mamba-chat","probe"):raise ValueError(mode)
        sources={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob("*")) if p.is_file()}
        config={"schema":"tcm-model-intake-experiment-v1","mode":mode,"sources":sources,
                "device":"cpu","backend":"hf" if mode=="mamba-chat" else "c-fp32-cpu",
                "candidate_backend":"c-openblas-fp32" if mode=="mamba-chat" else "c-fp32-cpu",
                "dtype":"fp32","cpu_threads":2 if mode=="mamba-chat" else 1,
                "backend_options":{"physical_batch_size":1,"full_prefill_logits":True},"seq_len":64,
                "physical_batch_size":1,"physical_work":"two native prefills, two cached own paths<=12tokens each and two full-prefix checks; release native bank; one C export; two C own paths plus suffix replay; one zero-writer arm plus replay; prefix+consumed<=64" if mode=="mamba-chat" else "one FX shape probe; one PAN source all-pairs batch; two C all-pairs arms; one first-row boundary capture",
                "max_retries":0}
        if mode=="mamba-chat":
            from mamba_chat_source import PROFILES,DEFAULT_PROFILE
            profile=PROFILES[DEFAULT_PROFILE]
            config.update(source_repository=profile["repository"],source_revision=profile["revision"],
                          source_weights_sha256=profile["sha256"],task_family="decode",max_seq_len=64,
                          build_dependency="scipy-openblas32==0.3.34.481.3",
                          acquisition={"transport":"http-no-xet","no_write_timeout_s":120,"budget_s":1200,
                                       "preserved_stalled_job":"job-372c21791726"},
                          lifecycle="acquire once; native FP32 bank then release; C compiler and executor sequential; never duplicate full banks")
            # The actual Torch+compiler floor was920.5MB in job-7373b61f6362,
            # above mrun's generic410MB prior. Preserve the planner but declare
            # this measured backend exception: floor+FP32 bank+source mmap+64MB
            # bounded workspaces, x1.2, rounded upward to64MB =>2048MB.
            config["reservation_measurement"]={"floor_job":"job-7373b61f6362","floor_mb":920.5,
                                                "fp32_bank_mb":profile["fp32_bank_bytes"]/1e6,"source_mmap_mb":profile["bytes"]/1e6,
                                                "workspace_mb":64,"factor":1.2,"round_mb":64}
            measured_ram=64*math.ceil((920.5+profile["fp32_bank_bytes"]/1e6+profile["bytes"]/1e6+64)*1.2/64)
            measured_disk=2*math.ceil((2*profile["bytes"]+profile["fp32_bank_bytes"]+128e6)/2e9)
        # Beast's global uv cache crosses mount points and cannot atomically
        # install overlay wheels (job-68dec70f2ede). A lease-local cache keeps
        # temporary and final wheel paths on the same filesystem.
        cmd=("env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=60 uv --cache-dir .uv-cache run --project {env} --with scipy-openblas32==0.3.34.481.3 python tools/intake_worker.py " if mode=="mamba-chat" else "uv run --offline --project {env} python tools/intake_worker.py ")+mode
        if mode=="probe":
            cmd="uv run --offline --project {env} python tools/mamba_chat_source.py --probe"
            config.update(physical_work="two shared model-source metadata observations15seconds apart; no tensor loads or model forward",task_family="source-metadata-probe")
        (stage/"source-manifest.json").write_text(json.dumps(config,indent=2)+"\n")
        job=launch(cmd,
                   experiment="cmachine-model-intake-"+mode,payload=stage,prefer_host="beast",pin="beast",
                   needs={},config=config,model=chat_model() if mode=="mamba-chat" else None,
                   env_alias="mrun",timeout_s=1800 if mode=="mamba-chat" else 60 if mode=="probe" else 300,detach=True,retry_on_kill=False,
                   ram_mb=measured_ram if mode=="mamba-chat" else None,cpu_threads=2 if mode=="mamba-chat" else None,
                   disk_gb=measured_disk if mode=="mamba-chat" else None,
                   preflight=True,guarded=True,
                   note="MambaSan Japanese instruction checkpoint; boundedHTTP+normalization; two CPU FP32 native own<=12 paths/full-prefix checks, release bank; compile one C graph; two own C paths+one zero-writer arm/suffix replay; 2threads,seq<=64; noCUDA/training/retries; stop after panel" if mode=="mamba-chat" else "Metadata-only source probe: two filesystem size/mtime observations15seconds apart; no tensor load or forward; stop after second observation" if mode=="probe" else "PAN intake: one checkpoint load; one native all-pairs batch, two C all-pairs arms, first-row boundary capture; CPU FP32 one thread; stop after finite panel; no training/retries")
        target=ROOT/"results"/job;target.mkdir(parents=True,exist_ok=True)
        # Retain our own staged source bytes before the temporary directory is
        # removed; server payload retrieval is reserved to its authorized agent.
        shutil.copytree(stage,target/"payload",dirs_exist_ok=True)
    target=ROOT/"results"/job;target.mkdir(parents=True,exist_ok=True)
    (target/"submission.json").write_text(json.dumps({"job_id":job,"config":config},indent=2)+"\n")
    print(json.dumps({"job_id":job,"result_directory":str(target)}))

def collect(job):
    from mrun.client.api import Api
    api=Api();record=api.json("GET",f"/api/jobs/{job}")
    target=ROOT/"results"/job;target.mkdir(parents=True,exist_ok=True)
    # Larger causal panels opt in through the locally retained submission. The
    # remote report cannot increase the collector's extraction authority.
    submission=target/'submission.json'
    budget=json.loads(submission.read_text()).get('config',{}).get('evidence_budget_bytes',64*1024**2) if submission.exists() else 64*1024**2
    if type(budget) is not int or not 0<budget<=256*1024**2:raise ValueError('invalid evidence budget')
    (target/"job.json").write_text(json.dumps(record,indent=2)+"\n")
    _,logs,_=api.request("GET",f"/api/jobs/{job}/logs?offset=0&wait_s=1",timeout_s=30)
    (target/"logs.txt").write_bytes(logs)
    report=None;expanded_bytes=0
    for line in logs.decode().splitlines():
        if line.startswith("TCM_RESULT_GZIP_B64="):
            report=json.loads(gzip.decompress(base64.b64decode(line.split("=",1)[1])))
        if line.startswith("TCM_FILES_TGZ_B64="):
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(line.split("=",1)[1])),mode="r:gz") as archive:
                members=archive.getmembers()
                expanded_bytes+=sum(m.size for m in members)
                if expanded_bytes>budget:raise ValueError("evidence budget exceeded")
                for m in members:
                    relative=Path(m.name)
                    # The first worker used absolute cwd paths in its tar member
                    # names. Recover only files explicitly listed in that sealed
                    # report, under raw/, without rerunning the experiment.
                    if relative.parts and relative.parts[0]!="raw" and report:
                        expected={str(Path(k)).lstrip("/"): "raw/"+Path(k).name for k in report.get("raw",{})}
                        parents={str(Path(k).parent).lstrip("/") for k in report.get("raw",{})}
                        for parent in parents:expected[parent+"/report.json"]="raw/report.json"
                        if m.name not in expected:raise ValueError("unlisted legacy evidence member")
                        relative=Path(expected[m.name])
                    if not m.isfile() or relative.is_absolute() or ".." in relative.parts or relative.parts[0]!="raw":raise ValueError("invalid evidence member")
                    path=target/relative;data=archive.extractfile(m).read()
                    if path.exists() and path.read_bytes()!=data:raise ValueError("immutable evidence changed")
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    if report:
        (target/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        summary=dict(report);summary.pop("raw",None);summary.pop("executed_sources",None)
        if "result" in summary:
            summary["result"]=dict(summary["result"]);summary["result"].pop("source",None)
        print(json.dumps({"state":record.get("state"),"report":summary}))
    else:print(json.dumps({"state":record.get("state"),"report":None,"tail":logs.decode()[-3000:]}))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("submit","collect"));parser.add_argument("target");args=parser.parse_args()
    (submit if args.action=="submit" else collect)(args.target)
