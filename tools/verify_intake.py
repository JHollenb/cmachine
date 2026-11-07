"""Verify collected intake artifacts and recompute bounded raw observations.

Hash/source consistency is a mechanics requirement. Model scores are reported
without a discovery/promotion threshold. No model is loaded or executed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from inspect_program import inspect

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def verify(root):
    root=Path(root);report=json.loads((root/"report.json").read_text());job=json.loads((root/"job.json").read_text())
    if job["state"]!="succeeded" or report["status"]!="completed":
        raise ValueError("run is not completed; preserve it as incomplete/error evidence")
    if json.loads((root/"raw/report.json").read_text())!=report:raise ValueError("raw report differs")
    checked=0
    for name,record in report["raw"].items():
        path=root/"raw"/Path(name).name
        if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError(f"raw custody differs: {name}")
        checked+=1
    for name,digest in report["executed_sources"].items():
        if job["config"]["sources"].get(name)!=digest:raise ValueError(f"executed source differs: {name}")
    payload=root/"payload"
    if payload.exists():
        for name,digest in job["config"]["sources"].items():
            if sha(payload/name)!=digest:raise ValueError(f"retained payload source differs: {name}")
    observed={}
    result=report["result"]
    if report["mode"]=="pan":
        mapping=json.loads((root/"raw/pan.map.json").read_text());symbols=inspect(mapping)
        data=np.load(root/"raw/pan-evidence.npz",allow_pickle=False)
        native,actual,target=data["native_logits"],data["c_logits"],data["target"]
        if result["baseline"]["bundle_digest"]!=mapping["bundle"]["content_sha256"]:raise ValueError("C program identity differs")
        delta=actual-native
        arm_data=(np.load(root/"raw/pan-intervention.npz",allow_pickle=False)
                  if (root/"raw/pan-intervention.npz").exists() else data)
        observed={"rows":len(target),"native_correct":int((native.argmax(1)==target).sum()),
                  "c_correct":int((actual.argmax(1)==target).sum()),
                  "top1_agreement_count":int((actual.argmax(1)==native.argmax(1)).sum()),
                  "zero_gate_correct":int((arm_data["zero_gate_predictions"]==target).sum()) if "zero_gate_predictions" in arm_data else None,
                  "max_abs_logit_error":float(np.abs(delta).max()),"mean_abs_logit_error":float(np.abs(delta).mean()),
                  "symbols":len(symbols["symbols"])}
    elif report["mode"]=="mamba-chat":
        mapping=json.loads((root/"raw/mamba-chat.map.json").read_text());symbols=inspect(mapping)
        observed={"symbols":len(symbols["symbols"]),"state_slots":len(symbols["state_slots"]),"prompts":[]}
        for item in result["prompts"]:
            if item["c"]["bundle_digest"]!=mapping["bundle"]["content_sha256"]:raise ValueError("C program identity differs")
            data=np.load(root/("raw/mamba-"+item["name"]+".npz"),allow_pickle=False)
            native,actual=data["native_logits"],data["c_logits"]
            a,b=item["generated_tokens"],item["c"]["generated_tokens"]
            common=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            n=min(len(native),len(actual),len(item["input_ids"])+common)
            delta=actual[:n]-native[:n]
            observed["prompts"].append({"name":item["name"],"common_generated_prefix":common,
                                       "comparable_rows":n,"max_abs_logit_error":float(np.abs(delta).max()),
                                       "mean_abs_logit_error":float(np.abs(delta).mean()),
                                       "source_tokens":a,"c_tokens":b})
    else:raise ValueError("unknown experiment mode")
    return {"schema":"tcm-intake-verification-v1","job_id":job["job_id"],"raw_files_checked":checked,
            "retained_payload_checked":payload.exists(),"observations":observed,
            "mechanics_status":"artifact-hashes-source-bindings-and-map-verified",
            "instrument_status":"raw-logit-observations-recomputed",
            "trend_status":"reported-without-a-score-threshold","terminal_status":"no-general-capability-or-speed-claim"}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.directory),indent=2))
