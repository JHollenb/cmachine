"""Post-hoc signed receiver/MLP interaction account; no model execution."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(root):
    root=Path(root);report=json.loads((root/"report.json").read_text());verification=json.loads((root/"verification.json").read_text());fingerprint=sha(root/"report.json")
    if verification["report_sha256"]!=fingerprint:raise ValueError("Verified report differs")
    rows=[];raw_records={};dot=lambda a,b:math.fsum(float(x)*float(y) for x,y in zip(a,b))
    for row in report["analysis"]["attention_reads"]:
        task,position=row["task"],row["position"];ports={}
        for port in ("raw_receiver","mlp_return","output"):
            arrays=[]
            for world in ("NN","PN","NP","PP"):
                name=f"{world}-{task}-p{position}-{port}.f32";path=root/"raw"/name;record=report["raw_files"][name]
                if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError("Raw audit dependency differs")
                raw_records[name]=record;a=np.fromfile(path,dtype="<f4").astype(float)
                if a.shape!=(1536,) or not np.isfinite(a).all():raise ValueError("Raw extent/finite contract differs")
                arrays.append(a)
            ports[port]=arrays[3]-arrays[1]-arrays[2]+arrays[0]
        a,m,y=(ports[k] for k in ("raw_receiver","mlp_return","output"));e=y-a-m
        rows.append({"task":task,"position":position,"receiver_interaction_power":dot(a,a),"mlp_interaction_power":dot(m,m),"receiver_mlp_cross":2*dot(a,m),"output_interaction_power":dot(y,y),"addition_remainder_power":dot(e,e),"receiver_remainder_cross":2*dot(a,e),"mlp_remainder_cross":2*dot(m,e),"addition_remainder_l2":math.sqrt(dot(e,e)),"pointwise_account_closure_max_abs":float(np.max(np.abs(y-(a+m+e))))})
    result={"schema":"tcm-attention-mlp-interaction-audit-v1","report_sha256":fingerprint,"verification_sha256":sha(root/"verification.json"),"audit_source_sha256":sha(__file__),"post_hoc":True,"models_evaluated":0,"raw_dependencies":raw_records,"rows":rows,"terminal_status":"not-requested"}
    data=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n";target=root/"mlp-interaction-audit.json"
    if target.exists() and target.read_text()!=data:raise ValueError("Immutable saved audit differs")
    target.write_text(data);print(json.dumps({"audit":str(target),"rows":rows}));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();audit(a.root)
