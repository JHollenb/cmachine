"""Offline carried/read geometry and continuation frontier, bound to raw output."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(root):
    report=json.loads((root/"report.json").read_text());raw=root/"raw"
    qr=np.fromfile(raw/"reuse-qr.f64",dtype="<f8");extra=np.fromfile(raw/"enrich-qr.f64",dtype="<f8")
    columns=[qr[:1536],qr[1536:3072],extra[:1536]]
    def data(name,port):return np.fromfile(raw/f"{name}-{port}.f32",dtype="<f4")
    result={"schema":"tcm-reuse-components-v2","report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),"scope":"posthoc FP64 geometric attribution and raw selected-KV timing; no semantic or causal importance assigned","correction":"v1 frontier reshaped kind-major arrays as layer-major; v2 follows the driver's explicit kind,layer,head order; raw evidence and geometric calculations unchanged","components":[],"next_kv_frontiers":[]}
    for group in ("Niko-A","Niko-B"):
        name=group+"-dense";x,a,r=(data(name,p).astype(float) for p in ("entry","attention-write","receiver"))
        for rank in (2,3):
            def omit(v):
                out=v.copy()
                for q in columns[:rank]:out-=q*np.dot(q,v)
                return out
            ex,ea=omit(x),omit(a);e=ex+ea;den=float(np.dot(e,e))
            actual=data(group+("-project2" if rank==2 else "-project"),"omitted").astype(float)
            result["components"].append({"group":group,"rank":rank,"carry_signed_share":float(np.dot(ex,e)/den),"attention_signed_share":float(np.dot(ea,e)/den),"carry_squared_norm":float(np.dot(ex,ex)),"attention_squared_norm":float(np.dot(ea,ea)),"cross_term":float(2*np.dot(ex,ea)),"sum_squared_norm":den,
                "projected_receiver_addition_rounding_l2":float(np.linalg.norm(omit(r-x-a))),"ideal_operand_sum_to_executed_omission_l2":float(np.linalg.norm(e-actual))})
        for mode in ("project2","project","correct"):
            candidate=data(group+"-"+mode,"next-kv").reshape(2,11,256);native=data(group+"-dense","next-kv").reshape(2,11,256)
            equal=[candidate[:,i].tobytes()==native[:,i].tobytes() for i in range(11)]
            result["next_kv_frontiers"].append({"group":group,"mode":mode,"layers":list(range(17,28)),"byte_equal":equal,"first_changed_layer":next((17+i for i,v in enumerate(equal) if not v),None)})
    (root/"receiver-components-v2.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);print(json.dumps(analyze(p.parse_args().root)))
