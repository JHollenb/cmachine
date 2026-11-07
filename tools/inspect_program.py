"""Derive a symbol/consumer table from a sealed C program source map.

Optional trace rows join numerical observations to exact source operations.
These are structural and observational records, never learned-role certificates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from program import canonical

ARITY={"constant":(),"linear":("a",),"add":("a","b"),"multiply":("a","b"),
       "remainder":("a",),"cos":("a",),"affine":("a",),"concat":("a","b"),
       "slice":("a",),"rms":("a",),"embedding":("a",),"causal-conv-silu":("a",),
       "selective-scan":("a","b","c","d"),"silu":("a",)}

def inspect(mapping, *, trace=None, row=0, job_id=None):
    body=dict(mapping);seal=body.pop("fingerprint")
    if hashlib.sha256(canonical(body)).hexdigest()!=seal:
        raise ValueError("source map fingerprint differs")
    symbols=[{**v,"value_id":i,"producer":None,"consumers":[]} for i,v in enumerate(mapping["values"])]
    operations=mapping["instructions"]
    for pc,op in enumerate(operations):
        if op["kind"] not in ARITY:raise ValueError("unknown instruction arity")
        symbol=symbols[op["out"]]
        if symbol["producer"] is not None:raise ValueError("multiple SSA producers")
        symbol["producer"]={"pc":pc,"kind":op["kind"],"source_ref":op["source"]}
        for field in ARITY[op["kind"]]:symbols[op[field]]["consumers"].append(pc)
    report={"schema":"tcm-program-symbol-table-v1","source_map_fingerprint":seal,
            "program_bundle_digest":mapping["bundle"]["content_sha256"],"source_identity":mapping["source_identity"],
            "symbols":symbols,"state_slots":mapping["state_slots"],
            "occurrence_key_fields":["program_bundle_digest","job_id","input_row","pc","value_id"],
            "authority":{"mechanics_status":"source-map-validated","instrument_status":"no-trace-supplied",
                         "trend_status":"not-analyzed","terminal_status":"no-learned-mathematical-role-claim"}}
    if trace is not None:
        import numpy as np
        width=sum(v["length"] for v in symbols)
        trace=Path(trace)
        if row<0 or not trace.stat().st_size or trace.stat().st_size%(4*width):raise ValueError("trace row/extent differs")
        data=np.memmap(trace,mode="r",dtype="<f4").reshape(-1,width)
        if row>=len(data):raise ValueError("trace row outside evidence")
        for symbol in symbols:
            x=data[row,symbol["offset"]:symbol["offset"]+symbol["length"]]
            if not np.isfinite(x).all():raise ValueError("non-finite trace evidence")
            symbol["observation"]={"job_id":job_id,"input_row":row,"l2":float(np.linalg.norm(x.astype(np.float64))),
                                   "min":float(x.min()),"max":float(x.max()),"mean":float(x.astype(np.float64).mean())}
        digest=hashlib.sha256()
        with trace.open("rb") as f:
            for block in iter(lambda:f.read(1<<20),b""):digest.update(block)
        report["trace"]={"sha256":digest.hexdigest(),"rows":len(data),"selected_row":row}
        report["authority"]["instrument_status"]="trace-row-observed"
    report["fingerprint"]=hashlib.sha256(canonical(report)).hexdigest()
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trace",type=Path);parser.add_argument("--row",type=int,default=0);parser.add_argument("--job-id")
    args=parser.parse_args()
    result=inspect(json.loads(args.mapping.read_text()),trace=args.trace,row=args.row,job_id=args.job_id)
    with args.output.open("xb") as f:f.write(canonical(result)+b"\n")
    print(json.dumps({"symbols":len(result["symbols"]),"state_slots":len(result["state_slots"]),"output":str(args.output)}))
