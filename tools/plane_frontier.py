"""Offline layerwise continuation-frontier analysis; no model execution.

Prediction written while job-b18afb3e1bbc was queued: changing token t's MLP16
can change its KV17+, but common token t+1 forms KV17 before reading that changed
memory. Its KV17 should therefore remain exact; later KV18+ can differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(root):
    report=json.loads((root/"report.json").read_text())
    def read(name,clock):
        path=root/"raw"/f"{name}-{clock}-kv.f32"
        data=np.fromfile(path,dtype="<f4")
        if data.size!=5632 or not np.isfinite(data).all():raise ValueError("invalid frontier raw array")
        return data.reshape(2,11,2,128)
    result={"schema":"tcm-cache-frontier-observation-v1","job_id":root.name,
        "report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),
        "analyzer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction":"same-history next-token layer17 KV stays exact after a previous-token MLP16 edit; higher layers may differ",
        "scope":"captured current-token and one common-next-token rows, layers17-27 only; no raw lower-layer cache observation", "rows":[]}
    for row in report["driver"]["rows"]:
        record={"name":row["name"],"dose":row["dose"],"clocks":{}}
        for clock in ("current","next"):
            a,b=read(row["name"],clock),read("native",clock);layers=[]
            for i in range(11):
                metrics={"layer":17+i}
                for kind,label in enumerate(("key","value")):
                    x,y=a[kind,i],b[kind,i];delta=x.astype(float)-y.astype(float)
                    metrics[label]={"bytes_equal":x.tobytes()==y.tobytes(),"max_abs":float(np.max(np.abs(delta))),"relative_l2":float(np.linalg.norm(delta)/max(np.linalg.norm(y.astype(float)),1e-30))}
                layers.append(metrics)
            record["clocks"][clock]=layers
        result["rows"].append(record)
    result["summary"]={"next_layer17_exact_rows":sum(r["clocks"]["next"][0]["key"]["bytes_equal"] and r["clocks"]["next"][0]["value"]["bytes_equal"] for r in result["rows"]),"rows":len(result["rows"])}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path);args=parser.parse_args()
    result=analyze(args.directory);path=args.directory/"continuation-frontier.json";path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["summary"]))
