"""Posthoc shared-direction hypothesis from retained B2 arrays; no model run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def diagnose(root):
    report=json.loads((root/"report.json").read_text());rows={r["name"]:r for r in report["driver"]["target_rows"]}
    def data(name,port):return np.fromfile(root/"raw"/f"{name}-{port}.f32",dtype="<f4").astype(float)
    em,en=(data(p+"-dense","omitted") for p in ("Mira","Niko"));beta=float(np.dot(em,en)/np.dot(em,em));left=en-beta*em
    result={"schema":"tcm-posthoc-reuse-hypothesis-v1","report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),
        "scope":"known-context posthoc linear-image forecast; prospective nonlinear write and transfer remain unmeasured here",
        "residual_cosine":float(np.dot(em,en)/(np.linalg.norm(em)*np.linalg.norm(en))),"beta":beta,
        "remaining_fraction_of_omitted_norm":float(np.linalg.norm(left)/np.linalg.norm(en)),
        "remaining_fraction_of_receiver_norm":float(np.linalg.norm(left)/rows["Niko-dense"]["receiver_norm"]),"channels":{}}
    scale=beta*rows["Niko-dense"]["rho"]/rows["Mira-dense"]["rho"]
    for channel in ("gate","up"):
        dense=data("Niko-dense","mlp_"+channel);project=data("Niko-project","mlp_"+channel);forecast=project+scale*data("Mira-correct","correction-"+channel)
        result["channels"][channel]={"b2_relative_error":float(np.linalg.norm(project-dense)/np.linalg.norm(dense)),"posthoc_extra_direction_relative_error":float(np.linalg.norm(forecast-dense)/np.linalg.norm(dense))}
    (root/"residual-transfer-hypothesis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("root",type=Path);print(json.dumps(diagnose(parser.parse_args().root)))
