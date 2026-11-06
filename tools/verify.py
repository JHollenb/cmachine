"""Offline custody, receipt and raw-readout verification; no model execution."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def cancellation_bound(record,count,key):
    """FP64 reduction/assembly closure; a mechanics contract with input scale."""
    if count<1:raise ValueError("empty cancellation identity")
    u=2.0**-53;steps=2*count+8
    if steps*u>=1:raise ValueError("unsupported cancellation extent")
    gamma=steps*u/(1-steps*u)
    if key=="identity_roundoff":
        scale=abs(record["before_squared"])+abs(record["after_squared"])+2*abs(record["uncentered_cross_inner_product"])+abs(record["error_change_squared"])
    elif key=="roundoff":
        if set(record["term_squared_norms"])!={"carry","read","interaction"} or len(record["twice_pair_inner_products"])!=3:raise ValueError("unsupported energy identity family")
        scale=abs(record["joint_squared_norm"])+sum(map(abs,record["term_squared_norms"].values()))+sum(map(abs,record["twice_pair_inner_products"].values()))
    else:raise ValueError("unknown cancellation diagnostic")
    bound=gamma*scale
    if not math.isfinite(scale) or not math.isfinite(bound) or not math.isfinite(record[key]):raise ValueError("nonfinite cancellation diagnostic")
    if abs(record[key])>bound:raise ValueError("FP64 cancellation identity exceeds declared forward-error bound")
    return bound


def verify(root):
    report=json.loads((root/"report.json").read_text())
    submission=json.loads((root/"submission.json").read_text())
    job=json.loads((root/"job.json").read_text())
    if job["job_id"]!=submission["job_id"]:raise ValueError("job identity differs")
    if sha(root/"logs.txt")!=job["result"]["artifacts"]["job_log"]["sha256"]:
        raise ValueError("collected log differs from scheduler log seal")
    source_report=None
    for line in (root/"logs.txt").read_text().splitlines():
        if line.startswith("TCM_RESULT_GZIP_B64="):
            source_report=json.loads(gzip.decompress(base64.b64decode(line.split("=",1)[1])))
    if report!=source_report:raise ValueError("collected report differs from sealed job log")
    if (root/"raw/report.json").exists() and json.loads((root/"raw/report.json").read_text())!=report:
        raise ValueError("direct raw report differs from logged report")
    checked_sources=0
    for name,digest in submission["config"]["sources"].items():
        path=root/"payload"/name
        if path.exists():
            if sha(path)!=digest:raise ValueError(f"payload source differs: {name}")
            checked_sources+=1
    if report.get("schema") in ("tcm-stack-experiment-v1","tcm-plane-experiment-v1","tcm-reuse-experiment-v1","tcm-enrichment-experiment-v1","tcm-components-experiment-v1","tcm-formation-experiment-v1","tcm-writer-experiment-v1","tcm-lineage-mechanics-v1") and checked_sources!=len(submission["config"]["sources"]):
        raise ValueError("assay source custody is incomplete")
    receipts=0
    for key in ("saturn_server_receipt","saturn_isa_receipt"):
        if key not in report:continue
        wrapper=report[key];body=dict(wrapper["receipt"]);fingerprint=body.pop("fingerprint")
        if hashlib.sha256(canonical(body)).hexdigest()!=fingerprint:raise ValueError("public receipt seal differs")
        if body["parent"]!=wrapper["before"] or body["result"]!=wrapper["after"]:raise ValueError("public receipt boundary differs")
        receipts+=1
    raw_readouts=0
    if report.get("schema")=="tcm-qwen-experiment-v1":
        import numpy as np
        for case in report["cases"]:
            name=case["name"];a=np.fromfile(root/"raw"/f"qwen-{name}-c-first.f32",dtype="<f4").astype(np.float64)
            b=np.load(root/"raw"/f"qwen-{name}-native-first.npy").astype(np.float64)
            error=float(np.max(np.abs(a-b)))
            if abs(error-case["comparison"]["first_logits_max_abs_error"])>1e-12:
                raise ValueError("raw readout comparison differs")
            raw_readouts+=1
    raw_ports=0
    if report.get("schema")=="tcm-lineage-mechanics-v1":
        import numpy as np
        for name,record in report["raw_files"].items():
            p=root/"raw"/name
            if sha(p)!=record["sha256"] or p.stat().st_size!=record["bytes"]:raise ValueError("lineage raw identity differs")
        left=np.fromfile(root/"raw/lineage-historical.f32",dtype="<f4");right=np.fromfile(root/"raw/lineage-candidate.f32",dtype="<f4")
        paths=report["ordinary_paths"]
        if left.shape!=(151936,) or right.shape!=left.shape or not np.isfinite(left).all() or not np.isfinite(right).all() or left.tobytes()!=right.tobytes():raise ValueError("lineage native readout parity differs")
        if [t["id"] for t in paths["historical"]["tokens"]]!=[t["id"] for t in paths["candidate"]["tokens"]] or not paths["candidate"]["checkpoint_exact"]:raise ValueError("lineage continuation parity differs")
        if not report["stack_mechanics"]["writer_parent_lineage_checks"] or report["summary"]["host_marker_bytes"]!=9216 or report["summary"]["new_scientific_samples"]!=0:raise ValueError("lineage mechanics instrument differs")
        raw_readouts=2
    if report.get("schema")=="tcm-writer-experiment-v1":
        from writer_analysis import verify_saved
        writer_audit=verify_saved(root,report);raw_readouts=writer_audit["raw_readouts"];raw_ports=writer_audit["raw_ports"]
    if report.get("schema") in ("tcm-plane-experiment-v1","tcm-reuse-experiment-v1","tcm-enrichment-experiment-v1","tcm-components-experiment-v1","tcm-formation-experiment-v1"):
        import math
        import numpy as np
        reuse=report["schema"] in ("tcm-reuse-experiment-v1","tcm-enrichment-experiment-v1","tcm-components-experiment-v1","tcm-formation-experiment-v1")
        if report["schema"]=="tcm-formation-experiment-v1":from formation_analysis import analyze
        elif reuse:from reuse_analysis import analyze
        else:from plane_analysis import analyze
        for name,record in report["raw_files"].items():
            path=root/"raw"/name
            if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError("plane raw identity/extent differs")
        driver_file="reuse-driver.json" if reuse else "plane-driver.json"
        if sha(root/"raw"/driver_file)!=report["driver_raw_sha256"]:raise ValueError("raw driver identity differs")
        unannotated=json.loads(json.dumps(report["driver"]))
        for row in unannotated["rows"]+unannotated.get("calibration_rows",[])+unannotated.get("target_rows",[]):
            row["own"].pop("text")
            for question in row["questions"]:question["output"].pop("text")
        if reuse:unannotated["rebuilt_source_own"].pop("text")
        if unannotated!=json.loads((root/"raw"/driver_file).read_text()):raise ValueError("raw driver/report join differs")
        if report["schema"]=="tcm-formation-experiment-v1" and any(row["input_token_id"]!=report["input"]["targets"][0]["ids"][-1] for row in report["driver"]["target_rows"]):raise ValueError("formation token/frozen input join differs")
        formation=report["schema"]=="tcm-formation-experiment-v1"
        consumer_version=1 if formation and report["consumer_contrasts"]["schema"].endswith("v1") else 2
        recomputed=analyze(root/"raw",report["driver"],consumer_version=consumer_version) if formation else analyze(root/"raw",report["driver"])
        roundoff_diagnostics=[]
        def compare(a,b,count=None,path=""):
            if isinstance(a,dict):
                if set(a)!=set(b):raise ValueError("analysis fields differ")
                count=a.get("count",count)
                for k in a:
                    if count is not None and ((k=="identity_roundoff" and "before_squared" in a) or (k=="roundoff" and "term_squared_norms" in a)):
                        # Numerical integrity only: forward-error bound for FP64
                        # dot/reduction plus scalar assembly, not a quality gate.
                        ba,bb=cancellation_bound(a,count,k),cancellation_bound(b,count,k)
                        roundoff_diagnostics.append({"path":path+"/"+k,"local":a[k],"producer":b[k],"local_bound":ba,"producer_bound":bb,"count":count})
                    else:compare(a[k],b[k],count,path+"/"+k)
            elif isinstance(a,list):
                if len(a)!=len(b):raise ValueError("analysis rows differ")
                for index,(x,y) in enumerate(zip(a,b)):compare(x,y,count,path+"/"+str(index))
            elif isinstance(a,float):
                if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-14):raise ValueError("plane raw arithmetic differs")
            elif a!=b:raise ValueError("plane observation differs")
        compare(recomputed["comparisons"],report["comparisons"])
        if report["schema"]=="tcm-formation-experiment-v1":compare(recomputed["consumer_contrasts"],report["consumer_contrasts"])
        for k,v in recomputed["summary"].items():compare(v,report["summary"][k])
        for row in report["driver"]["rows"]+report["driver"].get("calibration_rows",[])+report["driver"].get("target_rows",[]):
            first=root/"raw"/f"{row['name']}-first.f32"
            if sha(first)!=row["first"]["sha256"]:raise ValueError("first readout seal differs")
            a=np.fromfile(first,dtype="<f4")
            if int(np.argmax(a))!=row["first"]["top"]:raise ValueError("raw argmax differs")
            raw_readouts+=1
            raw_ports+=sum((root/"raw"/f"{row['name']}-{p}.f32").exists() for p in ("mlp_input","mlp_gate","mlp_up","mlp_return","output"))
        if report["schema"]=="tcm-components-experiment-v1":raw_ports+=3
        if report["schema"]=="tcm-formation-experiment-v1":raw_ports+=28
    if report.get("schema")=="tcm-stack-experiment-v1":
        import numpy as np
        for case in report["cases"]:
            name=case["name"];a=np.fromfile(root/"raw"/f"stack-{name}-historical.f32",dtype="<f4")
            b=np.fromfile(root/"raw"/f"stack-{name}-candidate.f32",dtype="<f4")
            if report["model"]["profile_version"]!="qwen25-fp32-execution-v2" or a.shape!=b.shape or len(a)!=151936:
                raise ValueError("raw readout geometry differs")
            error=float(np.max(np.abs(a.astype(float)-b.astype(float))))
            if error!=case["comparison"]["first_logits_max_abs_error"] or (a.tobytes()==b.tobytes())!=case["comparison"]["full_first_logits_bytes_equal"]:
                raise ValueError("raw stack readout comparison differs")
            raw_readouts+=1
            for port in case["ports"]:
                path=root/"raw"/port["file"]
                if sha(path)!=port["sha256"] or path.stat().st_size!=port["count"]*4:
                    raise ValueError("raw port identity/extent differs")
                raw_ports+=1
    result={"schema":"tcm-offline-verification-v1","job_id":submission["job_id"],"report_sha256":sha(root/"report.json"),
            "verifier_sha256":sha(Path(__file__)),
            "scheduler_log_seal":True,"report_log_join":True,"copied_payload_sources_verified":checked_sources,
            "public_receipts_verified":receipts,"raw_readouts_recomputed":raw_readouts,"raw_ports_verified":raw_ports,
            "scope":"custody, receipt identities and raw-readout arithmetic; scientific observations remain in the report"}
    if report.get("schema")=="tcm-formation-experiment-v1":
        result["fp64_cancellation_policy"]={"id":"tcm-f64-cancellation-v1","purpose":"verify floating reduction/assembly closure without comparing backend-dependent cancellation residuals to an unscaled absolute threshold","bound":"gamma_(2*n+8) times sum of absolute constituent powers/cross terms; u=2^-53","diagnostics":roundoff_diagnostics}
    # A small source-bound stream lets the standard evidence plane inspect this
    # custom experiment without pretending it understands every report field.
    observations=[]
    if report.get("schema")=="tcm-writer-experiment-v1":
        result["writer_audit"]=writer_audit
        observations=[{"schema":"tcm-observation-v1","kind":"observation","case":x["name"],"model_id":report["model"]["identity"],"metrics":x,"report_sha256":result["report_sha256"]} for x in report["comparisons"]]
        observations.extend({"schema":"tcm-observation-v1","kind":"observation","case":f"writer-consumer-{i}","model_id":report["model"]["identity"],"metrics":x,"report_sha256":result["report_sha256"]} for i,x in enumerate(report["consumer_contrasts"]["records"]))
    elif report.get("schema") in ("tcm-reuse-experiment-v1","tcm-enrichment-experiment-v1","tcm-components-experiment-v1","tcm-formation-experiment-v1"):
        observations=[{"schema":"tcm-observation-v1","kind":"observation","case":x["name"],"model_id":report["model"]["identity"],"metrics":x,"report_sha256":result["report_sha256"]} for x in report["comparisons"]]
        if report["schema"]=="tcm-components-experiment-v1":observations.append({"schema":"tcm-observation-v1","kind":"observation","case":"Niko-C-component-diagnostics","model_id":report["model"]["identity"],"metrics":report["summary"]["component_diagnostics"],"report_sha256":result["report_sha256"]})
    elif report.get("schema")=="tcm-plane-experiment-v1":
        observations=[{"schema":"tcm-observation-v1","kind":"observation","case":str(x["dose"]),"model_id":report["model"]["identity"],"metrics":x,"report_sha256":result["report_sha256"]} for x in report["comparisons"]]
    elif "cases" in report:
        observations=[{"schema":"tcm-observation-v1","kind":"observation","case":x["name"],"model_id":report["model"]["identity"],"metrics":x["comparison"],"report_sha256":result["report_sha256"]} for x in report["cases"]]
    elif "paths" in report:
        observations=[{"schema":"tcm-observation-v1","kind":"observation","case":str(x["index"]),"metrics":{"own_state":x["own_state"],"checkpoint_exact":x["checkpoint_exact"]},"report_sha256":result["report_sha256"]} for x in report["paths"]]
    else:
        observations=[{"schema":"tcm-observation-v1","kind":"observation","case":"continuation","metrics":report.get("summary",{}),"report_sha256":result["report_sha256"]}]
    if report.get("schema")=="tcm-components-experiment-v1":
        # Additive posthoc records. The original report/log/raw rows stay sealed.
        # Keep the executed full-correction endpoint distinct from the local
        # split-joint diagnostic, whose continuation was never executed.
        from consumer_contrasts import write as write_consumer_contrasts
        consumers=write_consumer_contrasts(root,result)
        consumer_sha=sha(root/"consumer-contrasts.json")
        result["consumer_contrasts"]={"records":len(consumers["records"]),
            "raw_files_verified":len(consumers["raw_files"]),"sha256":consumer_sha,
            "analyzer_sha256":consumers["analyzer_sha256"]}
        for index,row in enumerate(consumers["records"]):
            observations.append({"schema":"tcm-observation-v1","kind":"observation",
                "case":f"Niko-C-consumer-{index}","model_id":report["model"]["identity"],
                "report_sha256":result["report_sha256"],"consumer_analysis_sha256":consumer_sha,
                "binding":{k:v for k,v in row.items() if k!="metrics"},"metrics":row["metrics"]})
    if report.get("schema")=="tcm-formation-experiment-v1":
        from formation_analysis import analyze as formation_analyze
        corrected=formation_analyze(root/"raw",report["driver"],consumer_version=2)
        consumers={**corrected["consumer_contrasts"],"report_sha256":result["report_sha256"],
            "model":report["model"],"scope":"verified native formation panel; full-carried reference still omits layer16 read"}
        analysis_sources={}
        for name in ("formation_analysis.py","consumer_contrasts.py","reuse_analysis.py","verify.py"):
            path=Path(__file__).parent/name;snapshot=root/"analysis-source"/name;snapshot.parent.mkdir(exist_ok=True);snapshot.write_bytes(path.read_bytes());analysis_sources[name]=sha(path)
        consumers["analysis_sources"]=analysis_sources
        if consumer_version==1:
            corrections=root/"corrections";corrections.mkdir(exist_ok=True)
            (corrections/"consumer-reference-v2.json").write_text(json.dumps({"schema":"tcm-analysis-correction-v1","report_sha256":result["report_sha256"],"issue":"producer v1 full_carried_reference_error_l2.dense erroneously used the substituted full-carried reference as dense, reporting zero; native_error_l2 and all four intervention errors are unaffected","producer_schema":report["consumer_contrasts"]["schema"],"corrected_schema":consumers["schema"],"affected_records":[{"port":r["port"],"selection":r["selection"],"layer":r.get("layer"),"corrected_dense_vs_full_carried_l2":r["metrics"]["full_carried_reference_error_l2"]["dense"]} for r in consumers["records"]],"original_report_unchanged":True},indent=2)+"\n")
        (root/"formation-consumers.json").write_text(json.dumps(consumers,indent=2,allow_nan=False)+"\n")
        result["consumer_contrasts"]={"records":len(consumers["records"]),"sha256":sha(root/"formation-consumers.json")}
        observations.append({"schema":"tcm-observation-v1","kind":"observation","case":"Niko-D-formation-diagnostics",
            "model_id":report["model"]["identity"],"report_sha256":result["report_sha256"],"metrics":report["summary"]["formation_diagnostics"]})
        for index,row in enumerate(consumers["records"]):
            observations.append({"schema":"tcm-observation-v1","kind":"observation","case":f"Niko-D-consumer-{index}",
                "model_id":report["model"]["identity"],"report_sha256":result["report_sha256"],
                "binding":{**{k:v for k,v in row.items() if k!="metrics"},"role_definitions":consumers["definitions"]},"metrics":row["metrics"]})
    (root/"observations.jsonl").write_text("".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in observations))
    files={str(p.relative_to(root)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("verification.json","custody.json")}
    (root/"custody.json").write_text(json.dumps({"job_id":submission["job_id"],"source":"exact completed mrun job work directory and scheduler log","files":files},indent=2)+"\n")
    (root/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.directory)))
