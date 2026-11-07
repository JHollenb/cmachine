"""Additive saved-only joins and trend analysis; does not alter the executed report."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(root):
    root=Path(root);report=json.loads((root/"report.json").read_text());verification=json.loads((root/"verification.json").read_text());job=json.loads((root/"job.json").read_text());submission=json.loads((root/"submission.json").read_text())
    if sha(root/"report.json")!=verification["report_sha256"]:raise ValueError("Verification report join differs")
    for key in ("sources","source_manifest_sha256"):
        if job["config"][key]!=submission["config"][key]:raise ValueError("Server/client source identity differs")
    questions={q["name"]:q for q in report["inputs"]["questions"]};driver=report["driver"]
    joins=0
    for diagnostic,groups in ((False,driver["arms"]),(True,driver["diagnostics"])):
        for group in groups:
            for task,turn in zip(("correction","target","collateral"),group["turns"]):
                q=questions[task];file_ids=list(map(int,(root/"raw"/q["file"]).read_text().split()))
                if q["ids"]!=file_ids or turn["template_ids"]!=file_ids or turn["budget"]!=q["budget"]:raise ValueError("Declared input/template join differs")
                joins+=1
    def array(name,port):
        path=root/"raw"/f"{name}-{port}.f32";record=report["raw_files"][path.name]
        if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError("Audit array identity differs")
        return np.fromfile(path,dtype="<f4")
    replay_checks=0
    for diagnostic in driver["diagnostics"]:
        label=diagnostic["candidate"]
        for task,turn in zip(("correction","target","collateral"),diagnostic["turns"]):
            native=next(t for t in driver["arms"][0]["turns"] if t["name"]=="native-"+task)
            if turn["parent_history_ids"]!=native["parent_history_ids"] or [s["input_id"] for s in turn["steps"]+turn["generation_steps"]]!=[s["input_id"] for s in native["steps"]+native["generation_steps"]]:raise ValueError("New trend assumes matching native histories")
            for row in turn["steps"]+turn["generation_steps"]:
                a=f"{turn['name']}-s{row['step']:03d}";b=f"{native['name']}-s{row['step']:03d}"
                for port in ("input","raw_receiver","mlp_input","mlp_return","output","kv16-17"):
                    if array(a,port).tobytes()!=array(b,port).tobytes():raise ValueError("Native dense diagnostic replay differs")
                    replay_checks+=1
            for port in ("prefill","final"):
                if array(turn["name"],port).tobytes()!=array(native["name"],port).tobytes() or turn[port+"_kv_sha256"]!=native[port+"_kv_sha256"]:raise ValueError("Dense diagnostic complete boundary differs")
                replay_checks+=2
    rows=[]
    for arm in driver["arms"]:
        for task,turn in zip(("correction","target","collateral"),arm["turns"]):
            logits=array(turn["name"],"prefill");top=int(logits.argmax());other=logits.copy();other[top]=-np.inf
            comparison=next((c for c in report["analysis"]["same_text_comparisons"] if c["policy"]==arm["name"] and c["task"]==task),None)
            rows.append({"policy":arm["name"],"task":task,"text":report["texts"][turn["name"]],"selected_token_id":top,"top_vs_rest_margin":float(logits[top]-other.max()),"green_minus_blue":float(logits[13250]-logits[12203]),"prefill_logit_relative_l2":comparison["boundaries"]["prefill"]["logit_difference"]["relative_l2"] if comparison else 0.,"post_consumed_eos_logit_relative_l2":comparison["boundaries"]["final"]["logit_difference"]["relative_l2"] if comparison else 0.})
    values=[r["write_difference"]["relative_l2"] for c in report["analysis"]["same_text_comparisons"] if c["policy"]=="project" for r in c["steps"]]
    stderr=(root/"raw/driver.stderr.txt").read_text().splitlines();seal_index=next(i for i,line in enumerate(stderr) if line.startswith("sealed all primary"))
    if sum(line.startswith("completed ") for line in stderr[:seal_index])!=12 or not all(not line.startswith("completed diag-") for line in stderr[:seal_index]) or sum(line.startswith("completed diag-") for line in stderr[seal_index+1:])!=9:raise ValueError("Observed diagnostic chronology differs")
    signed=[]
    for task in ("correction","target","collateral"):
        for port in ("prefill","final"):
            native,project,zero=(array(label+"-"+task,port).astype(np.float64) for label in ("native","project","zero"))
            e_zero=zero-native;delta=project-zero;e_project=project-native
            z2=float(e_zero@e_zero);d2=float(delta@delta);cross=float(2*(e_zero@delta));p2=float(e_project@e_project)
            signed.append({"task":task,"boundary":port,"zero_error_power":z2,"added_policy_effect_power":d2,"twice_error_effect_inner_product":cross,"project_error_power":p2,"power_closure_residual":p2-(z2+d2+cross),"pointwise_closure_max_abs":float(np.max(np.abs(e_project-(e_zero+delta)))),"added_effect_over_zero_error_power":d2/z2,"cross_over_zero_error_power":cross/z2,"project_over_zero_error_power":p2/z2,"scope":"complete PROJECT versus zero policy effect, including formed cache; not an isolated writer attribution"})
    result={"schema":"tcm-whole-turn-additive-audit-v2","report_sha256":sha(root/"report.json"),"verification_sha256":sha(root/"verification.json"),"audit_source_sha256":sha(__file__),"models_evaluated":0,"declared_template_joins":joins,"dense_diagnostic_native_exact_checks":replay_checks,"primary_seal_chronology_checked":True,"project_local_worse_than_zero":{"count":sum(v>1 for v in values),"total":len(values)},"boundary_rows":rows,"signed_policy_effects":signed,"scope":"posthoc saved-only audit; v1, initial report and executed analysis unchanged","terminal_status":"not-requested"}
    (root/"audit-source-v2.py").write_bytes(Path(__file__).read_bytes())
    (root/"additive-audit-v2.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(json.dumps({k:v for k,v in result.items() if k not in ("boundary_rows","signed_policy_effects")}));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();audit(a.root)
