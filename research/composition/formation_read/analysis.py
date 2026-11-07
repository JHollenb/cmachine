"""Authenticated historical joins and signed episode-policy factorials."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from whole_turn_analysis import validate_turn,token_hash,final_history,steps
from legacy_analysis import same,distance
D,VOCAB=1536,151936
TASKS=("correction","target","collateral")
HERE=Path(__file__).resolve().parent

def factorial(y00,y10,y01,y11):
    arrays=[np.asarray(y,dtype=np.float64) for y in (y00,y10,y01,y11)]
    if len({a.shape for a in arrays})!=1 or not all(np.isfinite(a).all() for a in arrays):raise ValueError("Invalid factorial arrays")
    n,pn,np_,pp=arrays;a=pn-n;b=np_-n;i=pp-pn-np_+n;d=pp-n
    dot=lambda x,y:float(np.vdot(x,y));norm=lambda x:float(np.linalg.norm(x));d2=dot(d,d);native=norm(n)
    terms={"formation_power":dot(a,a),"question_power":dot(b,b),"interaction_power":dot(i,i),"formation_question_cross":2*dot(a,b),"formation_interaction_cross":2*dot(a,i),"question_interaction_cross":2*dot(b,i)}
    return {"relative_l2_vs_NN":{label:norm(y-n)/native if native else None for label,y in zip(("NN","PN","NP","PP"),arrays)},"norms":{"formation":norm(a),"question":norm(b),"interaction":norm(i),"joint":norm(d),"formation_after_project_question":norm(pp-np_),"question_after_project_formation":norm(pp-pn)},"interaction_over_joint":norm(i)/norm(d) if d2 else None,"signed_projections_on_joint":{label:dot(x,d)/d2 if d2 else None for label,x in (("formation",a),("question",b),("interaction",i))},"signed_energy":terms,"joint_power":d2,"power_closure_residual":d2-sum(terms.values()),"pointwise_closure_max_abs":float(np.max(np.abs(d-(a+b+i))))}

def histories_match(turns,boundary):
    def history(t):return t["parent_history_ids"]+[s["input_id"] for s in (t["steps"] if boundary=="prefill" else steps(t))]
    histories=[history(t) for t in turns]
    return all(h==histories[0] for h in histories[1:])

def analyze(raw,bank,driver,texts):
    raw=Path(raw);ref=json.loads((HERE/"origin.json").read_text())
    if driver["schema"]!="tcm-formation-read-driver-v1" or driver["layer"]!=16 or driver["capacity"]!=512 or not driver["source_replay_checked_before_candidates"] or driver["historical_endpoint_vectors_read_by_driver"]:raise ValueError("Hybrid contract differs")
    if driver["source_history_ids"]!=ref["source_history_ids"] or driver["source_kv_sha256"]!=ref["source_kv_sha256"]:raise ValueError("Source parent differs")
    def load_new(stem,port,count=D):
        path=raw/f"{stem}-{port}.f32";a=np.fromfile(path,dtype="<f4")
        if a.shape!=(count,) or not np.isfinite(a).all():raise ValueError("Invalid new port/readout")
        return a
    def load_old(stem,port,count):
        name=f"{stem}-{port}.f32";path=HERE/"historical"/name;record=ref["raw_records"][name]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError("Historical readout identity differs")
        a=np.fromfile(path,dtype="<f4")
        if a.shape!=(count,) or not np.isfinite(a).all():raise ValueError("Invalid historical field")
        return a
    arms=driver["arms"]
    if [a["name"] for a in arms]!=["PN","NP"]:raise ValueError("Hybrid membership differs")
    observations=[];turns_by_name={};replay_checks=0;local=[];work={"transitions":0,"gate_up_calls":0,"down_calls":0,"compile_calls":0}
    for arm in arms:
        fpolicy,rpolicy=(2,0) if arm["name"]=="PN" else (0,2)
        if arm["formation_policy"]!=fpolicy or arm["reader_policy"]!=rpolicy or not arm["formation_checked_before_readers"] or not arm["source_prefix_intact"] or len(arm["turns"])!=3:raise ValueError("Episode policy/parent differs")
        correction=arm["turns"][0];original=ref["formations"]["project" if fpolicy else "native"]
        if correction["parent_history_ids"]!=driver["source_history_ids"] or correction["parent_kv_sha256"]!=driver["source_kv_sha256"]:raise ValueError("Correction source differs")
        if final_history(correction)!=final_history(original) or correction["final_position"]!=original["final_position"] or steps(correction)!=steps(original):raise ValueError("Correction replay history/metadata differs")
        for name,record in ref["raw_records"].items():
            if name.startswith(original["name"]+"-"):
                current=raw/(correction["name"]+name[len(original["name"]):])
                if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest()!=record["sha256"] or current.stat().st_size!=record["bytes"]:raise ValueError("Correction port replay differs")
                replay_checks+=1
        for boundary in ("prefill","final"):
            if correction[boundary]!=original[boundary] or correction[boundary+"_kv_sha256"]!=original[boundary+"_kv_sha256"]:raise ValueError("Correction complete boundary differs")
        for task,t in zip(TASKS,arm["turns"]):
            if t["name"]!=arm["name"]+"-"+task or t["policy"]!=(fpolicy if task=="correction" else rpolicy) or not t["autonomous"]:raise ValueError("Turn identity differs")
            validate_turn(t);turns_by_name[t["name"]]=t
            q=next(q for q in ref["inputs"]["questions"] if q["name"]==task)
            if t["template_ids"]!=q["ids"] or list(map(int,(raw/q["file"]).read_text().split()))!=q["ids"]:raise ValueError("Declared template differs")
            if task!="correction" and (t["parent_history_ids"]!=final_history(correction) or t["parent_kv_sha256"]!=correction["final_kv_sha256"]):raise ValueError("Reader did not use own correction parent")
            for row in steps(t):
                stem=f"{t['name']}-s{row['step']:03d}";receiver,write,out=(load_new(stem,p) for p in ("raw_receiver","mlp_return","output"));load_new(stem,"input");load_new(stem,"mlp_input");load_new(stem,"kv16-17",1024)
                if not same(out,np.add(receiver,write)):raise ValueError("Actual skip differs")
                if row["substitution"]:load_new(stem,"reconstruction");load_new(stem,"omitted")
                work["transitions"]+=1
                for field in ("gate_up_calls","down_calls","compile_calls"):work[field]+=row[field]
            for boundary,step in (("prefill",t["steps"][-1]),("final",t["generation_steps"][-1])):
                y=load_new(t["name"],boundary,VOCAB)
                if int(np.argmax(y))!=t[boundary]["top"] or t[boundary]["top"]!=step["top"] or hashlib.sha256(y.tobytes()).hexdigest()!=t[boundary]["sha256"]:raise ValueError("New readout/selection differs")
            expected={"correction":"OK","target":"green","collateral":"red"}[task]
            observations.append({"kind":"behavior","policy":arm["name"],"task":task,"text":texts[t["name"]],"answer_ids":t["answer_ids"],"expected_literal":expected,"literal_match":texts[t["name"]].strip().casefold()==expected.casefold(),"stopped":t["stopped"],"truncated":t["truncated"]})
            if task!="correction":
                old=ref["readers"]["NN" if rpolicy==0 else "PP"][task]
                for new_step,old_step in zip(steps(t),steps(old)):
                    common=new_step["history_sha256"]==old_step["history_sha256"]
                    if not common:continue
                    new=f"{t['name']}-s{new_step['step']:03d}";past=f"{old['name']}-s{old_step['step']:03d}"
                    exact={port:hashlib.sha256((raw/f"{new}-{port}.f32").read_bytes()).hexdigest()==ref["raw_records"][f"{past}-{port}.f32"]["sha256"] for port in ("input","raw_receiver","mlp_input","mlp_return","output")}
                    kv,prior_kv=load_new(new,"kv16-17",1024),load_old(past,"kv16-17",1024)
                    exact["kv16"]=same(kv[np.r_[0:256,512:768]],prior_kv[np.r_[0:256,512:768]])
                    exact["kv17"]=same(kv[np.r_[256:512,768:1024]],prior_kv[np.r_[256:512,768:1024]])
                    row={"kind":"matched-history-local-control","policy":arm["name"],"task":task,"step":new_step["step"],"position":new_step["position"],"reference":"NN" if rpolicy==0 else "PP","exact":exact};local.append(row);observations.append(row)
    panels=[]
    for task in ("target","collateral"):
        old_n,old_p=ref["readers"]["NN"][task],ref["readers"]["PP"][task]
        new_pn,new_np=turns_by_name["PN-"+task],turns_by_name["NP-"+task]
        worlds=(old_n,new_pn,new_np,old_p)
        for boundary in ("prefill","final"):
            arrays=(load_old(old_n["name"],boundary,VOCAB),load_new(new_pn["name"],boundary,VOCAB),load_new(new_np["name"],boundary,VOCAB),load_old(old_p["name"],boundary,VOCAB))
            detail=factorial(*arrays);detail.update(task=task,boundary=boundary,common_consumed_history=histories_match(worlds,boundary),positions={label:(t["parent_position"]+len(t["steps"]) if boundary=="prefill" else t["final_position"]) for label,t in zip(("NN","PN","NP","PP"),worlds)},scope="complete episode-policy effects; fixed-history only where actual input histories match")
            detail["worlds"]={}
            for label,t,y in zip(("NN","PN","NP","PP"),worlds,arrays):
                top=int(np.argmax(y));other=y.copy();other[top]=-np.inf
                detail["worlds"][label]={"top":top,"top_vs_rest_margin":float(y[top]-other.max()),"answer_ids":t["answer_ids"],"stopped":t["stopped"],"truncated":t["truncated"],"red_minus_blue":float(y[1151]-y[12203]),"green_minus_blue":float(y[13250]-y[12203]),"historical":label in ("NN","PP")}
            panels.append(detail);observations.append({"kind":"episode-factorial","task":task,"boundary":boundary,"common_consumed_history":detail["common_consumed_history"],**{k:detail[k] for k in ("relative_l2_vs_NN","norms","interaction_over_joint","signed_projections_on_joint","signed_energy","joint_power","power_closure_residual","pointwise_closure_max_abs")}})
    summary={"new_hybrid_policies":2,"new_turns":6,"historical_endpoint_policies":2,"correction_replay_file_checks":replay_checks,"matched_history_local_comparisons":len(local),"all_local_controls_exact":all(all(r["exact"].values()) for r in local),"behavior":[r for r in observations if r["kind"]=="behavior"],"factorials":[{k:p[k] for k in ("task","boundary","common_consumed_history","relative_l2_vs_NN","interaction_over_joint")} for p in panels]}
    return {"schema":"tcm-formation-read-analysis-v1","summary":summary,"work":work,"panels":panels,"local_controls":local,"observations":observations,"terminal_status":"not-requested"}
