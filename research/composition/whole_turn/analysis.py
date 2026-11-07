"""Saved-only audit: token authority, whole-turn policies and sealed same-text replay."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from legacy_analysis import seqdot,same,distance

D,H,VOCAB=1536,8960,151936
LABELS=("native","project","zero","corrected")
POLICIES=(0,2,4,3)
TASKS=("correction","target","collateral")

def steps(turn):return turn["steps"]+turn["generation_steps"]
def final_history(turn):return turn["parent_history_ids"]+[s["input_id"] for s in steps(turn)]
def token_hash(ids):return hashlib.sha256(np.asarray(ids,dtype="<i4").tobytes()).hexdigest()

def validate_turn(turn):
    parent=turn["parent_history_ids"];template=turn["template_ids"];skip=turn["template_skip"]
    if not parent or not template or template[0]!=151645 or skip!=int(parent[-1]==151645):raise ValueError("Template closure differs")
    if turn["host_supplied_closing_token"]!=(skip==0):raise ValueError("Host closure metadata differs")
    prefill=turn["steps"];generated=turn["generation_steps"];answers=turn["answer_ids"]
    if [s["input_id"] for s in prefill]!=template[skip:] or len(prefill)!=turn["prefill_count"]:raise ValueError("Whole template ingestion differs")
    if not answers or len(answers)>turn["budget"] or len(generated)!=len(answers) or not turn["all_delivered_consumed"]:raise ValueError("Consumed answer closure differs")
    stopped=answers[-1] in (151645,151643)
    if any(x in (151645,151643) for x in answers[:-1]) or turn["stopped"]!=stopped or turn["truncated"]==stopped or (not stopped and len(answers)!=turn["budget"]):raise ValueError("Termination metadata differs")
    history=list(parent)
    for j,row in enumerate(steps(turn)):
        history.append(row["input_id"])
        if row["step"]!=j or row["position"]!=len(parent)+j or row["history_sha256"]!=token_hash(history):raise ValueError("History clock/identity differs")
        expected_role=("host-template-close" if not skip and j==0 else "template-user") if j<len(prefill) else ("own-generated" if turn["autonomous"] else "sealed-replay-answer")
        if row["role"]!=expected_role:raise ValueError("Token source role differs")
        policy=turn["policy"]
        if policy not in POLICIES or row["substitution"]!=(policy!=0) or row["gate_up_calls"]!=(2 if policy in (0,3) else 0) or row["down_calls"]!=(0 if policy==4 else 1) or row["compile_calls"]!=0:raise ValueError("Selected-layer work differs")
    for j,token in enumerate(answers):
        if generated[j]["input_id"]!=token:raise ValueError("Delivered answer was not consumed")
        if turn["autonomous"] and token!=(prefill[-1] if j==0 else generated[j-1])["top"]:raise ValueError("Greedy own token replaced")
    if turn["parent_position"]!=len(parent) or turn["final_position"]!=len(history):raise ValueError("Boundary position differs")
    return history

def analyze(raw,bank,driver,texts):
    raw,bank=Path(raw),Path(bank)
    if driver["schema"]!="tcm-whole-turn-driver-v1" or driver["layer"]!=16 or driver["capacity"]!=512 or not driver["source_replay_checked_before_candidates"]:raise ValueError("Whole-turn contract differs")
    payload=(raw/"driver.json").read_bytes();prefix=payload[:driver["consumer_prefix_seal_bytes"]]
    if hashlib.sha256(prefix).hexdigest()!=driver["consumer_prefix_seal_sha256"] or not driver["diagnostics_after_seal"]:raise ValueError("Primary seal differs")
    sealed=json.loads(prefix+b"}")
    if sealed["arms"]!=driver["arms"] or "diagnostics" in sealed:raise ValueError("Seal does not cover primary outputs")
    source=driver["source_history_ids"]
    if len(source)!=driver["source_position"] or source[-1]!=151645:raise ValueError("Source clock differs")
    colors=json.loads((raw/"colors.json").read_text())
    def load(stem,port,n=D):
        a=np.fromfile(raw/f"{stem}-{port}.f32",dtype="<f4")
        if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f"Invalid retained array {stem}-{port}")
        return a
    base=np.fromfile(bank/"reuse-program.f32",dtype="<f4");extra=np.fromfile(bank/"enrich-program.f32",dtype="<f4")
    qr=np.fromfile(bank/"reuse-qr.f64",dtype="<f8");qe=np.fromfile(bank/"enrich-qr.f64",dtype="<f8")
    if tuple(map(len,(base,extra,qr,qe)))!=(40448,19456,3076,1539):raise ValueError("Bank extents differ")
    basis=(base[:D],base[D:2*D],extra[:D]);q=(qr[:D],qr[D:2*D],qe[:D]);r00,r01,r11,_=qr[2*D:];r02,r12,r22=qe[D:]
    observations=[];all_turns={};work={"primary_transitions":0,"diagnostic_transitions":0,"gate_up_calls":0,"down_calls":0,"compile_calls":0}
    arms=driver["arms"];diagnostics=driver["diagnostics"]
    if [a["name"] for a in arms]!=list(LABELS) or [a["policy"] for a in arms]!=list(POLICIES) or [a["candidate"] for a in diagnostics]!=list(LABELS[1:]):raise ValueError("Panel membership differs")
    for diagnostic,groups in ((False,arms),(True,diagnostics)):
        for group in groups:
            if not group["source_prefix_intact"] or len(group["turns"])!=3:raise ValueError("Parent restoration differs")
            correction=group["turns"][0]
            if correction["parent_history_ids"]!=source or correction["parent_kv_sha256"]!=driver["source_kv_sha256"]:raise ValueError("Common source parent differs")
            for task,turn in zip(TASKS,group["turns"]):
                expected_name=("diag-"+group["candidate"] if diagnostic else group["name"])+"-"+task
                if turn["name"]!=expected_name or turn["autonomous"]==diagnostic or turn["policy"]!=(0 if diagnostic else group["policy"]):raise ValueError("Turn identity differs")
                validate_turn(turn);all_turns[turn["name"]]=turn
                if task!="correction" and (turn["parent_history_ids"]!=final_history(correction) or turn["parent_kv_sha256"]!=correction["final_kv_sha256"]):raise ValueError("Readers did not fork their own correction parent")
                for row in steps(turn):
                    stem=f"{turn['name']}-s{row['step']:03d}"
                    receiver,write,out=(load(stem,p) for p in ("raw_receiver","mlp_return","output"));load(stem,"input");load(stem,"mlp_input");load(stem,"kv16-17",1024)
                    if not same(out,np.add(receiver,write)):raise ValueError("Actual residual skip differs")
                    if turn["policy"]==4 and np.any(write!=0):raise ValueError("Zero policy wrote nonzero values")
                    if row["substitution"]:
                        v0,v1,v2=(seqdot(column,receiver) for column in q);c2=v2/r22;c1=(v1-r12*c2)/r11;c0=(v0-r01*c1-r02*c2)/r00
                        calculated=np.asarray((c0,c1,c2));c=np.asarray(row["executed"],dtype=np.float32)
                        if not np.allclose(calculated,row["coefficients"],rtol=1e-12,atol=1e-14) or not same(c,calculated.astype(np.float32)):raise ValueError("Executed coordinate solve differs")
                        reconstructed=np.add(np.add(c[0]*basis[0],c[1]*basis[1]),c[2]*basis[2])
                        omitted=np.add(np.add(receiver,-np.float32(1)*reconstructed),np.float32(0)*reconstructed)
                        if not same(reconstructed,load(stem,"reconstruction")) or not same(omitted,load(stem,"omitted")):raise ValueError("Actual omitted operand differs")
                    work["diagnostic_transitions" if diagnostic else "primary_transitions"]+=1
                    for field in ("gate_up_calls","down_calls","compile_calls"):work[field]+=row[field]
                for port,reference in (("prefill",turn["steps"][-1]),("final",turn["generation_steps"][-1])):
                    logits=load(turn["name"],port,VOCAB)
                    if int(np.argmax(logits))!=turn[port]["top"] or reference["top"]!=turn[port]["top"] or hashlib.sha256(logits.tobytes()).hexdigest()!=turn[port]["sha256"]:raise ValueError("Readout/selection differs")
                if not diagnostic:
                    expected={"correction":"OK","target":"green","collateral":"red"}[task]
                    observations.append({"kind":"behavior","policy":group["name"],"task":task,"text":texts[turn["name"]],"answer_ids":turn["answer_ids"],"expected_literal":expected,"literal_match":texts[turn["name"]].strip().casefold()==expected.casefold(),"stopped":turn["stopped"],"truncated":turn["truncated"],"host_supplied_closing_token":turn["host_supplied_closing_token"]})
    comparisons=[]
    for label in LABELS[1:]:
        for task in TASKS:
            candidate=all_turns[f"{label}-{task}"];dense=all_turns[f"diag-{label}-{task}"]
            if candidate["parent_history_ids"]!=dense["parent_history_ids"] or [r["input_id"] for r in steps(candidate)]!=[r["input_id"] for r in steps(dense)] or candidate["answer_ids"]!=dense["answer_ids"]:raise ValueError("Diagnostic is not the candidate's actual history")
            rows=[]
            for a,b in zip(steps(candidate),steps(dense)):
                stem=f"{candidate['name']}-s{a['step']:03d}";other=f"{dense['name']}-s{b['step']:03d}"
                exact={p:same(load(stem,p),load(other,p)) for p in ("input","raw_receiver","mlp_input")}
                kv,reference=load(stem,"kv16-17",1024),load(other,"kv16-17",1024)
                exact["kv16"]=same(kv[np.r_[0:256,512:768]],reference[np.r_[0:256,512:768]])
                qualified=all(exact.values())
                row={"kind":"same-text-local-write","policy":label,"task":task,"step":a["step"],"position":a["position"],"role":a["role"],"lower_ports_exact":exact,"same_input_reference_valid":qualified,"write_difference":distance(load(stem,"mlp_return"),load(other,"mlp_return")),"kv17_difference":distance(kv[np.r_[256:512,768:1024]],reference[np.r_[256:512,768:1024]])}
                rows.append(row);observations.append(row)
            boundaries={}
            for port in ("prefill","final"):
                ca,de=load(candidate["name"],port,VOCAB),load(dense["name"],port,VOCAB)
                boundaries[port]={"logit_difference":distance(ca,de),"full_logits_exact":same(ca,de),"full_valid_kv_exact":candidate[port+"_kv_sha256"]==dense[port+"_kv_sha256"],"greedy_matches_dense_same_text":int(np.argmax(ca))==int(np.argmax(de)),"color_logits":{color:{"candidate":float(ca[ids[0]]),"dense_same_text":float(de[ids[0]])} for color,ids in colors.items() if len(ids)==1}}
            comparisons.append({"policy":label,"task":task,"all_lower_ports_exact":all(r["same_input_reference_valid"] for r in rows),"steps":rows,"boundaries":boundaries})
    native_comparisons=[]
    for label in LABELS[1:]:
        for task in TASKS:
            c,n=all_turns[f"{label}-{task}"],all_turns[f"native-{task}"]
            native_comparisons.append({"policy":label,"task":task,"same_ingested_history":c["parent_history_ids"]+[r["input_id"] for r in c["steps"]]==n["parent_history_ids"]+[r["input_id"] for r in n["steps"]],"same_complete_history":final_history(c)==final_history(n),"own_answer_ids_equal":c["answer_ids"]==n["answer_ids"],"prefill_logit_difference":distance(load(c["name"],"prefill",VOCAB),load(n["name"],"prefill",VOCAB)),"scope":"own trajectories; same-input only if full consumed prefix matches"})
    summary={"primary_turns":12,"paid_diagnostic_segments":9,"all_same_text_lower_ports_exact":all(c["all_lower_ports_exact"] for c in comparisons),"behavior":[r for r in observations if r["kind"]=="behavior"],"policy_write_errors":{label:{"count":len(values),"min":min(values),"max":max(values),"mean":float(np.mean(values))} for label in LABELS[1:] if (values:=[r["write_difference"]["relative_l2"] for r in observations if r["kind"]=="same-text-local-write" and r["policy"]==label and r["same_input_reference_valid"] and r["write_difference"]["relative_l2"] is not None])}}
    return {"schema":"tcm-whole-turn-analysis-v1","summary":summary,"work":work,"same_text_comparisons":comparisons,"native_own_trajectory_comparisons":native_comparisons,"observations":observations,"consumer_seal_verified":True,"terminal_status":"not-requested"}
