"""Native-consumed inherited/new carry split; full-carried is NOT full native."""
from __future__ import annotations

import numpy as np
from consumer_contrasts import contrast
from reuse_analysis import analyze as analyze_reuse


def consumer_metrics(arrays,consumer_version=2):
    if consumer_version not in (1,2):raise ValueError("unknown formation consumer schema")
    metrics=contrast(arrays)
    reference=contrast({**arrays,"dense":arrays["correct"]})
    # v1 replaced dense with F while computing transitions and then reused its
    # error dictionary. Reproduce that original cell only for immutable audits.
    metrics["full_carried_reference_error_l2"]=(reference["native_error_l2"] if consumer_version==1 else
        {role:float(np.linalg.norm(value.astype(float)-arrays["correct"].astype(float))) for role,value in arrays.items()})
    metrics["full_carried_reference_error_transitions"]=reference["native_error_transitions"]
    return metrics


def analyze(directory,driver,consumer_version=2):
    if consumer_version not in (1,2):raise ValueError("unknown formation consumer schema")
    result=analyze_reuse(directory,driver)
    rows={r["name"].removeprefix("Niko-D-"):r for r in driver["target_rows"]}
    roles={"dense":1,"project":2,"inherited":9,"new":9,"carry":7}
    if set(rows)!=set(roles):raise ValueError("formation panel roles differ")
    def load(arm,port,n=1536):
        a=np.fromfile(directory/f"Niko-D-{arm}-{port}.f32",dtype="<f4")
        if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError("invalid formation array")
        return a
    def same(a,b):return a.tobytes()==b.tobytes()
    def measure(a,b):
        a,b=a.astype(float),b.astype(float);den=np.linalg.norm(b)
        return {"l2":float(np.linalg.norm(a-b)),"relative_l2":float(np.linalg.norm(a-b)/den) if den else None,"max_abs":float(np.max(np.abs(a-b)))}
    native=rows["dense"]
    if native["common_next_input_token_id"]!=native["first"]["top"]:raise ValueError("formation common token is not native pending ID")
    for arm,row in rows.items():
        if row["mode"]!=roles[arm] or row["rank"]!=3:raise ValueError("formation role/rank mismatch")
        for key in ("position","parent_kv_prefix_sha256","rho","input_token_id","common_next_input_token_id"):
            if row[key]!=native[key]:raise ValueError("formation consumer ancestry differs")
    for port in ("input","attention_write","raw_receiver","mlp_return","output"):
        for arm in rows:
            if not same(load(arm,"l15-"+port),load("dense","l15-"+port)):raise ValueError("formation earlier-layer operands differ")
    x,a,r,m,out=(load("dense","l15-"+p) for p in ("input","attention_write","raw_receiver","mlp_return","output"))
    if not same(r,np.add(x,a)) or not same(out,np.add(r,m)) or not same(out,load("dense","entry")):
        raise ValueError("formation native layer15/16 closure differs")
    qr=np.fromfile(directory/"reuse-qr.f64",dtype="<f8");extra=np.fromfile(directory/"enrich-qr.f64",dtype="<f8")
    q=(qr[:1536],qr[1536:3072],extra[:1536])
    def omitted(v):
        v=v.astype(float);dots=[]
        for column in q:
            value=0.0
            for u,w in zip(column,v):value+=float(u)*float(w)
            dots.append(value)
        for column,value in zip(q,dots):v-=column*value
        return v
    inherited=omitted(x).astype(np.float32);carry=omitted(out).astype(np.float32)
    new=np.add(np.add(carry,np.multiply(np.float32(-1),inherited)),np.multiply(np.float32(0),inherited))
    for arm in rows:
        for port,expected in (("inherited-operand",inherited),("carried-operand",carry),("new-write-operand",new)):
            if not same(load(arm,port),expected):raise ValueError("formation split operand differs")
    for arm,expected in (("inherited",inherited),("new",new)):
        if not same(load(arm,"explicit-operand"),expected):raise ValueError("formation correction consumed another operand")
    if not same(load("carry","carry-omitted"),carry):raise ValueError("full-carried reference uses another omission")
    if driver["split_joint"]!={"gate_up_matvecs":0,"down_matvecs":1,"session_metadata_unchanged":True,"kv_prefix_sha256":driver["source_answer_kv_prefix_sha256"],"state_advanced":False}:raise ValueError("formation joint diagnostic advanced state")
    for channel in ("gate","up"):
        joined=np.add(np.add(load("project","mlp_"+channel,8960),load("inherited","correction-"+channel,8960)),load("new","correction-"+channel,8960))
        actual=np.fromfile(directory/f"split-joint-mlp_{channel}.f32",dtype="<f4")
        if not same(joined,actual):raise ValueError("formation split-joint assembly differs")
    joint=np.fromfile(directory/"split-joint-mlp_return.f32",dtype="<f4")
    if joint.shape!=(1536,) or not np.isfinite(joint).all():raise ValueError("invalid joint write")
    local_interaction=joint.astype(float)-load("inherited","mlp_return")-load("new","mlp_return")+load("project","mlp_return")
    detail={"scope":"formation of frozen-B3 omission, not a semantic badge operation; full-carried reference omits current read complement",
        "operand_split_closure":measure(inherited.astype(float)+new.astype(float),carry),
        "new_complement_vs_independent_projected_writes":measure(new,omitted(a.astype(float)+m.astype(float))),
        "native_addition_rounding":measure(out.astype(float),x.astype(float)+a.astype(float)+m.astype(float)),
        "inherited_norm":float(np.linalg.norm(inherited.astype(float))),"new_write_norm":float(np.linalg.norm(new.astype(float))),
        "split_joint_vs_full_carried_write":measure(joint,load("carry","mlp_return")),
        "local_split_joint_interaction_l2":float(np.linalg.norm(local_interaction)),
        "local_split_joint_scope":"pure local diagnostic; no downstream split-joint continuation or error bound"}
    records=[];mapping={"project":"project","carry":"inherited","read":"new","correct":"carry","dense":"dense"}
    for port,n in (("mlp_return",1536),("output",1536),("first",151936),("current-kv",5632),("next-kv",5632)):
        arrays={role:load(arm,port,n) for role,arm in mapping.items()}
        binding={"port":port,"position":native["position"]+int(port=="next-kv"),"selection":"full-retained-array",
                 "common_next_input_token_id":native["common_next_input_token_id"] if port=="next-kv" else None}
        metrics=consumer_metrics(arrays,consumer_version)
        records.append({**binding,"metrics":metrics})
        if port.endswith("kv"):
            shaped={role:v.reshape(2,11,256) for role,v in arrays.items()}
            for kind,label in enumerate(("key","value")):
                for index in range(11):
                    values={role:v[kind,index] for role,v in shaped.items()};metrics=consumer_metrics(values,consumer_version)
                    records.append({**binding,"selection":label,"layer":17+index,"metrics":metrics})
    result["summary"].update(formation_diagnostics=detail,
        full_carried_reference_write_relative_error=measure(load("carry","mlp_return"),load("dense","mlp_return"))["relative_l2"])
    result["summary"]["source_residual_corrected_write_relative_error"]=result["summary"].pop("max_residual_corrected_write_relative_error")
    result["consumer_contrasts"]={"schema":f"tcm-formation-consumer-contrasts-v{consumer_version}",
        "definitions":{"carry":"inherited correction","read":"new-write complement correction","correct":"executed full-carried reference, not full omission/native","interaction":"Y_fullcarried-Y_inherited-Y_new+Y_project; operational endpoints, not literal split-joint continuation"},
        "binding":{"layer":16,"position":native["position"],"parent_kv_prefix_sha256":native["parent_kv_prefix_sha256"],
            "program_sha256":driver["program_sha256"],"extension_program_sha256":driver["extension_program_sha256"],
            "common_next_input_token_id":native["common_next_input_token_id"],"input_token_id":native["input_token_id"],
            "own_token_histories_equal":all(row["own"]["ids"]==native["own"]["ids"] for row in rows.values())},"records":records}
    return result
