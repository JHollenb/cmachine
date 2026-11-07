"""Four sequential writer policies, retained ports and actual native futures."""
from __future__ import annotations
import hashlib
import math
import numpy as np


def projected(v, columns):
    dots=[]
    for column in columns:
        value=0.0
        for x,y in zip(column,v):value+=float(x)*float(y)
        dots.append(value)
    out=[]
    for j in range(len(v)):
        value=0.0
        for column,dot in zip(columns,dots):value+=float(column[j])*dot
        out.append(value)
    return np.asarray(out,dtype=np.float32)


def measure(a,b):
    a,b=a.astype(np.float64),b.astype(np.float64);e=a-b;den=float(np.linalg.norm(b))
    return {"l2":float(np.linalg.norm(e)),"relative_l2":float(np.linalg.norm(e)/den) if den else None,
            "max_abs":float(np.max(np.abs(e))),"count":a.size}


def factorial(arrays):
    y={k:v.astype(np.float64) for k,v in arrays.items()};zero=np.zeros_like(y["00"])
    effects={"attention_isolated":y["10"]-y["00"],"mlp_isolated":y["01"]-y["00"],
             "attention_given_mlp":y["11"]-y["01"],"mlp_given_attention":y["11"]-y["10"],
             "joint":y["11"]-y["00"],"interaction":y["11"]-y["10"]-y["01"]+y["00"]}
    record={"effects":{k:measure(v,zero) for k,v in effects.items()},
            "native_differences":{k:measure(v,y["00"]) for k,v in y.items()}}
    for label in ("attention_isolated","mlp_isolated","interaction"):
        den=float(np.linalg.norm(effects["joint"]))
        record["effects"][label]["l2_over_joint_l2"]=float(np.linalg.norm(effects[label])/den) if den else None
    # Signed inner products are geometric observations; no mediation claim.
    record["twice_attention_mlp_inner_product"]=2*math.fsum(float(x)*float(y) for x,y in zip(effects["attention_isolated"].ravel(),effects["mlp_isolated"].ravel()))
    return record


def analyze(directory,driver):
    if driver["schema"]!="tcm-subspace-writer-driver-v1":raise ValueError("writer schema differs")
    def load(name,n):
        x=np.fromfile(directory/name,dtype="<f4")
        if x.size!=n or not np.isfinite(x).all():raise ValueError("invalid writer array: "+name)
        return x
    def same(a,b):return a.tobytes()==b.tobytes()
    def port(name,clock,p):return load(f"{name}-{clock}-{p}.f32",8960 if p in ("mlp_gate","mlp_up") else 256 if p in ("key","value") else 1536)
    raw=(directory/"reuse-program.f32").read_bytes();qr_raw=(directory/"reuse-qr.f64").read_bytes()
    extra=(directory/"enrich-program.f32").read_bytes();eqr_raw=(directory/"enrich-qr.f64").read_bytes()
    if len(raw)!=161792 or len(qr_raw)!=24608 or hashlib.sha256(raw+qr_raw).hexdigest()!=driver["program_sha256"]:raise ValueError("writer base program differs")
    if len(extra)!=77824 or len(eqr_raw)!=12312 or hashlib.sha256(extra+eqr_raw).hexdigest()!=driver["extension_program_sha256"]:raise ValueError("writer extension differs")
    qr=np.frombuffer(qr_raw,dtype="<f8");eqr=np.frombuffer(eqr_raw,dtype="<f8");columns=(qr[:1536],qr[1536:3072],eqr[:1536])
    if not all(np.isfinite(v).all() for v in columns):raise ValueError("nonfinite writer basis")
    binding=driver["writer_binding"]
    if binding!={"compile_layer":16,"edit_layer":15,"kv_first_layer":15,"gpu_bytes":6144,"host_bytes":43008,"projection":"sequential-fp64-qqt-then-fp32","fires_per_arm":1}:raise ValueError("writer binding differs")
    rows={r["name"]:r for r in driver["target_rows"]}
    expected={"writer-ordinary":-1,"writer-00":0,"writer-10":1,"writer-01":2,"writer-11":3}
    if len(driver["target_rows"])!=5 or {k:r["policy"] for k,r in rows.items()}!=expected:raise ValueError("writer arms differ")
    native=rows["writer-00"]
    if native["common_next_input_token_id"]!=native["first"]["top"]:raise ValueError("common token is not native pending ID")
    ports=("input","query","key","value","attention","mlp_input","mlp_gate","mlp_up","mlp_return","output","attention_write","raw_receiver","consumed_attention_write","consumed_mlp_write")
    comparisons=[];consumer_records=[]
    for name,row in rows.items():
        for key in ("position","input_token_id","common_next_input_token_id","parent_kv_prefix_sha256"):
            if row[key]!=native[key]:raise ValueError("writer parent/token differs")
        if not row["prefix_intact"]:raise ValueError("writer parent corrupted")
        for p in ("input","query","key","value","attention_write"):
            if not same(port(name,"l15",p),port("writer-00","l15",p)):raise ValueError("writer shared prefix differs")
        for clock in ("l15","l16","next-l16"):
            for p in ports:port(name,clock,p)
            x,a,r,m,y=(port(name,clock,p) for p in ("input","consumed_attention_write","raw_receiver","consumed_mlp_write","output"))
            if not same(np.add(x,a),r) or not same(np.add(r,m),y):raise ValueError("consumed writer/addition closure differs")
            for pre,consumed,mask in (("attention_write","consumed_attention_write",1),("mlp_return","consumed_mlp_write",2)):
                source=port(name,clock,pre);want=projected(source,columns) if clock=="l15" and row["policy"]>=0 and row["policy"]&mask else source
                if not same(want,port(name,clock,consumed)):raise ValueError("writer projection/materialization differs")
        if not same(port(name,"l15","output"),port(name,"l16","input")):raise ValueError("writer next input join differs")
        for rawport,key in (("first","first"),("common-next","common_next")):
            array=load(f"{name}-{rawport}.f32",151936)
            if hashlib.sha256(array.tobytes()).hexdigest()!=row[key]["sha256"] or int(np.argmax(array))!=row[key]["top"]:raise ValueError("writer readout identity differs")
        current=load(f"{name}-current-kv.f32",6656).reshape(2,13,256)
        nxt=load(f"{name}-next-kv.f32",6656).reshape(2,13,256)
        base_current=load("writer-00-current-kv.f32",6656).reshape(2,13,256)
        base_next=load("writer-00-next-kv.f32",6656).reshape(2,13,256)
        if not same(current[:,0],base_current[:,0]) or not same(nxt[:,:2],base_next[:,:2]):raise ValueError("writer effect precedes declared clock")
        def first_changed(a,b):
            return next((15+i for i in range(13) if not same(a[:,i],b[:,i])),None)
        if name!="writer-ordinary":
            comparisons.append({"name":name,"policy":row["policy"],"own_text":row["own"]["text"],"own_ids_equal":row["own"]["ids"]==native["own"]["ids"],
                "first":measure(load(f"{name}-first.f32",151936),load("writer-00-first.f32",151936)),
                "common_next":measure(load(f"{name}-common-next.f32",151936),load("writer-00-common-next.f32",151936)),
                "current_kv":measure(current,base_current),"next_kv":measure(nxt,base_next),
                "first_changed_current_kv_layer":first_changed(current,base_current),"first_changed_common_next_kv_layer":first_changed(nxt,base_next),
                "questions":[{"name":q["name"],"text":q["output"]["text"],"native_text":n["output"]["text"],"same_own_history":row["own"]["ids"]==native["own"]["ids"]} for q,n in zip(row["questions"],native["questions"])]})
    # No-op checks compare all selected ports, full readouts, memory and own IDs.
    for clock in ("l15","l16","next-l16"):
        for p in ports:
            if not same(port("writer-ordinary",clock,p),port("writer-00",clock,p)):raise ValueError("zero writer port parity differs")
    for suffix,n in (("first",151936),("common-next",151936),("current-kv",6656),("next-kv",6656)):
        if not same(load(f"writer-ordinary-{suffix}.f32",n),load(f"writer-00-{suffix}.f32",n)):raise ValueError("zero writer future parity differs")
    if rows["writer-ordinary"]["own"]!=native["own"] or rows["writer-ordinary"]["questions"]!=native["questions"]:raise ValueError("zero writer continuation differs")
    for left,right in (("00","01"),("10","11")):
        for p in ("raw_receiver","mlp_input","mlp_gate","mlp_up","mlp_return"):
            if not same(port("writer-"+left,"l15",p),port("writer-"+right,"l15",p)):raise ValueError("MLP policy changed its pre-edit computation")
    arrays={}
    for clock in ("l15","l16","next-l16"):
        for p in ("attention_write","consumed_attention_write","raw_receiver","mlp_return","consumed_mlp_write","output"):
            arrays[clock+"/"+p]={arm:port("writer-"+arm,clock,p) for arm in ("00","10","01","11")}
    incoming={arm:port("writer-"+arm,"l16","input") for arm in ("00","10","01","11")}
    arrays["layer16-input"]=incoming
    # This diagnostic uses the same sequential FP64 (I-P) policy as formation,
    # rather than subtracting an already rounded projected FP32 vector.
    def omitted(v):
        dots=[]
        for column in columns:
            value=0.0
            for x,y in zip(column,v):value+=float(x)*float(y)
            dots.append(value)
        result=v.astype(float)
        for column,dot in zip(columns,dots):result-=column*dot
        return result.astype(np.float32)
    arrays["layer16-omission"]={arm:omitted(v) for arm,v in incoming.items()}
    omissions=[]
    for arm in incoming:
        input_norm=float(np.linalg.norm(incoming[arm].astype(float)))
        omission_norm=float(np.linalg.norm(arrays["layer16-omission"][arm].astype(float)))
        omissions.append({"arm":arm,"input_l2":input_norm,"omission_l2":omission_norm,
                          "omission_relative_l2":omission_norm/input_norm if input_norm else None})
    for p,n in (("first",151936),("common-next",151936),("current-kv",6656),("next-kv",6656)):
        arrays[p]={arm:load(f"writer-{arm}-{p}.f32",n) for arm in ("00","10","01","11")}
    for key,values in arrays.items():consumer_records.append({"consumer":key,"scope":"sequential policy contrast",**factorial(values)})
    for p in ("current-kv","next-kv"):
        shaped={arm:values.reshape(2,13,256) for arm,values in arrays[p].items()}
        for kind,label in enumerate(("key","value")):
            for i in range(13):consumer_records.append({"consumer":p,"kind":label,"layer":15+i,"scope":"sequential policy contrast",**factorial({arm:v[kind,i] for arm,v in shaped.items()})})
    removed=[]
    for arm in ("10","01","11"):
        for pre,consumed,bit in (("attention_write","consumed_attention_write",1),("mlp_return","consumed_mlp_write",2)):
            if rows["writer-"+arm]["policy"]&bit:removed.append({"arm":arm,"write":pre,**measure(port("writer-"+arm,"l15",consumed),port("writer-"+arm,"l15",pre))})
    return {"comparisons":comparisons,"consumer_contrasts":{"schema":"tcm-writer-consumers-v1","records":consumer_records},
        "summary":{"scientific_arms":4,"ordinary_target_controls":1,"source_controls":len(driver["rows"]),"calibration_rows":len(driver["calibration_rows"]),
        "zero_policy_ports_logits_kv_continuation_exact":True,"writer_gpu_bytes":6144,"writer_host_bytes":43008,
        "frozen_payload_bytes":len(raw)+len(qr_raw)+len(extra)+len(eqr_raw),"removed_writes":removed,
        "layer16_omissions":omissions,"omission_policy":"sequential FP64 x-minus-three-projector-terms, then FP32",
        "source_text":driver["rows"][0]["own"]["text"],"rebuilt_source_ids_equal":driver["rebuilt_source_own"]["ids"]==driver["rows"][0]["own"]["ids"],
        "clock_boundaries_observed":True,"consumer_records":len(consumer_records)}}


def verify_saved(root,report):
    import json
    raw=root/"raw"
    for name,record in report["raw_files"].items():
        data=(raw/name).read_bytes()
        if len(data)!=record["bytes"] or hashlib.sha256(data).hexdigest()!=record["sha256"]:raise ValueError("writer raw source differs: "+name)
    if hashlib.sha256((raw/"reuse-driver.json").read_bytes()).hexdigest()!=report["driver_raw_sha256"]:raise ValueError("writer driver digest differs")
    unannotated=json.loads(json.dumps(report["driver"]))
    for row in unannotated["rows"]+unannotated["calibration_rows"]+unannotated["target_rows"]:
        row["own"].pop("text")
        for q in row["questions"]:q["output"].pop("text")
    unannotated["rebuilt_source_own"].pop("text")
    if unannotated!=json.loads((raw/"reuse-driver.json").read_text()):raise ValueError("writer driver annotation join differs")
    for row in report["driver"]["rows"]+report["driver"]["calibration_rows"]:
        data=(raw/(row["name"]+"-first.f32")).read_bytes();values=np.frombuffer(data,dtype="<f4")
        if len(values)!=151936 or not np.isfinite(values).all() or hashlib.sha256(data).hexdigest()!=row["first"]["sha256"] or int(np.argmax(values))!=row["first"]["top"]:raise ValueError("source/calibration readout differs")
    if any(row["input_token_id"]!=report["input"]["targets"][0]["ids"][-1] for row in report["driver"]["target_rows"]):raise ValueError("writer frozen target differs")
    expected=analyze(raw,report["driver"])
    def compare(a,b):
        if isinstance(a,dict):
            if set(a)!=set(b):raise ValueError("writer analysis fields differ")
            for k in a:compare(a[k],b[k])
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError("writer analysis rows differ")
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-14):raise ValueError("writer analysis arithmetic differs")
        elif a!=b:raise ValueError("writer analysis value differs")
    for key,value in expected.items():compare(value,({k:report[key][k] for k in value} if key=="summary" else report[key]))
    return {"raw_readouts":15,"raw_ports":210,"raw_files":len(report["raw_files"]),"consumer_records":len(expected["consumer_contrasts"]["records"])}
