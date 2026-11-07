"""Frozen bank predictions first, original-span audit afterward; no model calls."""
from __future__ import annotations
import hashlib,json,math,struct
from pathlib import Path
import numpy as np
from whole_turn_analysis import validate_turn,final_history,token_hash,steps
HERE=Path(__file__).resolve().parent
ARMS=("native","exact","one","eight","delete")
SCALE=float(np.float32(1/math.sqrt(128)))
PORTS={"input":1536,"query":1536,"key":256,"value":256,"attention":1536,"mlp_input":1536,"mlp_gate":8960,"mlp_up":8960,"mlp_return":1536,"output":1536,"attention_write":1536,"raw_receiver":1536}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(np.ravel(a),np.ravel(b)))
def norm(a):return math.sqrt(dot(a,a))
def metric(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);d=a-b;den=norm(b)
    return {"l2":norm(d),"max_abs":float(np.max(np.abs(d))),"relative_l2":norm(d)/den if den else None}
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def bank_read(path):
    data=Path(path).read_bytes()
    if data[:8]!=b"TCMSPAN1":raise ValueError("Bank magic differs")
    version,heads,dim,rows,m,reserved=struct.unpack("<6I",data[8:32])
    if (version,heads,dim,rows,reserved)!=(1,2,128,34,0) or m not in (1,8,34) or len(data)!=32+2*m*257*4:raise ValueError("Bank extent differs")
    a=np.frombuffer(data[32:],dtype="<f4").copy();n=2*m
    if not np.isfinite(a).all():raise ValueError("Nonfinite bank")
    return {"m":m,"counts":a[:n].reshape(2,m),"keys":a[n:n+n*128].reshape(2,m,128),"values":a[n+n*128:].reshape(2,m,128)}

def response(query,keys,values,counts=None):
    """Only the supplied response operands; no filesystem or reference access."""
    keys=np.asarray(keys,dtype=float);values=np.asarray(values,dtype=float);q=np.asarray(query,dtype=float)
    counts=np.ones(keys.shape[:2]) if counts is None else np.asarray(counts,dtype=float)
    out=[]
    for h in range(12):
        kh=h//6;scores=[dot(q[h],k)*SCALE+math.log(float(n)) for k,n in zip(keys[kh],counts[kh])];maximum=max(scores);e=[math.exp(s-maximum) for s in scores];total=math.fsum(e)
        mean=[math.fsum(w*float(v[d]) for w,v in zip(e,values[kh]))/total for d in range(128)];out.append([maximum+math.log(total),*mean])
    return np.asarray(out,dtype=float)
def mix(complement,span):
    out=[]
    for d,c in zip(complement,span):
        maximum=max(d[0],c[0]);zd=math.exp(d[0]-maximum);zc=math.exp(c[0]-maximum);out.append((zd*d[1:]+zc*c[1:])/(zd+zc))
    return np.asarray(out)
def softmax(scores):
    rows=[]
    for row in scores:
        maximum=max(row);e=[math.exp(float(x)-maximum) for x in row];den=math.fsum(e);rows.append([x/den for x in e])
    return np.asarray(rows)
def context(probabilities,values):
    return np.asarray([[math.fsum(float(w)*float(v[d]) for w,v in zip(probabilities[h],values[h//6])) for d in range(128)] for h in range(12)])

def analyze(raw,bank,driver,texts):
    raw=Path(raw);ref=json.loads((HERE/"origin.json").read_text());questions=json.loads((HERE/"questions.json").read_text())
    if driver["schema"]!="tcm-span-response-driver-v1" or driver["layer"]!=17 or driver["capacity"]!=512 or not driver["programs_frozen_before_question_files_opened"] or driver["constructor_future_queries"]!=0 or driver["constructor_position"]!=93:raise ValueError("Prospective construction contract differs")
    if driver["source_history_ids"]!=ref["source_history_ids"] or driver["source_kv_sha256"]!=ref["source_kv_sha256"]:raise ValueError("Source identity differs")
    correction=driver["correction"];old=ref["correction"];validate_turn(correction)
    for k in ("parent_history_ids","parent_position","parent_kv_sha256","template_ids","prefill_count","answer_ids","final_position","prefill_kv_sha256","final_kv_sha256","prefill","final"):
        if correction[k]!=old[k]:raise ValueError("Historical correction differs")
    for a,b in zip(steps(correction),steps(old)):
        for k in ("step","position","input_id","role","history_sha256","top","substitution","gate_up_calls","down_calls","compile_calls"):
            if a[k]!=b[k]:raise ValueError("Historical correction transition differs")
    def load(name,label,shape,dtype="<f4"):
        a=np.fromfile(raw/f"{name}-{label}",dtype=dtype)
        if a.size!=math.prod(shape) or not np.isfinite(a).all():raise ValueError(f"Invalid array {name}-{label}")
        return a.reshape(shape)
    for label in ("prefill","final"):
        if sha(raw/f"correction-{label}.f32")!=old[label]["sha256"]:raise ValueError("Correction readout differs")
    programs={};program_rows=[]
    if [p["name"] for p in driver["programs"]]!=["one","eight","exact"]:raise ValueError("Program membership differs")
    for p,after in zip(driver["programs"],driver["programs_after"]):
        name=p["name"];path=raw/p["file"];b=bank_read(path);m=b["m"];logs=load("bank-"+name,"logweights.f32",(2,m));gpu_raw=b["keys"].tobytes()+b["values"].tobytes()+logs.tobytes()
        if sha(path)!=p["file_sha256"] or path.stat().st_size!=p["bytes"] or m!=p["clusters"] or hashlib.sha256(gpu_raw).hexdigest()!=p["gpu_before_sha256"] or after!={"name":name,"gpu_after_sha256":p["gpu_before_sha256"]}:raise ValueError("Frozen program bytes differ")
        if p["float_payload_bytes"]!=2*m*257*4 or p["gpu_bytes"]!=2*m*257*4:raise ValueError("Program storage accounting differs")
        masks=np.asarray(p["membership"],dtype=np.uint64).reshape(2,m)
        for h in range(2):
            union=0
            for j,mask in enumerate(masks[h]):
                mask=int(mask)
                if mask<=0 or union&mask or mask>>34 or b["counts"][h,j]!=mask.bit_count():raise ValueError("Invalid cluster membership")
                union|=mask
            if union!=(1<<34)-1:raise ValueError("Incomplete span coverage")
        programs[name]={**b,"masks":masks,"logs":logs};program_rows.append({"name":name,"float_payload_bytes":p["float_payload_bytes"],"file_bytes":p["bytes"],"raw_original_span_bytes":69632,"file_over_raw_span":p["bytes"]/69632})
    # Qualify primary outputs and load the already sealed bank-only predictions.
    cases=[];observations=[];identity=[];total_own=0;primary={}
    if len(driver["questions"])!=2:raise ValueError("Question membership differs")
    for qi,group in enumerate(driver["questions"]):
        task=group["task"];q=questions[qi]
        if task!=q["name"] or not group["correction_parent_intact"] or len(group["arms"])!=5:raise ValueError("Question contract differs")
        template=list(map(int,(raw/q["file"]).read_text().split()));skip=int(final_history(correction)[-1]==151645)
        if template!=group["template_ids"] or group["template_skip"]!=skip:raise ValueError("Question template differs")
        expected_parent=final_history(correction)+template[skip:-1];arms=group["arms"];native=arms[0];captures={}
        for label,mode,a in zip(ARMS,(-1,34,1,8,0),arms):
            name=task+"-"+label
            if a["name"]!=name or a["mode"]!=mode or a["budget"]!=q["budget"] or a["parent_history_ids"]!=expected_parent or a["parent_position"]!=len(expected_parent) or a["input_id"]!=template[-1] or a["parent_kv_sha256"]!=native["parent_kv_sha256"] or not a["parent_intact"] or a["selected_read_applications"]!=1 or a["later_attention_policy"]!="native":raise ValueError("Common cut or intervention differs")
            history=expected_parent+[a["input_id"]]
            if a["history_sha256"]!=token_hash(history):raise ValueError("First cut history differs")
            ans=a["answer_ids"];gen=a["generation_steps"];pending=a["prefill"]["top"]
            if not ans or len(ans)>q["budget"] or len(gen)!=len(ans) or a["stopped"]!=(ans[-1] in (151645,151643)) or a["truncated"]==a["stopped"] or any(x in (151645,151643) for x in ans[:-1]) or not a["all_delivered_consumed"] or (a["truncated"] and len(ans)!=q["budget"]):raise ValueError("Generation closure differs")
            for j,(token,s) in enumerate(zip(ans,gen)):
                history.append(token)
                if token!=pending or s["input_id"]!=token or s["position"]!=len(history)-1 or s["step"]!=j or s["history_sha256"]!=token_hash(history):raise ValueError("Greedy own token or clock differs")
                pending=s["top"]
            if a["final_position"]!=len(history):raise ValueError("Final clock differs")
            total_own+=len(ans);cap={p:load(name,p+".f32",(n,)) for p,n in PORTS.items()};length=a["parent_position"]+1;m=34 if mode<0 else mode
            if a["packed_length"]!=length-34+m or a["original_span_rows_read_by_selected_attention"]!=(34 if mode<0 else 0) or a["span_dot_and_value_macs"]!=2*12*m*128 or a["complement_copy_bytes"]!=(0 if mode<0 else (length-34)*2048) or a["virtual_copy_bytes"]!=(0 if mode<=0 else mode*2048):raise ValueError("Physical read accounting differs")
            cap["probabilities"]=load(name,"probabilities.f32",(12,length-34+m));cap["kv"]=load(name,"kv17-18-27.f32",(2,3,2,128))
            for boundary in ("prefill","final"):
                y=load(name,boundary+".f32",(151936,));cap[boundary]=y
                if hashlib.sha256(y.tobytes()).hexdigest()!=a[boundary]["sha256"] or int(np.argmax(y))!=a[boundary]["top"]:raise ValueError("Readout identity differs")
            if not same(np.add(cap["input"],cap["attention_write"]),cap["raw_receiver"]) or not same(np.add(cap["raw_receiver"],cap["mlp_return"]),cap["output"]):raise ValueError("Native residual addition differs")
            if label in programs:
                pred=load(name,"response.f64",(12,129),"<f8")
                if sha(raw/f"{name}-response.f64")!=a["prediction_sha256"]:raise ValueError("Sealed primary prediction differs")
                primary[name]=pred
            elif a["prediction_sha256"] is not None:raise ValueError("Unexpected predictor")
            if mode>=0:cap["unscaled_scores"]=load(name,"unscaled-scores.f32",(12,length-34+m))
            captures[label]=cap;observations.append({"kind":"behavior","case":name,"text":texts[name],"answer_ids":ans,"stopped":a["stopped"],"truncated":a["truncated"],"recall_literal_match":texts[name].strip()=="Niko" if task=="recall" else None})
        for p in (*PORTS,"probabilities","kv","prefill","final"):
            if not same(captures["native"][p],captures["exact"][p]):raise ValueError("Exact34 native-consumer parity differs")
        for k in ("answer_ids","generation_steps","prefill_kv_sha256","final_kv_sha256","prefill","final","final_position"):
            if arms[0][k]!=arms[1][k]:raise ValueError("Exact34 continuation/cache differs")
        for cap in captures.values():
            if not same(cap["query"],captures["native"]["query"]) or not same(cap["kv"][:,0],captures["native"]["kv"][:,0]):raise ValueError("Query/current KV changed before the selected read")
        identity.append({"task":task,"exact34_all_ports_logits_cache_and_tokens":True});cases.append({"task":task,"group":group,"captures":captures})
    # Only now load original C for exact-span scoring and centroid audit.
    keys=load("audit-span","keys.f32",(2,34,128));values=load("audit-span","values.f32",(2,34,128));ideal_banks={};construction=[]
    for name,b in programs.items():
        ideal_k=[];ideal_v=[]
        for h in range(2):
            hk=[];hv=[]
            for mask in b["masks"][h]:
                ids=[i for i in range(34) if int(mask)&(1<<i)];hk.append([math.fsum(float(keys[h,i,d]) for i in ids)/len(ids) for d in range(128)]);hv.append([math.fsum(float(values[h,i,d]) for i in ids)/len(ids) for d in range(128)])
            ideal_k.append(hk);ideal_v.append(hv)
        ideal_banks[name]={**b,"keys":np.asarray(ideal_k),"values":np.asarray(ideal_v)};construction.append({"name":name,"key_rounding":metric(b["keys"],ideal_k),"value_rounding":metric(b["values"],ideal_v)})
    if not same(programs["exact"]["keys"],keys) or not same(programs["exact"]["values"],values):raise ValueError("Exact34 program differs from original span")
    comparisons=[];jensen=[]
    for case in cases:
        task=case["task"];group=case["group"];captures=case["captures"];q=captures["native"]["query"].reshape(12,128);length=group["arms"][0]["parent_position"]+1;compk=load(task+"-native","complement-keys.f32",(2,length-34,128));compv=load(task+"-native","complement-values.f32",(2,length-34,128))
        exact=response(q,keys,values);complement=response(q,compk,compv);native_ideal=mix(complement,exact);ideals={name:response(q,b["keys"],b["values"],b["counts"]) for name,b in ideal_banks.items()}
        for h in range(12):
            row={"task":task,"head":h,"kv_head":h//6,"ideal_logZ_one_minus_eight":float(ideals["one"][h,0]-ideals["eight"][h,0]),"ideal_logZ_eight_minus_exact":float(ideals["eight"][h,0]-exact[h,0]),"stored_one_logZ_minus_ideal":float(primary[task+"-one"][h,0]-ideals["one"][h,0]),"stored_eight_logZ_minus_ideal":float(primary[task+"-eight"][h,0]-ideals["eight"][h,0])};jensen.append(row);observations.append({"kind":"denominator-order","case":task,"step":h,"metrics":row})
        for label in ("exact","one","eight","delete"):
            name=task+"-"+label;cap=captures[label];native=captures["native"];per_head=[];nterms=[];zterms=[]
            if label!="delete":
                b=programs[label];pred=primary[name];bank_ref=response(q,b["keys"],b["values"],b["counts"]);pred_context=mix(complement,pred);pred_discrepancy=metric(pred,bank_ref)
                ktable=np.concatenate((compk[:,:59],b["keys"],compk[:,59:]),axis=1);vtable=np.concatenate((compv[:,:59],b["values"],compv[:,59:]),axis=1);bias=np.zeros((12,ktable.shape[1]),dtype="<f4")
                for h in range(12):bias[h,59:59+b["m"]]=b["logs"][h//6]
            else:
                pred=None;pred_context=complement[:,1:];pred_discrepancy=None;ktable,vtable=compk,compv;bias=np.zeros((12,ktable.shape[1]),dtype="<f4")
            for h in range(12):
                logd,mu_d=complement[h,0],complement[h,1:];loge,mu_e=exact[h,0],exact[h,1:];logp,mu_p=(pred[h,0],pred[h,1:]) if pred is not None else (-math.inf,np.zeros(128));g=max(logd,loge,logp);zd=math.exp(logd-g);ze=math.exp(loge-g);zp=math.exp(logp-g) if pred is not None else 0.;true=(zd*mu_d+ze*mu_e)/(zd+ze);dn=zp*mu_p-ze*mu_e;dz=zp-ze;nt=dn/(zd+zp);zt=-true*dz/(zd+zp);nterms.append(nt);zterms.append(zt)
                ratio=math.exp(logp-loge) if pred is not None else 0.;per_head.append({"head":h,"Z_ratio":ratio,"signed_Z_relative_error":ratio-1,"numerator":metric(ratio*mu_p,mu_e),"span_mean":metric(mu_p,mu_e),"full_context":metric(pred_context[h],native_ideal[h])})
            nterms=np.asarray(nterms);zterms=np.asarray(zterms);ideal_delta=pred_context-native_ideal;native_delta=(cap["attention"].astype(float)-native["attention"].astype(float)).reshape(12,128);roundoff_delta=native_delta-ideal_delta
            score64=np.asarray([[dot(q[h],k) for k in ktable[h//6]] for h in range(12)]);scaled=np.add(np.multiply(cap["unscaled_scores"],np.float32(SCALE)),bias).astype(float);prob_ref=softmax(scaled);ctx_ref=context(cap["probabilities"].astype(float),vtable)
            result={"task":task,"arm":label,"per_head":per_head,"bank_primary_vs_fsum":pred_discrepancy,"full_ideal_context_vs_native_ideal":metric(pred_context,native_ideal),"actual_context_vs_native":metric(cap["attention"],native["attention"]),"actual_context_vs_bank_prediction":metric(cap["attention"].reshape(12,128),pred_context),"native_context_vs_exact_ideal":metric(native["attention"].reshape(12,128),native_ideal),"signed_context_account":{"numerator_term_norm":norm(nterms),"denominator_term_norm":norm(zterms),"cross":2*dot(nterms,zterms),"ideal_delta_norm":norm(ideal_delta),"ideal_closure_max_abs":float(np.max(np.abs(ideal_delta-nterms-zterms))),"native_arithmetic_delta_norm":norm(roundoff_delta)},"consumer_ports":{p:metric(cap[p],native[p]) for p in ("attention_write","raw_receiver","mlp_input","mlp_return","output","prefill","final")},"current_KV":{"kv17":metric(cap["kv"][:,0],native["kv"][:,0]),"kv18":metric(cap["kv"][:,1],native["kv"][:,1]),"kv27":metric(cap["kv"][:,2],native["kv"][:,2])},"execution_arithmetic":{"native_QK_vs_fsum":metric(cap["unscaled_scores"],score64),"native_probabilities_vs_ideal_softmax_of_rounded_logits":metric(cap["probabilities"],prob_ref),"native_context_vs_prob_value_product":metric(cap["attention"].reshape(12,128),ctx_ref)}}
            arm_record=group["arms"][ARMS.index(label)];native_record=group["arms"][0];same_final=arm_record["answer_ids"]==native_record["answer_ids"]
            first_difference=next((j for j,(a,b) in enumerate(zip(arm_record["answer_ids"],native_record["answer_ids"])) if a!=b),None)
            if not same_final and first_difference is None:first_difference=min(len(arm_record["answer_ids"]),len(native_record["answer_ids"]))
            result["history_domain"]={"first_read_common_complete_parent":True,"final_complete_history_matches_native":same_final,"final_clock_matches_native":arm_record["final_position"]==native_record["final_position"],"final_comparison_kind":"same-history-fidelity" if same_final else "different-own-history-trajectory","first_different_emitted_token":first_difference}
            comparisons.append(result);observations.append({"kind":"span-response-consumer","case":name,"metrics":result})
    summary={"prompts":2,"native_query_heads":24,"branches":10,"selected_read_applications":10,"source_advances":59,"correction_advances":len(steps(correction)),"question_prefix_advances":sum(len(g["template_ids"])-g["template_skip"]-1 for g in driver["questions"]),"own_continuation_advances":total_own,"identity_controls":identity,"program_storage":program_rows,"answers":texts,"scope":"once-only frozen span response; dense complement/prefix/later native reads retained; no whole-chat compression or speedup claim"}
    return {"summary":summary,"program_construction":construction,"jensen_observations":jensen,"comparisons":comparisons,"observations":observations}
