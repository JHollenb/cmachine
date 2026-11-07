"""Read-only native attention account; arithmetic uses deterministic FP64 fsum."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
from whole_turn_analysis import validate_turn,final_history,steps,token_hash
HERE=Path(__file__).resolve().parent
WORLDS=("NN","PN","NP","PP")
PORTS={"input":1536,"query":1536,"key":256,"value":256,"attention":1536,"mlp_input":1536,"mlp_gate":8960,"mlp_up":8960,"mlp_return":1536,"output":1536,"attention_write":1536,"raw_receiver":1536}

def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(np.ravel(a),np.ravel(b)))
def norm(a):return math.sqrt(dot(a,a))
def metric(a,b=None):
    a=np.asarray(a,dtype=np.float64);delta=a if b is None else a-np.asarray(b,dtype=np.float64)
    return {"l2":norm(delta),"max_abs":float(np.max(np.abs(delta))) if delta.size else 0.,"relative_l2":None if b is None or norm(b)==0 else norm(delta)/norm(b)}
def mm(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError("Matrix shape differs")
    rows=a.tolist();cols=b.T.tolist()
    return np.asarray([[math.fsum(x*y for x,y in zip(row,col)) for col in cols] for row in rows],dtype=np.float64)
def head_mm(a,b):return np.asarray([mm(x,y) for x,y in zip(a,b)])
def interaction(a):return a[3]-a[1]-a[2]+a[0]
def factorial(a):
    a=[np.asarray(v,dtype=np.float64) for v in a];formation=a[1]-a[0];question=a[2]-a[0];i=interaction(a);joint=a[3]-a[0]
    return {"formation":metric(formation),"question":metric(question),"interaction":metric(i),"joint":metric(joint),"interaction_over_joint":norm(i)/norm(joint) if norm(joint) else None,"cross_terms":{"formation_question":2*dot(formation,question),"formation_interaction":2*dot(formation,i),"question_interaction":2*dot(question,i)}}
def softmax64(x):
    result=[]
    for row in x:
        maximum=max(row);exps=[math.exp(float(v)-maximum) for v in row];total=math.fsum(exps);result.append([v/total for v in exps])
    return np.asarray(result)
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def value_terms(probabilities,values):
    p=[np.asarray(x,dtype=np.float64) for x in probabilities];v=[np.asarray(x,dtype=np.float64) for x in values]
    pairs=((interaction(p),v[0]),(p[3]-p[1],v[1]-v[0]),(p[3]-p[2],v[2]-v[0]),(p[3],interaction(v)))
    return [head_mm(a[:,None,:],b)[:,0,:] for a,b in pairs]

def normalizer_response(before_scores,after_scores,before_values,after_values,band):
    """Conditional finite-read update; both worlds use one stable exponential gauge."""
    eb=[];ea=[]
    for b,a in zip(before_scores,after_scores):
        maximum=max(float(max(b)),float(max(a)));eb.append([math.exp(float(x)-maximum) for x in b]);ea.append([math.exp(float(x)-maximum) for x in a])
    eb=np.asarray(eb);ea=np.asarray(ea);zb=np.asarray([math.fsum(row) for row in eb]);za=np.asarray([math.fsum(row) for row in ea])
    pb=eb/zb[:,None];pa=ea/za[:,None]
    cb=head_mm(pb[:,None,:],before_values)[:,0,:];ca=head_mm(pa[:,None,:],after_values)[:,0,:]
    dz=np.asarray([math.fsum(a[band])-math.fsum(b[band]) for b,a in zip(eb,ea)])
    dn=head_mm(ea[:,None,band],after_values[:,band,:])[:,0,:]-head_mm(eb[:,None,band],before_values[:,band,:])[:,0,:]
    prediction=(dn-cb*dz[:,None])/za[:,None]
    return {"prediction":prediction,"direct":ca-cb,"ratio":zb/za,"delta_z_over_after_z":dz/za}

def analyze(raw,bank,driver,texts):
    raw=Path(raw);ref=json.loads((HERE/"origin.json").read_text())
    if driver["schema"]!="tcm-attention-read-driver-v1" or driver["layer"]!=17 or driver["replacement_layer"]!=16 or driver["capacity"]!=512 or driver["selected_positions"]!=[116,118] or (driver["qheads"],driver["kvheads"],driver["head_dimension"])!=(12,2,128):raise ValueError("Attention contract differs")
    if not driver["source_replay_checked_before_candidates"] or driver["source_history_ids"]!=ref["source_history_ids"] or driver["source_kv_sha256"]!=ref["source_kv_sha256"]:raise ValueError("Source parent differs")
    def load(stem,port,shape):
        a=np.fromfile(raw/f"{stem}-{port}.f32",dtype="<f4")
        if a.size!=math.prod(shape) or not np.isfinite(a).all():raise ValueError(f"Invalid capture {stem}-{port}")
        return a.reshape(shape)
    turns={};observations=[];work={"observed_transitions":0,"gate_up_calls":0,"down_calls":0,"compile_calls":0};joins=0
    def qualify(t,old):
        nonlocal joins
        validate_turn(t)
        for key in ("policy","parent_position","parent_history_ids","parent_kv_sha256","template_ids","template_skip","host_supplied_closing_token","prefill_count","answer_ids","stopped","truncated","final_position","prefill_kv_sha256","final_kv_sha256","prefill","final"):
            if t[key]!=old[key]:raise ValueError(f"Historical endpoint identity differs: {t['name']}:{key}")
        if final_history(t)!=final_history(old):raise ValueError("Historical consumed history differs")
        for a,b in zip(steps(t),steps(old)):
            for key in ("step","position","input_id","role","history_sha256","top","substitution","gate_up_calls","down_calls","compile_calls"):
                if a[key]!=b[key]:raise ValueError("Historical transition differs")
        for boundary in ("prefill","final"):
            y=load(t["name"],boundary,(151936,))
            if hashlib.sha256(y.tobytes()).hexdigest()!=old[boundary]["sha256"] or int(np.argmax(y))!=t[boundary]["top"]:raise ValueError("Uninstrumented endpoint bytes differ")
            joins+=1
        for row in steps(t):
            work["observed_transitions"]+=1
            for k in ("gate_up_calls","down_calls","compile_calls"):work[k]+=row[k]
        observations.append({"kind":"behavior","case":t["name"],"text":texts[t["name"]],"answer_ids":t["answer_ids"],"historical_logits_and_cache_exact":True})
    if len(driver["formations"])!=2:raise ValueError("Formation membership differs")
    for alpha,group in zip(("N","P"),driver["formations"]):
        if group["policy"]!=(0 if alpha=="N" else 2) or not group["formation_checked_before_readers"] or not group["source_prefix_intact"]:raise ValueError("Formation policy differs")
        correction=group["correction"];qualify(correction,ref["formations"][alpha])
        if correction["name"]!=alpha+"-correction" or len(group["readers"])!=4:raise ValueError("Correction identity differs")
        for beta in ("N","P"):
            for task in ("target","collateral"):
                name=alpha+beta+"-"+task;t=next(t for t in group["readers"] if t["name"]==name);qualify(t,ref["readers"][alpha+beta][task]);turns[name]=t
                if t["parent_history_ids"]!=final_history(correction) or t["parent_kv_sha256"]!=correction["final_kv_sha256"]:raise ValueError("Sibling correction parent differs")
                if [row["position"] for row in steps(t) if row["attention_trace"]]!=[116,118] or t["steps"][-1]["position"]!=116 or t["generation_steps"][-1]["position"]!=118 or t["generation_steps"][-1]["input_id"]!=151645:raise ValueError("Selected clock/history differs")
    operator=load("operator","o17",(1536,1536)).astype(np.float64);rows=[]
    for task in ("target","collateral"):
        for position in (116,118):
            length=position+1;names=[w+"-"+task for w in WORLDS]
            histories=[t["parent_history_ids"]+[s["input_id"] for s in steps(t) if s["position"]<=position] for t in (turns[n] for n in names)]
            if not all(h==histories[0] for h in histories[1:]):raise ValueError("Factorial is not on common consumed history")
            ports={p:[load(f"{name}-p{position}",p,(n,)) for name in names] for p,n in PORTS.items()}
            queries=[a.reshape(12,128) for a in ports["query"]]
            keys=[load(f"{name}-p{position}","keys",(2,length,128)) for name in names]
            values=[load(f"{name}-p{position}","values",(2,length,128)) for name in names]
            probs=[load(f"{name}-p{position}","probabilities",(12,length)) for name in names]
            scores=[load(f"{name}-p{position}","unscaled-scores",(12,length)) for name in names]
            scales=[load(f"{name}-p{position}","scale",(1,)) for name in names]
            if not all(same(s,scales[0]) for s in scales):raise ValueError("Native scale identity differs")
            mapped_keys=[a[np.arange(12)//6].astype(np.float64) for a in keys];mapped_values=[a[np.arange(12)//6].astype(np.float64) for a in values]
            q64=[a.astype(np.float64) for a in queries];score64=[head_mm(a[:,None,:],b.transpose(0,2,1))[:,0,:] for a,b in zip(q64,mapped_keys)]
            scores64=[a.astype(np.float64) for a in scores];prob64=[a.astype(np.float64) for a in probs]
            score_remainders=[s-e for s,e in zip(scores64,score64)];ideal_interaction=interaction(score64)
            scaled32=[np.multiply(s,np.float32(scales[0][0])).astype(float) for s in scores]
            scale_constant=[s*(float(scales[0][0])-1/math.sqrt(128)) for s in scores64]
            scale_multiply=[s-r*float(scales[0][0]) for s,r in zip(scaled32,scores64)]
            bilinear=head_mm((q64[2]-q64[0])[:,None,:],(mapped_keys[1]-mapped_keys[0]).transpose(0,2,1))[:,0,:]
            native_interaction=interaction(scores64);prob_interaction=interaction(prob64)
            source=slice(0,driver["source_position"]);correction=slice(driver["source_position"],turns[names[0]]["parent_position"]);question=slice(turns[names[0]]["parent_position"],length)
            bands={"source":source,"correction":correction,"question":question}
            dependencies={"query_fixed_by_question_policy":same(queries[0],queries[1]) and same(queries[2],queries[3]),"entry_fixed_by_question_policy":same(ports["input"][0],ports["input"][1]) and same(ports["input"][2],ports["input"][3])}
            for label,arrays in (("keys",keys),("values",values)):
                dependencies[label+"_source_fixed"]=all(same(a[:,source],arrays[0][:,source]) for a in arrays)
                dependencies[label+"_correction_fixed_by_formation"]=same(arrays[0][:,correction],arrays[2][:,correction]) and same(arrays[1][:,correction],arrays[3][:,correction])
                dependencies[label+"_question_fixed_by_question_policy"]=same(arrays[0][:,question],arrays[1][:,question]) and same(arrays[2][:,question],arrays[3][:,question])
            current_rows_exact=all(same(k[:,-1,:].reshape(-1),ports["key"][i]) and same(v[:,-1,:].reshape(-1),ports["value"][i]) for i,(k,v) in enumerate(zip(keys,values)))
            if not current_rows_exact:raise ValueError("Captured cache/current operand join differs")
            contexts=[head_mm(p[:,None,:],v)[:,0,:] for p,v in zip(prob64,mapped_values)]
            actual=[a.reshape(12,128).astype(np.float64) for a in ports["attention"]];value_remainders=[a-b for a,b in zip(actual,contexts)]
            terms=value_terms(prob64,mapped_values);term_names=("probability_interaction_on_baseline_values","conditional_question_probability_on_changed_correction_values","conditional_formation_probability_on_changed_question_values","value_interaction")
            native_context_interaction=interaction(actual);remainder_interaction=interaction(value_remainders);prediction=sum(terms)+remainder_interaction
            denominator=dot(native_context_interaction,native_context_interaction)
            signed={name:{**metric(t),"projection_on_native_interaction":dot(t,native_context_interaction)/denominator if denominator else None} for name,t in zip(term_names+("native_value_roundoff_interaction",),terms+[remainder_interaction])}
            pairwise={f"{a}:{b}":2*dot(x,y) for i,(a,x) in enumerate(zip(term_names,terms)) for b,y in list(zip(term_names,terms))[i+1:]}
            normalizers={}
            offband=np.r_[np.arange(driver["source_position"]),np.arange(turns[names[0]]["parent_position"],length)]
            for beta,before,after in (("N",0,1),("P",2,3)):
                update=normalizer_response(scaled32[before],scaled32[after],mapped_values[before],mapped_values[after],correction)
                native_delta=actual[after]-actual[before];ratio=update["ratio"]
                normalizers[beta]={"ideal_update_closure":metric(update["direct"],update["prediction"]),"native_context_delta":metric(native_delta),"prediction":metric(update["prediction"]),"native_minus_prediction":metric(native_delta,update["prediction"]),"offband_native_probability_ratio_residual":metric(prob64[after][:,offband],ratio[:,None]*prob64[before][:,offband]),"per_head_Z_before_over_Z_after":ratio.tolist(),"per_head_delta_Z_over_Z_after":update["delta_z_over_after_z"].tolist()}
            projected=mm(operator,native_context_interaction.reshape(1536,1))[:,0];actual_write_interaction=interaction([a.astype(np.float64) for a in ports["attention_write"]])
            skips=[];return_skips=[]
            for j in range(4):
                entry=ports["input"][j];write=ports["attention_write"][j];receiver=ports["raw_receiver"][j];mlp=ports["mlp_return"][j];out=ports["output"][j]
                if not same(np.add(entry,write),receiver) or not same(np.add(receiver,mlp),out):raise ValueError("Native residual addition differs")
                skips.append(receiver.astype(float)-entry.astype(float)-write.astype(float));return_skips.append(out.astype(float)-receiver.astype(float)-mlp.astype(float))
            local_factorials={p:factorial(a) for p,a in ports.items() if p in ("input","query","attention","attention_write","raw_receiver","mlp_input","mlp_return","output")}
            record={"task":task,"position":position,"boundary":"pre-answer" if position==116 else "post-consumed-EOS","length":length,"common_history_sha256":token_hash(histories[0]),"dependencies":dependencies,"native_scale_hex":scales[0].tobytes().hex(),"native_scale":float(scales[0][0]),"native_scale_minus_mathematical":float(scales[0][0])-1/math.sqrt(128),
                    "scores":{"native_factorial":metric(native_interaction),"ideal_factorial":metric(ideal_interaction),"bilinear_prediction":metric(bilinear),"ideal_minus_bilinear":metric(ideal_interaction,bilinear),"native_minus_bilinear":metric(native_interaction,bilinear),"native_roundoff_factorial":metric(interaction(score_remainders)),"account_closure":metric(native_interaction-bilinear-interaction(score_remainders)),"world_raw_dot_roundoff":{w:metric(e) for w,e in zip(WORLDS,score_remainders)},"bands":{b:{"native_interaction":metric(native_interaction[:,sl]),"ideal_interaction":metric(ideal_interaction[:,sl]),"bilinear":metric(bilinear[:,sl])} for b,sl in bands.items()}},
                    "probabilities":{"factorial":metric(prob_interaction),"row_sum_interaction_max_abs":max(abs(math.fsum(row)) for row in prob_interaction),"world_sum_minus_one_max_abs":max(abs(math.fsum(row)-1) for a in prob64 for row in a),"native_vs_fp64_softmax_of_native_scaled_scores":{w:metric(p,softmax64(np.multiply(s,np.float32(scales[0][0])).astype(float))) for w,p,s in zip(WORLDS,prob64,scores)},"bands":{b:{**metric(prob_interaction[:,sl]),"absolute_mass":math.fsum(abs(float(v)) for v in prob_interaction[:,sl].ravel()),"signed_mass":math.fsum(float(v) for v in prob_interaction[:,sl].ravel())} for b,sl in bands.items()},"per_head": [{"head":h,"kv_head":h//6,"interaction_l2":norm(prob_interaction[h]),"source_absolute_mass":math.fsum(abs(float(x)) for x in prob_interaction[h,source]),"correction_absolute_mass":math.fsum(abs(float(x)) for x in prob_interaction[h,correction]),"question_absolute_mass":math.fsum(abs(float(x)) for x in prob_interaction[h,question])} for h in range(12)]},
                    "values":{"factorial":metric(interaction(mapped_values))},"context":{"native_interaction":metric(native_context_interaction),"world_native_gemm_remainders":{w:metric(e) for w,e in zip(WORLDS,value_remainders)},"terms":signed,"ideal_term_pairwise_cross":pairwise,"account_closure":metric(native_context_interaction,prediction)},"conditional_normalizer_updates":normalizers,
                    "score_scaling":{"native_scaled_factorial":metric(interaction(scaled32)),"mathematical_bilinear_prediction":metric(bilinear/math.sqrt(128)),"constant_quantization_factorial":metric(interaction(scale_constant)),"multiplication_rounding_factorial":metric(interaction(scale_multiply)),"raw_dot_rounding_scaled_factorial":metric(interaction(score_remainders)/math.sqrt(128)),"account_closure":metric(interaction(scaled32)-bilinear/math.sqrt(128)-interaction(scale_constant)-interaction(scale_multiply)-interaction(score_remainders)/math.sqrt(128))},
                    "local_factorials":local_factorials,"attention_projection":{"ideal_O_on_context_interaction":metric(projected),"native_write_interaction":metric(actual_write_interaction),"native_minus_linear_projection":metric(actual_write_interaction,projected)},"residual_additions":{"attention_add_interaction_remainder":metric(interaction(skips)),"mlp_add_interaction_remainder":metric(interaction(return_skips))}}
            rows.append(record);observations.append({"kind":"attention-read","case":task,"step":position,"metrics":record})
    return {"summary":{"historical_full_readout_joins":joins,"historical_valid_cache_joins":joins,"selected_traces":16,"worlds":list(WORLDS),"work":work,"answers":texts,"no_compact_state_or_semantic_necessity_claim":True},"attention_reads":rows,"observations":observations}
