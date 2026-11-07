"""Own-query response audit and actual recurring consumer, using saved bytes only."""
from pathlib import Path
import hashlib,json,math,struct
import numpy as np
from span_parent_analysis import sha,metric,same,bank_read,PORTS,SCALE,validate_turn,final_history,token_hash,steps,response as fsum_response
HERE=Path(__file__).resolve().parent
POLICIES=("native","exact","eight","byte","delete")

def byte_read(path):
    data=Path(path).read_bytes()
    if len(data)!=17984 or data[:8]!=b"TCMBYTE1" or struct.unpack("<6I",data[8:32])!=(1,2,128,34,8,0):raise ValueError("Byte format differs")
    scales=np.frombuffer(data[32:576],dtype="<f4").reshape(2,2,34).copy();codes=np.frombuffer(data[576:],dtype=np.int8).reshape(2,2,34,128).copy()
    if not np.isfinite(scales).all() or not (scales>0).all() or (codes==-128).any():raise ValueError("Byte operands invalid")
    decoded=np.multiply(codes.astype(np.float32),scales[...,None],dtype=np.float32)
    return {"m":34,"counts":np.ones((2,34),dtype=np.float32),"keys":decoded[0],"values":decoded[1],"scales":scales,"codes":codes}

def response(q,k,v,counts=None):
    q=np.asarray(q,dtype=np.float64).reshape(12,128);k=np.asarray(k,dtype=np.float64);v=np.asarray(v,dtype=np.float64);c=np.ones(k.shape[:2]) if counts is None else counts
    grouped=k[np.arange(12)//6];scores=np.einsum("hd,hid->hi",q,grouped,optimize=False)*SCALE+np.log(c[np.arange(12)//6]);g=np.max(scores,axis=1);e=np.exp(scores-g[:,None]);z=np.sum(e,axis=1,dtype=np.float64)
    mean=np.einsum("hi,hid->hd",e,v[np.arange(12)//6],optimize=False)/z[:,None]
    return np.column_stack((g+np.log(z),mean))

def distribution(values):
    a=np.asarray(values,dtype=float).ravel()
    return {"n":a.size,"mean":float(np.mean(a)),"min":float(np.min(a)),"median":float(np.median(a)),"max":float(np.max(a))}

def analyze(raw,bank,driver,texts):
    raw=Path(raw);ref=json.loads((HERE/"origin.json").read_text());questions=json.loads((HERE/"questions.json").read_text())
    if driver["schema"]!="tcm-span-recurrent-driver-v1" or driver["layer"]!=17 or driver["capacity"]!=512 or not driver["programs_frozen_before_question_files_opened"] or driver["constructor_future_queries"] or driver["constructor_position"]!=93:raise ValueError("Constructor boundary differs")
    if driver["source_history_ids"]!=ref["source_history_ids"] or driver["source_kv_sha256"]!=ref["source_kv_sha256"]:raise ValueError("Source differs")
    correction=driver["correction"];validate_turn(correction)
    if correction!=ref["correction"]:raise ValueError("Native correction record differs")
    for label in ("prefill","final"):
        if sha(raw/f"correction-{label}.f32")!=correction[label]["sha256"]:raise ValueError("Correction bytes differ")
    def load(file,shape,dtype="<f4"):
        x=np.fromfile(raw/file,dtype=dtype)
        if x.size!=math.prod(shape) or not np.isfinite(x).all():raise ValueError("Invalid evidence array: "+file)
        return x.reshape(shape)
    programs={};storage=[]
    if [p["name"] for p in driver["programs"]]!=["eight","exact","byte"]:raise ValueError("Program set differs")
    for p,after in zip(driver["programs"],driver["programs_after"]):
        name=p["name"];path=raw/p["file"]
        if sha(path)!=p["file_sha256"] or path.stat().st_size!=p["bytes"] or after!={"name":name,"gpu_after_sha256":p["gpu_before_sha256"]}:raise ValueError("Frozen bank changed")
        b=byte_read(path) if name=="byte" else bank_read(path);programs[name]=b
        if name=="byte":gpu=path.read_bytes()[32:]
        else:
            logs=load("bank-"+name+"-logweights.f32",(2,b["m"]));gpu=b["keys"].tobytes()+b["values"].tobytes()+logs.tobytes()
        if hashlib.sha256(gpu).hexdigest()!=p["gpu_before_sha256"] or len(gpu)!=p["gpu_bytes"]:raise ValueError("GPU bank identity differs")
        storage.append({"name":name,"file_bytes":p["bytes"],"gpu_bytes":p["gpu_bytes"],"raw_span_bytes":69632,"file_over_raw_span":p["bytes"]/69632})
    poison=driver["poisoned_root"];expected_poison=hashlib.sha256(struct.pack("<I",0x7fc00000)*(69632//4)).hexdigest()
    if poison["position"]!=93 or poison["epoch"]!=poison["native_epoch"]+1 or poison["pattern_u32"]!=0x7fc00000 or poison["span_bytes_sha256"]!=expected_poison or poison["bytes"]!=69632 or not poison["old_frames_discarded"] or not poison["dense_L17_fallback_refused"] or poison["physical_cache_allocation_released"]:raise ValueError("Poison/root policy differs")
    if [c["policy"] for c in driver["chains"]]!=list(POLICIES):raise ValueError("Policy set differs")
    all_rows=[];consumers=[];observations=[];captures={};total_reads=0;all_steps={};native_turns=driver["chains"][0]["turns"]
    # Validate native/own history first, before consulting original audit span.
    for chain,mode in zip(driver["chains"],(-1,34,8,34,0)):
        label=chain["policy"];history=final_history(correction);chain_steps=[];poisoned=label!="native"
        if chain["mode"]!=mode or chain["poisoned"]!=poisoned or chain["root_epoch"]!=(poison["epoch"] if poisoned else poison["native_epoch"]) or not chain["parent_intact"] or chain["poison_intact"]!=poisoned or len(chain["turns"])!=3:raise ValueError("Recurring root differs")
        if chain["parent_complement_cache_sha256"]!=driver["chains"][0]["parent_complement_cache_sha256"] or poisoned and chain["parent_raw_cache_sha256"]!=poison["raw_cache_sha256"]:raise ValueError("Root cache differs")
        for ti,(turn,q) in enumerate(zip(chain["turns"],questions)):
            name=f"{label}-t{ti}";template=list(map(int,(raw/q["file"]).read_text().split()));skip=int(history[-1]==template[0]);start_history=list(history)
            if turn["name"]!=name or turn["turn"]!=ti or turn["parent_position"]!=len(history) or turn["parent_history_ids"]!=history or turn["template_ids"]!=template or turn["template_skip"]!=skip or turn["budget"]!=q["budget"]:raise ValueError("Linked user history differs")
            prefill=turn["steps"];gen=turn["generation_steps"];answers=turn["answer_ids"]
            if len(prefill)!=len(template)-skip or not answers or len(answers)>q["budget"] or len(gen)!=len(answers) or not turn["all_delivered_consumed"] or turn["stopped"]!=(answers[-1] in (151645,151643)) or turn["truncated"]==turn["stopped"] or any(t in (151645,151643) for t in answers[:-1]) or turn["truncated"] and len(answers)!=q["budget"]:raise ValueError("Generation closure differs")
            pending=None
            for index,step in enumerate(prefill+gen):
                own=index>=len(prefill);token=answers[index-len(prefill)] if own else template[skip+index]
                if own and token!=pending:raise ValueError("Not own greedy token")
                history.append(token);m=34 if mode<0 else mode
                if step["step"]!=index or step["position"]!=len(history)-1 or step["input_id"]!=token or step["history_sha256"]!=token_hash(history) or step["role"]!=("own-generated" if own else "template-user") or step["selected_read_applications"]!=1 or step["packed_length"]!=len(history)-34+m or step["original_span_rows_read_by_selected_attention"]!=(34 if mode<0 else 0) or step["span_dot_and_value_macs"]!=2*12*m*128:raise ValueError("Read execution/clock differs")
                qv=load(step["query_file"],(12,128));pending=step["top"];chain_steps.append((step,qv));total_reads+=1
            if turn["final_position"]!=len(history):raise ValueError("Turn final clock differs")
            for clock,step in (("prefill",prefill[-1]),("final",gen[-1])):
                point=f"{name}-{clock}";port={p:load(f"{point}-{p}.f32",(n,)) for p,n in PORTS.items()};port["kv"]=load(f"{point}-kv17-18-27.f32",(2,3,2,128));port["logits"]=load(f"{name}-{clock}.f32",(151936,))
                if step["capture"]!=point or turn[clock]["top"]!=step["top"] or int(np.argmax(port["logits"]))!=step["top"] or sha(raw/f"{name}-{clock}.f32")!=turn[clock]["sha256"]:raise ValueError("Readout/port capture differs")
                if not same(np.add(port["input"],port["attention_write"]),port["raw_receiver"]) or not same(np.add(port["raw_receiver"],port["mlp_return"]),port["output"]):raise ValueError("Residual additions differ")
                if label in programs:port["prediction"]=load(f"{point}-response.f64",(12,129),"<f8")
                captures[label,ti,clock]=port
                if label!="native":
                    native=captures["native",ti,clock];native_history=list(native_turns[ti]["parent_history_ids"])+template[int(native_turns[ti]["template_skip"]):]
                    if clock=="final":native_history+=native_turns[ti]["answer_ids"]
                    actual_history=history if clock=="final" else start_history+template[skip:]
                    consumer={"policy":label,"turn":ti,"clock":clock,"same_complete_history_as_native":actual_history==native_history,"same_clock_as_native":len(actual_history)==len(native_history),"ports":{p:metric(port[p],native[p]) for p in ("query","attention","attention_write","mlp_return","output","logits","kv")},"complement_cache_equal_native":turn[f"{clock}_complement_cache_sha256"]==native_turns[ti][f"{clock}_complement_cache_sha256"],"raw_cache_equal_native":turn[f"{clock}_kv_sha256"]==native_turns[ti][f"{clock}_kv_sha256"]}
                    consumers.append(consumer);observations.append({"kind":"recurring-span-consumer","case":point,"metrics":consumer})
            observations.append({"kind":"own-behavior","case":name,"text":texts[name],"answer_ids":answers,"stopped":turn["stopped"],"truncated":turn["truncated"]})
        all_steps[label]=chain_steps
    # Exact34 reproduces every native query and sparse consumer; raw poison differs by design.
    if len(all_steps["native"])!=len(all_steps["exact"]):raise ValueError("Exact recurrence length differs")
    for (a,q),(b,r) in zip(all_steps["native"],all_steps["exact"]):
        if any(a[k]!=b[k] for k in ("position","input_id","history_sha256","top","packed_length")) or not same(q,r):raise ValueError("Exact34 recurrent token/query parity differs")
    for t in range(3):
        if native_turns[t]["answer_ids"]!=driver["chains"][1]["turns"][t]["answer_ids"]:raise ValueError("Exact34 own answer differs")
        for clock in ("prefill","final"):
            a,b=captures["native",t,clock],captures["exact",t,clock]
            if any(not same(a[p],b[p]) for p in (*PORTS,"kv","logits")):raise ValueError("Exact34 sparse port parity differs")
            c=next(c for c in consumers if c["policy"]=="exact" and c["turn"]==t and c["clock"]==clock)
            if not c["complement_cache_equal_native"] or c["raw_cache_equal_native"]:raise ValueError("Masked/raw cache distinction differs")
    # Primary predictions above are already sealed. Original span now enters audit only.
    keys=load("audit-span-keys.f32",(2,34,128));values=load("audit-span-values.f32",(2,34,128))
    if not same(programs["exact"]["keys"],keys) or not same(programs["exact"]["values"],values):raise ValueError("Exact bank/source identity differs")
    byte=programs["byte"];source=np.stack((keys,values));scales=np.max(np.abs(source),axis=-1)/np.float32(127);scales=np.where(scales>0,scales,np.float32(1)).astype(np.float32);ratio=np.divide(source,scales[...,None],dtype=np.float32).astype(float);codes=np.clip(np.copysign(np.floor(np.abs(ratio)+0.5),ratio),-127,127).astype(np.int8)
    if not same(scales,byte["scales"]) or not same(codes,byte["codes"]):raise ValueError("Memory-only byte constructor differs")
    key_error=np.linalg.norm(byte["keys"].astype(float)-keys,axis=-1);value_error=np.linalg.norm(byte["values"].astype(float)-values,axis=-1);eta_k=np.max(key_error,axis=1);eta_v=np.max(value_error,axis=1);vmax=np.max(np.linalg.norm(values.astype(float),axis=-1),axis=1)
    summaries=[]
    for label in POLICIES:
        rows=[]
        for step,q in all_steps[label]:
            exact=response(q,keys,values)
            if label=="delete":ratio=np.zeros(12);mean=np.zeros((12,128))
            else:
                b=programs["exact" if label=="native" else label];pred=response(q,b["keys"],b["values"],b["counts"]);ratio=np.exp(pred[:,0]-exact[:,0]);mean=pred[:,1:]
                if step["capture"] and label in programs:
                    primary=load(step["capture"]+"-response.f64",(12,129),"<f8");reference=fsum_response(q,b["keys"],b["values"],b["counts"]);observations.append({"kind":"primary-bank-arithmetic","case":step["capture"],"metrics":metric(primary,reference)})
            numerator=ratio[:,None]*mean-exact[:,1:];den=np.linalg.norm(exact[:,1:],axis=1);ne=np.linalg.norm(numerator,axis=1)/den;me=np.linalg.norm(mean-exact[:,1:],axis=1)/den
            row={"policy":label,"query_file":step["query_file"],"position":step["position"],"history_sha256":step["history_sha256"],"Z_ratios":ratio.tolist(),"numerator_relative_errors":ne.tolist(),"span_mean_relative_errors":me.tolist()}
            if label=="byte":
                eps=np.linalg.norm(q.astype(float),axis=1)*SCALE*eta_k[np.arange(12)//6];bound=eta_v[np.arange(12)//6]+2*vmax[np.arange(12)//6]*np.tanh(eps/2);actual=np.linalg.norm(mean-exact[:,1:],axis=1)
                row.update(score_error_bound=eps.tolist(),log_Z_ratio=np.log(ratio).tolist(),conditional_mean_absolute_bound=bound.tolist(),conditional_mean_absolute_error=actual.tolist(),query_norms=np.linalg.norm(q.astype(float),axis=1).tolist())
            rows.append(row);all_rows.append(row)
        summary={"policy":label,"selected_reads":len(rows),"correlated_head_reads":12*len(rows),"Z_ratio":distribution([r["Z_ratios"] for r in rows]),"numerator_relative_error":distribution([r["numerator_relative_errors"] for r in rows]),"span_mean_relative_error":distribution([r["span_mean_relative_errors"] for r in rows])}
        summaries.append(summary);observations.append({"kind":"own-query-response-summary","case":label,"metrics":summary})
    summary={"policies":5,"linked_turns_per_policy":3,"source_advances":59,"correction_advances":len(steps(correction)),"selected_read_applications":total_reads,"exact34_native_every_query_sparse_ports_logits_and_complement_cache":True,"original_C17_unavailable_to_candidate_read_path":True,"physical_cache_allocation_released":False,"program_storage":storage,"own_query_responses":summaries,"answers":texts,"scope":"recurring compressed fixed L17 span only; dense complement, other layers and full weights/allocation retained; no compressed ingestion or serving speedup claim"}
    return {"summary":summary,"byte_source_reconstruction":{"key_max_row_l2":eta_k.tolist(),"value_max_row_l2":eta_v.tolist(),"original_value_max_row_l2":vmax.tolist()},"own_query_response_rows":all_rows,"consumer_comparisons":consumers,"observations":observations}
