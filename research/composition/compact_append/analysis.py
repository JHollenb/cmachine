"""Audit growing owned caches, own-row codec error, and cross-layer consumers."""
from pathlib import Path
import hashlib,json,math,struct
import numpy as np
from span_parent_analysis import sha,metric,same,PORTS,SCALE,validate_turn,final_history,token_hash
HERE=Path(__file__).resolve().parent
POLICIES=("native","float","byte17","byte17_18")

def encode(x):
    x=np.asarray(x,dtype=np.float32);s=np.max(np.abs(x),axis=-1)/np.float32(127);s=np.where(s>0,s,np.float32(1)).astype(np.float32);r=np.divide(x,s[...,None],dtype=np.float32).astype(np.float64);codes=np.clip(np.copysign(np.floor(np.abs(r)+.5),r),-127,127).astype(np.int8)
    return codes,s,np.multiply(codes.astype(np.float32),s[...,None],dtype=np.float32)

def read_program(path):
    raw=Path(path).read_bytes()
    if raw[:8]!=b"TCMAPND1" or len(raw)<32:raise ValueError("Cache program magic differs")
    version,layer,heads,dim,n,bits=struct.unpack("<6I",raw[8:32])
    if (version,heads,dim,bits)!=(1,2,128,8) or layer not in (17,18) or not 0<n<=512 or len(raw)!=32+n*528:raise ValueError("Cache program schema differs")
    scales=np.frombuffer(raw[32:32+16*n],dtype="<f4").copy().reshape(2,2,n);codes=np.frombuffer(raw[32+16*n:],dtype=np.int8).copy().reshape(2,2,n,128)
    if not np.isfinite(scales).all() or not (scales>0).all() or (codes==-128).any():raise ValueError("Invalid cache program operands")
    return layer,codes,scales,np.multiply(codes.astype(np.float32),scales[...,None],dtype=np.float32)

def response(q,k,v):
    q=np.asarray(q,dtype=np.float64);k=np.asarray(k,dtype=np.float64);v=np.asarray(v,dtype=np.float64);group=np.arange(12)//6
    score=np.einsum("hd,hid->hi",q,k[group],optimize=False)*SCALE;g=np.max(score,axis=1);e=np.exp(score-g[:,None]);z=np.sum(e,axis=1,dtype=np.float64);mean=np.einsum("hi,hid->hd",e,v[group],optimize=False)/z[:,None]
    return np.column_stack((g+np.log(z),mean))

def stats(x):
    x=np.asarray(x,dtype=float).ravel();return {"n":int(x.size),"mean":float(np.mean(x)),"median":float(np.median(x)),"min":float(np.min(x)),"max":float(np.max(x))}

def analyze(raw,bank,driver,texts):
    raw=Path(raw);origin=json.loads((HERE/"origin.json").read_text());questions=json.loads((HERE/"questions.json").read_text())
    if driver["schema"]!="tcm-compact-append-driver-v1" or driver["capacity"]!=512 or driver["layers"]!=[17,18] or not driver["prefix_programs_frozen_before_question_files_opened"] or driver["constructor_future_queries"]:raise ValueError("Construction boundary differs")
    if driver["source_history_ids"]!=origin["source_history_ids"] or driver["source_kv_sha256"]!=origin["source_kv_sha256"]:raise ValueError("Source differs")
    correction=driver["correction"];validate_turn(correction)
    if correction!=origin["correction"]:raise ValueError("Native correction differs")
    def load(name,shape,dtype="<f4"):
        a=np.fromfile(raw/name,dtype=dtype)
        if a.size!=math.prod(shape) or not np.isfinite(a).all():raise ValueError("Invalid array: "+name)
        return a.reshape(shape)
    for clock in ("prefill","final"):
        if sha(raw/f"correction-{clock}.f32")!=correction[clock]["sha256"]:raise ValueError("Historical correction readout differs")
    root=load("root-kv.f32",(2,28,2,93,128))
    if sha(raw/"root-kv.f32")!=correction["final_kv_sha256"]:raise ValueError("Native hydration prefix differs")
    prefix={}
    for li,layer in enumerate((17,18)):
        l,c,s,d=read_program(raw/f"prefix-l{layer}.bin");ec,es,ed=encode(root[:,layer])
        if l!=layer or not same(c,ec) or not same(s,es) or not same(d,ed) or sha(raw/f"prefix-l{layer}.bin")!=driver["prefix_program_sha256"][li]:raise ValueError("Source-only prefix encoding differs")
        prefix[layer]=(c,s,d)
    if len(questions)!=3 or [c["policy"] for c in driver["chains"]]!=list(POLICIES) or any(len(c["turns"])!=3 for c in driver["chains"]):raise ValueError("Policy/turn membership differs")
    all_steps={};all_caps={};consumers=[];rows=[];observations=[];memory=[];codec_rows=[];histories={};summary_response=[];native=driver["chains"][0];total_steps=0
    for mode,chain in enumerate(driver["chains"]):
        policy=chain["policy"];selected=[] if mode==0 else [17] if mode==2 else [17,18];byte_layers=selected if mode>=2 else [];dense=28-len(selected);arena=len(selected)*(270336 if mode>=2 else 1048576);workspace=524288 if byte_layers else 0;status=4 if byte_layers else 0
        expected_session=native["session_gpu_bytes"]-len(selected)*1048576
        if chain["mode"]!=mode or chain["dense_layers"]!=dense or chain["session_gpu_bytes"]!=expected_session or chain["cache_arena_bytes"]!=arena or chain["decode_workspace_bytes"]!=workspace or chain["status_bytes"]!=status or chain["steady_state_gpu_bytes"]!=expected_session+arena+workspace+status or chain["tracked_live_gpu_bytes"]!=driver["model_live_gpu_bytes"]+chain["steady_state_gpu_bytes"] or chain["native_session_already_destroyed"]!=(mode>0):raise ValueError("Live allocation/layout differs")
        memory.append({"policy":policy,"dense_layers":dense,"session_gpu_bytes":expected_session,"selected_arena_bytes":arena,"shared_decode_bytes":workspace,"status_bytes":status,"total_state_gpu_bytes":chain["steady_state_gpu_bytes"],"state_saving_vs_native_bytes":native["steady_state_gpu_bytes"]-chain["steady_state_gpu_bytes"],"total_model_and_state_gpu_bytes":chain["tracked_live_gpu_bytes"]})
        mech=chain["mechanics"];injected=mode>=2;partial=[93,0] if mode==2 else [94,93] if mode==3 else [0,0]
        if not mech["suffix_replaced"] or not mech["token_and_encoded_root_restored"] or mech["two_token_logical_cache_sha256"]!=mech["replayed_logical_cache_sha256"] or mech["injected_unpublished_append_failure"]!=injected or mech["abort_restored_root"]!=injected or mech["partial_cursors"]!=partial or sha(raw/f"{policy}-rollback-reference.f32")!=mech["full_logits_sha256"]:raise ValueError("Rollback/abort mechanics differs")
        history=final_history(correction);step_records=[];caches={l:root[:,l].copy() for l in (17,18)};decoded={l:(prefix[l][2].copy() if l in byte_layers else root[:,l].copy()) for l in (17,18)};response_by_layer={17:[],18:[]};snapshot_count=0
        for ti,(turn,q) in enumerate(zip(chain["turns"],questions)):
            name=f"{policy}-t{ti}";template=list(map(int,(raw/q["file"]).read_text().split()));skip=int(history[-1]==template[0]);start=list(history);prefill=turn["steps"];gen=turn["generation_steps"];answer=turn["answer_ids"]
            if turn["name"]!=name or turn["turn"]!=ti or turn["budget"]!=q["budget"] or turn["parent_position"]!=len(history) or turn["parent_history_ids"]!=history or turn["template_ids"]!=template or turn["template_skip"]!=skip or len(prefill)!=len(template)-skip:raise ValueError("Linked turn inputs differ")
            if not answer or len(answer)>q["budget"] or len(gen)!=len(answer) or not turn["all_delivered_consumed"] or turn["stopped"]!=(answer[-1] in (151645,151643)) or turn["truncated"]==turn["stopped"] or any(x in (151645,151643) for x in answer[:-1]) or turn["truncated"] and len(answer)!=q["budget"]:raise ValueError("Own generation closure differs")
            pending=None
            for index,step in enumerate(prefill+gen):
                own=index>=len(prefill);token=answer[index-len(prefill)] if own else template[skip+index]
                if own and token!=pending:raise ValueError("Not own greedy continuation")
                history.append(token)
                if step["step"]!=index or step["position"]!=len(history)-1 or step["input_id"]!=token or step["role"]!=("own-generated" if own else "template-user") or step["history_sha256"]!=token_hash(history) or step["encoded_cursors"]!=[len(history)]*len(selected):raise ValueError("Append clock/cursor differs")
                pending=step["top"];entry={"step":step,"turn":ti,"name":name,"layers":{}};total_steps+=1
                for layer in (17,18):
                    stem=f"{name}-s{index:03d}-l{layer}";query=load(stem+"-query.f32",(12,128));key=load(stem+"-key.f32",(2,128));value=load(stem+"-value.f32",(2,128));own_row=np.stack((key,value));caches[layer]=np.concatenate((caches[layer],own_row[:,:,None,:]),axis=2)
                    code,scale,effective=encode(own_row);effective=effective if layer in byte_layers else own_row;decoded[layer]=np.concatenate((decoded[layer],effective[:,:,None,:]),axis=2)
                    original_response=response(query,caches[layer][0],caches[layer][1]);encoded_response=response(query,decoded[layer][0],decoded[layer][1]);ratio=np.exp(encoded_response[:,0]-original_response[:,0]);mu=original_response[:,1:];den=np.linalg.norm(mu,axis=1);ne=np.linalg.norm(ratio[:,None]*encoded_response[:,1:]-mu,axis=1)/den;me=np.linalg.norm(encoded_response[:,1:]-mu,axis=1)/den
                    rr={"policy":policy,"turn":ti,"position":step["position"],"layer":layer,"history_sha256":step["history_sha256"],"Z_ratios":ratio.tolist(),"numerator_relative_errors":ne.tolist(),"mean_relative_errors":me.tolist()};rows.append(rr);response_by_layer[layer].append(rr)
                    entry["layers"][layer]={"query":query,"key":key,"value":value}
                    if layer in byte_layers:codec_rows.append({"policy":policy,"layer":layer,"position":step["position"],"key_codec":metric(effective[0],key),"value_codec":metric(effective[1],value)})
                    if step["capture"]:
                        point=f"{name}-{step['capture']}-l{layer}";ports={p:load(f"{point}-{p}.f32",(n,)) for p,n in PORTS.items()}
                        if not same(ports["query"].reshape(12,128),query) or not same(ports["key"].reshape(2,128),key) or not same(ports["value"].reshape(2,128),value) or not same(np.add(ports["input"],ports["attention_write"]),ports["raw_receiver"]) or not same(np.add(ports["raw_receiver"],ports["mlp_return"]),ports["output"]):raise ValueError("Native port identities differ")
                        ports["encoded_context_ideal"]=encoded_response[:,1:];all_caps[policy,ti,step["capture"],layer]=ports
                        if layer in byte_layers:
                            l,bc,bs,bd=read_program(raw/f"{point}.bin");ec,es,ed=encode(caches[layer])
                            if l!=layer or not same(bc,ec) or not same(bs,es) or not same(bd,ed):raise ValueError("Immediate GPU codec/cache ownership differs")
                            snapshot_count+=1
                step_records.append(entry)
            if turn["final_position"]!=len(history):raise ValueError("Final turn clock differs")
            for clock in ("prefill","final"):
                y=load(f"{name}-{clock}.f32",(151936,));step=prefill[-1] if clock=="prefill" else gen[-1]
                if sha(raw/f"{name}-{clock}.f32")!=turn[clock]["sha256"] or int(np.argmax(y))!=step["top"] or turn[clock]["top"]!=step["top"]:raise ValueError("Full readout identity differs")
                actual_history=history if clock=="final" else start+template[skip:];histories[policy,ti,clock]=list(actual_history)
                all_caps[policy,ti,clock,"logits"]=y
                if mode:
                    same_history=actual_history==histories["native",ti,clock];native_y=all_caps["native",ti,clock,"logits"];consumer={"policy":policy,"turn":ti,"clock":clock,"same_complete_history_as_native":same_history,"same_clock_as_native":len(actual_history)==len(histories["native",ti,clock]),"full_logits":metric(y,native_y),"layers":{}}
                    for layer in (17,18):
                        cap=all_caps[policy,ti,clock,layer];base=all_caps["native",ti,clock,layer];consumer["layers"][str(layer)]={"ports":{p:metric(cap[p],base[p]) for p in ("query","key","value","attention","attention_write","mlp_return","output")},"actual_context_vs_own_encoded_ideal":metric(cap["attention"].reshape(12,128),cap["encoded_context_ideal"])}
                    consumers.append(consumer);observations.append({"kind":"compact-append-consumer","case":f"{name}-{clock}","metrics":consumer})
                if policy=="native":
                    old=origin["historical_native_turns"][ti]
                    if turn[clock]!=old[clock] or turn[f"{clock}_logical_cache_sha256"]!=old[f"{clock}_kv_sha256"] or turn["answer_ids"]!=old["answer_ids"]:raise ValueError("Historical native trajectory differs")
                if policy=="float":
                    if not same(y,all_caps["native",ti,clock,"logits"]) or turn[f"{clock}_logical_cache_sha256"]!=native["turns"][ti][f"{clock}_logical_cache_sha256"]:raise ValueError("Exact callback readout/cache parity differs")
                    for layer in (17,18):
                        if any(not same(all_caps[policy,ti,clock,layer][p],all_caps["native",ti,clock,layer][p]) for p in PORTS):raise ValueError("Exact callback port parity differs")
            observations.append({"kind":"own-behavior","case":name,"text":texts[name],"answer_ids":answer,"stopped":turn["stopped"],"truncated":turn["truncated"]})
        all_steps[policy]=step_records
        if chain["cursor_rewinds"]!=(0 if mode==0 else 3 if mode==1 else 6):raise ValueError("Encoded rewind calls differ")
        expected_calls=len(selected)*(len(step_records)+5)+(0 if mode<2 else 2*len(selected))
        if chain["owned_step_calls_including_mechanics"]!=expected_calls or chain["published_rows_including_mechanics"]!=expected_calls-(1 if mode>=2 else 0):raise ValueError("Append physical work differs")
        for l,rs in response_by_layer.items():
            r={"policy":policy,"layer":l,"reads":len(rs),"correlated_head_reads":12*len(rs),"Z_ratio":stats([x["Z_ratios"] for x in rs]),"numerator_relative_error":stats([x["numerator_relative_errors"] for x in rs]),"mean_relative_error":stats([x["mean_relative_errors"] for x in rs])};summary_response.append(r);observations.append({"kind":"owned-cache-response","case":f"{policy}-l{l}","metrics":r})
    # Full exact callback/native query+row parity at every transition, beyond sparse ports.
    if len(all_steps["native"])!=len(all_steps["float"]):raise ValueError("Exact callback clocks differ")
    for a,b in zip(all_steps["native"],all_steps["float"]):
        if any(a["step"][k]!=b["step"][k] for k in ("step","position","input_id","role","top","history_sha256","capture")):raise ValueError("Exact callback transition differs")
        for l in (17,18):
            if any(not same(a["layers"][l][p],b["layers"][l][p]) for p in ("query","key","value")):raise ValueError("Exact callback every-row/query parity differs")
    composition=[]
    for policy,baseline in (("byte17","native"),("byte17_18","native"),("byte17_18","byte17")):
        for a,b in zip(all_steps[policy],all_steps[baseline]):
            domain=a["step"]["history_sha256"]==b["step"]["history_sha256"] and a["step"]["position"]==b["step"]["position"]
            composition.append({"policy":policy,"baseline":baseline,"position":a["step"]["position"],"same_history_and_clock":domain,"Q17":metric(a["layers"][17]["query"],b["layers"][17]["query"]),"Q18":metric(a["layers"][18]["query"],b["layers"][18]["query"]),"K18":metric(a["layers"][18]["key"],b["layers"][18]["key"]),"V18":metric(a["layers"][18]["value"],b["layers"][18]["value"])})
    for policy,baseline in (("byte17","native"),("byte17_18","native"),("byte17_18","byte17")):
        cs=[c for c in composition if c["policy"]==policy and c["baseline"]==baseline and c["same_history_and_clock"]];observations.append({"kind":"cross-layer-query-composition","case":policy+"-vs-"+baseline,"metrics":{"same_history_rows":len(cs),"Q18_relative_l2":stats([c["Q18"]["relative_l2"] for c in cs]) if cs else None}})
    summary={"sessions":4,"linked_turns_per_session":3,"main_token_transitions":total_steps,"selected_byte_layer_appends":sum(1 for r in codec_rows),"initial_prefix_rows":93,"capacity":512,"source_advances":59,"correction_advances":34,"exact_callback_native_ports_queries_rows_readouts_and_logical_cache":True,"historical_native_boundaries":6,"selected_layers_have_no_dense_slots":True,"current_encoded_row_consumed":True,"rollback_and_partial_abort_closed":True,"allocation":memory,"response_summaries":summary_response,"answers":texts,"scope":"growing row-quantized selected-layer caches; full model weights and other26/27layers dense; finite supplied-fact conversation; no constant-size semantic state or serving speedup claim"}
    return {"summary":summary,"own_codec_rows":codec_rows,"own_response_rows":rows,"consumer_comparisons":consumers,"cross_layer_query_rows":composition,"observations":observations}
