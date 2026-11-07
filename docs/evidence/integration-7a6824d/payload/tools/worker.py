"""Finite Beast worker: compile C, retain native comparisons and own-state paths."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "TCM_RESULT_GZIP_B64="
TRACE_PORT_NAMES=("input","query","key","value","attention","mlp_input","mlp_gate","mlp_up","mlp_return","output","attention_write","raw_receiver","consumed_attention_write","consumed_mlp_write")


def validate_trace_ports(descriptors):
    if len(descriptors)!=len(TRACE_PORT_NAMES) or {x["port"] for x in descriptors}!=set(TRACE_PORT_NAMES):
        raise ValueError("selected-port instrument is incomplete or duplicated")


def preserve(report):
    path=ROOT/"results/partial-report.json"
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def run(command, **kwargs):
    print("[run]", " ".join(map(str,command)), flush=True)
    result = subprocess.run(list(map(str,command)), capture_output=True, text=True, timeout=180, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr[-8000:]} {result.stdout[-1000:]}")
    return result


def read_bundle(path):
    import numpy as np
    raw = Path(path).read_bytes()
    if raw[:8] != b"TCM1LE\0\0" or hashlib.sha256(raw[40:]).digest()!=raw[8:40]:
        raise ValueError("bundle identity invalid")
    arch,count = struct.unpack_from("<II",raw,40)
    tensors = {}
    for i in range(count):
        name,rank,dtype,*rest=struct.unpack_from("<64sII4QQQ",raw,176+120*i)
        shape,offset,size=rest[:rank],rest[4],rest[5]
        tensors[name.split(b"\0")[0].decode()]=np.frombuffer(raw,dtype={1:"<f4",2:"<u2",3:"<f8"}[dtype],count=size//{1:4,2:2,3:8}[dtype],offset=offset).reshape(shape).copy()
    return arch,tensors


def environment():
    paths = ["/usr/local/cuda/bin/nvcc", "/mnt/hdd8tb/llamacpp-kv4/llama.cpp/include/llama.h",
             "/mnt/hdd8tb/llamacpp-kv4/llama.cpp/ggml/include/ggml.h",
             "/mnt/hdd8tb/llamacpp-kv4/llama.cpp/build/bin/libllama.so"]
    return {"platform":platform.platform(),"python":sys.version,"paths":{p:Path(p).exists() for p in paths},
            "cc":shutil.which("cc"),"nvcc":shutil.which("nvcc"),"job_id":os.environ.get("MRUN_JOB_ID")}


def build_cuda():
    build=run(["make","-C",ROOT,"cuda"])
    return {"environment":environment(),"build_stdout":build.stdout,"build_stderr":build.stderr,
            "binary_sha256":hashlib.sha256((ROOT/"build/tcm-cuda").read_bytes()).hexdigest(),
            "scope":"CPU compilation only; no GPU kernels or model execution"}


def c_call(model, program, registers, flags, *, steps=4, extra=(), name="call"):
    output=ROOT/"results"/f"{name}.json"
    cmd=[ROOT/"build/tcm","isa",model,"--program",program,"--registers",",".join(f"{x:016x}" for x in registers),
         "--flags","".join(map(str,flags)),"--steps",str(steps),"--output",output,*extra]
    run(cmd)
    return json.loads(output.read_text())


def native_call(model, code, registers, flags, *, anchored_input=None):
    import torch
    window=code[:15]+b"\0"*max(0,15-len(code))
    wb=torch.tensor([[(b>>i)&1 for b in window for i in range(8)]],dtype=torch.float32)
    rb=torch.tensor([[[(word>>i)&1 for i in range(64)] for word in registers]],dtype=torch.float32)
    fb=torch.tensor([flags],dtype=torch.float32)
    handle=None
    if anchored_input is not None:
        proposed=torch.tensor(anchored_input,dtype=torch.float32)
        def install(_module,_args,output):
            result=output.clone();result[:,8,:]=proposed;return result
        handle=model.mixers[2].register_forward_hook(install)
    try:
        with torch.inference_mode():
            output=model(wb,rb,fb)
    finally:
        if handle is not None:handle.remove()
    result={k:v[0].detach().cpu().tolist() for k,v in output.items()}
    return result


def decode(heads):
    registers=[sum((1<<i) for i,x in enumerate(row) if x>0) for row in heads["regs"]]
    flags=[int(x>0) for x in heads["flags"]]
    length=max(range(15),key=lambda i:heads["length"][i])+1
    return registers,flags,length


def comparisons(c,native):
    import numpy as np
    rows={}
    for name in native:
        a,b=np.asarray(c[name]),np.asarray(native[name])
        diff=a-b
        rows[name]={"max_abs_error":float(np.max(np.abs(diff))),"rms_error":float(np.sqrt(np.mean(diff*diff))),
                    "sign_mismatches":int(np.sum((a>0)!=(b>0))),"elements":int(a.size)}
        if name in ("dest","src","width","length"):
            rows[name]["argmax_equal"]=bool(np.argmax(a)==np.argmax(b))
    return rows


def isa_experiment():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/"native"))
    from direct_effect.model import Executor
    _,weights=read_bundle(ROOT/"models/isa-r3.tcm")
    model=Executor("mamba",store=True,aligned=True)
    model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()},strict=True);model.eval()
    run(["make","-C",ROOT])
    report={"schema":"tcm-isa-experiment-v1","environment":environment(),"paths":[],"quadratic":[],
            "mechanics_status":"complete","instrument_status":"complete","trend_status":"observations","terminal_status":"not-requested"}
    programs=["4883c701"*4,"4883c001"*4,"4883c7014883ef014883c7024883ef02",
              "4883c701"*4]
    for index,program in enumerate(programs):
        rng=random.Random(0xC011AB1E+index)
        registers=[rng.getrandbits(64) for _ in range(8)]
        registers[0 if index==1 else 7]=[0x012345FF,0x34567FFF,0x5678FFFF,0x789FFFFF][index]
        flags=[0,0,0,0] if index<3 else [1,0,1,1]
        c=c_call(ROOT/"models/isa-r3.tcm",program,registers,flags,name=f"equations-{index}")
        check=c_call(ROOT/"models/isa-r3.tcm",program,registers,flags,extra=("--checkpoint-row","31"),name=f"checkpoint-{index}")
        if not check["checkpoint_exact"]:raise ValueError("same-C checkpoint closure failed")
        native_rows=[];offset=0;current_regs=registers;current_flags=flags
        for step in range(4):
            if offset>=len(program)//2:break
            before={"registers":[f"{x:016x}" for x in current_regs],"flags":current_flags,"rip":offset,"instruction":step}
            heads=native_call(model,bytes.fromhex(program)[offset:],current_regs,current_flags)
            next_regs,next_flags,length=decode(heads)
            native_rows.append({"before":before,"heads":heads,"after":{"registers":[f"{x:016x}" for x in next_regs],"flags":next_flags,"rip":offset+length,"instruction":step+1}})
            current_regs,current_flags,offset=next_regs,next_flags,offset+length
        paired=[]
        for cr,nr in zip(c["steps"],native_rows):
            paired.append({"same_input":cr["before"]==nr["before"],"same_committed_state":cr["after"]==nr["after"],"heads":comparisons(cr["heads"],nr["heads"])})
        own_state=all(c["steps"][i]["before"]==c["steps"][i-1]["after"] for i in range(1,len(c["steps"])))
        if not own_state:raise ValueError("C trajectory did not consume its own prior output")
        first_divergence=next((i for i,row in enumerate(paired) if not row["same_committed_state"]),None)
        report["paths"].append({"index":index,"program":program,"c":c,"native":native_rows,"paired":paired,"own_state":own_state,"checkpoint_exact":check["checkpoint_exact"],
            "first_committed_divergence":first_divergence,"unmatched_c_steps":len(c["steps"])-len(paired),"unmatched_native_steps":len(native_rows)-len(paired)})
        preserve(report)
    # Controlled anchored intervention in both C arms. It bypasses the natural
    # receiver selection question, which remains outside this first law assay.
    law_path=ROOT/"models/post2-quadratic.tcm"
    _,law=read_bundle(law_path)
    for index,(s,t) in enumerate(((0,0),(.25,.75),(.75,-.25),(-.5,.25),(.25,-.75))):
        extra=("--law",str(law_path),"--anchor",f"{s},{t}")
        dense=c_call(ROOT/"models/isa-r3.tcm","4883c001"*2,[3,5,15,31,63,127,255,511],[0,0,0,0],steps=2,extra=extra,name=f"law-dense-{index}")
        compact=c_call(ROOT/"models/isa-r3.tcm","4883c001"*2,[3,5,15,31,63,127,255,511],[0,0,0,0],steps=2,extra=(*extra,"--quadratic"),name=f"law-compact-{index}")
        if compact["law_rows"]!=1 or compact["dense_post_rows"]+1!=dense["dense_post_rows"]:
            raise ValueError("quadratic law did not skip exactly one dense post row")
        x=np.asarray(dense["steps"][0]["law_input"],dtype=np.float32).astype(np.float64)
        gamma=weights["post.2.0.weight"].astype(np.float64);beta=weights["post.2.0.bias"].astype(np.float64)
        z=(x-x.mean())/np.sqrt(np.mean((x-x.mean())**2)+1e-5)*gamma+beta
        from math import erf,sqrt
        a=weights["post.2.1.weight"].astype(np.float64)@z+weights["post.2.1.bias"]
        gelu=np.array([.5*v*(1+erf(v/sqrt(2))) for v in a])
        formula=weights["post.2.3.weight"].astype(np.float64)@gelu+weights["post.2.3.bias"]
        branch=np.asarray(compact["steps"][0]["law_branch"])
        native_rows=[];current_regs=[3,5,15,31,63,127,255,511];current_flags=[0,0,0,0];offset=0
        for step in range(2):
            native=native_call(model,bytes.fromhex("4883c001"*2)[offset:],current_regs,current_flags,
                anchored_input=x if step==0 else None)
            current_regs,current_flags,length=decode(native);offset+=length
            native_rows.append({"heads":native,"after":{"registers":[f"{z:016x}" for z in current_regs],"flags":current_flags,"rip":offset,"instruction":step+1}})
        actual_delta=formula-law["anchor.branch"].astype(np.float64)
        predicted_delta=branch-law["anchor.branch"].astype(np.float64)
        response_norm=float(np.linalg.norm(actual_delta));response_error=float(np.linalg.norm(predicted_delta-actual_delta))
        report["quadratic"].append({"coordinates":[s,t],"c_dense":dense,"c_quadratic":compact,
            "branch_relative_error":float(np.linalg.norm(branch-formula)/np.linalg.norm(formula)),
            "response_absolute_error":response_error,"response_norm":response_norm,
            "response_relative_error":response_error/response_norm if (s or t) and response_norm else None,
            "zero_update_anchor_exact":bool(np.array_equal(branch.astype(np.float32),law["anchor.branch"])) if not(s or t) else None,
            "dense_vs_formula_max_abs":float(np.max(np.abs(np.asarray(dense["steps"][0]["law_branch"])-formula))),
            "same_committed_states":[a["after"]==b["after"] for a,b in zip(dense["steps"],compact["steps"])],
            "head_comparisons":[comparisons(a["heads"],b["heads"]) for a,b in zip(compact["steps"],dense["steps"])],
            "native_anchored":native_rows,
            "native_dense_comparisons":[comparisons(a["heads"],b["heads"]) for a,b in zip(dense["steps"],native_rows)],
            "native_quadratic_committed_equal":[a["after"]==b["after"] for a,b in zip(compact["steps"],native_rows)],
            "scope":"frozen anchored receiver intervention in both C arms; complete equations elsewhere; no natural applicability claim"})
        preserve(report)
    report["summary"]={"paired_calls":sum(len(x["paired"]) for x in report["paths"]),
        "same_committed_states":sum(y["same_committed_state"] for x in report["paths"] for y in x["paired"]),
        "max_same_input_head_error":max((z["max_abs_error"] for x in report["paths"] for y in x["paired"] if y["same_input"] for z in y["heads"].values()),default=None),
        "all_own_state":all(x["own_state"] for x in report["paths"]),"all_checkpoint_exact":all(x["checkpoint_exact"] for x in report["paths"]),
        "law_substitutions":sum(x["c_quadratic"]["law_rows"] for x in report["quadratic"])}
    return report


def qwen_experiment():
    import gc
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bundle import export_qwen

    source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    if not source.is_dir():raise FileNotFoundError(source)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run(["make","-C",ROOT,"cuda"])
    run(["make","-C",ROOT,"gpu-mechanics"])
    gpu_mechanics=json.loads(run([ROOT/"build/gpu-mechanics"]).stdout)
    model_path=Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
    metadata_path=model_path.with_suffix(".json")
    source_identity=file_sha256(source/"model.safetensors")
    reusable=False
    if model_path.exists() and metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        reusable=(metadata.get("profile_version")=="qwen25-fp32-execution-v2" and metadata.get("identity")==source_identity)
    if not reusable:
        metadata=export_qwen(source,model_path)
        metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    for filename,digest in metadata["tokenizer_files"].items():
        if file_sha256(source/filename)!=digest:raise ValueError("source tokenizer/config identity changed")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    cases=[
        ("math","Compute 37 * 24. Explain the calculation briefly.",64),
        ("recall","Remember these assignments: Mira's code is 7319. Niko's code is 2846. "
         +"The garden has a wooden gate. The afternoon sky is clear. "*12
         +"What is Mira's code? Answer with the code only.",24),
        ("chat","Why does bread rise when yeast is added? Explain in a few sentences.",80),
    ]
    prepared=[]
    for name,text,budget in cases:
        ids=tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True,return_dict=False)
        if not isinstance(ids,list) or not all(isinstance(i,int) for i in ids):raise TypeError("tokenizer must return a flat token-ID list")
        token_path=ROOT/"results"/f"qwen-{name}-input.txt"
        token_path.write_text(" ".join(map(str,ids))+"\n")
        prepared.append({"name":name,"prompt":text,"ids":ids,"budget":budget,"tokens_path":str(token_path)})
    report={"schema":"tcm-qwen-experiment-v1","environment":environment(),"model":metadata,"cases":prepared,
        "gpu_mechanics":gpu_mechanics,
        "mechanics_status":"complete","instrument_status":"complete","trend_status":"observations","terminal_status":"not-requested",
        "consumer":"greedy generation from each execution's own state; tokenizer/template outside numerical C runtime",
        "torch_version":torch.__version__}
    import inspect
    import transformers
    from transformers.models.qwen2 import modeling_qwen2
    from transformers import cache_utils
    report["native_implementation"]={"transformers_version":transformers.__version__,
        "qwen2_source_sha256":file_sha256(inspect.getfile(modeling_qwen2)),
        "cache_source_sha256":file_sha256(inspect.getfile(cache_utils))}
    preserve(report)
    # One native model resident at a time. Cache and temporary logits are released
    # before the C subprocess acquires its independent parameter bank and cache.
    for label,dtype in (("native_fp32",torch.float32),("native_bf16",torch.bfloat16)):
        print("[native-load]",label,flush=True)
        model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=dtype,attn_implementation="eager",device_map={"":"cuda"},local_files_only=True).eval()
        for case in prepared:
            ids=torch.tensor([case["ids"]],dtype=torch.long,device="cuda")
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():out=model(input_ids=ids,use_cache=True,return_dict=True)
            torch.cuda.synchronize();prefill=time.monotonic()-start
            logits=out.logits[0,-1].float()
            if label=="native_fp32":
                first=logits.cpu().numpy().copy();np.save(ROOT/"results"/f"qwen-{case['name']}-native-first.npy",first)
            cache=out.past_key_values;generated=[];margins=[];start=time.monotonic()
            with torch.inference_mode():
                for step in range(case["budget"]):
                    token=int(torch.argmax(logits).item());generated.append(token)
                    top=torch.topk(logits,2).values;margins.append(float((top[0]-top[1]).item()))
                    if token in metadata["native_eos_token_ids"] or step+1==case["budget"]:break
                    out=model(input_ids=torch.tensor([[token]],device="cuda"),past_key_values=cache,use_cache=True,return_dict=True)
                    cache=out.past_key_values;logits=out.logits[0,-1].float()
            torch.cuda.synchronize()
            case[label]={"ids":generated,"text":tokenizer.decode(generated,skip_special_tokens=False),"margins":margins,
                         "prefill_seconds":prefill,"generation_seconds":time.monotonic()-start}
            preserve(report)
            del out,cache,logits,ids
        del model;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    for case in prepared:
        name=case["name"];output=ROOT/"results"/f"qwen-{name}-c.json";logits=ROOT/"results"/f"qwen-{name}-c-first.f32"
        print("[c-generate]",name,flush=True)
        run([ROOT/"build/tcm-cuda","qwen",model_path,"--tokens-file",case["tokens_path"],"--new-tokens",case["budget"],
             "--context",1024,"--first-logits",logits,"--checkpoint-test","--output",output])
        candidate=json.loads(output.read_text());candidate_ids=[x["id"] for x in candidate["tokens"]]
        if candidate["model_identity"]!=source_identity or candidate["model_digest"]!=metadata["content_sha256"]:raise ValueError("C/native model or bundle identity differs")
        candidate["text"]=tokenizer.decode(candidate_ids,skip_special_tokens=False);case["c_equations"]=candidate
        reference=np.load(ROOT/"results"/f"qwen-{name}-native-first.npy")
        actual=np.fromfile(logits,dtype="<f4")
        if actual.shape!=reference.shape:raise ValueError("C/native readout geometry differs")
        delta=actual.astype(np.float64)-reference
        native_ids=case["native_fp32"]["ids"]
        first_divergence=next((i for i,(a,b) in enumerate(zip(candidate_ids,native_ids)) if a!=b),None)
        if first_divergence is None and len(candidate_ids)!=len(native_ids):first_divergence=min(len(candidate_ids),len(native_ids))
        case["comparison"]={"first_logits_max_abs_error":float(np.max(np.abs(delta))),"first_logits_rms_error":float(np.sqrt(np.mean(delta*delta))),
            "first_argmax_equal":int(np.argmax(reference))==int(np.argmax(actual)),"first_token_divergence":first_divergence,
            "same_generated_ids":candidate_ids==native_ids,"native_fp32_length":len(native_ids),"c_length":len(candidate_ids),
            "own_state_kv_invariant":candidate["kv_rows"]==len(case["ids"])+len(candidate_ids)-1,
            "native_fp32_bf16_equal":case["native_fp32"]["ids"]==case["native_bf16"]["ids"]}
        if not case["comparison"]["own_state_kv_invariant"] or not candidate["checkpoint_exact"]:
            raise ValueError("C token/cache/checkpoint mechanics failed")
        del case["tokens_path"]
        preserve(report)
    report["summary"]={"cases":len(prepared),"same_fp32_trajectories":sum(c["comparison"]["same_generated_ids"] for c in prepared),
        "same_first_argmax":sum(c["comparison"]["first_argmax_equal"] for c in prepared),
        "all_checkpoint_exact":all(c["c_equations"]["checkpoint_exact"] for c in prepared),
        "max_first_logits_error":max(c["comparison"]["first_logits_max_abs_error"] for c in prepared)}
    return report


def server_controller(request):
    """Candidate execution: future appends depend only on its own emitted IDs."""
    process=subprocess.Popen([str(ROOT/"build/tcm-cuda"),"serve",request["model_path"],"--context","1024"],
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    events=[]
    def receive(command=None):
        if command is not None:process.stdin.write(command+"\n");process.stdin.flush()
        line=process.stdout.readline()
        if not line:raise RuntimeError(f"C server exited: {process.poll()}")
        result=json.loads(line);events.append({"command":command,"result":result});return result
    def command(text):
        result=receive(text)
        if "error" in result:raise RuntimeError(result["error"])
        return result
    def append(ids):return command("A "+" ".join(map(str,ids)))
    try:
        ready=receive()
        if ready["model_identity"]!=request["model_identity"] or ready["model_digest"]!=request["model_digest"]:
            raise ValueError("persistent server model identity differs")
        prompt=request["math_ids"];append(prompt);command("C")
        split=command("G 2")["ids"]+command("G 3")["ids"]
        command("R");whole=command("G 5")["ids"]
        if split!=whole:raise ValueError("split generation or checkpoint policy failed")
        # Consume the server's own last delivered token before a new user turn.
        transcript=prompt+whole
        def user_turn(segment,budget):
            nonlocal transcript
            last=transcript[-1]
            tail=[last]+([] if last in request["eos"] else [151645])+segment
            append(tail)
            full_input=transcript[:-1]+tail
            result=command("G "+str(budget))["ids"]
            transcript=full_input+result
            return full_input,result
        _,remember=user_turn(request["remember_segment"],16)
        recall_prompt,recall=user_turn(request["recall_segment"],16)
        command("Z");append(recall_prompt);fresh=command("G 16")["ids"]
        if fresh!=recall:raise ValueError("appended context and full-prefix replay differ")
        # Oversized append is refused before mutation; ordinary generation still works.
        before_refusal=command("S");refused=receive("A "+" ".join(["1"]*1025))
        if "error" not in refused:raise ValueError("oversized append was accepted")
        if command("S")!=before_refusal:raise ValueError("oversized append mutated server state")
        command("Z");stale=receive("R")
        if "error" not in stale:raise ValueError("reset retained a stale checkpoint")
        append(request["unassisted_ids"]);unassisted=command("G 32")["ids"]
        command("Z");append(request["assisted_ids"]);assisted=command("G 16")["ids"]
        return {"ready":ready,"events":events,"split_equals_whole":True,"append_equals_fresh_prefix":True,
                "math_ids":whole,"remember_ids":remember,"recall_ids":recall,"recall_prompt":recall_prompt,
                "unassisted_ids":unassisted,"assisted_ids":assisted,"oversized_append_refused":True,"reset_invalidates_checkpoint":True}
    finally:
        if process.poll() is None:process.stdin.write("Q\n");process.stdin.flush()
        process.wait(timeout=10)


def continuation_experiment():
    import gc
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    sys.path.insert(0,str(ROOT/"public"))
    from saturn_bridge import invoke
    source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    model_path=Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
    metadata=json.loads(model_path.with_suffix(".json").read_text())
    if file_sha256(source/"model.safetensors")!=metadata["identity"]:raise ValueError("native source identity differs")
    run(["make","-C",ROOT,"cuda"]);run(["make","-C",ROOT,"all"])
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    def prompt(text):
        return tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True,return_dict=False)
    def segment(text):
        return tokenizer.encode("\n<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n",add_special_tokens=False)
    request={"model_path":str(model_path),"model_identity":metadata["identity"],"model_digest":metadata["content_sha256"],
        "binary_sha256":file_sha256(ROOT/"build/tcm-cuda"),"controller_sha256":file_sha256(__file__),
        "math_ids":prompt("What is 37 times 24? Answer with the integer only."),
        "remember_segment":segment("Remember that Mira's code is 7319. Reply OK."),
        "recall_segment":segment("What is Mira's code? Answer with the code only."),
        "unassisted_ids":prompt("What is Mira's code? Answer with the code only."),
        "assisted_ids":prompt("Additional explicit memory: Mira's code is 7319. What is Mira's code? Answer with the code only."),
        "eos":metadata["native_eos_token_ids"]}
    candidate,receipt=invoke(["C persistent server with declared controller"],None,request,controller=server_controller)
    for name in ("math","remember","recall","unassisted","assisted"):
        candidate[name+"_text"]=tokenizer.decode(candidate[name+"_ids"],skip_special_tokens=False)
    report={"schema":"tcm-continuation-experiment-v1","environment":environment(),"model":metadata,
        "candidate":candidate,"saturn_server_receipt":receipt,"native":{},
        "mechanics_status":"complete","instrument_status":"complete","trend_status":"observations","terminal_status":"not-requested"}
    preserve(report)
    # A separate typed ISA invocation validates the public bridge over the CPU adapter.
    isa_output=ROOT/"results/typed-isa.json"
    isa_command=[ROOT/"build/tcm","isa",ROOT/"models/isa-r3.tcm","--program","4883c001"*2,
                 "--registers","3,5,f,1f,3f,7f,ff,1ff","--steps","2","--checkpoint-row","31","--output",isa_output]
    isa_request={"program":"4883c001"*2,"registers":[3,5,15,31,63,127,255,511],"flags":[0,0,0,0],
                 "model_bundle_sha256":file_sha256(ROOT/"models/isa-r3.tcm"),"binary_sha256":file_sha256(ROOT/"build/tcm")}
    report["typed_isa"],report["saturn_isa_receipt"]=invoke(isa_command,isa_output,isa_request)
    preserve(report)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.float32,attn_implementation="eager",device_map={"":"cuda"},local_files_only=True).eval()
    for name,ids,budget in (("math",request["math_ids"],16),("recall_same_history",candidate["recall_prompt"],16),
                            ("unassisted",request["unassisted_ids"],32),("assisted",request["assisted_ids"],16)):
        generated=[]
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([ids],device="cuda"),use_cache=True,return_dict=True)
            cache=out.past_key_values
            for i in range(budget):
                token=int(torch.argmax(out.logits[0,-1]).item());generated.append(token)
                if token in metadata["native_eos_token_ids"] or i+1==budget:break
                out=model(input_ids=torch.tensor([[token]],device="cuda"),past_key_values=cache,use_cache=True,return_dict=True);cache=out.past_key_values
        report["native"][name]={"ids":generated,"text":tokenizer.decode(generated,skip_special_tokens=False)};preserve(report)
        del out,cache
    del model;gc.collect();torch.cuda.empty_cache()
    report["summary"]={"split_equals_whole":candidate["split_equals_whole"],"append_equals_fresh_prefix":candidate["append_equals_fresh_prefix"],
        "math_native_equal":candidate["math_ids"]==report["native"]["math"]["ids"],
        "recall_native_same_history_equal":candidate["recall_ids"]==report["native"]["recall_same_history"]["ids"],
        "math_answer":candidate["math_text"],"recall_answer":candidate["recall_text"],"assisted_answer":candidate["assisted_text"],
        "typed_isa_checkpoint_exact":report["typed_isa"]["checkpoint_exact"]}
    return report


def client_smoke():
    run(["make","-C",ROOT,"cuda"])
    source="/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    model="/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm"
    transcript="\n".join(["/save","Remember that Mira's code is 7319. Reply OK.","/save",
        "What is Mira's code? Answer with the code only.","/rewind",
        "What is Mira's code? Answer with the code only.","filler "*1000,
        "What is Mira's code? Answer with the code only.","/quit",""])
    result=run([sys.executable,ROOT/"tools/chat.py","--model",model,"--tokenizer",source,"--context","512","--new-tokens","16"],input=transcript)
    required=("Send a message before saving a parent.","current state is preserved.")
    if not all(x in result.stdout for x in required):raise ValueError("client lifecycle control was not exercised")
    return {"schema":"tcm-client-smoke-v1","environment":environment(),"transcript":transcript,"stdout":result.stdout,"stderr":result.stderr,
        "summary":{"client_exited_normally":True,"save_before_prompt_refused":True,"oversized_turn_preserved_state":True,
                   "code_answer_occurrences":result.stdout.count("model> 7319")},
        "scope":"actual tokenizer and persistent C process; supplied code fact; three answer observations with rewind/capacity controls",
        "mechanics_status":"complete","instrument_status":"complete","trend_status":"observations","terminal_status":"not-requested"}


def stack_experiment():
    """Bounded engineering panel; no recovered semantic law is claimed."""
    import numpy as np
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ROOT/"public"))
    from saturn_bridge import invoke
    source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    model=Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
    metadata=json.loads(model.with_suffix(".json").read_text())
    baseline=Path("/mnt/big/saturn-cmachine/builds/70b5a1df0c9af09e/build/tcm-cuda")
    expected="70b5a1df0c9af09e603e39c83d33264f49510da522606fe92f12d08452bde009"
    if file_sha256(baseline)!=expected:raise ValueError("historical reference binary differs")
    if file_sha256(source/"model.safetensors")!=metadata["identity"]:raise ValueError("source checkpoint differs")
    for name,digest in metadata["tokenizer_files"].items():
        if file_sha256(source/name)!=digest:raise ValueError("tokenizer configuration differs")
    run(["make","-C",ROOT,"cuda","stack-mechanics","gpu-mechanics"])
    report={"schema":"tcm-stack-experiment-v1","environment":environment(),"model":metadata,
        "baseline_binary_sha256":expected,"candidate_binary_sha256":file_sha256(ROOT/"build/tcm-cuda"),
        "cases":[],"mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested",
        "scope":"historical-C/new-C numerical compatibility, nested lifecycle and selected observations; no new semantic law or task-accuracy qualification"}
    preserve(report)
    report["gpu_mechanics"]=json.loads(run([ROOT/"build/gpu-mechanics"]).stdout);preserve(report)
    request={"model_path":str(model),"model_digest":metadata["content_sha256"],"binary_sha256":file_sha256(ROOT/"build/stack-mechanics"),"inputs":"fixed synthetic token IDs from state_mechanics.c;12token capacity"}
    def controller(_):return json.loads(run([ROOT/"build/stack-mechanics",model]).stdout)
    report["stack_mechanics"],report["saturn_server_receipt"]=invoke([ROOT/"build/stack-mechanics",model],None,request,controller=controller);preserve(report)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    for name,text in (("math","What is 37 times 24? Answer with the integer only."),
                      ("recall","Mira's code is 7319. Niko's code is 2846. What is Mira's code? Answer with the code only."),
                      ("chat","Why does bread rise when yeast is added? Explain in a few sentences.")):
        ids=tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=True,add_generation_prompt=True,return_dict=False)
        if not isinstance(ids,list) or not all(isinstance(i,int) for i in ids):raise TypeError("tokenizer IDs are not flat")
        tokens=ROOT/"results"/f"stack-{name}-input.txt";tokens.write_text(" ".join(map(str,ids))+"\n")
        prefix=ROOT/"results"/f"stack-{name}-trace"
        row={"name":name,"prompt":text,"ids":ids,"budget":32,"runs":{}}
        for arm,binary in (("historical",baseline),("candidate",ROOT/"build/tcm-cuda")):
            output=ROOT/"results"/f"stack-{name}-{arm}.json";logits=output.with_suffix(".f32")
            command=[binary,"qwen",model,"--tokens-file",tokens,"--new-tokens","32","--context","256","--output",output,"--first-logits",logits,"--checkpoint-test"]
            if arm=="candidate":command.extend(["--trace-prefix",prefix,"--trace-layer","16"])
            run(command);value=json.loads(output.read_text())
            if value["model_digest"]!=metadata["content_sha256"]:raise ValueError("executed model differs")
            value["text"]=tokenizer.decode([x["id"] for x in value["tokens"]],skip_special_tokens=False);row["runs"][arm]=value
        left=np.fromfile(ROOT/"results"/f"stack-{name}-historical.f32",dtype="<f4")
        right=np.fromfile(ROOT/"results"/f"stack-{name}-candidate.f32",dtype="<f4")
        descriptors=[json.loads(s) for s in prefix.with_suffix(".jsonl").read_text().splitlines()]
        validate_trace_ports(descriptors)
        for item in descriptors:
            path=Path(str(prefix)+f"-{item['occurrence']:04d}-{item['port']}.f32")
            if path.stat().st_size!=4*item["count"] or item["position"]!=len(ids)-1 or item["layer"]!=16:raise ValueError("port geometry differs")
            item.update(file=path.name,sha256=file_sha256(path))
        row["ports"]=descriptors
        row["comparison"]={"full_first_logits_bytes_equal":left.tobytes()==right.tobytes(),
            "first_logits_max_abs_error":float(np.max(np.abs(left.astype(float)-right.astype(float)))),
            "own_generated_ids_equal":[x["id"] for x in row["runs"]["historical"]["tokens"]]==[x["id"] for x in row["runs"]["candidate"]["tokens"]],
            "checkpoint_exact":row["runs"]["candidate"]["checkpoint_exact"],"selected_ports":len(descriptors)}
        report["cases"].append(row);preserve(report)
    # Exercise the actual client: old snapshots are immutable references; each
    # model append changes only the current branch's transcript.
    transcript="\n".join(["/push","Remember that Mira's code is 7319. Reply OK.","/save","/push","/rewind",
        "What is Mira's code? Answer with the code only.","/push","Correction: Mira's code is 2846. Reply OK.",
        "/push","What is Mira's code? Answer with the code only.","/rollback","/commit","/rollback",
        "What is Mira's code? Answer with the code only.","/rollback","/quit",""])
    result=run([sys.executable,ROOT/"tools/chat.py","--model",model,"--tokenizer",source,"--context","512","--new-tokens","16"],input=transcript)
    (ROOT/"results/stack-client.stdout.txt").write_text(result.stdout)
    (ROOT/"results/stack-client.stderr.txt").write_text(result.stderr)
    report["client"]={"transcript":transcript,"stdout":result.stdout,"stderr":result.stderr,"model_code_7319_occurrences":result.stdout.count("model> 7319")};preserve(report)
    if result.stdout.count("'event': 'pushed'")!=3 or result.stdout.count("'event': 'rolled_back'")!=2 or result.stdout.count("'event': 'committed'")!=1:
        raise ValueError("nested client lifecycle did not execute as declared")
    if "before rewinding" not in result.stdout or "No nested frame." not in result.stdout:raise ValueError("client guards did not execute")
    report["summary"]={"cases":len(report["cases"]),"all_old_new_first_logits_exact":all(x["comparison"]["full_first_logits_bytes_equal"] for x in report["cases"]),
        "all_old_new_own_tokens_equal":all(x["comparison"]["own_generated_ids_equal"] for x in report["cases"]),
        "ports_captured":sum(x["comparison"]["selected_ports"] for x in report["cases"]),"nested_client_exited_normally":True,
        "host_marker_bytes":report["stack_mechanics"]["host_marker_bytes"]}
    report.update(mechanics_status="complete",instrument_status="complete");preserve(report)
    if not report["summary"]["all_old_new_first_logits_exact"] or not report["summary"]["all_old_new_own_tokens_equal"]:
        raise ValueError("token-step refactor changed the historical arithmetic trajectory")
    return report


def plane_experiment():
    from transformers import AutoTokenizer
    from plane_analysis import analyze
    sys.path.insert(0,str(ROOT/"public"))
    from saturn_bridge import invoke
    source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    model=Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
    metadata=json.loads(model.with_suffix(".json").read_text())
    historical=Path("/mnt/big/saturn-cmachine/builds/33c1fe26a55654f0/build/tcm-cuda")
    expected="33c1fe26a55654f0af4531573e240a6c98be02f70d63bbe64aecaad32b74d28e"
    if file_sha256(historical)!=expected or file_sha256(source/"model.safetensors")!=metadata["identity"]:raise ValueError("historical binary/checkpoint differs")
    for name,sha in metadata["tokenizer_files"].items():
        if file_sha256(source/name)!=sha:raise ValueError("tokenizer identity differs")
    run(["make","-C",ROOT,"cuda","plane-experiment","stack-mechanics"])
    report={"schema":"tcm-plane-experiment-v1","environment":environment(),"model":metadata,
        "historical_binary_sha256":expected,"candidate_binary_sha256":file_sha256(ROOT/"build/tcm-cuda"),
        "driver_binary_sha256":file_sha256(ROOT/"build/plane-experiment"),
        "mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested",
        "scope":"explicit compiled nonlinear receiver plane; controlled operands and C-native arithmetic; no semantic program, general chat, compression or whole-model speedup claim"}
    preserve(report)
    report["stack_mechanics"]=json.loads(run([ROOT/"build/stack-mechanics",model]).stdout);preserve(report)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    text="Mira's badge is red. Niko's badge is blue. What color is Mira's badge? Answer with the color only."
    messages=[{"role":"user","content":text}]
    prompt_text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    if ids!=tokenizer.encode(prompt_text,add_special_tokens=False):raise ValueError("native template identity differs")
    prompt=ROOT/"results/plane-input.txt";prompt.write_text(" ".join(map(str,ids))+"\n")
    questions=[]
    for name,person in (("target","Mira"),("collateral","Niko")):
        question=f"What color is {person}'s badge? Answer with the color only."
        extended=tokenizer.apply_chat_template(messages+[{"role":"assistant","content":""},{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
        if not extended.startswith(prompt_text):raise ValueError("native followup template has no exact prefix")
        suffix=tokenizer.encode(extended[len(prompt_text):],add_special_tokens=False)
        if tokenizer.encode(extended,add_special_tokens=False)!=ids+suffix:raise ValueError("native followup token concatenation differs")
        if suffix[0]!=151645:raise ValueError("expected native assistant end delimiter")
        path=ROOT/"results"/f"plane-{name}-suffix.txt";path.write_text(" ".join(map(str,suffix))+"\n")
        questions.append({"name":name,"text":question,"suffix_ids":suffix,"path":str(path)})
    color_ids=[tokenizer.encode(color,add_special_tokens=False) for color in ("red","blue")]
    if any(len(x)!=1 for x in color_ids):raise ValueError("expected single-token colors")
    report["input"]={"text":text,"ids":ids,"questions":questions,"red_id":color_ids[0][0],"blue_id":color_ids[1][0]}
    report["ordinary_paths"]={}
    for name,binary in (("historical",historical),("candidate",ROOT/"build/tcm-cuda")):
        output=ROOT/"results"/f"plane-ordinary-{name}.json"
        run([binary,"qwen",model,"--tokens-file",prompt,"--new-tokens","8","--context","512","--output",output,"--first-logits",output.with_suffix(".f32")])
        report["ordinary_paths"][name]=json.loads(output.read_text())
    preserve(report)
    command=[ROOT/"build/plane-experiment",model,prompt,questions[0]["path"],questions[1]["path"],ROOT/"results",*map(str,(color_ids[0][0],color_ids[1][0]))]
    request={"model_digest":metadata["content_sha256"],"binary_sha256":report["driver_binary_sha256"],"prompt_ids":ids,"doses":[[0,0],[.25,0],[-.25,0],[-1,0],[0,.25],[.5,.5]],"plan_sha256":file_sha256(ROOT/"PLANE_EXPERIMENT.md")}
    def controller(_):
        result=run(command)
        (ROOT/"results/plane-driver.stdout.txt").write_text(result.stdout)
        (ROOT/"results/plane-driver.stderr.txt").write_text(result.stderr)
        return json.loads((ROOT/"results/plane-driver.json").read_text())
    driver,report["saturn_server_receipt"]=invoke(command,None,request,controller=controller)
    if driver["model_digest"]!=metadata["content_sha256"] or driver["position"]!=len(ids)-1:raise ValueError("executed plane specimen differs")
    report["driver_raw_sha256"]=file_sha256(ROOT/"results/plane-driver.json")
    # Keep the receipt's raw returned object immutable. Text is a tokenizer-only
    # annotation on a separate report projection.
    driver=json.loads(json.dumps(driver))
    for row in driver["rows"]:
        row["own"]["text"]=tokenizer.decode(row["own"]["ids"],skip_special_tokens=False)
        for question in row["questions"]:question["output"]["text"]=tokenizer.decode(question["output"]["ids"],skip_special_tokens=False)
    report["driver"]=driver
    report.update(analyze(ROOT/"results",driver));preserve(report)
    ordinary=report["ordinary_paths"]
    ordinary_ids={k:[x["id"] for x in v["tokens"]] for k,v in ordinary.items()}
    historical_bytes=(ROOT/"results/plane-ordinary-historical.f32").read_bytes()
    candidate_bytes=(ROOT/"results/plane-ordinary-candidate.f32").read_bytes()
    native_bytes=(ROOT/"results/native-first.f32").read_bytes()
    report["summary"].update(ordinary_first_logits_exact=historical_bytes==candidate_bytes==native_bytes,
        ordinary_own_ids_equal=ordinary_ids["historical"]==ordinary_ids["candidate"]==driver["rows"][0]["own"]["ids"])
    # Numerical approximation quality never stops this exploratory panel. This
    # engineering check protects the original ordinary executor's behavior.
    if not report["summary"]["ordinary_first_logits_exact"] or not report["summary"]["ordinary_own_ids_equal"]:raise ValueError("ordinary executor changed")
    report["raw_files"]={p.name:{"sha256":file_sha256(p),"bytes":p.stat().st_size} for p in sorted((ROOT/"results").glob("*")) if p.is_file() and p.suffix in (".f32",".txt")}
    report.update(mechanics_status="complete",instrument_status="complete");preserve(report)
    return report


def reuse_experiment(enrich=False,components=False,formation=False,writer=False):
    enrich=enrich or components or formation or writer
    from transformers import AutoTokenizer
    if writer:from writer_analysis import analyze
    elif formation:from formation_analysis import analyze
    else:from reuse_analysis import analyze
    sys.path.insert(0,str(ROOT/"public"))
    from saturn_bridge import invoke
    source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    model=Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
    metadata=json.loads(model.with_suffix(".json").read_text())
    historical=Path("/mnt/big/saturn-cmachine/builds/43065f592580b3d2/build/tcm-cuda")
    expected="43065f592580b3d2679cdb0206b1c26c0a36318a8947d16a434c9938bcd23ca6"
    if components:
        historical=Path("/mnt/big/saturn-cmachine/builds/8a1f58d28aa01ab4/build/tcm-cuda")
        expected="8a1f58d28aa01ab429df27b24936eb4e22440c28c981b18ede6995dac06abf35"
    if formation:
        historical=Path("/mnt/big/saturn-cmachine/builds/e97520668fe8d614/build/tcm-cuda")
        expected="e97520668fe8d6149f5b1f705bcaa0da476fb8e96b89152314f98491b70ad8d3"
    if writer:
        historical=Path("/mnt/big/saturn-cmachine/builds/d4ba05c0161a737d/build/tcm-cuda")
        expected="d4ba05c0161a737d9d8c227a679ecb72ea6673773045bf638ca65df888995613"
    if file_sha256(historical)!=expected or file_sha256(source/"model.safetensors")!=metadata["identity"]:raise ValueError("historical binary/checkpoint differs")
    for name,sha in metadata["tokenizer_files"].items():
        if file_sha256(source/name)!=sha:raise ValueError("tokenizer identity differs")
    run(["make","-C",ROOT,"cuda","reuse-experiment","stack-mechanics"])
    report={"schema":"tcm-writer-experiment-v1" if writer else "tcm-formation-experiment-v1" if formation else "tcm-components-experiment-v1" if components else "tcm-enrichment-experiment-v1" if enrich else "tcm-reuse-experiment-v1","environment":environment(),"model":metadata,
        "historical_binary_sha256":expected,"candidate_binary_sha256":file_sha256(ROOT/"build/tcm-cuda"),"driver_binary_sha256":file_sha256(ROOT/"build/reuse-experiment"),
        "mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested",
        "scope":"one explicit Mira calibration operand frozen before two fresh Niko paraphrases; prospective B2/B3 transfer, actual residual/native nonlinear consumer retained; no target refresh/fallback or general semantic claim" if enrich else "frozen source receiver subspace reused at two naturally formed later questions; actual residual and native nonlinear consumer retained; no target refresh/fallback or general semantic claim"}
    preserve(report);report["stack_mechanics"]=json.loads(run([ROOT/"build/stack-mechanics",model]).stdout);preserve(report)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    text="Mira's badge is red. Niko's badge is blue. What color is Mira's badge? Answer with the color only."
    messages=[{"role":"user","content":text}]
    source_text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    if ids!=tokenizer.encode(source_text,add_special_tokens=False):raise ValueError("native source template differs")
    prompt=ROOT/"results/reuse-input.txt";prompt.write_text(" ".join(map(str,ids))+"\n")
    def suffix(question,name):
        extended=tokenizer.apply_chat_template(messages+[{"role":"assistant","content":""},{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
        if not extended.startswith(source_text):raise ValueError("native suffix has no exact string prefix")
        result=tokenizer.encode(extended[len(source_text):],add_special_tokens=False)
        if tokenizer.encode(extended,add_special_tokens=False)!=ids+result or result[0]!=151645:raise ValueError("native suffix token concatenation differs")
        path=ROOT/"results"/f"reuse-{name}-suffix.txt";path.write_text(" ".join(map(str,result))+"\n")
        return {"text":question,"ids":result,"path":str(path)}
    if writer:
        targets=[suffix("Which badge color belongs to Niko? Answer only that color.","target-Niko-E")]
        report["scope"]="four sequential layer15 native/projected attention/projected MLP/both policies plus ordinary target control; supplied frozen layer16 B3; actual recomputed receivers, dense own-state suffix/KV/answers; no semantic-selectivity or chat-accuracy claim"
    elif formation:
        targets=[suffix("Name the color of the badge belonging to Niko. Return just that color.","target-Niko-D")]
        report["scope"]="fixed predecessor B3; one fresh question with dense/project/inherited/new-write/full-carried arms, actual layer15 lineage and native continuation; full-carried still omits layer16 read; pure local joint diagnostic; no semantic-origin or accuracy claim"
    elif components:
        targets=[suffix("Niko is wearing which badge color? Respond with the color alone.","target-Niko-C")]
        report["scope"]="one frozen B3 program; prospective carry/read-complement corrections at one new question, native continuation in5arms, plus one pure nonlinear/down diagnostic; no target refresh/fallback or semantic importance claim"
    elif enrich:targets=[suffix(question,name) for question,name in (("What is the color of Niko's badge? Reply with one color.","target-Niko-A"),("Which color badge does Niko have? Give only its color.","target-Niko-B"))]
    else:targets=[suffix(f"Tell me {person}'s badge color, just the color.",f"target-{person}") for person in ("Niko","Mira")]
    calibration=suffix("Tell me Mira's badge color, just the color.","calibration-Mira") if enrich else None
    questions=[suffix(f"What color is {person}'s badge? Answer with the color only.",f"future-{person}") for person in ("Mira","Niko")]
    # Verify the same delimiter suffixes in a later-turn template too. Actual
    # generated IDs remain authoritative; no decoded assistant text is reencoded.
    for target in targets:
        later=messages+[{"role":"assistant","content":""},{"role":"user","content":target["text"]}]
        prefix=tokenizer.apply_chat_template(later,tokenize=False,add_generation_prompt=True)
        for question in questions:
            extended=tokenizer.apply_chat_template(later+[{"role":"assistant","content":""},{"role":"user","content":question["text"]}],tokenize=False,add_generation_prompt=True)
            if not extended.startswith(prefix) or tokenizer.encode(extended[len(prefix):],add_special_tokens=False)!=question["ids"]:raise ValueError("later-turn native delimiter suffix differs")
    colors=[tokenizer.encode(c,add_special_tokens=False) for c in ("red","blue")]
    if any(len(c)!=1 for c in colors):raise ValueError("single-token lowercase colors expected")
    report["input"]={"text":text,"ids":ids,"targets":targets,"questions":questions,"red_id":colors[0][0],"blue_id":colors[1][0]};report["ordinary_paths"]={}
    if enrich:report["input"]["calibration"]=calibration
    for name,binary in (("historical",historical),("candidate",ROOT/"build/tcm-cuda")):
        output=ROOT/"results"/f"reuse-ordinary-{name}.json";run([binary,"qwen",model,"--tokens-file",prompt,"--new-tokens","8","--context","512","--output",output,"--first-logits",output.with_suffix(".f32")]);report["ordinary_paths"][name]=json.loads(output.read_text())
    preserve(report)
    command=[ROOT/"build/reuse-experiment",model,prompt,*[t["path"] for t in targets],*[q["path"] for q in questions],ROOT/"results",str(colors[0][0]),str(colors[1][0])]
    if enrich:command.extend(["writer" if writer else "formation" if formation else "components" if components else "enrich",calibration["path"]])
    request={"model_digest":metadata["content_sha256"],"binary_sha256":report["driver_binary_sha256"],"prompt_ids":ids,"target_ids":[t["ids"] for t in targets],"future_ids":[q["ids"] for q in questions],"plan_sha256":file_sha256(ROOT/("WRITER_INTERVENTION_EXPERIMENT.md" if writer else "CARRY_FORMATION_EXPERIMENT.md" if formation else "CARRY_READ_EXPERIMENT.md" if components else "ENRICH_EXPERIMENT.md" if enrich else "REUSE_EXPERIMENT.md"))}
    if enrich:request["calibration_ids"]=calibration["ids"]
    def controller(_):
        result=run(command);(ROOT/"results/reuse-driver.stdout.txt").write_text(result.stdout);(ROOT/"results/reuse-driver.stderr.txt").write_text(result.stderr)
        return json.loads((ROOT/"results/reuse-driver.json").read_text())
    raw_driver,report["saturn_server_receipt"]=invoke(command,None,request,controller=controller);report["driver_raw_sha256"]=file_sha256(ROOT/"results/reuse-driver.json");driver=json.loads(json.dumps(raw_driver))
    if driver["model_digest"]!=metadata["content_sha256"] or driver["source_position"]!=len(ids)-1:raise ValueError("executed reuse specimen differs")
    if (components or formation or writer) and (driver["program_sha256"]!="1a12d8a22bfd00575039b9409d28bbdd44bdaa8f8b207fe1f5f4f5bfb4424d3e" or driver["extension_program_sha256"]!="d33759a30b44473612f219ba00df332ea0330db4b7100c7d7a22e0519c43ddad"):raise ValueError("declared predecessor B3 program changed")
    if (formation or writer) and any(row["input_token_id"]!=targets[0]["ids"][-1] for row in driver["target_rows"]):raise ValueError("writer/formation token differs from frozen target input")
    for row in driver["rows"]+driver.get("calibration_rows",[])+driver["target_rows"]:
        row["own"]["text"]=tokenizer.decode(row["own"]["ids"],skip_special_tokens=False)
        for question in row["questions"]:question["output"]["text"]=tokenizer.decode(question["output"]["ids"],skip_special_tokens=False)
    driver["rebuilt_source_own"]["text"]=tokenizer.decode(driver["rebuilt_source_own"]["ids"],skip_special_tokens=False)
    report["driver"]=driver;report.update(analyze(ROOT/"results",driver));preserve(report)
    ordinary=report["ordinary_paths"];ordinary_ids={k:[x["id"] for x in v["tokens"]] for k,v in ordinary.items()}
    first=[(ROOT/"results"/name).read_bytes() for name in ("reuse-ordinary-historical.f32","reuse-ordinary-candidate.f32","source-native-first.f32")]
    report["summary"].update(ordinary_first_logits_exact=first[0]==first[1]==first[2],ordinary_own_ids_equal=ordinary_ids["historical"]==ordinary_ids["candidate"]==driver["rows"][0]["own"]["ids"])
    if not report["summary"]["ordinary_first_logits_exact"] or not report["summary"]["ordinary_own_ids_equal"] or not report["summary"]["rebuilt_source_ids_equal"]:raise ValueError("native executor/source replay changed")
    report["raw_files"]={p.name:{"sha256":file_sha256(p),"bytes":p.stat().st_size} for p in sorted((ROOT/"results").glob("*")) if p.is_file() and p.suffix in (".f32",".f64",".txt")}
    report.update(mechanics_status="complete",instrument_status="complete");preserve(report);return report


def lineage_experiment():
    """Engineering check only: parent ownership, restoration and native parity."""
    import numpy as np
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ROOT/"public"))
    from saturn_bridge import invoke
    source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    model=Path("/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm")
    metadata=json.loads(model.with_suffix(".json").read_text())
    historical=Path("/mnt/big/saturn-cmachine/builds/d4ba05c0161a737d/build/tcm-cuda")
    if file_sha256(historical)!="d4ba05c0161a737d9d8c227a679ecb72ea6673773045bf638ca65df888995613":raise ValueError("historical binary differs")
    for name,digest in metadata["tokenizer_files"].items():
        if file_sha256(source/name)!=digest:raise ValueError("lineage tokenizer differs")
    run(["make","-C",ROOT,"cuda","stack-mechanics"])
    report={"schema":"tcm-lineage-mechanics-v1","model":metadata,"environment":environment(),"mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"engineering-only","terminal_status":"not-requested", "scope":"retained-parent lineage and ordinary C parity; no new scientific intervention sample","candidate_binary_sha256":file_sha256(ROOT/"build/tcm-cuda"),"ordinary_paths":{}}
    preserve(report)
    command=[ROOT/"build/stack-mechanics",model]
    report["stack_mechanics"],report["saturn_server_receipt"]=invoke(command,None,{"model_digest":metadata["content_sha256"],"binary_sha256":file_sha256(ROOT/"build/stack-mechanics"),"purpose":"pretrained retained-parent lineage mechanics"},controller=lambda _:json.loads(run(command).stdout))
    if not report["stack_mechanics"].get("writer_parent_lineage_checks"):raise ValueError("lineage instrument did not run")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    ids=tokenizer.apply_chat_template([{"role":"user","content":"Mira's badge is red. Niko's badge is blue. What color is Mira's badge? Answer with the color only."}],tokenize=True,add_generation_prompt=True,return_dict=False)
    prompt=ROOT/"results/lineage-input.txt";prompt.write_text(" ".join(map(str,ids))+"\n");report["input_ids"]=ids
    for name,binary in (("historical",historical),("candidate",ROOT/"build/tcm-cuda")):
        output=ROOT/"results"/f"lineage-{name}.json"
        run([binary,"qwen",model,"--tokens-file",prompt,"--new-tokens","8","--context","512","--checkpoint-test","--output",output,"--first-logits",output.with_suffix(".f32")])
        report["ordinary_paths"][name]=json.loads(output.read_text())
        if report["ordinary_paths"][name]["model_digest"]!=metadata["content_sha256"]:raise ValueError("lineage executed model differs")
    first=[(ROOT/"results"/f"lineage-{name}.f32").read_bytes() for name in ("historical","candidate")]
    paths=report["ordinary_paths"]
    report["summary"]={"first_logits_bytes_equal":first[0]==first[1],"own_ids_equal":[t["id"] for t in paths["historical"]["tokens"]]==[t["id"] for t in paths["candidate"]["tokens"]],"checkpoint_exact":paths["candidate"]["checkpoint_exact"],"host_marker_bytes":report["stack_mechanics"]["host_marker_bytes"],"new_scientific_samples":0}
    if len(first[0])!=151936*4 or not np.isfinite(np.frombuffer(first[0],dtype="<f4")).all() or not np.isfinite(np.frombuffer(first[1],dtype="<f4")).all():raise ValueError("lineage readout shape/finite contract differs")
    if not all(report["summary"][k] for k in ("first_logits_bytes_equal","own_ids_equal","checkpoint_exact")):raise ValueError("ordinary lineage parity differs")
    report["raw_files"]={p.name:{"sha256":file_sha256(p),"bytes":p.stat().st_size} for p in sorted((ROOT/"results").glob("*")) if p.suffix in (".f32",".txt")}
    report.update(mechanics_status="complete",instrument_status="complete");preserve(report);return report


def emit_stack_files(limit_mib=16):
    """Small raw evidence/source bundle in the scheduler-sealed log."""
    import io,tarfile
    files=[]
    for directory,archive_root in (("results","raw"),("src","payload/src"),("tools","payload/tools"),("tests","payload/tests"),("public","payload/public")):
        for path in sorted((ROOT/directory).rglob("*")):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                files.append((path,archive_root+"/"+str(path.relative_to(ROOT/directory))))
    files.extend((ROOT/name,"payload/"+name) for name in ("Makefile","source-manifest.json","PLANE_EXPERIMENT.md","REUSE_EXPERIMENT.md","ENRICH_EXPERIMENT.md","CARRY_READ_EXPERIMENT.md","CARRY_FORMATION_EXPERIMENT.md","WRITER_INTERVENTION_EXPERIMENT.md","PARENT_LINEAGE.md") if (ROOT/name).exists())
    if sum(p.stat().st_size for p,_ in files)>limit_mib*1024**2:raise ValueError("raw evidence bundle exceeds declared limit")
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode="w:gz") as archive:
        for path,name in files:archive.add(path,arcname=name,recursive=False)
    print("TCM_FILES_TGZ_B64="+base64.b64encode(buffer.getvalue()).decode(),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("probe","isa","qwen","buildcuda","continuation","client","stack","plane","reuse","enrich","components","formation","writer","lineage"));args=parser.parse_args()
    (ROOT/"results").mkdir(exist_ok=True)
    start=time.monotonic()
    try:
        report={"probe":environment,"isa":isa_experiment,"qwen":qwen_experiment,"buildcuda":build_cuda,"continuation":continuation_experiment,"client":client_smoke,"stack":stack_experiment,"plane":plane_experiment,"reuse":reuse_experiment,"enrich":lambda:reuse_experiment(True),"components":lambda:reuse_experiment(components=True),"formation":lambda:reuse_experiment(formation=True),"writer":lambda:reuse_experiment(writer=True),"lineage":lineage_experiment}[args.mode]()
        report["worker_seconds"]=time.monotonic()-start
        code=0
    except Exception:
        import traceback
        partial=ROOT/"results/partial-report.json"
        report=json.loads(partial.read_text()) if partial.exists() else {"environment":environment()}
        report.update(error=traceback.format_exc(),mechanics_status="incomplete",instrument_status="incomplete")
        code=1
    raw=json.dumps(report,sort_keys=True,allow_nan=False).encode()
    (ROOT/"results/report.json").write_bytes(raw+b"\n")
    print(MARKER+base64.b64encode(gzip.compress(raw,mtime=0)).decode(),flush=True)
    if args.mode in ("stack","plane","reuse","enrich","components","formation","writer","lineage"):emit_stack_files(32 if args.mode in ("plane","reuse","enrich","components","formation","writer","lineage") else 16)
    return code


if __name__=="__main__":raise SystemExit(main())
