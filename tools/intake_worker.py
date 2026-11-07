"""Finite mrun worker: one source load, graph compilation, C consumer evidence.

No nested jobs, training, automatic retries or Python numerical serving fallback.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
import tarfile
import time
import traceback

ROOT=Path.cwd()
RAW=Path("raw")
sys.path.insert(0,str(ROOT))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()

def command(args):
    result=subprocess.run([str(x) for x in args],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args}\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
    return result

def snapshot(name, value):
    """Publish each completed evidence boundary before optional work starts."""
    with (RAW/(name+".json")).open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())

def optional_arm(name, baseline, experiment):
    snapshot(name+"-baseline-summary",baseline)
    try:
        result=experiment()
    except Exception:
        result={"status":"error","error":traceback.format_exc(),
                "baseline_status":"completed; preserved independently"}
    snapshot(name+"-intervention-summary",result)
    return result

def c_run(bundle,rows,name,*,generate=0,eos=(),zero=None,trace=False,state=False,binary="build/tcm"):
    import numpy as np
    input_path=RAW/(name+".input.f32");np.asarray(rows,dtype=np.float32).tofile(input_path)
    args=[binary,"program",bundle,"--input",input_path,"--output",RAW/(name+".logits.f32"),"--replay"]
    if generate:args += ["--generate",str(generate)]
    for token in eos:args += ["--eos",str(token)]
    if zero is not None:args += ["--zero",str(zero)]
    if trace:args += ["--trace",RAW/(name+".trace.f32")]
    if state:args += ["--state-file",RAW/(name+".state.f32")]
    run=command(args);record=json.loads(run.stdout)
    (RAW/(name+".json")).write_text(json.dumps(record,indent=2)+"\n")
    return record,np.fromfile(RAW/(name+".logits.f32"),dtype=np.float32)

def pan():
    import numpy as np
    import torch
    from pan.models.pan import PhaseAccumulatorNetwork
    from program import lower_fx
    torch.set_num_threads(1)
    checkpoint=ROOT/"inputs/pan.pt"
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if payload["arch"]!="PAN":raise ValueError("checkpoint is not PAN")
    cfg=payload["config"];p=int(cfg["p"]);n=3 if cfg["task_kind"]=="mod_two_step" else 2
    model=PhaseAccumulatorNetwork(p,int(cfg["k_freqs"]),n_inputs=n,freq_init=cfg["freq_init"]).eval()
    model.load_state_dict(payload["state_dict"],strict=True)
    identity=sha(checkpoint)
    source={"checkpoint_sha256":identity,"config":cfg,"source_files":{str(x):sha(x) for x in sorted(Path("pan").rglob("*.py"))}}
    mapping=lower_fx(model,n,identity,RAW/"pan.tcm",source_record=source)
    if n!=2:raise ValueError("finite all-pairs panel requires two inputs")
    rows=np.array([(a,b) for a in range(p) for b in range(p)],dtype=np.int64)
    kind=cfg["task_kind"]
    if kind=="mod_add":target=(rows[:,0]+rows[:,1])%p
    elif kind=="mod_sub":target=(rows[:,0]-rows[:,1])%p
    elif kind=="mod_mul":target=(rows[:,0]*rows[:,1])%p
    else:raise ValueError(f"panel task not implemented: {kind}")
    print(f"PHASE pan source loaded; task={kind}; p={p}; rows={len(rows)}; C ops={len(mapping['instructions'])}",flush=True)
    with torch.no_grad():native=model(torch.from_numpy(rows)).numpy()
    baseline,flat=c_run(RAW/"pan.tcm",rows,"pan-baseline",trace=True)
    actual=flat.reshape(native.shape)
    difference=actual-native;pred=actual.argmax(1);native_pred=native.argmax(1)
    np.savez_compressed(RAW/"pan-evidence.npz",inputs=rows,target=target,native_logits=native,c_logits=actual,
                        baseline_predictions=pred,
                        per_row_max_abs=np.abs(difference).max(1))
    # The source map retains every vector boundary; this identifies where the
    # two engines first differ instead of reducing evidence to final accuracy.
    from torch.fx import Interpreter
    import torch.fx
    graph=torch.fx.symbolic_trace(model)
    observed={}
    class Capture(Interpreter):
        def run_node(self,node):
            result=super().run_node(node)
            if isinstance(result,torch.Tensor):observed[node.name]=result.detach().numpy().reshape(-1)
            return result
    with torch.no_grad():Capture(graph).run(torch.from_numpy(rows[:1]))
    trace=np.fromfile(RAW/"pan-baseline.trace.f32",dtype=np.float32)
    width=sum(v["length"] for v in mapping["values"]);trace=trace.reshape(len(rows),width)[0]
    boundaries=[]
    for value in mapping["values"][1:]:
        if value["name"] in observed:
            other=observed[value["name"]];got=trace[value["offset"]:value["offset"]+value["length"]]
            if len(other)==len(got):boundaries.append({"value":value["name"],"max_abs":float(np.abs(other-got).max())})
    # Keep compact immutable raw evidence, not duplicate uncompressed row banks.
    for name in ("pan-baseline.logits.f32","pan-baseline.trace.f32"):(RAW/name).unlink()
    result={"specimen":"PAN", "source":source,"parameter_count":sum(v.numel() for v in model.parameters()),
            "task_kind":kind,"modulus":p,"rows":len(rows),"operations":len(mapping["instructions"]),
            "state_slots":len(mapping["state_slots"]),"native_accuracy":float((native_pred==target).mean()),
            "c_accuracy":float((pred==target).mean()),"top1_agreement":float((pred==native_pred).mean()),
            "max_abs_logit_error":float(np.abs(difference).max()),"mean_abs_logit_error":float(np.abs(difference).mean()),
            "boundaries_first_row":boundaries,"baseline":baseline,
            "authority":{"mechanics_status":"executed","instrument_status":"source-and-C-compared",
                         "trend_status":"see-signed-task-and-intervention-observations",
                         "terminal_status":"bounded-specimen-results-only; no-general-math-or-speed-claim"}}
    def intervene():
        gate=next(i for i,o in enumerate(mapping["instructions"]) if o["kind"]=="linear" and o["w"]=="decoder.weight")-1
        arm,ablated_flat=c_run(RAW/"pan.tcm",rows,"pan-zero-gate",zero=gate)
        ablated=ablated_flat.reshape(native.shape);arm_pred=ablated.argmax(1)
        np.savez_compressed(RAW/"pan-intervention.npz",zero_gate_predictions=arm_pred,
                            target_logit_delta=ablated[np.arange(len(rows)),target]-actual[np.arange(len(rows)),target])
        (RAW/"pan-zero-gate.logits.f32").unlink()
        return {"status":"completed","operation":gate,"kind":"zero computed gate before decoder",
                "accuracy":float((arm_pred==target).mean()),
                "accuracy_delta":float((arm_pred==target).mean()-(pred==target).mean()),"run":arm}
    result["intervention"]=optional_arm("pan",result,intervene)
    return result

def mamba_chat():
    import ctypes
    import gc
    import numpy as np
    import torch
    from transformers import AutoTokenizer,MambaConfig,MambaForCausalLM
    from mamba_chat_source import acquire
    from program import lower_mrun
    torch.set_num_threads(2)
    import scipy_openblas32 as blas
    cpp=f"-I{shlex.quote(blas.get_include_dir())} -DTCM_SCIPY_OPENBLAS"
    ld=f"-L{shlex.quote(blas.get_lib_dir())} -Wl,-rpath,{shlex.quote(blas.get_lib_dir())} -l{blas.get_library()}"
    command(["make","program-blas",f"BLAS_CPPFLAGS={cpp}",f"BLAS_LDLIBS={ld}"])
    source,normalization=acquire()
    (RAW/"normalization.json").write_text(json.dumps(normalization,indent=2)+"\n")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
    physical=normalization["physical_rows"];semantic=normalization["semantic_vocab"]
    config=MambaConfig.from_pretrained(source,local_files_only=True)
    # The model constructor allocates physical row counts. The canonical source
    # config/IO still owns the smaller semantic token domain, masked below.
    config.vocab_size=physical
    print("PHASE load independent HF source oracle FP32 CPU, physical-vocab override recorded",flush=True)
    model=MambaForCausalLM.from_pretrained(source,config=config,torch_dtype=torch.float32,local_files_only=True).eval()
    prompts=([("capital","日本の首都はどこですか？",12),("addition","7足す5はいくつですか？",12)]
             if normalization["language"]=="Japanese" else [("greeting","Reply with one word: hello.",8),("addition","What is 7 plus 5? Answer with a number.",12)])
    native=[]
    for name,text,budget in prompts:
        messages=[{"role":"user","content":text}]
        if tokenizer.chat_template:
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            prompt_contract="pinned-source-tokenizer-chat-template"
        else:
            rendered="<|user|>\n"+text+"\n<|assistant|>\n"
            prompt_contract="official-model-card-Zephyr-role-markers"
        ids=tokenizer.encode(rendered,add_special_tokens=False)
        if any(i<0 or i>=semantic for i in ids):raise ValueError("tokenizer emitted unmapped token")
        if len(ids)+budget-1>64:raise ValueError("prompt/continuation exceeds declared64token geometry")
        print(f"PHASE native {name}: prompt_tokens={len(ids)}, own_budget={budget}",flush=True)
        with torch.no_grad():
            out=model(torch.tensor([ids]),use_cache=True)
            logits=[out.logits[0].float().numpy().copy()];selected=[]
            for t in range(budget):
                token=int(out.logits[0,-1,:semantic].argmax());selected.append(token)
                if token==tokenizer.eos_token_id or t+1==budget:break
                out=model(torch.tensor([[token]]),cache_params=out.cache_params,use_cache=True)
                logits.append(out.logits[0].float().numpy().copy())
            full=model(torch.tensor([ids+selected[:-1]]),use_cache=False).logits[0,-1].float().numpy()
            last=out.logits[0,-1].float().numpy()
            cache=out.cache_params
            state={}
            for i in range(config.num_hidden_layers):
                conv=cache.layers[i].conv_states[0] if hasattr(cache,"layers") else cache.conv_states[i]
                recurrent=cache.layers[i].recurrent_states[0] if hasattr(cache,"layers") else cache.ssm_states[i]
                state[f"layers.{i}.conv_state"]=conv.float().numpy().copy()
                state[f"layers.{i}.recurrent_state"]=recurrent.float().numpy().copy()
            native.append({"name":name,"prompt":text,"rendered":rendered,"input_ids":ids,"budget":budget,
                           "prompt_contract":prompt_contract,"generated_tokens":selected,
                           "text":tokenizer.decode(selected,skip_special_tokens=True),
                           "logits":np.concatenate(logits),"state":state,
                           "cached_vs_full_prefix_max_abs":float(np.abs(full-last).max())})
            snapshot("mamba-"+name+"-native-summary",{k:v for k,v in native[-1].items() if k not in ("logits","state")})
        del out,cache,full,last,logits,state
    del model;gc.collect()
    try:ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:pass
    print("PHASE native model released; canonical source import and C program compilation",flush=True)
    mapping=lower_mrun(source,RAW/"mamba-chat.tcm",ir_destination=RAW/"canonical-source-ir.json")
    from program_chat import verify_tokenizer_binding
    tokenizer_binding=verify_tokenizer_binding(RAW/"mamba-chat.tcm",source)
    gc.collect()
    try:ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:pass
    environment={"torch":torch.__version__,"threads":2,
                 "tokenizer_binding":tokenizer_binding,
                 "openblas_package_config":blas.get_pkg_config(),
                 "openblas_threads":os.environ.get("OPENBLAS_NUM_THREADS"),
                 "blas_linkage":command(["ldd","build/tcm-blas"]).stdout,
                 "note":"ffp-contract=off applies to compiled C; OpenBLAS controls its own reductions/FMA; this is measured cross-profile evidence"}
    outputs=[]
    for item in native:
        name=item["name"]
        print(f"PHASE C {name}: own tokens plus checkpoint suffix replay",flush=True)
        record,flat=c_run(RAW/"mamba-chat.tcm",np.asarray(item["input_ids"]).reshape(-1,1),"mamba-"+name,
                          generate=item["budget"],eos=[tokenizer.eos_token_id],state=True,binary="build/tcm-blas")
        actual=flat.reshape(-1,physical);expected=item["logits"]
        a,b=record["generated_tokens"],item["generated_tokens"]
        common=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        comparable=min(len(actual),len(expected),len(item["input_ids"])+common)
        delta=actual[:comparable]-expected[:comparable]
        cstate=np.fromfile(RAW/("mamba-"+name+".state.f32"),dtype=np.float32)
        state_metrics=[];selected_state={}
        if a[:-1]==b[:-1]:
            for slot in mapping["state_slots"]:
                start=slot["offset"];size=int(np.prod(slot["shape"]))
                cs=cstate[start:start+size].reshape(slot["shape"]);ns=item["state"][slot["name"]].reshape(slot["shape"])
                state_metrics.append({"slot":slot["name"],"max_abs":float(np.abs(cs-ns).max()),"mean_abs":float(np.abs(cs-ns).mean())})
                if slot["name"].startswith(("layers.0.",f"layers.{config.num_hidden_layers-1}.")):
                    selected_state["c."+slot["name"]]=cs;selected_state["native."+slot["name"]]=ns
        np.savez_compressed(RAW/("mamba-"+name+".npz"),native_logits=expected,c_logits=actual,
                            per_row_max_abs=np.abs(delta).max(1),**selected_state)
        (RAW/("mamba-"+name+".logits.f32")).unlink();(RAW/("mamba-"+name+".state.f32")).unlink()
        result={k:v for k,v in item.items() if k not in ("logits","state")}
        result.update(c=record,c_text=tokenizer.decode(a,skip_special_tokens=True),common_generated_prefix=common,
                      comparable_rows=comparable,first_divergence=common if common<min(len(a),len(b)) else None,
                      max_abs_logit_error=float(np.abs(delta).max()),mean_abs_logit_error=float(np.abs(delta).mean()),
                      common_input_top1_agreement=float((actual[:comparable,:semantic].argmax(1)==expected[:comparable,:semantic].argmax(1)).mean()),
                      state_comparison=state_metrics)
        outputs.append(result)
        snapshot("mamba-"+name+"-comparison-summary",result)
    # A typed intervention at a mapped consumed writer, replaying from zero state.
    result={"specimen":normalization["repository"],"revision":normalization["revision"],"normalization":normalization,
            "native_profile":f"HF Mamba FP32 CPU sequential scan and cached continuation; physical rows{physical}; semantic selection{semantic}",
            "environment":environment,"operations":len(mapping["instructions"]),"state_slots":len(mapping["state_slots"]),
            "state_floats":sum(int(np.prod(s["shape"])) for s in mapping["state_slots"]),"prompts":outputs,
            "authority":{"mechanics_status":"executed; one-row commits and complete token-boundary suffix replay",
                         "instrument_status":"source-and-C-common-prefix-comparison",
                         "trend_status":"bounded numerical, continuation and causal observations; inspect per-slot and per-row differences",
                         "terminal_status":"no-native-BF16, full-StateIR-transactions, chat-quality-improvement or speedup claim"}}
    def intervene():
        writer=next(i for i,o in enumerate(mapping["instructions"]) if o["source"].endswith("/layers.0.mixer.out_proj"))
        first=native[0]
        print("PHASE C source-mapped zero-writer exploratory arm",flush=True)
        arm,flat=c_run(RAW/"mamba-chat.tcm",np.asarray(first["input_ids"]).reshape(-1,1),"mamba-zero-writer",
                        generate=first["budget"],eos=[tokenizer.eos_token_id],zero=writer,binary="build/tcm-blas")
        actual=flat.reshape(-1,physical);ninput=len(first["input_ids"])
        with np.load(RAW/("mamba-"+first["name"]+".npz")) as evidence:baseline=evidence["c_logits"]
        signed=actual[:ninput]-baseline[:ninput]
        np.savez_compressed(RAW/"mamba-writer-intervention.npz",baseline_prefill=baseline[:ninput],arm_prefill=actual[:ninput],
                            per_row_signed_mean=signed.mean(1),per_row_max_abs=np.abs(signed).max(1))
        (RAW/"mamba-zero-writer.logits.f32").unlink()
        return {"status":"completed","operation":writer,"source":mapping["instructions"][writer]["source"],"run":arm,
                "text":tokenizer.decode(arm["generated_tokens"],skip_special_tokens=True),
                "prefill_max_abs_change":float(np.abs(signed).max())}
    result["intervention"]=optional_arm("mamba",result,intervene)
    return result

def main(experiments=None,*,retain_bundles=False,evidence_budget_bytes=64*1024**2):
    if type(evidence_budget_bytes) is not int or not 0<evidence_budget_bytes<=256*1024**2:
        raise ValueError('invalid bounded evidence budget')
    threads="2" if sys.argv[1]=="mamba-chat" else "1"
    os.environ["OMP_NUM_THREADS"]=threads;os.environ["OPENBLAS_NUM_THREADS"]=threads
    RAW.mkdir(exist_ok=False);start=time.time();report={"schema":"tcm-model-intake-experiment-v1","mode":sys.argv[1]}
    try:
        command(["make"])
        report["result"]=(experiments or {"pan":pan,"mamba-chat":mamba_chat})[sys.argv[1]]()
        report["status"]="completed"
    except BaseException:
        report["status"]="error";report["error"]=traceback.format_exc();print(report["error"],flush=True)
    report["wall_seconds"]=time.time()-start
    report["executed_sources"]={str(p):sha(p) for p in sorted(Path("src").glob("*")) if p.is_file()}
    report["raw"]={str(p):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(RAW.rglob("*")) if p.is_file() and (retain_bundles or p.suffix!=".tcm")}
    (RAW/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("TCM_RESULT_GZIP_B64="+base64.b64encode(gzip.compress(json.dumps(report).encode())).decode(),flush=True)
    files=[p for p in sorted(RAW.rglob("*")) if p.is_file() and (retain_bundles or p.suffix!=".tcm")]
    if sum(p.stat().st_size for p in files)>evidence_budget_bytes:raise ValueError("expanded evidence exceeds declared budget")
    # Separate bounded archives avoid one giant log line and keep all raw
    # comparisons. The collector enforces the combined expanded-byte budget.
    groups=[];group=[];size=0
    for p in files:
        if group and size+p.stat().st_size>8*1024**2:
            groups.append(group);group=[];size=0
        group.append(p);size+=p.stat().st_size
    if group:groups.append(group)
    for group in groups:
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode="w:gz") as archive:
            for p in group:archive.add(p,arcname=str(p),recursive=False)
        print("TCM_FILES_TGZ_B64="+base64.b64encode(stream.getvalue()).decode(),flush=True)
    return 0 if report["status"]=="completed" else 1

if __name__=="__main__":raise SystemExit(main())
