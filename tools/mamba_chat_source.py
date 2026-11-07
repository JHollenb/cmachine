"""Pinned, byte-preserving normalization of the original Mamba-Chat checkpoint.

Source acquisition runs only inside the admitted fleet worker. No pickle, remote
Python, quantization or inferred semantic labels. The receipt retains both the
source and derived identities; the original snapshot is never edited.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

REPO="havenhq/mamba-chat"
REVISION="84fa8e355ece8f4f55cf0f2a8b81cf189b3eeee0"
EXPECTED_SHA="1bd09ef683484ab9bede711703cd12fa57168e6c3dde6a75412f848af6d24e81"
PROFILES={
    "mamba-chat-2.8b":{"repository":REPO,"revision":REVISION,"sha256":EXPECTED_SHA,"bytes":5547903216,
                      "hidden":2560,"layers":64,"vocab":50277,"physical_rows":50280,"bos":0,"eos":0,
                      "language":"English","sizing_label":"2.8b","fp32_bank_bytes":11073382400},
    "mambasan-130m-instruct":{"repository":"loiccabannes/MambaSan-130m-instruct",
                              "revision":"b2eb0b6f88b5904a6166afc69e9bd7180920c63b",
                              "sha256":"a33f55c181141bade9280388354751830a35204fcef0d3b3a9d7c2a05ad1e007",
                              "bytes":231472040,"hidden":768,"layers":24,"vocab":32000,"physical_rows":32000,
                              "bos":31996,"eos":31999,"language":"Japanese","sizing_label":"130m","fp32_bank_bytes":460385280}}
DEFAULT_PROFILE="mambasan-130m-instruct"

def download_http(original,profile_key):
    """The child imports HF only after explicit transport environment is set."""
    from huggingface_hub import snapshot_download
    profile=PROFILES[profile_key]
    snapshot_download(profile["repository"],revision=profile["revision"],local_dir=original,
                      allow_patterns=["model.safetensors","config.json","tokenizer*","special_tokens_map.json","README.md","vocab.txt","vocab.json","merges.txt","emoji.json"],
                      max_workers=2)

def bounded_download(original,profile_key):
    env={**os.environ,"HF_HUB_OFFLINE":"0","HF_HUB_DISABLE_XET":"1","HF_HUB_DOWNLOAD_TIMEOUT":"60"}
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--download-http",str(original),profile_key],env=env)
    last=None;last_progress=time.monotonic();started=last_progress
    try:
        while child.poll() is None:
            rows=[(str(p.relative_to(original)),p.stat().st_size,p.stat().st_mtime_ns)
                  for p in sorted(original.rglob("*")) if p.is_file() and (p.name.endswith(".incomplete") or p.name=="model.safetensors")]
            if rows!=last:last=rows;last_progress=time.monotonic()
            print(json.dumps({"phase":"source-download","bytes_visible":sum(r[1] for r in rows),
                              "seconds_without_write":round(time.monotonic()-last_progress,1)}),flush=True)
            if time.monotonic()-last_progress>120:raise RuntimeError("source-download-incomplete: no weight-file writes for120seconds")
            if time.monotonic()-started>1200:raise RuntimeError("source-download-incomplete: declared1200second acquisition budget reached")
            time.sleep(10)
        if child.returncode:raise RuntimeError(f"source-download-incomplete: transport child exited{child.returncode}")
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()

def acquire(profile_key=DEFAULT_PROFILE):
    from mrun.paths import models_root
    cache=models_root()/"cmachine-model-intake"
    profile=PROFILES[profile_key]
    repo,revision,expected_sha=profile["repository"],profile["revision"],profile["sha256"]
    stem=repo.rsplit("/",1)[-1]+"-"+revision[:8]
    # Keep the stalled Xet partial from job-372c21791726 unchanged; the HTTP
    # transport uses a separate resumable source root with the same pinned bytes.
    original=cache/(stem+"-http-source-v1")
    print("PHASE acquiring pinned Mamba-Chat safetensors and tokenizer",flush=True)
    bounded_download(original,profile_key)
    weights=original/"model.safetensors"
    if weights.stat().st_size!=profile["bytes"] or sha(weights)!=expected_sha:
        raise ValueError("pinned Mamba-Chat weight custody mismatch")
    config=json.loads((original/"config.json").read_text())
    expected={"d_model":profile["hidden"],"n_layer":profile["layers"],"vocab_size":profile["vocab"],"rms_norm":True,
              "residual_in_fp32":True,"fused_add_norm":True,"pad_vocab_size_multiple":8}
    for k,v in expected.items():
        if config.get(k)!=v:raise ValueError(f"unexpected original config: {k}")
    if config.get("ssm_cfg",{})!={}:raise ValueError("non-default SSM config unsupported")
    with weights.open("rb") as f:
        old_length=struct.unpack("<Q",f.read(8))[0];header=json.loads(f.read(old_length))
    metadata=header.get("__metadata__",{})
    if metadata.get("lm_head.weight")!="backbone.embedding.weight" or "lm_head.weight" in header:
        raise ValueError("expected original tied-head alias evidence absent")
    if header["backbone.embedding.weight"]["shape"]!=[profile["physical_rows"],profile["hidden"]]:
        raise ValueError("original lexical row shape differs")
    # Rename only a header key. Tensor offsets and all payload bytes are copied
    # unchanged. There is still exactly one lexical physical allocation.
    header["backbone.embeddings.weight"]=header.pop("backbone.embedding.weight")
    header["__metadata__"]={**metadata,"lm_head.weight":"backbone.embeddings.weight"}
    derived=cache/(stem+"-hf-tied-v1")
    recipe={"schema":"tcm-mamba-chat-normalization-v1","profile":profile_key,"repository":repo,"revision":revision,
            "original_weight_sha256":expected_sha,"original_config":config,"language":profile["language"],
            "rename":{"backbone.embedding.weight":"backbone.embeddings.weight"},
            "alias_evidence":metadata,"semantic_vocab":profile["vocab"],"physical_rows":profile["physical_rows"],
            "tensor_payload":"byte-preserved; no dtype conversion",
            "state_defaults":"Mamba1 state16,conv4,expand2,rankceil(hidden/16),SiLU,RMS epsilon1e-5; tensor shapes checked by canonical decompiler"}
    if not derived.exists():
        temporary=cache/(derived.name+".partial")
        if temporary.exists():raise ValueError("incomplete previous normalization requires inspection")
        temporary.mkdir(parents=True)
        for item in original.iterdir():
            if item.is_file() and item.name not in ("model.safetensors","config.json"):
                shutil.copy2(item,temporary/item.name)
        encoded=json.dumps(header,separators=(",",":"),sort_keys=True).encode()
        encoded += b" "*((-len(encoded))%8)
        with weights.open("rb") as source,(temporary/"model.safetensors").open("xb") as out:
            source.seek(8+old_length);out.write(struct.pack("<Q",len(encoded)));out.write(encoded)
            shutil.copyfileobj(source,out,4<<20)
        hf_config={"architectures":["MambaForCausalLM"],"model_type":"mamba","hidden_size":profile["hidden"],
                   "intermediate_size":2*profile["hidden"],"num_hidden_layers":profile["layers"],"state_size":16,"conv_kernel":4,
                   "time_step_rank":(profile["hidden"]+15)//16,"vocab_size":profile["vocab"],"tie_word_embeddings":True,
                   "layer_norm_epsilon":1e-5,"rms_norm":True,"residual_in_fp32":True,
                   "fused_add_norm":True,"pad_vocab_size_multiple":8,"use_bias":False,
                   "use_conv_bias":True,"hidden_act":"silu","expand":2,
                   "bos_token_id":profile["bos"],"eos_token_id":profile["eos"],"pad_token_id":profile["eos"]}
        (temporary/"config.json").write_text(json.dumps(hf_config,indent=2)+"\n")
        recipe["derived_weights_sha256"]=sha(temporary/"model.safetensors")
        recipe["assets"]={p.name:sha(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        (temporary/"normalization-receipt.json").write_text(json.dumps(recipe,indent=2)+"\n")
        temporary.rename(derived)
    else:
        recorded=json.loads((derived/"normalization-receipt.json").read_text())
        for k,v in recipe.items():
            if recorded[k]!=v:raise ValueError("normalization receipt changed")
        for name,digest in recorded["assets"].items():
            if sha(derived/name)!=digest:raise ValueError("normalized source changed")
        recipe=recorded
    print("PHASE pinned Mamba-Chat normalized; tied head and semantic vocabulary retained",flush=True)
    return derived,recipe

def probe():
    """Metadata-only acquisition diagnosis; does not open model tensor data."""
    from mrun.paths import models_root
    root=models_root()/"cmachine-model-intake"/("mamba-chat-"+REVISION[:8]+"-source")
    for observation in range(2):
        files=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,
                "mtime_age_s":round(time.time()-p.stat().st_mtime,1)}
               for p in sorted(root.rglob("*")) if p.is_file()]
        print(json.dumps({"root":str(root),"observation":observation,"files":files}),flush=True)
        if observation==0:time.sleep(15)

if __name__=="__main__":
    if sys.argv[1:]==["--probe"]:probe()
    elif len(sys.argv)==4 and sys.argv[1]=="--download-http":download_http(Path(sys.argv[2]),sys.argv[3])
    else:raise SystemExit("use --probe for metadata-only inspection")
