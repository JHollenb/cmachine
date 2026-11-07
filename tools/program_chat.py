"""Tokenization client for a compiled C program; C owns numerical continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from transformers import AutoTokenizer
from bundle import MAGIC
from inspect_program import inspect

def verify_tokenizer_binding(model, source):
    """Check the sealed model/map join and every non-weight source asset.

    The compiler's complete source inventory also covers auxiliary tokenizer
    files, such as Japanese emoji.json, outside canonical IOIR's asset roles.
    A tokenizer-only copy may omit weights but must retain all bound metadata.
    """
    model,source=Path(model),Path(source)
    mapping=json.loads(model.with_suffix(".map.json").read_text())
    inspect(mapping)
    with model.open("rb") as stream:prefix=stream.read(40)
    if prefix[:8]!=MAGIC or prefix[8:].hex()!=mapping["bundle"]["content_sha256"]:
        raise ValueError("model and source map differ")
    origin=mapping.get("origin") or {}
    files=origin.get("source_files")
    if not files:raise ValueError("source map lacks codec custody; recompile with the current compiler")
    expected={item["path"]:item for item in files if item["role"]!="weight-shard"}
    if not any(item["role"] in ("tokenizer","tokenizer-config") for item in expected.values()):
        raise ValueError("source map has no bound tokenizer assets")
    for name,item in expected.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("invalid source asset path")
        path=source/relative;digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
        if path.stat().st_size!=item["byte_count"] or digest.hexdigest()!=item["sha256"]:
            raise ValueError(f"tokenizer source asset differs: {name}")
    actual=set()
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix==".safetensors":continue
        relative=path.relative_to(source)
        if relative.parts[:3]==(".cache","huggingface","download"):continue
        actual.add(relative.as_posix())
    if actual!=set(expected):raise ValueError("unbound or missing tokenizer source assets")
    return {"source_fingerprint":origin["source_fingerprint"],
            "bundle_digest":mapping["bundle"]["content_sha256"],
            "assets":{name:item["sha256"] for name,item in expected.items()}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--binary",type=Path,default=Path("build/tcm-blas"))
    parser.add_argument("--prompt",required=True)
    parser.add_argument("--new-tokens",type=int,default=32)
    parser.add_argument("--zephyr-roles",action="store_true",help="use documented original Mamba-Chat role markers when tokenizer has no template")
    args=parser.parse_args()
    if not 1<=args.new_tokens<=1024:parser.error("new-tokens must be 1..1024")
    binding=verify_tokenizer_binding(args.model,args.tokenizer)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    if tokenizer.chat_template:
        text=tokenizer.apply_chat_template([{"role":"user","content":args.prompt}],tokenize=False,add_generation_prompt=True)
    elif args.zephyr_roles:text="<|user|>\n"+args.prompt+"\n<|assistant|>\n"
    else:parser.error("tokenizer has no template; provide an explicit prompt-format option")
    ids=tokenizer.encode(text,add_special_tokens=False)
    if verify_tokenizer_binding(args.model,args.tokenizer)!=binding:
        raise ValueError("tokenizer source changed during encoding")
    with tempfile.TemporaryDirectory(prefix="tcm-chat-") as temporary:
        root=Path(temporary);np.asarray(ids,dtype=np.float32).tofile(root/"input.f32")
        cmd=[str(args.binary.resolve()),"program",str(args.model.resolve()),"--input",str(root/"input.f32"),
             "--output",str(root/"logits.f32"),"--generate",str(args.new_tokens)]
        if tokenizer.eos_token_id is not None:cmd += ["--eos",str(tokenizer.eos_token_id)]
        run=subprocess.run(cmd,capture_output=True,text=True,check=True)
        record=json.loads(run.stdout)
        if record["bundle_digest"]!=binding["bundle_digest"]:raise ValueError("executed model differs from tokenizer binding")
        record["tokenizer_binding"]=binding
        print(tokenizer.decode(record["generated_tokens"],skip_special_tokens=True))
        print(json.dumps(record))

if __name__=="__main__":main()
