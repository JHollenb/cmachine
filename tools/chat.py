"""Text-only client for the persistent C machine; tokenizer is outside inference.

The exact token history controls cache reuse. If rendering the next chat turn
changes an already processed prefix, explicitly reset and recompute that prompt.
Optional memory text is supplied visibly as user context, never hidden state.
Run the executable and this client inside an admitted mrun GPU lease on Beast.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,required=True)
    parser.add_argument("--binary",type=Path)
    parser.add_argument("--cache-policy",choices=("native","full-byte","bounded"),default="native")
    parser.add_argument("--anchors",type=int,default=16)
    parser.add_argument("--recent",type=int,default=64)
    parser.add_argument("--reservoir",type=int,default=32)
    parser.add_argument("--context",type=int,default=4096)
    parser.add_argument("--new-tokens",type=int,default=128)
    parser.add_argument("--memory",type=Path)
    parser.add_argument("--prompt")
    args=parser.parse_args()
    if not 1<=args.context<=32768 or not 1<=args.new_tokens<=4096:
        parser.error("context must be 1..32768 and new-tokens must be 1..4096")
    if args.cache_policy=="bounded" and (args.anchors<0 or args.recent<1 or args.reservoir<0 or args.anchors+args.recent+args.reservoir>args.context):
        parser.error("bounded support needs nonnegative anchors/reservoir, positive recent rows, and total support <= context")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    metadata_path=args.model.with_suffix(".json")
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        for filename,digest in metadata.get("tokenizer_files",{}).items():
            if hashlib.sha256((args.tokenizer/filename).read_bytes()).hexdigest()!=digest:
                raise ValueError("tokenizer/template configuration differs from model bundle")
    if args.prompt is not None and args.prompt.startswith("/"):
        raise ValueError("one-shot mode expects a prompt, not an interactive command")
    binary=args.binary or Path(__file__).resolve().parents[1]/"build/tcm-cuda"
    command=[str(binary),"serve",str(args.model),"--context",str(args.context)]
    if args.cache_policy!="native":command += ["--cache-policy",args.cache_policy,"--anchors",str(args.anchors),"--recent",str(args.recent),"--reservoir",str(args.reservoir)]
    machine=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def receive():
        line=machine.stdout.readline()
        if not line:raise RuntimeError(f"C machine exited: {machine.poll()}")
        result=json.loads(line)
        if "error" in result:raise RuntimeError(result["error"])
        return result
    def send(command):
        machine.stdin.write(command+"\n");machine.stdin.flush();return receive()
    ready=receive();print(json.dumps(ready),flush=True)
    if ready.get("cache_policy","native")!=args.cache_policy:
        machine.terminate();machine.wait(timeout=10);raise RuntimeError("C machine did not activate the requested cache policy")
    if args.cache_policy=="bounded" and any(ready.get(k)!=getattr(args,k) for k in ("anchors","recent","reservoir")):
        machine.terminate();machine.wait(timeout=10);raise RuntimeError("C machine did not activate the requested bounded support")
    if args.cache_policy!="native":print("[experimental "+args.cache_policy+" memory; /state reports the live allocation]",flush=True)
    stack_limit=ready.get("stack_limit",64);legacy_checkpoint=ready.get("legacy_checkpoint",True)
    messages=[];processed=[];saved=None;frames=[]
    memory=args.memory.read_text() if args.memory else None
    if memory:print("Explicit memory context:\n"+memory,flush=True)
    try:
        while True:
            text=args.prompt if args.prompt is not None else input("you> ")
            if text in ("/quit","/exit"):break
            if text=="/state":print(send("S"));continue
            if text=="/save":
                if not legacy_checkpoint:print("This memory policy uses /push and /rollback; legacy /save is unavailable.");continue
                if not processed:print("Send a message before saving a parent.");continue
                print(send("C"));saved=(messages,processed);continue
            if text=="/rewind":
                if not legacy_checkpoint:print("This memory policy uses /push and /rollback; legacy /rewind is unavailable.");continue
                if saved is None:print("No saved parent.");continue
                if frames:print("Commit or roll back the nested frames before rewinding the saved parent.");continue
                print(send("R"));messages,processed=saved;continue
            if text=="/push":
                if not processed:print("Send a message before pushing a frame.");continue
                if len(frames)>=stack_limit:print("Nested frame limit reached; current state is preserved.");continue
                result=send("P");frames.append((result["frame"],messages,processed));print(result);continue
            if text in ("/rollback","/commit"):
                if not frames:print("No nested frame.");continue
                frame,old_messages,old_processed=frames[-1]
                result=send(("B " if text=="/rollback" else "K ")+str(frame))
                frames.pop()
                if text=="/rollback":messages,processed=old_messages,old_processed;saved=None
                print(result);continue
            if memory and not messages:text="Additional explicit memory:\n"+memory+"\n\nUser message:\n"+text
            proposed=messages+[{"role":"user","content":text}]
            ids=tokenizer.apply_chat_template(proposed,tokenize=True,add_generation_prompt=True,return_dict=False)
            if not isinstance(ids,list) or not all(isinstance(i,int) for i in ids):raise TypeError("tokenizer must return a flat token-ID list")
            if len(ids)+args.new_tokens-1>args.context:
                print("This turn and generation budget exceed the configured context; the current state is preserved.",flush=True)
                if args.prompt is not None:break
                continue
            if ids[:len(processed)]!=processed:
                send("Z");processed=[];saved=None;frames=[];print("[recomputed prompt after token-prefix change; saved parent and nested frames invalidated]",flush=True)
            suffix=ids[len(processed):]
            if suffix:send("A "+" ".join(map(str,suffix)))
            result=send("G "+str(args.new_tokens));generated=result["ids"]
            processed=ids+generated[:-1] if generated else ids
            answer=tokenizer.decode(generated,skip_special_tokens=True)
            messages=proposed+[{"role":"assistant","content":answer}]
            print("model> "+answer,flush=True)
            if args.prompt is not None:break
    except EOFError:
        pass
    finally:
        if machine.poll() is None:
            machine.stdin.write("Q\n");machine.stdin.flush()
        machine.wait(timeout=10)


if __name__=="__main__":main()
