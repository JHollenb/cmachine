"""Sixteen read-only traces inside a single finite GPU lease."""
from pathlib import Path
import base64,gzip,io,json,os,platform,sys,tarfile,time,traceback
import support
from support import ROOT,RAW,sha,command,preserve
support.LIMIT=48*1024**2

def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze,final_history,token_hash
    sys.path.insert(0,str(ROOT/"public"));from saturn_bridge import invoke
    baseline=json.loads((ROOT/"baseline.json").read_text());manifest=json.loads((ROOT/"source-manifest.json").read_text());ref=json.loads((ROOT/"origin.json").read_text())
    for name,expected in manifest["sources"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Shipped source differs")
    if ref["predecessors"]!=manifest["predecessors"] or ref["model"]!=baseline["model"]:raise ValueError("Historical model differs")
    for name,record in baseline["bank_files"].items():
        if sha(ROOT/"bank"/name)!=record["sha256"] or (ROOT/"bank"/name).stat().st_size!=record["bytes"]:raise ValueError("Frozen bank differs")
    model=baseline["model"];source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name,expected in model["tokenizer_files"].items():
        if sha(source/name)!=expected:raise ValueError("Tokenizer differs")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);messages=[{"role":"user","content":baseline["source_template_text"]}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(text,add_special_tokens=False)
    if ids!=baseline["source_template_ids"] or ids!=ref["inputs"]["source_ids"]:raise ValueError("Source template differs")
    (RAW/"source.txt").write_text(" ".join(map(str,ids))+"\n");(RAW/"historical-source-own.txt").write_text(" ".join(map(str,baseline["historical_source_own_ids"]))+"\n")
    questions=ref["inputs"]["questions"]
    for q in questions:
        full=tokenizer.apply_chat_template(messages+[{"role":"assistant","content":""},{"role":"user","content":q["text"]}],tokenize=False,add_generation_prompt=True)
        suffix=tokenizer.encode(full[len(text):],add_special_tokens=False)
        if not full.startswith(text) or suffix!=q["ids"] or tokenizer.encode(full,add_special_tokens=False)!=ids+suffix or suffix[0]!=151645 or len(suffix)>64 or q["budget"]!=8:raise ValueError("Append grammar differs")
        (RAW/q["file"]).write_text(" ".join(map(str,suffix))+"\n")
    maxima=[len(q["ids"])+8 for q in questions];payload_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "results" not in p.relative_to(ROOT).parts)
    trace_floats=8*1536+2*256+2*8960+2*2*512*128+2*12*512+1
    bound=20*151936*4+16*trace_floats*4+1536*1536*4+4*1024**2+payload_bytes
    if len(ids)+len(baseline["historical_source_own_ids"])+maxima[0]+max(maxima[1:])>512 or bound>support.LIMIT:raise ValueError("Capacity/custody bound exceeded")
    build=command(["make"])
    report={"schema":"tcm-attention-read-experiment-v1","model":model,"baseline":baseline,"inputs":ref["inputs"],"predecessors":ref["predecessors"],"environment":{"platform":platform.platform(),"python":sys.version,"job_id":os.environ.get("MRUN_JOB_ID")},"binary_sha256":sha(ROOT/"build/composition"),"build_stderr":build.stderr,"budget":{"selected_traces":16,"max_observed_transitions":2*maxima[0]+4*sum(maxima[1:]),"raw_and_payload_byte_bound":bound,"raw_archive_limit_bytes":support.LIMIT},"mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested"};preserve(report)
    args=[ROOT/"build/composition",model["path"],RAW/"source.txt",*[RAW/q["file"] for q in questions],RAW,ROOT/"bank",RAW/"historical-source-own.txt",baseline["historical_source_kv_sha256"],baseline["bank_concat_sha256"],ref["formations"]["N"]["final_kv_sha256"],ref["formations"]["P"]["final_kv_sha256"],token_hash(final_history(ref["formations"]["N"])),token_hash(final_history(ref["formations"]["P"]))]
    request={"model_digest":model["content_sha256"],"binary_sha256":report["binary_sha256"],"inputs":report["inputs"],"bank_files":baseline["bank_files"],"plan_sha256":manifest["plan_sha256"],"predecessors":ref["predecessors"],"origin_sha256":sha(ROOT/"origin.json"),"budget":report["budget"]}
    def controller(_):
        result=command(args);(RAW/"driver.stdout.txt").write_text(result.stdout);(RAW/"driver.stderr.txt").write_text(result.stderr);return json.loads((RAW/"driver.json").read_text())
    driver,report["saturn_receipt"]=invoke(args,None,request,controller=controller);report["driver"]=driver
    if driver["model_digest"]!=model["content_sha256"] or driver["source_history_ids"]!=ref["source_history_ids"] or driver["source_kv_sha256"]!=ref["source_kv_sha256"]:raise ValueError("Actual source/model differs")
    for key in ("frozen_bank_before_sha256","frozen_bank_after_sha256"):
        if driver[key]!=baseline["bank_concat_sha256"]:raise ValueError("Frozen bank changed")
    turns=[t for group in driver["formations"] for t in [group["correction"]]+group["readers"]];report["texts"]={t["name"]:tokenizer.decode(t["answer_ids"],skip_special_tokens=True) for t in turns}
    preserve(report);report["analysis"]=analyze(RAW,ROOT/"bank",driver,report["texts"])
    report["raw_files"]={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in (".f32",".txt") or p.name=="driver.json"}
    report.update(mechanics_status="complete",instrument_status="complete");return report

def emit_files():
    files=[(p,"raw/"+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    for p in sorted(ROOT.rglob("*")):
        rel=p.relative_to(ROOT)
        if p.is_file() and "__pycache__" not in rel.parts and (len(rel.parts)==1 or rel.parts[0] in ("runtime","bank","public")):files.append((p,"payload/"+str(rel)))
    if sum(p.stat().st_size for p,_ in files)>support.LIMIT:raise ValueError("Archive exceeds declared bound")
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode="w:gz") as archive:
        for path,name in files:archive.add(path,arcname=name,recursive=False)
    print("COMPOSITION_FILES_TGZ_B64="+base64.b64encode(buffer.getvalue()).decode(),flush=True)

if __name__=="__main__":
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:
        report=json.loads((RAW/"partial-report.json").read_text()) if (RAW/"partial-report.json").exists() else {};report.update(error=traceback.format_exc(),mechanics_status="incomplete",instrument_status="incomplete");code=1
    report["worker_seconds"]=time.monotonic()-started;raw=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/"report.json").write_bytes(raw+b"\n")
    print("COMPOSITION_RESULT_GZIP_B64="+base64.b64encode(gzip.compress(raw,mtime=0)).decode(),flush=True);emit_files();raise SystemExit(code)
