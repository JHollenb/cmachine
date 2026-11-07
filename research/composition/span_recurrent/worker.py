"""One source-only constructor and five recurring, linked native-consumer chains."""
from pathlib import Path
import base64,gzip,io,json,os,platform,sys,tarfile,time,traceback
import support
from support import ROOT,RAW,sha,command,preserve
support.LIMIT=48*1024**2

def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze,final_history,token_hash
    sys.path.insert(0,str(ROOT/"public"));from saturn_bridge import invoke
    baseline=json.loads((ROOT/"baseline.json").read_text());manifest=json.loads((ROOT/"source-manifest.json").read_text());ref=json.loads((ROOT/"origin.json").read_text());questions=json.loads((ROOT/"questions.json").read_text())
    for name,expected in manifest["sources"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Shipped source differs")
    if ref["predecessor_report_sha256"]!=manifest["predecessor_report_sha256"] or ref["model"]!=baseline["model"]:raise ValueError("Historical model differs")
    model=baseline["model"];source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name,expected in model["tokenizer_files"].items():
        if sha(source/name)!=expected:raise ValueError("Tokenizer differs")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);messages=[{"role":"user","content":baseline["source_template_text"]}];text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(text,add_special_tokens=False)
    if ids!=baseline["source_template_ids"]:raise ValueError("Source template differs")
    (RAW/"source.txt").write_text(" ".join(map(str,ids))+"\n");(RAW/"source-own.txt").write_text(" ".join(map(str,baseline["historical_source_own_ids"]))+"\n")
    correction=ref["correction_input"];(RAW/"correction.txt").write_text(" ".join(map(str,correction["ids"]))+"\n")
    inputs=[]
    for q in [correction]+questions:
        full=tokenizer.apply_chat_template(messages+[{"role":"assistant","content":""},{"role":"user","content":q["text"]}],tokenize=False,add_generation_prompt=True);suffix=tokenizer.encode(full[len(text):],add_special_tokens=False)
        if not full.startswith(text) or tokenizer.encode(full,add_special_tokens=False)!=ids+suffix or not suffix or suffix[0]!=151645 or len(suffix)>96:raise ValueError("Append grammar differs")
        if q is correction and suffix!=q["ids"]:raise ValueError("Correction template differs")
        if q is not correction:
            (RAW/q["file"]).write_text(" ".join(map(str,suffix))+"\n");inputs.append({**q,"ids":suffix})
    if 93+sum(len(q["ids"])+q["budget"] for q in inputs)>512:raise ValueError("Linked context bound exceeded")
    payload_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "results" not in p.relative_to(ROOT).parts)
    # Source/correction capture,30readouts,30sparseports, every future query, and bounded JSON.
    transitions=5*sum(len(q["ids"])+q["budget"] for q in inputs)
    bound=32*151936*4+40*8*1536*4+30*(10*1536+2*256+2*8960)*4+transitions*1536*4+30*12*129*8+8*1024**2+payload_bytes
    if bound>support.LIMIT:raise ValueError("Custody bound exceeded")
    build=command(["make"]);report={"schema":"tcm-span-recurrent-experiment-v1","model":model,"inputs":{"source_ids":ids,"correction":correction,"questions":inputs},"predecessor_report_sha256":ref["predecessor_report_sha256"],"environment":{"platform":platform.platform(),"python":sys.version,"job_id":os.environ.get("MRUN_JOB_ID")},"binary_sha256":sha(ROOT/"build/composition"),"build_stderr":build.stderr,"budget":{"linked_turns":15,"selected_read_application_bound":transitions,"own_token_bound":240,"raw_and_payload_byte_bound":bound,"raw_archive_limit_bytes":support.LIMIT},"mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested"};preserve(report)
    args=[ROOT/"build/composition",model["path"],RAW/"source.txt",RAW/"correction.txt",*[RAW/q["file"] for q in questions],RAW,RAW/"source-own.txt",ref["source_kv_sha256"],ref["correction"]["final_kv_sha256"],token_hash(final_history(ref["correction"]))]
    request={"model_digest":model["content_sha256"],"binary_sha256":report["binary_sha256"],"inputs":report["inputs"],"plan_sha256":manifest["plan_sha256"],"predecessor_report_sha256":ref["predecessor_report_sha256"],"origin_sha256":sha(ROOT/"origin.json"),"constructor_visibility":"native correction KV17 only; no future queries; freeze before question files; candidate phase poisoned C17 and no dense L17 fallback","budget":report["budget"]}
    def controller(_):
        result=command(args);(RAW/"driver.stdout.txt").write_text(result.stdout);(RAW/"driver.stderr.txt").write_text(result.stderr);return json.loads((RAW/"driver.json").read_text())
    driver,report["saturn_receipt"]=invoke(args,None,request,controller=controller);report["driver"]=driver
    if driver["model_digest"]!=model["content_sha256"]:raise ValueError("Actual model differs")
    report["texts"]={a["name"]:tokenizer.decode(a["answer_ids"],skip_special_tokens=True) for group in driver["chains"] for a in group["turns"]};preserve(report)
    report["analysis"]=analyze(RAW,ROOT,driver,report["texts"]);report["raw_files"]={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in (".f32",".f64",".bin",".txt") or p.name=="driver.json"}
    report.update(mechanics_status="complete",instrument_status="complete");return report

def emit_files():
    files=[(p,"raw/"+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    for p in sorted(ROOT.rglob("*")):
        rel=p.relative_to(ROOT)
        if p.is_file() and "__pycache__" not in rel.parts and (len(rel.parts)==1 or rel.parts[0] in ("runtime","public")):files.append((p,"payload/"+str(rel)))
    if sum(p.stat().st_size for p,_ in files)>support.LIMIT:raise ValueError("Archive exceeds bound")
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
