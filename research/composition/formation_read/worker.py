"""Execute only PN/NP; historical endpoint arrays are analysis-only inputs."""
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
    if ref["report_sha256"]!=manifest["predecessor_report_sha256"] or ref["model"]!=baseline["model"]:raise ValueError("Historical/model identity differs")
    for name,record in baseline["bank_files"].items():
        if sha(ROOT/"bank"/name)!=record["sha256"] or (ROOT/"bank"/name).stat().st_size!=record["bytes"]:raise ValueError("Bank differs")
    model=baseline["model"];source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name,expected in model["tokenizer_files"].items():
        if sha(source/name)!=expected:raise ValueError("Tokenizer differs")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True);messages=[{"role":"user","content":baseline["source_template_text"]}]
    source_text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tokenizer.encode(source_text,add_special_tokens=False)
    if ids!=baseline["source_template_ids"] or ids!=ref["inputs"]["source_ids"]:raise ValueError("Source template differs")
    (RAW/"source.txt").write_text(" ".join(map(str,ids))+"\n");(RAW/"historical-source-own.txt").write_text(" ".join(map(str,baseline["historical_source_own_ids"]))+"\n")
    questions=ref["inputs"]["questions"]
    for q in questions:
        full=tokenizer.apply_chat_template(messages+[{"role":"assistant","content":""},{"role":"user","content":q["text"]}],tokenize=False,add_generation_prompt=True)
        if not full.startswith(source_text):raise ValueError("Template text prefix differs")
        suffix=tokenizer.encode(full[len(source_text):],add_special_tokens=False)
        if suffix!=q["ids"] or tokenizer.encode(full,add_special_tokens=False)!=ids+suffix or suffix[0]!=151645 or len(suffix)>64 or q["budget"]!=8:raise ValueError("Append grammar differs")
        (RAW/q["file"]).write_text(" ".join(map(str,suffix))+"\n")
    (RAW/"colors.json").write_text(json.dumps(ref["colors"],sort_keys=True)+"\n")
    maxima=[len(q["ids"])+8 for q in questions];per_path=sum(maxima)
    payload_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "results" not in p.relative_to(ROOT).parts)
    bound=2*per_path*(5*1536+1024)*4+per_path*2*1536*4+12*151936*4+6*2*8960*4+4*1024**2+payload_bytes
    if len(ids)+len(baseline["historical_source_own_ids"])+maxima[0]+max(maxima[1:])>512 or bound>support.LIMIT:raise ValueError("Capacity/custody bound exceeded")
    build=command(["make"])
    report={"schema":"tcm-formation-read-experiment-v1","model":model,"baseline":baseline,"inputs":ref["inputs"],"predecessor_report_sha256":ref["report_sha256"],"colors":ref["colors"],"environment":{"platform":platform.platform(),"python":sys.version,"job_id":os.environ.get("MRUN_JOB_ID")},"binary_sha256":sha(ROOT/"build/composition"),"build_stderr":build.stderr,"budget":{"max_selected_layer_transitions":2*per_path,"raw_and_payload_byte_bound":bound,"raw_archive_limit_bytes":support.LIMIT},"mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested"}
    preserve(report)
    args=[ROOT/"build/composition",model["path"],RAW/"source.txt",*[RAW/q["file"] for q in questions],RAW,ROOT/"bank",RAW/"historical-source-own.txt",baseline["historical_source_kv_sha256"],baseline["bank_concat_sha256"],ref["formations"]["native"]["final_kv_sha256"],ref["formations"]["project"]["final_kv_sha256"],token_hash(final_history(ref["formations"]["native"])),token_hash(final_history(ref["formations"]["project"]))]
    request={"model_digest":model["content_sha256"],"binary_sha256":report["binary_sha256"],"inputs":report["inputs"],"bank_files":baseline["bank_files"],"plan_sha256":manifest["plan_sha256"],"predecessor_report_sha256":ref["report_sha256"],"origin_sha256":sha(ROOT/"origin.json"),"budget":report["budget"]}
    def controller(_):
        result=command(args);(RAW/"driver.stdout.txt").write_text(result.stdout);(RAW/"driver.stderr.txt").write_text(result.stderr);return json.loads((RAW/"driver.json").read_text())
    driver,report["saturn_receipt"]=invoke(args,None,request,controller=controller);report["driver"]=driver
    if driver["model_digest"]!=model["content_sha256"] or driver["source_history_ids"]!=ref["source_history_ids"] or driver["source_kv_sha256"]!=ref["source_kv_sha256"]:raise ValueError("Actual source/model differs")
    for key in ("frozen_bank_before_sha256","frozen_bank_after_sha256"):
        if driver[key]!=baseline["bank_concat_sha256"]:raise ValueError("Owned bank changed")
    report["texts"]={t["name"]:tokenizer.decode(t["answer_ids"],skip_special_tokens=True) for arm in driver["arms"] for t in arm["turns"]}
    preserve(report);report["analysis"]=analyze(RAW,ROOT/"bank",driver,report["texts"])
    report["raw_files"]={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in (".f32",".txt") or p.name in ("driver.json","colors.json")}
    report.update(mechanics_status="complete",instrument_status="complete");return report

def emit_files():
    files=[(p,"raw/"+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    for p in sorted(ROOT.rglob("*")):
        relative=p.relative_to(ROOT)
        if p.is_file() and "__pycache__" not in relative.parts and (len(relative.parts)==1 or relative.parts[0] in ("runtime","bank","public","historical")):files.append((p,"payload/"+str(relative)))
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
