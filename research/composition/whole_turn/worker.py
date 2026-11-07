"""One resident C model; seal primary consumers before paid diagnostic replay."""
from pathlib import Path
import base64,gzip,hashlib,json,os,platform,sys,time,traceback
import support
from support import ROOT,RAW,sha,command,preserve
support.LIMIT=96*1024**2

QUESTIONS=(
    ("correction","Correction: Niko's badge is now green. Keep the other person's badge unchanged. Reply only with OK."),
    ("target","What color is Niko's badge now? Reply with the color only."),
    ("collateral","What badge color does Mira still have? Reply with the color only."),
)

def experiment():
    from transformers import AutoTokenizer
    from analysis import analyze
    sys.path.insert(0,str(ROOT/"public"));from saturn_bridge import invoke
    baseline=json.loads((ROOT/"baseline.json").read_text());manifest=json.loads((ROOT/"source-manifest.json").read_text())
    for name,expected in manifest["sources"].items():
        if sha(ROOT/name)!=expected:raise ValueError("Shipped source differs")
    for name,record in baseline["bank_files"].items():
        if sha(ROOT/"bank"/name)!=record["sha256"] or (ROOT/"bank"/name).stat().st_size!=record["bytes"]:raise ValueError("Frozen bank differs")
    model=baseline["model"];source=Path("/mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    for name,expected in model["tokenizer_files"].items():
        if sha(source/name)!=expected:raise ValueError("Tokenizer differs")
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    messages=[{"role":"user","content":baseline["source_template_text"]}]
    source_text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    ids=tokenizer.encode(source_text,add_special_tokens=False)
    if ids!=baseline["source_template_ids"]:raise ValueError("Source template differs")
    (RAW/"source.txt").write_text(" ".join(map(str,ids))+"\n")
    (RAW/"historical-source-own.txt").write_text(" ".join(map(str,baseline["historical_source_own_ids"]))+"\n")
    inputs=[]
    for name,text in QUESTIONS:
        full=tokenizer.apply_chat_template(messages+[{"role":"assistant","content":""},{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
        if not full.startswith(source_text):raise ValueError("Template text prefix differs")
        suffix=tokenizer.encode(full[len(source_text):],add_special_tokens=False)
        if tokenizer.encode(full,add_special_tokens=False)!=ids+suffix or suffix[0]!=151645 or len(suffix)>64:raise ValueError("Template concatenation/64-token envelope differs")
        # Verify the same append grammar following a nonempty assistant response.
        parent_messages=messages+[{"role":"assistant","content":"red"},{"role":"user","content":QUESTIONS[0][1]}]
        for body in ("OK","An unexpected acknowledgment."):
            parent=tokenizer.apply_chat_template(parent_messages,tokenize=False,add_generation_prompt=True)
            extended=tokenizer.apply_chat_template(parent_messages+[{"role":"assistant","content":body},{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
            expected=tokenizer.encode(parent,add_special_tokens=False)+tokenizer.encode(body,add_special_tokens=False)+suffix
            if tokenizer.encode(extended,add_special_tokens=False)!=expected:raise ValueError("Later-turn template grammar differs")
        path=RAW/f"{name}.txt";path.write_text(" ".join(map(str,suffix))+"\n")
        inputs.append({"name":name,"text":text,"budget":8,"ids":suffix,"file":path.name})
    colors={c:tokenizer.encode(c,add_special_tokens=False) for c in ("red","blue","green")}
    (RAW/"colors.json").write_text(json.dumps(colors,sort_keys=True)+"\n")
    # Worst case counts include an explicit closing delimiter at each reader.
    maxima=[len(q["ids"])+8 for q in inputs];per_path=sum(maxima)
    max_transitions=7*per_path
    first_gu=21*2*8960*4
    local_bytes=7*per_path*(5*1536+1024)*4+3*per_path*2*1536*4
    bound=local_bytes+42*151936*4+first_gu+8*1024**2
    if len(ids)+len(baseline["historical_source_own_ids"])+maxima[0]+max(maxima[1:])>512 or bound>support.LIMIT:raise ValueError("Declared capacity/evidence bound exceeded")
    build=command(["make"])
    report={"schema":"tcm-whole-turn-experiment-v1","model":model,"baseline":baseline,"inputs":{"source_ids":ids,"questions":inputs},"colors":colors,
            "environment":{"platform":platform.platform(),"python":sys.version,"job_id":os.environ.get("MRUN_JOB_ID")},"binary_sha256":sha(ROOT/"build/composition"),"build_stderr":build.stderr,
            "budget":{"max_selected_layer_transitions":max_transitions,"raw_byte_bound":bound,"raw_archive_limit_bytes":support.LIMIT},"mechanics_status":"in-progress","instrument_status":"in-progress","trend_status":"observations","terminal_status":"not-requested"}
    preserve(report)
    args=[ROOT/"build/composition",model["path"],RAW/"source.txt",*[RAW/q["file"] for q in inputs],RAW,ROOT/"bank",RAW/"historical-source-own.txt",baseline["historical_source_kv_sha256"],baseline["bank_concat_sha256"]]
    request={"model_digest":model["content_sha256"],"binary_sha256":report["binary_sha256"],"inputs":report["inputs"],"bank_files":baseline["bank_files"],"plan_sha256":manifest["plan_sha256"],"budget":report["budget"]}
    def controller(_):
        result=command(args);(RAW/"driver.stdout.txt").write_text(result.stdout);(RAW/"driver.stderr.txt").write_text(result.stderr)
        return json.loads((RAW/"driver.json").read_text())
    driver,report["saturn_receipt"]=invoke(args,None,request,controller=controller);report["driver"]=driver
    if driver["model_digest"]!=model["content_sha256"] or driver["source_position"]!=baseline["historical_source_position"] or driver["source_kv_sha256"]!=baseline["historical_source_kv_sha256"]:raise ValueError("Source/model replay differs")
    for key in ("frozen_bank_before_sha256","frozen_bank_after_sha256"):
        if driver[key]!=baseline["bank_concat_sha256"]:raise ValueError("Owned program changed")
    report["texts"]={t["name"]:tokenizer.decode(t["answer_ids"],skip_special_tokens=True) for a in driver["arms"]+driver["diagnostics"] for t in a["turns"]}
    preserve(report);report["analysis"]=analyze(RAW,ROOT/"bank",driver,report["texts"])
    report["raw_files"]={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(RAW.iterdir()) if p.suffix in (".f32",".txt") or p.name in ("driver.json","colors.json")}
    report.update(mechanics_status="complete",instrument_status="complete");return report

if __name__=="__main__":
    RAW.mkdir(exist_ok=True);started=time.monotonic()
    try:report=experiment();code=0
    except Exception:
        report=json.loads((RAW/"partial-report.json").read_text()) if (RAW/"partial-report.json").exists() else {}
        report.update(error=traceback.format_exc(),mechanics_status="incomplete",instrument_status="incomplete");code=1
    report["worker_seconds"]=time.monotonic()-started
    raw=json.dumps(report,sort_keys=True,allow_nan=False).encode();(RAW/"report.json").write_bytes(raw+b"\n")
    print("COMPOSITION_RESULT_GZIP_B64="+base64.b64encode(gzip.compress(raw,mtime=0)).decode(),flush=True)
    support.emit_files();raise SystemExit(code)
