"""Authenticate a completed C capture and repair only the historical-key join.

Original scheduler status, source payload, failed report and raw files stay immutable.
"""
from pathlib import Path
import argparse,base64,gzip,hashlib,importlib.util,io,json,sys,tarfile
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write_immutable(path,data):
    if path.exists() and path.read_bytes()!=data:raise ValueError('Derived immutable artifact differs: '+str(path))
    path.write_bytes(data)
def repair(root):
    root=Path(root).resolve();report=json.loads((root/'report.json').read_text());job=json.loads((root/'job.json').read_text());sub=json.loads((root/'submission.json').read_text())
    if job['state']!='failed' or job['job_id']!=sub['job_id'] or 'KeyError: \'prefill_kv_sha256\'' not in report.get('error','') or 'driver' not in report or len(report['driver']['chains'])!=3:raise ValueError('Not the named completed-capture incident')
    if sha(root/'logs.txt')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Scheduler log seal differs')
    lines=(root/'logs.txt').read_text().splitlines();logged=[json.loads(gzip.decompress(base64.b64decode(s.split('=',1)[1]))) for s in lines if s.startswith('COMPOSITION_RESULT_GZIP_B64=')]
    if logged!=[report] or json.loads((root/'raw/report.json').read_text())!=report:raise ValueError('Original report join differs')
    archive_lines=[s for s in lines if s.startswith('COMPOSITION_FILES_TGZ_B64=')]
    if len(archive_lines)!=1:raise ValueError('Archive membership differs')
    inventory={}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_lines[0].split('=',1)[1])),mode='r:gz') as archive:
        seen=set();members=archive.getmembers()
        if sum(m.size for m in members)>128*1024**2:raise ValueError('Archive extent exceeds plan')
        for m in members:
            rel=Path(m.name)
            if not m.isfile() or rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ('raw','payload') or m.name in seen:raise ValueError('Invalid archive member')
            seen.add(m.name);data=archive.extractfile(m).read();path=root/rel
            if path.read_bytes()!=data:raise ValueError('Archived bytes differ: '+m.name)
            if rel.parts[0]=='raw':inventory[str(Path(*rel.parts[1:]))]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    for n,h in sub['config']['sources'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Frozen source differs')
    if sha(root/'payload/source-manifest.json')!=sub['config']['source_manifest_sha256']:raise ValueError('Source manifest differs')
    for k in ('sources','source_manifest_sha256','predecessor_report_sha256'):
        if job['config'][k]!=sub['config'][k]:raise ValueError('Scheduler request differs')
    reference=json.loads((root/'payload/origin.json').read_text());prior=HERE.parent/'compact_append/runs/job-2a28feeb7e5b/report.json'
    if sha(prior)!=reference['predecessor_report_sha256']:raise ValueError('Historical parent seal differs')
    before="turn[c+'_logical_cache_sha256']!=old[c+'_kv_sha256']";after="turn[c+'_logical_cache_sha256']!=old.get(c+'_logical_cache_sha256',old.get(c+'_kv_sha256'))"
    original=(root/'payload/analysis.py').read_text();corrected=(HERE/'analysis.py').read_text()
    if original.count(before)!=1 or original.replace(before,after)!=corrected:raise ValueError('Repair exceeds the named field join')
    write_immutable(root/'analysis-repair-source.py',corrected.encode());sys.path.insert(0,str(root/'payload'))
    spec=importlib.util.spec_from_file_location('all_cache_repaired_analyzer',root/'analysis-repair-source.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.HERE=root/'payload'
    analysis=module.analyze(root/'raw',root/'payload',report['driver'],report['texts']);write_immutable(root/'reanalysis.json',canonical(analysis)+b'\n');write_immutable(root/'raw-inventory.json',json.dumps(inventory,sort_keys=True,indent=2).encode()+b'\n')
    wrapper=report['saturn_receipt'];receipt=dict(wrapper['receipt']);fingerprint=receipt.pop('fingerprint')
    if hashlib.sha256(canonical(receipt)).hexdigest()!=fingerprint or receipt['parent']!=wrapper['before'] or receipt['result']!=wrapper['after']:raise ValueError('Public invocation receipt differs')
    verification={'schema':'tcm-compact-all-saved-repair-v1','report_sha256':sha(root/'report.json'),'producer_job_status':'failed-in-postprocessing','original_failure_preserved':True,'scheduler_log_and_archive_verified':True,'source_files_verified':len(sub['config']['sources']),'raw_files_verified':len(inventory),'raw_bytes_verified':sum(r['bytes'] for r in inventory.values()),'analysis_source_sha256':sha(root/'analysis-repair-source.py'),'reanalysis_sha256':sha(root/'reanalysis.json'),'repair':'historical logical-cache field name only; no equations, raw rows or tolerances changed','whole_invocation_receipt':True,'analysis_recomputed':True,'models_evaluated':0,'mechanics_status':'complete-under-saved-only-repair','terminal_status':'not-requested'}
    write_immutable(root/'verification.json',json.dumps(verification,sort_keys=True,indent=2).encode()+b'\n');write_immutable(root/'repair-verifier-source.py',Path(__file__).read_bytes())
    observations=[{**row,'schema':'tcm-composition-observation-v1','model_id':report['model']['identity'],'report_sha256':verification['report_sha256'],'reanalysis_sha256':verification['reanalysis_sha256']} for row in analysis['observations']];write_immutable(root/'observations.jsonl',b''.join(canonical(row)+b'\n' for row in observations));print(json.dumps(verification));print(json.dumps(analysis['summary']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();repair(a.root)
