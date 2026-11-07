"""Freeze additive interpretation separately from immutable scientific payloads."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');blog=ROOT.parents[2]/'obsidian/blog/2026-11-06-144600-a-program-that-produces-queries-and-updates-its-own-memory.md'
    for job in ('job-2230480e1c18','job-077becbef4c9','job-c7334bbc5bb8'):
        result=ROOT/'runs'/job;verification=json.loads((result/'verification.json').read_text())
        if verification['report_sha256']!=sha(result/'report.json'):raise ValueError('Report identity differs')
        destination=result/f'interpretation-{stamp}';destination.mkdir();files={}
        sources=[(ROOT/name,'source/'+name) for name in ('README.md','FINDINGS.md','MATH.md','REVIEW.md','PLAN.md','study.py','evidence.py','plot.py','publish.py')]
        sources += [(blog,'source/'+blog.name),(result/'study.json','study.json')]
        sources += [(ROOT/'index'/job/f'evidence-{action}.json',f'evidence-{action}.json') for action in ('run','verify')]
        if (result/'query-memory.png').exists():sources.append((result/'query-memory.png','query-memory.png'))
        for source,name in sources:
            digest=sha(source);target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            if sha(target)!=digest or sha(source)!=digest:raise ValueError('Interpretation source changed during copy')
            files[name]={'sha256':digest,'bytes':target.stat().st_size,'source':str(source)}
        manifest={'schema':'tcm-query-memory-interpretation-v1','timestamp':stamp,'job_id':job,'report_sha256':sha(result/'report.json'),'verification_sha256':sha(result/'verification.json'),'scope':'additive saved-only diagnostics and cross-run interpretation; no model calls; executed payload/raw/report unchanged','files':files}
        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'job':job,'manifest':str(destination/'manifest.json'),'sha256':sha(destination/'manifest.json')}))
if __name__=='__main__':main()
