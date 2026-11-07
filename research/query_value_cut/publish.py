"""Freeze additive saved-only interpretation without changing executed artifacts."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result=HERE/'runs/job-27a597ebd5f8';verification=json.loads((result/'verification.json').read_text())
    if verification['report_sha256']!=sha(result/'report.json'):raise ValueError('Verified report differs')
    destination=result/('interpretation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));destination.mkdir();files={}
    sources=[(HERE/name,'source/'+name) for name in ('README.md','PLAN.md','MATH.md','FINDINGS.md','ADVICE.md','PRE_RUN_REVIEW.md','REVIEW.md','study.py','evidence.py','plot.py','publish.py')]
    blog=HERE.parents[2]/'obsidian/blog/2026-11-06-144600-a-program-that-produces-queries-and-updates-its-own-memory.md';sources.append((blog,'source/'+blog.name))
    sources.extend((result/name,name) for name in ('study.json','query-value-cut.png'))
    sources.extend((HERE/'index/job-27a597ebd5f8'/f'evidence-{action}.json',f'evidence-{action}.json') for action in ('run','verify'))
    for source,name in sources:
        before=sha(source);target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=before or sha(source)!=before:raise ValueError('Interpretation changed while copying')
        files[name]={'sha256':before,'bytes':target.stat().st_size,'source':str(source)}
    manifest={'schema':'tcm-query-value-interpretation-v1','report_sha256':sha(result/'report.json'),'verification_sha256':sha(result/'verification.json'),'scope':'additive saved-only joins, evidence and interpretations; no model call; executed payload/raw/report unchanged','files':files}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'manifest':str(destination/'manifest.json'),'sha256':sha(destination/'manifest.json')}))
if __name__=='__main__':main()
