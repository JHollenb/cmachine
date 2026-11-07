"""Freeze additive saved-only interpretation without changing executed artifacts."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(job):
    result=HERE/'runs'/job;verification=json.loads((result/'verification.json').read_text())
    if verification['report_sha256']!=sha(result/'report.json'):raise ValueError('Verified report differs')
    destination=result/('interpretation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));destination.mkdir();files={}
    sources=[(HERE/name,'source/'+name) for name in ('README.md','PLAN.md','MATH.md','FINDINGS.md','ADVICE.md','PRE_RUN_REVIEW.md','REVIEW.md','study.py','evidence.py','plot.py','publish.py')]
    blog=HERE.parents[2]/'obsidian/blog/2026-11-06-161445-a-better-query-from-the-same-small-code.md';sources.append((blog,'source/'+blog.name))
    sources.extend((result/name,name) for name in ('study.json','query-program.png'))
    sources.extend((HERE/name,'source/'+name) for name in ('compiler.py','fresh.json','audit.py','stable_metrics.py','audit_replay.py','audit_worker.py','verify_replay.py','analysis.py','run.py','test_analysis.py'))
    sources.extend((result/name,name) for name in ('fp64-query-metrics.json','verification-refusal-diagnostic.json','audit-replay.json','audit-replay-submission.json','audit-replay-job.json','audit-replay-logs.txt','verification-source.py'))
    sources.extend((HERE/'index'/job/f'evidence-{action}.json',f'evidence-{action}.json') for action in ('run','verify'))
    for source,name in sources:
        before=sha(source);target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=before or sha(source)!=before:raise ValueError('Interpretation changed while copying')
        files[name]={'sha256':before,'bytes':target.stat().st_size,'source':str(source)}
    manifest={'schema':'tcm-query-program-interpretation-v1','report_sha256':sha(result/'report.json'),'verification_sha256':sha(result/'verification.json'),'scope':'additive saved-only joins, evidence and interpretations; no model call; executed payload/raw/report unchanged','files':files}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'manifest':str(destination/'manifest.json'),'sha256':sha(destination/'manifest.json')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');main(p.parse_args().job)
