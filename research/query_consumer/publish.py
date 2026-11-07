"""Seal additive interpretation; leave executed payload/raw/report unchanged."""
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(job):
    root=HERE/'runs'/job;verified=json.loads((root/'verification.json').read_text());digest=sha(root/'report.json')
    if verified['report_sha256']!=digest:raise ValueError('Verified report differs')
    out=root/('interpretation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir();sources=[]
    sources.extend((p,'source/'+p.name) for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in ('.py','.md','.json','.c','.h'))
    sources.extend((root/name,name) for name in ('verification.json','analysis-recovery.json','study.json','horizon-audit.json','query-consumer.png'))
    sources.extend((HERE/'index'/job/f'evidence-{a}.json',f'evidence-{a}.json') for a in ('run','verify'))
    domains=HERE.parents[2];docs=('obsidian/blog/2026-11-06-173328-the-consumer-metric-needs-a-clock.md','obsidian/blog/2026-11-06-164143-a-query-can-be-less-native-and-more-useful.md','obsidian/blog/2026-08-14-what-saturn-can-actually-do.md','saturn/cmachine/README.md','saturn/cmachine/FINDINGS.md','saturn/cmachine/STACKED_STATE_PLAN.md','saturn/docs/CAPABILITY-MACHINE.md')
    sources.extend((domains/p,'docs/'+p) for p in docs);sources.extend((HERE/'runs/job-ad7b8088f482'/name,'interrupted-attempt/'+name) for name in ('job.json','logs.txt','submission.json'));files={}
    for source,name in sources:
        before=sha(source);target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(source)!=before or sha(target)!=before:raise ValueError('Interpretation drift while copying')
        files[name]={'sha256':before,'bytes':target.stat().st_size,'source':str(source)}
    if sha(root/'report.json')!=digest:raise ValueError('Report changed')
    manifest={'schema':'tcm-query-consumer-interpretation-v1','report_sha256':digest,'verification_sha256':sha(root/'verification.json'),'scope':'Additive saved-only interpretation, horizon audit and qualified recovery. Original scheduler failure/report/raw/payload unchanged; current analyzer fixes exactly two summary-field accesses. Interrupted attempt retained without a scientific outcome.','files':files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'manifest':str(out/'manifest.json'),'sha256':sha(out/'manifest.json'),'files':len(files)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');main(p.parse_args().job)
