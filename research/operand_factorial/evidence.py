"""Project verified factorial observations into a target-local evidence sweep."""
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;DOMAINS=HERE.parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def immutable(path,text):
    if path.exists() and path.read_text()!=text:raise ValueError('immutable projection differs: '+str(path))
    path.write_text(text)

def project(job):
    root=HERE/'runs'/job;report=json.loads((root/'report.json').read_text());v=json.loads((root/'verification.json').read_text())
    if v['report_sha256']!=sha(root/'report.json') or not v['saved_analysis_reproduced']:raise ValueError('verification join')
    if json.loads((root/'job.json').read_text())['job_id']!=job:raise ValueError('job identity')
    index=HERE/'index'/job;target=index/'target-streams';target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for kind in ['decomposition','observations','mechanics']:
        for r in report['analysis'][kind]:
            # Full signed Gram/term vectors remain in the authoritative report.
            scalars={k:x for k,x in r.items() if x is None or isinstance(x,(str,int,float,bool))}
            rows.append({'schema':'tcm-operand-factorial-observation-v1','kind':'observation','observable':kind,'report_sha256':v['report_sha256'],'model_id':report['model']['identity'],'model_content_sha256':report['model']['content_sha256'],**scalars})
    immutable(target/'observations.jsonl',''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    sources=[{'path':str((root/n).resolve()),'sha256':sha(root/n)} for n in ['report.json','verification.json','job.json','submission.json','raw/driver.json']]
    receipt={'schema':'tcm-operand-factorial-evidence-source-v1','mechanics_status':report['mechanics_status'],'instrument_status':report['instrument_status'],'trend_status':'observations','terminal_status':'not-requested','model':report['model'],'sources':sources,'observations':len(rows),'scope':'Scalar projection of verified raw C factorial and saved decomposition. No new model calls, no semantic promotion gates, no physical address trace claimed.'}
    immutable(target/'receipt.json',json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    result={}
    for action in ['run','verify']:
        args=[str(DOMAINS/'common/bin/common'),'exec','saturn-evidence',action,str(target),'--watch-root',str(index/'watch'),'--registry',str(index/'claims.jsonl'),'--results-root',str(target),'--json']
        if action=='run':args+=['--xref-db',str(index/'xref.sqlite'),'--xref-root',str(target)]
        r=subprocess.run(args,capture_output=True,text=True)
        (index/f'evidence-{action}.json').write_text(r.stdout);(index/f'evidence-{action}.stderr.txt').write_text(r.stderr)
        result[action]={'returncode':r.returncode,'summary':json.loads(r.stdout)}
        if r.returncode not in [0,3]:raise RuntimeError(r.stderr[-2000:])
    (index/'coverage.json').write_text(json.dumps({'report_sha256':v['report_sha256'],'rows':len(rows),'counts':{k:len(report['analysis'][k]) for k in ['decomposition','observations','mechanics']},'scope':receipt['scope']},indent=2)+'\n')
    print(json.dumps({'job':job,'rows':len(rows),'returncodes':{k:v['returncode'] for k,v in result.items()}}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');project(p.parse_args().job)
