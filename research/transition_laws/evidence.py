"""Run Saturn's evidence instruments on this experiment's own observations."""
import argparse,contextlib,hashlib,io,json,shutil,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evidence(job):
    result=HERE/'runs'/job;index=HERE/'index'/job;target=index/'target-streams';target.mkdir(parents=True,exist_ok=True)
    verification=json.loads((result/'verification.json').read_text())
    if verification.get('mechanics_status')!='valid':raise ValueError('Evidence mechanics not verified')
    shutil.copy2(result/'observations.jsonl',target/'observations.jsonl')
    (target/'source.json').write_text(json.dumps({'job':job,'report_sha256':sha(result/'report.json'),'analysis_sha256':sha(result/'analysis.json'),'verification_sha256':sha(result/'verification.json'),'scope':'All28 rank256 receiver-factor experiment; conditional same-ID shadow metrics; source/monitor/fresh evaluation; exploratory trends only'},indent=2)+'\n')
    workspace=next(p for p in HERE.parents if (p/'saturn/src/saturn').is_dir())
    package=types.ModuleType('saturn');package.__path__=[str(workspace/'saturn/src/saturn')];sys.modules['saturn']=package
    from saturn.evidence_pipeline import main
    for action in ['run','verify']:
        args=[action,str(target),'--watch-root',str(index/'watch'),'--registry',str(index/'claims.jsonl'),'--results-root',str(target),'--json']
        if action=='run':args+=['--xref-root',str(target),'--xref-db',str(index/'xref.sqlite')]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):code=main(args)
        (index/f'evidence-{action}.json').write_text(output.getvalue());print(json.dumps({'action':action,'code':code}))
        if code not in [0,3]:raise RuntimeError('Evidence instrument refused')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');evidence(p.parse_args().job)
