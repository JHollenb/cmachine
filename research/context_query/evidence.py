"""Post-job evidence sweep over the verified, curated observation stream."""
import argparse,contextlib,hashlib,io,json,shutil,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
def evidence(job):
    root=HERE/'runs'/job;index=HERE/'index'/job;target=index/'target-streams';target.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if json.loads((root/'verification.json').read_text())['report_sha256']!=sha(root/'report.json'):raise ValueError('Verification identity differs')
    shutil.copy2(root/'observations.jsonl',target/'observations.jsonl');(target/'source.json').write_text(json.dumps({'job':job,'report_sha256':sha(root/'report.json'),'verification_sha256':sha(root/'verification.json'),'scope':'Curated source/fresh own-token turn observations only. Custody, declared updates and stream arithmetic qualified separately. No semantic or speed certification.'},indent=2)+'\n')
    package=types.ModuleType('saturn');package.__path__=[str(HERE.parents[2]/'saturn/src/saturn')];sys.modules['saturn']=package
    from saturn.evidence_pipeline import main
    for action in ('run','verify'):
        args=[action,str(target),'--watch-root',str(index/'watch'),'--registry',str(index/'claims.jsonl'),'--results-root',str(target),'--json']
        if action=='run':args+=['--xref-root',str(target),'--xref-db',str(index/'xref.sqlite')]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):code=main(args)
        (index/f'evidence-{action}.json').write_text(output.getvalue());print(json.dumps({'action':action,'code':code,'result':json.loads(output.getvalue())}))
        if code not in (0,3):raise RuntimeError('Evidence instrument refused input')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');evidence(p.parse_args().job)
