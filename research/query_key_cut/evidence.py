"""Standard offline sweep over the verified bounded intervention observations."""
import argparse,contextlib,hashlib,io,json,shutil,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
def evidence(job):
    root=HERE/'runs'/job;index=HERE/'index'/job;target=index/'target-streams';target.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if json.loads((root/'verification.json').read_text())['report_sha256']!=sha(root/'report.json'):raise ValueError('Verification identity differs')
    report=json.loads((root/'report.json').read_text())
    scope=('Eight own-token paths on four known development conversations, two frozen operand masks; not new heldout generalization or independent same-state effects. Numerical custody, native same-ID shadows and physical work audited separately. No general semantic or speed certification.' if report.get('schema')=='qk18-on-policy-v1' else 'Curated36logical intervention rows, including4aliases; not independent specimens. Numerical custody, native replay and inner C cache recovery audited separately. No general semantic or speed certification.')
    shutil.copy2(root/'observations.jsonl',target/'observations.jsonl');(target/'source.json').write_text(json.dumps({'job':job,'report_sha256':sha(root/'report.json'),'verification_sha256':sha(root/'verification.json'),'scope':scope},indent=2)+'\n')
    package=types.ModuleType('saturn');package.__path__=[str(HERE.parents[2]/'saturn/src/saturn')];sys.modules['saturn']=package
    from saturn.evidence_pipeline import main
    for action in ('run','verify'):
        args=[action,str(target),'--watch-root',str(index/'watch'),'--registry',str(index/'claims.jsonl'),'--results-root',str(target),'--json']
        if action=='run':args+=['--xref-root',str(target),'--xref-db',str(index/'xref.sqlite')]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):code=main(args)
        (index/f'evidence-{action}.json').write_text(output.getvalue());print(json.dumps({'action':action,'code':code,'instrument_status':json.loads(output.getvalue()).get('instrument_status')}))
        if code not in (0,3):raise RuntimeError('Evidence instrument refused input')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');evidence(p.parse_args().job)
