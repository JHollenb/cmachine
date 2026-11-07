"""Isolated Saturn sweep over verified composition and scientific streams."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evidence(job):
    result=HERE/'runs'/job;index=HERE/'index'/job;target=index/'target-streams';target.mkdir(parents=True,exist_ok=True)
    if json.loads((result/'verification.json').read_text())['report_sha256']!=sha(result/'report.json'):raise ValueError('Verification/report identity differs')
    if json.loads((result/'analysis/scientific-analysis.json').read_text())['report_sha256']!=sha(result/'report.json'):raise ValueError('Scientific/report identity differs')
    streams=[result/'observations.jsonl',result/'analysis/observations.jsonl']
    (target/'observations.jsonl').write_text(''.join(p.read_text() for p in streams))
    (target/'source.json').write_text(json.dumps({'job':job,'report_sha256':sha(result/'report.json'),
        'verification_sha256':sha(result/'verification.json'),'scientific_analysis_sha256':sha(result/'analysis/scientific-analysis.json'),
        'reader_attribution_sha256':sha(result/'analysis/reader-attribution.json'),
        'scope':'Six short conversations and matched-history writer/reader composition; numerical and trend stream coverage only; no terminal certification'},indent=2)+'\n')
    package=types.ModuleType('saturn');package.__path__=[str(HERE.parents[1]/'src/saturn')];sys.modules['saturn']=package
    from saturn.evidence_pipeline import main
    for action in ('run','verify'):
        args=[action,str(target),'--watch-root',str(index/'watch'),'--registry',str(index/'claims.jsonl'),'--results-root',str(target),'--json']
        if action=='run':args+=['--xref-root',str(target),'--xref-db',str(index/'xref.sqlite')]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):code=main(args)
        (index/f'evidence-{action}.json').write_text(output.getvalue())
        print(json.dumps({'job':job,'action':action,'code':code}))
        if code not in (0,3):raise RuntimeError('Evidence instrument refused inputs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');evidence(p.parse_args().job)
