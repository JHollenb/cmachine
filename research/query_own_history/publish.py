"""Seal additive interpretation and executed-source identities separately."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(job):
    root = HERE / 'runs' / job
    digest = sha(root / 'report.json')
    if json.loads((root / 'verification.json').read_text())['report_sha256'] != digest:
        raise ValueError('Verified report differs')
    manifest = json.loads((root / 'payload/source-manifest.json').read_text())
    sources = []
    for name, expected in manifest['sources'].items():
        p = root / 'payload' / name
        if sha(p) != expected:
            raise ValueError('Executed source differs: '+name)
        sources.append((p, 'executed-payload/'+name))
    sources.append((root / 'payload/source-manifest.json', 'executed-payload/source-manifest.json'))
    for name in ('study.py', 'plot.py', 'evidence.py', 'watch_audit.py', 'publish.py', 'README.md', 'PLAN.md', 'MATH.md', 'FINDINGS.md', 'REVIEW.md', 'NEXT.md'):
        sources.append((HERE / name, 'interpretation-source/'+name))
    for name in ('verification.json', 'study.json', 'watch-audit.json', 'query-own-history.png', 'observations.jsonl'):
        sources.append((root / name, name))
    for action in ('run', 'verify'):
        sources.append((HERE / 'index' / job / f'evidence-{action}.json', f'evidence-{action}.json'))
    fires = list((HERE / 'index' / job / 'watch').glob('**/*.fires.jsonl'))
    if len(fires) != 1:
        raise ValueError('Watch ledger differs')
    sources.append((fires[0], 'entropy-watch.fires.jsonl'))
    domains = HERE.parents[2]
    docs = ('obsidian/blog/2026-11-06-184252-the-same-query-program-visits-a-different-future.md', 'obsidian/blog/2026-11-06-181408-a-query-program-has-to-live-in-the-future-it-creates.md', 'obsidian/blog/2026-11-06-183300-longer-chat-context-still-needs-the-right-relationships.md', 'obsidian/blog/2026-08-14-what-saturn-can-actually-do.md', 'obsidian/images/job-47cc27324445-query-own-history.png', 'saturn/cmachine/README.md', 'saturn/cmachine/FINDINGS.md', 'saturn/cmachine/STACKED_STATE_PLAN.md', 'saturn/cmachine/query_horizon/NEXT.md', 'saturn/docs/CAPABILITY-MACHINE.md')
    sources.extend((domains / name, 'docs/'+name) for name in docs)
    output = root / ('interpretation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir()
    records = {}
    for source, name in sources:
        expected = sha(source)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(source) != expected or sha(target) != expected:
            raise ValueError('Interpretation drift: '+name)
        records[name] = {'sha256': expected, 'bytes': target.stat().st_size, 'source': str(source)}
    if sha(root / 'report.json') != digest:
        raise ValueError('Original report changed')
    seal = {'schema': 'tcm-query-own-history-interpretation-v1', 'report_sha256': digest, 'verification_sha256': sha(root / 'verification.json'), 'scope': 'Additive matched-schedule history interpretation, loss/support and saved reader factorization, offline entropy watch audit, documentation and next proposals. Executed report/raw/payload unchanged; no further model calls, fitted image or serving policy change.', 'files': records}
    path = output / 'manifest.json'
    path.write_text(json.dumps(seal, indent=2)+'\n')
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'files': len(records)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('job')
    publish(p.parse_args().job)
