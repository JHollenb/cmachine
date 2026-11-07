"""Seal interpretation separately from immutable executed payload/report/raw."""
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
    report_hash = sha(root / 'report.json')
    verification = json.loads((root / 'verification.json').read_text())
    if verification['report_sha256'] != report_hash:
        raise ValueError('Verified report differs')
    manifest = json.loads((root / 'payload/source-manifest.json').read_text())
    sources = []
    # Capture executed sources, not the subsequently edited working driver/protocol.
    for name, digest in manifest['sources'].items():
        path = root / 'payload' / name
        if sha(path) != digest:
            raise ValueError('Executed source differs: ' + name)
        sources.append((path, 'executed-payload/' + name))
    sources.append((root / 'payload/source-manifest.json', 'executed-payload/source-manifest.json'))
    for name in ('study.py', 'plot.py', 'evidence.py', 'publish.py', 'README.md', 'EXECUTED_PROTOCOL.md', 'MATH.md', 'FINDINGS.md', 'REVIEW.md', 'NEXT.md'):
        sources.append((HERE / name, 'interpretation-source/' + name))
    for name in ('verification.json', 'study.json', 'query-horizon.png'):
        sources.append((root / name, name))
    for action in ('run', 'verify'):
        sources.append((HERE / 'index' / job / f'evidence-{action}.json', f'evidence-{action}.json'))
    domains = HERE.parents[2]
    for name in ('obsidian/blog/2026-11-06-181408-a-query-program-has-to-live-in-the-future-it-creates.md', 'obsidian/blog/2026-11-06-173328-the-consumer-metric-needs-a-clock.md', 'obsidian/blog/2026-08-14-what-saturn-can-actually-do.md', 'obsidian/images/job-309bf320370d-query-horizon.png', 'saturn/cmachine/README.md', 'saturn/cmachine/FINDINGS.md', 'saturn/cmachine/STACKED_STATE_PLAN.md', 'saturn/cmachine/query_consumer/NEXT.md', 'saturn/docs/CAPABILITY-MACHINE.md'):
        sources.append((domains / name, 'docs/' + name))
    output = root / ('interpretation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir()
    records = {}
    for source, name in sources:
        digest = sha(source)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(source) != digest or sha(target) != digest:
            raise ValueError('Interpretation drift while copying ' + name)
        records[name] = {'sha256': digest, 'bytes': target.stat().st_size, 'source': str(source)}
    if sha(root / 'report.json') != report_hash:
        raise ValueError('Report changed')
    seal = {'schema': 'tcm-query-horizon-interpretation-v1', 'report_sha256': report_hash, 'verification_sha256': sha(root / 'verification.json'), 'scope': 'Additive saved-only interpretation of completed32-token forced-source/fresh-own-target run. Executed payload explicitly captured separately from subsequent working64-token/context development. Original report/raw/payload unchanged. No model calls or serving policy change.', 'files': records}
    path = output / 'manifest.json'
    path.write_text(json.dumps(seal, indent=2) + '\n')
    print(json.dumps({'manifest': str(path), 'sha256': sha(path), 'files': len(records)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('job')
    publish(p.parse_args().job)
