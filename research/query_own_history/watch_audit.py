"""Qualify offline entropy watch fires without changing the original ledger."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def audit(job):
    here = Path(__file__).resolve().parent
    root, index = here / 'runs' / job, here / 'index' / job
    rows = [json.loads(line) for line in (root / 'observations.jsonl').read_text().splitlines()]
    ledgers = list((index / 'watch').glob('**/*.fires.jsonl'))
    if len(ledgers) != 1:
        raise ValueError('Expected one entropy watch ledger')
    fires = [json.loads(line) for line in ledgers[0].read_text().splitlines()]
    counts = collections.Counter()
    examples = []
    for fire in fires:
        i = fire['record_index']
        row = rows[i]
        if fire['field'] != 'native_entropy_nats' or fire['predicate_id'] != 'ent-spike-native-entropy-nats' or row['native_entropy_nats'] != fire['observed']:
            raise ValueError('Unexpected watch finding')
        previous = rows[i-1] if i else None
        kind = 'between-pairs' if previous and previous['pair'] != row['pair'] else 'prediction-role-change' if previous and previous['prediction_turn'] != row['prediction_turn'] else 'within-role'
        counts[kind] += 1
        if len(examples) < 6:
            examples.append({'pair': row['pair'], 'step': row['step'], 'prediction_turn': row['prediction_turn'], 'reference_entropy_nats': fire['reference'], 'entropy_nats': fire['observed'], 'kind': kind})
    result = {'schema': 'tcm-own-history-watch-audit-v1', 'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(), 'ledger_sha256': hashlib.sha256(ledgers[0].read_bytes()).hexdigest(), 'fires': len(fires), 'all_fires': 'Relative native-entropy increases over standard factor3 threshold', 'categories': dict(counts), 'examples': examples, 'scope': 'Offline diagnostic watch of heterogeneous token/phase histories, not an invalidity gate. Independent finite-value checks and raw identities pass; no numerical corruption or semantic conclusion follows from the entropy ratio alone. Original ledger unchanged.'}
    (root / 'watch-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('job')
    audit(p.parse_args().job)
