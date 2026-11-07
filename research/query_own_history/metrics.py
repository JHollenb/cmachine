"""Independent FP64 full-vocabulary audit and explicit decision clocks."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(native, candidate):
    a, b = [np.asarray(x, dtype=float) for x in (native, candidate)]
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid distribution operands')
    def logp(x):
        z = x - x.max()
        return z - np.log(np.exp(z).sum())
    ln, lc = logp(a), logp(b)
    p = np.exp(ln)
    top = int(np.argmax(a))
    return {'kl_nats': float(p @ (ln - lc)), 'relative_logit_l2': float(np.linalg.norm(b-a) / max(float(np.linalg.norm(a)), 1e-30)), 'native_top_logp_delta_nats': float(lc[top]-ln[top]), 'native_entropy_nats': float(-p @ ln), 'native_top': top, 'candidate_top': int(np.argmax(b))}


def future_scores(rows, trace):
    if len(rows) != trace['count'] or [r['step'] for r in rows] != list(range(trace['count'])):
        raise ValueError('Clock extent differs')
    mask = [-1] * trace['count']
    means = []
    for j, turn in enumerate(trace['turns']):
        clocks = turn['prediction_clocks']
        expected = list(range(turn['prefill_end']-1, turn['final']-1))
        if not clocks or clocks != expected or len(clocks) != len(turn['answer_ids']):
            raise ValueError('Decision clock geometry differs')
        for t, token in zip(clocks, turn['answer_ids']):
            if trace['consumed_ids'][t+1] != token or rows[t]['expected_next_token'] != token:
                raise ValueError('Prediction token differs')
            mask[t] = j
        means.append(float(np.mean([rows[t]['kl_nats'] for t in clocks])))
    if trace['prediction_turn'] != mask or [r['prediction_turn'] for r in rows] != mask:
        raise ValueError('Decision mask differs')
    if any(r['expected_next_token'] != -1 for r in rows if r['prediction_turn'] < 0):
        raise ValueError('Unused row has token authority')
    return float(np.mean(means)), means


def audit_pair(root, index):
    root = Path(root)
    trace = json.loads((root / f'pair{index}-compact-trace.json').read_text())
    measured = json.loads((root / f'pair{index}-metrics.json').read_text())
    seal = json.loads((root / f'pair{index}-candidate-seal.json').read_text())
    if measured['pair'] != index or seal['phase'] != 'compact-complete-before-native-shadow':
        raise ValueError('Pair identity differs')
    for row in seal['files']:
        if sha(root / row['name']) != row['sha256']:
            raise ValueError('Pre-shadow candidate seal differs')
    n, vocab = trace['count'], 151936
    if any((root / name).stat().st_size != n*vocab*4 for name in ('native-logits.tmp', 'candidate-logits.tmp')):
        raise ValueError('Full stream shape differs')
    if sha(root / 'candidate-logits.tmp') != seal['candidate_stream_sha256']:
        raise ValueError('Candidate changed after shadow started')
    streams = [np.memmap(root / name, dtype='<f4', mode='r', shape=(n, vocab)) for name in ('native-logits.tmp', 'candidate-logits.tmp')]
    maxima = {k: 0. for k in ('kl_nats', 'relative_logit_l2', 'native_top_logp_delta_nats', 'native_entropy_nats')}
    if len(measured['rows']) != n:
        raise ValueError('Metric rows differ')
    for t, row in enumerate(measured['rows']):
        a, b = [x[t] for x in streams]
        for label, values in [('native', a), ('candidate', b)]:
            if hashlib.sha256(values.tobytes()).hexdigest() != row[label+'_logits_sha256']:
                raise ValueError('Full stream row hash differs')
        actual = distribution(a, b)
        for key, value in actual.items():
            if key.endswith('_top'):
                if value != row[key]:
                    raise ValueError('Distribution argmax differs')
            else:
                maxima[key] = max(maxima[key], abs(value-row[key]))
                if abs(value-row[key]) > 1e-10*max(1., abs(value), abs(row[key])):
                    raise ValueError('C/Python full-vocabulary metric differs')
    mean, turns = future_scores(measured['rows'], trace)
    if abs(mean-measured['future_mean_kl_nats']) > 1e-10 or not np.allclose(turns, measured['per_turn_kl_nats'], rtol=1e-10, atol=1e-12):
        raise ValueError('Future aggregation differs')
    result = {'schema': 'tcm-own-history-live-audit-v1', 'pair': index, 'rows': n, 'future_mean_kl_nats': mean, 'per_turn_kl_nats': turns, 'maximum_c_python_discrepancy': maxima, 'native_stream_sha256': sha(root / 'native-logits.tmp'), 'candidate_stream_sha256': sha(root / 'candidate-logits.tmp'), 'trace_sha256': sha(root / f'pair{index}-compact-trace.json'), 'candidate_seal_sha256': sha(root / f'pair{index}-candidate-seal.json'), 'scope': 'Contemporary all-clock independent calculations. Full streams transient; four decision vectors/evaluator remain.'}
    (root / f'pair{index}-audit.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result
