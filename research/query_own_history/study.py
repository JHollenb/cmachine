"""Saved-only occupancy/support arithmetic and Q/key/aggregation diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from analysis import arrays
from reader import Memory, full_read, rel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study(root):
    root = Path(root)
    report = json.loads((root / 'report.json').read_text())
    if json.loads((root / 'verification.json').read_text())['report_sha256'] != sha(root / 'report.json'):
        raise ValueError('Verified report differs')
    sources, read_rows = [], []
    for c in range(3):
        per_turn = []
        for j in range(2):
            histories = []
            for i in (c, c+3):
                trace = json.loads((root / 'raw' / f'pair{i}-compact-trace.json').read_text())
                rows = json.loads((root / 'raw' / f'pair{i}-metrics.json').read_text())['rows']
                clocks = trace['turns'][j]['prediction_clocks']
                histories.append([rows[t]['kl_nats'] for t in clocks])
            a, b = histories
            common = min(len(a), len(b))
            paired_window = float(np.mean(b[:common])-np.mean(a[:common]))
            support = float((np.mean(b)-np.mean(b[:common]))-(np.mean(a)-np.mean(a[:common])))
            total = float(np.mean(b)-np.mean(a))
            if abs(total-paired_window-support) > 1e-14:
                raise ValueError('Finite support decomposition differs')
            per_turn.append({'turn': j, 'native_decisions': len(a), 'own_decisions': len(b), 'common_budget': common, 'native_history_kl_nats': float(np.mean(a)), 'own_history_kl_nats': float(np.mean(b)), 'total_change_nats': total, 'common_budget_summary_change_nats': paired_window, 'unequal_tail_weight_change_nats': support, 'scope': 'Exact mean arithmetic; common decision indices after divergence are not identical histories or paired causal states.'})
        sources.append({'source': c, 'turns': per_turn})
    for i in range(6):
        trace = json.loads((root / 'raw' / f'pair{i}-compact-trace.json').read_text())
        metrics = json.loads((root / 'raw' / f'pair{i}-metrics.json').read_text())['rows']
        n = trace['count']
        nc = arrays(root / 'raw', f'pair{i}-native', n)
        cc = arrays(root / 'raw', f'pair{i}-compact', n)
        mask = {t for turn in trace['turns'] for t in turn['prediction_clocks']}
        memory = Memory()
        for t in range(n):
            memory.append(cc['k'][t], cc['v'][t])
            if t not in mask:
                continue
            # Native values agree on the entire identical text prefix within the pair.
            nn = full_read(nc['q'][t], nc['k'][:t+1], nc['v'][:t+1])
            qn = full_read(cc['q'][t], nc['k'][:t+1], nc['v'][:t+1])
            nk = full_read(nc['q'][t], cc['k'][:t+1], nc['v'][:t+1])
            qk = full_read(cc['q'][t], cc['k'][:t+1], nc['v'][:t+1])
            nb = memory.read(nc['q'][t]).astype(float)
            qb = memory.read(cc['q'][t]).astype(float)
            if not np.array_equal(qb.astype('<f4'), cc['attention'][t]):
                raise ValueError('Actual bounded reader differs')
            q, k, interaction, aggregation, numeric = qn-nn, nk-nn, qk-qn-nk+nn, qb-qk, nn-nc['attention'][t].astype(float)
            joint = qb-nc['attention'][t].astype(float)
            den = max(float(np.linalg.norm(nc['attention'][t])), 1e-30)
            closure = joint-q-k-interaction-aggregation-numeric
            if np.linalg.norm(closure) > 1e-12*max(float(np.linalg.norm(joint)), 1):
                raise ValueError('Reader vector account differs')
            turn = trace['prediction_turn'][t]
            read_rows.append({'pair': i, 'source': i % 3, 'history': 'native-generated' if i < 3 else 'compact-generated', 'step': t, 'turn': turn, 'answer_token_index': trace['turns'][turn]['prediction_clocks'].index(t), 'kl_nats': metrics[t]['kl_nats'], 'native_reconstruction_relative': float(np.linalg.norm(numeric)/den), 'query_only_relative': float(np.linalg.norm(q)/den), 'key_only_relative': float(np.linalg.norm(k)/den), 'query_key_interaction_relative': float(np.linalg.norm(interaction)/den), 'both_full_relative': rel(qk, nc['attention'][t]), 'aggregation_at_native_query_relative': float(np.linalg.norm(nb-nk)/den), 'aggregation_at_compact_query_relative': float(np.linalg.norm(aggregation)/den), 'query_aggregation_interaction_relative': float(np.linalg.norm((qb-qk)-(nb-nk))/den), 'both_bounded_relative': float(np.linalg.norm(joint)/den), 'finite_closure_relative': float(np.linalg.norm(closure)/den)})
    reader_summary = []
    keys = [k for k in read_rows[0] if k.endswith('_relative')]
    for i in range(6):
        for turn in range(2):
            rows = [r for r in read_rows if r['pair'] == i and r['turn'] == turn]
            reader_summary.append({'pair': i, 'source': i % 3, 'turn': turn, 'history': 'native-generated' if i < 3 else 'compact-generated', 'decisions': len(rows), 'means': {k: float(np.mean([r[k] for r in rows])) for k in keys}, 'max_native_reconstruction_relative': max(r['native_reconstruction_relative'] for r in rows)})
    summary = report['analysis']['summary']
    result = {'schema': 'tcm-own-history-saved-study-v1', 'report_sha256': sha(root / 'report.json'), 'relative_own_history_kl_change': summary['compact_history_future_kl_nats']/summary['native_history_future_kl_nats']-1, 'loss_support_decomposition': sources, 'reader_summary': reader_summary, 'reader_rows': read_rows, 'field_qualification': {'native_greedy_prefix_before_decision': 'Cumulative agreement of native and compact generated-token choices, not native on-policy status in the native-generated condition. Historical raw field name retained unchanged.'}, 'scope': 'Saved-only analysis; no model calls/fitting. Reader crossings compute alternative attention equations on recorded same-text operands, without native suffix continuation. Native GPU numerical-profile remainder retained. The aggregation term includes the bounded reader final FP32 rounding; full reads use FP64. Closure is an accounting identity. Component norms do not order logit KL or measure semantic shares.', 'terminal_status': 'not-requested'}
    (root / 'study.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'relative_own_history_kl_change': result['relative_own_history_kl_change'], 'reader_decision_rows': len(read_rows), 'maximum_native_attention_reconstruction_relative': max(r['native_reconstruction_relative'] for r in read_rows), 'max_finite_closure_relative': max(r['finite_closure_relative'] for r in read_rows)}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    study(p.parse_args().root)
