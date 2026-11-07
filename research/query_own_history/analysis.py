"""Independent saved-operand, authority, historical and clock verification."""
import hashlib
import json
from pathlib import Path
import numpy as np
from metrics import distribution, future_scores, sha
from reader import Memory, rel

V = 151936
FIELDS = {'norm': 1536, 'q': 1536, 'k': 256, 'v': 256, 'attention': 1536, 'output': 1536, 'coeff': 128}


def arrays(root, name, n):
    result = {}
    for key, width in FIELDS.items():
        values = np.fromfile(root / f'{name}-{key}.f32', dtype='<f4')
        if values.size != n*width or not np.isfinite(values).all():
            raise ValueError('Operand shape/finite contract differs')
        result[key] = values.reshape(n, 12, 128) if key in ('q', 'attention') else values.reshape(n, 2, 128) if key in ('k', 'v') else values.reshape(n, width)
    result['tops'] = np.fromfile(root / f'{name}-tops.i32', dtype='<i4')
    if result['tops'].shape != (n,):
        raise ValueError('Pending stream shape differs')
    return result


def trace_check(trace, inputs, tops, own):
    history = []
    phases = []
    eos = set(inputs['native_eos_token_ids'])
    for j, turn in enumerate(trace['turns']):
        supplied = inputs['prompt_ids'] if j == 0 else inputs['followup_ids']
        skip = int(bool(history) and history[-1] == supplied[0])
        if turn['start'] != len(history) or turn['template_skip'] != skip:
            raise ValueError('Template clock differs')
        history += supplied[skip:]
        phases += [0 if j == 0 else 2] * (len(supplied)-skip)
        if turn['prefill_end'] != len(history):
            raise ValueError('Prefill clock differs')
        answer = turn['answer_ids']
        if not answer or len(answer) > 32 or any(t in eos for t in answer[:-1]):
            raise ValueError('Answer extent/EOS differs')
        if turn['stopped'] != (answer[-1] in eos) or turn['truncated'] == turn['stopped'] or (not turn['stopped'] and len(answer) != 32):
            raise ValueError('EOS/truncation contract differs')
        for token in answer:
            if own and token != int(tops[len(history)-1]):
                raise ValueError('Autonomous generated-token authority differs')
            history.append(token)
            phases.append(1 if j == 0 else 3)
        if turn['final'] != len(history):
            raise ValueError('Final clock differs')
    if history != trace['consumed_ids'] or phases != trace['input_phases'] or len(history) != trace['count']:
        raise ValueError('Consumed history differs')


def analyze(raw, bank, driver, texts):
    del bank
    raw = Path(raw)
    inputs = json.loads((raw / 'input-manifest.json').read_text())
    lineage = json.loads((raw / 'lineage.json').read_text())
    if driver['schema'] != 'tcm-query-own-history-driver-v1' or driver['capacity'] != 512 or driver['layer'] != 17 or driver['memory_bytes'] != 98464 or not driver['native_sham_logits_and_kv_byte_exact']:
        raise ValueError('Runtime geometry differs')
    if driver['input_manifest_sha256'] != sha(raw / 'input-manifest.json') or driver['query_image_sha256'] != lineage['query_image_sha256']:
        raise ValueError('Input/image binding differs')
    for name, record in lineage['program_files'].items():
        if sha(raw / name) != record['sha256']:
            raise ValueError('Frozen image/bank differs')
    old_traces = json.loads((raw / 'historical-traces.json').read_text())['sources']
    old_metrics = json.loads((raw / 'historical-metrics.json').read_text())['rows']
    old_by_source = [[r for r in old_metrics if r['source'] == c] for c in range(3)]
    pairs, observations, records = [], [], {}
    reads = snapshots = durable = historical_arrays = historical_rows = generated = 0
    for index, pair in enumerate(driver['pairs']):
        source, own = index % 3, index >= 3
        if pair['pair'] != index or pair['source'] != source or pair['history'] != ('compact-generated' if own else 'native-generated') or not pair['candidate_sealed_before_native']:
            raise ValueError('History/evaluator geometry differs')
        c_name, n_name = f'pair{index}-compact', f'pair{index}-native'
        trace = json.loads((raw / f'{c_name}-trace.json').read_text())
        n = trace['count']
        cs, ns = arrays(raw, c_name, n), arrays(raw, n_name, n)
        trace_check(trace, inputs['sources'][source], cs['tops'], own)
        if not own:
            trace_check(trace, inputs['sources'][source], ns['tops'], True)
            if trace != {k: v for k, v in old_traces[source].items() if k != 'source'}:
                raise ValueError('Archived source trace differs')
        if not np.array_equal(cs['norm'], ns['norm']) or not np.array_equal(cs['v'], ns['v']):
            raise ValueError('Same-history lower receiver/native V differs')
        metric = json.loads((raw / f'pair{index}-metrics.json').read_text())
        audit = json.loads((raw / f'pair{index}-audit.json').read_text())
        seal = json.loads((raw / f'pair{index}-candidate-seal.json').read_text())
        expected_seal = {p.name for p in raw.glob(c_name+'-*')}
        if {r['name'] for r in seal['files']} != expected_seal or len(seal['files']) != len(expected_seal) or audit['candidate_seal_sha256'] != sha(raw / f'pair{index}-candidate-seal.json'):
            raise ValueError('Candidate seal membership differs')
        for record in seal['files']:
            if sha(raw / record['name']) != record['sha256']:
                raise ValueError('Candidate changed after shadow')
        mean, turns = future_scores(metric['rows'], trace)
        if audit['rows'] != n or audit['trace_sha256'] != sha(raw / f'{c_name}-trace.json') or abs(mean-audit['future_mean_kl_nats']) > 1e-10 or abs(mean-metric['future_mean_kl_nats']) > 1e-10:
            raise ValueError('Audit/aggregation differs')
        memory = Memory()
        boundary_positions = {t[stage]: (j, stage) for j, t in enumerate(trace['turns']) for stage in ('prefill_end', 'final')}
        keep = {t['prediction_clocks'][j] for t in trace['turns'] for j in (0, -1)}
        first = None
        prefix = True
        for t, row in enumerate(metric['rows']):
            if row['native_top'] != int(ns['tops'][t]) or row['candidate_top'] != int(cs['tops'][t]) or row['native_greedy_prefix_before_decision'] != prefix:
                raise ValueError('Distribution/pending authority differs')
            turn = row['prediction_turn']
            if turn >= 0:
                generated += int(own)
                if row['native_top'] != row['candidate_top']:
                    if first is None:
                        first = {'turn': turn, 'answer_token_index': trace['turns'][turn]['prediction_clocks'].index(t), 'clock': t, 'native_token': row['native_top'], 'compact_token': row['candidate_top']}
                    prefix = False
            memory.append(cs['k'][t], cs['v'][t])
            if not np.array_equal(memory.read(cs['q'][t]), cs['attention'][t]):
                raise ValueError('Compact memory law replay differs')
            reads += 1
            if t+1 in boundary_positions:
                j, stage = boundary_positions[t+1]
                label = 'prefill' if stage == 'prefill_end' else 'final'
                if (raw / f'{c_name}-t{j}-{label}-memory.bin').read_bytes() != memory.bytes():
                    raise ValueError('Memory snapshot replay differs')
                snapshots += 1
            if not own:
                old = old_by_source[source][t]
                for key in ('native_logits_sha256', 'candidate_logits_sha256', 'native_top', 'candidate_top', 'prediction_turn', 'expected_next_token'):
                    if row[key] != old[key]:
                        raise ValueError('Archived full-clock identity differs')
                if hashlib.sha256(cs['norm'][t].tobytes()).hexdigest() != old['receiver_sha256']:
                    raise ValueError('Archived receiver hash differs')
                historical_rows += 1
            if t in keep:
                a = np.fromfile(raw / f'{n_name}-t{t}-logits.f32', dtype='<f4')
                b = np.fromfile(raw / f'{c_name}-t{t}-logits.f32', dtype='<f4')
                if a.shape != (V,) or b.shape != (V,) or sha(raw / f'{n_name}-t{t}-logits.f32') != row['native_logits_sha256'] or sha(raw / f'{c_name}-t{t}-logits.f32') != row['candidate_logits_sha256']:
                    raise ValueError('Durable decision vector differs')
                for key, value in distribution(a, b).items():
                    if abs(value-row[key]) > 1e-10*max(1., abs(value), abs(row[key])):
                        raise ValueError('Durable probability replay differs')
                durable += 1
            observations.append({'kind': 'history-conditioned-distribution', 'pair': index, 'source': source, 'history': pair['history'], 'step': t, 'prediction_turn': turn, 'kl_nats': row['kl_nats'], 'relative_logit_l2': row['relative_logit_l2'], 'native_entropy_nats': row['native_entropy_nats'], 'same_text_history': True})
        for evaluator, name in zip(pair['evaluators'], (c_name, n_name)):
            compact = name == c_name
            if evaluator['name'] != name or evaluator['count'] != n or evaluator['compact'] != compact or evaluator['linear_calls'] != (198 if compact else 197)*n or evaluator['linear_macs'] != (1543569408-3145728+819200 if compact else 1543569408)*n:
                raise ValueError('Evaluator work ledger differs')
            if compact and (evaluator['compact_reads'] != n or not evaluator['selected_full_kv_untouched'] or evaluator['serving_transfer_bytes'] != 14336*n):
                raise ValueError('Compact boundary ledger differs')
        if not own:
            for suffix, description in lineage['historical_arrays'][str(source)].items():
                role, field = suffix.split(':')
                name = c_name if role == 'compact' else n_name
                if sha(raw / f'{name}-{field}') != description['sha256']:
                    raise ValueError('Historical operand array differs')
                historical_arrays += 1
        records[index] = (trace, cs, ns)
        windows = []
        for j, turn in enumerate(trace['turns']):
            clocks = turn['prediction_clocks']
            windows.append({'turn': j, 'first8_count': len(clocks[:8]), 'first8_kl_nats': float(np.mean([metric['rows'][t]['kl_nats'] for t in clocks[:8]])), 'after8_count': len(clocks[8:]), 'after8_kl_nats': float(np.mean([metric['rows'][t]['kl_nats'] for t in clocks[8:]])) if clocks[8:] else None})
        pairs.append({'pair': index, 'source': source, 'history': pair['history'], 'advances_per_evaluator': n, 'future_mean_kl_nats': mean, 'per_turn_kl_nats': turns, 'decision_windows': windows, 'answer_lengths': [len(t['answer_ids']) for t in trace['turns']], 'truncated': [t['truncated'] for t in trace['turns']], 'first_evaluator_greedy_disagreement': first, 'texts': texts[c_name], 'maximum_c_python_discrepancy': audit['maximum_c_python_discrepancy']})
    contrasts = []
    for c in range(3):
        nt, nc, nn = records[c]
        ot, oc, on = records[c+3]
        common = 0
        for a, b in zip(nt['consumed_ids'], ot['consumed_ids']):
            if a != b:
                break
            common += 1
        for native, own_arrays in ((nc, oc), (nn, on)):
            for key in (*FIELDS, 'tops'):
                if not np.array_equal(native[key][:common], own_arrays[key][:common]):
                    raise ValueError('Cross-history shared-prefix state differs')
        windows = []
        for turn in range(2):
            nrows = json.loads((raw / f'pair{c}-metrics.json').read_text())['rows']
            orows = json.loads((raw / f'pair{c+3}-metrics.json').read_text())['rows']
            ncl, ocl = nt['turns'][turn]['prediction_clocks'], ot['turns'][turn]['prediction_clocks']
            count = min(len(ncl), len(ocl))
            windows.append({'turn': turn, 'native_count': len(ncl), 'compact_count': len(ocl), 'common_decision_count': count, 'native_history_common_window_kl_nats': float(np.mean([nrows[t]['kl_nats'] for t in ncl[:count]])), 'compact_history_common_window_kl_nats': float(np.mean([orows[t]['kl_nats'] for t in ocl[:count]])), 'same_states_after_divergence': False})
        contrasts.append({'source': c, 'common_consumed_prefix': common, 'native_history_future_kl_nats': pairs[c]['future_mean_kl_nats'], 'compact_history_future_kl_nats': pairs[c+3]['future_mean_kl_nats'], 'change_nats': pairs[c+3]['future_mean_kl_nats']-pairs[c]['future_mean_kl_nats'], 'common_budget_windows': windows, 'candidate_answers_identical': [a['answer_ids'] == b['answer_ids'] for a, b in zip(nt['turns'], ot['turns'])]})
    physical = sum(e['count'] for p in driver['pairs'] for e in p['evaluators'])+32
    if driver['all_linear_calls'] != sum(e['linear_calls'] for p in driver['pairs'] for e in p['evaluators'])+197*32 or driver['all_linear_macs'] != sum(e['linear_macs'] for p in driver['pairs'] for e in p['evaluators'])+1543569408*32:
        raise ValueError('Whole invocation ledger differs')
    means = [float(np.mean([p['future_mean_kl_nats'] for p in pairs[start:start+3]])) for start in (0, 3)]
    summary = {'physical_trajectories': 12, 'all_physical_advances': physical, 'sham_advances': 32, 'native_history_evaluator_advances': 2*sum(p['advances_per_evaluator'] for p in pairs[:3]), 'compact_history_evaluator_advances': 2*sum(p['advances_per_evaluator'] for p in pairs[3:]), 'compact_autonomous_answer_tokens': generated, 'compact_reads_exact': reads, 'compact_snapshots_byte_exact': snapshots, 'full_clock_dual_calculation_rows': sum(p['advances_per_evaluator'] for p in pairs), 'durable_metric_pairs': durable, 'durable_full_vectors': durable*2, 'historical_arrays_exact': historical_arrays, 'historical_full_clock_pairs_exact': historical_rows, 'native_history_future_kl_nats': means[0], 'compact_history_future_kl_nats': means[1], 'own_minus_native_history_kl_nats': means[1]-means[0], 'query_image_unchanged': True, 'selected_macs_per_advance': 819200, 'memory_bytes': 98464, 'complete_model_emulation': False, 'measured_speedup': None}
    return {'summary': summary, 'pairs': pairs, 'history_contrasts': contrasts, 'observations': observations, 'scope': 'Two history measures with independently built native/compact states on identical text within each pair; no pure memory attribution or fitting.', 'terminal_status': 'not-requested'}
