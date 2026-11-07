"""Prepare native-Q responses in actual compact memory; no fit or model execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'query_own_history'
SPEC = importlib.util.spec_from_file_location('retained_reader', SOURCE / 'reader.py')
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)
REPORT_SHA = 'd043d1490e7bf10bc8d688e20e178a07ad36be221518690a78d1ff11b974ba39'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, output):
    root, output = Path(root), Path(output)
    report = json.loads((root / 'report.json').read_text())
    verification = json.loads((root / 'verification.json').read_text())
    if sha(root / 'report.json') != REPORT_SHA or verification['report_sha256'] != REPORT_SHA:
        raise ValueError('Source report identity differs')
    if output.exists():
        raise ValueError('Refusing to overwrite target evidence')
    used = {}

    def raw(name):
        path = root / 'raw' / name
        record = report['raw_files'][name]
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Source custody differs: {name}')
        used[name] = record
        return path

    targets, baselines, native_full, coeffs, rows = [], [], [], [], []
    total_clocks = 0
    for pair in (3, 4, 5):
        trace = json.loads(raw(f'pair{pair}-compact-trace.json').read_text())
        n = trace['count']
        total_clocks += n

        def array(policy, field, width):
            a = np.fromfile(raw(f'pair{pair}-{policy}-{field}.f32'), dtype='<f4')
            if a.size != n * width or not np.isfinite(a).all():
                raise ValueError('Operand shape/finite contract differs')
            return a.reshape(n, width)

        k = array('compact', 'k', 256).reshape(n, 2, 128)
        v = array('compact', 'v', 256).reshape(n, 2, 128)
        q = array('compact', 'q', 1536).reshape(n, 12, 128)
        nq = array('native', 'q', 1536).reshape(n, 12, 128)
        attention = array('compact', 'attention', 1536)
        native_attention = array('native', 'attention', 1536)
        coeff = array('compact', 'coeff', 128)
        if not np.array_equal(v.reshape(n, 256), array('native', 'v', 256)):
            raise ValueError('Native V differs on same token history')
        if not np.array_equal(array('compact', 'norm', 1536), array('native', 'norm', 1536)):
            raise ValueError('Same-text lower receiver differs')
        decision = {}
        for turn_index, turn in enumerate(trace['turns']):
            clocks = turn['prediction_clocks']
            if len(clocks) != len(turn['answer_ids']) or len(clocks) == 0:
                raise ValueError('Decision clock contract differs')
            for index, clock in enumerate(clocks):
                if clock in decision:
                    raise ValueError('Overlapping decision clocks')
                decision[clock] = (turn_index, index, 1 / (3 * 2 * len(clocks)))
        memory = READER.Memory()
        for clock in range(n):
            memory.append(k[clock], v[clock])
            # Replay every actual read, including prefill and unused final outputs.
            if not np.array_equal(memory.read(q[clock]).reshape(-1), attention[clock]):
                raise ValueError('Actual bounded reader does not replay exactly')
            if clock not in decision:
                continue
            turn, index, weight = decision[clock]
            teacher = memory.read(nq[clock]).reshape(-1)
            baseline, full = attention[clock], native_attention[clock]
            e = baseline.astype(float) - teacher.astype(float)
            d = teacher.astype(float) - full.astype(float)
            same = float(e @ e)
            cross = float(2 * e @ d)
            displacement = float(d @ d)
            original = float((e + d) @ (e + d))
            remainder = original - same - cross - displacement
            if abs(remainder) > 1e-12 * max(1., original, same, abs(cross), displacement):
                raise ValueError('Loss account does not close')
            row = {'row': len(rows), 'pair': pair, 'source': pair - 3,
                   'turn': turn, 'clock': clock, 'answer_decision': index,
                   'selected_token_id': trace['turns'][turn]['answer_ids'][index],
                   'weight': weight, 'memory_sha256': hashlib.sha256(memory.bytes()).hexdigest(),
                   'same_memory_squared_response': same, 'cross_term': cross,
                   'teacher_displacement_squared': displacement,
                   'full_native_squared_response': original, 'closure_remainder': remainder}
            targets.append(teacher)
            baselines.append(baseline)
            native_full.append(full)
            coeffs.append(coeff[clock])
            rows.append(row)
    if len(rows) != 162 or total_clocks != 420 or abs(sum(r['weight'] for r in rows)-1) > 1e-14:
        raise ValueError('Source support/weight differs')
    arrays = {'native-q-compact-memory.f32': targets,
              'baseline-compact-response.f32': baselines,
              'native-full-response.f32': native_full,
              'receiver-coeff.f32': coeffs}
    for values in arrays.values():
        if not np.isfinite(np.asarray(values)).all():
            raise ValueError('Nonfinite target')
    output.mkdir(parents=True)
    files = {}
    for name, values in arrays.items():
        a = np.asarray(values, dtype='<f4')
        path = output / name
        path.write_bytes(a.tobytes())
        files[name] = {'shape': list(a.shape), 'dtype': '<f4', 'bytes': path.stat().st_size,
                       'sha256': sha(path)}
    aggregate = {key: sum(r['weight'] * r[key] for r in rows) for key in (
        'same_memory_squared_response', 'cross_term', 'teacher_displacement_squared',
        'full_native_squared_response', 'closure_remainder')}
    result = {'schema': 'tcm-own-state-response-targets-v1', 'source_report_sha256': REPORT_SHA,
              'source_run': str(root.resolve()), 'builder_sha256': sha(Path(__file__)),
              'reader_sha256': sha(SOURCE / 'reader.py'),
              'source_verification_sha256': sha(root / 'verification.json'),
              'source_files': used, 'files': files, 'rows': rows,
              'summary': {'source_histories': 3, 'replayed_actual_reads': total_clocks,
                          'answer_decisions': len(rows), 'weight_sum': sum(r['weight'] for r in rows),
                          'weighted_loss_account': aggregate},
              'teacher': 'Paid source-native Q evaluated through the exact actual compact memory.',
              'scope': 'Source-only cached preparation. No drawer operands, fitting, optimizer, new query image, native suffix continuation or model calls. Counterfactual teachers are not demonstrated downstream repairs.',
              'new_model_calls': 0, 'mechanics_status': 'verified',
              'instrument_status': 'actual bounded reads replay byte exactly',
              'trend_status': 'target hypothesis prepared; no candidate evaluated',
              'terminal_status': 'not-requested'}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE / 'runs/job-47cc27324445')
    parser.add_argument('--output', type=Path, default=HERE / 'source-targets-v1')
    args = parser.parse_args()
    build(args.source, args.output)
