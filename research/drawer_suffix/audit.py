"""Independent saved-only consumer audit; no model execution or primary-file writes."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
DOMAINS = HERE.parents[2]
RUN = DOMAINS / 'artifacts/qwen-relational-reader-cut-20261106/runs/job-935667de6e2f'
PRIOR = DOMAINS / 'artifacts/qwen-query-horizon-expanded-context-20261106/runs/job-54fcb8fc75e3'
V = 151936


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4*1024**2), b''):
            h.update(data)
    return h.hexdigest()


def floats(path, count):
    a = np.fromfile(path, dtype='<f4')
    if a.size != count or not np.isfinite(a).all():
        raise ValueError('Invalid operand: '+str(path))
    return a


def distribution(native, candidate):
    a, b = native.astype(float), candidate.astype(float)
    def logp(x):
        y = x-x.max()
        return y-math.log(math.fsum(np.exp(y)))
    ln, lc = logp(a), logp(b)
    p = np.exp(ln)
    return {'native_kl_nats': math.fsum(p*(ln-lc)), 'native_logit_relative_l2': float(np.linalg.norm(b-a)/np.linalg.norm(a)), 'Willow_first_token_probability': math.exp(lc[9945]), 'Cobalt_first_token_probability': math.exp(lc[34]), 'Willow_minus_Cobalt_first_token_logit': float(b[9945]-b[34])}


def main():
    r, sub, job = [read(RUN / name) for name in ('report.json', 'submission.json', 'job.json')]
    raw, payload = RUN / 'raw', RUN / 'payload'
    if job['job_id'] != 'job-935667de6e2f' or sub['job_id'] != job['job_id'] or job['state'] != 'succeeded':
        raise ValueError('Job identity/state differs')
    if sha(RUN / 'logs.txt') != job['result']['artifacts']['job_log']['sha256']:
        raise ValueError('Scheduler log seal differs')
    logged = []
    with (RUN / 'logs.txt').open('rb') as f:
        for line in f:
            if line.startswith(b'COMPOSITION_RESULT_GZIP_B64='):
                logged.append(json.loads(gzip.decompress(base64.b64decode(line.split(b'=', 1)[1]))))
    if logged != [r] or read(raw / 'report.json') != r or read(raw / 'driver.json') != r['driver']:
        raise ValueError('Report/log/raw join differs')
    for name, digest in sub['config']['sources'].items():
        if sha(payload / name) != digest:
            raise ValueError('Executed source differs: '+name)
    if sha(payload / 'source-manifest.json') != sub['config']['source_manifest_sha256']:
        raise ValueError('Source manifest differs')
    for name, record in r['raw_files'].items():
        if sha(raw / name) != record['sha256'] or (raw / name).stat().st_size != record['bytes']:
            raise ValueError('Raw differs: '+name)
    wrapper = r['saturn_receipt']
    receipt = dict(wrapper['receipt'])
    fingerprint = receipt.pop('fingerprint')
    if hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() != fingerprint or receipt['parent'] != wrapper['before'] or receipt['result'] != wrapper['after']:
        raise ValueError('Whole invocation receipt differs')
    d, inputs = r['driver'], r['inputs']
    if d['model_digest'] != r['model']['content_sha256'] or d['model_identity'] != r['model']['identity'] or d['capacity'] != 2048 or d['layer'] != 17 or d['memory_bytes'] != 98464:
        raise ValueError('Model/profile differs')
    if sha(PRIOR / 'report.json') != inputs['predecessor_report_sha256']:
        raise ValueError('Predecessor differs')
    for item in inputs['lineage']:
        source = PRIOR / 'raw' / item['source']
        if sha(source) != item['source_sha256'] or sha(payload / item['file']) != item['sha256']:
            raise ValueError('Operand lineage differs')
    reader_spec = importlib.util.spec_from_file_location('drawer_reader', PRIOR / 'payload/reader.py')
    reader = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(reader)
    cases = {c['task']: c for c in inputs['cases']}
    records, logits, rows, controls, reconstructions = {}, {}, [], [], []
    token_choices = prefix_advances = 0
    for parent in d['parents']:
        task, history = parent['task'], parent['history']
        case, branches = cases[task], parent['branches']
        n = case['context_tokens']
        if parent['prefix_advances'] != n-1 or [b['arm'] for b in branches] != list(range(9)):
            raise ValueError('Parent geometry differs')
        prefix_advances += n-1
        if len({b['parent_valid_kv_sha256'] for b in branches}) != 1 or len({b['parent_memory_sha256'] for b in branches}) != 1:
            raise ValueError('Parent cache/memory differs between branches')
        qs = [floats(raw / f'case{task}-{label}-q.f32', 1536).reshape(12, 128) for label in ('native', 'compact')]
        qs.append(floats(raw / f'case{task}-earlier-query.f32', 1536).reshape(12, 128))
        ks = [floats(raw / f'case{task}-{label}-k.f32', n*256).reshape(n, 2, 128) for label in ('native', 'compact')]
        vs = [floats(raw / f'case{task}-{label}-v.f32', n*256).reshape(n, 2, 128) for label in ('native', 'compact')]
        if vs[0].tobytes() != vs[1].tobytes():
            raise ValueError('Native values differ')
        memory = reader.Memory()
        for key, value in zip(ks[1], vs[0]):
            memory.append(key, value)
        native_reference = floats(raw / f'case{task}-native-reference.f32', V)
        for branch in branches:
            name, arm = branch['name'], branch['arm']
            if branch['parent_lineage'] != parent['parent_lineage'] or branch['parent_position'] != n-1 or branch['interventions'] != 1:
                raise ValueError('Intervention/lineage differs')
            tops = np.fromfile(raw / (name+'-tops.i32'), dtype='<i4')
            if tops.size != branch['count']-(n-1):
                raise ValueError('Pending stream differs')
            tokens = list(case['prompt_ids'])
            for turn in branch['turns']:
                j = turn['turn']
                if j:
                    skip = int(tokens[-1] == case['followup_ids'][0])
                    if turn['start'] != len(tokens) or turn['template_skip'] != skip or turn['ingested'] != len(case['followup_ids'])-skip:
                        raise ValueError('Followup authority differs')
                    tokens.extend(case['followup_ids'][skip:])
                elif turn['start'] != 0 or turn['ingested'] != n or turn['template_skip'] != 0:
                    raise ValueError('Prompt authority differs')
                if turn['prefill']['position'] != len(tokens):
                    raise ValueError('Prefill clock differs')
                answer = turn['answer_ids']
                eos = (151645, 151643)
                if not answer or len(answer) > turn['budget'] or any(t in eos for t in answer[:-1]) or turn['budget'] != (16 if j == 0 else 32):
                    raise ValueError('Answer/EOS geometry differs')
                for token in answer:
                    if token != int(tops[len(tokens)-1-(n-1)]):
                        raise ValueError('Own generated token differs')
                    tokens.append(token)
                    token_choices += 1
                if turn['final']['position'] != len(tokens) or turn['stopped'] != (answer[-1] in eos) or turn['truncated'] == turn['stopped'] or (not turn['stopped'] and len(answer) != turn['budget']):
                    raise ValueError('Answer completion differs')
                for stage in ('prefill', 'final'):
                    z = floats(raw / f'{name}-t{j}-{stage}-logits.f32', V)
                    record = floats(raw / f'{name}-t{j}-{stage}.record', 6784)
                    if int(np.argmax(z)) != turn[stage]['top'] or int(tops[turn[stage]['position']-1-(n-1)]) != turn[stage]['top']:
                        raise ValueError('Boundary argmax differs')
                    logits[(task, history, arm, j, stage)] = z
                    records[(task, history, arm, j, stage)] = record
            if tokens != branch['consumed_ids'] or len(tokens) != branch['count'] or branch['physical_advances'] != branch['count']-(n-1):
                raise ValueError('Consumed history differs')
            if branch['linear_calls'] != branch['physical_advances']*(198 if history else 197) or branch['linear_macs'] != branch['physical_advances']*(1541242880 if history else 1543569408):
                raise ValueError('Branch linear ledger differs')
            natural = floats(raw / (name+'-natural.f32'), 1536)
            consumed = floats(raw / (name+'-consumed.f32'), 1536)
            if consumed.tobytes() != records[(task, history, arm, 0, 'prefill')][3584:5120].tobytes():
                raise ValueError('Consumer attention observer differs')
            if arm <= 5 or arm == 8:
                qi, ri = (2, 0) if arm == 8 else divmod(arm, 3)
                predicted = (memory.read(qs[qi]) if ri == 2 else reader.full_read(qs[qi], ks[ri], vs[0])).reshape(-1)
                error = float(np.linalg.norm(consumed.astype(float)-predicted)/max(float(np.linalg.norm(predicted)), 1e-30))
                if ri == 2 and consumed.tobytes() != predicted.astype('<f4').tobytes():
                    raise ValueError('Exact bounded reader replay differs')
                reconstructions.append({'task': task, 'history': history, 'arm': arm, 'native_full_vs_cpu64_relative': error if ri != 2 else None, 'bounded_read_exact': ri == 2})
            z = logits[(task, history, arm, 0, 'prefill')]
            text = r['texts'][name]
            rows.append({'name': name, 'task': task, 'context_tokens': n, 'history': history, 'arm': arm, 'query': ('native' if arm < 3 else 'compiled') if arm <= 5 else ('untouched' if arm == 6 else 'sham' if arm == 7 else 'earlier-compiled'), 'reader': ('native-full', 'compiled-full', 'compiled-bounded')[arm % 3] if arm <= 5 else 'control', 'answer': text[0], 'correct_initial_binding': text[0].strip().strip('.!').lower() == 'willow', 'followup': text[1], 'answer_lengths': [len(t['answer_ids']) for t in branch['turns']], 'truncated': [t['truncated'] for t in branch['turns']], **distribution(native_reference, z)})
        baseline = branches[6]
        reference_name = f'case{task}-'+('compact' if history else 'native')+'-reference.f32'
        if logits[(task, history, 6, 0, 'prefill')].tobytes() != (raw / reference_name).read_bytes():
            raise ValueError('Historical baseline logits differ')
        for other_arm in (7, 5 if history else 0):
            other = branches[other_arm]
            if baseline['consumed_ids'] != other['consumed_ids'] or baseline['turns'] != other['turns']:
                raise ValueError('Exact replay tokens/caches differ')
            for turn in (0, 1):
                for stage in ('prefill', 'final'):
                    for suffix in ('-logits.f32', '.record')+ (('-memory.bin',) if history else ()):
                        if (raw / f'{baseline["name"]}-t{turn}-{stage}{suffix}').read_bytes() != (raw / f'{other["name"]}-t{turn}-{stage}{suffix}').read_bytes():
                            raise ValueError('Sham/natural endpoint differs')
            controls.append({'task': task, 'history': history, 'baseline': 6, 'other_arm': other_arm, 'all_boundaries_tokens_and_valid_caches_exact': True})
        if history:
            for branch in branches:
                if (raw / f'{branch["name"]}-t0-prefill-memory.bin').read_bytes() != memory.bytes():
                    raise ValueError('Current cut changed compact memory')
    history_comparisons, factorials = [], []
    for task in (4, 5):
        for arm in (*range(6), 8):
            a, b = [records[(task, h, arm, 0, 'prefill')] for h in (0, 1)]
            if any(a[start:end].tobytes() != b[start:end].tobytes() for start, end in ((0, 1536), (3584, 5120), (5120, 6656))):
                raise ValueError('Fixed-current norm/attention/postMLP state differs')
            za, zb = [logits[(task, h, arm, 0, 'prefill')].astype(float) for h in (0, 1)]
            history_comparisons.append({'task': task, 'arm': arm, 'norm_attention_and_post_MLP_residual_byte_exact': True, 'later_history_margin_shift': float((zb[9945]-zb[34])-(za[9945]-za[34])), **distribution(za, zb)})
        for h in (0, 1):
            z = [logits[(task, h, a, 0, 'prefill')].astype(float) for a in range(6)]
            margin = [float(x[9945]-x[34]) for x in z]
            q = z[3]-z[0]
            k = z[4]-z[3]
            aggregation = z[5]-z[4]
            total = z[5]-z[0]
            interactions = [z[4]-z[3]-z[1]+z[0], z[5]-z[4]-z[2]+z[1]]
            factorials.append({'task': task, 'history': h, 'ordered_query_margin': margin[3]-margin[0], 'ordered_key_margin': margin[4]-margin[3], 'ordered_aggregation_margin': margin[5]-margin[4], 'joint_margin': margin[5]-margin[0], 'ordered_vector_identity_max_abs': float(np.max(np.abs(total-q-k-aggregation))), 'Q_key_margin_interaction': float(interactions[0][9945]-interactions[0][34]), 'Q_aggregation_margin_interaction': float(interactions[1][9945]-interactions[1][34])})
    branches = [b for p in d['parents'] for b in p['branches']]
    linear_calls = sum(p['prefix_advances']*(198 if p['history'] else 197) for p in d['parents'])+sum(b['linear_calls'] for b in branches)
    linear_macs = sum(p['prefix_advances']*(1541242880 if p['history'] else 1543569408) for p in d['parents'])+sum(b['linear_macs'] for b in branches)
    if d['physical_advances'] != prefix_advances+sum(b['physical_advances'] for b in branches) or d['all_linear_calls'] != linear_calls or d['all_linear_macs'] != linear_macs:
        raise ValueError('Global work ledger differs')
    result = {'schema': 'drawer-suffix-independent-audit-v1', 'report_sha256': sha(RUN / 'report.json'), 'primary_run': str(RUN), 'source_files_verified': len(sub['config']['sources']), 'raw_files_verified': len(r['raw_files']), 'scheduler_log_seal': True, 'whole_invocation_receipt': True, 'models_evaluated_by_this_audit': 0, 'generated_token_decisions_verified': token_choices, 'controls': controls, 'reader_reconstructions': reconstructions, 'results': rows, 'history_comparisons': history_comparisons, 'factorials': factorials, 'requested_subset': [row for row in rows if row['task'] == 4 and row['history'] == 1 and row['arm'] < 6], 'cost': {'broader_run_physical_advances': d['physical_advances'], 'prefix_advances': prefix_advances, 'broader_run_branches': len(branches), 'nominal_model_linear_calls': linear_calls, 'nominal_model_linear_macs': linear_macs, 'worker_seconds': r['worker_seconds'], 'runner_seconds': job['result']['elapsed_s'], 'peak_rss_mb': job['result']['peak_rss_mb'], 'peak_vram_mb': job['result']['peak_vram_mb'], 'diagnostic_attention_and_operand_acquisition_in_linear_ledger': False}, 'mechanics_status': 'verified', 'instrument_status': 'verified-with-paid-operands-and-known-case-scope', 'terminal_status': 'not-requested', 'scope': 'Saved-only independent audit of reused concurrent job. Primary question is256-token compact-parent sixbranch subset; all completed parent/length controls retained. One current read, original future policy, no new model calls or fitted program. First-token contrast is not complete-label likelihood; later cross-parent futures also change recurring policy.'}
    output = HERE / 'audit-v1'
    output.mkdir(exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'report_sha256': result['report_sha256'], 'sources': result['source_files_verified'], 'raw': result['raw_files_verified'], 'token_decisions': token_choices, 'controls': len(controls), 'cost': result['cost'], 'requested_subset': result['requested_subset']}))


if __name__ == '__main__':
    main()
