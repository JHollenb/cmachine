"""Independent stdlib-only check of immutable composition readouts; no models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def subtract(a, b):
    if len(a) != len(b):
        raise ValueError('Vector extent mismatch')
    return [x-y for x, y in zip(a, b)]


def kl(native, candidate):
    def logp(x):
        maximum = max(x)
        shifted = [v-maximum for v in x]
        partition = math.log(math.fsum(math.exp(v) for v in shifted))
        return [v-partition for v in shifted]
    p, q = logp(native), logp(candidate)
    return math.fsum(math.exp(a)*(a-b) for a, b in zip(p, q))


def analyze(root):
    root = Path(root)
    report = json.loads((root/'report.json').read_text())
    paths = {p['name']: p for p in report['driver']['paths']}
    checked = {}

    def read(name, count):
        path = root/'raw'/name
        data = path.read_bytes()
        record = report['raw_files'][name]
        actual = hashlib.sha256(data).hexdigest()
        if len(data) != 4*count or len(data) != record['bytes'] or actual != record['sha256']:
            raise ValueError('Raw identity/extent mismatch: '+name)
        values = struct.unpack('<'+'f'*count, data)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite raw values: '+name)
        checked[name] = actual
        return values

    rows, attention = [], []
    for task in range(6):
        names = [f'case{task}-replay-{p}' for p in ('00','10','01')]+[f'case{task}-own-11']
        reference = paths[names[0]]
        if any(paths[n]['consumed_ids'] != reference['consumed_ids'] for n in names):
            raise ValueError('Not a matched-history square')
        n = reference['count']
        fields = [read(name+'-attention.f32', n*1536) for name in names]
        for turn in range(2):
            for stage in ('prefill','final'):
                position = reference['turns'][turn][stage]['position']
                if any(paths[name]['turns'][turn][stage]['position'] != position for name in names):
                    raise ValueError('Boundary clock mismatch')
                y = [read(f'{name}-t{turn}-{stage}-logits.f32',151936) for name in names]
                differences = [subtract(x,y[0]) for x in y[1:]]
                interaction = [c-a-b for a,b,c in zip(*differences)]
                denominator = norm(y[0])
                rows.append({'task':task,'turn':turn,'stage':stage,'position':position,
                             'relative_l2':dict(zip(('10','01','11'),[norm(d)/denominator for d in differences])),
                             'native_to_candidate_kl':dict(zip(('10','01','11'),[kl(y[0],x) for x in y[1:]])),
                             'interaction_relative':norm(interaction)/denominator})
                i = position-1
                a = [x[i*1536:(i+1)*1536] for x in fields]
                native = norm(subtract(a[2],a[0]))/norm(a[0])
                cheap = norm(subtract(a[3],a[1]))/norm(a[1])
                attention.append({'task':task,'turn':turn,'stage':stage,
                                  'native_writer_relative':native,'cheap_writer_relative':cheap,
                                  'ratio':cheap/native})
    mean = lambda xs: math.fsum(xs)/len(xs)
    summary = {
        'mean_relative_logit_l2':{p:mean([r['relative_l2'][p] for r in rows]) for p in ('10','01','11')},
        'initial_mean_relative_logit_l2':{p:mean([r['relative_l2'][p] for r in rows if r['turn']==0 and r['stage']=='prefill']) for p in ('10','01','11')},
        'mean_native_to_candidate_kl':{p:mean([r['native_to_candidate_kl'][p] for r in rows]) for p in ('10','01','11')},
        'mean_logit_interaction':mean([r['interaction_relative'] for r in rows]),
        'max_logit_interaction':max(r['interaction_relative'] for r in rows),
        'joint_vs_writer_l2_wins':sum(r['relative_l2']['11']<r['relative_l2']['10'] for r in rows),
        'joint_vs_writer_kl_wins':sum(r['native_to_candidate_kl']['11']<r['native_to_candidate_kl']['10'] for r in rows),
        'attention_native_writer_mean':mean([r['native_writer_relative'] for r in attention]),
        'attention_cheap_writer_mean':mean([r['cheap_writer_relative'] for r in attention]),
        'attention_ratio_mean':mean([r['ratio'] for r in attention]),
        'attention_ratio_min':min(r['ratio'] for r in attention),
        'attention_ratio_max':max(r['ratio'] for r in attention),
    }
    return {'schema':'astra-composition-independent-check-v1',
            'report_sha256':sha(root/'report.json'),'analysis_source_sha256':sha(__file__),
            'method':'Little-endian FP32 decoding, Python double arithmetic and math.fsum; no NumPy or model execution',
            'raw_file_hashes':checked,'matched_boundary_count':len(rows),
            'summary':summary,'logit_rows':rows,'attention_rows':attention,
            'models_evaluated':0,'scope':'Matched recurring policies on joint11 consumed IDs; no new native future or semantic certification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.root/'analysis'/'astra-check.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
