"""Independent offline arithmetic on collected operand-basis arrays; no model calls."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/job-51df6b792d28'
RAW = RUN / 'raw'
r = json.loads((RUN / 'report.json').read_text())
def arr(name):
    p = RAW / name
    b = p.read_bytes()
    assert hashlib.sha256(b).hexdigest() == r['raw_files'][name]['sha256']
    a = np.frombuffer(b, dtype='<f4').astype(np.float64)
    assert np.isfinite(a).all()
    return a
def error(a,b):
    d = a-b
    return {'l2':float(np.linalg.norm(d)), 'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(b)),
            'max_abs':float(np.abs(d).max()),'top_equal':int(np.argmax(a))==int(np.argmax(b))}
local=[]; readouts=[]; paired={}; exact=0
for group, refs in zip(r['driver']['primary_cases'],r['driver']['references']):
    for p, ref in zip(group['arms'],refs['same_ids']):
        assert ref['primary_name']==p['name']
        history=[]
        for s,t in zip(p['steps'],ref['steps']):
            assert s['input_id']==t['input_id']
            history.append(s['input_id'])
            for port in ['raw_receiver','mlp_input']:
                left=s['stem']+'-'+port+'.f32'; right=t['stem']+'-'+port+'.f32'
                assert np.array_equal(arr(left),arr(right))
                assert (RAW/left).read_bytes()==(RAW/right).read_bytes()
            exact+=1
            w=arr(s['stem']+'-mlp_return.f32'); native=arr(s['native_write_file'])
            e=error(w,native)
            x={'case':p['case'],'arm':p['arm'],'step':s['step'],'native_sq':float(native@native),'error_sq':float((w-native)@(w-native)),
               'write_to_native_norm':float(np.linalg.norm(w)/np.linalg.norm(native)),**e}
            local.append(x)
            key=(p['case'],tuple(history))
            paired.setdefault(key,{})[p['arm']]=x
        for clock in ['first','after_answer0','after_followup','final']:
            a=arr(p[clock]['logits']['file']); b=arr(ref[clock]['logits']['file'])
            readouts.append({'case':p['case'],'arm':p['arm'],'clock':clock,**error(a,b)})
def aggregate(rows):
    return {'n':len(rows),'mean_relative_l2':float(np.mean([x['relative_l2'] for x in rows])),
            'pooled_relative_l2':float(np.sqrt(sum(x['error_sq'] for x in rows)/sum(x['native_sq'] for x in rows))),
            'better_than_zero':sum(x['relative_l2']<1 for x in rows),
            'mean_write_to_native_norm':float(np.mean([x['write_to_native_norm'] for x in rows]))}
operands=[]
for group in r['driver']['primary_cases']:
    for p in group['arms']:
        for s in p['steps']:
            stem=s['stem']
            if stem+'-mlp_gate.f32' not in r['raw_files']: continue
            g=arr(stem+'-mlp_gate.f32'); u=arr(stem+'-mlp_up.f32')
            gn=arr(stem+'-native_gate.f32'); un=arr(stem+'-native_up.f32')
            operands.append({'arm':p['arm'],'case':p['case'],'step':s['step'],
                 'gate':error(g,gn),'up':error(u,un)})
summary={'report_sha256':hashlib.sha256((RUN/'report.json').read_bytes()).hexdigest(),
 'same_id_receiver_and_norm_exact_steps':exact,
 'local_by_arm':{arm:aggregate([x for x in local if x['arm']==arm]) for arm in ['source64','random64','b3','zero']},
 'local_by_case_arm':{f'{c}-{arm}':aggregate([x for x in local if x['arm']==arm and x['case']==c]) for c in range(3) for arm in ['source64','random64','b3','zero']},
 'readouts_candidate_vs_same_id':readouts,
 'paired_source_contrasts':{},'selected_operands':operands,'texts':r['texts']}
for arm in ['random64','b3','zero']:
    pairs=[(x['source64'],x[arm]) for x in paired.values() if 'source64'in x and arm in x]
    summary['paired_source_contrasts'][arm]={'same_history_pairs':len(pairs),'source_lower_write_error':sum(a['error_sq']<b['error_sq'] for a,b in pairs),
      'source_pooled_error':aggregate([a for a,b in pairs])['pooled_relative_l2'],
      'other_pooled_error':aggregate([b for a,b in pairs])['pooled_relative_l2']}
(RUN/'advisor-independent-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['texts','readouts_candidate_vs_same_id','selected_operands']},indent=2))
for arm in ['source64','random64','b3']:
    o=[x for x in operands if x['arm']==arm]
    if o: print('OPERANDS',arm,len(o),{k:float(np.mean([x[k]['relative_l2'] for x in o])) for k in ['gate','up']})
for c in range(3):
    for arm in ['source64','random64','b3','zero']:
        rows=[x for x in readouts if x['case']==c and x['arm']==arm]
        print(c,arm,' '.join(f"{x['clock']}={x['relative_l2']:.6f}/{int(x['top_equal'])}" for x in rows))
