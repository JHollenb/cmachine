"""Small, saved-only scientific contrasts; no new model execution or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

LABELS=('00','10','01','11')
FIELDS={'mlp-write':1536,'receiver':1536,'q':1536,'k':256,'v':256,
        'attention':1536,'output':1536,'kv18':512}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(raw,name,shape):
    a=np.fromfile(raw/name,dtype='<f4')
    if a.size!=int(np.prod(shape)) or not np.isfinite(a).all(): raise ValueError(name)
    return a.reshape(shape).astype(np.float64)
def softmax(x):
    logp=x-np.max(x); logp-=np.log(np.exp(logp).sum())
    return np.exp(logp),logp
def fidelity(candidate,native):
    p,lp=softmax(native); q,lq=softmax(candidate)
    return {'relative_logit_l2':float(np.linalg.norm(candidate-native)/max(np.linalg.norm(native),1e-30)),
            'native_to_candidate_kl':float(p@(lp-lq)),
            'native_top':int(np.argmax(native)),'candidate_top':int(np.argmax(candidate)),
            'same_top':bool(np.argmax(native)==np.argmax(candidate))}
def summary(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return {'n':int(a.size),'mean':float(a.mean()),'median':float(np.median(a)),
            'min':float(a.min()),'max':float(a.max())} if a.size else {'n':0}


def square(values):
    y00,y10,y01,y11=values
    a=y10-y00; b=y01-y00; i=y11-y10-y01+y00; joint=y11-y00
    den=max(float(y00@y00),1e-30)
    gram=np.stack((a,b,i))@np.stack((a,b,i)).T/den
    native_reader_change=float(np.linalg.norm(b)/max(np.linalg.norm(y00),1e-30))
    cheap_reader_change=float(np.linalg.norm(y11-y10)/max(np.linalg.norm(y10),1e-30))
    return {'writer_relative':float(np.linalg.norm(a)/np.sqrt(den)),
            'reader_relative':native_reader_change,'joint_relative':float(np.linalg.norm(joint)/np.sqrt(den)),
            'interaction_relative':float(np.linalg.norm(i)/np.sqrt(den)),
            'reader_change_under_cheap_writer_relative':cheap_reader_change,
            'reader_error_ratio_cheap_over_native_writer':cheap_reader_change/native_reader_change if native_reader_change>1e-14 else None,
            'gram_scaled':gram.tolist(),
            'writer_reader_cross_scaled':float(2*gram[0,1]),
            'closure_max_abs':float(np.max(np.abs(joint-a-b-i)))}


def analyze(root):
    root=Path(root); raw=root/'raw'; report=json.loads((root/'report.json').read_text())
    driver=report['driver']; paths=driver['paths']; by_name={p['name']:p for p in paths}
    matched=[]; own=[]; dynamics=[]; component_boundaries=[]; outcomes=[]
    for task in range(6):
        quad=[by_name[f'case{task}-replay-{p}'] for p in LABELS[:3]]+[by_name[f'case{task}-own-11']]
        n=quad[0]['count']; ids=quad[0]['consumed_ids']
        if any(p['consumed_ids']!=ids for p in quad): raise ValueError('Matched histories differ')
        arrays={field:[load(raw,f'{p["name"]}-{field}.f32',(n,width)) for p in quad] for field,width in FIELDS.items()}
        for field,values in arrays.items():
            for position in range(n):
                row={'task':task,'position':position+1,'field':field,**square([v[position] for v in values])}
                dynamics.append(row)
            for turn in quad[3]['turns']:
                for stage in ('prefill','final'):
                    pos=turn[stage]['position']
                    component_boundaries.append({'task':task,'turn':turn['turn'],'stage':stage,
                                                 'position':pos,'field':field,**square([v[pos-1] for v in values])})
        for turn in quad[3]['turns']:
            for stage in ('prefill','final'):
                ti=turn['turn']; vectors=[load(raw,f'{p["name"]}-t{ti}-{stage}-logits.f32',(151936,)) for p in quad]
                row={'task':task,'turn':ti,'stage':stage,'position':turn[stage]['position'],
                     **square(vectors),
                     'fidelity':{label:fidelity(vectors[j],vectors[0]) for j,label in enumerate(LABELS[1:],1)}}
                matched.append(row)
        native=by_name[f'case{task}-own-00']
        for policy,label in enumerate(LABELS):
            path=by_name[f'case{task}-own-{label}']
            outcomes.append({'task':task,'policy':label,'answers':report['texts'][path['name']],
                             'stopped':[t['stopped'] for t in path['turns']],
                             'truncated':[t['truncated'] for t in path['turns']]})
            if policy==0: continue
            for ti in range(2):
                for stage in ('prefill','final'):
                    a=load(raw,f'{path["name"]}-t{ti}-{stage}-logits.f32',(151936,))
                    b=load(raw,f'{native["name"]}-t{ti}-{stage}-logits.f32',(151936,))
                    na=path['turns'][ti][stage]['position']; nb=native['turns'][ti][stage]['position']
                    own.append({'task':task,'policy':label,'turn':ti,'stage':stage,
                                'same_consumed_ids':path['consumed_ids'][:na]==native['consumed_ids'][:nb],
                                **fidelity(a,b)})
    initial=[r for r in matched if r['turn']==0 and r['stage']=='prefill']
    by_policy={p:{'initial_logit_error':summary(r['fidelity'][p]['relative_logit_l2'] for r in initial),
                  'all_boundary_logit_error':summary(r['fidelity'][p]['relative_logit_l2'] for r in matched),
                  'all_boundary_native_kl':summary(r['fidelity'][p]['native_to_candidate_kl'] for r in matched),
                  'same_top_boundaries':sum(r['fidelity'][p]['same_top'] for r in matched)} for p in LABELS[1:]}
    attention=[r for r in component_boundaries if r['field']=='attention']
    metrics={'policies':by_policy,
             'logit_interaction':summary(r['interaction_relative'] for r in matched),
             'attention_reader_error_native_writer':summary(r['reader_relative'] for r in attention),
             'attention_reader_error_cheap_writer':summary(r['reader_change_under_cheap_writer_relative'] for r in attention),
             'attention_reader_error_ratio':summary(r['reader_error_ratio_cheap_over_native_writer'] for r in attention),
             'joint_vs_writer_logit_l2_wins':sum(r['fidelity']['11']['relative_logit_l2']<r['fidelity']['10']['relative_logit_l2'] for r in matched),
             'joint_vs_writer_kl_wins':sum(r['fidelity']['11']['native_to_candidate_kl']<r['fidelity']['10']['native_to_candidate_kl'] for r in matched),
             'negative_writer_reader_logit_cross_terms':sum(r['writer_reader_cross_scaled']<0 for r in matched),
             'matched_boundary_rows':len(matched),'own_comparison_rows':len(own),
             'dynamics_rows':len(dynamics),'component_boundary_rows':len(component_boundaries)}
    return {'schema':'operand-composition-scientific-analysis-v1','report_sha256':sha(root/'report.json'),
            'analysis_source_sha256':sha(__file__),'summary':metrics,'matched_readouts':matched,
            'own_readouts':own,'component_boundaries':component_boundaries,'dynamics':dynamics,
            'own_outcomes':outcomes,'terminal_status':'not-requested',
            'interpretation':'Matched squares are recurring policies on joint IDs. Own readouts with different histories are whole-policy comparisons. All signed directions remain reconstructible from the immutable four raw vectors.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=analyze(a.root); out=a.root/'analysis';out.mkdir(exist_ok=True)
    (out/'scientific-analysis.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
    rows=[]
    for kind in ('matched_readouts','own_readouts','component_boundaries','dynamics'):
        rows.extend({'kind':kind,'report_sha256':result['report_sha256'],**row} for row in result[kind])
    (out/'observations.jsonl').write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows))
    print(json.dumps(result['summary'],indent=2))
