"""Artifact-only covariance/write audit; no model imports or inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
raw = root / 'raw'
report = json.loads((root / 'report.json').read_text())
def load(name, shape):
    x = np.fromfile(raw / name, dtype='<f4').reshape(shape).astype(np.float64)
    assert np.isfinite(x).all(), name
    return x
def energy(x):
    return float(np.sum(x*x, dtype=np.float64))

bases = {p: [load(f'{p}/E-{l}.f32', (256,1536)) for l in range(28)] for p in ('p0','p1')}
banks = {b: load(f'source-{b}-selected.f32', (384,28,1536)) for b in ('native','compact')}
out = {'scope':'artifact-only FP64 reconstruction; fixed source samples; no native inference',
       'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),
       'geometry':{}, 'local_writes':{}, 'policy_metrics':{}}
for p,es in bases.items():
    for b,xs in banks.items():
        rows=[]
        for l,e in enumerate(es):
            x=xs[:,l,:]; residual=x-(x@e.T)@e
            rows.append({'layer':l,'input_energy':energy(x),'residual_energy':energy(residual),
                         'relative_residual_energy':energy(residual)/energy(x),
                         'orthogonality_max_abs':float(np.max(np.abs(e@e.T-np.eye(256))))})
        out['geometry'][f'{p}_on_{b}']={'layers':rows,
            'pooled_residual_energy_fraction':sum(r['residual_energy'] for r in rows)/sum(r['input_energy'] for r in rows),
            'mean_layer_residual_energy_fraction':float(np.mean([r['relative_residual_energy'] for r in rows]))}
for label,run in report['runs'].items():
    rows=[]
    for path in run['driver']['paths']:
        for ti,t in enumerate(path['turns']):
            for stage,b in t['boundaries'].items():
                w=load(b['write_file'],(28,1536)); n=load(b['native_write_file'],(28,1536))
                for l in range(28):
                    ne=energy(n[l]); de=energy(w[l]-n[l])
                    rows.append({'case_id':path['case_id'],'turn':ti,'stage':stage,'layer':l,
                                 'reference_energy':ne,'difference_energy':de,
                                 'relative_L2':(de/ne)**.5 if ne else None})
    selected=range(28) if label!='heldout-p0-layer16' else [16]
    selected_rows=[x for x in rows if x['layer'] in selected]
    summaries=[]
    for l in range(28):
        a=[x for x in rows if x['layer']==l]
        summaries.append({'layer':l,'count':len(a),'pooled_relative_L2':(sum(x['difference_energy'] for x in a)/sum(x['reference_energy'] for x in a))**.5,
                          'mean_relative_L2':float(np.mean([x['relative_L2'] for x in a])),
                          'max_relative_L2':max(x['relative_L2'] for x in a)})
    out['local_writes'][label]={'layers':summaries,'rows':rows,
         'selected_layers_pooled_relative_L2':(sum(x['difference_energy'] for x in selected_rows)/sum(x['reference_energy'] for x in selected_rows))**.5,
         'selected_layers_mean_relative_L2':float(np.mean([x['relative_L2'] for x in selected_rows]))}
    m=run['metrics']; ds=m['decisions']
    out['policy_metrics'][label]={'equal_case_turn_KL':m['equal_case_turn_KL'],
       'decisions':len(ds),'conditional_top_matches':sum(d['emitted']==d['native_shadow_top'] for d in ds),
       'answers':[{'case_id':t['case_id'],'turn':t['turn'],'answer':t['answer'],'tokens':t['tokens'],'stopped':t['stopped'],'mean_KL':t['mean_KL']} for t in m['turns']]}
target=root/'analysis'/'own_geometry.json'
target.parent.mkdir(exist_ok=True)
assert not target.exists(),target
target.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'geometry':{k:{a:v for a,v in x.items() if a!='layers'} for k,x in out['geometry'].items()},
 'local_writes':{k:{a:v for a,v in x.items() if a not in ('layers','rows')} for k,x in out['local_writes'].items()},
 'policy_metrics':{k:{a:v for a,v in x.items() if a!='answers'} for k,x in out['policy_metrics'].items()}}))
