"""Independent saved-array analysis of the thirteen-cut factorial; no model calls."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/job-eb06becf984f'
r=json.loads((RUN/'report.json').read_text())
checked=set()
def raw(name):
    b=(RUN/'raw/arrays'/name).read_bytes()
    assert hashlib.sha256(b).hexdigest()==r['raw_files']['arrays/'+name]['sha256']
    checked.add(name)
    return b
def a(name):
    v=np.frombuffer(raw(name),dtype='<f4').astype(np.float64)
    assert np.isfinite(v).all()
    return v
def sq(x): return float(np.dot(x,x))
def norm(x): return float(np.linalg.norm(x))
def states(b,h): return b['current'] if h==0 else b['future'][h-1]
rows=[]
for cut in r['driver']['cuts']:
    worlds={x['current_mode']:x for x in cut['branches']}
    worlds['H']=cut['references']['H']
    writes={k:a(v['current']['stem']+'-mlp_return.f32') for k,v in worlds.items()}
    native=writes['y11']; assert np.array_equal(native,writes['H'])
    assert raw(worlds['y11']['current']['stem']+'-output.f32')==raw(worlds['H']['current']['stem']+'-output.f32')
    for port,pairs in [('mlp_gate',[('y00','y01'),('y10','y11')]),('mlp_up',[('y00','y10'),('y01','y11')])]:
        for x,y in pairs:
            assert raw(worlds[x]['current']['stem']+'-'+port+'.f32')==raw(worlds[y]['current']['stem']+'-'+port+'.f32')
    inter=writes['y11']-writes['y10']-writes['y01']+writes['y00']
    ga=writes['y10']-writes['y00'];up=writes['y01']-writes['y00'];joint=native-writes['y00']
    projected=a(cut['interaction']['write_file'])
    four=a(cut['interaction']['four_write_file'])
    row={'case':cut['case'],'clock':cut['clock'],'position':cut['position'],'retrospective':cut['retrospective'],
         'native_write_sq':sq(native),'write_error_sq':{k:sq(v-native)for k,v in writes.items()},
         'write_relative_error':{k:norm(v-native)/norm(native)for k,v in writes.items()},
         'write_interaction_l2':norm(inter),'projected_interaction_relative_remainder':norm(inter-projected)/norm(inter),
         'write_factorial_energy':{'gate_sq':sq(ga),'up_sq':sq(up),'interaction_sq':sq(inter),'cross_gate_up':2*float(ga@up),
                                  'cross_gate_interaction':2*float(ga@inter),'cross_up_interaction':2*float(up@inter),'joint_sq':sq(joint)},
         'gpu_fourwrite_relative_remainder':norm(inter-four)/norm(inter),'horizons':[]}
    for h in range(1+len(cut['continuation_ids'])):
        st={k:states(v,h)for k,v in worlds.items()}
        z={k:a(v['readout']['file'])for k,v in st.items()}
        H=z['H']; c_l=z['y00']-z['y11'];l_h=z['y11']-H
        A=z['y10']-z['y00'];B=z['y01']-z['y00'];I=z['y11']-z['y10']-z['y01']+z['y00'];J=z['y11']-z['y00']
        kv={k:np.frombuffer(raw(v['current_kv_row_file']),dtype='<u4').reshape(2,28,2,128)for k,v in st.items()}
        first={}
        for base in ['y00','H']:
            first[base]={}
            for k,v in kv.items():
                changed=[l for l in range(28)if not np.array_equal(v[:,l],kv[base][:,l])]
                first[base][k]=changed[0]if changed else None
        hist={'horizon':h,'H_sq':sq(H),'error_sq':{k:sq(v-H)for k,v in z.items()},
              'relative_errors':{k:norm(v-H)/norm(H)for k,v in z.items()},
              'top':{k:int(np.argmax(v))for k,v in z.items()},
              'C_L_sq':sq(c_l),'L_H_sq':sq(l_h),'C_H_sq':sq(z['y00']-H),'cross_2dot':2*float(c_l@l_h),
              'C_L_relative':norm(c_l)/norm(H),'L_H_relative':norm(l_h)/norm(H),'cross_normalized':2*float(c_l@l_h)/sq(H),
              'factorial_effects_l2':{'gate_at_up0':norm(A),'up_at_gate0':norm(B),'interaction':norm(I),'joint':norm(J)},
              'factorial_energy':{'gate_sq':sq(A),'up_sq':sq(B),'interaction_sq':sq(I),'cross_gate_up':2*float(A@B),'cross_gate_interaction':2*float(A@I),'cross_up_interaction':2*float(B@I),'joint_sq':sq(J)},
              'first_changed_newKV_layer':first}
        if cut['retrospective']and h==0:
            hist['correct6_minus_wrong5']={k:float(v[21]-v[20])for k,v in z.items()}
        assert abs(hist['C_H_sq']-hist['C_L_sq']-hist['L_H_sq']-hist['cross_2dot'])<1e-8*max(1,hist['C_H_sq'])
        row['horizons'].append(hist)
    rows.append(row)
fixed=[x for x in rows if not x['retrospective']]
arms=['y00','y10','y01','y11','zero']
summary={'report_sha256':hashlib.sha256((RUN/'report.json').read_bytes()).hexdigest(),
 'scope':'twelve fixed cuts separately from one retrospective arithmetic digit; correlated clocks',
 'fixed_write':{},'fixed_horizons':{},'rows':rows}
for k in arms:
    summary['fixed_write'][k]={'pooled_error':float(np.sqrt(sum(x['write_error_sq'][k]for x in fixed)/sum(x['native_write_sq']for x in fixed))),
      'closer_than_Y00':sum(x['write_error_sq'][k]<x['write_error_sq']['y00']for x in fixed),
      'closer_than_zero':sum(x['write_error_sq'][k]<x['write_error_sq']['zero']for x in fixed)}
for h in range(3):
    hs=[x['horizons'][h]for x in fixed if len(x['horizons'])>h];den=sum(x['H_sq']for x in hs)
    summary['fixed_horizons'][str(h)]={'n':len(hs),'arms':{k:{'pooled_H_relative_error':float(np.sqrt(sum(x['error_sq'][k]for x in hs)/den)),
      'closer_than_Y00':sum(x['error_sq'][k]<x['error_sq']['y00']for x in hs),
      'closer_than_zero':sum(x['error_sq'][k]<x['error_sq']['zero']for x in hs),
      'H_top_matches':sum(x['top'][k]==x['top']['H']for x in hs)}for k in arms},
      'pooled_C_L':float(np.sqrt(sum(x['C_L_sq']for x in hs)/den)),
      'pooled_L_H':float(np.sqrt(sum(x['L_H_sq']for x in hs)/den)),
      'normalized_cross':sum(x['cross_2dot']for x in hs)/den,
      'negative_cross_rows':sum(x['cross_2dot']<0 for x in hs)}
summary['hash_checked_arrays']=len(checked)
summary['fixed_write_factorial_sum_energy']={k:sum(x['write_factorial_energy'][k]for x in fixed)for k in fixed[0]['write_factorial_energy']}
summary['common_nine_current']={}
common=[x for x in fixed if len(x['horizons'])==3]
den=sum(x['horizons'][0]['H_sq']for x in common)
summary['common_nine_current']={'n':len(common),'C_L':float(np.sqrt(sum(x['horizons'][0]['C_L_sq']for x in common)/den)),
                              'L_H':float(np.sqrt(sum(x['horizons'][0]['L_H_sq']for x in common)/den))}
(RUN/'advisor-independent-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items()if k!='rows'},indent=2))
for x in rows:
    h=x['horizons'][0]
    print('CUT',x['case'],x['clock'],'writes',[round(x['write_relative_error'][k],6)for k in arms],
      'logits',[round(h['relative_errors'][k],6)for k in arms],
      'C_L_L_H_cross',[round(h[k],6)for k in ['C_L_relative','L_H_relative','cross_normalized']])
print('INTERACTION_REMAINDER_RANGE',min(x['projected_interaction_relative_remainder']for x in rows),max(x['projected_interaction_relative_remainder']for x in rows))
for h in range(3):
    hs=[x['horizons'][h]for x in rows if len(x['horizons'])>h]
    for base in ['y00','H']:
        print('KV',h,base,{k:sorted(set(str(x['first_changed_newKV_layer'][base][k])for x in hs))for k in arms})
