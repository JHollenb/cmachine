"""Saved-array algebra and consumer observations; no scientific quality gates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(path, count=None):
    x=np.fromfile(path,dtype='<f4')
    if count is not None and x.size!=count: raise ValueError('shape: '+str(path))
    if not np.isfinite(x).all(): raise ValueError('nonfinite: '+str(path))
    return x


def silu(x):
    x=np.asarray(x,dtype=np.float64);e=np.exp(-np.abs(x))
    return x*np.where(x>=0,1/(1+e),e/(1+e))


def terms(down,g,u,gh,uh):
    s,sh=silu(g),silu(gh);du=u.astype(np.float64)-uh;ds=s-sh
    return down@np.column_stack((ds*uh,sh*du,ds*du))


def rel(x,y):
    norm=float(np.linalg.norm(y));return float(np.linalg.norm(x-y)/norm) if norm else None


def analyze(raw,saved,driver,plan,save_projected=False):
    raw,saved=Path(raw),Path(saved);d,h=1536,8960
    down=array(raw/'down.f32',d*h).reshape(d,h).astype(np.float64)
    saved_rows=json.loads((saved/'clocks.json').read_text());decomposed=[];projected={}
    for row in saved_rows:
        stem=row['stem'];load=lambda suffix:array(saved/(stem+'-'+suffix+'.f32'))
        g,u,gh,uh=[load(s) for s in ['native_gate','native_up','mlp_gate','mlp_up']]
        if any(x.size!=h for x in [g,u,gh,uh]):raise ValueError('operand shape')
        native,candidate=load('native_write').astype(np.float64),load('mlp_return').astype(np.float64)
        t=terms(down,g,u,gh,uh);error=native-candidate;remainder=error-t.sum(axis=1)
        energy=float(error@error);gram=t.T@t
        gram_n=gram/energy if energy else gram*0
        normal=float(native@native)
        ys=down@np.column_stack((silu(gh)*uh,silu(g)*uh,silu(gh)*u,silu(g)*u))
        projected[stem]=np.column_stack((t,remainder))
        decomposed.append({**row,'relative_write_error':rel(candidate,native),
            'term_norm_over_native':(np.linalg.norm(t,axis=0)/np.linalg.norm(native)).tolist(),
            'signed_gram_over_observed_error_energy':gram_n.tolist(),
            'term_alignment_with_error_over_error_energy':(t.T@error/energy).tolist() if energy else [0,0,0],
            'finite_remainder_over_error':float(np.linalg.norm(remainder)/np.linalg.norm(error)),
            'finite_remainder_over_native':float(np.linalg.norm(remainder)/np.linalg.norm(native)),
            'ideal_local_restoration_errors':{name:rel(ys[:,i],native) for i,name in enumerate(['00','10','01','11'])},
            'candidate_norm_over_native':float(np.linalg.norm(candidate)/np.linalg.norm(native)),
            'twice_alignment_minus_candidate_energy_over_native_energy':float((2*native@candidate-candidate@candidate)/normal)})
    if save_projected:
        np.savez(raw/'signed-projections.npz',**projected)
    records=driver['records'];groups=defaultdict(dict);observations=[];mechanics=[]
    by_clock={c['key']:c for c in plan['clocks']}
    for r in records:groups[(r['clock'],r['parent'])][r['arm']]=r
    for (key,parent),arms in groups.items():
        if set(arms)!={'00','10','01','11','zero','native','repeat00'}:raise ValueError('incomplete factorial')
        hashes={v['prefix_kv_sha256'] for v in arms.values()}
        mechanics.append({'kind':'common_parent','clock':key,'parent':parent,'exact':len(hashes)==1})
        for a,b in [('11','native'),('00','repeat00')]:
            eq=all((raw/(arms[a]['stem']+'-'+suffix+'.f32')).read_bytes()==(raw/(arms[b]['stem']+'-'+suffix+'.f32')).read_bytes() for suffix in ['write','logits'])
            eq=eq and arms[a]['continuation']==arms[b]['continuation']
            mechanics.append({'kind':'replay_'+a+'_'+b,'clock':key,'parent':parent,'exact':eq})
        clock=by_clock[key];expected=clock['expected_carried_logits'] if parent=='carried' else clock['expected_native_logits'];arm='00' if parent=='carried' else 'native'
        mechanics.append({'kind':'historical_readout','clock':key,'parent':parent,'exact':(raw/(arms[arm]['stem']+'-logits.f32')).read_bytes()==(saved/expected).read_bytes()})
        for suffix,old in [('receiver','raw_receiver'),('native_gate','native_gate'),('native_up','native_up'),('candidate_gate','mlp_gate'),('candidate_up','mlp_up')]:
            mechanics.append({'kind':'historical_operand_'+suffix,'clock':key,'parent':parent,'exact':(raw/(arms['00']['stem']+'-'+suffix+'.f32')).read_bytes()==(saved/(clock['saved_stem']+'-'+old+'.f32')).read_bytes()})
        native_reference=groups[(key,'native')]['native'];native_logits=array(raw/(native_reference['stem']+'-logits.f32')).astype(np.float64)
        local_native=array(raw/(arms['native']['stem']+'-write.f32')).astype(np.float64)
        current_native_logits=array(raw/(arms['native']['stem']+'-logits.f32')).astype(np.float64)
        z={name:array(raw/(r['stem']+'-logits.f32')).astype(np.float64) for name,r in arms.items()}
        write_interaction=sum((coef*array(raw/(arms[name]['stem']+'-write.f32')).astype(np.float64) for name,coef in [('11',1),('10',-1),('01',-1),('00',1)]))
        logit_interaction=z['11']-z['10']-z['01']+z['00']
        for name,r in arms.items():
            write=array(raw/(r['stem']+'-write.f32')).astype(np.float64)
            observations.append({'clock':key,'case':clock['case'],'policy':clock['policy'],'boundary':clock['boundary'],'parent':parent,'arm':name,
                'write_relative_error':rel(write,local_native),'logit_relative_error_vs_native_same_ID':rel(z[name],native_logits),
                'logit_relative_error_vs_same_parent_native':rel(z[name],current_native_logits),
                'top':r['logits']['top'],'native_same_ID_top':native_reference['logits']['top'],'continuation':r['continuation'],
                'write_interaction_over_native':float(np.linalg.norm(write_interaction)/np.linalg.norm(local_native)),
                'logit_interaction_over_native_same_ID':float(np.linalg.norm(logit_interaction)/np.linalg.norm(native_logits))})
    return {'schema':'tcm-operand-factorial-analysis-v1','mechanics':mechanics,'decomposition':decomposed,'observations':observations,
        'scope':'36 selected development clocks for algebra; six policy-specific clocks x two history parents x seven arms, immediate full vocabulary and four native continuation tokens. Signed terms include cancellation. No independent-token statistics or terminal quality gates.',
        'summary':{'decomposition_rows':len(decomposed),'factorial_rows':len(observations),'mechanics_checks':len(mechanics),'mechanics_mismatches':sum(not x['exact'] for x in mechanics),
                   'term_order':['gate','payload','interaction'],'max_finite_remainder_over_native':max(r['finite_remainder_over_native'] for r in decomposed)}}
