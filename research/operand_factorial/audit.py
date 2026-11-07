"""Independent raw-array checks of the factorial's write-space interaction."""
import argparse,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def audit(job):
    root=HERE/'runs'/job;r=json.loads((root/'report.json').read_text());raw=root/'raw'
    down=np.fromfile(raw/'down.f32',dtype='<f4').reshape(1536,8960).astype('float64')
    groups={}
    for row in r['driver']['records']:groups.setdefault((row['clock'],row['parent']),{})[row['arm']]=row
    def load(stem,suffix):return np.fromfile(raw/(stem+'-'+suffix+'.f32'),dtype='<f4').astype('float64')
    # tanh form independently evaluates the same ideal SiLU identity.
    phi=lambda x:.5*x*(1+np.tanh(x/2))
    rows=[]
    for (clock,parent),arms in groups.items():
        stem=arms['00']['stem'];g,u,gh,uh=[load(stem,s) for s in ['native_gate','native_up','candidate_gate','candidate_up']]
        ideal=down@((phi(g)-phi(gh))*(u-uh))
        actual=load(arms['11']['stem'],'write')-load(arms['10']['stem'],'write')-load(arms['01']['stem'],'write')+load(arms['00']['stem'],'write')
        native=load(arms['native']['stem'],'write');norm=np.linalg.norm(native)
        rows.append({'clock':clock,'parent':parent,'write_interaction_identity_remainder_over_native':float(np.linalg.norm(actual-ideal)/norm),
            'native_vs_both_write_exact':bool(np.array_equal(native,load(arms['11']['stem'],'write'))),
            'native_vs_both_logits_exact':bool(np.array_equal(load(arms['native']['stem'],'logits'),load(arms['11']['stem'],'logits'))),
            'repeat00_exact':bool(np.array_equal(load(stem,'logits'),load(arms['repeat00']['stem'],'logits')))})
    across=[]
    for clock in sorted({k[0] for k in groups}):
        a,b=groups[(clock,'carried')]['11']['stem'],groups[(clock,'native')]['11']['stem']
        for suffix in ['receiver','native_gate','native_up','write']:
            across.append({'clock':clock,'observable':suffix,'byte_exact':(raw/(a+'-'+suffix+'.f32')).read_bytes()==(raw/(b+'-'+suffix+'.f32')).read_bytes()})
    result={'schema':'tcm-operand-factorial-independent-array-audit-v1','rows':rows,'cross_parent_current_computation':across,'cross_parent_mismatches':sum(not x['byte_exact'] for x in across),'max_interaction_remainder_over_native':max(x['write_interaction_identity_remainder_over_native'] for x in rows),'model_executions':0,'scope':'Independent tanh-SiLU algebra and cross-parent byte checks on saved weights/operands; finite remainder is observed, no scientific threshold.'}
    (root/'independent-array-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');audit(p.parse_args().job)
