"""Independent FP64 full-vocabulary distribution calculations."""
import hashlib,json
from pathlib import Path
import numpy as np
DOSES=(0.,-1.,.25,.5,1.,2.)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def distribution(native,candidate):
    a,b=[np.asarray(x,dtype=float) for x in (native,candidate)]
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Distribution operands invalid')
    def logp(x):
        x=x-x.max();return x-np.log(np.exp(x).sum())
    ln,lc=logp(a),logp(b);p=np.exp(ln);top=int(np.argmax(a))
    return {'kl_nats':float(p@(ln-lc)),'relative_logit_l2':float(np.linalg.norm(b-a)/max(float(np.linalg.norm(a)),1e-30)),'native_top_logp_delta_nats':float(lc[top]-ln[top]),'native_entropy_nats':float(-p@ln),'native_top':top,'candidate_top':int(np.argmax(b))}
def choose(means):return min(range(6),key=lambda i:(means[i],abs(DOSES[i]),i))
def audit_source(root,index):
    root=Path(root);report=json.loads((root/f'source-metrics-{index}.json').read_text());rows=report['rows'];n=193;vocab=151936
    if report['proposal']!=index or report['dose']!=DOSES[index] or len(rows)!=n:raise ValueError('Source metric geometry differs')
    sources=[]
    for name in ('source-native-logits.tmp','source-candidate-logits.tmp'):
        if (root/name).stat().st_size!=n*vocab*4:raise ValueError('Source full-vocabulary extent differs')
        sources.append(np.memmap(root/name,dtype='<f4',mode='r',shape=(n,vocab)))
    values=[];maximum={key:0. for key in ('kl_nats','relative_logit_l2','native_top_logp_delta_nats','native_entropy_nats')}
    for t,row in enumerate(rows):
        a,b=[x[t] for x in sources]
        for key,x in [('native',a),('candidate',b)]:
            if hashlib.sha256(x.tobytes()).hexdigest()!=row[key+'_logits_sha256']:raise ValueError('C/Python source row identity differs')
        measured=distribution(a,b)
        for key,value in measured.items():
            if key.endswith('_top'):
                if row[key]!=value:raise ValueError('C/Python source argmax differs')
            else:
                error=abs(value-row[key]);maximum[key]=max(maximum[key],error)
                if error>1e-10*max(1.,abs(value),abs(row[key])):raise ValueError('C/Python source distribution calculation differs')
        values.append({'source':row['source'],'step':row['step'],'region':row['region'],**measured})
    means=[float(np.mean([r['kl_nats'] for r in values if r['source']==c])) for c in range(3)];mean=float(np.mean(means))
    if abs(mean-report['equal_trajectory_mean_kl_nats'])>1e-10:raise ValueError('Source mean differs')
    result={'schema':'tcm-native-source-live-audit-v1','proposal':index,'dose':DOSES[index],'rows':values,'per_trajectory_mean_kl_nats':means,'equal_trajectory_mean_kl_nats':mean,'maximum_c_python_discrepancy':maximum,'native_stream_sha256':sha(root/'source-native-logits.tmp'),'candidate_stream_sha256':sha(root/'source-candidate-logits.tmp'),'source_logits_consumed':n*vocab*2,'scope':'Contemporaneous full-clock C/Python agreement on exact FP32 row bytes; transient full streams removed after source selection. Only nine checkpoint logits per policy have durable independent replay.'}
    (root/f'source-audit-{index}.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');return result
