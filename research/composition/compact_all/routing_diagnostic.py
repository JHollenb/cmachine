"""Saved-only L0 routing/codec audit; no model, fitting or new trajectory."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(root):
    root=Path(root).resolve();verification=json.loads((root/'verification.json').read_text());inventory=json.loads((root/'raw-inventory.json').read_text())
    if sha(root/'report.json')!=verification['report_sha256'] or sha(root/'reanalysis.json')!=verification['reanalysis_sha256']:raise ValueError('Qualified report binding differs')
    for name in ('byte_all-rows.f32','byte_all-queries.f32'):
        if sha(root/'raw'/name)!=inventory[name]['sha256']:raise ValueError('Raw input differs')
    sys.path.insert(0,str(root/'payload'))
    from append_analysis import encode
    from span_parent_analysis import SCALE
    report=json.loads((root/'report.json').read_text());n=report['driver']['chains'][2]['main_transitions'];rows=np.fromfile(root/'raw/byte_all-rows.f32',dtype='<f4').reshape(n,28,2,2,128);queries=np.fromfile(root/'raw/byte_all-queries.f32',dtype='<f4').reshape(n,4,12,128)
    k=rows[:,0,0];v=rows[:,0,1];_,_,oldk=encode(k);_,_,newk=encode(k.reshape(n,2,8,16));newk=newk.reshape(n,2,128);_,_,decodedv=encode(v);group=np.arange(12)//6;output=[]
    def read(q,keys,values):
        scores=np.einsum('hd,hid->hi',q.astype(np.float64),keys.transpose(1,0,2)[group].astype(np.float64),optimize=False)*SCALE;maximum=scores.max(axis=1);e=np.exp(scores-maximum[:,None]);z=e.sum(axis=1);p=e/z[:,None];mu=np.einsum('hi,hid->hd',p,values.transpose(1,0,2)[group].astype(np.float64),optimize=False);return p,mu,maximum+np.log(z)
    for position in range(n):
        p,mu,z=read(queries[position,0],k[:position+1],v[:position+1]);oldp,oldmu,oldz=read(queries[position,0],oldk[:position+1],decodedv[:position+1]);newp,newmu,newz=read(queries[position,0],newk[:position+1],decodedv[:position+1]);den=np.linalg.norm(mu,axis=1)
        for h in range(12):output.append({'position':position,'head':h,'native_top':int(np.argmax(p[h])),'row_key_top':int(np.argmax(oldp[h])),'block16_key_top':int(np.argmax(newp[h])),'row_context_error':float(np.linalg.norm(oldmu[h]-mu[h])/den[h]),'block16_context_error':float(np.linalg.norm(newmu[h]-mu[h])/den[h]),'row_probability_l1':float(np.sum(np.abs(oldp[h]-p[h]))),'block16_probability_l1':float(np.sum(np.abs(newp[h]-p[h]))),'row_logZ_difference':float(oldz[h]-z[h]),'block16_logZ_difference':float(newz[h]-z[h])})
    def summary(key):
        a=np.asarray([r[key] for r in output]);return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'max':float(a.max()),'min':float(a.min())}
    result={'schema':'tcm-saved-routing-codec-diagnostic-v1','report_sha256':verification['report_sha256'],'reanalysis_sha256':verification['reanalysis_sha256'],'source_sha256':sha(__file__),'input_sha256':{name:inventory[name]['sha256'] for name in ('byte_all-rows.f32','byte_all-queries.f32')},'models_evaluated':0,'profile':'posthoc same saved own operands; block16 keys fixed, values row-int8; no new trajectory','summary':{k:summary(k) for k in ('row_context_error','block16_context_error','row_probability_l1','block16_probability_l1')},'improved_reads':sum(r['block16_context_error']<r['row_context_error'] for r in output),'examples':[r for r in output if (r['position'],r['head']) in ((7,3),(30,3))],'rows':output,'terminal_status':'not-requested'}
    data=json.dumps(result,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n';target=root/'routing-diagnostic.json'
    if target.exists() and target.read_bytes()!=data:raise ValueError('Saved diagnostic changed')
    target.write_bytes(data);(root/'routing-diagnostic-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
