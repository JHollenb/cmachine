"""One response-gradient proposal from source-owned histories; no target access."""
import hashlib,json
from pathlib import Path
import numpy as np
from reader import Memory
from image_transactions import rope,query_transactions

DOSES=(0.,-1.,.5,1.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array(p,width):return np.fromfile(p,dtype='<f4').reshape(-1,width).astype(float)
def rows(memory,g):
    c=int(memory.clusters[g]);slots=[(memory.head+16-memory.recent+j)%16 for j in range(memory.recent)]
    return (np.concatenate((memory.ks[g,:c]/memory.count[g,:c,None],memory.keys[g,slots].astype(float))),np.concatenate((memory.vs[g,:c]/memory.count[g,:c,None],memory.values[g,slots].astype(float))),np.concatenate((memory.count[g,:c].astype(float),np.ones(memory.recent))))
def compile_source(root,out):
    root,out=Path(root),Path(out);manifest=json.loads((root/'targets/manifest.json').read_text());nodes=[];q=[];z=[];positions=[];weights=[];truth=[]
    for c in range(3):
        trace=json.loads((root/f'parity{c}-trace.json').read_text());k=array(root/f'parity{c}-k.f32',256).reshape(-1,2,128);v=array(root/f'parity{c}-v.f32',256).reshape(-1,2,128);query=array(root/f'parity{c}-q.f32',1536);coeff=array(root/f'parity{c}-coeff.f32',128);memory=Memory();chosen={r['clock']:r for r in manifest['rows'] if r['source']==c}
        for clock in range(trace['count']):
            memory.append(k[clock].astype('<f4'),v[clock].astype('<f4'))
            if clock not in chosen:continue
            row=chosen[clock]
            if hashlib.sha256(memory.bytes()).hexdigest()!=row['memory_sha256']:raise ValueError('Source memory identity differs')
            nodes.append([rows(memory,g) for g in range(2)]);q.append(query[clock]);z.append(coeff[clock]);positions.append(clock);weights.append(row['weight'])
    q=np.asarray(q).reshape(-1,12,128);z=np.asarray(z);positions=np.asarray(positions);weights=np.asarray(weights);truth=array(root/'targets/native-q-compact-memory.f32',1536).reshape(q.shape);base=array(root/'query-fit.f32',128)
    def evaluate(image,gradient=False,tangent=None):
        query=q+rope(z@(image-base).T,positions).reshape(q.shape);dq=rope(z@tangent.T,positions).reshape(q.shape) if tangent is not None else None;g=np.zeros_like(q);loss=curvature=0.
        for t in range(len(q)):
            for h in range(12):
                key,value,mass=nodes[t][h//6];score=key@query[t,h]/np.sqrt(128)+np.log(mass);p=np.exp(score-score.max());p/=p.sum();a=p@value;e=a-truth[t,h];loss+=weights[t]*float(e@e)
                if gradient:g[t,h]=2*weights[t]*(p*((value-a)@e))@key/np.sqrt(128)
                if tangent is not None:
                    ds=key@dq[t,h]/np.sqrt(128);da=(p*(ds-p@ds))@value;curvature+=weights[t]*float(da@da)
        return loss,rope(g.reshape(-1,1536),positions,inverse=True),curvature
    loss,grad,_=evaluate(base,True);gram=z.T@z;ridge=float(np.linalg.eigvalsh(gram)[-1]*1e-3);direction=-np.linalg.solve(gram+ridge*np.eye(128),z.T@grad).T;dq=z@direction.T;gd=float(np.sum(grad*dq));curvature=evaluate(base,tangent=direction)[2];alpha=max(0.,-gd/(2*curvature)) if curvature>0 else 0.;cap=.15*np.linalg.norm(q)/max(np.linalg.norm(dq),1e-30);alpha=min(alpha,float(cap))
    eps=1e-5*max(float(np.linalg.norm(q)),1.)/max(float(np.linalg.norm(dq)),1e-30);fd=(evaluate(base+eps*direction)[0]-evaluate(base-eps*direction)[0])/(2*eps);discrepancy=abs(fd-gd)/max(abs(fd),abs(gd),1e-30)
    if not np.isfinite([loss,alpha,fd]).all() or discrepancy>1e-4:raise ValueError('Response derivative instrument broken')
    images=[];proposals=[]
    for i,dose in enumerate(DOSES):
        image=np.asarray(base+dose*alpha*direction,dtype='<f4');image.tofile(out/f'query-proposal-{i}.f32');images.append(image);proposals.append({'index':i,'dose':dose,'step':dose*alpha,'local_source_loss':evaluate(image)[0],'sha256':sha(out/f'query-proposal-{i}.f32')})
    np.asarray(direction,dtype='<f8').tofile(out/'source-direction.f64')
    tx=query_transactions(base,images,0,sha(root/'targets/manifest.json'));(out/'image-transactions.json').write_text(json.dumps(tx,indent=2)+'\n')
    result={'schema':'context-query-source-proposals-v1','source_target_manifest_sha256':sha(root/'targets/manifest.json'),'source_rows':len(q),'source_clock_reads':420,'teacher':'paid source-native Q through unchanged actual compact memory','objective':'equal source/turn mean squared same-memory attention response','baseline_loss':loss,'ridge':ridge,'gradient_dot_direction':gd,'curvature':curvature,'alpha':alpha,'query_step_relative_cap':.15,'finite_difference':fd,'derivative_relative_discrepancy':discrepancy,'proposals':proposals,'target_inputs_read':False,'selection':'later source-only own-token native-suffix KL, equally weighted sources and turns','mutable_closure':'query image only; exact Saturn parent restoration for each proposal; no optimizer or training RNG'}
    (out/'fit.json').write_text(json.dumps(result,indent=2)+'\n');return result
