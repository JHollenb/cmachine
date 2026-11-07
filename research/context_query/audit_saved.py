"""Independent retained-boundary and source-reader audit without a model or Torch."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def audit(root):
    root=Path(root);raw=root/'raw';payload=root/'payload';report=read(root/'report.json');analysis=read(root/'analysis.json');spec=importlib.util.spec_from_file_location('audit_memory',payload/'reader.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    spec=importlib.util.spec_from_file_location('audit_image_math',payload/'image_transactions.py');maths=importlib.util.module_from_spec(spec);spec.loader.exec_module(maths)
    boundaries=memories=0;maximum=0.
    for row in analysis['rows']:
        name=row['name'];trace=read(raw/(name+'.json'));metrics=read(raw/(name+'-metrics.json'))['samples']
        for s in metrics:
            if not s['boundary']:continue
            stem=f'{name}-turn{s["turn"]}-b{s["boundary"]}';n=np.fromfile(raw/(stem+'-native.f32'),dtype='<f4');c=np.fromfile(raw/(stem+'-candidate.f32'),dtype='<f4')
            if n.shape!=(151936,) or c.shape!=n.shape or not np.isfinite(n).all() or not np.isfinite(c).all() or sha(raw/(stem+'-native.f32'))!=s['native_sha256'] or sha(raw/(stem+'-candidate.f32'))!=s['candidate_sha256']:raise ValueError('Retained boundary identity differs')
            n=n.astype(float);c=c.astype(float);n-=n.max();c-=c.max();ln=n-np.log(np.exp(n).sum());lc=c-np.log(np.exp(c).sum());kl=float(np.sum(np.exp(ln)*(ln-lc)));error=abs(kl-s['kl']);maximum=max(maximum,error)
            if error>1e-10*max(1,abs(kl)):raise ValueError('Retained boundary KL differs')
            boundaries+=1
        for p in raw.glob(name+'-turn*-memory*.bin'):
            data=p.read_bytes()
            if len(data)!=98464:raise ValueError('Prototype shape differs')
            count=np.frombuffer(data,dtype='<u4',count=32,offset=65536).reshape(2,16);clusters=np.frombuffer(data,dtype='<u4',count=2,offset=65664);next0,recent,head=map(int,np.frombuffer(data[-24:],dtype='<u8'));turn=int(p.name.split('-turn')[1][0]);expected=sum(phase<2 for phase in trace['input_phases']) if turn==0 else trace['count']
            if next0!=expected or recent!=min(16,next0) or head!=next0%16 or any(int(count[g].sum())+recent!=next0 or clusters[g]>16 for g in (0,1)):raise ValueError('Prototype mass/cursor differs')
            memories+=1
    target=read(payload/'targets/manifest.json');truth=np.fromfile(payload/'targets/native-q-compact-memory.f32',dtype='<f4').reshape(-1,1536);base=np.fromfile(payload/'query-fit.f32',dtype='<f4').reshape(1536,128).astype(float);images=[np.fromfile(raw/f'query-proposal-{i}.f32',dtype='<f4').reshape(base.shape).astype(float) for i in range(4)];loss=np.zeros(4);clock_reads=0
    for c in range(3):
        trace=read(payload/f'parity{c}-trace.json');n=trace['count'];q=np.fromfile(payload/f'parity{c}-q.f32',dtype='<f4').reshape(n,1536).astype(float);k=np.fromfile(payload/f'parity{c}-k.f32',dtype='<f4').reshape(n,2,128);v=np.fromfile(payload/f'parity{c}-v.f32',dtype='<f4').reshape(n,2,128);z=np.fromfile(payload/f'parity{c}-coeff.f32',dtype='<f4').reshape(n,128).astype(float);a=np.fromfile(payload/f'parity{c}-attention.f32',dtype='<f4').reshape(n,1536);memory=reader.Memory();chosen={r['clock']:r for r in target['rows'] if r['source']==c}
        for t in range(n):
            memory.append(k[t],v[t]);actual=memory.read(q[t].reshape(12,128)).reshape(-1)
            if not np.array_equal(actual,a[t]):raise ValueError('Historical actual read differs')
            clock_reads+=1
            if t not in chosen:continue
            r=chosen[t]
            if hashlib.sha256(memory.bytes()).hexdigest()!=r['memory_sha256']:raise ValueError('Source memory hash differs')
            for i,image in enumerate(images):
                query=q[t]+maths.rope((z[t]@(image-base).T)[None,:],np.asarray([t]))[0];response=memory.read(query.reshape(12,128)).reshape(-1).astype(float);error=response-truth[r['row']].astype(float);loss[i]+=r['weight']*float(error@error)
    expected=np.asarray([r['local_source_loss'] for r in report['fit']['proposals']]);relative=np.abs(loss-expected)/np.maximum(expected,1e-30)
    if relative.max()>1e-6:raise ValueError('Independent rounded reader response loss differs')
    selected=report['selected_index'];x=np.concatenate([np.fromfile(raw/f's{c}-p{selected}-norm18.f32',dtype='<f4').reshape(-1,1536) for c in range(3)]).astype(float);b=np.fromfile(raw/'basis18.f32',dtype='<f4').reshape(128,1536).astype(float);energy=float(np.sum((x@b.T)**2)/np.sum(x*x));orth=float(np.linalg.norm(b@b.T-np.eye(128),ord=2))
    if abs(energy-report['basis18']['captured_source_energy'])>1e-12 or orth>1e-6:raise ValueError('Basis source energy/orthogonality differs')
    result={'schema':'context-query-independent-saved-audit-v1','report_sha256':sha(root/'report.json'),'retained_full_logit_pairs':boundaries,'maximum_retained_kl_discrepancy':maximum,'prototype_snapshots':memories,'historical_actual_reads_exact':clock_reads,'source_response_losses_with_independent_fp32_output_rounding':loss.tolist(),'maximum_relative_smooth_vs_rounded_loss_difference':float(relative.max()),'later_basis_source_rows':len(x),'later_basis_source_energy':energy,'later_basis_orthogonality_spectral_error':orth,'new_model_calls':0,'scope':'Independent saved arrays, memory law and retained logits. Does not replay discarded streams, compile original layer18 weight images again, or certify general chat.'};(root/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');audit(p.parse_args().root)
