"""One source-response direction; no target parsing, gradients or selection."""
import hashlib,json,mmap,struct,time
from pathlib import Path
import numpy as np

DOSES=(0.,-1.,.25,.5,1.,2.)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return {'name':path.name,'bytes':path.stat().st_size,'sha256':sha(path)}
def save(out,name,array,dtype='<f8'):
    path=out/name;path.write_bytes(np.asarray(array,dtype=dtype).tobytes());return record(path)

def rope(x,positions,inverse=False):
    x=np.asarray(x,dtype=float);shape=x.shape;y=x.reshape(len(positions),-1,128)
    angle=np.asarray(positions)[:,None,None]*1e6**(-2*np.arange(64,dtype=float)/128)[None,None,:]
    c,s=np.cos(angle),np.sin(angle)*(-1 if inverse else 1);a,b=y[...,:64],y[...,64:]
    return np.concatenate((a*c-b*s,b*c+a*s),axis=-1).reshape(shape)

def response(q,k,v,mass):
    score=k@q/np.sqrt(128.)+np.log(mass);p=np.exp(score-score.max());p/=p.sum()
    return p@v,p

def reader_pullback(q,k,v,mass,error):
    a,p=response(q,k,v,mass)
    return (p*((v-a)@error))@k/np.sqrt(128.)

def source_states(kn,vn,kc,lengths):
    from analysis import Memory
    nodes=[];teacher=[];positions=[];start=0
    for n in lengths:
        m=Memory();group=[]
        for t in range(n):
            m.append(kc[start+t].reshape(2,128),vn[start+t].reshape(2,128).astype('<f4'))
            slots=[(m.head+16-m.recent+i)%16 for i in range(m.recent)];row=[]
            for g in range(2):
                c=int(m.clusters[g]);row.append((np.concatenate((m.ks[g,:c]/m.count[g,:c,None],m.keys[g,slots].astype(float))),np.concatenate((m.vs[g,:c]/m.count[g,:c,None],m.values[g,slots].astype(float))),np.concatenate((m.count[g,:c].astype(float),np.ones(m.recent)))))
            nodes.append(row);positions.append(t);group.append((start,start+t+1))
        teacher.extend(group);start+=n
    if start!=len(kn):raise ValueError('Source trajectory extent differs')
    return nodes,teacher,np.asarray(positions)

def prepare(qn,kn,vn,kc,bias,lengths):
    nodes,ranges,positions=source_states(kn,vn,kc,lengths);qn=rope(qn+bias,positions).reshape(-1,12,128);truth=[]
    for t,(start,stop) in enumerate(ranges):
        truth.append([response(qn[t,h],kn[start:stop].reshape(-1,2,128)[:,h//6],vn[start:stop].reshape(-1,2,128)[:,h//6],np.ones(stop-start))[0] for h in range(12)])
    truth=np.asarray(truth);weights=[];offset=0
    for n in lengths:
        energy=float(np.sum(truth[offset:offset+n]**2));weights.extend([1/(len(lengths)*max(energy,1e-30))]*n);offset+=n
    return nodes,truth,positions,np.asarray(weights)

def evaluate(image,z,bias,prepared,gradient=False,tangent=None):
    nodes,truth,positions,weights=prepared;q=rope(z@np.asarray(image,dtype=float).T+bias,positions).reshape(-1,12,128)
    dq=rope(z@tangent.T,positions).reshape(q.shape) if tangent is not None else None
    ys=np.empty_like(q);g=np.zeros_like(q);curvature=0.;entropies=[]
    for t in range(len(q)):
        for h in range(12):
            k,v,mass=nodes[t][h//6];a,p=response(q[t,h],k,v,mass);ys[t,h]=a;e=a-truth[t,h]
            if gradient:g[t,h]=2*weights[t]*reader_pullback(q[t,h],k,v,mass,e)
            if dq is not None:
                ds=k@dq[t,h]/np.sqrt(128.);da=(p*(ds-p@ds))@v;curvature+=weights[t]*float(da@da)
            entropies.append(float(-np.sum(p*np.log(np.maximum(p,1e-300)))))
    errors=np.sum((ys-truth)**2,axis=(1,2));loss=float(weights@errors)
    grad=rope(g.reshape(len(q),1536),positions,inverse=True) if gradient else None
    return loss,grad,{'weighted_loss':loss,'mean_node_attention_entropy_nats':float(np.mean(entropies)),'curvature':float(curvature)},errors

def propose(z,baseline,qn,kn,vn,kc,bias,lengths):
    prepared=prepare(qn,kn,vn,kc,bias,lengths);loss,grad,details,_=evaluate(baseline,z,bias,prepared,gradient=True)
    coefficients,_,rank,_=np.linalg.lstsq(z,grad,rcond=None)
    if rank!=128:raise ValueError('Source feature rank differs')
    direction=-coefficients.T;dq=z@direction.T;gd=float(np.sum(grad*dq));_,_,linear,_=evaluate(baseline,z,bias,prepared,tangent=direction)
    curvature=linear['curvature'];alpha=max(0.,-gd/(2*curvature)) if curvature>0 else 0.
    # A centered source derivative check diagnoses the gradient instrument, not quality.
    epsilon=1e-5*max(float(np.linalg.norm(z@baseline.T)),1.)/max(float(np.linalg.norm(dq)),1e-30)
    lp=evaluate(baseline+epsilon*direction,z,bias,prepared)[0];lm=evaluate(baseline-epsilon*direction,z,bias,prepared)[0];fd=(lp-lm)/(2*epsilon)
    derivative_relative_discrepancy=abs(fd-gd)/max(abs(fd),abs(gd),1e-30)
    if not np.isfinite([loss,gd,alpha,fd]).all() or derivative_relative_discrepancy>1e-4:raise ValueError('Source derivative instrument invalid')
    rows=[];images=[]
    for i,dose in enumerate(DOSES):
        image=np.asarray(baseline+dose*alpha*direction,dtype='<f4');value,_,stats,errors=evaluate(image,z,bias,prepared);per=[];offset=0
        for n in lengths:
            per.append(float(np.sqrt(np.sum(errors[offset:offset+n])/max(float(np.sum(prepared[1][offset:offset+n]**2)),1e-30))));offset+=n
        rows.append({'index':i,'dose':dose,'step':dose*alpha,'source_response_loss':value,'trajectory_relative_response_errors':per,'source_query_relative_error':float(np.linalg.norm(z@image.astype(float).T-qn)/np.linalg.norm(qn)),'source_query_displacement_relative_to_ls':float(np.linalg.norm(z@(image.astype(float)-baseline).T)/np.linalg.norm(z@baseline.T)),'mean_node_attention_entropy_nats':stats['mean_node_attention_entropy_nats']});images.append(image)
    selected=min(range(len(rows)),key=lambda i:(rows[i]['source_response_loss'],abs(rows[i]['dose']),i))
    return images,rows,selected,direction,{'gradient_direction_inner_product':gd,'linearized_response_curvature':curvature,'predicted_line_step':alpha,'finite_difference_directional_derivative':fd,'finite_difference_epsilon':epsilon,'derivative_relative_discrepancy':derivative_relative_discrepancy,'source_baseline_loss':loss}

def tensors(model):
    digest=hashlib.sha256()
    with Path(model['path']).open('rb') as stream:
        prefix=stream.read(40)
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    if prefix[:8]!=b'TCM1LE\0\0' or prefix[8:].hex()!=model['content_sha256'] or digest.hexdigest()!=model['content_sha256']:raise ValueError('Compiler model body differs')
    result={};records={}
    with Path(model['path']).open('rb') as stream:
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
        if mapping[112:176].decode()!=model['identity']:raise ValueError('Model identity differs')
        for i in range(struct.unpack_from('<I',mapping,44)[0]):
            row=struct.unpack_from('<64sII4QQQ',mapping,176+i*120);name=row[0].split(b'\0')[0].decode()
            for kind in ('q','k','v'):
                for field in ('weight','bias'):
                    if name!=f'model.layers.17.self_attn.{kind}_proj.{field}':continue
                    dims=tuple(row[3:3+row[1]]);expected=((1536 if kind=='q' else 256),1536) if field=='weight' else ((1536 if kind=='q' else 256),)
                    dtype,offset,nbytes=row[2],row[-2],row[-1]
                    if dims!=expected or dtype not in (1,2) or nbytes!=int(np.prod(dims))*(2 if dtype==2 else 4) or offset+nbytes>len(mapping):raise ValueError('Tensor geometry differs')
                    raw=mapping[offset:offset+nbytes];a=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4') if dtype==2 else np.frombuffer(raw,dtype='<f4')
                    result[kind+field]=a.reshape(dims).astype(float);records[kind+field]={'name':name,'shape':dims,'dtype':dtype,'offset':offset,'bytes':nbytes,'sha256':hashlib.sha256(raw).hexdigest()}
        mapping.close()
    if len(result)!=6 or not all(np.isfinite(a).all() for a in result.values()):raise ValueError('Missing/nonfinite compiler tensors')
    return result,records

def query_transactions(baseline,images,selected,source_binding):
    import torch
    from saturn_pub import Adapter,Act,Session,SlotSpec,SurfaceManifest
    class QueryAdapter(Adapter):
        model_identity=source_binding;execution={'family':'source-query-response','adapter':'bounded-image-proposals'}
        def boundary(self,state):return 'query-image'
        def addresses(self,state):return ('query',)
        def validate(self,state):
            q=state['query']
            if set(state)!={'query'} or q.shape!=(1536,128) or q.dtype!=torch.float32 or not torch.isfinite(q).all():raise ValueError('Query image state differs')
        def advance(self,state):raise RuntimeError('Only declared image Acts are allowed')
        def surface(self,state):return SurfaceManifest((SlotSpec('query',writable=True,producer='source-only response compiler',consumer='compact memory reader'),),'query-image-f32-1536x128','source response and native C continuation')
    session=Session(QueryAdapter(),{'query':torch.from_numpy(np.asarray(baseline,dtype='<f4').copy())});parent=session.capture();rows=[]
    for i,image in enumerate(images):
        candidate=torch.from_numpy(image.copy())
        def install(reads):return {'query':candidate.clone()}
        receipt=session.apply(Act('source-response-query-image',('query',),('query',),install,parameters={'proposal_index':i,'source_binding':source_binding}))
        future=session.capture(retain=False);restored=session.restore(parent);after=session.capture(retain=False)
        if after.fingerprint!=parent.fingerprint:raise ValueError('Image rollback failed')
        rows.append({'index':i,'apply':receipt.to_dict(),'future':future.fingerprint,'restore':restored.to_dict(),'restored_parent_exact':True})
    return {'scope':'Exact image-only mutable closure; no optimizer, RNG, cursor, hooks or mutable model cache in compiler. Native C histories are separate later trials.','parent':parent.fingerprint,'selected_index':selected,'proposals':rows}

def compile_image(model,root,out):
    started=time.monotonic();root,out=Path(root),Path(out);source=json.loads((root/'source-provenance.json').read_text())
    if source['model_digest']!=model['content_sha256'] or source['model_identity']!=model['identity']:raise ValueError('Source model differs')
    for name,r in source['files'].items():
        if record(root/name)!={'name':name,**r}:raise ValueError('Source bytes differ')
    x=np.fromfile(root/'source-inputs.f32',dtype='<f4').reshape(193,1536).astype(float);b=np.fromfile(root/'program128-basis.f32',dtype='<f4').reshape(128,1536).astype(float)
    base=np.fromfile(root/'program128-q.f32',dtype='<f4').reshape(1536,128).astype(float);km=np.fromfile(root/'program128-k.f32',dtype='<f4').reshape(256,128).astype(float)
    if sha(root/'program128-q.f32')!='99e6fa43c6fbed8d01a78f2255b7a3ba12d57b49393f0a32392e3c292656e1c6':raise ValueError('LS parent differs')
    ts,recipe=tensors(model);lengths=[t['advances'] for t in source['source_trajectories']];positions=np.concatenate([np.arange(n) for n in lengths]);z=x@b.T
    qn=x@ts['qweight'].T;kn=rope(x@ts['kweight'].T+ts['kbias'],positions);vn=x@ts['vweight'].T+ts['vbias'];kc=rope(z@km.T+ts['kbias'],positions).astype('<f4');bias=ts['qbias']
    arrays={'source-query-targets.f64':qn,'source-native-keys.f64':kn,'source-native-values.f64':vn,'source-query-bias.f64':bias}
    files={name:save(out,name,a) for name,a in arrays.items()};files['source-compiled-keys.f32']=save(out,'source-compiled-keys.f32',kc,'<f4')
    images,rows,selected,direction,details=propose(z,base,qn,kn,vn,kc,bias,lengths)
    files['response-direction.f64']=save(out,'response-direction.f64',direction)
    for row,image in zip(rows,images):row['image']=save(out,f'query-proposal-{row["index"]}.f32',image,'<f4')
    image_record=save(out,'query-fit.f32',images[selected],'<f4')
    transactions=query_transactions(base,images,selected,sha(root/'source-provenance.json'));(out/'query-transactions.json').write_text(json.dumps(transactions,sort_keys=True,indent=2)+'\n')
    result={'schema':'tcm-source-response-query-image-v1','model_digest':model['content_sha256'],'model_identity':model['identity'],'source_provenance':source,'tensors':recipe,'basis_sha256':sha(root/'program128-basis.f32'),'baseline_query_sha256':sha(root/'program128-q.f32'),'query_image':image_record,'source_query_targets':files['source-query-targets.f64'],'source_arrays':files,'source_rows':193,'source_trajectories':3,'trajectory_lengths':lengths,'target_inputs_read_by_compiler':False,'fresh_prompts_parsed_after_image_seal':True,'target_fitting':False,'new_source_model_advances':0,'serving_extra_macs':0,'serving_extra_linear_calls':0,'serving_extra_state_bytes':0,'teacher_profile':'CPU64 original-weight affine + bias/half-split RoPE/native-full attention, separate source clocks; not CUDA-byte-exact','candidate_profile':'CPU64 compact reader with FP32 rounded compiled keys/native values and unchanged FP64 centroid law','selection':'minimum source-only response loss; smaller absolute dose then index on exact tie; no target outcomes','details':details,'proposals':rows,'selected_index':selected,'transactions_sha256':sha(out/'query-transactions.json'),'nominal_cpu_affine_source_macs':193*(1536*1536+2*256*1536+1536*128+256*128),'source_gradient_line_search_and_diagnostics_additional_paid':True,'seconds':time.monotonic()-started}
    (out/'query-compiler.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');return result

def verify_fit(raw,driver):
    raw=Path(raw);c=json.loads((raw/'query-compiler.json').read_text());seal=json.loads((raw/'query-image-seal.json').read_text())
    if c['model_digest']!=driver['model_digest'] or c['model_identity']!=driver['model_identity'] or c['target_fitting'] or c['target_inputs_read_by_compiler']:raise ValueError('Compiler identity/visibility differs')
    if seal['query_image']!=c['query_image'] or seal['compiler_report_sha256']!=sha(raw/'query-compiler.json') or seal['phase']!='before-parsing-fresh-prompts-or-tokenizing-targets':raise ValueError('Query image seal differs')
    for r in [*c['source_arrays'].values(),c['query_image'],*[p['image'] for p in c['proposals']]]:
        if record(raw/r['name'])!=r:raise ValueError('Source proposal bytes differ')
    for name,r in c['source_provenance']['files'].items():
        if record(raw/name)!={'name':name,**r}:raise ValueError('Source acquisition differs')
    def load(name,shape,dtype='<f8'):return np.fromfile(raw/name,dtype=dtype).reshape(shape)
    z=load('source-inputs.f32',(193,1536),'<f4').astype(float)@load('program128-basis.f32',(128,1536),'<f4').astype(float).T
    base=load('program128-q.f32',(1536,128),'<f4').astype(float)
    args=(load('source-query-targets.f64',(193,1536)),load('source-native-keys.f64',(193,256)),load('source-native-values.f64',(193,256)),load('source-compiled-keys.f32',(193,256),'<f4'),load('source-query-bias.f64',(1536,)),c['trajectory_lengths'])
    images,rows,selected,direction,details=propose(z,base,*args)
    # FP64 diagnostics compare through the outer verifier; emitted images have their own seals.
    if selected!=c['selected_index'] or sha(raw/'query-transactions.json')!=c['transactions_sha256']:raise ValueError('Selection/transaction identity differs')
    tx=json.loads((raw/'query-transactions.json').read_text())
    if len(tx['proposals'])!=len(DOSES) or not all(p['restored_parent_exact'] and p['restore']['result']==tx['parent'] for p in tx['proposals']):raise ValueError('Rollback record differs')
    image=load('query-fit.f32',(1536,128),'<f4');chosen=load(c['proposals'][selected]['image']['name'],(1536,128),'<f4')
    if not np.array_equal(image,chosen) or not np.array_equal(base,load('query-proposal-0.f32',(1536,128),'<f4')):raise ValueError('Selected/zero-dose image differs')
    verify_update(raw)
    return {'source_rows':193,'source_trajectories':3,'target_inputs_used_for_fit':False,'proposals':rows,'selected_index':selected,'details':{key:details[key] for key in ('gradient_direction_inner_product','linearized_response_curvature','predicted_line_step','source_baseline_loss')},'query_image_sha256':c['query_image']['sha256'],'same_serving_macs_calls_and_state':True,'image_only_rollback_exact':True,'source_target_arithmetic':c['teacher_profile']}

def verify_update(raw):
    """Tie saved image bytes to the declared dose and saved source direction."""
    raw=Path(raw);c=json.loads((raw/'query-compiler.json').read_text());base=np.fromfile(raw/'program128-q.f32',dtype='<f4').reshape(1536,128).astype(float);direction=np.fromfile(raw/'response-direction.f64',dtype='<f8').reshape(base.shape);rows=[]
    for p in c['proposals']:
        saved=np.fromfile(raw/p['image']['name'],dtype='<f4').reshape(base.shape);expected=np.asarray(base+p['step']*direction,dtype='<f4')
        if not np.array_equal(saved,expected):raise ValueError('Proposal does not equal declared update')
        if p['step']!=p['dose']*c['details']['predicted_line_step']:raise ValueError('Dose step differs')
        rows.append({'index':p['index'],'exact_declared_update':True,'image_sha256':sha(raw/p['image']['name'])})
    return {'schema':'tcm-query-response-update-verification-v1','scope':'Additive saved-array integrity check; original report/payload unchanged. No cross-platform refit image byte claim.','proposals':rows,'all_declared_updates_byte_exact':True}
