"""One source-only, same-cost conditional query image. No target access."""
import hashlib,json,mmap,struct,time
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fit_image(x,basis,weight):
    """Argmin_F ||(X B.T) F.T - X Wq.T||_F; no ridge/rank search."""
    x,basis,weight=[np.asarray(a,dtype=float) for a in (x,basis,weight)]
    z=x@basis.T;target=x@weight.T
    return fit_targets(z,target)

def fit_targets(z,target):
    coefficients,residuals,rank,singular=np.linalg.lstsq(z,target,rcond=None)
    if rank!=z.shape[1] or not np.isfinite(coefficients).all():raise ValueError('Source coordinate solve invalid')
    return np.asarray(coefficients.T,dtype='<f4'),{'rank':int(rank),'coordinate_singular_values':singular.tolist(),'source_target_norm':float(np.linalg.norm(target)),'source_fit_relative_error':float(np.linalg.norm(z@coefficients-target)/max(float(np.linalg.norm(target)),1e-30))}

def compile_image(model,root,out):
    started=time.monotonic();root,out=Path(root),Path(out);source=json.loads((root/'source-provenance.json').read_text())
    if source['model_digest']!=model['content_sha256'] or source['model_identity']!=model['identity']:raise ValueError('Source model differs')
    for name,record in source['files'].items():
        if sha(root/name)!=record['sha256'] or (root/name).stat().st_size!=record['bytes']:raise ValueError('Source artifact differs')
    x=np.fromfile(root/'source-inputs.f32',dtype='<f4').reshape(193,1536)
    basis=np.fromfile(root/'program128-basis.f32',dtype='<f4').reshape(128,1536)
    baseline=np.fromfile(root/'program128-q.f32',dtype='<f4').reshape(1536,128)
    # Bind the exact original bundle body before reading its selected tensor.
    digest=hashlib.sha256()
    with Path(model['path']).open('rb') as stream:
        prefix=stream.read(40)
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    if prefix[:8]!=b'TCM1LE\0\0' or prefix[8:].hex()!=model['content_sha256'] or digest.hexdigest()!=model['content_sha256']:raise ValueError('Compiler model body differs')
    tensor='model.layers.17.self_attn.q_proj.weight'
    with Path(model['path']).open('rb') as stream:
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
        if mapping[112:176].decode()!=model['identity']:raise ValueError('Compiler model identity differs')
        count=struct.unpack_from('<I',mapping,44)[0];entry=None
        for i in range(count):
            row=struct.unpack_from('<64sII4QQQ',mapping,176+i*120)
            if row[0].split(b'\0')[0].decode()==tensor:entry=row;break
        if entry is None:raise ValueError('Native Q tensor missing')
        dtype,shape,offset,nbytes=entry[2],tuple(entry[3:5]),entry[-2],entry[-1]
        if entry[1]!=2 or shape!=(1536,1536) or dtype not in (1,2) or offset+nbytes>len(mapping):raise ValueError('Native Q tensor contract differs')
        raw=mapping[offset:offset+nbytes]
        if nbytes!=1536*1536*(2 if dtype==2 else 4):raise ValueError('Native Q tensor extent differs')
        weight=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4') if dtype==2 else np.frombuffer(raw,dtype='<f4')
        weight=weight.reshape(shape).astype(float);mapping.close()
    if not np.isfinite(weight).all():raise ValueError('Native Q tensor nonfinite')
    z=x.astype(float)@basis.astype(float).T;target=x.astype(float)@weight.T;den=max(float(np.linalg.norm(target)),1e-30)
    image,details=fit_targets(z,target);file=out/'query-fit.f32';file.write_bytes(image.tobytes());targetfile=out/'source-query-targets.f64';targetfile.write_bytes(np.asarray(target,dtype='<f8').tobytes())
    details.update(source_baseline_relative_error=float(np.linalg.norm(z@baseline.astype(float).T-target)/den),source_stored_fit_relative_error=float(np.linalg.norm(z@image.astype(float).T-target)/den))
    # Image is frozen before the worker parses fresh.json or tokenizes targets.
    result={'schema':'tcm-source-conditional-query-image-v1','model_digest':model['content_sha256'],'model_identity':model['identity'],'source_provenance':source,'tensor':{'name':tensor,'dtype':dtype,'shape':list(shape),'offset':offset,'bytes':nbytes,'sha256':hashlib.sha256(raw).hexdigest()},'basis_sha256':sha(root/'program128-basis.f32'),'baseline_query_sha256':sha(root/'program128-q.f32'),'query_image':{'name':file.name,'sha256':sha(file),'bytes':file.stat().st_size},'source_query_targets':{'name':targetfile.name,'sha256':sha(targetfile),'bytes':targetfile.stat().st_size},'construction':'FP64 ordinary source least squares; full128coordinate rank; no ridge/dose/rank search; native bias/RoPE unchanged','source_rows':193,'source_trajectories':3,'target_inputs_read_by_compiler':False,'fresh_prompts_parsed_after_image_seal':True,'target_fitting':False,'new_source_model_advances':0,'serving_extra_macs':0,'serving_extra_linear_calls':0,'serving_extra_state_bytes':0,'nominal_cpu_target_construction_macs':193*1536*1536,'nominal_cpu_feature_construction_macs':193*1536*128,'lstsq_and_diagnostics_additional_paid':True,'details':details,'seconds':time.monotonic()-started}
    (out/'query-compiler.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result
