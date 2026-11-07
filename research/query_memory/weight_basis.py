"""Offline checkpoint-only right-subspace compiler; no activation or target fitting."""
import hashlib
import json
import mmap
from pathlib import Path
import struct
import time
import numpy as np

def compile_basis(model,out):
    started=time.monotonic();path=Path(model['path']);out=Path(out)
    # Hash the declared bundle body independently of the native C loader.
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        prefix=stream.read(40)
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    if prefix[:8]!=b'TCM1LE\0\0' or prefix[8:].hex()!=model['content_sha256'] or digest.hexdigest()!=model['content_sha256']:raise ValueError('Weight compiler model digest differs')
    matrices=[];recipes=[]
    with path.open('rb') as stream:
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ);count=struct.unpack_from('<I',mapping,44)[0]
        if mapping[112:176].decode()!=model['identity']:raise ValueError('Weight compiler model identity differs')
        table={}
        for i in range(count):
            row=struct.unpack_from('<64sII4QQQ',mapping,176+i*120);name=row[0].split(b'\0')[0].decode();table[name]=row
        for family,width in (('q',1536),('k',256),('v',256)):
            name=f'model.layers.17.self_attn.{family}_proj.weight';row=table[name];dtype=row[2];shape=tuple(row[3:5]);offset,nbytes=row[-2:]
            if row[1]!=2 or shape!=(width,1536) or dtype not in (1,2) or offset+nbytes>len(mapping):raise ValueError('QKV tensor contract differs')
            raw=mapping[offset:offset+nbytes]
            if dtype==2:value=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').reshape(shape).astype(float)
            else:value=np.frombuffer(raw,dtype='<f4').reshape(shape).astype(float)
            if not np.isfinite(value).all():raise ValueError('Nonfinite original QKV weight')
            norm=float(np.linalg.norm(value));matrices.append(value/norm);recipes.append({'tensor':name,'shape':list(shape),'dtype':dtype,'offset':offset,'bytes':nbytes,'sha256':hashlib.sha256(raw).hexdigest(),'frobenius_norm':norm})
        mapping.close()
    joint=np.concatenate(matrices);rng=np.random.default_rng(20261106);width=144
    q=np.linalg.qr(joint.T@rng.standard_normal((2048,width)),mode='reduced')[0]
    for _ in range(3):
        left=np.linalg.qr(joint@q,mode='reduced')[0]
        q=np.linalg.qr(joint.T@left,mode='reduced')[0]
    small=joint@q;_,singular,vh=np.linalg.svd(small,full_matrices=False);directions=q@vh[:128].T
    # Freeze column signs for readable, reproducible ownership; eigenspace rotations
    # can still differ between math-library profiles, so emitted bytes are canonical.
    for j in range(128):
        pivot=int(np.argmax(np.abs(directions[:,j])))
        if directions[pivot,j]<0:directions[:,j]*=-1
    records=[]
    for rank in (32,128):
        basis=np.ascontiguousarray(directions[:,:rank].T,dtype='<f4');file=out/f'weight-basis{rank}.f32';file.write_bytes(basis.tobytes())
        errors=[]
        start=0
        for matrix in matrices:
            captured=float(np.linalg.norm(matrix@basis.astype(float).T)**2);errors.append(float(np.sqrt(max(0.,1-captured))));start+=matrix.shape[0]
        records.append({'rank':rank,'basis_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'basis_bytes':file.stat().st_size,'family_relative_frobenius_residuals':errors})
    result={'schema':'tcm-weight-qkv-rightspace-v1','model_digest':model['content_sha256'],'model_identity':model['identity'],'uses_target_activations':False,'uses_calibration_activations':False,'family_scaling':'equal Frobenius norm for Q,K,V','seed':20261106,'oversampled_width':144,'power_iterations':3,'qr_after_each_multiplication':True,'small_svd_singular_values':singular.tolist(),'tensor_recipes':recipes,'programs':records,'nominal_joint_matrix_macs':8*2048*1536*144,'nominal_final_basis_rotation_macs':1536*144*128,'nominal_family_diagnostic_macs':2048*1536*(32+128),'qr_svd_normalization_and_scalar_work_additional':True,'seconds':time.monotonic()-started}
    (out/'weight-compiler.json').write_text(json.dumps(result,indent=2)+'\n');return result
