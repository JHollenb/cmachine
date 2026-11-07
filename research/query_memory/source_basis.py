"""Expand an authenticated source-only basis without target fitting or new forwards."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def compile_basis(model,out):
    started=time.monotonic();root=Path(__file__).resolve().parent;out=Path(out)
    provenance=json.loads((root/'source-provenance.json').read_text())
    if provenance['model_digest']!=model['content_sha256'] or provenance['model_identity']!=model['identity']:raise ValueError('Source model differs')
    for name,digest in provenance['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source differs')
    x=np.fromfile(root/'source-inputs.f32',dtype='<f4').reshape(193,1536).astype(float)
    prefix=np.fromfile(root/'source-prefix32.f32',dtype='<f4').reshape(32,1536).astype(float)
    residual=x.copy();basis=list(prefix);pivots=[]
    for direction in prefix:
        for _ in range(2):residual-=(residual@direction)[:,None]*direction
    for rank in range(32,128):
        pivot=int(np.argmax(np.sum(residual*residual,axis=1)));direction=residual[pivot].copy()
        for _ in range(2):
            for previous in basis:direction-=np.dot(direction,previous)*previous
        norm=np.linalg.norm(direction)
        if not np.isfinite(norm) or norm<=0:raise ValueError('Source span exhausted')
        direction/=norm;basis.append(direction);pivots.append(pivot)
        for _ in range(2):residual-=(residual@direction)[:,None]*direction
    basis=np.asarray(basis,dtype='<f4');records=[]
    for rank in (32,128):
        file=out/f'weight-basis{rank}.f32';file.write_bytes(basis[:rank].tobytes())
        records.append({'rank':rank,'basis_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'basis_bytes':file.stat().st_size})
    result={'schema':'tcm-source-qkv-expanded-rightspace-v1','model_digest':model['content_sha256'],'model_identity':model['identity'],'uses_target_activations':False,'uses_calibration_activations':True,'source_rows':193,'source_provenance':provenance,'programs':records,'extension_pivots':pivots,'prefix32_byte_exact':(out/'weight-basis32.f32').read_bytes()==(root/'source-prefix32.f32').read_bytes(),'source_acquisition_native_advances_paid_previously':193,'new_source_acquisition_forwards':0,'nominal_source_residual_macs':2*2*193*1536*128,'nominal_extension_reorthogonalization_macs':4*1536*sum(range(32,128)),'pivot_norm_scaling_and_scalar_work_additional':True,'seconds':time.monotonic()-started}
    (out/'weight-compiler.json').write_text(json.dumps(result,indent=2)+'\n');return result
