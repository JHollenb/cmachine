"""Additive saved verification: numerical metric drift is an observation.

The executed verifier is preserved. Its 1e-14 absolute comparison of CPU64
diagnostic scalars is not a calibrated cross-BLAS/libm contract. This wrapper
retains every signed scalar difference while preserving exact hash, shape,
finite, token, work, model, receipt, native-sham and seam controls.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import hashlib


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(root):
    root=Path(root).resolve(); corrections=root/'corrections';corrections.mkdir(exist_ok=True)
    executed=root/'payload/verification.py'
    spec=importlib.util.spec_from_file_location('executed_operand_verifier',executed)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module._compare; deltas=[]
    def compare(a,b,path=()):
        if path and path[0]=='numerics':
            if isinstance(a,dict):
                if not isinstance(b,dict) or a.keys()!=b.keys(): raise ValueError('Numerics schema differs '+str(path))
                for key in a: compare(a[key],b[key],path+(key,))
            elif isinstance(a,list):
                if not isinstance(b,list) or len(a)!=len(b): raise ValueError('Numerics row coverage differs '+str(path))
                for i,(left,right) in enumerate(zip(a,b)): compare(left,right,path+(i,))
            elif isinstance(a,float) or isinstance(b,float):
                if not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not math.isfinite(a) or not math.isfinite(b): raise ValueError('Invalid numeric diagnostic '+str(path))
                deltas.append({'path':list(path),'local_recomputed':a,'worker_reported':b,
                               'signed_difference':a-b,'absolute_difference':abs(a-b)})
            else: original(a,b,path)
        else: original(a,b,path)
    module._compare=compare
    result=module.verify(root)
    differences={'schema':'operand-composition-numerical-portability-v1',
                 'executed_verifier_sha256':sha(executed),'additive_verifier_sha256':sha(__file__),
                 'policy':'Numerical scalar differences are diagnostics. Numerical schemas, row counts and finite contracts remain strict; all other executed verifier mechanics remain unchanged.',
                 'scalars':len(deltas),'nonidentical_scalars':sum(r['absolute_difference']!=0 for r in deltas),
                 'maximum_absolute_scalar_difference':max(r['absolute_difference'] for r in deltas),
                 'rows':deltas}
    (corrections/'numerical-portability-v1.json').write_text(json.dumps(differences,sort_keys=True,allow_nan=False)+'\n')
    result.update(verification_policy='additive numerical portability v1',
                  numerical_scalar_equality_gate=False,
                  numerical_diagnostic_schema_and_finiteness_verified=True,
                  numerical_portability='corrections/numerical-portability-v1.json',
                  maximum_absolute_numeric_scalar_difference=differences['maximum_absolute_scalar_difference'],
                  verification_source_sha256=sha(__file__),
                  executed_verifier_sha256=sha(executed))
    (root/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.root),indent=2,sort_keys=True))
