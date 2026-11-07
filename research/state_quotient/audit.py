"""Independent stdlib exact-rank audit, using float.as_integer_ratio instead of bits."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

def audit(root):
    root = Path(root)
    report = json.loads((root/'report.json').read_text())
    raw = (root/'raw/qwen-down-minor.f32').read_bytes()
    n,prime = 256,2147483647
    if len(raw)!=n*n*4 or hashlib.sha256(raw).hexdigest()!=report['driver']['qwen_minor']['sha256']:
        raise ValueError('Minor identity differs')
    start = time.monotonic()
    values = []
    for (value,) in struct.iter_unpack('<f',raw):
        numerator,denominator = value.as_integer_ratio()
        values.append(numerator*pow(denominator,-1,prime)%prime)
    matrix = [values[i*n:(i+1)*n] for i in range(n)]
    rank,determinant = 0,1
    for col in range(n):
        pivot_row = next((i for i in range(rank,n) if matrix[i][col]),None)
        if pivot_row is None:continue
        if pivot_row!=rank:
            matrix[rank],matrix[pivot_row] = matrix[pivot_row],matrix[rank]
            determinant = -determinant%prime
        pivot = matrix[rank][col];determinant = determinant*pivot%prime
        inverse = pow(pivot,-1,prime)
        for i in range(rank+1,n):
            factor = matrix[i][col]*inverse%prime
            if factor:
                matrix[i][col:] = [(x-factor*y)%prime for x,y in zip(matrix[i][col:],matrix[rank][col:])]
        rank += 1
        if rank==n:break
    if rank<n:determinant=0
    expected = report['analysis']['rank_witness']
    if rank!=expected['rank'] or determinant!=expected['determinant_mod_prime']:
        raise ValueError('Independent rank arithmetic differs')
    own_adds=0
    for desc in report['driver']['paths']:
        name,r,steps = desc['name'],desc['rank'],desc['steps']
        unpack = lambda suffix:list(struct.unpack('<'+'f'*((root/'raw'/f'{name}-{suffix}.f32').stat().st_size//4),(root/'raw'/f'{name}-{suffix}.f32').read_bytes()))
        states,writes = unpack('states'),unpack('writes')
        for t in range(steps):
            for j in range(r):
                rounded = struct.pack('<f',states[t*r+j]+writes[t*r+j])
                if rounded!=struct.pack('<f',states[(t+1)*r+j]):raise ValueError('Own update differs')
                own_adds+=1
    result = {'schema':'tcm-quotient-independent-audit-v1','report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),'minor_sha256':hashlib.sha256(raw).hexdigest(),'field':prime,'rank':rank,'determinant_mod_prime':determinant,'independent_mapping':'stdlib float.as_integer_ratio and modular inverse','independent_elimination':'unnormalized rows,Python arbitrary-precision integers','retained_state_scalar_adds_byte_checked':own_adds,'seconds':time.monotonic()-start,'qwen_forwards':0,'semantic_certification':False}
    (root/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');audit(p.parse_args().root)
