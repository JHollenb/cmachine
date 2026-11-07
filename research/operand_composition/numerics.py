"""Independent CPU64 equations on the C process's saved, actual operands.

These comparisons diagnose the numerical instrument. CPU64 BLAS, transcendental
functions and reduction orders differ from C/CUDA FP32; fidelity magnitudes do
not become exploratory quality gates. Missing, malformed or nonfinite inputs
are integrity errors. No weights or native model invocation is needed.
"""
from pathlib import Path
import numpy as np

D, H, R, Q = 1536, 8960, 64, 128


def load(raw, name, shape):
    value = np.fromfile(Path(raw)/name, dtype='<f4')
    if value.size != int(np.prod(shape)) or not np.isfinite(value).all():
        raise ValueError('Invalid numerical audit input '+name)
    return value.reshape(shape)


def discrepancy(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError('Numerical audit shape differs')
    delta = a-b
    return {'absolute_l2': float(np.linalg.norm(delta)),
            'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(b), 1e-30)),
            'max_abs': float(np.max(np.abs(delta)))}


def rope(value, position, theta):
    value = np.asarray(value, dtype=np.float64).reshape(-1,128)
    angle = position/np.power(float(theta), 2*np.arange(64)/128.)
    c, s = np.cos(angle), np.sin(angle)
    a, b = value[:,:64], value[:,64:]
    return np.concatenate((a*c-b*s,b*c+a*s),axis=1).reshape(-1)


def full_read(query, keys, values):
    result=[]
    for h, q in enumerate(np.asarray(query,dtype=np.float64).reshape(12,128)):
        score=keys[:,h//6].astype(np.float64)@q/np.sqrt(128.)
        probability=np.exp(score-score.max()); probability/=probability.sum()
        result.append(probability@values[:,h//6].astype(np.float64))
    return np.asarray(result).reshape(-1)


def audit_numerics(raw, driver):
    raw=Path(raw)
    E=load(raw,'mlp-E.f32',(R,D)).astype(np.float64)
    Lg=load(raw,'mlp-Lg.f32',(H,R)).astype(np.float64)
    Lu=load(raw,'mlp-Lu.f32',(H,R)).astype(np.float64)
    B=load(raw,'query-B.f32',(Q,D)).astype(np.float64)
    F=load(raw,'query-F.f32',(D,Q)).astype(np.float64)
    K=load(raw,'query-K.f32',(256,Q)).astype(np.float64)
    gamma=load(raw,'attention17-input-norm.f32',(D,)).astype(np.float64)
    bq=load(raw,'attention17-q-bias.f32',(D,)).astype(np.float64)
    bk=load(raw,'attention17-k-bias.f32',(256,)).astype(np.float64)
    rows=[]
    for path in driver['paths']:
        name,n=path['name'],path['count']
        fields={'mlp-norm':D,'mlp-output':D,'receiver':D,'q':D,'k':256,'v':256,
                'attention':D,'mlp-code':R,'query-code':Q}
        a={key:load(raw,f'{name}-{key}.f32',(n,width)) for key,width in fields.items()}
        keys=a['k'].reshape(n,2,128); values=a['v'].reshape(n,2,128)
        for turn in path['turns']:
            for kind in ('prefill','final'):
                i=turn[kind]['position']-1
                r=a['mlp-output'][i].astype(np.float64)
                norm=(r/np.sqrt(np.mean(r*r)+driver['epsilon']))*gamma
                record={'path':name,'task':path['task'],'policy':path['policy'],
                        'mode':path['mode'],'turn':turn['turn'],'stage':kind,'position':i+1,
                        'receiver_normalization':discrepancy(norm,a['receiver'][i]),
                        'own_full_attention':discrepancy(full_read(a['q'][i],keys[:i+1],values[:i+1]),a['attention'][i])}
                if path['writer']:
                    z=a['mlp-code'][i].astype(np.float64)
                    record['mlp_coordinate']=discrepancy(E@a['mlp-norm'][i].astype(np.float64),z)
                    for label,image in (('gate',Lg),('up',Lu)):
                        target=load(raw,f'{name}-t{turn["turn"]}-{kind}-{label}.f32',(H,))
                        record['mlp_'+label+'_from_executed_code']=discrepancy(image@z,target)
                if path['reader']:
                    z=a['query-code'][i].astype(np.float64)
                    record['query_coordinate']=discrepancy(B@a['receiver'][i].astype(np.float64),z)
                    record['query_bias_and_rope']=discrepancy(rope(F@z+bq,i,driver['theta']),a['q'][i])
                    record['key_bias_and_rope']=discrepancy(rope(K@z+bk,i,driver['theta']),a['k'][i])
                rows.append(record)
    kinds=sorted({key for row in rows for key,value in row.items() if isinstance(value,dict)})
    summary={key:{'rows':sum(key in row for row in rows),
                  'max_relative_l2':max(row[key]['relative_l2'] for row in rows if key in row),
                  'max_absolute_l2':max(row[key]['absolute_l2'] for row in rows if key in row)} for key in kinds}
    return {'schema':'operand-composition-independent-numerics-v1',
            'profile':'CPU64 equations applied to saved C FP32 operands; no native model calls',
            'rows':rows,'summary':summary,'models_evaluated':0,
            'terminal_status':'not-requested'}
