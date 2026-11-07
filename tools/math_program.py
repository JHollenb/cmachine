"""Source-mapped math execution, exact polynomial analysis and FieldLM reads.

All semantic specifications are supplied. The analyzer establishes its admitted
polynomial structure before classifying an operation. C owns numerical kernels;
native FieldLM retains parsing, frame extension and graph mutation.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from math import prod
import numpy as np
from program import Program,canonical,OPCODES
from inspect_program import inspect


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def read_program(path,mapping):
    inspect(mapping)
    raw=Path(path).read_bytes()
    if len(raw)<176 or raw[:8]!=b"TCM1LE\0\0" or hashlib.sha256(raw[40:]).digest()!=raw[8:40] or raw[8:40].hex()!=mapping['bundle']['content_sha256']:
        raise ValueError('program/map identity differs')
    architecture,count=struct.unpack_from('<II',raw,40)
    if architecture!=4 or 176+120*count>len(raw):raise ValueError('not a valid vector program')
    bank={};indices={}
    for i in range(count):
        name,rank,dtype,*tail=struct.unpack_from('<64sII4QQQ',raw,176+120*i)
        width={1:4,3:8}.get(dtype)
        key=name.split(b'\0')[0].decode()
        if (width is None or not 1<=rank<=4 or key in bank or not all(tail[:rank]) or any(tail[rank:4])
            or tail[4]<176+120*count or tail[4]%8 or tail[5]!=prod(tail[:rank])*width or tail[4]+tail[5]>len(raw)):
            raise ValueError('invalid program tensor descriptor')
        bank[key]=np.frombuffer(raw,dtype={1:'<f4',3:'<f8'}[dtype],count=tail[5]//width,offset=tail[4]).reshape(tail[:rank]);indices[key]=i
        if not np.isfinite(bank[key]).all():raise ValueError('nonfinite binary program tensor')
    def equal(actual,expected,field):
        if not np.array_equal(actual,expected):raise ValueError('binary/source-map '+field+' differs')
    if any(key not in bank or bank[key].dtype!=np.dtype('<f8') for key in ('__ops','__values')):
        raise ValueError('binary/source-map schedule dtype differs')
    equal(architecture,mapping['bundle']['architecture'],'architecture')
    equal(count,mapping['bundle']['tensor_count'],'tensor count')
    equal(len(raw),mapping['bundle']['bytes'],'byte count')
    equal(raw[112:176].decode(),mapping['source_identity'],'source identity')
    config=[1,len(mapping['values']),len(mapping['instructions']),sum(v['length'] for v in mapping['values']),
        sum(prod(s['shape']) for s in mapping['state_slots']),0,mapping['output_value'],mapping['input_width'],
        int(mapping['token_mode']),mapping['readout_domain'],0,0,0,0,0,0]
    equal(struct.unpack_from('<16I',raw,48),config,'header/output/state')
    equal(bank.get('__values'),np.asarray([[v['offset'],v['length']] for v in mapping['values']],dtype='<f8'),'value table')
    parameters={item['source_name']:item['bundle_name'] for item in mapping['parameters']}
    if len(parameters)!=len(mapping['parameters']) or set(bank)!={'__ops','__values',*parameters.values()}:
        raise ValueError('binary/source-map parameter inventory differs')
    for item in mapping['parameters']:
        tensor=bank[item['bundle_name']]
        equal(tensor.shape,tuple(item['shape']) or (1,),'parameter shape')
        if tensor.dtype!=np.dtype('<f4') or item['execution_dtype']!='float32':raise ValueError('parameter is not executed FP32')
    expected=[]
    for op in mapping['instructions']:
        expected.append([OPCODES[op['kind']],op['out'],op['a'],op['b'],op['c'],op['d'],
            indices[parameters[op['w']]] if op['w'] is not None else -1,
            indices[parameters[op['bias']]] if op['bias'] is not None else -1,
            op['offset'],op['arg'],op['scale'],op['shift']])
    equal(bank.get('__ops'),np.asarray(expected,dtype='<f8'),'instruction table')
    return {key:bank[name] for key,name in parameters.items()}


def polynomial_audit(path,mapping):
    """Expand a small pure graph, then test the ordered matrix2 specification.

    Coefficients are the exact binary rationals actually exported to FP32.
    Unsupported nonlinear operations produce an explicit instrument refusal.
    """
    if mapping['input_width']!=8 or mapping['values'][mapping['output_value']]['length']!=4:
        return {'status':'not-applicable','reason':'matrix2 candidate requires two ordered four-value inputs'}
    weights=read_program(path,mapping)
    def clean(poly):return {key:value for key,value in poly.items() if value}
    def add(a,b):
        out=dict(a)
        for key,value in b.items():out[key]=out.get(key,Fraction(0))+value
        return clean(out)
    def mul(a,b):
        out={}
        for ka,va in a.items():
            for kb,vb in b.items():
                key=tuple(sorted(ka+kb))
                if len(key)>2:raise ValueError('polynomial degree exceeds bounded quadratic analyzer')
                out[key]=out.get(key,Fraction(0))+va*vb
        return clean(out)
    def constant(x):return clean({():Fraction(float(np.float32(x)))})
    values={0:[{(i,):Fraction(1)} for i in range(8)]}
    try:
        for op in mapping['instructions']:
            kind=op['kind'];a=values[op['a']];b=values[op['b']]
            if kind=='slice':out=a[op['offset']:op['offset']+mapping['values'][op['out']]['length']]
            elif kind=='concat':out=a+b
            elif kind=='constant':out=[constant(x) for x in weights[op['w']].flat]
            elif kind=='affine':out=[add(mul(x,constant(op['scale'])),constant(op['shift'])) for x in a]
            elif kind in ('add','multiply'):
                fn=add if kind=='add' else mul
                out=[fn(a[0 if len(a)==1 else i],b[0 if len(b)==1 else i]) for i in range(max(len(a),len(b)))]
            elif kind=='linear':
                out=[]
                for i,row in enumerate(weights[op['w']]):
                    total=constant(weights[op['bias']][i]) if op['bias'] is not None else {}
                    for coefficient,term in zip(row,a,strict=True):total=add(total,mul(constant(coefficient),term))
                    out.append(total)
            else:raise ValueError('unsupported polynomial instruction: '+kind)
            values[op['out']]=out
    except ValueError as error:return {'status':'instrument-not-applicable','reason':str(error)}
    output=values[mapping['output_value']]
    bilinear=all(len(key)==2 and key[0]<4<=key[1] for row in output for key in row)
    if not bilinear:return {'status':'not-bilinear-under-declared-ports','polynomial_identity':digest([{str(k):str(v) for k,v in row.items()} for row in output])}
    coefficients=[]
    for k,row in enumerate(output):
        for i in range(4):
            for j in range(4):
                actual=row.get((i,j+4),Fraction(0));expected=int(i//2==k//2 and j%2==k%2 and i%2==j//2)
                coefficients.append({'out':k,'left':i,'right':j,'actual':str(actual),'expected':expected,'residual':str(actual-expected)})
    residuals=[abs(Fraction(row['residual'])) for row in coefficients]
    return {'status':'exact-ordered-matrix2-polynomial' if not any(residuals) else 'approximate-ordered-matrix2-polynomial',
            'bilinearity_established':True,'coefficient_identity':digest(coefficients),'coefficients':coefficients,
            'nonzero_residuals':sum(bool(x) for x in residuals),'max_abs_coefficient_residual':float(max(residuals)),
            'specification':'supplied row-major ordered 2x2 multiplication; left input columns0:4, right4:8',
            'numerical_scope':'formal rational polynomial of deployed FP32 coefficients; not an unrestricted floating execution or uint64 certificate'}


def factor_audit(rule):
    """Audit original JSON coefficients without silently narrowing or rounding."""
    u,v,w=[rule[k] for k in ('u','v','w')];rows=[]
    for o in range(4):
        for i in range(4):
            for j in range(4):
                value=sum((Fraction(x)*Fraction(y)*Fraction(z) for x,y,z in zip(w[o],[r[i] for r in u],[r[j] for r in v],strict=True)),Fraction(0))
                expected=int(i//2==o//2 and j%2==o%2 and i%2==j//2)
                rows.append({'out':o,'left':i,'right':j,'actual':str(value),'expected':expected,'residual':str(value-expected)})
    return {'coefficient_identity':digest(rows),'nonzero_residuals':sum(Fraction(r['residual'])!=0 for r in rows),
            'max_abs_coefficient_residual':max(float(abs(Fraction(r['residual']))) for r in rows),'coefficients':rows}


class CExecutor:
    """Small-bundle research client; every call is charged, with no native fallback."""
    def __init__(self,binary,ledger=None):
        self.binary=Path(binary).resolve();self.ledger=Path(ledger) if ledger else None
        self.calls=0;self.seconds=0.
    def run(self,path,rows,*,zero=None):
        rows=np.asarray(rows,dtype='<f4')
        if rows.ndim!=2 or not len(rows):raise ValueError('C executor needs a nonempty row batch')
        start=time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='tcm-math-call-') as tmp:
            root=Path(tmp);rows.tofile(root/'input.f32')
            cmd=[str(self.binary),'program',str(Path(path).resolve()),'--input',str(root/'input.f32'),'--output',str(root/'output.f32')]
            if zero is not None:cmd+=['--zero',str(zero)]
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            if run.returncode:raise RuntimeError('C math call failed: '+run.stderr[-3000:])
            record=json.loads(run.stdout);out=np.fromfile(root/'output.f32',dtype='<f4').reshape(len(rows),-1)
        elapsed=time.perf_counter()-start;self.calls+=1;self.seconds+=elapsed
        if self.ledger:
            with self.ledger.open('a') as stream:stream.write(json.dumps({'call':self.calls,'rows':len(rows),'input_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),
                'output_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'wall_seconds_including_process':elapsed,'run':record})+'\n')
        return out


class ProjectedField:
    """Compile each current immutable record view to generic equality filters.

    UTF-8 strings become content-bound integer coordinates outside C. C filters
    actual records; the native FieldFabric consumer extends its own frames.
    """
    def __init__(self,view,root,executor):
        self.view,self.root,self.executor=view,Path(root),executor
        self.root.mkdir(parents=True,exist_ok=True);self.native_lookup=view.lookup
        self.banks={};self.calls=0;self.comparisons=[];self.memo={}
        coordinates=view.state() if hasattr(view,'state') else {'adapter':'generated-test-view'}
        coordinates.pop('records',None)
        self.coordinate_identity=digest(coordinates)
    def lookup(self,relation,bindings,*,namespace,scope):
        records=self.view.records
        if not records:return []
        state=[record.__dict__ for record in records];identity=digest(state)
        if identity not in self.banks:
            import torch
            arity=max(len(r.roles) for r in records)
            fields=[(r.namespace,r.scope,int(r.relation),*r.roles) for r in records]
            dictionary={text:i+1 for i,text in enumerate(sorted({json.dumps(value,sort_keys=True) for row in fields for value in row}))}
            if len(dictionary)>2**24:raise ValueError('projected dictionary exceeds exact FP32 integer coordinates')
            table=np.full((len(records),3+arity),-2,dtype=np.float32)
            for i,row in enumerate(fields):
                for j,value in enumerate(row):table[i,j]=dictionary[json.dumps(value,sort_keys=True)]
            path=self.root/('field-'+identity+'.tcm')
            if not path.exists():
                p=Program(identity,3+arity,origin={'front_end':'fieldlm-projected-records','records_sha256':identity,
                    'native_coordinates_sha256':self.coordinate_identity,
                    'scope':'native parser and frame extension remain on host; C executes relation/namespace/scope/role equality',
                    'missing_role':-2,'wildcard':-1})
                weight=p.parameter('records',torch.from_numpy(table))
                p.output=p.emit('match-rows',len(records),w=weight,source='fieldlm://'+identity+'/projected-read')
                p.save(path)
                (self.root/('field-'+identity+'.records.json')).write_bytes(canonical({'records':state,'dictionary':dictionary})+b'\n')
            self.banks[identity]=(path,dictionary,arity)
        path,dictionary,arity=self.banks[identity]
        query=np.full(3+arity,-1,dtype=np.float32)
        for j,value in enumerate((namespace,scope,int(relation))):query[j]=dictionary.get(json.dumps(value,sort_keys=True),0)
        for axis,value in bindings.items():
            if self.view.active_axes is not None and axis not in self.view.active_axes:continue
            if type(axis) is not int or axis<0:raise ValueError('invalid FieldLM query axis')
            if axis>=arity:query[0]=0;continue
            query[3+axis]=dictionary.get(json.dumps(value,sort_keys=True),0)
        key=(identity,tuple(int(x) for x in query))
        if key not in self.memo:
            mask=self.executor.run(path,query[None,:])[0]
            if not np.isin(mask,[0,1]).all():raise ValueError('invalid C projected-read mask')
            self.memo[key]=([i for i,x in enumerate(mask) if x],len(self.comparisons))
            self.comparisons.append({'bank':identity,'query':list(key[1]),'selected':self.memo[key][0],
                                     'native_equal':True,'logical_reads':0})
        indices,entry=self.memo[key];found=[records[i] for i in indices]
        self.calls+=1
        # Audit only: the required consumer receives C-selected records.
        expected=self.native_lookup(relation,bindings,namespace=namespace,scope=scope)
        self.comparisons[entry]['native_equal'] &= found==expected
        self.comparisons[entry]['logical_reads']+=1
        return found


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Audit a captured vector program against the supplied ordered matrix2 specification')
    parser.add_argument('program',type=Path);args=parser.parse_args()
    mapping=json.loads(args.program.with_suffix('.map.json').read_text())
    print(json.dumps(polynomial_audit(args.program,mapping),indent=2))
