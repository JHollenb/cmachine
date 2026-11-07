"""Generated mechanics only. Archived math models execute through mrun."""
from dataclasses import dataclass
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from program import Program,lower_fx
from math_program import CExecutor,ProjectedField,polynomial_audit,digest,read_program


class Product(nn.Module):
    """Generated rank-eight schoolbook cell, unrelated to learned checkpoints."""
    def __init__(self,perturb=0.):
        super().__init__()
        u=torch.zeros(8,4);v=torch.zeros(8,4);w=torch.zeros(4,8)
        for k in range(4):
            for j in range(2):
                q=2*k+j;u[q,2*(k//2)+j]=1;v[q,2*j+k%2]=1;w[k,q]=1
        w[0,0]+=perturb
        self.u=nn.Parameter(u);self.v=nn.Parameter(v);self.w=nn.Parameter(w)
    def forward(self,x):
        a=x[:,:4];b=x[:,4:]
        return ((a@self.u.T)*(b@self.v.t()))@self.w.transpose(0,1)


@dataclass(frozen=True)
class Record:
    namespace:str
    scope:str
    relation:int
    roles:tuple
    source:str='toy'
    position:int=0


class View:
    active_axes=None
    def __init__(self):
        self.records=[Record('n','s',3,('a','b')),Record('n','s',3,('a',)),
                      Record('n','other',3,('a','b')),Record('other','s',4,('b','a'))]
    def lookup(self,relation,bindings,*,namespace,scope):
        return [r for r in self.records if (r.namespace,r.scope,r.relation)==(namespace,scope,relation)
                and all(axis<len(r.roles) and r.roles[axis]==value for axis,value in bindings.items()
                        if self.active_axes is None or axis in self.active_axes)]


class MathMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(['make','-C',str(ROOT)],capture_output=True,check=True)
        torch.set_num_threads(1)
    def executor(self):return CExecutor(os.environ.get('TCM_TEST_EXECUTABLE',str(ROOT/'build/tcm')))

    def test_polynomial_proof_then_classification_and_source_C_execution(self):
        rows=np.random.default_rng(8).integers(-3,4,(24,8)).astype(np.float32)
        truth=(rows[:,:4].reshape(-1,2,2)@rows[:,4:].reshape(-1,2,2)).reshape(-1,4)
        with tempfile.TemporaryDirectory() as tmp:
            for label,model,classification in [('exact',Product(),'exact'),('near',Product(.125),'approximate')]:
                path=Path(tmp)/(label+'.tcm');mapping=lower_fx(model,8,'1'*64,path)
                certificate=polynomial_audit(path,mapping)
                self.assertTrue(certificate['bilinearity_established'])
                self.assertEqual(len(certificate['coefficients']),64)
                self.assertEqual(certificate['status'],classification+'-ordered-matrix2-polynomial')
                with torch.no_grad():expected=model(torch.from_numpy(rows)).numpy()
                actual=self.executor().run(path,rows)
                np.testing.assert_array_equal(actual,expected)
                if label=='exact':np.testing.assert_array_equal(actual,truth)
            raw=path.read_bytes();path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError,'identity differs'):polynomial_audit(path,mapping)

    def test_resealed_sidecar_cannot_change_executed_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'exact.tcm';mapping=lower_fx(Product(),8,'8'*64,path)
            def changed(edit):
                other=copy.deepcopy(mapping);edit(other);other.pop('fingerprint');other['fingerprint']=digest(other)
                with self.assertRaisesRegex(ValueError,'binary/source-map'):read_program(path,other)
            changed(lambda m:m['instructions'][-2].update(kind='add'))
            changed(lambda m:m['instructions'][-1].update(w=m['instructions'][2]['w']))
            changed(lambda m:m['values'][1].update(offset=1))
            changed(lambda m:m.update(output_value=1))
            changed(lambda m:m.update(source_identity='9'*64))

    def test_nonlinear_graph_refuses_polynomial_instrument(self):
        model=nn.Sequential(Product(),nn.ReLU())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'relu.tcm';mapping=lower_fx(model,8,'2'*64,path)
            result=polynomial_audit(path,mapping)
            self.assertEqual(result['status'],'instrument-not-applicable')
            self.assertIn('relu',result['reason'])

    def test_normalization_and_two_scorer_shapes(self):
        class Scorer(nn.Module):
            def __init__(self,width):
                super().__init__();self.register_buffer('mean',torch.linspace(-1,1,width))
                self.register_buffer('std',torch.linspace(.2,2,width));self.net=nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,1))
            def forward(self,x):return self.net((x-self.mean)/self.std).squeeze(-1)
        with tempfile.TemporaryDirectory() as tmp:
            for width in (21,23):
                torch.manual_seed(width);model=Scorer(width).eval();path=Path(tmp)/f'scorer{width}.tcm'
                mapping=lower_fx(model,width,'3'*64,path)
                self.assertEqual([op['kind'] for op in mapping['instructions']].count('divide'),1)
                rows=np.random.default_rng(width).normal(size=(64,width)).astype(np.float32)
                with torch.no_grad():expected=model(torch.from_numpy(rows)).numpy()
                actual=self.executor().run(path,rows).reshape(-1)
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)

    def test_projected_read_namespace_arity_active_axes_mutation_and_memo(self):
        view=View();executor=self.executor()
        with tempfile.TemporaryDirectory() as tmp:
            reader=ProjectedField(view,Path(tmp),executor)
            for relation,bindings,namespace,scope in [(3,{},'n','s'),(3,{1:'b'},'n','s'),
                    (3,{0:'unknown'},'n','s'),(3,{2:'a'},'n','s'),(3,{},'n','other'),(4,{0:'b'},'other','s')]:
                self.assertEqual(reader.lookup(relation,bindings,namespace=namespace,scope=scope),
                                 view.lookup(relation,bindings,namespace=namespace,scope=scope))
            calls=executor.calls;reader.lookup(3,{},namespace='n',scope='s');self.assertEqual(calls,executor.calls)
            view.active_axes={0}
            self.assertEqual(reader.lookup(3,{1:'absent'},namespace='n',scope='s'),view.records[:2])
            view.records.append(Record('n','s',3,('x','y')))
            self.assertEqual(len(reader.lookup(3,{},namespace='n',scope='s')),3)
            self.assertEqual(len(reader.banks),2)
            self.assertTrue(all(r['native_equal'] for r in reader.comparisons))

    def test_refuse_bad_keys_nonfinite_and_divide_by_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Program('4'*64,1);w=p.parameter('keys',torch.tensor([[1.],[2.]]));p.output=p.emit('match-rows',2,w=w)
            path=Path(tmp)/'keys.tcm';p.save(path)
            for rows in ([[.5]],[[float('nan')]],[[float('inf')]], [[-2.]]):
                with self.assertRaisesRegex(RuntimeError,'C math call failed'):self.executor().run(path,rows)
            bad=Program('5'*64,1);w=bad.parameter('keys',torch.tensor([[-1.]]));bad.output=bad.emit('match-rows',1,w=w)
            other=Path(tmp)/'badkeys.tcm';bad.save(other)
            with self.assertRaisesRegex(RuntimeError,'shape or state'):self.executor().run(other,[[1.]])
            class Divide(nn.Module):
                def forward(self,x):return x/0.
            path=Path(tmp)/'zero.tcm';lower_fx(Divide(),1,'6'*64,path)
            with self.assertRaisesRegex(RuntimeError,'non-finite operation'):self.executor().run(path,[[1.]])

    def test_reject_shape_change_outside_vector_abi(self):
        class TransposedRows(nn.Module):
            def forward(self,x):return x.reshape(2,2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'single-row vector ABI'):
                lower_fx(TransposedRows(),4,'7'*64,Path(tmp)/'bad.tcm')
            class ConstantMatrix(nn.Module):
                def __init__(self):super().__init__();self.register_buffer('matrix',torch.eye(2))
                def forward(self,x):return self.matrix
            with self.assertRaisesRegex(ValueError,'single-row vector ABI'):
                lower_fx(ConstantMatrix(),4,'7'*64,Path(tmp)/'bad-constant.tcm')

if __name__=='__main__':unittest.main()
