import unittest
import numpy as np
from analysis import Memory,full_read,vector_factorial,probability_row,rel
from audit import MembershipMemory
class MemoryMath(unittest.TestCase):
    def test_relative_metric_promotes_reference_before_norm(self):
        reference=np.asarray([1e20,3e20],dtype='<f4')
        self.assertAlmostEqual(rel(2*reference,reference),1.,places=14)
    def test_mass_and_values(self):
        m=Memory();ks=[];vs=[]
        for t in range(65):
            k=np.full((2,128),t%7-3,dtype='<f4');v=np.full((2,128),t%11-5,dtype='<f4');ks.append(k);vs.append(v);m.append(k,v)
            self.assertTrue(np.all(m.count.sum(1)+m.recent==t+1))
        out=m.read(np.zeros((12,128),'<f4'));self.assertLess(float(np.max(np.abs(out-np.mean(vs,axis=0,dtype=float)[0]))),1e-7);self.assertEqual(len(m.bytes()),98464)
    def test_exact_before_merge(self):
        m=Memory();r=np.random.default_rng(7);k=r.normal(size=(20,2,128)).astype('<f4');v=r.normal(size=(20,2,128)).astype('<f4');q=r.normal(size=(12,128)).astype('<f4')
        for key,value in zip(k,v):m.append(key,value)
        self.assertTrue(np.array_equal(m.read(q),full_read(q,k,v).astype('<f4')))
    def test_query_changes_read(self):
        m=Memory();a=np.zeros((2,128),'<f4');b=a.copy();a[:,0]=2;b[:,0]=-2;m.append(a,a);m.append(b,b);q=np.zeros((12,128),'<f4');q[:,0]=8
        self.assertTrue(np.all(m.read(q)[:,0]>0));self.assertTrue(np.all(m.read(-q)[:,0]<0))
    def test_nonzero_address_value_interaction(self):
        pn=np.asarray([.75,.25]);pc=pn[::-1];vn=np.asarray([[1.,2.],[3.,4.]]);vc=np.asarray([[2.,1.],[7.,0.]])
        nn,cn,nc,cc=pn@vn,pc@vn,pn@vc,pc@vc
        np.testing.assert_array_equal(cc-cn-nc+nn,[1.5,-1.5]);np.testing.assert_array_equal((pc-pn)@(vc-vn),[1.5,-1.5])
        result=vector_factorial(nn,cn,nc,cc);self.assertGreater(result['interaction_relative'],0);self.assertEqual(result['vector_closure_relative'],0.)
    def test_triangular_probability_clock(self):
        data=np.concatenate([np.full(12*(t+1),t,dtype='<f4') for t in range(5)])
        for t in range(5):
            value=probability_row(data,t);self.assertEqual(value.shape,(12,t+1));self.assertTrue(np.all(value==t))
    def test_lifted_weights_and_value_memory_interaction(self):
        rng=np.random.default_rng(91);k=rng.normal(size=(65,2,128)).astype('<f4');vn=rng.normal(size=k.shape).astype('<f4');vc=(vn+rng.normal(size=k.shape)*.5).astype('<f4');q=rng.normal(size=(12,128)).astype('<f4')
        mn,mc=MembershipMemory(),MembershipMemory()
        for key,n,c in zip(k,vn,vc):mn.append(key,n);mc.append(key,c)
        self.assertEqual(mn.key_signature(),mc.key_signature());w=mn.weights(q);np.testing.assert_allclose(w.sum(1),1,atol=2e-15)
        native=[];compiled=[];full_weights=[]
        for h in range(12):
            score=k[:,h//6].astype(float)@q[h].astype(float)/np.sqrt(128.);p=np.exp(score-score.max());p/=p.sum();full_weights.append(p)
            native.append(w[h]@vn[:,h//6].astype(float));compiled.append(w[h]@vc[:,h//6].astype(float))
        np.testing.assert_array_equal(np.asarray(native,dtype='<f4'),mn.read(q));np.testing.assert_array_equal(np.asarray(compiled,dtype='<f4'),mc.read(q))
        p=np.asarray(full_weights);actual=np.asarray(compiled)-full_read(q,k,vc)-np.asarray(native)+full_read(q,k,vn)
        formula=np.asarray([(w[h]-p[h])@(vc[:,h//6].astype(float)-vn[:,h//6].astype(float)) for h in range(12)])
        np.testing.assert_allclose(actual,formula,rtol=1e-12,atol=1e-15);self.assertGreater(float(np.linalg.norm(formula)),0)
    def test_query_memory_interaction_on_identical_state(self):
        rng=np.random.default_rng(72);k=rng.normal(size=(65,2,128)).astype('<f4');v=rng.normal(size=k.shape).astype('<f4');qs=[rng.normal(size=(12,128)).astype('<f4') for _ in range(2)];m=MembershipMemory()
        for key,value in zip(k,v):m.append(key,value)
        weights=[]
        for q in qs:
            full=[]
            for h in range(12):
                score=k[:,h//6].astype(float)@q[h].astype(float)/np.sqrt(128.);p=np.exp(score-score.max());full.append(p/p.sum())
            weights.extend([np.asarray(full),m.weights(q)])
        mixed=[np.asarray([p[h]@v[:,h//6].astype(float) for h in range(12)]) for p in weights]
        actual=[full_read(qs[0],k,v),m.read(qs[0]).astype(float),full_read(qs[1],k,v),m.read(qs[1]).astype(float)]
        effect=actual[3]-actual[2]-actual[1]+actual[0];formula=np.asarray([(weights[3][h]-weights[1][h]-weights[2][h]+weights[0][h])@v[:,h//6].astype(float) for h in range(12)])
        remainders=[a-b for a,b in zip(actual,mixed)];finite=remainders[3]-remainders[2]-remainders[1]+remainders[0]
        np.testing.assert_allclose(effect-formula-finite,0,atol=1e-15);self.assertGreater(float(np.linalg.norm(formula)),0)
if __name__=='__main__':unittest.main()
