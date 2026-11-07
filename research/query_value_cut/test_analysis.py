import unittest
import numpy as np
from analysis import Memory,full_read,vector_factorial,probability_row
class MemoryMath(unittest.TestCase):
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
if __name__=='__main__':unittest.main()
