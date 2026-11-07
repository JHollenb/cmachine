"""Model-free math checks for the declared attention account."""
import unittest,sys
from pathlib import Path
import numpy as np
from analysis import interaction,mm,head_mm,value_terms,softmax64,normalizer_response

class AttentionMath(unittest.TestCase):
    def test_receipt_bridge_import_closure(self):
        sys.path.insert(0,str(Path(__file__).resolve().parent/"public"))
        from saturn_bridge import invoke
        self.assertTrue(callable(invoke))
    def test_bilinear_support_and_gqa(self):
        rng=np.random.default_rng(31);q0=rng.normal(size=(12,8));dq=rng.normal(size=(12,8));k0=rng.normal(size=(2,7,8));da=np.zeros_like(k0);db=np.zeros_like(k0);da[:,2:4]=rng.normal(size=(2,2,8));db[:,4:]=rng.normal(size=(2,3,8))
        qs=[q0,q0,q0+dq,q0+dq];ks=[k0,k0+da,k0+db,k0+da+db];ks=[k[np.arange(12)//6] for k in ks]
        s=[head_mm(q[:,None],k.transpose(0,2,1))[:,0] for q,k in zip(qs,ks)];prediction=head_mm(dq[:,None],da[np.arange(12)//6].transpose(0,2,1))[:,0]
        np.testing.assert_allclose(interaction(s),prediction,rtol=1e-13,atol=1e-13);np.testing.assert_array_equal(prediction[:,:2],0);np.testing.assert_array_equal(prediction[:,4:],0)
    def test_general_value_factorial(self):
        rng=np.random.default_rng(4);p=[rng.normal(size=(3,5)) for _ in range(4)];v=[rng.normal(size=(3,5,7)) for _ in range(4)]
        y=[head_mm(a[:,None],b)[:,0] for a,b in zip(p,v)];np.testing.assert_allclose(interaction(y),sum(value_terms(p,v)),rtol=1e-13,atol=1e-13)
    def test_softmax_spreads_off_band(self):
        a=np.zeros((1,4));b=a.copy();b[0,1]=2
        self.assertNotEqual(float((softmax64(b)-softmax64(a))[0,0]),0)
    def test_fsum_projection_orientation(self):
        np.testing.assert_array_equal(mm([[1,2],[3,4]],[[5],[7]]),[[19],[43]])
    def test_conditional_normalizer_update(self):
        rng=np.random.default_rng(9);s=rng.normal(size=(3,7));t=s.copy();t[:,2:4]+=2;v=rng.normal(size=(3,7,5));w=v.copy();w[:,2:4]+=rng.normal(size=(3,2,5))
        r=normalizer_response(s,t,v,w,slice(2,4));np.testing.assert_allclose(r["direct"],r["prediction"],rtol=1e-13,atol=1e-13)
        off=[0,1,4,5,6];np.testing.assert_allclose(softmax64(t)[:,off],r["ratio"][:,None]*softmax64(s)[:,off],rtol=1e-13,atol=1e-13)

if __name__=="__main__":unittest.main()
