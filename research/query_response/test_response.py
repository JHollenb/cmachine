import unittest
import numpy as np
from compiler import rope,response,reader_pullback,source_states

class ResponseMath(unittest.TestCase):
    def test_reader_gradient_with_count_mass(self):
        r=np.random.default_rng(41);q=r.normal(size=128);k=r.normal(size=(5,128));v=r.normal(size=(5,128));mass=np.asarray([8.,3.,1.,1.,1.]);target=r.normal(size=128);direction=r.normal(size=128)
        a,_=response(q,k,v,mass);grad=reader_pullback(q,k,v,mass,a-target)
        def loss(x):
            error=response(x,k,v,mass)[0]-target;return .5*error@error
        eps=1e-5;fd=(loss(q+eps*direction)-loss(q-eps*direction))/(2*eps)
        self.assertAlmostEqual(fd,float(grad@direction),places=8)
    def test_half_split_rotation_and_adjoint(self):
        r=np.random.default_rng(3);x=r.normal(size=(4,1536));g=r.normal(size=x.shape);pos=np.asarray([0,1,21,63]);rot=rope(x,pos)
        np.testing.assert_allclose(rope(rot,pos,True),x,atol=2e-15);self.assertAlmostEqual(float(np.sum(g*rot)),float(np.sum(rope(g,pos,True)*x)),places=11)
        only=np.zeros((1,128));only[0,0]=1.;y=rope(only,[1]);self.assertAlmostEqual(y[0,0],np.cos(1.));self.assertAlmostEqual(y[0,64],np.sin(1.));self.assertEqual(np.count_nonzero(y),2)
    def test_source_trajectories_reset_memory_and_clock(self):
        k=np.zeros((5,256));v=np.arange(5.)[:,None]*np.ones((5,256));nodes,ranges,positions=source_states(k,v,k.astype('<f4'),[3,2])
        self.assertEqual(positions.tolist(),[0,1,2,0,1]);self.assertEqual(ranges,[(0,1),(0,2),(0,3),(3,4),(3,5)]);self.assertEqual(nodes[3][0][1].shape,(1,128));self.assertEqual(nodes[3][0][1][0,0],3.)
    def test_one_key_has_no_query_gradient(self):
        q=np.ones(128);k=np.ones((1,128));v=np.full((1,128),2.)
        np.testing.assert_array_equal(reader_pullback(q,k,v,np.ones(1),np.ones(128)),np.zeros(128))
if __name__=='__main__':unittest.main()
