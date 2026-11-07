import unittest
import numpy as np
from compiler import fit_image
class QueryCompiler(unittest.TestCase):
    def test_predictable_omission_and_fixed_interface_limit(self):
        # x2 is predictable from retained x0/x1 in discovery, but not universally.
        z=np.asarray([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]])
        x=np.column_stack((z,2*z[:,0]-z[:,1]));b=np.asarray([[1.,0.,0.],[0.,1.,0.]])
        w=np.asarray([[0.,0.,1.],[1.,1.,2.]])
        fitted,detail=fit_image(x,b,w);prediction=x@b.T@fitted.T
        np.testing.assert_allclose(prediction,x@w.T,atol=1e-6);self.assertEqual(detail['rank'],2)
        # Equal codes conceal a changed third coordinate: no refit can recover both.
        a,c=np.asarray([1.,1.,0.]),np.asarray([1.,1.,5.])
        np.testing.assert_array_equal(b@a,b@c);self.assertFalse(np.array_equal(w@a,w@c))
    def test_rank_deficient_features_refuse(self):
        with self.assertRaises(ValueError):fit_image(np.ones((4,3)),np.eye(3)[:2],np.eye(3))
if __name__=='__main__':unittest.main()
