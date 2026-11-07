"""Small model-free checks of the signed nonlinear decomposition instrument."""
import unittest
import numpy as np
from analysis import silu,terms

class DecompositionTest(unittest.TestCase):
    def test_exact_finite_difference_and_signed_gram(self):
        rng=np.random.default_rng(1126);d=rng.normal(size=(7,11));g,u,gh,uh=rng.normal(size=(4,11))
        delta=d@(silu(g)*u-silu(gh)*uh);t=terms(d,g,u,gh,uh)
        np.testing.assert_allclose(t.sum(axis=1),delta,rtol=1e-13,atol=1e-13)
        self.assertAlmostEqual(float((t.T@t).sum()),float(delta@delta),places=10)
    def test_restoring_each_operand_removes_its_error(self):
        rng=np.random.default_rng(17);d=rng.normal(size=(3,5));g,u,gh,uh=rng.normal(size=(4,5))
        np.testing.assert_array_equal(terms(d,g,u,g,u),np.zeros((3,3)))
        np.testing.assert_array_equal(terms(d,g,u,g,uh)[:,[0,2]],np.zeros((3,2)))
        np.testing.assert_array_equal(terms(d,g,u,gh,u)[:,[1,2]],np.zeros((3,2)))
    def test_silu_extremes_are_finite(self):
        y=silu(np.array([-1000.,0.,1000.]));self.assertTrue(np.isfinite(y).all());self.assertEqual(y[2],1000.)

if __name__=='__main__':unittest.main()
