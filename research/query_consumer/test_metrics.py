import unittest
import numpy as np
from metrics import distribution,choose
from reader import Memory
class ConsumerMetrics(unittest.TestCase):
    def test_logit_shift_changes_l2_not_distribution(self):
        a=np.asarray([2.,-1.,0.]);r=distribution(a,a+7.)
        self.assertAlmostEqual(r['kl_nats'],0.,places=14);self.assertAlmostEqual(r['native_top_logp_delta_nats'],0.,places=14);self.assertGreater(r['relative_logit_l2'],0.)
    def test_kl_direction_and_vocab_not_averaged(self):
        a=np.asarray([5.,1.,-2.]);b=a[::-1];r=distribution(a,b);p=np.exp(a-a.max());p/=p.sum();q=np.exp(b-b.max());q/=q.sum()
        self.assertAlmostEqual(r['kl_nats'],float(np.sum(p*np.log(p/q))),places=13);self.assertGreater(r['kl_nats'],6.)
    def test_ties_use_smallest_abs_dose_then_index(self):
        self.assertEqual(choose([1.,1.,1.,1.,1.,1.]),0);self.assertEqual(choose([2.,1.,1.,1.,1.,1.]),2);self.assertEqual(choose([2.,2.,2.,2.,1.,1.]),4)
    def test_fp64_norm_and_bad_operands(self):
        x=np.asarray([1e20,3e20],dtype='<f4');self.assertAlmostEqual(distribution(x,2*x)['relative_logit_l2'],1.)
        with self.assertRaises(ValueError):distribution([float('nan')],[0.])
    def test_memory_reader_unchanged(self):
        m=Memory();z=np.zeros((2,128),'<f4')
        for t in range(65):m.append(z,np.full((2,128),t,dtype='<f4'))
        np.testing.assert_array_equal(m.read(np.zeros((12,128),'<f4')),np.full((12,128),32,dtype='<f4'));self.assertEqual(len(m.bytes()),98464)
if __name__=='__main__':unittest.main()
