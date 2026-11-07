import unittest
import numpy as np
from analysis import field_values, eliminate, PRIME

class ExactRank(unittest.TestCase):
    def test_dyadics_and_signs(self):
        values = field_values(np.array([0.,-0.,1.,-1.,.5,-.75],'<f4')).tolist()
        self.assertEqual(values, [0,0,1,PRIME-1,pow(2,PRIME-2,PRIME),-3*pow(4,PRIME-2,PRIME)%PRIME])
    def test_subnormal(self):
        value = np.array([1],'<u4').view('<f4')
        self.assertEqual(int(field_values(value)[0]),pow(2,(-149)%(PRIME-1),PRIME))
    def test_nonfinite(self):
        with self.assertRaises(ValueError):field_values([float('inf')])
    def test_elimination(self):
        self.assertEqual(eliminate(np.eye(4,dtype=np.int64))['determinant_mod_prime'],1)
        self.assertEqual(eliminate([[1,2],[2,4]])['rank'],1)
        self.assertEqual(eliminate([[0,2],[3,4]])['determinant_mod_prime'],PRIME-6)
        self.assertEqual(eliminate([[1,PRIME-1],[PRIME-1,1]])['rank'],1)
    def test_overflow_bound(self):
        self.assertLess((PRIME-1)**2+PRIME,2**63)

if __name__=='__main__':unittest.main()
