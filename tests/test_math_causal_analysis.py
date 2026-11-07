"""Independent arithmetic work instruments use structural sharing and domains."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_math_causal import expression_work


class CausalWorkAnalysis(unittest.TestCase):
    def test_shared_subexpression_is_charged_once_but_both_products_remain(self):
        expression=['prod','v1',['sum','v0','v1'],['sum','v0','v1']]
        self.assertEqual(expression_work(expression,'integer-commutative'),(3,{'add':1,'mul':2}))
    def test_matrix_domain_charges_declared_scalar_work(self):
        expression=['prod',['sum','A','B'],['sum','C','D']]
        self.assertEqual(expression_work(expression,'integer-commutative'),(3,{'add':2,'mul':1}))
        self.assertEqual(expression_work(expression,'matrix2-noncommutative'),(20,{'add':2,'mul':1}))


if __name__=='__main__':unittest.main()
