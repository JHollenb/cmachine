import unittest
import numpy as np
from analysis import factorial,histories_match

class FactorialMechanics(unittest.TestCase):
    def test_signed_interaction_and_conditional_effect(self):
        n=np.array([1.,2.]);a=np.array([2.,-1.]);b=np.array([-1.,1.]);i=np.array([.5,-.25]);r=factorial(n,n+a,n+b,n+a+b+i)
        self.assertAlmostEqual(r["norms"]["interaction"],np.linalg.norm(i));self.assertAlmostEqual(r["norms"]["question_after_project_formation"],np.linalg.norm(b+i));self.assertAlmostEqual(r["power_closure_residual"],0.)
    def test_zero_joint_keeps_cancelling_effects(self):
        n=np.array([1.,2.]);a=np.array([2.,-1.]);r=factorial(n,n+a,n-a,n)
        self.assertIsNone(r["interaction_over_joint"]);self.assertGreater(r["norms"]["formation"],0);self.assertLess(r["signed_energy"]["formation_question_cross"],0)
    def test_prefill_and_generated_histories_are_distinct(self):
        a={"parent_history_ids":[1],"steps":[{"input_id":2}],"generation_steps":[{"input_id":3}]};b={**a,"generation_steps":[{"input_id":4}]}
        self.assertTrue(histories_match([a,b],"prefill"));self.assertFalse(histories_match([a,b],"final"))
    def test_invalid_array_geometry_is_refused(self):
        with self.assertRaisesRegex(ValueError,"Invalid"):factorial([1],[1,2],[1],[1])

if __name__=="__main__":unittest.main()
