import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from formation_analysis import consumer_metrics
from verify import cancellation_bound


class FormationAnalysisMechanics(unittest.TestCase):
    def test_native_and_full_carried_are_distinct_references(self):
        arrays={"project":np.array([0.,0.]),"carry":np.array([1.,0.]),
                "read":np.array([0.,1.]),"correct":np.array([1.,1.]),"dense":np.array([4.,5.])}
        current=consumer_metrics(arrays)
        self.assertEqual(current["full_carried_reference_error_l2"]["dense"],5.)
        self.assertEqual(current["full_carried_reference_error_l2"]["correct"],0.)
        self.assertEqual(current["native_error_l2"]["correct"],5.)
        self.assertEqual(consumer_metrics(arrays,1)["full_carried_reference_error_l2"]["dense"],0.)

    def test_roundoff_contract_scales_but_rejects_wrong_identity(self):
        record={"joint_squared_norm":2e5,"term_squared_norms":{"carry":1e5,"read":1e5,"interaction":0.},
                "twice_pair_inner_products":{"carry / read":0.,"carry / interaction":0.,"read / interaction":0.},"roundoff":1e-8}
        bound=cancellation_bound(record,151936,"roundoff")
        self.assertGreater(bound,abs(record["roundoff"]))
        bad={**record,"roundoff":0.1}
        with self.assertRaises(ValueError):cancellation_bound(bad,151936,"roundoff")
        with self.assertRaises(ValueError):cancellation_bound(record,0,"roundoff")
        with self.assertRaises(ValueError):cancellation_bound({**record,"roundoff":float("nan")},151936,"roundoff")


if __name__=="__main__":unittest.main()
