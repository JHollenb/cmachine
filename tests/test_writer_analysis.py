import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from writer_analysis import factorial,projected
from worker import TRACE_PORT_NAMES,validate_trace_ports


class WriterAnalysisTests(unittest.TestCase):
    def test_trace_contract_rejects_missing_or_duplicate_consumed_port(self):
        rows=[{"port":p} for p in TRACE_PORT_NAMES]
        validate_trace_ports(rows)
        with self.assertRaises(ValueError):validate_trace_ports(rows[:-1])
        with self.assertRaises(ValueError):validate_trace_ports(rows[:-1]+[rows[0]])

    def test_sequential_policy_interaction(self):
        # Both's M is conditional on attention: it is not a sum of fixed writes.
        a={k:np.asarray([v],dtype=np.float32) for k,v in {"00":1,"10":4,"01":6,"11":12}.items()}
        out=factorial(a)["effects"]
        self.assertEqual(out["attention_isolated"]["l2"],3)
        self.assertEqual(out["mlp_isolated"]["l2"],5)
        self.assertEqual(out["attention_given_mlp"]["l2"],6)
        self.assertEqual(out["mlp_given_attention"]["l2"],8)
        self.assertEqual(out["interaction"]["l2"],3)
        self.assertIsNone(out["interaction"]["relative_l2"])

    def test_negative_cross_effect_is_retained(self):
        a={k:np.asarray(v,dtype=np.float32) for k,v in {"00":[0,0],"10":[2,1],"01":[-3,1],"11":[-1,2]}.items()}
        out=factorial(a)
        self.assertEqual(out["twice_attention_mlp_inner_product"],-10)
        self.assertEqual(out["effects"]["interaction"]["l2"],0)

    def test_projector_uses_all_three_columns(self):
        q=np.eye(5,dtype=float)[:3]
        x=np.asarray([1,-2,3,4,-5],dtype=np.float32)
        self.assertEqual(projected(x,q).tobytes(),np.asarray([1,-2,3,0,0],dtype=np.float32).tobytes())


if __name__=="__main__":unittest.main()
