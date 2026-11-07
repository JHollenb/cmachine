import unittest
import numpy as np
from analysis import archive_bound, polynomial_product, validate_path

class Contracts(unittest.TestCase):
    def path(self):
        return {"policy": 1, "all_delivered_consumed": True, "consumed_ids": [5, 7], "answers": [{"ids": [7]}, {"ids": []}], "steps": [
            {"step": 0, "position": 10, "input_id": 5, "role": "initial-prompt", "top": 7, "mode": 1, "gate_up_calls": 2, "down_calls": 1, "native_swiglu": 0, "polynomial_swiglu": 1},
            {"step": 1, "position": 11, "input_id": 7, "role": "assistant", "top": 9, "mode": 0, "gate_up_calls": 2, "down_calls": 1, "native_swiglu": 1, "polynomial_swiglu": 0}]}
    def test_once_is_not_once_per_turn(self):
        p = self.path()
        validate_path(p, 10, True)
        p["steps"][1].update(mode=1, native_swiglu=0, polynomial_swiglu=1)
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_path(p, 10, True)
    def test_teacher_tokens_are_distinct(self):
        p = self.path()
        p["steps"][1]["input_id"] = p["consumed_ids"][1] = p["answers"][0]["ids"][0] = 8
        with self.assertRaisesRegex(ValueError, "Teacher"):
            validate_path(p, 10, True)
        validate_path(p, 10, False)
    def test_consumption_clock(self):
        p = self.path()
        p["steps"][1]["position"] += 1
        with self.assertRaisesRegex(ValueError, "clock"):
            validate_path(p, 10, True)
    def test_polynomial_at_zero_anchor(self):
        g = np.array([-2, 0, 2], dtype=np.float32)
        law = np.array([[0]*3, [.5]*3, [.25]*3, [0]*3, [0]*3], dtype=np.float32)
        np.testing.assert_array_equal(polynomial_product(g, np.ones(3, dtype=np.float32), law), .5*g+.25*g*g)
    def test_anchor_uses_stored_finite_silu(self):
        source = np.array([-20, -.3, 20], dtype=np.float32)
        p = np.array([2e-9, .4255, 1], dtype=np.float32)
        silu = np.array([-4e-8, -.12765, 20], dtype=np.float32)
        law = np.array([source, p, p*(1-p), .5*p*(1-p)*(1-2*p), silu])
        np.testing.assert_array_equal(polynomial_product(source, np.ones(3, dtype=np.float32), law), silu)
    def test_evidence_geometry_is_bounded_before_execution(self):
        self.assertLess(archive_bound([40, 40]), 256*1024**2)
        self.assertGreater(archive_bound([100, 100]), 256*1024**2)
        with self.assertRaises(ValueError):
            archive_bound([40])

if __name__ == "__main__":
    unittest.main()
