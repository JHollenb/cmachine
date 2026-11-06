"""Mechanics of finite contrasts, undefined ratios and common-parent custody."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from consumer_contrasts import ARMS, MODES, common_parent, contrast


class ConsumerContrastMechanics(unittest.TestCase):
    def test_harmful_isolated_read_can_help_conditionally(self):
        # A coupled two-coordinate consumer, with no monotonicity assumption.
        result = contrast({"project": [0., 0.], "carry": [1., 0.],
                           "read": [-1., 1.], "correct": [1., 1.], "dense": [1., 1.]})
        effects = result["native_error_transitions"]
        self.assertLess(effects["read_alone"]["error_norm_reduction"], 0)
        self.assertEqual(effects["read_given_carry"]["error_norm_reduction"], 1)
        self.assertEqual(result["terms"]["interaction"]["l2"], 1)
        self.assertEqual(result["tensor_identity_roundoff_max_abs"], 0)
        self.assertEqual(result["energy_identity"]["roundoff"], 0)
        for transition in effects.values():
            self.assertEqual(transition["paired_error_identity"]["identity_roundoff"], 0)

    def test_zero_response_stays_undefined(self):
        result = contrast({arm: [2., 3.] for arm in ARMS})
        self.assertIsNone(result["terms"]["interaction"]["norm_over_joint"])
        self.assertIsNone(result["terms"]["carry"]["signed_projection_onto_joint"])
        self.assertIsNone(result["native_error_transitions"]["read_alone"]["error_norm_reduction"])

    def test_invalid_arrays_are_mechanics_errors(self):
        for bad in ([float("nan"), 1.], [], [1.]):
            arrays = {arm: [0., 1.] for arm in ARMS}
            arrays["read"] = bad
            with self.assertRaises(ValueError):
                contrast(arrays)

    def test_nonzero_effects_survive_cancelling_joint(self):
        result = contrast({"project": [0., 0.], "carry": [1., 0.],
                           "read": [-1., 0.], "correct": [0., 0.], "dense": [0., 0.]})
        self.assertEqual(result["joint"]["l2"], 0)
        self.assertEqual(result["terms"]["carry"]["l2"], 1)
        self.assertEqual(result["terms"]["read"]["l2"], 1)
        self.assertIsNone(result["terms"]["carry"]["norm_over_joint"])
        self.assertEqual(result["energy_identity"]["twice_pair_inner_products"]["carry / read"], -2)
        self.assertEqual(result["energy_identity"]["roundoff"], 0)

    def test_different_parent_or_clock_is_rejected(self):
        rows = {arm: dict(mode=MODES[arm], position=81, parent_kv_prefix_sha256="parent", rho=0.75,
                          rank=3, prefix_intact=True) for arm in ARMS}
        common_parent(rows)
        for key, bad in (("position", 82), ("parent_kv_prefix_sha256", "other"),
                         ("rho", 0.5), ("rank", 2), ("prefix_intact", False), ("mode", 1)):
            changed = {arm: dict(row) for arm, row in rows.items()}
            changed["read"][key] = bad
            with self.assertRaises(ValueError):
                common_parent(changed)


if __name__ == "__main__":
    unittest.main()
