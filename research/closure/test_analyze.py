import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analyze import (
    CODE_BYTES,
    ClosureInputError,
    array_metrics,
    code_digest,
    evaluate_exact_code_pairs,
    exact_code,
    exact_code_equal,
    gram_projection,
    random_spherical_candidates,
    spherical_candidate,
    analyze,
)


class ExactCodeTests(unittest.TestCase):
    def test_code_is_raw_fp32_and_preserves_adjacent_bits(self):
        coefficients = np.asarray([1.0, -0.0, 0.5], dtype="<f4")
        rho = np.asarray(0.75, dtype="<f4")
        code = exact_code(coefficients, rho)
        self.assertEqual(len(code), CODE_BYTES)
        adjacent = coefficients.copy()
        adjacent[0] = np.nextafter(adjacent[0], np.float32(2.0), dtype=np.float32)
        other = exact_code(adjacent, rho)
        self.assertFalse(exact_code_equal(code, other))
        self.assertNotEqual(code_digest(code), code_digest(other))

    def test_code_does_not_round_json_like_nearby_values(self):
        left = exact_code(np.asarray([0.12345678, 0.2, 0.3], dtype="<f4"), np.float32(0.7))
        right = exact_code(np.asarray([0.12345679, 0.2, 0.3], dtype="<f4"), np.float32(0.7))
        self.assertNotEqual(left, right)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.basis = rng.standard_normal((12, 3)).astype("<f4")
        self.source = rng.standard_normal(12).astype("<f4")

    def test_general_gram_projection_is_idempotent(self):
        split = gram_projection(self.basis, self.source)
        projected = gram_projection(self.basis, split.projection).projection
        np.testing.assert_allclose(projected, split.projection, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(np.dot(split.projection, split.omitted)), 0.0, places=10)

    def test_spherical_candidate_preserves_projection_and_omitted_norm_diagnostically(self):
        candidate = spherical_candidate(self.source, self.basis, np.arange(12), name="x", construction_seed=1)
        source_split = gram_projection(self.basis, self.source)
        candidate_split = gram_projection(self.basis, candidate.receiver)
        np.testing.assert_allclose(candidate_split.projection, source_split.projection, rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(
            float(np.linalg.norm(candidate_split.omitted)),
            float(np.linalg.norm(source_split.omitted)),
            places=5,
        )
        self.assertFalse(np.array_equal(candidate.receiver, self.source))

    def test_random_candidates_are_deterministic(self):
        first = random_spherical_candidates(self.source, self.basis, count=3, seed=17)
        second = random_spherical_candidates(self.source, self.basis, count=3, seed=17)
        self.assertEqual([x.receiver.tobytes() for x in first], [x.receiver.tobytes() for x in second])


class SelectionOrderTests(unittest.TestCase):
    def test_outcomes_are_loaded_only_after_exact_code_filter(self):
        entries = [
            {"name": "a", "coefficients_f32": [1.0, 2.0, 3.0], "rho_f32": 4.0},
            {"name": "b", "coefficients_f32": [1.0, 2.0, 3.0], "rho_f32": 4.0},
            {"name": "unique", "coefficients_f32": [1.0, 2.0, 3.000001], "rho_f32": 4.0},
        ]
        loaded = []

        def loader(entry):
            loaded.append(entry["name"])
            return entry["name"] + "-outcome"

        result = evaluate_exact_code_pairs(entries, loader)
        self.assertEqual(loaded, ["a", "b"])
        self.assertEqual(len(result), 1)
        self.assertEqual([x["name"] for x in result[0]["entries"]], ["a", "b"])

    def test_target_metadata_is_rejected_before_outcomes(self):
        entries = [
            {"name": "a", "coefficients_f32": [1.0, 2.0, 3.0], "rho_f32": 4.0,
             "metadata": {"target_ids": [99]}},
        ]
        with self.assertRaises(ClosureInputError):
            evaluate_exact_code_pairs(entries, lambda _: self.fail("outcome loaded"))

    def test_array_metrics_keeps_exact_equality_separate_from_magnitude(self):
        left = np.asarray([1.0, 2.0], dtype="<f4")
        right = left.copy()
        right[0] = np.nextafter(right[0], np.float32(2.0), dtype=np.float32)
        metrics = array_metrics(left, right)
        self.assertFalse(metrics["bytes_equal"])
        self.assertGreater(metrics["l2"], 0.0)


class SavedReportTests(unittest.TestCase):
    @staticmethod
    def _save(root, prefix, port, array):
        Path(root, f"{prefix}-{port}.f32").write_bytes(np.asarray(array, dtype="<f4").tobytes())

    def test_analyze_exact_pair_and_code_only_bound(self):
        with TemporaryDirectory() as directory:
            raw = Path(directory)
            hidden, intermediate = 4, 3
            code = exact_code([0.125, -0.25, 0.5], np.float32(0.75))
            source = np.arange(hidden, dtype="<f4")
            fiber = source + np.asarray([0.0, 1.0, 0.0, 0.0], dtype="<f4")
            source_gate = np.arange(intermediate, dtype="<f4")
            fiber_gate = source_gate + np.float32(0.25)
            source_up = source_gate + np.float32(2.0)
            fiber_up = source_up + np.float32(0.5)
            source_write = np.arange(hidden, dtype="<f4")
            fiber_write = source_write + np.float32(2.0)
            for port, array in (("receiver", source), ("normalized", source), ("gate", source_gate), ("up", source_up), ("write", source_write)):
                self._save(raw, "recall-source", port, array)
            for port, array in (("receiver", fiber), ("normalized", fiber), ("gate", fiber_gate), ("up", fiber_up), ("write", fiber_write)):
                self._save(raw, "recall-fiber", port, array)
            for port, array in (("gate", source_gate), ("up", source_up), ("write", source_write)):
                self._save(raw, "recall-project", port, array)
                self._save(raw, "recall-fiber-project", port, array)
            for port, array in (("receiver", fiber), ("normalized", fiber), ("gate", fiber_gate), ("up", fiber_up), ("write", fiber_write)):
                self._save(raw, "recall-repair", port, array)
            driver = {
                "schema": "tcm-state-closure-driver-v1",
                "layer": 16,
                "capacity": 512,
                "source_replay_checked": True,
                "cases": [{
                    "task": "recall",
                    "source_code_hex": code.hex(),
                    "actual_runtime_encoder_verified": True,
                    "candidates": [{"id": "alt", "index": 0, "code_hex": code.hex(), "same_code": True, "different_receiver": True}],
                    "selected_id": "alt",
                    "writes": {name: {"prefix": ("recall-fiber-project" if name == "fiber_project" else f"recall-{name}")} for name in ("source", "fiber", "project", "fiber_project", "repair")},
                    "arms": [],
                }],
            }
            result = analyze(raw, {}, driver, geometry={"hidden": hidden, "intermediate": intermediate})
            self.assertEqual(result["summary"]["selected_exact_fiber_cases"], 1)
            self.assertEqual(result["exact_fiber_codes"][0]["first_eligible_exact_fiber_id"], "alt")
            identity = result["per_case_write_differences"][0]["project_equal_code_identity"]
            self.assertTrue(identity["exact"])
            bound = result["code_only_minimax_half"][0]
            self.assertAlmostEqual(bound["deterministic_code_only_minimax_half_l2"], np.linalg.norm(source_write - fiber_write) / 2.0)

    def test_no_selected_collision_does_not_load_outcomes(self):
        with TemporaryDirectory() as directory:
            code = exact_code([0.125, -0.25, 0.5], np.float32(0.75))
            driver = {
                "schema": "tcm-state-closure-driver-v1", "layer": 16, "capacity": 512,
                "source_replay_checked": True,
                "cases": [{
                    "task": "recall", "source_code_hex": code.hex(),
                    "actual_runtime_encoder_verified": True,
                    "candidates": [{"index": 0, "code_hex": exact_code([0.125, -0.25, 0.5001], np.float32(0.75)).hex(), "same_code": False, "different_receiver": True}],
                    "selected_index": -1,
                    "writes": {"source": {"prefix": "missing-source"}}, "arms": [],
                }],
            }
            result = analyze(directory, {}, driver)
            self.assertEqual(result["summary"]["instrument_status"], "evidence-incomplete")
            self.assertEqual(result["per_case_write_differences"][0]["status"], "evidence-incomplete-no-exact-fiber-selection")


if __name__ == "__main__":
    unittest.main()
