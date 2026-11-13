"""Exact affine-envelope geometry, including point-only and parallel ties."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decision_envelope", ROOT / "tools/decision_envelope.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
envelope = module.envelope


def winner(rows, dose):
    scores = [(Fraction(r["intercept"])+dose*Fraction(r["slope"]), r["token_id"]) for r in rows]
    best = max(s for s, _ in scores)
    return sorted(i for s, i in scores if s == best)


class DecisionEnvelopes(unittest.TestCase):
    def test_retained_translation_regime_and_probe(self):
        data = json.loads((ROOT / "tests/fixtures/translation-envelope.json").read_text())
        report = envelope(data["lines"])
        self.assertEqual(report["coverage"], "provided-lines-only")
        middle = report["segments"][1]
        self.assertEqual(middle["winner"], 650)
        self.assertEqual(middle["left"], "324474813684256/4315771582720603")
        self.assertEqual(middle["right"], "207832459388728/939467298396787")
        self.assertEqual(middle["probe"]["binary64_dose"], .125)
        self.assertNotIn(650, [winner(data["lines"], Fraction(i, 4))[0] for i in range(5)])
        self.assertEqual(winner(data["lines"], Fraction(1, 8))[0], 650)

    def test_point_only_tie_survives_hull_pruning(self):
        rows = [{"token_id": i, "intercept": 0, "slope": s} for i, s in [(9, -1), (0, 0), (8, 1)]]
        report = envelope(rows, -1, 1)
        self.assertEqual([s["winner"] for s in report["segments"]], [9, 8])
        middle = next(p for p in report["points"] if p["dose"] == "0")
        self.assertEqual(middle["tied_token_ids"], [0, 8, 9])
        self.assertEqual(middle["winner"], 0)

    def test_equivalent_and_dominated_parallel_lines(self):
        rows = [{"token_id": 9, "intercept": 1, "slope": 0},
                {"token_id": 2, "intercept": 1, "slope": 0},
                {"token_id": 0, "intercept": 0, "slope": 0}]
        result = envelope(rows)
        self.assertEqual(result["segments"][0]["tied_line_ids"], [2, 9])
        self.assertTrue(all(p["tied_token_ids"] == [2, 9] for p in result["points"]))

    def test_random_hulls_against_full_line_competition(self):
        rng = random.Random(239)
        for _ in range(60):
            rows = [{"token_id": i, "intercept": rng.randint(-20, 20), "slope": rng.randint(-10, 10)} for i in range(24)]
            result = envelope(rows, -2, 3, vocabulary_size=24)
            for segment in result["segments"]:
                left, right = Fraction(segment["left"]), Fraction(segment["right"])
                self.assertEqual(winner(rows, (left+right)/2), segment["tied_line_ids"])
                probe = Fraction(segment["probe"]["binary64_dose"])
                self.assertTrue(left < probe < right)
            for point in result["points"]:
                self.assertEqual(winner(rows, Fraction(point["dose"])), point["tied_token_ids"])

    def test_narrow_regions_remain_even_without_binary64_interior(self):
        # Region (1,1+2^-1000) exists over rationals but has no binary64 interior.
        rows = [{"token_id": 0, "intercept": 0, "slope": 0},
                {"token_id": 1, "intercept": -1, "slope": 1},
                {"token_id": 2, "intercept": str(-2-Fraction(1, 2**1000)), "slope": 2}]
        result = envelope(rows, 0, 2)
        middle = next(s for s in result["segments"] if s["winner"] == 1)
        self.assertIsNone(middle["probe"]["binary64_dose"])

    def test_degenerate_interval_and_full_vocabulary_contract(self):
        rows = [{"token_id": 0, "intercept": 1, "slope": 0}]
        self.assertEqual(envelope(rows, 1, 1)["segments"], [])
        with self.assertRaises(ValueError):
            envelope(rows, vocabulary_size=2)
        with self.assertRaises(ValueError):
            envelope(rows*2)
        with self.assertRaises(ValueError):
            envelope([{**rows[0], "slope": float("nan")}])


if __name__ == "__main__":
    unittest.main()
