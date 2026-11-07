"""Model-free mechanics tests for the source-only operand compiler."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compiler


class CompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h = 19
        self.d = 13
        self.rank = 4
        rng = np.random.default_rng(42)
        self.gate = rng.normal(size=(self.h, self.d)).astype("<f4")
        self.up = rng.normal(size=(self.h, self.d)).astype("<f4")
        self.gamma = rng.normal(loc=1.0, scale=0.2, size=(self.d,)).astype("<f4")

    def test_compile_is_deterministic_and_source_only(self) -> None:
        first = compiler.compile_banks(
            gate=self.gate,
            up=self.up,
            gamma=self.gamma,
            layer=16,
            rank=self.rank,
            oversample=3,
            power_iterations=2,
            seed=17,
            random_seed=18,
        )
        second = compiler.compile_banks(
            gate=self.gate,
            up=self.up,
            gamma=self.gamma,
            layer=16,
            rank=self.rank,
            oversample=3,
            power_iterations=2,
            seed=17,
            random_seed=18,
        )
        for bank in ("source", "matched_random"):
            for component in ("q_t", "gate", "up"):
                self.assertEqual(
                    first.banks[bank][component].tobytes(),
                    second.banks[bank][component].tobytes(),
                )
        self.assertFalse(
            np.array_equal(first.banks["source"]["q_t"], first.banks["matched_random"]["q_t"])
        )
        self.assertLess(first.metrics["source_basis"]["orthonormal_frobenius_error"], 2e-6)
        self.assertLess(first.metrics["matched_random_basis"]["orthonormal_frobenius_error"], 2e-6)
        self.assertEqual(first.metrics["source_basis"]["basis_dtype"], "<f4")
        self.assertEqual(first.metrics["matched_random_basis"]["basis_dtype"], "<f4")
        self.assertIn("basis_cast_difference", first.metrics["images"]["source_gate"])
        self.assertTrue(first.metrics["images"]["source_gate"]["finite"])

    def test_folded_images_match_definition(self) -> None:
        result = compiler.compile_banks(
            gate=self.gate,
            up=self.up,
            gamma=self.gamma,
            rank=self.rank,
            oversample=3,
            power_iterations=1,
            seed=4,
            random_seed=5,
        )
        folded_gate = self.gate.astype(np.float64) * self.gamma.astype(np.float64)[None, :]
        folded_up = self.up.astype(np.float64) * self.gamma.astype(np.float64)[None, :]
        for bank in ("source", "matched_random"):
            q_t = result.banks[bank]["q_t"].astype(np.float64)
            expected_gate = folded_gate @ q_t.T
            expected_up = folded_up @ q_t.T
            np.testing.assert_allclose(result.banks[bank]["gate"], expected_gate, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(result.banks[bank]["up"], expected_up, rtol=2e-6, atol=2e-6)

    def test_manifest_load_and_exact_output_custody(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name, arr in (("gate", self.gate), ("up", self.up), ("gamma", self.gamma)):
                path = root / f"{name}.f32"
                path.write_bytes(np.asarray(arr, dtype="<f4", order="C").tobytes())
                paths[name] = path.name
            source_manifest = root / "source.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "weights": {
                            "gate": {"path": paths["gate"], "shape": [self.h, self.d]},
                            "up": {"path": paths["up"], "shape": [self.h, self.d]},
                            "gamma": {"path": paths["gamma"], "shape": [self.d]},
                        }
                    }
                )
            )
            source = compiler.load_source_weights(
                manifest=source_manifest,
                hidden=self.d,
                intermediate=self.h,
            )
            self.assertEqual(
                source.records["gate"].sha256,
                hashlib.sha256((root / paths["gate"]).read_bytes()).hexdigest(),
            )
            result = compiler.compile_banks(source, rank=self.rank, oversample=3, power_iterations=1)
            output = root / "banks"
            manifest = compiler.write_banks(result, output)
            on_disk = json.loads((output / "bank_manifest.json").read_text())
            self.assertEqual(on_disk["schema"], compiler.SCHEMA)
            self.assertEqual(on_disk["profile"]["hidden"], self.d)
            self.assertFalse(on_disk["provenance"]["future_activation_inputs"])
            self.assertTrue(on_disk["provenance"]["source_weights_only"])
            for bank_entries in on_disk["banks"].values():
                for entry in bank_entries.values():
                    data = (output / entry["file"]).read_bytes()
                    self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
                    self.assertEqual(entry["byte_count"], len(data))
            self.assertEqual(manifest["manifest_sha256"], hashlib.sha256((output / "bank_manifest.json").read_bytes()).hexdigest())

    def test_loader_rejects_nonfinite_or_bad_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = np.zeros((self.h, self.d), dtype="<f4")
            bad[0, 0] = np.nan
            bad_path = root / "bad.f32"
            bad_path.write_bytes(bad.tobytes())
            up_path = root / "up.f32"
            up_path.write_bytes(self.up.tobytes())
            gamma_path = root / "gamma.f32"
            gamma_path.write_bytes(self.gamma.tobytes())
            with self.assertRaises(ValueError):
                compiler.load_source_weights(
                    gate=bad_path,
                    up=up_path,
                    gamma=gamma_path,
                    hidden=self.d,
                    intermediate=self.h,
                )
            wrong_path = root / "wrong.f32"
            wrong_path.write_bytes(np.zeros((self.h, self.d - 1), dtype="<f4").tobytes())
            with self.assertRaises(ValueError):
                compiler.load_source_weights(
                    gate=wrong_path,
                    up=up_path,
                    gamma=gamma_path,
                    hidden=self.d,
                    intermediate=self.h,
                )

    def test_input_dir_cli_matches_c_driver_abi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, arr in (("G", self.gate), ("U", self.up), ("gamma", self.gamma)):
                (root / f"{name}.f32").write_bytes(np.asarray(arr, dtype="<f4", order="C").tobytes())
            (root / "input-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "tcm-operand-basis-input-v1",
                        "G": {"file": "G.f32", "rows": self.h, "cols": self.d},
                        "U": {"file": "U.f32", "rows": self.h, "cols": self.d},
                        "gamma": {"file": "gamma.f32", "count": self.d},
                    }
                )
            )
            output = root / "compiled"
            self.assertEqual(
                compiler.main(
                    [
                        "--input-dir",
                        str(root),
                        "--output-dir",
                        str(output),
                        "--hidden",
                        str(self.d),
                        "--intermediate",
                        str(self.h),
                        "--rank",
                        str(self.rank),
                        "--oversample",
                        "3",
                        "--power-iterations",
                        "1",
                    ]
                ),
                0,
            )
            self.assertTrue((output / "source-QT.f32").exists())
            self.assertTrue((output / "source-GQ.f32").exists())
            self.assertTrue((output / "source-UQ.f32").exists())
            self.assertTrue((output / "random-QT.f32").exists())
            manifest = json.loads((output / "bank_manifest.json").read_text())
            self.assertEqual(manifest["source"]["weights"]["gate"]["shape"], [self.h, self.d])


if __name__ == "__main__":
    unittest.main()
