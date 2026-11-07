"""Offline tests for closure.geometry; no model, CUDA, or mrun dependency."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geometry import ANGLES, SEED, generate, historical_coefficients


class GeometryTests(unittest.TestCase):
    def test_nonorthogonal_projection_and_fiber(self) -> None:
        rng = np.random.default_rng(17)
        columns = rng.normal(size=(9, 3)).astype(np.float64)
        coefficients = np.asarray((0.7, -1.1, 0.25), dtype=np.float64)
        raw = columns @ coefficients + rng.normal(size=9)
        base = columns @ np.linalg.solve(columns.T @ columns, columns.T @ raw)
        omitted = raw - base
        self.assertLess(float(np.max(np.abs(columns.T @ omitted))), 1e-11)
        self.assertAlmostEqual(
            float(np.linalg.norm(raw) ** 2),
            float(np.linalg.norm(base) ** 2 + np.linalg.norm(omitted) ** 2),
            places=10,
        )
        seeded = np.random.Generator(np.random.PCG64(SEED))
        proposal = seeded.normal(size=9)
        constraints = np.column_stack((columns, omitted))
        direction = proposal - constraints @ np.linalg.solve(
            constraints.T @ constraints, constraints.T @ proposal
        )
        direction *= np.linalg.norm(omitted) / np.linalg.norm(direction)
        self.assertLess(float(np.max(np.abs(columns.T @ direction))), 1e-11)
        self.assertLess(abs(float(np.dot(omitted, direction))), 1e-11)
        for angle in ANGLES:
            point = base + omitted * np.cos(angle) + direction * np.sin(angle)
            self.assertAlmostEqual(
                float(np.linalg.norm(point)),
                float(np.linalg.norm(raw)),
                places=10,
            )

    def test_historical_sequential_coefficients(self) -> None:
        columns = np.eye(5, 3, dtype=np.float64)
        vector = np.asarray((2.0, -3.0, 4.0, 1.0, -2.0), dtype=np.float32)
        factors = (1.0, 0.0, 1.0, 0.0, 0.0, 1.0)
        fp64, fp32 = historical_coefficients(vector, columns, factors)
        np.testing.assert_array_equal(fp64, np.asarray((2.0, -3.0, 4.0)))
        self.assertEqual(fp32.dtype, np.dtype("<f4"))
        self.assertEqual(fp32.tobytes().hex(), "00000040000040c000008040")

    def test_generate_is_deterministic_and_source_bound(self) -> None:
        dimension = 9
        rng = np.random.default_rng(31)
        columns, _ = np.linalg.qr(rng.normal(size=(dimension, 3)))
        columns = np.asarray(columns[:, :3], dtype=np.float64)
        reuse = np.concatenate(
            (columns[:, 0], columns[:, 1], np.asarray((1.0, 0.0, 1.0, 1.0)))
        )
        enrich = np.concatenate((columns[:, 2], np.asarray((0.0, 0.0, 1.0))))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            bank = root / "bank"
            out_a = root / "out-a"
            out_b = root / "out-b"
            raw.mkdir()
            bank.mkdir()
            (bank / "reuse-qr.f64").write_bytes(
                np.asarray(reuse, dtype="<f8").tobytes()
            )
            (bank / "enrich-qr.f64").write_bytes(
                np.asarray(enrich, dtype="<f8").tobytes()
            )
            for index, task in enumerate(("recall", "math", "prose")):
                values = columns @ np.asarray((1.0 + index, -0.5, 0.25))
                values += np.asarray(
                    (0.0, 0.0, 0.0, 1.0 + index, -2.0, 0.5, 1.5, -0.75, 0.25)
                )
                np.asarray(values, dtype="<f4").tofile(
                    raw / f"{task}-native-s000-raw_receiver.f32"
                )
            first = generate(raw, bank, out_a)
            second = generate(raw, bank, out_b)
            self.assertEqual(first["schema"], "tcm-closure-geometry-v2")
            self.assertEqual(first["seed"], SEED)
            self.assertEqual(first["candidates_per_task"], 64)
            self.assertEqual(len(first["tasks"]), 3)
            for task_a, task_b in zip(first["tasks"], second["tasks"]):
                self.assertEqual(
                    [
                        (record["id"], record["coefficient_bytes_hex"])
                        for record in task_a["candidates"]
                    ],
                    [
                        (record["id"], record["coefficient_bytes_hex"])
                        for record in task_b["candidates"]
                    ],
                )
                source = (
                    raw / f"{task_a['task']}-native-s000-raw_receiver.f32"
                ).read_bytes()
                self.assertEqual(source, Path(task_a["source"]["file"]).read_bytes())
                self.assertEqual(
                    Path(task_a["candidate_file"]).stat().st_size,
                    64 * dimension * 4,
                )
                self.assertEqual(len(task_a["candidates"]), 64)
                self.assertEqual(
                    [record["index"] for record in task_a["candidates"]],
                    list(range(64)),
                )
                self.assertTrue(
                    all(record["finite"] for record in task_a["candidates"])
                )
            parsed = json.loads((out_a / "geometry.json").read_text())
            self.assertEqual(parsed["schema"], first["schema"])


if __name__ == "__main__":
    unittest.main()
