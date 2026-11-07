"""Small CPU-only tests for operand composition arithmetic and joins."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from .analysis import factorial_contrast, token_authority, analyze
    from .audit import audit
except ImportError:  # direct invocation from this directory
    from analysis import factorial_contrast, token_authority, analyze  # type: ignore
    from audit import audit  # type: ignore


class AnalysisMathTests(unittest.TestCase):
    def test_factorial_identity_and_signed_interaction(self) -> None:
        y00 = np.array([1, 2, 3], dtype=np.float32)
        y10 = np.array([2, 4, 7], dtype=np.float32)
        y01 = np.array([5, 8, 11], dtype=np.float32)
        y11 = np.array([9, 15, 22], dtype=np.float32)
        result = factorial_contrast({"00": y00, "10": y10, "01": y01, "11": y11})
        self.assertEqual(result["status"], "valid")
        # The signed cross-term is [3, 5, 7], rather than a norm-only effect.
        self.assertEqual(result["components"]["interaction"]["max_abs"], 7.0)
        self.assertEqual(result["gram_names"], ["mlp_at_reader_native", "reader_at_mlp_native", "interaction"])

    def test_rank_factor_orientation(self) -> None:
        # E=Q^T maps hidden -> rank, then Lg/Lu map rank -> intermediate.
        e = np.array([[1, 0, 1], [0, 1, -1]], dtype=np.float32)
        lg = np.array([[2, 3], [5, 7]], dtype=np.float32)
        lu = np.array([[11, 13], [17, 19]], dtype=np.float32)
        x = np.array([2, 3, 5], dtype=np.float32)
        z = e @ x
        self.assertTrue(np.array_equal(z, np.array([7, -2], dtype=np.float32)))
        self.assertTrue(np.array_equal(lg @ z, np.array([8, 21], dtype=np.float32)))
        self.assertTrue(np.array_equal(lu @ z, np.array([51, 81], dtype=np.float32)))

    def test_token_authority_statuses(self) -> None:
        path = {"name": "p", "consumed_ids": [10, 11, 12], "turns": [{"ingested": [10], "answer_ids": [11, 12]}]}
        self.assertEqual(token_authority(path)["status"], "exact")
        path["consumed_ids"] = [10, 12, 11]
        self.assertEqual(token_authority(path)["status"], "mismatch")
        count_path = {"consumed_ids": [10, 11], "turns": [{"ingested": 1, "answer_ids": [11]}]}
        self.assertEqual(token_authority(count_path)["status"], "declared_only")


class SavedJoinTests(unittest.TestCase):
    def _write_f32(self, root: Path, name: str, values: np.ndarray) -> str:
        path = root / name
        np.asarray(values, dtype="<f4").tofile(path)
        return name

    def test_synthetic_boundary_factorial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"hidden": 4, "intermediate": 6, "vocab": 8, "rank": 2, "query_rank": 3, "kv": 2}
            paths = []
            for policy, offset in ((0, 0), (1, 1), (2, 2), (3, 6)):
                name = f"case0-own-{policy:02d}"
                logits = np.arange(8, dtype=np.float32) + offset
                self._write_f32(root, f"{name}-t0-prefill-logits.f32", logits)
                paths.append({"name": name, "task": 0, "policy": policy, "mode": "own", "consumed_ids": [101], "turns": [{"turn": 0, "ingested": [101], "answer_ids": []}]})
                if policy < 3:
                    replay = f"case0-replay-{policy:02d}"
                    self._write_f32(root, f"{replay}-t0-prefill-logits.f32", logits)
                    paths.append({"name": replay, "task": 0, "policy": policy, "mode": "replay", "replay_of": "case0-own-11", "consumed_ids": [101], "turns": [{"turn": 0, "ingested": [101], "answer_ids": []}]})
            driver = {"schema": "tcm-operand-composition-driver-v1", **profile, "paths": paths}
            report = analyze(root, driver)
            rows = [r for r in report["factorial"] if r.get("status") == "valid"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["clock"], "prefill")
            audit_report = audit(root, driver)
            self.assertEqual(audit_report["mechanics_status"], "valid")
            self.assertEqual(audit_report["geometry"]["own"], 4)

    def test_path_contiguous_fields_are_sliced_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"hidden": 4, "intermediate": 6, "vocab": 8, "rank": 2, "query_rank": 3, "kv": 2}
            name = "case0-own-11"
            # Two token rows in the producer's path-contiguous format.
            self._write_f32(root, f"{name}-mlp-norm.f32", np.arange(8, dtype=np.float32))
            self._write_f32(root, f"{name}-mlp-write.f32", np.arange(8, dtype=np.float32) + 1)
            driver = {"schema": "tcm-operand-composition-driver-v1", **profile, "paths": [{"name": name, "task": 0, "policy": 3, "mode": "own", "count": 2, "consumed_ids": [1, 2], "turns": []}]}
            report = analyze(root, driver)
            path_report = next(r for r in report["producer_rows"] if r["path"] == name)
            rows = path_report["rows"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["fields"]["mlp-norm"]["l2"], np.linalg.norm(np.arange(4, dtype=np.float32)))
            self.assertEqual(rows[1]["fields"]["mlp-norm"]["l2"], np.linalg.norm(np.arange(4, 8, dtype=np.float32)))

    def test_own_driver_seal_is_json_serializable_and_joined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {"hidden": 4, "intermediate": 6, "vocab": 8, "rank": 2, "query_rank": 3, "kv": 2}
            path = {"name": "case0-own-11", "task": 0, "policy": 3, "mode": "own", "writer": True, "reader": True, "count": 0, "consumed_ids": [], "turns": []}
            driver = {"schema": "tcm-operand-composition-driver-v1", **profile, "paths": [path]}
            own_driver = root / "own-driver.json"
            own_driver.write_text(json.dumps({**driver, "paths": [path]}, separators=(",", ":")) + "\n")
            import hashlib
            own_hash = hashlib.sha256(own_driver.read_bytes()).hexdigest()
            (root / "own-seal.json").write_text(json.dumps({"files": [{"name": "own-driver.json", "bytes": own_driver.stat().st_size, "sha256": own_hash}]}))
            result = audit(root, driver)
            self.assertEqual(result["mechanics_status"], "valid")
            self.assertTrue(any(x.get("kind") == "own_driver_join" and x.get("status") == "exact" for x in result["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
