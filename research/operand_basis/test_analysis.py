"""Model-free evidence and instrumentation tests for operand_basis.analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analysis


class AnalysisTest(unittest.TestCase):
    def _write(self, root: Path, name: str, values: np.ndarray) -> str:
        path = root / name
        path.write_bytes(np.asarray(values, dtype="<f4").tobytes())
        return name

    def _build_fixture(self, root: Path, *, nonfinite: bool = False) -> dict:
        hidden, vocab = 4, 6
        case = {
            "case": 0,
            "prompt_tokens": [10, 11],
            "followup_tokens": [99, 100],
            "arms": [],
        }
        all_primary_files = []
        def make_path(name: str, arm: str, ids: list[int], write_delta: float) -> dict:
            path = {"name": name, "case": 0, "arm": arm, "autonomous": True, "followup_skip": 0,
                    "answer0": {"ids": [ids[2]], "stopped": True, "truncated": False},
                    "followup_ids": [99, 100], "answer1": {"ids": [ids[-1]], "stopped": True, "truncated": False},
                    "all_delivered_consumed": True, "steps": []}
            for index, token in enumerate(ids):
                stem = f"{name}-s{index:03d}"
                row = {"step": index, "position": index, "input_id": token,
                       "basis_q_calls": 1 if arm in ("source64", "random64") else 0,
                       "basis_g_calls": 1 if arm in ("source64", "random64") else 0,
                       "basis_u_calls": 1 if arm in ("source64", "random64") else 0,
                       "selected_gate_up_calls": 0 if arm != "native" else 2,
                       "selected_down_calls": 1 if arm in ("source64", "random64") else (1 if arm == "native" else 0),
                       "native_write_file": f"{stem}-native_write.f32"}
                path["steps"].append(row)
                receiver = np.arange(hidden, dtype=np.float32) + index
                norm = receiver * 0.5
                candidate_write = receiver * (0.1 + write_delta)
                if nonfinite and index == 2 and arm == "source64":
                    candidate_write[0] = np.nan
                native_write = receiver * 0.1
                self._write(root, f"{stem}-raw_receiver.f32", receiver)
                self._write(root, f"{stem}-mlp_input.f32", norm)
                self._write(root, f"{stem}-mlp_return.f32", candidate_write)
                self._write(root, f"{stem}-native_write.f32", native_write)
                all_primary_files.extend([f"{stem}-raw_receiver.f32", f"{stem}-mlp_input.f32", f"{stem}-mlp_return.f32"])
            for boundary, stem in (("first", f"{name}-s001"), ("final", name)):
                logits = np.arange(vocab, dtype=np.float32) + (0.25 if arm == "random64" else 0.0)
                file = f"{stem}-logits.f32"
                self._write(root, file, logits)
                path[boundary] = {"valid_kv_sha256": "kv", "logits": {"file": file, "top": int(np.argmax(logits)), "top_margin": 1.0}}
            return path

        source = make_path("case0-source64", "source64", [10, 11, 3, 99, 100, 4], 0.02)
        random = make_path("case0-random64", "random64", [10, 11, 3, 99, 100, 4], 0.03)
        native = make_path("case0-native", "native", [10, 11, 3, 99, 100, 4], 0.0)
        same = make_path("case0-source64-native-same-id", "native_same_id", [10, 11, 3, 99, 100, 4], 0.0)
        same["parent_name"] = source["name"]
        case["arms"] = [source, random]

        seal_files = []
        for filename in sorted(set(all_primary_files)):
            path = root / filename
            seal_files.append({"name": filename, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

        compiled = root / "compiled"
        compiled.mkdir()
        basis_names = ("source-QT.f32", "source-GQ.f32", "source-UQ.f32", "random-QT.f32", "random-GQ.f32", "random-UQ.f32")
        basis_sha = {}
        for filename in basis_names:
            path = compiled / filename
            path.write_bytes(np.arange(8, dtype="<f4").tobytes())
            basis_sha[filename.replace("-QT.f32", "_qt").replace("-GQ.f32", "_gq").replace("-UQ.f32", "_uq")] = hashlib.sha256(path.read_bytes()).hexdigest()
        input_dir = compiled / "compile-input"
        input_dir.mkdir()
        source_hashes = {}
        for key, filename in (("G", "G.f32"), ("U", "U.f32"), ("gamma", "gamma.f32")):
            values = np.ones(4, dtype="<f4")
            self._write(input_dir, filename, values)
            source_hashes[key] = hashlib.sha256((input_dir / filename).read_bytes()).hexdigest()
        (input_dir / "input-manifest.json").write_text(json.dumps({"G": {"sha256": source_hashes["G"]}, "U": {"sha256": source_hashes["U"]}, "gamma": {"sha256": source_hashes["gamma"]}}))
        bank_manifest = {
            "source": {"weights": {key: {"sha256": digest} for key, digest in (("gate", source_hashes["G"]), ("up", source_hashes["U"]), ("gamma", source_hashes["gamma"]))}},
        }
        (compiled / "bank_manifest.json").write_text(json.dumps(bank_manifest))

        prefix = b'{"sealed_prefix":true}'
        (root / "driver.json").write_bytes(prefix + b'{"rest":true}')
        prefix_hash = hashlib.sha256(prefix).hexdigest()
        (root / "primary-seal.json").write_text(json.dumps({"driver_prefix_bytes": len(prefix), "driver_prefix_sha256": prefix_hash, "files": seal_files}))
        driver = {
            "schema": "tcm-operand-basis-driver-v1",
            "profile": {"hidden": hidden, "vocab": vocab},
            "primary_seal_bytes": len(prefix), "primary_seal_sha256": prefix_hash,
            "compiled_dir": str(compiled), "compiler_input_dir": str(input_dir), "basis_sha256": basis_sha,
            "primary_cases": [case], "references": [{"case": 0, "native": native, "same_ids": [same]}],
            "work": {"primary_steps": 10},
        }
        return driver

    def test_seals_hash_join_common_history_and_same_id_readouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = self._build_fixture(root)
            result = analysis.analyze(root, root / "unused", driver, {"case0-source64": ["source"], "case0-random64": ["random"]})
            self.assertEqual(result["integrity"]["primary_seal"]["status"], "ok")
            self.assertEqual(result["integrity"]["compiled_program"]["status"], "observed")
            self.assertTrue(result["integrity"]["compiled_program"]["all_expected_files_match"])
            self.assertEqual(result["summary"]["common_history_receiver_mismatches"], 0)
            self.assertGreater(result["summary"]["full_vocab_native_same_id_readouts"], 0)
            self.assertEqual(result["summary"]["token_authority_mismatches"], 0)
            self.assertGreater(result["summary"]["op_counts"]["derived_projection_calls"], 0)
            self.assertTrue(any(row["kind"] == "local_write" for row in result["observations"]))

    def test_nonfinite_write_is_retained_as_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = self._build_fixture(root, nonfinite=True)
            result = analysis.analyze(root, None, driver, {})
            nonfinite = [row for row in result["local_writes"] if row["write_error"].get("status") == "nonfinite"]
            self.assertTrue(nonfinite)
            self.assertIn("scientific_quality_gate", result["summary"])

    def test_common_history_uses_full_prefix_not_token_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = self._build_fixture(root)
            # Make random diverge after the shared prefix, while reusing token
            # 3 later.  A token-ID-only grouping would incorrectly compare it.
            random_path = driver["primary_cases"][0]["arms"][1]
            random_path["steps"][2]["input_id"] = 8
            random_path["answer0"]["ids"] = [8]
            random_path["steps"][5]["input_id"] = 3
            result = analysis.analyze(root, None, driver, {})
            self.assertEqual(result["summary"]["common_history_receiver_mismatches"], 0)
            self.assertIn("complete_consumed_token_prefix", result["common_history_raw_receiver"][0]["grouping"])

    def test_derived_report_is_relocation_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            first_root = parent / "producer-a"
            second_root = parent / "producer-b"
            first_root.mkdir()
            driver = self._build_fixture(first_root)
            shutil.copytree(first_root, second_root)
            left = analysis.analyze(first_root, None, driver, {})
            right = analysis.analyze(second_root, None, driver, {})
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
