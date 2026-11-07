import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import _history_digest, analyze, local_tail_identity, policy_factorial, response_family


class MemoryResponseMath(unittest.TestCase):
    def test_signed_tail_identity_and_support_expansion(self):
        native_p = np.asarray([0.5, 0.3, 0.2])
        # Candidate retained rows 0 and 2; row 1 is omitted from its compact
        # support.  The policy value map is intentionally different so the
        # address/value decomposition is exercised together.
        policy_p = np.asarray([0.8, 0.2])
        native_v = np.asarray([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        policy_v = np.asarray([[1.0, 0.0], [4.5, 0.0]])
        result = local_tail_identity(native_p, policy_p, native_v, policy_v, [0, 2])
        self.assertEqual(result["selected_rows"], 2)
        self.assertEqual(result["tail_rows"], 1)
        self.assertAlmostEqual(result["signed_tail_mass"]["sum"], -0.3)
        self.assertLess(result["identity_residual"]["l2"], 1e-12)
        # The omitted row contributes a signed value term; it is not replaced
        # by an absolute omitted-mass percentage.
        self.assertAlmostEqual(result["signed_tail_value_contribution"]["sum"], -0.6)

    def test_factorial_keeps_signed_interaction(self):
        native = np.asarray([1.0, 2.0])
        address = np.asarray([2.0, 2.0])
        value = np.asarray([1.0, 4.0])
        both = np.asarray([4.0, 8.0])
        result = policy_factorial(native, address, value, both)
        np.testing.assert_array_equal(both - address - value + native, [2.0, 4.0])
        self.assertGreater(result["interaction_relative_l2"], 0.0)
        self.assertAlmostEqual(result["closure_relative_l2"], 0.0)

    def test_response_family_gqa_and_support_diagnostics(self):
        rng = np.random.default_rng(12)
        q = rng.normal(size=(12, 8))
        nk = rng.normal(size=(2, 6, 8))
        nv = rng.normal(size=(2, 6, 8))
        # A byte decoder is allowed to differ from native rows; this is an
        # audit-only conditional response, not a claim about future history.
        bk = (nk * 0.98).astype(np.float32)
        bv = (nv * 1.01).astype(np.float32)
        result = response_family(q, nk, nv, bk, bv, top_size=4, recent_size=4, active_start=2)
        self.assertEqual(len(result["top_support"]), 12)
        self.assertEqual(len(result["top_support"][0]), 4)
        self.assertEqual(len(result["recent_support"][0]), 4)
        top = result["responses"]["byte_top4"]["tail"]
        self.assertEqual(top["heads"], 12)
        self.assertEqual(top["rows"], 6)
        self.assertLess(top["identity_residual"]["relative_to_native"], 1e-12)
        self.assertIn("history_deleted", result["responses"])
        self.assertLess(result["responses"]["history_deleted"]["tail"]["identity_residual"]["relative_to_native"], 1e-12)

    def test_analysis_labels_same_id_and_post_divergence(self):
        p = np.asarray([0.5, 0.3, 0.2])
        v = np.asarray([[0.0], [1.0], [2.0]])
        driver = {
            "schema": "tcm-memory-response-driver-v1",
            "cases": [
                {
                    "case_id": "task-native",
                    "task": "binding0",
                    "policy": "native",
                    "question": "Mira",
                    "consumed_ids": [10, 11, 12],
                    "answer_ids": [12],
                    "truth": {"answer_ids": [12]},
                    "readouts": [
                        {"clock": "initial", "position": 2, "logits": [0.0, 1.0]},
                        {"clock": "final", "position": 3, "logits": [1.0, 0.0]},
                    ],
                    "attention": [
                        {
                            "layer": 17,
                            "position": 2,
                            "native_weights": p,
                            "policy_weights": p,
                            "native_values": v,
                            "selected_indices": [0, 1, 2],
                        }
                    ],
                    "work": {"linear_calls": 10, "state_bytes": 100},
                },
                {
                    "case_id": "task-byte",
                    "task": "binding0",
                    "policy": "byte_full",
                    "question": "Mira",
                    "consumed_ids": [10, 11, 13],
                    "answer_ids": [13],
                    "truth": {"answer_ids": [12]},
                    "readouts": [
                        {"clock": "initial", "position": 2, "logits": [0.0, 1.0]},
                        {"clock": "final", "position": 3, "logits": [0.0, 1.0]},
                    ],
                    "attention": [
                        {
                            "layer": 17,
                            "position": 2,
                            "native_weights": p,
                            "policy_weights": p,
                            "native_values": v,
                            "selected_indices": [0, 1, 2],
                        }
                    ],
                    "work": {"linear_calls": 12, "state_bytes": 80},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            result = analyze(Path(temp), None, driver)
        domains = [x["comparison_domain"] for x in result["readout_comparisons"]]
        self.assertEqual(domains, ["same_id", "post_divergence"])
        self.assertEqual(result["summary"]["native_fidelity_same_id_rows"], 1)
        self.assertEqual(result["summary"]["post_divergence_rows"], 1)
        self.assertEqual(result["summary"]["truth_matches"], {"matched": 1, "known": 2})
        self.assertEqual(result["attention_tail_rows"][0]["metrics"]["tail_rows"], 0)

    def test_history_deleted_alias_and_work_are_preserved(self):
        driver = {
            "cases": [
                {
                    "id": "native",
                    "task": "binding0",
                    "policy": "native",
                    "consumed_ids": [1],
                    "answer_ids": [1],
                    "readouts": [{"clock": "q", "position": 1, "logits": [1.0, 0.0]}],
                },
                {
                    "id": "deleted",
                    "task": "binding0",
                    "policy": "history_deleted",
                    "consumed_ids": [1],
                    "answer_ids": [1],
                    "state_intervention": {"kind": "native_parent_history_excluded", "fork": 1},
                    "readouts": [{"clock": "q", "position": 1, "logits": [0.0, 1.0]}],
                    "work": {"memory_bytes": 24, "cache_bytes": 48},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            result = analyze(Path(temp), None, driver)
        deleted = next(x for x in result["cases"] if x["case_id"] == "deleted")
        self.assertEqual(deleted["policy"], "history_deleted")
        self.assertEqual(deleted["state_intervention"]["kind"], "native_parent_history_excluded")
        self.assertEqual(result["work_rows"][0]["state_bytes"], 24)
        self.assertEqual(result["work_rows"][0]["dense_cache_bytes"], 48)

    def test_color_truth_is_case_insensitive_but_format_strict(self):
        driver = {
            "cases": [
                {
                    "name": "branch-good",
                    "task": 0,
                    "policy": "native",
                    "question": "Mira",
                    "consumed_ids": [1],
                    "answer_ids": [2, 3],
                    "readouts": [],
                },
                {
                    "name": "branch-punct",
                    "task": 0,
                    "policy": "byte_full",
                    "question": "Niko",
                    "consumed_ids": [1],
                    "answer_ids": [4],
                    "readouts": [],
                },
                {
                    "name": "branch-verbose",
                    "task": 0,
                    "policy": "byte_top4",
                    "question": "Mira",
                    "consumed_ids": [1],
                    "answer_ids": [5],
                    "readouts": [],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            (root / "inputs" / "runtime-inputs.json").write_text(
                '{"cases":[{"id":0,"expected":["red","blue"]}]}\n'
            )
            result = analyze(
                root,
                None,
                driver,
                {"branch-good": "RED", "branch-punct": "blue!", "branch-verbose": "blue and red"},
            )
        outcomes = {x["case_id"]: x["truth"] for x in result["cases"]}
        self.assertTrue(outcomes["branch-good"]["matched"])
        self.assertTrue(outcomes["branch-punct"]["matched"])
        self.assertFalse(outcomes["branch-verbose"]["matched"])
        self.assertEqual(outcomes["branch-verbose"]["format_status"], "format_miss")

    def test_runtime_manifest_coverage_is_required(self):
        driver = {
            "cases": [
                {
                    "name": "case0-native-q0",
                    "task": 0,
                    "policy": "native",
                    "question": "Mira",
                    "consumed_ids": [1],
                    "answer_ids": [2],
                    "readouts": [],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            (root / "inputs" / "runtime-inputs.json").write_text(
                '{"cases":[{"id":0,"expected":["red"]}],"policies":["native"],"questions":["Mira"]}\n'
            )
            result = analyze(root, None, driver, {"case0-native-q0": "red"})
        self.assertEqual(result["summary"]["coverage"]["status"], "complete")
        self.assertEqual(result["summary"]["coverage"]["expected_rows"], 1)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            (root / "inputs" / "runtime-inputs.json").write_text(
                '{"cases":[{"id":0,"expected":["red"]}],"policies":["native"],"questions":["Mira","Niko"]}\n'
            )
            with self.assertRaises(ValueError):
                analyze(root, None, driver, {"case0-native-q0": "red"})

    def test_flat_path_with_nested_queries_preserves_prefix_authority(self):
        parent = [1, 2]
        prefix_hash = _history_digest(parent)

        def path(policy):
            def query(index, color):
                name = f"case0-{policy}-query{index}"
                input_ids = [10 + index]
                answer_ids = [20 + index]
                return {
                    "name": name,
                    "query_id": index,
                    "input_ids": input_ids,
                    "answer_ids": answer_ids,
                    "readouts": [
                        {
                            "kind": f"query{index}_input",
                            "position": 3,
                            "history_sha256": _history_digest(parent + input_ids),
                            "logits": [1.0, 0.0] if color == "red" else [0.0, 1.0],
                        },
                        {
                            "kind": f"query{index}_final",
                            "position": 4,
                            "history_sha256": _history_digest(parent + input_ids + answer_ids),
                            "logits": [1.0, 0.0] if color == "red" else [0.0, 1.0],
                        },
                    ],
                }

            return {
                "case_id": 0,
                "policy": policy,
                "parent_history_ids": parent,
                "parent_history_sha256": prefix_hash,
                "prefix_boundaries": [
                    {"kind": "common_parent", "position": 2, "history_sha256": prefix_hash}
                ],
                "queries": [query(0, "red"), query(1, "blue")],
            }

        driver = {"schema": "tcm-memory-response-driver-v1", "cases": [path("native"), path("byte_full")]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            (root / "inputs" / "runtime-inputs.json").write_text(
                '{"cases":[{"id":0,"initial_bit":0,"update_bit":0,"expected":["red","blue"]}],'
                '"policies":["native","byte_full"],"questions":["Mira","Niko"]}\n'
            )
            texts = {
                "case0-native-query0": "red",
                "case0-native-query1": "blue",
                "case0-byte_full-query0": "red",
                "case0-byte_full-query1": "blue",
            }
            result = analyze(root, None, driver, texts)
        self.assertEqual(result["summary"]["coverage"]["status"], "complete")
        self.assertEqual(len(result["cases"]), 4)
        self.assertEqual(len(result["prefix_boundaries"]), 2)
        self.assertEqual(result["summary"]["native_fidelity_same_id_rows"], 4)
        self.assertTrue(all(row["history_check"]["verified"] for row in result["readout_declarations"]))
        self.assertEqual({row["frozen_case"]["initial_bit"] for row in result["cases"]}, {0})

    def test_declared_prefix_hash_mismatch_is_mechanics_error(self):
        driver = {
            "cases": [
                {
                    "case_id": 0,
                    "policy": "native",
                    "parent_history_ids": [1, 2],
                    "parent_history_sha256": "0" * 64,
                    "queries": [{"name": "case0-native-query0", "query_id": 0, "input_ids": [3], "answer_ids": [4]}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            analyze(Path(temp), None, driver, {})

    def test_conditional_audit_packet_reconstructs_byte_and_top4_reads(self):
        rng = np.random.default_rng(22)
        q = rng.normal(size=(12, 8)).tolist()
        k = rng.normal(size=(2, 6, 8)).tolist()
        v = rng.normal(size=(2, 6, 8)).tolist()
        driver = {
            "cases": [
                {
                    "name": "audit",
                    "task": 0,
                    "policy": "byte_full",
                    "question": "Mira",
                    "consumed_ids": [1],
                    "answer_ids": [2],
                    "attention_audit": [
                        {"layer": 17, "position": 1, "query": q, "native_keys": k, "native_values": v,
                         "byte_keys": k, "byte_values": v}
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            result = analyze(Path(temp), None, driver)
        self.assertEqual(len(result["attention_tail_rows"]), 3)
        self.assertTrue(all("metrics" in row for row in result["attention_tail_rows"]))
        self.assertEqual({row["support_kind"] for row in result["attention_tail_rows"]}, {"byte_full", "byte_top4", "byte_recent4"})

    def test_structured_own_query_audit_separates_conditional_and_actual_drift(self):
        rng = np.random.default_rng(31)
        query = rng.normal(size=(4, 3))
        original_keys = rng.normal(size=(2, 6, 3))
        original_values = rng.normal(size=(2, 6, 3))
        byte_keys = (original_keys * 0.99).astype(np.float32)
        byte_values = (original_values * 1.01).astype(np.float32)
        family = response_family(
            query, original_keys, original_values, byte_keys, byte_values, active_start=2
        )
        actual = family["responses"]["byte_recent4"]
        actual_weights = actual["weights"][:, 2:]
        driver = {
            "cases": [
                {
                    "name": "own-query",
                    "task": "case0",
                    "policy": "byte_recent4",
                    "question": "Mira",
                    "consumed_ids": [1],
                    "answer_ids": [2],
                    "audit_rows": [
                        {
                            "layer": 17,
                            "position": 1,
                            "key_count": 6,
                            "query": query.tolist(),
                            "original_keys": original_keys.tolist(),
                            "original_values": original_values.tolist(),
                            "byte_keys": byte_keys.tolist(),
                            "byte_values": byte_values.tolist(),
                            "actual_context": actual["output"].tolist(),
                            "actual_weights": actual_weights.tolist(),
                            "active_start": 2,
                            "codec_audit_exact": True,
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp:
            result = analyze(Path(temp), None, driver)
        self.assertEqual(result["summary"]["attention_response_rows"], 1)
        packet = result["attention_response_rows"][0]["metrics"]
        self.assertEqual(packet["key_count"], 6)
        self.assertEqual(packet["active_start"], 2)
        self.assertTrue(packet["codec_audit_exact"])
        effects = packet["conditional_effects"]
        self.assertIn("codec_byte_full_minus_native", effects)
        self.assertIn("selection_top4_minus_byte_full", effects)
        self.assertIn("selection_recent4_minus_byte_full", effects)
        self.assertNotIn("history_deleted_minus_native", effects)
        self.assertIn("tail_decomposition", effects["selection_top4_minus_byte_full"])
        self.assertIn("signed_tail_value_contribution", effects["selection_top4_minus_byte_full"]["tail_decomposition"])
        self.assertIn("identity_residual", effects["selection_recent4_minus_byte_full"]["tail_decomposition"])
        self.assertLess(packet["actual_vs_conditional"]["context"]["l2"], 1e-12)
        self.assertLess(packet["actual_vs_conditional"]["weights"]["l2"], 1e-12)

        # A failed byte-cache identity is an instrument limitation for the
        # conditional attribution, not a reason to erase the primary policy
        # outcome or abort the analysis.
        driver["cases"][0]["audit_rows"][0]["codec_audit_exact"] = False
        with tempfile.TemporaryDirectory() as temp:
            result = analyze(Path(temp), None, driver)
        packet = result["attention_response_rows"][0]["metrics"]
        self.assertEqual(packet["codec_reconstruction_status"], "instrument-broken")
        self.assertFalse(packet["actual_vs_conditional"]["usable_for_actual_policy_claim"])

    def test_readout_color_margins_and_four_corner_signed_contrasts(self):
        cases = []
        texts = {}
        margins = {(0, 0): 0.0, (0, 1): 2.0, (1, 0): 1.0, (1, 1): 6.0}
        for case_id, (initial, update) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            name = f"case{case_id}-native-query0"
            logits = [0.0] * 10
            logits[1] = margins[(initial, update)]
            # Token id 4 is the exact bare blue token; alternate surfaces are
            # included to prove they remain explicit diagnostics.
            logits[2] = 0.25
            logits[5] = -0.5
            cases.append(
                {
                    "case_id": case_id,
                    "policy": "native",
                    "queries": [
                        {
                            "name": name,
                            "query_id": 0,
                            "consumed_ids": [1],
                            "answer_ids": [2],
                            "readouts": [{"kind": "query0_final", "position": 1, "logits": logits}],
                        }
                    ],
                }
            )
            texts[name] = "red"
        driver = {"schema": "tcm-memory-response-driver-v1", "cases": cases}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            (root / "inputs" / "runtime-inputs.json").write_text(
                '{"cases":['
                '{"id":0,"initial_bit":0,"update_bit":0,"expected":["red"]},'
                '{"id":1,"initial_bit":0,"update_bit":1,"expected":["red"]},'
                '{"id":2,"initial_bit":1,"update_bit":0,"expected":["red"]},'
                '{"id":3,"initial_bit":1,"update_bit":1,"expected":["red"]}],'
                '"policies":["native"],"questions":["Mira"],'
                '"color_token_variants":{"red":{"red":[1]," red":[2],"Red":[3]},'
                '"blue":{"blue":[4]," blue":[5],"Blue":[6]}}}\n'
            )
            result = analyze(root, None, driver, texts)
        self.assertEqual(result["summary"]["color_margin_readouts"], 4)
        contrasts = result["four_corner_contrasts"]
        self.assertEqual(len(contrasts), 1)
        contrast = contrasts[0]
        self.assertEqual(contrast["status"], "complete")
        self.assertAlmostEqual(contrast["initial_signed_contrast"], 2.5)
        self.assertAlmostEqual(contrast["update_signed_contrast"], 3.5)
        self.assertAlmostEqual(contrast["mixed_signed_interaction"], 3.0)
        self.assertAlmostEqual(contrast["initial_edge_update0"], 1.0)
        self.assertAlmostEqual(contrast["initial_edge_update1"], 4.0)
        self.assertAlmostEqual(contrast["update_edge_initial0"], 2.0)
        self.assertAlmostEqual(contrast["update_edge_initial1"], 5.0)
        self.assertTrue(contrast["same_absolute_position"])
        readout = result["readout_declarations"][0]["color_margin"]
        self.assertAlmostEqual(readout["primary"]["red_minus_blue"], 0.0)
        self.assertEqual(readout["primary"]["status"], "single_token_exact")
        self.assertEqual(readout["variant_pooled_single_token"]["status"], "diagnostic_variant_pooled_single_token")
        self.assertAlmostEqual(readout["variant_margins"]["leading_lower"]["red_minus_blue"], 0.75)


if __name__ == "__main__":
    unittest.main()
