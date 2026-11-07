"""Tiny synthetic source/IR mechanics; no pretrained or numerical model run."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import intake as module
from intake import intake


def fixture(root, family="qwen2", hidden=4, fill=0):
    """Declarative checkpoint with inert bytes, not a trained test model."""
    root = Path(root)
    root.mkdir()
    heads = hidden // 2
    config = {
        "architectures": ["Qwen2ForCausalLM" if family == "qwen2" else "Qwen3ForCausalLM"],
        "model_type": family, "hidden_size": hidden, "intermediate_size": 8,
        "num_hidden_layers": 1, "num_attention_heads": heads, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": 8, "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        "hidden_act": "silu", "tie_word_embeddings": True,
        "bos_token_id": 1, "eos_token_id": [2, 3], "pad_token_id": 0,
        "attention_dropout": 0.0, "use_cache": True,
    }
    if family == "qwen3":
        config["attention_bias"] = False
    shapes = {
        "model.embed_tokens.weight": (8, hidden), "model.norm.weight": (hidden,),
        "model.layers.0.input_layernorm.weight": (hidden,),
        "model.layers.0.post_attention_layernorm.weight": (hidden,),
        "model.layers.0.self_attn.q_proj.weight": (hidden, hidden),
        "model.layers.0.self_attn.k_proj.weight": (2, hidden),
        "model.layers.0.self_attn.v_proj.weight": (2, hidden),
        "model.layers.0.self_attn.o_proj.weight": (hidden, hidden),
        "model.layers.0.mlp.gate_proj.weight": (8, hidden),
        "model.layers.0.mlp.up_proj.weight": (8, hidden),
        "model.layers.0.mlp.down_proj.weight": (hidden, 8),
    }
    if family == "qwen2":
        shapes.update({f"model.layers.0.self_attn.{name}_proj.bias": (width,)
                       for name, width in (("q", hidden), ("k", 2), ("v", 2))})
    else:
        shapes.update({f"model.layers.0.self_attn.{name}_norm.weight": (2,)
                       for name in ("q", "k")})
    header = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for name, shape in sorted(shapes.items()):
        end = cursor + 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [cursor, end]}
        cursor = end
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes([fill]) * cursor)
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}\n")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (root / "generation_config.json").write_text(json.dumps({"do_sample": False, "max_new_tokens": 8}))
    return root


class IntakeMechanics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_two_declared_architectures_produce_exact_refs_without_execution_claim(self):
        from manalysis.decompiler_bridge import validate_exact_graph

        for family, hidden in (("qwen2", 4), ("qwen3", 6)):
            with self.subTest(family=family):
                source = fixture(self.root / family, family, hidden)
                output = self.root / (family + "-intake")
                report = intake(source=source, output=output, revision="a" * 40)
                self.assertEqual(report["status"], "structurally-imported")
                self.assertFalse(report["authority"]["c_execution_qualified"])
                self.assertFalse(report["authority"]["semantic_claim"])
                self.assertEqual(report["inventory"]["state_slot_count"], 3)
                graph = json.loads((output / report["artifacts"]["graph"]).read_text())
                validate_exact_graph(graph)
                refs = {node["ref"] for node in graph["nodes"] if node["kind"] == "operation"}
                self.assertEqual(refs, {op["exact_ref"] for op in report["inventory"]["operations"]})
                priors = json.loads((output / report["artifacts"]["role_priors"]).read_text())
                self.assertFalse(priors["claim_discipline"]["semantic_claim"])
                self.assertEqual(priors["claim_discipline"]["status"], "hypothesis-only")

    def test_roundtrip_preserves_exact_identity_and_weaker_raw_ir_scope(self):
        source = fixture(self.root / "source")
        first = intake(source=source, output=self.root / "first")
        bundle_path = self.root / "first" / first["artifacts"]["bundle"]
        second = intake(ir=bundle_path, output=self.root / "second")
        self.assertEqual(first["bundle_fingerprint"], second["bundle_fingerprint"])
        self.assertEqual(second["source_validation_scope"], "coordinated-ir-bundle")
        self.assertEqual(second["workspace"]["validation"]["scope"], "coordinated-ir-bundle")
        self.assertIsNone(second["source_report"])
        wrapper = intake(ir=self.root / "first" / "input.json", output=self.root / "wrapper")
        self.assertEqual(wrapper["source_validation_scope"], "coordinated-ir-bundle")
        self.assertIsNone(wrapper["source_report"])

    def test_changed_parameter_bytes_change_source_bound_graph_identity(self):
        source = fixture(self.root / "source", fill=0)
        one = intake(source=source, output=self.root / "one-out")
        weights = source / "model.safetensors"
        raw = bytearray(weights.read_bytes())
        raw[-1] = 1
        weights.write_bytes(raw)
        two = intake(source=source, output=self.root / "two-out")
        self.assertNotEqual(one["bundle_fingerprint"], two["bundle_fingerprint"])
        self.assertNotEqual(one["workspace"]["graph_fingerprint"], two["workspace"]["graph_fingerprint"])

    def test_unknown_family_preserves_refusal_without_guessing_graph(self):
        source = fixture(self.root / "source")
        config = json.loads((source / "config.json").read_text())
        config["model_type"] = "unknown-family"
        config["architectures"] = ["UnknownArchitecture"]
        (source / "config.json").write_text(json.dumps(config))
        result = intake(source=source, output=self.root / "output")
        self.assertIn(result["status"], ("unsupported", "rejected"))
        self.assertIsNone(result["bundle_fingerprint"])
        self.assertIsNone(result["inventory"])
        self.assertNotIn("graph", result["artifacts"])
        self.assertTrue(result["source_report"]["failures"])

    def test_tampered_ir_is_rejected_before_output_creation(self):
        report = intake(source=fixture(self.root / "source"), output=self.root / "first")
        raw = json.loads((self.root / "first" / report["artifacts"]["bundle"]).read_text())
        raw["model"]["architecture_id"] = "forged"
        path = self.root / "tampered.json"
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            intake(ir=path, output=self.root / "refused")
        self.assertFalse((self.root / "refused").exists())

    def test_output_cannot_overwrite_or_mutate_frozen_source(self):
        source = fixture(self.root / "source")
        output = self.root / "output"
        intake(source=source, output=output)
        before = (output / "intake.json").read_bytes()
        with self.assertRaises(FileExistsError):
            intake(source=source, output=output)
        self.assertEqual(before, (output / "intake.json").read_bytes())
        with self.assertRaises(ValueError):
            intake(source=source, output=source / "output")
        self.assertFalse((source / "output").exists())

    def test_canonical_mutation_refusal_is_retained(self):
        from mrun.decompiler import SafetensorsTensorIndexer

        source = fixture(self.root / "source")
        original = SafetensorsTensorIndexer.build

        def mutate(indexer, frozen):
            index = original(indexer, frozen)
            config = source / "config.json"
            config.write_text(config.read_text() + " ")
            return index

        with mock.patch.object(SafetensorsTensorIndexer, "build", mutate):
            report = intake(source=source, output=self.root / "output")
        self.assertEqual(report["status"], "rejected")
        self.assertIsNone(report["bundle_fingerprint"])
        self.assertTrue((self.root / "output" / "intake.json").exists())
        self.assertTrue(report["source_report"]["failures"])

    def test_late_source_mutation_cannot_publish_success(self):
        source = fixture(self.root / "source")
        original = module.inventory

        def mutate(bundle):
            result = original(bundle)
            config = source / "config.json"
            config.write_text(config.read_text() + " ")
            return result

        with mock.patch.object(module, "inventory", mutate):
            with self.assertRaises(RuntimeError):
                intake(source=source, output=self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_completion_marker_is_published_from_complete_bytes(self):
        source = fixture(self.root / "source")
        output = self.root / "output"
        original = module.os.link
        checked = []

        def link(temporary, target):
            if Path(target).name == "intake.json":
                self.assertFalse(Path(target).exists())
                report = json.loads(Path(temporary).read_text())
                fingerprint = report.pop("fingerprint")
                self.assertEqual(module.sealed(report)["fingerprint"], fingerprint)
                self.assertTrue((output / "REPORT.md").exists())
                checked.append(True)
            return original(temporary, target)

        with mock.patch.object(module.os, "link", link):
            intake(source=source, output=output)
        self.assertEqual(checked, [True])
        self.assertTrue((output / "intake.json").exists())


if __name__ == "__main__":
    unittest.main()
