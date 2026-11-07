"""File-identity checks for a real tokenizer-cache layout; no model execution."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from chat_branch import ChatError, _tokenizer_manifest


class TokenizerIdentityTests(unittest.TestCase):
    def test_snapshot_symlink_is_hashed_and_changed_content_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            blob = root / "blobs" / "config"
            blob.parent.mkdir()
            blob.write_bytes(b'{"model_type":"qwen2"}\n')
            (snapshot / "config.json").symlink_to(blob)
            expected = hashlib.sha256(blob.read_bytes()).hexdigest()
            sidecar = {"tokenizer_files": {"config.json": expected}}
            report = _tokenizer_manifest(snapshot, sidecar)
            self.assertEqual(report["files"]["config.json"]["sha256"], expected)
            blob.write_bytes(b"changed")
            with self.assertRaisesRegex(ChatError, "hash mismatch"):
                _tokenizer_manifest(snapshot, sidecar)

    def test_manifest_parent_traversal_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ChatError, "escapes tokenizer directory"):
                _tokenizer_manifest(Path(directory), {"tokenizer_files": {"../config.json": "unused"}})

    def test_absolute_manifest_address_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ChatError, "escapes tokenizer directory"):
                _tokenizer_manifest(Path(directory), {"tokenizer_files": {"/config.json": "unused"}})


if __name__ == "__main__":
    unittest.main()
