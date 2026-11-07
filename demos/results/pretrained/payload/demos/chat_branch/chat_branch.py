#!/usr/bin/env python3
"""A small, public, reversible Qwen chat session.

The C token server owns model state and token generation.  This program owns
only the tokenizer-side conversation view and the protocol journal.  In
particular, generated token IDs are always taken from a ``G`` response; a
branch replay recomputes its prompt suffix and asks the model to generate
again rather than teacher-forcing a saved answer.

The server protocol is documented in ``cmachine/README.md`` and in
``cmachine/src/qwen_serve.c``.  Keeping this client independent of the C
implementation makes the example useful with an explicitly supplied binary,
model bundle, and tokenizer on any compatible machine.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


DEFAULT_PROMPT = "Mira's badge is red. Niko's badge is blue. Reply OK."
DEFAULT_UPDATE_A = "Correction: Mira's badge is green. What color is Mira's badge? Reply with only the color."
DEFAULT_UPDATE_B = "Correction: Mira's badge is yellow. What color is Mira's badge? Reply with only the color."


class ChatError(RuntimeError):
    """A protocol, tokenizer, or state-contract error."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _jsonable_ids(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        # JSON and tokenizer backends sometimes return numpy integer scalars.
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ChatError(f"tokenizer returned a non-integer token: {value!r}") from exc
        if integer < 0:
            raise ChatError(f"tokenizer returned a negative token: {integer}")
        result.append(integer)
    return result


def _state_view(state: dict[str, Any]) -> dict[str, Any]:
    """Return stable state fields used by replay comparisons."""

    fields = (
        "position",
        "pending_token",
        "delivered",
        "epoch",
        "cache_sha256",
        "stack_depth",
        "top_frame",
    )
    return {key: state.get(key) for key in fields}


def _state_match(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "position",
        "pending_token",
        "delivered",
        "epoch",
        "cache_sha256",
        "stack_depth",
        "top_frame",
    )
    equal = {key: left.get(key) == right.get(key) for key in keys}
    return {
        "equal": all(equal.values()),
        "fields": equal,
        "left": _state_view(left),
        "right": _state_view(right),
    }


@dataclass
class Conversation:
    """Tokenizer-side state matching ``tools/chat.py``'s ownership rule."""

    messages: list[dict[str, str]] = field(default_factory=list)
    processed: list[int] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "messages": copy.deepcopy(self.messages),
            "processed": list(self.processed),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.messages = copy.deepcopy(snapshot["messages"])
        self.processed = list(snapshot["processed"])


class TokenServer:
    """Line-oriented client for ``qwen serve``."""

    def __init__(self, binary: Path, model: Path, context: int, policy: str,
                 anchors: int, recent: int, reservoir: int) -> None:
        command = [str(binary), "serve", str(model), "--context", str(context)]
        if policy != "native":
            command += ["--cache-policy", policy]
            if policy == "bounded":
                command += [
                    "--anchors", str(anchors),
                    "--recent", str(recent),
                    "--reservoir", str(reservoir),
                ]
        self.command_line = command
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.events: list[dict[str, Any]] = []
        self.closed = False
        self.ready = self._read_json("ready")

    def _read_json(self, label: str) -> dict[str, Any]:
        if self.process.stdout is None:
            raise ChatError("token server stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read()[-2000:]
            raise ChatError(f"token server ended while waiting for {label}; {stderr}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ChatError(f"token server emitted non-JSON for {label}: {line!r}") from exc
        if not isinstance(value, dict):
            raise ChatError(f"token server emitted non-object for {label}: {value!r}")
        return value

    def send(self, command: str, label: Optional[str] = None) -> dict[str, Any]:
        if self.closed or self.process.stdin is None:
            raise ChatError("token server is closed")
        started = time.monotonic()
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        response = self._read_json(label or command.split()[0])
        elapsed = time.monotonic() - started
        event = {
            "label": label or command.split()[0],
            "command": command,
            "response": response,
            "wall_seconds": elapsed,
        }
        self.events.append(event)
        if response.get("event") == "error" or response.get("error"):
            raise ChatError(f"token server rejected {command!r}: {response}")
        return response

    def state(self, label: str) -> dict[str, Any]:
        response = self.send("S", label)
        required = {
            "event",
            "position",
            "pending_token",
            "delivered",
            "stack_depth",
            "top_frame",
            "epoch",
            "cache_sha256",
            "logits_sha256",
        }
        missing = sorted(required.difference(response))
        if response.get("event") != "state" or missing:
            raise ChatError(f"state response does not satisfy the token-server contract: {response}")
        # Native qwen_serve reports these two explicit lifecycle fields.
        # Bounded/full-byte servers expose the same hashes and position fields
        # but omit them; derive only the diagnostic validity bit there and
        # keep checkpoint_serial absent rather than inventing a serial.
        response.setdefault("checkpoint_serial", None)
        response.setdefault("logits_valid", bool(response.get("logits_sha256")))
        return response

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write("Q\n")
                self.process.stdin.flush()
                self.process.stdout.readline() if self.process.stdout is not None else None
            except (BrokenPipeError, OSError):
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)

    def __enter__(self) -> "TokenServer":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ChatError(
            "the tokenizer dependency is missing; install transformers in the "
            "environment used for the demo"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # tokenizer libraries expose several exception types
        raise ChatError(f"could not load local tokenizer {path}: {exc}") from exc


def tokenize_chat(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        values = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except Exception as exc:
        raise ChatError(f"tokenizer chat template failed: {exc}") from exc
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise ChatError(f"chat template returned {type(values).__name__}, expected a token list")
    # Some tokenizers return a one-element batch for this API.
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ChatError("chat template returned a batch instead of one conversation")
        values = list(values[0])
    return _jsonable_ids(values)


def decode_tokens(tokenizer: Any, ids: list[int], skip_special_tokens: bool) -> str:
    try:
        return str(tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))
    except Exception as exc:
        raise ChatError(f"tokenizer decode failed: {exc}") from exc


def _tokenizer_manifest(tokenizer_path: Path, model_sidecar: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Hash local tokenizer files that are named by an optional model manifest."""

    names: set[str] = set()
    if isinstance(model_sidecar, dict):
        declared = model_sidecar.get("tokenizer_files", {})
        if isinstance(declared, dict):
            names.update(str(name) for name in declared)
        elif isinstance(declared, list):
            names.update(str(name) for name in declared)
    if not names:
        names.update(
            name for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "merges.txt",
                "vocab.json",
                "added_tokens.json",
                "tokenizer.model",
                "spiece.model",
            ) if (tokenizer_path / name).is_file()
        )
    files: dict[str, Any] = {}
    declared_hashes = model_sidecar.get("tokenizer_files", {}) if isinstance(model_sidecar, dict) else {}
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ChatError(f"tokenizer manifest escapes tokenizer directory: {name}")
        # Hugging Face snapshots link into a sibling blobs directory. Validate
        # the declared relative address, then hash the actual linked content.
        candidate = tokenizer_path / relative
        if not candidate.is_file():
            raise ChatError(f"declared tokenizer file is missing: {name}")
        actual = sha256_file(candidate)
        expected = None
        if isinstance(declared_hashes, dict):
            item = declared_hashes.get(name)
            if isinstance(item, str):
                expected = item
            elif isinstance(item, dict):
                expected = item.get("sha256") or item.get("content_sha256")
        if expected is not None and actual != expected:
            raise ChatError(f"tokenizer hash mismatch for {name}: {actual} != {expected}")
        files[name] = {"sha256": actual, "bytes": candidate.stat().st_size}
    return {"root": str(tokenizer_path), "files": files}


def _model_manifest(model: Path, ready: dict[str, Any]) -> dict[str, Any]:
    sidecar_path = model.with_suffix(".json")
    sidecar: Optional[dict[str, Any]] = None
    if sidecar_path.is_file():
        try:
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ChatError(f"could not read model identity sidecar {sidecar_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ChatError(f"model identity sidecar is not an object: {sidecar_path}")
        sidecar = loaded
    expected_digest = None if sidecar is None else (
        sidecar.get("content_sha256") or sidecar.get("model_digest")
    )
    actual_digest = ready.get("model_digest")
    if expected_digest is not None and actual_digest is not None and expected_digest != actual_digest:
        raise ChatError(f"model digest disagrees with server identity: {expected_digest} != {actual_digest}")
    expected_identity = None if sidecar is None else sidecar.get("model_identity") or sidecar.get("identity")
    actual_identity = ready.get("model_identity")
    if expected_identity is not None and actual_identity is not None and expected_identity != actual_identity:
        raise ChatError(f"model identity disagrees with server: {expected_identity!r} != {actual_identity!r}")
    return {
        "path": str(model),
        "identity_sidecar": str(sidecar_path) if sidecar is not None else None,
        "sidecar": sidecar,
        "server_identity": actual_identity,
        "server_digest": actual_digest,
    }


def _turn(
    server: TokenServer,
    tokenizer: Any,
    conversation: Conversation,
    user_text: str,
    budget: int,
    label: str,
) -> dict[str, Any]:
    proposed = conversation.messages + [{"role": "user", "content": user_text}]
    full_ids = tokenize_chat(tokenizer, proposed)
    prefix_match = full_ids[: len(conversation.processed)] == conversation.processed
    if not prefix_match:
        raise ChatError(
            f"chat template no longer preserves processed token prefix at {label}; "
            "refusing to reset a branch implicitly"
        )
    suffix = full_ids[len(conversation.processed) :]
    append_response: Optional[dict[str, Any]] = None
    if suffix:
        append_response = server.send("A " + " ".join(str(token) for token in suffix), f"{label}.append")
    generation = server.send(f"G {budget}", f"{label}.generate")
    generated = _jsonable_ids(generation.get("ids", []))
    if not generated:
        raise ChatError(f"server returned no generated IDs at {label}")
    assistant_text = decode_tokens(tokenizer, generated, skip_special_tokens=True)
    raw_assistant_text = decode_tokens(tokenizer, generated, skip_special_tokens=False)
    # The final generated token remains pending in the native server.  It is
    # intentionally omitted from processed, just as tools/chat.py does.
    conversation.messages = proposed + [{"role": "assistant", "content": assistant_text}]
    conversation.processed = full_ids + generated[:-1]
    return {
        "label": label,
        "user_text": user_text,
        "input_ids": full_ids,
        "appended_ids": suffix,
        "append_response": append_response,
        "generated_ids": generated,
        "generated_text": assistant_text,
        "generated_text_with_special_tokens": raw_assistant_text,
        "generation_response": generation,
        "state_conversation": conversation.snapshot(),
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ChatError(message)


def _run_demo(args: argparse.Namespace) -> dict[str, Any]:
    binary = Path(args.binary).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    for label, path in (("binary", binary), ("model", model), ("tokenizer", tokenizer_path)):
        if not path.exists():
            raise ChatError(f"{label} path does not exist: {path}")
    if args.context < 1 or args.context > 32768:
        raise ChatError("context must be between 1 and 32768")
    if args.new_tokens < 1 or args.new_tokens > 1024:
        raise ChatError("new-tokens must be between 1 and 1024")

    tokenizer = load_tokenizer(tokenizer_path)
    server: Optional[TokenServer] = None
    started = time.monotonic()
    try:
        server = TokenServer(
            binary,
            model,
            args.context,
            args.cache_policy,
            args.anchors,
            args.recent,
            args.reservoir,
        )
        ready = server.ready
        model_manifest = _model_manifest(model, ready)
        tokenizer_manifest = _tokenizer_manifest(tokenizer_path, model_manifest.get("sidecar"))
        conversation = Conversation()
        initial = _turn(server, tokenizer, conversation, args.prompt, args.new_tokens, "initial")
        initial_state = server.state("initial.state")
        root_snapshot = conversation.snapshot()
        root_state = server.state("root.state")

        parent = server.send("P", "parent.push")
        parent_frame = parent.get("frame", parent.get("id"))
        parent_state = server.state("parent.state")

        branch_a = _turn(server, tokenizer, conversation, args.update_a, args.new_tokens, "branch_a")
        branch_a_state = server.state("branch_a.state")

        rollback_a = server.send(f"B {parent_frame}", "branch_a.rollback")
        rollback_a_state = server.state("branch_a.rollback.state")
        conversation.restore(root_snapshot)

        _assert(_state_match(root_state, rollback_a_state)["equal"], "branch A rollback did not restore root state")
        if args.cache_policy == "native":
            _assert(not bool(rollback_a_state.get("logits_valid")), "native rollback unexpectedly retained valid logits")

        parent_replay = server.send("P", "parent.replay.push")
        replay_frame = parent_replay.get("frame", parent_replay.get("id"))
        parent_replay_state = server.state("parent.replay.state")
        branch_a_replay = _turn(server, tokenizer, conversation, args.update_a, args.new_tokens, "branch_a.replay")
        branch_a_replay_state = server.state("branch_a.replay.state")

        replay_equal = {
            "generated_ids": branch_a["generated_ids"] == branch_a_replay["generated_ids"],
            "cache_sha256": branch_a_state.get("cache_sha256") == branch_a_replay_state.get("cache_sha256"),
            "logits_sha256": branch_a_state.get("logits_sha256") == branch_a_replay_state.get("logits_sha256"),
            "position": branch_a_state.get("position") == branch_a_replay_state.get("position"),
            "pending_token": branch_a_state.get("pending_token") == branch_a_replay_state.get("pending_token"),
        }
        _assert(all(replay_equal.values()), "branch A replay diverged from the same generated history")

        rollback_replay = server.send(f"B {replay_frame}", "branch_a.replay.rollback")
        rollback_replay_state = server.state("branch_a.replay.rollback.state")
        conversation.restore(root_snapshot)
        _assert(_state_match(root_state, rollback_replay_state)["equal"], "replay rollback did not restore root")

        parent_b = server.send("P", "parent.branch_b.push")
        branch_b_frame = parent_b.get("frame", parent_b.get("id"))
        branch_b_parent_state = server.state("parent.branch_b.state")
        branch_b = _turn(server, tokenizer, conversation, args.update_b, args.new_tokens, "branch_b")
        branch_b_state = server.state("branch_b.state")
        _assert(args.update_a != args.update_b, "branch updates must be different")
        commit_b = server.send(f"K {branch_b_frame}", "branch_b.commit")
        final_state = server.state("final.state")

        frame_ids = [parent_frame, replay_frame, branch_b_frame]
        _assert(len(set(frame_ids)) == len(frame_ids), "rollback did not consume branch markers uniquely")
        _assert(final_state.get("stack_depth") == 0, "committed branch left an open stack frame")
        rollback_serial_incremented = (
            args.cache_policy != "native"
            or rollback_a_state.get("checkpoint_serial") == root_state.get("checkpoint_serial", 0) + 1
        )
        _assert(rollback_serial_incremented, "native rollback did not advance checkpoint serial")
        return {
            "schema": "tcm-chat-branch-demo-v1",
            "status": "ok",
            "mechanics_status": "valid",
            "instrument_status": "native_token_server_events",
            "trend_status": "observational_only",
            "terminal_status": "not_requested",
            "profile": {
                "cache_policy": args.cache_policy,
                "context": args.context,
                "new_tokens_per_turn": args.new_tokens,
                "arithmetic_contract": "native Qwen C runtime: FP32 cuBLAS pedantic contractions; BF16 weights widened once",
                "logit_readout": "server state logits_sha256 is a full-vocabulary FP32 readout hash when logits_valid=true",
                "timing_scope": "wall time around line-protocol commands; not a kernel benchmark",
            },
            "inputs": {
                "prompt": args.prompt,
                "update_a": args.update_a,
                "update_b": args.update_b,
                "tokenizer": tokenizer_manifest,
                "model": model_manifest,
                "binary": {"path": str(binary), "sha256": sha256_file(binary), "bytes": binary.stat().st_size},
                "client": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
            "server": {
                "command": server.command_line,
                "ready": ready,
                "model_identity": ready.get("model_identity"),
                "model_digest": ready.get("model_digest"),
                "capacity": ready.get("capacity"),
                "gpu_weight_bytes": ready.get("gpu_weight_bytes"),
                "gpu_session_bytes": ready.get("gpu_session_bytes"),
            },
            "token_authority": {
                "generated_ids_source": "G responses from the native server",
                "replay_mode": "same user text and tokenizer suffix; no generated answer IDs were injected",
                "pending_ownership": "processed excludes the final pending generated token until the next A/G transition",
            },
            "turns": {
                "initial": initial,
                "branch_a": branch_a,
                "branch_a_replay": branch_a_replay,
                "branch_b": branch_b,
            },
            "states": {
                "initial": initial_state,
                "root": root_state,
                "parent": parent_state,
                "branch_a": branch_a_state,
                "branch_a_rollback": rollback_a_state,
                "parent_replay": parent_replay_state,
                "branch_a_replay": branch_a_replay_state,
                "branch_a_replay_rollback": rollback_replay_state,
                "branch_b_parent": branch_b_parent_state,
                "branch_b": branch_b_state,
                "final": final_state,
            },
            "branch_frames": {
                "first": parent_frame,
                "replay": replay_frame,
                "branch_b": branch_b_frame,
                "unique_after_rollback": True,
            },
            "comparisons": {
                "rollback_to_root": _state_match(root_state, rollback_a_state),
                "replay_rollback_to_root": _state_match(root_state, rollback_replay_state),
                "replay_exact": replay_equal,
                "native_rollback_logits_invalid_expected": args.cache_policy == "native",
                "rollback_logits_valid_observed": bool(rollback_a_state.get("logits_valid")),
                "native_checkpoint_serial_incremented": rollback_serial_incremented,
                "branch_updates_different": args.update_a != args.update_b,
            },
            "cost": {
                "command_events": len(server.events),
                "append_token_count": sum(len(turn["appended_ids"]) for turn in (initial, branch_a, branch_a_replay, branch_b)),
                "generated_token_count": sum(len(turn["generated_ids"]) for turn in (initial, branch_a, branch_a_replay, branch_b)),
                "server_wall_seconds": sum(float(event["wall_seconds"]) for event in server.events),
                "client_wall_seconds": time.monotonic() - started,
                "gpu_weight_bytes": ready.get("gpu_weight_bytes"),
                "gpu_session_bytes": ready.get("gpu_session_bytes"),
            },
            "protocol_events": server.events,
            "process_returncode_before_close": server.process.poll(),
        }
    finally:
        if server is not None:
            server.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="path to the built cmachine binary")
    parser.add_argument("--model", required=True, help="path to the explicit model bundle")
    parser.add_argument("--tokenizer", required=True, help="local tokenizer directory")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--context", type=int, default=512, choices=(512, 768), help="server context capacity")
    parser.add_argument("--new-tokens", type=int, default=8, help="generation budget per turn")
    parser.add_argument("--cache-policy", choices=("native", "full-byte", "bounded"), default="native")
    parser.add_argument("--anchors", type=int, default=32, help="bounded cache anchor rows")
    parser.add_argument("--recent", type=int, default=64, help="bounded cache recent rows")
    parser.add_argument("--reservoir", type=int, default=128, help="bounded cache reservoir rows")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="initial user prompt")
    parser.add_argument("--update-a", default=DEFAULT_UPDATE_A, help="first branch user update")
    parser.add_argument("--update-b", default=DEFAULT_UPDATE_B, help="second branch user update")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any]
    try:
        report = _run_demo(args)
    except Exception as exc:
        report = {
            "schema": "tcm-chat-branch-demo-v1",
            "status": "error",
            "mechanics_status": "incomplete",
            "instrument_status": "incomplete",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        print(f"chat branch demo failed: {exc}", file=sys.stderr)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report.get("status") != "ok":
        return 2
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "initial": report["turns"]["initial"]["generated_text"],
        "branch_a": report["turns"]["branch_a"]["generated_text"],
        "branch_b": report["turns"]["branch_b"]["generated_text"],
        "replay_exact": report["comparisons"]["replay_exact"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
