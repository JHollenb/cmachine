# Reversible chat branch demo

**Fresh result:** guarded CUDA job `job-995c8e81e7ca` produced `Green`, rolled back and regenerated `Green` with identical generated IDs, cache and full-logit hashes, then rolled back and committed the alternative `Yellow` branch. Read the [complete result and protocol journal](../results/pretrained/README.md).

This is a small public client for the C machine's persistent Qwen token
server.  It keeps tokenization outside inference and asks the server to do all
model execution.  The demo creates one parent conversation, generates an
answer, pushes a native branch frame, runs two different user updates, and
replays the first update after rollback.  Replay is generated again from the
same user text and tokenizer state; saved answer IDs are reported for audit
but are never injected into the server.

The report records the actual user and assistant text, every generated token
ID, token-clock positions, pending-token ownership, cache hashes, full-vocab
logit hashes, rollback metadata, server identity, source hashes, and protocol
wall times.  Native rollback intentionally invalidates the current logits
readout.  The demo checks the restored cache/pending/root state and only
compares logits after the replay has recomputed them.

## Requirements

* A CUDA build of `cmachine` containing the `serve` protocol.  Build it in
  the repository with `make cuda` on a compatible host.
* An explicit Qwen2.5-1.5B-Instruct model bundle accepted by that binary.
* A local Hugging Face tokenizer directory for the same model.  Install
  `transformers` in the Python environment used to launch the client.  The
  tokenizer is loaded with `local_files_only=True`; no network fetch is part
  of the demo.

The client does not contain model or tokenizer paths.  Pass them explicitly,
which also makes the command suitable for an admitted mrun worker.

## Usage

```bash
python3 cmachine/demos/chat_branch/chat_branch.py \
  --binary /path/to/tcm-cuda \
  --model /path/to/qwen-model.bundle \
  --tokenizer /path/to/qwen-tokenizer \
  --context 512 \
  --new-tokens 16 \
  --output /path/to/chat-branch.json
```

The default initial prompt is a short color binding.  `--prompt`,
`--update-a`, and `--update-b` make the same protocol useful for a short prose
prompt or another explicitly chosen test.  `--context 768` is also supported.
`native` is the default cache policy.  `full-byte` and `bounded` are exposed
for diagnostics when the corresponding server build supports them, but the
branch result should be interpreted with the policy named in the report.

The client follows the token ownership rule used by `tools/chat.py`: after a
generation, the final generated ID remains the server's pending token and is
therefore omitted from the client's processed prefix until the next append.
This is why the branch code can restore the Python conversation snapshot and
reconstruct the next chat-template suffix without teacher-forcing an answer.

## Report interpretation

`comparisons.rollback_to_root` checks position, pending token, delivered flag,
epoch, cache digest, and stack metadata.  `checkpoint_serial` is intentionally
reported separately because native rollback increments it.  A rollback state
with `logits_valid: false` is expected.  `comparisons.replay_exact` compares
the replayed generated IDs, cache digest, pending position, and full-vocabulary
logit digest.  These are mechanics observations; the generated prose and
color behavior are recorded as observations rather than a quality gate.

The cost section reports protocol command counts, generated/appended token
counts, server wall time, model/session allocation sizes, and the server's
declared arithmetic profile.  Command wall time is orchestration timing, not a
kernel benchmark.

The stable top-level report shape is:

```text
schema/status/mechanics_status/instrument_status/trend_status/terminal_status
profile/inputs/server/token_authority
turns/{initial,branch_a,branch_a_replay,branch_b}
states/{initial,root,parent,branch_a,branch_a_rollback,parent_replay,
        branch_a_replay,branch_a_replay_rollback,branch_b_parent,branch_b,final}
branch_frames/comparisons/cost/protocol_events
```

Each turn contains the template `input_ids`, the newly appended suffix,
server-generated `generated_ids`, decoded text, and its generation response.
Each state is the server's own `S` response, including position,
`pending_token`, cache digest, optional full-vocabulary logits digest, and
rollback metadata.  An error report keeps `status: "error"` and the error
type/message rather than presenting partial evidence as a successful replay.
