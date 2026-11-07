# Fresh mrun reproduction plan (not submitted)

This file is a staged execution plan only.  It deliberately contains no
machine-local model path and has not submitted a job.  A worker should receive
the model and tokenizer locations through the mrun lease or an explicit input
manifest, then invoke `chat_branch.py` with those paths.

## Source and identity

* Build from the reviewed cmachine commit selected by the caller.  Stage the
  demo script, README, and the exact C server sources used by the binary from
  that commit.
* Require the server `ready` event's `model_identity` and `model_digest`.
  If the model bundle has an adjacent identity JSON, the client checks its
  declared digest and identity before the first append.
* Load tokenizer files locally and record their SHA-256 values.  A model
  identity manifest may declare expected tokenizer hashes; a mismatch stops
  the run before any model state is mutated.
* Keep the output JSON, the invocation receipt, the staged source manifest,
  and the binary/model/tokenizer identity records together in the mrun result
  directory.  Preserve a failed or partial report rather than rewriting it.

## Smallest reproduction

Use one guarded CUDA lease, one model process, one short color-binding prompt,
and a generation budget of 8 or 16.  Run the same client with `--context 512`
first; repeat at 768 only when the first run is mechanically valid.  The
invocation is conceptually:

```text
python3 cmachine/demos/chat_branch/chat_branch.py \
  --binary <leased-binary> --model <leased-model> \
  --tokenizer <leased-tokenizer> --context 512 --new-tokens 16 \
  --output <result>/chat-branch.json
```

The worker must not provide answer IDs or a cache dump as inputs.  The server
generates each answer.  The report's own IDs, cache digests, and full-vocab
logit digests are the output authority.

## Expected lifecycle evidence

1. Record `ready`, append the template-generated initial prompt suffix, and
   generate an answer.  Capture the parent `S` state, then push `P`.
2. Run update A and generate its own answer.  Roll back with `B`, capture the
   restored cache/pending/root state, and record that native logits are
   invalid until a new forward pass.
3. Push a fresh frame, run update A again, and require exact generated IDs,
   cache hash, pending position, and post-generation full-vocabulary logits
   hash.  Roll it back, push another frame, run the different update B, and
   commit it.
4. Preserve all line-protocol events and the Python token snapshots.  A
   failure is classified as an identity/schema/finite/rollback mechanics
   failure; answer quality remains an observation.

## mrun staging hook

The repository's existing commit-bound integration harness is
[`../../validation/run.py`](../../validation/run.py).  Reuse its source
staging and guarded `mrun.client.submit.launch` pattern, but make this demo a
separate one-process experiment: request CUDA, use one finite lease, disable
retries, and pass model/tokenizer paths from the lease rather than copying the
private paths used by historical fleet scripts.  The final submission should
bind the selected commit, source-manifest hash, model identity/digest,
tokenizer manifest, context, policy, and output schema in its request and
receipt.  Do not submit this plan automatically; root review is the final
approval step.
