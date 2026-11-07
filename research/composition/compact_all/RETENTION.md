# Bounded retention needs bounded recovery

Prospective design, November 6, 2026. The [subsequent bounded-retention experiment](../bounded_retention/FINDINGS.md) now executes the fixed-snapshot alternative below, including anchors/recent rows and a paired-row reservoir. Bounded allocation and recovery work; all bounded policies fail its four delayed questions. The fixed-horizon undo-journal alternative remains design-stage. Fixed storage is not a promise of unrestricted exact recall.

## State and append

Separate the model's logical position limit from memory capacity. Keep A structural anchor rows and R recent rows per layer in a circular byte arena, with absolute post-RoPE positions. Consume their union once when anchor and recent sets overlap. Score/decode scratch scales with A+R, not the total logical cursor. New actual K/V is encoded before the current attention read, retaining the established contract. Retired rows are inaccessible to serving attention.

Start with an explicit support policy, without centroid fitting or semantic parsing: first A rows plus the most recent R rows. A full-retention byte baseline supplies the representation control. A recent-only arm separates anchor effects; native references retain their own mistakes. Later source-only merging must compete with this simple support baseline and preserve multiplicity/score-value response rather than silently repeating first-moment loss.

## Undo must restore overwritten data

A cursor restore cannot recover a ring slot that a later token overwrote. Use a fixed rewind horizon H, with an undo arena containing each overwritten encoded row and its original absolute address. For L layers, per-token row size b=528 bytes, serving plus undo rows cost approximately

\[
B_{\rm rows}=Lb(A+R+H).
\]

Add slot addresses, validity, checkpoint metadata, native scratch, score scratch and shared decode storage explicitly. Here H includes the current uncommitted token transaction: processing a token with H−1 completed overwritten tokens still recoverable uses the last slot. Guaranteeing H completed-token rewinds during another pending token instead requires H+1 slots. H is an allocation parameter, not the elapsed conversation length. An unbounded journal or retained full-history shadow invalidates the bounded-storage claim.

Before overwriting a slot, save its previous bytes once under the current token transaction. Publish the valid cursor only after encoding validation. A last-layer failure may leave earlier layers published and the current layer physically modified but unpublished; abort must undo both kinds of mutation. In particular, the failed layer's old reachable ring row must be restored even though its valid cursor never advanced. Sibling replay must reproduce encoded memory and full logits exactly under this new contract.

Marker admission and append need an explicit safety policy. A live marker cannot silently become unrestorable. Refuse an append before mutation if it would destroy the last undo record needed by a live marker. The controller may explicitly commit that marker and continue. Expired detached checkpoints must be refused without mutation. A fixed number of full compact snapshots is an alternative, with its complete storage counted. The existing callback cannot silently assert after restoring session metadata; recovery eligibility must be checked before any rollback mutation.

## Discriminating consumer

Supply several independent, bounded-length values under distinct names, update one later, then append enough neutral user/template text to evict the relevant rows. Ask for an old unchanged value, the revised value, a distractor detail, a small computation over retained facts, and an ordinary explanation. Include early and late query timing. Generate and consume each candidate's own tokens, with no native refresh or answer repair.

Record whether the relevant source rows are actually outside support. Preserve native, full-byte, anchor+recent and recent-only outputs, truncation, logit differences at matching histories, delayed-answer truth, support membership, memory allocation and rollback mechanics. Native parity is not the truth metric. A failed task score does not erase successful bounded execution or a directional retention effect.

The initial aim is a finite development panel that exposes what the state forgets. Only then test more names, context lengths, paraphrases and held-out query families. Do not enlarge the benchmark before understanding the first loss and its possible repair.

## What can be sufficient

For a declared family of continuations, histories are equivalent when every permitted continuation receives the same required output distribution. The relation must also be preserved by allowed appends: h~h' implies hx~h'x whenever x is applicable to both histories. Compact sufficient state represents an append-closed relation and tracks its applicability domain; an arbitrary restricted query family alone does not guarantee this update law. Arbitrary independent facts plus unrestricted later exact recall require the number of distinguishable states to grow with the facts. Fixed finite storage can instead target a bounded fact vocabulary, a finite native context, approximate answers under a declared query distribution, or retrieval backed by counted external storage.

These are assumptions about the consumer, not findings that a ring is already sufficient. The experiment should expose the tradeoff and quantify it rather than rename a window as semantic memory.
