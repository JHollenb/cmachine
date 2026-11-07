# Astra review of persistent bounded chat

The existing user-requested Astra advisor reviewed the server, client and installation before execution and independently audited the saved run afterward. It found no blocking ownership or publication bug. Its requested split-generation and delivered-EOS checks were added before freezing the executed payload.

The independent saved-only audit authenticates 75 source hashes, 23 raw files/171,960 bytes, scheduler log/report/raw joins, the seven-command whole-invocation receipt and 79-file installation manifest. Both policies reproduce eight historical answers and 16 prefill/final logit hashes each. Raw traces confirm `G 1; G 1` equals `G 2`, repeated generation after delivered EOS returns empty without mutation, all 11 refusals per policy preserve state, and actual client rollback/replay reproduces cache, logits and RNG. All five text-client cases have zero prefix recomputations.

Astra confirms the ownership boundary: the actual client root 363 retains delivered EOS 151645, while the assay consumes it and reaches 364. The next append includes that pending token exactly once. These states should not be compared as identical token boundaries.

The observed short conversation gives sensible prose, 46 for explicit 17+29 and correct recent recall. Prose under the tiled profile reaches its 32-token budget without EOS. Delayed recall remains wrong in all four bounded children. The longer digit string uses 32 tokens versus the historical 8; it supplies no evidence of improved recall. The full-byte 40 sum remains wrong too.

Resource accounting independently agrees: 9,279,620 session bytes including snapshots; 3,133,714,584 tracked model-plus-state bytes. The lease's 6,394 MB measured VRAM peak includes the original native reference. It is not a bounded-only peak or a speed result. Installation binds the separate opt-in target and unchanged stable native hash; native remains the default.

The resulting claim is persistent bounded execution on the declared specimen. Conversation persistence ends with the process, host transcript storage remains additional, logical capacity is 768 in this trial, and useful general compact sufficient chat remains open. The integration removes an interface gap while preserving the known scientific failure.
