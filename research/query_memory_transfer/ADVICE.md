# Native-value restoration through recurrent compact memory

2026-11-06. Design advice for the next authorized cut; no model execution or
shared-runtime edits performed by this reviewer.

The proposed 18-path design is preferable to repeating all old arms plus three
new ones. It directly tests whether the native-V improvement survives compact
memory, while preserving all endpoints needed to interpret that interaction.

## Minimal panel

Use three programs crossed with two memory laws:

| Program | Full selected attention | Bounded memory |
|---|---|---|
| NN: native Q/K/V | historical | historical |
| CC: compiled Q/K/V | historical | historical |
| CN: compiled Q/K, native V | historical | new |

Across three existing two-turn cases this gives eighteen paths, fifteen
historical endpoint joins, and three new CN+compact trajectories. Fifteen
non-native candidates seal before three native references. NC and deletion need
not repeat: neither answers the new transfer question, and their previous
evidence remains available. Preserve the same budgets, source128 bank, layer17,
native receiver formation, and numerical profiles. No new fit, source acquisition,
image compilation, or native target-query refresh is needed.

The main reference should remain NN+full. Compare CC versus CN under full memory,
then CC versus CN under compact memory, relative to that same reference. For
example, retain both signed squared-error gains

    G_full = ||Y_CC,full − Y_NN,full||² − ||Y_CN,full − Y_NN,full||²
    G_comp = ||Y_CC,comp − Y_NN,full||² − ||Y_CN,comp − Y_NN,full||².

Positive gain means native V moves that observable closer to the original
reference. This definition does not prescribe a quality threshold. Keep absolute
and relative effects, native-consumer logits, answers, and follow-up trajectories.
NN+compact remains the separate memory-only control; substituting it as the
reference on one side would answer a different question. A transfer ratio is
optional and unstable when the original gain is near zero.

## Effective-weight interaction

On the maximal common consumed-ID prefix of CC/full, CC/compact, CN/full, and
CN/compact, verify normalized receiver bytes, actual Q/K rows, full probability
pairs, and compact key-state/partition pairs. Equal Q/K and the key-only routing
rule should yield identical cluster memberships, key sums, counts, recent slots,
and compact response weights. V differs, but must not affect routing.

For a head, let a compact node's normalized softmax weight be w_j. Each historical
member of a prototype with count m_j receives effective weight w_j/m_j; a recent
row receives its individual node weight. These lifted weights form Ptilde over
the original positions. The ideal interaction is

    I = Y_CC,comp − Y_CC,full − Y_CN,comp + Y_CN,full
      = (Ptilde − P)(V_C − V_N),

where P is the actual saved native full-attention probability matrix for the
common compiled Q/K. Reconstruct membership and Ptilde after sealing; membership
lists and full retired histories are audit data, not extra serving state.

Define each finite remainder as the saved attention output minus the FP64
weighted mixture of its original per-position V rows. Keep the signed sum of all
four remainders. Compact remainders include stored FP64 value-sum accumulation,
the order of divisions and reductions, and the final FP32 cast. They are not
solely final-cast error. Native full-attention remainders include its FP32 value
mixture. This definition keeps the identity accountable without pretending
stored prototype means equal exact real means.

The first 32 appended rows have no multi-member prototype merge: there are up to
sixteen recent rows plus sixteen single-member retired prototypes. Nevertheless,
the CPU64 compact score/mix profile differs from native FP32 attention. Label
that interval a numerical-profile control. After the first real merge at the
33rd appended row, Ptilde−P also includes information discarded by aggregation.
Do not label the entire interaction, including the pre-merge interval, as lost
covariance or compression error.

The exact formula applies only where paired operands and partitions are verified.
If a pair differs, retain the trajectories but mark the formula instrument
inapplicable there. Beyond history divergence, report recurring-policy outcomes;
do not claim a common-cache local intervention. The NN memory comparison is also
useful, but changes Q/K and clustering as well as V relative to CC. It does not
share this same-partition value-only identity.

## Minimal implementation checks

- Reuse the authenticated bank and runtime from `job-27a597ebd5f8`, whose report
  SHA256 is `da86f60e282cfdf93cd72dffabd98c91cabb7bfb9c3b46450421cc08ca207052`.
  Bind the original model identity, all four bank hashes, and profile as before.
- Policy7 uses compiled Q/K, native V, and the unchanged compact reader. It has
  one fresh coordinate call, two compiled maps, and one native V map per token.
  Selected-QKV cost stays 819,200 MAC/update, plus the paid compact reader,
  routing, and transfers. No native selected-cache read should occur.
- Preserve the compact selected-cache sentinel, per-token append count and mass,
  exact snapshot replay, own emitted-token authority, and ordinary dense parity.
  These prove execution; fidelity measurements remain observations.
- Capture raw diagnostic Q/K/V during execution as needed, but do not consume
  full-history reference reads or lifted membership weights in the primary
  compact serving path. Analyze them only after candidate sealing.
- Join all fifteen historical paths, including histories, cache-bound records,
  full logits, ports, coefficient arrays, and compact snapshots. New analysis
  may add observations without modifying predecessor raw results.

With the old maximum lengths 107/108/132, the previous bound formula adjusted to
six arms per case, three full-probability paths per case, and nine compact
paths totals 116,975,016 bytes (111.56 MiB), including its 8 MiB overhead allowance.
Recompute the final bound from the actual recorder; optional saved membership or
lifted matrices require their own additional account. An in-memory post-hoc audit
does not require retaining quadratic lifted matrices in the physical archive.

## Expected information and boundaries

Native V can improve full attention while increasing sensitivity to information
lost inside a centroid. That would be a useful negative transfer observation,
not a broken approximation or a reason to erase the path. Conversely, retained
native-V gain need not require a small interaction norm. Its direction relative
to the existing error and the native suffix is what determines usefulness.

This cut should precede choosing covariance moments or query-sensitive splits:
it tests whether the proposed cheaper program's observed repair transfers through
its actual repeated memory law. It still pays native V, dense lower layers,
other caches, and the native output consumer. The repeated cases remain
development evidence and have previously been insensitive to deletion at the
level of basic answers. Neither gain transfer nor exact replay establishes
semantic necessity, improved arithmetic truth, latency benefit, or whole-model
forward-pass-free closure.
