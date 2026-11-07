# Compiling a nonlinear receiver plane

Prospective plan, 2026-11-05. This is an explicit computation intervention in the C Qwen executor. It is separate from OVC7/OVC8's native query-matrix response forecasts. The first panel uses Qwen2.5-1.5B-Instruct, layer 16, the final prompt token, one binding context, six declared doses and four arms. No training or target-dependent fitting occurs.

## The computation

Let x be the incoming residual, a the attention output-projection write, and r0 the materialized native FP32 sum x+a. Save three raw basis vectors (r0,a,x). Materialize each receiver with the same FP32 schedule in every edited arm:

    r(s,t) = (r0 + s*a) + t*x.

For RMSNorm weights gamma, gate matrix G, up matrix U and down matrix D, compile six vectors once before any target evaluation:

    Aj = G (gamma * rj), Bj = U (gamma * rj), j in {0,a,x}.

The executable program then evaluates:

    rho = rsqrt(sum_i r_i(s,t)^2 / d + epsilon)
    g = rho * ((A0 + s*Aa) + t*Ax)
    u = rho * ((B0 + s*Ba) + t*Bx)
    write = D [SiLU(g) * u]
    next_residual = r(s,t) + write.

The existing native RMS reduction exports rho. SiLU and D execute normally. The selected gate/up matrix-vector calls actually do not execute in the compiled arm. Exact equality holds in real arithmetic on the declared plane; FP32 receiver materialization and projection reassociation introduce arithmetic differences. Three coefficients preserve the independently rounded r0; deriving A0 as Ax+Aa would add another error. No zero-dose correction or target fitting is allowed. Capture the zero-dose numerical floor.

Because r0 is approximately x+a, this plane mainly varies the attention/carry mixture and the overall residual magnitude. RMSNorm nearly removes positive common scale, apart from epsilon and rounding, whereas the residual skip retains it. These coordinates are not two discovered semantic variables.

## Smallest informative panel

Prompt: “Mira's badge is red. Niko's badge is blue. What color is Mira's badge? Answer with the color only.” Use the source tokenizer's native chat template. Freeze the token-boundary prefix before its last token. Compile on the unchanged final token, save/hash the complete program, then evaluate all targets from that same parent.

Doses: (0,0), (+0.25,0), (-0.25,0), (-1,0), (0,+0.25), (+0.5,+0.5). The last is the common-scale control. The -1 attention dose is a controlled ablation, not a natural prompt.

Arms: unmodified native once; dense MLP on each materialized receiver; compiled nonlinear MLP on each identical receiver; each edited receiver with the original MLP write held fixed. The held-write arm separates the direct residual path from the changed nonlinear write. Its absent gate/up computations must not emit fabricated ports.

For every arm save selected receiver/gate/up/write/output ports where actually computed, first full logits, the current token's new higher-layer KV rows, one same-history continuation token's logits/KV, and eight-token own-generated output. Append exact native role delimiters and ask the target and collateral questions on separate children of each arm's own generated state. Preserve IDs, text, answer margins and truncation. A later answer is an observation, not a general free-form-chat claim.

## Mechanics and resource accounting

The program binds to the loaded model, session, reset epoch, token position and layer. Exact finite live-operand guards check x and a. It changes no retained prefix KV. It is an explicit operation argument; no persistent intervention policy is hidden in checkpoint state. Metadata refusals occur before writes; a read-guard refusal leaves the incoming residual intact, and later finite failures restore it. The token remains partial and requires an explicit layer recomputation or reset. Counters describe physical projection calls and are not rollback state.

Compilation requires six extra GEMVs. The frozen payload is six intermediate-width coefficients plus four hidden-width vectors (three operands and the native write); one scalar is execution scratch. All original weights stay resident because other tokens/layers still use them. There is no one-shot latency, whole-model compression or speedup claim. Use one finite mrun CUDA lease on Beast, one resident model, 512-token capacity, retries disabled. Run model-free lifecycle/plane checks first and retain ordinary-path compatibility against the previously installed binary.

Mechanics authority is limited to identity, shape, finite values, frozen source/program integrity, applicable execution and replay closure. Port/state/logit errors, target/collateral answers and work savings are observations. Failed numerical agreement must retain all applicable controls. OVC7's nonlinear-reader errors motivate retaining nonlinearities; they do not establish this program's numerical or semantic performance.

Related: [stacked-state plan](STACKED_STATE_PLAN.md), [findings](FINDINGS.md), [OVC7 findings](../experiments/2026-11-05-operational-variable-closure/FINDINGS-OVC7.md), [OVC8 plan](../experiments/2026-11-05-operational-variable-closure/PLAN-OVC8.md).
