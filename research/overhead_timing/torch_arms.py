"""PyTorch baselines for the analysis-overhead timing (research/overhead_timing).

    python torch_arms.py MODEL_DIR PROMPTS_TXT OUT_JSONL --steps 64 --warmup-steps 8 --smoke-steps 4
                         [--deadline S] [--phase-file P]

HF AutoModelForCausalLM, FP32, sdpa, batch 1, greedy, TF32 off. Per prompt every arm runs the same
prompt from scratch (its own prefill, timed separately) and then N decode steps, each bracketed by
torch.cuda.synchronize(). Arm order rotates per prompt. Arms:
  T0   plain KV-cache decode, no hooks
  T1   T0 + stacked_memory style hooks: per-layer forward-pre-hooks (counting) + a pre-hook on the final
       norm cloning the last-position hidden, downloaded with .cpu().numpy() every step
  T1b  T0 + a per-layer forward-pre-hook cloning each layer's last-position input hidden, one stacked
       .cpu() download per step (28 x 1536 floats, the same bytes as C2)
  T2   mdb DecoderSplice loop (capture_kv=range(28), kv injected back and torch.cat every step), as in
       DecoderSplice.decode / the mdb_chain native arm
  T3   DecoderSplice-style manual layer loop (explicit rotary/position_ids, per-layer calls through the
       DecoderSplice object's layers) over a standard HF DynamicCache: capture_kv / inject / our own
       torch.cat all off. DecoderSplice.run cannot decode without capture (it has no cache of its own).
T2/T3 are skipped when the vendored mdb modules do not import. Smoke (first prompt, few tokens, every arm)
runs first and exits 3 when an arm's token ids differ from T0 or the hooks captured nothing."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

pc = time.perf_counter


def phase(label):
    path = os.environ.get("CM_PHASE_FILE")
    if path:
        tmp = path + ".tmp"
        Path(tmp).write_text(label)
        os.replace(tmp, path)


def read_prompts(path):
    out = []
    for line in Path(path).read_text().splitlines():
        v = list(map(int, line.split()))
        if v:
            assert v[0] == len(v) - 1
            out.append(v[1:])
    return out


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--smoke-steps", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=1e9)
    a = ap.parse_args()
    started = pc()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prompts = read_prompts(a.prompts)
    dev = "cuda"
    phase("torch_load")
    t = pc()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True).to(dev).eval()
    sync()
    load_s = pc() - t
    nL = model.config.num_hidden_layers
    decoder = model.get_decoder()
    head_w = model.get_output_embeddings()
    mdb_ok, mdb_err, helper = True, "", None
    try:
        from mdb.compiled_state_runner import DecoderSplice
        helper = DecoderSplice(model, read_prompts(a.prompts)[0][:4])
    except Exception as e:  # noqa: BLE001
        mdb_ok, mdb_err = False, repr(e)

    # ---- hooks (installed once, switched by flags so T0 runs with no work in them) ----
    hook_state = {"mode": 0, "calls": 0, "raw": None, "layers": [None] * nL, "bytes": 0}

    def count_hook(_m, _args):
        if hook_state["mode"] == 1:
            hook_state["calls"] += 1

    def norm_hook(_m, args):
        if hook_state["mode"] == 1:
            hook_state["raw"] = args[0][:, -1].detach().clone()

    def make_layer_capture(i):
        def hook(_m, args, kwargs):
            if hook_state["mode"] == 2:
                h = args[0] if args else kwargs["hidden_states"]
                hook_state["layers"][i] = h[:, -1].detach().clone()
        return hook

    def install(mode):
        """Hooks exist only while a T1/T1b arm decodes, so T0/T2/T3 run on an unhooked model."""
        handles = []
        if mode == 1:
            handles += [layer.register_forward_pre_hook(count_hook) for layer in decoder.layers]
            handles.append(decoder.norm.register_forward_pre_hook(norm_hook))
        elif mode == 2:
            handles += [layer.register_forward_pre_hook(make_layer_capture(i), with_kwargs=True) for i, layer in enumerate(decoder.layers)]
        return handles

    def argmax_tok(logits):
        return logits[:, -1].argmax(-1)

    @torch.no_grad()
    def plain(ids, steps, mode):
        hook_state["mode"] = 0
        x = torch.tensor([ids], device=dev)
        sync(); t0 = pc()
        out = model(input_ids=x, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = argmax_tok(out.logits)
        sync(); prefill = (pc() - t0) * 1e3
        toks, ms, nbytes = [nxt], [], 0
        hook_state["mode"] = mode
        handles = install(mode)
        ts = time.time()
        for _ in range(steps):
            sync(); t0 = pc()
            out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            nxt = argmax_tok(out.logits)
            if mode == 1:
                nbytes = hook_state["raw"].cpu().numpy().nbytes
            elif mode == 2:
                nbytes = torch.stack(hook_state["layers"]).cpu().numpy().nbytes
            sync(); ms.append((pc() - t0) * 1e3)
            toks.append(nxt)
        te = time.time()
        for h in handles:
            h.remove()
        hook_state["mode"] = 0
        return [int(v) for v in torch.cat(toks)], ms, prefill, ts, te, nbytes

    @torch.no_grad()
    def splice(ids, steps, capture):
        suffix_n = 4
        prefix, suffix = ids[:-suffix_n], ids[-suffix_n:]
        sync(); t0 = pc()
        runner = DecoderSplice(model, prefix)
        sx = torch.tensor([suffix], device=dev)
        if capture:
            x, _, kv = runner.native(sx, capture_kv=range(nL))
            nxt = runner.tokens(runner.readout(x[:, -1]))
        else:  # T3: HF cache over the whole prompt through the model, then the manual layer loop
            out = model(input_ids=torch.tensor([ids], device=dev), use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            nxt = argmax_tok(out.logits)
        sync(); prefill = (pc() - t0) * 1e3
        toks, ms, nbytes = [nxt], [], 0
        ts = time.time()
        for k in range(1, steps + 1):
            sync(); t0 = pc()
            if capture:
                p = suffix_n + k - 1
                xk, _, kvn = runner.run(runner.embed(toks[-1][:, None]), 0, positions=[p], kv=kv, kv_positions=range(p), capture_kv=range(nL))
                kv = {layer: (torch.cat([kv[layer][0], kvn[layer][0]], 2), torch.cat([kv[layer][1], kvn[layer][1]], 2)) for layer in range(nL)}
                nbytes = sum(kvn[layer][0].numel() + kvn[layer][1].numel() for layer in range(nL)) * 4
                nxt = runner.tokens(runner.readout(xk[:, -1]))
            else:
                pos = len(ids) + k - 1
                position_ids = torch.tensor([[pos]], device=dev)
                hidden = decoder.get_input_embeddings()(toks[-1][:, None])
                rope = decoder.rotary_emb(hidden, position_ids)
                for layer in helper.layers:
                    o = layer(hidden, attention_mask=None, position_ids=position_ids, past_key_values=cache, use_cache=True, position_embeddings=rope)
                    hidden = o[0] if isinstance(o, tuple) else o
                nxt = helper.tokens(helper.readout(hidden[:, -1]))
            sync(); ms.append((pc() - t0) * 1e3)
            toks.append(nxt)
        te = time.time()
        return [int(v) for v in torch.cat(toks)], ms, prefill, ts, te, nbytes

    arms = [("T0", lambda i, n: plain(i, n, 0)), ("T1", lambda i, n: plain(i, n, 1)), ("T1b", lambda i, n: plain(i, n, 2))]
    if mdb_ok:
        arms += [("T2", lambda i, n: splice(i, n, True)), ("T3", lambda i, n: splice(i, n, False))]
    names = [n for n, _ in arms]
    out = open(a.out, "w")
    out.write(json.dumps({"type": "header", "load_s": load_s, "arms": names, "mdb_ok": mdb_ok, "mdb_error": mdb_err, "steps": a.steps,
                          "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                          "device": torch.cuda.get_device_name(0)}) + "\n")
    out.flush()

    def emit(kind, p, pos, name, ids_len, res):
        ids, ms, prefill, ts, te, nbytes = res
        out.write(json.dumps({"type": kind, "prompt": p, "arm": name, "order_pos": pos, "prompt_tokens": ids_len, "prefill_ms": prefill,
                              "t_start": ts, "t_end": te, "ids": ids, "ms": ms, "capture_bytes_per_token": nbytes}) + "\n")
        out.flush()

    # ---- smoke ----
    phase("torch_smoke")
    res = {n: fn(prompts[0], a.smoke_steps) for n, fn in arms}
    for pos, n in enumerate(names):
        emit("smoke", 0, pos, n, len(prompts[0]), res[n])
    why = ""
    for n in names[1:]:
        if res[n][0] != res["T0"][0]:
            why = f"{n} token ids differ from T0: {res[n][0]} vs {res['T0'][0]}"
    if hook_state["calls"] == 0:
        why = why or "count hooks never fired"
    if res["T1"][5] == 0 or res["T1b"][5] == 0:
        why = why or "hidden download captured nothing"
    out.write(json.dumps({"type": "smoke_result", "ok": not why, "why": why}) + "\n")
    out.flush()
    print("[torch smoke]", "ok" if not why else why, flush=True)
    if why:
        sys.exit(3)

    # ---- warmup ----
    phase("torch_warmup")
    for n, fn in arms:
        fn(prompts[0], a.warmup_steps)

    # ---- main ----
    mstart, done, truncated = pc(), 0, False
    for p, ids in enumerate(prompts):
        if pc() - mstart > a.deadline:
            truncated = True
            break
        for k in range(len(arms)):
            n, fn = arms[(k + p) % len(arms)]
            phase(n)
            emit("run", p, k, n, len(ids), fn(ids, a.steps))
        done += 1
    phase("torch_done")
    out.write(json.dumps({"type": "footer", "prompts_done": done, "truncated": truncated, "main_s": pc() - mstart, "wall_s": pc() - started,
                          "peak_torch_alloc_bytes": torch.cuda.max_memory_allocated(), "hook_calls": hook_state["calls"]}) + "\n")
    out.close()


if __name__ == "__main__":
    main()
