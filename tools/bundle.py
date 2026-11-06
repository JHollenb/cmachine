"""Small, deterministic, mmap-friendly tensor bundle shared by C adapters.

SHA256 covers every byte after the 40-byte magic/digest prefix, including
architecture, numerical configuration, model identity, tensor table and data.
Export is offline; the C serving executable imports no Python or model framework.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import struct
from pathlib import Path

MAGIC = b"TCM1LE\0\0"
DTYPES = {"float32": 1, "bfloat16": 2, "float64": 3}


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def write_bundle(path, architecture, identity, config, tensors):
    """tensors maps names to (dtype, shape, contiguous little-endian bytes)."""
    path = Path(path)
    if len(identity) != 64 or any(x not in "0123456789abcdef" for x in identity):
        raise ValueError("model identity must be a SHA256")
    if len(config) > 16:
        raise ValueError("configuration too long")
    config = list(config) + [0] * (16-len(config))
    names = sorted(tensors)
    offset = 176 + 120 * len(names)
    records, blocks = [], []
    for name in names:
        dtype, shape, raw = tensors[name]
        encoded = name.encode()
        if len(encoded) >= 64 or len(shape) > 4 or b"\0" in encoded:
            raise ValueError("invalid tensor descriptor")
        item = {"float32": 4, "bfloat16": 2, "float64": 8}[dtype]
        count = 1
        for d in shape:
            if d <= 0:
                raise ValueError("empty dimension")
            count *= d
        if len(raw) != item*count:
            raise ValueError("tensor payload size differs")
        padding = (-offset) % 8
        blocks.append(b"\0"*padding)
        offset += padding
        records.append(struct.pack("<64sII4QQQ", encoded, len(shape), DTYPES[dtype],
                                   *(list(shape)+[0]*(4-len(shape))), offset, len(raw)))
        blocks.append(raw)
        offset += len(raw)
    prefix = struct.pack("<II16I64s", architecture, len(names), *config, identity.encode())
    # Streaming writes keep the Qwen exporter from duplicating the complete bank.
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        stream.write(MAGIC + b"\0"*32)
        for block in [prefix, *records, *blocks]:
            stream.write(block)
            digest.update(block)
        stream.seek(8)
        stream.write(digest.digest())
    return {"path": str(path), "bytes": path.stat().st_size,
            "content_sha256": digest.hexdigest(), "identity": identity,
            "architecture": architecture, "tensor_count": len(names)}


def export_isa(archive, destination):
    import torch

    payload = torch.load(archive, map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    tensors = {}
    for name, value in payload["parameters"].items():
        value = value.detach().cpu().contiguous()
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("ISA export requires finite FP32 parameters")
        raw = value.numpy().tobytes()
        record = manifest["parameter_records"][name]
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"parameter custody differs: {name}")
        tensors[name] = ("float32", tuple(value.shape), raw)
    config = manifest["model_config"]
    if (config["architecture"], config["width"], config["layers"], config["inner"],
            config["state"], config["kernel"], config["store"], config["aligned"]) != ("mamba",48,3,96,4,3,True,True):
        raise ValueError("unsupported ISA profile")
    body = dict(manifest)
    seal = body.pop("program_sha256")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != seal:
        raise ValueError("stream manifest seal differs")
    result = write_bundle(destination, 1, manifest["source_binding"]["checkpoint_sha256"],
                          [48,3,float_bits(config["post_norm_epsilon"])], tensors)
    result["source_program_sha256"] = seal
    result["source_archive_sha256"] = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    return result


def export_law(plan_path, destination):
    plan = json.loads(Path(plan_path).read_text())
    predictor = plan["predictor"]
    def check_seal(document, key):
        body = dict(document)
        expected = body.pop(key)
        raw = json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError(f"invalid {key} seal")
    check_seal(plan,"fingerprint")
    check_seal(predictor,"fingerprint")
    baseline = plan["baselines"]["10"]
    values = baseline["post.2.input"]["values"]
    flat = [x for batch in values for row in batch for x in row]
    receiver_raw = struct.pack("<"+"f"*len(flat),*flat)
    if hashlib.sha256(receiver_raw).hexdigest()!=predictor["receiver_M_sha256"]:
        raise ValueError("receiver identity differs from sealed predictor")
    x0 = baseline["post.2.input"]["values"][0][8]
    p0 = baseline["post.2.output"]["values"][0][8]
    def tensor(values, dtype):
        shape = (len(values), len(values[0])) if isinstance(values[0], list) else (len(values),)
        flat = [x for row in values for x in row] if len(shape)==2 else values
        code = "d" if dtype=="float64" else "f"
        return dtype, shape, struct.pack("<"+code*len(flat), *flat)
    tensors = {"anchor.input": tensor(x0,"float32"), "anchor.branch": tensor(p0,"float32"),
               "basis.0": tensor(predictor["h0"],"float64"), "basis.1": tensor(predictor["h1"],"float64"),
               "coefficients": tensor(predictor["coefficients"],"float64")}
    result = write_bundle(destination,3,predictor["model_identity"],[2,8],tensors)
    result.update(source_plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
                  original_receiver_sha256=predictor["receiver_M_sha256"],
                  predictor_fingerprint=predictor["fingerprint"],
                  scope="explicit anchored intervention at post.2 row 8; original receiver evidence identity retained; not natural live applicability")
    return result


def export_qwen(source, destination):
    """Copy original BF16/FP32 safetensor bytes; widening belongs to the backend."""
    source=Path(source)
    config=json.loads((source/"config.json").read_text())
    expected={"model_type":"qwen2","hidden_size":1536,"num_hidden_layers":28,
              "num_attention_heads":12,"num_key_value_heads":2,"intermediate_size":8960,
              "vocab_size":151936,"tie_word_embeddings":True,"hidden_act":"silu",
              "max_position_embeddings":32768}
    for key,value in expected.items():
        if config.get(key)!=value:
            raise ValueError(f"unsupported Qwen configuration: {key}")
    if config.get("rope_scaling") is not None or config.get("use_sliding_window",False):
        raise ValueError("scaled RoPE or sliding attention is outside the adapter")
    safetensors=source/"model.safetensors"
    digest=hashlib.sha256()
    with safetensors.open("rb") as file:
        for block in iter(lambda:file.read(1<<20),b""):digest.update(block)
    identity=digest.hexdigest()
    generation=json.loads((source/"generation_config.json").read_text()) if (source/"generation_config.json").exists() else {}
    eos=generation.get("eos_token_id",config.get("eos_token_id"))
    # The checkpoint has multiple native stop tokens. Keep the complete list in
    # metadata; the initial C ABI uses the conversation end token (151645).
    eos_list=eos if isinstance(eos,list) else [eos]
    primary=151645 if 151645 in eos_list else eos_list[0]
    if len(eos_list)>2:raise ValueError("adapter supports at most two checkpoint EOS tokens")
    extra=next((x for x in eos_list if x!=primary),0)
    tensors={}
    with safetensors.open("rb") as file:
        mapped=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
        header_size=struct.unpack_from("<Q",mapped,0)[0]
        header=json.loads(mapped[8:8+header_size])
        data_start=8+header_size
        for name,record in header.items():
            if name=="__metadata__":continue
            dtype={"BF16":"bfloat16","F32":"float32"}.get(record["dtype"])
            if dtype is None:raise ValueError("Qwen export supports BF16 or FP32 tensors")
            a,b=record["data_offsets"]
            tensors[name]=(dtype,tuple(record["shape"]),memoryview(mapped)[data_start+a:data_start+b])
        cfg=[1536,28,12,2,128,8960,151936,float_bits(config["rms_norm_eps"]),float_bits(config["rope_theta"]),primary,32768,1,extra]
        result=write_bundle(destination,2,identity,cfg,tensors)
        for _,_,view in tensors.values():view.release()
        tensors.clear();mapped.close()
    result.update(source_weights_sha256=identity,native_eos_token_ids=eos_list,
                  profile_version="qwen25-fp32-execution-v2",
                  tokenizer_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [source/"tokenizer.json",source/"tokenizer_config.json",source/"config.json"]},
                  numerical_contract="original tensor bytes; C widens BF16 to FP32 once on GPU; FP32 execution")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("isa", "law", "qwen"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = {"isa":export_isa,"law":export_law,"qwen":export_qwen}[args.kind](args.source,args.destination)
    args.destination.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
