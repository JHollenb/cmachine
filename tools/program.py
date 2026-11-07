"""Compile bounded, source-mapped vector graphs to the C program executor.

FX is an opt-in trusted-source capture front end, not arbitrary repository code
loading. The canonical mrun front end never imports model repository code.
No family names reach the C dispatcher. Unsupported graph operations fail closed.
"""
from __future__ import annotations
import hashlib
import builtins
import json
import math
import operator
import os
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle import write_bundle

OPCODES = {"linear": 1, "add": 2, "multiply": 3, "remainder": 4,
           "cos": 5, "affine": 6, "concat": 7, "slice": 8, "rms": 9,
           "embedding": 10, "causal-conv-silu": 11, "selective-scan": 12,
           "silu": 13, "constant": 14, "relu": 15, "divide": 16, "match-rows": 17}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Program:
    """Target-specific schedule, retaining the upstream graph's exact addresses."""
    def __init__(self, identity, input_width, *, token_mode=False, vocab=None, origin=None):
        if len(identity) != 64:
            raise ValueError("source identity must be a SHA256")
        self.identity, self.input_width = identity, input_width
        self.tokens, self.vocab, self.origin = token_mode, vocab, origin
        self.values = [{"name": "input", "offset": 0, "length": input_width}]
        self.ops, self.weights, self.slots, self.source_dtypes = [], {}, [], {}
        self.output = None

    def length(self, value):
        return self.values[value]["length"]

    def parameter(self, name, tensor):
        import torch
        if name not in self.weights:
            self.source_dtypes[name] = str(tensor.dtype)
            tensor = tensor.detach().cpu().float().contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite parameter: {name}")
            self.weights[name] = tensor
        return name

    def emit(self, kind, n, a=0, b=0, c=0, d=0, w=None, bias=None,
             offset=0, arg=0, scale=0, shift=0, source=None, name=None):
        if kind not in OPCODES or type(n) is not int or n <= 0:
            raise ValueError(f"unsupported instruction {kind}/{n}")
        index = len(self.values)
        self.values.append({"name": name or f"value.{index}", "length": n,
                            "offset": sum(v["length"] for v in self.values)})
        self.ops.append({"kind": kind, "out": index, "a": a, "b": b, "c": c, "d": d,
                         "w": w, "bias": bias, "offset": offset, "arg": arg,
                         "scale": scale, "shift": shift, "source": source})
        return index

    def state_slot(self, name, shape):
        offset = sum(math.prod(s["shape"]) for s in self.slots)
        self.slots.append({"name": name, "shape": list(shape), "offset": offset,
                           "initialization": "zero", "commit": "one-input-row"})
        return offset

    def save(self, destination, *, source_guard=None):
        if self.output is None:
            raise ValueError("graph output unbound")
        destination = Path(destination)
        if destination.exists() or destination.with_suffix(".map.json").exists():
            raise FileExistsError("refusing to overwrite compiled program")
        names, banks = {}, {}
        for name in sorted(self.weights):
            tensor = self.weights[name]
            key = (tensor.data_ptr(), tuple(tensor.shape))
            if key not in banks:banks[key] = f"w.{len(banks):04d}"
            names[name] = banks[key]
        table_names = sorted(["__ops", "__values", *set(names.values())])
        indices = {n: i for i, n in enumerate(table_names)}
        tensors = {}
        for name, tensor in self.weights.items():
            tensors[names[name]] = ("float32", tuple(tensor.shape) or (1,), memoryview(tensor.numpy()).cast("B"))
        rows = []
        for op in self.ops:
            rows.append([OPCODES[op["kind"]], op["out"], op["a"], op["b"], op["c"], op["d"],
                         indices[names[op["w"]]] if op["w"] is not None else -1,
                         indices[names[op["bias"]]] if op["bias"] is not None else -1,
                         op["offset"], op["arg"], op["scale"], op["shift"]])
        def table(values):
            flat = [x for row in values for x in row]
            return "float64", (len(values), len(values[0])), struct.pack("<"+"d"*len(flat), *flat)
        tensors["__ops"] = table(rows)
        tensors["__values"] = table([[v["offset"], v["length"]] for v in self.values])
        arena = sum(v["length"] for v in self.values)
        state = sum(math.prod(s["shape"]) for s in self.slots)
        cfg = [1, len(self.values), len(self.ops), arena, state, 0, self.output,
               self.input_width, int(self.tokens), self.vocab or self.length(self.output)]
        destination.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(prefix=".program-",dir=destination.parent)
        os.close(fd)
        temporary=Path(temporary)
        try:
            result = write_bundle(temporary, 4, self.identity, cfg, tensors)
            if source_guard:source_guard()
            os.link(temporary,destination)
            result["path"]=str(destination)
        finally:
            temporary.unlink(missing_ok=True)
        mapping = {"schema": "tcm-program-source-map-v1", "bundle": result,
                   "source_identity": self.identity, "origin": self.origin,
                   "values": self.values, "instructions": self.ops,
                   "parameters": [{"source_name": name, "bundle_name": names[name],
                                   "shape": list(t.shape), "source_dtype": self.source_dtypes.get(name,str(t.dtype)),
                                   "execution_dtype": "float32"} for name, t in sorted(self.weights.items())],
                   "state_slots": self.slots, "input_width": self.input_width,
                   "output_value": self.output, "token_mode": self.tokens,
                   "readout_domain": self.vocab or self.length(self.output),
                   "arithmetic": "FP32; source coefficients converted offline with original dtype retained; C libm; contraction disabled; selected scalar or BLAS reductions",
                   "authority": {"mechanics_status": "compiled-unexecuted", "instrument_status": "unmeasured",
                                 "trend_status": "unmeasured", "terminal_status": "no-parity-or-semantic-claim",
                                 "semantic_roles": "upstream-declarations-and-hypotheses"}}
        mapping["fingerprint"] = hashlib.sha256(canonical(mapping)).hexdigest()
        destination.with_suffix(".map.json").write_bytes(canonical(mapping)+b"\n")
        return mapping


def lower_fx(module, input_width, identity, destination, *, source_record=None):
    """Capture trusted Python computation; lower operations, not a model-name switch.

    Bounded ABI: one row, flattened vectors, no mutation, data-dependent Python
    control flow or random operations. Shape-only unsqueeze/cast are explicit in
    the retained FX graph, and erased only for the FP32 vector target.
    """
    import torch
    import torch.fx
    graph = torch.fx.symbolic_trace(module.eval())
    from torch.fx.passes.shape_prop import ShapeProp
    with torch.no_grad():
        ShapeProp(graph).propagate(torch.zeros((1,input_width),dtype=torch.float32))
    reference_identity=hashlib.sha256(canonical({"checkpoint":identity,"source":source_record,
                                                "graph":str(graph.graph),"abi":"single-row-fp32-vector-v1"})).hexdigest()
    p = Program(identity, input_width, origin={"front_end": "trusted-torch-fx", "source": source_record,
                                               "program_identity":reference_identity,"graph": str(graph.graph), "scope": "single-row-vector-abi"})
    env = {}
    def value(node):
        item = env[node]
        if isinstance(item, str):
            t = p.weights[item]
            return p.emit("constant", t.numel(), w=item, source=f"fx://{reference_identity}/{node.name}", name=node.name)
        return item
    def attr(target):
        result = graph
        for part in target.split("."):
            result = getattr(result, part)
        return result
    def linear(node, x, w, bias=None):
        return p.emit("linear", p.weights[w].shape[0], a=value(x), w=w, bias=bias,
                      source=f"fx://{reference_identity}/{node.name}", name=node.name)
    for node in graph.graph.nodes:
        source = f"fx://{reference_identity}/{node.name}"
        target, args = node.target, node.args
        if node.op == "placeholder":
            if env:
                raise ValueError("only one input port supported")
            env[node] = 0
        elif node.op == "get_attr":
            env[node] = p.parameter(str(target), attr(target))
        elif node.op == "call_method" and target in ("float", "unsqueeze", "squeeze", "reshape", "view", "flatten"):
            if target == "unsqueeze" and args[1] not in (-1, 0):
                raise ValueError("only row/vector unsqueeze supported")
            item=env[args[0]]
            if isinstance(item,str):
                tensor=getattr(p.weights[item],target)(*args[1:],**node.kwargs)
                env[node]=p.parameter(item+"."+node.name,tensor)
            else:env[node]=item
        elif node.op == "call_method" and target in ("t", "transpose"):
            key=env[args[0]]
            if not isinstance(key,str) or p.weights[key].ndim!=2 or (target=="transpose" and tuple(args[1:]) not in ((0,1),(1,0))):
                raise ValueError("only static matrix-parameter transpose supported")
            env[node]=p.parameter(key+".transpose",p.weights[key].T)
        elif node.op == "call_module" and isinstance(attr(target), torch.nn.Linear):
            layer = attr(target)
            w = p.parameter(str(target)+".weight", layer.weight)
            bias = p.parameter(str(target)+".bias", layer.bias) if layer.bias is not None else None
            env[node] = linear(node, args[0], w, bias)
        elif node.op=="call_module" and isinstance(attr(target),torch.nn.ReLU):
            if attr(target).inplace:raise ValueError("in-place ReLU unsupported")
            a=value(args[0]);env[node]=p.emit("relu",p.length(a),a=a,source=source,name=node.name)
        elif node.op == "call_function":
            kw = {"source": source, "name": node.name}
            if target is operator.getitem:
                index = args[1]
                if not (isinstance(index,tuple) and len(index)==2 and index[0]==slice(None)):
                    raise ValueError("only static row-vector column selection supported")
                a=value(args[0]);width=p.length(a);column=index[1]
                if type(column) is int:
                    start=column+width if column<0 else column;stop=start+1
                    if not 0<=start<width:raise ValueError("column outside vector")
                elif isinstance(column,slice):
                    start,stop,stride=column.indices(width)
                    if stride!=1 or stop<=start:raise ValueError("empty or strided vector slice unsupported")
                else:raise ValueError("dynamic vector slice unsupported")
                env[node]=p.emit("slice",stop-start,a=a,offset=start,**kw)
            elif target is builtins.getattr and args[1]=="T":
                key=env[args[0]]
                if not isinstance(key,str) or p.weights[key].ndim!=2:raise ValueError("only matrix-parameter transpose supported")
                env[node]=p.parameter(key+".transpose",p.weights[key].T)
            elif target in (operator.matmul,torch.matmul):
                key=env[args[1]]
                if not isinstance(key,str) or p.weights[key].ndim!=2:raise ValueError("matmul requires a static right parameter")
                weight=p.parameter(key+".linear",p.weights[key].T)
                env[node]=linear(node,args[0],weight)
            elif target in (torch._C._nn.linear, torch.nn.functional.linear):
                if node.kwargs:
                    raise ValueError("keyword linear bindings require explicit lowering")
                env[node] = linear(node, args[0], env[args[1]], env[args[2]] if len(args)>2 and args[2] is not None else None)
            elif target in (operator.mod, torch.remainder):
                a = value(args[0]); env[node] = p.emit("remainder", p.length(a), a=a, scale=float(args[1]), **kw)
            elif target in (operator.mul, operator.add, operator.sub, operator.truediv):
                left, right = args
                if isinstance(left, (float, int)) or isinstance(right, (float, int)):
                    if isinstance(left, (float, int)):
                        if target not in (operator.mul, operator.add):
                            raise ValueError("unsupported left-scalar arithmetic")
                        left, right = right, left
                    a = value(left)
                    if target is operator.truediv:
                        key=p.parameter(node.name+".divisor",torch.tensor([right],dtype=torch.float32))
                        b=p.emit("constant",1,w=key,source=source)
                        env[node]=p.emit("divide",p.length(a),a=a,b=b,**kw)
                    else:
                        scale, shift = (float(right), 0) if target is operator.mul else (1, float(right)) if target is operator.add else (1, -float(right))
                        env[node] = p.emit("affine", p.length(a), a=a, scale=scale, shift=shift, **kw)
                else:
                    a, b = value(left), value(right)
                    if target is operator.sub:
                        b = p.emit("affine", p.length(b), a=b, scale=-1, source=source)
                    env[node] = p.emit("multiply" if target is operator.mul else "divide" if target is operator.truediv else "add", max(p.length(a),p.length(b)), a=a,b=b, **kw)
            elif target in (torch.relu,torch.nn.functional.relu):
                if node.kwargs.get("inplace",False):raise ValueError("in-place ReLU unsupported")
                a=value(args[0]);env[node]=p.emit("relu",p.length(a),a=a,**kw)
            elif target is torch.cos:
                a=value(args[0]);env[node]=p.emit("cos",p.length(a),a=a,**kw)
            elif target is torch.cat:
                if node.kwargs.get("dim", args[1] if len(args)>1 else 0) != -1:
                    raise ValueError("only final-vector concat supported")
                items=[value(n) for n in args[0]];a=items[0]
                for b in items[1:]:
                    a=p.emit("concat",p.length(a)+p.length(b),a=a,b=b,**kw)
                env[node]=a
            else:
                raise ValueError(f"unimplemented FX operation: {node.format_node()}")
        elif node.op == "output":
            shape=tuple(args[0].meta['tensor_meta'].shape)
            if not shape or len(shape)>2 or (len(shape)==2 and shape[0]!=1):
                raise ValueError(f"output outside single-row vector ABI: {shape}")
            p.output = value(args[0])
        else:
            raise ValueError(f"unimplemented FX node: {node.format_node()}")
        if node.op!="output" and isinstance(env.get(node),int) and node.meta.get("tensor_meta") is not None:
            shape=tuple(node.meta["tensor_meta"].shape)
            if not shape or len(shape)>2 or (len(shape)==2 and shape[0]!=1):
                raise ValueError(f"tensor outside single-row vector ABI: {node.name}/{shape}")
            if math.prod(shape)!=p.length(env[node]):
                raise ValueError(f"flattened shape differs: {node.name}")
    return p.save(destination)


def lower_mrun(source, destination, *, ir_destination=None):
    """Lower canonical Mamba1 ModelIR operations and exact parameter views.

    This target owns streaming semantics for the two declared Mamba state slots.
    These are target obligations, not inherited numerical qualification from U2.
    """
    import torch
    from safetensors import safe_open
    from mrun.decompiler import decompile_source
    from manalysis.decompiler_bridge import build_exact_graph
    source=Path(source).resolve()
    for target in (destination,ir_destination):
        if target is None:continue
        target=Path(target).resolve()
        if target==source or source in target.parents:
            raise ValueError("compiled output must be outside the frozen source")
        if target.exists():raise FileExistsError(f"refusing to overwrite {target}")
    result = decompile_source(source)
    if ir_destination:
        with Path(ir_destination).open("xb") as f:f.write(canonical(result.as_dict())+b"\n")
    if result.ir_bundle is None:
        raise ValueError(f"canonical source refusal: {result.report.as_dict()}")
    bundle = result.ir_bundle
    if bundle.model.architecture_id != "mamba1-selective-state-space-causal-decoder":
        raise ValueError("this backend has no lowering for the declared architecture")
    exact = build_exact_graph(bundle)
    refs = {n["operation_id"]:n["ref"] for n in exact["nodes"] if n["kind"]=="operation"}
    p=Program(bundle.fingerprint,1,token_mode=True,vocab=bundle.model.dimensions.vocab_size,
              origin={"front_end":"canonical-mrun-ir", "bundle_fingerprint":bundle.fingerprint,
                      "source_fingerprint":bundle.model.source_fingerprint,
                      "source_files":[record.as_dict() for record in result.source.files],
                      "runtime_controls":{"sequence_length":"consumed_rows in run receipt",
                                          "pending_token":"unconsumed last selected token in run receipt"},
                      "state_contract":bundle.state.as_dict(),"io_contract":bundle.io.as_dict()})
    allocations={a.allocation_id:a for a in bundle.physical_weights.allocations}
    loaded = {}
    for view in bundle.physical_weights.views:
        if any(t.kind!="identity" for t in view.transforms):
            raise ValueError("non-identity parameter view unsupported")
        allocation=allocations[view.allocation_id]
        if allocation.allocation_id not in loaded:
            with safe_open(str(Path(source)/allocation.source_file),framework="pt",device="cpu") as f:
                t=f.get_tensor(allocation.source_tensor)
            if not torch.isfinite(t).all():raise ValueError(f"non-finite source allocation: {allocation.source_tensor}")
            source_dtype=str(t.dtype)
            widened=t.float().contiguous()
            if not torch.isfinite(widened).all():raise ValueError(f"source cannot be represented as finite FP32: {allocation.source_tensor}")
            loaded[allocation.allocation_id]=(widened,source_dtype)
        t,source_dtype=loaded[allocation.allocation_id]
        p.weights[view.logical_name]=t
        p.source_dtypes[view.logical_name]=source_dtype
    values={"token_ids":0}
    for op in bundle.model.operations:
        a=[values[x] for x in op.inputs];attrs=op.attributes;params=list(op.parameters)
        kw={"source":refs[op.operation_id],"name":op.outputs[0]}
        def param(suffix, optional=False):
            found=[n for n in params if n.endswith(suffix)]
            if len(found)==1:return found[0]
            if not found and optional:return None
            raise ValueError(f"ambiguous parameter binding: {op.operation_id}/{suffix}")
        if op.kind=="token-embedding":
            w=params[0];out=p.emit("embedding",p.weights[w].shape[1],a=a[0],w=w,**kw)
        elif op.kind in ("linear","linear-readout"):
            w=param(".weight");out=p.emit("linear",p.weights[w].shape[0],a=a[0],w=w,bias=param(".bias",True),**kw)
        elif op.kind=="mamba-rms-norm":
            out=p.emit("rms",p.length(a[0]),a=a[0],w=params[0],scale=attrs["epsilon"],**kw)
        elif op.kind in ("mamba-input-gate-split","mamba-selection-split"):
            widths=[attrs["split_width"]]*2 if op.kind=="mamba-input-gate-split" else [attrs["time_step_rank"],attrs["state_size"],attrs["state_size"]]
            if sum(widths)!=p.length(a[0]):raise ValueError("split width differs")
            offset=0
            for name,width in zip(op.outputs,widths):
                values[name]=p.emit("slice",width,a=a[0],offset=offset,source=refs[op.operation_id],name=name);offset+=width
            continue
        elif op.kind=="mamba-causal-depthwise-convolution":
            n=p.length(a[0]);k=attrs["kernel_size"];slot=p.state_slot(attrs["state_slot"],(n,k))
            out=p.emit("causal-conv-silu",n,a=a[0],w=param(".weight"),bias=param(".bias",True),offset=slot,arg=k,**kw)
        elif op.kind=="mamba-selective-scan":
            n=p.length(a[0]);s=attrs["state_size"];slot=p.state_slot(attrs["state_slot"],(n,s))
            out=p.emit("selective-scan",n,a=a[0],b=a[1],c=a[2],d=a[3],w=param(".A_log"),bias=param(".D"),offset=slot,arg=s,**kw)
        elif op.kind=="silu":out=p.emit("silu",p.length(a[0]),a=a[0],**kw)
        elif op.kind in ("elementwise-multiply","residual-add"):
            out=p.emit("multiply" if op.kind=="elementwise-multiply" else "add",p.length(a[0]),a=a[0],b=a[1],**kw)
        else:raise ValueError(f"unimplemented canonical operation: {op.operation_id}/{op.kind}")
        if len(op.outputs)!=1:raise ValueError("unhandled multi-output operation")
        values[op.outputs[0]]=out
    p.output=values["logits"]
    result.source.assert_unchanged()
    return p.save(destination,source_guard=result.source.assert_unchanged)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Lower canonical source IR to cmachine C program")
    parser.add_argument("source",type=Path);parser.add_argument("destination",type=Path)
    parser.add_argument("--retain-ir",type=Path)
    args=parser.parse_args()
    mapping=lower_mrun(args.source,args.destination,ir_destination=args.retain_ir)
    print(json.dumps({"bundle":mapping["bundle"],"source_map":str(args.destination.with_suffix('.map.json')),
                      "instructions":len(mapping["instructions"]),"state_slots":len(mapping["state_slots"])}))
