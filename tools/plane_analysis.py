"""Source-bound analysis of the finite receiver-plane panel, without a model."""
from __future__ import annotations

import hashlib
import numpy as np


def analyze(directory, driver):
    def array(name, count):
        value=np.fromfile(directory/name,dtype="<f4")
        if value.size!=count or not np.isfinite(value).all():raise ValueError(f"invalid plane raw array: {name}")
        return value.astype(np.float64)
    def measure(candidate,reference):
        difference=candidate-reference
        return {"max_abs":float(np.max(np.abs(difference))),"l2":float(np.linalg.norm(difference)),
                "relative_l2":float(np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-30)),
                "bytes_equal":bool(np.array_equal(candidate.view(np.uint64),reference.view(np.uint64)))}
    def data(name,port):return array(f"{name}-{port}.f32",8960 if port in ("mlp_gate","mlp_up") else 151936 if port=="first" else 5632 if port.endswith("kv") else 1536)
    rows={x["name"]:x for x in driver["rows"]}
    if len(rows)!=19:raise ValueError("receiver-plane panel is incomplete")
    program=directory/"plane-program.f32"
    if program.stat().st_size!=driver["program_bytes"] or hashlib.sha256(program.read_bytes()).hexdigest()!=driver["program_sha256"]:
        raise ValueError("frozen program identity differs")
    if driver["program_bytes"]!=239616:raise ValueError("program extent differs from declared profile")
    if not all(r["prefix_intact"] and r["program_unchanged"] for r in rows.values()):raise ValueError("parent/program integrity differs")
    comparisons=[]
    for dose in range(6):
        dense,compiled,held=(rows[f"{name}-{dose}"] for name in ("dense","compiled","held"))
        expected=({"gate_up_matvecs":2,"down_matvecs":1,"compile_matvecs":0},
                  {"gate_up_matvecs":0,"down_matvecs":1,"compile_matvecs":0},
                  {"gate_up_matvecs":0,"down_matvecs":0,"compile_matvecs":0})
        for row,work in zip((dense,compiled,held),expected):
            if row["work"]!=work:raise ValueError("declared projection bypass did not execute")
        # Both arms must receive the same actual normalized materialization.
        receiver=data(dense["name"],"mlp_input")
        if not np.array_equal(receiver.view(np.uint64),data(compiled["name"],"mlp_input").view(np.uint64)):
            raise ValueError("compiled and dense receiver instrument differs")
        if not np.array_equal(data(held["name"],"mlp_return").view(np.uint64),data("native","mlp_return").view(np.uint64)):
            raise ValueError("held-write control did not hold the native write")
        result={"dose":dense["dose"],"compiled_vs_dense":{},"dense_vs_native":{},"held_vs_dense":{},
                "own_ids_equal":compiled["own"]["ids"]==dense["own"]["ids"],
                "same_history_next_top_equal":compiled["same_history_next"]["top"]==dense["same_history_next"]["top"],
                "same_history_next_full_logits_hash_equal":compiled["same_history_next"]["sha256"]==dense["same_history_next"]["sha256"],
                "same_history_next_red_blue_margin_error":compiled["same_history_next"]["red_minus_blue"]-dense["same_history_next"]["red_minus_blue"],
                "questions":[{"name":a["name"],"dense_text":a["output"]["text"],"compiled_text":b["output"]["text"],"held_text":c["output"]["text"],
                              "compiled_dense_own_history_equal":compiled["own"]["ids"]==dense["own"]["ids"],
                              "compiled_minus_dense_red_blue_margin":b["first"]["red_minus_blue"]-a["first"]["red_minus_blue"]}
                             for a,b,c in zip(dense["questions"],compiled["questions"],held["questions"])]}
        for port in ("mlp_gate","mlp_up","mlp_return","output","first","current-kv","next-kv"):
            d,c,n=data(dense["name"],port),data(compiled["name"],port),data("native",port)
            result["compiled_vs_dense"][port]=measure(c,d)
            result["dense_vs_native"][port]=measure(d,n)
            if port not in ("mlp_gate","mlp_up"):result["held_vs_dense"][port]=measure(data(held["name"],port),d)
            response=d-data("dense-0",port)
            prediction=c-data("compiled-0",port)
            result["compiled_vs_dense"][port]["response_norm"]=float(np.linalg.norm(response))
            result["compiled_vs_dense"][port]["response_error_l2"]=float(np.linalg.norm(prediction-response))
            result["compiled_vs_dense"][port]["response_relative_error"]=float(np.linalg.norm(prediction-response)/np.linalg.norm(response)) if dose and np.linalg.norm(response)>0 else None
        comparisons.append(result)
    native=rows["native"]
    return {"comparisons":comparisons,"summary":{
        "rows":19,"doses":6,"compiled_calls":6,"compile_gate_up_matvecs":6,"target_gate_up_matvecs_skipped":12,
        "max_compiled_write_relative_error":max(x["compiled_vs_dense"]["mlp_return"]["relative_l2"] for x in comparisons),
        "max_compiled_first_logits_abs_error":max(x["compiled_vs_dense"]["first"]["max_abs"] for x in comparisons),
        "matching_compiled_dense_own_paths":sum(x["own_ids_equal"] for x in comparisons),
        "matching_compiled_dense_common_next_top":sum(x["same_history_next_top_equal"] for x in comparisons),
        "native_text":native["own"]["text"],"program_bytes":driver["program_bytes"],
        "dense_zero_vs_native_first_logits_exact":measure(data("dense-0","first"),data("native","first"))["bytes_equal"],
        "held_zero_vs_native_first_logits_exact":measure(data("held-0","first"),data("native","first"))["bytes_equal"]}}
