"""Raw arithmetic/continuation analysis for a frozen subspace reused in context."""
from __future__ import annotations
import hashlib
import numpy as np


def analyze(directory,driver):
    formation=driver["schema"]=="tcm-subspace-formation-driver-v1"
    components=driver["schema"]=="tcm-subspace-components-driver-v1"
    enrich=components or formation or driver["schema"]=="tcm-subspace-enrichment-driver-v1"
    def seqdot(a,b):
        result=0.0
        for x,y in zip(a,b):result+=float(x)*float(y)
        return result
    def load(name,n):
        a=np.fromfile(directory/name,dtype="<f4")
        if len(a)!=n or not np.isfinite(a).all():raise ValueError(f"invalid reuse array: {name}")
        return a
    def data(name,port):return load(f"{name}-{port}.f32",8960 if port in ("mlp_gate","mlp_up","correction-gate","correction-up") else 151936 if port=="first" else 5632 if port.endswith("kv") else 1536)
    def equal(a,b):return a.tobytes()==b.tobytes()
    def measure(a,b):
        a,b=a.astype(float),b.astype(float);delta=a-b
        return {"relative_l2":float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30)),"l2":float(np.linalg.norm(delta)),"max_abs":float(np.max(np.abs(delta))),"bytes_equal":equal(a,b)}
    raw=(directory/"reuse-program.f32").read_bytes();qr_raw=(directory/"reuse-qr.f64").read_bytes()
    if len(raw)!=161792 or len(qr_raw)!=24608 or hashlib.sha256(raw+qr_raw).hexdigest()!=driver["program_sha256"]:raise ValueError("reusable program identity/extent differs")
    program=np.frombuffer(raw,dtype="<f4");qr=np.frombuffer(qr_raw,dtype="<f8")
    if not np.isfinite(program).all() or not np.isfinite(qr).all():raise ValueError("nonfinite frozen program")
    x,a=program[:1536],program[1536:3072];baseline=program[-1536:]
    q0,q1=qr[:1536],qr[1536:3072];r00,r01,r11,condition=qr[3072:]
    extra_bytes=extra_qr_bytes=0
    if enrich:
        extra=(directory/"enrich-program.f32").read_bytes();extra_qr=(directory/"enrich-qr.f64").read_bytes()
        extra_bytes,extra_qr_bytes=len(extra),len(extra_qr)
        if extra_bytes!=77824 or extra_qr_bytes!=12312 or hashlib.sha256(extra+extra_qr).hexdigest()!=driver["extension_program_sha256"]:raise ValueError("extension identity/extent differs")
        extra=np.frombuffer(extra,dtype="<f4");extra_qr=np.frombuffer(extra_qr,dtype="<f8")
        if not np.isfinite(extra).all() or not np.isfinite(extra_qr).all() or driver["extension_compile_projections"]!=2:raise ValueError("extension compile instrument differs")
        b=extra[:1536];q2=extra_qr[:1536];r02,r12,r22=extra_qr[1536:]
        # Reproduce the declared calibration transaction from its captured input.
        candidate=data("calibration-dense","omitted").astype(float)
        for _ in range(2):
            for q in (q0,q1):candidate-=seqdot(q,candidate)*q
        if not equal(candidate.astype(np.float32),b):raise ValueError("extension did not use the declared calibration operand")
        reconstructed=q0*r02+q1*r12+q2*r22
        if not np.allclose(reconstructed,b.astype(float),rtol=1e-12,atol=1e-12):raise ValueError("extension QR describes another operand")
    if condition!=driver["basis_condition"] or not driver["program_unchanged"]:raise ValueError("program metadata differs")
    rows=driver["rows"]+driver.get("calibration_rows",[])+driver["target_rows"]
    if len(rows)!=(10 if components or formation else 13 if enrich else 14):raise ValueError("reuse panel incomplete")
    by_name={r["name"]:r for r in rows}
    for row in rows:
        name=row["name"];r,rhat,e=(data(name,p) for p in ("receiver","reconstruction","omitted"));c=np.asarray(row["executed_coefficients"],dtype=np.float32)
        expected_rhat=np.add(np.multiply(c[0],x),np.multiply(c[1],a))
        if len(c)==3:expected_rhat=np.add(expected_rhat,np.multiply(c[2],b))
        expected_e=np.add(np.add(r,np.multiply(np.float32(-1),rhat)),np.multiply(np.float32(0),rhat))
        if not equal(rhat,expected_rhat) or not equal(e,expected_e):raise ValueError("executed residual materialization differs")
        # Sequential float64 sums mirror the declared QR backsolve independently.
        v0=seqdot(q0,r);v1=seqdot(q1,r)
        if len(c)==3:
            v2=seqdot(q2,r);c2=v2/r22;c1=(v1-r12*c2)/r11;calculated=np.asarray([(v0-r01*c1-r02*c2)/r00,c1,c2])
        else:calculated=np.asarray([(v0-r01*(v1/r11))/r00,v1/r11])
        if not np.allclose(calculated,row["coefficients"],rtol=1e-12,atol=1e-14):raise ValueError("QR coefficient arithmetic differs")
        if not equal(c,np.asarray(row["coefficients"],dtype=np.float32)):raise ValueError("executed coefficient cast differs")
        norm=float(np.linalg.norm(e.astype(float)))
        if not np.isclose(norm,row["omitted_norm"],rtol=1e-12,atol=1e-14):raise ValueError("omitted operand norm differs")
        if not equal(data(name,"output"),np.add(r,data(name,"mlp_return"))):raise ValueError("actual residual skip was not retained")
        if not row["prefix_intact"]:raise ValueError("retained prefix changed")
        if hashlib.sha256((directory/f"{name}-first.f32").read_bytes()).hexdigest()!=row["first"]["sha256"]:raise ValueError("first-logit seal differs")
        if int(np.argmax(data(name,"first")))!=row["first"]["top"]:raise ValueError("full raw argmax differs")
        expected={"gate_up_matvecs":0 if row["mode"] in (2,5,6) else 2,"down_matvecs":0 if row["mode"]==5 else 1,"compile_matvecs":4 if row["mode"]==0 else 0}
        if row["work"]!=expected:raise ValueError("physical-work instrument differs")
        if enrich and not equal(r,np.add(data(name,"entry"),data(name,"attention-write"))):raise ValueError("receiver operand lineage differs")
        if row["mode"] in (2,6):
            for k,channel in enumerate(("gate","up")):
                images=program[3072+k*17920:3072+(k+1)*17920].reshape(2,8960)
                projection=np.add(np.multiply(c[0],images[0]),np.multiply(c[1],images[1]))
                if len(c)==3:projection=np.add(projection,np.multiply(c[2],extra[1536+k*8960:1536+(k+1)*8960]))
                if not equal(data(name,"mlp_"+channel),np.multiply(np.float32(row["rho"]),projection)):raise ValueError("compiled projection arithmetic differs")
        if row["mode"]==5:
            if not equal(data(name,"mlp_return"),baseline):raise ValueError("held write changed")
            if any((directory/f"{name}-{p}.f32").exists() for p in ("mlp_gate","mlp_up")):raise ValueError("fabricated held-write projection ports")
    comparisons=[]
    for group in (("source","Niko-D") if formation else ("source","Niko-C") if components else ("source","Niko-A","Niko-B") if enrich else ("source","Niko","Mira")):
        dense=by_name["source-native" if group=="source" else group+"-dense"]
        modes=("project","correct","reassoc") if group=="source" else (("project","inherited","new","carry") if formation else ("project","carry","read","correct") if components else ("project2","project","correct") if enrich else ("project","correct","reassoc","held"))
        for mode in modes:
            row=by_name[f"{group}-{mode}"];name=row["name"]
            if row["parent_kv_prefix_sha256"]!=dense["parent_kv_prefix_sha256"] or row["position"]!=dense["position"]:raise ValueError("comparison KV prefixes differ")
            if not equal(data(name,"receiver"),data(dense["name"],"receiver")) or not equal(data(name,"mlp_input"),data(dense["name"],"mlp_input")):raise ValueError("live receiver differs across arms")
            if row["rho"]!=dense["rho"]:raise ValueError("full receiver scale differs across arms")
            if enrich and any(not equal(data(name,p),data(dense["name"],p)) for p in ("entry","attention-write")):raise ValueError("pre-add operands differ across arms")
            record={"group":group,"mode":mode,"name":name,"omitted_relative_norm":row["omitted_norm"]/max(row["receiver_norm"],1e-30),"coefficients":row["executed_coefficients"],"ports":{},
                    "own_ids_equal":row["own"]["ids"]==dense["own"]["ids"],"own_text":row["own"]["text"],"dense_text":dense["own"]["text"],
                    "first_red_blue_margin_delta":row["first"]["red_minus_blue"]-dense["first"]["red_minus_blue"],
                    "common_next_top_equal":row["common_next"]["top"]==dense["common_next"]["top"],
                    "questions":[{"name":q["name"],"text":q["output"]["text"],"dense_text":d["output"]["text"],"same_own_history":row["own"]["ids"]==dense["own"]["ids"],"red_blue_margin_delta":q["first"]["red_minus_blue"]-d["first"]["red_minus_blue"]} for q,d in zip(row["questions"],dense["questions"])]}
            for port in ("mlp_gate","mlp_up","mlp_return","output","first","current-kv","next-kv"):
                if mode=="held" and port in ("mlp_gate","mlp_up"):continue
                record["ports"][port]=measure(data(name,port),data(dense["name"],port))
            comparisons.append(record)
        corrected=by_name[group+("-carry" if formation and group!="source" else "-correct")];projected=by_name[group+"-project"]
        for channel in ("gate","up"):
            if not equal(data(corrected["name"],"mlp_"+channel),np.add(data(projected["name"],"mlp_"+channel),data(corrected["name"],"correction-"+channel))):raise ValueError("residual correction did not add to the actual candidate")
        if (components or formation) and group!="source":
            for mode in (("inherited","new") if formation else ("carry","read")):
                for channel in ("gate","up"):
                    if not equal(data(group+"-"+mode,"mlp_"+channel),np.add(data(projected["name"],"mlp_"+channel),data(group+"-"+mode,"correction-"+channel))):raise ValueError("component correction assembly differs")
    target_project=[r for r in comparisons if r["group"]!="source" and r["mode"]=="project"]
    corrections=[r for r in comparisons if r["mode"]=="correct"]
    summary={
        "rows":len(rows),"later_contexts":1 if components or formation else 2,"source_compile_projections":4,"target_projections_skipped":2 if components or formation else 8 if enrich else 4,
        "frozen_device_program_bytes":len(raw),"frozen_qr_bytes":len(qr_raw),"basis_condition":float(condition),
        "target_project_write_relative_errors":[r["ports"]["mlp_return"]["relative_l2"] for r in target_project],
        "target_project_omitted_relative_norms":[r["omitted_relative_norm"] for r in target_project],
        "matching_target_project_own_paths":sum(r["own_ids_equal"] for r in target_project),
        "max_residual_corrected_write_relative_error":max(r["ports"]["mlp_return"]["relative_l2"] for r in corrections),
        "source_text":driver["rows"][0]["own"]["text"],"rebuilt_source_ids_equal":driver["rebuilt_source_own"]["ids"]==driver["rows"][0]["own"]["ids"]}
    if enrich:
        summary.update(extension_compile_projections=2,extension_device_bytes=extra_bytes,extension_qr_bytes=extra_qr_bytes,
            target_project2_write_relative_errors=[r["ports"]["mlp_return"]["relative_l2"] for r in comparisons if r["mode"]=="project2"],
            target_project2_omitted_relative_norms=[r["omitted_relative_norm"] for r in comparisons if r["mode"]=="project2"],
            basis3_condition=float(np.linalg.cond(np.stack((x,a,b),axis=1).astype(float))))
        def omitted(v):
            v=v.astype(float);return v-q0*np.dot(q0,v)-q1*np.dot(q1,v)
        for row in comparisons:
            if row["group"]=="source":continue
            ex,ea=(omitted(data(row["name"],p)) for p in ("entry","attention-write"));e=ex+ea;den=max(float(np.dot(e,e)),1e-30)
            row["omitted_attribution"]={"carried_squared_norm":float(np.dot(ex,ex)),"attention_squared_norm":float(np.dot(ea,ea)),"cross_term":float(2*np.dot(ex,ea)),"sum_squared_norm":float(np.dot(e,e)),
                "carried_signed_energy_fraction":float(np.dot(ex,e)/den),"attention_signed_energy_fraction":float(np.dot(ea,e)/den),
                "scope":"FP64 B2 projection of the two captured pre-add operands; signed energy attribution, not semantic attribution"}
    if components:
        base="Niko-C";q=(q0,q1,q2);entry=data(base+"-dense","entry").astype(float);attention=data(base+"-dense","attention-write").astype(float)
        def project_away(v):
            dots=[seqdot(column,v) for column in q];out=v.copy()
            for column,dot in zip(q,dots):out-=column*dot
            return out
        ex=project_away(entry).astype(np.float32);e=data(base+"-project","omitted")
        ea=np.add(np.add(e,np.multiply(np.float32(-1),ex)),np.multiply(np.float32(0),ex))
        for mode in ("carry","read"):
            if not equal(data(base+"-"+mode,"carry-omitted"),ex) or not equal(data(base+"-"+mode,"read-complement"),ea):raise ValueError("component operands describe another split")
            for port in ("entry","attention-write","reconstruction","omitted"):
                if not equal(data(base+"-"+mode,port),data(base+"-project",port)):raise ValueError("component receiver ancestry differs")
        diagnostic=driver["split_joint"]
        if diagnostic!={"gate_up_matvecs":0,"down_matvecs":1,"session_metadata_unchanged":True,"kv_prefix_sha256":driver["source_answer_kv_prefix_sha256"],"state_advanced":False}:raise ValueError("split-joint work/state instrument differs")
        for channel in ("gate","up"):
            joined=np.add(np.add(data(base+"-project","mlp_"+channel),data(base+"-carry","correction-"+channel)),data(base+"-read","correction-"+channel))
            if not equal(load(f"split-joint-mlp_{channel}.f32",8960),joined):raise ValueError("split-joint actual operands differ")
        joint=load("split-joint-mlp_return.f32",1536).astype(float);m0=data(base+"-project","mlp_return").astype(float);mx=data(base+"-carry","mlp_return").astype(float);ma=data(base+"-read","mlp_return").astype(float);mf=data(base+"-correct","mlp_return").astype(float);native=data(base+"-dense","mlp_return").astype(float)
        interaction=joint-mx-ma+m0;total=joint-m0;closure=e.astype(float)-ex.astype(float)-ea.astype(float)
        detail={"scope":"five primary native-continuation arms plus one pure SiLU/down diagnostic; read-side complement includes recorded rounding differences",
            "carry_operand_norm":float(np.linalg.norm(ex.astype(float))),"read_complement_norm":float(np.linalg.norm(ea.astype(float))),
            "operand_split_closure_l2":float(np.linalg.norm(closure)),"read_complement_vs_independent_attention_projection":measure(ea,project_away(attention)),
            "split_joint_vs_full_corrected_write":measure(joint,mf),"nonlinear_interaction":measure(interaction,np.zeros_like(interaction)),
            "interaction_l2_over_native_write_norm":float(np.linalg.norm(interaction)/np.linalg.norm(native)),
            "interaction_l2_over_total_correction_norm":float(np.linalg.norm(interaction)/max(np.linalg.norm(total),1e-30)),
            "additive_write_forecast_error":measure(mx+ma-m0,joint),"projection_split_mismatch":{},"error_repair":{}}
        # A zero reference does not define a meaningful relative interaction error.
        detail["nonlinear_interaction"].pop("relative_l2")
        for channel in ("gate","up"):
            summed=data(base+"-carry","correction-"+channel).astype(float)+data(base+"-read","correction-"+channel).astype(float)
            detail["projection_split_mismatch"][channel]=measure(summed,data(base+"-correct","correction-"+channel))
        for port in ("mlp_return","first","current-kv","next-kv"):
            reference=data(base+"-dense",port).astype(float);baseline=data(base+"-project",port).astype(float);err=baseline-reference;den=max(float(np.dot(err,err)),1e-30)
            detail["error_repair"][port]={}
            for mode in ("carry","read","correct"):
                actual=data(base+"-"+mode,port).astype(float);delta=actual-baseline
                detail["error_repair"][port][mode]={"signed_error_direction_removed":float(-np.dot(delta,err)/den),"error_norm_reduction":float(1-np.linalg.norm(actual-reference)/max(np.linalg.norm(err),1e-30))}
        summary["component_diagnostics"]=detail
    return {"comparisons":comparisons,"summary":summary}
