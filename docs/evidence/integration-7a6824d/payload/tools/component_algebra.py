"""Finite SwiGLU interaction on saved ports; no model or down projection runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(root):
    def load(name):
        value=np.fromfile(root/"raw"/name,dtype="<f4").astype(float)
        if value.shape!=(8960,) or not np.isfinite(value).all():raise ValueError("invalid saved MLP port")
        return value
    def silu(v):
        z=np.exp(-np.abs(v));sigmoid=np.where(v>=0,1/(1+z),z/(1+z));return v*sigmoid
    def port(mode,channel):return load(f"Niko-C-{mode}-mlp_{channel}.f32")
    g0,gx,ga=(port(mode,"gate") for mode in ("project","carry","read"));gj=load("split-joint-mlp_gate.f32")
    u0,ux,ua=(port(mode,"up") for mode in ("project","carry","read"));uj=load("split-joint-mlp_up.f32")
    s0,sx,sa,sj=map(silu,(g0,gx,ga,gj));dxu,dau=ux-u0,ua-u0;eta=uj-ux-ua+u0
    terms={"gate_curvature":(sj-sx-sa+s0)*u0,"read_gate_on_carried_up":(sj-sx)*dxu,"carried_gate_on_read_up":(sj-sa)*dau,"up_assembly_rounding":sj*eta}
    combined=sum(terms.values());direct=sj*uj-sx*ux-sa*ua+s0*u0;den=max(float(np.dot(combined,combined)),1e-30)
    result={"schema":"tcm-finite-swiglu-algebra-v1","report_sha256":hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),"analyzer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope":"FP64 SiLU replay on four actual FP32 gate/up ports; exact finite port identity with up-assembly remainder; neither native-CUDA SiLU parity nor down-weighted causal importance",
        "interaction_product_l2":float(np.linalg.norm(combined)),"identity_roundoff_max_abs":float(np.max(np.abs(combined-direct))),"up_assembly_eta_l2":float(np.linalg.norm(eta)),"gate_assembly_eta_l2":float(np.linalg.norm(gj-gx-ga+g0)),"terms":{}}
    for name,value in terms.items():result["terms"][name]={"l2":float(np.linalg.norm(value)),"signed_projection_share":float(np.dot(value,combined)/den)}
    result["pairwise_inner_products"]={a+" / "+b:float(np.dot(terms[a],terms[b])) for i,a in enumerate(terms) for b in list(terms)[i+1:]}
    # A second-order cross term is a diagnostic approximation to this finite
    # response, never an alternative to the executed nonlinear consumer.
    z=np.exp(-np.abs(g0));sigma=np.where(g0>=0,1/(1+z),z/(1+z));sp=sigma+g0*sigma*(1-sigma);spp=sigma*(1-sigma)*(2+g0*(1-2*sigma));dxg,dag=gx-g0,ga-g0
    quadratic=spp*dxg*dag*u0+sp*(dag*dxu+dxg*dau)
    result["quadratic_product_interaction_relative_error"]=float(np.linalg.norm(quadratic-combined)/max(np.linalg.norm(combined),1e-30))
    result["quadratic_product_interaction_cosine"]=float(np.dot(quadratic,combined)/max(np.linalg.norm(quadratic)*np.linalg.norm(combined),1e-30))
    # A fixed positive-gate chart gives a different conditional quadratic law.
    # Its success here would only concern products before the learned down map.
    mask=g0>0
    fixed=mask*(gj*uj-gx*ux-ga*ua+g0*u0)
    result["fixed_sign_chart_interaction_relative_error"]=float(np.linalg.norm(fixed-combined)/max(np.linalg.norm(combined),1e-30))
    result["fixed_sign_chart_interaction_cosine"]=float(np.dot(fixed,combined)/max(np.linalg.norm(fixed)*np.linalg.norm(combined),1e-30))
    result["fixed_sign_chart_baseline_relative_error"]=float(np.linalg.norm(mask*g0*u0-s0*u0)/max(np.linalg.norm(s0*u0),1e-30))
    frozen=sigma*(gj*uj-gx*ux-ga*ua+g0*u0)
    result["frozen_sigmoid_chart_interaction_relative_error"]=float(np.linalg.norm(frozen-combined)/max(np.linalg.norm(combined),1e-30))
    result["frozen_sigmoid_chart_interaction_cosine"]=float(np.dot(frozen,combined)/max(np.linalg.norm(frozen)*np.linalg.norm(combined),1e-30))
    source_gate=load("source-native-mlp_gate.f32");zsource=np.exp(-np.abs(source_gate));source_sigma=np.where(source_gate>=0,1/(1+zsource),zsource/(1+zsource))
    source_frozen=source_sigma*(gj*uj-gx*ux-ga*ua+g0*u0)
    result["source_frozen_sigmoid_chart_interaction_relative_error"]=float(np.linalg.norm(source_frozen-combined)/max(np.linalg.norm(combined),1e-30))
    result["source_frozen_sigmoid_chart_interaction_cosine"]=float(np.dot(source_frozen,combined)/max(np.linalg.norm(source_frozen)*np.linalg.norm(combined),1e-30))
    source_product=source_sigma*g0*u0;candidate_product=s0*u0;dense_product=silu(port("dense","gate"))*port("dense","up")
    result["source_chart_gate_freezing_error_relative_candidate_product"]=float(np.linalg.norm(source_product-candidate_product)/max(np.linalg.norm(candidate_product),1e-30))
    result["source_chart_total_error_relative_dense_product"]=float(np.linalg.norm(source_product-dense_product)/max(np.linalg.norm(dense_product),1e-30))
    result["candidate_error_relative_dense_product"]=float(np.linalg.norm(candidate_product-dense_product)/max(np.linalg.norm(dense_product),1e-30))
    result["chart_scope"]="fixed-sign and baseline-exact frozen-sigmoid charts use the target candidate and are posthoc local diagnostics; source-frozen chart uses only the recorded source native gate; all concern products before the down map, not a serving promotion"
    result["gate_sign_changes_from_candidate"]={name:int(np.count_nonzero((value>0)!=mask)) for name,value in (("carry",gx),("read",ga),("joint",gj))}
    (root/"finite-swiglu-algebra.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);print(json.dumps(analyze(p.parse_args().root)))
