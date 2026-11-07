"""Post-hoc dispersion/covariance omission account, using saved evidence only."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(np.ravel(a),np.ravel(b)))
def norm(a):return math.sqrt(dot(a,a))
def audit(root):
    root=Path(root);report=json.loads((root/"report.json").read_text());verification=json.loads((root/"verification.json").read_text())
    if verification["report_sha256"]!=sha(root/"report.json"):raise ValueError("Verified report differs")
    dependencies={}
    def checked(name):
        p=root/"raw"/name;record=report["raw_files"][name]
        if sha(p)!=record["sha256"] or p.stat().st_size!=record["bytes"]:raise ValueError("Raw audit dependency differs")
        dependencies[name]=record;return p
    def load(name,shape,dtype="<f4"):
        x=np.fromfile(checked(name),dtype=dtype).astype(float)
        if x.size!=math.prod(shape) or not np.isfinite(x).all():raise ValueError("Invalid audit array")
        return x.reshape(shape)
    keys=load("audit-span-keys.f32",(2,34,128));values=load("audit-span-values.f32",(2,34,128));scale=float(np.float32(1/math.sqrt(128)));programs={p["name"]:p for p in report["driver"]["programs"]};rows=[]
    for task in ("recall","prose"):
        query=load(task+"-native-query.f32",(12,128))
        for label in ("one","eight"):
            p=programs[label];m=p["clusters"];checked(p["file"]);members=np.asarray(p["membership"],dtype=np.uint64).reshape(2,m);prediction=load(task+"-"+label+"-response.f64",(12,129),"<f8");all_d=[];all_c=[];all_r=[];all_error=[];zrows=[];closure=0.
            for h in range(12):
                head=h//6;q=query[h];scores=[dot(q,k)*scale for k in keys[head]];g=max(scores);e=[math.exp(s-g) for s in scores];z=math.fsum(e);n=np.asarray([math.fsum(w*float(v[d]) for w,v in zip(e,values[head])) for d in range(128)]);ideal_n=np.zeros(128);ideal_z=0.;dispersion=np.zeros(128);covariance=np.zeros(128);dz=0.
                for mask in members[head]:
                    ids=[i for i in range(34) if int(mask)&(1<<i)];count=len(ids);kb=np.asarray([math.fsum(float(keys[head,i,d]) for i in ids)/count for d in range(128)]);vb=np.asarray([math.fsum(float(values[head,i,d]) for i in ids)/count for d in range(128)]);sbar=dot(q,kb)*scale;base=math.exp(sbar-g);centered_exp=[math.exp(scores[i]-sbar) for i in ids];excess=math.fsum(centered_exp)-count
                    ideal_z+=count*base;ideal_n+=count*base*vb;dz+=base*excess;dispersion+=base*excess*vb
                    covariance+=base*np.asarray([math.fsum(w*(float(values[head,i,d])-vb[d]) for w,i in zip(centered_exp,ids)) for d in range(128)])
                pred_z=math.exp(float(prediction[h,0])-g);pred_n=pred_z*prediction[h,1:];remainder=ideal_n-pred_n;error=n-pred_n;difference=error-dispersion-covariance-remainder;closure=max(closure,float(np.max(np.abs(difference/z))))
                all_d.append(dispersion/z);all_c.append(covariance/z);all_r.append(remainder/z);all_error.append(error/z);zrows.append({"head":h,"relative_Z_omission":(z-pred_z)/z,"dispersion_relative_Z_omission":dz/z,"stored_primary_relative_Z_remainder":(ideal_z-pred_z)/z,"Z_account_closure":(z-pred_z-dz-(ideal_z-pred_z))/z})
            d,c,r,error=map(np.asarray,(all_d,all_c,all_r,all_error));rows.append({"task":task,"bank":label,"normalization":"each head numerator divided by its exact span Z before concatenating heads","numerator_omission_norm":norm(error),"dispersion_term_norm":norm(d),"key_value_dependence_term_norm":norm(c),"ideal_to_stored_primary_remainder_norm":norm(r),"dispersion_dependence_cross":2*dot(d,c),"dispersion_remainder_cross":2*dot(d,r),"dependence_remainder_cross":2*dot(c,r),"vector_account_closure_max_abs":closure,"Z_heads":zrows})
    result={"schema":"tcm-span-omission-audit-v1","report_sha256":sha(root/"report.json"),"verification_sha256":sha(root/"verification.json"),"audit_source_sha256":sha(__file__),"post_hoc":True,"models_evaluated":0,"raw_dependencies":dependencies,"rows":rows,"terminal_status":"not-requested"};data=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n";target=root/"omission-audit.json"
    if target.exists() and target.read_text()!=data:raise ValueError("Immutable audit differs")
    target.write_text(data);print(json.dumps({"audit":str(target),"rows":[{k:v for k,v in r.items() if k!="Z_heads"} for r in rows]}));return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");a=p.parse_args();audit(a.root)
