"""Posthoc fixed-query factoring of saved support, payload and query changes.

This does not construct a serving summary or execute another model trajectory.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def context(state,q,wi):return state.context(q.astype(np.float64),wi)
def states(root,module,arm):
    raw=root/'raw';ai=['native','full_byte','recent','anchors','reservoir'].index(arm['name']);a=16 if ai>=3 else 0;r=768 if ai==1 else 64;m=32 if ai==4 else 0;s=module.State(a,r,m);count=sum(len(t['steps'])+len(t['generation_steps']) for t in arm['turns']);rows=np.fromfile(raw/(arm['name']+'-rows.bin'),dtype='<f4').reshape(count,3,2,2,128);offset=0;result={};hist=[];source=None;source_hist=None;parent=None;parent_hist=None
    for ti,t in enumerate(arm['turns']):
        if ti==2:s=source.clone();hist=list(source_hist)
        if ti>=4:s=parent.clone();hist=list(parent_hist)
        for j,step in enumerate(t['steps']+t['generation_steps']):
            s.append(rows[offset]);offset+=1;hist.append(step[1])
            if j==len(t['steps'])-1:result[t['input_index']]=(s.clone(),list(hist),t['name'])
        if ti==0:source=s.clone();source_hist=list(hist)
        if ti==3:parent=s.clone();parent_hist=list(hist)
    return result
def distribution(state,q,wi,h):
    order=np.flatnonzero(state.pos>=0);order=order[np.argsort(state.pos[order])];kh=h//6;k=(state.codes[wi,0,kh,order].astype(np.float32)*state.scales[wi,0,kh,order,None]).astype(np.float32).astype(np.float64);n=min(state.n,state.m);logw=[math.log(state.n/n) if n and i>=state.a+state.r else 0 for i in order];scores=[math.fsum(float(x)*float(y) for x,y in zip(row,q[h]))/math.sqrt(128)+w for row,w in zip(k,logw)];maximum=max(scores);p=[math.exp(z-maximum) for z in scores];z=math.fsum(p);return state.pos[order],np.array([x/z for x in p]),maximum+math.log(z)
def run(root):
    root=Path(root);r=json.loads((root/'report.json').read_text());v=json.loads((root/'verification.json').read_text());manifest=json.loads((root/'payload/source-manifest.json').read_text())
    if v['report_sha256']!=sha(root/'report.json') or sha(root/'payload/analysis.py')!=manifest['sources']['analysis.py']:raise ValueError('Source verification differs')
    for name,record in r['raw_files'].items():
        p=root/'raw'/name
        if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Raw byte differs')
    sp=importlib.util.spec_from_file_location('frozen_bounded_analyzer',root/'payload/analysis.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);full=states(root,mod,r['driver']['arms'][1]);candidate=states(root,mod,r['driver']['arms'][4]);records=[];headrows=[]
    for ix in range(4,8):
        fs,fh,fn=full[ix];cs,ch,cn=candidate[ix]
        if fh!=ch:records.append({'input_index':ix,'instrument_status':'inapplicable-different-history'});continue
        ideal=cs.clone()
        for slot,p in enumerate(cs.pos):
            if p>=0:ideal.codes[:,:,:,slot]=fs.codes[:,:,:,int(p)];ideal.scales[:,:,:,slot]=fs.scales[:,:,:,int(p)]
        facts=r['analysis']['summary']['roots']['reservoir']['fact_support'];fact_positions={p for f in facts for p in f['absolute_positions']}
        for wi,l in enumerate((0,17,27)):
            qf=np.fromfile(root/'raw'/f'{fn}-prefill-l{l}-query.f32',dtype='<f4').reshape(12,128);qc=np.fromfile(root/'raw'/f'{cn}-prefill-l{l}-query.f32',dtype='<f4').reshape(12,128);yf=np.fromfile(root/'raw'/f'{fn}-prefill-l{l}-attention.f32',dtype='<f4').reshape(12,128);yc=np.fromfile(root/'raw'/f'{cn}-prefill-l{l}-attention.f32',dtype='<f4').reshape(12,128)
            unit=ideal.clone();unit.m=0;base=context(fs,qf,wi);omitted=context(unit,qf,wi);support=context(ideal,qf,wi);payload=context(cs,qf,wi);query=context(cs,qc,wi);terms=[omitted-base,support-omitted,payload-support,query-payload];delta=query-base;den=max(float(np.linalg.norm(base)),1e-30);executed_delta=yc.astype(np.float64)-yf.astype(np.float64);arithmetic=(yc-query)-(yf-base);closure=executed_delta-(sum(terms)+arithmetic)
            row={'input_index':ix,'layer':l,'same_history':True,'query_relative_l2':mod.metric(qc,qf)['relative_l2'],'omission_relative_norm':float(np.linalg.norm(terms[0])/den),'multiplicity_relative_norm':float(np.linalg.norm(terms[1])/den),'support_combined_relative_norm':float(np.linalg.norm(support-base)/den),'payload_relative_norm':float(np.linalg.norm(terms[2])/den),'query_relative_norm':float(np.linalg.norm(terms[3])/den),'total_ideal_relative_norm':float(np.linalg.norm(delta)/den),'executed_context_relative_norm':float(np.linalg.norm(executed_delta)/den),'arithmetic_relative_norm':float(np.linalg.norm(arithmetic)/den),'ordered_term_gram_over_base_energy':[[float(np.vdot(x,y)/den**2) for y in terms] for x in terms],'signed_context_terms':{k:x.tolist() for k,x in zip(('omission','multiplicity','payload','query'),terms)},'absolute_closure_l2':float(np.linalg.norm(closure))};records.append(row)
            for h in range(12):
                positions,p,logz=distribution(fs,qf.astype(np.float64),wi,h);selected,ps,logzs=distribution(ideal,qf.astype(np.float64),wi,h);_,pu,logzu=distribution(unit,qf.astype(np.float64),wi,h);lookup={int(a):float(b) for a,b in zip(selected,ps)};unit_lookup={int(a):float(b) for a,b in zip(selected,pu)};aligned=np.array([lookup.get(int(a),0) for a in positions]);unaligned=np.array([unit_lookup.get(int(a),0) for a in positions]);missing=np.array([int(a) not in lookup for a in positions]);factmask=np.array([int(a) in fact_positions for a in positions]);headrows.append({'input_index':ix,'layer':l,'head':h,'full_byte_probability_on_omitted_rows':float(p[missing].sum()),'probability_on_literal_fact_clause_rows':float(p[factmask].sum()),'probability_on_omitted_literal_fact_clause_rows':float(p[missing&factmask].sum()),'unit_support_probability_l1':float(np.abs(unaligned-p).sum()),'unit_support_log_partition_change':logzu-logz,'weighted_support_probability_l1':float(np.abs(aligned-p).sum()),'weighted_support_log_partition_change':logzs-logz})
    summary=[]
    for l in (0,17,27):
        rr=[x for x in records if x.get('layer')==l];hh=[x for x in headrows if x['layer']==l];summary.append({'layer':l,**{k:math.fsum(x[k] for x in rr)/len(rr) for k in ('omission_relative_norm','multiplicity_relative_norm','support_combined_relative_norm','payload_relative_norm','query_relative_norm','total_ideal_relative_norm','executed_context_relative_norm')},'mean_full_byte_probability_on_omitted_rows':math.fsum(x['full_byte_probability_on_omitted_rows'] for x in hh)/len(hh),'max_full_byte_probability_on_omitted_rows':max(x['full_byte_probability_on_omitted_rows'] for x in hh)})
    out={'schema':'tcm-bounded-retention-ordered-factor-v2','report_sha256':sha(root/'report.json'),'source_sha256':sha(__file__),'models_evaluated':0,'scope':'posthoc saved-operand numeric factoring at four matching-history late prefills, full-byte and own reservoir; ordered omission/multiplicity/payload/query path, not causal answer shares or a new serving summary','summary':summary,'contrasts':records,'head_rows':headrows};data=json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'
    for path,content in ((root/'diagnostic-v2.json',data.encode()),(root/'diagnostic-v2-source.py',Path(__file__).read_bytes())):
        if path.exists() and path.read_bytes()!=content:raise ValueError('Immutable diagnostic differs')
        path.write_bytes(content)
    print(json.dumps({'summary':summary,'contrasts':len(records),'head_rows':len(headrows),'models_evaluated':0},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
