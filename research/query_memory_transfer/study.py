"""Saved-only historical endpoint joins and marginal work account."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def study(root,previous):
    root,previous=Path(root),Path(previous);report=json.loads((root/'report.json').read_text());old=json.loads((previous/'report.json').read_text())
    if json.loads((root/'verification.json').read_text())['report_sha256']!=sha(root/'report.json'):raise ValueError('New verification identity differs')
    if sha(previous/'report.json')!=report['program_provenance']['source_report_sha256']:raise ValueError('Predecessor report identity differs')
    labels={0:'native',1:'compiled',2:'memory',3:'compiled-memory',5:'values-native'};joins=[]
    for p in (p for p in report['driver']['paths'] if p['policy'] in labels):
        oldname=f'case{p["task"]}-{labels[p["policy"]]}';q=next(q for q in old['driver']['paths'] if q['name']==oldname);name=p['name'];matches=[]
        if p['consumed_ids']!=q['consumed_ids'] or p['turns']!=q['turns']:raise ValueError('Historical history or cache-bound endpoint differs: '+name)
        suffixes=['norm.f32','q.f32','k.f32','v.f32','attention.f32','output.f32','coeff.f32','tops.i32']
        if not p['compact']:suffixes.append('probabilities.f32')
        for suffix in suffixes:
            a,b=root/'raw'/f'{name}-{suffix}',previous/'raw'/f'{oldname}-{suffix}'
            if a.read_bytes()!=b.read_bytes():raise ValueError('Historical array differs: '+a.name)
            matches.append(a.name)
        for turn in range(2):
            for stage in ('prefill','final'):
                for suffix in (['logits.f32','memory.bin'] if p['compact'] else ['logits.f32']):
                    a,b=root/'raw'/f'{name}-t{turn}-{stage}-{suffix}',previous/'raw'/f'{oldname}-t{turn}-{stage}-{suffix}'
                    if a.read_bytes()!=b.read_bytes():raise ValueError('Historical boundary differs: '+a.name)
                    matches.append(a.name)
        joins.append({'path':name,'historical_path':oldname,'history_and_cache_bound_turn_records_exact':True,'byte_exact_arrays':len(matches)})
    bank=[]
    for kind in ('basis','q','k','v'):
        name=f'program128-{kind}.f32';a,b=root/'raw'/name,previous/'raw'/name
        if a.read_bytes()!=b.read_bytes():raise ValueError('Historical bank differs')
        bank.append({'file':name,'sha256':sha(a),'bytes':a.stat().st_size})
    economics=[]
    for p in report['driver']['paths']:
        selected=(196608 if p['factor'] else 0)+(229376 if p['qk_factor'] else 2752512)+(32768 if p['v_factor'] else 393216)
        n=p['count'];economics.append({'path':p['name'],'selected_qkv_macs_per_update':selected,'selected_qkv_saving_fraction':1-selected/3145728,'whole_token_linear_saving_fraction':(3145728-selected)/1543569408,'actual_advances':n,'linear_calls':p['linear_calls'],'linear_macs':p['linear_macs'],'reader_products':p['reader_products'],'coordinate_calls':p['coordinate_calls'],'serving_transfer_bytes':p['serving_transfer_bytes'],'diagnostic_probability_bytes':6*n*(n+1)*4 if not p['compact'] and not p['zero'] else 0,'instrumented_path_seconds':p['seconds']})
    initial=[]
    for task in range(3):
        policies=['native','values-native','values-native-memory'];paths=[next(p for p in report['driver']['paths'] if p['name']==f'case{task}-{label}') for label in policies];stop=paths[0]['turns'][0]['prefill']['position']
        histories=[p['consumed_ids'][:p['turns'][0]['prefill']['position']] for p in paths]
        entry={'task':task,'same_initial_consumed_ids':all(h==histories[0] for h in histories),'scope':'Operational path decomposition; full-logit increments also include policy-specific upper memory; norms are not additive causal shares.'}
        for port,suffix in (('attention','attention.f32'),('logits','t0-prefill-logits.f32')):
            ys=[np.fromfile(root/'raw'/f'{p["name"]}-{suffix}',dtype='<f4').astype(float) for p in paths]
            if port=='attention':ys=[y.reshape(-1,12,128)[:stop].reshape(-1) for y in ys]
            formation,memory=ys[1]-ys[0],ys[2]-ys[1];den=max(float(np.linalg.norm(ys[0])),1e-30);product=float(np.linalg.norm(formation)*np.linalg.norm(memory))
            entry[port]={'formation_relative':float(np.linalg.norm(formation)/den),'memory_increment_relative':float(np.linalg.norm(memory)/den),'combined_relative':float(np.linalg.norm(ys[2]-ys[0])/den),'formation_memory_cosine':float(formation@memory/product) if product else None,'signed_cross_scaled':float(2*formation@memory/den**2),'vector_closure_relative':float(np.linalg.norm(ys[2]-ys[0]-formation-memory)/den)}
        initial.append(entry)
    result={'schema':'tcm-query-memory-transfer-study-v1','scope':'saved-only exact historical joins, operational initial error decomposition and nominal work; no model calls or speedup claim','report_sha256':sha(root/'report.json'),'previous_report_sha256':sha(previous/'report.json'),'historical_joins':joins,'historical_arrays_exact':sum(x['byte_exact_arrays'] for x in joins),'source_bank_exact':bank,'new_compilation_or_calibration_forwards':0,'source_acquisition_and_compilation_paid_historically':True,'economics':economics,'initial_error_decomposition':initial,'terminal_status':'not-requested'}
    (root/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report_sha256':result['report_sha256'],'historical_paths_exact':len(joins),'historical_arrays_exact':result['historical_arrays_exact'],'bank_arrays_exact':len(bank)}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--previous',default=str(Path(__file__).resolve().parent.parent/'query_value_cut/runs/job-27a597ebd5f8'));a=p.parse_args();study(a.root,a.previous)
