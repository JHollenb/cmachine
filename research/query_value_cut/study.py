"""Saved-only historical endpoint joins and marginal work account."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def study(root,previous):
    root,previous=Path(root),Path(previous);report=json.loads((root/'report.json').read_text());old=json.loads((previous/'report.json').read_text())
    if json.loads((root/'verification.json').read_text())['report_sha256']!=sha(root/'report.json'):raise ValueError('New verification identity differs')
    if sha(previous/'report.json')!=report['program_provenance']['source_report_sha256']:raise ValueError('Predecessor report identity differs')
    labels={0:'native',1:'project128',2:'memory',3:'both128',4:'zero'};joins=[]
    for p in (p for p in report['driver']['paths'] if p['policy'] in labels):
        oldname=f'case{p["task"]}-{labels[p["policy"]]}';q=next(q for q in old['driver']['paths'] if q['name']==oldname);name=p['name'];matches=[]
        if p['consumed_ids']!=q['consumed_ids'] or p['turns']!=q['turns']:raise ValueError('Historical history or cache-bound endpoint differs: '+name)
        for suffix in ('norm.f32','q.f32','k.f32','v.f32','attention.f32','output.f32','coeff.f32','tops.i32'):
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
    result={'schema':'tcm-query-value-study-v1','scope':'saved-only exact historical joins and nominal work; no model calls or speedup claim','report_sha256':sha(root/'report.json'),'previous_report_sha256':sha(previous/'report.json'),'historical_joins':joins,'historical_arrays_exact':sum(x['byte_exact_arrays'] for x in joins),'source_bank_exact':bank,'new_compilation_or_calibration_forwards':0,'source_acquisition_and_compilation_paid_historically':True,'economics':economics,'terminal_status':'not-requested'}
    (root/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report_sha256':result['report_sha256'],'historical_paths_exact':len(joins),'historical_arrays_exact':result['historical_arrays_exact'],'bank_arrays_exact':len(bank)}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--previous',default=str(Path(__file__).resolve().parent.parent/'query_memory/runs/job-c7334bbc5bb8'));a=p.parse_args();study(a.root,a.previous)
