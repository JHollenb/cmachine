"""Additive saved-only control joins and economic/operand diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):return {'mean':float(np.mean(x)),'max':float(np.max(x)),'min':float(np.min(x))}
def relative(a,b):return float(np.linalg.norm(a.astype(float)-b.astype(float))/max(float(np.linalg.norm(b.astype(float))),1e-30))
def study(root,previous=None):
    root=Path(root);report=json.loads((root/'report.json').read_text());paths=report['driver']['paths'];rows=[];joins=[];operand_rows=[]
    for desc in paths:
        n=desc['count'];compact=desc['compact'];rank=desc.get('rank',32 if desc['factor'] else 0);name=desc['name'];obs=[r for r in report['analysis']['observations'] if r.get('kind')=='own-query-attention' and r['path']==name]
        k=min(n,32);full_products=3072*n*(n+1)//2;entries=sum(min(t,32) for t in range(1,n+1));prototypes=sum(min(max(t-16,0),16) for t in range(1,n+1));recent_ring_rows=sum(min(t,16) for t in range(1,n+1))
        row={'path':name,'rank':rank,'advances':n,'first_unmerged_32_own_kv_attention_error':stats([r['own_kv_attention_relative_error'] for r in obs[:k]]),'postmerge_own_kv_attention_error':stats([r['own_kv_attention_relative_error'] for r in obs[32:]]) if n>32 else None,'selected_qkv_macs_per_update':3584*rank if rank else 3145728,'full_attention_products_at_own_length':full_products,'actual_selected_read_products':desc['reader_products'],'read_product_saving_fraction_at_own_length':1-desc['reader_products']/full_products,'selected_valid_native_kv_bytes_at_own_length':2048*n,'compact_logical_state_bytes':98464 if compact else None,'compact_vs_valid_native_fraction':98464/(2048*n) if compact else None,'selected_full_capacity_arena_bytes_still_paid':1048576,'compact_scalar_work':None}
        if compact:
            row['compact_scalar_work']={'exp_calls':12*entries,'log_calls':12*prototypes,'read_divisions':24*128*prototypes+12*entries+1536*n,'routing_divisions':4096*max(n-32,0),'sum_updates':512*max(n-16,0),'recent_scalar_writes':512*n,'serving_cpu_gpu_bytes':desc['serving_transfer_bytes'],'recording_hashing_basis_compilation_and_native_scaffold_additional':True}
        rows.append(row)
    # Same initial prompt histories permit direct native operand comparisons.
    for desc in (p for p in paths if p['factor'] or p['compact']):
        native=next(p for p in paths if p['task']==desc['task'] and p['policy']==0);n=desc['turns'][0]['prefill']['position'];nn=native['turns'][0]['prefill']['position']
        if n!=nn or desc['consumed_ids'][:n]!=native['consumed_ids'][:nn]:raise ValueError('Initial common history differs')
        item={'candidate':desc['name'],'initial_prompt_positions':n,'same_consumed_ids':True}
        for field,width in (('norm',1536),('q',1536),('k',256),('v',256),('attention',1536),('output',1536)):
            a=np.fromfile(root/'raw'/f'{desc["name"]}-{field}.f32',dtype='<f4').reshape(-1,width)[:n];b=np.fromfile(root/'raw'/f'{native["name"]}-{field}.f32',dtype='<f4').reshape(-1,width)[:n]
            item[field+'_byte_exact']=a.tobytes()==b.tobytes();item[field+'_relative_l2']=relative(a,b)
        operand_rows.append(item)
    if previous:
        previous=Path(previous);old=json.loads((previous/'report.json').read_text())
        expanded=report.get('weight_compiler',{}).get('prefix32_byte_exact',False)
        for p in (p for p in paths if p['policy'] in ((0,1,2,3,4) if expanded else (0,2,4))):
            name=p['name'];oldname=name.replace('project32','project').replace('both32','both') if p['policy'] in (1,3) else name
            q=next(q for q in old['driver']['paths'] if q['name']==oldname);matches=[]
            if p['consumed_ids']!=q['consumed_ids'] or p['turns']!=q['turns']:raise ValueError('Unaffected control history/boundary differs: '+name)
            for field in ('norm','q','k','v','attention','output','tops'):
                suffix='i32' if field=='tops' else 'f32';a=root/'raw'/f'{name}-{field}.{suffix}';b=previous/'raw'/f'{oldname}-{field}.{suffix}'
                if a.read_bytes()!=b.read_bytes():raise ValueError('Unaffected control array differs: '+a.name)
                matches.append(a.name)
            for turn in (0,1):
                for stage in ('prefill','final'):
                    for suffix in (['logits.f32','memory.bin'] if p['compact'] else ['logits.f32']):
                        namefile=f'{name}-t{turn}-{stage}-{suffix}'
                        if (root/'raw'/namefile).read_bytes()!=(previous/'raw'/f'{oldname}-t{turn}-{stage}-{suffix}').read_bytes():raise ValueError('Unaffected control boundary array differs: '+namefile)
                        matches.append(namefile)
            newcoeff=np.fromfile(root/'raw'/f'{name}-coeff.f32',dtype='<f4').reshape(p['count'],128);oldcoeff=np.fromfile(previous/'raw'/f'{oldname}-coeff.f32',dtype='<f4').reshape(q['count'],32)
            if not np.array_equal(newcoeff[:,:32],oldcoeff) or np.any(newcoeff[:,32:]!=0):raise ValueError('Unaffected coefficient padding differs')
            joins.append({'path':name,'historical_path':oldname,'history_and_boundaries_exact':True,'arrays_byte_exact':len(matches),'coefficient_prefix_and_zero_tail_exact':True})
    result={'schema':'tcm-query-memory-study-v1','report_sha256':sha(root/'report.json'),'previous_report_sha256':sha(previous/'report.json') if previous else None,'scope':'saved-only work and own-query operand account; no native model evaluation','economics':rows,'common_initial_history_operands':operand_rows,'unaffected_control_joins':joins,'semantic_certification':False}
    (root/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report_sha256':result['report_sha256'],'paths':len(rows),'unchanged_control_paths':len(joins),'unchanged_arrays':sum(j['arrays_byte_exact'] for j in joins)}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--previous');a=p.parse_args();study(a.root,a.previous)
