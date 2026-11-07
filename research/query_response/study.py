"""Saved-only historical endpoint joins and marginal work account."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def study(root,previous):
    root,previous=Path(root),Path(previous);report=json.loads((root/'report.json').read_text());old=json.loads((previous/'report.json').read_text())
    if json.loads((root/'verification.json').read_text())['report_sha256']!=sha(root/'report.json'):raise ValueError('New verification identity differs')
    if sha(previous/'report.json')!=report['program_provenance']['source_report_sha256']:raise ValueError('Predecessor report identity differs')
    labels={0:'native',5:'query-fit',7:'query-fit-memory'};joins=[]
    for p in (p for p in report['driver']['paths'] if p['policy'] in labels and p['task']<3):
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
        name=f'program128-{kind}.f32';a,b=root/'raw'/name,previous/'raw'/('query-fit.f32' if kind=='q' else name)
        if a.read_bytes()!=b.read_bytes():raise ValueError('Historical bank differs')
        bank.append({'file':name,'sha256':sha(a),'bytes':a.stat().st_size})
    economics=[]
    for p in report['driver']['paths']:
        selected=(196608 if p['factor'] else 0)+(196608 if p['q_factor'] else 2359296)+(32768 if p['k_factor'] else 393216)+393216
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
    paired=[]
    for task in range(6):
        oldname,newname=f'case{task}-values-native-memory',f'case{task}-query-fit-memory';a,b=[next(p for p in report['driver']['paths'] if p['name']==name) for name in (oldname,newname)]
        common=0
        for ids in zip(a['consumed_ids'],b['consumed_ids']):
            if ids[0]!=ids[1]:break
            common+=1
        matched={}
        for suffix,width in [('norm.f32',1536),('k.f32',256),('v.f32',256),('coeff.f32',128)]:
            aa=np.fromfile(root/'raw'/f'{oldname}-{suffix}',dtype='<f4').reshape(-1,width);bb=np.fromfile(root/'raw'/f'{newname}-{suffix}',dtype='<f4').reshape(-1,width)
            matched[suffix]=bool(np.array_equal(aa[:common],bb[:common]))
        snapshots=[]
        for t in range(2):
            for stage in ('prefill','final'):
                stopa=a['turns'][t][stage]['position'];stopb=b['turns'][t][stage]['position'];same_clock=stopa==stopb
                suffix=f't{t}-{stage}-memory.bin';same=(root/'raw'/f'{oldname}-{suffix}').read_bytes()==(root/'raw'/f'{newname}-{suffix}').read_bytes()
                snapshots.append({'turn':t,'stage':stage,'same_clock':same_clock,'snapshot_bytes_exact':same,'matched_consumed_history':a['consumed_ids'][:stopa]==b['consumed_ids'][:stopb]})
        paired.append({'task':task,'old':oldname,'new':newname,'consumed_ids_exact':a['consumed_ids']==b['consumed_ids'],'common_consumed_prefix':common,'selected_operand_prefix_exact':matched,'snapshots':snapshots,'scope':'Selected layer only; upper caches and writes remain policy-specific.'})

    truth=[]
    for p in report['driver']['paths']:
        task=p['task'];answers=report['texts'][p['name']]
        if task%3==1:
            values=[re.match(r'\s*([+-]?\d+)',x) for x in answers];values=[int(x.group(1)) if x else None for x in values];initial_expected=266 if task==1 else 414;delta=13 if task==1 else 27
            truth.append({'path':p['name'],'kind':'arithmetic-output-only-evaluator','initial_expected':initial_expected,'initial_observed':values[0],'initial_correct':values[0]==initial_expected,'followup_expected_from_own_first_answer':values[0]+delta if values[0] is not None else None,'followup_observed':values[1],'followup_correct_on_own_answer':values[0] is not None and values[1]==values[0]+delta,'explanation_truncated':p['turns'][1]['truncated'],'model_output_unmodified':True})
    distributions=[]
    def log_softmax(z):
        shifted=z-z.max();return shifted-np.log(np.exp(shifted).sum())
    for row in report['analysis']['factorials']:
        task,turn,stage=row['task'],row['turn'],row['stage'];labels=('native','values-native-memory','query-fit-memory')
        logits=[np.fromfile(root/'raw'/f'case{task}-{label}-t{turn}-{stage}-logits.f32',dtype='<f4').astype(float) for label in labels];logs=[log_softmax(z) for z in logits];probability=np.exp(logs[0]);native_top=int(np.argmax(logits[0]))
        kl=[float(probability@(logs[0]-p)) for p in logs[1:]]
        distributions.append({'task':task,'turn':turn,'stage':stage,'old_new_consumed_ids_exact':row['old_new_compact_same_consumed_ids'],'both_histories_match_native':row['consumed_history_matches_native']['values-native-memory'] and row['consumed_history_matches_native']['query-fit-memory'],'native_to_old_kl_nats':kl[0],'native_to_fitted_kl_nats':kl[1],'kl_improvement_nats':kl[0]-kl[1],'native_top_log_probability_improvement_nats':float(logs[2][native_top]-logs[1][native_top]),'scope':'Native-distribution fidelity, not task truth. After native history divergence this is an operational-policy comparison.'})
    result={'schema':'tcm-query-response-study-v1','scope':'saved-only exact joins, output-only arithmetic evaluation, native-distribution fidelity, operational initial error decomposition and nominal work; no model calls or speedup claim','report_sha256':sha(root/'report.json'),'previous_report_sha256':sha(previous/'report.json'),'historical_joins':joins,'historical_arrays_exact':sum(x['byte_exact_arrays'] for x in joins),'source_bank_exact':bank,'new_calibration_model_advances':0,'new_query_image_compiled':True,'source_acquisition_and_original_image_compilation_paid_historically':True,'economics':economics,'initial_error_decomposition':initial,'old_new_selected_state_pairs':paired,'arithmetic_outcomes':truth,'readout_distributions':distributions,'terminal_status':'not-requested'}
    (root/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report_sha256':result['report_sha256'],'historical_paths_exact':len(joins),'historical_arrays_exact':result['historical_arrays_exact'],'bank_arrays_exact':len(bank)}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--previous',default=str(Path(__file__).resolve().parent.parent/'query_program/runs/job-bd80dd16c1d5'));a=p.parse_args();study(a.root,a.previous)
