"""Saved-only historical joins, objective contrasts and answer evaluation."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def study(root,previous):
    root,previous=Path(root),Path(previous);r=json.loads((root/'report.json').read_text());old=json.loads((previous/'report.json').read_text())
    recovery=json.loads((root/'analysis-recovery.json').read_text()) if (root/'analysis-recovery.json').exists() else None
    if recovery is not None:
        if recovery['original_report_sha256']!=sha(root/'report.json'):raise ValueError('Recovery report identity differs')
        r['analysis']=recovery['analysis']
    if json.loads((root/'verification.json').read_text())['report_sha256']!=sha(root/'report.json') or sha(previous/'report.json')!=r['family_provenance']['source_report_sha256']:raise ValueError('Study lineage differs')
    joins=[]
    for p in r['driver']['paths']:
        if p['task']>=3 or p['policy'] not in (0,7,12):continue
        name=p['name'];oldname=f'case{p["task"]}-'+{0:'native',7:'values-native-memory',12:'query-fit-memory'}[p['policy']];q=next(x for x in old['driver']['paths'] if x['name']==oldname)
        if p['consumed_ids']!=q['consumed_ids'] or p['turns']!=q['turns']:raise ValueError('Historical token/cache-bound endpoint differs')
        suffixes=['norm.f32','q.f32','k.f32','v.f32','attention.f32','output.f32','coeff.f32','tops.i32']+(['probabilities.f32'] if not p['compact'] else [])
        suffixes += [f't{t}-{stage}-{suffix}' for t in range(2) for stage in ('prefill','final') for suffix in (['logits.f32','memory.bin'] if p['compact'] else ['logits.f32'])]
        for suffix in suffixes:
            if (root/'raw'/f'{name}-{suffix}').read_bytes()!=(previous/'raw'/f'{oldname}-{suffix}').read_bytes():raise ValueError('Historical array differs: '+name+'-'+suffix)
        joins.append({'path':name,'historical_path':oldname,'arrays_byte_exact':len(suffixes),'token_history_and_cache_bound_endpoints_exact':True})
    family=[]
    for name in [f'program128-{k}.f32' for k in ('basis','q','k','v')]+[f'query-proposal-{i}.f32' for i in range(6)]:
        if (root/'raw'/name).read_bytes()!=(previous/'raw'/name).read_bytes():raise ValueError('Frozen program family changed')
        family.append({'name':name,'sha256':sha(root/'raw'/name),'bytes':(root/'raw'/name).stat().st_size})
    comparisons=r['analysis']['comparisons'];counts={}
    for baseline in ('ls','response'):
        rows=[]
        for x in comparisons:
            s,b=x['metrics']['selected'],x['metrics'][baseline];rows.append({'task':x['task'],'turn':x['turn'],'stage':x['stage'],'split':x['split'],'kl_improvement_nats':b['kl_nats']-s['kl_nats'],'logit_l2_improvement':b['relative_logit_l2']-s['relative_logit_l2'],'native_top_logp_improvement_nats':s['native_top_logp_delta_nats']-b['native_top_logp_delta_nats'],'all_native_histories_match':all(x['native_history_matches'].values()),'selected_baseline_ids_match':x['selected_history_matches_'+baseline]})
        counts[baseline]={'kl_improvements':sum(x['kl_improvement_nats']>0 for x in rows),'kl_regressions':sum(x['kl_improvement_nats']<0 for x in rows),'l2_improvements':sum(x['logit_l2_improvement']>0 for x in rows),'l2_regressions':sum(x['logit_l2_improvement']<0 for x in rows),'top_logp_improvements':sum(x['native_top_logp_improvement_nats']>0 for x in rows),'matching_native_history_rows':sum(x['all_native_histories_match'] for x in rows),'matching_native_history_kl_improvements':sum(x['kl_improvement_nats']>0 for x in rows if x['all_native_histories_match']),'rows':rows}
    truth=[]
    for p in r['driver']['paths']:
        if p['task']%3!=1:continue
        answers=r['texts'][p['name']];values=[re.match(r'\s*([+-]?\d+)',s) for s in answers];values=[int(x.group(1)) if x else None for x in values];expected=266 if p['task']==1 else 408;delta=13 if p['task']==1 else -19
        equation=re.match(r'\s*([+-]?\d+)\s*-\s*(\d+)\s*=\s*([+-]?\d+)',answers[1]);equation_values=[int(equation.group(i)) for i in (1,2,3)] if equation else None
        equation_correct=bool(equation_values and delta<0 and equation_values[0]==values[0] and equation_values[1]==-delta and equation_values[2]==values[0]+delta)
        leading_correct=values[0] is not None and values[1]==values[0]+delta
        truth.append({'path':p['name'],'initial_expected':expected,'initial_observed':values[0],'initial_correct':values[0]==expected,'followup_expected_from_own_answer':values[0]+delta if values[0] is not None else None,'followup_leading_integer':values[1],'followup_leading_integer_correct':leading_correct,'followup_explicit_subtraction_equation':equation_values,'followup_explicit_equation_correct':equation_correct,'followup_arithmetic_correct_on_own_answer':leading_correct or equation_correct,'explanation_truncated':p['turns'][1]['truncated'],'evaluator_scope':'Post-hoc output-only evaluator; separately records leading integer and explicit subtraction equation. Model tokens unchanged.'})
    source=[]
    for row,prior in zip(r['analysis']['source_selection_audit']['source_curves'],r['family_provenance']['source_response_proposals']):
        source.append({'proposal':row['proposal'],'dose':row['dose'],'source_native_kl_nats':row['equal_trajectory_mean_kl_nats'],'source_c_attention_loss':float(np.mean(np.square(row['source_attention_relative_error']))),'prior_cpu_source_attention_loss':prior['source_response_loss'],'prior_cpu_source_query_error':prior['source_query_relative_error'],'source_region_mean_kl_nats':row['pooled_region_mean_kl_nats']})
    result={'schema':'tcm-native-consumer-query-study-v1','report_sha256':sha(root/'report.json'),'previous_report_sha256':sha(previous/'report.json'),'historical_joins':joins,'historical_arrays_exact':sum(x['arrays_byte_exact'] for x in joins),'frozen_family_arrays_exact':family,'source_objective_contrast':source,'target_gains':counts,'arithmetic_outcomes':truth,'history_comparisons':r['analysis']['history_comparisons'],'scope':'Saved-only joins, fixed-family objective comparison, native-distribution fidelity and output-only truth. No target fitting or new model call.','terminal_status':'not-requested'}
    (root/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'historical_paths':len(joins),'historical_arrays':result['historical_arrays_exact'],'family_arrays':len(family),'target_counts':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in counts.items()}}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--previous',default=str(Path(__file__).resolve().parent.parent/'query_response/runs/job-b141eadfe952'));a=p.parse_args();study(a.root,a.previous)
