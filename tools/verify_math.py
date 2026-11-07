"""Verify frozen math intake custody and recompute observations without forwards.

Structural/numerical differences remain observations. No accuracy, score or
search-quality threshold is used to assign a scientific pass/fail verdict.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from inspect_program import inspect
from math_program import digest,factor_audit,polynomial_audit,read_program


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)
def same(actual,expected,message):require(actual==expected,message)

def polynomial(expr,domain):
    if isinstance(expr,str):return {(expr,):1}
    if isinstance(expr,int):return {():expr} if expr else {}
    if expr[0]=='const':return polynomial(expr[1],domain)
    terms=[polynomial(child,domain) for child in expr[1:]]
    result=terms[0]
    for term in terms[1:]:
        out=dict(result) if expr[0] in ('add','sum','sub') else {}
        if expr[0] in ('add','sum','sub'):
            sign=-1 if expr[0]=='sub' else 1
            for key,value in term.items():out[key]=out.get(key,0)+sign*value
        elif expr[0] in ('mul','prod'):
            for left,a in result.items():
                for right,b in term.items():
                    key=left+right
                    if domain=='integer-commutative':key=tuple(sorted(key))
                    out[key]=out.get(key,0)+a*b
        else:raise ValueError('unsupported exact expression instrument: '+expr[0])
        result={key:value for key,value in out.items() if value}
    return result


def verify(root):
    root=Path(root);report=load(root/'report.json');job=load(root/'job.json');result=report.get('result',{})
    require(job['state']=='succeeded' and report['status']=='completed','execution incomplete; retain raw failure evidence')
    same(report['mode'],'math','wrong experiment mode');same(load(root/'raw/report.json'),report,'raw report differs')
    for name,record in report['raw'].items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.parts[0]=='raw','unsafe raw evidence path')
        same(sha(root/path),record['sha256'],'raw hash differs: '+name)
        same((root/path).stat().st_size,record['bytes'],'raw size differs: '+name)
    for name,identity in report['executed_sources'].items():same(job['config']['sources'][name],identity,'executed C source differs')
    require((root/'payload').is_dir(),'retained staged payload missing')
    for name,identity in job['config']['sources'].items():same(sha(root/'payload'/name),identity,'staged source differs: '+name)
    custody=job['payload_custody']
    for key in ('sha256','size_bytes'):
        same(custody['declared'][key],custody['sealed'][key],'payload seal differs')
        same(custody['declared'][key],custody['executed'][key],'executed payload differs')
    maps={};symbols={};bundles=0
    for path in sorted((root/'raw').rglob('*.map.json')):
        mapping=load(path);symbol=inspect(mapping);maps[mapping['bundle']['content_sha256']]=mapping
        symbols[str(path.relative_to(root/'raw'))]={'values':len(symbol['symbols']),'instructions':len(mapping['instructions'])}
        bundle=path.with_suffix('').with_suffix('.tcm')
        if bundle.exists():read_program(bundle,mapping);bundles+=1
    calls=[json.loads(line) for line in (root/'raw/c-calls.jsonl').read_text().splitlines()]
    by_io=defaultdict(list)
    for number,call in enumerate(calls,1):
        same(call['call'],number,'C ledger order differs');run=call['run']
        require(run['bundle_digest'] in maps,'C call has no bound source map')
        mapping=maps[run['bundle_digest']]
        same(run['model_identity'],mapping['source_identity'],'C source identity differs')
        same(run['ops_per_row'],len(mapping['instructions']),'C operation count differs')
        same(run['input_rows'],call['rows'],'C row count differs');same(run['framework_fallback'],False,'C fallback reported')
        same(run['state_floats'],0,'unexpected mutable C state')
        by_io[(run['bundle_digest'],call['input_sha256'],call['output_sha256'])].append(call)
    def consumed(mapping,inputs,outputs):
        key=(mapping['bundle']['content_sha256'],hashlib.sha256(np.asarray(inputs,dtype='<f4').tobytes()).hexdigest(),
             hashlib.sha256(np.asarray(outputs,dtype='<f4').tobytes()).hexdigest())
        require(key in by_io,'raw numerical rows do not bind to a C invocation')
    raw=root/'raw';observations={'cells':[]}
    rules=load(root/'payload/inputs/exact-rules.json')
    rules={'learned'+str(row['seed']):row['rule'] for row in rules}
    for seed in (137,149,163):rules['approx'+str(seed)]=load(root/f'payload/inputs/seed{seed}-bilinear7.json')
    for item in result['cells']:
        name=item['name'];mapping=load(raw/(name+'.map.json'));data=np.load(raw/(name+'.npz'),allow_pickle=False)
        a,b,source64,rows=data['c'],data['native_fp32'],data['native_fp64'],data['inputs']
        truth=(rows[:,:4].reshape(-1,2,2)@rows[:,4:].reshape(-1,2,2)).reshape(-1,4)
        np.testing.assert_array_equal(data['truth'],truth)
        metrics={'name':name,'rows':len(rows),'source_fp32_c_max_abs':float(np.abs(a-b).max()),
            'source_fp64_c_max_abs':float(np.abs(a-source64).max()),'math_truth_c_max_abs':float(np.abs(a-truth).max()),
            'swapped_operand_changed_rows':int(np.any(data['swapped']!=a,axis=1).sum())}
        for key in metrics:
            if key not in ('name','rows'):same(metrics[key],item[key],'cell observation differs: '+key)
        consumed(mapping,rows,a);consumed(mapping,np.concatenate([rows[:,4:],rows[:,:4]],axis=1),data['swapped'])
        certificate=load(raw/(name+'-algebra.json'));same(factor_audit(rules[name]),certificate['original'],'original coefficient audit differs')
        bundle=raw/(name+'.tcm')
        if bundle.exists():same(polynomial_audit(bundle,mapping),certificate['deployed'],'deployed polynomial audit differs')
        metrics['classification']=certificate['deployed']['status'];metrics['coefficient_residuals']=certificate['deployed']['nonzero_residuals']
        current=rows[16:32,:4].copy();right=rows[16:32,4:]
        continuation=[]
        for step in item['own_output_continuation']:
            actual=np.asarray(step['c_outputs'],dtype=np.float32)
            consumed(mapping,np.concatenate([current,right],axis=1),actual);current=actual
            observed={'step':step['step'],'C_chain_bound':True,'source_c_max_abs':None}
            if 'native_outputs' in step:
                error=float(np.abs(np.asarray(step['native_outputs'],dtype=np.float32)-actual).max())
                same(error,step['max_abs_source_c'],'own-output source/C observation differs');observed['source_c_max_abs']=error
            continuation.append(observed)
        metrics['own_output_continuation']=continuation
        observations['cells'].append(metrics)
    fields=load(raw/'field-lookups.json');lookup_reads=0;unique_queries=0;differences=[]
    for epoch in fields['epochs']:
        same(epoch['calls'],sum(c['logical_reads'] for c in epoch['comparisons']),'logical read count differs')
        lookup_reads+=epoch['calls'];unique_queries+=len(epoch['comparisons'])
        for comparison in epoch['comparisons']:
            identity=comparison['bank'];state=load(raw/'field-banks'/('field-'+identity+'.records.json'))
            same(digest(state['records']),identity,'projected records identity differs')
            mapping=load(raw/'field-banks'/('field-'+identity+'.map.json'))
            same(mapping['source_identity'],identity,'projected bank binding differs')
            table=[];arity=len(comparison['query'])-3
            for record in state['records']:
                values=[record['namespace'],record['scope'],int(record['relation']),*record['roles']]
                encoded=[state['dictionary'][json.dumps(value,sort_keys=True)] for value in values]
                table.append(encoded+[-2]*(3+arity-len(encoded)))
            query=np.asarray(comparison['query'],dtype=np.float32);table=np.asarray(table,dtype=np.float32)
            selected=np.flatnonzero(np.all((query==-1)|(query==table),axis=1)).tolist()
            same(selected,comparison['selected'],'projected query result differs')
            output=np.zeros((1,len(table)),dtype=np.float32);output[0,selected]=1
            consumed(mapping,query[None,:],output)
            if not comparison['native_equal']:differences.append({'bank':identity,'query':comparison['query']})
    same(lookup_reads,result['lookup_calls'],'reported read count differs')
    same(not differences,result['lookup_all_equal'],'reported native equality differs')
    observations['projected_reads']={'logical_reads':lookup_reads,'unique_queries':unique_queries,'native_disagreements':differences}
    scores=load(raw/'scorer-rows.json');score_metrics=[]
    for row in scores:
        native=np.asarray(row['native'],dtype=np.float32);actual=np.asarray(row['c'],dtype=np.float32)
        native_order=np.argsort(-native,kind='stable');c_order=np.argsort(-actual,kind='stable');k=row['selected_count']
        same(native_order.tolist(),row['native_order'],'native score order differs');same(c_order.tolist(),row['c_order'],'C score order differs')
        same(native_order[:k].tolist(),row['native_selected'],'native selected order differs')
        same(c_order[:k].tolist(),row['c_selected'],'C selected order differs')
        consumed(load(raw/('scorer-'+row['version']+'.map.json')),row['features'],actual[:,None])
        score_metrics.append({'goal':row['goal'],'arm':row['arm'],'round':row['round'],'version':row['version'],
            'rows':len(actual),'max_abs':float(np.abs(actual-native).max()),
            'selected_order_equal':bool(np.array_equal(native_order[:k],c_order[:k])),
            'selected_set_equal':set(native_order[:k])==set(c_order[:k]),
            'native_cutoff_margin':row['native_cutoff_margin'],'c_cutoff_margin':row['c_cutoff_margin']})
    observations['scorers']={'batches':len(scores),'rows':sum(row['rows'] for row in score_metrics),
        'max_abs':max((row['max_abs'] for row in score_metrics),default=None),
        'selection_order_disagreements':[row for row in score_metrics if not row['selected_order_equal']],
        'selection_set_disagreements':[row for row in score_metrics if not row['selected_set_equal']]}
    same(observations['scorers']['max_abs'],result['scorer_max_abs'],'reported score error differs')
    same(len(observations['scorers']['selection_order_disagreements']),result['scorer_selection_order_mismatched_batches'],'reported score-order differences differ')
    pairs=[]
    def numerical_pair(name,item):
        if 'before' not in item:return
        before_map=load(raw/(name+'-before.map.json'));after_map=load(raw/(name+'-after.map.json'))
        before_source=before_map['origin']['source'];after_source=after_map['origin']['source']
        same(before_source['domain'],after_source['domain'],'expression domain differs')
        same(polynomial(before_source['expression'],before_source['domain'])==polynomial(after_source['expression'],after_source['domain']),
             item['exact_polynomial'],'exact expression polynomial observation differs')
        for label in ('before','after'):
            data=np.load(raw/(name+'-'+label+'.npz'),allow_pickle=False);mapping=load(raw/(name+'-'+label+'.map.json'))
            np.testing.assert_array_equal(data['c'],item[label]['outputs']);consumed(mapping,data['inputs'],data['c'])
            same(float(np.abs(data['native_fp32']-data['c']).max()),item[label]['source_c_max_abs'],'expression source/C error differs')
            same(len(mapping['instructions']),item[label]['operations'],'expression operation count differs')
            macs=sum(mapping['values'][op['a']]['length']*mapping['values'][op['out']]['length'] for op in mapping['instructions'] if op['kind']=='linear')
            same(macs,item[label]['linear_macs_per_row'],'expression linear work count differs')
        before=np.asarray(item['before']['outputs']);after=np.asarray(item['after']['outputs'])
        same(float(np.abs(before-after).max()),item['c_before_after_max_abs'],'before/after error differs')
        pairs.append({'name':name,'exact_polynomial':item['exact_polynomial'],
            'C_operations':[item['before']['operations'],item['after']['operations']],
            'linear_macs':[item['before']['linear_macs_per_row'],item['after']['linear_macs_per_row']],
            'C_before_after_max_abs':item['c_before_after_max_abs'],
            'source_C_max_abs':max(item[label]['source_c_max_abs'] for label in ('before','after'))})
    concepts=result['frozen_fields']['concepts'];laws=result['frozen_fields']['metaconcepts']
    for name in ('concepts','metaconcepts','scorer-v1','scorer-v2a'):
        same(sha(raw/(name+'.source.json')),job['config']['model_artifacts'][name][1],'frozen archive identity differs')
    for i,item in enumerate(concepts['cases']):numerical_pair('concept-'+str(i),item)
    for path in sorted(raw.glob('law-*-summary.json')):
        item=load(path);numerical_pair(path.name.removesuffix('-summary.json'),item)
    for item in result['search']:
        if item['status']!='error':numerical_pair(item['goal']+'-'+item['arm'],item)
    observations['compiled_consumers']=pairs
    observations['search']=[{k:item.get(k) for k in ('goal','arm','status','final_cost','applications','wall_seconds_including_C_and_extraction','error')} for item in result['search']]
    observations['fields']={'concept_count':concepts['concept_count'],'learned_rewrites':concepts.get('rewrite_count'),
        'laws':laws['rule_count'],'body_equality':concepts['bodies_equal'],
        'concept_consumer_equal':all(item['native_c_consumer_equal'] for item in concepts['cases']),
        'law_catalog_equal':all(item['native_c_equal'] for item in laws['catalogs']),
        'law_match_equal':all(item['native_c_matches_equal'] for item in laws['cases']),
        'namespace_scope_missing_controls':concepts['wrong_namespace_scope_missing_returns']}
    same(len(calls),result['C_calls'],'C call total differs')
    return {'schema':'tcm-math-verification-v1','job_id':job['job_id'],'payload_sha256':custody['executed']['sha256'],
        'raw_files_checked':len(report['raw']),'programs':len(symbols),'program_bundles_checked':bundles,'C_calls_checked':len(calls),
        'observations':observations,'mechanics_status':'payload-and-raw-hashes-source-maps-and-C-invocations-verified',
        'instrument_status':'raw-errors-rankings-projected-queries-work-counts-and-C-continuation-recomputed; binary-bound-polynomials-where-bundles-retained',
        'trend_status':'per-case-observations-preserved-without-quality-threshold',
        'terminal_status':'bounded integration; no general reasoning, unrestricted numeric-domain, or speedup claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=verify(args.directory);encoded=json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(encoded)
        print(json.dumps({key:result[key] for key in ('job_id','raw_files_checked','programs','program_bundles_checked','C_calls_checked','mechanics_status')}))
    else:print(encoded)
