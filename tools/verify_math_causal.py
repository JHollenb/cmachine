"""Offline custody, cut/receipt, choice, C-consumption and signed-outcome audit.

No learned model forwards or search reruns. Quality outcomes are observations;
only evidence identity and internal consistency can fail this verifier.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from math_program import digest,read_program
from verify_math import load,sha,require,same,polynomial


def described_value(value):
    """Decode only the scalar/container subset used by these sealed cuts."""
    kind=value['kind']
    if kind=='scalar':return value['value']
    if kind in ('list','tuple'):
        items=[described_value(v) for v in value['items']]
        return tuple(items) if kind=='tuple' else items
    if kind=='mapping':return {described_value(v['key']):described_value(v['value']) for v in value['items']}
    raise ValueError('unsupported typed cut value: '+kind)


def graph_value(value):
    if isinstance(value,list):return [graph_value(v) for v in value]
    if isinstance(value,dict):
        if 'dict' in value:return {graph_value(k):graph_value(v) for k,v in value['dict']}
        if 'tuple' in value:return tuple(graph_value(v) for v in value['tuple'])
        raise ValueError('unsupported graph metadata witness')
    return value


def expression_work(expression,domain):
    """Independent structural CSE and the declared scalar/matrix cost model."""
    seen=set();ops=Counter()
    def visit(expr):
        if isinstance(expr,(str,int)) or expr[0]=='const':return
        key=digest(expr)
        if key in seen:return
        seen.add(key);op=expr[0];children=expr[1:]
        if op in ('sum','add','sub'):ops['add']+=len(children)-1
        elif op in ('prod','mul'):
            nonconstant=sum(not (isinstance(c,int) or isinstance(c,list) and c[0]=='const') for c in children)
            ops['mul']+=max(nonconstant-1,0)
        else:raise ValueError('unsupported arithmetic work instrument: '+op)
        for child in children:visit(child)
    visit(expression)
    weights={'add':1,'mul':1} if domain=='integer-commutative' else {'add':4,'mul':12}
    return sum(weights[k]*v for k,v in ops.items()),dict(+ops)


def verify(root):
    root=Path(root);raw=root/'raw';report=load(root/'report.json');job=load(root/'job.json');result=report['result']
    same(report['status'],'completed','worker incomplete');same(job['state'],'succeeded','fleet execution incomplete')
    same(report['mode'],'math-causal','wrong panel');same(load(raw/'report.json'),report,'raw report differs')
    for path,record in report['raw'].items():
        p=Path(path);require(not p.is_absolute() and '..' not in p.parts and p.parts[0]=='raw','unsafe evidence path')
        same(sha(root/p),record['sha256'],'raw identity differs: '+path);same((root/p).stat().st_size,record['bytes'],'raw size differs')
    for path,identity in job['config']['sources'].items():same(sha(root/'payload'/path),identity,'payload source differs: '+path)
    sys.path.insert(0,str((root/'payload/public').resolve()))
    import saturn_pub
    sdk=Path(saturn_pub.__file__).parent
    for name in ('__init__.py','core.py','contracts.py','values.py'):
        same(sha(sdk/name),job['config']['sources']['public/saturn_pub/'+name],'verification SDK differs from executed SDK')
    for path,identity in report['executed_sources'].items():same(identity,job['config']['sources'][path],'executed C source differs')
    for field in ('sha256','size_bytes'):
        custody=job['payload_custody'];same(custody['declared'][field],custody['sealed'][field],'payload seal differs')
        same(custody['declared'][field],custody['executed'][field],'executed payload differs')
    maps={};files={}
    for path in raw.rglob('*.map.json'):
        mapping=load(path);bundle=path.with_suffix('').with_suffix('.tcm');read_program(bundle,mapping)
        maps[mapping['bundle']['content_sha256']]=mapping;files[sha(bundle)]=mapping
    calls=[json.loads(line) for line in (raw/'c-calls.jsonl').read_text().splitlines()]
    by_io=set()
    for i,call in enumerate(calls,1):
        same(call['call'],i,'physical C ledger sequence differs');record=call['run'];mapping=maps[record['bundle_digest']]
        same(record['model_identity'],mapping['source_identity'],'C source identity differs')
        same(record['framework_fallback'],False,'native fallback');same(record['state_floats'],0,'stateful memo authority')
        same(record['input_rows'],call['rows'],'C geometry differs');same(record['ops_per_row'],len(mapping['instructions']),'C work differs')
        by_io.add((record['bundle_digest'],call['input_sha256'],call['output_sha256']))
    def consumed(mapping,inputs,outputs):
        key=(mapping['bundle']['content_sha256'],hashlib.sha256(np.asarray(inputs,dtype='<f4').tobytes()).hexdigest(),
             hashlib.sha256(np.asarray(outputs,dtype='<f4').tobytes()).hexdigest())
        require(key in by_io,'consumer arrays have no physical C invocation')
    memo=load(raw/'causal-c-memo.json');same(memo['physical_calls'],len(calls),'memo physical count differs')
    same(len(memo['entries']),len(calls),'memo did not bind every physical call')
    same(sum(e['calls'] for e in memo['entries']),memo['logical_calls'],'logical C count differs')
    require(len({e['physical_call'] for e in memo['entries']})==len(calls),'duplicate physical call binding')
    for entry in memo['entries']:
        same(sha(root/entry['program']),entry['bundle_sha256'],'memo program differs')
        key=digest({'binary':memo['binary_sha256'],'profile':'cpu-fp32-ordered-no-fallback',
            'bundle':entry['bundle_sha256'],'shape':entry['shape'],'input':entry['input_sha256'],'zero':entry['zero']})
        same(entry['key'],key,'memo key differs');call=calls[entry['physical_call']-1]
        same(entry['input_sha256'],call['input_sha256'],'memo input differs');same(entry['output_sha256'],call['output_sha256'],'memo output differs')
        same(files[entry['bundle_sha256']]['bundle']['content_sha256'],call['run']['bundle_digest'],'memo bundle differs')
    witnesses={}
    for path in (raw/'causal-witnesses').glob('*.json'):
        body=load(path);same(digest(body),path.stem,'witness seal differs');witnesses[path.stem]=body
        if body.get('schema')=='saturn-pub-statecut-v2':raise ValueError('unwrapped cut witness')
        for receipt in body.get('receipts',[]):
            same(digest({k:v for k,v in receipt.items() if k!='fingerprint'}),receipt['fingerprint'],'Saturn receipt seal differs')
        if 'manifest' in body:
            from saturn_pub.values import describe
            same(digest(body['manifest']),body['fingerprint'],'Saturn cut seal differs')
            same(describe(body['payload']),body['manifest']['payload'],'Saturn payload differs')
            same(body['payload']['context'],body['manifest']['model_identity'],'cut model differs')
    cuts_by_fingerprint={body['fingerprint']:body for body in witnesses.values() if 'manifest' in body}
    def receipts_close(row,end):
        from saturn_pub import StateCut
        body=witnesses[row['receipts']]
        initial=(witnesses[row['cut']] if 'cut' in row else cuts_by_fingerprint[body['cut']])
        payload=initial['payload'];manifest=initial['manifest'];parent=manifest['parent']
        point=described_value(manifest['execution_point']);surface=described_value(manifest['surface'])
        def fingerprint(state,parent,point,surface):
            return StateCut(manifest['model_identity'],manifest['execution'],point['phase'],parent,state,
                execution_point=point,surface=surface).fingerprint
        same(fingerprint(payload,parent,point,surface),initial['fingerprint'],'initial cut does not close')
        for receipt in body['receipts']:
            before=fingerprint(payload,parent,point,surface)
            same(receipt['parent'],before,'receipt parent does not bind current cut')
            same(receipt['model_identity'],manifest['model_identity'],'receipt model differs')
            same(receipt['execution'],manifest['execution'],'receipt execution differs')
            if receipt['operation']=='restore':
                same(receipt['result'],initial['fingerprint'],'restore did not reinstall cut');continue
            if receipt['operation']=='apply':
                same(receipt['act']['writes'],['choice'],'Act writes beyond choice')
                same(receipt['act']['invalidates'],[],'Act invalidates beyond choice')
                writes=described_value(receipt['writes']);same(set(writes),{'choice'},'Act physical write differs')
                payload={**payload,**writes}
            elif receipt['operation']=='continue':
                transitions=receipt['transitions'];same(len(transitions),1,'unexpected continuation geometry')
                same(transitions[0]['before'],point,'consumer entry point differs')
                same(transitions[0]['native_operation']['writes'],['prefix','frontier','choice','endpoint'],'consumer footprint differs')
                entry={'frontier':payload['frontier'],'choice':payload['choice']}
                same(end['decisions'][:len(payload['prefix'])+1],payload['prefix']+[entry],'continuation did not consume edited choice')
                payload={'context':payload['context'],'prefix':end['decisions'],'frontier':None,'choice':[],'endpoint':row['endpoint']}
                point=transitions[0]['after']
                surface={**surface,'slots':[{**s,'writable':False} for s in surface['slots']]}
            else:raise ValueError('unexpected Saturn receipt operation')
            parent=before;same(fingerprint(payload,parent,point,surface),receipt['result'],'receipt result does not bind declared state')
        same(payload['endpoint'],row['endpoint'],'receipt chain did not reach endpoint')
    scores=load(raw/'causal-scorer-audits.json');score_arrays={};score_max=0.
    for identity,row in scores.items():
        with np.load(raw/('causal-scores-'+identity+'.npz')) as data:
            inputs,native,actual=(data[k].copy() for k in ('inputs','native','c'))
        same(digest({'version':row['version'],'input_sha256':hashlib.sha256(inputs.tobytes()).hexdigest(),'shape':list(inputs.shape)}),identity,'score input identity differs')
        same(np.argsort(-actual,kind='stable').tolist(),row['c_order'],'C score ranking differs')
        same(np.argsort(-native,kind='stable').tolist(),row['native_order'],'source score ranking differs')
        same(float(np.abs(native-actual).max()),row['source_c_max_abs'],'score residual differs')
        consumed(load(raw/('scorer-'+row['version']+'.map.json')),inputs,actual[:,None]);score_max=max(score_max,row['source_c_max_abs'])
        score_arrays[(row['version'],hashlib.sha256(inputs.tobytes()).hexdigest(),inputs.shape)]=row
    v1names=load(raw/'scorer-v1.map.json')['origin']['source']['feature_order']
    ranking_changes=[]
    for identity,body in witnesses.items():
        if 'candidate_ids' not in body:continue
        epoch=witnesses[body['epoch']];shape=tuple(epoch['features']['shape'])
        path=raw/'causal-witnesses'/(epoch['features']['sha256']+'.f32')
        same(sha(path),epoch['features']['sha256'],'raw feature bytes differ')
        X=np.fromfile(path,dtype='<f4').reshape(shape);require(np.isfinite(X).all(),'nonfinite features')
        same(body['candidate_ids'],[digest({'epoch':body['epoch'],'ordinal':i}) for i in range(len(X))],'candidate addresses differ')
        for key in ('context','round','remaining','prefix_sha256'):same(body[key],epoch[key],'frontier/epoch differs')
        for version,proposal in body['proposals'].items():
            names=load(raw/('scorer-'+version+'.map.json'))['origin']['source']['feature_order']
            inputs=np.column_stack([X[:,v1names.index(n)] if n in v1names else np.zeros(len(X),dtype=np.float32) for n in names]).astype('<f4')
            row=score_arrays[(version,hashlib.sha256(inputs.tobytes()).hexdigest(),inputs.shape)]
            same(proposal,[body['candidate_ids'][i] for i in row['c_order'][:body['count']]],'proposal does not consume C scores')
            k=body['count'];native=row['native_order'][:k];actual=row['c_order'][:k]
            if native!=actual:ranking_changes.append({'frontier':identity,'version':version,'round':body['round'],
                'same_selected_set':set(native)==set(actual),'native':native,'c':actual})
    fields=load(raw/'causal-field-lookups.json');banks={}
    for row in fields:
        identity=row['bank']
        if identity not in banks:
            state=load(raw/'field-banks'/('field-'+identity+'.records.json'));same(digest(state['records']),identity,'Field records differ')
            mapping=load(raw/'field-banks'/('field-'+identity+'.map.json'));arity=len(row['query'])-3;table=[]
            for record in state['records']:
                values=[record['namespace'],record['scope'],int(record['relation']),*record['roles']]
                encoded=[state['dictionary'][json.dumps(v,sort_keys=True)] for v in values];table.append(encoded+[-2]*(3+arity-len(encoded)))
            table=np.asarray(table,dtype='<f4');weights=read_program(raw/'field-banks'/('field-'+identity+'.tcm'),mapping)
            np.testing.assert_array_equal(weights['records'],table);banks[identity]=(table,mapping)
        table,mapping=banks[identity];query=np.asarray(row['query'],dtype='<f4')
        selected=np.flatnonzero(np.all((query==-1)|(query==table),axis=1)).tolist();same(selected,row['selected'],'C projected results differ')
        mask=np.zeros((1,len(table)),dtype='<f4');mask[0,selected]=1;consumed(mapping,query[None,:],mask)
    numerics=load(raw/'causal-numerics.json');numeric_maps={}
    for identity,row in numerics.items():
        path=root/row['program'];mapping=load(path.with_suffix('.map.json'))
        numeric_maps[identity]=mapping
        source=mapping['origin']['source']
        same(digest({'expression':source['expression'],'domain':source['domain'],'variables':source['variable_order']}),identity,'numeric expression identity differs')
        same(len(mapping['instructions']),row['operations'],'numerical work differs from binary schedule')
        with np.load(path.with_suffix('.npz')) as data:
            consumed(mapping,data['inputs'],data['c']);np.testing.assert_array_equal(data['c'],row['outputs'])
            same(float(np.abs(data['native']-data['c']).max()),row['source_c_max_abs'],'numeric source/C residual differs')
    observations=[];endpoints=0;controls=0;branch_errors=[]
    identity=digest({'sources':job['config']['sources'],
        'scorers':{v:load(raw/('scorer-'+v+'.map.json'))['bundle']['content_sha256'] for v in ('v1','v2a')},
        'C_executable':memo['binary_sha256'],'python_hash_seed':job['config']['python_hash_seed']})
    caps={'rounds':6,'per_round':8,'selections':48,'nodes':768,'matches':1024,
          'search_seconds':120,'extract_passes':1,'extract_seconds':.5}
    for path in sorted(raw.glob('causal-*-*.json')):
        body=load(path)
        if type(body) is not dict or 'baselines' not in body:continue
        original=body['expression'];domain=body['domain'];baseline=body['baselines']['v2a'];baseline_cost=baseline['cost']
        same(body['context'],digest({'executor':identity,'expression':original,'domain':domain,'caps':caps}),'context source/cap identity differs')
        def endpoint(end):
            nonlocal endpoints
            endpoints+=1;prefix=[];used=0
            for decision in end['decisions']:
                frontier=witnesses[decision['frontier']];same(frontier['prefix_sha256'],digest(prefix),'endpoint prefix differs')
                same(frontier['remaining'],48-used,'remaining selection budget differs')
                choice=decision['choice'];same(len(choice),frontier['count'],'choice cardinality differs')
                require(len(set(choice))==len(choice) and set(choice)<=set(frontier['candidate_ids']),'invalid choice')
                used+=len(choice);prefix.append(decision)
            same(used,end['selection_budget_used'],'selected work differs');require(used<=48,'budget exceeded')
            same(sum(t['applied'] for t in end['trace']),end['applications'],'applied work differs')
            exact=polynomial(original,domain)==polynomial(end['expression'],domain);same(exact,end['exact_polynomial'],'polynomial result differs')
            cost,ops=expression_work(end['expression'],domain)
            same(cost,end['cost'],'reported DAG cost differs from expression');same(ops,dict(+Counter(end['ops'])),'DAG operations differ')
            numerical=end['numerical'];target=numerics[numerical['program']];reference=numerics[numerical['reference']]
            names=set()
            def variables(expr):
                if isinstance(expr,str):names.add(expr)
                elif isinstance(expr,list) and expr[0]!='const':
                    for child in expr[1:]:variables(child)
            variables(original)
            for identity,expression in ((numerical['program'],end['expression']),(numerical['reference'],original)):
                source=numeric_maps[identity]['origin']['source']
                same(source['expression'],expression,'emitted program is not the declared endpoint expression')
                same(source['domain'],domain,'emitted domain differs');same(source['variable_order'],sorted(names),'emitted input ports differ')
            same(numerical['operations'],target['operations'],'numerical operation count differs')
            same(numerical['C_reference_max_abs'],float(np.abs(np.asarray(target['outputs'])-np.asarray(reference['outputs'])).max()),'C consumer residual differs')
        for end in body['baselines'].values():endpoint(end)
        for key in ('batch_swaps','controls','single_exchanges','doses','reciprocal'):
            for row in body[key]:
                if row.get('status','completed')!='completed':
                    if row.get('status')=='error':branch_errors.append({'goal':body['goal'],**row})
                    continue
                end=witnesses[row['endpoint']];endpoint(end);same(row['cost'],end['cost'],'branch cost differs')
                receipts_close(row,end)
                if key!='reciprocal':same(row['cost_reduction'],baseline_cost-row['cost'],'signed effect differs')
                if 'timing' in row:
                    timing=row['timing'];same(timing['endpoint'],row['endpoint'],'timing belongs to different endpoint')
                    same(timing['search_clock_proven_unhit'],timing['search_wall_seconds']<caps['search_seconds'] and end['search_status']!='time-cap','search clock qualification differs')
                    same(timing['extraction_clock_proven_unhit'],timing['extraction_wall_seconds']<caps['extract_seconds']/2,'extract clock qualification differs')
                    same(row['causal_qualified'],end['exact_polynomial'] and timing['search_clock_proven_unhit'] and timing['extraction_clock_proven_unhit'],'repair qualification differs')
                if key=='controls' and row['label'] in ('untouched','sham','A-B-A'):
                    same(end,baseline,'restoration or sham does not close');controls+=1
        observations.append({'goal':body['goal'],'baseline_costs':{k:v['cost'] for k,v in body['baselines'].items()},
            'batch_swaps':[{'round':r['round'],'cost':r.get('cost'),'status':r['status']} for r in body['batch_swaps']],
            'single_exchange_costs':dict(sorted((str(cost),sum(r.get('cost')==cost for r in body['single_exchanges']))
                for cost in {r.get('cost') for r in body['single_exchanges']} if cost is not None)),
            'batch_summary':body['batch_summary'],'exchange_summary':body['exchange_summary']})
        observations[-1]['alternative_membership_controls']=[{'label':r['label'],'cost':r.get('cost'),
            'cost_reduction':r.get('cost_reduction'),'status':r['status']} for r in body['controls'] if r['label'].startswith('alternative')]
        if body['single_exchanges']:
            round_=body['single_exchanges'][0]['round'];frontier=witnesses[next(d['frontier'] for d in baseline['decisions']
                if witnesses[d['frontier']]['round']==round_)]
            recipient=frontier['proposals']['v2a'];donor=frontier['proposals']['v1'];metas=graph_value(witnesses[frontier['epoch']]['metas'])
            expected={(slot,new) for slot,old in enumerate(recipient) if old not in donor for new in donor if new not in recipient}
            same({(r['slot'],r['inserted']) for r in body['single_exchanges']},expected,'single exchange panel is not exhaustive over batch differences')
            groups=[]
            for candidate in dict.fromkeys(r['inserted'] for r in body['single_exchanges']):
                ordinal=frontier['candidate_ids'].index(candidate);meta=metas[ordinal]
                rows=[r for r in body['single_exchanges'] if r['inserted']==candidate]
                for r in rows:
                    choice=recipient.copy();choice[r['slot']]=candidate;same(r['choice'],choice,'single exchange changed other slots')
                groups.append({'ordinal':ordinal,'rule':meta['rule'],'group':meta.get('group'),
                    'cost_counts':dict(Counter(str(r.get('cost')) for r in rows))})
            observations[-1]['exchanges_by_inserted_candidate']=groups
            reciprocal=next((r for r in body['reciprocal'] if r.get('status','completed')=='completed'),None)
            swap=next(r for r in body['batch_swaps'] if r['round']==round_)
            if reciprocal:
                reciprocal_cut=witnesses[reciprocal['cut']]['payload']
                observations[-1]['choice_suffix_cross']={'round':round_,
                    'same_parent_frontier':reciprocal_cut['frontier']==next(d['frontier'] for d in baseline['decisions'] if witnesses[d['frontier']]['round']==round_),
                    'v1_choice_v1_suffix':body['baselines']['v1']['cost'],'v1_choice_v2a_suffix':swap['cost'],
                    'v2a_choice_v1_suffix':reciprocal['cost'],'v2a_choice_v2a_suffix':baseline_cost}
    audits=load(raw/'causal-replay-audits.json');attempts=[a for e in audits for a in e['attempts']]
    confounded=sum(not a['search_clock_proven_unhit'] or not a.get('extraction_clock_proven_unhit',True) for a in attempts)
    same(confounded,result['potentially_clock_confounded_attempts'],'clock classification differs')
    same(sum(e['replays'] for e in audits),result['replays'],'replay count differs')
    same(memo['logical_calls'],result['logical_C_calls'],'reported logical C calls differ');same(len(calls),result['physical_C_calls'],'reported physical C calls differ')
    return {'schema':'tcm-math-causal-verification-v1','mechanics_status':'verified',
        'instrument_status':'clock-confounded-rows-retained' if confounded else 'applicable-at-recorded-boundaries',
        'terminal_status':'bounded causal observations only; no promotion or generalization',
        'raw_files':len(report['raw']),'raw_bytes':sum(r['bytes'] for r in report['raw'].values()),
        'programs':len(maps),'witnesses':len(witnesses),'verified_endpoint_consumptions':endpoints,
        'restoration_controls':controls,'branch_errors':branch_errors,'potentially_clock_confounded_attempts':confounded,
        'physical_C_calls':len(calls),'logical_C_calls':memo['logical_calls'],'unique_field_queries':len(fields),
        'field_logical_reads':sum(r['logical_reads'] for r in fields),'field_native_disagreements':sum(not r['native_equal'] for r in fields),
        'scorer_batches':len(scores),'scorer_max_abs':score_max,'selection_order_changes':ranking_changes,
        'selection_set_changes':sum(not r['same_selected_set'] for r in ranking_changes),'observations':observations}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify(args.result);text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')
