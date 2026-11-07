"""Offline capability evidence audit. No learned forwards or search reruns.

Recompute selection from discovery arrays, bind physically pruned parameters,
C invocation arrays, Saturn cuts, proposals and native endpoint costs. Quality
differences remain signed observations, never scientific pass/fail gates.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
from math_program import digest,read_program
from verify_math import load,sha,require,same,polynomial
from verify_math_causal import described_value,expression_work


def array_sha(a):return hashlib.sha256(np.asarray(a,dtype='<f4').tobytes()).hexdigest()


def verify(root):
    root=Path(root);raw=root/'raw';job=load(root/'job.json');report=load(root/'report.json');result=report['result']
    same(job['state'],'succeeded','fleet execution incomplete');same(report['status'],'completed','worker incomplete')
    same(report['mode'],'math-capability','wrong experiment mode');same(load(raw/'report.json'),report,'raw report differs')
    for name,record in report['raw'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts and path.parts[0]=='raw','unsafe evidence path')
        same(sha(root/path),record['sha256'],'raw identity differs: '+name)
        same((root/path).stat().st_size,record['bytes'],'raw size differs')
    for name,identity in job['config']['sources'].items():same(sha(root/'payload'/name),identity,'payload source differs: '+name)
    for name,identity in report['executed_sources'].items():same(identity,job['config']['sources'][name],'executed C source differs')
    for key in ('sha256','size_bytes'):
        custody=job['payload_custody'];same(custody['declared'][key],custody['sealed'][key],'payload seal differs')
        same(custody['declared'][key],custody['executed'][key],'executed payload differs')
    sys.path.insert(0,str((root/'payload/public').resolve()))
    import saturn_pub
    from saturn_pub import StateCut
    from saturn_pub.values import describe
    for name in ('__init__.py','core.py','contracts.py','values.py'):
        same(sha(Path(saturn_pub.__file__).parent/name),job['config']['sources']['public/saturn_pub/'+name],'verification SDK differs')
    maps={};banks={};files={}
    for path in raw.glob('*.map.json'):
        mapping=load(path);bundle=path.with_suffix('').with_suffix('.tcm');bank=read_program(bundle,mapping)
        maps[str(bundle.relative_to(root))]=mapping;banks[str(bundle.relative_to(root))]=bank
        files[sha(bundle)]=mapping
    calls=[json.loads(line) for line in (raw/'c-calls.jsonl').read_text().splitlines()];by_io=set()
    by_digest={m['bundle']['content_sha256']:m for m in maps.values()}
    for i,call in enumerate(calls,1):
        same(call['call'],i,'physical C ledger order differs');run=call['run'];mapping=by_digest[run['bundle_digest']]
        same(run['model_identity'],mapping['source_identity'],'C source differs');same(run['framework_fallback'],False,'framework fallback')
        same(run['input_rows'],call['rows'],'C geometry differs');same(run['state_floats'],0,'stateful C program')
        same(run['ops_per_row'],len(mapping['instructions']),'C instruction work differs')
        by_io.add((run['bundle_digest'],call['input_sha256'],call['output_sha256']))
    def consumed(program,inputs,outputs):
        require((maps[program]['bundle']['content_sha256'],array_sha(inputs),array_sha(outputs)) in by_io,'arrays have no physical C invocation')
    memo=load(raw/'causal-c-memo.json');same(memo['physical_calls'],len(calls),'memo count differs')
    same(len(memo['entries']),len(calls),'memo physical binding incomplete')
    same(sum(e['calls'] for e in memo['entries']),memo['logical_calls'],'logical C count differs')
    same(len({e['physical_call'] for e in memo['entries']}),len(calls),'duplicate memo call binding')
    for entry in memo['entries']:
        same(sha(root/entry['program']),entry['bundle_sha256'],'memo bundle differs')
        same(entry['key'],digest({'binary':memo['binary_sha256'],'profile':'cpu-fp32-ordered-no-fallback',
            'bundle':entry['bundle_sha256'],'shape':entry['shape'],'input':entry['input_sha256'],'zero':entry['zero']}),'memo key differs')
        call=calls[entry['physical_call']-1]
        same(entry['input_sha256'],call['input_sha256'],'memo input differs');same(entry['output_sha256'],call['output_sha256'],'memo output differs')
        same(files[entry['bundle_sha256']]['bundle']['content_sha256'],call['run']['bundle_digest'],'memo program differs')
    full={}
    for version in ('v1','v2a'):
        source=load(raw/('scorer-'+version+'.source.json'));program='raw/scorer-'+version+'.tcm'
        full[version]={k:np.asarray(v,dtype='<f4') for k,v in source.items()}
        for key,array in full[version].items():np.testing.assert_array_equal(banks[program][key],array)
        require(np.all(full[version]['std']>0),'invalid normalization')
        record=maps[program]['origin']['source']
        same(record['constructor_sha256'],job['config']['sources']['sources/'+version+'/train.py'],'scorer constructor differs')
        same(record['feature_source_sha256'],job['config']['sources']['sources/'+version+'/scheduler.py'],'feature source differs')
    scores=load(raw/'capability-scorer-audits.json');score_arrays={};score_max=0.
    for identity,row in scores.items():
        with np.load(raw/('capability-score-'+identity+'.npz')) as data:
            inputs,native,actual=(data[k].copy() for k in ('inputs','native','c'))
        same(digest({'program_sha256':sha(root/row['program']),'inputs_sha256':array_sha(inputs),'shape':list(inputs.shape)}),identity,'score input key differs')
        same(np.argsort(-actual,kind='stable').tolist(),row['c_order'],'C score order differs')
        same(np.argsort(-native,kind='stable').tolist(),row['native_order'],'native score order differs')
        same(float(np.abs(native-actual).max()),row['source_c_max_abs'],'score residual differs')
        consumed(row['program'],inputs,actual[:,None]);score_arrays[identity]=(inputs,native,actual)
        score_max=max(score_max,row['source_c_max_abs'])
    corpus=load(raw/'capability-corpus.json');discovery=load(raw/'capability-discovery.json');seal=load(raw/'capability-seal.json')
    discovery_ids={r['id'] for r in corpus['discovery']};cases={r['id']:r for r in corpus['test']}
    same(seal['discovery_ids'],[r['id'] for r in corpus['discovery']],'discovery split differs')
    same(seal['test_ids'],list(cases),'test split differs');require(not discovery_ids.intersection(cases),'split IDs overlap')
    same(seal['manalysis_source_sha256'],job['config']['sources']['sources/conditional_family.py'],'coalition implementation differs')
    data=np.load(raw/'capability-discovery.npz');X=data['inputs'];H=data['hidden_C'];Y=data['full_C'];discovery_rows=len(X)
    for key,array in banks['raw/capability-hidden.tcm'].items():np.testing.assert_array_equal(array,full['v1'][key.removeprefix('model.')])
    consumed('raw/capability-hidden.tcm',X,H);consumed('raw/scorer-v1.tcm',X,Y[:,None])
    contributions=H.astype(np.float64)*full['v1']['net.2.weight'].astype(np.float64)[0][None,:]
    centered=contributions.copy();target=Y.astype(np.float64);weights=np.zeros(len(X));families=[];offset=0
    for row in discovery['boundaries']:
        require(row['goal'] in discovery_ids,'test row entered discovery');same(row['offset'],offset,'discovery row offset differs')
        n=row['rows'];sl=slice(offset,offset+n);centered[sl]-=centered[sl].mean(axis=0);target[sl]-=target[sl].mean()
        weights[sl]=1/n;families.extend([row['family']]*n);offset+=n
        np.testing.assert_array_equal(score_arrays[row['score']][0],X[sl])
    same(offset,len(X),'discovery coverage differs');np.testing.assert_array_equal(data['centered_contributions'],centered)
    np.testing.assert_array_equal(data['centered_target'],target);np.testing.assert_array_equal(data['weights'],weights)
    np.testing.assert_array_equal(data['families'],families);families=np.asarray(families)
    rankings={}
    for name in ('all','binomial','planted'):
        include=np.ones(len(X),bool) if name=='all' else families==name
        a,y,w=centered[include],target[include],weights[include];w=w/w.sum();prediction=np.zeros(len(y));units=[]
        for trace in discovery['traces'][name]:
            losses={i:float(np.sum(w*(prediction+a[:,i]-y)**2)) for i in range(32) if i not in units}
            same(trace['candidates'],[{'unit':i,'loss':v} for i,v in losses.items()],'greedy alternatives differ')
            winner=min(losses,key=lambda i:(losses[i],i));same(trace['added'],winner,'discovery selection differs')
            same(trace['loss'],losses[winner],'discovery loss differs');prediction+=a[:,winner];units.append(winner)
        rankings[name]=units
    same(rankings,discovery['rankings'],'saved rankings differ');same(digest(rankings),seal['rankings_sha256'],'ranking seal differs')
    for unit,row in enumerate(load(raw/'capability-ablations.json')):
        same(row['unit'],unit,'ablation inventory differs');program=row['program'];bank=banks[program]
        for key,array in full['v1'].items():
            expected=array.copy()
            if key=='net.2.weight':expected[:,unit]=0
            np.testing.assert_array_equal(bank[key],expected)
        with np.load((root/program).with_suffix('.npz')) as ablation:
            np.testing.assert_array_equal(ablation['inputs'],X);np.testing.assert_array_equal(ablation['full_c'],Y)
            consumed(program,X,ablation['c'][:,None]);losses=[]
            for boundary in discovery['boundaries']:
                sl=slice(boundary['offset'],boundary['offset']+boundary['rows']);delta=ablation['c'][sl]-Y[sl]
                losses.append(float(np.mean((delta-delta.mean())**2)))
        same(losses,row['frontier_losses'],'direct C ablation losses differ')
        same(float(np.mean(losses)),row['mean_centered_score_change'],'ablation mean differs')
    variants={v['label']:v for v in seal['variants']}
    for width in (4,8):
        union=sorted(set(rankings['binomial'][:width])|set(rankings['planted'][:width]))
        same(variants[f'v1-union{width}']['units'],union,'neuron union differs')
        same(variants[f'v1-matched-global-union{width}']['units'],sorted(rankings['all'][:len(union)]),'matched global differs')
        catalog=seal['catalogs'][str(width)]
        require(any(r['family']=={'0':union} and r['component_count']==len(union) and
            f'pair:binomial{width}+planted{width}' in r['sources'] for r in catalog),'canonical union catalog differs')
    for width in (4,8,16):same(variants[f'v1-global{width}']['units'],sorted(rankings['all'][:width]),'global units differ')
    for family in ('binomial','planted'):same(variants['v1-specialist-'+family]['units'],sorted(rankings[family][:8]),'specialist differs')
    width=variants['v1-union8']['hidden_width'];source=full['v1']
    energy=(np.square(source['net.0.weight'].astype(np.float64)).sum(axis=1)+np.square(source['net.0.bias'].astype(np.float64)))*np.square(source['net.2.weight'].astype(np.float64)[0])
    same(variants['v1-energy-matched']['units'],sorted(np.argsort(-energy,kind='stable')[:width].tolist()),'energy control differs')
    for seed in (17,29):same(variants[f'v1-random-matched-{seed}']['units'],sorted(np.random.default_rng(seed).choice(32,width,replace=False).tolist()),'random mask differs')
    for spec in variants.values():
        bank=banks[spec['program']];units=spec['units'];h=len(units);d=len(full[spec['version']]['mean'])
        for key,array in full[spec['version']].items():
            expected=array[units] if key in ('net.0.weight','net.0.bias') else array[:,units] if key=='net.2.weight' else array
            np.testing.assert_array_equal(bank[key],expected)
        same(spec['hidden_width'],h,'hidden width differs');same(spec['learned_coefficients'],h*(d+2)+1,'parameter count differs')
        same(spec['normalization_buffers'],2*d,'buffer count differs');same(spec['linear_macs_per_candidate'],(d+1)*h,'MAC count differs')
        same(spec['bundle_bytes'],(root/spec['program']).stat().st_size,'binary bytes differ')
        same(spec['bundle_digest'],maps[spec['program']]['bundle']['content_sha256'],'variant bundle differs')
    witnesses={}
    for path in (raw/'capability-witnesses').glob('*.json.gz'):
        body=json.loads(gzip.decompress(path.read_bytes()));identity=path.name.removesuffix('.json.gz')
        same(digest(body),identity,'witness seal differs');witnesses[identity]=body
        if 'manifest' in body:
            same(digest(body['manifest']),body['fingerprint'],'cut seal differs')
            same(describe(body['payload']),body['manifest']['payload'],'cut payload differs')
        for receipt in body.get('receipts',[]):same(digest({k:v for k,v in receipt.items() if k!='fingerprint'}),receipt['fingerprint'],'receipt seal differs')
    def features(frontier):
        epoch=witnesses[frontier['epoch']];f=epoch['features'];path=raw/'capability-witnesses'/(f['sha256']+'.f32')
        same(sha(path),f['sha256'],'feature seal differs');X=np.fromfile(path,dtype='<f4').reshape(f['shape'])
        require(np.isfinite(X).all(),'nonfinite features')
        for key in ('context','round','remaining','prefix_sha256'):same(frontier[key],epoch[key],'frontier epoch differs')
        same(frontier['candidate_ids'],[digest({'epoch':frontier['epoch'],'ordinal':i}) for i in range(len(X))],'candidate addresses differ')
        return X
    numerics=load(raw/'capability-numerics.json');numeric_arrays={}
    for identity,row in numerics.items():
        program=row['program'];mapping=maps[program];source=mapping['origin']['source']
        same(digest({'expression':source['expression'],'domain':source['domain'],'variables':source['variable_order'],
            'input_sha256':source['input_sha256']}),identity,'numeric program identity differs')
        with np.load((root/program).with_suffix('.npz')) as data:
            inputs,native,actual=(data[k].copy() for k in ('inputs','native','c'))
        same(array_sha(inputs),source['input_sha256'],'numeric input binding differs');consumed(program,inputs,actual)
        np.testing.assert_array_equal(actual,row['outputs']);same(float(np.abs(native-actual).max()),row['source_c_max_abs'],'numeric source/C residual differs')
        same(len(mapping['instructions']),row['operations'],'numeric instruction count differs');numeric_arrays[identity]=(inputs,actual)
    rows=load(raw/'capability-results.json');errors=[];observations=[];order_changes=[];endpoint_count=0;caps=None
    names=maps['raw/scorer-v1.tcm']['origin']['source']['feature_order']
    for row in rows:
        case=cases[row['goal']];spec=variants[row['variant']]
        same(row['family'],case['family'],'row family differs')
        if row['status']=='error':errors.append(row);continue
        end=witnesses[row['endpoint']];endpoint_count+=1
        identity=digest({'sources':job['config']['sources'],'variant':spec,'seed':case['seed']})
        expected_caps={'rounds':6,'per_round':8,'selections':48,'nodes':768,'matches':1024,'search_seconds':120,'extract_passes':1,'extract_seconds':.5}
        same(end['context'],digest({'executor':identity,'expression':case['expression'],'domain':case['domain'],'caps':expected_caps}),'endpoint context differs')
        cost,ops=expression_work(end['expression'],case['domain']);same(cost,end['cost'],'endpoint cost differs')
        same(ops,dict(+Counter(end['ops'])),'endpoint operation counts differ');same(row['cost'],cost,'row cost differs')
        same(polynomial(end['expression'],case['domain'])==polynomial(case['expression'],case['domain']),end['exact_polynomial'],'polynomial result differs')
        same(row['exact_polynomial'],end['exact_polynomial'],'row polynomial result differs');same(row['numerical'],end['numerical'],'numeric consumer differs')
        same(row['reference_cost'],case['reference_cost'],'reference cost differs')
        same(row['reference_gap'],None if case['reference_cost'] is None else cost-case['reference_cost'],'reference gap differs')
        same(row['reference_reached'],case['reference_cost'] is not None and cost<=case['reference_cost'],'reference attainment differs')
        prefix=[];used=0;candidate_count=0;rng=np.random.default_rng(case['seed'])
        same(len(row['decisions']),len(end['decisions']),'decision counts differ')
        for step,(decision,actual) in enumerate(zip(row['decisions'],end['decisions'])):
            frontier=witnesses[actual['frontier']];X=features(frontier);count=frontier['count']
            same(frontier['prefix_sha256'],digest(prefix),'decision prefix differs');same(frontier['remaining'],48-used,'remaining budget differs')
            same(count,min(8,48-used,len(X)),'selection cap differs');same(decision['candidate_count'],len(X),'candidate count differs')
            if spec['random_policy']:order=rng.permutation(len(X)).tolist()
            elif spec['heuristic']:
                columns={n:X[:,i] for i,n in enumerate(names)}
                score=(4.0*columns['target_on_route']+columns['est_saving']+0.5*columns['f_on_route']-0.1*columns['log_arity'])
                order=np.argsort(-score,kind='stable').tolist()
            elif spec.get('broad_policy',False):
                broad=(X[:,names.index('rule_factor')]>0)&(X[:,names.index('log_group')]>np.float32(np.log1p(2)))
                order=sorted(range(len(X)),key=lambda i:(0,-float(X[i,names.index('target_on_route')]),
                    -float(X[i,names.index('log_group')]),i) if broad[i] else (1,0,0,i))
            else:
                key=decision['score'];audit=scores[key];columns=maps[spec['program']]['origin']['source']['feature_order']
                inputs=np.column_stack([X[:,names.index(n)] if n in names else np.zeros(len(X),dtype='<f4') for n in columns]).astype('<f4')
                np.testing.assert_array_equal(inputs,score_arrays[key][0]);same(sha(root/audit['program']),sha(root/spec['program']),'wrong scorer consumed')
                order=audit['native_order'] if spec.get('native_policy',False) else audit['c_order'];candidate_count+=len(X)
                if audit['native_order'][:count]!=audit['c_order'][:count]:order_changes.append({'goal':row['goal'],'variant':row['variant'],'step':step,
                    'same_selected_set':set(audit['native_order'][:count])==set(audit['c_order'][:count])})
            choice=list(order[:count]);edits=[]
            if spec['quota']:
                broad=(X[:,names.index('rule_factor')]>0)&(X[:,names.index('log_group')]>np.float32(np.log1p(2)))
                pool=[i for i in order if i not in choice and (not bool(broad[i]) if spec['alternative'] else bool(broad[i]))]
                if spec['alternative']:random.Random(case['seed']+step).shuffle(pool)
                else:
                    rank={v:i for i,v in enumerate(order)}
                    pool.sort(key=lambda i:(-float(X[i,names.index('target_on_route')]),-float(X[i,names.index('log_group')]),rank[i]))
                for slot,new in zip(range(count-1,max(-1,count-spec['quota']-1),-1),pool):
                    edits.append({'slot':slot,'removed':choice[slot],'inserted':new});choice[slot]=new
            same(choice,decision['selected'],'proposal selection differs');same(edits,decision['edits'],'proposal edits differ')
            same(actual['choice'],[frontier['candidate_ids'][i] for i in choice],'native consumer did not use proposal')
            same(frontier['proposals']['active'],actual['choice'],'active proposal differs');used+=count;prefix.append(actual)
        same(used,row['selected'],'row work differs');same(used,end['selection_budget_used'],'endpoint work differs');require(used<=48,'selection budget exceeded')
        same(row['applications'],sum(t['applied'] for t in end['trace']),'applications differ')
        same(row['unions'],sum(t['unions'] for t in end['trace']),'egraph union count differs')
        same(row['scorer_logical_candidates'],candidate_count,'scorer candidate work differs')
        same(row['scorer_linear_macs'],candidate_count*spec['linear_macs_per_candidate'],'scorer MAC work differs')
        numeric=end['numerical'];target=numerics[numeric['program']];reference=numerics[numeric['reference']]
        for key,expression in ((numeric['program'],end['expression']),(numeric['reference'],case['expression'])):
            source=maps[numerics[key]['program']]['origin']['source'];same(source['expression'],expression,'wrong numeric expression')
            same(source['domain'],case['domain'],'wrong numeric domain')
            ports=set()
            def variables(e):
                if isinstance(e,str):ports.add(e)
                elif isinstance(e,list) and e[0]!='const':
                    for child in e[1:]:variables(child)
            variables(case['expression']);same(source['variable_order'],sorted(ports),'numeric input ports differ')
        np.testing.assert_array_equal(numeric_arrays[numeric['program']][0],numeric_arrays[numeric['reference']][0])
        same(numeric['operations'],target['operations'],'emitted operation count differs')
        same(numeric['C_reference_max_abs'],float(np.abs(numeric_arrays[numeric['program']][1]-numeric_arrays[numeric['reference']][1]).max()),'numeric continuation differs')
        clocks=row['clocks'];same(clocks['endpoint'],row['endpoint'],'clock belongs to different endpoint')
        same(clocks['search_clock_proven_unhit'],clocks['search_wall_seconds']<120 and end['search_status']!='time-cap','search clock differs')
        same(clocks['extraction_clock_proven_unhit'],clocks['extraction_wall_seconds']<.25,'extraction clock differs')
        if row['cut']:
            cut=witnesses[row['cut']];manifest=cut['manifest'];payload=cut['payload']
            point=described_value(manifest['execution_point']);surface=described_value(manifest['surface'])
            initial=StateCut(manifest['model_identity'],manifest['execution'],point['phase'],manifest['parent'],payload,execution_point=point,surface=surface)
            same(initial.fingerprint,cut['fingerprint'],'initial StateCut does not close');same(payload['context'],end['context'],'cut context differs')
            same(payload['prefix']+[{'frontier':payload['frontier'],'choice':payload['choice']}],end['decisions'][:len(payload['prefix'])+1],'initial choice not consumed')
            same(len(cut['receipts']),1,'unexpected receipt geometry');receipt=cut['receipts'][0]
            same(receipt['operation'],'continue','unexpected receipt operation');same(receipt['parent'],cut['fingerprint'],'receipt parent differs')
            same(receipt['model_identity'],manifest['model_identity'],'receipt model differs');same(receipt['execution'],manifest['execution'],'receipt execution differs')
            transition=receipt['transitions'][0];same(transition['before'],point,'receipt entry differs')
            same(transition['native_operation']['writes'],['prefix','frontier','choice','endpoint'],'consumer footprint differs')
            final={'context':payload['context'],'prefix':end['decisions'],'frontier':None,'choice':[],'endpoint':row['endpoint']}
            after=transition['after'];surface={**surface,'slots':[{**s,'writable':False} for s in surface['slots']]}
            final_cut=StateCut(manifest['model_identity'],manifest['execution'],after['phase'],cut['fingerprint'],final,execution_point=after,surface=surface)
            same(final_cut.fingerprint,receipt['result'],'receipt endpoint does not close')
        else:same(end['decisions'],[],'uncaptured run had choices')
        observations.append({'goal':row['goal'],'family':row['family'],'variant':row['variant'],'cost':cost,
            'split':'replication' if row['goal'].startswith('replication-') else 'original',
            'reference_reached':row['reference_reached'],'exact_polynomial':end['exact_polynomial'],
            'C_reference_max_abs':numeric['C_reference_max_abs'],'scorer_macs':row['scorer_linear_macs'],
            'emitted_C_instructions':numeric['operations'],'union_insertions':sum(len(d['edits']) for d in row['decisions']),
            'decisions':len(row['decisions']),'selection_pressure':any(d['candidate_count']>len(d['selected']) for d in row['decisions'])})
    declared=seal.get('case_variants',{goal:list(variants) for goal in cases})
    same({(r['goal'],r['variant']) for r in rows},{(goal,label) for goal,labels in declared.items() for label in labels},'declared panel rows missing')
    same(len({(r['goal'],r['variant']) for r in rows}),len(rows),'duplicate panel rows')
    summaries=[];pairs=[]
    for spec in variants.values():
        selected=[r for r in observations if r['variant']==spec['label']]
        summary={'variant':spec['label'],'width':spec['hidden_width'],
            'serving_learned_coefficients':spec['learned_coefficients'] if spec['serving_uses_neural_scores'] else 0,
            'serving_normalization_buffers':spec['normalization_buffers'] if spec['serving_uses_neural_scores'] else 0,
            'completed':len(selected),'errors':sum(r['variant']==spec['label'] for r in errors),
            'mean_cost':float(np.mean([r['cost'] for r in selected])) if selected else None,
            'reference_reached':sum(r['reference_reached'] for r in selected),'exact_polynomial':sum(r['exact_polynomial'] for r in selected),
            'total_scorer_linear_macs':sum(r['scorer_macs'] for r in selected),
            'total_emitted_C_instructions':sum(r['emitted_C_instructions'] for r in selected)}
        summaries.append(summary)
        for family in sorted({r['family'] for r in selected}):
            family_rows=[r for r in selected if r['family']==family]
            pairs.append({'variant':spec['label'],'family':family,'n':len(family_rows),
                'mean_cost':float(np.mean([r['cost'] for r in family_rows])),
                'reference_reached':sum(r['reference_reached'] for r in family_rows)})
    same(summaries,load(raw/'capability-summary.json'),'summary differs')
    lookup={(r['variant'],r['goal']):r for r in observations};comparisons=[]
    for donor,recipient in [('v1-global16','v1-full'),('v1-matched-global-union8','v1-full'),
        ('v1-union8','v1-matched-global-union8'),('v1-specialist-planted','v1-full'),
        ('v1-random-matched-29','v1-full'),('v2a-proposal-union1','v2a-full'),
        ('v2a-proposal-union2','v2a-full'),('v2a-proposal-union4','v2a-full'),('v2a-alternative1','v2a-full'),
        ('v1-proposal-union1','v1-full'),('v1-global14-proposal-union1','v1-matched-global-union8'),
        ('broad-only','v2a-full'),('v1-native','v1-full'),('v1-planted8-native','v1-specialist-planted'),
        ('v1-random14-native','v1-random-matched-29')]:
        effects=[]
        for goal in cases:
            if (donor,goal) in lookup and (recipient,goal) in lookup:
                a,b=lookup[donor,goal],lookup[recipient,goal]
                effects.append({'goal':goal,'family':a['family'],'split':a['split'],'before':b['cost'],'after':a['cost'],'reduction':b['cost']-a['cost']})
        comparisons.append({'candidate':donor,'reference':recipient,'n':len(effects),
            'wins':sum(r['reduction']>0 for r in effects),'ties':sum(r['reduction']==0 for r in effects),
            'losses':sum(r['reduction']<0 for r in effects),'effects':effects})
    clocks=load(raw/'capability-clocks.json');confounded=sum(not a['search_clock_proven_unhit'] or not a.get('extraction_clock_proven_unhit',True) for a in clocks)
    same(confounded,result['clock_confounded'],'clock count differs');same(endpoint_count,result['completed'],'completed count differs')
    same(len(errors),result['errors'],'error count differs');same(len(calls),result['physical_C_calls'],'physical C count differs')
    same(memo['logical_calls'],result['logical_C_calls'],'logical C count differs');same(score_max,result['scorer_max_abs'],'maximum scorer error differs')
    return {'schema':'tcm-math-capability-verification-v1','mechanics_status':'verified',
        'instrument_status':'partial-endpoint-coverage' if errors else 'all-declared-goal-variant-pairs-consumed',
        'trend_status':'signed-per-goal-observations','terminal_status':'exploratory bounded model/search assay; no general reasoning or minimality claim',
        'job_id':root.name,'raw_files':len(report['raw']),'raw_bytes':sum(r['bytes'] for r in report['raw'].values()),
        'programs':len(maps),'witnesses':len(witnesses),'discovery_goals':len(discovery_ids),'discovery_frontiers':len(discovery['boundaries']),
        'discovery_candidate_rows':discovery_rows,'test_goals':len(cases),'variants':len(variants),'completed':endpoint_count,'errors':errors,
        'physical_C_calls':len(calls),'logical_C_calls':memo['logical_calls'],'scorer_batches':len(scores),'scorer_max_abs':score_max,
        'numeric_source_C_max_abs':max(r['source_c_max_abs'] for r in numerics.values()),
        'numeric_reference_max_abs':max(r['C_reference_max_abs'] for r in observations),
        'clock_confounded':confounded,'selection_order_changes':order_changes,
        'selection_set_changes':sum(not r['same_selected_set'] for r in order_changes),
        'summaries':summaries,'family_summaries':pairs,'comparisons':comparisons,'observations':observations}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=verify(args.result);body=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(body)
    else:print(body,end='')
