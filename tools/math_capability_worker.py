"""Finite discovery/locked-test capability and efficiency panel for math scorers."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'public'))
from saturn_pub import Session
from math_capability import (Hidden,PackedWitnesses,coalition_catalog,corpus,greedy_units,
                             proposal_union,replication_corpus,structural_order,subnetwork)
from math_causal_worker import ExactMemo
from math_debugger import ReplayEngine,ChoiceAdapter
from math_program import CExecutor,digest
from math_worker import RAW,Expression,load,scorers,trusted_class
from intake_worker import main,sha,snapshot
from program import lower_fx


def run():
    torch.set_num_threads(1);manifest=json.loads(Path('source-manifest.json').read_text())
    for path,identity in manifest['sources'].items():
        if sha(path)!=identity:raise ValueError('staged source changed: '+path)
    executor=CExecutor('build/tcm',RAW/'c-calls.jsonl');memo=ExactMemo(executor);frozen=scorers()
    cg=load('concept_graph','sources/v1/concept_graph.py');sch=load('scheduler','sources/v1/scheduler.py')
    goals=load('goals','sources/v1/goals.py');discovery,test=corpus(goals,cg)
    original_test_ids=[c['id'] for c in test];test+=replication_corpus(goals,cg)
    snapshot('capability-corpus',{'discovery':discovery,'test':test,
        'unit_of_analysis':'paired goal; binomial repeats share family/arity; held-shape families share one variable seed',
        'scope':'fresh assay rows from known families; no claim of exclusion from original checkpoint training data'})
    store=PackedWitnesses(RAW/'capability-witnesses');score_audits={};frontiers=[];numerics={}
    def rows_for(X,version):
        names=frozen[version][1];columns={k:i for i,k in enumerate(sch.FEATURES)}
        return np.column_stack([X[:,columns[k]] if k in columns else np.zeros(len(X),dtype='<f4') for k in names]).astype('<f4')
    def score(path,model,rows):
        key=digest({'program_sha256':sha(path),'inputs_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'shape':list(rows.shape)})
        if key not in score_audits:
            actual=memo.run(path,rows).reshape(-1)
            with torch.no_grad():native=model(torch.from_numpy(rows)).numpy().reshape(-1)
            np.savez_compressed(RAW/('capability-score-'+key+'.npz'),inputs=rows,native=native,c=actual)
            score_audits[key]={'program':str(path),'rows':len(rows),'source_c_max_abs':float(np.abs(native-actual).max()),
                'native_order':np.argsort(-native,kind='stable').tolist(),'c_order':np.argsort(-actual,kind='stable').tolist(),'uses':0}
        score_audits[key]['uses']+=1
        return score_audits[key]['c_order'],key
    def scout(case):
        graph=cg.EGraph(case['domain'],'ac');roots=[goals._insert_nary(graph,case['expression'])];graph.rebuild()
        records=[];remaining=48
        class Policy:
            def select(self,g,rs,metas,X,round_):
                nonlocal remaining
                rows=np.asarray(X,dtype='<f4');count=min(8,remaining,len(rows));order,key=score(RAW/'scorer-v1.tcm',frozen['v1'][0],rows)
                records.append({'goal':case['id'],'family':case['family'],'round':round_,'score':key,'rows':rows.copy(),'count':count})
                remaining-=count;return order[:count]
        cg.saturate(graph,roots,'ac',index_cls=cg.DictIndex,scheduler=Policy(),featurize=sch.featurize,
            iter_cap=6,node_cap=768,match_cap=1024,time_cap=120)
        return records
    for case in discovery:
        print('PHASE capability discovery '+case['id'],flush=True);frontiers.extend(scout(case))
    # Capture actual C post-ReLU outputs once on prepared discovery rows.
    X=np.concatenate([r['rows'] for r in frontiers]);hidden=Hidden(frozen['v1'][0]).eval()
    hidden_map=lower_fx(hidden,21,digest({'checkpoint':frozen['v1'][2]['source_identity'],'role':'post-ReLU-discovery'}),
        RAW/'capability-hidden.tcm',source_record={'role':'original hidden activations','feature_order':list(sch.FEATURES)})
    H=memo.run(RAW/'capability-hidden.tcm',X)
    full=memo.run(RAW/'scorer-v1.tcm',X).reshape(-1)
    weights=frozen['v1'][0].net[2].weight.detach().numpy()[0]
    contributions=H.astype(np.float64)*weights.astype(np.float64)[None,:]
    centered=contributions.copy();target=full.astype(np.float64);families=[];row_weights=np.zeros(len(X));offset=0;boundaries=[]
    for row in frontiers:
        n=len(row['rows']);sl=slice(offset,offset+n);centered[sl]-=centered[sl].mean(axis=0);target[sl]-=target[sl].mean()
        row_weights[sl]=1/n;families.extend([row['family']]*n);boundaries.append({'offset':offset,'rows':n,**{k:v for k,v in row.items() if k!='rows'}});offset+=n
    families=np.asarray(families);rankings={};traces={}
    for name in ('all','binomial','planted'):
        include=np.ones(len(X),dtype=bool) if name=='all' else families==name
        rankings[name],traces[name]=greedy_units(centered[include],target[include],row_weights[include],16)
    np.savez_compressed(RAW/'capability-discovery.npz',inputs=X,hidden_C=H,full_C=full,centered_contributions=centered,
        centered_target=target,weights=row_weights,families=families)
    snapshot('capability-discovery',{'boundaries':boundaries,'rankings':rankings,'traces':traces,
        'selection':'centered score-error forward coordinate search, FP64 diagnostic from actual C hidden states; no fitted coefficients or held-out outcomes',
        'hidden_map':hidden_map['bundle']['content_sha256']})
    # Direct C zero-output-weight ablations keep the hidden representation and
    # original reduction order; these calibrate coordinate sensitivity.
    ablations=[]
    for unit in range(32):
        model=copy.deepcopy(frozen['v1'][0]);model.net[2].weight.data[:,unit]=0
        path=RAW/f'ablate-v1-unit{unit}.tcm'
        lower_fx(model,21,digest({'checkpoint':frozen['v1'][2]['source_identity'],'zero_post_relu_unit':unit}),path,
            source_record={'parent_checkpoint':frozen['v1'][2]['source_identity'],'zero_post_relu_unit':unit,'feature_order':list(sch.FEATURES)})
        actual=memo.run(path,X).reshape(-1)
        losses=[]
        for boundary in boundaries:
            sl=slice(boundary['offset'],boundary['offset']+boundary['rows']);delta=actual[sl]-full[sl]
            losses.append(float(np.mean((delta-delta.mean())**2)))
        ablations.append({'unit':unit,'mean_centered_score_change':float(np.mean(losses)),
            'frontier_losses':losses,'program':str(path)})
        np.savez_compressed(path.with_suffix('.npz'),inputs=X,c=actual,full_c=full)
    snapshot('capability-ablations',ablations)
    variants=[];variant_models={};catalogs={}
    def variant(label,version,units=None,*,quota=0,alternative=False,heuristic=False,random_policy=False,broad_policy=False,native_policy=False):
        model=frozen[version][0] if units is None else subnetwork(frozen[version][0],units)
        if units is None:path=RAW/('scorer-'+version+'.tcm');mapping=frozen[version][2]
        else:
            path=RAW/(label+'.tcm');mapping=lower_fx(model,len(frozen[version][1]),
                digest({'checkpoint':frozen[version][2]['source_identity'],'kept_units':sorted(units)}),path,
                source_record={'parent_checkpoint':frozen[version][2]['source_identity'],'kept_units':sorted(units),
                    'feature_order':frozen[version][1],'deployment':'frozen coefficients copied; original unit order; no retraining'})
        h=model.net[0].out_features;d=len(frozen[version][1]);trainable=sum(p.numel() for p in model.parameters())
        item={'label':label,'version':version,'units':list(range(32)) if units is None else sorted(units),
            'hidden_width':h,'learned_coefficients':trainable,'normalization_buffers':2*d,
            'linear_macs_per_candidate':(d+1)*h,'bundle_bytes':path.stat().st_size,'program':str(path),
            'bundle_digest':mapping['bundle']['content_sha256'],'quota':quota,'alternative':alternative,
            'heuristic':heuristic,'random_policy':random_policy,'broad_policy':broad_policy,'native_policy':native_policy,
            'serving_uses_neural_scores':not(heuristic or random_policy or broad_policy)}
        variants.append(item);variant_models[label]=(model,path)
    variant('v1-full','v1');variant('v2a-full','v2a')
    for width in (4,8,16):variant(f'v1-global{width}','v1',rankings['all'][:width])
    for width in (4,8):
        catalog=coalition_catalog('sources/conditional_family.py',
            [{'key':f'{family}{width}','family':{0:rankings[family][:width]}} for family in ('binomial','planted')],2*width)
        catalogs[str(width)]=catalog
        for row in catalog:
            if any(source.startswith('pair:') for source in row['sources']):
                units=row['family']['0'];variant(f'v1-union{width}','v1',units)
                variant(f'v1-matched-global-union{width}','v1',rankings['all'][:len(units)])
    for family in ('binomial','planted'):variant('v1-specialist-'+family,'v1',rankings[family][:8])
    size=next(v['hidden_width'] for v in variants if v['label']=='v1-union8')
    model=frozen['v1'][0]
    energy=(model.net[0].weight.detach().double().square().sum(dim=1)+model.net[0].bias.detach().double().square())*model.net[2].weight.detach().double()[0].square()
    variant('v1-energy-matched','v1',np.argsort(-energy.numpy(),kind='stable')[:size].tolist())
    for seed in (17,29):variant('v1-random-matched-'+str(seed),'v1',np.random.default_rng(seed).choice(32,size,replace=False).tolist())
    for quota in (1,2,4):variant('v2a-proposal-union'+str(quota),'v2a',quota=quota)
    variant('v2a-alternative1','v2a',quota=1,alternative=True)
    variant('heuristic','v1',heuristic=True);variant('random','v1',random_policy=True)
    original_variants=[v['label'] for v in variants]
    variant('v1-proposal-union1','v1',quota=1)
    variant('v1-global14-proposal-union1','v1',rankings['all'][:size],quota=1)
    variant('broad-only','v1',broad_policy=True)
    added_variants=[v['label'] for v in variants if v['label'] not in original_variants]
    variant('v1-native','v1',native_policy=True)
    variant('v1-planted8-native','v1',rankings['planted'][:8],native_policy=True)
    random_units=next(v['units'] for v in variants if v['label']=='v1-random-matched-29')
    variant('v1-random14-native','v1',random_units,native_policy=True)
    native_variants=[v['label'] for v in variants if v['native_policy']]
    replication_variants=['v1-full','v1-matched-global-union8','v1-proposal-union1','v1-global14-proposal-union1',
        'v2a-full','v2a-proposal-union1','broad-only']
    case_variants={c['id']:(original_variants+added_variants if c['id'] in original_test_ids else replication_variants.copy()) for c in test}
    for key in ('test-trinomial-3','replication-trinomial-3'):case_variants[key]+=native_variants
    case_variants['replication-trinomial-3']+=['v1-specialist-planted','v1-random-matched-29']
    # This seal is written before any held-out model execution. The source/input
    # corpus was fixed before discovery; no test result can change a variant.
    snapshot('capability-seal',{'schema':'tcm-math-capability-seal-v1','variants':variants,'catalogs':catalogs,
        'discovery_ids':[c['id'] for c in discovery],'test_ids':[c['id'] for c in test],
        'rankings_sha256':digest(rankings),'manalysis_source_sha256':sha('sources/conditional_family.py'),
        'case_variants':case_variants,'followup':{'prior_job':'job-2f36448406a4',
            'reason':'preserve partial first panel; repair constant-left method lowering; native tie-order control; scorer-free broad-only and fixed-budget reduced-scorer composition',
            'original_variants':original_variants,'added_variants':added_variants,
            'replication_ids':[c['id'] for c in test if c['id'] not in original_test_ids],
            'lead_selection':'planted8 and random14/29 were selected after their original trinomial3 result; fresh layout is exploratory replication, not an independent structure'}})
    Product=trusted_class('sources/law.py','ProductMLP');rule=next(r['rule'] for r in json.loads(Path('inputs/exact-rules.json').read_text()) if r['seed']==811)
    cell=Product(len(rule['u'])).eval();cell.load_state_dict({k:torch.tensor(rule[k],dtype=torch.float32) for k in ('u','v','w')},strict=True)
    def numerical_consumer(case):
        names=sorted(goals.variables(case['expression']));matrix=case['domain']==cg.MATRIX
        inputs=np.random.default_rng(case['seed']).integers(-2,3,(16,len(names)*(4 if matrix else 1))).astype('<f4')
        def evaluate(expression):
            input_sha=hashlib.sha256(inputs.tobytes()).hexdigest()
            identity=digest({'expression':expression,'domain':case['domain'],'variables':names,'input_sha256':input_sha})
            if identity not in numerics:
                module=Expression(expression,names,cell if matrix else None);path=RAW/('capability-numeric-'+identity+'.tcm')
                mapping=lower_fx(module,inputs.shape[1],identity,path,source_record={'expression':expression,'domain':case['domain'],
                    'variable_order':names,'cell':'learned811' if matrix else None,'input_sha256':input_sha})
                with torch.no_grad():native=module(torch.from_numpy(inputs)).numpy()
                actual=memo.run(path,inputs);np.savez_compressed(path.with_suffix('.npz'),inputs=inputs,native=native,c=actual)
                numerics[identity]={'program':str(path),'operations':len(mapping['instructions']),
                    'source_c_max_abs':float(np.abs(native-actual).max()),'outputs':actual.tolist()}
            return identity,numerics[identity]
        reference_id,reference=evaluate(case['expression'])
        def consume(expression):
            identity,item=evaluate(expression)
            return {'program':identity,'reference':reference_id,'operations':item['operations'],
                'C_reference_max_abs':float(np.abs(np.asarray(item['outputs'])-np.asarray(reference['outputs'])).max())}
        return consume
    results=[];clock_audits=[]
    for case in test:
        consume=numerical_consumer(case)
        for spec in variants:
            if spec['label'] not in case_variants[case['id']]:continue
            label=spec['label'];print('PHASE capability test '+case['id']+' '+label,flush=True)
            decisions=[];score_uses=[];rng=np.random.default_rng(case['seed'])
            model,path=variant_models[label]
            def policy(X,count):
                if spec['heuristic']:order=np.argsort(-sch.heuristic_score(X),kind='stable').tolist();key=None
                elif spec['random_policy']:order=rng.permutation(len(X)).tolist();key=None
                elif spec['broad_policy']:order=structural_order(X,sch.FEATURES);key=None
                else:
                    order,key=score(path,model,rows_for(X,spec['version']))
                    if spec['native_policy']:order=score_audits[key]['native_order']
                choice=order[:count];edits=[]
                if spec['quota']:
                    choice,edits=proposal_union(order,X,sch.FEATURES,count,spec['quota'],alternative=spec['alternative'],seed=case['seed']+len(decisions))
                decisions.append({'score':key,'selected':choice,'edits':edits,'candidate_count':len(X)})
                if key:score_uses.append(key)
                return {'active':choice}
            engine=ReplayEngine(cg,goals,sch.featurize,cg.DictIndex,policy,store,case['expression'],case['domain'],
                digest({'sources':manifest['sources'],'variant':spec,'seed':case['seed']}),consume=consume)
            try:
                # Recompute a sealed first boundary through the public Saturn
                # consumer adapter. Random policy restarts its seed on replay.
                try:initial=engine.run(pause_at=0,suffix='active')
                except ValueError as error:
                    if str(error)!='requested choice boundary was not reached':raise
                    initial=None
                decisions.clear();score_uses.clear();rng=np.random.default_rng(case['seed']);cut_id=None
                if initial is None:
                    end_id=engine.run(suffix='active')['endpoint'];end=store.get(end_id)
                else:
                    adapter=ChoiceAdapter(engine,suffix='active');session=Session(adapter,initial);cut=session.capture(retain=False)
                    session.continue_();end_id=session.capture(retain=False).payload['endpoint'];end=store.get(end_id)
                    cut_id=store.put({'manifest':cut.manifest(),'fingerprint':cut.fingerprint,'payload':cut.payload,
                        'receipts':[r.to_dict() for r in session.receipts]})
                clocks=engine.replay_audits[-1];clock_audits.extend(engine.replay_audits)
                results.append({'status':'completed','goal':case['id'],'family':case['family'],'variant':label,
                    'endpoint':end_id,'cut':cut_id,'choice_boundary':'captured' if cut_id else 'not-applicable-no-fresh-rewrites',
                    'cost':end['cost'],'reference_cost':case['reference_cost'],
                    'reference_gap':None if case['reference_cost'] is None else end['cost']-case['reference_cost'],
                    'reference_reached':case['reference_cost'] is not None and end['cost']<=case['reference_cost'],
                    'exact_polynomial':end['exact_polynomial'],'numerical':end['numerical'],'decisions':decisions,
                    'selected':end['selection_budget_used'],'applications':end['applications'],
                    'unions':sum(t['unions'] for t in end['trace']),
                    'scorer_logical_candidates':sum(score_audits[k]['rows'] for k in score_uses),
                    'scorer_linear_macs':sum(score_audits[k]['rows']*spec['linear_macs_per_candidate'] for k in score_uses),
                    'clocks':clocks})
            except Exception:results.append({'status':'error','goal':case['id'],'family':case['family'],'variant':label,'error':traceback.format_exc()})
    memo.finish();snapshot('capability-scorer-audits',score_audits);snapshot('capability-numerics',numerics)
    snapshot('capability-results',results);snapshot('capability-clocks',clock_audits)
    summaries=[]
    for spec in variants:
        rows=[r for r in results if r['variant']==spec['label'] and r['status']=='completed']
        summaries.append({'variant':spec['label'],'width':spec['hidden_width'],
            'serving_learned_coefficients':spec['learned_coefficients'] if spec['serving_uses_neural_scores'] else 0,
            'serving_normalization_buffers':spec['normalization_buffers'] if spec['serving_uses_neural_scores'] else 0,
            'completed':len(rows),'errors':sum(r['variant']==spec['label'] and r['status']=='error' for r in results),
            'mean_cost':float(np.mean([r['cost'] for r in rows])) if rows else None,
            'reference_reached':sum(r['reference_reached'] for r in rows),'exact_polynomial':sum(r['exact_polynomial'] for r in rows),
            'total_scorer_linear_macs':sum(r['scorer_linear_macs'] for r in rows),
            'total_emitted_C_instructions':sum(r['numerical']['operations'] for r in rows)})
    snapshot('capability-summary',summaries)
    return {'goals':len(test),'discovery_goals':len(discovery),'variants':len(variants),'summaries':summaries,
        'completed':sum(r['status']=='completed' for r in results),'errors':sum(r['status']=='error' for r in results),
        'physical_C_calls':executor.calls,'logical_C_calls':memo.logical,
        'scorer_batches':len(score_audits),'scorer_max_abs':max(r['source_c_max_abs'] for r in score_audits.values()),
        'clock_confounded':sum(not a['search_clock_proven_unhit'] or not a.get('extraction_clock_proven_unhit',True) for a in clock_audits),
        'contracts':{'training':'none; discovery selects original V1 ReLU coordinates; no shared coordinate claim across versions',
            'union':'canonical manalysis bounded neuron-set unions; recipient-only structural proposal quota, same8choicecap',
            'consumer':'original V1 DictIndex search and extractor on host, C scorers and emitted programs; FieldLM C-read bridge qualified separately',
            'quality':'exact function, reference-program cost, fresh goal-family performance and signed per-goal effects; no general language reasoning score',
            'efficiency':'scorer-only coefficients/buffers/binary bytes/MACs; native reference bank remains in the audit process',
            'terminal_status':'exploratory quality-cost frontier; no globally minimal circuit or scheduler promotion'}}


if __name__=='__main__':raise SystemExit(main({'math-capability':run},retain_bundles=True,evidence_budget_bytes=256*1024**2))
