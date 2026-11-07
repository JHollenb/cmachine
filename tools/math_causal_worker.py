"""Finite, frozen same-parent rewrite interventions; real work runs under mrun."""
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
from saturn_pub import Act,Session
from math_debugger import ChoiceAdapter,ReplayEngine,Witnesses,single_exchanges,outcome_summary
from math_program import CExecutor,ProjectedField,digest
from math_worker import RAW,Expression,load,scorers,trusted_class
from intake_worker import main,sha,snapshot
from program import lower_fx


class ExactMemo:
    """Stateless C reuse, bound to executable, full bundle and exact input bytes."""
    def __init__(self,executor):
        self.executor=executor;self.binary_sha=sha(executor.binary);self.files={};self.memo={};self.logical=0
    def run(self,path,rows,*,zero=None):
        path=Path(path).resolve();stat=path.stat();stamp=(stat.st_size,stat.st_mtime_ns)
        if path not in self.files:self.files[path]=(stamp,sha(path))
        if self.files[path][0]!=stamp:raise ValueError('C program changed during memoized execution')
        rows=np.ascontiguousarray(rows,dtype='<f4');input_sha=hashlib.sha256(rows.tobytes()).hexdigest()
        key=digest({'binary':self.binary_sha,'profile':'cpu-fp32-ordered-no-fallback',
            'bundle':self.files[path][1],'shape':list(rows.shape),'input':input_sha,'zero':zero})
        self.logical+=1
        if key not in self.memo:
            out=self.executor.run(path,rows,zero=zero)
            self.memo[key]={'output':out.copy(),'calls':0,'physical_call':self.executor.calls,
                'program':str(path.relative_to(Path.cwd())),'bundle_sha256':self.files[path][1],
                'shape':list(rows.shape),'input_sha256':input_sha,
                'output_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'zero':zero}
        entry=self.memo[key];entry['calls']+=1
        return entry['output'].copy()
    def finish(self):
        if sha(self.executor.binary)!=self.binary_sha:raise ValueError('C executable changed')
        for path,(_,identity) in self.files.items():
            if sha(path)!=identity:raise ValueError('C memo source program changed')
        snapshot('causal-c-memo',{'binary_sha256':self.binary_sha,'logical_calls':self.logical,
            'physical_calls':self.executor.calls,'entries':[{**{k:v for k,v in row.items() if k!='output'},'key':key}
                for key,row in self.memo.items()]})


def run():
    torch.set_num_threads(1)
    manifest=json.loads(Path('source-manifest.json').read_text())
    for path,identity in manifest['sources'].items():
        if sha(path)!=identity:raise ValueError('staged source changed: '+path)
    executor=CExecutor('build/tcm',RAW/'c-calls.jsonl');memo=ExactMemo(executor)
    models=scorers()
    cg=load('concept_graph','sources/v1/concept_graph.py')
    sch=load('scheduler','sources/v1/scheduler.py');goals=load('goals','sources/v1/goals.py')
    native_store=load('native_store','sources/v1/native_store.py')
    store=Witnesses(RAW/'causal-witnesses');score_cache={};lookup_rows={};current=[]
    def flush():
        if not current:return
        reader=current.pop()
        for row in reader.comparisons:
            key=digest({k:row[k] for k in ('bank','query','selected')})
            if key not in lookup_rows:lookup_rows[key]={**row,'logical_reads':0}
            lookup_rows[key]['logical_reads']+=row['logical_reads']
            lookup_rows[key]['native_equal'] &= row['native_equal']
    class CIndex(native_store.NativeIndex):
        def __init__(self,g):
            flush();super().__init__(g)
            reader=ProjectedField(self.view,RAW/'field-banks',memo)
            self.view.lookup=reader.lookup;current.append(reader)
    def score(X,count):
        proposals={};source={key:i for i,key in enumerate(sch.FEATURES)}
        for version,(model,names,mapping) in models.items():
            rows=np.column_stack([X[:,source[key]] if key in source else np.zeros(len(X),dtype=np.float32)
                                  for key in names]).astype('<f4')
            key=digest({'version':version,'input_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'shape':list(rows.shape)})
            if key not in score_cache:
                native=model.score_numpy(rows);actual=memo.run(RAW/('scorer-'+version+'.tcm'),rows).reshape(-1)
                path=RAW/('causal-scores-'+key+'.npz')
                np.savez_compressed(path,inputs=rows,native=native,c=actual)
                score_cache[key]={'version':version,'rows':len(rows),'source_c_max_abs':float(np.abs(native-actual).max()),
                    'c_order':np.argsort(-actual,kind='stable').tolist(),
                    'native_order':np.argsort(-native,kind='stable').tolist(),'uses':0}
            row=score_cache[key];row['uses']+=1;proposals[version]=row['c_order'][:count]
        return proposals
    # Only the exact learned cell is needed to consume ordered-matrix controls.
    Product=trusted_class('sources/law.py','ProductMLP')
    rule=next(r['rule'] for r in json.loads(Path('inputs/exact-rules.json').read_text()) if r['seed']==811)
    cell=Product(len(rule['u'])).eval()
    cell.load_state_dict({key:torch.tensor(rule[key],dtype=torch.float32) for key in ('u','v','w')},strict=True)
    numerics={}
    def consumer(original,domain):
        variables=sorted(goals.variables(original));matrix=domain==cg.MATRIX;width=4 if matrix else 1
        rows=np.random.default_rng(741).integers(-2,3,(32,len(variables)*width)).astype('<f4')
        def evaluate(expr):
            key=digest({'expression':expr,'domain':domain,'variables':variables})
            if key not in numerics:
                module=Expression(expr,variables,cell if matrix else None)
                path=RAW/('causal-numeric-'+key+'.tcm')
                mapping=lower_fx(module,rows.shape[1],key,path,source_record={'expression':expr,
                    'domain':domain,'variable_order':variables,'cell':'frozen learned811' if matrix else None,
                    'cell_source_sha256':sha('inputs/exact-rules.json') if matrix else None})
                with torch.no_grad():native=module(torch.from_numpy(rows)).numpy()
                actual=memo.run(path,rows)
                np.savez_compressed(path.with_suffix('.npz'),inputs=rows,native=native,c=actual)
                numerics[key]={'program':str(path),'operations':len(mapping['instructions']),
                    'source_c_max_abs':float(np.abs(native-actual).max()),'outputs':actual.tolist()}
            return key,numerics[key]
        original_id,reference=evaluate(original)
        def consume(expr):
            identity,result=evaluate(expr)
            return {'program':identity,'reference':original_id,'operations':result['operations'],
                'source_c_max_abs':result['source_c_max_abs'],
                'C_reference_max_abs':float(np.abs(np.asarray(result['outputs'])-np.asarray(reference['outputs'])).max())}
        return consume
    expressions=[('scalar-product',cg.SCALAR,goals.binomial_goal(3)[0][0]),
        ('scalar-planted',cg.SCALAR,goals.planted_goal(74117,n_vars=4,n_ops=5,max_monos=24)[0][0]),
        ('ordered-matrix',cg.MATRIX,['add',['add',['mul','A','C'],['mul','B','C']],['add',['mul','A','D'],['mul','B','D']]])]
    results=[];engines=[]
    identity=digest({'sources':manifest['sources'],'scorers':{v:m[2]['bundle']['content_sha256'] for v,m in models.items()},
        'C_executable':memo.binary_sha,'python_hash_seed':manifest['python_hash_seed']})
    for name,domain,original in expressions:
        engine=ReplayEngine(cg,goals,sch.featurize,CIndex,score,store,original,domain,identity,
            consume=consumer(original,domain));engines.append(engine)
        baselines={}
        for version in ('v1','v2a'):
            print('PHASE causal baseline '+name+' '+version,flush=True)
            state=engine.run(suffix=version);baselines[version]=store.get(state['endpoint'])
        target=baselines['v2a'];adapter=ChoiceAdapter(engine);rows=[];controls=[];cuts=[]
        def branch(label,round_,cut,choices=None,*,restore=False):
            print('PHASE causal branch '+name+' '+label+' round='+str(round_),flush=True)
            try:
                session=Session.from_cut(adapter,cut)
                if choices is not None:session.apply(Act.replace('choice',choices,name=label))
                if restore:session.apply(Act.replace('choice',cut.payload['choice'],name='restore-original-choice'))
                session.continue_();state=session.capture(retain=False).payload;end=store.get(state['endpoint'])
                receipts=[r.to_dict() for r in session.receipts]
                receipt_id=store.put({'label':label,'cut':cut.fingerprint,'receipts':receipts})
                clock=engine.replay_audits[-1]
                timing=clock['search_clock_proven_unhit'] and clock['extraction_clock_proven_unhit']
                changed_choices=[];changed_graphs=[]
                for a,b in zip(target['decisions'],end['decisions']):
                    fa,fb=store.get(a['frontier']),store.get(b['frontier'])
                    if a['choice']!=b['choice']:changed_choices.append(fb['round'])
                    if store.get(fa['epoch'])['graph']!=store.get(fb['epoch'])['graph']:changed_graphs.append(fb['round'])
                result={'status':'completed','label':label,'round':round_,'endpoint':state['endpoint'],
                    'cost':end['cost'],'cost_reduction':target['cost']-end['cost'],'receipts':receipt_id,
                    'same_endpoint_as_target':end==target,'exact_polynomial':end['exact_polynomial'],
                    'timing':clock,'causal_qualified':bool(end['exact_polynomial'] and timing),
                    'first_changed_choice_round':min(changed_choices,default=None),
                    'first_changed_graph_boundary_round':min(changed_graphs,default=None),
                    'choice':choices if choices is not None else cut.payload['choice']}
                return result
            except Exception:return {'status':'error','label':label,'round':round_,'error':traceback.format_exc()}
        for pos,decision in enumerate(target['decisions']):
            frontier=store.get(decision['frontier']);round_=frontier['round']
            state=engine.run(target['decisions'][:pos],pause_at=round_)
            if state['frontier']!=decision['frontier'] or state['choice']!=decision['choice']:
                raise ValueError('baseline cut reconstruction differs')
            session=Session(adapter,state);cut=session.capture(retain=False)
            cut_id=store.put({'manifest':cut.manifest(),'fingerprint':cut.fingerprint,'payload':cut.payload})
            cuts.append({'round':round_,'cut':cut_id,'frontier':state['frontier']})
            donor=frontier['proposals']['v1'];recipient=state['choice']
            rows.append(branch('coherent-v1-batch',round_,cut,donor))
            if name=='scalar-product':
                controls.append(branch('untouched',round_,cut))
                controls.append(branch('sham',round_,cut,recipient))
                controls.append(branch('A-B-A',round_,cut,donor,restore=True))
                controls.append(branch('same-membership-reversed',round_,cut,list(reversed(recipient))))
        summary=outcome_summary(rows,target['cost'])
        exchanges=[];doses=[];refusals=[];reciprocal=[]
        # Exploratory branch selection follows the observed trend even when a
        # claim remains unqualified; no proxy gate erases the follow-up evidence.
        first=summary['earliest_observed_cost_reduction_round']
        if name=='scalar-product' and first is not None:
            pos=next(i for i,d in enumerate(target['decisions']) if store.get(d['frontier'])['round']==first)
            state=engine.run(target['decisions'][:pos],pause_at=first)
            cut=Session(adapter,state).capture(retain=False);frontier=store.get(state['frontier'])
            recipient=state['choice'];donor=frontier['proposals']['v1']
            alternatives=[x for x in frontier['candidate_ids'] if x not in recipient and x not in donor]
            if len(alternatives)>=len(recipient):
                controls.append(branch('alternative-membership-batch',first,cut,alternatives[:len(recipient)]))
                for slot,candidate in enumerate(alternatives[:len(recipient)]):
                    choice=recipient.copy();choice[slot]=candidate
                    controls.append(branch('alternative-single-'+str(slot),first,cut,choice))
            else:controls.append({'status':'not-applicable','label':'alternative-membership','round':first,
                'reason':'fewer than one batch of candidates outside both policy choices'})
            for slot,candidate,choice in single_exchanges(recipient,donor):
                row=branch('single-exchange-'+str(slot)+'-'+str(frontier['candidate_ids'].index(candidate)),first,cut,choice)
                row.update(slot=slot,removed=recipient[slot],inserted=candidate);exchanges.append(row)
            # Coherent increasing-dose replacements preserve recipient slot order.
            differing=[i for i,x in enumerate(recipient) if x not in donor]
            inserts=[x for x in donor if x not in recipient]
            for dose in sorted({1,2,4,len(differing)}):
                if dose>len(differing) or not dose:continue
                choice=recipient.copy()
                for slot,value in zip(differing[:dose],inserts[:dose],strict=True):choice[slot]=value
                doses.append({**branch('coherent-dose-'+str(dose),first,cut,choice),'dose':dose})
            bad_choices={'wrong-length':recipient[:-1],
                'wrong-type':[0]*len(recipient),'wrong-parent':[('0'*64)]+recipient[1:]}
            if len(recipient)>1:bad_choices['duplicate']=[recipient[0]]*len(recipient)
            else:refusals.append({'case':'duplicate','status':'not-applicable','reason':'choice length is at most one'})
            for label,choice in bad_choices.items():
                try:Session.from_cut(adapter,cut).apply(Act.replace('choice',choice,name=label))
                except ValueError as error:refusals.append({'case':label,'refused':True,'reason':str(error)})
                else:raise ValueError('invalid choice was admitted: '+label)
            stale=copy.deepcopy(cut.payload)
            if stale['prefix']:stale['prefix']=[]
            else:stale['prefix']=[{'frontier':state['frontier'],'choice':recipient}]
            try:Session(adapter,stale)
            except ValueError as error:refusals.append({'case':'wrong-prefix','refused':True,'reason':str(error)})
            else:raise ValueError('stale prefix was admitted')
            # Reciprocal: V2a proposal on the V1 parent, then unchanged V1 suffix.
            v1=baselines['v1'];v1pos=next((i for i,d in enumerate(v1['decisions']) if store.get(d['frontier'])['round']==first),None)
            if v1pos is None:reciprocal.append({'status':'not-applicable','round':first,'reason':'V1 terminated before this boundary'})
            else:
                try:
                    v1state=engine.run(v1['decisions'][:v1pos],pause_at=first,suffix='v1')
                    v1adapter=ChoiceAdapter(engine,suffix='v1');v1session=Session(v1adapter,v1state)
                    v1cut=v1session.capture(retain=False);v1frontier=store.get(v1state['frontier'])
                    v1session.apply(Act.replace('choice',v1frontier['proposals']['v2a'],name='reciprocal-v2a-on-v1-parent'))
                    v1session.continue_();v1end=v1session.capture(retain=False).payload['endpoint']
                    reciprocal.append({'status':'completed','suffix':'v1','round':first,'endpoint':v1end,'cost':store.get(v1end)['cost'],
                        'baseline_cost':v1['cost'],'cut':store.put({'manifest':v1cut.manifest(),'fingerprint':v1cut.fingerprint,'payload':v1cut.payload}),
                        'receipts':store.put({'receipts':[r.to_dict() for r in v1session.receipts]})})
                except Exception:reciprocal.append({'status':'error','round':first,'error':traceback.format_exc()})
        result={'goal':name,'domain':domain,'expression':original,'context':engine.context,
            'baselines':baselines,'cuts':cuts,'batch_swaps':rows,'batch_summary':summary,'controls':controls,
            'single_exchanges':exchanges,'exchange_summary':outcome_summary(exchanges,target['cost']),
            'doses':doses,'refusals':refusals,'reciprocal':reciprocal}
        snapshot('causal-'+name,result);results.append(result)
    flush();memo.finish();snapshot('causal-field-lookups',list(lookup_rows.values()))
    snapshot('causal-scorer-audits',score_cache);snapshot('causal-numerics',numerics)
    audits=[{'context':e.context,'replays':e.replays,'wall_seconds_including_audits':e.seconds,'attempts':e.replay_audits} for e in engines]
    snapshot('causal-replay-audits',audits)
    return {'goals':[{k:r[k] for k in ('goal','batch_summary','exchange_summary')} for r in results],
        'branches':sum(len(r['batch_swaps'])+len(r['controls'])+len(r['single_exchanges'])+len(r['doses'])+len(r['reciprocal']) for r in results),
        'replays':sum(e.replays for e in engines),'physical_C_calls':executor.calls,'logical_C_calls':memo.logical,
        'unique_field_queries':len(lookup_rows),'field_logical_reads':sum(r['logical_reads'] for r in lookup_rows.values()),
        'field_all_equal':all(r['native_equal'] for r in lookup_rows.values()),
        'scorer_batches':len(score_cache),'scorer_max_abs':max(r['source_c_max_abs'] for r in score_cache.values()),
        'potentially_clock_confounded_attempts':sum(not a['search_clock_proven_unhit'] or not a.get('extraction_clock_proven_unhit',True)
            for e in engines for a in e.replay_audits),
        'contracts':{'mechanics':'sealed prefix and ordered graph/features; original matcher/rebuild/extractor; Saturn choice Acts; exact C memo',
            'instrument':'supplied V1 base library; V2a scorer features remapped by name; no V2a acquired-library claim',
            'consumer':'host search/FieldLM parser; C projected lookup, frozen scores and emitted numerical programs',
            'trend':'all signed branch costs preserved; earliest tested repairable batch boundary only',
            'terminal_status':'no training, promotion, general reasoning or unique failure-source claim'}}


if __name__=='__main__':raise SystemExit(main({'math-causal':run},retain_bundles=True,evidence_budget_bytes=256*1024**2))
