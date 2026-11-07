"""Finite math intake panel: frozen cells, live FieldLM reads and C-scored search."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
from program import lower_fx,canonical
from math_program import CExecutor,ProjectedField,digest,polynomial_audit,factor_audit
from intake_worker import main,snapshot,sha

RAW=Path('raw')
ARCHIVE=Path('/mnt/hdd8tb/cold-storage/mbp1/domains/2026-11-04-artifact-offload/model-experiments/experiments')
PINS={
 'scorer-v1':('2026-11-02-concept-graph-v1/results/main-v2/study__train__scorer.json','b1ec6adebad641f0ee51e2d9f97523e926e29f370bf939f14c0a6e9406b63861'),
 'scorer-v2a':('2026-11-02-concept-graph-v2a/results/main-v1/study__train__scorer.json','d4155d521cb0b71ff70d4842de1d40208e1a5ec23e4244a7c8bfe16b99c3b0ef'),
 'concepts':('2026-10-30-fieldlm-mathematical-concepts/results/transfer-main-v1/study__learning-101__frozen-learner.json','e85091819bd1e05c45c4357635714c8fdb69676d9597ef8909d275e8271624f3'),
 'metaconcepts':('2026-10-30-fieldlm-mathematical-metaconcepts/results/transfer-main-v1/study__learning-101__frozen.json','aa6fde9f31a977e74a4390766680bd2028f21a4cba987824d745426fb09eff6b')}

def acquire(name):
    relative,expected=PINS[name];source=ARCHIVE/relative;data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('archived source changed: '+name)
    (RAW/(name+'.source.json')).write_bytes(data)
    return json.loads(data),expected

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def trusted_class(path,name):
    tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    namespace={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

class RowCell(nn.Module):
    def __init__(self,cell):super().__init__();self.cell=cell
    def forward(self,x):return self.cell(x[:,:4],x[:,4:])

class Expression(nn.Module):
    def __init__(self,expression,names,cell=None):
        super().__init__();self.expression=expression;self.names=names;self.cell=cell;self.width=4 if cell else 1
        self.constants={}
        def constant(value):
            if value not in self.constants:
                name='constant_'+str(len(self.constants));self.constants[value]=name
                self.register_buffer(name,torch.tensor([[value,0,0,value]] if cell else [[value]],dtype=torch.float32))
        def scan(expr):
            if isinstance(expr,str):return
            if isinstance(expr,int):constant(expr);return
            if expr[0]=='const':
                constant(expr[1])
            else:
                for child in expr[1:]:scan(child)
        scan(expression)
    def forward(self,x):
        cache={}
        def evaluate(expr):
            key=json.dumps(expr,sort_keys=True)
            if key in cache:return cache[key]
            out=compute(expr);cache[key]=out;return out
        def compute(expr):
            if isinstance(expr,str):
                i=self.names.index(expr)*self.width;return x[:,i:i+self.width]
            if isinstance(expr,int):return getattr(self,self.constants[expr])
            op=expr[0]
            if op=='const':return getattr(self,self.constants[expr[1]])
            terms=[evaluate(child) for child in expr[1:]];out=terms[0]
            for term in terms[1:]:
                if op in ('add','sum'):out=out+term
                elif op=='sub':out=out-term
                elif op in ('mul','prod'):out=self.cell(out,term) if self.cell else out*term
                else:raise ValueError(op)
            return out
        return evaluate(self.expression)

def cells(executor):
    Product=trusted_class('sources/law.py','ProductMLP')
    rules=json.loads(Path('inputs/exact-rules.json').read_text())
    specimens=[('learned'+str(item['seed']),item['rule'],'inputs/exact-rules.json',False) for item in rules]
    for seed in (137,149,163):
        path=f'inputs/seed{seed}-bilinear7.json';specimens.append(('approx'+str(seed),json.loads(Path(path).read_text()),path,True))
    rng=np.random.default_rng(74031)
    basis=np.asarray([np.r_[np.eye(4)[i],np.eye(4)[j]] for i in range(4) for j in range(4)],dtype=np.float32)
    rows=np.concatenate([basis,rng.integers(-3,4,(256,8)).astype(np.float32)])
    result=[];models={}
    for name,rule,source,is_approx in specimens:
        print('PHASE compile and audit '+name,flush=True)
        original=Product(len(rule['u'])).double().eval()
        original.load_state_dict({key:torch.tensor(rule[key],dtype=torch.float64) for key in ('u','v','w')},strict=True)
        deployed=copy.deepcopy(original).float().eval();models[name]=deployed
        wrapper=RowCell(deployed);identity=sha(source)
        mapping=lower_fx(wrapper,8,identity,RAW/(name+'.tcm'),source_record={'artifact':source,'sha256':identity,
            'constructor_sha256':sha('sources/law.py'),'checkpoint_numeric_type':'FP64 JSON export' if is_approx else 'frozen integer rule',
            'deployment':'explicit FP32 conversion; no integer rounding or retraining','roles':['row-major left matrix','row-major right matrix']})
        with torch.no_grad():
            native=wrapper(torch.from_numpy(rows)).numpy()
            native64=RowCell(original)(torch.from_numpy(rows).double()).numpy()
        actual=executor.run(RAW/(name+'.tcm'),rows)
        truth=(rows[:,:4].reshape(-1,2,2)@rows[:,4:].reshape(-1,2,2)).reshape(-1,4)
        swap=executor.run(RAW/(name+'.tcm'),np.concatenate([rows[:,4:],rows[:,:4]],axis=1))
        a=rows[16:32,:4].copy();b=rows[16:32,4:].copy();nc=a.copy();cc=a.copy();curve=[]
        with torch.no_grad():
            for step in range(3):
                nc=deployed(torch.from_numpy(nc),torch.from_numpy(b)).numpy()
                cc=executor.run(RAW/(name+'.tcm'),np.concatenate([cc,b],axis=1))
                curve.append({'step':step+1,'max_abs_source_c':float(np.abs(nc-cc).max()),'c_outputs':cc.tolist(),'native_outputs':nc.tolist()})
        certificate=polynomial_audit(RAW/(name+'.tcm'),mapping);source_certificate=factor_audit(rule)
        snapshot(name+'-algebra',{'deployed':certificate,'original':source_certificate})
        np.savez_compressed(RAW/(name+'.npz'),inputs=rows,native_fp32=native,native_fp64=native64,c=actual,truth=truth,swapped=swap)
        item={'name':name,'source_sha256':identity,'operations':len(mapping['instructions']),
            'source_fp32_c_max_abs':float(np.abs(actual-native).max()),'source_fp32_c_mean_abs':float(np.abs(actual-native).mean()),
            'source_fp64_c_max_abs':float(np.abs(actual-native64).max()),'math_truth_c_max_abs':float(np.abs(actual-truth).max()),
            'swapped_operand_changed_rows':int(np.any(swap!=actual,axis=1).sum()),'own_output_continuation':curve,
            'original_coefficient_residuals':source_certificate['nonzero_residuals'],
            'deployed_coefficient_residuals':certificate.get('nonzero_residuals'),'deployed_classification':certificate['status']}
        snapshot(name+'-summary',item);result.append(item)
    return result,models

def feature_names(path):
    tree=ast.parse(Path(path).read_text());scope={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('RULES','FEATURES') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return list(scope['FEATURES'])

def scorers():
    out={}
    for version in ('v1','v2a'):
        Scorer=trusted_class('sources/'+version+'/train.py','Scorer')
        state,identity=acquire('scorer-'+version);names=feature_names('sources/'+version+'/scheduler.py')
        model=Scorer(len(names)).eval();model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in state.items()},strict=True)
        if not bool(torch.all(model.std>0)):raise ValueError('nonpositive scorer normalization')
        mapping=lower_fx(model,len(names),identity,RAW/('scorer-'+version+'.tcm'),source_record={
            'archived_checkpoint_sha256':identity,'feature_order':names,'feature_source_sha256':sha('sources/'+version+'/scheduler.py'),
            'constructor_sha256':sha('sources/'+version+'/train.py'),'normalization':'subtract then divide; no folding','version':version})
        out[version]=(model,names,mapping)
    return out

def numerical_pair(name,domain,before,after,goals,models,executor,origin=None):
    """Compile and consume both expressions with their own C numerical outputs."""
    matrix=domain=='matrix2-noncommutative';variables=sorted(goals.variables(before));width=4 if matrix else 1
    rows=np.random.default_rng(741).integers(-2,3,(32,len(variables)*width)).astype(np.float32)
    numerical=[]
    for label,expr in (('before',before),('after',after)):
        module=Expression(expr,variables,models['learned811'] if matrix else None)
        path=RAW/(name+'-'+label+'.tcm')
        mapping=lower_fx(module,rows.shape[1],digest({'expression':expr,'domain':domain,'origin':origin,
            'cell':'learned811' if matrix else None}),path,source_record={'expression':expr,'domain':domain,
            'variable_order':variables,'host_source':origin,'cell_certificate':'learned811-algebra.json' if matrix else None})
        with torch.no_grad():native=module(torch.from_numpy(rows)).numpy()
        actual=executor.run(path,rows)
        np.savez_compressed(RAW/(name+'-'+label+'.npz'),inputs=rows,native_fp32=native,c=actual)
        macs=sum(mapping['values'][o['a']]['length']*mapping['values'][o['out']]['length'] for o in mapping['instructions'] if o['kind']=='linear')
        numerical.append({'linear_macs_per_row':macs,'operations':len(mapping['instructions']),
            'source_c_max_abs':float(np.abs(actual-native).max()),'outputs':actual.tolist()})
    return {'before':numerical[0],'after':numerical[1],
        'c_before_after_max_abs':float(np.max(np.abs(np.asarray(numerical[0]['outputs'])-np.asarray(numerical[1]['outputs']))))}

def frozen_fields(goals,models,executor,lookups):
    """Use archived learned bodies through original FieldFabric consumers."""
    store_module=load('field_store','sources/field_store.py')
    algebra=load('algebra','sources/algebra.py');concepts=load('concepts','sources/concepts.py')
    dimensions=load('native_dimensions','sources/native_dimensions.py')
    def attach(view):
        reader=ProjectedField(view,RAW/'field-banks',executor);view.lookup=reader.lookup;lookups.append(reader);return reader
    def cold(fabric):
        # Only derived query results are evicted, on a fresh restored arm.
        fabric.abstractions.clear();fabric.frames.clear()
    payload,identity=acquire('concepts')
    native=store_module.Store.from_state(copy.deepcopy(payload['store']))
    compiled=store_module.Store.from_state(copy.deepcopy(payload['store']))
    if native.backend!='field' or compiled.backend!='field':raise ValueError('frozen concept store is not native FieldLM')
    cold(native.fabric);cold(compiled.fabric);reader=attach(compiled.fabric.view(store_module.VIEW_NAME))
    expected=native.list(domain=None,kind=None);actual=compiled.list(domain=None,kind=None)
    if not expected or expected!=actual:raise ValueError('frozen concept bodies changed on C read')
    cases=[]
    for i,rule in enumerate(actual):
        bindings={key:'fresh'+str(j) for j,key in enumerate(rule['parameters'])}
        expression=algebra.substitute(rule['lhs'],bindings)
        left=concepts.optimize([expression],native,max_steps=12);right=concepts.optimize([expression],compiled,max_steps=12)
        row={'id':rule['id'],'parents':rule['parents'],'source_sha256':identity,'expression':expression,
            'native_c_consumer_equal':left['outputs']==right['outputs'] and left['accepted']==right['accepted'],
            'exact_polynomial':algebra.equivalent(expression,right['outputs'][0]),'optimization':right}
        row.update(numerical_pair('concept-'+str(i),'integer-commutative',expression,right['outputs'][0],goals,models,executor,
            {'artifact_sha256':identity,'concept':rule['id'],'consumer_sha256':sha('sources/concepts.py')}))
        cases.append(row);snapshot('concept-'+str(i)+'-summary',row)
    wrong=[compiled.get(actual[0]['id'],namespace=compiled.namespace+'-wrong'),
           compiled.get(actual[0]['id'],scope=compiled.scope+'-wrong'),compiled.get('absent-concept')]
    concept_result={'source_sha256':identity,'concept_count':len(actual),'bodies_equal':expected==actual,
        'rewrite_count':sum(r['kind']=='rewrite' for r in actual),'primitive_count':sum(r['kind']!='rewrite' for r in actual),
        'cases':cases,'wrong_namespace_scope_missing_returns':wrong,'C_logical_reads':reader.calls}
    snapshot('concept-store-summary',concept_result)
    payload,identity=acquire('metaconcepts')
    native=dimensions.DimensionalLibrary.from_state(copy.deepcopy(payload['library']))
    compiled=dimensions.DimensionalLibrary.from_state(copy.deepcopy(payload['library']))
    if native.backend!='field' or compiled.backend!='field':raise ValueError('frozen law library is not native FieldLM')
    cold(native.fabric);cold(compiled.fabric);catalog=attach(compiled.catalog_view);graph=attach(compiled.graph_view)
    cases=[];catalogs=[]
    def substitute(expr,bindings):
        if isinstance(expr,str):return copy.deepcopy(bindings.get(expr,expr))
        if isinstance(expr,int):return expr
        return [expr[0],*[substitute(child,bindings) for child in expr[1:]]]
    for i,(domain,op) in enumerate(sorted({(r['domain'],r['lhs'][0]) for r in native.rules})):
        left=native.lookup(domain,op);right=compiled.lookup(domain,op)
        catalogs.append({'domain':domain,'root_op':op,'native_c_equal':left==right,'ids':[r['id'] for r in right]})
        for j,rule in enumerate(right):
            placeholders=sorted(dimensions._placeholders(rule['lhs']))
            expression=substitute(rule['lhs'],{key:'fresh'+str(k) for k,key in enumerate(placeholders)})
            left_matches=native.match(rule,expression,domain);right_matches=compiled.match(rule,expression,domain)
            row={'id':rule['id'],'domain':domain,'expression':expression,'native_c_matches_equal':left_matches==right_matches,
                'matches':right_matches,'wrong_domain_matches':compiled.match(rule,expression,'wrong-domain')}
            if right_matches:
                replacement=substitute(rule['rhs'],right_matches[0]);row['replacement']=replacement
                row['exact_polynomial']=goals.poly(expression,domain)==goals.poly(replacement,domain)
                row.update(numerical_pair('law-'+str(i)+'-'+str(j),domain,expression,replacement,goals,models,executor,
                    {'artifact_sha256':identity,'rule':rule['id'],'consumer_sha256':sha('sources/native_dimensions.py')}))
            # A changed repeated placeholder and reversed matrix ports are observed controls.
            def change_one(expr,needle):
                if expr==needle:return 'mismatched-placeholder',True
                if isinstance(expr,(str,int)):return expr,False
                out=list(expr)
                for k in range(1,len(out)):
                    out[k],changed=change_one(out[k],needle)
                    if changed:return out,True
                return out,False
            if placeholders:
                changed,_=change_one(expression,'fresh0')
                a=native.match(rule,changed,domain);b=compiled.match(rule,changed,domain)
                row['changed_placeholder_control']={'native_c_equal':a==b,'matches':b,'expression':changed}
            if domain=='matrix2-noncommutative':
                def reverse(expr):
                    if isinstance(expr,(str,int)):return expr
                    children=[reverse(x) for x in expr[1:]]
                    return [expr[0],*(children[::-1] if expr[0]=='mul' else children)]
                changed=reverse(expression);a=native.match(rule,changed,domain);b=compiled.match(rule,changed,domain)
                row['reversed_ports_control']={'native_c_equal':a==b,'matches':b,'expression':changed}
            cases.append(row);snapshot('law-'+str(i)+'-'+str(j)+'-summary',row)
    result={'source_sha256':identity,'rule_count':len(native.rules),'catalogs':catalogs,'cases':cases,
        'C_catalog_logical_reads':catalog.calls,'C_graph_logical_reads':graph.calls}
    snapshot('metaconcept-summary',result)
    return {'concepts':concept_result,'metaconcepts':result}

def run():
    torch.set_num_threads(1);executor=CExecutor('build/tcm',RAW/'c-calls.jsonl')
    cell_results,models=cells(executor);score_models=scorers()
    cg=load('concept_graph','sources/v1/concept_graph.py')
    sch=load('scheduler','sources/v1/scheduler.py');goals=load('goals','sources/v1/goals.py')
    native_store=load('native_store','sources/v1/native_store.py')
    panels=[];lookups=[];scores=[]
    fields=frozen_fields(goals,models,executor,lookups)
    class CIndex(native_store.NativeIndex):
        def __init__(self,g):
            super().__init__(g);reader=ProjectedField(self.view,RAW/'field-banks',executor)
            self.view.lookup=reader.lookup;lookups.append(reader)
    expressions=[('scalar-product',cg.SCALAR,goals.binomial_goal(3)[0][0]),
        ('scalar-planted',cg.SCALAR,goals.planted_goal(74117,n_vars=4,n_ops=5,max_monos=24)[0][0]),
        ('ordered-matrix',cg.MATRIX,['add',['add',['mul','A','C'],['mul','B','C']],['add',['mul','A','D'],['mul','B','D']]])]
    def search(name,domain,expression,arm):
            print('PHASE FieldLM search '+name+' '+arm,flush=True)
            graph=cg.EGraph(domain,'ac');roots=[goals._insert_nary(graph,expression)];graph.rebuild();selection=[]
            class Policy:
                def __init__(self):self.remaining=48;self.rng=np.random.default_rng(74031)
                def select(self,g,roots,metas,X,round_):
                    if arm=='enumeration':order=np.arange(len(metas))
                    elif arm=='random':order=self.rng.permutation(len(metas))
                    elif arm=='heuristic':order=np.argsort(-sch.heuristic_score(X),kind='stable')
                    else:
                        version='v2a' if arm=='c-v2a-base-library' else 'v1';model,names,_=score_models[version]
                        source={key:i for i,key in enumerate(sch.FEATURES)}
                        rows=np.column_stack([X[:,source[key]] if key in source else np.zeros(len(X),dtype=np.float32) for key in names]).astype(np.float32)
                        expected=model.score_numpy(rows)
                        actual=executor.run(RAW/('scorer-'+version+'.tcm'),rows).reshape(-1)
                        which=expected if arm=='native-v1' else actual
                        order=np.argsort(-which,kind='stable')
                        native_order=np.argsort(-expected,kind='stable');c_order=np.argsort(-actual,kind='stable')
                        k=min(self.remaining,len(order),8)
                        scores.append({'goal':name,'arm':arm,'round':round_,'version':version,'features':rows.tolist(),
                            'native':expected.tolist(),'c':actual.tolist(),'native_order':np.argsort(-expected,kind='stable').tolist(),
                            'c_order':c_order.tolist(),'selected_count':k,
                            'native_selected':native_order[:k].tolist(),'c_selected':c_order[:k].tolist(),
                            'selected_order_equal':bool(np.array_equal(native_order[:k],c_order[:k])),
                            'full_order_equal':bool(np.array_equal(native_order,c_order)),
                            'native_cutoff_margin':float(expected[native_order[k-1]]-expected[native_order[k]]) if 0<k<len(order) else None,
                            'c_cutoff_margin':float(actual[c_order[k-1]]-actual[c_order[k]]) if 0<k<len(order) else None})
                    count=min(self.remaining,len(order),8);chosen=order[:count]
                    self.remaining-=count;selection.append({'round':round_,'candidates':len(metas),'chosen':chosen.tolist(),
                        'chosen_meta':[metas[int(i)] for i in chosen]})
                    return [int(x) for x in chosen]
            policy=Policy();start=time.perf_counter()
            result=cg.saturate(graph,roots,'ac',index_cls=CIndex if arm.startswith('c-') else native_store.NativeIndex,
                scheduler=policy,featurize=sch.featurize,node_cap=768,time_cap=120,iter_cap=6,match_cap=1024)
            choice=cg.extract_dag(graph,roots,passes=1,budget_s=0.5);extracted=cg.to_expr(graph,choice,roots[0]);cost,ops=cg.dag_cost(graph,choice,roots)
            exact=goals.poly(expression,domain)==goals.poly(extracted,domain)
            numerical=numerical_pair(name+'-'+arm,domain,expression,extracted,goals,models,executor,
                {'host_search':'supplied V1 base library','scheduler':arm,'source_sha256':sha('sources/v1/concept_graph.py')})
            item={'goal':name,'arm':arm,'domain':domain,'expression':expression,'extracted':extracted,'exact_polynomial':exact,
                'final_cost':cost,'final_ops':dict(ops),'status':result['status'],'trace':result['trace'],'selection':selection,
                'selection_budget_used':48-policy.remaining,'applications':sum(row['applied'] for row in result['trace']),
                'wall_seconds_including_C_and_extraction':time.perf_counter()-start,**numerical}
            return item
    for name,domain,expression in expressions:
        for arm in ('native-v1','c-v1','c-v2a-base-library','heuristic','random','enumeration'):
            try:item=search(name,domain,expression,arm)
            except Exception:item={'goal':name,'arm':arm,'status':'error','error':traceback.format_exc()}
            snapshot(name+'-'+arm+'-summary',item);panels.append(item)
    snapshot('field-lookups',{'epochs':[{'calls':r.calls,'banks':len(r.banks),'comparisons':r.comparisons} for r in lookups]})
    snapshot('scorer-rows',scores)
    return {'cells':cell_results,'frozen_fields':fields,'search':panels,'lookup_calls':sum(r.calls for r in lookups),
        'lookup_all_equal':all(row['native_equal'] for reader in lookups for row in reader.comparisons),
        'scorer_batches':len(scores),'scorer_max_abs':max((max(abs(a-b) for a,b in zip(row['native'],row['c'],strict=True)) for row in scores),default=None),
        'scorer_selection_order_mismatched_batches':sum(not row['selected_order_equal'] for row in scores),
        'search_completed':sum(row['status']!='error' for row in panels),'search_errors':sum(row['status']=='error' for row in panels),
        'C_calls':executor.calls,'C_wall_seconds_including_processes':executor.seconds,
        'contracts':{'host':'FieldLM parsing, frame extension, rewrite rules, features, stable selection, congruence and extraction',
          'C':'projected record filtering, frozen learned scoring and emitted numerical programs',
          'budgets':'6 rounds,8 selections/round,48 selected applications,768 nodes,1024 matches/round for every arm; final extraction1pass,0.5s; search timeout120s safety bound; enumeration preserves original candidate order',
          'v2a':'23-feature scorer on named V1 base-library features; no V2a acquired divide/expand library claim',
          'training':'none; archived immutable learned artifacts','terminal_status':'bounded integration and observations; no general reasoning or speedup claim'}}

if __name__=='__main__':raise SystemExit(main({'math':run},retain_bundles=True))
