"""Saturn choice cuts over the original equality-saturation consumer.

A cut seals a replay capsule, not a pickled e-graph. Reconstruction reruns the
original engine and checks every ordered frontier before consuming a choice.
Wall clocks and physical audit/cache counters deliberately live outside cuts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from saturn_pub import (Adapter, ExecutionPoint, NumericalContract, SlotSpec,
                        SurfaceManifest, TransitionSpec)
from math_program import digest
from program import canonical


def typed(value):
    """Witness raw ordering/types without invoking mutating graph helpers."""
    if isinstance(value, dict):return {'dict': [[typed(k), typed(v)] for k,v in value.items()]}
    if isinstance(value, tuple):return {'tuple': [typed(v) for v in value]}
    if isinstance(value, set):return {'set_iteration': [typed(v) for v in value]}
    if isinstance(value, list):return [typed(v) for v in value]
    if value is None or type(value) in (str,int,bool,float):return value
    raise TypeError('unsupported graph witness type: '+type(value).__name__)


class Witnesses:
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.writes=0

    def put(self, body):
        identity=digest(body);path=self.root/(identity+'.json');data=canonical(body)+b'\n'
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('immutable witness changed')
        else:path.write_bytes(data);self.writes+=1
        return identity

    def get(self, identity):
        if type(identity) is not str or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('invalid witness identity')
        body=json.loads((self.root/(identity+'.json')).read_text())
        if digest(body)!=identity:raise ValueError('stale witness content')
        return body


class Paused(Exception):
    def __init__(self, state):self.state=state


class ReplayEngine:
    CAPS={'rounds':6,'per_round':8,'selections':48,'nodes':768,'matches':1024,
          'search_seconds':120,'extract_passes':1,'extract_seconds':.5}

    def __init__(self,cg,goals,featurize,index_cls,score,store,expression,domain,identity,
                 *,consume=None,caps=None):
        self.cg,self.goals,self.featurize,self.index_cls=cg,goals,featurize,index_cls
        self.score,self.store,self.consume=score,store,consume
        self.expression,self.domain=copy.deepcopy(expression),domain
        self.caps={**self.CAPS,**(caps or {})}
        self.context=digest({'executor':identity,'expression':expression,'domain':domain,'caps':self.caps})
        self.replays=0;self.seconds=0.;self.replay_audits=[]

    def validate_choice(self, frontier, choice):
        body=self.store.get(frontier)
        if body['context']!=self.context:raise ValueError('wrong replay context')
        allowed=body['candidate_ids']
        if (type(choice) is not list or len(choice)!=body['count'] or
            any(type(i) is not str or i not in allowed for i in choice) or len(set(choice))!=len(choice)):
            raise ValueError('choice must be a unique, same-length list from this frontier')
        return body

    def run(self,prefix=(),*,pause_at=None,suffix='v2a'):
        """Original matcher, application order, rebuild and extractor; no forked engine."""
        cg=self.cg;caps=self.caps;engine=self;prefix=copy.deepcopy(list(prefix));decisions=[]
        graph=cg.EGraph(self.domain,'ac')
        roots=[self.goals._insert_nary(graph,self.expression)];graph.rebuild()
        remaining=caps['selections'];targets=[];started=time.perf_counter();self.replays+=1
        def features(g,rs,metas,ts,round_):
            targets[:]=ts
            return engine.featurize(g,rs,metas,ts,round_)
        class Policy:
            def select(self,g,rs,metas,X,round_):
                nonlocal remaining
                rows=np.ascontiguousarray(X,dtype='<f4')
                raw=rows.tobytes();feature_sha=hashlib.sha256(raw).hexdigest()
                path=engine.store.root/(feature_sha+'.f32')
                if path.exists():
                    if path.read_bytes()!=raw:raise ValueError('feature witness changed')
                else:path.write_bytes(raw)
                expected_fields={'domain','mode','parent','nodes','memo','const_classes','n_nodes',
                                 'created','deadline','match_limit'}
                if not set(vars(g))<=expected_fields:raise ValueError('unsealed mutable graph field')
                parent={'domain':g.domain,'mode':g.mode,'parent':typed(g.parent),'nodes':typed(g.nodes),
                    'memo':typed(g.memo),'const_classes':typed(g.const_classes),'n_nodes':g.n_nodes,
                    'created':typed(g.created),'roots':typed(rs)}
                epoch={'context':engine.context,'round':round_,'remaining':remaining,
                    'prefix_sha256':digest(decisions),'graph':parent,'targets':typed(targets),
                    'metas':typed(metas),'features':{'sha256':feature_sha,'shape':list(rows.shape),'dtype':'<f4'}}
                epoch_id=engine.store.put(epoch)
                ids=[digest({'epoch':epoch_id,'ordinal':i}) for i in range(len(metas))]
                count=min(remaining,len(metas),caps['per_round'])
                proposals=engine.score(rows,count)
                for order in proposals.values():
                    if (type(order) is not list or len(order)!=count or
                        any(type(i) is not int or not 0<=i<len(metas) for i in order) or len(set(order))!=count):
                        raise ValueError('invalid policy proposal')
                body={'context':engine.context,'epoch':epoch_id,'round':round_,'remaining':remaining,
                    'prefix_sha256':digest(decisions),'candidate_ids':ids,'count':count,
                    'proposals':{name:[ids[i] for i in order] for name,order in proposals.items()}}
                frontier=engine.store.put(body)
                if len(decisions)<len(prefix):
                    recorded=prefix[len(decisions)]
                    if recorded['frontier']!=frontier:raise ValueError('prefix replay frontier differs')
                    chosen=recorded['choice'];engine.validate_choice(frontier,chosen)
                else:
                    chosen=body['proposals'][suffix]
                    if pause_at==round_:
                        raise Paused({'context':engine.context,'prefix':copy.deepcopy(decisions),
                            'frontier':frontier,'choice':chosen,'endpoint':None})
                decisions.append({'frontier':frontier,'choice':copy.deepcopy(chosen)})
                remaining-=count
                return [ids.index(i) for i in chosen]
        try:
            search=cg.saturate(graph,roots,'ac',index_cls=self.index_cls,scheduler=Policy(),
                featurize=features,node_cap=caps['nodes'],time_cap=caps['search_seconds'],
                iter_cap=caps['rounds'],match_cap=caps['matches'],trace_extract=True)
            search_wall=time.perf_counter()-started
            if len(decisions)<len(prefix):raise ValueError('prefix extends beyond consumer termination')
            if pause_at is not None:raise ValueError('requested choice boundary was not reached')
            graph_before_extract=digest({'parent':typed(graph.parent),'nodes':typed(graph.nodes),'memo':typed(graph.memo)})
            t=time.perf_counter()
            choice=cg.extract_dag(graph,roots,passes=caps['extract_passes'],budget_s=caps['extract_seconds'])
            extract_wall=time.perf_counter()-t
            expression=cg.to_expr(graph,choice,roots[0]);cost,ops=cg.dag_cost(graph,choice,roots)
            # Wall less than the earliest internal threshold proves clock limits
            # unhit. Longer executions stay visible as potentially confounded.
            clocks={'search_wall_seconds':search_wall,'extraction_wall_seconds':extract_wall,
                'search_clock_proven_unhit':search_wall<caps['search_seconds'] and search['status']!='time-cap',
                'extraction_clock_proven_unhit':extract_wall<caps['extract_seconds']/2}
            body={'context':self.context,'decisions':decisions,'expression':expression,'cost':cost,
                'ops':dict(ops),'graph_before_extract':graph_before_extract,
                'exact_polynomial':self.goals.poly(self.expression,self.domain)==self.goals.poly(expression,self.domain),
                'selection_budget_used':caps['selections']-remaining,
                'applications':sum(row['applied'] for row in search['trace']),
                'search_status':search['status'],
                'trace':[{k:v for k,v in row.items() if k!='seconds'} for row in search['trace']]}
            if self.consume:body['numerical']=self.consume(expression)
            endpoint=self.store.put(body)
            self.replay_audits.append({'endpoint':endpoint,**clocks})
            return {'context':self.context,'prefix':decisions,'frontier':None,'choice':[],'endpoint':endpoint}
        except Paused as pause:
            self.replay_audits.append({'frontier':pause.state['frontier'],'search_wall_seconds':time.perf_counter()-started,
                'search_clock_proven_unhit':time.perf_counter()-started<caps['search_seconds']})
            return pause.state
        finally:self.seconds+=time.perf_counter()-started


class ChoiceAdapter(Adapter):
    """Only the current choice is writable; native continuation replays the prefix."""
    def __init__(self,engine,*,suffix='v2a'):
        self.engine,self.suffix=engine,suffix;self.model_identity=engine.context
        self.execution={'family':'fieldlm-math-search','adapter':'prefix-replay-choice-v1',
            'suffix':suffix,'caps':engine.caps,'closure':'sealed decisions + verified ordered graph/frontier/features',
            'physical_work':'reconstruction, audits, clocks and exact C cache outside rollback'}

    def addresses(self,state):return ('context','prefix','frontier','choice','endpoint')
    def boundary(self,state):return 'complete' if state['endpoint'] else 'rewrite-choice'
    def validate(self,state):
        if set(state)!=set(self.addresses(state)) or state['context']!=self.engine.context:
            raise ValueError('invalid replay closure')
        if type(state['prefix']) is not list:raise ValueError('invalid replay prefix')
        for entry in state['prefix']:
            if set(entry)!= {'frontier','choice'}:raise ValueError('invalid prefix decision')
            self.engine.validate_choice(entry['frontier'],entry['choice'])
        if state['endpoint']:
            endpoint=self.engine.store.get(state['endpoint'])
            if (state['frontier'] is not None or state['choice']!=[] or
                endpoint['context']!=self.engine.context or endpoint['decisions']!=state['prefix']):
                raise ValueError('invalid completed replay')
        else:
            frontier=self.engine.validate_choice(state['frontier'],state['choice'])
            if frontier['prefix_sha256']!=digest(state['prefix']):raise ValueError('wrong prefix for frontier')

    def surface(self,state):
        return SurfaceManifest(tuple(SlotSpec(name,writable=name=='choice' and not state['endpoint'],
            role='control' if name=='choice' else 'runtime') for name in self.addresses(state)),
            'fieldlm-prefix-capsule-v1','original-saturate-extract-C')
    def point(self,state):
        round_=self.engine.store.get(state['frontier'])['round'] if state['frontier'] else self.engine.caps['rounds']
        return ExecutionPoint('fieldlm-math-search',round_,self.boundary(state),
            operator='rewrite-batch',next_operation='original-search-suffix')
    def transition(self,state):
        return TransitionSpec('original-search-suffix',reads=self.addresses(state),
            writes=('prefix','frontier','choice','endpoint'),consumer='original-saturate-extract-C',footprint='declared')
    def numerical_contract(self):
        return NumericalContract(program='original-search-and-C-fp32',comparator='exact-prefix-and-consumer-witness')
    def advance(self,state):
        if state['endpoint']:raise ValueError('search already complete')
        self.validate(state)
        prefix=state['prefix']+[{'frontier':state['frontier'],'choice':state['choice']}]
        return self.engine.run(prefix,suffix=self.suffix)


def single_exchanges(recipient,donor):
    """All donor-only insertions at each recipient-only slot, preserving order."""
    return [(slot,candidate,recipient[:slot]+[candidate]+recipient[slot+1:])
            for slot,old in enumerate(recipient) if old not in donor
            for candidate in donor if candidate not in recipient]


def outcome_summary(rows,baseline_cost):
    """No monotonicity assumption or promotion gate; retain signed effects."""
    valid=[r for r in rows if r.get('status')=='completed']
    improved=[r for r in valid if r['cost']<baseline_cost]
    qualified=[r for r in improved if r.get('causal_qualified',False)]
    return {'completed':len(valid),'errors':len(rows)-len(valid),
        'improved':len(improved),'equal':sum(r['cost']==baseline_cost for r in valid),
        'worse':sum(r['cost']>baseline_cost for r in valid),
        'earliest_observed_cost_reduction_round':min((r['round'] for r in improved),default=None),
        'qualified_repairs':len(qualified),
        'earliest_tested_repairable_round':min((r['round'] for r in qualified),default=None),
        'minimum_observed_cost':min((r['cost'] for r in valid),default=None)}
