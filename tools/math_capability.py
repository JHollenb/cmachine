"""Frozen-coordinate discovery and explicit unions for the math capability assay.

Selection sees discovery frontiers only. No coefficients are fitted; exported
subnetworks copy original units in original order. A structural proposal policy
is supplied host computation, separately accounted from learned parameters.
"""
from __future__ import annotations
import ast
from collections.abc import Mapping,Sequence
import gzip
import itertools
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from math_program import digest
from math_debugger import Witnesses
from program import canonical


class PackedWitnesses(Witnesses):
    """Same content identity/validation as choice cuts; compressed raw storage."""
    def put(self,body):
        identity=digest(body);path=self.root/(identity+'.json.gz');data=canonical(body)+b'\n'
        if path.exists():
            if gzip.decompress(path.read_bytes())!=data:raise ValueError('immutable witness changed')
        else:path.write_bytes(gzip.compress(data,mtime=0));self.writes+=1
        return identity
    def get(self,identity):
        if type(identity) is not str or len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('invalid witness identity')
        body=json.loads(gzip.decompress((self.root/(identity+'.json.gz')).read_bytes()))
        if digest(body)!=identity:raise ValueError('stale witness content')
        return body


def coalition_catalog(path,rows,cap):
    """Run the canonical manalysis set-union implementation from pinned source."""
    tree=ast.parse(Path(path).read_text());names={'_normalize_family','_family_key','build_family_coalition_catalog'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):raise ValueError('canonical coalition API changed')
    scope={'Any':object,'Mapping':Mapping,'Sequence':Sequence,'combinations':itertools.combinations}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['build_family_coalition_catalog'](rows,max_order=2,max_components=cap)


class Hidden(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model.net[1](self.model.net[0]((x-self.model.mean)/self.model.std))


def subnetwork(model,units):
    if not units or len(set(units))!=len(units) or any(type(i) is not int or not 0<=i<32 for i in units):
        raise ValueError('invalid hidden-unit subset')
    units=sorted(units)
    result=type(model)(len(model.mean),hidden=len(units)).eval()
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    state['net.0.weight']=state['net.0.weight'][units];state['net.0.bias']=state['net.0.bias'][units]
    state['net.2.weight']=state['net.2.weight'][:,units]
    result.load_state_dict(state,strict=True)
    return result


def greedy_units(contributions,targets,weights,limit=16):
    """Greedy deletion-complement search on centered discovery score error.

    Every accepted/rejected candidate loss is retained. Loss chooses which
    frozen coordinate to add; it does not certify an execution circuit.
    """
    a=np.asarray(contributions,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if (a.ndim!=2 or y.shape!=(len(a),) or w.shape!=y.shape or not np.isfinite(a).all()
        or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0):
        raise ValueError('invalid discovery arrays')
    w=w/w.sum()
    selected=[];prediction=np.zeros(len(a));trace=[]
    for step in range(min(limit,a.shape[1])):
        losses={i:float(np.sum(w*(prediction+a[:,i]-y)**2)) for i in range(a.shape[1]) if i not in selected}
        winner=min(losses,key=lambda i:(losses[i],i));selected.append(winner);prediction+=a[:,winner]
        trace.append({'step':step+1,'added':winner,'loss':losses[winner],
            'candidates':[{'unit':i,'loss':v} for i,v in losses.items()]})
    return selected,trace


def proposal_union(order,X,names,count,quota=1,*,alternative=False,seed=0):
    """Reserve bounded slots for broader candidates from this actual frontier."""
    chosen=list(order[:count]);columns={name:i for i,name in enumerate(names)}
    factor=X[:,columns['rule_factor']]>0
    broad=factor & (X[:,columns['log_group']]>np.float32(np.log1p(2)))
    pool=[int(i) for i in order if i not in chosen and bool(broad[i])]
    if alternative:
        pool=[int(i) for i in order if i not in chosen and not bool(broad[i])]
        random.Random(seed).shuffle(pool)
    else:
        # The original neural order breaks ties; labels, donors and task answers
        # are unavailable to this proposal rule.
        rank={v:i for i,v in enumerate(order)}
        pool.sort(key=lambda i:(-float(X[i,columns['target_on_route']]),-float(X[i,columns['log_group']]),rank[i]))
    replacements=[]
    for slot,new in zip(range(count-1,max(-1,count-quota-1),-1),pool):
        replacements.append({'slot':slot,'removed':chosen[slot],'inserted':new});chosen[slot]=new
    if len(chosen)!=count or len(set(chosen))!=count:raise ValueError('proposal union violated selection budget')
    return chosen,replacements


def structural_order(X,names):
    """Scorer-free broader-first rule; unselected ordinary rows fill by ordinal."""
    columns={name:i for i,name in enumerate(names)}
    broad=(X[:,columns['rule_factor']]>0)&(X[:,columns['log_group']]>np.float32(np.log1p(2)))
    return sorted(range(len(X)),key=lambda i:(0,-float(X[i,columns['target_on_route']]),
        -float(X[i,columns['log_group']]),i) if broad[i] else (1,0,0,i))


def randomized_goal(expression,seed,*,commutative=True):
    rng=random.Random(seed)
    variables=set()
    def scan(e):
        if isinstance(e,str):variables.add(e)
        elif e[0]!='const':
            for child in e[1:]:scan(child)
    scan(expression);labels=list(sorted(variables));rng.shuffle(labels)
    mapping={key:'s'+str(seed)+'_'+str(i) for i,key in enumerate(labels)}
    def copy(e):
        if isinstance(e,str):return mapping[e]
        if e[0]=='const':return list(e)
        children=[copy(c) for c in e[1:]]
        if e[0] in ('add','sum') or commutative and e[0] in ('mul','prod'):rng.shuffle(children)
        return [e[0],*children]
    return copy(expression)


def validate_input_expression(expression):
    """The native insertion API accepts binary input trees, not extracted n-ary trees."""
    if isinstance(expression,str):return
    if not isinstance(expression,list) or not expression:raise ValueError('invalid input expression')
    if expression[0]=='const':
        if len(expression)!=2 or type(expression[1]) is not int:raise ValueError('invalid input constant')
        return
    if len(expression)!=3 or expression[0] not in ('add','sub','mul'):raise ValueError('input expression must be binary add/sub/mul')
    for child in expression[1:]:validate_input_expression(child)


def corpus(goals,cg):
    """Seed banks fixed before discovery; test goals never select units/policies."""
    discovery=[];test=[]
    for split,base,target in [('discovery',31100,discovery),('test',71900,test)]:
        for j,n in enumerate((2,3,4)):
            for rep in range(2):
                seed=base+100*j+rep;expression,reference=goals.binomial_goal(n)
                target.append({'id':f'{split}-binomial-{n}-{rep}','family':'binomial','seed':seed,
                    'domain':cg.SCALAR,'expression':randomized_goal(expression[0],seed),'reference_cost':reference,
                    'reference_kind':'cost of supplied factored expression; no global minimum claim'})
        for j in range(6):
            seed=base+1000+j;expression,reference,planted=goals.planted_goal(seed,n_vars=4,n_ops=6,max_monos=24)
            target.append({'id':f'{split}-planted-{j}','family':'planted','seed':seed,'domain':cg.SCALAR,
                'expression':expression[0],'reference_cost':reference,'reference_expression':planted,
                'reference_kind':'supplied planted circuit upper bound'})
    # Out-of-construction transfer arms. Names/order alter scalar layout; matrix
    # products retain their ordered ports.
    for n in (2,3):
        expression,reference=goals.trinomial_goal(n)
        test.append({'id':f'test-trinomial-{n}','family':'trinomial','seed':81900+n,'domain':cg.SCALAR,
            'expression':randomized_goal(expression[0],81900+n),'reference_cost':reference,
            'reference_kind':'supplied factored expression'})
    for domain in (cg.SCALAR,cg.MATRIX):
        for label,expressions in goals.held_graphs(seed=82901):
            if label not in ('right-factor-add','right-factor-sub','context-congruence','no-shared-factor','mixed-factor-sides'):continue
            test.append({'id':'test-'+domain+'-'+label,'family':'ordered-matrix' if domain==cg.MATRIX else 'held-shape',
                'seed':82901,'domain':domain,'expression':expressions[0],
                'reference_cost':goals.HELD_PRIOR.get((domain,label)),
                'reference_kind':'historical supplied-family reference; not a fresh-task oracle'})
    return discovery,test


def replication_corpus(goals,cg):
    """Six follow-up rows, fixed after the first exploratory panel was inspected."""
    rows=[]
    expression,reference=goals.trinomial_goal(3)
    rows.append({'id':'replication-trinomial-3','family':'trinomial','seed':131903,'domain':cg.SCALAR,
        'expression':randomized_goal(expression[0],131903),'reference_cost':reference,
        'reference_kind':'supplied factored expression; fresh layout of known construction'})
    for seed in (142301,142302):
        expression,reference,planted=goals.planted_goal(seed,n_vars=4,n_ops=6,max_monos=24)
        rows.append({'id':'replication-planted-'+str(seed),'family':'planted','seed':seed,'domain':cg.SCALAR,
            'expression':expression[0],'reference_cost':reference,'reference_expression':planted,
            'reference_kind':'supplied planted circuit upper bound'})
    expression=goals._sum_tree([['mul',a,b] for a in ('a','b','c') for b in ('d','e')])
    rows.append({'id':'replication-mixed-arity','family':'mixed-arity','seed':148003,'domain':cg.SCALAR,
        'expression':randomized_goal(expression,148003),'reference_cost':4,
        'reference_kind':'supplied product of three-term and two-term sums'})
    expression=['add',['mul','x',goals._sum_tree([['mul','a',v] for v in ('b','c','d')])],['mul','x','e']]
    rows.append({'id':'replication-nested-factor','family':'nested-factor','seed':148004,'domain':cg.SCALAR,
        'expression':randomized_goal(expression,148004),'reference_cost':5,
        'reference_kind':'supplied x*(a*(b+c+d)+e) cost'})
    expression=next(exprs[0] for label,exprs in goals.held_graphs(seed=149901) if label=='right-factor-sub')
    rows.append({'id':'replication-matrix-sub','family':'ordered-matrix','seed':149901,'domain':cg.MATRIX,
        'expression':expression,'reference_cost':goals.HELD_PRIOR.get((cg.MATRIX,'right-factor-sub')),
        'reference_kind':'historical supplied-family reference; fresh variable seed'})
    for row in rows:validate_input_expression(row['expression'])
    return rows
