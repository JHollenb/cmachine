"""Generated choice organism: mechanics only, no learned checkpoint execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
HAS_SATURN=importlib.util.find_spec('saturn_pub') is not None
if HAS_SATURN:
    from saturn_pub import Act,Session
    from math_debugger import ChoiceAdapter,ReplayEngine,Witnesses,single_exchanges,outcome_summary


class ToyGraph:
    def __init__(self,domain,mode):
        self.domain,self.mode=domain,mode;self.parent=[0];self.nodes={0:{('var',('x',))}}
        self.memo={('var',('x',)):0};self.const_classes=set();self.n_nodes=1;self.created=None
    @property
    def repaired(self):return len(self.parent)>1
    def rebuild(self):pass
    def find(self,*args):raise AssertionError('witness called mutating find')


class ToyGoals:
    @staticmethod
    def _insert_nary(g,expression):return 0
    @staticmethod
    def poly(expression,domain):return 'same supplied toy function'


class ToyConsumer:
    """Round0 divergence irrelevant; choosing candidate1 at round1 repairs cost."""
    EGraph=ToyGraph
    @staticmethod
    def saturate(g,roots,ruleset,*,scheduler,featurize,iter_cap,**kwargs):
        trace=[]
        for round_ in range(iter_cap):
            metas=[{'rule':'toy0','key':(round_,0)},{'rule':'toy1','key':(round_,1)}]
            X=featurize(g,roots,metas,[0,0],round_)
            choices=scheduler.select(g,roots,metas,X,round_)
            if round_==1 and choices==[1]:g.parent.append(0)
            trace.append({'iteration':round_+1,'applied':len(choices),'seconds':0})
        return {'status':'iter-cap','trace':trace}
    @staticmethod
    def extract_dag(g,roots,**kwargs):return {}
    @staticmethod
    def to_expr(g,choice,root):return ['toy-repaired' if g.repaired else 'toy-original','x']
    @staticmethod
    def dag_cost(g,choice,roots):return (5 if g.repaired else 9),{}


@unittest.skipUnless(HAS_SATURN,'Saturn public SDK is optional; fleet payload pins its source')
class ChoiceMechanics(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.feature_offset=0
        def features(g,roots,metas,targets,round_):
            return np.asarray([[round_,self.feature_offset],[round_,1+self.feature_offset]],dtype=np.float32)
        self.engine=ReplayEngine(ToyConsumer,ToyGoals,features,object,
            lambda X,k:{'v1':[1],'v2a':[0]},Witnesses(self.tmp.name),'x','toy','frozen-generated-source',
            caps={'rounds':2,'per_round':1,'selections':2})
        self.adapter=ChoiceAdapter(self.engine)

    def test_irrelevant_first_divergence_and_actual_repair(self):
        baseline=self.engine.store.get(self.engine.run()['endpoint']);self.assertEqual(baseline['cost'],9)
        rows=[]
        for i,entry in enumerate(baseline['decisions']):
            state=self.engine.run(baseline['decisions'][:i],pause_at=i)
            session=Session(self.adapter,state);cut=session.capture(retain=False)
            frontier=self.engine.store.get(state['frontier'])
            branch=session.fork(cut);branch.apply(Act.replace('choice',frontier['proposals']['v1']))
            branch.continue_();end=self.engine.store.get(branch.capture(retain=False).payload['endpoint'])
            rows.append({'status':'completed','round':i,'cost':end['cost'],'causal_qualified':True})
            # Captured source remains unchanged after editing/continuing its fork.
            self.assertEqual(session.read('choice'),cut.payload['choice']);cut.verify()
        self.assertEqual([r['cost'] for r in rows],[9,5])
        self.assertEqual(outcome_summary(rows,9)['earliest_tested_repairable_round'],1)

    def test_sham_A_B_A_and_native_continuation_receipts(self):
        baseline=self.engine.run();state=self.engine.run(pause_at=0)
        session=Session(self.adapter,state);cut=session.capture(retain=False)
        other=self.engine.store.get(state['frontier'])['proposals']['v1']
        session.apply(Act.replace('choice',other));session.apply(Act.replace('choice',cut.payload['choice']))
        session.continue_()
        self.assertEqual(session.capture(retain=False).payload['endpoint'],baseline['endpoint'])
        self.assertEqual(len(session.receipts),3)
        self.assertEqual([s.name for s in self.adapter.surface(state).slots if s.writable],['choice'])

    def test_refuse_malformed_stale_or_wrong_parent_choices(self):
        state=self.engine.run(pause_at=0);cut=Session(self.adapter,state).capture(retain=False)
        later=self.engine.run(pause_at=1)['choice']
        for invalid in ([],state['choice']*2,[0],['0'*64],later):
            with self.assertRaises(ValueError):Session.from_cut(self.adapter,cut).apply(Act.replace('choice',invalid))
        with self.assertRaises(ValueError):Session.from_cut(self.adapter,cut).apply(Act.replace('context','other'))
        self.feature_offset=1
        with self.assertRaisesRegex(ValueError,'frontier differs'):Session.from_cut(self.adapter,cut).continue_()

    def test_sealed_witness_and_prefix_validation(self):
        state=self.engine.run(pause_at=1)
        bad=copy.deepcopy(state);bad['prefix']=[]
        with self.assertRaisesRegex(ValueError,'wrong prefix'):Session(self.adapter,bad)
        path=self.engine.store.root/(state['frontier']+'.json')
        path.write_text(path.read_text().replace('"count":1','"count":2'))
        with self.assertRaisesRegex(ValueError,'stale witness'):Session(self.adapter,state)

    def test_exchange_order_and_nonmonotone_summary(self):
        choices=single_exchanges(['a','b','c'],['c','d','e'])
        self.assertEqual(len(choices),4)
        self.assertEqual(choices[0],(0,'d',['d','b','c']))
        summary=outcome_summary([{'status':'completed','round':r,'cost':c,'causal_qualified':True} for r,c in [(0,9),(1,5),(2,12),(3,4)]],9)
        self.assertEqual(summary['earliest_tested_repairable_round'],1)
        self.assertEqual((summary['improved'],summary['equal'],summary['worse']),(2,1,1))
        unqualified=outcome_summary([{'status':'completed','round':0,'cost':1,'causal_qualified':False}],9)
        self.assertEqual(unqualified['improved'],1)
        self.assertIsNone(unqualified['earliest_tested_repairable_round'])


if __name__=='__main__':unittest.main()
