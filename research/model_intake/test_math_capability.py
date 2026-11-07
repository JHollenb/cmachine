"""Generated frozen-coordinate and proposal-union mechanics; no learned assets."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
HAS_SATURN=importlib.util.find_spec('saturn_pub') is not None
if HAS_SATURN:from math_capability import PackedWitnesses,greedy_units,proposal_union,structural_order,subnetwork,randomized_goal,validate_input_expression


class GeneratedScorer(nn.Module):
    def __init__(self,width,hidden=32):
        super().__init__();self.register_buffer('mean',torch.zeros(width));self.register_buffer('std',torch.ones(width))
        self.net=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x):return self.net((x-self.mean)/self.std).squeeze(-1)


@unittest.skipUnless(HAS_SATURN,'optional Saturn public SDK required')
class CapabilityMechanics(unittest.TestCase):
    def test_pruned_units_copy_parameters_and_match_zeroed_consumer(self):
        torch.manual_seed(711);source=GeneratedScorer(21).eval();units=[4,1,9,25]
        small=subnetwork(source,units);self.assertEqual(sum(p.numel() for p in small.parameters()),93)
        source.net[2].weight.data[:,[i for i in range(32) if i not in units]]=0
        rows=torch.randn(18,21)
        with torch.no_grad():np.testing.assert_allclose(small(rows).numpy(),source(rows).numpy(),rtol=1e-5,atol=1e-7)
        np.testing.assert_array_equal(small.net[0].weight.detach().numpy(),source.net[0].weight.detach().numpy()[sorted(units)])
    def test_greedy_discovery_keeps_all_signed_candidate_losses(self):
        a=np.eye(4);target=np.array([2.,0.,1.,0.]);contributions=a*np.array([2.,3.,1.,4.])
        selected,trace=greedy_units(contributions,target,np.ones(4),2)
        self.assertEqual(selected,[0,2]);self.assertEqual(trace[-1]['loss'],0)
        self.assertEqual([len(r['candidates']) for r in trace],[4,3])
        with self.assertRaises(ValueError):greedy_units(contributions,target,np.zeros(4))
    def test_proposal_union_preserves_cardinality_and_uses_own_features(self):
        names=['rule_factor','log_group','target_on_route'];X=np.zeros((12,3),dtype=np.float32)
        X[:,0]=1;X[:,1]=np.log1p(2);X[9,1]=np.log1p(4);X[9,2]=1
        choice,edits=proposal_union(list(range(12)),X,names,8,quota=2)
        self.assertEqual(choice,[0,1,2,3,4,5,6,9]);self.assertEqual(len(edits),1)
        alt,_=proposal_union(list(range(12)),X,names,8,quota=1,alternative=True,seed=71)
        self.assertNotIn(9,alt);self.assertEqual(len(set(alt)),8)
    def test_compressed_witness_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=PackedWitnesses(tmp);key=store.put({'prefix':[1,2]})
            self.assertEqual(store.get(key),{'prefix':[1,2]});self.assertEqual(store.put({'prefix':[1,2]}),key)
            import gzip
            (Path(tmp)/(key+'.json.gz')).write_bytes(gzip.compress(json.dumps({'prefix':[2,1]}).encode()))
            with self.assertRaisesRegex(ValueError,'stale witness'):store.get(key)
    def test_scorer_free_structure_has_ordinal_fallback(self):
        names=['rule_factor','log_group','target_on_route'];X=np.zeros((12,3),dtype=np.float32)
        X[[9,10],0]=1;X[[9,10],1]=np.log1p(4);X[10,2]=1
        self.assertEqual(structural_order(X,names),[10,9,0,1,2,3,4,5,6,7,8,11])
    def test_native_input_trees_are_binary(self):
        validate_input_expression(['add',['mul','x','y'],['const',-1]])
        with self.assertRaisesRegex(ValueError,'binary'):validate_input_expression(['prod','x','y'])
        with self.assertRaisesRegex(ValueError,'binary'):validate_input_expression(['add','x','y','z'])
    def test_ordered_matrix_ports_survive_layout_variation(self):
        result=randomized_goal(['mul','A','B'],713,commutative=False)
        self.assertEqual(result[0],'mul');self.assertNotEqual(result[1],result[2])
        self.assertEqual(result,randomized_goal(['mul','A','B'],713,commutative=False))


if __name__=='__main__':unittest.main()
