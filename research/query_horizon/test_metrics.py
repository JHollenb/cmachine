import unittest
import numpy as np
from metrics import distribution,choose
from reader import Memory
from metrics import future_scores
from analysis import validate_trace
class ConsumerMetrics(unittest.TestCase):
    def test_logit_shift_changes_l2_not_distribution(self):
        a=np.asarray([2.,-1.,0.]);r=distribution(a,a+7.)
        self.assertAlmostEqual(r['kl_nats'],0.,places=14);self.assertAlmostEqual(r['native_top_logp_delta_nats'],0.,places=14);self.assertGreater(r['relative_logit_l2'],0.)
    def test_kl_direction_and_vocab_not_averaged(self):
        a=np.asarray([5.,1.,-2.]);b=a[::-1];r=distribution(a,b);p=np.exp(a-a.max());p/=p.sum();q=np.exp(b-b.max());q/=q.sum()
        self.assertAlmostEqual(r['kl_nats'],float(np.sum(p*np.log(p/q))),places=13);self.assertGreater(r['kl_nats'],6.)
    def test_ties_use_smallest_abs_dose_then_index(self):
        self.assertEqual(choose([1.,1.,1.,1.,1.,1.]),0);self.assertEqual(choose([2.,1.,1.,1.,1.,1.]),2);self.assertEqual(choose([2.,2.,2.,2.,1.,1.]),4)
    def test_fp64_norm_and_bad_operands(self):
        x=np.asarray([1e20,3e20],dtype='<f4');self.assertAlmostEqual(distribution(x,2*x)['relative_logit_l2'],1.)
        with self.assertRaises(ValueError):distribution([float('nan')],[0.])
    def test_memory_reader_unchanged(self):
        m=Memory();z=np.zeros((2,128),'<f4')
        for t in range(65):m.append(z,np.full((2,128),t,dtype='<f4'))
        np.testing.assert_array_equal(m.read(np.zeros((12,128),'<f4')),np.full((12,128),32,dtype='<f4'));self.assertEqual(len(m.bytes()),98464)
    def test_prediction_clock_includes_eos_excludes_unused_states(self):
        trace={'count':5,'prompt_length':1,'followup_length':2,'consumed_ids':[100,10,11,200,12],'input_phases':[0,1,1,2,3],'prediction_turn':[0,0,-1,1,-1],'turns':[{'turn':0,'start':0,'prefill_end':1,'final':3,'template_skip':0,'answer_ids':[10,11],'prediction_clocks':[0,1],'stopped':True,'truncated':False},{'turn':1,'start':3,'prefill_end':4,'final':5,'template_skip':1,'answer_ids':[12],'prediction_clocks':[3],'stopped':True,'truncated':False}]}
        validate_trace(trace,{'prompt_ids':[100],'followup_ids':[11,200]},[11,12])
        rows=[{'source':0,'step':t,'kl_nats':value,'prediction_turn':trace['prediction_turn'][t]} for t,value in enumerate([2.,4.,1000.,8.,1000.])]
        score,per,turns=future_scores(rows,[trace]);self.assertEqual(score,5.5);self.assertEqual(turns,[[3.,8.]])
        trace['turns'][0]['prediction_clocks']=[1,2]
        with self.assertRaises(ValueError):future_scores(rows,[trace])
    def test_empty_future_cannot_silently_change_turn_weight(self):
        trace={'count':1,'consumed_ids':[1],'prediction_turn':[-1],'turns':[{'prefill_end':1,'answer_ids':[],'prediction_clocks':[],'final':1}]}
        with self.assertRaises(ValueError):future_scores([{'source':0,'step':0,'kl_nats':0.,'prediction_turn':-1}],[trace])
if __name__=='__main__':unittest.main()
