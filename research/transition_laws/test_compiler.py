"""Model-free checks for the experiment's sampling and answer-decision instrument."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
SPEC=importlib.util.spec_from_file_location('transition_worker',Path(__file__).with_name('worker.py'))
worker=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(worker)

def test_first_decision_uses_last_consumed_row_and_eos_decision(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'RAW',tmp_path);monkeypatch.setattr(worker,'V',3)
    # The second emitted ID stands for EOS; both decisions count, post-EOS does not.
    logits=np.asarray([[0,3,1],[4,1,0]],dtype='<f4')
    logits.tofile(tmp_path/'candidate');logits.tofile(tmp_path/'shadow')
    run={'paths':[{'case_id':'fixture','candidate_logits_file':'candidate','shadow_logits_file':'shadow','turns':[{'prefill':7,'answer_ids':[1,0],'stopped':True,'decision_positions':[6,7]}]}]}
    class Tokenizer:
        def decode(self,ids,skip_special_tokens):return str(ids)
    result=worker.metrics(run,'fixture',Tokenizer())
    assert result['equal_case_turn_KL']==0
    assert [r['position'] for r in result['decisions']]==[6,7]
    run['paths'][0]['turns'][0]['decision_positions']=[7,8]
    with pytest.raises(ValueError,match='Decision clocks'):worker.metrics(run,'wrong',Tokenizer())

def test_source_sampling_retains_actual_clock_lineage(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'RAW',tmp_path);monkeypatch.setattr(worker,'L',2);monkeypatch.setattr(worker,'D',3)
    values=np.arange(70*2*3,dtype='<f4').reshape(70,2,3);values.tofile(tmp_path/'receivers')
    samples=worker.capture_samples({'paths':[{'case_id':'fixture','consumed_ids':list(range(70)),'receiver_file':'receivers'}]},'source-native')
    metadata=json.loads((tmp_path/'source-native-sampling.json').read_text())
    indices=metadata['lineage'][0]['indices']
    assert len(indices)==len(set(indices))==64
    assert indices[0]==0 and indices[-1]==69
    np.testing.assert_array_equal(samples,values[indices])

def test_nominal_cost_includes_code_formation_and_original_down_map():
    d,h,r=1536,8960,256
    native=3*d*h;factor=d*r+2*h*r+d*h
    assert native-factor==22_544_384
    assert 28*(native-factor)==631_242_752
    assert 1_543_569_408-28*(native-factor)==912_326_656
