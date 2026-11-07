"""No model: exercise token authority and explicit template closure."""
import copy
import unittest
from analysis import validate_turn,token_hash

def fixture(parent=(1,151645),answers=(42,151645),autonomous=True):
    template=[151645,198,5];skip=int(parent[-1]==151645);history=list(parent);prefill=[];generated=[]
    for j,token in enumerate(template[skip:]+list(answers)):
        history.append(token);is_generation=j>=len(template)-skip
        row={"step":j,"position":len(parent)+j,"input_id":token,"history_sha256":token_hash(history),"top":42 if not is_generation else 151645,"role":("own-generated" if autonomous else "sealed-replay-answer") if is_generation else ("host-template-close" if not skip and j==0 else "template-user"),"substitution":True,"gate_up_calls":0,"down_calls":1,"compile_calls":0}
        (generated if is_generation else prefill).append(row)
    return {"parent_history_ids":list(parent),"template_ids":template,"template_skip":skip,"host_supplied_closing_token":not bool(skip),"steps":prefill,"generation_steps":generated,"prefill_count":len(prefill),"answer_ids":list(answers),"budget":len(answers),"all_delivered_consumed":True,"stopped":answers[-1] in (151645,151643),"truncated":answers[-1] not in (151645,151643),"autonomous":autonomous,"parent_position":len(parent),"final_position":len(history),"policy":2}

class TokenContracts(unittest.TestCase):
    def test_own_tokens_consumed(self):validate_turn(fixture())
    def test_wrong_selected_token_refused(self):
        t=fixture();t["steps"][-1]["top"]=99
        with self.assertRaisesRegex(ValueError,"Greedy"):validate_turn(t)
    def test_truncated_parent_gets_explicit_closure(self):
        t=fixture(parent=(1,17));validate_turn(t);self.assertEqual(t["steps"][0]["role"],"host-template-close")
    def test_other_eos_is_not_template_close(self):
        t=fixture(parent=(1,151643));validate_turn(t);self.assertEqual(t["template_skip"],0)
    def test_forced_diagnostic_is_not_own_generation(self):
        t=fixture(autonomous=False);t["steps"][-1]["top"]=99;validate_turn(t)
    def test_false_history_join_refused(self):
        t=fixture();t["generation_steps"][-1]["history_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"History"):validate_turn(t)
    def test_policy_must_run_during_ingestion(self):
        t=fixture();t["steps"][0]["substitution"]=False
        with self.assertRaisesRegex(ValueError,"work"):validate_turn(t)
    def test_budget_endpoint_consumed(self):validate_turn(fixture(answers=(42,43),autonomous=False))

if __name__=="__main__":unittest.main()
