import copy
import unittest
from analysis import history_domains, validate_token_chain


class TokenAuthority(unittest.TestCase):
    def fixture(self):
        return {"budget": 2, "steps": [{"input_id": 198, "top": 37}, {"input_id": 37, "top": 151645}, {"input_id": 151645, "top": 1}],
                "own": {"ids": [37, 151645], "stopped": True, "truncated": False, "all_delivered_consumed": True}}

    def test_final_eos_must_be_consumed(self):
        arm = self.fixture();validate_token_chain(arm)
        arm["steps"].pop()
        with self.assertRaises(ValueError):validate_token_chain(arm)

    def test_native_token_cannot_replace_own_prediction(self):
        arm = self.fixture();arm["steps"][1]["input_id"] = 38
        with self.assertRaises(ValueError):validate_token_chain(arm)

    def test_next_token_equality_does_not_restore_history(self):
        native = self.fixture();arm = copy.deepcopy(native);arm["steps"][1]["input_id"] = 38
        self.assertEqual(history_domains(arm, native), [True, False, False])

    def test_different_prediction_only_changes_next_input_domain(self):
        native = self.fixture();arm = copy.deepcopy(native);arm["steps"][0]["top"] = 38
        self.assertEqual(history_domains(arm, native), [True, True, True])
        arm["steps"][1]["input_id"] = 38
        self.assertEqual(history_domains(arm, native), [True, False, False])

    def test_truncation_is_not_a_stop(self):
        arm = self.fixture();arm["own"]["stopped"] = False;arm["own"]["truncated"] = True
        with self.assertRaises(ValueError):validate_token_chain(arm)

    def test_generation_cannot_continue_after_eos(self):
        arm = self.fixture();arm["own"]["ids"][0] = 151645;arm["steps"][0]["top"] = 151645;arm["steps"][1]["input_id"] = 151645
        with self.assertRaises(ValueError):validate_token_chain(arm)


if __name__ == "__main__":unittest.main()
