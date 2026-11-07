import unittest
from metrics import distribution, future_scores


class ClockTests(unittest.TestCase):
    def specimen(self):
        trace = {'count': 8, 'consumed_ids': [4, 5, 6, 99, 7, 8, 9, 99], 'prediction_turn': [-1, 0, 0, -1, -1, 1, 1, -1], 'turns': [{'prefill_end': 2, 'final': 4, 'answer_ids': [6, 99], 'prediction_clocks': [1, 2]}, {'prefill_end': 6, 'final': 8, 'answer_ids': [9, 99], 'prediction_clocks': [5, 6]}]}
        rows = [{'step': i, 'prediction_turn': turn, 'expected_next_token': trace['consumed_ids'][i+1] if turn >= 0 else -1, 'kl_nats': float(i)} for i, turn in enumerate(trace['prediction_turn'])]
        return trace, rows

    def test_first_and_eos_decisions_include_no_unused_row(self):
        trace, rows = self.specimen()
        mean, turns = future_scores(rows, trace)
        self.assertEqual(turns, [1.5, 5.5])
        self.assertEqual(mean, 3.5)

    def test_shifted_mask_refused(self):
        trace, rows = self.specimen()
        trace['turns'][0]['prediction_clocks'] = [2, 3]
        with self.assertRaises(ValueError):
            future_scores(rows, trace)

    def test_native_to_candidate_kl_direction_and_shift(self):
        a, b = [2., 0., -1.], [0., 1., -2.]
        self.assertGreater(distribution(a, b)['kl_nats'], 0)
        self.assertNotEqual(distribution(a, b)['kl_nats'], distribution(b, a)['kl_nats'])
        self.assertAlmostEqual(distribution(a, b)['kl_nats'], distribution([x+1000 for x in a], [x-100 for x in b])['kl_nats'])


if __name__ == '__main__':
    unittest.main()
