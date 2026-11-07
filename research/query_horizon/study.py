"""Saved-only horizon contrasts and output checks; no model calls or fitting."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study(root):
    root = Path(root)
    report = json.loads((root / 'report.json').read_text())
    verification = json.loads((root / 'verification.json').read_text())
    if verification['report_sha256'] != sha(root / 'report.json'):
        raise ValueError('Verified report identity differs')
    audit = report['analysis']['source_selection_audit']
    curves = audit['source_curves']
    family = report['family_provenance']
    images = []
    for i in range(6):
        path = root / 'raw' / f'query-proposal-{i}.f32'
        if sha(path) != family['files'][path.name]['sha256']:
            raise ValueError('Frozen query image differs')
        images.append({'proposal': i, 'sha256': sha(path), 'bytes': path.stat().st_size})
    selected = report['source_selection']['selected_index']
    if (root / 'raw' / 'selected-query.f32').read_bytes() != (root / 'raw' / f'query-proposal-{selected}.f32').read_bytes():
        raise ValueError('Selected image differs')
    rankings = {}
    for window in curves[0]['window_equal_source_kl_nats']:
        rows = sorted([{'proposal': c['proposal'], 'dose': c['dose'], 'kl_nats': c['window_equal_source_kl_nats'][window]} for c in curves], key=lambda x: (x['kl_nats'], abs(x['dose']), x['proposal']))
        rankings[window] = rows
    per_turn = []
    for c in range(3):
        for turn in range(2):
            winner = min(curves, key=lambda x: (x['per_source_per_turn_kl_nats'][c][turn], abs(x['dose']), x['proposal']))
            per_turn.append({'source': c, 'turn': turn, 'winner_dose': winner['dose'], 'dose1_kl_nats': curves[4]['per_source_per_turn_kl_nats'][c][turn], 'dose2_kl_nats': curves[5]['per_source_per_turn_kl_nats'][c][turn]})
    prefix = []
    for i in range(6):
        rows = json.loads((root / 'raw' / f'source-metrics-{i}.json').read_text())['rows']
        scored = [x for x in rows if x['prediction_turn'] >= 0]
        prefix.append({'proposal': i, 'generated_decision_rows': len(scored), 'own_prefix_before_decision': sum(x['greedy_prefix_before_decision'] for x in scored), 'already_forced_after_disagreement': sum(not x['greedy_prefix_before_decision'] for x in scored), 'own_prefix_and_greedy_choice_matches': sum(x['greedy_prefix_before_decision'] and x['candidate_top'] == x['expected_next_token'] for x in scored)})
    gains = {}
    for metric in ('kl_nats', 'relative_logit_l2'):
        rows = []
        for x in report['analysis']['comparisons']:
            rows.append({'task': x['task'], 'turn': x['turn'], 'stage': x['stage'], 'improvement': x['metrics']['response'][metric] - x['metrics']['selected'][metric], 'native_history_matches': x['native_history_matches']['selected']})
        gains[metric] = {'improves': sum(x['improvement'] > 0 for x in rows), 'worsens': sum(x['improvement'] < 0 for x in rows), 'ties': sum(x['improvement'] == 0 for x in rows), 'matching_native_history_rows': sum(x['native_history_matches'] for x in rows), 'matching_native_history_improves': sum(x['native_history_matches'] and x['improvement'] > 0 for x in rows), 'rows': rows}
    outcomes = []
    for path in report['driver']['paths']:
        texts = report['texts'][path['name']]
        item = {'path': path['name'], 'task': path['task'], 'answers': texts, 'truncated': [x['truncated'] for x in path['turns']]}
        if path['task'] == 0:
            item.update(expected=['ceramic', 'steel'], exact_answers_correct=[t.strip().lower() == expected for t, expected in zip(texts, ['ceramic', 'steel'])])
        elif path['task'] == 1:
            leading = [re.match(r'\s*([+-]?\d+)', t) for t in texts]
            leading = [int(m.group(1)) if m else None for m in leading]
            item.update(expected=[418, 401], leading_integers=leading, initial_correct=leading[0] == 418, followup_leading_integer_correct=leading[1] == 401, followup_correct_from_own_answer=leading[0] is not None and leading[1] == leading[0] - 17, complete_explanation=not path['turns'][1]['truncated'])
        else:
            item['evaluation'] = 'Texts retained for qualitative inspection; no automatic prose correctness score.'
        outcomes.append(item)
    aliases = report['driver']['aliases']
    for x in report['analysis']['comparisons']:
        if selected == 5 and (x['physical_names']['selected'] != x['physical_names']['allclock'] or x['metrics']['selected'] != x['metrics']['allclock']):
            raise ValueError('Aliased policy differs')
    result = {'schema': 'tcm-query-horizon-study-v1', 'report_sha256': sha(root / 'report.json'), 'frozen_family': images, 'selected_index': selected, 'source_future_relative_gain_vs_response': 1 - curves[selected]['future_mean_kl_nats'] / curves[4]['future_mean_kl_nats'], 'source_future_relative_gain_vs_ls': 1 - curves[selected]['future_mean_kl_nats'] / curves[0]['future_mean_kl_nats'], 'window_rankings': rankings, 'per_source_turn_contrast': per_turn, 'source_greedy_prefix_counts': prefix, 'source_greedy_first_disagreements': audit['greedy_prefix_audit'], 'source_answer_lengths': [[len(t['answer_ids']) for t in s['turns']] for s in audit['source_traces']], 'source_answer_truncations': [[t['truncated'] for t in s['turns']] for s in audit['source_traces']], 'selected_vs_response': gains, 'selected_equals_allclock_by_construction': selected == 5, 'logical_aliases': aliases, 'target_output_checks': outcomes, 'scope': 'Saved-only interpretation. Twelve correlated readout clocks from three new conversations; six images along one old direction. Source rows after first greedy disagreement are native-forced evidence. Output-only truth checks are post hoc and do not select images.', 'terminal_status': 'not-requested'}
    (root / 'study.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'source_future_relative_gain_vs_response': result['source_future_relative_gain_vs_response'], 'source_future_relative_gain_vs_ls': result['source_future_relative_gain_vs_ls'], 'window_winner_doses': {k: v[0]['dose'] for k, v in rankings.items()}, 'source_prefix_counts': prefix[0], 'target_metric_counts': {k: {a: b for a, b in v.items() if a != 'rows'} for k, v in gains.items()}}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    study(parser.parse_args().root)
