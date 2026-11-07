"""Scientific figure from immutable report and saved-only study."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    root = Path(root)
    report = json.loads((root / 'report.json').read_text())
    study = json.loads((root / 'study.json').read_text())
    curves = sorted(report['analysis']['source_selection_audit']['source_curves'], key=lambda c: c['dose'])
    doses = [c['dose'] for c in curves]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    for key, label in [('first8-decisions', 'First 8 decisions'), ('later-first-answer-decisions', 'Remaining first answer'), ('post-followup-decisions', 'After follow-up')]:
        ax.plot(doses, [c['window_equal_source_kl_nats'][key] for c in curves], 'o-', label=label)
    ax.set(yscale='log', xlabel='Frozen query dose', ylabel='Equal-source native KL (nats)', title='The preferred dose depends on future clocks')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for source in range(3):
        ax.plot(doses, [c['per_source_future_kl_nats'][source] for c in curves], 'o-', label=f'Source {source + 1}')
    ax.plot(doses, [c['future_mean_kl_nats'] for c in curves], 'o--', color='black', label='Declared equal-source/turn objective')
    ax.set(xlabel='Frozen query dose', ylabel='Native KL (nats)', title='Dose 2 helps two contexts and worsens one')
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for source in range(3):
        # Derived again from the archived first disagreement, not a visual estimate.
        record = next(x for x in study['source_greedy_first_disagreements'] if x['proposal'] == 5 and x['source'] == source)
        d = record['first_disagreement']
        matched = sum(record['native_answer_lengths'][:d['turn']]) + d['answer_token_index']
        total = sum(record['native_answer_lengths'])
        ax.barh(source, matched, color='#2a9d8f', label='Greedy-matching prefix' if source == 0 else None)
        ax.barh(source, total - matched, left=matched, color='#e9c46a', label='First mismatch + later forced tokens' if source == 0 else None)
    ax.set(yticks=range(3), yticklabels=['Source 1', 'Source 2', 'Source 3'], xlabel='Answer tokens across both turns', title='Longer source replay eventually becomes forced')
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1, 1]
    rows = study['selected_vs_response']['kl_nats']['rows']
    values = [x['improvement'] for x in rows]
    ax.bar(np.arange(len(rows)), values, color=['#2a9d8f' if v > 0 else '#e76f51' for v in values])
    ax.axhline(0, color='black', linewidth=.7)
    ax.set_xticks(np.arange(len(rows)), [f'{["Recall", "Math", "Prose"][x["task"]]}\nT{x["turn"] + 1} {x["stage"]}' for x in rows], rotation=65, ha='right', fontsize=8)
    wins = study['selected_vs_response']['kl_nats']['improves']
    ax.set(ylabel='KL(dose 1) − KL(dose 2), nats', title=f'New targets: {wins}/{len(rows)} KL boundaries improve')
    fig.suptitle('Longer query futures: same six images, same memory, different clock objective\nSelected dose 2 aliases the prior all-clock control; target text matches dose 1', fontsize=13)
    target = root / 'query-horizon.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    plot(p.parse_args().root)
