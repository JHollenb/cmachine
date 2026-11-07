"""Publication figure from verified histories and saved-only reader diagnostics."""
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
    pairs = report['analysis']['pairs']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    x = np.arange(3)
    ax = axes[0, 0]
    for offset, start, label, color in [(-.19, 0, 'Native-generated history', '#457b9d'), (.19, 3, 'Compact-generated history', '#e9a23b')]:
        ax.bar(x+offset, [pairs[i+start]['future_mean_kl_nats'] for i in range(3)], .36, label=label, color=color)
    ax.set(xticks=x, xticklabels=['Notebook', 'Cards', 'Explanation'], ylabel='Equal-turn conditional KL, nats', title='Mean rises; two contexts improve and one worsens')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    for c, label in enumerate(['Notebook', 'Cards', 'Explanation']):
        ax.plot([1, 2], pairs[c]['per_turn_kl_nats'], 'o--', color=f'C{c}', alpha=.65)
        ax.plot([1, 2], pairs[c+3]['per_turn_kl_nats'], 's-', color=f'C{c}', label=label)
    ax.set(xticks=[1, 2], xticklabels=['First answer', 'Follow-up answer'], ylabel='Conditional KL, nats', title='Solid: own history; dashed: native history')
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for start, offset, label, color in [(0, -.19, 'Native history', '#457b9d'), (3, .19, 'Own history', '#e9a23b')]:
        values = [pairs[start+c]['answer_lengths'][1] for c in range(3)]
        ax.bar(x+offset, values, .36, label=label, color=color)
    ax.axhline(32, color='grey', linestyle=':', linewidth=1)
    ax.set(xticks=x, xticklabels=['Notebook', 'Cards', 'Explanation'], ylabel='Actual answer tokens, including EOS', title='The correct card paraphrase ends earlier')
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    rows = study['reader_summary']
    keys = ['query_only_relative', 'key_only_relative', 'aggregation_at_compact_query_relative', 'both_bounded_relative']
    labels = ['Q only', 'K only', 'Aggregation\nat compact Q', 'Actual\nbounded']
    for start, offset, name, color in [(0, -.19, 'Native history', '#457b9d'), (3, .19, 'Own history', '#e9a23b')]:
        # Each source/turn contributes equally, matching the declared quality weighting.
        values = [np.mean([r['means'][key] for r in rows if start <= r['pair'] < start+3]) for key in keys]
        ax.bar(np.arange(4)+offset, values, .36, label=name, color=color)
    ax.set(xticks=np.arange(4), xticklabels=labels, ylabel='Native attention-relative norm', title='Saved reader diagnostics; norms are not causal shares')
    ax.legend(fontsize=9)
    fig.suptitle('One fixed query program on two history distributions\nSame prompts, 32-token budgets and 98,464-byte memory; native shadows consume compact IDs', fontsize=13)
    target = root / 'query-own-history.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    plot(p.parse_args().root)
