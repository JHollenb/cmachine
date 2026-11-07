"""Publication figures from the immutable composition measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def present(root):
    root=Path(root);a=json.loads((root/'analysis/scientific-analysis.json').read_text())
    out=root/'analysis';job=root.name
    colors={'10':'#5377aa','01':'#d79435','11':'#378b73'}
    initial=[r for r in a['matched_readouts'] if r['turn']==0 and r['stage']=='prefill']
    attention=[r for r in a['component_boundaries'] if r['field']=='attention']
    fig,axes=plt.subplots(1,3,figsize=(14.4,4.4),layout='constrained')
    x=np.arange(6)
    for j,(policy,label) in enumerate((('10','MLP only'),('01','Q/K only'),('11','Both'))):
        axes[0].bar(x+(j-1)*.25,[r['fidelity'][policy]['relative_logit_l2']*100 for r in initial],width=.23,color=colors[policy],label=label)
    axes[0].set(xticks=x,xticklabels=['Bind 1','Bind 2','Math 1','Math 2','Prose 1','Prose 2'],ylabel='Relative full-logit L2 (%)',title='Initial readout on common prompts')
    axes[0].tick_params(axis='x',labelrotation=30);axes[0].legend(frameon=False,fontsize=9)
    xs=np.asarray([r['reader_relative']*100 for r in attention]);ys=np.asarray([r['reader_change_under_cheap_writer_relative']*100 for r in attention])
    for task in range(6):
        idx=[i for i,r in enumerate(attention) if r['task']==task]
        axes[1].scatter(xs[idx],ys[idx],s=35,label=f'Case {task+1}',alpha=.85)
    lim=max(xs.max(),ys.max())*1.08
    axes[1].plot([0,lim],[0,lim],color='#8c969f',ls='--',lw=1)
    axes[1].set(xlim=(0,lim),ylim=(0,lim),xlabel='Reader error with native MLP (%)',ylabel='Reader error with compiled MLP (%)',title='Reader discrepancy survives the handoff')
    for task in range(6):
        rows=[r for r in a['matched_readouts'] if r['task']==task]
        axes[2].plot(np.arange(4),[r['interaction_relative']*100 for r in rows],marker='o',lw=1.3,label=f'Case {task+1}')
    axes[2].set(xticks=np.arange(4),xticklabels=['Turn 1\nprefill','Turn 1\nend','Turn 2\nprefill','Turn 2\nend'],ylabel='Interaction / native logit norm (%)',title='The two errors do not simply add')
    axes[2].legend(frameon=False,ncol=2,fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.suptitle('Frozen MLP16 → Q/K17 composition | Qwen2.5-1.5B | full FP32 memory',fontsize=13)
    fig.savefig(out/f'{job}-operand-composition.png',dpi=180)
    plt.close(fig)
    return out/f'{job}-operand-composition.png'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(present(a.root))
