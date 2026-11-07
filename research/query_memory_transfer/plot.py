"""Saved-only scientific figure; no model calls."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main(root):
    root=Path(root);r=json.loads((root/'report.json').read_text());rows=r['analysis']['factorials'];x=np.arange(3)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.7),layout='constrained');names=['Recall','Arithmetic','Prose']
    policies=['compiled','values-native','compiled-memory','values-native-memory'];labels=['CC/full','CN/full','CC/compact','CN/compact'];colors=['#7182a8','#29998b','#b38a69','#dcad43']
    for turn,ax in zip((0,1),axes[0]):
        for j,(policy,label,color) in enumerate(zip(policies,labels,colors)):
            vals=[100*next(row for row in rows if row['task']==i and row['turn']==turn and row['stage']=='prefill')['relative_logit_difference'][policy] for i in range(3)]
            ax.bar(x+(j-1.5)*.20,vals,.19,label=label,color=color)
            for i,v in enumerate(vals):ax.text(x[i]+(j-1.5)*.20,v+.25,f'{v:.2f}',ha='center',fontsize=7)
        ax.set(xticks=x,xticklabels=names,ylabel='Relative full-logit difference (%)',title=('A. First prompt: same consumed IDs' if turn==0 else 'B. Followup: own carried state and text'));ax.legend(ncol=2,fontsize=8)
        ax.set_ylim(0,ax.get_ylim()[1]*1.13)
    for j,(field,label,color) in enumerate((('full_gain_signed_squared_error','Full memory','#29998b'),('compact_gain_signed_squared_error','Bounded memory','#dcad43'))):
        vals=[next(row for row in rows if row['task']==i and row['turn']==0 and row['stage']=='prefill')[field] for i in range(3)]
        axes[1,0].bar(x+(j-.5)*.32,vals,.30,label=label,color=color)
    axes[1,0].axhline(0,color='#666',lw=.8);axes[1,0].set(xticks=x,xticklabels=names,ylabel='Native-normalized squared-error reduction',title='C. Native-V gain against the same NN/full reference');axes[1,0].legend(fontsize=8)
    tasks=r['analysis']['common_prefix_factorials']['tasks']
    for i,t in enumerate(tasks):
        selected=[row for row in r['analysis']['common_prefix_factorials']['rows'] if row['task']==i]
        axes[1,1].plot([z['step']+1 for z in selected],[100*z['interaction_relative_to_cn_full'] for z in selected],label=names[i])
    axes[1,1].axvline(32.5,color='#777',ls='--',lw=.8);axes[1,1].set(xlabel='Shared consumed-token position',ylabel='Interaction / CN-full attention norm (%)',title='D. Value × memory interaction (not signed benefit)');axes[1,1].legend(fontsize=8)
    fig.suptitle('Does native-value fidelity survive repeated bounded-memory reads?',fontsize=15)
    target=root/'query-memory-transfer.png';fig.savefig(target,dpi=160);plt.close(fig);print(target)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
