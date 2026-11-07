"""Saved-only fixed-cost query-program comparison; no model invocation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    root=Path(root);report=json.loads((root/'report.json').read_text());rows=report['analysis']['factorials'];x=np.arange(6)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');names=['Recall','Math','Prose','New recall','New math','New prose'];colors=['#78849b','#278e83']
    for turn,ax in zip((0,1),axes[0]):
        for j,(policy,label,color) in enumerate(zip(('values-native-memory','query-fit-memory'),('Original query','Source-fitted query'),colors)):
            vals=[100*next(row for row in rows if row['task']==i and row['turn']==turn and row['stage']=='prefill')['relative_logit_difference'][policy] for i in range(6)]
            ax.bar(x+(j-.5)*.32,vals,.30,label=label,color=color)
            for i,v in enumerate(vals):ax.text(x[i]+(j-.5)*.32,v+.25,f'{v:.2f}',ha='center',fontsize=7)
        ax.axvline(2.5,color='#999',ls='--',lw=.8);ax.set(xticks=x,xticklabels=names,ylabel='Relative full-logit difference (%)',title=('A. Initial compact read: identical prompt histories' if turn==0 else 'B. Followup compact read: own earlier text/state'));ax.tick_params(axis='x',labelrotation=18);ax.legend(fontsize=8);ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    ops=json.loads((root/'fp64-query-metrics.json').read_text())['rows'] if (root/'fp64-query-metrics.json').exists() else report['analysis']['query_operand_comparisons']
    for j,(policy,label,color) in enumerate(zip(('values-native-memory','query-fit-memory'),('Original query','Source-fitted query'),colors)):
        vals=[100*next(row for row in ops if row['task']==i and row['region']=='initial_prompt' and row['candidate']==f'case{i}-{policy}')['query_relative_error'] for i in range(6)]
        axes[1,0].bar(x+(j-.5)*.32,vals,.30,label=label,color=color)
        for i,v in enumerate(vals):axes[1,0].text(x[i]+(j-.5)*.32,v+.3,f'{v:.2f}',ha='center',fontsize=7)
    axes[1,0].axvline(2.5,color='#999',ls='--',lw=.8);axes[1,0].set(xticks=x,xticklabels=names,ylabel='Pooled initial native-query error (%)',title='C. Query reconstruction and consumer effect are separate');axes[1,0].tick_params(axis='x',labelrotation=18);axes[1,0].legend(fontsize=8)
    axes[1,0].set_ylim(0,28)
    work=np.asarray([819200,819200,2981888,3145728]);axes[1,1].barh(np.arange(4),work/1e6,color=colors+['#d6a14d','#ababab']);axes[1,1].invert_yaxis();axes[1,1].set(yticks=np.arange(4),yticklabels=['Original CN','Fitted query CN','Paid native-Q control','Native QKV'],xlim=(0,3.7),xlabel='Selected QKV nominal MAC/update (millions)',title='D. Fitted image replaces old image at unchanged cost')
    for i,cost in enumerate(work):axes[1,1].text(cost/1e6+.04,i,f'{cost/1e6:.3f}',va='center',fontsize=9)
    axes[1,1].set_ylim(4.5,-.6)
    axes[1,1].text(.01,.02,'Same 128 coordinates and 98,464-byte logical memory.\nWhole-token linear saving remains only 0.1507%; no latency claim.',transform=axes[1,1].transAxes,fontsize=8)
    fig.suptitle('A source-only query image at unchanged runtime cost',fontsize=16)
    target=root/'query-program.png';fig.savefig(target,dpi=160);plt.close(fig);print(target)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
