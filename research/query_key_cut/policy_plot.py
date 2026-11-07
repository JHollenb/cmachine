"""Compare frozen operand masks without averaging away task heterogeneity."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot(root):
    root=Path(root)
    rows=json.loads((root/'analysis.json').read_text())['observations']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True,layout='constrained')
    labels=['17 only','17 + Q/K18','17 + Q18','17 + K18']
    colors=['#547f9d','#bd6b31','#13816d','#8c4774']
    for task,ax in enumerate(axes):
        for i,(label,color) in enumerate(zip(labels,colors)):
            values=[]
            for context in (128,1024):
                q=next(x for x in rows if x['task']==task and x['context_tokens']==context and x['policy']=='query-only18')
                k=next(x for x in rows if x['task']==task and x['context_tokens']==context and x['policy']=='key-only18')
                values.append([q['prior_updated17_kl'],q['prior_both18_kl'],q['future_kl'],k['future_kl']][i])
            ax.bar(np.arange(2)+(i-1.5)*.18,values,width=.18,label=label,color=color)
        ax.set_xticks([0,1],['128 tokens','1,024 tokens'])
        ax.set_title(['Inventory and update','Explanation and follow-up'][task])
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean future KL(native || candidate), lower is closer')
    axes[1].legend(fontsize=9)
    fig.suptitle('Native keys fix the inventory path; explanation fidelity still degrades',fontsize=13)
    fig.supxlabel('Known development conversations · each policy owns its history · no new fitting',fontsize=9)
    fig.savefig(root/'on-policy-masks.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');plot(p.parse_args().root)
