"""Publication figure from independently recomputed saved results."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def plot(root):
    root=Path(root);a=json.loads((root/'analysis.json').read_text());fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');colors=['#567c99','#127c63','#b56b28','#914e84'];src=a['source_feedback'];ax[0].plot([r['dose'] for r in sorted(src,key=lambda r:r['dose'])],[r['future_kl'] for r in sorted(src,key=lambda r:r['dose'])],color=colors[1],marker='o',linewidth=2)
    for c in range(3):ax[0].plot([r['dose'] for r in sorted(src,key=lambda r:r['dose'])],[r['per_source'][c] for r in sorted(src,key=lambda r:r['dose'])],alpha=.35,linewidth=1)
    selected=src[a['selected_index']];ax[0].scatter([selected['dose']],[selected['future_kl']],s=100,facecolor='white',edgecolor=colors[1],linewidth=2,zorder=5);ax[0].set(xlabel='Signed source update dose',ylabel='Mean future KL (nats)',title='Source own-token futures choose the update');ax[0].grid(axis='y',alpha=.2)
    x=np.arange(len(a['fresh']));width=.18;labels=['Frozen layer17','Updated layer17','Updated17 + Q/K18','Updated17 + Q/K18 + compact18']
    for j in range(4):ax[1].bar(x+(j-1.5)*width,[case['arms'][j+1]['future_kl'] for case in a['fresh']],width,color=colors[j],label=labels[j])
    ax[1].set_xticks(x,[case['kind']+'\n'+str(case['context_tokens'])+' tokens' for case in a['fresh']]);ax[1].set(ylabel='Mean future KL (nats)',title='Frozen programs on fresh two-turn histories');ax[1].grid(axis='y',alpha=.2);ax[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('A better contextual query transfers; extending the compiler adds error',fontsize=14);fig.savefig(root/'context-query.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');plot(p.parse_args().root)
