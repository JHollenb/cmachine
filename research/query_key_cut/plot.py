"""First-wrong-binding margins and the first inherited-state consumer."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def plot(root):
    root=Path(root);a=json.loads((root/'analysis.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained');arms=[2,3,4,5,6,7,8];labels=['C Q\nC K','N Q\nC K','C Q\nN K','N Q\nN K','Older Q\nN K','Zero\nread','N Q/K\nreverse V']
    for group in a['groups']:
        i=int(group['context']==1024);j=group['history'];ax=axes[i,j];rows=[group['rows'][k] for k in arms];colors=['#13816d' if row['first_whistle_location']=='silver' else '#bd6b31' for row in rows];ax.bar(np.arange(7),[row['silver_minus_amber_logit'] for row in rows],color=colors);ax.scatter(np.arange(7),[row['silver_minus_amber_logit'] for row in rows],c=colors,s=22,zorder=3);ax.axhline(0,color='#333333',linewidth=.8);ax.set_xticks(np.arange(7),labels,fontsize=8);ax.set_title(f'{group["context"]} tokens · '+('compiled history18' if not j else 'native history18'));ax.set_ylabel('Silver minus amber logit');ax.grid(axis='y',alpha=.2)
    fig.suptitle('At the wrong location token, keys and prior history both matter',fontsize=15);fig.savefig(root/'query-key-margins.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
    for row,color in zip(a['historical_native_current_contrasts'],['#547f9d','#8c4774']):ax.plot([x['layer'] for x in row['changes']],[x['output'] for x in row['changes']],marker='o',label=str(row['context'])+' tokens',color=color)
    ax.set(xlabel='Layer',ylabel='L2 output difference across histories',title='Native current layer18 closes exactly; divergence returns at attention19');ax.set_xticks(range(18,28));ax.legend();ax.grid(alpha=.2);fig.savefig(root/'history-consumer.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');plot(p.parse_args().root)
