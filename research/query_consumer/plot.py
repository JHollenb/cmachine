"""Saved-only scientific figure; original failed report stays unchanged."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot(root):
    root=Path(root);s=json.loads((root/'study.json').read_text());h=json.loads((root/'horizon-audit.json').read_text());rows=sorted(s['source_objective_contrast'],key=lambda x:x['dose']);x=[r['dose'] for r in rows];blue='#246b94';orange='#df8c37';green='#4e8b68'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(12,8.3));ax=axs[0,0];base=next(r for r in rows if r['dose']==0)
    for key,label,color in [('source_c_attention_loss','Attention response',blue),('source_native_kl_nats','Native full-distribution KL',orange)]:ax.plot(x,[100*r[key]/base[key] for r in rows],'-o',label=label,color=color)
    ax.set(xlabel='Frozen query dose',ylabel='Source loss / LS baseline (%)',title='A. Actual C consumers prefer different doses');ax.legend(frameon=False)
    ax=axs[0,1];y=np.arange(3);curves=json.loads((root/'analysis-recovery.json').read_text())['analysis']['source_selection_audit']['source_curves']
    for offset,index,label,color in [(-.18,4,'Dose 1',blue),(.18,5,'Dose 2',orange)]:ax.bar(y+offset,np.asarray(curves[index]['per_trajectory_mean_kl_nats'])*1000,.36,label=label,color=color)
    ax.set(xticks=y,xticklabels=['Source 1','Source 2','Source 3'],ylabel='Mean native KL (millinats)',title='B. Aggregate gain trades off source contexts');ax.legend(frameon=False)
    ax=axs[1,0]
    for region,label,color in [('shared-header','Header',green),('prompt-body','Prompt body',orange),('historical-own-suffix','Later suffix',blue)]:
        data=h['regional_selector_diagnostics'][region]['curves'];den=data[4]['equal_trajectory_mean_kl_nats'];data=[r for r in data if r['dose']>=0];ax.plot([r['dose'] for r in data],[100*r['equal_trajectory_mean_kl_nats']/den for r in data],'-o',label=label,color=color)
    ax.axhline(100,color='#555555',lw=.6);ax.set(xlabel='Frozen query dose',ylabel='Regional KL / dose 1 (%)',title='C. Header, prompt and future rank doses differently');ax.legend(frameon=False)
    ax=axs[1,1];initial=[r for r in s['target_gains']['response']['rows'] if r['turn']==0 and r['stage']=='prefill'];g=[1000*r['kl_improvement_nats'] for r in initial];ax.bar(np.arange(6),g,color=[blue if v>=0 else orange for v in g]);ax.axhline(0,color='#333333',lw=.7);ax.set(xticks=np.arange(6),xticklabels=['Recall','Math','Prose','Ticket recall','New math','Camping'],ylabel='Dose 2 KL gain vs dose 1 (millinats)',title='D. Initial target KL improves in only two cases');ax.tick_params(axis='x',rotation=24)
    fig.suptitle('The consumer metric needs a context and a clock',fontsize=17,y=.987);fig.text(.5,.012,'Same six frozen images / B128 / 98,464-byte memory / 819,200 selected MACs\n1,351 source advances + 2,434 target advances + 32 sham. Answers unchanged; post-processing failure recovered offline.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.065,1,.96),h_pad=2,w_pad=2);out=root/'query-consumer.png';fig.savefig(out,dpi=170);plt.close(fig);print(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');plot(p.parse_args().root)
