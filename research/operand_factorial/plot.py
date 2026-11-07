"""Static scientific figures from the collected report; no model execution."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def plot(job):
    report=json.loads((HERE/'runs'/job/'report.json').read_text());a=report['analysis'];out=HERE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharey=True)
    for i,policy in enumerate(['source64','random64']):
        for j,(case,boundary,title) in enumerate([(0,'first','Recall · initial prompt'),(1,'first','Arithmetic · initial prompt'),(1,'after_followup','Arithmetic · followup')]):
            ax=axs[i,j];rows=[x for x in a['observations'] if x['policy']==policy and x['case']==case and x['boundary']==boundary]
            names=['00','10','01','11','zero'];x=np.arange(len(names))
            for offset,parent,color in [(-.19,'carried','#c05b36'),(.19,'native','#326d9c')]:
                values=[100*next(r['logit_relative_error_vs_native_same_ID'] for r in rows if r['arm']==name and r['parent']==parent) for name in names]
                ax.bar(x+offset,values,.36,color=color,label='Candidate-carried memory' if parent=='carried' else 'Native same-ID memory')
            ax.set_xticks(x,['Neither','Gate','Payload','Both','Delete']);ax.set_title(title+'\n'+policy);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if j==0:ax.set_ylabel('Relative full-logit error (%)')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.01))
    fig.suptitle('Exact current operands do not erase earlier memory changes',fontsize=16,y=.99)
    fig.text(.5,.055,'Reference: native execution of identical consumed IDs. Four native continuation tokens follow each one-write intervention.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(out/(job+'-restoration.png'));plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5));policies=['source64','random64','b3'];x=np.arange(3)
    for j,(name,color) in enumerate(zip(['Gate error','Payload error','Interaction'],['#4c78a8','#f2a541','#a45a8a'])):
        y=[np.median([r['term_norm_over_native'][j] for r in a['decomposition'] if r['policy']==p]) for p in policies]
        axs[0].bar(x+(j-1)*.25,y,.24,label=name,color=color)
    axs[0].set_xticks(x,policies);axs[0].set_ylabel('Median projected term norm / native write norm');axs[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.1),ncol=3);axs[0].set_title('The conjunction term is substantial')
    for j,(name,color) in enumerate(zip(['00','10','01'],['#707070','#4c78a8','#f2a541'])):
        y=[np.mean([r['ideal_local_restoration_errors'][name] for r in a['decomposition'] if r['policy']==p]) for p in policies]
        axs[1].bar(x+(j-1)*.25,y,.24,label={'00':'Neither restored','10':'Gate restored','01':'Payload restored'}[name],color=color)
    axs[1].axhline(1,color='black',linestyle='--',linewidth=1,label='Deletion');axs[1].set_xticks(x,policies);axs[1].set_ylabel('Mean local relative write error');axs[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2);axs[1].set_title('Source operands have conditional usefulness')
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('36 saved clocks · 12 per policy · development evidence',fontsize=14)
    fig.text(.5,.015,'Terms are signed vectors and can cancel; their norms are not additive shares. Means use each policy’s saved receiver distribution.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(out/(job+'-decomposition.png'));plt.close(fig)
    print(json.dumps({'figures':[str(p) for p in sorted(out.glob(job+'-*.png'))]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');plot(p.parse_args().job)
