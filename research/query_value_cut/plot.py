"""Research figure from immutable saved observations; no model calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
    root=HERE/'runs/job-27a597ebd5f8';r=json.loads((root/'report.json').read_text());tasks=r['analysis']['common_prefix_factorials']['tasks'];x=np.arange(3)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.7),layout='constrained');names=['Recall','Arithmetic','Prose'];colors=['#697aa5','#178e83','#cb824b'];labels=['Both compiled (CC)','Native values (CN)','Native query/key (NC)']
    for i,(field,policy,label,color) in enumerate(zip(['both_relative','qk_only_relative','value_only_relative'],['compiled','values-native','address-native'],labels,colors)):
        values=[100*t['regions']['initial_prompt']['attention'][field] for t in tasks]
        axes[0,0].bar(x+(i-1)*.24,values,.23,label=label,color=color)
        for j,value in enumerate(values):axes[0,0].text(x[j]+(i-1)*.24,value+.5,f'{value:.1f}',ha='center',fontsize=8)
        logits=[100*next(v['relative_logit_difference'] for v in r['analysis']['comparisons'] if v['candidate']==f'case{t}-{policy}' and v['turn']==0 and v['stage']=='prefill') for t in range(3)]
        axes[0,1].bar(x+(i-1)*.24,logits,.23,label=label,color=color)
        for j,value in enumerate(logits):axes[0,1].text(x[j]+(i-1)*.24,value+.2,f'{value:.2f}',ha='center',fontsize=8)
    axes[0,0].set(xticks=x,xticklabels=names,ylim=(0,58),ylabel='Relative attention-output error (%)',title='A. Common initial prompt: both families matter');axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].set(xticks=x,xticklabels=names,ylim=(0,19),ylabel='Relative full-logit difference (%)',title='B. Later consumer changes the ranking');axes[0,1].legend(fontsize=8,ncol=2)
    terms=[('qk_only_relative','QK²','#697aa5'),('value_only_relative','V²','#178e83'),('interaction_relative','I²','#cb824b'),('qk_value_cross_scaled','2 QK·V','#9981b0'),('interaction_qk_cross_scaled','2 QK·I','#bc6868'),('interaction_value_cross_scaled','2 V·I','#8d6b48')]
    for i,(field,label,color) in enumerate(terms):
        values=[t['regions']['initial_prompt']['attention'][field]**2 if i<3 else t['regions']['initial_prompt']['attention'][field] for t in tasks]
        axes[1,0].bar(x+(i-2.5)*.13,values,.12,label=label,color=color)
    axes[1,0].axhline(0,color='#555555',lw=.8);axes[1,0].set(xticks=x,xticklabels=names,ylim=(-.16,.20),ylabel='Signed error power / native attention power',title='C. Interaction partly cancels the operand errors');axes[1,0].legend(fontsize=8,ncol=3)
    work=np.asarray([458752,819200,2981888,3145728]);barcolors=colors+['#aaaaaa'];worklabels=['Both compiled (CC)','Native values (CN)','Native query/key (NC)','Native (NN)']
    axes[1,1].barh(np.arange(4),work/1e6,color=barcolors);axes[1,1].invert_yaxis();axes[1,1].set(yticks=np.arange(4),yticklabels=worklabels,xlim=(0,3.5),xlabel='Selected QKV nominal MAC/update (millions)',title='D. Local work tradeoff; no latency claim')
    for i,cost in enumerate(work):axes[1,1].text(cost/1e6+.04,i,f'{cost/1e6:.3f}',va='center',fontsize=9)
    axes[1,1].text(.02,.02,'Full-token linear saving: CC 0.174% · CN 0.151% · NC 0.0106%',transform=axes[1,1].transAxes,fontsize=8)
    fig.suptitle('A native QK × value cut: local fidelity, consumer reversal and paid work',fontsize=15)
    target=root/'query-value-cut.png';fig.savefig(target,dpi=160);plt.close(fig);print(target)
if __name__=='__main__':main()
