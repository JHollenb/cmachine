"""Saved-artifact scientific figure; no model evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root):
    root=Path(root);r=json.loads((root/'report.json').read_text());s=json.loads((root/'study.json').read_text());c=r['query_compiler'];labels=['Recall','Math','Prose','Silk recall','New math','Seed prose'];blue='#246b94';orange='#df8c37'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8.2));ax=axs[0,0]
    rows=sorted(c['proposals'],key=lambda p:p['dose']);ax.plot([p['dose'] for p in rows],[p['source_response_loss'] for p in rows],'-o',color=blue)
    chosen=c['proposals'][c['selected_index']];ax.scatter([chosen['dose']],[chosen['source_response_loss']],s=90,color=orange,zorder=3);ax.set(xlabel='Dose × predicted response step',ylabel='Source response squared loss',title='A. One direction, source-only selection')
    ax=axs[0,1];x=np.arange(6);rows=[x for x in r['analysis']['factorials'] if x['turn']==0 and x['stage']=='prefill'];a=[100*p['relative_logit_difference']['values-native-memory'] for p in rows];b=[100*p['relative_logit_difference']['query-fit-memory'] for p in rows]
    ax.bar(x-.18,a,.36,label='LS query',color=blue);ax.bar(x+.18,b,.36,label='Response query',color=orange);ax.set(xticks=x,xticklabels=labels,ylabel='Relative native-logit L2 (%)',title='B. All six initial logit distances improve');ax.tick_params(axis='x',rotation=23);ax.legend(frameon=False)
    ax=axs[1,0];rows=[p for p in s['readout_distributions'] if p['turn']==0 and p['stage']=='prefill'];g=[1000*p['kl_improvement_nats'] for p in rows];ax.bar(x,g,color=[blue if v>=0 else orange for v in g]);ax.axhline(0,color='#333333',linewidth=.7);ax.set(xticks=x,xticklabels=labels,ylabel='KL improvement (millinats)',title='C. Initial native-distribution fidelity is mixed');ax.tick_params(axis='x',rotation=23)
    ax=axs[1,1];q=c['proposals'];base=q[0];sel=q[c['selected_index']];ax.bar([0,1],[100*base['source_query_relative_error'],100*sel['source_query_relative_error']],color=[blue,orange]);ax.set(xticks=[0,1],xticklabels=['LS query','Response query'],ylabel='Source query-vector error (%)',title='D. Better response uses a less native query',ylim=(0,6.7));ax.text(.5,6.3,'Reader loss ↓8.36%; serving cost unchanged',ha='center',fontsize=10)
    fig.suptitle('Predict the memory response, then test the native continuation',fontsize=16,y=.985);fig.text(.5,.015,'Qwen2.5-1.5B / layer17 • same B128, compiled K, native V, 98,464-byte memory • 24 paths / 2,412 advances\nSource: 193 correlated rows in 3 trajectories. Answers unchanged; no general chat or latency claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.955),h_pad=2,w_pad=2);target=root/'query-response.png';fig.savefig(target,dpi=170);plt.close(fig);print(target)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');plot(p.parse_args().root)
