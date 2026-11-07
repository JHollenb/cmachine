"""Saved-only research figure; no model calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def load(job,name):return json.loads((ROOT/'runs'/job/name).read_text())
def main():
    original=load('job-2230480e1c18','study.json');weights=load('job-077becbef4c9','study.json');source=load('job-c7334bbc5bb8','study.json');report=load('job-c7334bbc5bb8','report.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained');x=np.arange(3);names=['Recall','Arithmetic','Prose'];colors=['#7583a5','#be7a4b','#e0ad77','#178e83','#6c568f']
    series=[('Activation 32',original,'project'),('Weights 32',weights,'project32'),('Weights 128',weights,'project128'),('Activation 128',source,'project128')]
    for i,(label,data,suffix) in enumerate(series):
        values=[100*next(r['attention_relative_l2'] for r in data['common_initial_history_operands'] if r['candidate']==f'case{t}-{suffix}') for t in range(3)]
        axes[0,0].bar(x+(i-1.5)*.19,values,.18,label=label,color=colors[i])
    axes[0,0].axhline(100,color='#555555',ls=':',lw=1,label='Zero attention')
    axes[0,0].set(xticks=x,xticklabels=names,ylim=(0,165),ylabel='Relative attention-output error (%)',title='A. Query compiler: common initial prompt')
    axes[0,0].legend(fontsize=8,ncol=2)
    for i,(suffix,label,color) in enumerate([('project32','Activation 32',colors[0]),('project128','Activation 128',colors[3]),('both128','Activation 128 + memory',colors[4]),('memory','Memory only','#67a658'),('zero','Zero attention','#9a9a9a')]):
        values=[100*next(r['relative_logit_difference'] for r in report['analysis']['comparisons'] if r['candidate']==f'case{t}-{suffix}' and r['turn']==0 and r['stage']=='prefill') for t in range(3)]
        axes[0,1].bar(x+(i-2)*.16,values,.15,label=label,color=color)
    axes[0,1].set(xticks=x,xticklabels=names,ylim=(0,26),ylabel='Relative full-logit difference (%)',title='B. First generation readout on identical history');axes[0,1].legend(fontsize=8,ncol=2)
    for suffix,label,color in [('memory','Native QKV + memory','#67a658'),('both32','Activation 32 + memory',colors[0]),('both128','Activation 128 + memory',colors[3])]:
        observations=[r for r in report['analysis']['observations'] if r.get('kind')=='own-query-attention' and r['path']==f'case2-{suffix}']
        axes[1,0].plot(np.arange(1,len(observations)+1),[100*r['own_kv_attention_relative_error'] for r in observations],label=label,color=color,lw=1.3)
    axes[1,0].axvline(32,color='#555555',ls=':',lw=1);axes[1,0].set(xlabel='Own consumed-token position',ylabel='Relative own-KV attention error (%)',title='C. Compression loss on each prose branch’s own Q/K/V');axes[1,0].legend(fontsize=8)
    n=np.arange(1,513);axes[1,1].plot(n,2048*n/1024,label='Native selected KV: valid prefix',color='#7583a5');axes[1,1].axhline(98464/1024,label='Compact logical state: 96.16 KiB',color='#178e83');axes[1,1].axhline(1024,label='Original arena still allocated: 1024 KiB',color='#9a9a9a',ls='--');axes[1,1].set(xlabel='Consumed tokens',ylabel='Selected-layer KiB',title='D. Logical reader size versus retained allocation');axes[1,1].legend(fontsize=8)
    fig.suptitle('An own-query memory loop: improved source compilation, remaining approximation',fontsize=15)
    destination=ROOT/'runs/job-c7334bbc5bb8/query-memory.png';fig.savefig(destination,dpi=160);plt.close(fig);print(destination)
if __name__=='__main__':main()
