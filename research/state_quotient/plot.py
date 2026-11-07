"""Publication figure from immutable saved trajectories; no model execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root):
    root=Path(root);report=json.loads((root/'report.json').read_text());paths={d['name']:d for d in report['driver']['paths']}
    def states(name):return np.fromfile(root/'raw'/f'{name}-states.f32',dtype='<f4').reshape(-1,paths[name]['rank'])
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained')
    color={'closed-plus':'#264d73','escape-plus':'#197a68','escape-minus':'#8358a3','escape-stale':'#bd4f38'}
    x=np.arange(17)
    ax=axes[0]
    ax.plot(x,states('closed-plus')[:,0],color=color['closed-plus'],label='Full + / full −')
    ax.scatter(x,states('closed-compact')[:,0],s=19,facecolors='none',edgecolors='#e69d22',label='Reduced own state')
    ax.set(title='Closed reads and writes',xlabel='Own updates',ylabel='Declared output x₀')
    ax.legend(fontsize=8)
    ax=axes[1]
    for name in ('escape-plus','escape-minus','escape-stale'):
        ax.plot(x,states(name)[:,0],color=color[name],label={'escape-plus':'Full +','escape-minus':'Full −','escape-stale':'Old reduced state'}[name])
    for name,ref in (('repair-plus','escape-plus'),('repair-minus','escape-minus')):
        ax.scatter(x,states(name)[:,0],s=21,facecolors='none',edgecolors=color[ref])
    ax.set(title='One escaping write; one-coordinate repair',xlabel='Own updates',ylabel='Declared output x₀')
    ax.legend(fontsize=8)
    ax=axes[2]
    for cand,ref,label,col in (('escape-stale','escape-plus','Stale vs full +','#197a68'),('escape-stale','escape-minus','Stale vs full −','#8358a3'),('repair-plus','escape-plus','Repaired vs full +','#e69d22')):
        a,b=states(cand),states(ref)
        err=np.linalg.norm(a[:,:2].astype(float)-b[:,:2].astype(float),axis=1)
        ax.plot(x,err,color=col,label=label)
    ax.set(title='Observed output discrepancy',xlabel='Own updates',ylabel='L2 difference in (x₀, x₁)')
    ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A compact nonlinear state works when its writes preserve what it discards',fontsize=13)
    target=root/'state-quotient.png';fig.savefig(target,dpi=180);plt.close(fig);print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');plot(p.parse_args().root)
