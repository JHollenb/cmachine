"""Render the recorded scorer budgets and paired scalar cases, without forwards."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(result,output):
    result=Path(result);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    verification=json.loads((result/'verification.json').read_text())
    seal=json.loads((result/'raw/capability-seal.json').read_text());variants={v['label']:v for v in seal['variants']}
    rows={(r['variant'],r['goal']):r for r in verification['observations']}
    labels=[('v1-full','Full V1'),('v1-matched-global-union8','Global 14'),('v1-union8','Specialist union 14'),
        ('v1-specialist-planted','Planted specialist 8'),('v1-global14-proposal-union1','Global 14 + one proposal'),
        ('v2a-full','Full V2a'),('v2a-proposal-union1','V2a + one proposal'),('broad-only','Structural rule only')]
    goals=[('test-binomial-3-0','Binomial 3\nlayout A',5),('test-binomial-4-0','Binomial 4\nlayout A',7),
        ('test-binomial-4-1','Binomial 4\nlayout B',7),('test-trinomial-3','Trinomial 3\noriginal',8),
        ('replication-trinomial-3','Trinomial 3\nnew layout',8)]
    values=np.array([[rows.get((label,goal),{}).get('cost',np.nan) for goal,_,_ in goals] for label,_ in labels])
    gap=values-np.array([ref for _,_,ref in goals])[None,:]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,6.9),facecolor='white');grid=fig.add_gridspec(1,2,width_ratios=[1,1.6],wspace=.58)
    left=fig.add_subplot(grid[0]);positions=np.arange(4)
    budgets=[('v1-full','Full V1'),('v1-matched-global-union8','Global 14'),('v1-union8','Union 14'),('v1-specialist-planted','Specialist 8')]
    left.barh(positions,[variants[k]['learned_coefficients'] for k,_ in budgets],color=['#8a97a7','#327cad','#aa6e40','#439b86'],height=.54)
    left.set_yticks(positions,[label for _,label in budgets]);left.invert_yaxis();left.set_xlim(0,1000)
    left.set_xlabel('Learned scorer coefficients');left.set_title('Smaller numerical scorers',loc='left',fontweight='bold',pad=24)
    left.spines[['left','bottom']].set_visible(False);left.tick_params(length=0);left.xaxis.grid(True,color='#e5e9ef');left.set_axisbelow(True)
    for i,(key,_) in enumerate(budgets):
        spec=variants[key];left.text(spec['learned_coefficients']+12,i,f"{spec['learned_coefficients']}\n{spec['linear_macs_per_candidate']} MACs / candidate",va='center',fontsize=10)
    left.text(0,-.2,'42 normalization floats remain in each V1 scorer.\nBinary size: full 5,172 B; global 14 is 3,516 B.',transform=left.transAxes,va='top',fontsize=10,color='#4b5563')
    right=fig.add_subplot(grid[1]);image=right.imshow(np.ma.masked_invalid(np.log1p(np.maximum(gap,0))),cmap='YlOrRd',vmin=0,vmax=np.log1p(53),aspect='auto')
    right.set_yticks(np.arange(len(labels)),[label for _,label in labels]);right.set_xticks(np.arange(len(goals)),[label for _,label,_ in goals],fontsize=10)
    right.xaxis.tick_top();right.tick_params(length=0,pad=9)
    for i in range(len(labels)):
        for j in range(len(goals)):
            value=values[i,j];right.text(j,i,'—' if np.isnan(value) else str(int(value)),ha='center',va='center',color='white' if not np.isnan(value) and gap[i,j]>20 else '#24313f',fontweight='bold',fontsize=12)
    right.set_title('Extracted scalar program cost · lower is better',loc='left',fontweight='bold',pad=52)
    right.spines[:].set_visible(False)
    right.text(0,-.12,'Constructive reference costs: 5, 7, 7, 8, 8. Color shows cost above each reference.\nA dash means that policy was not run on that follow-up case.',transform=right.transAxes,va='top',fontsize=10,color='#4b5563')
    fig.suptitle('Compression and composition have different effects',x=.06,y=.98,ha='left',fontsize=18,fontweight='bold',color='#24313f')
    fig.text(.06,.015,'Frozen weights · original search budgets · paired cases · both trinomial layouts share one mathematical construction',fontsize=10,color='#4b5563')
    fig.subplots_adjust(left=.12,right=.98,top=.77,bottom=.22)
    for extension in ('png','svg'):fig.savefig(output/('math-capability-unions.'+extension),dpi=170,facecolor='white')
    svg=output/'math-capability-unions.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (output/'math-capability-unions.json').write_text(json.dumps({'job_id':result.name,'labels':labels,'goals':goals,
        'costs':[[None if np.isnan(x) else int(x) for x in row] for row in values]},indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    render(args.result,args.output)
