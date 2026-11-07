"""Saved-report cost and behavior summary; no model execution."""
import argparse,hashlib,json,re
from pathlib import Path
LABELS=['heldout-native','heldout-p0-layer16','heldout-p0','heldout-p1']
def summarize(job):
    report=json.loads((job/'report.json').read_text())
    cases={c['id']:c for c in json.loads((job/'raw/heldout-inputs.json').read_text())}
    rows=[]
    for label in LABELS:
        run=report['runs'][label];turns=run['metrics']['turns'];checks=[]
        for turn in turns:
            case=cases[turn['case_id']]
            if case.get('format') not in ['word','integer']:continue
            expected=case['expected'][turn['turn']];text=turn['answer'].strip()
            if case['format']=='word':correct=text.lower().rstrip('.!')==str(expected).lower()
            else:correct=bool(re.fullmatch(r'[+-]?\d+',text)) and int(text)==expected
            checks.append({'case_id':turn['case_id'],'turn':turn['turn'],'answer':text,'expected':expected,'strict_correct':correct,'truncated':turn['truncated']})
        costs=set(p['linear_macs']//p['count'] for p in run['driver']['paths'])
        calls=set(p['linear_calls']//p['count'] for p in run['driver']['paths'])
        if len(costs)!=1 or len(calls)!=1:raise ValueError('Nonuniform whole-token geometry')
        rows.append({'label':label,'MAC_per_token':costs.pop(),'calls_per_token':calls.pop(),'equal_case_turn_KL':run['metrics']['equal_case_turn_KL'],'equal_case_turn_logit_L2':run['metrics']['equal_case_turn_logit_relative_L2'],'strict_correct':sum(x['strict_correct'] for x in checks),'structured_answers':len(checks),'structured_checks':checks,'conditional_top_matches':sum(x['emitted']==x['native_shadow_top'] for x in run['metrics']['decisions']),'answer_decisions':len(run['metrics']['decisions'])})
    total_primary=sum(p['count'] for r in report['runs'].values() for p in r['driver']['paths'])
    return {'report_sha256':hashlib.sha256((job/'report.json').read_bytes()).hexdigest(),'rows':rows,'primary_transitions':total_primary,'native_shadow_transitions':total_primary,'sham_transitions':64,'whole_invocation_linear_calls':report['execution_totals']['total_linear_calls'],'whole_invocation_linear_MAC':report['execution_totals']['total_linear_macs'],'scope':'Six correlated two-turn conversations; eight structured answers plus prose; strict answer format and conditional-native fidelity are separate observations.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--image',type=Path);a=p.parse_args();summary=summarize(a.job)
    out=a.job/'analysis';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    if a.image:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=['Native','One layer','All layers\nnative fit','All layers\nstate refresh'];colors=['#34495e','#458a8d','#c87545','#795b9b']
        fig,axes=plt.subplots(1,3,figsize=(12,4));series=[([r['MAC_per_token']/1e9 for r in summary['rows']],'Nominal linear MACs per token','billions'),([r['equal_case_turn_KL'] for r in summary['rows']],'Conditional native KL','nats; lower is closer'),([r['strict_correct'] for r in summary['rows']],'Structured answers correct','out of 8; strict format')]
        for ax,(values,title,ylabel) in zip(axes,series):
            ax.bar(range(4),values,color=colors,width=.68);ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel,fontsize=9);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(values)*1.2+.01)
            for i,v in enumerate(values):ax.text(i,v+max(values)*.025,f'{v:.3g}',ha='center',fontsize=9)
        fig.suptitle('Rank 256 across 28 MLPs reduces arithmetic but loses these conversations',fontsize=13)
        fig.text(.5,.015,'Six fresh two-turn cases. Full residual, memory, attention and down maps retained. MAC counts are not measured serving speed.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.92]);a.image.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.image,dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
