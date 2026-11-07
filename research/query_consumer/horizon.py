"""Saved-only clock/trajectory selector contrast and pending-token authority."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from metrics import choose,DOSES
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root):
    root=Path(root);source=json.loads((root/'raw/source-provenance.json').read_text());curves=[json.loads((root/f'raw/source-metrics-{i}.json').read_text()) for i in range(6)];rows=[];counts=source['source_trajectories']
    for c,s in enumerate(counts):
        prompt=len(s['prompt_ids']);expected=s['own_ids'];offset=sum(x['advances'] for x in counts[:c])
        for label,index in [('native',None)]+[(f'proposal{i}',i) for i in range(6)]:
            file=root/f'raw/source{c}-{label}-tops.i32';tops=np.fromfile(file,dtype='<i4');metrics=curves[0 if index is None else index]['rows'][offset:offset+s['advances']];field='native_top' if index is None else 'candidate_top'
            if tops.shape!=(s['advances'],) or tops.tolist()!=[m[field] for m in metrics]:raise ValueError('Source pending stream differs from independently audited full-vocabulary argmax')
            predicted=tops[prompt-1:prompt+7].tolist();first=next((i for i,(a,b) in enumerate(zip(predicted,expected)) if a!=b),None)
            rows.append({'source':c,'policy':label,'tops_sha256':sha(file),'predicted_suffix_ids':predicted,'forced_suffix_ids':expected,'all_eight_pending_predictions_match':len(predicted)==8 and first is None,'first_disagreement':first})
    regional={}
    for region in ('shared-header','prompt-body','historical-own-suffix'):
        by_image=[]
        for image in curves:
            per=[float(np.mean([r['kl_nats'] for r in image['rows'] if r['source']==c and r['region']==region])) for c in range(3)];by_image.append({'proposal':image['proposal'],'dose':DOSES[image['proposal']],'per_trajectory_mean_kl_nats':per,'equal_trajectory_mean_kl_nats':float(np.mean(per))})
        regional[region]={'curves':by_image,'equal_trajectory_selected_index':choose([r['equal_trajectory_mean_kl_nats'] for r in by_image]),'per_trajectory_selected_indices':[choose([r['per_trajectory_mean_kl_nats'][c] for r in by_image]) for c in range(3)]}
    result={'schema':'tcm-source-query-horizon-audit-v1','report_sha256':sha(root/'report.json'),'analysis_recovery_sha256':sha(root/'analysis-recovery.json'),'source_pending_checks':rows,'all_eight_pending_predictions_match':all(r['all_eight_pending_predictions_match'] for r in rows),'individually_verified_next_token_choices':len(rows)*8,'regional_selector_diagnostics':regional,'scope':'Post-hoc saved-only diagnostic; these regional choices do not change the sealed selected image or target results. Source selection was forced by design; saved argmax induction shows the same eight source suffix tokens would be produced greedily on these exact traces. No claim beyond eight tokens or a new autonomous source execution.'}
    (root/'horizon-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'source_traces':len(rows),'all_eight_pending_match':result['all_eight_pending_predictions_match'],'region_selected_doses':{k:DOSES[v['equal_trajectory_selected_index']] for k,v in regional.items()}}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');audit(p.parse_args().root)
