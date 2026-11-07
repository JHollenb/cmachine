"""Join the precise wrong-token cut to immutable prior metrics and actual choices."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(root,prior):
    root,prior=Path(root),Path(prior);r=read(root/'report.json');old=read(prior/'report.json');raw=root/'raw';payload=root/'payload'
    if sha(prior/'report.json')!=r['baseline_report_sha256']:raise ValueError('Prior report differs')
    joins=[];boundaries=0;parent_groups={}
    for case in r['inputs']:
        context=case['context_tokens'];name=f'c0-n{context}-arm3-metrics.json';source=prior/'raw'/name
        if sha(source)!=old['raw_files'][name]['sha256']:raise ValueError('Prior metrics custody differs')
        row=next(x for x in read(source)['samples'] if x['clock']==case['cut_position']);baseline=next(x for x in r['driver']['rows'] if x.get('name')==f'case{context}-h0-arm0')
        if sha(raw/(baseline['name']+'-cut-logits.f32'))!=row['candidate_sha256'] or row['candidate_top']!=case['binding_token_id'] or row['native_top']!=case['labels']['correct']['ids'][0]:raise ValueError('Exact original binding-decision join differs')
        previous=read(payload/f'case{context}-previous.json')
        if baseline['consumed_ids']!=previous['consumed_ids']:raise ValueError('Historical complete baseline replay differs')
        if (raw/f'case{context}-h0-parent-memory.bin').read_bytes()!=(raw/f'case{context}-h1-parent-memory.bin').read_bytes():raise ValueError('Lower prototype parent differs')
        joins.append({'context':context,'cut_position':case['cut_position'],'prior_metrics_sha256':sha(source),'original_compiled_cut_logits_sha256':row['candidate_sha256'],'exact_current_bad_baseline_join':True,'original_full_native_shadow_top':row['native_top'],'compiled_top':row['candidate_top'],'lower_prototype_parent_byte_exact':True})
    for row in r['driver']['rows']:
        if 'alias_of' in row:continue
        case=next(x for x in r['inputs'] if x['context_tokens']==row['context']);prefix=case['parent_ids']+[case['cut_token']]
        if row['consumed_ids'][:len(prefix)]!=prefix:raise ValueError('Shared input prefix differs')
        key=(row['context'],row['history']);parent_groups.setdefault(key,set()).add(row['parent_hash'])
        for suffix,ids in [('cut-logits',row['answers'][0]),('answer-last-logits',row['answers'][0][-1:]),('followup-logits',row['answers'][1]),('followup-last-logits',row['answers'][1][-1:])]:
            logits=np.fromfile(raw/(row['name']+'-'+suffix+'.f32'),dtype='<f4')
            if logits.shape!=(151936,) or int(logits.argmax())!=ids[0]:raise ValueError('Native greedy boundary choice differs')
            boundaries+=1
    if any(len(x)!=1 for x in parent_groups.values()):raise ValueError('Branch parent fingerprint differs')
    result={'schema':'qk18-cut-lineage-audit-v1','report_sha256':sha(root/'report.json'),'prior_report_sha256':sha(prior/'report.json'),'binding_clock_joins':joins,'greedy_boundary_choices_verified':boundaries,'exact_parent_groups':len(parent_groups),'new_model_calls':0,'scope':'Independent immutable prior/current joins, lower parent identity, supplied shared prefix and retained native greedy choices. The first compass clause was already in the supplied prefix; it is not a fresh collateral outcome.'};(root/'lineage-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--prior',default=str(Path(__file__).resolve().parent.parent/'context_query/runs/job-fe51e7f4f0ee'));a=p.parse_args();audit(a.root,a.prior)
