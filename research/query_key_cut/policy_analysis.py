"""Audit frozen mask policies against same-ID native shadows and prior policies."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def audit(root,prior):
    root,prior=Path(root),Path(prior);r=read(root/'report.json');raw=root/'raw';payload=root/'payload';job=read(root/'job.json');submission=read(root/'submission.json');old=read(prior/'report.json')
    if sha(prior/'report.json')!=r['baseline_report_sha256'] or sha(root/'logs.txt')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Prior/log custody differs')
    reports=[json.loads(gzip.decompress(base64.b64decode(x.split('=',1)[1]))) for x in (root/'logs.txt').read_text().splitlines() if x.startswith('COMPOSITION_RESULT_GZIP_B64=')]
    if reports!=[r] or read(raw/'report.json')!=r:raise ValueError('Report log/raw differs')
    for p,h in submission['config']['sources'].items():
        if sha(payload/p)!=h:raise ValueError('Source hash differs '+p)
    for p,m in r['raw_files'].items():
        if sha(raw/p)!=m['sha256'] or (raw/p).stat().st_size!=m['bytes']:raise ValueError('Raw custody differs '+p)
    wrapper=r['saturn_receipt'];receipt=dict(wrapper['receipt']);fingerprint=receipt.pop('fingerprint')
    if hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=fingerprint or receipt['parent']!=wrapper['before'] or receipt['result']!=wrapper['after']:raise ValueError('Invocation receipt differs')
    observations=[];boundaries=0;max_error=0.
    for row in r['trials']:
        trace=read(raw/(row['name']+'.json'));metrics=read(raw/(row['name']+'-metrics.json'))['samples'];n=trace['count'];macs=1998848 if row['policy']==0 else 3801088
        if trace['operand_work']!={'applications':2*n,'code_calls':2*n,'compiled_calls':3*n,'native_calls':3*n,'linear_macs':macs*n} or trace['prototype_reads']!=n:raise ValueError('Physical operand work differs')
        if hashlib.sha256(np.asarray(trace['consumed_ids'],dtype='<i4').tobytes()).hexdigest()!=trace['trace_sha256']:raise ValueError('Trace differs')
        turns=[]
        for turn in (0,1):
            scores=[m for m in metrics if m['future'] and m['turn']==turn]
            if len(scores)!=len(trace['answers'][turn]):raise ValueError('Decision accounting differs')
            for m in scores:
                if m['candidate_top']!=trace['consumed_ids'][m['clock']+1]:raise ValueError('Own-token authority differs')
            turns.append(float(np.mean([m['kl'] for m in scores])))
        if turns!=row['per_turn_future_kl'] or sum(turns)/2!=row['future_mean_kl']:raise ValueError('Future aggregate differs')
        for m in metrics:
            if not m['boundary']:continue
            prefix=f'{row["name"]}-turn{m["turn"]}-b{m["boundary"]}';p=raw/(prefix+'-native.f32');q=raw/(prefix+'-candidate.f32')
            if sha(p)!=m['native_sha256'] or sha(q)!=m['candidate_sha256']:raise ValueError('Boundary hash differs')
            a=np.fromfile(p,dtype='<f4').astype(float);b=np.fromfile(q,dtype='<f4').astype(float)
            if int(np.argmax(a))!=m['native_top'] or int(np.argmax(b))!=m['candidate_top']:raise ValueError('Retained greedy boundary differs')
            a-=a.max();b-=b.max();ln=a-np.log(np.exp(a).sum());lc=b-np.log(np.exp(b).sum());error=abs(float(np.exp(ln)@(ln-lc))-m['kl']);max_error=max(max_error,error)
            if error>1e-10*max(1,abs(m['kl'])):raise ValueError('Boundary KL differs')
            boundaries+=1
        stem=f'c{row["task"]}-n{row["context_tokens"]}';old_names=[stem+'-arm2',stem+'-arm3'];comparison=[]
        for name in old_names:
            source=prior/'raw'/(name+'-audit.json')
            if sha(source)!=old['raw_files'][source.name]['sha256']:raise ValueError('Prior audit differs')
            comparison.append(read(source)['future_mean_kl'])
        observations.append({'schema':'qk18-on-policy-observation-v1','model_id':r['model']['identity'],'name':row['name'],'policy':'query-only18' if row['policy']==0 else 'key-only18','context_tokens':row['context_tokens'],'task':row['task'],'future_kl':row['future_mean_kl'],'per_turn_kl':turns,'prior_updated17_kl':comparison[0],'prior_both18_kl':comparison[1],'texts':r['texts'][row['name']],'stopped':trace['stopped'],'selected_linear_macs_per_token':macs,'nominal_whole_token_linear_macs':1543569408-2*3145728+macs})
    means={policy:float(np.mean([x['future_kl'] for x in observations if x['policy']==policy])) for policy in ('query-only18','key-only18')};a={'schema':'qk18-on-policy-analysis-v1','observations':observations,'means':means,'retained_logit_pairs':boundaries,'max_retained_kl_discrepancy':max_error,'scope':'Known development conversations, frozen source images/masks, no fit/selection/new heldout. Each policy has its own history and corresponding native same-ID shadow, so mean changes are policy trends, not isolated same-state effects.'};(root/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');(root/'observations.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in observations));v={'schema':'qk18-on-policy-verification-v1','report_sha256':sha(root/'report.json'),'source_files':len(submission['config']['sources']),'raw_files':len(r['raw_files']),'own_token_paths':8,'retained_logit_pairs':boundaries,'max_retained_kl_discrepancy':max_error,'work_counters_exact':True,'new_model_calls':0,'terminal_status':'not-requested'};(root/'verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'verification':v,'means':means}));return a
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--prior',default=str(Path(__file__).resolve().parent.parent/'context_query/runs/job-fe51e7f4f0ee'));a=p.parse_args();audit(a.root,a.prior)
