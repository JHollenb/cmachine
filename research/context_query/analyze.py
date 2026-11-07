"""Saved-only custody, declared update, own-token and native-consumer analysis."""
import argparse,base64,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def analyze(root):
    root=Path(root);raw=root/'raw';r=read(root/'report.json');inputs=read(raw/'input-manifest.json');rows=[];observations=[]
    for trial in r['trials']:
        if 'alias_of' in trial:continue
        name=trial['name'];trace=read(raw/(name+'.json'));metrics=read(raw/(name+'-metrics.json'));audit=read(raw/(name+'-audit.json'));per=[]
        if hashlib.sha256(np.asarray(trace['consumed_ids'],dtype='<i4').tobytes()).hexdigest()!=trace['trace_sha256']:raise ValueError('Consumed trace hash differs')
        work=trace['operand_work'];sites=int(trace['policy']>=0)+int(trace['policy']>=4)
        if work!={'applications':sites*trace['count'],'code_calls':sites*trace['count'],'compiled_calls':2*sites*trace['count'],'native_calls':sites*trace['count'],'linear_macs':819200*sites*trace['count']}:raise ValueError('Physical program counters differ')
        if trace['prototype_reads']!=(int(trace['policy']>=0)+int(trace['policy']==5))*trace['count']:raise ValueError('Memory append/read coverage differs')
        for t in (0,1):
            future=[m for m in metrics['samples'] if m['future'] and m['turn']==t];prefill=[m for m in metrics['samples'] if not m['future'] and m['turn']==t]
            if len(future)!=len(trace['answers'][t]) or not future:raise ValueError('Future sample coverage differs')
            for m in future:
                if m['candidate_top']!=trace['consumed_ids'][m['clock']+1] or trace['prediction_turn'][m['clock']]!=t:raise ValueError('Future own-token attribution differs')
            per.append(float(np.mean([m['kl'] for m in future])))
            observations.append({'schema':'context-query-observation-v1','model_id':r['model']['identity'],'name':name,'source':trial['source'],'policy':trial['policy'],'turn':t,'future_decisions':len(future),'future_kl_nats':per[-1],'prefill_samples':len(prefill),'sampled_prefill_kl_nats':float(np.mean([m['kl'] for m in prefill])) if prefill else None,'native_top_agreement':float(np.mean([m['native_top']==m['candidate_top'] for m in future])),'stopped':trace['stopped'][t]})
        if per!=audit['per_turn_future_kl'] or sum(per)/2!=trial['future_mean_kl']:raise ValueError('Future aggregation differs')
        if trace['policy']==-1 and any(m['kl'] or m['native_sha256']!=m['candidate_sha256'] for m in metrics['samples']):raise ValueError('Native identity arm differs')
        rows.append({'name':name,'policy':trial['policy'],'source':trial['source'],'count':trace['count'],'future_kl':sum(per)/2,'per_turn_future_kl':per,'sampled_prefill_kl':audit['sampled_prefill_kl'],'future_native_top_agreement':audit['native_top_agreement_future'],'answer_lengths':[len(a) for a in trace['answers']],'stopped':trace['stopped'],'candidate_seconds':trace['candidate_seconds'],'shadow_seconds':trace['shadow_seconds'],'session_gpu_bytes':trace['allocated_session_bytes'],'selected_program_macs':work['linear_macs'],'prototype_reads':trace['prototype_reads']})
    chosen=r['selected_index'];feedback=[]
    for i in range(4):
        group=[x for x in rows if x['source'] and x['policy']==i];feedback.append({'index':i,'dose':r['fit']['proposals'][i]['dose'],'future_kl':float(np.mean([x['future_kl'] for x in group])),'per_source':[x['future_kl'] for x in group],'local_response_loss':r['fit']['proposals'][i]['local_source_loss']})
    expected=min(range(4),key=lambda i:(feedback[i]['future_kl'],abs(feedback[i]['dose']),i))
    if expected!=chosen:raise ValueError('Selection differs')
    for actual,saved in zip(feedback,r['feedback']):
        if actual['future_kl']!=saved['future_kl'] or actual['index']!=saved['index']:raise ValueError('Feedback aggregate differs')
    fresh=[]
    for case in inputs:
        group=[]
        for arm in range(5):
            name=case['stem']+f'-arm{arm}';trial=next(x for x in r['trials'] if x['name']==name);actual=trial.get('alias_of',name);row=next(x for x in rows if x['name']==actual);group.append({'arm':arm,**row,'name':name,'actual_name':actual,'texts':r['texts'][actual]})
        fresh.append({'kind':case['kind'],'context_tokens':case['context_tokens'],'arms':group,'query_update_relative_change':group[2]['future_kl']/group[1]['future_kl']-1,'later_qk_delta_kl':group[3]['future_kl']-group[2]['future_kl'],'later_memory_delta_kl':group[4]['future_kl']-group[3]['future_kl']})
    means=[float(np.mean([case['arms'][i]['future_kl'] for case in fresh])) for i in range(5)]
    return {'schema':'context-query-analysis-v1','source_feedback':feedback,'selected_index':chosen,'source_relative_change':feedback[chosen]['future_kl']/feedback[0]['future_kl']-1,'fresh':fresh,'fresh_arm_mean_kl':means,'fresh_query_update_relative_change':means[2]/means[1]-1,'source_direction_local_relative_change':r['fit']['proposals'][chosen]['local_source_loss']/r['fit']['baseline_loss']-1,'observations':observations,'rows':rows,'scope':'Exploratory conditional fidelity, equally weighted conversations and turns; each policy supplies its own answer history. Different candidate domains prevent interpreting differences as isolated same-state causal effects. No task-truth, universal closure or speed certification.'}
def verify(root):
    root=Path(root);raw=root/'raw';r=read(root/'report.json');submission=read(root/'submission.json');job=read(root/'job.json');payload=root/'payload'
    if job['job_id']!=submission['job_id'] or sha(root/'logs.txt')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Scheduler log join differs')
    logged=[json.loads(gzip.decompress(base64.b64decode(line.split('=',1)[1]))) for line in (root/'logs.txt').read_text().splitlines() if line.startswith('COMPOSITION_RESULT_GZIP_B64=')]
    if logged!=[r] or read(raw/'report.json')!=r:raise ValueError('Report log/raw join differs')
    for name,h in submission['config']['sources'].items():
        if sha(payload/name)!=h:raise ValueError('Payload source differs '+name)
    for name,record in r['raw_files'].items():
        if sha(raw/name)!=record['sha256'] or (raw/name).stat().st_size!=record['bytes']:raise ValueError('Raw custody differs '+name)
    fit=read(raw/'fit.json');base=np.fromfile(payload/'query-fit.f32',dtype='<f4').reshape(1536,128).astype(float);direction=np.fromfile(raw/'source-direction.f64',dtype='<f8').reshape(base.shape)
    for p in fit['proposals']:
        image=np.fromfile(raw/f'query-proposal-{p["index"]}.f32',dtype='<f4').reshape(base.shape)
        if not np.array_equal(image,np.asarray(base+p['step']*direction,dtype='<f4')) or p['step']!=p['dose']*fit['alpha'] or sha(raw/f'query-proposal-{p["index"]}.f32')!=p['sha256']:raise ValueError('Declared update differs')
    tx=read(raw/'image-transactions.json')
    if len(tx['proposals'])!=4 or not all(p['restored_parent_exact'] and p['restore']['result']==tx['parent'] for p in tx['proposals']):raise ValueError('Image restoration differs')
    # The imported transaction helper records accepted parent index0. The later
    # suffix selection seal is the sole authority for choosing a serving image.
    seal=read(raw/'selection-seal.json');freshseal=read(raw/'fresh-seal.json')
    if seal['selected_index']!=r['selected_index'] or seal['fit_sha256']!=sha(raw/'fit.json') or freshseal['selection_sha256']!=sha(raw/'selection-seal.json') or freshseal['basis18_sha256']!=sha(raw/'basis18.f32'):raise ValueError('Visibility seal differs')
    wrapper=r['saturn_receipt'];receipt=dict(wrapper['receipt']);fingerprint=receipt.pop('fingerprint')
    if hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=fingerprint or receipt['parent']!=wrapper['before'] or receipt['result']!=wrapper['after']:raise ValueError('Invocation receipt differs')
    analysis=analyze(root);write(root/'analysis.json',analysis);(root/'observations.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in analysis['observations']))
    result={'schema':'context-query-verification-v1','report_sha256':sha(root/'report.json'),'scheduler_log_seal':True,'source_files':len(submission['config']['sources']),'raw_files':len(r['raw_files']),'exact_declared_updates':4,'exact_image_restorations':4,'own_token_turns':len(analysis['observations']),'native_identity_paths':sum(row['policy']==-1 for row in analysis['rows']),'analysis_sha256':sha(root/'analysis.json'),'new_model_calls':0,'terminal_status':'not-requested','qualification':'Runtime assertions and in-worker C/Python sampled-stream checks are hash-bound. This verifier independently recomputes saved aggregations and image bytes, not discarded full streams or a model forward.'};write(root/'verification.json',result);print(json.dumps({'verification':result,'summary':{k:analysis[k] for k in ('selected_index','source_relative_change','fresh_arm_mean_kl','fresh_query_update_relative_change')}}));return analysis
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');verify(p.parse_args().root)
