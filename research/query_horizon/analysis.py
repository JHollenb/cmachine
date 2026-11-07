"""Source prediction-clock authority, memory replay and sealed target comparison."""
import hashlib,json
from pathlib import Path
import numpy as np
from reader import Memory,full_read,rel,stats
from metrics import distribution,choose,DOSES,future_scores
V=151936
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(raw,name,shape):
    a=np.fromfile(raw/name,dtype='<f4')
    if a.size!=int(np.prod(shape)) or not np.isfinite(a).all():raise ValueError('Invalid recorded operand '+name)
    return a.reshape(shape)
def operands(raw,name,n):return {key:load(raw,f'{name}-{key}.f32',shape) for key,shape in {'norm':(n,1536),'q':(n,12,128),'k':(n,2,128),'v':(n,2,128),'attention':(n,12,128),'output':(n,1536),'coeff':(n,128)}.items()}
def source_operands(raw,name,n):return {key:load(raw,f'{name}-{key}.f32',shape) for key,shape in {'q':(n,12,128),'k':(n,2,128),'v':(n,2,128),'attention':(n,12,128)}.items()}
def validate_trace(trace,inputs,eos):
    history=[];phases=[];prediction=[-1]*trace['count']
    if trace['prompt_length']!=len(inputs['prompt_ids']) or trace['followup_length']!=len(inputs['followup_ids']) or len(trace['turns'])!=2:raise ValueError('Source prompt/followup/turn geometry differs')
    for turn,s in enumerate(trace['turns']):
        ids=inputs['prompt_ids'] if turn==0 else inputs['followup_ids'];skip=int(turn==1 and history[-1]==ids[0])
        if s['turn']!=turn or s['start']!=len(history) or s['template_skip']!=skip:raise ValueError('Source template start differs')
        history+=ids[skip:];phases += [0 if turn==0 else 2]*(len(ids)-skip)
        if s['prefill_end']!=len(history) or not 1<=len(s['answer_ids'])<=32:raise ValueError('Source prefill/budget differs')
        clocks=list(range(len(history)-1,len(history)-1+len(s['answer_ids'])))
        if s['prediction_clocks']!=clocks:raise ValueError('Source prediction clock is not before consumed token')
        for t in clocks:prediction[t]=turn
        history+=s['answer_ids'];phases += [1 if turn==0 else 3]*len(s['answer_ids'])
        if s['final']!=len(history) or s['stopped']!=(s['answer_ids'][-1] in eos) or s['truncated']==s['stopped'] or (not s['stopped'] and len(s['answer_ids'])!=32):raise ValueError('Source EOS/final/budget authority differs')
    if history!=trace['consumed_ids'] or phases!=trace['input_phases'] or prediction!=trace['prediction_turn'] or len(history)!=trace['count']:raise ValueError('Source complete token/phase trace differs')
    return prediction

def source_audit(raw,driver):
    selection=json.loads((raw/'source-selection.json').read_text());seal=json.loads((raw/'selection-seal.json').read_text());family=json.loads((raw/'family-provenance.json').read_text());original=json.loads((raw/'source-provenance.json').read_text());inputs=json.loads((raw/'source-input-manifest.json').read_text());traces=json.loads((raw/'source-traces.json').read_text())['sources'];prefix=json.loads((raw/'prefix-provenance.json').read_text());historical_control=json.loads((raw/'historical-control.json').read_text());total=sum(t['count'] for t in traces)
    if sha(raw/'source-selection.json')!=driver['source_selection_sha256'] or sha(raw/'source-selection.json')!=seal['selection_sha256'] or seal['phase']!='before-parsing-fresh-prompts-or-tokenizing-targets' or selection['source_advances']!=7*total or selection['source_native_advances']!=total or selection['source_candidate_advances']!=6*total or selection['source_lengths']!=[t['count'] for t in traces] or selection['source_traces_sha256']!=sha(raw/'source-traces.json'):raise ValueError('Source selection identity/geometry differs')
    for name,r in family['files'].items():
        if sha(raw/name)!=r['sha256'] or (raw/name).stat().st_size!=r['bytes']:raise ValueError('Frozen image family differs')
    selected=selection['selected_index']
    if selected!=driver['selected_index'] or sha(raw/'selected-query.f32')!=family['files'][f'query-proposal-{selected}.f32']['sha256'] or sha(raw/'selected-query.f32')!=seal['image_sha256'] or selection['selected_image_sha256']!=seal['image_sha256'] or historical_control['selection']['selected_index']!=5 or sha(raw/'program128-q.f32')!=family['files']['query-proposal-5.f32']['sha256']:raise ValueError('Selected/historical image differs')
    expected_native=load(raw,'source-inputs.f32',(193,1536));old_offset=0;native=[];native_norm=[];native_norm_hash=[];prefix_arrays=0
    def prefix_check(name,array,c):
        n=prefix['source_lengths'][c];data=np.asarray(array[:n]).tobytes();record=prefix['files'][name]
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Historical source prefix operand differs: '+name)
    for c,t in enumerate(traces):
        validate_trace(t,inputs['sources'][c],inputs['native_eos_token_ids']);s=original['source_trajectories'][c];old_ids=s['prompt_ids']+s['own_ids']
        if t['source']!=c or t['consumed_ids'][:len(old_ids)]!=old_ids or t['turns'][0]['answer_ids'][:8]!=s['own_ids']:raise ValueError('Original source prefix IDs differ')
        n=t['count'];a=source_operands(raw,f'source{c}-native',n);norm=load(raw,f'source{c}-native-norm.f32',(n,1536));oldn=prefix['source_lengths'][c]
        if not np.array_equal(norm[:oldn],expected_native[old_offset:old_offset+oldn]):raise ValueError('Native source prefix receiver differs')
        old_offset+=oldn;native.append(a);native_norm.append(norm);native_norm_hash.append([hashlib.sha256(row.tobytes()).hexdigest() for row in norm]);tops=np.fromfile(raw/f'source{c}-native-tops.i32',dtype='<i4')
        for key,array in [*a.items(),('tops',tops)]:prefix_check(f'source{c}-native-{key}.'+('i32' if key=='tops' else 'f32'),array,c);prefix_arrays+=1
    curves=[];snapshot_rows=[];observations=[];policy_rows=[];means=[];prefix_logits=0
    for i in range(6):
        primary=json.loads((raw/f'source-metrics-{i}.json').read_text());audit=json.loads((raw/f'source-audit-{i}.json').read_text());old=json.loads((raw/f'historical-source-metrics-{i}.json').read_text())
        if sha(raw/f'source-audit-{i}.json')!=seal['source_audit_sha256'][i] or primary['proposal']!=i or audit['proposal']!=i or len(primary['rows'])!=total or len(audit['rows'])!=total:raise ValueError('Source live audit identity differs')
        recomputed,per,turnmeans=future_scores(primary['rows'],traces)
        if abs(recomputed-primary['equal_trajectory_mean_kl_nats'])>1e-12 or abs(recomputed-selection['mean_kl_nats'][i])>1e-12:raise ValueError('Future selector aggregation differs')
        means.append(recomputed);offset=0;old_offset=0;replay=[];response=[];query=[];window_values={key:[] for key in ('first8-decisions','later-first-answer-decisions','post-followup-decisions','old-eight-consumed-rows','extended-all-clock')}
        for c,t in enumerate(traces):
            n=t['count'];a=source_operands(raw,f'source{c}-proposal{i}',n);m=Memory();tops=np.fromfile(raw/f'source{c}-proposal{i}-tops.i32',dtype='<i4');native_tops=np.fromfile(raw/f'source{c}-native-tops.i32',dtype='<i4');oldn=prefix['source_lengths'][c]
            for key,array in [*a.items(),('tops',tops)]:prefix_check(f'source{c}-proposal{i}-{key}.'+('i32' if key=='tops' else 'f32'),array,c);prefix_arrays+=1
            if not np.array_equal(a['v'],native[c]['v']) or tops.shape!=(n,):raise ValueError('Native V/source pending extent differs')
            keep=sorted(set([clock for turn in t['turns'] for clock in (turn['prediction_clocks'][0],turn['prediction_clocks'][-1])]))
            on_policy=True;first=None;turn_first=[None,None]
            for step in range(n):
                row=primary['rows'][offset+step];other=audit['rows'][offset+step];turn=t['prediction_turn'][step];region='first-answer' if turn==0 else 'post-followup-answer' if turn==1 else 'prefill' if t['input_phases'][step]==0 else 'followup-ingest' if t['input_phases'][step]==2 else 'unused-answer-state'
                if row['source']!=c or row['step']!=step or row['region']!=region or row['prediction_turn']!=turn or row['receiver_sha256']!=native_norm_hash[c][step] or row['greedy_prefix_before_decision']!=on_policy or row['candidate_top']!=int(tops[step]) or row['native_top']!=int(native_tops[step]):raise ValueError('Source clock/receiver/authority/greedy prefix differs')
                for key in ('kl_nats','relative_logit_l2','native_top_logp_delta_nats','native_entropy_nats'):
                    if abs(row[key]-other[key])>1e-10*max(1.,abs(row[key]),abs(other[key])):raise ValueError('Stored dual source calculation disagrees')
                if turn>=0:
                    expected=t['consumed_ids'][step+1]
                    if row['expected_next_token']!=expected or row['native_top']!=expected:raise ValueError('Native source generated-token authority differs')
                    if row['candidate_top']!=expected:
                        if turn_first[turn] is None:turn_first[turn]={'clock':step,'answer_token_index':t['turns'][turn]['prediction_clocks'].index(step),'predicted_token':row['candidate_top'],'forced_token':expected}
                        if first is None:first={'turn':turn,**turn_first[turn]}
                        on_policy=False
                elif row['expected_next_token']!=-1:raise ValueError('Non-generation source clock has a forced next token')
                if step<oldn:
                    oldrow=old['rows'][old_offset+step]
                    if row['native_logits_sha256']!=oldrow['native_logits_sha256'] or row['candidate_logits_sha256']!=oldrow['candidate_logits_sha256']:raise ValueError('Historical source full-logit prefix differs')
                    prefix_logits+=1
                m.append(a['k'][step],a['v'][step]);read=m.read(a['q'][step]);replay.append(rel(read,a['attention'][step]))
                if not np.array_equal(read,a['attention'][step]):raise ValueError('Source compact read replay differs')
                if step in keep:
                    nn=f'source{c}-native-t{step}-logits.f32';cn=f'source{c}-proposal{i}-t{step}-logits.f32';metric=distribution(load(raw,nn,(V,)),load(raw,cn,(V,)))
                    if sha(raw/nn)!=row['native_logits_sha256'] or sha(raw/cn)!=row['candidate_logits_sha256']:raise ValueError('Durable source vector differs from stream')
                    for key,value in metric.items():
                        if abs(value-row[key])>1e-10*max(1.,abs(value),abs(row[key])):raise ValueError('Durable source metric replay differs')
                    snapshot_rows.append({'proposal':i,'source':c,'step':step,'kl_nats':metric['kl_nats']})
                observations.append({'kind':'source-future-distribution','proposal':i,'source':c,'step':step,'prediction_turn':turn,'region':region,'kl_nats':row['kl_nats'],'greedy_prefix_before_decision':row['greedy_prefix_before_decision'],'compact_read_replay_relative':replay[-1]})
            policy_rows.append({'proposal':i,'source':c,'native_answer_lengths':[len(x['answer_ids']) for x in t['turns']],'first_disagreement':first,'turn_specific_first_disagreements':turn_first,'complete_greedy_trace_matches':first is None})
            source_rows=primary['rows'][offset:offset+n];clocks0=t['turns'][0]['prediction_clocks'];clocks1=t['turns'][1]['prediction_clocks'];windows={'first8-decisions':clocks0[:8],'later-first-answer-decisions':clocks0[8:],'post-followup-decisions':clocks1,'old-eight-consumed-rows':list(range(t['prompt_length'],t['prompt_length']+8)),'extended-all-clock':list(range(n))}
            for name,clocks in windows.items():window_values[name].append(float(np.mean([source_rows[k]['kl_nats'] for k in clocks])) if clocks else None)
            response.append(rel(a['attention'],native[c]['attention']));query.append(rel(a['q'],native[c]['q']));offset+=n;old_offset+=oldn
        allmean=float(np.mean(window_values['extended-all-clock']))
        if abs(allmean-primary['extended_all_clock_mean_kl_nats'])>1e-12:raise ValueError('Extended all-clock aggregation differs')
        curves.append({'proposal':i,'dose':DOSES[i],'future_mean_kl_nats':recomputed,'per_source_future_kl_nats':per,'per_source_per_turn_kl_nats':turnmeans,'window_per_source_kl_nats':window_values,'window_equal_source_kl_nats':{k:float(np.mean(v)) if all(x is not None for x in v) else None for k,v in window_values.items()},'source_attention_relative_error':response,'source_query_relative_error':query,'compact_replay_max_relative':max(replay),'maximum_c_python_discrepancy':audit['maximum_c_python_discrepancy']})
    if choose(means)!=selected:raise ValueError('Saved future image selection differs')
    return {'selected_index':selected,'source_native_advances':total,'source_advances':total*7,'source_compact_reads_exact':total*6,'full_clock_dual_calculation_rows':total*6,'durable_metric_replay_rows':len(snapshot_rows),'distinct_durable_full_logit_vectors':len(snapshot_rows)//6*7,'historical_prefix_arrays_exact':prefix_arrays,'historical_prefix_full_logit_pairs_exact':prefix_logits,'native_source_receiver_prefix_exact':True,'source_curves':curves,'source_snapshot_metrics':snapshot_rows,'greedy_prefix_audit':policy_rows,'source_traces':traces,'observations':observations,'scope':'Future prediction clocks before consuming token; fixed native source IDs, candidate own memory/caches. Greedy claim stops permanently after first disagreeing generated-token decision. Old eight-consumed-row diagnostic remains one clock shifted.'}

def analyze(raw,bank,driver,texts):
    raw=Path(raw);selected=driver['selected_index'];candidate_policies=(7,12) if selected in (5,4) else (7,12,11);definitions=[(t,p) for t in range(3) for p in candidate_policies]+[(t,0) for t in range(3)]
    if [(p['task'],p['policy']) for p in driver['paths']]!=definitions or driver['layer']!=17 or driver['capacity']!=512 or driver['memory_bytes']!=98464 or not driver['native_sham_logits_and_kv_byte_exact']:raise ValueError('Execution geometry/instrument differs')
    expected_aliases=[{'logical_name':f'case{t}-query-fit-memory','physical_name':f'case{t}-'+('values-native-memory' if selected==5 else 'response-reference-memory'),'reason':'identical selected query image and complete execution policy'} for t in range(3)] if selected in (5,4) else []
    if driver['aliases']!=expected_aliases:raise ValueError('Logical alias map differs')
    seal=json.loads((raw/'candidate-seal.json').read_text());expected={p.name for p in raw.iterdir() if p.suffix in ('.f32','.bin','.i32','.f64') and not any(p.name.startswith(f'case{i}-native-') for i in range(3))}
    if {r['name'] for r in seal['files']}!=expected or len(seal['files'])!=len(expected):raise ValueError('Candidate seal membership differs')
    for row in seal['files']:
        if sha(raw/row['name'])!=row['sha256'] or (raw/row['name']).stat().st_size!=row['bytes']:raise ValueError('Candidate seal differs')
    provenance=json.loads((raw/'program-provenance.json').read_text())
    if provenance['model_digest']!=driver['model_digest'] or provenance['model_identity']!=driver['model_identity']:raise ValueError('Program identity differs')
    for name,r in provenance['files'].items():
        if sha(raw/name)!=r['sha256']:raise ValueError('Program bytes differ')
    src=source_audit(raw,driver);inputs=json.loads((raw/'input-manifest.json').read_text())
    if sha(raw/'input-manifest.json')!=driver['target_input_manifest_sha256'] or inputs['selection_seal']['selection_sha256']!=driver['source_selection_sha256']:raise ValueError('Target-after-source binding differs')
    paths=[];records={};observations=list(src['observations']);snapshots=0;reads=0
    for p in driver['paths']:
        name,n=p['name'],p['count'];compact=p['policy']!=0;selected_macs=819200 if compact else 3145728
        if p['compact']!=compact or p['q_factor']!=compact or p['k_factor']!=compact or p['v_factor'] or p['query_fit']!=(p['policy'] in (11,12)) or p['linear_calls']!=(198 if compact else 197)*n or p['linear_macs']!=(1543569408-3145728+selected_macs)*n:raise ValueError('Target affine/policy ledger differs')
        if compact and (not p['selected_full_kv_untouched'] or p['compact_reads']!=n or p['serving_transfer_bytes']!=14336*n):raise ValueError('Target compact boundary differs')
        a=operands(raw,name,n);records[name]=(p,a);tops=np.fromfile(raw/f'{name}-tops.i32',dtype='<i4')
        if tops.shape!=(n,):raise ValueError('Pending token stream extent differs')
        history=[];boundaries={}
        for turn in p['turns']:
            ids=inputs['cases'][p['task']]['prompt_ids' if turn['turn']==0 else 'followup_ids'];skip=int(bool(history) and history[-1]==ids[0])
            if turn['start']!=len(history) or turn['template_skip']!=skip or turn['ingested']!=len(ids)-skip:raise ValueError('Target template differs')
            history+=ids[skip:]
            if turn['prefill']['position']!=len(history):raise ValueError('Target prefill clock differs')
            for token in turn['answer_ids']:
                if token!=int(tops[len(history)-1]):raise ValueError('Generated-token authority differs')
                history.append(token)
            if turn['final']['position']!=len(history) or turn['truncated']==turn['stopped']:raise ValueError('Target final clock differs')
            for stage in ('prefill','final'):
                t=turn[stage]['position'];z=load(raw,f'{name}-t{turn["turn"]}-{stage}-logits.f32',(V,))
                if int(np.argmax(z))!=turn[stage]['top'] or int(tops[t-1])!=turn[stage]['top']:raise ValueError('Target readout authority differs')
                boundaries[t]=(turn['turn'],stage)
        if history!=p['consumed_ids'] or len(history)!=n:raise ValueError('Target consumed history differs')
        errors=[];m=Memory()
        for t in range(n):
            m.append(a['k'][t],a['v'][t]);error=rel(a['attention'][t],full_read(a['q'][t],a['k'][:t+1],a['v'][:t+1]));errors.append(error)
            if compact:
                reads+=1
                if not np.array_equal(m.read(a['q'][t]),a['attention'][t]):raise ValueError('Target compact read replay differs')
                if t+1 in boundaries:
                    turn,stage=boundaries[t+1]
                    if (raw/f'{name}-t{turn}-{stage}-memory.bin').read_bytes()!=m.bytes():raise ValueError('Target snapshot replay differs')
                    snapshots+=1
            observations.append({'kind':'target-own-query-read','path':name,'step':t,'own_kv_attention_relative_error':error,'compact':compact})
        paths.append({'name':name,'task':p['task'],'policy':p['policy'],'advances':n,'answers':texts.get(name),'own_kv_attention_error':stats(errors),'selected_macs_per_advance':selected_macs})
    native_steps=src['source_native_advances'];source_calls=197*native_steps+198*6*native_steps;source_macs=1543569408*native_steps+(1543569408-3145728+819200)*6*native_steps
    if driver['source_linear_calls']!=source_calls or driver['source_linear_macs']!=source_macs or driver['all_linear_calls']!=197*32+source_calls+sum(p['linear_calls'] for p in driver['paths']) or driver['all_linear_macs']!=1543569408*32+source_macs+sum(p['linear_macs'] for p in driver['paths']):raise ValueError('Source/whole-invocation ledger differs')
    aliases={r['logical_name']:r['physical_name'] for r in driver['aliases']}
    def resolve(name):return aliases.get(name,name)
    comparisons=[];operands_rows=[];history_rows=[]
    for task in range(3):
        names={key:resolve(f'case{task}-{label}') for key,label in [('native','native'),('allclock','values-native-memory'),('response','response-reference-memory'),('selected','query-fit-memory')]};native,na=records[names['native']];initial=native['turns'][0]['prefill']['position']
        for key in ('allclock','response','selected'):
            desc,a=records[names[key]];common=0
            for ids in zip(native['consumed_ids'],desc['consumed_ids']):
                if ids[0]!=ids[1]:break
                common+=1
            if common<initial or not np.array_equal(a['norm'][:common],na['norm'][:common]):raise ValueError('Common history lower receiver differs')
            operands_rows.append({'task':task,'candidate':key,'physical_name':names[key],'common_prefix':common,'query_initial_relative_error':rel(a['q'][:initial],na['q'][:initial]),'attention_initial_relative_error':rel(a['attention'][:initial],na['attention'][:initial]),'output_initial_relative_error':rel(a['output'][:initial],na['output'][:initial])})
        history_rows.append({'task':task,'selected_physical_name':names['selected'],'selected_equals_allclock_ids':records[names['selected']][0]['consumed_ids']==records[names['allclock']][0]['consumed_ids'],'selected_equals_response_ids':records[names['selected']][0]['consumed_ids']==records[names['response']][0]['consumed_ids']})
        for turn in range(2):
            for stage in ('prefill','final'):
                nn=load(raw,f'{names["native"]}-t{turn}-{stage}-logits.f32',(V,));metrics={};histories={}
                for key in ('allclock','response','selected'):
                    p,_=records[names[key]];metrics[key]=distribution(nn,load(raw,f'{names[key]}-t{turn}-{stage}-logits.f32',(V,)));stop=p['turns'][turn][stage]['position'];histories[key]=p['consumed_ids'][:stop]
                stop=native['turns'][turn][stage]['position'];hn=native['consumed_ids'][:stop];row={'kind':'target-native-distribution','task':task,'turn':turn,'stage':stage,'split':inputs['cases'][task]['split'],'physical_names':names,'metrics':metrics,'native_history_matches':{k:h==hn for k,h in histories.items()},'selected_history_matches_allclock':histories['selected']==histories['allclock'],'selected_history_matches_response':histories['selected']==histories['response']};comparisons.append(row);observations.append(row)
    summary={'physical_paths':len(paths),'logical_target_policies':12,'target_own_advances':sum(p['count'] for p in driver['paths']),'source_forced_advances':7*native_steps,'sham_physical_advances':32,'all_physical_advances':7*native_steps+32+sum(p['count'] for p in driver['paths']),'target_compact_paths':sum(p['compact'] for p in driver['paths']),'target_compact_reads_exact':reads,'target_compact_snapshots_byte_exact':snapshots,'source_compact_reads_exact':6*native_steps,'source_full_clock_dual_calculation_rows':6*native_steps,'source_durable_full_vectors':src['distinct_durable_full_logit_vectors'],'selected_index':selected,'selected_dose':DOSES[selected],'memory_bytes':98464,'selected_macs_per_advance':819200,'dense_receiver_and_remaining_model_paid':True,'complete_model_emulation':False,'measured_speedup':None}
    return {'summary':summary,'source_selection_audit':src,'paths':paths,'comparisons':comparisons,'query_operand_comparisons':operands_rows,'history_comparisons':history_rows,'observations':observations,'terminal_status':'not-requested'}
