"""Saved-only memory, source instrument, selector and target-consumer audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from reader import Memory,full_read,rel,stats
from metrics import distribution,choose,DOSES
V=151936
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(raw,name,shape):
    a=np.fromfile(raw/name,dtype='<f4')
    if a.size!=int(np.prod(shape)) or not np.isfinite(a).all():raise ValueError('Invalid recorded operand '+name)
    return a.reshape(shape)
def operands(raw,name,n):return {key:load(raw,f'{name}-{key}.f32',shape) for key,shape in {'norm':(n,1536),'q':(n,12,128),'k':(n,2,128),'v':(n,2,128),'attention':(n,12,128),'output':(n,1536),'coeff':(n,128)}.items()}
def source_audit(raw,driver):
    selection=json.loads((raw/'source-selection.json').read_text());seal=json.loads((raw/'selection-seal.json').read_text());family=json.loads((raw/'family-provenance.json').read_text());source=json.loads((raw/'source-provenance.json').read_text())
    if sha(raw/'source-selection.json')!=driver['source_selection_sha256'] or sha(raw/'source-selection.json')!=seal['selection_sha256'] or seal['phase']!='before-parsing-fresh-prompts-or-tokenizing-targets' or selection['source_advances']!=1351 or selection['source_lengths']!=[66,63,64] or selection['prompt_lengths']!=[58,55,56]:raise ValueError('Source selection identity/geometry differs')
    for name,r in family['files'].items():
        if sha(raw/name)!=r['sha256'] or (raw/name).stat().st_size!=r['bytes']:raise ValueError('Frozen image family differs')
    selected=selection['selected_index']
    if selected!=driver['selected_index'] or sha(raw/'selected-query.f32')!=family['files'][f'query-proposal-{selected}.f32']['sha256'] or sha(raw/'selected-query.f32')!=seal['image_sha256'] or selection['selected_image_sha256']!=seal['image_sha256']:raise ValueError('Selected image differs')
    expected_native=load(raw,'source-inputs.f32',(193,1536));offset=0;native=[];counts=selection['source_lengths'];checks=[];curves=[];snapshots=[];means=[];observation=[]
    for c,n in enumerate(counts):
        original=source['source_trajectories'][c];ids=np.loadtxt(raw/f'source{c}.txt',dtype=int,ndmin=1).tolist()
        if ids!=original['prompt_ids']+original['own_ids'] or n!=original['advances']:raise ValueError('Source forced IDs differ')
        a=operands(raw,f'source{c}-native',n)
        if not np.array_equal(a['norm'],expected_native[offset:offset+n]):raise ValueError('Reacquired source receiver differs from original capture')
        native.append(a);offset+=n
    for i in range(6):
        primary=json.loads((raw/f'source-metrics-{i}.json').read_text());audit=json.loads((raw/f'source-audit-{i}.json').read_text())
        if sha(raw/f'source-audit-{i}.json')!=seal['source_audit_sha256'][i] or primary['proposal']!=i or audit['proposal']!=i or len(primary['rows'])!=193 or len(audit['rows'])!=193:raise ValueError('Source live audit identity differs')
        per=[];offset=0;regions={};replay=[];response=[];query=[]
        for c,n in enumerate(counts):
            a=operands(raw,f'source{c}-proposal{i}',n);m=Memory();errors=[]
            if not np.array_equal(a['norm'],native[c]['norm']) or not np.array_equal(a['v'],native[c]['v']):raise ValueError('Source fixed-ID receiver/native-value contract differs')
            for t in range(n):
                row=primary['rows'][offset+t];other=audit['rows'][offset+t];region='historical-own-suffix' if t>=selection['prompt_lengths'][c] else 'shared-header' if t<24 else 'prompt-body'
                if row['source']!=c or row['step']!=t or row['region']!=region or (other['source'],other['step'],other['region'])!=(c,t,region):raise ValueError('Source clock/region differs')
                for key in ('kl_nats','relative_logit_l2','native_top_logp_delta_nats','native_entropy_nats'):
                    if abs(row[key]-other[key])>1e-10*max(1.,abs(row[key]),abs(other[key])):raise ValueError('Stored full-clock dual calculation disagrees')
                errors.append(row['kl_nats']);regions.setdefault(region,[]).append(row['kl_nats']);m.append(a['k'][t],a['v'][t]);replay.append(rel(m.read(a['q'][t]),a['attention'][t]))
                if not np.array_equal(m.read(a['q'][t]),a['attention'][t]):raise ValueError('Source compact read replay differs')
                if t in (15,selection['prompt_lengths'][c]-1,n-1):
                    nn=f'source{c}-native-t{t}-logits.f32';cn=f'source{c}-proposal{i}-t{t}-logits.f32';metric=distribution(load(raw,nn,(V,)),load(raw,cn,(V,)))
                    if sha(raw/nn)!=row['native_logits_sha256'] or sha(raw/cn)!=row['candidate_logits_sha256']:raise ValueError('Durable source vector differs from live stream')
                    for key,value in metric.items():
                        if abs(value-row[key])>1e-10*max(1.,abs(value),abs(row[key])):raise ValueError('Durable source metric replay differs')
                    snapshots.append({'proposal':i,'source':c,'step':t,'kl_nats':metric['kl_nats'],'native_logits_sha256':sha(raw/nn),'candidate_logits_sha256':sha(raw/cn)})
                observation.append({'kind':'source-native-distribution','proposal':i,'source':c,'step':t,'region':region,'kl_nats':row['kl_nats'],'relative_logit_l2':row['relative_logit_l2'],'compact_reader_replay_relative':replay[-1]})
            per.append(float(np.mean(errors)));response.append(rel(a['attention'],native[c]['attention']));query.append(rel(a['q'],native[c]['q']));offset+=n
        mean=float(np.mean(per));means.append(mean)
        if abs(mean-primary['equal_trajectory_mean_kl_nats'])>1e-12 or abs(mean-selection['mean_kl_nats'][i])>1e-12:raise ValueError('Source selector aggregation differs')
        curves.append({'proposal':i,'dose':DOSES[i],'equal_trajectory_mean_kl_nats':mean,'per_trajectory_mean_kl_nats':per,'pooled_region_mean_kl_nats':{k:float(np.mean(v)) for k,v in regions.items()},'source_attention_relative_error':response,'source_query_relative_error':query,'source_memory_replay_max_relative':max(replay),'live_maximum_c_python_discrepancy':audit['maximum_c_python_discrepancy']})
    if choose(means)!=selected:raise ValueError('Saved source selection differs')
    return {'selected_index':selected,'source_advances':1351,'native_source_receiver_byte_exact_to_original':True,'full_clock_dual_calculation_rows':1158,'durable_metric_replay_rows':len(snapshots),'distinct_durable_full_logit_vectors':63,'source_curves':curves,'source_snapshot_metrics':snapshots,'scope':'Forced original source IDs; C and Python agreed on all full-vocabulary source clocks while streams existed; durable independent logit replay is limited to nine clocks per policy.','observations':observation}

def analyze(raw,bank,driver,texts):
    raw=Path(raw);selected=driver['selected_index'];candidate_policies=(7,12) if selected in (0,4) else (7,12,11);definitions=[(t,p) for t in range(6) for p in candidate_policies]+[(t,0) for t in range(6)]
    if [(p['task'],p['policy']) for p in driver['paths']]!=definitions or driver['layer']!=17 or driver['capacity']!=512 or driver['memory_bytes']!=98464 or not driver['native_sham_logits_and_kv_byte_exact']:raise ValueError('Execution geometry/instrument differs')
    expected_aliases=[{'logical_name':f'case{t}-query-fit-memory','physical_name':f'case{t}-'+('values-native-memory' if selected==0 else 'response-reference-memory'),'reason':'identical selected query image and complete execution policy'} for t in range(6)] if selected in (0,4) else []
    if driver['aliases']!=expected_aliases:raise ValueError('Logical alias map differs')
    seal=json.loads((raw/'candidate-seal.json').read_text());expected={p.name for p in raw.iterdir() if p.suffix in ('.f32','.bin','.i32','.f64') and not any(p.name.startswith(f'case{i}-native-') for i in range(6))}
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
    source_calls=197*193+198*1158;source_macs=1543569408*193+(1543569408-3145728+819200)*1158
    if driver['source_linear_calls']!=source_calls or driver['source_linear_macs']!=source_macs or driver['all_linear_calls']!=197*32+source_calls+sum(p['linear_calls'] for p in driver['paths']) or driver['all_linear_macs']!=1543569408*32+source_macs+sum(p['linear_macs'] for p in driver['paths']):raise ValueError('Source/whole-invocation ledger differs')
    aliases={r['logical_name']:r['physical_name'] for r in driver['aliases']}
    def resolve(name):return aliases.get(name,name)
    comparisons=[];operands_rows=[];history_rows=[]
    for task in range(6):
        names={key:resolve(f'case{task}-{label}') for key,label in [('native','native'),('ls','values-native-memory'),('response','response-reference-memory'),('selected','query-fit-memory')]};native,na=records[names['native']];initial=native['turns'][0]['prefill']['position']
        for key in ('ls','response','selected'):
            desc,a=records[names[key]];common=0
            for ids in zip(native['consumed_ids'],desc['consumed_ids']):
                if ids[0]!=ids[1]:break
                common+=1
            if common<initial or not np.array_equal(a['norm'][:common],na['norm'][:common]):raise ValueError('Common history lower receiver differs')
            operands_rows.append({'task':task,'candidate':key,'physical_name':names[key],'common_prefix':common,'query_initial_relative_error':rel(a['q'][:initial],na['q'][:initial]),'attention_initial_relative_error':rel(a['attention'][:initial],na['attention'][:initial]),'output_initial_relative_error':rel(a['output'][:initial],na['output'][:initial])})
        history_rows.append({'task':task,'selected_physical_name':names['selected'],'selected_equals_ls_ids':records[names['selected']][0]['consumed_ids']==records[names['ls']][0]['consumed_ids'],'selected_equals_response_ids':records[names['selected']][0]['consumed_ids']==records[names['response']][0]['consumed_ids']})
        for turn in range(2):
            for stage in ('prefill','final'):
                nn=load(raw,f'{names["native"]}-t{turn}-{stage}-logits.f32',(V,));metrics={};histories={}
                for key in ('ls','response','selected'):
                    p,_=records[names[key]];metrics[key]=distribution(nn,load(raw,f'{names[key]}-t{turn}-{stage}-logits.f32',(V,)));stop=p['turns'][turn][stage]['position'];histories[key]=p['consumed_ids'][:stop]
                stop=native['turns'][turn][stage]['position'];hn=native['consumed_ids'][:stop];row={'kind':'target-native-distribution','task':task,'turn':turn,'stage':stage,'split':inputs['cases'][task]['split'],'physical_names':names,'metrics':metrics,'native_history_matches':{k:h==hn for k,h in histories.items()},'selected_history_matches_ls':histories['selected']==histories['ls'],'selected_history_matches_response':histories['selected']==histories['response']};comparisons.append(row);observations.append(row)
    summary={'physical_paths':len(paths),'logical_target_policies':24,'target_own_advances':sum(p['count'] for p in driver['paths']),'source_forced_advances':1351,'sham_physical_advances':32,'all_physical_advances':1351+32+sum(p['count'] for p in driver['paths']),'target_compact_paths':sum(p['compact'] for p in driver['paths']),'target_compact_reads_exact':reads,'target_compact_snapshots_byte_exact':snapshots,'source_compact_reads_exact':1158,'source_full_clock_dual_calculation_rows':1158,'source_durable_full_vectors':63,'selected_index':selected,'selected_dose':DOSES[selected],'memory_bytes':98464,'selected_macs_per_advance':819200,'dense_receiver_and_remaining_model_paid':True,'complete_model_emulation':False,'measured_speedup':None}
    return {'summary':summary,'source_selection_audit':src,'paths':paths,'comparisons':comparisons,'query_operand_comparisons':operands_rows,'history_comparisons':history_rows,'observations':observations,'terminal_status':'not-requested'}
