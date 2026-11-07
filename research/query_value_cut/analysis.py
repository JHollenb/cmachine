"""Reconstruct candidate memory and own-query reads from sealed operands."""
import hashlib
import json
from pathlib import Path
import numpy as np

DIM=128
class Memory:
    def __init__(self):
        self.ks=np.zeros((2,16,128));self.vs=np.zeros_like(self.ks);self.count=np.zeros((2,16),'<u4');self.clusters=np.zeros(2,'<u4')
        self.keys=np.zeros((2,16,128),'<f4');self.values=np.zeros_like(self.keys);self.next=self.recent=self.head=0
    def append(self,k,v):
        if self.recent==16:
            for g in range(2):
                key,value=self.keys[g,self.head],self.values[g,self.head]
                if self.clusters[g]<16:c=int(self.clusters[g]);self.clusters[g]+=1
                else:
                    distances=[]
                    for slot in range(16):
                        delta=key.astype(float)-self.ks[g,slot]/int(self.count[g,slot])
                        distances.append(sum(float(x)*float(x) for x in delta))
                    c=min(range(16),key=lambda i:distances[i])
                self.count[g,c]+=1;self.ks[g,c]+=key;self.vs[g,c]+=value
        else:self.recent+=1
        self.keys[:,self.head]=k;self.values[:,self.head]=v;self.head=(self.head+1)%16;self.next+=1
    def read(self,q):
        result=[]
        for h,query in enumerate(q):
            g=h//6;c=int(self.clusters[g]);slots=[(self.head+16-self.recent+i)%16 for i in range(self.recent)]
            k=np.concatenate((self.ks[g,:c]/self.count[g,:c,None],self.keys[g,slots].astype(float)))
            v=np.concatenate((self.vs[g,:c]/self.count[g,:c,None],self.values[g,slots].astype(float)))
            mass=np.concatenate((self.count[g,:c].astype(float),np.ones(self.recent)))
            score=k@query.astype(float)/np.sqrt(128.)+np.log(mass);weight=np.exp(score-score.max());result.append(weight@v/weight.sum())
        return np.asarray(result,dtype='<f4')
    def bytes(self):
        return b''.join(x.tobytes() for x in (self.ks.astype('<f8'),self.vs.astype('<f8'),self.count,self.clusters,self.keys,self.values,np.asarray([self.next,self.recent,self.head],'<u8')))

def full_read(q,k,v):
    k=k.transpose(1,0,2).astype(float);v=v.transpose(1,0,2).astype(float);result=[]
    for h,query in enumerate(q):
        score=k[h//6]@query.astype(float)/np.sqrt(128.);weight=np.exp(score-score.max());result.append(weight@v[h//6]/weight.sum())
    return np.asarray(result)

def rel(a,b):return float(np.linalg.norm(np.asarray(a,dtype=float)-b)/max(float(np.linalg.norm(b)),1e-30))
def stats(values):
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),'max':float(np.max(values)),'min':float(np.min(values))}

def vector_factorial(nn,cn,nc,cc):
    nn,cn,nc,cc=[np.asarray(x,dtype=float).reshape(-1) for x in (nn,cn,nc,cc)]
    qk,value,interaction=cn-nn,nc-nn,cc-cn-nc+nn;joint=cc-nn;den=max(float(np.linalg.norm(nn)),1e-30)
    return {'qk_only_relative':float(np.linalg.norm(qk)/den),'value_only_relative':float(np.linalg.norm(value)/den),'both_relative':float(np.linalg.norm(joint)/den),'interaction_relative':float(np.linalg.norm(interaction)/den),'qk_value_cross_scaled':float(2*qk@value/den**2),'interaction_qk_cross_scaled':float(2*interaction@qk/den**2),'interaction_value_cross_scaled':float(2*interaction@value/den**2),'vector_closure_relative':float(np.linalg.norm(joint-qk-value-interaction)/den)}

def probability_row(values,t):
    return values[6*t*(t+1):6*(t+1)*(t+2)].reshape(12,t+1).astype(float)

def common_prefix_factorials(driver,records):
    rows=[];tasks=[]
    for task in range(3):
        labels=('native','values-native','address-native','compiled');parts=[records[f'case{task}-{label}'] for label in labels];descs=[x[0] for x in parts];arrays=[x[1] for x in parts]
        common=0
        for ids in zip(*(p['consumed_ids'] for p in descs)):
            if len(set(ids))!=1:break
            common+=1
        initial=descs[0]['turns'][0]['prefill']['position']
        if common<initial:raise ValueError('Common history lost inside initial prompt')
        if any(a['norm'][:common].tobytes()!=arrays[0]['norm'][:common].tobytes() for a in arrays[1:]):raise ValueError('Common lower receiver profile differs')
        formula=[];remainders=[];probability_equal=True;operand_pairs_equal=True;mix_errors=[];normalization=[]
        for t in range(common):
            probs=[probability_row(a['probabilities'],t) for a in arrays]
            p_equal=np.array_equal(probs[0],probs[2]) and np.array_equal(probs[1],probs[3]);probability_equal &= p_equal
            operands_equal=np.array_equal(arrays[0]['q'][t],arrays[2]['q'][t]) and np.array_equal(arrays[0]['k'][t],arrays[2]['k'][t]) and np.array_equal(arrays[1]['q'][t],arrays[3]['q'][t]) and np.array_equal(arrays[1]['k'][t],arrays[3]['k'][t]) and np.array_equal(arrays[0]['v'][t],arrays[1]['v'][t]) and np.array_equal(arrays[2]['v'][t],arrays[3]['v'][t]);operand_pairs_equal &= operands_equal
            mixed=[]
            for p,a in zip(probs,arrays):
                if np.any(p.sum(axis=1)<=0):raise ValueError('Empty probability row')
                normalization.append(float(np.max(np.abs(p.sum(axis=1)-1))))
                mixed.append(np.asarray([p[h]@a['v'][:t+1,h//6].astype(float) for h in range(12)]))
            predicted=np.asarray([(probs[3][h]-probs[0][h])@(arrays[3]['v'][:t+1,h//6].astype(float)-arrays[0]['v'][:t+1,h//6].astype(float)) for h in range(12)])
            native=[a['attention'][t].astype(float) for a in arrays];actual=native[3]-native[1]-native[2]+native[0];rest=[a-b for a,b in zip(native,mixed)]
            finite_remainder=rest[3]-rest[1]-rest[2]+rest[0];formula.append(predicted);remainders.append(actual-predicted)
            row={'kind':'common-input-qk-value','task':task,'step':t,'token_id':descs[0]['consumed_ids'][t],'formula_instrument_status':'applicable' if p_equal and operands_equal else 'operand-pair-mismatch',**vector_factorial(*native),'interaction_formula_relative_discrepancy':rel(actual,predicted),'interaction_native_mix_remainder_relative_to_native':float(np.linalg.norm(finite_remainder)/max(float(np.linalg.norm(native[0])),1e-30)),'factorization_account_remainder_relative_to_native':float(np.linalg.norm(actual-predicted-finite_remainder)/max(float(np.linalg.norm(native[0])),1e-30))}
            rows.append(row);mix_errors.extend(rel(a,b) for a,b in zip(native,mixed))
        regions={}
        for region,n in (('initial_prompt',initial),('maximal_common_prefix',common)):
            outputs=[a['attention'][:n] for a in arrays];actual=outputs[3].astype(float)-outputs[1]-outputs[2]+outputs[0]
            regions[region]={'positions':n,'attention':vector_factorial(*outputs),'layer_exit':vector_factorial(*(a['output'][:n] for a in arrays)),'interaction_formula_error_relative_to_native':float(np.linalg.norm(actual-np.asarray(formula[:n]))/max(float(np.linalg.norm(outputs[0].astype(float))),1e-30))}
        tasks.append({'task':task,'maximal_common_consumed_prefix':common,'initial_prompt_positions':initial,'normalized_receivers_byte_exact':True,'probability_pairs_byte_exact':bool(probability_equal),'operand_pairs_byte_exact':bool(operand_pairs_equal),'formula_instrument_status':'applicable' if probability_equal and operand_pairs_equal else 'operand-pair-mismatch','native_mix_relative_roundoff':stats(mix_errors),'probability_normalization_max_error':max(normalization),'regions':regions})
    return {'scope':'same exact consumed-ID prefix and selected receiver; full suffix policies retain their own historical upper cache','tasks':tasks,'rows':rows}

def analyze(raw,bank,driver,texts):
    raw=Path(raw);definitions=[(task,p) for task in range(3) for p in range(1,7)]+[(task,0) for task in range(3)]
    if [(p['task'],p['policy']) for p in driver['paths']]!=definitions or driver['layer']!=17 or driver['capacity']!=512 or driver['memory_bytes']!=98464 or not driver['native_sham_logits_and_kv_byte_exact']:
        raise ValueError('Execution geometry/instrument differs')
    seal=json.loads((raw/'candidate-seal.json').read_text());names=[x['name'] for x in seal['files']]
    expected={p.name for p in raw.iterdir() if p.suffix in ('.f32','.bin','.i32') and not any(p.name.startswith(f'case{i}-native-') for i in range(3))}
    if set(names)!=expected or len(names)!=len(expected):raise ValueError('Candidate seal membership differs')
    for row in seal['files']:
        data=(raw/row['name']).read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Candidate seal differs')
    def load(name,shape):
        a=np.fromfile(raw/name,dtype='<f4')
        if a.size!=int(np.prod(shape)) or not np.isfinite(a).all():raise ValueError('Invalid recorded operand '+name)
        return a.reshape(shape)
    if [p['rank'] for p in driver['programs']]!=[128] or driver['sources']:raise ValueError('Reused program geometry differs')
    bases={128:load('program128-basis.f32',(128,1536))}
    provenance=json.loads((raw/'program-provenance.json').read_text())
    if provenance['model_digest']!=driver['model_digest'] or provenance['model_identity']!=driver['model_identity'] or provenance['source_report_sha256']!='f033d61f0e02a047af0991bfa89845600eaafd3afd96d34fe41218caaffd933e':raise ValueError('Source program identity differs')
    for name,description in provenance['files'].items():
        data=(raw/name).read_bytes()
        if len(data)!=description['bytes'] or hashlib.sha256(data).hexdigest()!=description['sha256']:raise ValueError('Source program bytes differ')
    if driver['programs'][0]['compile_linear_calls'] or driver['programs'][0]['compile_linear_macs']:raise ValueError('Unexpected new program compilation')
    basis_gram_error={str(rank):float(np.max(np.abs(b.astype(float)@b.astype(float).T-np.eye(rank)))) for rank,b in bases.items()}
    inputs=json.loads((raw/'input-manifest.json').read_text());observations=[];path_rows=[];all_records={}
    for desc in driver['paths']:
        name,n=desc['name'],desc['count'];qk_factor=desc['policy'] in (1,3,5);v_factor=desc['policy'] in (1,3,6);factor=qk_factor or v_factor;compact=desc['policy'] in (2,3);zero=desc['policy']==4;rank=128 if factor else 0
        if (desc['factor'],desc['compact'],desc['zero'])!=(factor,compact,zero):raise ValueError('Policy flags differ')
        if desc['rank']!=rank or desc['qk_factor']!=qk_factor or desc['v_factor']!=v_factor:raise ValueError('Dynamic program policy differs')
        selected_macs=(196608 if factor else 0)+(229376 if qk_factor else 2752512)+(32768 if v_factor else 393216)
        expected_calls=(198 if factor else 197)*n;expected_macs=(1543569408-3145728+selected_macs)*n
        compiled_maps=2*int(qk_factor)+int(v_factor)
        if desc['linear_calls']!=expected_calls or desc['linear_macs']!=expected_macs or desc['native_qkv_calls']!=(3-compiled_maps)*n or desc['compiled_qkv_calls']!=compiled_maps*n or desc['coordinate_calls']!=int(factor)*n:raise ValueError('Affine ledger differs')
        if compact and (not desc['selected_full_kv_untouched'] or desc['compact_reads']!=n or desc['serving_transfer_bytes']!=14336*n):raise ValueError('Bounded-memory execution differs')
        shapes={'norm':(n,1536),'q':(n,12,128),'k':(n,2,128),'v':(n,2,128),'attention':(n,12,128),'output':(n,1536),'coeff':(n,128)}
        arrays={k:load(f'{name}-{k}.f32',shape) for k,shape in shapes.items()};all_records[name]=(desc,arrays)
        if not compact and not zero:
            probabilities=load(f'{name}-probabilities.f32',(6*n*(n+1),))
            if np.any(probabilities<0):raise ValueError('Negative native probability')
            arrays['probabilities']=probabilities
        elif (raw/f'{name}-probabilities.f32').exists():raise ValueError('Non-native reader captured stale probabilities')
        tops=np.fromfile(raw/f'{name}-tops.i32',dtype='<i4')
        if tops.shape!=(n,) or np.any(tops<0) or np.any(tops>=151936):raise ValueError('Invalid pending-token stream')
        history=[];case=inputs['cases'][desc['task']]
        for turn in desc['turns']:
            ids=case['prompt_ids'] if turn['turn']==0 else case['followup_ids'];skip=int(bool(history) and history[-1]==ids[0])
            if turn['start']!=len(history) or turn['template_skip']!=skip or turn['ingested']!=len(ids)-skip:raise ValueError('Template authority differs')
            history+=ids[skip:]
            if turn['prefill']['position']!=len(history):raise ValueError('Prefill clock differs')
            for token in turn['answer_ids']:
                if token!=int(tops[len(history)-1]):raise ValueError('Generated-token authority differs')
                history.append(token)
            if turn['final']['position']!=len(history) or turn['truncated']==turn['stopped']:raise ValueError('Final clock/status differs')
            for stage in ('prefill','final'):
                point=turn[stage]['position'];logits=load(f'{name}-t{turn["turn"]}-{stage}-logits.f32',(151936,))
                if int(np.argmax(logits))!=turn[stage]['top'] or int(tops[point-1])!=turn[stage]['top']:raise ValueError('Readout authority differs')
        if history!=desc['consumed_ids']:raise ValueError('Consumed token history differs')
        boundaries={}
        for turn in desc['turns']:
            for stage in ('prefill','final'):
                point=turn[stage]['position'];boundaries[point]=(turn['turn'],stage)
        memory=Memory();errors=[];replay=[];snapshot_count=0;energy=[]
        for t in range(n):
            memory.append(arrays['k'][t],arrays['v'][t]);reference=full_read(arrays['q'][t],arrays['k'][:t+1],arrays['v'][:t+1]);error=rel(arrays['attention'][t].astype(float),reference);errors.append(error)
            replay_error=rel(memory.read(arrays['q'][t]).astype(float),arrays['attention'][t].astype(float)) if compact else None
            if compact:replay.append(replay_error)
            projection=arrays['coeff'][t,:rank].astype(float)@bases[rank].astype(float) if factor else arrays['norm'][t].astype(float)
            omitted=rel(projection,arrays['norm'][t].astype(float));energy.append(omitted)
            observations.append({'kind':'own-query-attention','path':name,'step':t,'token_id':desc['consumed_ids'][t],'own_kv_attention_relative_error':error,'compact_law_replay_relative_error':replay_error,'receiver_projection_relative_error':omitted,'history_rows':t+1,'represented_entries_per_head':min(t+1,32) if compact else t+1})
            if compact and t+1 in boundaries:
                turn,stage=boundaries[t+1];packet=(raw/f'{name}-t{turn}-{stage}-memory.bin').read_bytes()
                if packet!=memory.bytes():raise ValueError('Compact state replay differs: '+name+str(t))
                for g in range(2):
                    if int(memory.count[g].sum())+memory.recent!=t+1:raise ValueError('Memory mass differs')
                snapshot_count+=1
        if len(desc['consumed_ids'])!=n or len(desc['turns'])!=2:raise ValueError('History extent differs')
        row={'name':name,'task':desc['task'],'policy':desc['policy'],'rank':rank,'advances':n,'own_kv_attention_error':stats(errors),'receiver_projection_error':stats(energy),'memory_replay_error':stats(replay) if replay else None,'memory_snapshots_byte_exact':snapshot_count,'answers':texts.get(name),'truncated':[turn['truncated'] for turn in desc['turns']],'nominal_linear_macs':desc['linear_macs'],'reader_products':desc['reader_products']}
        path_rows.append(row)
    comparisons=[];factorials=[]
    for task in range(3):
        native=next(p for p in driver['paths'] if p['task']==task and p['policy']==0)
        for desc in (p for p in driver['paths'] if p['task']==task and p['policy']!=0):
            for turn in range(2):
                for stage in ('prefill','final'):
                    a=load(f'{desc["name"]}-t{turn}-{stage}-logits.f32',(151936,)).astype(float);b=load(f'{native["name"]}-t{turn}-{stage}-logits.f32',(151936,)).astype(float)
                    stopa,stopb=desc['turns'][turn][stage]['position'],native['turns'][turn][stage]['position']
                    same=desc['consumed_ids'][:stopa]==native['consumed_ids'][:stopb]
                    row={'kind':'trajectory-readout','task':task,'candidate':desc['name'],'turn':turn,'stage':stage,'same_consumed_ids':same,'relative_logit_difference':rel(a,b),'native_top':native['turns'][turn][stage]['top'],'candidate_top':desc['turns'][turn][stage]['top']};comparisons.append(row);observations.append(row)
        # Full-cache QK x V policy outcomes; same token history does not equal the same upper cache.
        for turn in range(2):
            for stage in ('prefill','final'):
                labels=('native','values-native','address-native','compiled');paths=[next(p for p in driver['paths'] if p['name']==f'case{task}-{label}') for label in labels]
                y=[load(f'{p["name"]}-t{turn}-{stage}-logits.f32',(151936,)).astype(float) for p in paths]
                a,b=y[1]-y[0],y[2]-y[0];interaction=y[3]-y[1]-y[2]+y[0];joint=y[3]-y[0];den=max(float(np.linalg.norm(y[0])),1e-30)
                histories=[p['consumed_ids'][:p['turns'][turn][stage]['position']] for p in paths]
                row={'kind':'qk-value-policy-factorial','task':task,'turn':turn,'stage':stage,'all_same_consumed_ids':all(h==histories[0] for h in histories),'qk_only_relative':float(np.linalg.norm(a)/den),'value_only_relative':float(np.linalg.norm(b)/den),'both_relative':float(np.linalg.norm(joint)/den),'interaction_relative':float(np.linalg.norm(interaction)/den),'qk_value_cross':float(2*a@b),'interaction_qk_cross':float(2*interaction@a),'interaction_value_cross':float(2*interaction@b)};factorials.append(row);observations.append(row)
    expected_calls=197*32+sum(p['linear_calls'] for p in driver['paths'])
    expected_macs=1543569408*32+sum(p['linear_macs'] for p in driver['paths'])
    if driver['all_linear_calls']!=expected_calls or driver['all_linear_macs']!=expected_macs:raise ValueError('Whole-invocation affine ledger differs')
    prefix_factorials=common_prefix_factorials(driver,all_records)
    observations.extend(prefix_factorials['rows'])
    summary={'paths':len(path_rows),'own_advances':sum(p['advances'] for p in path_rows),'compact_paths':6,'source_rows':193,'basis_orthogonality_max_abs':basis_gram_error,'compact_memory_bytes':98464,'native_selected_cache_capacity_bytes':1048576,'candidate_sealed_files':len(expected),'compact_snapshots_byte_exact':sum(p['memory_snapshots_byte_exact'] for p in path_rows),'compact_law_max_relative_replay_error':max(p['memory_replay_error']['max'] for p in path_rows if p['memory_replay_error']),'full_global_cache_still_allocated':True,'other_27_layers_and_full_weights_retained':True,'measured_speedup':None,'complete_model_emulation':False}
    summary['source_acquisition_advances_this_job']=0
    return {'summary':summary,'paths':path_rows,'comparisons':comparisons,'factorials':factorials,'common_prefix_factorials':prefix_factorials,'observations':observations,'terminal_status':'not-requested'}
