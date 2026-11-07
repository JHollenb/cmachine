"""Saved-only all-layer construction audit. Quality measurements never gate discovery."""
from pathlib import Path
import json,math,struct,hashlib
import numpy as np
from append_analysis import encode,response,metric,same,stats
from span_parent_analysis import token_hash,validate_turn
HERE=Path(__file__).resolve().parent
POLICIES=('native','float_all','byte_all')
WATCHED=(0,17,18,27)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def logical(rows):return hashlib.sha256(np.ascontiguousarray(rows.transpose(2,1,3,0,4)).tobytes()).hexdigest()
def analyze(raw,bank,driver,texts):
    raw=Path(raw);origin=json.loads((HERE/'origin.json').read_text());baseline=json.loads((HERE/'baseline.json').read_text());questions=json.loads((HERE/'questions.json').read_text())
    if driver['schema']!='tcm-compact-all-driver-v1' or not driver['starts_empty'] or driver['capacity']!=512 or driver['watched_layers']!=list(WATCHED) or len(questions)!=5 or [c['policy'] for c in driver['chains']]!=list(POLICIES):raise ValueError('Membership/construction differs')
    def load(name,shape):
        a=np.fromfile(raw/name,dtype='<f4')
        if a.size!=math.prod(shape) or not np.isfinite(a).all():raise ValueError('Array extent/finite differs: '+name)
        return a.reshape(shape)
    observations=[];memory=[];consumers=[];query_rows=[];response_rows=[];codec_rows=[];by_policy={};total=0
    for mode,chain in enumerate(driver['chains']):
        policy=chain['policy'];turns=chain['turns']+chain['later_turns'];n=chain['main_transitions'];byte=mode==2
        if len(chain['turns'])!=2 or len(chain['later_turns'])!=3 or len(turns)!=5 or not 0<n<=512:raise ValueError('Turn cardinality differs')
        rows=load(policy+'-rows.f32',(n,28,2,2,128));queries=load(policy+'-queries.f32',(n,4,12,128));codes,scales,decoded=encode(rows);effective=decoded if byte else rows
        arena=28*(270336 if byte else 1048576) if mode else 0;workspace=524288 if byte else 0;status=4 if byte else 0;session=736768 if mode else 30096896
        if chain['mode']!=mode or chain['dense_layers']!=(0 if mode else 28) or chain['session_gpu_bytes']!=session or chain['cache_arena_bytes']!=arena or chain['decode_workspace_bytes']!=workspace or chain['status_bytes']!=status or chain['steady_state_gpu_bytes']!=session+arena+workspace+status or chain['tracked_live_gpu_bytes']!=driver['model_live_gpu_bytes']+chain['steady_state_gpu_bytes']:raise ValueError('Allocation differs')
        memory.append({'policy':policy,'dense_layers':chain['dense_layers'],'state_gpu_bytes':chain['steady_state_gpu_bytes'],'saving_vs_native_bytes':30096896-chain['steady_state_gpu_bytes'],'model_plus_state_gpu_bytes':chain['tracked_live_gpu_bytes']})
        hist=[];steps=[];boundaries={};offset=0;rs={l:[] for l in WATCHED}
        for ti,(turn,question) in enumerate(zip(turns,questions)):
            name=f'{policy}-t{ti}';template=list(map(int,(raw/question['file']).read_text().split()));skip=int(bool(hist) and hist[-1]==template[0]);pre=turn['steps'];gen=turn['generation_steps'];answer=turn['answer_ids'];start=list(hist)
            if turn['turn']!=ti or turn['name']!=name or turn['budget']!=question['budget'] or turn['parent_position']!=len(hist) or turn['parent_history_ids']!=hist or turn['template_ids']!=template or turn['template_skip']!=skip or len(pre)!=len(template)-skip:raise ValueError('Turn inputs differ')
            if not answer or len(gen)!=len(answer) or len(answer)>question['budget'] or not turn['all_delivered_consumed'] or turn['stopped']!=(answer[-1] in (151645,151643)) or turn['truncated']==turn['stopped'] or any(i in (151645,151643) for i in answer[:-1]) or turn['truncated'] and len(answer)!=question['budget']:raise ValueError('Own generation closure differs')
            pending=None
            for ix,step in enumerate(pre+gen):
                own=ix>=len(pre);token=answer[ix-len(pre)] if own else template[skip+ix]
                if own and token!=pending:raise ValueError('Not own greedy token')
                hist.append(token)
                if step['step']!=ix or step['position']!=offset or step['input_id']!=token or step['role']!=('own-generated' if own else 'template-user') or step['history_sha256']!=token_hash(hist) or step['encoded_cursor_count']!=(28 if mode else 0) or not step['all_encoded_cursors_at_position']:raise ValueError('Token/cursor clock differs')
                pending=step['top'];steps.append(step)
                if byte:
                    for qi,l in enumerate(WATCHED):
                        original=response(queries[offset,qi],rows[:offset+1,l,0].transpose(1,0,2),rows[:offset+1,l,1].transpose(1,0,2));represented=response(queries[offset,qi],effective[:offset+1,l,0].transpose(1,0,2),effective[:offset+1,l,1].transpose(1,0,2));ratio=np.exp(represented[:,0]-original[:,0]);mu=original[:,1:];den=np.linalg.norm(mu,axis=1);rr={'position':offset,'layer':l,'Z_ratios':ratio.tolist(),'numerator_relative_errors':(np.linalg.norm(ratio[:,None]*represented[:,1:]-mu,axis=1)/den).tolist(),'mean_relative_errors':(np.linalg.norm(represented[:,1:]-mu,axis=1)/den).tolist()};rs[l].append(rr);response_rows.append(rr)
                if step['capture']:
                    for qi,l in enumerate(WATCHED):
                        ports={p:load(f'{name}-{step["capture"]}-l{l}-{p}.f32',(1536,)) for p in ('input','attention','attention_write','raw_receiver','mlp_return','output')}
                        if not same(np.add(ports['input'],ports['attention_write']),ports['raw_receiver']) or not same(np.add(ports['raw_receiver'],ports['mlp_return']),ports['output']):raise ValueError('Residual identities differ')
                offset+=1
            if turn['final_position']!=len(hist):raise ValueError('Final clock differs')
            for clock,count in (('prefill',offset-len(gen)),('final',offset)):
                y=load(f'{name}-{clock}.f32',(151936,));ref=pre[-1] if clock=='prefill' else gen[-1];state_hash=logical(effective[:count]);history=hist[:count]
                if sha(raw/f'{name}-{clock}.f32')!=turn[clock]['sha256'] or int(np.argmax(y))!=ref['top'] or turn[clock]['top']!=ref['top'] or state_hash!=turn[clock+'_logical_cache_sha256']:raise ValueError('Readout/logical cache differs')
                boundaries[ti,clock]={'history':history,'y':y,'cache':state_hash}
                if mode:
                    base=by_policy['native']['boundaries'][ti,clock];c={'policy':policy,'turn':ti,'clock':clock,'same_history_and_clock':history==base['history'],'full_logits':metric(y,base['y'])};consumers.append(c);observations.append({'kind':'all-layer-consumer','case':name+'-'+clock,'metrics':c})
                    if mode==1 and (not c['same_history_and_clock'] or not same(y,base['y']) or state_hash!=base['cache']):raise ValueError('Exact callback readout parity differs')
            if byte:
                data=(raw/f'{name}-cache.bin').read_bytes();header=struct.unpack('<6I',data[8:32]);count=offset
                if data[:8]!=b'TCMALL01' or header!=(1,28,2,128,count,8) or len(data)!=32+28*count*528:raise ValueError('Snapshot schema differs')
                p=32
                for l in range(28):
                    bs=np.frombuffer(data[p:p+16*count],dtype='<f4').reshape(2,2,count);p+=16*count;bc=np.frombuffer(data[p:p+512*count],dtype=np.int8).reshape(2,2,count,128);p+=512*count
                    if not same(bs,scales[:count,l].transpose(1,2,0)) or not same(bc,codes[:count,l].transpose(1,2,0,3)):raise ValueError('Immediate GPU encoding/valid prefix differs')
            if mode==0:
                if ti==0 and (answer!=baseline['historical_source_own_ids'] or turn['final_logical_cache_sha256']!=baseline['historical_source_kv_sha256']):raise ValueError('Historical source differs')
                old=origin['correction'] if ti==1 else origin['historical_native_turns'][ti-2] if ti>=2 else None
                if old and (answer!=old['answer_ids'] or any(turn[c]!=old[c] or turn[c+'_logical_cache_sha256']!=old.get(c+'_logical_cache_sha256',old.get(c+'_kv_sha256')) for c in ('prefill','final'))):raise ValueError('Historical native boundary differs')
            observations.append({'kind':'own-behavior','case':name,'text':texts[name],'answer_ids':answer,'stopped':turn['stopped'],'truncated':turn['truncated']})
        if offset!=n:raise ValueError('Trace cardinality differs')
        mech=chain['mechanics'];root_position=turns[1]['final_position'];partial=[root_position+1]*27+[root_position] if byte else [0]*28
        if mech['root_position']!=root_position or mech['root_logical_cache_sha256']!=logical(effective[:root_position]) or not mech['suffix_replaced'] or not mech['root_restored'] or mech['partial_abort']!=byte or mech['partial_cursors']!=partial or mech['two_token_logical_cache_sha256']!=mech['replayed_logical_cache_sha256'] or mech['full_logits_sha256']!=sha(raw/f'{policy}-rollback-reference.f32'):raise ValueError('Rollback/partial abort differs')
        calls=28*(n+(7 if byte else 5)) if mode else 0
        if chain['owned_step_calls_including_mechanics']!=calls or chain['published_rows_including_mechanics']!=calls-int(byte) or chain['cursor_rewinds']!=(6 if byte else 3 if mode else 0):raise ValueError('Physical calls differ')
        if mode==1:
            base=by_policy['native']
            if steps!=base['steps'] and any({k:v for k,v in a.items() if k!='encoded_cursor_count'}!={k:v for k,v in b.items() if k!='encoded_cursor_count'} for a,b in zip(steps,base['steps'])):raise ValueError('Exact transition parity differs')
            if n!=len(base['rows']) or not same(rows,base['rows']) or not same(queries,base['queries']):raise ValueError('Exact own-row/query parity differs')
            for ti in range(5):
                for clock in ('prefill','final'):
                    for l in WATCHED:
                        for port in ('input','attention','attention_write','raw_receiver','mlp_return','output'):
                            if (raw/f'{policy}-t{ti}-{clock}-l{l}-{port}.f32').read_bytes()!=(raw/f'native-t{ti}-{clock}-l{l}-{port}.f32').read_bytes():raise ValueError('Exact sparse port parity differs')
        if byte:
            base=by_policy['native'];common=min(n,len(base['rows']))
            for i in range(common):
                domain=steps[i]['history_sha256']==base['steps'][i]['history_sha256'] and steps[i]['position']==base['steps'][i]['position']
                for qi,l in enumerate(WATCHED):query_rows.append({'position':i,'layer':l,'same_history_and_clock':domain,'query':metric(queries[i,qi],base['queries'][i,qi])})
            for l in range(28):codec_rows.append({'layer':l,'keys':metric(decoded[:,l,0],rows[:,l,0]),'values':metric(decoded[:,l,1],rows[:,l,1])})
            for l,rr in rs.items():observations.append({'kind':'all-layer-own-response','case':f'byte_all-l{l}','metrics':{'correlated_head_reads':12*len(rr),'Z_ratio':stats([r['Z_ratios'] for r in rr]),'numerator_relative_error':stats([r['numerator_relative_errors'] for r in rr]),'mean_relative_error':stats([r['mean_relative_errors'] for r in rr])}})
        by_policy[policy]={'rows':rows,'queries':queries,'steps':steps,'boundaries':boundaries};total+=n
    for l in WATCHED:
        rr=[r['query']['relative_l2'] for r in query_rows if r['layer']==l and r['same_history_and_clock']];observations.append({'kind':'all-layer-query-composition','case':f'byte_all-l{l}','metrics':stats(rr) if rr else None})
    summary={'sessions':3,'turns_per_session':5,'all_start_at_token_zero':True,'native_hydration':False,'main_token_transitions':total,'byte_layer_appends':28*driver['chains'][2]['main_transitions'],'byte_dense_cache_layers':0,'exact_all_callback_parity':True,'historical_source_and_eight_native_boundaries':True,'rollback_and_last_layer_abort_closed':True,'allocation':memory,'answers':texts,'scope':'all-layer byte KV from token zero; full weights and full retention; one supplied-fact development conversation; no bounded-memory general chat or speed claim'}
    return {'summary':summary,'consumer_comparisons':consumers,'own_response_rows':response_rows,'query_comparisons':query_rows,'codec_rows':codec_rows,'observations':observations}
