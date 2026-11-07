"""Validate lossless storage under matched tiled computation and own-state continuation."""
from pathlib import Path
import hashlib,json,struct,math
import numpy as np
from append_analysis import encode,metric,same
from span_parent_analysis import token_hash,validate_turn
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logical(rows):return hashlib.sha256(np.ascontiguousarray(rows.transpose(2,1,3,0,4)).tobytes()).hexdigest()
def analyze(raw,bank,driver,texts):
    raw=Path(raw);origin=json.loads((HERE/'origin.json').read_text());questions=json.loads((HERE/'questions.json').read_text())
    if driver['schema']!='tcm-weight-tiles-driver-v1' or len(driver['profiles'])!=3 or len(questions)!=5:raise ValueError('Profile membership differs')
    observations=[];memory=[];consumers=[];records=[]
    def load(name,shape):
        a=np.fromfile(raw/name,dtype='<f4')
        if a.size!=math.prod(shape) or not np.isfinite(a).all():raise ValueError('Invalid array '+name)
        return a.reshape(shape)
    for mode,entry in enumerate(driver['profiles']):
        d=entry['driver'];prefix=f'w{mode}-';chain=d['chains'][0];turns=chain['turns']+chain['later_turns'];n=chain['main_transitions'];matrix=1543569408*(2 if mode==2 else 4);vectors=579584;tile=36700160 if mode else 0;model=matrix+vectors+tile+16404
        if entry['profile']!=mode or d['weight_profile']!=mode or d['schema']!='tcm-compact-all-driver-v1' or not d['starts_empty'] or d['capacity']!=512 or d['model_digest']!=origin['model']['content_sha256'] or len(d['chains'])!=1 or chain['mode']!=2 or chain['policy']!='byte_all' or len(chain['turns'])!=2 or len(chain['later_turns'])!=3 or n<1 or n>512:raise ValueError('Execution profile differs')
        if d['matrix_count']!=197 or d['permanent_matrix_bytes']!=matrix or d['rank1_bytes']!=vectors or d['weight_tile_bytes']!=tile or d['model_weight_bytes']!=matrix+vectors or d['model_live_gpu_bytes']!=model:raise ValueError('Weight allocation differs')
        if chain['dense_layers'] or chain['session_gpu_bytes']!=736768 or chain['cache_arena_bytes']!=7569408 or chain['decode_workspace_bytes']!=524288 or chain['status_bytes']!=4 or chain['steady_state_gpu_bytes']!=8830468 or chain['tracked_live_gpu_bytes']!=model+8830468:raise ValueError('State allocation differs')
        if d['embedding_calls']!=n+7 or d['matvec_calls']!=197*(n+6)+192 or d['tile_calls']!=(877*(n+6)+706 if mode else 0):raise ValueError('Physical weight work differs')
        memory.append({'profile':mode,'permanent_matrix_bytes':matrix,'rank1_bytes':vectors,'weight_decode_tile_bytes':tile,'model_with_workspace_gpu_bytes':model,'cache_and_session_gpu_bytes':8830468,'model_and_state_gpu_bytes':model+8830468,'matvec_calls':d['matvec_calls'],'tile_calls':d['tile_calls'],'seconds':d['seconds']})
        rows=load(prefix+'byte_all-rows.f32',(n,28,2,2,128));queries=load(prefix+'byte_all-queries.f32',(n,4,12,128));codes,scales,decoded=encode(rows);hist=[];offset=0;steps=[];boundaries={}
        for ti,(t,q) in enumerate(zip(turns,questions)):
            template=list(map(int,(raw/q['file']).read_text().split()));skip=int(bool(hist) and hist[-1]==template[0]);pre=t['steps'];gen=t['generation_steps'];answer=t['answer_ids'];name=f'byte_all-t{ti}'
            if t['name']!=name or t['turn']!=ti or t['budget']!=q['budget'] or t['parent_position']!=len(hist) or t['parent_history_ids']!=hist or t['template_ids']!=template or t['template_skip']!=skip or len(pre)!=len(template)-skip or not pre:raise ValueError('Turn input differs')
            if not answer or len(gen)!=len(answer) or len(answer)>q['budget'] or not t['all_delivered_consumed'] or t['stopped']!=(answer[-1] in (151645,151643)) or t['truncated']==t['stopped'] or any(i in (151645,151643) for i in answer[:-1]) or t['truncated'] and len(answer)!=q['budget']:raise ValueError('Generation closure differs')
            pending=None
            for ix,step in enumerate(pre+gen):
                own=ix>=len(pre);token=answer[ix-len(pre)] if own else template[skip+ix]
                if own and token!=pending:raise ValueError('Not own greedy token')
                hist.append(token)
                if step['step']!=ix or step['position']!=offset or step['input_id']!=token or step['role']!=('own-generated' if own else 'template-user') or step['history_sha256']!=token_hash(hist) or step['encoded_cursor_count']!=28 or not step['all_encoded_cursors_at_position']:raise ValueError('Append clock differs')
                pending=step['top'];steps.append(step);offset+=1
            if t['final_position']!=offset:raise ValueError('Final cursor differs')
            for clock,count in (('prefill',offset-len(gen)),('final',offset)):
                y=load(prefix+name+'-'+clock+'.f32',(151936,));ref=pre[-1] if clock=='prefill' else gen[-1]
                if sha(raw/(prefix+name+'-'+clock+'.f32'))!=t[clock]['sha256'] or int(np.argmax(y))!=ref['top'] or t[clock]['top']!=ref['top'] or logical(decoded[:count])!=t[clock+'_logical_cache_sha256']:raise ValueError('Readout/cache differs')
                boundaries[ti,clock]={'history':hist[:count],'y':y}
                if mode:
                    ref=records[0]['boundaries'][ti,clock];c={'profile':mode,'turn':ti,'clock':clock,'same_history_and_clock':hist[:count]==ref['history'],'full_logits':metric(y,ref['y'])};consumers.append(c);observations.append({'kind':'weight-storage-consumer','case':prefix+name+'-'+clock,'metrics':c})
            data=(raw/(prefix+name+'-cache.bin')).read_bytes();header=struct.unpack('<6I',data[8:32]);count=offset
            if data[:8]!=b'TCMALL01' or header!=(1,28,2,128,count,8) or len(data)!=32+28*count*528:raise ValueError('Snapshot extent differs')
            p=32
            for l in range(28):
                bs=np.frombuffer(data[p:p+16*count],dtype='<f4').reshape(2,2,count);p+=16*count;bc=np.frombuffer(data[p:p+512*count],dtype=np.int8).reshape(2,2,count,128);p+=512*count
                if not same(bs,scales[:count,l].transpose(1,2,0)) or not same(bc,codes[:count,l].transpose(1,2,0,3)):raise ValueError('Own encoded snapshot differs')
            observations.append({'kind':'own-behavior','case':prefix+name,'text':texts[prefix+name],'answer_ids':answer,'stopped':t['stopped'],'truncated':t['truncated']})
        if offset!=n:raise ValueError('Trace cardinality differs')
        mech=chain['mechanics'];p=turns[1]['final_position']
        if mech['root_position']!=p or mech['root_logical_cache_sha256']!=logical(decoded[:p]) or mech['partial_cursors']!=[p+1]*27+[p] or not mech['partial_abort'] or not mech['root_restored'] or not mech['suffix_replaced'] or mech['two_token_logical_cache_sha256']!=mech['replayed_logical_cache_sha256'] or mech['full_logits_sha256']!=sha(raw/(prefix+'byte_all-rollback-reference.f32')) or chain['owned_step_calls_including_mechanics']!=28*(n+7) or chain['published_rows_including_mechanics']!=28*(n+7)-1 or chain['cursor_rewinds']!=6:raise ValueError('Rollback/work differs')
        if mode==0:
            if chain!=origin['historical_byte_chain']:raise ValueError('Historical byte-chain differs')
            for filename,record in origin['historical_byte_raw'].items():
                if sha(raw/(prefix+filename))!=record['sha256'] or (raw/(prefix+filename)).stat().st_size!=record['bytes']:raise ValueError('Historical byte raw differs')
        if mode==2:
            one=records[1]
            if {k:v for k,v in chain.items() if k!='tracked_live_gpu_bytes'}!={k:v for k,v in one['chain'].items() if k!='tracked_live_gpu_bytes'} or not same(rows,one['rows']) or not same(queries,one['queries']):raise ValueError('Lossless tiled storage trajectory differs')
            for path in raw.glob('w1-byte_all-*'):
                if path.read_bytes()!=(raw/('w2-'+path.name[3:])).read_bytes():raise ValueError('Matched tiled raw bytes differ')
        records.append({'chain':chain,'rows':rows,'queries':queries,'boundaries':boundaries})
    summary={'profiles':3,'turns_per_profile':5,'all28_caches_byte_from_token_zero':True,'source_bf16_matrix_storage_lossless_under_matched_tiling':True,'historical_all_byte_raw_matched':len(origin['historical_byte_raw']),'no_permanent_fp32_matrix_shadow_in_bf16_profile':True,'allocation':memory,'answers':texts,'scope':'lossless source storage and matched FP32 tile computation; full learned parameter information retained; full-retention linear byte caches; no forward elimination or general chat claim'}
    return {'summary':summary,'consumer_comparisons':consumers,'observations':observations}
