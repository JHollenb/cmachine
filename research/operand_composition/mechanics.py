"""Strict mechanics for the declared, fixed 42-path composition instrument."""
import hashlib
import json
from pathlib import Path
import numpy as np

WIDTHS={'mlp-norm':1536,'mlp-write':1536,'mlp-output':1536,'receiver':1536,
        'q':1536,'k':256,'v':256,'attention':1536,'output':1536,
        'mlp-code':64,'query-code':128,'kv16':512,'kv18':512}


def require(condition, message):
    if not condition:
        raise ValueError('Composition instrument: '+message)


def read(raw, name, shape, dtype='<f4'):
    value=np.fromfile(Path(raw)/name,dtype=dtype)
    require(value.size==int(np.prod(shape)) and np.isfinite(value).all(),name+' shape/finite')
    return value.reshape(shape)


def verify_mechanics(raw, driver, inputs):
    raw=Path(raw); paths=driver['paths']; eos=(151645,151643)
    require(driver['schema']=='tcm-operand-composition-driver-v1','schema')
    require(driver['writer_layer']==16 and driver['reader_layer']==17 and driver['capacity']==512,'sites/capacity')
    require(driver['memory_policy']=='full-fp32-all-layers','memory profile')
    require(driver['record_bytes']==49920,'record layout')
    require(driver['native_sham_steps']==16 and driver['native_sham_logits_and_kv_byte_exact'],'native sham')
    require(driver['own_sealed_before_replay'] and driver['frozen_programs_byte_unchanged'],'execution seals')
    expected=[(t,p,'own') for t in range(6) for p in range(4)]+[(t,p,'replay') for t in range(6) for p in range(3)]
    require([(p['task'],p['policy'],p['mode']) for p in paths]==expected,'42-path coverage/order')
    require(len({p['name'] for p in paths})==42,'unique paths')
    cases=inputs['cases']; by_name={p['name']:p for p in paths}; details=[]
    for p in paths:
        name,n=p['name'],p['count']; own=p['mode']=='own'; w=bool(p['policy']&1); r=bool(p['policy']&2)
        require(p['writer']==w and p['reader']==r and p['own_generation']==own,name+' flags')
        require(0<n<=512 and len(p['consumed_ids'])==n,name+' extent')
        require(all(type(i) is int and 0<=i<151936 for i in p['consumed_ids']),name+' IDs')
        tops=read(raw,name+'-tops.i32',(n,),'<i4'); require(np.all((tops>=0)&(tops<151936)),name+' tops')
        for field,width in WIDTHS.items(): read(raw,name+'-'+field+'.f32',(n,width))
        history=[]; require(len(p['turns'])==2,name+' turns')
        for ti,t in enumerate(p['turns']):
            require(t['turn']==ti and t['start']==len(history),name+' turn start')
            ids=cases[p['task']]['prompt_ids' if ti==0 else 'followup_ids']
            skip=int(bool(history) and history[-1]==ids[0])
            require(t['template_skip']==skip and t['ingested']==len(ids)-skip,name+' template')
            history+=ids[skip:]
            require(t['prefill']['position']==len(history),name+' prefill clock')
            require(t['budget']==cases[p['task']]['budget'] and 0<len(t['answer_ids'])<=t['budget'],name+' answer budget')
            for j,token in enumerate(t['answer_ids']):
                if own: require(token==int(tops[len(history)-1]),name+' generated token authority')
                require(token not in eos or j==len(t['answer_ids'])-1,name+' early EOS')
                history.append(token)
            require(t['stopped']==(t['answer_ids'][-1] in eos) and t['truncated']==(not t['stopped']),name+' stop flags')
            require(t['stopped'] or len(t['answer_ids'])==t['budget'],name+' truncation budget')
            require(t['final']['position']==len(history),name+' final clock')
            for stage in ('prefill','final'):
                point=t[stage]['position']; logits=read(raw,f'{name}-t{ti}-{stage}-logits.f32',(151936,))
                require(int(np.argmax(logits))==t[stage]['top']==int(tops[point-1]),name+' readout authority')
                for operand in ('gate','up'): read(raw,f'{name}-t{ti}-{stage}-{operand}.f32',(8960,))
        require(history==p['consumed_ids'],name+' complete token history')
        if not own:
            parent=by_name[p['replay_of']]
            require(parent['policy']==3 and parent['task']==p['task'] and parent['mode']=='own',name+' replay parent')
            require(parent['consumed_ids']==p['consumed_ids'],name+' matched IDs')
            for a,b in zip(p['turns'],parent['turns']):
                for key in ('turn','start','ingested','template_skip','budget','answer_ids','stopped','truncated'):
                    require(a[key]==b[key],name+' replay turn '+key)
                for stage in ('prefill','final'): require(a[stage]['position']==b[stage]['position'],name+' replay clock')
        expected_counters={'linear_calls':(197+w+r)*n,
            'linear_macs':(1543569408-26279936*w-2326528*r)*n,
            'native_gate_up_calls':2*(not w)*n,'compiled_gate_up_calls':2*w*n,'mlp_coordinate_calls':w*n,
            'native_qk_calls':2*(not r)*n,'compiled_qk_calls':2*r*n,'query_coordinate_calls':r*n,
            'native_attention_calls':n,'reader_products':12*128*n*(n+1)}
        for key,value in expected_counters.items(): require(p[key]==value,name+' '+key)
        details.append({'name':name,'count':n,'own_generation':own,'token_authority':True,'counters':True})
    require(len({p['session_gpu_bytes'] for p in paths})==1,'fixed session allocations')
    require(driver['all_linear_calls']==32*197+sum(p['linear_calls'] for p in paths),'global linear calls including sham')
    require(driver['all_linear_macs']==32*1543569408+sum(p['linear_macs'] for p in paths),'global MACs including sham')
    seal=json.loads((raw/'own-seal.json').read_text()); members=seal['files']
    require(len({m['name'] for m in members})==len(members),'unique seal entries')
    expected_files={p.name for p in raw.iterdir() if p.suffix in ('.f32','.i32') and '-replay-' not in p.name}|{'own-driver.json'}
    require({m['name'] for m in members}==expected_files,'complete own array/metadata seal')
    for m in members:
        data=(raw/m['name']).read_bytes()
        require(len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256'],'own seal '+m['name'])
    own_driver=json.loads((raw/'own-driver.json').read_text())
    require(own_driver['paths']==paths[:24],'own metadata seal join')
    for key,value in own_driver.items():
        if key!='paths': require(driver[key]==value,'own root metadata '+key)
    controls=[]
    for task in range(6):
        names=[f'case{task}-replay-00',f'case{task}-replay-10',f'case{task}-replay-01',f'case{task}-own-11']
        n=by_name[names[0]]['count']
        for field in ('mlp-norm','kv16'):
            values=[read(raw,name+'-'+field+'.f32',(n,WIDTHS[field])) for name in names]
            require(all(v.tobytes()==values[0].tobytes() for v in values[1:]),f'case{task} lower shared '+field)
        for field in ('mlp-write','mlp-output','receiver','v'):
            values=[read(raw,name+'-'+field+'.f32',(n,WIDTHS[field])) for name in names]
            require(values[0].tobytes()==values[2].tobytes() and values[1].tobytes()==values[3].tobytes(),f'case{task} reader-independent '+field)
        controls.append({'task':task,'matched_rows':n,'lower_shared_byte_exact':True,'reader_independent_byte_exact':True})
    return {'schema':'operand-composition-mechanics-v1','paths':details,'seam_controls':controls,
            'own_sealed_files':len(members),'complete_transitions':sum(p['count'] for p in paths),
            'fixed_session_gpu_bytes':paths[0]['session_gpu_bytes'],'models_evaluated':0,
            'mechanics_status':'complete','terminal_status':'not-requested'}
