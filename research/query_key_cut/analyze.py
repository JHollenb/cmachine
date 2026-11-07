"""Saved-only reader/circuit factorial, historical contrast and custody audit."""
import argparse,base64,gzip,hashlib,json,re
from pathlib import Path
import numpy as np
D=1536;K=256;V=151936
FIELDS=('input','attention','attention_write','receiver','mlp_input','mlp_write','output')
RFIELDS=(('norm',D),('qc',D),('qn',D),('used',D),('kc',K),('kn',K),('v',K),('computed',D),('consumed',D))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vector(raw,name,n):
    a=np.fromfile(raw/name,dtype='<f4')
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Finite shape differs: '+name)
    return a
def reader(raw,name):
    a=vector(raw,name+'-reader.f32',sum(n for _,n in RFIELDS));out={};offset=0
    for key,n in RFIELDS:out[key]=a[offset:offset+n];offset+=n
    return out
def distribution(n,c):
    n=n.astype(float);c=c.astype(float);n-=n.max();c-=c.max();ln=n-np.log(np.exp(n).sum());lc=c-np.log(np.exp(c).sum());return float(np.exp(ln)@(ln-lc))
def norm(a):return float(np.linalg.norm(np.asarray(a,dtype=float)))
def location(text,object0):
    m=re.search(r'\b'+object0+r'\b.{0,100}?\b(silver|amber|violet)\b',text,re.I|re.S);return m.group(1).lower() if m else None
def analyze(raw,report):
    raw=Path(raw);rows=report['driver']['rows'];actual=[r for r in rows if 'alias_of' not in r];observations=[];reader_rows=[];groups=[];shams=0;addition_remainder=0.;reader_error=[]
    def row(context,history,arm):
        result=next(r for r in rows if (r['context'],r['history'],r['arm'])==(context,history,arm))
        return row(context,history,result['alias_of']) if 'alias_of' in result else result
    for context in (128,1024):
        case=next(x for x in report['inputs'] if x['context_tokens']==context);silver=case['labels']['correct']['ids'][0];amber=case['labels']['incorrect']['ids'][0]
        for history in (0,1):
            kc=vector(raw,f'case{context}-h{history}-keys-compiled.f32',(case['cut_position']+1)*K).reshape(2,-1,128).astype(float);kn=vector(raw,f'case{context}-h{history}-keys-native.f32',kc.size).reshape(kc.shape).astype(float);values=vector(raw,f'case{context}-h{history}-values.f32',kc.size).reshape(kc.shape).astype(float)
            baseline=row(context,history,0);sham=row(context,history,1)
            if baseline['answers']!=sham['answers'] or baseline['consumed_ids']!=sham['consumed_ids'] or baseline['final_cache_sha256']!=sham['final_cache_sha256']:raise ValueError('Sham future differs')
            for suffix in ('reader.f32','layers.f32','probabilities.f32','cut-logits.f32','answer-last-logits.f32','followup-logits.f32','followup-last-logits.f32'):
                if (raw/(baseline['name']+'-'+suffix)).read_bytes()!=(raw/(sham['name']+'-'+suffix)).read_bytes():raise ValueError('Sham arrays differ '+suffix)
            shams+=1;local=[]
            reference=vector(raw,row(context,history,5)['name']+'-cut-logits.f32',V)
            for arm in range(9):
                r=row(context,history,arm);name=r['name'];a=reader(raw,name);layers=vector(raw,name+'-layers.f32',10*7*D).reshape(10,7,D);logits=vector(raw,name+'-cut-logits.f32',V);text=report['texts'][name]
                if not np.array_equal(a['consumed'],layers[0,1]):raise ValueError('Installed read differs from consumer')
                if arm==7:
                    if np.any(a['consumed']):raise ValueError('Zero read is nonzero')
                elif not np.array_equal(a['computed'],a['consumed']):raise ValueError('Unchanged attention write differs')
                for layer in layers:
                    addition_remainder=max(addition_remainder,norm(layer[3].astype(float)-np.add(layer[0],layer[2],dtype=np.float32).astype(float)),norm(layer[6].astype(float)-np.add(layer[3],layer[5],dtype=np.float32).astype(float)))
                qt=history;kt=history
                if arm in (2,3,4,5):qt=int(arm in (3,5));kt=int(arm in (4,5))
                if arm in (6,8):qt=kt=1
                expected_query=a['qn'] if qt else a['qc']
                if arm==6:expected_query=vector(raw,f'case{context}-h{history}-native-queries.f32',(case['cut_position']+1)*D).reshape(-1,D)[case['cut_position']-16]
                if not np.array_equal(a['used'],expected_query) or not np.array_equal(a['kc'],kc[:,-1].astype('<f4').reshape(-1)) or not np.array_equal(a['kn'],kn[:,-1].astype('<f4').reshape(-1)) or not np.array_equal(a['v'],values[:,-1].astype('<f4').reshape(-1)):raise ValueError('Declared query/key/value source differs')
                key=kn if kt else kc;val=values[:,::-1] if arm==8 else values;queries=a['used'].reshape(12,128).astype(float);expected=[];prob=[]
                for h in range(12):
                    scores=key[h//6]@queries[h]/np.sqrt(128.);p=np.exp(scores-scores.max());p/=p.sum();expected.append(p@val[h//6]);prob.append(p)
                expected=np.asarray(expected).reshape(-1);probs=vector(raw,name+'-probabilities.f32',12*key.shape[1]).reshape(12,-1).astype(float);relative=norm(a['computed'].astype(float)-expected)/max(norm(expected),1e-30);reader_error.append(relative)
                result={'context':context,'history':history,'arm':arm,'actual_name':name,'alias':r['arm']!=arm,'silver_minus_amber_logit':float(logits[silver]-logits[amber]),'kl_to_same_parent_native_qk':distribution(reference,logits),'first_whistle_location':location(text[0],'whistle'),'first_compass_location':location(text[0],'compass'),'followup_whistle_location':location(text[1],'whistle'),'stopped':r['stopped'],'first_answer':text[0],'followup_answer':text[1]};local.append(result);observations.append({'schema':'qk18-cut-observation-v1','model_id':report['model']['identity'],**result})
                reader_rows.append({'context':context,'history':history,'arm':arm,'relative_cpu64_attention_remainder':relative,'maximum_cpu64_probability_remainder':float(np.max(np.abs(probs-np.asarray(prob))))})
            def values_for(arm,suffix,n):return vector(raw,row(context,history,arm)['name']+'-'+suffix,n).astype(float)
            interaction=values_for(5,'cut-logits.f32',V)-values_for(3,'cut-logits.f32',V)-values_for(4,'cut-logits.f32',V)+values_for(2,'cut-logits.f32',V)
            layer_interaction=values_for(5,'layers.f32',10*7*D)-values_for(3,'layers.f32',10*7*D)-values_for(4,'layers.f32',10*7*D)+values_for(2,'layers.f32',10*7*D);layer_interaction=layer_interaction.reshape(10,7,D)
            groups.append({'context':context,'history':history,'parent_hash':baseline['parent_hash'],'rows':local,'logit_interaction_l2':norm(interaction),'margin_interaction':float(interaction[silver]-interaction[amber]),'interaction_by_layer':[{'layer':18+i,**{field:norm(layer_interaction[i,j]) for j,field in enumerate(FIELDS)}} for i in range(10)]})
        for suffix,n in [('keys-compiled.f32',(case['cut_position']+1)*K),('keys-native.f32',(case['cut_position']+1)*K),('values.f32',(case['cut_position']+1)*K),('norms.f32',(case['cut_position']+1)*D),('native-queries.f32',(case['cut_position']+1)*D)]:
            if not np.array_equal(vector(raw,f'case{context}-h0-'+suffix,n),vector(raw,f'case{context}-h1-'+suffix,n)):raise ValueError('Fixed lower receiver/key/value operand differs across histories '+suffix)
    history_rows=[]
    for context in (128,1024):
        bad=row(context,0,5);good=row(context,1,5);b=vector(raw,bad['name']+'-layers.f32',10*7*D).reshape(10,7,D);g=vector(raw,good['name']+'-layers.f32',10*7*D).reshape(10,7,D);changes=[{'layer':18+i,**{field:norm(b[i,j].astype(float)-g[i,j].astype(float)) for j,field in enumerate(FIELDS)}} for i in range(10)]
        history_rows.append({'context':context,'native_current_layer18_byte_exact':bool(np.array_equal(b[0],g[0])),'native_current_layer19_input_byte_exact':bool(np.array_equal(b[1,0],g[1,0])),'changes':changes,'first_nonzero_field':next(({'layer':c['layer'],'field':field,'l2':c[field]} for c in changes for field in FIELDS if c[field]>0),None)})
    for r in actual:
        ids=np.asarray(r['consumed_ids'],dtype='<i4')
        if hashlib.sha256(ids.tobytes()).hexdigest()!=r['consumed_ids_sha256'] or not r['parent_replayed_exact']:raise ValueError('Trace or restore identity differs')
    return {'schema':'qk18-cut-analysis-v1','groups':groups,'historical_native_current_contrasts':history_rows,'reader_arithmetic':reader_rows,'max_relative_cpu64_reader_remainder':max(reader_error),'maximum_fp32_residual_add_remainder':addition_remainder,'exact_sham_pairs':shams,'physical_branches':len(actual),'logical_branches':len(rows),'observations':observations,'scope':'Known-failure diagnosis, single current read followed by original policy. Paid same-receiver native operands and full memory; no fitted producer, new generalization or speed claim. First-token margin and completed location are separate observations.'}
def verify(root):
    root=Path(root);r=read(root/'report.json');job=read(root/'job.json');submission=read(root/'submission.json');raw=root/'raw';payload=root/'payload'
    if job['job_id']!=submission['job_id'] or sha(root/'logs.txt')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Scheduler log differs')
    reports=[json.loads(gzip.decompress(base64.b64decode(line.split('=',1)[1]))) for line in (root/'logs.txt').read_text().splitlines() if line.startswith('COMPOSITION_RESULT_GZIP_B64=')]
    if reports!=[r] or read(raw/'report.json')!=r:raise ValueError('Report identity differs')
    for p,h in submission['config']['sources'].items():
        if sha(payload/p)!=h:raise ValueError('Source hash differs '+p)
    for p,record in r['raw_files'].items():
        if sha(raw/p)!=record['sha256'] or (raw/p).stat().st_size!=record['bytes']:raise ValueError('Raw custody differs '+p)
    if read(raw/'driver.json')!=r['driver']:raise ValueError('Driver differs')
    wrapper=r['saturn_receipt'];receipt=dict(wrapper['receipt']);fingerprint=receipt.pop('fingerprint')
    if hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=fingerprint or receipt['parent']!=wrapper['before'] or receipt['result']!=wrapper['after']:raise ValueError('Invocation receipt differs')
    a=analyze(raw,r)
    def compare(x,y):
        if isinstance(x,dict):
            if x.keys()!=y.keys():raise ValueError('Analysis schema differs')
            for k in x:compare(x[k],y[k])
        elif isinstance(x,list):
            if len(x)!=len(y):raise ValueError('Analysis count differs')
            for xx,yy in zip(x,y):compare(xx,yy)
        elif isinstance(x,float):
            if not np.isclose(x,y,rtol=1e-9,atol=1e-12):raise ValueError('Analysis arithmetic differs')
        elif x!=y:raise ValueError('Analysis observation differs')
    compare(a,r['analysis']);(root/'analysis.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n');(root/'observations.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in a['observations']))
    result={'schema':'qk18-cut-verification-v1','report_sha256':sha(root/'report.json'),'source_files':len(submission['config']['sources']),'raw_files':len(r['raw_files']),'exact_sham_pairs':a['exact_sham_pairs'],'parent_replayed_branches':a['physical_branches'],'logical_rows':a['logical_branches'],'native_current_layer18_closed_contexts':sum(x['native_current_layer18_byte_exact'] for x in a['historical_native_current_contrasts']),'maximum_reader_arithmetic_remainder':a['max_relative_cpu64_reader_remainder'],'analysis_recomputed':True,'new_model_calls':0,'terminal_status':'not-requested'};(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');verify(p.parse_args().root)
