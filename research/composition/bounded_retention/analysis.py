"""Saved-only token authority, bounded support, own-row codec and consumer audit."""
from pathlib import Path
import hashlib,json,struct,math,re
import numpy as np
HERE=Path(__file__).resolve().parent
MASK=(1<<64)-1
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);return {'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),'max_abs':float(np.max(np.abs(a-b)))}
def codec(rows):
    x=np.asarray(rows,dtype=np.float32);maximum=np.max(np.abs(x),axis=-1);s=np.where(maximum>0,np.divide(maximum,np.float32(127)),np.float32(1)).astype(np.float32);r=np.divide(x,s[...,None]);absolute=np.abs(r);base=np.floor(absolute);q=np.copysign(base+(absolute-base>=np.float32(.5)),r);return np.clip(q,-127,127).astype(np.int8),s
class State:
    def __init__(self,a,r,m):
        self.a=a;self.r=r;self.m=m;self.c=a+r+m;self.p=0;self.n=0;self.rng=7;self.pos=np.full(self.c,-1,dtype='<i4');self.codes=np.zeros((3,2,2,self.c,128),dtype=np.int8);self.scales=np.zeros((3,2,2,self.c),dtype='<f4')
    def clone(self):
        s=State(self.a,self.r,self.m);s.p=self.p;s.n=self.n;s.rng=self.rng;s.pos=self.pos.copy();s.codes=self.codes.copy();s.scales=self.scales.copy();return s
    def draw(self,n):
        threshold=((-n)&MASK)%n
        while True:
            self.rng=(self.rng+0x9e3779b97f4a7c15)&MASK;z=self.rng;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK;z=((z^(z>>27))*0x94d049bb133111eb)&MASK;z=z^(z>>31)
            if z>=threshold:return z%n
    def append(self,row):
        slot=self.p if self.p<self.a else self.a+(self.p-self.a)%self.r
        if self.pos[slot]>=0:
            self.n+=1
            if self.m:
                j=self.n-1 if self.n<=self.m else self.draw(self.n)
                if j<self.m:
                    to=self.a+self.r+j;self.pos[to]=self.pos[slot];self.codes[:,:,:,to]=self.codes[:,:,:,slot];self.scales[:,:,:,to]=self.scales[:,:,:,slot]
        codes,scales=codec(row);self.codes[:,:,:,slot]=codes;self.scales[:,:,:,slot]=scales;self.pos[slot]=self.p;self.p+=1
    def context(self,q,layer):
        order=np.flatnonzero(self.pos>=0);order=order[np.argsort(self.pos[order])];k=(self.codes[layer,0]*self.scales[layer,0,:, :,None]).astype(np.float32)[:,order].astype(np.float64);v=(self.codes[layer,1]*self.scales[layer,1,:,:,None]).astype(np.float32)[:,order].astype(np.float64);logs=np.zeros(len(order),dtype=np.float64)
        m=min(self.n,self.m)
        if m:logs[order>=self.a+self.r]=math.log(self.n/m)
        result=[]
        for h in range(12):
            scores=[math.fsum(float(a)*float(b) for a,b in zip(row,q[h]))/math.sqrt(128)+float(w) for row,w in zip(k[h//6],logs)];maximum=max(scores);p=[math.exp(z-maximum) for z in scores];den=math.fsum(p);p=[x/den for x in p];result.append([math.fsum(w*float(row[d]) for w,row in zip(p,v[h//6])) for d in range(128)])
        return np.array(result)
def load_snapshot(p):
    data=Path(p).read_bytes();h=struct.unpack_from('<7I',data,8);n,rng=struct.unpack_from('<2Q',data,36);c=h[2];positions=np.frombuffer(data,offset=52,count=c,dtype='<i4').copy()
    if data[:8]!=b'TCMRING1' or h[:2]!=(1,28) or len(data)!=52+4*c+28*c*528:raise ValueError('Snapshot geometry differs')
    raw=data[52+4*c:];codes=np.empty((28,2,2,c,128),dtype=np.int8);scales=np.empty((28,2,2,c),dtype='<f4')
    for l in range(28):
        row=raw[l*c*528:(l+1)*c*528];codes[l]=np.frombuffer(row[:512*c],dtype=np.int8).reshape(2,2,c,128);scales[l]=np.frombuffer(row[512*c:],dtype='<f4').reshape(2,2,c)
    return h,n,rng,positions,codes,scales
def analyze(raw,bank,driver,texts):
    raw=Path(raw);origin=json.loads((HERE/'origin.json').read_text());panel=json.loads((HERE/'panel.json').read_text());inputs=json.loads((raw/'inputs.json').read_text());names=['native','full_byte','recent','anchors','reservoir'];watched=[0,17,27];observations=[];answers=[];allocation=[];local=[];comparisons=[];reference={};allsteps=0;totalrows=0;rootstates={}
    if len(inputs)!=len(panel) or any(any(q[k]!=planned[k] for k in planned) or q['ids']!=list(map(int,(raw/q['file']).read_text().split())) for q,planned in zip(inputs,panel)):raise ValueError('Declared inputs differ')
    if driver['schema']!='tcm-bounded-retention-driver-v1' or [a['name'] for a in driver['arms']]!=names or driver['logical_capacity']!=768 or driver['model_digest']!=origin['model']['content_sha256']:raise ValueError('Specimen/arm binding differs')
    def load(name,shape):
        x=np.fromfile(raw/name,dtype='<f4')
        if x.size!=math.prod(shape) or not np.isfinite(x).all():raise ValueError('Invalid captured array '+name)
        return x.reshape(shape)
    model_bytes=3087138816+579584+36700160+16404
    for ai,arm in enumerate(driver['arms']):
        name=arm['name'];a=16 if ai>=3 else 0;r=768 if ai<=1 else 64;m=32 if ai==4 else 0;c=a+r+m;s=State(a,r,m) if ai else None;turns=arm['turns'];nsteps=sum(len(t['steps'])+len(t['generation_steps']) for t in turns);allsteps+=nsteps
        rows=load(name+'-rows.bin',(nsteps,3,2,2,128)) if ai else None;row_index=0;hist=[];roots={};source_last=None;last=-1;source_state=None;source_hist=None;source_answer=None;root=None;root_hist=None;root_last=None
        if len(turns)!=8 or [t['input_index'] for t in turns]!=[0,3,1,2,4,5,6,7]:raise ValueError('Turn order differs')
        for ti,t in enumerate(turns):
            ix=t['input_index'];q=panel[ix]
            if ti==2:hist=list(source_hist);last=source_answer;s=source_state.clone() if ai else None
            if ti>=4:hist=list(root_hist);last=root_last;s=root.clone() if ai else None
            ids=list(map(int,(raw/f'input-{ix}.txt').read_text().split()));skip=int(last==ids[0]);pre=t['steps'];gen=t['generation_steps'];answer=t['answer_ids'];start=len(hist)
            if t['name']!=name+'-'+q['name'] or t['parent_position']!=start or t['template_skip']!=skip or len(pre)!=len(ids)-skip or not pre or len(gen)!=len(answer) or not answer or len(answer)>q['budget'] or not t['all_delivered_consumed'] or t['stopped']!=(answer[-1] in (151645,151643)) or t['truncated']==t['stopped'] or any(x in (151645,151643) for x in answer[:-1]) or t['truncated'] and len(answer)!=q['budget']:raise ValueError('Turn/generation closure differs')
            pending=None
            for j,step in enumerate(pre+gen):
                own=j>=len(pre);token=answer[j-len(pre)] if own else ids[j+skip]
                if step[:2]!=[len(hist),token] or own and token!=pending:raise ValueError('Not own-token continuation')
                hist.append(token);pending=step[2]
                if ai:s.append(rows[row_index]);row_index+=1
                if j in (len(pre)-1,len(pre)+len(gen)-1):
                    clock='prefill' if j==len(pre)-1 else 'final';y=load(t['name']+'-'+clock+'.f32',(151936,))
                    if int(np.argmax(y))!=pending or t[clock]['top']!=pending or sha(raw/(t['name']+'-'+clock+'.f32'))!=t[clock]['sha256']:raise ValueError('Readout/token binding differs')
                    key=(ix,clock)
                    if ai==0:reference[key]=(list(hist),y)
                    else:
                        refhist,refy=reference[key];same=hist==refhist;comparisons.append({'arm':name,'turn':q['name'],'clock':clock,'same_history':same,'full_logits':metric(y,refy) if same else None})
                        for wi,l in enumerate(watched):
                            query=load(f'{t["name"]}-{clock}-l{l}-query.f32',(12,128));context=load(f'{t["name"]}-{clock}-l{l}-attention.f32',(12,128));estimated=s.context(query.astype(np.float64),wi);local.append({'arm':name,'turn':q['name'],'clock':clock,'layer':l,'ideal_weighting_vs_executed_context':metric(context,estimated)})
            if t['final_position']!=len(hist):raise ValueError('Final clock differs')
            last=answer[-1]
            if 'expected' in q:
                normalized=re.sub(r'[^a-z0-9]+','',texts[t['name']].lower());row={'arm':name,'turn':q['name'],'text':texts[t['name']],'expected':q['expected'],'exact_expected_response':normalized==q['expected'],'stopped':t['stopped'],'truncated':t['truncated']};answers.append(row);observations.append({'kind':'delayed-recall-behavior','case':t['name'],'metrics':row})
            if ti==0:source_last=len(hist);source_hist=list(hist);source_answer=last;source_state=s.clone() if ai else None
            if ti==3:
                root=s.clone() if ai else None;root_hist=list(hist);root_last=last;correction_end=turns[2]['final_position'];recent_start=len(hist)-64
                if correction_end>=recent_start or len(hist)!=arm['root_position']:raise ValueError('Facts/correction not actually evicted from recent support')
                rootstates[name]={'root_position':len(hist),'source_end':source_last,'correction_start':turns[2]['parent_position'],'correction_end':correction_end,'recent_start':recent_start}
                if ai:
                    h,n,rng,pos,codes,scales=load_snapshot(raw/(name+'-root.bin'))
                    if h!=(1,28,c,a,r,m,s.p) or n!=s.n or rng!=s.rng or not np.array_equal(pos,s.pos) or not np.array_equal(codes[watched],s.codes) or not np.array_equal(scales[watched],s.scales) or not np.isfinite(scales).all():raise ValueError('Own-row/support reconstruction differs')
                    valid=pos[pos>=0];sampled=pos[a+r:];recent=pos[a:a+r];fact_support=[]
                    for input_index,offset,skipped in ((0,0,turns[0]['template_skip']),(1,turns[2]['parent_position'],turns[2]['template_skip'])):
                        for span in inputs[input_index]['fact_spans']:
                            absolute=[offset+j-skipped for j in span['token_indices'] if j>=skipped]
                            if ai>=2 and any(p<16 or p>=len(hist)-64 for p in absolute):raise ValueError('Fact clause not outside anchors/recent')
                            fact_support.append({'clause':span['clause'],'absolute_positions':absolute,'resident_positions':[p for p in absolute if p in valid],'sample_positions':[p for p in absolute if p in sampled]})
                    support={'arm':name,'positions':valid.tolist(),'sample_positions':sampled.tolist(),'population':s.n,'sample_count':int((sampled>=0).sum()),'fact_support':fact_support,'source_rows_in_recent':int(((recent>=16)&(recent<source_last)).sum()),'source_rows_in_sample':int(((sampled>=16)&(sampled<source_last)).sum()),'correction_rows_in_sample':int(((sampled>=turns[2]['parent_position'])&(sampled<correction_end)).sum())};observations.append({'kind':'bounded-support','case':name,'metrics':support});rootstates[name].update(support)
            observations.append({'kind':'own-turn','case':t['name'],'text':texts[t['name']],'stopped':t['stopped'],'truncated':t['truncated']})
        if ai and row_index!=nsteps:raise ValueError('Captured own-row count differs')
        totalrows+=nsteps*3*4 if ai else 0
        recovery=3*(28*c*528+151936*4) if ai else 0;session=712192+48*c;gpu=session+(4*28*c*528+1032*c+4+3*151936*4 if ai else 28*768*2048)
        if arm['state_gpu_bytes']!=gpu or arm['recovery_gpu_bytes']!=recovery or arm['tracked_model_state_gpu_bytes']!=model_bytes+gpu or arm['support_capacity']!=c or arm['cache_gpu_bytes']!=(28*c*528 if ai else 28*768*2048):raise ValueError('Allocation account differs')
        mech=arm['mechanics']
        if not mech['exact_replay'] or not mech['root_restored'] or mech['root_sha256']!=arm['root_sha256'] or ai and (not mech['partial_last_layer_abort'] or mech['nonmutating_refusals']!=4):raise ValueError('Lifecycle coverage differs')
        allocation.append({'arm':name,'support_capacity':c,'logical_capacity':768,'state_gpu_bytes':gpu,'recovery_gpu_bytes':recovery,'model_and_state_gpu_bytes':model_bytes+gpu,'cache_host_bytes':arm['cache_host_bytes'],'main_transitions':nsteps,'physical_cache_calls':arm['physical_cache_calls'],'occupied_sample_replacement_abort':mech['occupied_sample_replacement_abort'],'occupied_sample_abort_instrument_status':'covered' if mech['occupied_sample_replacement_abort'] else ('not-applicable' if ai!=4 else 'not-observed-in-bounded-search')});observations.append({'kind':'bounded-allocation','case':name,'metrics':allocation[-1]})
    if driver['matrix_count']!=197 or driver['matrix_bytes']!=3087138816 or driver['vector_bytes']!=579584 or driver['weight_tile_bytes']!=36700160:raise ValueError('Weight policy differs')
    summary={'arms':5,'main_transitions':allsteps,'own_encoded_rows_reconstructed':totalrows,'answers':answers,'allocation':allocation,'roots':rootstates,'scope':'bounded resident support and fixed recovery snapshots; full weights/forward retained; one development ledger, no general chat sufficiency or speed claim'}
    return {'summary':summary,'consumer_comparisons':comparisons,'local_attention_arithmetic':local,'observations':observations}
