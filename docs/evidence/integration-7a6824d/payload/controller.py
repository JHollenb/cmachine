"""Real stdin/stdout protocol with explicitly recorded delivered/consumed ownership."""
from pathlib import Path
import json,subprocess
class Protocol:
    def __init__(self,args,stderr):
        self.stderr=Path(stderr).open('w');self.process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1);self.rows=[];self.ready=self.receive();self.rows.append({'command':None,'reply':self.ready})
    def receive(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Persistent C process exited: '+str(self.process.poll()))
        return json.loads(line)
    def send(self,line,allow_error=False):
        self.process.stdin.write(line+'\n');self.process.stdin.flush();reply=self.receive();self.rows.append({'command':line,'reply':reply})
        if 'error' in reply and not allow_error:raise RuntimeError(reply['error'])
        return reply
    def close(self):
        if self.process.poll() is None:self.process.stdin.write('Q\n');self.process.stdin.flush()
        code=self.process.wait(timeout=15);self.stderr.close()
        if code:raise RuntimeError('Persistent process exit '+str(code))
def protocol_case(args,raw,policy,origin,case_name=None):
    raw=Path(raw);label=case_name or policy;p=Protocol(args,raw/(label+'-protocol.stderr.txt'));turns=[];refusals=[];generation_checks=[]
    def semantic(value):
        if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k!='top_frame'}
        if isinstance(value,list):return [semantic(v) for v in value]
        return value
    def check_refusal(command):
        before=p.send('S');reply=p.send(command,True);after=p.send('S')
        if 'error' not in reply or before!=after:raise ValueError('Refusal mutated persistent state: '+command[:80])
        refusals.append({'command':command[:80],'state_unchanged':True})
    def turn(i,last):
        q=origin['inputs'][i];ids=q['ids'];skip=int(last==ids[0]);p.send('A '+' '.join(map(str,ids[skip:])));before=p.send('S')
        if i==0:
            frame=p.send('P')['frame'];one=p.send('G 1');two=p.send('G 1');split=p.send('S');again=p.send('G 1');unchanged=p.send('S')
            if not two['ids'] or two['ids'][-1] not in (151645,151643) or again['ids'] or split!=unchanged:raise ValueError('Delivered EOS is not stable under repeated generation')
            p.send('B '+str(frame));frame=p.send('P')['frame'];joined=p.send('G 2');combined=p.send('S')
            if joined['ids']!=one['ids']+two['ids'] or semantic(split)!=semantic(combined):raise ValueError('Split and joined generation differ')
            p.send('B '+str(frame));generation_checks.append({'split_and_joined_ids':joined['ids'],'state_equal_except_marker_id':True,'repeated_eos_empty_and_nonmutating':True})
        reply=p.send('G '+str(q['budget']));generated=reply['ids']
        if not generated:raise ValueError('First generation unexpectedly empty')
        delivered=p.send('S')
        if not delivered['delivered'] or delivered['pending_token']!=generated[-1]:raise ValueError('Delivered token lost ownership')
        # The reference assay consumes every delivered token, including final EOS.
        p.send('A '+str(generated[-1]));after=p.send('S');t={'input_index':i,'template_skip':skip,'prefill':before,'generated':reply,'delivered_state':delivered,'final':after};turns.append(t);return generated[-1]
    try:
        ready=p.ready
        if ready['cache_policy']!=policy or ready['model_digest']!=origin['model']['content_sha256'] or ready['stack_limit']!=2 or ready['legacy_checkpoint']:raise ValueError('Activated persistent profile differs')
        for bad in ('A -1','A 151936','A 1x','G 0','C','R'):check_refusal(bad)
        last=turn(0,-1);mark=p.send('P')['frame'];nested=p.send('P')['frame'];check_refusal('P');check_refusal('B '+str(mark));p.send('K '+str(nested));turn(3,last);p.send('B '+str(mark));check_refusal('B '+str(mark));check_refusal('A '+' '.join(['1']*769));last=turn(1,last);last=turn(2,last);root=p.send('S')
        for i in range(4,8):
            frame=p.send('P')['frame'];turn(i,last);p.send('B '+str(frame));restored=p.send('S')
            if any(restored[k]!=root[k] for k in ('position','pending_token','delivered','cache_sha256','logits_sha256','evictions','rng')):raise ValueError('Persistent sibling failed exact root restore')
        # Replay a complete child, including generation, after overwriting ring slots.
        children=[]
        for repetition in range(2):
            frame=p.send('P')['frame'];turn(4,last);children.append(turns[-1]);p.send('B '+str(frame))
        if semantic(children[0])!=semantic(children[1]):raise ValueError('Persistent own child replay differs')
        stale=p.send('P')['frame'];p.send('Z');check_refusal('B '+str(stale));reset=p.send('S')
        if reset['position'] or reset['resident_rows'] or reset['evictions'] or reset['stack_depth']:raise ValueError('Persistent reset left reachable memory')
        result={'policy':policy,'ready':ready,'turns':turns,'refusals':refusals,'generation_checks':generation_checks,'delayed_root':root,'exact_sibling_replay':True,'reset':reset}
    finally:
        p.close();(raw/(label+'-protocol.json')).write_text(json.dumps(p.rows,sort_keys=True)+'\n')
    return result

def native_case(args,raw,label,origin):
    """Compare original and integrated native cores through one reporting view.

    Native cursor rollback restores pending/top metadata and the valid prefix;
    its full logits become valid again after the next completed token.
    """
    raw=Path(raw);p=Protocol(args,raw/(label+'-protocol.stderr.txt'));turns=[]
    def turn(i,last):
        q=origin['inputs'][i];ids=q['ids'];skip=int(last==ids[0]);p.send('A '+' '.join(map(str,ids[skip:])));before=p.send('S');reply=p.send('G '+str(q['budget']));delivered=p.send('S');p.send('A '+str(reply['ids'][-1]));after=p.send('S');turns.append({'input_index':i,'prefill':before,'generated':reply,'delivered_state':delivered,'final':after});return reply['ids'][-1]
    try:
        if p.ready['model_digest']!=origin['model']['content_sha256']:raise ValueError('Native reference model differs')
        last=turn(0,-1);mark=p.send('P')['frame'];turn(3,last);p.send('B '+str(mark));last=turn(1,last);last=turn(2,last);root=p.send('S')
        for i in (4,5,6,7,4,4):
            frame=p.send('P')['frame'];turn(i,last);p.send('B '+str(frame));restored=p.send('S')
            if any(restored[k]!=root[k] for k in ('position','pending_token','delivered','cache_sha256')) or restored['logits_valid']:raise ValueError('Native cursor rollback contract differs')
        p.send('Z');reset=p.send('S')
        return {'ready':p.ready,'turns':turns,'delayed_root':root,'reset':reset}
    finally:
        p.close();(raw/(label+'-protocol.json')).write_text(json.dumps(p.rows,sort_keys=True)+'\n')
