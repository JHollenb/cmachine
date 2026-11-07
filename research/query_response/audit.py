"""Post-seal diagnostics. Memberships and dense histories are never serving state."""
import numpy as np
from analysis import Memory,probability_row,rel,stats

class MembershipMemory(Memory):
    def __init__(self):
        super().__init__();self.members=[[[] for _ in range(16)] for _ in range(2)];self.slots=[None]*16
    def append(self,k,v):
        if self.recent==16:
            for g in range(2):
                if self.clusters[g]<16:c=int(self.clusters[g])
                else:
                    distances=[]
                    for j in range(16):
                        delta=self.keys[g,self.head].astype(float)-self.ks[g,j]/int(self.count[g,j])
                        distances.append(sum(float(x)*float(x) for x in delta))
                    c=min(range(16),key=lambda j:distances[j])
                self.members[g][c].append(self.slots[self.head])
        self.slots[self.head]=self.next
        super().append(k,v)
    def key_signature(self):
        return (self.ks.tobytes(),self.count.tobytes(),self.clusters.tobytes(),self.keys.tobytes(),self.next,self.recent,self.head,tuple(self.slots),tuple(tuple(tuple(m) for m in g) for g in self.members))
    def weights(self,q):
        """Lift each prototype's mass to its original rows for diagnostic algebra."""
        result=np.zeros((12,self.next));slots=[(self.head+16-self.recent+i)%16 for i in range(self.recent)]
        for h,query in enumerate(q):
            g=h//6;c=int(self.clusters[g])
            keys=np.concatenate((self.ks[g,:c]/self.count[g,:c,None],self.keys[g,slots].astype(float)))
            mass=np.concatenate((self.count[g,:c].astype(float),np.ones(self.recent)))
            score=keys@query.astype(float)/np.sqrt(128.)+np.log(mass);w=np.exp(score-score.max());w/=w.sum()
            for j in range(c):
                if len(self.members[g][j])!=int(self.count[g,j]):raise ValueError('Audit membership mass differs')
                result[h,self.members[g][j]]=w[j]/int(self.count[g,j])
            for j,slot in enumerate(slots):result[h,self.slots[slot]]=w[c+j]
        return result

def source_fit_audit(raw,basis,driver):
    from compiler import verify_fit
    return verify_fit(raw,driver)


def common_prefix_factorials(driver,records):
    rows=[];tasks=[]
    for task in range(3):
        labels=('values-native','values-native-memory','query-fit','query-fit-memory')
        parts=[records[f'case{task}-{label}'] for label in labels];descs=[p[0] for p in parts];arrays=[p[1] for p in parts]
        common=0
        for ids in zip(*(d['consumed_ids'] for d in descs)):
            if len(set(ids))!=1:break
            common+=1
        initial=descs[0]['turns'][0]['prefill']['position']
        if common<initial:raise ValueError('Common prefix lost inside prompt')
        memories=[MembershipMemory(),MembershipMemory()];flags=[];actuals=[];residuals=[];mix_errors=[]
        for t in range(common):
            for m,a in zip(memories,(arrays[1],arrays[3])):m.append(a['k'][t],a['v'][t])
            ps=[probability_row(arrays[i]['probabilities'],t) for i in (0,2)]
            ws=[m.weights(arrays[i]['q'][t]) for m,i in zip(memories,(1,3))];weights=(ps[0],ws[0],ps[1],ws[1])
            equal=(all(np.array_equal(a['norm'][t],arrays[0]['norm'][t]) and np.array_equal(a['k'][t],arrays[0]['k'][t]) and np.array_equal(a['v'][t],arrays[0]['v'][t]) for a in arrays[1:]) and np.array_equal(arrays[0]['q'][t],arrays[1]['q'][t]) and np.array_equal(arrays[2]['q'][t],arrays[3]['q'][t]) and memories[0].bytes()==memories[1].bytes() and memories[0].key_signature()==memories[1].key_signature())
            flags.append(bool(equal));equal=all(flags)
            mixed=[np.asarray([p[h]@a['v'][:t+1,h//6].astype(float) for h in range(12)]) for p,a in zip(weights,arrays)]
            ys=[a['attention'][t].astype(float) for a in arrays];actual=ys[3]-ys[2]-ys[1]+ys[0]
            predicted=np.asarray([(weights[3][h]-weights[1][h]-weights[2][h]+weights[0][h])@arrays[0]['v'][:t+1,h//6].astype(float) for h in range(12)])
            rest=[y-m for y,m in zip(ys,mixed)];finite=rest[3]-rest[2]-rest[1]+rest[0];den=max(float(np.linalg.norm(ys[0])),1e-30)
            actuals.append(actual);residuals.append(actual-predicted-finite);mix_errors.extend(rel(y,m) for y,m in zip(ys,mixed))
            rows.append({'kind':'common-input-query-memory','task':task,'step':t,'information_merge_active':t>=32,'formula_instrument_status':'applicable' if equal else 'operand-or-partition-mismatch','interaction_relative_to_baseline_full':float(np.linalg.norm(actual)/den),'formula_discrepancy_relative_to_baseline_full':float(np.linalg.norm(actual-predicted)/den),'four_mix_remainders_relative_to_baseline_full':[float(np.linalg.norm(r)/den) for r in rest],'factorization_account_remainder_relative_to_baseline_full':float(np.linalg.norm(actual-predicted-finite)/den)})
        regions={}
        for name,start,stop in (('initial_prompt',0,initial),('maximal_common_prefix',0,common),('numerical_profile_only',0,min(32,common)),('merged_memory',32,common)):
            if stop<=start:continue
            den=max(float(np.linalg.norm(arrays[0]['attention'][start:stop].astype(float))),1e-30)
            regions[name]={'positions':stop-start,'interaction_relative_to_baseline_full':float(np.linalg.norm(actuals[start:stop])/den),'factorization_account_remainder_relative_to_baseline_full':float(np.linalg.norm(residuals[start:stop])/den)}
        tasks.append({'task':task,'initial_prompt_positions':initial,'maximal_common_consumed_prefix':common,'formula_instrument_status':'applicable' if all(flags) else 'operand-or-partition-mismatch','paired_receivers_queries_keys_values_and_memory_byte_exact':all(flags),'mix_remainder':stats(mix_errors),'regions':regions})
    return {'scope':'Development four-policy common consumed-ID prefix only; query changes at identical selected K/V and compact states. Upper cache remains policy-specific. Lifted weights/membership are post-seal diagnostics. Fresh asymmetric panel does not identify a memory interaction.','tasks':tasks,'rows':rows}

def policy_transfers(raw,driver):
    rows=[]
    for task in range(6):
        for turn in range(2):
            for stage in ('prefill','final'):
                labels=['native','values-native-memory','query-fit-memory']
                if task<3:labels+=['values-native','query-fit']
                paths=[next(p for p in driver['paths'] if p['name']==f'case{task}-{label}') for label in labels]
                ys={label:np.fromfile(raw/f'{p["name"]}-t{turn}-{stage}-logits.f32',dtype='<f4').astype(float) for label,p in zip(labels,paths)}
                histories={label:p['consumed_ids'][:p['turns'][turn][stage]['position']] for label,p in zip(labels,paths)}
                den=max(float(ys['native']@ys['native']),1e-60);errors={label:float((y-ys['native'])@(y-ys['native'])/den) for label,y in ys.items()}
                compact_gain=errors['values-native-memory']-errors['query-fit-memory'];full_gain=errors['values-native']-errors['query-fit'] if task<3 else None
                rows.append({'kind':'same-cost-query-gain','task':task,'split':'development' if task<3 else 'prospective-transfer','turn':turn,'stage':stage,'all_same_consumed_ids':all(h==histories['native'] for h in histories.values()),'consumed_history_matches_native':{label:h==histories['native'] for label,h in histories.items()},'old_new_compact_same_consumed_ids':histories['values-native-memory']==histories['query-fit-memory'],'relative_logit_difference':{label:float(np.sqrt(error)) for label,error in errors.items()},'full_gain_signed_squared_error':full_gain,'compact_gain_signed_squared_error':compact_gain,'gain_transfer_difference':compact_gain-full_gain if full_gain is not None else None})
    return rows

def query_operand_comparisons(records):
    rows=[]
    for task in range(6):
        labels=('native','values-native-memory','query-fit-memory');parts=[records[f'case{task}-{label}'] for label in labels]
        descs=[p[0] for p in parts];arrays=[p[1] for p in parts];initial=descs[0]['turns'][0]['prefill']['position'];common=0
        for ids in zip(*(p['consumed_ids'] for p in descs)):
            if len(set(ids))!=1:break
            common+=1
        if common<initial:raise ValueError('Initial operand comparison lost common history')
        for region,n in (('initial_prompt',initial),('maximal_common_prefix',common)):
            same=all(np.array_equal(a['norm'][:n],arrays[0]['norm'][:n]) for a in arrays[1:])
            for label,a in zip(labels[1:],arrays[1:]):
                rows.append({'kind':'query-program-operand-response','task':task,'split':'development' if task<3 else 'prospective-transfer','candidate':f'case{task}-{label}','region':region,'positions':n,'lower_receivers_byte_exact':same,'query_relative_error':rel(a['q'][:n],arrays[0]['q'][:n]),'attention_relative_error':rel(a['attention'][:n],arrays[0]['attention'][:n]),'layer_output_relative_error':rel(a['output'][:n],arrays[0]['output'][:n])})

    return rows
