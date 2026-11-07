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

def common_prefix_factorials(driver,records):
    rows=[];tasks=[]
    for task in range(3):
        labels=('values-native','values-native-memory','compiled','compiled-memory')
        parts=[records[f'case{task}-{label}'] for label in labels];descs=[p[0] for p in parts];arrays=[p[1] for p in parts]
        common=0
        for ids in zip(*(d['consumed_ids'] for d in descs)):
            if len(set(ids))!=1:break
            common+=1
        initial=descs[0]['turns'][0]['prefill']['position']
        if common<initial:raise ValueError('Common prefix lost inside initial prompt')
        memories=[MembershipMemory(),MembershipMemory()];formulas=[];actuals=[];residuals=[];pair_flags=[];mix_errors=[]
        for t in range(common):
            for m,a in zip(memories,(arrays[1],arrays[3])):m.append(a['k'][t],a['v'][t])
            ps=[probability_row(arrays[i]['probabilities'],t) for i in (0,2)]
            ws=[m.weights(arrays[i]['q'][t]) for m,i in zip(memories,(1,3))]
            equal=(all(np.array_equal(a['norm'][t],arrays[0]['norm'][t]) and np.array_equal(a['q'][t],arrays[0]['q'][t]) and np.array_equal(a['k'][t],arrays[0]['k'][t]) for a in arrays[1:]) and np.array_equal(arrays[0]['v'][t],arrays[1]['v'][t]) and np.array_equal(arrays[2]['v'][t],arrays[3]['v'][t]) and np.array_equal(ps[0],ps[1]) and memories[0].key_signature()==memories[1].key_signature() and np.array_equal(ws[0],ws[1]))
            pair_flags.append(bool(equal))
            equal=all(pair_flags)  # Every earlier V row participates in this mixture.
            weights=(ps[0],ws[0],ps[1],ws[1]);mixed=[np.asarray([p[h]@a['v'][:t+1,h//6].astype(float) for h in range(12)]) for p,a in zip(weights,arrays)]
            ys=[a['attention'][t].astype(float) for a in arrays];actual=ys[3]-ys[2]-ys[1]+ys[0]
            formula=np.asarray([(ws[0][h]-ps[0][h])@(arrays[2]['v'][:t+1,h//6].astype(float)-arrays[0]['v'][:t+1,h//6].astype(float)) for h in range(12)])
            rest=[y-m for y,m in zip(ys,mixed)];finite=rest[3]-rest[2]-rest[1]+rest[0];den=max(float(np.linalg.norm(ys[0])),1e-30)
            actuals.append(actual);formulas.append(formula);residuals.append(actual-formula-finite);mix_errors.extend(rel(y,m) for y,m in zip(ys,mixed))
            rows.append({'kind':'common-input-value-memory','task':task,'step':t,'token_id':descs[0]['consumed_ids'][t],'information_merge_active':t>=32,'formula_instrument_status':'applicable' if equal else 'operand-or-partition-mismatch','interaction_relative_to_cn_full':float(np.linalg.norm(actual)/den),'formula_discrepancy_relative_to_cn_full':float(np.linalg.norm(actual-formula)/den),'four_mix_remainders_relative_to_cn_full':[float(np.linalg.norm(r)/den) for r in rest],'signed_mix_remainder_relative_to_cn_full':float(np.linalg.norm(finite)/den),'factorization_account_remainder_relative_to_cn_full':float(np.linalg.norm(actual-formula-finite)/den)})
        regions={}
        for name,start,stop in (('initial_prompt',0,initial),('maximal_common_prefix',0,common),('numerical_profile_only',0,min(32,common)),('merged_memory',32,common)):
            if stop<=start:continue
            den=max(float(np.linalg.norm(arrays[0]['attention'][start:stop].astype(float))),1e-30)
            regions[name]={'positions':stop-start,'interaction_relative_to_cn_full':float(np.linalg.norm(actuals[start:stop])/den),'formula_discrepancy_relative_to_cn_full':float(np.linalg.norm(np.asarray(actuals[start:stop])-formulas[start:stop])/den),'factorization_account_remainder_relative_to_cn_full':float(np.linalg.norm(residuals[start:stop])/den)}
        tasks.append({'task':task,'initial_prompt_positions':initial,'maximal_common_consumed_prefix':common,'formula_instrument_status':'applicable' if all(pair_flags) else 'operand-or-partition-mismatch','paired_operands_probabilities_and_key_partitions_byte_exact':all(pair_flags),'mix_remainder':stats(mix_errors),'regions':regions})
    return {'scope':'Common exact consumed-ID prefix and selected receiver only; upper caches remain policy-specific. Lifted weights/memberships are post-seal diagnostics, not serving state. First32 positions compare numerical profiles; merges begin at33.','tasks':tasks,'rows':rows}

def policy_transfers(raw,driver):
    """Signed native-V gains under both readers use the same NN/full reference."""
    rows=[]
    for task in range(3):
        for turn in range(2):
            for stage in ('prefill','final'):
                labels=('native','memory','compiled','compiled-memory','values-native','values-native-memory')
                paths=[next(p for p in driver['paths'] if p['name']==f'case{task}-{label}') for label in labels]
                ys=[np.fromfile(raw/f'{p["name"]}-t{turn}-{stage}-logits.f32',dtype='<f4').astype(float) for p in paths]
                den=max(float(ys[0]@ys[0]),1e-60);errors=[float((y-ys[0])@(y-ys[0])/den) for y in ys]
                full_gain=errors[2]-errors[4];compact_gain=errors[3]-errors[5]
                histories=[p['consumed_ids'][:p['turns'][turn][stage]['position']] for p in paths]
                interaction=ys[3]-ys[2]-ys[5]+ys[4]
                rows.append({'kind':'native-value-gain-transfer','task':task,'turn':turn,'stage':stage,'all_same_consumed_ids':all(h==histories[0] for h in histories),'relative_logit_difference':{label:float(np.sqrt(error)) for label,error in zip(labels,errors)},'full_gain_signed_squared_error':full_gain,'compact_gain_signed_squared_error':compact_gain,'gain_transfer_difference':compact_gain-full_gain,'value_memory_policy_interaction_relative_to_native':float(np.linalg.norm(interaction)/np.sqrt(den))})
    return rows
