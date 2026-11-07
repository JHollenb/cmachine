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

def rel(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-30))
def stats(values):
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),'max':float(np.max(values)),'min':float(np.min(values))}

def vector_factorial(nn,cn,nc,cc):
    nn,cn,nc,cc=[np.asarray(x,dtype=float).reshape(-1) for x in (nn,cn,nc,cc)]
    qk,value,interaction=cn-nn,nc-nn,cc-cn-nc+nn;joint=cc-nn;den=max(float(np.linalg.norm(nn)),1e-30)
    return {'qk_only_relative':float(np.linalg.norm(qk)/den),'value_only_relative':float(np.linalg.norm(value)/den),'both_relative':float(np.linalg.norm(joint)/den),'interaction_relative':float(np.linalg.norm(interaction)/den),'qk_value_cross_scaled':float(2*qk@value/den**2),'interaction_qk_cross_scaled':float(2*interaction@qk/den**2),'interaction_value_cross_scaled':float(2*interaction@value/den**2),'vector_closure_relative':float(np.linalg.norm(joint-qk-value-interaction)/den)}

def probability_row(values,t):
    return values[6*t*(t+1):6*(t+1)*(t+2)].reshape(12,t+1).astype(float)

