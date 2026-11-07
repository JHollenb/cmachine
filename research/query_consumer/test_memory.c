#include "memory.h"
#include <assert.h>
#include <stdio.h>
int main(void){
    QMMemory m={0};float keys[256],values[256],queries[1536]={0},out[1536];double sum[256]={0};
    assert(qm_read(&m,queries,out)==-1);assert(qm_append(&m,1,keys,values)==-1);
    for(int t=0;t<65;t++){
        for(int j=0;j<256;j++){keys[j]=(float)((t+j)%7-3);values[j]=(float)((t*3+j)%11-5);sum[j]+=values[j];}
        assert(!qm_append(&m,t,keys,values));assert(qm_append(&m,t,keys,values)==-1);
        assert(!qm_read(&m,queries,out));
        for(int g=0;g<2;g++){
            uint64_t count=m.recent;for(unsigned c=0;c<m.clusters[g];c++)count+=m.count[g][c];assert(count==m.next);
            for(int j=0;j<128;j++){
                double total=0;for(unsigned c=0;c<m.clusters[g];c++)total+=m.value_sum[g][c][j];for(unsigned i=0;i<m.recent;i++)total+=m.values[g][i][j];assert(total==sum[g*128+j]);
                for(int h=g*6;h<(g+1)*6;h++)assert(fabs(out[h*128+j]-(float)(sum[g*128+j]/(t+1)))<1e-6);
            }
        }
    }
    QMMemory restored;memcpy(&restored,&m,sizeof(m));for(int j=0;j<1536;j++)queries[j]=(float)(j%5-2);float before[1536],after[1536];assert(!qm_read(&m,queries,before)&&!qm_read(&restored,queries,after)&&!memcmp(before,after,sizeof(before)));
    keys[0]=NAN;QMMemory parent=m;assert(qm_append(&m,65,keys,values)==-1&&!memcmp(&parent,&m,sizeof(m)));
    printf("memory mechanics: mass/value conservation,causal append,zeroquery,GQA,restoration,nonfinite refusal; state=%zu bytes\n",sizeof(m));return 0;
}
