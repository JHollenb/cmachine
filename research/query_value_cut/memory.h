#ifndef QM_MEMORY_H
#define QM_MEMORY_H
#include <stdint.h>
#include <string.h>
#include <math.h>
#define QM_DIM 128
#define QM_GROUPS 2
#define QM_HEADS 12
#define QM_CLUSTERS 16
#define QM_RECENT 16
typedef struct {
    double key_sum[QM_GROUPS][QM_CLUSTERS][QM_DIM];
    double value_sum[QM_GROUPS][QM_CLUSTERS][QM_DIM];
    uint32_t count[QM_GROUPS][QM_CLUSTERS],clusters[QM_GROUPS];
    float keys[QM_GROUPS][QM_RECENT][QM_DIM],values[QM_GROUPS][QM_RECENT][QM_DIM];
    uint64_t next,recent,head;
} QMMemory;
static int qm_append(QMMemory *m,uint64_t position,const float *keys,const float *values){
    if(position!=m->next||m->next>=512)return -1;
    for(int i=0;i<QM_GROUPS*QM_DIM;i++)if(!isfinite(keys[i])||!isfinite(values[i]))return -1;
    if(m->recent==QM_RECENT){
        for(int g=0;g<QM_GROUPS;g++){
            const float *k=m->keys[g][m->head],*v=m->values[g][m->head];unsigned selected;
            if(m->clusters[g]<QM_CLUSTERS)selected=m->clusters[g]++;
            else{
                selected=0;double best=INFINITY;
                for(unsigned c=0;c<QM_CLUSTERS;c++){
                    double distance=0;for(int j=0;j<QM_DIM;j++){double e=(double)k[j]-m->key_sum[g][c][j]/m->count[g][c];distance+=e*e;}
                    if(distance<best){best=distance;selected=c;}
                }
            }
            m->count[g][selected]++;
            for(int j=0;j<QM_DIM;j++){m->key_sum[g][selected][j]+=k[j];m->value_sum[g][selected][j]+=v[j];}
        }
    }else m->recent++;
    for(int g=0;g<QM_GROUPS;g++){
        memcpy(m->keys[g][m->head],keys+g*QM_DIM,QM_DIM*4);
        memcpy(m->values[g][m->head],values+g*QM_DIM,QM_DIM*4);
    }
    m->head=(m->head+1)%QM_RECENT;m->next++;return 0;
}
static int qm_read(const QMMemory *m,const float *queries,float *out){
    if(!m->next||m->recent>QM_RECENT)return -1;
    for(int j=0;j<QM_HEADS*QM_DIM;j++)if(!isfinite(queries[j]))return -1;
    for(int h=0;h<QM_HEADS;h++){
        int g=h/(QM_HEADS/QM_GROUPS);double score[QM_RECENT+QM_CLUSTERS],maximum=-INFINITY,denominator=0;
        unsigned ccount=m->clusters[g],n=ccount+(unsigned)m->recent;
        for(unsigned c=0;c<ccount;c++){
            if(!m->count[g][c])return -1;double dot=0;
            for(int j=0;j<QM_DIM;j++)dot+=(double)queries[h*QM_DIM+j]*(m->key_sum[g][c][j]/m->count[g][c]);
            score[c]=dot/sqrt((double)QM_DIM)+log((double)m->count[g][c]);if(score[c]>maximum)maximum=score[c];
        }
        for(unsigned i=0;i<m->recent;i++){
            unsigned slot=(unsigned)((m->head+QM_RECENT-m->recent+i)%QM_RECENT);double dot=0;
            for(int j=0;j<QM_DIM;j++)dot+=(double)queries[h*QM_DIM+j]*m->keys[g][slot][j];
            score[ccount+i]=dot/sqrt((double)QM_DIM);if(score[ccount+i]>maximum)maximum=score[ccount+i];
        }
        for(unsigned i=0;i<n;i++){score[i]=exp(score[i]-maximum);denominator+=score[i];}
        if(!(denominator>0)||!isfinite(denominator))return -1;
        for(int j=0;j<QM_DIM;j++){
            double value=0;
            for(unsigned c=0;c<ccount;c++)value+=score[c]*(m->value_sum[g][c][j]/m->count[g][c]);
            for(unsigned i=0;i<m->recent;i++){unsigned slot=(unsigned)((m->head+QM_RECENT-m->recent+i)%QM_RECENT);value+=score[ccount+i]*m->values[g][slot][j];}
            out[h*QM_DIM+j]=(float)(value/denominator);if(!isfinite(out[h*QM_DIM+j]))return -1;
        }
    }return 0;
}
#endif
