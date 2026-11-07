#include "qwen.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void){
    TcmGpu *gpu=gpu_open();
    float w[12]={1,2,3,4,4,3,2,1,-1,0,1,2},x[4]={1,-1,2,0};
    float *dw=gpu_weight(w,TCM_F32,12),*dx=gpu_weight(x,TCM_F32,4),*dy=gpu_alloc(12);float actual[16];
    gpu_matvec(gpu,dw,4,3,dx,dy);gpu_read(actual,dy,12);
    assert(actual[0]==5&&actual[1]==5&&actual[2]==1);
    gpu_free(dw);gpu_free(dx);gpu_free(dy);
    float q[16],keys[2][5][4]={0},values[2][5][4]={0};
    for(int i=0;i<16;i++)q[i]=(float)((i%5)-2)/4;
    for(int h=0;h<2;h++)for(int t=0;t<3;t++)for(int d=0;d<4;d++){keys[h][t][d]=(float)(h+t-d)/8;values[h][t][d]=(float)(h*3+t+d)/4;}
    float *dq=gpu_weight(q,TCM_F32,16),*dk=gpu_alloc(sizeof(keys)),*dv=gpu_alloc(sizeof(values)),*scores=gpu_alloc(4*5*4),*out=gpu_alloc(16*4);
    gpu_zero(dk,sizeof(keys));gpu_zero(dv,sizeof(values));
    for(int t=0;t<3;t++){
        float rowk[8],rowv[8];for(int h=0;h<2;h++)for(int d=0;d<4;d++){rowk[h*4+d]=keys[h][t][d];rowv[h*4+d]=values[h][t][d];}
        float *a=gpu_weight(rowk,TCM_F32,8),*b=gpu_weight(rowv,TCM_F32,8);gpu_cache(dk,a,2,4,5,t);gpu_cache(dv,b,2,4,5,t);gpu_free(a);gpu_free(b);
    }
    gpu_attention(gpu,out,dq,dk,dv,scores,4,2,4,5,3);gpu_read(actual,out,sizeof(actual));
    double max_error=0;
    for(int h=0;h<4;h++){
        double p[3],sum=0;for(int t=0;t<3;t++){double dot=0;for(int d=0;d<4;d++)dot+=q[h*4+d]*keys[h/2][t][d];p[t]=exp(dot/2);sum+=p[t];}
        for(int d=0;d<4;d++){double expected=0;for(int t=0;t<3;t++)expected+=p[t]/sum*values[h/2][t][d];double e=fabs(actual[h*4+d]-expected);if(e>max_error)max_error=e;}
    }
    assert(max_error<2e-5);
    gpu_rope(dq,4,4,7,1e6f);gpu_read(actual,dq,sizeof(actual));
    double rope_error=0;for(int h=0;h<4;h++)for(int d=0;d<2;d++){double angle=7/pow(1e6,2.0*d/4),a=q[h*4+d],b=q[h*4+d+2];double e=fabs(actual[h*4+d]-(a*cos(angle)-b*sin(angle)));if(e>rope_error)rope_error=e;e=fabs(actual[h*4+d+2]-(b*cos(angle)+a*sin(angle)));if(e>rope_error)rope_error=e;}
    assert(rope_error<2e-5);
    gpu_free(dq);gpu_free(dk);gpu_free(dv);gpu_free(scores);gpu_free(out);
    float logits[600]={0};logits[0]=5;logits[511]=5;float *dl=gpu_weight(logits,TCM_F32,600);TcmTop2 top=gpu_top2(gpu,dl,600);assert(top.first==0&&top.second==511);gpu_free(dl);
    QwenModel model={0};model.gpu=gpu;model.hidden=8;model.layers=1;model.qheads=4;model.kvheads=2;model.head=2;model.intermediate=12;model.vocab=16;model.max_position=32;memset(model.digest,'a',64);
    QwenSession a,b;assert(!qwen_session_init(&a,&model,8));assert(!qwen_session_init(&b,&model,8));QwenCheckpoint ca,cb;
    assert(qwen_checkpoint(&a,&ca)==-1);a.position=b.position=1;a.pending=b.pending=3;a.phase=b.phase=1;
    assert(!qwen_checkpoint(&a,&ca));assert(!qwen_checkpoint(&b,&cb));assert(qwen_restore(&b,&ca)==-1);assert(!qwen_restore(&a,&ca));
    assert(!qwen_checkpoint(&a,&cb));assert(qwen_restore(&a,&ca)==-1);assert(!qwen_restore(&a,&cb));
    qwen_session_free(&a);qwen_session_free(&b);gpu_close(gpu);
    printf("{\"matvec_exact\":true,\"attention_max_error\":%.17g,\"rope_max_error\":%.17g,\"top2_ties\":true,\"checkpoint_ownership\":true}\n",max_error,rope_error);
    return 0;
}
