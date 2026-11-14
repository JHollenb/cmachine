/* Tiny CPU implementation for model-free state-lifecycle tests. This does not
 * qualify CUDA numerical behavior and is never linked into the real executor. */
#include "gpu.h"
#include "qwen_ledger.h"
int gpu_weight_profile(void){return 0;}
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
struct TcmGpu { int unused; };
TcmGpu *gpu_open(void){return calloc(1,sizeof(TcmGpu));}
void gpu_close(TcmGpu *g){free(g);}
void *gpu_alloc(size_t n){void *p=calloc(1,n);assert(p);return p;}
void gpu_free(void *p){free(p);}
void gpu_zero(void *p,size_t n){memset(p,0,n);}
void gpu_copy(void *a,const void *b,size_t n){memcpy(a,b,n);}
void gpu_read(void *a,const void *b,size_t n){memcpy(a,b,n);}
void gpu_write(void *a,const void *b,size_t n){memcpy(a,b,n);}
float *gpu_weight(const void *p,uint32_t type,size_t n){
    float *out=gpu_alloc(n*4);if(type==1)memcpy(out,p,n*4);
    else{assert(type==2);for(size_t i=0;i<n;i++){uint32_t bits=(uint32_t)((const uint16_t*)p)[i]<<16;memcpy(out+i,&bits,4);}}return out;
}
void gpu_matvec(TcmGpu *g,const float *w,int in,int out,const float *x,float *y){
    (void)g;for(int j=0;j<out;j++){float sum=0;for(int i=0;i<in;i++)sum+=w[(size_t)j*in+i]*x[i];y[j]=sum;}
}
void gpu_rms(float *y,const float *x,const float *w,int n,float epsilon){
    float sum=0;for(int i=0;i<n;i++)sum+=x[i]*x[i];float r=1/sqrtf(sum/n+epsilon);for(int i=0;i<n;i++)y[i]=x[i]*r*w[i];
}
void gpu_rms_scale(float *y,const float *x,const float *w,int n,float e,float *r){float sum=0;for(int i=0;i<n;i++)sum+=x[i]*x[i];*r=1/sqrtf(sum/n+e);gpu_rms(y,x,w,n,e);}
void gpu_multiply(float *out,const float *a,const float *b,int n){for(int i=0;i<n;i++)out[i]=a[i]*b[i];}
void gpu_plane(float *out,const float *base,const float *a,const float *b,int n,float s,float t){for(int i=0;i<n;i++)out[i]=(base[i]+s*a[i])+t*b[i];}
void gpu_plane_scaled(float *out,const float *base,const float *a,const float *b,int n,float s,float t,const float *r){gpu_plane(out,base,a,b,n,s,t);for(int i=0;i<n;i++)out[i]*=*r;}
void gpu_linear2(float *out,const float *a,const float *b,int n,float ca,float cb,const float *scale){for(int i=0;i<n;i++){float v=ca*a[i];if(b)v+=cb*b[i];out[i]=scale?(*scale)*v:v;}}
int gpu_identical_finite(TcmGpu *g,const float *a,const float *b,int n){(void)g;for(int i=0;i<n;i++)if(!isfinite(a[i])||!isfinite(b[i])||memcmp(a+i,b+i,4))return 0;return 1;}
void gpu_add(float *a,const float *b,int n){for(int i=0;i<n;i++)a[i]+=b[i];}
void gpu_bias(float *a,const float *b,int n){gpu_add(a,b,n);}
void gpu_rope(float *q,int heads,int d,int position,float theta){
    for(int h=0;h<heads;h++)for(int i=0;i<d/2;i++){float angle=position/powf(theta,2.0f*i/d),a=q[h*d+i],b=q[h*d+i+d/2];q[h*d+i]=a*cosf(angle)-b*sinf(angle);q[h*d+i+d/2]=b*cosf(angle)+a*sinf(angle);}
}
void gpu_cache(float *cache,const float *row,int heads,int d,int capacity,int position){
    for(int h=0;h<heads;h++)memcpy(cache+((size_t)h*capacity+position)*d,row+h*d,(size_t)d*4);
}
static void attention(float *out,const float *q,const float *k,const float *v,float *scores,int qheads,int kvheads,int d,int capacity,int n,float *ledger){
    for(int h=0;h<qheads;h++){
        int group=h/(qheads/kvheads);float *p=scores+h*n,max=-INFINITY,sum=0;double partition=0;
        for(int t=0;t<n;t++){float dot=0;for(int i=0;i<d;i++)dot+=q[h*d+i]*k[((size_t)group*capacity+t)*d+i];p[t]=dot/sqrtf((float)d);if(p[t]>max)max=p[t];}
        for(int t=0;t<n;t++){if(ledger)partition+=exp((double)p[t]-(double)max);p[t]=expf(p[t]-max);sum+=p[t];}
        for(int i=0;i<d;i++){float a=0;for(int t=0;t<n;t++)a+=p[t]/sum*v[((size_t)group*capacity+t)*d+i];out[h*d+i]=a;}
        /* Match CUDA's published packed, normalized score scratch. Preserve
         * the stub's historical return arithmetic above. */
        for(int t=0;t<n;t++)p[t]/=sum;
        if(ledger){
            /* Ledger fields from this pass's own max/partition and native probabilities. */
            double entropy=0;float best=-1;int index=0;
            for(int t=0;t<n;t++){float x=p[t];if(x>0.0f)entropy-=(double)(x*logf(x));if(p[t]>best){best=p[t];index=t;}}
            ledger[QL_H_LOGZ*qheads+h]=(float)((double)max+log(partition));ledger[QL_H_PFIRST*qheads+h]=p[0];
            ledger[QL_H_ENTROPY*qheads+h]=(float)entropy;ledger[QL_H_PMAX*qheads+h]=best;ledger[QL_H_ARGMAX*qheads+h]=(float)index;
        }
    }
}
void gpu_attention(TcmGpu *g,float *out,const float *q,const float *k,const float *v,float *scores,int qheads,int kvheads,int d,int capacity,int n){
    (void)g;attention(out,q,k,v,scores,qheads,kvheads,d,capacity,n,NULL);
}
void gpu_attention_ledger(TcmGpu *g,float *out,const float *q,const float *k,const float *v,float *scores,int qheads,int kvheads,int d,int capacity,int n,float *ledger){
    (void)g;attention(out,q,k,v,scores,qheads,kvheads,d,capacity,n,ledger);
}
void gpu_ledger_dot(float *dot,float *norm2,const float *x,const float *w,int n,float *retain){
    double a=0,b=0;for(int i=0;i<n;i++){double v=w[i];if(x)a+=(double)x[i]*v;b+=v*v;if(retain)retain[i]=w[i];}
    if(dot)*dot=(float)a;if(norm2)*norm2=(float)b;
}
void gpu_ledger_gate(float *out,const float *g,const float *u,int n,double *scratch){
    (void)scratch;int small=0,off=0,on=0;double even2=0,odd2=0,cross=0;
    for(int i=0;i<n;i++){
        float gi=g[i];if(fabsf(gi)<1.0f)small++;if(gi<=-5.0f)off++;if(gi>=5.0f)on++;
        float even=(gi*u[i])*0.5f,odd=even*tanhf(gi*0.5f);even2+=(double)(even*even);odd2+=(double)(odd*odd);cross+=(double)(even*odd); /* FP32 terms, double sums (as the GPU) */
    }
    out[0]=(float)small;out[1]=(float)off;out[2]=(float)on;out[3]=(float)even2;out[4]=(float)odd2;out[5]=(float)cross;
}
void gpu_ledger_margin(float *row,const float *embedding,const float *gamma,const float *writes,int layers,int block_floats,int n,int a,int b,double *scratch){
    (void)scratch;
    /* Stub weight profile is native FP32 only. */
    double rho=row[QL_RHO_FINAL],total=0,norm=0;
    for(int i=0;i<n;i++){double c=(double)gamma[i]*((double)embedding[(size_t)a*n+i]-(double)embedding[(size_t)b*n+i]);norm+=c*c;}
    row[QL_C_NORM]=(float)sqrt(norm);
    for(int j=0;j<2*layers+1;j++){
        double dot=0;for(int i=0;i<n;i++){double c=(double)gamma[i]*((double)embedding[(size_t)a*n+i]-(double)embedding[(size_t)b*n+i]);dot+=c*(double)writes[(size_t)j*n+i];}
        double term=rho*dot;total+=term;
        if(!j)row[QL_MARGIN_EMBED]=(float)term;else row[QL_HEAD+(size_t)((j-1)/2)*block_floats+((j-1)%2?QL_MARGIN_MLP:QL_MARGIN_ATTN)]=(float)term;
    }
    row[QL_MARGIN_SUM]=(float)total;
}
void gpu_swiglu(float *gate,const float *up,int n){for(int i=0;i<n;i++)gate[i]=gate[i]/(1+expf(-gate[i]))*up[i];}
void gpu_gate_anchor(const float *g,float *p,float *first,float *second,float *s,int n){for(int i=0;i<n;i++){float denominator=1.0f+expf(-g[i]);p[i]=1.0f/denominator;first[i]=p[i]*(1.0f-p[i]);second[i]=(0.5f*first[i])*(1.0f-2.0f*p[i]);s[i]=g[i]/denominator;}}
void gpu_gate_swiglu(float *g,const float *u,const float *source,const float *p,const float *first,const float *second,const float *s,int n){for(int i=0;i<n;i++){float d=g[i]-source[i];float sigma=(p[i]+first[i]*d)+(second[i]*d)*d;float predicted=s[i]+(g[i]*sigma-source[i]*p[i]);g[i]=predicted*u[i];}}
TcmTop2 gpu_top2(TcmGpu *g,const float *p,int n){
    (void)g;TcmTop2 r={-1,-1,-INFINITY,-INFINITY};
    for(int i=0;i<n;i++)if(p[i]>r.first_value){r.second=r.first;r.second_value=r.first_value;r.first=i;r.first_value=p[i];}else if(p[i]>r.second_value){r.second=i;r.second_value=p[i];}return r;
}
void gpu_sync(void){}
size_t gpu_free_bytes(void){return 0;}
float *gpu_owned_weight(TcmGpu *g,const void *p,uint32_t type,size_t n,int rank,int a,int b){(void)g;(void)rank;(void)a;(void)b;return gpu_weight(p,type,n);}
size_t gpu_owned_weight_bytes(const float *p,size_t n){(void)p;return n*4;}
void gpu_owned_embedding(float *out,const float *p,int token,int width){gpu_copy(out,p+(size_t)token*width,(size_t)width*4);}
