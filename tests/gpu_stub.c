/* Tiny CPU implementation for model-free state-lifecycle tests. This does not
 * qualify CUDA numerical behavior and is never linked into the real executor. */
#include "gpu.h"
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
void gpu_attention(TcmGpu *g,float *out,const float *q,const float *k,const float *v,float *scores,int qheads,int kvheads,int d,int capacity,int n){
    (void)g;for(int h=0;h<qheads;h++){
        int group=h/(qheads/kvheads);float *p=scores+h*capacity,max=-INFINITY,sum=0;
        for(int t=0;t<n;t++){float dot=0;for(int i=0;i<d;i++)dot+=q[h*d+i]*k[((size_t)group*capacity+t)*d+i];p[t]=dot/sqrtf((float)d);if(p[t]>max)max=p[t];}
        for(int t=0;t<n;t++){p[t]=expf(p[t]-max);sum+=p[t];}
        for(int i=0;i<d;i++){float a=0;for(int t=0;t<n;t++)a+=p[t]/sum*v[((size_t)group*capacity+t)*d+i];out[h*d+i]=a;}
    }
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
