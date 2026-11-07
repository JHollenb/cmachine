#ifndef QUERY_DISTRIBUTION_H
#define QUERY_DISTRIBUTION_H
#include <math.h>
#include <stddef.h>
typedef struct {double kl,l2,top_logp_delta,native_entropy;int native_top,candidate_top;} Distribution;
static int distribution(const float *native,const float *candidate,size_t n,Distribution *out){
    if(!n)return -1;double mn=-INFINITY,mc=-INFINITY;size_t nt=0,ct=0;
    for(size_t i=0;i<n;i++){if(!isfinite(native[i])||!isfinite(candidate[i]))return -1;if(native[i]>mn){mn=native[i];nt=i;}if(candidate[i]>mc){mc=candidate[i];ct=i;}}
    double sn=0,sc=0;for(size_t i=0;i<n;i++){sn+=exp((double)native[i]-mn);sc+=exp((double)candidate[i]-mc);}
    double ln=log(sn),lc=log(sc),kl=0,energy=0,error=0,entropy=0;
    for(size_t i=0;i<n;i++){
        double a=(double)native[i]-mn-ln,b=(double)candidate[i]-mc-lc,p=exp(a),d=(double)candidate[i]-native[i];
        kl+=p*(a-b);entropy-=p*a;error+=d*d;energy+=(double)native[i]*native[i];
    }
    *out=(Distribution){kl,sqrt(error/fmax(energy,1e-60)),((double)candidate[nt]-mc-lc)-((double)native[nt]-mn-ln),entropy,(int)nt,(int)ct};
    return isfinite(kl)&&isfinite(out->l2)&&isfinite(out->top_logp_delta)?0:-1;
}
#endif
