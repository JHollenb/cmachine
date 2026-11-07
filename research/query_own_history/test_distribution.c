#include "distribution.h"
#include <assert.h>
#include <stdio.h>
int main(void){
    float a[]={2,-1,0},b[]={9,6,7};Distribution m;assert(!distribution(a,a,3,&m));assert(m.kl==0&&m.l2==0&&m.native_top==0&&m.candidate_top==0);assert(!distribution(a,b,3,&m));assert(fabs(m.kl)<1e-14&&fabs(m.top_logp_delta)<1e-14&&m.l2>0);assert(distribution(a,b,0,&m));b[0]=NAN;assert(distribution(a,b,3,&m));puts("distribution mechanics: nats,sum over vocabulary,shift invariance,finite/extent contracts");return 0;
}
