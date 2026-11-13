#include "circuit_expansions.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int vec(const double *x,size_t n){if(!x||!n)return 0;for(size_t i=0;i<n;i++)if(!isfinite(x[i]))return 0;return 1;}
static int shape(size_t t,size_t n){return t&&n&&t<=4096&&n<=4096&&t<=SIZE_MAX/n/sizeof(double);}
double cm_gelu(double x){return .5*x*erfc(-x/sqrt(2.0));}
double cm_gelu_first(double x){if(fabs(x)>40)return x>0?1:0;return .5*erfc(-x/sqrt(2.0))+x*exp(-.5*x*x)/sqrt(2*acos(-1.0));}
double cm_gelu_second(double x){if(fabs(x)>40)return 0;return (2-x*x)*exp(-.5*x*x)/sqrt(2*acos(-1.0));}
double cm_gelu_secant(double a,double b){
    if(!isfinite(a)||!isfinite(b))return NAN;
    if(a==b)return cm_gelu_first(a);
    double delta=b-a,mid=.5*a+.5*b;
    /* Four-point Gauss integration avoids subtracting nearly equal GELUs.
     * The mathematical secant is the average derivative on this segment. */
    if(fabs(delta)<1e-4){
        const double x[]={.3399810435848562648,.8611363115940525752};
        const double w[]={.6521451548625461426,.3478548451374538574};
        double sum=0;for(int i=0;i<2;i++)sum+=.5*w[i]*(cm_gelu_first(mid+.5*delta*x[i])+cm_gelu_first(mid-.5*delta*x[i]));
        return sum;
    }return (cm_gelu(b)-cm_gelu(a))/delta;
}
int cm_gelu_split(double a,double observed,double *linear,double *nonlinear,double *remainder){
    if(!linear||!nonlinear||!remainder||!isfinite(a)||!isfinite(observed))return -1;
    double l=.5*a,n=.5*a*erf(a/sqrt(2.0)),r=observed-l-n;
    if(!isfinite(l)||!isfinite(n)||!isfinite(r))return -1;
    *linear=l;*nonlinear=n;*remainder=r;return 0;
}
static int producers(size_t t,size_t n,const double *r,const double *b,const double *c,const double *dt){
    if(!shape(t,n)||!vec(r,t*n)||!vec(b,t*n)||!vec(c,t*n)||!vec(dt,t))return 0;
    for(size_t i=0;i<t*n;i++)if(r[i]<0||r[i]>1)return 0;
    for(size_t i=0;i<t;i++)if(dt[i]<0)return 0;
    return 1;
}
int cm_ssm_kernel(size_t t,size_t n,const double *r,const double *b,const double *c,const double *dt,double *out){
    if(!out||!producers(t,n,r,b,c,dt)||!shape(t,t))return -1;
    memset(out,0,t*t*sizeof(double));
    double *state=malloc(n*sizeof(double));if(!state)return -1;
    for(size_t s=0;s<t;s++){
        for(size_t j=0;j<n;j++)state[j]=dt[s]*b[s*n+j];
        for(size_t target=s;target<t;target++){
            if(target>s)for(size_t j=0;j<n;j++)state[j]*=r[target*n+j];
            for(size_t j=0;j<n;j++)out[target*t+s]+=c[target*n+j]*state[j];
        }
    }free(state);return vec(out,t*t)?0:-1;
}
int cm_ssm_cut(size_t t,size_t n,size_t cut,const double *r,const double *b,const double *c,const double *dt,double *left,double *entry){
    if(!left||!entry||!producers(t,n,r,b,c,dt)||cut==0||cut>=t)return -1;
    for(size_t j=0;j<n;j++){
        double product=1;
        for(size_t target=cut;target<t;target++){product*=r[target*n+j];left[(target-cut)*n+j]=c[target*n+j]*product;}
        product=1;
        for(size_t s=cut;s-->0;){entry[j*cut+s]=dt[s]*b[s*n+j]*product;product*=r[s*n+j];}
    }return vec(left,(t-cut)*n)&&vec(entry,n*cut)?0:-1;
}
int cm_ssm_path(size_t t,size_t n,const double *r,const double *injection,const double *c,size_t source,size_t target,double *out){
    if(!out||!shape(t,n)||source>=t||target>=t||source>target||!vec(r,t*n)||!vec(injection,t*n)||!vec(c,t*n))return -1;
    for(size_t j=0;j<n;j++){
        double value=injection[source*n+j];for(size_t k=source+1;k<=target;k++)value*=r[k*n+j];out[j]=value*c[target*n+j];
    }return vec(out,n)?0:-1;
}
int cm_ssm_finite(size_t n,const double *r0,const double *r1,const double *h0,const double *h1,const double *i0,const double *i1,const double *y0,const double *y1,double *out){
    if(!out||!vec(r0,n)||!vec(r1,n)||!vec(h0,n)||!vec(h1,n)||!vec(i0,n)||!vec(i1,n)||!vec(y0,n)||!vec(y1,n))return -1;
    for(size_t j=0;j<n;j++){
        out[j]=r1[j]*(h1[j]-h0[j]);out[n+j]=(r1[j]-r0[j])*h0[j];out[2*n+j]=i1[j]-i0[j];
        out[3*n+j]=(y1[j]-y0[j])-out[j]-out[n+j]-out[2*n+j];
    }return vec(out,4*n)?0:-1;
}
int cm_consumer_metric(size_t consumers,size_t n,const double *jacobian,const double *error,const double *weight,double *signed_effect,double *energy){
    if(!signed_effect||!energy||!shape(consumers,n)||!vec(jacobian,consumers*n)||!vec(error,n)||!vec(weight,consumers))return -1;
    for(size_t j=0;j<consumers;j++)if(weight[j]<0)return -1;
    double total=0;
    for(size_t j=0;j<consumers;j++){
        double value=0;for(size_t i=0;i<n;i++)value+=jacobian[j*n+i]*error[i];signed_effect[j]=value;total+=weight[j]*value*value;
    }if(!vec(signed_effect,consumers)||!isfinite(total))return -1;
    *energy=total;return 0;
}
int cm_unpatchify(size_t h,size_t w,const double *phase,const double *scale,const double *bias,double *out){
    if(!out||out==phase||!shape(h,w)||h*w>4096||!vec(phase,4*h*w)||!vec(scale,4)||!vec(bias,4))return -1;
    for(size_t a=0;a<2;a++)for(size_t b=0;b<2;b++)for(size_t y=0;y<h;y++)for(size_t x=0;x<w;x++){
        size_t p=2*a+b;out[(2*y+a)*(2*w)+2*x+b]=phase[p*h*w+y*w+x]*scale[p]+bias[p];
    }return vec(out,4*h*w)?0:-1;
}
int cm_dft2(size_t h,size_t w,const CmComplex *in,CmComplex *out,int inverse){
    if(!in||!out||in==out||!shape(h,w)||h*w>16384||(inverse!=0&&inverse!=1))return -1;
    size_t longest=h>w?h:w;CmComplex *tmp=malloc(h*w*sizeof(*tmp)),*col=malloc(2*longest*sizeof(*col));
    if(!tmp||!col){free(tmp);free(col);return -1;}
    for(size_t y=0;y<h;y++)if(cm_dft(w,in+y*w,tmp+y*w,inverse))goto fail;
    for(size_t x=0;x<w;x++){
        for(size_t y=0;y<h;y++)col[y]=tmp[y*w+x];
        if(cm_dft(h,col,col+longest,inverse))goto fail;
        for(size_t y=0;y<h;y++)out[y*w+x]=col[longest+y];
    }free(tmp);free(col);return 0;
fail:free(tmp);free(col);return -1;
}
int cm_polyphase_combine(size_t h,size_t w,const CmComplex *phase,CmComplex *out){
    if(!phase||!out||phase==out||!shape(h,w)||h*w>4096)return -1;
    double pi=acos(-1.0);
    for(size_t k=0;k<2*h;k++)for(size_t l=0;l<2*w;l++){
        double re=0,im=0;
        for(size_t a=0;a<2;a++)for(size_t b=0;b<2;b++){
            CmComplex z=phase[(2*a+b)*h*w+(k%h)*w+l%w];if(!isfinite(z.re)||!isfinite(z.im))return -1;
            double angle=-pi*(k*(double)a/h+l*(double)b/w),c=cos(angle),s=sin(angle);
            re+=c*z.re-s*z.im;im+=c*z.im+s*z.re;
        }if(!isfinite(re)||!isfinite(im))return -1;
        out[k*(2*w)+l]=(CmComplex){re,im};
    }return 0;
}
int cm_polyphase_audit(size_t h,size_t w,const double *phase,const double *scale,const double *bias,CmPolyphaseAudit *out){
    if(!out||!shape(h,w)||h*w>4096)return -1;
    size_t n=h*w;double *spatial=malloc(4*n*sizeof(double));CmComplex *buf=calloc(20*n,sizeof(*buf));
    if(!spatial||!buf){free(spatial);free(buf);return -1;}
    CmComplex *planes=buf,*frequency=buf+4*n,*predicted=buf+8*n,*actual=buf+12*n,*input=buf+16*n;
    CmPolyphaseAudit report={0};
    if(cm_unpatchify(h,w,phase,scale,bias,spatial))goto fail;
    for(size_t p=0;p<4;p++)for(size_t j=0;j<n;j++)planes[p*n+j].re=phase[p*n+j]*scale[p]+bias[p];
    for(size_t p=0;p<4;p++)if(cm_dft2(h,w,planes+p*n,frequency+p*n,0))goto fail;
    if(cm_polyphase_combine(h,w,frequency,predicted))goto fail;
    for(size_t j=0;j<4*n;j++){input[j].re=spatial[j];report.spatial_energy+=spatial[j]*spatial[j];}
    if(cm_dft2(2*h,2*w,input,actual,0))goto fail;
    for(size_t j=0;j<4*n;j++)report.max_spectral_error=fmax(report.max_spectral_error,hypot(actual[j].re-predicted[j].re,actual[j].im-predicted[j].im));
    for(size_t j=0;j<n;j++){
        double mean=0;for(size_t p=0;p<4;p++)mean+=planes[p*n+j].re/4;
        for(size_t p=0;p<4;p++)report.phase_average_loss_l2=hypot(report.phase_average_loss_l2,planes[p*n+j].re-mean);
    }
    if(!isfinite(report.spatial_energy)||!isfinite(report.max_spectral_error))goto fail;
    *out=report;free(spatial);free(buf);return 0;
fail:free(spatial);free(buf);return -1;
}
