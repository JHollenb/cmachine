#include "circuit_math.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int finite_vector(const double *p,size_t n){
    if(!p||!n)return 0;
    for(size_t i=0;i<n;i++)if(!isfinite(p[i]))return 0;
    return 1;
}
static int extent(size_t a,size_t b){return a&&b&&a<=SIZE_MAX/b/sizeof(double);}
static double sigmoid(double x){if(x>=0)return 1/(1+exp(-x));double e=exp(x);return e/(1+e);}
int cm_read_geometry(size_t n,const double *g,const double *u,CmReadGeometry *out){
    if(!out||!finite_vector(g,n)||!finite_vector(u,n))return -1;
    CmReadGeometry r={0};
    for(size_t i=0;i<n;i++){r.gate_norm=hypot(r.gate_norm,g[i]);r.up_norm=hypot(r.up_norm,u[i]);r.dot+=g[i]*u[i];}
    r.eigen_plus=.5*(r.dot+r.gate_norm*r.up_norm);r.eigen_minus=.5*(r.dot-r.gate_norm*r.up_norm);
    double check[]={r.gate_norm,r.up_norm,r.dot,r.eigen_plus,r.eigen_minus};if(!finite_vector(check,5))return -1;
    *out=r;return 0;
}
int cm_neuron_even_odd(double gate,double up,double *even,double *odd){
    if(!even||!odd||!isfinite(gate)||!isfinite(up))return -1;
    double e=.5*gate*up,o=e*tanh(.5*gate);if(!isfinite(e)||!isfinite(o))return -1;
    *even=e;*odd=o;return 0;
}
double cm_silu(double x){return x*sigmoid(x);}
double cm_silu_first(double x){double p=sigmoid(x);return p+x*p*(1-p);}
double cm_silu_second(double x){double p=sigmoid(x);return p*(1-p)*(2+x*(1-2*p));}
double cm_silu_secant(double a,double b){
    if(!isfinite(a)||!isfinite(b))return NAN;
    if(a==b)return cm_silu_first(a);
    /* expm1 avoids cancellation in sigmoid differences for close endpoints.
     * For separated endpoints the bounded direct difference is well behaved. */
    double delta=b-a;
    if(fabs(delta)<.5){
        double p=sigmoid(a),q=sigmoid(b);
        double dp=delta>0?p*(1-q)*expm1(delta):-q*(1-p)*expm1(-delta);
        return .5*(p+q)+.5*(a+b)*(dp/delta);
    }
    return (cm_silu(b)-cm_silu(a))/delta;
}
int cm_neuron_delta(double g0,double u0,double f0,double g1,double u1,double f1,CmNeuronDelta *out){
    double args[]={g0,u0,f0,g1,u1,f1};if(!out||!finite_vector(args,6))return -1;
    double s0=cm_silu(g0),s1=cm_silu(g1),ds=s1-s0,du=u1-u0;
    CmNeuronDelta r={u0*ds,s0*du,ds*du,(f1-s1*u1)-(f0-s0*u0),f1-f0,cm_silu_secant(g0,g1)};
    double check[]={r.gate,r.up,r.interaction,r.remainder,r.total,r.secant};
    if(!finite_vector(check,6))return -1;
    *out=r;return 0;
}
int cm_neuron_curvature(size_t n,const double *g,const double *u,const double *x,const double *a,const double *b,double *out){
    if(!out||!finite_vector(g,n)||!finite_vector(u,n)||!finite_vector(x,n)||!finite_vector(a,n)||!finite_vector(b,n))return -1;
    double gx=0,ux=0,ga=0,gb=0,ua=0,ub=0;
    for(size_t i=0;i<n;i++){gx+=g[i]*x[i];ux+=u[i]*x[i];ga+=g[i]*a[i];gb+=g[i]*b[i];ua+=u[i]*a[i];ub+=u[i]*b[i];}
    double v=ux*cm_silu_second(gx)*ga*gb+cm_silu_first(gx)*(ga*ub+gb*ua);
    if(!isfinite(v))return -1;
    *out=v;return 0;
}
static double scale(size_t n,const double *x,double eps){
    if(!n||!isfinite(eps)||eps<=0||!finite_vector(x,n))return NAN;
    double q=eps;for(size_t i=0;i<n;i++)q+=x[i]*x[i]/n;
    return isfinite(q)?1/sqrt(q):NAN;
}
static int rms(size_t n,const double *x,const double *gamma,double eps,const double *v,double *out,int adjoint){
    double rho=scale(n,x,eps);if(!out||!isfinite(rho)||!finite_vector(gamma,n)||!finite_vector(v,n))return -1;
    double dot=0;for(size_t i=0;i<n;i++)dot+=x[i]*v[i]*(adjoint?gamma[i]:1);
    if(!isfinite(dot))return -1;
    for(size_t i=0;i<n;i++)out[i]=rho*(adjoint?gamma[i]*v[i]-x[i]*rho*rho*dot/n:gamma[i]*(v[i]-x[i]*rho*rho*dot/n));
    return finite_vector(out,n)?0:-1;
}
int cm_rms_jvp(size_t n,const double *x,const double *g,double e,const double *v,double *o){return rms(n,x,g,e,v,o,0);}
int cm_rms_vjp(size_t n,const double *x,const double *g,double e,const double *v,double *o){return rms(n,x,g,e,v,o,1);}
int cm_rms_finite(size_t n,const double *a,const double *b,const double *gamma,double eps,double *out){
    double r0=scale(n,a,eps),r1=scale(n,b,eps);if(!out||!isfinite(r0)||!isfinite(r1)||!finite_vector(gamma,n))return -1;
    double dq=0;for(size_t i=0;i<n;i++)dq+=(b[i]-a[i])*(b[i]+a[i])/n;
    double dr=-dq*r0*r1/(1/r0+1/r1);
    for(size_t i=0;i<n;i++)out[i]=gamma[i]*(r1*(b[i]-a[i])+dr*a[i]);
    return finite_vector(out,n)?0:-1;
}
int cm_writer_reader(size_t n,const double *w,const double *v,const double *r,const double *g,double eps,double *fixed,double *radial,double *total){
    double rho=scale(n,r,eps);if(!fixed||!radial||!total||!isfinite(rho)||!finite_vector(w,n)||!finite_vector(v,n)||!finite_vector(g,n))return -1;
    double a=0,b=0,c=0;for(size_t i=0;i<n;i++){a+=w[i]*g[i]*v[i];b+=w[i]*g[i]*r[i];c+=r[i]*v[i];}
    double correction=-rho*rho*rho*b*c/n,value=rho*a+correction;
    if(!isfinite(a)||!isfinite(correction)||!isfinite(value))return -1;
    *fixed=a;*radial=correction;*total=value;return 0;
}
int cm_rms_second(size_t n,const double *x,const double *gamma,double eps,const double *a,const double *b,double *out){
    double rho=scale(n,x,eps);if(!out||!isfinite(rho)||!finite_vector(gamma,n)||!finite_vector(a,n)||!finite_vector(b,n))return -1;
    double xa=0,xb=0,ab=0;for(size_t i=0;i<n;i++){xa+=x[i]*a[i];xb+=x[i]*b[i];ab+=a[i]*b[i];}
    for(size_t i=0;i<n;i++)out[i]=gamma[i]*(-rho*rho*rho/n*(xb*a[i]+xa*b[i]+ab*x[i])+3*rho*rho*rho*rho*rho/(n*(double)n)*xa*xb*x[i]);
    return finite_vector(out,n)?0:-1;
}
static int probabilities(size_t n,const double *p){
    if(!finite_vector(p,n))return 0;
    double sum=0;for(size_t i=0;i<n;i++){if(p[i]<0)return 0;sum+=p[i];}
    /* Input contract, not scientific gate: native FP32 normalized rows round. */
    return fabs(sum-1)<=1e-5;
}
int cm_attention_vjp(size_t n,size_t d,const double *p,const double *q,const double *k,const double *v,const double *a,const double *lambda,double *dq,double *dk,double *dv,double *dl){
    if(!extent(n,d)||!dq||!dk||!dv||!dl||!probabilities(n,p)||!finite_vector(q,d)||!finite_vector(k,n*d)||!finite_vector(v,n*d)||!finite_vector(a,d)||!finite_vector(lambda,d))return -1;
    memset(dq,0,d*sizeof(double));double inv=1/sqrt((double)d);
    for(size_t j=0;j<n;j++){
        double dot=0;for(size_t i=0;i<d;i++)dot+=(v[j*d+i]-a[i])*lambda[i];dl[j]=p[j]*dot;
        for(size_t i=0;i<d;i++){dq[i]+=dl[j]*k[j*d+i]*inv;dk[j*d+i]=dl[j]*q[i]*inv;dv[j*d+i]=p[j]*lambda[i];}
    }
    return finite_vector(dq,d)&&finite_vector(dk,n*d)&&finite_vector(dv,n*d)&&finite_vector(dl,n)?0:-1;
}
int cm_attention_query_second(size_t n,size_t d,const double *p,const double *k,const double *v,const double *a,const double *direction,double *out){
    if(!extent(n,d)||!out||!probabilities(n,p)||!finite_vector(k,n*d)||!finite_vector(v,n*d)||!finite_vector(a,d)||!finite_vector(direction,d))return -1;
    double mean=0;for(size_t j=0;j<n;j++)for(size_t i=0;i<d;i++)mean+=p[j]*k[j*d+i]*direction[i]/sqrt((double)d);
    memset(out,0,d*sizeof(double));
    for(size_t j=0;j<n;j++){
        double s=-mean;for(size_t i=0;i<d;i++)s+=k[j*d+i]*direction[i]/sqrt((double)d);
        for(size_t i=0;i<d;i++)out[i]+=p[j]*(v[j*d+i]-a[i])*s*s;
    }
    return finite_vector(out,d)?0:-1;
}
int cm_attention_finite(size_t n,size_t d,const double *p,const double *v,const double *dl,const double *dv,double *out){
    if(!extent(n,d)||!out||!probabilities(n,p)||!finite_vector(v,n*d)||!finite_vector(dl,n)||!finite_vector(dv,n*d))return -1;
    double max=-INFINITY;for(size_t j=0;j<n;j++)if(p[j]>0){double x=log(p[j])+dl[j];if(x>max)max=x;}
    if(!isfinite(max))return -1;
    double den=0;for(size_t j=0;j<n;j++)if(p[j]>0)den+=exp(log(p[j])+dl[j]-max);
    for(size_t i=0;i<d;i++){
        double value=0;for(size_t j=0;j<n;j++){
            double p1=p[j]>0?exp(log(p[j])+dl[j]-max)/den:0;
            value+=(p1-p[j])*v[j*d+i]+p1*dv[j*d+i];
        }out[i]=value;
    }return finite_vector(out,d)?0:-1;
}
static int complex_vector(const CmComplex *v,size_t n){
    if(!v||!n||n>4096)return 0;
    for(size_t i=0;i<n;i++)if(!isfinite(v[i].re)||!isfinite(v[i].im))return 0;
    return 1;
}
int cm_dft(size_t n,const CmComplex *in,CmComplex *out,int inverse){
    if(!out||in==out||!complex_vector(in,n)||(inverse!=0&&inverse!=1))return -1;
    double tau=2*acos(-1.0);
    for(size_t k=0;k<n;k++){
        double re=0,im=0;for(size_t j=0;j<n;j++){
            double angle=(inverse?1:-1)*tau*k*j/n,c=cos(angle),s=sin(angle);
            re+=in[j].re*c-in[j].im*s;im+=in[j].re*s+in[j].im*c;
        }out[k].re=re/(inverse?n:1);out[k].im=im/(inverse?n:1);
    }return complex_vector(out,n)?0:-1;
}
int cm_circular_convolution(size_t n,const CmComplex *a,const CmComplex *b,CmComplex *out){
    if(!out||out==a||out==b||!complex_vector(a,n)||!complex_vector(b,n))return -1;
    for(size_t k=0;k<n;k++){
        double re=0,im=0;for(size_t j=0;j<n;j++){size_t l=(k+n-j)%n;re+=a[j].re*b[l].re-a[j].im*b[l].im;im+=a[j].re*b[l].im+a[j].im*b[l].re;}
        out[k]=(CmComplex){re,im};
    }return complex_vector(out,n)?0:-1;
}
int cm_shift_commutator(size_t n,const double *j,double *out){
    if(!out||!extent(n,n)||n>4096||!finite_vector(j,n*n))return -1;
    double sum=0;for(size_t t=0;t<n;t++)for(size_t s=0;s<n;s++){
        double v=j[t*n+(s+1)%n]-j[((t+n-1)%n)*n+s];sum+=v*v;
    }if(!isfinite(sum))return -1;
    *out=sum;return 0;
}
int cm_coalitions(unsigned bits,double *values,int inverse){
    if(bits>12||(inverse!=0&&inverse!=1)||!finite_vector(values,(size_t)1<<bits))return -1;
    size_t n=(size_t)1<<bits;
    for(unsigned b=0;b<bits;b++)for(size_t s=0;s<n;s++)if(s&((size_t)1<<b))values[s]+=(inverse?1:-1)*values[s^((size_t)1<<b)];
    return finite_vector(values,n)?0:-1;
}
static int interface_valid(const CmInterfaceRow *r){return r&&r->scope&&*r->scope&&r->occurrence&&*r->occurrence&&r->arguments&&r->argument_bytes&&r->result&&r->result_bytes;}
int cm_interface_compare(const CmInterfaceRow *a,const CmInterfaceRow *b,CmInterfaceRelation *out){
    if(!out||!interface_valid(a)||!interface_valid(b))return -1;
    if(strcmp(a->scope,b->scope))*out=CM_INTERFACE_OTHER_SCOPE;
    else if(a->argument_bytes!=b->argument_bytes||memcmp(a->arguments,b->arguments,a->argument_bytes))*out=CM_INTERFACE_OTHER_ARGUMENTS;
    else if(a->result_bytes!=b->result_bytes||memcmp(a->result,b->result,a->result_bytes))*out=CM_INTERFACE_WITNESS;
    else *out=CM_INTERFACE_AGREES;
    return 0;
}
