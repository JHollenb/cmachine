#include "isa_circuit.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double *slot(CmIsaRow *r,IsaPort p,size_t *n){
    switch(p){
#define PORT(tag,field,count) case tag:*n=count;return r->field
    PORT(ISA_MIX_INPUT,input,48);PORT(ISA_MIX_U,u,96);PORT(ISA_SELECTION,selection,9);PORT(ISA_TIMESTEP,dt,96);
    PORT(ISA_RETENTION,retention,384);PORT(ISA_INJECTION,injection,384);PORT(ISA_SCAN_STATE,state,384);
    PORT(ISA_SCAN_READ,read,96);PORT(ISA_OUTPUT_GATE,gate,96);PORT(ISA_GATED,gated,96);
    PORT(ISA_POST_INPUT,post_input,48);PORT(ISA_POST_NORMALIZED,normalized,48);PORT(ISA_POST_PRE,pre,96);
    PORT(ISA_POST_ACTIVATION,activation,96);PORT(ISA_POST_WRITE,write,48);PORT(ISA_BLOCK_OUTPUT,output,48);
#undef PORT
    default:return NULL;
    }
}
static void capture(void *ctx,unsigned row,unsigned layer,IsaPort port,const float *values,size_t count){
    CmIsaCapture *c=ctx;if(c->error||layer!=c->layer)return;
    if(row>=64){c->error=1;return;}
    size_t n=0;double *dst=slot(c->rows+row,port,&n);
    if(!dst||n!=count||(c->rows[row].seen&(1u<<port))){c->error=1;return;}
    for(size_t i=0;i<n;i++){dst[i]=values[i];if(!isfinite(dst[i]))c->error=1;}
    c->rows[row].seen|=1u<<port;
}
static int live(const CmIsaCapture *c){
    if(!(c&&c->ready&&c->session&&c->session->model==c->model&&c->session->state.phase==2&&c->session->state.row==64&&
        !memcmp(&c->session->state.arch,&c->root_arch,sizeof(IsaArch))&&!c->session->policy.enabled))return 0;
    unsigned char bytes[32];char digest[65];tcm_sha256(&c->session->state,sizeof(IsaState),bytes);tcm_hex(bytes,32,digest);
    return !strcmp(digest,c->result_digest);
}
void cm_isa_capture_free(CmIsaCapture *c){if(c){free(c->rows);memset(c,0,sizeof(*c));}}
int cm_isa_capture(CmIsaCapture *c,IsaSession *s,const unsigned char *window,size_t n,unsigned layer){
    if(!c||c->rows||!s||!s->model||!s->model->bundle||s->observer||s->policy.enabled||layer>=3||n>15||(n&&!window)||(s->state.phase!=0&&s->state.phase!=3))return -1;
    c->rows=calloc(64,sizeof(*c->rows));if(!c->rows)return -1;
    c->model=s->model;c->session=s;c->layer=layer;c->root_arch=s->state.arch;c->window_length=n;if(n)memcpy(c->window,window,n);
    isa_checkpoint(s,&c->parent);void *old=s->observer_context;s->observer=capture;s->observer_context=c;
    int error=isa_begin(s,window,n);
    while(!error&&!c->error&&s->state.phase==1)error=isa_advance(s);
    s->observer=NULL;s->observer_context=old;
    unsigned required=(1u<<ISA_PORT_COUNT)-1;
    for(size_t i=0;i<64;i++)if(c->rows[i].seen!=required)c->error=1;
    if(error||c->error||s->state.phase!=2){(void)isa_restore(s,&c->parent);cm_isa_capture_free(c);return -2;}
    unsigned char bytes[32];tcm_sha256(&s->state,sizeof(IsaState),bytes);tcm_hex(bytes,32,c->result_digest);
    c->ready=1;return 0;
}
static int scan_operands(const CmIsaCapture *c,unsigned channel,double *r,double *b,double *read,double *dt){
    if(!live(c)||channel>=96)return -1;
    for(size_t t=0;t<64;t++){
        dt[t]=c->rows[t].dt[channel];
        for(size_t j=0;j<4;j++){r[t*4+j]=c->rows[t].retention[channel*4+j];b[t*4+j]=c->rows[t].selection[1+j];read[t*4+j]=c->rows[t].selection[5+j];}
    }return 0;
}
int cm_isa_scan_kernel(const CmIsaCapture *c,unsigned channel,double *out){
    double r[256],b[256],read[256],dt[64];if(scan_operands(c,channel,r,b,read,dt))return -1;
    return cm_ssm_kernel(64,4,r,b,read,dt,out);
}
int cm_isa_scan_cut(const CmIsaCapture *c,unsigned channel,size_t cut,double *left,double *entry){
    double r[256],b[256],read[256],dt[64];if(scan_operands(c,channel,r,b,read,dt))return -1;
    return cm_ssm_cut(64,4,cut,r,b,read,dt,left,entry);
}
int cm_isa_scan_path(const CmIsaCapture *c,unsigned channel,size_t source,size_t target,double *out){
    double r[256],b[256],read[256],dt[64],injection[256];if(scan_operands(c,channel,r,b,read,dt))return -1;
    for(size_t t=0;t<64;t++)for(size_t j=0;j<4;j++)injection[t*4+j]=c->rows[t].injection[channel*4+j];
    return cm_ssm_path(64,4,r,injection,read,source,target,out);
}
int cm_isa_gelu_write(const CmIsaCapture *c,size_t row,const double *consumer,double *out,double *projection){
    if(!live(c)||row>=64||!out||!projection)return -1;
    const CmIsaRow *p=c->rows+row;const IsaMixer *m=c->model->mixer+c->layer;
    memset(out,0,144*sizeof(double));double sums[3]={0};
    for(size_t i=0;i<48;i++){
        double affine=m->post2.b?m->post2.b[i]:0,nonlinear=0;
        for(size_t j=0;j<96;j++){
            double a=m->post1.b?m->post1.b[j]:0;
            for(size_t k=0;k<48;k++)a+=(double)m->post1.w[j*48+k]*p->normalized[k];
            affine+=.5*m->post2.w[i*96+j]*a;
            nonlinear+=.5*m->post2.w[i*96+j]*a*erf(a/sqrt(2.0));
        }
        out[i]=affine;out[48+i]=nonlinear;out[96+i]=p->write[i]-affine-nonlinear;
        double w=consumer?consumer[i]:1;
        if(!isfinite(w)||!isfinite(out[96+i]))return -2;
        for(size_t arm=0;arm<3;arm++)sums[arm]+=w*out[arm*48+i];
    }for(size_t arm=0;arm<3;arm++)projection[arm]=sums[arm];return 0;
}
int cm_isa_state_effect(const CmIsaCapture *c,size_t cut,unsigned channel,unsigned coordinate,float dose,unsigned reg,unsigned bit,double *effect){
    if(!effect||!live(c)||cut<1||cut>=64||channel>=96||coordinate>=4||reg>=8||bit>=64||!isfinite(dose)||c->session->observer)return -1;
    IsaSession *s=(IsaSession *)c->session;IsaCheckpoint saved;isa_checkpoint(s,&saved);
    /* Restore exact root and regenerate the requested row-boundary cut. */
    int error=isa_restore(s,&c->parent);
    if(!error)error=isa_begin(s,c->window,c->window_length);
    while(!error&&s->state.row<cut)error=isa_advance(s);
    if(!error){float changed=s->state.scan[c->layer][channel][coordinate]+dose;if(!isfinite(changed))error=-1;else s->state.scan[c->layer][channel][coordinate]=changed;}
    while(!error&&s->state.phase==1)error=isa_advance(s);
    double value=!error?(double)s->state.output.regs[reg][bit]-saved.state.output.regs[reg][bit]:0;
    if(isa_restore(s,&saved))return -2;
    if(error||!isfinite(value))return -2;
    *effect=value;return 0;
}
int cm_isa_register_account(const CmIsaCapture *c,size_t row,unsigned reg,CmIsaRegisterAccount *out){
    if(!out||!live(c)||c->layer!=2||row>=64||reg>=8)return -1;
    double lambda[48],terms[144],projection[3];
    for(size_t i=0;i<48;i++)lambda[i]=c->model->reg.w[reg*48+i];
    if(cm_isa_gelu_write(c,row,lambda,terms,projection))return -2;
    CmIsaRegisterAccount a={0};a.copy_prior=c->session->state.before[reg][row]*8-4;
    a.head_bias=c->model->reg.b?c->model->reg.b[reg]:0;
    for(size_t i=0;i<48;i++)a.carry+=lambda[i]*c->rows[row].post_input[i];
    a.gelu_affine=projection[0];a.gelu_nonlinear=projection[1];a.mlp_remainder=projection[2];
    a.native_logit=c->session->state.output.regs[reg][row];
    a.readout_remainder=a.native_logit-a.copy_prior-a.head_bias-a.carry-a.gelu_affine-a.gelu_nonlinear-a.mlp_remainder;
    if(!isfinite(a.readout_remainder))return -2;
    *out=a;return 0;
}
int cm_isa_report(FILE *f,const CmIsaCapture *c,unsigned channel,size_t source,size_t target){
    if(!f||!live(c)||channel>=96||source>target||target>=64)return -1;
    double paths[4];if(cm_isa_scan_path(c,channel,source,target,paths))return -1;
    for(int i=0;i<64;i++){char x=c->model->bundle->digest[i];if(!((x>='a'&&x<='f')||(x>='0'&&x<='9')))return -1;}
    fprintf(f,"{\"schema\":\"tcm-isa-scan-inspection-v1\",\"model_digest\":\"%.64s\",\"instruction\":%llu,\"scan_clock\":\"bit-position-reset-per-instruction\",\"layer\":%u,\"channel\":%u,\"source_bit\":%zu,\"receiver_bit\":%zu,\"native_read\":%.17g,\"signed_state_paths\":[%.17g,%.17g,%.17g,%.17g],\"kernel_scope\":\"fixed-producers-scan-read-before-skip-gate-projection\",\"cross_cut_rank_bound_per_channel\":4,\"convolution_tail_per_channel\":2,\"producer_arrays\":[\"u\",\"dt\",\"selection-B-C\",\"retention\",\"injection\",\"state\",\"read\",\"gate\"],\"parameter_refs\":[\"mixers.%u.a_log\",\"mixers.%u.select.weight\",\"mixers.%u.dt_proj.weight\"],\"terminal_status\":\"no-semantic-certification\"}\n",c->model->bundle->digest,(unsigned long long)c->root_arch.instruction,c->layer,channel,source,target,c->rows[target].read[channel],paths[0],paths[1],paths[2],paths[3],c->layer,c->layer,c->layer);
    return ferror(f)?-1:0;
}
