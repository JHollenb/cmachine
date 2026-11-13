#include "machine.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static const float *param(const TcmModel *m,const char *name,int rank,int d0,int d1,int d2) {
    const TcmTensor *t=tcm_tensor(m,name);
    if(!t||t->dtype!=TCM_F32||t->rank!=(unsigned)rank)return NULL;
    int dims[3]={d0,d1,d2};for(int i=0;i<rank;i++)if(t->dim[i]!=(uint64_t)dims[i])return NULL;
    return t->data;
}
static int bind_linear(IsaLinear *l,const TcmModel *m,const char *name,int in,int out,int bias) {
    char key[64];snprintf(key,sizeof(key),"%s.weight",name);l->w=param(m,key,2,out,in,0);
    snprintf(key,sizeof(key),"%s.bias",name);l->b=bias?param(m,key,1,out,0,0):NULL;l->in=in;l->out=out;
    return !l->w||(bias&&!l->b)?-1:0;
}
int isa_model_init(IsaModel *m,const TcmModel *b,char *err,size_t cap) {
    memset(m,0,sizeof(*m));m->bundle=b;
    if(b->architecture!=TCM_ISA||b->config[0]!=48||b->config[1]!=3||b->count!=87){snprintf(err,cap,"unsupported ISA profile or tensor count");return -1;}
    memcpy(&m->post_epsilon,&b->config[2],4);
    if(!isfinite(m->post_epsilon)||m->post_epsilon<=0){snprintf(err,cap,"invalid post LayerNorm epsilon");return -1;}
    for(unsigned i=0;i<b->count;i++){
        const TcmTensor *t=&b->tensor[i];if(t->dtype!=TCM_F32){snprintf(err,cap,"ISA parameters require FP32");return -1;}
        const float *p=t->data;for(uint64_t j=0;j<t->bytes/4;j++)if(!isfinite(p[j])){snprintf(err,cap,"non-finite ISA parameter");return -1;}
    }
    int bad=0;
#define LIN(field,name,in,out) bad |= bind_linear(&m->field,b,name,in,out,1)
    LIN(decoder1,"decoder.0",120,256);LIN(decoder2,"decoder.2",256,256);
    LIN(dest,"dest",256,8);LIN(src,"src",256,8);LIN(query,"query",272,32);
    LIN(offset,"offset_head",256,8);LIN(input,"inp",293,48);LIN(flag_query,"flag_query",256,64);
    LIN(reg,"reg_out",48,8);LIN(imm,"imm_out",48,1);LIN(store,"store_out",48,2);
    LIN(flags1,"flags.0",500,128);LIN(flags2,"flags.2",128,4);
    LIN(width,"width_head",256,3);LIN(length,"length",256,15);LIN(syscall,"syscall",256,1);LIN(store_flag,"store_flag",256,1);
#undef LIN
    m->pos=param(b,"pos",2,64,16,0);m->ins_pos=param(b,"ins_pos",2,122,16,0);
    bad|=!m->pos||!m->ins_pos;
    for(unsigned i=0;i<3;i++){
        IsaMixer *x=&m->mixer[i];char key[64];
#define MIXLIN(field,suffix,in,out,bias) snprintf(key,sizeof(key),"mixers.%u." suffix,i);bad|=bind_linear(&x->field,b,key,in,out,bias)
        MIXLIN(input,"in_proj",48,192,1);MIXLIN(select,"select",96,9,0);MIXLIN(dt,"dt_proj",1,96,1);MIXLIN(output,"out_proj",96,48,1);
#undef MIXLIN
        snprintf(key,sizeof(key),"post.%u.1",i);bad|=bind_linear(&x->post1,b,key,48,96,1);
        snprintf(key,sizeof(key),"post.%u.3",i);bad|=bind_linear(&x->post2,b,key,96,48,1);
#define P(field,fmt,rank,a,c,d) snprintf(key,sizeof(key),fmt,i);x->field=param(b,key,rank,a,c,d);bad|=!x->field
        P(conv,"mixers.%u.conv.weight",3,96,1,3);P(conv_bias,"mixers.%u.conv.bias",1,96,0,0);
        P(a_log,"mixers.%u.a_log",2,96,4,0);P(d,"mixers.%u.d",1,96,0,0);
        P(gamma,"post.%u.0.weight",1,48,0,0);P(beta,"post.%u.0.bias",1,48,0,0);
#undef P
    }
    if(bad){snprintf(err,cap,"ISA tensor names or dimensions differ from R3 stream program");return -1;}return 0;
}
int isa_law_init(IsaLaw *law,const TcmModel *b,const IsaModel *m,char *err,size_t cap) {
    memset(law,0,sizeof(*law));
    if(b->architecture!=TCM_ISA_LAW||strcmp(b->identity,m->bundle->identity)||b->config[0]!=2||b->config[1]!=8||b->count!=5){snprintf(err,cap,"law model/site binding differs");return -1;}
    law->bundle=b;law->layer=b->config[0];law->row=b->config[1];
    law->x0=param(b,"anchor.input",1,48,0,0);law->p0=param(b,"anchor.branch",1,48,0,0);
    const TcmTensor *a=tcm_tensor(b,"basis.0"),*c=tcm_tensor(b,"basis.1"),*d=tcm_tensor(b,"coefficients");
    if(!law->x0||!law->p0||!a||!c||!d||a->dtype!=TCM_F64||c->dtype!=TCM_F64||d->dtype!=TCM_F64||a->rank!=1||c->rank!=1||d->rank!=2||a->dim[0]!=48||c->dim[0]!=48||d->dim[0]!=5||d->dim[1]!=48){snprintf(err,cap,"invalid quadratic law tensors");return -1;}
    law->h0=a->data;law->h1=c->data;law->coeff=d->data;
    for(unsigned i=0;i<b->count;i++){
        const TcmTensor *t=&b->tensor[i];for(uint64_t j=0;j<t->bytes/(t->dtype==TCM_F64?8:4);j++){
            double v=t->dtype==TCM_F64?((const double*)t->data)[j]:((const float*)t->data)[j];
            if(!isfinite(v)){snprintf(err,cap,"non-finite law parameter");return -1;}
        }
    }
    double aa=0,bb=0,cc=0;for(int i=0;i<48;i++){aa+=law->h0[i]*law->h0[i];bb+=law->h0[i]*law->h1[i];cc+=law->h1[i]*law->h1[i];}
    if(!(aa*cc-bb*bb>0)){snprintf(err,cap,"singular law basis");return -1;}return 0;
}
static void linear(const IsaLinear *l,const float *x,float *y) {
    for(int i=0;i<l->out;i++){float sum=0;const float *w=l->w+(size_t)i*l->in;for(int j=0;j<l->in;j++)sum+=w[j]*x[j];y[i]=sum+(l->b?l->b[i]:0);}
}
static float gelu(float x){return 0.5f*x*(1.0f+erff(x*0.7071067811865475244f));}
static float silu(float x){return x/(1.0f+expf(-x));}
static float softplus(float x){return x>20.0f?x:log1pf(expf(x));}
const char *isa_port_name(IsaPort port){
    static const char *names[]={"mix_input","mix_u","selection","timestep","retention","injection","scan_state","scan_read","output_gate","gated","post_input","post_normalized","post_pre","post_activation","post_write","block_output"};
    return port>=0&&port<ISA_PORT_COUNT?names[port]:NULL;
}
static void isa_observe(IsaSession *s,unsigned layer,IsaPort port,const float *data,size_t count){
    if(s&&s->observer)s->observer(s->observer_context,s->state.row,layer,port,data,count);
}
static void softmax(float *x,unsigned n){float max=x[0],sum=0;for(unsigned i=1;i<n;i++)if(x[i]>max)max=x[i];for(unsigned i=0;i<n;i++){x[i]=expf(x[i]-max);sum+=x[i];}for(unsigned i=0;i<n;i++)x[i]/=sum;}
unsigned isa_argmax(const float *x,unsigned n){unsigned k=0;for(unsigned i=1;i<n;i++)if(x[i]>x[k])k=i;return k;}
uint64_t isa_bits(const float *x){uint64_t v=0;for(unsigned i=0;i<64;i++)if(x[i]>0)v|=UINT64_C(1)<<i;return v;}
void isa_session_init(IsaSession *s,const IsaModel *m,const IsaArch *a){memset(s,0,sizeof(*s));s->model=m;if(a)s->state.arch=*a;}
int isa_begin(IsaSession *s,const unsigned char *window,size_t n) {
    if(n>15||(s->state.phase!=0&&s->state.phase!=3))return -1;
    IsaArch arch=s->state.arch;uint64_t dense=s->state.dense_post_rows,laws=s->state.law_rows;
    memset(&s->state,0,sizeof(s->state));IsaState *v=&s->state;v->arch=arch;v->dense_post_rows=dense;v->law_rows=laws;
    const IsaModel *m=s->model;float bits[120],temp[256],pd[8],ps[8],fq[64];
    for(unsigned i=0;i<120;i++){v->window[i]=(i/8<n)?(float)((window[i/8]>>(i%8))&1):0;bits[i]=v->window[i]*2-1;}
    for(unsigned i=0;i<8;i++)for(unsigned j=0;j<64;j++)v->before[i][j]=(float)((arch.registers[i]>>j)&1);
    for(unsigned i=0;i<4;i++){if(arch.flags[i]>1)return -1;v->before_flags[i]=(float)arch.flags[i];}
    linear(&m->decoder1,bits,temp);for(unsigned i=0;i<256;i++)temp[i]=gelu(temp[i]);
    linear(&m->decoder2,temp,v->z);for(unsigned i=0;i<256;i++)v->z[i]=gelu(v->z[i]);
    linear(&m->dest,v->z,v->dest);linear(&m->src,v->z,v->src);memcpy(pd,v->dest,sizeof(pd));memcpy(ps,v->src,sizeof(ps));softmax(pd,8);softmax(ps,8);
    for(unsigned i=0;i<64;i++)for(unsigned j=0;j<8;j++){v->a[i]+=pd[j]*v->before[j][i];v->s[i]+=ps[j]*v->before[j][i];}
    linear(&m->offset,v->z,v->offsets);softmax(v->offsets,8);
    linear(&m->flag_query,v->z,fq);
    for(unsigned k=0;k<4;k++){
        for(unsigned i=0;i<64;i++){float sum=0;for(unsigned d=0;d<16;d++)sum+=fq[k*16+d]*m->pos[i*16+d];v->flag_point[k][i]=sum;}softmax(v->flag_point[k],64);
    }
    for(unsigned i=0;i<64;i++){
        float qin[272],q[32],scores[122];memcpy(qin,m->pos+16*i,16*sizeof(float));memcpy(qin+16,v->z,sizeof(v->z));linear(&m->query,qin,q);
        for(unsigned h=0;h<2;h++){
            for(unsigned j=0;j<122;j++){float sum=0;for(unsigned d=0;d<16;d++)sum+=q[h*16+d]*m->ins_pos[j*16+d];scores[j]=sum*4.0f;}softmax(scores,122);
            float sum=scores[121];for(unsigned j=0;j<120;j++)sum+=scores[j]*v->window[j];v->imm[i][h]=sum;
        }
    }
    v->phase=1;return 0;
}
static void mixer(IsaSession *s,float hidden[48],unsigned layer) {
    IsaState *v=&s->state;const IsaMixer *m=&s->model->mixer[layer];float norm[48],p[192],u[96],sel[9],dt[96],gated[96],out[48];
    float retention[96][4],injection[96][4],steps[96],reads[96],gates[96];
    isa_observe(s,layer,ISA_MIX_INPUT,hidden,48);
    float sq=0;for(unsigned i=0;i<48;i++)sq+=hidden[i]*hidden[i];float scale=1.0f/sqrtf(sq/48+1e-5f);
    for(unsigned i=0;i<48;i++)norm[i]=hidden[i]*scale;linear(&m->input,norm,p);
    for(unsigned i=0;i<96;i++){
        float c=v->conv[layer][0][i]*m->conv[i*3]+v->conv[layer][1][i]*m->conv[i*3+1]+p[i]*m->conv[i*3+2]+m->conv_bias[i];
        v->conv[layer][0][i]=v->conv[layer][1][i];v->conv[layer][1][i]=p[i];u[i]=silu(c);
    }
    linear(&m->select,u,sel);linear(&m->dt,sel,dt);
    isa_observe(s,layer,ISA_MIX_U,u,96);isa_observe(s,layer,ISA_SELECTION,sel,9);
    for(unsigned i=0;i<96;i++){
        float d=softplus(dt[i]),read=0;
        for(unsigned j=0;j<4;j++){
            float decay=expf(d*-expf(m->a_log[i*4+j]));float writer=(d*u[i])*sel[1+j];
            if(s->observer){retention[i][j]=decay;injection[i][j]=writer;}
            v->scan[layer][i][j]=decay*v->scan[layer][i][j]+writer;read+=v->scan[layer][i][j]*sel[5+j];
        }
        gated[i]=(read+m->d[i]*u[i])*silu(p[96+i]);
        if(s->observer){steps[i]=d;reads[i]=read;gates[i]=silu(p[96+i]);}
    }
    if(s->observer){
        isa_observe(s,layer,ISA_TIMESTEP,steps,96);isa_observe(s,layer,ISA_RETENTION,&retention[0][0],384);
        isa_observe(s,layer,ISA_INJECTION,&injection[0][0],384);isa_observe(s,layer,ISA_SCAN_STATE,&v->scan[layer][0][0],384);
        isa_observe(s,layer,ISA_SCAN_READ,reads,96);isa_observe(s,layer,ISA_OUTPUT_GATE,gates,96);isa_observe(s,layer,ISA_GATED,gated,96);
    }
    linear(&m->output,gated,out);for(unsigned i=0;i<48;i++)hidden[i]+=out[i];
}
static void post(const IsaModel *model,unsigned layer,const float x[48],float branch[48],IsaSession *observer) {
    const IsaMixer *m=&model->mixer[layer];float mean=0,var=0,norm[48],ff[96];
    for(unsigned i=0;i<48;i++)mean+=x[i];mean/=48;
    for(unsigned i=0;i<48;i++){float c=x[i]-mean;var+=c*c;}
    float inv=1.0f/sqrtf(var/48+model->post_epsilon);
    for(unsigned i=0;i<48;i++)norm[i]=(x[i]-mean)*inv*m->gamma[i]+m->beta[i];
    isa_observe(observer,layer,ISA_POST_INPUT,x,48);isa_observe(observer,layer,ISA_POST_NORMALIZED,norm,48);
    linear(&m->post1,norm,ff);isa_observe(observer,layer,ISA_POST_PRE,ff,96);
    for(unsigned i=0;i<96;i++)ff[i]=gelu(ff[i]);isa_observe(observer,layer,ISA_POST_ACTIVATION,ff,96);
    linear(&m->post2,ff,branch);isa_observe(observer,layer,ISA_POST_WRITE,branch,48);
}
static double dot(const double *a,const double *b){double sum=0,c=0;for(unsigned i=0;i<48;i++){double y=a[i]*b[i]-c,t=sum+y;c=(t-sum)-y;sum=t;}return sum;}
static void law_post(IsaSession *s,float hidden[48],unsigned layer) {
    IsaState *v=&s->state;const IsaLaw *law=s->law;double delta[48];float branch[48];
    for(unsigned i=0;i<48;i++){float step=(float)(s->policy.s*law->h0[i]+s->policy.t*law->h1[i]);hidden[i]=law->x0[i]+step;delta[i]=(double)hidden[i]-law->x0[i];v->law_input[i]=hidden[i];}
    double a=dot(law->h0,law->h0),b=dot(law->h0,law->h1),c=dot(law->h1,law->h1),d=dot(law->h0,delta),e=dot(law->h1,delta),det=a*c-b*b;
    double u=(c*d-b*e)/det,t=(a*e-b*d)/det,terms[5]={u,t,u*u,t*t,u*t};v->law_coordinates[0]=u;v->law_coordinates[1]=t;v->off_plane_norm=0;
    for(unsigned i=0;i<48;i++){double r=delta[i]-u*law->h0[i]-t*law->h1[i];v->off_plane_norm+=r*r;}
    v->off_plane_norm=sqrt(v->off_plane_norm);
    if(s->policy.quadratic){
        for(unsigned i=0;i<48;i++){double sum=0;for(unsigned k=0;k<5;k++)sum+=terms[k]*law->coeff[k*48+i];branch[i]=(float)((double)law->p0[i]+sum);}v->law_rows++;
    }else{post(s->model,layer,hidden,branch,s);v->dense_post_rows++;}
    for(unsigned i=0;i<48;i++){v->law_branch[i]=branch[i];hidden[i]+=branch[i];v->law_hidden[i]=hidden[i];}
}
int isa_advance(IsaSession *s) {
    IsaState *v=&s->state;const IsaModel *m=s->model;if(v->phase!=1||v->row>=64)return -1;
    if(s->policy.enabled&&(!s->law||!isfinite(s->policy.s)||!isfinite(s->policy.t)))return -1;
    unsigned r=v->row;float input[293],hidden[48];unsigned n=0;
    for(unsigned j=0;j<8;j++)input[n++]=v->before[j][r];input[n++]=v->a[r];input[n++]=v->s[r];input[n++]=v->imm[r][0];input[n++]=v->imm[r][1];
    memcpy(input+n,m->pos+16*r,16*sizeof(float));n+=16;memcpy(input+n,v->z,sizeof(v->z));n+=256;
    float aligned=0;for(unsigned j=0;j<8;j++){float x=v->window[r+8*j];input[n++]=x*2-1;aligned+=v->offsets[j]*x;}input[n++]=aligned*2-1;
    if(n!=293)return -1;linear(&m->input,input,hidden);
    for(unsigned l=0;l<3;l++){
        mixer(s,hidden,l);
        if(s->policy.enabled&&v->arch.instruction==s->policy.instruction&&l==s->law->layer&&r==s->law->row)law_post(s,hidden,l);
        else{float branch[48];post(m,l,hidden,branch,s);for(unsigned i=0;i<48;i++)hidden[i]+=branch[i];v->dense_post_rows++;}
        isa_observe(s,l,ISA_BLOCK_OUTPUT,hidden,48);
    }
    float reg[8],store[2];linear(&m->reg,hidden,reg);linear(&m->imm,hidden,&v->output.imm[r]);linear(&m->store,hidden,store);
    for(unsigned i=0;i<8;i++)v->output.regs[i][r]=v->before[i][r]*8-4+reg[i];
    v->output.store_addr[r]=store[0];v->output.store_value[r]=store[1];
    for(unsigned i=0;i<48;i++){if(!isfinite(hidden[i]))return -1;v->hidden_sum[i]+=hidden[i];for(unsigned k=0;k<4;k++)v->flag_reads[k][i]+=v->flag_point[k][r]*hidden[i];}
    v->row++;
    if(v->row==64){
        float features[500],ff[128];memcpy(features,v->z,sizeof(v->z));memcpy(features+256,v->before_flags,sizeof(v->before_flags));memcpy(features+260,v->flag_reads,sizeof(v->flag_reads));
        for(unsigned i=0;i<48;i++)features[452+i]=v->hidden_sum[i]/64;
        linear(&m->flags1,features,ff);for(unsigned i=0;i<128;i++)ff[i]=gelu(ff[i]);linear(&m->flags2,ff,v->output.flags);
        for(unsigned i=0;i<4;i++)v->output.flags[i]+=v->before_flags[i]*8-4;
        memcpy(v->output.dest,v->dest,sizeof(v->dest));memcpy(v->output.src,v->src,sizeof(v->src));
        linear(&m->width,v->z,v->output.width);linear(&m->length,v->z,v->output.length);linear(&m->syscall,v->z,&v->output.syscall);linear(&m->store_flag,v->z,&v->output.store);v->phase=2;
    }
    return 0;
}
int isa_commit(IsaSession *s) {
    IsaState *v=&s->state;if(v->phase!=2)return -1;
    const float *values=(const float*)&v->output;
    for(size_t i=0;i<sizeof(v->output)/sizeof(float);i++)if(!isfinite(values[i]))return -1;
    unsigned length=isa_argmax(v->output.length,15)+1;if(v->arch.rip>UINT64_MAX-length||v->arch.instruction==UINT64_MAX)return -1;
    for(unsigned i=0;i<8;i++)v->arch.registers[i]=isa_bits(v->output.regs[i]);for(unsigned i=0;i<4;i++)v->arch.flags[i]=v->output.flags[i]>0;
    v->arch.rip+=length;v->arch.instruction++;v->phase=3;return 0;
}
void isa_checkpoint(const IsaSession *s,IsaCheckpoint *c){memset(c,0,sizeof(*c));memcpy(c->model_digest,s->model->bundle->digest,65);if(s->law)memcpy(c->law_digest,s->law->bundle->digest,65);c->policy=s->policy;c->state=s->state;}
int isa_restore(IsaSession *s,const IsaCheckpoint *c){if(strcmp(c->model_digest,s->model->bundle->digest)||strcmp(c->law_digest,s->law?s->law->bundle->digest:""))return -1;if(c->state.row>64||c->state.phase>3||(c->state.phase==1&&c->state.row>=64)||(c->state.phase>=2&&c->state.row!=64)||(c->state.phase==0&&c->state.row!=0))return -1;s->policy=c->policy;s->state=c->state;return 0;}
