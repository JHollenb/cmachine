#define _POSIX_C_SOURCE 200809L
#include "machine.h"
#include "program.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#ifdef TCM_USE_CBLAS
#include <cblas.h>
#ifdef TCM_SCIPY_OPENBLAS
#define cblas_sgemv scipy_cblas_sgemv
#endif
#endif

typedef struct { size_t off,n; } Value;
typedef struct { int code,out,a,b,c,d,w,bias; size_t offset,arg; float scale,shift; } Op;
typedef struct {
    TcmModel model; Value *values; Op *ops;
    size_t nv,no,na,ns,output,input_width,vocab;
    int tokens; float *arena,*state;
} Program;
static int error(const char *s){fprintf(stderr,"program refused: %s\n",s);return -1;}
static int integer(double x,int low,int high){return isfinite(x)&&x>=low&&x<=high&&floor(x)==x;}
static const TcmTensor *weight(Program *p,int i){return i<0?NULL:&p->model.tensor[i];}
static size_t len(Program *p,int i){return i<0?0:p->values[i].n;}
static int shape(const TcmTensor *w,size_t a,size_t b,size_t c){
    return w&&w->dtype==TCM_F32&&w->rank==(c?3:b?2:1)&&w->dim[0]==a&&(!b||w->dim[1]==b)&&(!c||w->dim[2]==c);
}
static int bias_ok(Program *p,Op *o,size_t n){return o->bias<0||shape(weight(p,o->bias),n,0,0);}
static int load(Program *p,const char *file){
    char err[256]; if(tcm_load(&p->model,file,err,sizeof(err)))return error(err);
    uint32_t *cfg=p->model.config;
    if(p->model.architecture!=TCM_PROGRAM||cfg[0]!=1||!cfg[1]||cfg[1]>100000||!cfg[2]||cfg[2]>100000||!cfg[3]||cfg[3]>100000000||cfg[4]>100000000||cfg[5]!=0||cfg[6]>=cfg[1]||!cfg[7]||cfg[8]>1)return error("invalid graph header");
    p->nv=cfg[1];p->no=cfg[2];p->na=cfg[3];p->ns=cfg[4];p->output=cfg[6];p->input_width=cfg[7];p->tokens=cfg[8];p->vocab=cfg[9];
    const TcmTensor *v=tcm_tensor(&p->model,"__values"),*ops=tcm_tensor(&p->model,"__ops");
    if(!v||v->dtype!=TCM_F64||v->rank!=2||v->dim[0]!=p->nv||v->dim[1]!=2||!ops||ops->dtype!=TCM_F64||ops->rank!=2||ops->dim[0]!=p->no||ops->dim[1]!=12)return error("invalid schedule tables");
    p->values=calloc(p->nv,sizeof(Value));p->ops=calloc(p->no,sizeof(Op));p->arena=calloc(p->na,sizeof(float));p->state=calloc(p->ns?p->ns:1,sizeof(float));
    if(!p->values||!p->ops||!p->arena||!p->state)return error("allocation failed");
    const double *vd=v->data,*od=ops->data;size_t end=0;
    for(size_t i=0;i<p->nv;i++){
        if(!integer(vd[2*i],0,(int)p->na)||!integer(vd[2*i+1],1,(int)p->na)||vd[2*i]!=end||vd[2*i+1]>p->na-end)return error("invalid SSA arena extent");
        p->values[i]=(Value){end,(size_t)vd[2*i+1]};end+=p->values[i].n;
    }
    if(end!=p->na||len(p,0)!=p->input_width||!p->vocab||p->vocab>len(p,(int)p->output)||(p->tokens&&p->input_width!=1))return error("invalid graph IO");
    unsigned char *defined=calloc(p->nv,1),*slots=calloc(p->ns?p->ns:1,1);if(!defined||!slots){free(defined);free(slots);return error("validation allocation failed");}defined[0]=1;
    int valid=1;
    for(size_t i=0;i<p->no&&valid;i++){
        const double *r=od+12*i;Op *o=p->ops+i;
        for(int j=0;j<10;j++)if(!integer(r[j],j==6||j==7?-1:0,j==6||j==7?(int)p->model.count-1:j==8||j==9?100000000:100000))valid=0;
        if(!valid||!isfinite(r[10])||!isfinite(r[11])){valid=0;break;}
        *o=(Op){(int)r[0],(int)r[1],(int)r[2],(int)r[3],(int)r[4],(int)r[5],(int)r[6],(int)r[7],(size_t)r[8],(size_t)r[9],(float)r[10],(float)r[11]};
        if(o->out<=0||(size_t)o->out>=p->nv||defined[o->out]){valid=0;break;}
        int ids[4]={o->a,o->b,o->c,o->d};for(int j=0;j<4;j++)if((size_t)ids[j]>=p->nv||!defined[ids[j]])valid=0;if(!valid)break;
        size_t n=len(p,o->out),a=len(p,o->a),b=len(p,o->b),count=0;const TcmTensor *w=weight(p,o->w);
        switch(o->code){
        case P_CONST: valid=w&&w->dtype==TCM_F32&&w->bytes/4==n;break;
        case P_LINEAR:valid=shape(w,n,a,0)&&bias_ok(p,o,n);break;
        case P_ADD:case P_MUL:case P_DIV:valid=n==(a>b?a:b)&&(a==b||a==1||b==1);break;
        case P_REMAINDER:valid=n==a&&o->scale>0;break;
        case P_COS:case P_AFFINE:case P_SILU:case P_RELU:valid=n==a;break;
        case P_MATCH_ROWS:
            valid=shape(w,n,a,0);
            if(valid){const float *keys=w->data;for(size_t k=0;k<n*a;k++)if(!integer(keys[k],-2,16777216)||keys[k]==-1)valid=0;}
            break;
        case P_CONCAT:valid=n==a+b;break;
        case P_SLICE:valid=o->offset<=a&&n<=a-o->offset;break;
        case P_RMS:valid=n==a&&o->scale>0&&shape(w,n,0,0);break;
        case P_EMBED:valid=a==1&&w&&w->dtype==TCM_F32&&w->rank==2&&w->dim[1]==n;break;
        case P_CONV:valid=n==a&&o->arg>0&&o->arg<=1024&&shape(w,n,1,o->arg)&&bias_ok(p,o,n);count=n*o->arg;break;
        case P_SCAN:valid=n==a&&a==b&&o->arg>0&&o->arg==len(p,o->c)&&o->arg==len(p,o->d)&&shape(w,n,o->arg,0)&&shape(weight(p,o->bias),n,0,0);count=n*o->arg;break;
        default:valid=0;
        }
        if(count){if(o->offset>p->ns||count>p->ns-o->offset)valid=0;else for(size_t j=o->offset;j<o->offset+count;j++){if(slots[j])valid=0;slots[j]=1;}}
        defined[o->out]=1;
    }
    if(!defined[p->output])valid=0;
    for(size_t j=0;j<p->ns;j++)if(!slots[j])valid=0;
    for(size_t j=0;j<p->nv;j++)if(!defined[j])valid=0;
    free(defined);free(slots);return valid?0:error("unsupported opcode, def-use, shape or state contract");
}
static void unload(Program *p){free(p->values);free(p->ops);free(p->arena);free(p->state);tcm_unload(&p->model);}
static float silu(float x){return x/(1.0f+expf(-x));}
static int step(Program *p,const float *input,int zero,FILE *trace){
    for(size_t i=0;i<p->input_width;i++)if(!isfinite(input[i]))return error("non-finite input");
    if(p->tokens&&(!isfinite(input[0])||input[0]<0||input[0]>=p->vocab||floorf(input[0])!=input[0]))return error("input outside semantic token domain");
    memcpy(p->arena,input,p->input_width*sizeof(float));
    for(size_t pc=0;pc<p->no;pc++){
        Op *o=p->ops+pc;float *y=p->arena+p->values[o->out].off;
        const float *a=p->arena+p->values[o->a].off,*b=p->arena+p->values[o->b].off,*c=p->arena+p->values[o->c].off,*d=p->arena+p->values[o->d].off;
        const float *w=o->w<0?NULL:weight(p,o->w)->data,*bias=o->bias<0?NULL:weight(p,o->bias)->data;
        size_t n=len(p,o->out),m=len(p,o->a),nb=len(p,o->b);
        switch(o->code){
        case P_CONST:memcpy(y,w,n*sizeof(float));break;
        case P_LINEAR:
#ifdef TCM_USE_CBLAS
            cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)n,(int)m,1,w,(int)m,a,1,0,y,1);
            if(bias)for(size_t i=0;i<n;i++)y[i]+=bias[i];
#else
            for(size_t i=0;i<n;i++){float sum=0;for(size_t j=0;j<m;j++)sum+=w[i*m+j]*a[j];y[i]=sum+(bias?bias[i]:0);}
#endif
            break;
        case P_ADD:for(size_t i=0;i<n;i++)y[i]=a[m==1?0:i]+b[nb==1?0:i];break;
        case P_MUL:for(size_t i=0;i<n;i++)y[i]=a[m==1?0:i]*b[nb==1?0:i];break;
        case P_DIV:for(size_t i=0;i<n;i++)y[i]=a[m==1?0:i]/b[nb==1?0:i];break;
        case P_RELU:for(size_t i=0;i<n;i++)y[i]=a[i]>0?a[i]:0;break;
        case P_MATCH_ROWS:
            for(size_t j=0;j<m;j++)if(!integer(a[j],-1,16777216))return error("invalid projected lookup key");
            for(size_t i=0;i<n;i++){int match=1;for(size_t j=0;j<m;j++)if(a[j]!=-1&&a[j]!=w[i*m+j]){match=0;break;}y[i]=(float)match;}
            break;
        case P_REMAINDER:for(size_t i=0;i<n;i++){float r=fmodf(a[i],o->scale);y[i]=r<0?r+o->scale:r;}break;
        case P_COS:for(size_t i=0;i<n;i++)y[i]=cosf(a[i]);break;
        case P_AFFINE:for(size_t i=0;i<n;i++)y[i]=a[i]*o->scale+o->shift;break;
        case P_CONCAT:memcpy(y,a,m*4);memcpy(y+m,b,nb*4);break;
        case P_SLICE:memcpy(y,a+o->offset,n*4);break;
        case P_RMS:{float ss=0;for(size_t i=0;i<n;i++)ss+=a[i]*a[i];float r=1.0f/sqrtf(ss/(float)n+o->scale);for(size_t i=0;i<n;i++)y[i]=(a[i]*r)*w[i];break;}
        case P_EMBED:if(!isfinite(a[0])||a[0]<0||a[0]>=weight(p,o->w)->dim[0]||floorf(a[0])!=a[0])return error("input token outside embedding");memcpy(y,w+(size_t)a[0]*n,n*4);break;
        case P_CONV:for(size_t i=0;i<n;i++){float *s=p->state+o->offset+i*o->arg;memmove(s,s+1,(o->arg-1)*4);s[o->arg-1]=a[i];float sum=0;for(size_t j=0;j<o->arg;j++)sum+=s[j]*w[i*o->arg+j];y[i]=silu(sum+(bias?bias[i]:0));}break;
        case P_SCAN:for(size_t i=0;i<n;i++){float dt=b[i]>20?b[i]:log1pf(expf(b[i])),sum=0;for(size_t j=0;j<o->arg;j++){size_t k=i*o->arg+j;float decay=expf((-expf(w[k]))*dt),drive=(dt*c[j])*a[i];float s=decay*p->state[o->offset+k]+drive;p->state[o->offset+k]=s;sum+=s*d[j];}y[i]=sum+a[i]*bias[i];}break;
        case P_SILU:for(size_t i=0;i<n;i++)y[i]=silu(a[i]);break;
        }
        if((int)pc==zero)memset(y,0,n*4);
        for(size_t i=0;i<n;i++)if(!isfinite(y[i]))return error("non-finite operation output");
    }
    for(size_t i=0;i<p->ns;i++)if(!isfinite(p->state[i]))return error("non-finite recurrent state");
    if(trace&&fwrite(p->arena,4,p->na,trace)!=p->na)return error("trace write failed");
    return 0;
}
static unsigned top(Program *p){const float *x=p->arena+p->values[p->output].off;unsigned k=0;for(size_t i=1;i<p->vocab;i++)if(x[i]>x[k])k=(unsigned)i;return k;}
static int stopped(unsigned token,int *eos,size_t neos){for(size_t i=0;i<neos;i++)if(token==(unsigned)eos[i])return 1;return 0;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int tcm_program_main(int argc,char **argv){
    const char *input=NULL,*output=NULL,*tracepath=NULL,*statepath=NULL;int generate=0,zero=-1,replay=0,eos[16];size_t neos=0;
    if(argc<3)return error("usage: tcm program BUNDLE --input FP32_ROWS --output FP32_LOGITS [--generate N] [--eos ID] [--replay] [--trace FILE] [--zero OP]")?2:2;
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"--replay")){replay=1;continue;}if(i+1>=argc)return 2;
        const char *key=argv[i++],*v=argv[i];char *end;long n;
        if(!strcmp(key,"--input"))input=v;else if(!strcmp(key,"--output"))output=v;else if(!strcmp(key,"--trace"))tracepath=v;
        else if(!strcmp(key,"--state-file"))statepath=v;
        else{n=strtol(v,&end,10);if(!*v||*end||n<0||n>1000000)return 2;if(!strcmp(key,"--generate"))generate=(int)n;else if(!strcmp(key,"--zero"))zero=(int)n;else if(!strcmp(key,"--eos")&&neos<16)eos[neos++]=(int)n;else return 2;}
    }
    Program p={0};int status=2;FILE *in=NULL,*out=NULL,*trace=NULL;float *row=NULL,*saved=NULL,*saved_logits=NULL,*end_state=NULL,*end_logits=NULL;unsigned *selected=NULL,*again=NULL;
    if(!input||!output||load(&p,argv[2]))goto done;
    if((generate&&!p.tokens)||(zero>=0&&(size_t)zero>=p.no))goto done;
    in=fopen(input,"rb");out=fopen(output,"wb");trace=tracepath?fopen(tracepath,"wb"):NULL;
    if(!in||!out||(tracepath&&!trace))goto done;
    if(fseek(in,0,SEEK_END))goto done;
    long input_bytes=ftell(in);
    if(input_bytes<=0||(size_t)input_bytes%(p.input_width*4)||fseek(in,0,SEEK_SET))goto done;
    row=calloc(p.input_width,4);selected=calloc((size_t)generate+1,sizeof(unsigned));again=calloc((size_t)generate+1,sizeof(unsigned));
    if(!row||!selected||!again)goto done;
    size_t rows=0,got;double start=now();
    while((got=fread(row,4,p.input_width,in))){if(got!=p.input_width||step(&p,row,zero,trace))goto done;if(fwrite(p.arena+p.values[p.output].off,4,len(&p,(int)p.output),out)!=len(&p,(int)p.output))goto done;rows++;}
    if(ferror(in)||!rows)goto done;
    size_t ns=(p.ns?p.ns:1)*4,nlog=len(&p,(int)p.output)*4;float *logits=p.arena+p.values[p.output].off;
    saved=malloc(ns);saved_logits=malloc(nlog);end_state=malloc(ns);end_logits=malloc(nlog);if(!saved||!saved_logits||!end_state||!end_logits)goto done;
    memcpy(saved,p.state,ns);memcpy(saved_logits,logits,nlog);size_t nselected=0;int exact=1;
    for(int pass=0;pass<(replay?2:1);pass++){
        if(pass){memcpy(p.state,saved,ns);memcpy(logits,saved_logits,nlog);}
        size_t count=0;for(int t=0;t<generate;t++){
            unsigned token=top(&p);(pass?again:selected)[count++]=token;
            if(stopped(token,eos,neos)||t+1==generate)break;
            float next=(float)token;if(step(&p,&next,zero,pass?NULL:trace))goto done;
            if(!pass&&fwrite(logits,1,nlog,out)!=nlog)goto done;
        }
        if(!pass){nselected=count;memcpy(end_state,p.state,ns);memcpy(end_logits,logits,nlog);}
        else exact=count==nselected&&!memcmp(again,selected,count*sizeof(unsigned))&&!memcmp(end_state,p.state,ns)&&!memcmp(end_logits,logits,nlog);
    }
    unsigned char hash[32];char state_hash[65];tcm_sha256(p.state,p.ns*4,hash);tcm_hex(hash,32,state_hash);
    if(statepath){FILE *sf=fopen(statepath,"wb");if(!sf)goto done;int bad=fwrite(p.state,4,p.ns,sf)!=p.ns;if(fclose(sf)||bad)goto done;}
    printf("{\"schema\":\"tcm-program-run-v1\",\"model_identity\":\"%s\",\"bundle_digest\":\"%s\",\"profile\":\"fp32-%s-c-no-contraction-libm\",\"input_rows\":%zu,\"ops_per_row\":%zu,\"state_floats\":%zu,\"state_sha256\":\"%s\",\"generated_tokens\":[",p.model.identity,p.model.digest,
#ifdef TCM_USE_CBLAS
    "openblas",
#else
    "sequential",
#endif
    rows,p.no,p.ns,state_hash);
    for(size_t i=0;i<nselected;i++)printf("%s%u",i?",":"",selected[i]);
    printf("],\"consumed_rows\":%zu,\"pending_token\":%d,\"checkpoint_replay\":%s,\"replay_exact\":%s,\"zero_operation\":%d,\"compute_seconds\":%.9g,\"framework_fallback\":false}\n",rows+(nselected?nselected-1:0),nselected?(int)selected[nselected-1]:-1,replay?"true":"false",exact?"true":"false",zero,now()-start);
    status=exact?0:1;
done:
    if(in)fclose(in);if(out&&fclose(out))status=2;if(trace&&fclose(trace))status=2;
    free(row);free(saved);free(saved_logits);free(end_state);free(end_logits);free(selected);free(again);unload(&p);return status;
}
