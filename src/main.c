#define _POSIX_C_SOURCE 200809L
#include "machine.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "program.h"
#ifdef TCM_WITH_CUDA
#include "qwen.h"
#endif

static void usage(void){fprintf(stderr,"usage: tcm inspect MODEL\n       tcm isa MODEL --program HEX [--registers HEX,...(8)] [--flags 0101] [--steps N] [--law MODEL --anchor S,T [--quadratic]] [--checkpoint-row N] [--output FILE]\n");}
static int hexval(int c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;}
static int words(const char *text,uint64_t out[8]){
    for(unsigned i=0;i<8;i++){char *end=NULL;if(!*text||*text=='-')return -1;errno=0;out[i]=strtoull(text,&end,16);if(errno||end==text||*end!=(i==7?0:','))return -1;text=end+(i!=7);}return 0;
}
static void array(FILE *f,const float *p,unsigned n){fputc('[',f);for(unsigned i=0;i<n;i++){if(i)fputc(',',f);fprintf(f,"%.9g",p[i]);}fputc(']',f);}
static void arch(FILE *f,const IsaArch *a){fprintf(f,"{\"registers\":[");for(unsigned i=0;i<8;i++)fprintf(f,"%s\"%016llx\"",i?",":"",(unsigned long long)a->registers[i]);fprintf(f,"],\"flags\":[%u,%u,%u,%u],\"rip\":%llu,\"instruction\":%llu}",a->flags[0],a->flags[1],a->flags[2],a->flags[3],(unsigned long long)a->rip,(unsigned long long)a->instruction);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int output_finite(const IsaOutput *o){const float *x=(const float*)o;for(size_t i=0;i<sizeof(*o)/sizeof(float);i++)if(!isfinite(x[i]))return 0;return 1;}
static void heads(FILE *f,const IsaOutput *o){
    fprintf(f,"{\"regs\":[");for(unsigned i=0;i<8;i++){if(i)fputc(',',f);array(f,o->regs[i],64);}fprintf(f,"],\"flags\":");array(f,o->flags,4);
#define H(name,n) fprintf(f,",\"" #name "\":");array(f,o->name,n)
    H(dest,8);H(src,8);H(width,3);H(length,15);fprintf(f,",\"syscall\":%.9g",o->syscall);H(imm,64);fprintf(f,",\"store\":%.9g",o->store);H(store_addr,64);H(store_value,64);
#undef H
    fputc('}',f);
}
int main(int argc,char **argv){
    if(argc>=2&&!strcmp(argv[1],"program"))return tcm_program_main(argc,argv);
#ifdef TCM_WITH_CUDA
    if(argc>=2&&!strcmp(argv[1],"qwen"))return qwen_main(argc,argv);
    if(argc>=2&&!strcmp(argv[1],"serve"))return qwen_serve_main(argc,argv);
#endif
    if(argc<3){usage();return 2;}TcmModel bundle={0},law_bundle={0};char err[256];double load_start=now();
    if(tcm_load(&bundle,argv[2],err,sizeof(err))){fprintf(stderr,"%s\n",err);return 2;}double load_s=now()-load_start;
    if(!strcmp(argv[1],"inspect")){printf("{\"architecture\":%u,\"tensors\":%u,\"bundle_bytes\":%zu,\"identity\":\"%s\",\"digest\":\"%s\"}\n",bundle.architecture,bundle.count,bundle.mapping_bytes,bundle.identity,bundle.digest);tcm_unload(&bundle);return 0;}
    if(strcmp(argv[1],"isa")){usage();tcm_unload(&bundle);return 2;}
    const char *program=NULL,*output_path=NULL,*law_path=NULL;IsaArch initial={0};unsigned steps=1,checkpoint_row=0;IsaLawPolicy policy={0};
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"--quadratic")){policy.quadratic=1;continue;}
        if(i+1>=argc){usage();return 2;}const char *value=argv[++i];
        if(!strcmp(argv[i-1],"--program"))program=value;
        else if(!strcmp(argv[i-1],"--output"))output_path=value;
        else if(!strcmp(argv[i-1],"--law"))law_path=value;
        else if(!strcmp(argv[i-1],"--registers")){if(words(value,initial.registers)){fprintf(stderr,"invalid register list\n");return 2;}}
        else if(!strcmp(argv[i-1],"--flags")){if(strlen(value)!=4)return 2;for(unsigned k=0;k<4;k++){if(value[k]!='0'&&value[k]!='1')return 2;initial.flags[k]=(unsigned char)(value[k]-'0');}}
        else if(!strcmp(argv[i-1],"--steps")||!strcmp(argv[i-1],"--checkpoint-row")){
            char *end;errno=0;unsigned long n=strtoul(value,&end,10);if(errno||!*value||*end||n>1024)return 2;
            if(!strcmp(argv[i-1],"--steps"))steps=(unsigned)n;else checkpoint_row=(unsigned)n;
        }else if(!strcmp(argv[i-1],"--anchor")){
            char *end;policy.s=strtod(value,&end);if(end==value||*end!=',')return 2;const char *tail=end+1;policy.t=strtod(tail,&end);if(end==tail||*end||!isfinite(policy.s)||!isfinite(policy.t))return 2;policy.enabled=1;
        }else{usage();return 2;}
    }
    if(!program||!steps||checkpoint_row>63||strlen(program)%2||strlen(program)>131072||policy.enabled!=!!law_path||(policy.quadratic&&!policy.enabled)){usage();return 2;}
    size_t bytes=strlen(program)/2;unsigned char *code=malloc(bytes?bytes:1);if(!code)return 2;
    for(size_t i=0;i<bytes;i++){int a=hexval(program[2*i]),b=hexval(program[2*i+1]);if(a<0||b<0){fprintf(stderr,"invalid program hex\n");return 2;}code[i]=(unsigned char)(a*16+b);}
    IsaModel model;IsaLaw law;IsaSession session;
    if(isa_model_init(&model,&bundle,err,sizeof(err))){fprintf(stderr,"%s\n",err);return 2;}
    if(law_path&&(tcm_load(&law_bundle,law_path,err,sizeof(err))||isa_law_init(&law,&law_bundle,&model,err,sizeof(err)))){fprintf(stderr,"%s\n",err);return 2;}
    isa_session_init(&session,&model,&initial);session.policy=policy;if(law_path)session.law=&law;
    FILE *f=output_path?fopen(output_path,"w"):stdout;if(!f){perror("output");return 2;}
    fprintf(f,"{\"schema\":\"tcm-isa-run-v1\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"law_digest\":\"%s\",\"mode\":\"%s\",\"arithmetic\":\"FP32 C sequential reductions; contraction disabled\",\"consumer\":\"registers,flags,predicted instruction length; all heads retained\",\"model_bytes\":%zu,\"session_bytes\":%zu,\"load_seconds\":%.9g,\"steps\":[",bundle.identity,bundle.digest,law_path?law_bundle.digest:"",policy.enabled?(policy.quadratic?"anchored-quadratic":"anchored-dense"):"equations",bundle.mapping_bytes,sizeof(session),load_s);
    double elapsed=0;unsigned completed=0;int exact_checkpoint=1;
    for(unsigned i=0;i<steps&&session.state.arch.rip<bytes;i++){
        IsaArch before=session.state.arch;size_t remaining=bytes-(size_t)before.rip,n=remaining<15?remaining:15;
        double start=now();if(isa_begin(&session,code+before.rip,n)){fprintf(stderr,"cannot begin ISA call\n");return 2;}
        IsaCheckpoint checkpoint;int saved=0;
        while(session.state.row<64){if(checkpoint_row&&session.state.row==checkpoint_row){isa_checkpoint(&session,&checkpoint);saved=1;}if(isa_advance(&session)){fprintf(stderr,"ISA transition failed\n");return 2;}}
        if(!output_finite(&session.state.output)){fprintf(stderr,"non-finite output\n");return 2;}
        if(saved){IsaState end=session.state;if(isa_restore(&session,&checkpoint))return 2;while(session.state.row<64)if(isa_advance(&session))return 2;if(memcmp(&end,&session.state,sizeof(end)))exact_checkpoint=0;}
        if(isa_commit(&session)){fprintf(stderr,"ISA commit failed\n");return 2;}elapsed+=now()-start;
        if(completed++)fputc(',',f);fprintf(f,"{\"before\":");arch(f,&before);fprintf(f,",\"after\":");arch(f,&session.state.arch);fprintf(f,",\"heads\":");heads(f,&session.state.output);
        if(policy.enabled&&before.instruction==policy.instruction){fprintf(f,",\"law_coordinates\":[%.17g,%.17g],\"off_plane_norm\":%.17g,\"law_input\":",session.state.law_coordinates[0],session.state.law_coordinates[1],session.state.off_plane_norm);array(f,session.state.law_input,48);fprintf(f,",\"law_branch\":");array(f,session.state.law_branch,48);fprintf(f,",\"law_hidden\":");array(f,session.state.law_hidden,48);}
        fputc('}',f);
    }
    fprintf(f,"],\"completed_steps\":%u,\"compute_seconds\":%.9g,\"dense_post_rows\":%llu,\"law_rows\":%llu,\"checkpoint_row\":%u,\"checkpoint_exact\":%s}\n",completed,elapsed,(unsigned long long)session.state.dense_post_rows,(unsigned long long)session.state.law_rows,checkpoint_row,exact_checkpoint?"true":"false");
    if(output_path&&fclose(f))return 2;free(code);tcm_unload(&law_bundle);tcm_unload(&bundle);return exact_checkpoint?0:1;
}
