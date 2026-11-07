#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"bounded-chat %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#include "qwen.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
static void digest(const void *p,size_t n,char text[65]){unsigned char raw[32];tcm_sha256(p,n,raw);tcm_hex(raw,32,text);}
static float *read_floats(const float *p,size_t n){float *h=malloc(n*4);REQUIRE(h);gpu_read(h,p,n*4);for(size_t i=0;i<n;i++)REQUIRE(isfinite(h[i]));return h;}
#include "bounded_cache.h"
#include <errno.h>
extern void gpu_weight_configure(int);
extern void gpu_weight_release(void);
static int number(const char *text,int minimum,int maximum,int *out){char *end;errno=0;long n=strtol(text,&end,10);if(errno||end==text||*end||n<minimum||n>maximum)return -1;*out=(int)n;return 0;}
static int handle(const char *p,uint64_t *out){while(*p==' '||*p=='\t')p++;if(*p<'0'||*p>'9')return -1;char *end;errno=0;unsigned long long n=strtoull(p,&end,10);while(*end==' '||*end=='\t'||*end=='\r'||*end=='\n')end++;if(errno||*end||!n)return -1;*out=(uint64_t)n;return 0;}
static void hashes(Ring *c,char cache[65],char logits[65]){
    size_t n=c->arena_bytes+20+(size_t)c->capacity*4;unsigned char *p=malloc(n);REQUIRE(p);gpu_read(p,c->arena,c->arena_bytes);size_t off=c->arena_bytes;memcpy(p+off,&c->view.cursor,4);off+=4;memcpy(p+off,&c->view.population,8);off+=8;memcpy(p+off,&c->view.rng,8);off+=8;memcpy(p+off,c->positions,(size_t)c->capacity*4);digest(p,n,cache);free(p);float *y=read_floats(c->session->logits,(size_t)c->session->model->vocab);digest(y,(size_t)c->session->model->vocab*4,logits);free(y);
}
static void state(Ring *c,const char *policy){
    QwenSession *s=c->session;char cache[65],logits[65];hashes(c,cache,logits);int valid=0;for(int i=0;i<c->capacity;i++)valid+=c->positions[i]>=0;
    printf("{\"event\":\"state\",\"cache_policy\":\"%s\",\"position\":%d,\"pending_token\":%d,\"delivered\":%d,\"stack_depth\":%u,\"top_frame\":%llu,\"epoch\":%llu,\"logical_capacity\":%d,\"resident_capacity\":%d,\"resident_rows\":%d,\"evictions\":%llu,\"rng\":%llu,\"cache_sha256\":\"%s\",\"logits_sha256\":\"%s\",\"gpu_state_bytes\":%zu,\"tracked_gpu_bytes\":%zu}\n",policy,s->position,s->pending,s->delivered,s->depth,(unsigned long long)(s->depth?s->frames[s->depth-1].id:0),(unsigned long long)s->epoch,s->capacity,c->capacity,valid,(unsigned long long)c->view.population,(unsigned long long)c->view.rng,cache,logits,s->allocated_bytes+c->gpu_bytes,gpu_live_bytes());
}
int qwen_bounded_serve_main(int argc,char **argv){
    if(argc<3||strcmp(argv[1],"serve")){fprintf(stderr,"usage: tcm-cuda serve MODEL [--context N] [--cache-policy bounded|full-byte] [--anchors N --recent N --reservoir N]\n");return 2;}
    int logical=768,a=16,r=64,m=32,weight_profile=2;const char *policy="bounded";
    for(int i=3;i<argc;i+=2){if(i+1>=argc)return 2;const char *key=argv[i],*value=argv[i+1];
        if(!strcmp(key,"--cache-policy")){if(strcmp(value,"bounded")&&strcmp(value,"full-byte"))return 2;policy=value;}
        else if(!strcmp(key,"--weight-profile")){if(!strcmp(value,"tiled-fp32"))weight_profile=1;else if(strcmp(value,"source-bf16"))return 2;}
        else if(!strcmp(key,"--context")){if(number(value,1,32768,&logical))return 2;}
        else if(!strcmp(key,"--anchors")){if(number(value,0,32768,&a))return 2;}
        else if(!strcmp(key,"--recent")){if(number(value,1,32768,&r))return 2;}
        else if(!strcmp(key,"--reservoir")){if(number(value,0,32768,&m))return 2;}
        else return 2;
    }
    int full=!strcmp(policy,"full-byte");if(full){a=0;r=logical;m=0;}if(a+r+m>logical){fprintf(stderr,"physical support exceeds logical capacity\n");return 2;}
    gpu_weight_configure(weight_profile);TcmModel bundle;QwenModel model;QwenSession session;Ring cache;char error[256];
    if(tcm_load(&bundle,argv[2],error,sizeof(error))){fprintf(stderr,"%s\n",error);return 2;}if(qwen_model_init(&model,&bundle,error,sizeof(error))){fprintf(stderr,"%s\n",error);tcm_unload(&bundle);return 2;}tcm_unload(&bundle);ring_open(&cache,&session,&model,logical,a,r,m,full);
    printf("{\"event\":\"ready\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"capacity\":%d,\"cache_policy\":\"%s\",\"anchors\":%d,\"recent\":%d,\"reservoir\":%d,\"resident_capacity\":%d,\"stack_limit\":2,\"legacy_checkpoint\":false,\"weight_storage\":\"%s\",\"weight_profile\":%d,\"gpu_weight_bytes\":%zu,\"gpu_session_bytes\":%zu,\"tracked_gpu_bytes\":%zu}\n",model.identity,model.digest,logical,policy,a,r,m,cache.capacity,weight_profile==2?"source-BF16 matrices / FP32 tile1024":"FP32 matrices / FP32 tile1024",weight_profile,model.weight_bytes,session.allocated_bytes+cache.gpu_bytes,gpu_live_bytes());fflush(stdout);
    int *tokens=malloc((size_t)logical*4);REQUIRE(tokens);char *line=NULL;size_t bytes=0;
    while(getline(&line,&bytes,stdin)>=0){
        if(line[0]=='Q')break;
        if(line[0]=='A'){
            char *p=line+1;int count=0,bad=0;while(*p){while(*p==' '||*p=='\t'||*p=='\n'||*p=='\r')p++;if(!*p)break;char *end;errno=0;long value=strtol(p,&end,10);if(errno||end==p||value<0||value>=model.vocab||count>=logical){bad=1;break;}tokens[count++]=(int)value;p=end;}
            if(bad||!count||count>logical-session.position)puts("{\"error\":\"invalid append or context capacity\"}");
            else{for(int i=0;i<count;i++)REQUIRE(!ring_push(&cache,tokens[i]));printf("{\"event\":\"appended\",\"count\":%d,\"position\":%d}\n",count,session.position);}
        }else if(line[0]=='G'){
            char *end;errno=0;long budget=strtol(line+1,&end,10);while(*end==' '||*end=='\t'||*end=='\n'||*end=='\r')end++;
            if(errno||end==line+1||*end||budget<1||budget>4096||session.pending<0||budget>logical-session.position+(session.delivered?0:1))puts("{\"error\":\"invalid generation budget or state\"}");
            else{printf("{\"event\":\"generated\",\"ids\":[");int count=0;for(long i=0;i<budget;i++){
                if(session.delivered){if(session.pending==model.eos||(model.eos_extra>0&&session.pending==model.eos_extra))break;REQUIRE(!ring_push(&cache,session.pending));}
                if(count++)putchar(',');printf("%d",session.pending);session.delivered=1;if(session.pending==model.eos||(model.eos_extra>0&&session.pending==model.eos_extra))break;
            }printf("],\"count\":%d,\"position\":%d,\"pending_token\":%d}\n",count,session.position,session.pending);}
        }else if(line[0]=='S')state(&cache,policy);
        else if(line[0]=='P'){uint64_t id;if(ring_mark(&cache,&id))puts("{\"error\":\"push requires a token boundary and one of two snapshot slots\"}");else printf("{\"event\":\"pushed\",\"frame\":%llu,\"depth\":%u,\"position\":%d}\n",(unsigned long long)id,session.depth,session.position);}
        else if(line[0]=='B'||line[0]=='K'){
            uint64_t id;int back=line[0]=='B';if(handle(line+1,&id)||(back?ring_rollback(&cache,id):ring_commit(&cache,id)))puts("{\"error\":\"no compatible top snapshot\"}");else printf("{\"event\":\"%s\",\"frame\":%llu,\"depth\":%u,\"position\":%d,\"pending_token\":%d}\n",back?"rolled_back":"committed",(unsigned long long)id,session.depth,session.position,session.pending);
        }else if(line[0]=='C'||line[0]=='R')puts("{\"error\":\"legacy cursor checkpoints unavailable; use P and B/K\"}");
        else if(line[0]=='Z'){REQUIRE(!qwen_reset(&session));gpu_zero(session.logits,(size_t)model.vocab*4);puts("{\"event\":\"reset\"}");}
        else puts("{\"error\":\"unknown command\"}");fflush(stdout);
    }
    free(line);free(tokens);ring_close(&cache);qwen_session_free(&session);qwen_model_free(&model);gpu_weight_release();REQUIRE(!gpu_live_bytes());return 0;
}
