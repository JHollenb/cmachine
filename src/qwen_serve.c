#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One live session, a legacy parent and a strict nested continuation stack.
 * A appends; G emits; C/R save/rewind; P/B/K push/rollback/commit; Z resets.
 * All numerical execution stays in the C/CUDA process. */
static int frame_id(const char *line,uint64_t *id){
    while(*line==' '||*line=='\t')line++;if(*line<'0'||*line>'9')return -1;
    char *end;errno=0;unsigned long long value=strtoull(line,&end,10);
    while(*end==' '||*end=='\t'||*end=='\r'||*end=='\n')end++;
    if(errno||*end||!value)return -1;*id=(uint64_t)value;return 0;
}
static void state_hashes(const QwenSession *s,char cache[65],char logits[65]){
    const QwenModel *m=s->model;size_t row=(size_t)s->position*m->head,n=(size_t)2*m->layers*m->kvheads*row,offset=0;
    float *host=malloc((n?n:1)*4);if(!host)exit(2);
    for(int kind=0;kind<2;kind++)for(int l=0;l<m->layers;l++)for(int h=0;h<m->kvheads;h++){
        size_t address=((size_t)l*m->kvheads+h)*s->capacity*m->head;
        if(row)gpu_read(host+offset,(kind?s->values:s->keys)+address,row*4);offset+=row;
    }
    unsigned char digest[32];tcm_sha256(host,n*4,digest);tcm_hex(digest,32,cache);free(host);
    if(s->phase!=1){logits[0]=0;return;}
    host=malloc((size_t)m->vocab*4);if(!host)exit(2);gpu_read(host,s->logits,(size_t)m->vocab*4);tcm_sha256(host,(size_t)m->vocab*4,digest);tcm_hex(digest,32,logits);free(host);
}
int qwen_serve_main(int argc,char **argv){
    int explicit_policy=0;
    for(int i=3;i+1<argc;i+=2)if(!strcmp(argv[i],"--cache-policy")){explicit_policy=1;if(strcmp(argv[i+1],"native"))return qwen_bounded_serve_main(argc,argv);}
    const char *name=strrchr(argv[0],'/');name=name?name+1:argv[0];
    if(!explicit_policy&&!strcmp(name,"tcm-bounded-chat"))return qwen_bounded_serve_main(argc,argv);
    if(argc<3||!(argc%2)){fprintf(stderr,"usage: tcm-cuda serve MODEL [--context N] [--cache-policy native|full-byte|bounded]\n");return 2;}
    int capacity=4096;
    for(int i=3;i<argc;i+=2){if(!strcmp(argv[i],"--cache-policy")&&!strcmp(argv[i+1],"native"))continue;if(strcmp(argv[i],"--context"))return 2;char *end;errno=0;long value=strtol(argv[i+1],&end,10);if(errno||end==argv[i+1]||*end||value<1||value>32768)return 2;capacity=(int)value;}
    TcmModel bundle;QwenModel model;QwenSession session;QwenCheckpoint checkpoint;int have_checkpoint=0;char error[256];
    if(tcm_load(&bundle,argv[2],error,sizeof(error))){fprintf(stderr,"%s\n",error);return 2;}
    if(qwen_model_init(&model,&bundle,error,sizeof(error))){fprintf(stderr,"%s\n",error);tcm_unload(&bundle);return 2;}tcm_unload(&bundle);
    if(qwen_session_init(&session,&model,capacity))return 2;
    int *tokens=malloc((size_t)capacity*sizeof(int));if(!tokens)return 2;char *line=NULL;size_t bytes=0;
    printf("{\"event\":\"ready\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"capacity\":%d,\"gpu_weight_bytes\":%zu,\"gpu_session_bytes\":%zu}\n",model.identity,model.digest,capacity,model.weight_bytes,session.allocated_bytes);fflush(stdout);
    while(getline(&line,&bytes,stdin)>=0){
        if(line[0]=='Q')break;
        if(line[0]=='A'){
            char *p=line+1;int count=0,bad=0;
            while(*p){while(*p==' '||*p=='\t'||*p=='\n'||*p=='\r')p++;if(!*p)break;char *end;errno=0;long value=strtol(p,&end,10);if(errno||end==p||value<0||value>=model.vocab||count>=capacity){bad=1;break;}tokens[count++]=(int)value;p=end;}
            if(bad||!count||count>capacity-session.position){puts("{\"error\":\"invalid append or context capacity\"}");fflush(stdout);continue;}
            for(int i=0;i<count;i++)if(qwen_push(&session,tokens[i]))return 2;
            printf("{\"event\":\"appended\",\"count\":%d,\"position\":%d}\n",count,session.position);
        }else if(line[0]=='G'){
            char *end;errno=0;long budget=strtol(line+1,&end,10);while(*end==' '||*end=='\n'||*end=='\r')end++;
            if(errno||*end||budget<1||budget>4096||session.pending<0||budget>capacity-session.position+(session.delivered?0:1)){puts("{\"error\":\"invalid generation budget or state\"}");fflush(stdout);continue;}
            printf("{\"event\":\"generated\",\"ids\":[");int count=0;
            for(long i=0;i<budget;i++){
                if(session.delivered){if(session.pending==model.eos||(model.eos_extra>0&&session.pending==model.eos_extra))break;if(qwen_advance(&session))return 2;}
                if(count++)putchar(',');printf("%d",session.pending);session.delivered=1;
                if(session.pending==model.eos||(model.eos_extra>0&&session.pending==model.eos_extra))break;
            }
            printf("],\"count\":%d,\"position\":%d,\"pending_token\":%d}\n",count,session.position,session.pending);
        }else if(line[0]=='S'){
            char cache[65],logits[65];state_hashes(&session,cache,logits);
            printf("{\"event\":\"state\",\"position\":%d,\"pending_token\":%d,\"delivered\":%d,\"checkpoint_serial\":%llu,\"stack_depth\":%u,\"top_frame\":%llu,\"epoch\":%llu,\"cache_sha256\":\"%s\",\"logits_sha256\":\"%s\",\"logits_valid\":%s}\n",session.position,session.pending,session.delivered,(unsigned long long)session.checkpoint_serial,session.depth,(unsigned long long)(session.depth?session.frames[session.depth-1].id:0),(unsigned long long)session.epoch,cache,logits,session.phase==1?"true":"false");
        }else if(line[0]=='C'){
            if(qwen_checkpoint(&session,&checkpoint))puts("{\"error\":\"checkpoint requires a token boundary\"}");
            else{have_checkpoint=1;printf("{\"event\":\"checkpoint\",\"position\":%d,\"serial\":%llu}\n",session.position,(unsigned long long)checkpoint.serial);}
        }else if(line[0]=='R'){
            if(!have_checkpoint||qwen_restore(&session,&checkpoint))puts("{\"error\":\"no compatible saved parent\"}");
            else printf("{\"event\":\"restored\",\"position\":%d,\"pending_token\":%d}\n",session.position,session.pending);
        }else if(line[0]=='P'){
            uint64_t id;if(qwen_stack_push(&session,&id))puts("{\"error\":\"push requires a token boundary and stack capacity\"}");
            else printf("{\"event\":\"pushed\",\"frame\":%llu,\"depth\":%u,\"position\":%d}\n",(unsigned long long)id,session.depth,session.position);
        }else if(line[0]=='B'||line[0]=='K'){
            uint64_t id;int rollback=line[0]=='B';
            if(frame_id(line+1,&id)||(rollback?qwen_stack_rollback(&session,id):qwen_stack_commit(&session,id)))puts("{\"error\":\"no compatible top frame\"}");
            else{if(rollback)have_checkpoint=0;printf("{\"event\":\"%s\",\"frame\":%llu,\"depth\":%u,\"position\":%d,\"pending_token\":%d}\n",rollback?"rolled_back":"committed",(unsigned long long)id,session.depth,session.position,session.pending);}
        }else if(line[0]=='Z'){
            if(qwen_reset(&session))puts("{\"error\":\"reset epoch exhausted\"}");else{have_checkpoint=0;puts("{\"event\":\"reset\"}");}
        }else puts("{\"error\":\"unknown command\"}");
        fflush(stdout);
    }
    free(line);free(tokens);qwen_session_free(&session);qwen_model_free(&model);return 0;
}
