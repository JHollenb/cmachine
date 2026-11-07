#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define CAPACITY 512
#define ACK_BUDGET 4
#define ANSWER_BUDGET 8

/* The runner supplies complete token-id files.  This executable intentionally
 * has no tokenizer or chat-template knowledge: the byte sequence in each file
 * is the state-transition input and generated IDs are the only own continuation
 * authority. */
#define REQUIRE(condition) do { \
    if (!(condition)) { fprintf(stderr, "memory_response REQUIRE failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(2); } \
} while (0)

#include "runtime/assay_io.h"
#include "cache.h"
#include "audit.h"

typedef struct { int ids[CAPACITY]; int count; } IdHistory;
typedef struct { FILE *file; int first; } JsonList;

static const char *out_dir;
static uint64_t token_attempts, token_commits, native_score_macs, native_value_macs;

static void json_list_open(JsonList *list, FILE *f) { list->file=f; list->first=1; fputc('[',f); }
static void json_list_next(JsonList *list) { if (!list->first) fputc(',',list->file); list->first=0; }
static void json_list_close(JsonList *list) { fputc(']',list->file); }

static void write_ids(FILE *f, const int *ids, int count) {
    fputc('[',f); for (int i=0;i<count;i++) { if (i) fputc(',',f); fprintf(f,"%d",ids[i]); } fputc(']',f);
}
static void history_add(IdHistory *h, int id) { REQUIRE(h && h->count<CAPACITY); h->ids[h->count++]=id; }
static void history_hash(const IdHistory *h, char out[65]) { digest(h->ids,(size_t)h->count*sizeof(int),out); }

static int load_ids(const char *input_dir, int case_id, const char *name, int **ids, int *count) {
    char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s/case%d-%s.txt",input_dir,case_id,name)<(int)sizeof(path));
    *ids=tokens(path,count); return 0;
}

static void policy_cache_hash(const QwenSession *s, MemoryCache *cache, char out[65]) {
    if (cache) { REQUIRE(cache->count==s->position); memory_cache_digest(cache,s->position,out); }
    else memory_dense_digest(s,s->position,out);
}

static void write_boundary(JsonList *boundaries, QwenSession *s, MemoryCache *cache,
                           const IdHistory *history, int case_id, const char *policy,
                           const char *kind) {
    json_list_next(boundaries);
    char filename[256], logits_hash[65], cache_hash[65], history_sha[65];
    REQUIRE(s->phase==1); REQUIRE(snprintf(filename,sizeof(filename),"case%d-%s-%s-logits.f32",case_id,policy,kind)<(int)sizeof(filename));
    float *logits=read_floats(s->logits,(size_t)s->model->vocab); digest(logits,(size_t)s->model->vocab*sizeof(float),logits_hash); save(out_dir,filename,logits,(size_t)s->model->vocab*sizeof(float));
    policy_cache_hash(s,cache,cache_hash); history_hash(history,history_sha);
    fprintf(boundaries->file,"{\"kind\":\"%s\",\"position\":%d,\"pending\":%d,\"top\":%d,\"runner_up\":%d,\"top_margin\":%.9g,\"history_sha256\":\"%s\",\"cache_sha256\":\"%s\",\"logits\":{\"file\":\"%s\",\"shape\":[%d],\"dtype\":\"<f4\",\"sha256\":\"%s\"}}",kind,s->position,s->pending,s->top.first,s->top.second,s->top.first_value-s->top.second_value,history_sha,cache_hash,filename,s->model->vocab,logits_hash);
    free(logits);
}

static void push_one(QwenSession *s, MemoryCache *cache, IdHistory *history, int token) {
    REQUIRE(token>=0 && token<s->model->vocab); token_attempts++;
    REQUIRE(!qwen_begin_token(s,token));
    while(s->layer<s->model->layers) {
        s->observe_position=s->position;s->observe_layer=s->layer;
        REQUIRE(!qwen_advance_layer(s));
        if(!cache) { native_score_macs+=(uint64_t)s->model->qheads*s->model->head*(s->position+1); native_value_macs+=(uint64_t)s->model->qheads*s->model->head*(s->position+1); }
    }
    REQUIRE(!qwen_finish_token(s)); token_commits++;
    if (cache) REQUIRE(!memory_cache_commit(cache,s->position));
    if(history) history_add(history,token);
}

static void consume(QwenSession *s, MemoryCache *cache, IdHistory *history, const int *ids, int n) {
    for (int i=0;i<n;i++) push_one(s,cache,history,ids[i]);
}

/* The worker supplies each later chat segment as a suffix of the same
 * serialized conversation.  Its first token is sometimes the same literal
 * boundary token that was already consumed at the preceding assistant/user
 * join.  Drop that one token only on an exact adjacent-ID match.  In
 * particular, a different EOS ID does not authorize dropping 151645 (or any
 * other boundary token); the supplied token stream remains the authority. */
static int suffix_start(const IdHistory *history, const int *ids, int n) {
    if (!history || !ids || n <= 0 || history->count <= 0) return 0;
    return ids[0] == history->ids[history->count - 1] ? 1 : 0;
}

static void consume_suffix(QwenSession *s, MemoryCache *cache, IdHistory *history,
                           const int *ids, int n) {
    int start = suffix_start(history, ids, n);
    for (int i=start;i<n;i++) push_one(s,cache,history,ids[i]);
}

static int generate_answer(QwenSession *s, MemoryCache *cache, IdHistory *history, int budget, int *answer, int *stopped_out) {
    int n=0, stopped=0; for (int i=0;i<budget;i++) {
        int token=s->pending; REQUIRE(token>=0); answer[n++]=token; push_one(s,cache,history,token);
        if (token==s->model->eos || (s->model->eos_extra>0 && token==s->model->eos_extra)) { stopped=1; break; }
    }
    if (stopped_out) *stopped_out=stopped; return n;
}

static int logits_equal(const QwenSession *a, const QwenSession *b) {
    size_t n=(size_t)a->model->vocab; float *x=read_floats(a->logits,n),*y=read_floats(b->logits,n); int ok=!memcmp(x,y,n*sizeof(float)); free(x);free(y); return ok;
}

static void pair_push(QwenSession *candidate, MemoryCache *cache, QwenSession *native,
                      IdHistory *history, int token, int *parity) {
    push_one(native,NULL,NULL,token); push_one(candidate,cache,history,token); if (!logits_equal(candidate,native)) *parity=0;
}

static void pair_consume(QwenSession *candidate, MemoryCache *cache, QwenSession *native,
                         IdHistory *history, const int *ids, int n, int *parity) {
    for (int i=0;i<n;i++) pair_push(candidate,cache,native,history,ids[i],parity);
}

static void pair_consume_suffix(QwenSession *candidate, MemoryCache *cache, QwenSession *native,
                                IdHistory *history, const int *ids, int n, int *parity) {
    int start = suffix_start(history, ids, n);
    for (int i=start;i<n;i++) pair_push(candidate,cache,native,history,ids[i],parity);
}

static int pair_generate(QwenSession *candidate, MemoryCache *cache, QwenSession *native,
                         IdHistory *history, int budget, int *answer, int *stopped_out, int *parity) {
    int n=0, stopped=0; for (int i=0;i<budget;i++) { int token=candidate->pending; REQUIRE(token>=0); answer[n++]=token; pair_push(candidate,cache,native,history,token,parity); if (token==candidate->model->eos || (candidate->model->eos_extra>0&&token==candidate->model->eos_extra)) { stopped=1; break; } }
    if (stopped_out) *stopped_out=stopped; return n;
}

static void write_answer_ids(FILE *f, const int *ids, int n) { write_ids(f,ids,n); }

static void run_query(FILE *f, QwenSession *s, MemoryCache *cache, MemoryAudit *audit, IdHistory *history,
                      int case_id, const char *policy, int query_id, const int *query, int query_n,
                      uint64_t frame, MemoryMark *mark, int parent_history_count) {
    fprintf(f,"{\"name\":\"case%d-%s-query%d\",\"query_id\":%d,\"input_ids\":",case_id,policy,query_id,query_id); write_ids(f,query,query_n); fputs(",\"readouts\":",f);
    audit->query_id=query_id;audit->count=0;audit->capture_position=s->position+query_n-suffix_start(history,query,query_n)-1;
    JsonList readouts; json_list_open(&readouts,f); consume_suffix(s,cache,history,query,query_n); write_boundary(&readouts,s,cache,history,case_id,policy,query_id?"query1_input":"query0_input");
    REQUIRE(audit->count==2);audit->capture_position=-1;
    int answer[ANSWER_BUDGET], stopped=0; int answer_n=generate_answer(s,cache,history,ANSWER_BUDGET,answer,&stopped); write_boundary(&readouts,s,cache,history,case_id,policy,query_id?"query1_final":"query0_final"); json_list_close(&readouts);
    fprintf(f,",\"answer_ids\":"); write_answer_ids(f,answer,answer_n);
    fputs(",\"consumed_ids\":",f);write_ids(f,history->ids,history->count);
    fputs(",\"audit_rows\":",f);audit_write(f,audit);
    fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"final_position\":%d,\"final_pending\":%d}",stopped?"true":"false",stopped?"false":(answer_n==ANSWER_BUDGET?"true":"false"),s->position,s->pending);
    REQUIRE(!qwen_stack_rollback(s,frame)); history->count=parent_history_count; if(cache) REQUIRE(!memory_mark_restore(cache,mark));
}

/* A changed sibling, a repeated suffix, and a partial layer write all return
 * to this same parent. Counters deliberately retain every physical attempt. */
static void mechanics(FILE *f,QwenSession *s,MemoryCache *c,IdHistory *history,const int *query,int n,int enabled) {
    if(!enabled) { fputs("{\"tested\":false}",f);return; }
    REQUIRE(n>=6);int length=6,parent_count=history->count,parent_position=s->position,parent_pending=s->pending;
    char parent[65],reference[65],current[65];policy_cache_hash(s,c,parent);
    MemoryMark mark={0};if(c)REQUIRE(!memory_mark_capture(c,&mark));
    float *expected=NULL;uint64_t frame;int sibling[6];memcpy(sibling,query,sizeof(sibling));sibling[5]=(sibling[5]+1)%s->model->vocab;
    for(int branch=0;branch<3;branch++) {
        REQUIRE(!qwen_stack_push(s,&frame)); consume(s,c,history,branch==1?sibling:query,length);
        if(branch==0) { expected=read_floats(s->logits,(size_t)s->model->vocab);policy_cache_hash(s,c,reference); }
        if(branch==2) { float *actual=read_floats(s->logits,(size_t)s->model->vocab);REQUIRE(!memcmp(expected,actual,(size_t)s->model->vocab*sizeof(float)));free(actual);policy_cache_hash(s,c,current);REQUIRE(!strcmp(current,reference)); }
        REQUIRE(!qwen_stack_rollback(s,frame));if(c)REQUIRE(!memory_mark_restore(c,&mark));history->count=parent_count;
        policy_cache_hash(s,c,current);REQUIRE(!strcmp(parent,current));REQUIRE(s->position==parent_position&&s->pending==parent_pending);
    }
    token_attempts++;REQUIRE(!qwen_begin_token(s,query[0]));
    if(c){c->fail_layer=18;c->fail_position=parent_position;}
    int status=0;
    while(s->layer<s->model->layers) {
        int attempted_layer=s->layer;s->observe_position=s->position;s->observe_layer=s->layer;
        status=qwen_advance_layer(s);
        if(c&&attempted_layer==18) { REQUIRE(status==-77);break; }
        REQUIRE(!status);if(!c){native_score_macs+=(uint64_t)s->model->qheads*s->model->head*(s->position+1);native_value_macs+=(uint64_t)s->model->qheads*s->model->head*(s->position+1);if(s->layer==18)break;}
    }
    if(c)for(int l=0;l<s->model->layers;l++)REQUIRE(c->layers[l].cursor==parent_position+(l<18));
    REQUIRE(!qwen_abort_token(s));if(c)REQUIRE(!memory_mark_restore(c,&mark));
    policy_cache_hash(s,c,current);REQUIRE(!strcmp(parent,current));REQUIRE(s->position==parent_position&&s->pending==parent_pending);
    REQUIRE(!qwen_stack_push(s,&frame));consume(s,c,history,query,length);
    float *actual=read_floats(s->logits,(size_t)s->model->vocab);REQUIRE(!memcmp(expected,actual,(size_t)s->model->vocab*sizeof(float)));free(actual);free(expected);
    policy_cache_hash(s,c,current);REQUIRE(!strcmp(reference,current));
    REQUIRE(!qwen_stack_rollback(s,frame));if(c)REQUIRE(!memory_mark_restore(c,&mark));history->count=parent_count;
    policy_cache_hash(s,c,current);REQUIRE(!strcmp(parent,current));memory_mark_free(&mark);
    fprintf(f,"{\"tested\":true,\"sibling_changed_id\":true,\"replay_logits_exact\":true,\"replay_cache_exact\":true,\"partial_abort_parent_exact\":true,\"partial_failure_layer\":%d,\"partial_status\":%d,\"physical_committed_test_tokens\":24,\"aborted_attempts\":1}",c?18:17,status);
}

static void consume_prefix(JsonList *boundaries, QwenSession *s, MemoryCache *cache,
                           QwenSession *native, IdHistory *history, int *parity,
                           int case_id, const char *policy, const char *input_dir,
                           int ack_ids[3][ANSWER_BUDGET], int ack_counts[3]) {
    const char *names[] = {"initial","update","distractor"}; int *ids=NULL,n=0,answer[ANSWER_BUDGET];
    for (int segment=0;segment<3;segment++) {
        REQUIRE(!load_ids(input_dir,case_id,names[segment],&ids,&n));
        if (segment == 0) {
            if (native) pair_consume(s,cache,native,history,ids,n,parity); else consume(s,cache,history,ids,n);
        } else {
            if (native) pair_consume_suffix(s,cache,native,history,ids,n,parity); else consume_suffix(s,cache,history,ids,n);
        }
        write_boundary(boundaries,s,cache,history,case_id,policy,segment==0?"initial_input":segment==1?"update_input":"distractor_input"); free(ids); ids=NULL;
        int stopped=0; int answer_n=native?pair_generate(s,cache,native,history,ACK_BUDGET,answer,&stopped,parity):generate_answer(s,cache,history,ACK_BUDGET,answer,&stopped);
        ack_counts[segment]=answer_n; for (int j=0;j<answer_n;j++) ack_ids[segment][j]=answer[j];
        write_boundary(boundaries,s,cache,history,case_id,policy,segment==0?"initial_ack":segment==1?"update_ack":"common_parent");
    }
}

static void run_case(FILE *f, const QwenModel *model, const char *input_dir, int case_id, MemoryPolicy policy) {
    fprintf(stderr,"memory_response case%d %s begin\n",case_id,memory_policy_name(policy));fflush(stderr);
    double started=clock_seconds();size_t base_bytes=gpu_live_bytes();
    uint64_t calls_before=gpu_linear_calls(),macs_before=gpu_linear_macs(),attempts_before=token_attempts,commits_before=token_commits,score_before=native_score_macs,value_before=native_value_macs;
    const char *policy_name=memory_policy_name(policy); QwenSession s, native_ref; MemoryCache cache; memset(&cache,0,sizeof(cache)); MemoryCache *c=NULL; int native_control=policy==MEMORY_FP32_ALL, parity=1; int initialized_ref=0;
    uint64_t mask=(UINT64_C(1)<<MEMORY_RESPONSE_LAYERS)-1;
    if (policy==MEMORY_BYTE_ALL||policy==MEMORY_BYTE_TOP4||policy==MEMORY_BYTE_RECENT4||policy==MEMORY_FP32_ALL||policy==MEMORY_ZERO) {
        REQUIRE(!qwen_session_init_layout(&s,model,CAPACITY,mask,memory_cache_step,memory_cache_rewind,&cache)); REQUIRE(!memory_cache_open(&cache,&s,policy,CAPACITY)); memory_cache_bind(&cache); c=&cache;
    } else REQUIRE(!qwen_session_init(&s,model,CAPACITY));
    if (native_control) { REQUIRE(!qwen_session_init(&native_ref,model,CAPACITY)); initialized_ref=1; }
    MemoryAudit audit;audit_open(&audit,&s,c,out_dir,case_id,policy_name);s.observer=audit_observe;s.observer_context=&audit;
    size_t state_gpu_bytes=gpu_live_bytes()-base_bytes;
    IdHistory history={0}; int ack_ids[3][ANSWER_BUDGET]={{0}},ack_counts[3]={0}; JsonList boundaries; fprintf(f,"{\"case_id\":%d,\"policy\":\"%s\",\"prefix_boundaries\":",case_id,policy_name); json_list_open(&boundaries,f);
    consume_prefix(&boundaries,&s,c,initialized_ref?&native_ref:NULL,&history,&parity,case_id,policy_name,input_dir,ack_ids,ack_counts); json_list_close(&boundaries);
    fprintf(f,",\"ack_answers\":{\"initial\":"); write_ids(f,ack_ids[0],ack_counts[0]); fputs(",\"update\":",f); write_ids(f,ack_ids[1],ack_counts[1]); fputs(",\"distractor\":",f); write_ids(f,ack_ids[2],ack_counts[2]); fputc('}',f);
    char parent_cache[65],parent_history[65]; policy_cache_hash(&s,c,parent_cache); history_hash(&history,parent_history); int parent_position=s.position; int parent_pending=s.pending;
    fprintf(f,",\"parent_position\":%d,\"parent_pending\":%d,\"parent_history_ids\":",parent_position,parent_pending); write_ids(f,history.ids,history.count); fprintf(f,",\"parent_history_sha256\":\"%s\",\"parent_cache_sha256\":\"%s\",\"native_prefix_control\":{\"available\":%s,\"exact_logits_each_boundary\":%s}",parent_history,parent_cache,native_control?"true":"false",native_control?(parity?"true":"false"):"null");
    if (native_control) { char nh[65]; memory_dense_digest(&native_ref,native_ref.position,nh); fprintf(f,",\"native_reference_cache_sha256\":\"%s\",\"callback_reference_cache_sha256\":\"%s\",\"native_reference_position\":%d",nh,parent_cache,native_ref.position); REQUIRE(parity&&!strcmp(nh,parent_cache)&&native_ref.position==parent_position&&native_ref.pending==parent_pending); }
    int *q0=NULL,*q1=NULL,n0=0,n1=0;REQUIRE(!load_ids(input_dir,case_id,"query0",&q0,&n0));REQUIRE(!load_ids(input_dir,case_id,"query1",&q1,&n1));
    fputs(",\"mechanics\":",f);mechanics(f,&s,c,&history,q0,n0,case_id==0);
    if (native_control) REQUIRE(!memory_cache_set_floor(c,parent_position));
    char query_parent_cache[65];policy_cache_hash(&s,c,query_parent_cache);
    fprintf(f,",\"query_parent_cache_sha256\":\"%s\"",query_parent_cache);
    MemoryMark mark={0}; if (c) REQUIRE(!memory_mark_capture(c,&mark));
    fputs(",\"queries\":[",f);
    uint64_t frame; int parent_history_count=history.count; REQUIRE(!qwen_stack_push(&s,&frame)); run_query(f,&s,c,&audit,&history,case_id,policy_name,0,q0,n0,frame,&mark,parent_history_count); fputc(',',f); REQUIRE(!qwen_stack_push(&s,&frame)); run_query(f,&s,c,&audit,&history,case_id,policy_name,1,q1,n1,frame,&mark,parent_history_count); fputs("]",f);
    free(q0);free(q1); char final_cache[65]; policy_cache_hash(&s,c,final_cache);REQUIRE(!strcmp(final_cache,query_parent_cache)&&s.position==parent_position&&s.pending==parent_pending);
    fprintf(f,",\"after_rollback_position\":%d,\"after_rollback_pending\":%d,\"after_rollback_cache_sha256\":\"%s\",\"callbacks\":{\"calls\":%lu,\"published\":%lu,\"rewinds\":%lu,\"floor\":%d},\"state_bytes\":{\"session\":%zu,\"cache_arena\":%zu,\"workspace\":%zu,\"status\":%zu,\"recent_ring_snapshot_host\":%zu,\"audit_history_host\":%zu,\"aux_native_reference_session\":%zu,\"tracked_gpu_total\":%zu}",s.position,s.pending,final_cache,c?c->calls:0,c?c->published:0,c?c->rewinds:0,c?c->floor:-1,s.allocated_bytes,c?c->arena_bytes:0,c?c->workspace_bytes:0,c?sizeof(int):0,c&&c->policy==MEMORY_BYTE_RECENT4?mark.code_bytes*2+mark.scale_bytes:0,audit.host_bytes,initialized_ref?native_ref.allocated_bytes:0,state_gpu_bytes);
    fprintf(f,",\"work\":{\"physical_token_attempts\":%llu,\"physical_token_commits\":%llu,\"learned_linear_calls\":%llu,\"learned_linear_macs\":%llu,\"score_macs\":%llu,\"value_macs_nominal\":%llu,\"decoded_scalars\":%llu,\"value_probability_scans\":%llu,\"elapsed_seconds_including_audit_and_mechanics\":%.9g},\"callback_mask\":\"%s\",\"rollback_parent_exact\":true}",(unsigned long long)(token_attempts-attempts_before),(unsigned long long)(token_commits-commits_before),(unsigned long long)(gpu_linear_calls()-calls_before),(unsigned long long)(gpu_linear_macs()-macs_before),(unsigned long long)(native_score_macs-score_before+(c?c->score_macs:0)),(unsigned long long)(native_value_macs-value_before+(c?c->value_macs:0)),(unsigned long long)(c?c->decoded_scalars:0),(unsigned long long)(c?c->value_probability_scans:0),clock_seconds()-started,c?"0x0fffffff":"0x0");
    memory_mark_free(&mark);audit_close(&audit); if (initialized_ref) qwen_session_free(&native_ref); qwen_session_free(&s); if(c) memory_cache_close(c);REQUIRE(gpu_live_bytes()==base_bytes);
    fprintf(stderr,"memory_response case%d %s complete %.3fs\n",case_id,policy_name,clock_seconds()-started);fflush(stderr);fflush(f);
}

int main(int argc, char **argv) {
    REQUIRE(argc==5); const char *model_path=argv[1], *input_dir=argv[2]; out_dir=argv[3]; const char *lane=argv[4];
    REQUIRE(!mkdir(out_dir,0777)||errno==EEXIST); int cases=!strcmp(lane,"smoke")?1:4; REQUIRE(!strcmp(lane,"smoke")||!strcmp(lane,"panel")||!strcmp(lane,"full"));
    TcmModel bundle; QwenModel model; char error[256]; REQUIRE(!tcm_load(&bundle,model_path,error,sizeof(error))); size_t source_bytes=bundle.mapping_bytes; REQUIRE(!qwen_model_init(&model,&bundle,error,sizeof(error))); tcm_unload(&bundle);
    REQUIRE(model.layers==28&&model.hidden==1536&&model.intermediate==8960&&model.qheads==12&&model.kvheads==2&&model.head==128&&model.vocab==151936);
    char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",out_dir)<(int)sizeof(path)); FILE *f=fopen(path,"wb"); REQUIRE(f);
    fprintf(f,"{\"schema\":\"tcm-memory-response-driver-v1\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"source_bundle_bytes\":%zu,\"capacity\":%d,\"ack_budget\":%d,\"answer_budget\":%d,\"callback_layers\":%d,\"lane\":\"%s\",\"policies\":[\"native\",\"byte_full\",\"byte_top4\",\"byte_recent4\",\"history_deleted\"],\"cases\":[",model.identity,model.digest,source_bytes,CAPACITY,ACK_BUDGET,ANSWER_BUDGET,model.layers,lane);
    const MemoryPolicy policies[]={MEMORY_NATIVE, MEMORY_BYTE_ALL, MEMORY_BYTE_TOP4, MEMORY_BYTE_RECENT4, MEMORY_FP32_ALL};
    for (int p=0;p<5;p++) for (int case_id=0;case_id<cases;case_id++) {
        if (p||case_id) fputc(',',f);
        run_case(f,&model,input_dir,case_id,policies[p]);
    }
    fprintf(f,"] ,\"gpu_weight_bytes\":%zu,\"tracked_gpu_peak_bytes\":%zu,\"physical_token_attempts\":%llu,\"physical_token_commits\":%llu,\"learned_linear_calls\":%llu,\"learned_linear_macs\":%llu}\n",model.weight_bytes,gpu_peak_bytes(),(unsigned long long)token_attempts,(unsigned long long)token_commits,(unsigned long long)gpu_linear_calls(),(unsigned long long)gpu_linear_macs()); REQUIRE(!fclose(f)); qwen_model_free(&model);REQUIRE(!gpu_live_bytes()); return 0;
}
