#include "turn_helpers.h"

int main(int argc,char **argv){
    REQUIRE(argc==15);int n,ns,nt[3];int *prompt=tokens(argv[2],&n),*source_own=tokens(argv[8],&ns),*templates[]={tokens(argv[3],&nt[0]),tokens(argv[4],&nt[1]),tokens(argv[5],&nt[2])};
    REQUIRE(n+ns<=512&&ns>0);TcmModel bundle;QwenModel m;QwenSession s;QwenSubspace p;char error[256],path[4096],source_hash[65],after[65],bank_before[65],bank_after[65];double started=clock_seconds();History source={0};
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(!qwen_session_init(&s,&m,512)&&!qwen_subspace_init(&p,&m));import_bank(&p,argv[7]);bank_fingerprint(&p,bank_before);REQUIRE(!strcmp(bank_before,argv[10]));
    for(int j=0;j<n;j++){REQUIRE(!qwen_push(&s,prompt[j]));source.ids[source.count++]=prompt[j];}
    for(int j=0;j<ns;j++){REQUIRE(s.pending==source_own[j]&&!qwen_push(&s,source_own[j]));source.ids[source.count++]=source_own[j];}
    REQUIRE(source.count<512&&source_own[ns-1]==151645&&source.count==s.position);prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9]));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",argv[6])<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);Capture capture={argv[6],NULL};
    fprintf(f,"{\"schema\":\"tcm-formation-read-driver-v1\",\"model_digest\":\"%s\",\"layer\":16,\"capacity\":512,\"source_replay_checked_before_candidates\":true,\"source_position\":%d,\"source_kv_sha256\":\"%s\",\"source_history_ids\":",m.digest,s.position,source_hash);ints(f,source.ids,source.count);fprintf(f,",\"gpu_program_workspace_bytes\":%zu,\"host_array_bytes\":%zu,\"arms\":[",p.gpu_bytes,p.host_bytes);
    const char *names[]={"PN","NP"};const int formation[]={2,0},reader[]={0,2};
    for(int arm=0;arm<2;arm++){
        if(arm)fputc(',',f);uint64_t root,child;REQUIRE(!qwen_stack_push(&s,&root));History h=source;Turn result;char name[160];fprintf(f,"{\"name\":\"%s\",\"formation_policy\":%d,\"reader_policy\":%d,\"turns\":[",names[arm],formation[arm],reader[arm]);
        REQUIRE(snprintf(name,sizeof(name),"%s-correction",names[arm])<(int)sizeof(name));run_turn(f,&s,&p,formation[arm],name,templates[0],nt[0],&h,&capture,&result,NULL);History corrected=h;
        const char *expected_kv=argv[formation[arm]?12:11],*expected_history=argv[formation[arm]?14:13];
        for(int t=1;t<3;t++){
            fputc(',',f);h=corrected;prefix_digest(&s,s.position,after);REQUIRE(!strcmp(after,expected_kv)&&s.position==h.count);char history_hash[65];history_digest(&h,history_hash);REQUIRE(!strcmp(history_hash,expected_history));
            REQUIRE(!qwen_stack_push(&s,&child));REQUIRE(snprintf(name,sizeof(name),"%s-%s",names[arm],tasks[t])<(int)sizeof(name));run_turn(f,&s,&p,reader[arm],name,templates[t],nt[t],&h,&capture,&result,NULL);REQUIRE(!qwen_stack_rollback(&s,child));
        }
        REQUIRE(!qwen_stack_rollback(&s,root));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_hash,after));fprintf(f,"],\"formation_checked_before_readers\":true,\"source_prefix_intact\":true}");
    }
    bank_fingerprint(&p,bank_after);REQUIRE(!strcmp(bank_before,bank_after));fprintf(f,"],\"historical_endpoint_vectors_read_by_driver\":false,\"frozen_bank_before_sha256\":\"%s\",\"frozen_bank_after_sha256\":\"%s\",\"seconds\":%.9g}\n",bank_before,bank_after,clock_seconds()-started);REQUIRE(!fclose(f));
    qwen_subspace_free(&p);qwen_session_free(&s);qwen_model_free(&m);free(prompt);free(source_own);for(int j=0;j<3;j++)free(templates[j]);return 0;
}
