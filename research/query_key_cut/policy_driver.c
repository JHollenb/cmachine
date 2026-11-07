/* Reuse the hash-bound executor; only the finite main and immutable masks differ. */
#define main archived_context_query_main
#include "engine.c"
#undef main
int main(int argc,char **argv){
    REQUIRE(argc==3);root=argv[2];began=now();TcmModel bundle;char error[256];REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&model,&bundle,error,sizeof(error)));tcm_unload(&bundle);
    float *b17=load("basis17.f32",128*D),*q17=load("query17.f32",128*D),*k17=load("key17.f32",128*256),*b18=load("basis18.f32",128*D),*q18=load("query18.f32",128*D),*k18=load("key18.f32",128*256);
    REQUIRE(!qwen_qkv_program_init(programs,&model,17,128,3,b17,q17,k17,NULL));selected=0;
    for(int policy=0;policy<2;policy++){
        int mask=policy?QWEN_QKV_K:QWEN_QKV_Q;REQUIRE(!qwen_qkv_program_init(&later,&model,18,128,mask,b18,policy?NULL:q18,policy?k18:NULL,NULL));
        for(int context=128;context<=1024;context+=896)for(int task=0;task<2;task++){
            char name[96],prompt[96],follow[96];snprintf(name,sizeof(name),"p%d-c%d-n%d",policy,task,context);snprintf(prompt,sizeof(prompt),"c%d-n%d.txt",task,context);snprintf(follow,sizeof(follow),"c%d-n%d-follow.txt",task,context);run_pair(name,4,prompt,follow,0);printf("DONE %s\n",name);REQUIRE(!fflush(stdout)&&getchar()=='\n');
        }
        qwen_qkv_program_free(&later);
    }
    char text[512];int n=snprintf(text,sizeof(text),"{\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"advances\":%llu,\"seconds\":%.9g,\"gpu_peak_bytes\":%zu}\n",model.digest,model.identity,(unsigned long long)advances,now()-began,gpu_peak_bytes());save("execution.json",text,(size_t)n);free(b17);free(q17);free(k17);free(b18);free(q18);free(k18);qwen_qkv_program_free(programs);qwen_model_free(&model);return 0;
}
