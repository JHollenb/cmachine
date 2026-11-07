#define _POSIX_C_SOURCE 200809L
#define main retained_state_test_main
#include "state_mechanics.c"
#undef main
#include "prototype_attention.h"

static void programs(void){
    QwenModel m;fixture(&m);QwenSession a,b;assert(!qwen_session_init(&a,&m,32)&&!qwen_session_init(&b,&m,32));
    float identity[64]={0};for(int i=0;i<8;i++)identity[i*8+i]=1;
    QwenQkvProgram p[2]={{0}};
    for(int j=0;j<2;j++){assert(!qwen_qkv_program_init(p+j,&m,j+1,8,3,identity,m.layer[j+1].q,m.layer[j+1].k,NULL));assert(!qwen_qkv_bind(&a,j+1,p+j));}
    assert(qwen_qkv_bind(&a,0,p)==-1);QwenModel wrong=m;QwenQkvProgram bad=p[0];bad.model=&wrong;assert(qwen_qkv_bind(&a,1,&bad)==-1);
    for(int t=0;t<4;t++){assert(!qwen_push(&a,t+2)&&!qwen_push(&b,t+2));equal(&a,&b);}
    assert(a.operand_work.applications==8&&a.operand_work.code_calls==8&&a.operand_work.compiled_calls==16&&a.operand_work.native_calls==8&&a.operand_work.linear_macs==8*192);
    assert(qwen_qkv_bind(&a,1,NULL)==-1);QwenCheckpoint parent;assert(!qwen_checkpoint(&a,&parent));assert(!qwen_push(&a,8));float saved[16];gpu_read(saved,a.logits,sizeof(saved));uint64_t before=a.operand_work.code_calls;assert(!qwen_restore(&a,&parent)&&!qwen_push(&a,8));assert(!memcmp(saved,a.logits,sizeof(saved))&&a.operand_work.code_calls==before+2);
    assert(!qwen_reset(&a));assert(!qwen_qkv_bind(&a,1,NULL)&&!qwen_qkv_bind(&a,2,NULL));qwen_session_free(&a);qwen_session_free(&b);for(int j=0;j<2;j++)qwen_qkv_program_free(p+j);fixture_free(&m);
}
static void atoms(void){
    QwenModel m;fixture(&m);QwenSession a,b;assert(!qwen_session_init(&a,&m,32)&&!qwen_session_init(&b,&m,32));QwenLexicalProgram p={0},loaded={0};assert(!qwen_lexical_compile(&p,&m,NULL,16));assert(p.compile_linear_macs==16*8*16);
    assert(!qwen_lexical_save(&p,"build/operand-atoms-test.bin"));assert(!qwen_lexical_load(&loaded,&m,"build/operand-atoms-test.bin"));assert(!strcmp(p.fingerprint,loaded.fingerprint));assert(!qwen_lexical_bind(&a,&loaded));
    for(int t=0;t<12;t++){int token=t<4?t+2:b.pending;assert(!qwen_push(&a,token)&&!qwen_push(&b,token));equal(&a,&b);}assert(a.operand_work.lexical_hits==12&&a.operand_work.code_calls==0);
    assert(qwen_lexical_bind(&a,NULL)==-1);assert(!qwen_reset(&a)&&!qwen_lexical_bind(&a,NULL));qwen_lexical_free(&loaded);
    FILE *f=fopen("build/operand-atoms-test.bin","r+b");assert(f&&!fseek(f,-1,SEEK_END));int x=fgetc(f);assert(x!=EOF&&!fseek(f,-1,SEEK_END)&&fputc(x^1,f)!=EOF&&!fclose(f));assert(qwen_lexical_load(&loaded,&m,"build/operand-atoms-test.bin")==-1);remove("build/operand-atoms-test.bin");
    int ids[]={2,3,4};assert(!qwen_lexical_compile(&loaded,&m,ids,3)&&!qwen_lexical_bind(&a,&loaded));QwenSession snapshot=a;assert(qwen_push(&a,9)==-1&&!memcmp(&snapshot,&a,sizeof(a)));
    assert(!qwen_lexical_bind(&a,NULL));qwen_lexical_free(&p);qwen_lexical_free(&loaded);qwen_session_free(&a);qwen_session_free(&b);fixture_free(&m);
}
static void reader(void){
    QwenPrototypeMemory m={0};assert(sizeof(m)==98464);float k[256]={0},v[256]={0},q[1536]={0},a[1536];double e[1536]={0},gradient[1536];
    for(int g=0;g<2;g++){k[g*128]=-1;v[g*128]=-2;}assert(!qwen_prototype_append(&m,0,k,v));for(int g=0;g<2;g++){k[g*128]=1;v[g*128]=2;}assert(!qwen_prototype_append(&m,1,k,v));
    for(int h=0;h<12;h++)e[h*128]=1;assert(!qwen_prototype_read(&m,q,a)&&!qwen_prototype_vjp(&m,q,e,gradient));for(int h=0;h<12;h++){assert(a[h*128]==0&&fabs(gradient[h*128]-2/sqrt(128.))<1e-15);for(int j=1;j<128;j++)assert(gradient[h*128+j]==0);}
    for(int t=2;t<2048;t++){for(int j=0;j<256;j++){k[j]=(float)((j+t)%13-6)/16;v[j]=(float)((j+t)%17-8)/16;}assert(!qwen_prototype_append(&m,(uint64_t)t,k,v));}
    for(int g=0;g<2;g++){uint64_t mass=m.recent;for(int c=0;c<16;c++)mass+=m.count[g][c];assert(mass==2048);}QwenPrototypeMemory saved=m;k[0]=NAN;assert(qwen_prototype_append(&m,2048,k,v)==-1&&!memcmp(&m,&saved,sizeof(m)));
}
int main(void){programs();atoms();reader();puts("operand programs: multilayer parity, rewind, binding, lexical parity/coverage/tamper and covariance VJP pass");return 0;}
