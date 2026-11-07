#define _POSIX_C_SOURCE 200809L
#include "machine.h"
#include "gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"quotient %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#define DIM 8
#define WIDTH 4
#define HORIZON 16
static const char *directory;
static unsigned norm_calls,matvec_calls,swiglu_calls;
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void digest(const void *v,size_t bytes,char out[65]){unsigned char raw[32];tcm_sha256(v,bytes,raw);tcm_hex(raw,32,out);}
static void save(const char *name,const void *v,size_t bytes){char p[4096];REQUIRE(snprintf(p,sizeof(p),"%s/%s",directory,name)<(int)sizeof(p));FILE *f=fopen(p,"wb");REQUIRE(f&&fwrite(v,1,bytes,f)==bytes&&!fclose(f));}
static void floats(const char *name,const float *v,size_t n){for(size_t i=0;i<n;i++)REQUIRE(isfinite(v[i]));save(name,v,n*4);}
static void read_finite(float *host,const float *device,int n){gpu_read(host,device,(size_t)n*4);for(int i=0;i<n;i++)REQUIRE(isfinite(host[i]));}
static void matrices(float g[WIDTH*DIM],float u[WIDTH*DIM],float down[DIM*WIDTH],int escape){
    memset(g,0,WIDTH*DIM*4);memset(u,0,WIDTH*DIM*4);memset(down,0,DIM*WIDTH*4);
    g[0]=.5f;g[DIM+1]=-.75f;g[2*DIM]=.25f;g[3*DIM+1]=.5f;
    u[1]=.5f;u[DIM]=.25f;u[2*DIM+1]=-.5f;u[3*DIM]=.5f;
    down[0]=.125f;down[WIDTH+1]=-.125f;if(escape)down[2*WIDTH]=.25f;
}
static void parameters(const char *prefix,const float *g,const float *u,const float *d){char name[128];snprintf(name,sizeof(name),"%s-G.f32",prefix);floats(name,g,WIDTH*DIM);snprintf(name,sizeof(name),"%s-U.f32",prefix);floats(name,u,WIDTH*DIM);snprintf(name,sizeof(name),"%s-D.f32",prefix);floats(name,d,DIM*WIDTH);}
static void trajectory(FILE *f,TcmGpu *gpu,const char *name,int rank,int escape,int sign,int steps){
    float g[WIDTH*DIM],u[WIDTH*DIM],d[DIM*WIDTH];matrices(g,u,d,escape);int compact=rank<DIM;
    float cg[WIDTH*DIM]={0},cu[WIDTH*DIM]={0},cd[DIM*WIDTH]={0};
    for(int row=0;row<WIDTH;row++)for(int col=0;col<rank;col++){cg[row*rank+col]=g[row*DIM+col];cu[row*rank+col]=u[row*DIM+col];}
    for(int row=0;row<rank;row++)for(int col=0;col<WIDTH;col++)cd[row*WIDTH+col]=d[row*WIDTH+col];
    float *wg=gpu_weight(cg,TCM_F32,(size_t)WIDTH*rank),*wu=gpu_weight(cu,TCM_F32,(size_t)WIDTH*rank),*wd=gpu_weight(cd,TCM_F32,(size_t)rank*WIDTH);
    float one[DIM];for(int i=0;i<DIM;i++)one[i]=1;float *gamma=gpu_weight(one,TCM_F32,DIM);
    float *state=gpu_alloc((size_t)rank*4),*representative=gpu_alloc(DIM*4),*normal=gpu_alloc(DIM*4),*gate=gpu_alloc(WIDTH*4),*up=gpu_alloc(WIDTH*4),*write=gpu_alloc((size_t)rank*4),*rho=gpu_alloc(4);
    float initial[DIM]={.75f,-.5f,0,0,0,0,0,0};initial[2]=(float)sign;
    float energy=compact?(rank==2?1.0f:0.0f):0;gpu_write(state,initial,(size_t)rank*4);
    float states[(HORIZON+1)*DIM]={0},norms[HORIZON*DIM]={0},gates[HORIZON*WIDTH]={0},ups[HORIZON*WIDTH]={0},products[HORIZON*WIDTH]={0},writes[HORIZON*DIM]={0},scales[HORIZON]={0};read_finite(states,state,rank);
    for(int step=0;step<steps;step++){
        if(compact){float canonical[DIM]={0};read_finite(canonical,state,rank);canonical[rank]=sqrtf(energy);gpu_write(representative,canonical,DIM*4);}
        gpu_rms_scale(normal,compact?representative:state,gamma,DIM,1e-6f,rho);norm_calls++;
        read_finite(norms+(size_t)step*DIM,normal,DIM);gpu_read(scales+step,rho,4);
        gpu_matvec(gpu,wg,rank,WIDTH,normal,gate);gpu_matvec(gpu,wu,rank,WIDTH,normal,up);matvec_calls+=2;
        read_finite(gates+(size_t)step*WIDTH,gate,WIDTH);read_finite(ups+(size_t)step*WIDTH,up,WIDTH);
        gpu_swiglu(gate,up,WIDTH);swiglu_calls++;read_finite(products+(size_t)step*WIDTH,gate,WIDTH);
        gpu_matvec(gpu,wd,WIDTH,rank,gate,write);matvec_calls++;read_finite(writes+(size_t)step*rank,write,rank);
        gpu_add(state,write,rank);read_finite(states+(size_t)(step+1)*rank,state,rank);
    }
    const char *suffix[]={"states","norms","gates","ups","products","writes","rho"};const float *values[]={states,norms,gates,ups,products,writes,scales};size_t count[]={(size_t)(steps+1)*rank,(size_t)steps*DIM,(size_t)steps*WIDTH,(size_t)steps*WIDTH,(size_t)steps*WIDTH,(size_t)steps*rank,(size_t)steps};
    for(int i=0;i<7;i++){char file[128];snprintf(file,sizeof(file),"%s-%s.f32",name,suffix[i]);floats(file,values[i],count[i]);}
    fprintf(f,"{\"name\":\"%s\",\"rank\":%d,\"escape_write\":%s,\"sign\":%d,\"steps\":%d,\"compact\":%s,\"retained_energy\":%.9g,\"state_bytes\":%d,\"static_affine_bytes\":%d,\"affine_macs_per_step\":%d,\"normalization_width\":8}",name,rank,escape?"true":"false",sign,steps,compact?"true":"false",energy,4*(rank+compact),4*3*rank*WIDTH,3*rank*WIDTH);REQUIRE(!fflush(f));
    gpu_free(wg);gpu_free(wu);gpu_free(wd);gpu_free(gamma);gpu_free(state);gpu_free(representative);gpu_free(normal);gpu_free(gate);gpu_free(up);gpu_free(write);gpu_free(rho);
}
static void seal_candidates(void){
    DIR *dir=opendir(directory);REQUIRE(dir);char path[4096];snprintf(path,sizeof(path),"%s/candidate-seal.json",directory);FILE *out=fopen(path,"w");REQUIRE(out);fprintf(out,"{\"files\":[");int count=0;struct dirent *entry;
    while((entry=readdir(dir))){size_t n=strlen(entry->d_name);if(n<4||strcmp(entry->d_name+n-4,".f32"))continue;
        REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,entry->d_name)<(int)sizeof(path));struct stat st;REQUIRE(!stat(path,&st)&&S_ISREG(st.st_mode));unsigned char *data=malloc((size_t)st.st_size);FILE *in=fopen(path,"rb");REQUIRE(data&&in&&fread(data,1,(size_t)st.st_size,in)==(size_t)st.st_size&&!fclose(in));char hash[65];digest(data,(size_t)st.st_size,hash);free(data);if(count++)fputc(',',out);fprintf(out,"{\"name\":\"%s\",\"bytes\":%zu,\"sha256\":\"%s\"}",entry->d_name,(size_t)st.st_size,hash);
    }fprintf(out,"]}\n");REQUIRE(!fclose(out)&&!closedir(dir));
}
static void minor(FILE *f,const char *model_path){
    TcmModel model;char error[256];REQUIRE(!tcm_load(&model,model_path,error,sizeof(error)));REQUIRE(model.architecture==TCM_QWEN&&model.config[0]==1536&&model.config[5]==8960);
    const char *tensor_name="model.layers.16.mlp.down_proj.weight";const TcmTensor *tensor=tcm_tensor(&model,tensor_name);REQUIRE(tensor&&tensor->rank==2&&tensor->dim[0]==1536&&tensor->dim[1]==8960&&(tensor->dtype==TCM_BF16||tensor->dtype==TCM_F32));
    const int n=256;float *tile=malloc((size_t)n*n*4);REQUIRE(tile);
    for(int row=0;row<n;row++)for(int col=0;col<n;col++){
        size_t offset=(size_t)row*8960+col;if(tensor->dtype==TCM_BF16){uint32_t bits=(uint32_t)((const uint16_t*)tensor->data)[offset]<<16;memcpy(tile+(size_t)row*n+col,&bits,4);}else tile[(size_t)row*n+col]=((const float*)tensor->data)[offset];
    }
    floats("qwen-down-minor.f32",tile,(size_t)n*n);char hash[65];digest(tile,(size_t)n*n*4,hash);free(tile);
    fprintf(f,",\"qwen_minor\":{\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"tensor\":\"%s\",\"source_dtype\":%u,\"source_shape\":[1536,8960],\"rows\":[0,256],\"columns\":[0,256],\"shape\":[256,256],\"sha256\":\"%s\",\"qwen_forwards\":0,\"gpu_weight_bank_loaded\":false}",model.digest,model.identity,tensor_name,tensor->dtype,hash);tcm_unload(&model);
}
int main(int argc,char **argv){
    REQUIRE(argc==3);directory=argv[2];int mechanics=!strcmp(argv[1],"--mechanics"),steps=mechanics?1:HORIZON;double started=seconds();TcmGpu *gpu=gpu_open();
    float g[WIDTH*DIM],u[WIDTH*DIM],d[DIM*WIDTH];matrices(g,u,d,0);parameters("closed",g,u,d);matrices(g,u,d,1);parameters("escape",g,u,d);
    char path[4096];snprintf(path,sizeof(path),"%s/driver.json",directory);FILE *f=fopen(path,"w");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-state-quotient-driver-v1\",\"dimension\":8,\"width\":4,\"steps\":%d,\"supplied_organism\":true,\"paths\":[",steps);
    trajectory(f,gpu,"closed-compact",2,0,1,steps);fputc(',',f);trajectory(f,gpu,"escape-stale",2,1,1,steps);fputc(',',f);trajectory(f,gpu,"repair-plus",3,1,1,steps);fputc(',',f);trajectory(f,gpu,"repair-minus",3,1,-1,steps);
    seal_candidates();fprintf(stderr,"sealed four reduced-state trajectories before references\n");
    fputc(',',f);trajectory(f,gpu,"closed-plus",8,0,1,steps);fputc(',',f);trajectory(f,gpu,"closed-minus",8,0,-1,steps);fputc(',',f);trajectory(f,gpu,"escape-plus",8,1,1,steps);fputc(',',f);trajectory(f,gpu,"escape-minus",8,1,-1,steps);
    fprintf(f,"],\"candidates_sealed_before_full_references\":true,\"norm_calls\":%u,\"matvec_calls\":%u,\"swiglu_calls\":%u",norm_calls,matvec_calls,swiglu_calls);
    if(!mechanics)minor(f,argv[1]);fprintf(f,",\"seconds\":%.9g}\n",seconds()-started);REQUIRE(!fclose(f));gpu_close(gpu);return 0;
}
