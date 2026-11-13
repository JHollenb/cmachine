#ifndef CM_ISA_FIXTURE_H
#define CM_ISA_FIXTURE_H
#include "machine.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
/* Supplied deterministic R3-shaped organism, not a pretrained ISA model. */
typedef struct { TcmModel bundle;IsaModel model;float *owned[128];size_t count;uint32_t seed; } IsaFixture;
static float *isa_fixture_values(IsaFixture *f,size_t n,int kind){
    assert(f->count<128);float *p=malloc(n*sizeof(float));assert(p);f->owned[f->count++]=p;
    for(size_t i=0;i<n;i++){f->seed=f->seed*1664525u+1013904223u;p[i]=kind==1?1.0f:(kind==2?-.7f:((float)((f->seed>>16)%101)-50)*.001f);}
    return p;
}
static IsaLinear isa_fixture_linear(IsaFixture *f,int in,int out,int bias){
    IsaLinear l={isa_fixture_values(f,(size_t)in*out,0),bias?isa_fixture_values(f,(size_t)out,0):NULL,in,out};return l;
}
static void isa_fixture_init(IsaFixture *f){
    memset(f,0,sizeof(*f));f->seed=721;memset(f->bundle.digest,'b',64);memset(f->bundle.identity,'c',64);
    IsaModel *m=&f->model;m->bundle=&f->bundle;m->post_epsilon=1e-5f;
#define LINE(field,in,out) m->field=isa_fixture_linear(f,in,out,1)
    LINE(decoder1,120,256);LINE(decoder2,256,256);LINE(dest,256,8);LINE(src,256,8);LINE(query,272,32);
    LINE(offset,256,8);LINE(input,293,48);LINE(flag_query,256,64);LINE(reg,48,8);LINE(imm,48,1);LINE(store,48,2);
    LINE(flags1,500,128);LINE(flags2,128,4);LINE(width,256,3);LINE(length,256,15);LINE(syscall,256,1);LINE(store_flag,256,1);
#undef LINE
    m->pos=isa_fixture_values(f,64*16,0);m->ins_pos=isa_fixture_values(f,122*16,0);
    for(int i=0;i<3;i++){
        IsaMixer *x=m->mixer+i;x->input=isa_fixture_linear(f,48,192,1);x->select=isa_fixture_linear(f,96,9,0);x->dt=isa_fixture_linear(f,1,96,1);
        x->output=isa_fixture_linear(f,96,48,1);x->post1=isa_fixture_linear(f,48,96,1);x->post2=isa_fixture_linear(f,96,48,1);
        x->conv=isa_fixture_values(f,96*3,0);x->conv_bias=isa_fixture_values(f,96,0);x->a_log=isa_fixture_values(f,96*4,2);
        x->d=isa_fixture_values(f,96,0);x->gamma=isa_fixture_values(f,48,1);x->beta=isa_fixture_values(f,48,0);
    }
}
static void isa_fixture_free(IsaFixture *f){for(size_t i=0;i<f->count;i++)free(f->owned[i]);memset(f,0,sizeof(*f));}
#endif
