#include <stdio.h>
#include "span_bank.h"
#define CHECK(x) do{if(!(x)){fprintf(stderr,"bank smoke line%d\n",__LINE__);return 2;}}while(0)
static int write_file(const char *root,const char *name,const void *data,size_t bytes){char path[4096];snprintf(path,sizeof(path),"%s/%s",root,name);FILE *f=fopen(path,"wb");return !f||fwrite(data,1,bytes,f)!=bytes||fclose(f);}
int main(int argc,char **argv){
    CHECK(argc==2);float keys[2*34*128],values[2*34*128],q[12*128];SpanProgram programs[3];const int sizes[]={1,8,34};
    for(int h=0;h<2;h++)for(int i=0;i<34;i++)for(int d=0;d<128;d++){size_t ix=((size_t)h*34+i)*128+d;keys[ix]=(float)((h*7+i*3+d)%23-11)/16;values[ix]=(float)((h*3+i*7+d*5)%29-14)/32;}
    for(int h=0;h<12;h++)for(int d=0;d<128;d++)q[h*128+d]=(float)((d+h)%7-3)/32;
    CHECK(!write_file(argv[1],"original-keys.f32",keys,sizeof(keys))&&!write_file(argv[1],"original-values.f32",values,sizeof(values))&&!write_file(argv[1],"query.f32",q,sizeof(q)));
    double before[3][12*129];
    for(int b=0;b<3;b++){
        SpanProgram *p=&programs[b];int m=sizes[b];CHECK(!span_construct(p,m,keys,values));size_t n=(size_t)2*m,bytes=32+(n+2*n*128)*4;unsigned char *raw=malloc(bytes);CHECK(raw);memcpy(raw,"TCMSPAN1",8);uint32_t header[]={1,2,128,34,(uint32_t)m,0};memcpy(raw+8,header,sizeof(header));size_t offset=32;memcpy(raw+offset,p->counts,n*4);offset+=n*4;memcpy(raw+offset,p->keys,n*128*4);offset+=n*128*4;memcpy(raw+offset,p->values,n*128*4);
        char name[96];snprintf(name,sizeof(name),"fixture-%d.bin",m);CHECK(!write_file(argv[1],name,raw,bytes));free(raw);snprintf(name,sizeof(name),"members-%d.u64",m);CHECK(!write_file(argv[1],name,p->members,n*sizeof(uint64_t)));
        for(int h=0;h<12;h++)CHECK(!span_predict(p,q+h*128,h/6,1.0f/sqrtf(128.0f),before[b]+h*129,before[b]+h*129+1));snprintf(name,sizeof(name),"response-%d.f64",m);CHECK(!write_file(argv[1],name,before[b],sizeof(before[b])));
    }
    for(size_t i=0;i<2*34*128;i++){keys[i]=10000;values[i]=-1000;}
    for(int b=0;b<3;b++){double after[12*129];for(int h=0;h<12;h++)CHECK(!span_predict(&programs[b],q+h*128,h/6,1.0f/sqrtf(128.0f),after+h*129,after+h*129+1));CHECK(!memcmp(before[b],after,sizeof(after)));span_host_free(&programs[b]);}
    puts("bank-smoke-ok; predictions unchanged after original operands overwritten");return 0;
}
