#include "byte_bank.h"
#include <stdio.h>
#include <assert.h>
int main(int argc,char **argv){
    assert(argc==3);float k[8704]={0},v[8704]={0},q[128]={0};ByteProgram p;
    for(int row=0;row<68;row++){
        k[row*128]=127;k[row*128+1]=0.5f;k[row*128+2]=-0.5f;k[row*128+3]=1.5f;k[row*128+4]=-1.5f;
        for(int d=5;d<128;d++)k[row*128+d]=(float)((row+d)%23-11)*0.2f;
        if(row)for(int d=0;d<128;d++)v[row*128+d]=(float)((row*3+d)%31-15)*0.3f;
    }
    assert(!byte_construct(&p,k,v));assert(p.scales[0]==1&&p.scales[68]==1);assert(p.keys[1]==1&&p.keys[2]==-1&&p.keys[3]==2&&p.keys[4]==-2);
    for(int d=0;d<128;d++)assert(p.values[d]==0);
    double first[12*129],second[12*129];for(int d=0;d<128;d++)q[d]=(float)(d%7-3)*0.02f;
    for(int h=0;h<12;h++)assert(!byte_predict(&p,q,h/6,1.0f/sqrtf(128),first+h*129,first+h*129+1));
    memset(k,0,sizeof(k));memset(v,0,sizeof(v));
    for(int h=0;h<12;h++)assert(!byte_predict(&p,q,h/6,1.0f/sqrtf(128),second+h*129,second+h*129+1));assert(!memcmp(first,second,sizeof(first)));
    unsigned char raw[17984];memcpy(raw,"TCMBYTE1",8);uint32_t header[]={1,2,128,34,8,0};memcpy(raw+8,header,24);memcpy(raw+32,p.scales,544);memcpy(raw+576,p.keys,8704);memcpy(raw+9280,p.values,8704);
    FILE *f=fopen(argv[1],"wb");assert(f&&fwrite(raw,1,sizeof(raw),f)==sizeof(raw)&&!fclose(f));f=fopen(argv[2],"wb");assert(f&&fwrite(first,1,sizeof(first),f)==sizeof(first)&&!fclose(f));
    k[0]=NAN;assert(byte_construct(&p,k,v)==-1);return 0;
}
