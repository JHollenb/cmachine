#define _POSIX_C_SOURCE 200809L
#include "machine.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32-n)); }
static void sha_block(uint32_t h[8], const unsigned char b[64]) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for (unsigned i=0;i<16;i++) w[i]=(uint32_t)b[4*i]<<24|(uint32_t)b[4*i+1]<<16|(uint32_t)b[4*i+2]<<8|b[4*i+3];
    for (unsigned i=16;i<64;i++) {
        uint32_t x=w[i-15],y=w[i-2];
        w[i]=w[i-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-7]+(rotr(y,17)^rotr(y,19)^(y>>10));
    }
    uint32_t a=h[0],b0=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
    for (unsigned i=0;i<64;i++) {
        uint32_t t=v+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
        uint32_t u=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b0)^(a&c)^(b0&c));
        v=g;g=f;f=e;e=d+t;d=c;c=b0;b0=a;a=t+u;
    }
    h[0]+=a;h[1]+=b0;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
}
void tcm_sha256(const void *data, size_t n, unsigned char out[32]) {
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const unsigned char *p=data; size_t full=n/64;
    for (size_t i=0;i<full;i++) sha_block(h,p+64*i);
    unsigned char tail[128]={0}; size_t r=n%64;
    memcpy(tail,p+64*full,r);tail[r]=0x80;
    size_t used=r<56?64:128; uint64_t bits=(uint64_t)n*8;
    for (unsigned i=0;i<8;i++) tail[used-1-i]=(unsigned char)(bits>>(8*i));
    sha_block(h,tail);if(used==128)sha_block(h,tail+64);
    for(unsigned i=0;i<8;i++)for(unsigned j=0;j<4;j++)out[4*i+j]=(unsigned char)(h[i]>>(24-8*j));
}
void tcm_hex(const unsigned char *p, size_t n, char *out) {
    static const char hex[]="0123456789abcdef";
    for(size_t i=0;i<n;i++){out[2*i]=hex[p[i]>>4];out[2*i+1]=hex[p[i]&15];}out[2*n]=0;
}
static uint32_t u32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static uint64_t u64(const unsigned char *p){return (uint64_t)u32(p)|(uint64_t)u32(p+4)<<32;}
static int fail(TcmModel *m,char *err,size_t cap,const char *msg){snprintf(err,cap,"%s",msg);tcm_unload(m);return -1;}
int tcm_load(TcmModel *m,const char *path,char *err,size_t cap) {
    memset(m,0,sizeof(*m));
    uint32_t endian=1;if(*(unsigned char*)&endian!=1)return fail(m,err,cap,"little-endian host required");
    int fd=open(path,O_RDONLY);struct stat st;
    if(fd<0)return fail(m,err,cap,"cannot open model bundle");
    if(fstat(fd,&st)||st.st_size<176){close(fd);return fail(m,err,cap,"truncated model bundle");}
    m->mapping_bytes=(size_t)st.st_size;
    m->mapping=mmap(NULL,m->mapping_bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(m->mapping==MAP_FAILED){m->mapping=NULL;return fail(m,err,cap,"model mmap failed");}
    const unsigned char *p=m->mapping;
    if(memcmp(p,"TCM1LE\0\0",8))return fail(m,err,cap,"unsupported bundle magic");
    unsigned char digest[32];tcm_sha256(p+40,m->mapping_bytes-40,digest);
    if(memcmp(digest,p+8,32))return fail(m,err,cap,"bundle SHA256 mismatch");
    tcm_hex(digest,32,m->digest);m->architecture=u32(p+40);m->count=u32(p+44);
    if(m->count==0||m->count>4096)return fail(m,err,cap,"invalid tensor count");
    for(unsigned i=0;i<16;i++)m->config[i]=u32(p+48+4*i);
    memcpy(m->identity,p+112,64);m->identity[64]=0;
    for(unsigned i=0;i<64;i++)if(!((m->identity[i]>='0'&&m->identity[i]<='9')||(m->identity[i]>='a'&&m->identity[i]<='f')))return fail(m,err,cap,"invalid model identity");
    size_t table_end=176+(size_t)m->count*120;
    if(table_end>m->mapping_bytes)return fail(m,err,cap,"truncated tensor table");
    m->tensor=calloc(m->count,sizeof(*m->tensor));
    if(!m->tensor)return fail(m,err,cap,"tensor descriptor allocation failed");
    uint64_t previous_end=table_end;
    for(unsigned i=0;i<m->count;i++) {
        const unsigned char *r=p+176+(size_t)i*120;TcmTensor *t=&m->tensor[i];
        memcpy(t->name,r,64);t->name[64]=0;t->rank=u32(r+64);t->dtype=u32(r+68);
        if(!memchr(r,0,64)||!t->name[0]||t->rank>4)return fail(m,err,cap,"invalid tensor descriptor");
        uint64_t elements=1;
        for(unsigned j=0;j<4;j++){
            t->dim[j]=u64(r+72+8*j);
            if(j<t->rank){if(!t->dim[j]||elements>UINT64_MAX/t->dim[j])return fail(m,err,cap,"invalid tensor dimensions");elements*=t->dim[j];}
        }
        uint64_t offset=u64(r+104);t->bytes=u64(r+112);
        unsigned item=t->dtype==TCM_F32?4:t->dtype==TCM_BF16?2:t->dtype==TCM_F64?8:0;
        if(!item||elements>UINT64_MAX/item||t->bytes!=elements*item||offset%item||offset<previous_end||offset>m->mapping_bytes||t->bytes>m->mapping_bytes-offset)
            return fail(m,err,cap,"invalid tensor extent or dtype");
        for(unsigned j=0;j<i;j++)if(!strcmp(t->name,m->tensor[j].name))return fail(m,err,cap,"duplicate tensor name");
        t->data=p+offset;previous_end=offset+t->bytes;
    }
    if(previous_end!=m->mapping_bytes)return fail(m,err,cap,"trailing bundle bytes");
    return 0;
}
void tcm_unload(TcmModel *m){if(m->mapping)munmap(m->mapping,m->mapping_bytes);free(m->tensor);memset(m,0,sizeof(*m));}
const TcmTensor *tcm_tensor(const TcmModel *m,const char *name){for(unsigned i=0;i<m->count;i++)if(!strcmp(m->tensor[i].name,name))return &m->tensor[i];return NULL;}
