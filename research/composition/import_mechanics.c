/* No model: exercise the real import against a malloc-backed GPU stub. */
#define main composition_driver_main
#include "driver.c"
#undef main
int main(int argc,char **argv){
    REQUIRE(argc==4);QwenModel m={0};m.hidden=1536;m.intermediate=8960;QwenSubspace p;REQUIRE(!qwen_subspace_init(&p,&m));import_bank(&p,argv[1]);char fingerprint[65];bank_fingerprint(&p,fingerprint);REQUIRE(!strcmp(fingerprint,argv[3]));
    int d=m.hidden,h=m.intermediate;size_t n=(size_t)3*d+4*h,offset=0;float *base=malloc(n*4),*extra=malloc(((size_t)d+2*h)*4);double *qr=malloc(((size_t)2*d+4)*8),*qe=malloc(((size_t)d+3)*8);REQUIRE(base&&extra&&qr&&qe);
    for(int j=0;j<2;j++){gpu_read(base+offset,p.basis[j],(size_t)d*4);offset+=d;}
    for(int j=0;j<2;j++){gpu_read(base+offset,p.gate[j],(size_t)h*4);offset+=h;}
    for(int j=0;j<2;j++){gpu_read(base+offset,p.up[j],(size_t)h*4);offset+=h;}
    gpu_read(base+offset,p.baseline_write,(size_t)d*4);REQUIRE(offset+d==n);
    gpu_read(extra,p.basis[2],(size_t)d*4);gpu_read(extra+d,p.gate[2],(size_t)h*4);gpu_read(extra+d+h,p.up[2],(size_t)h*4);
    memcpy(qr,p.q0,(size_t)d*8);memcpy(qr+d,p.q1,(size_t)d*8);qr[2*d]=p.r00;qr[2*d+1]=p.r01;qr[2*d+2]=p.r11;qr[2*d+3]=p.condition;
    memcpy(qe,p.q2,(size_t)d*8);qe[d]=p.r02;qe[d+1]=p.r12;qe[d+2]=p.r22;
    save(argv[2],"reuse-program.f32",base,n*4);save(argv[2],"enrich-program.f32",extra,((size_t)d+2*h)*4);save(argv[2],"reuse-qr.f64",qr,((size_t)2*d+4)*8);save(argv[2],"enrich-qr.f64",qe,((size_t)d+3)*8);
    free(base);free(extra);free(qr);free(qe);qwen_subspace_free(&p);puts("import layout retained every frozen operand byte; no model evaluated");return 0;
}
