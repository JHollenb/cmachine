#include "isa_circuit.h"
#include "isa_fixture.h"
#include <math.h>
#include <stdio.h>

static void close_to(double a,double b,double tolerance){assert(isfinite(a)&&isfinite(b)&&fabs(a-b)<=tolerance*(1+fabs(a)+fabs(b)));}
static void transforms(void){
    double phase[]={1,-1,-1,1},scale[]={1,1,1,1},bias[]={0,0,0,0},spatial[4];
    CmPolyphaseAudit audit;assert(!cm_polyphase_audit(1,1,phase,scale,bias,&audit));assert(!cm_unpatchify(1,1,phase,scale,bias,spatial));
    assert(!memcmp(phase,spatial,sizeof(phase)));close_to(audit.phase_average_loss_l2,2,1e-15);close_to(audit.max_spectral_error,0,1e-14);
    double j[]={1,0,0,0,2,0},error[]={.1,10,.3},weights[]={1,.5},signed_effect[2],energy;
    assert(!cm_consumer_metric(2,3,j,error,weights,signed_effect,&energy));close_to(signed_effect[0],.1,1e-15);close_to(signed_effect[1],20,1e-15);close_to(energy,200.01,1e-15);
    double l,n,r;assert(!cm_gelu_split(-1,cm_gelu(-1),&l,&n,&r));close_to(l+n+r,cm_gelu(-1),1e-15);close_to(cm_gelu(1)-cm_gelu(-1),1,1e-15);
    close_to(cm_gelu_secant(1,1),cm_gelu_first(1),1e-15);close_to(cm_gelu_secant(1,1+1e-12),cm_gelu_first(1),1e-11);
    assert(cm_gelu_first(-1)<0);assert(!cm_gelu_split(1000,1000,&l,&n,&r));assert(isfinite(cm_gelu_second(1000)));
    printf("{\"check\":\"polyphase-checkerboard\",\"spatial_energy\":%.17g,\"phase_average_loss_l2\":%.17g,\"max_spectral_error\":%.17g}\n",audit.spatial_energy,audit.phase_average_loss_l2,audit.max_spectral_error);
}
int main(void){
    transforms();IsaFixture fixture;isa_fixture_init(&fixture);IsaModel *m=&fixture.model;
    IsaArch arch={0};arch.registers[0]=UINT64_C(0x12345678);arch.registers[1]=UINT64_C(0xabcdef01);
    IsaSession observed,native;isa_session_init(&observed,m,&arch);isa_session_init(&native,m,&arch);
    unsigned char window[]={0x48,0x89,0xd8};CmIsaCapture c={0};assert(!cm_isa_capture(&c,&observed,window,sizeof(window),2));
    assert(!isa_begin(&native,window,sizeof(window)));while(native.state.phase==1)assert(!isa_advance(&native));
    assert(!memcmp(&observed.state,&native.state,sizeof(IsaState)));
    double kernel[64*64],left[32*4],entry[4*32],paths[4];assert(!cm_isa_scan_kernel(&c,0,kernel));assert(!cm_isa_scan_cut(&c,0,32,left,entry));
    double factor_error=0,read_error=0;
    for(size_t t=32;t<64;t++)for(size_t s=0;s<32;s++){
        double value=0;for(size_t j=0;j<4;j++)value+=left[(t-32)*4+j]*entry[j*32+s];factor_error=fmax(factor_error,fabs(value-kernel[t*64+s]));close_to(value,kernel[t*64+s],1e-14);
    }
    for(size_t t=0;t<64;t++){
        double value=0;for(size_t s=0;s<=t;s++)value+=kernel[t*64+s]*c.rows[s].u[0];read_error=fmax(read_error,fabs(value-c.rows[t].read[0]));
    }
    assert(!cm_isa_scan_path(&c,0,8,31,paths));assert(!cm_isa_report(stdout,&c,0,8,31));
    double terms[144],consumer[48],projection[3];for(size_t i=0;i<48;i++)consumer[i]=m->reg.w[i];
    assert(!cm_isa_gelu_write(&c,31,consumer,terms,projection));
    CmIsaRegisterAccount account;assert(!cm_isa_register_account(&c,31,0,&account));
    close_to(account.copy_prior+account.head_bias+account.carry+account.gelu_affine+account.gelu_nonlinear+account.mlp_remainder+account.readout_remainder,observed.state.output.regs[0][31],1e-15);
    double closure=0;for(size_t i=0;i<48;i++){double sum=terms[i]+terms[48+i]+terms[96+i];closure=fmax(closure,fabs(sum-c.rows[31].write[i]));close_to(sum,c.rows[31].write[i],1e-14);}
    double plus,minus,no_op,earlier;IsaState before=observed.state;
    assert(!cm_isa_state_effect(&c,16,0,0,.02f,0,31,&plus));assert(!memcmp(&before,&observed.state,sizeof(before)));
    assert(!cm_isa_state_effect(&c,16,0,0,-.02f,0,31,&minus));assert(!cm_isa_state_effect(&c,16,0,0,0,0,31,&no_op));assert(no_op==0);
    assert(!cm_isa_state_effect(&c,16,0,0,.02f,0,8,&earlier));assert(earlier==0);assert(!memcmp(&before,&observed.state,sizeof(before)));
    printf("{\"check\":\"native-ISA-expansions\",\"capture_state_byte_identical\":true,\"cross_cut_factor_max_abs\":%.17g,\"ideal_kernel_native_read_max_abs\":%.17g,\"gelu_write_closure_max_abs\":%.17g,\"gelu_affine_consumer\":%.17g,\"gelu_nonlinear_consumer\":%.17g,\"gelu_arithmetic_remainder_consumer\":%.17g,\"state_edit_positive_effect\":%.17g,\"state_edit_negative_effect\":%.17g,\"earlier_bit_control\":%.17g,\"no_op_effect\":%.17g}\n",factor_error,read_error,closure,projection[0],projection[1],projection[2],plus,minus,earlier,no_op);
    const float doses[]={-4,-1,-.25f,0,.25f,1,4};
    for(size_t i=0;i<sizeof(doses)/sizeof(doses[0]);i++){
        double effect;assert(!cm_isa_state_effect(&c,31,0,0,doses[i],0,31,&effect));
        printf("{\"check\":\"native-ISA-nearby-state-dose\",\"cut\":31,\"target_bit\":31,\"dose\":%.9g,\"register_logit_delta\":%.17g}\n",doses[i],effect);
    }
    observed.state.scan[2][0][0]+=.01f;assert(cm_isa_scan_kernel(&c,0,kernel)==-1);observed.state=before;
    assert(!isa_commit(&observed));assert(cm_isa_scan_path(&c,0,8,31,paths)==-1);cm_isa_capture_free(&c);
    /* Capture failure restores the exact pre-instruction state, not just a
     * cursor; the callback never gains mutation authority. */
    isa_session_init(&observed,m,&arch);IsaState empty=observed.state;float *w=(float*)m->input.w;float saved=w[0];w[0]=NAN;
    assert(cm_isa_capture(&c,&observed,window,sizeof(window),0)==-2);assert(!memcmp(&observed.state,&empty,sizeof(empty)));w[0]=saved;
    isa_fixture_free(&fixture);puts("expanded circuit mechanics: passed");return 0;
}
