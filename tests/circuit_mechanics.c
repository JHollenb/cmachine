/* Reuse the unchanged tiny native executor fixture and lifecycle assertions. */
#define main lifecycle_main
#include "state_mechanics.c"
#undef main
#include "qwen_circuit.h"

static void close_to(double a,double b,double eps){if(!(isfinite(a)&&isfinite(b)&&fabs(a-b)<=eps*(1+fabs(a)+fabs(b))))fprintf(stderr,"mismatch %.17g %.17g tolerance %.17g\n",a,b,eps);assert(isfinite(a)&&isfinite(b)&&fabs(a-b)<=eps*(1+fabs(a)+fabs(b)));}
static double dot(const double *a,const double *b,size_t n){double x=0;for(size_t i=0;i<n;i++)x+=a[i]*b[i];return x;}
static void math_checks(void){
    double x[]={.5,-.7,1.2},g[]={.8,1.1,-.4},u[]={-.2,.7,1.3},a[]={.3,-.2,.6},b[]={-.4,.5,.2};
    double j[3],vjp[3],changed[3],finite[3];
    assert(!cm_rms_jvp(3,x,g,1e-6,a,j));assert(!cm_rms_vjp(3,x,g,1e-6,b,vjp));close_to(dot(j,b,3),dot(a,vjp,3),1e-14);
    for(int i=0;i<3;i++)changed[i]=x[i]+.4*a[i];assert(!cm_rms_finite(3,x,changed,g,1e-6,finite));
    double r0=1/sqrt(dot(x,x,3)/3+1e-6),r1=1/sqrt(dot(changed,changed,3)/3+1e-6);
    for(int i=0;i<3;i++)close_to(finite[i],g[i]*(r1*changed[i]-r0*x[i]),1e-14);
    double fixed,radial,total;assert(!cm_writer_reader(3,b,a,x,g,1e-6,&fixed,&radial,&total));close_to(total,dot(j,b,3),1e-14);close_to(total,r0*fixed+radial,1e-14);
    CmNeuronDelta nd;assert(!cm_neuron_delta(-.7,.2,cm_silu(-.7)*.2,1.1,-.3,cm_silu(1.1)*-.3,&nd));close_to(nd.total,nd.gate+nd.up+nd.interaction+nd.remainder,1e-14);
    close_to(cm_silu_secant(1,1),cm_silu_first(1),1e-15);
    close_to(cm_silu_secant(1,1+1e-12),cm_silu_first(1),1e-11);
    close_to(cm_silu_secant(1000,1001),1,1e-14);close_to(cm_silu_secant(-1000,-999),0,1e-14);
    close_to(cm_silu(.7)*.2+cm_silu(-.7)*-.2,.7*.2,1e-15);
    double even,odd;assert(!cm_neuron_even_odd(.7,.2,&even,&odd));close_to(even+odd,cm_silu(.7)*.2,1e-14);close_to(even-odd,cm_silu(-.7)*-.2,1e-14);
    CmReadGeometry geometry;assert(!cm_read_geometry(3,g,u,&geometry));close_to(geometry.eigen_plus+geometry.eigen_minus,dot(g,u,3),1e-14);
    double curvature;assert(!cm_neuron_curvature(3,g,u,x,a,b,&curvature));double e=1e-4,F[4];
    for(int arm=0;arm<4;arm++){double z[3];for(int i=0;i<3;i++)z[i]=x[i]+((arm&1)?1:-1)*e*a[i]+((arm&2)?1:-1)*e*b[i];F[arm]=cm_silu(dot(g,z,3))*dot(u,z,3);}
    close_to(curvature,(F[3]-F[1]-F[2]+F[0])/(4*e*e),1e-7);
    double p[]={.2,.3,.5},q[]={.2,-.1},keys[]={.1,.3,-.4,.2,.7,-.5},values[]={.7,.2,-.3,.4,.1,-.2},att[2]={0},lambda[]={.3,-.8};
    for(int j0=0;j0<3;j0++)for(int i=0;i<2;i++)att[i]+=p[j0]*values[2*j0+i];
    double dq[2],dk[6],dv[6],dl[3];assert(!cm_attention_vjp(3,2,p,q,keys,values,att,lambda,dq,dk,dv,dl));close_to(dl[0]+dl[1]+dl[2],0,1e-14);
    double logchange[]={1000,999,998},valchange[]={.1,.2,.3,.4,.5,.6},delta[2];assert(!cm_attention_finite(3,2,p,values,logchange,valchange,delta));
    double den=0,pp[3];for(int z=0;z<3;z++){pp[z]=p[z]*exp(logchange[z]-1000);den+=pp[z];}
    for(int i=0;i<2;i++){double want=-att[i];for(int z=0;z<3;z++)want+=pp[z]/den*(values[2*z+i]+valchange[2*z+i]);close_to(delta[i],want,1e-13);}
    CmComplex sig[5],ft[5],back[5],sq[5],sqft[5],conv[5];
    for(int i=0;i<5;i++){sig[i]=(CmComplex){sin(i*.7),0};sq[i]=(CmComplex){sig[i].re*sig[i].re,0};}
    assert(!cm_dft(5,sig,ft,0));assert(!cm_dft(5,ft,back,1));assert(!cm_dft(5,sq,sqft,0));assert(!cm_circular_convolution(5,ft,ft,conv));
    for(int i=0;i<5;i++){close_to(back[i].re,sig[i].re,1e-14);close_to(back[i].im,0,1e-14);close_to(sqft[i].re,conv[i].re/5,1e-14);close_to(sqft[i].im,conv[i].im/5,1e-14);}
    double eye[]={1,0,0,0,1,0,0,0,1},comm;assert(!cm_shift_commutator(3,eye,&comm));assert(comm==0);eye[3]=.1;assert(!cm_shift_commutator(3,eye,&comm));assert(comm>0);
    double lattice[]={2,5,6,12,1,4,5,11},original[8];memcpy(original,lattice,sizeof(lattice));assert(!cm_coalitions(3,lattice,0));close_to(lattice[3],3,1e-14);assert(!cm_coalitions(3,lattice,1));assert(!memcmp(original,lattice,sizeof(lattice)));
    int args[]={2,3},res0=4,res1=5;CmInterfaceRow row0={"model/profile/program/consumer/horizon","left",args,sizeof(args),&res0,sizeof(res0)},row1=row0;row1.occurrence="right";row1.result=&res1;CmInterfaceRelation relation;
    assert(!cm_interface_compare(&row0,&row1,&relation)&&relation==CM_INTERFACE_WITNESS);row1.scope="other";assert(!cm_interface_compare(&row0,&row1,&relation)&&relation==CM_INTERFACE_OTHER_SCOPE);
    assert(cm_rms_jvp(0,x,g,1e-6,a,j)==-1);assert(cm_dft(5,sig,sig,0)==-1);p[0]=NAN;assert(cm_attention_finite(3,2,p,values,logchange,valchange,delta)==-1);
}
static void native_checks(void){
    QwenModel m;fixture(&m);QwenSession a,b;assert(!qwen_session_init(&a,&m,12));assert(!qwen_session_init(&b,&m,12));prefix(&a);prefix(&b);
    QwenCheckpoint parent;assert(!qwen_checkpoint(&b,&parent));
    CmQwenCapture capture={0};assert(!cm_qwen_capture_token(&capture,&a,9));assert(!qwen_push(&b,9));equal(&a,&b);
    CmQwenAdjoint adjoint={0};assert(!cm_qwen_consumer(&adjoint,&capture,3,4));assert(adjoint.ready);
    double intercept[16],slope[16];CmReadoutNorm readout_norm;assert(!cm_qwen_final_readout_lines(&capture,intercept,slope,&readout_norm));
    const CmQwenLayer *last=capture.layer+m.layers-1;
    for(int token=0;token<16;token++){
        double a0=0,b0=0;for(int i=0;i<8;i++){a0+=(double)m.embedding[token*8+i]*m.norm[i]*last->receiver[i];b0+=(double)m.embedding[token*8+i]*m.norm[i]*last->write[i];}
        close_to(intercept[token],a0,1e-14);close_to(slope[token],b0,1e-14);
    }
    assert(readout_norm.A>0&&readout_norm.C>=0);
    const char *line_path=getenv("CM_READOUT_LINES");
    if(line_path){FILE *f=fopen(line_path,"wx");assert(f);assert(!cm_qwen_write_readout_lines(f,&capture));assert(!fclose(f));}
    const double *lambda=adjoint.layer[0].input;double orthogonal[8]={0},parallel[8];double length=0;
    for(int i=0;i<8;i++)length=hypot(length,lambda[i]);assert(length>0);
    orthogonal[0]=10*lambda[1];orthogonal[1]=-10*lambda[0];for(int i=0;i<8;i++)parallel[i]=.01*lambda[i]/length;
    CmOmissionMetric orth,along;assert(!cm_qwen_omission(&capture,&adjoint,0,"input",orthogonal,8,&orth));assert(!cm_qwen_omission(&capture,&adjoint,0,"input",parallel,8,&along));
    close_to(orth.signed_effect,0,1e-14);close_to(along.signed_effect,.01*length,1e-14);
    printf("{\"check\":\"consumer-weighted-omissions\",\"orthogonal_l2\":%.17g,\"orthogonal_signed_effect\":%.17g,\"parallel_l2\":%.17g,\"parallel_signed_effect\":%.17g}\n",orth.omission_l2,orth.signed_effect,along.omission_l2,along.signed_effect);
    for(int layer=0;layer<m.layers;layer++)for(int neuron=0;neuron<m.intermediate;neuron++){
        double column[8];assert(!cm_qwen_parameter(&m,layer,"down",(size_t)neuron,1,column,8));close_to(dot(column,adjoint.layer[layer].output,8),adjoint.layer[layer].product[neuron],1e-14);
    }
    /* Independent central differences through the actual full native suffix,
     * including attention current-row K/V. FP32 rounding remains visible. */
    double worst=0;float first_input[8];gpu_read(first_input,m.embedding+9*8,sizeof(first_input));
    for(int coord=0;coord<8;coord++){
        double result[2];for(int arm=0;arm<2;arm++){
            assert(!qwen_restore(&b,&parent));assert(!qwen_begin_token(&b,9));float edit[8];memcpy(edit,first_input,sizeof(edit));edit[coord]+=(arm?1:-1)*.002f;gpu_write(b.x,edit,sizeof(edit));complete(&b);
            float logits[16];gpu_read(logits,b.logits,sizeof(logits));result[arm]=(double)logits[3]-logits[4];
        }
        double fd=(result[1]-result[0])/.004,actual=adjoint.layer[0].input[coord];double error=fabs(fd-actual);if(error>worst)worst=error;close_to(fd,actual,5e-3);
    }
    double fixed,radial,total;assert(!cm_qwen_writer_reader(&m,0,2,1,"gate",3,capture.layer[1].receiver,&fixed,&radial,&total));assert(isfinite(total));
    double static_link;assert(!cm_qwen_static_link(&m,0,2,1,"gate",3,&static_link));assert(static_link==fixed);CmReadGeometry geometry;assert(!cm_qwen_read_geometry(&m,1,3,&geometry));assert(geometry.gate_norm>0&&geometry.up_norm>0);
    assert(!cm_qwen_report(stdout,&capture,&adjoint,1,2,1,3));
    printf("{\"check\":\"native-tiny-consumer-central-difference\",\"max_absolute_error\":%.17g,\"capture_host_bytes\":%zu,\"adjoint_host_bytes\":%zu}\n",worst,capture.host_bytes,adjoint.host_bytes);
    CmQwenAdjoint same={0};assert(!cm_qwen_consumer(&same,&capture,3,3));for(int i=0;i<8;i++)assert(same.layer[0].input[i]==0);cm_qwen_adjoint_free(&same);
    CmComplex bins[4];double native,ideal,remainder;
    assert(!cm_qwen_source_frequency(&capture,&adjoint,1,2,bins,4,&native,&ideal,&remainder));
    double sum=0,imag=0;for(int i=0;i<4;i++){sum+=bins[i].re;imag+=bins[i].im;}close_to(sum,ideal,1e-14);close_to(imag,0,1e-14);close_to(sum+remainder,native,1e-14);
    printf("{\"check\":\"consumer-source-frequency\",\"coefficient_sum\":%.17g,\"native_scalar\":%.17g,\"native_remainder\":%.17g}\n",sum,native,remainder);
    /* Source gradients traverse every upper native consumer. Two query heads
     * share KV group one; both must contribute to the same source edit. */
    double source_error=0;
    for(int key_or_value=0;key_or_value<2;key_or_value++){
        double prediction=0;
        for(int head=2;head<4;head++)prediction+=key_or_value?capture.layer[1].probability[head*4+1]*adjoint.layer[1].attention[head*2]:adjoint.layer[1].routing[head*4+1]*capture.layer[1].query[head*2]/sqrt(2.0);
        double outcome[2];
        for(int arm=0;arm<2;arm++){
            prefix(&b);size_t offset=((size_t)b.cache_slots[1]*m.kvheads+1)*b.capacity*m.head+1*m.head;
            float *port=(key_or_value?b.values:b.keys)+offset;float value;gpu_read(&value,port,sizeof(value));value+=(arm?1:-1)*.002f;gpu_write(port,&value,sizeof(value));
            assert(!qwen_push(&b,9));float logits[16];gpu_read(logits,b.logits,sizeof(logits));outcome[arm]=(double)logits[3]-logits[4];
        }
        double fd=(outcome[1]-outcome[0])/.004;source_error=fmax(source_error,fabs(fd-prediction));close_to(fd,prediction,1e-3);
    }
    printf("{\"check\":\"shared-KV-source-to-consumer\",\"max_absolute_error\":%.17g}\n",source_error);
    /* Analysis has not changed native logits/cache/control. */
    assert(!qwen_reset(&b));for(int id=2;id<=4;id++)assert(!qwen_push(&b,id));assert(!qwen_push(&b,9));equal(&a,&b);
    /* Compatible same-clock endpoint accounting with actual changed operands. */
    prefix(&b);CmQwenCapture changed={0};assert(!cm_qwen_capture_token(&changed,&b,10));
    double terms[32],attention_terms[8];CmQwenFiniteSummary summary;
    assert(!cm_qwen_mlp_finite(&capture,&changed,1,adjoint.layer[1].output,terms,&summary));
    for(int i=0;i<8;i++)close_to(terms[i]+terms[8+i]+terms[16+i]+terms[24+i],changed.layer[1].write[i]-capture.layer[1].write[i],1e-14);
    close_to(summary.gate+summary.up+summary.interaction+summary.remainder,summary.total,1e-14);
    assert(!cm_qwen_attention_finite(&capture,&changed,1,2,attention_terms));
    for(int i=0;i<2;i++)close_to(attention_terms[i]+attention_terms[2+i]+attention_terms[4+i]+attention_terms[6+i],changed.layer[1].attention[4+i]-capture.layer[1].attention[4+i],1e-14);
    double dn[8],secant[8];for(int i=0;i<8;i++)dn[i]=changed.layer[1].normalized[i]-capture.layer[1].normalized[i];assert(!cm_qwen_mlp_secant_action(&capture,&changed,1,dn,secant));
    printf("{\"check\":\"finite-native-MLP\",\"gate\":%.17g,\"up\":%.17g,\"interaction\":%.17g,\"arithmetic_remainder\":%.17g,\"total\":%.17g,\"normalization_remainder_max_abs\":%.17g,\"projection_remainder_max_abs\":%.17g,\"product_secant_remainder_max_abs\":%.17g}\n",summary.gate,summary.up,summary.interaction,summary.remainder,summary.total,summary.normalization_remainder,summary.projection_remainder,summary.secant_remainder);
    assert(!cm_qwen_mlp_finite(&capture,&capture,1,NULL,terms,&summary));for(int i=0;i<32;i++)assert(terms[i]==0);
    cm_qwen_capture_free(&changed);
    assert(!qwen_push(&a,6));CmQwenAdjoint stale={0};assert(cm_qwen_consumer(&stale,&capture,3,4)==-1);
    cm_qwen_adjoint_free(&adjoint);cm_qwen_capture_free(&capture);
    QwenSession before=b;b.observer=observer;before=b;assert(cm_qwen_capture_token(&capture,&b,4)==-1);assert(!memcmp(&before,&b,sizeof(b)));b.observer=NULL;
    QwenLexicalProgram lexical={0};b.lexical_program=&lexical;before=b;assert(cm_qwen_capture_token(&capture,&b,4)==-1);assert(!memcmp(&before,&b,sizeof(b)));b.lexical_program=NULL;
    QwenQkvProgram qkv={0};b.qkv_programs[0]=&qkv;before=b;assert(cm_qwen_capture_token(&capture,&b,4)==-1);assert(!memcmp(&before,&b,sizeof(b)));b.qkv_programs[0]=NULL;
    b.cache_mask=1;before=b;assert(cm_qwen_capture_token(&capture,&b,4)==-1);assert(!memcmp(&before,&b,sizeof(b)));b.cache_mask=0;
    prefix(&b);QwenSession good=b;float saved=m.layer[0].q[0];m.layer[0].q[0]=NAN;
    assert(cm_qwen_capture_token(&capture,&b,9)==-2);assert(!capture.ready&&!b.observer&&b.position==good.position&&b.lineage_id==good.lineage_id&&b.pending==good.pending);
    m.layer[0].q[0]=saved;prefix(&a);assert(!qwen_push(&a,9));assert(!qwen_push(&b,9));equal(&a,&b);
    qwen_session_free(&a);qwen_session_free(&b);fixture_free(&m);
}
int main(void){math_checks();native_checks();puts("circuit mechanics: passed");return 0;}
