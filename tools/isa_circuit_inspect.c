#include "isa_circuit.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int integer(const char *text,unsigned *out){
    if(!text||!*text||*text=='-')return -1;
    char *end;errno=0;unsigned long v=strtoul(text,&end,10);if(errno||*end||v>UINT_MAX)return -1;
    *out=(unsigned)v;return 0;
}
static int nibble(char x){if(x>='0'&&x<='9')return x-'0';if(x>='a'&&x<='f')return x-'a'+10;if(x>='A'&&x<='F')return x-'A'+10;return -1;}
int main(int argc,char **argv){
    if(argc<4){fprintf(stderr,"usage: tcm-isa-circuit BUNDLE --bytes HEX [--layer 0..2] [--channel 0..95] [--source BIT] [--receiver BIT] [--cut ROWS --dose FLOAT --register 0..7]\n");return 2;}
    const char *hex=NULL;unsigned layer=2,channel=0,source=0,receiver=63,cut=32,reg=0;float dose=0;
    for(int i=2;i<argc;i++){
        const char *key=argv[i];if(++i>=argc)return 2;
        if(!strcmp(key,"--bytes")){hex=argv[i];continue;}
        if(!strcmp(key,"--dose")){char *end;errno=0;dose=strtof(argv[i],&end);if(errno||*end||!*argv[i]||!isfinite(dose))return 2;continue;}
        unsigned *p=NULL;
        if(!strcmp(key,"--layer"))p=&layer;else if(!strcmp(key,"--channel"))p=&channel;
        else if(!strcmp(key,"--source"))p=&source;else if(!strcmp(key,"--receiver"))p=&receiver;
        else if(!strcmp(key,"--cut"))p=&cut;else if(!strcmp(key,"--register"))p=&reg;
        if(!p||integer(argv[i],p))return 2;
    }
    if(!hex||strlen(hex)>30||strlen(hex)%2||layer>=3||channel>=96||source>receiver||receiver>=64||cut<1||cut>=64||reg>=8)return 2;
    size_t count=strlen(hex)/2;unsigned char window[15]={0};
    for(size_t i=0;i<count;i++){int a=nibble(hex[2*i]),b=nibble(hex[2*i+1]);if(a<0||b<0)return 2;window[i]=(unsigned char)(16*a+b);}
    TcmModel bundle;IsaModel model;char error[256];
    if(tcm_load(&bundle,argv[1],error,sizeof(error))){fprintf(stderr,"%s\n",error);return 2;}
    if(isa_model_init(&model,&bundle,error,sizeof(error))){fprintf(stderr,"%s\n",error);tcm_unload(&bundle);return 2;}
    IsaSession session;IsaArch root={0};isa_session_init(&session,&model,&root);CmIsaCapture capture={0};int status=2;
    if(cm_isa_capture(&capture,&session,window,count,layer)||cm_isa_report(stdout,&capture,channel,source,receiver))goto done;
    double effect;if(cm_isa_state_effect(&capture,cut,channel,0,dose,reg,receiver,&effect))goto done;
    if(layer==2){
        CmIsaRegisterAccount a;if(cm_isa_register_account(&capture,receiver,reg,&a))goto done;
        printf("{\"schema\":\"tcm-isa-register-account-v1\",\"register\":%u,\"bit\":%u,\"copy_prior\":%.17g,\"head_bias\":%.17g,\"carry\":%.17g,\"gelu_affine\":%.17g,\"gelu_nonlinear\":%.17g,\"mlp_remainder\":%.17g,\"readout_remainder\":%.17g,\"native_logit\":%.17g}\n",reg,receiver,a.copy_prior,a.head_bias,a.carry,a.gelu_affine,a.gelu_nonlinear,a.mlp_remainder,a.readout_remainder,a.native_logit);
    }
    printf("{\"schema\":\"tcm-isa-root-recipe-v1\",\"model_digest\":\"%.64s\",\"window_hex\":\"%s\",\"initial_arch\":\"zero-registers-flags-rip-instruction\",\"profile\":\"original-C-FP32-libm\",\"layer\":%u,\"channel\":%u,\"state_coordinate\":0,\"cut_rows\":%u,\"dose\":%.9g,\"consumer_register\":%u,\"consumer_bit\":%u,\"native_register_logit_delta\":%.17g,\"parent_restored\":true}\n",bundle.digest,hex,layer,channel,cut,dose,reg,receiver,effect);
    status=0;
done:
    if(status)fprintf(stderr,"ISA circuit inspection incomplete\n");
    cm_isa_capture_free(&capture);tcm_unload(&bundle);return status;
}
