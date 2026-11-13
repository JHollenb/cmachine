#include "qwen_circuit.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int number(const char *text,int *out){
    char *end;errno=0;long v=strtol(text,&end,10);
    if(errno||!text[0]||*end||v<0||v>INT_MAX)return -1;
    *out=(int)v;return 0;
}
int main(int argc,char **argv){
    if(argc<5){fprintf(stderr,"usage: tcm-circuit BUNDLE --target ID [--contrast ID] [--layer L] [--head H] [--source S] [--neuron N] [--frequency] [--readout-lines NEW_FILE] --tokens ID ...\n");return 2;}
    int target=-1,contrast=-1,layer=0,head=0,source=0,neuron=0,start=-1,frequency=0;const char *readout_lines=NULL;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--tokens")){start=i+1;break;}
        if(!strcmp(argv[i],"--frequency")){frequency=1;continue;}
        if(!strcmp(argv[i],"--readout-lines")){if(++i>=argc||readout_lines)return 2;readout_lines=argv[i];continue;}
        int *dest=NULL;
        if(!strcmp(argv[i],"--target"))dest=&target;
        else if(!strcmp(argv[i],"--contrast"))dest=&contrast;
        else if(!strcmp(argv[i],"--layer"))dest=&layer;
        else if(!strcmp(argv[i],"--head"))dest=&head;
        else if(!strcmp(argv[i],"--source"))dest=&source;
        else if(!strcmp(argv[i],"--neuron"))dest=&neuron;
        if(!dest||++i>=argc||number(argv[i],dest))return 2;
    }
    if(start<0||start>=argc||target<0)return 2;
    int n=argc-start;int *ids=malloc((size_t)n*sizeof(int));if(!ids)return 2;
    for(int i=0;i<n;i++)if(number(argv[start+i],ids+i)){free(ids);return 2;}
    TcmModel bundle;QwenModel model;char error[256];
    if(tcm_load(&bundle,argv[1],error,sizeof(error))){fprintf(stderr,"%s\n",error);free(ids);return 2;}
    if(qwen_model_init(&model,&bundle,error,sizeof(error))){fprintf(stderr,"%s\n",error);tcm_unload(&bundle);free(ids);return 2;}
    tcm_unload(&bundle);int status=2;QwenSession session={0};CmQwenCapture capture={0};CmQwenAdjoint adjoint={0};
    if(n>model.max_position||target>=model.vocab||contrast>=model.vocab||layer>=model.layers||head>=model.qheads||source>=n||neuron>=model.intermediate)goto done;
    for(int i=0;i<n;i++)if(ids[i]>=model.vocab)goto done;
    if(qwen_session_init(&session,&model,n))goto done;
    for(int i=0;i<n-1;i++)if(qwen_push(&session,ids[i]))goto done;
    if(cm_qwen_capture_token(&capture,&session,ids[n-1])||cm_qwen_consumer(&adjoint,&capture,target,contrast)||cm_qwen_report(stdout,&capture,&adjoint,layer,head,source,neuron))goto done;
    /* This CLI starts empty and applies no edits, so these IDs plus the exact
     * original bundle and numerical profile specify its root replay. */
    printf("{\"schema\":\"tcm-circuit-root-recipe-v1\",\"model_digest\":\"%.64s\",\"initial_state\":\"empty\",\"execution_profile\":\"original-FP32-CUDA-cuBLAS\",\"tokens\":[",model.digest);
    for(int i=0;i<n;i++)printf("%s%d",i?",":"",ids[i]);
    printf("],\"target\":%d,\"contrast\":%d,\"layer\":%d,\"head\":%d,\"source\":%d,\"neuron\":%d,\"standalone_operand_cache\":false}\n",target,contrast,layer,head,source,neuron);
    if(frequency){
        CmComplex *bins=calloc((size_t)n,sizeof(*bins));double native,ideal,remainder;
        if(!bins||cm_qwen_source_frequency(&capture,&adjoint,layer,head,bins,(size_t)n,&native,&ideal,&remainder)){free(bins);goto done;}
        /* First sixteen bins plus complete unlisted sum, never silent pruning. */
        int shown=n<16?n:16;double re=0,im=0;
        printf("{\"schema\":\"tcm-source-frequency-v1\",\"axis\":\"source-at-fixed-receiver\",\"forward\":\"unnormalized\",\"window\":%d,\"native_scalar\":%.17g,\"ideal_scalar\":%.17g,\"native_remainder\":%.17g,\"bins\":[",n,native,ideal,remainder);
        for(int i=0;i<n;i++){
            if(i<shown)printf("%s{\"k\":%d,\"re\":%.17g,\"im\":%.17g}",i?",":"",i,bins[i].re,bins[i].im);
            else{re+=bins[i].re;im+=bins[i].im;}
        }printf("],\"unlisted_sum\":{\"re\":%.17g,\"im\":%.17g}}\n",re,im);free(bins);
    }
    if(readout_lines){
        FILE *f=fopen(readout_lines,"wx");if(!f)goto done;
        int error=cm_qwen_write_readout_lines(f,&capture);if(fclose(f))error=-1;if(error)goto done;
    }
    status=0;
done:
    if(status)fprintf(stderr,"circuit inspection incomplete; no semantic result inferred\n");
    cm_qwen_adjoint_free(&adjoint);cm_qwen_capture_free(&capture);if(session.model)qwen_session_free(&session);qwen_model_free(&model);free(ids);return status;
}
