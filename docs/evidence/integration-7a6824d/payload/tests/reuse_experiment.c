#define _POSIX_C_SOURCE 200809L
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"reuse driver %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#include "assay_io.h"
#include "formation_io.h"

static void program(const QwenSubspace *p,const char *directory,char hash[65]){
    int d=p->model->hidden,h=p->model->intermediate;size_t count=(size_t)3*d+(size_t)4*h,offset=0;float *raw=malloc(count*4);REQUIRE(raw);
    for(int j=0;j<2;j++){gpu_read(raw+offset,p->basis[j],(size_t)d*4);offset+=d;}
    for(int j=0;j<2;j++){gpu_read(raw+offset,p->gate[j],(size_t)h*4);offset+=h;}
    for(int j=0;j<2;j++){gpu_read(raw+offset,p->up[j],(size_t)h*4);offset+=h;}
    gpu_read(raw+offset,p->baseline_write,(size_t)d*4);offset+=d;REQUIRE(offset==count);for(size_t j=0;j<count;j++)REQUIRE(isfinite(raw[j]));
    size_t qcount=(size_t)2*d+4;double *qr=malloc(qcount*8);REQUIRE(qr);memcpy(qr,p->q0,(size_t)d*8);memcpy(qr+d,p->q1,(size_t)d*8);qr[2*d]=p->r00;qr[2*d+1]=p->r01;qr[2*d+2]=p->r11;qr[2*d+3]=p->condition;
    size_t bytes=count*4+qcount*8;unsigned char *joined=malloc(bytes);REQUIRE(joined);memcpy(joined,raw,count*4);memcpy(joined+count*4,qr,qcount*8);digest(joined,bytes,hash);
    if(directory){save(directory,"reuse-program.f32",raw,count*4);save(directory,"reuse-qr.f64",qr,qcount*8);}free(joined);free(raw);free(qr);
}
static void receiver(const QwenSubspace *p,const char *directory,const char *name){
    const float *arrays[]={p->receiver,p->reconstruction,p->omitted,p->entry,p->attention_write};const char *suffix[]={"receiver","reconstruction","omitted","entry","attention-write"};char file[128];
    for(int j=0;j<5;j++){float *h=read_floats(arrays[j],(size_t)p->model->hidden);snprintf(file,sizeof(file),"%s-%s.f32",name,suffix[j]);save(directory,file,h,(size_t)p->model->hidden*4);free(h);}
}
static void extension(const QwenSubspace *p,const char *directory,char hash[65]){
    REQUIRE(p->rank==3);int d=p->model->hidden,h=p->model->intermediate;size_t n=(size_t)d+2*h,q=(size_t)d+3,bytes=n*4+q*8;
    float *raw=malloc(n*4);double *qr=malloc(q*8);unsigned char *joined=malloc(bytes);REQUIRE(raw&&qr&&joined);
    gpu_read(raw,p->basis[2],(size_t)d*4);gpu_read(raw+d,p->gate[2],(size_t)h*4);gpu_read(raw+d+h,p->up[2],(size_t)h*4);
    memcpy(qr,p->q2,(size_t)d*8);qr[d]=p->r02;qr[d+1]=p->r12;qr[d+2]=p->r22;
    for(size_t j=0;j<n;j++)REQUIRE(isfinite(raw[j]));for(size_t j=0;j<q;j++)REQUIRE(isfinite(qr[j]));
    memcpy(joined,raw,n*4);memcpy(joined+n*4,qr,q*8);digest(joined,bytes,hash);
    if(directory){save(directory,"enrich-program.f32",raw,n*4);save(directory,"enrich-qr.f64",qr,q*8);}free(raw);free(qr);free(joined);
}
static void finish(QwenSession *s){while(s->layer<s->model->layers)REQUIRE(!qwen_advance_layer(s));REQUIRE(!qwen_finish_token(s));}
static void upload_port(const char *directory,const char *name,float *device,int count){
    char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"rb");REQUIRE(f);float *h=malloc((size_t)count*4);REQUIRE(h);
    REQUIRE(fread(h,4,(size_t)count,f)==(size_t)count&&fgetc(f)==EOF&&!ferror(f)&&!fclose(f));for(int j=0;j<count;j++)REQUIRE(isfinite(h[j]));gpu_write(device,h,(size_t)count*4);free(h);
}
static void split_joint(FILE *f,QwenSession *s,QwenSubspace *p,const char *directory,const char *group,const char *left,const char *right){
    /* Pure nonlinear/down diagnostic from captured projections. No token,
     * residual, KV, pending policy or program update occurs here. */
    QwenSession before=*s;char prefix[65],after[65],name[128];prefix_digest(s,s->position,prefix);int h=s->model->intermediate,d=s->model->hidden;
    for(int k=0;k<2;k++){
        float *target=k?s->up:s->gate;const char *channel=k?"up":"gate";
        snprintf(name,sizeof(name),"%s-project-mlp_%s.f32",group,channel);upload_port(directory,name,target,h);
        for(int j=0;j<2;j++){snprintf(name,sizeof(name),"%s-%s-correction-%s.f32",group,j?right:left,channel);upload_port(directory,name,p->correction_gate,h);gpu_add(target,p->correction_gate,h);}
        float *host=read_floats(target,(size_t)h);snprintf(name,sizeof(name),"split-joint-mlp_%s.f32",channel);save(directory,name,host,(size_t)h*4);free(host);
    }
    gpu_swiglu(s->gate,s->up,h);gpu_matvec(s->model->gpu,s->model->layer[16].down,h,d,s->gate,s->projected);
    float *host=read_floats(s->projected,(size_t)d);save(directory,"split-joint-mlp_return.f32",host,(size_t)d*4);free(host);
    prefix_digest(s,s->position,after);REQUIRE(!strcmp(prefix,after)&&!memcmp(&before,s,sizeof(before)));
    fprintf(f,",\"split_joint\":{\"gate_up_matvecs\":0,\"down_matvecs\":1,\"session_metadata_unchanged\":true,\"kv_prefix_sha256\":\"%s\",\"state_advanced\":false}",prefix);
}
static void append_suffix(QwenSession *s,const int *ids,int count,int already_end,int omit_last){int skip=already_end==s->model->eos?1:0;REQUIRE(!skip||ids[0]==s->model->eos);int observed=s->observe_position;s->observe_position=-1;for(int j=skip;j<count-omit_last;j++)REQUIRE(!qwen_push(s,ids[j]));s->observe_position=observed;}
#include "writer_io.h"
static int row(FILE *f,QwenSession *s,QwenSubspace *p,QwenSubspaceMode mode,const char *name,const char *group,int final_id,int *common,Capture *capture,const int *questions[2],const int question_lengths[2],int red,int blue,float *calibration_operand,Formation *form){
    uint64_t branch;REQUIRE(!qwen_stack_push(s,&branch));int position=s->position;char prefix[65],after[65],file[128];prefix_digest(s,position,prefix);
    char layer15_name[128];capture->name=name;s->observe_position=position;
    if(form){REQUIRE(snprintf(layer15_name,sizeof(layer15_name),"%s-l15",name)<(int)sizeof(layer15_name));form->capture=(Capture){capture->directory,layer15_name};s->observer=formation_port;s->observer_context=form;s->observe_layer=15;}
    REQUIRE(!qwen_begin_token(s,final_id));while(s->layer<16)REQUIRE(!qwen_advance_layer(s));QwenSubspaceStats stats;
    if(form){formation_prepare(form,s,p,capture);s->observer=port;s->observer_context=capture;s->observe_layer=16;}
    if(mode==QWEN_SUBSPACE_OPERAND_CORRECT)REQUIRE(!qwen_advance_layer_subspace_operand(s,p,form->select_new?form->new_write:form->inherited,(size_t)s->model->hidden,&stats));
    else REQUIRE(!qwen_advance_layer_subspace(s,p,mode,&stats));
    receiver(p,capture->directory,name);if(calibration_operand)gpu_copy(calibration_operand,p->omitted,(size_t)s->model->hidden*4);if(mode==QWEN_SUBSPACE_COMPILE)program(p,capture->directory,after);finish(s);
    if(mode==QWEN_SUBSPACE_OPERAND_CORRECT){float *h=read_floats(p->correction_operand,(size_t)s->model->hidden);snprintf(file,sizeof(file),"%s-explicit-operand.f32",name);save(capture->directory,file,h,(size_t)s->model->hidden*4);free(h);}
    if(mode==QWEN_SUBSPACE_CORRECT||mode>=QWEN_SUBSPACE_CARRY_CORRECT){const float *arrays[]={p->correction_gate,p->correction_up};for(int j=0;j<2;j++){float *h=read_floats(arrays[j],(size_t)s->model->intermediate);snprintf(file,sizeof(file),"%s-correction-%s.f32",name,j?"up":"gate");save(capture->directory,file,h,(size_t)s->model->intermediate*4);free(h);}}
    if(mode>=QWEN_SUBSPACE_CARRY_CORRECT){const float *arrays[]={p->carry_omitted,p->read_complement};for(int j=0;j<2;j++){float *h=read_floats(arrays[j],(size_t)s->model->hidden);snprintf(file,sizeof(file),"%s-%s.f32",name,j?"read-complement":"carry-omitted");save(capture->directory,file,h,(size_t)s->model->hidden*4);free(h);}}
    if(*common<0)*common=s->pending;
    int rank=mode==QWEN_SUBSPACE_PROJECT2?2:p->rank;
    fprintf(f,"{\"name\":\"%s\",\"group\":\"%s\",\"mode\":%d,\"rank\":%d,\"position\":%d,\"parent_kv_prefix_sha256\":\"%s\",\"work\":{\"gate_up_matvecs\":%u,\"down_matvecs\":%u,\"compile_matvecs\":%u},\"coefficients\":[",name,group,(int)mode,rank,position,prefix,stats.work.gate_up_matvecs,stats.work.down_matvecs,stats.work.compile_matvecs);
    for(int j=0;j<rank;j++){if(j)fputc(',',f);fprintf(f,"%.17g",stats.coefficients[j]);}fprintf(f,"],\"executed_coefficients\":[");for(int j=0;j<rank;j++){if(j)fputc(',',f);fprintf(f,"%.9g",stats.executed[j]);}
    fprintf(f,"],\"rho\":%.9g,\"receiver_norm\":%.17g,\"reconstruction_norm\":%.17g,\"omitted_norm\":%.17g,\"input_token_id\":%d,\"common_next_input_token_id\":%d,\"first\":",stats.rho,stats.receiver_norm,stats.reconstruction_norm,stats.omitted_norm,final_id,*common);
    snprintf(file,sizeof(file),"%s-first.f32",name);readout(f,s,red,blue,capture->directory,file);snprintf(file,sizeof(file),"%s-current-kv.f32",name);kv_row(s,position,capture->directory,file);
    uint64_t child;REQUIRE(!qwen_stack_push(s,&child));REQUIRE(!qwen_push(s,*common));fprintf(f,",\"common_next\":");readout(f,s,red,blue,NULL,NULL);snprintf(file,sizeof(file),"%s-next-kv.f32",name);kv_row(s,position+1,capture->directory,file);REQUIRE(!qwen_stack_rollback(s,child));
    fprintf(f,",\"own\":");int last=generated(f,s,8);fprintf(f,",\"questions\":[");
    if(questions)for(int q=0;q<2;q++){
        REQUIRE(!qwen_stack_push(s,&child));append_suffix(s,questions[q],question_lengths[q],last,0);if(q)fputc(',',f);fprintf(f,"{\"name\":\"%s\",\"first\":",q?"Niko":"Mira");readout(f,s,red,blue,NULL,NULL);fprintf(f,",\"output\":");generated(f,s,8);fputc('}',f);REQUIRE(!qwen_stack_rollback(s,child));
    }
    REQUIRE(!qwen_stack_rollback(s,branch));prefix_digest(s,position,after);REQUIRE(!strcmp(prefix,after));fprintf(f,"],\"prefix_intact\":true}");REQUIRE(!fflush(f));return last;
}
int main(int argc,char **argv){
    int writer=argc==11&&!strcmp(argv[9],"writer"),formation=argc==11&&!strcmp(argv[9],"formation"),components=writer||formation||(argc==11&&!strcmp(argv[9],"components"));REQUIRE(argc==10||(argc==12&&!strcmp(argv[10],"enrich"))||components);int enrich=argc==12||components,ncal=0;int *calibration=enrich?tokens(argv[11-components],&ncal):NULL;
    const char *directory=argv[7-components];int red=atoi(argv[8-components]),blue=atoi(argv[9-components]),n,nt[2]={0,0},nq[2];int *prompt=tokens(argv[2],&n),*targets[]={tokens(argv[3],&nt[0]),components?NULL:tokens(argv[4],&nt[1])},*questions[]={tokens(argv[5-components],&nq[0]),tokens(argv[6-components],&nq[1])};REQUIRE(n>1);
    TcmModel bundle;QwenModel m;QwenSession s;QwenSubspace p;char error[256];double started=clock_seconds();REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error)));REQUIRE(!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);
    REQUIRE(red>=0&&red<m.vocab&&blue>=0&&blue<m.vocab);REQUIRE(!qwen_session_init(&s,&m,512));REQUIRE(!qwen_subspace_init(&p,&m));for(int j=0;j<n-1;j++)REQUIRE(!qwen_push(&s,prompt[j]));
    char file[4096];REQUIRE(snprintf(file,sizeof(file),"%s/reuse-driver.json",directory)<(int)sizeof(file));FILE *f=fopen(file,"wb");REQUIRE(f);Capture capture={directory,NULL};s.observer=port;s.observer_context=&capture;s.observe_layer=16;
    Formation form={0};if(formation){form.inherited=gpu_alloc((size_t)m.hidden*4);form.carry=gpu_alloc((size_t)m.hidden*4);form.new_write=gpu_alloc((size_t)m.hidden*4);}
    fprintf(f,"{\"schema\":\"tcm-subspace-%s-driver-v1\",\"model_digest\":\"%s\",\"layer\":16,\"capacity\":512,\"source_position\":%d,\"program_f32_bytes\":%zu,\"program_f64_bytes\":%zu,\"gpu_workspace_and_program_bytes\":%zu,\"host_array_bytes\":%zu,\"rows\":[",writer?"writer":formation?"formation":components?"components":enrich?"enrichment":"reuse",m.digest,n-1,((size_t)3*m.hidden+(size_t)4*m.intermediate)*4,((size_t)2*m.hidden+4)*8,p.gpu_bytes,p.host_bytes);
    const QwenSubspaceMode source_modes[]={QWEN_SUBSPACE_COMPILE,QWEN_SUBSPACE_PROJECT,QWEN_SUBSPACE_CORRECT,QWEN_SUBSPACE_REASSOC};const char *source_names[]={"source-native","source-project","source-correct","source-reassoc"};int common=-1;char frozen[65],after[65];
    for(int j=0;j<4;j++){if(j)fputc(',',f);row(f,&s,&p,source_modes[j],source_names[j],"source",prompt[n-1],&common,&capture,NULL,NULL,red,blue,NULL,NULL);program(&p,NULL,after);if(!j)strcpy(frozen,after);else REQUIRE(!strcmp(frozen,after));}
    /* Rebuild the native source after sibling source controls. This is native
     * own generation, never an oracle string injected as assistant content. */
    s.observe_position=-1;REQUIRE(!qwen_push(&s,prompt[n-1]));fprintf(f,"],\"rebuilt_source_own\":");int source_end=generated(f,&s,8);char source_parent[65],extended[65];prefix_digest(&s,s.position,source_parent);fprintf(f,",\"source_answer_kv_prefix_sha256\":\"%s\",\"calibration_rows\":[",source_parent);
    if(enrich){
        uint64_t parent;REQUIRE(!qwen_stack_push(&s,&parent));append_suffix(&s,calibration,ncal,source_end,1);common=-1;
        float *operand=gpu_alloc((size_t)m.hidden*4);row(f,&s,&p,QWEN_SUBSPACE_DENSE,"calibration-dense","calibration",calibration[ncal-1],&common,&capture,NULL,NULL,red,blue,operand,NULL);
        QwenPlaneWork work;REQUIRE(!qwen_subspace_extend(&p,operand,&work));gpu_free(operand);REQUIRE(work.compile_matvecs==2&&!work.gate_up_matvecs&&!work.down_matvecs);
        extension(&p,directory,extended);program(&p,NULL,after);REQUIRE(!strcmp(frozen,after));REQUIRE(!qwen_stack_rollback(&s,parent));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_parent,after));
        if(components){REQUIRE(!strcmp(frozen,"1a12d8a22bfd00575039b9409d28bbdd44bdaa8f8b207fe1f5f4f5bfb4424d3e"));REQUIRE(!strcmp(extended,"d33759a30b44473612f219ba00df332ea0330db4b7100c7d7a22e0519c43ddad"));}
        fprintf(f,"],\"extension_program_sha256\":\"%s\",\"extension_compile_projections\":%u,\"target_rows\":[",extended,work.compile_matvecs);
    }else fprintf(f,"],\"target_rows\":[");
    const QwenSubspaceMode target_modes[]={QWEN_SUBSPACE_DENSE,QWEN_SUBSPACE_PROJECT,QWEN_SUBSPACE_CORRECT,QWEN_SUBSPACE_REASSOC,QWEN_SUBSPACE_HELD};const char *mode_names[]={"dense","project","correct","reassoc","held"};
    for(int target=0;target<(components?1:2);target++){
        uint64_t parent;REQUIRE(!qwen_stack_push(&s,&parent));append_suffix(&s,targets[target],nt[target],source_end,1);common=-1;
        if(writer){
            QwenWriter w;REQUIRE(!qwen_writer_init(&w,&s,&p,15));
            const int policies[]={-1,0,1,2,3};const char *names[]={"writer-ordinary","writer-00","writer-10","writer-01","writer-11"};
            const int *q[]={questions[0],questions[1]};
            for(int j=0;j<5;j++){if(j)fputc(',',f);writer_row(f,&s,&w,policies[j],names[j],targets[target][nt[target]-1],&common,&capture,q,nq,red,blue);}
            qwen_writer_free(&w);REQUIRE(!qwen_stack_rollback(&s,parent));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_parent,after));
            program(&p,NULL,after);REQUIRE(!strcmp(frozen,after));extension(&p,NULL,after);REQUIRE(!strcmp(extended,after));continue;
        }
        const QwenSubspaceMode enriched_modes[]={QWEN_SUBSPACE_DENSE,QWEN_SUBSPACE_PROJECT2,QWEN_SUBSPACE_PROJECT,QWEN_SUBSPACE_CORRECT};const char *enriched_names[]={"dense","project2","project","correct"};const char *group=enrich?(target?"Niko-B":"Niko-A"):(target?"Mira":"Niko");
        const QwenSubspaceMode component_modes[]={QWEN_SUBSPACE_DENSE,QWEN_SUBSPACE_PROJECT,QWEN_SUBSPACE_CARRY_CORRECT,QWEN_SUBSPACE_READ_CORRECT,QWEN_SUBSPACE_CORRECT};const char *component_names[]={"dense","project","carry","read","correct"};if(components)group="Niko-C";
        const QwenSubspaceMode formation_modes[]={QWEN_SUBSPACE_DENSE,QWEN_SUBSPACE_PROJECT,QWEN_SUBSPACE_OPERAND_CORRECT,QWEN_SUBSPACE_OPERAND_CORRECT,QWEN_SUBSPACE_CARRY_CORRECT};const char *formation_names[]={"dense","project","inherited","new","carry"};if(formation)group="Niko-D";
        for(int j=0;j<(components?5:enrich?4:5);j++){char name[64];snprintf(name,sizeof(name),"%s-%s",group,formation?formation_names[j]:components?component_names[j]:enrich?enriched_names[j]:mode_names[j]);if(target||j)fputc(',',f);const int *q[]={questions[0],questions[1]};form.select_new=j==3;row(f,&s,&p,formation?formation_modes[j]:components?component_modes[j]:enrich?enriched_modes[j]:target_modes[j],name,group,targets[target][nt[target]-1],&common,&capture,q,nq,red,blue,NULL,formation?&form:NULL);program(&p,NULL,after);REQUIRE(!strcmp(frozen,after));if(enrich){extension(&p,NULL,after);REQUIRE(!strcmp(extended,after));}}
        REQUIRE(!qwen_stack_rollback(&s,parent));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_parent,after));
    }
    fputc(']',f);if(components&&!writer){split_joint(f,&s,&p,directory,formation?"Niko-D":"Niko-C",formation?"inherited":"carry",formation?"new":"read");program(&p,NULL,after);REQUIRE(!strcmp(frozen,after));extension(&p,NULL,after);REQUIRE(!strcmp(extended,after));}
    if(writer)fprintf(f,",\"writer_binding\":{\"compile_layer\":16,\"edit_layer\":15,\"kv_first_layer\":15,\"gpu_bytes\":%zu,\"host_bytes\":%zu,\"projection\":\"sequential-fp64-qqt-then-fp32\",\"fires_per_arm\":1}",(size_t)m.hidden*4,(size_t)m.hidden*(3*sizeof(double)+sizeof(float)));
    fprintf(f,",\"program_sha256\":\"%s\",\"basis_condition\":%.17g,\"program_unchanged\":true,\"seconds\":%.9g}\n",frozen,p.condition,clock_seconds()-started);REQUIRE(!fclose(f));
    if(formation){gpu_free(form.inherited);gpu_free(form.carry);gpu_free(form.new_write);}free(calibration);free(prompt);for(int j=0;j<2;j++){free(targets[j]);free(questions[j]);}qwen_subspace_free(&p);qwen_session_free(&s);qwen_model_free(&m);return 0;
}
