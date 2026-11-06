#ifndef TCM_WRITER_IO_H
#define TCM_WRITER_IO_H
/* Included after the existing source/calibration helpers. */
static void writer_port(void *context,int position,int layer,QwenPort id,const float *data,size_t count){
    (void)position;(void)layer;Capture *c=context;char name[256];
    REQUIRE(snprintf(name,sizeof(name),"%s-%s.f32",c->name,qwen_port_name(id))<(int)sizeof(name));
    float *h=read_floats(data,count);save(c->directory,name,h,count*4);free(h);
}
static void writer_kv(const QwenSession *s,int position,const char *directory,const char *name){
    const QwenModel *m=s->model;size_t count=(size_t)2*(m->layers-15)*m->kvheads*m->head,offset=0;float *h=malloc(count*4);REQUIRE(h);
    for(int kind=0;kind<2;kind++)for(int l=15;l<m->layers;l++)for(int k=0;k<m->kvheads;k++){
        size_t address=(((size_t)l*m->kvheads+k)*s->capacity+position)*m->head;gpu_read(h+offset,(kind?s->values:s->keys)+address,(size_t)m->head*4);offset+=m->head;
    }for(size_t j=0;j<count;j++)REQUIRE(isfinite(h[j]));save(directory,name,h,count*4);free(h);
}
static void writer_row(FILE *f,QwenSession *s,QwenWriter *w,int policy,const char *name,int final_id,int *common,Capture *capture,const int *questions[2],const int question_lengths[2],int red,int blue){
    uint64_t branch,child;REQUIRE(!qwen_stack_push(s,&branch));int position=s->position;char prefix[65],after[65],file[128],label[128];prefix_digest(s,position,prefix);
    s->observer=writer_port;s->observer_context=capture;s->observe_position=position;s->observe_layer=15;
    REQUIRE(snprintf(label,sizeof(label),"%s-l15",name)<(int)sizeof(label));capture->name=label;
    REQUIRE(!qwen_begin_token(s,final_id));while(s->layer<15)REQUIRE(!qwen_advance_layer(s));
    if(policy<0)REQUIRE(!qwen_advance_layer(s));else REQUIRE(!qwen_advance_layer_writer(s,w,(unsigned)policy));
    s->observe_layer=16;REQUIRE(snprintf(label,sizeof(label),"%s-l16",name)<(int)sizeof(label));REQUIRE(!qwen_advance_layer(s));finish(s);
    if(*common<0)*common=s->pending;
    fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"position\":%d,\"input_token_id\":%d,\"common_next_input_token_id\":%d,\"parent_kv_prefix_sha256\":\"%s\",\"first\":",name,policy,position,final_id,*common,prefix);
    snprintf(file,sizeof(file),"%s-first.f32",name);readout(f,s,red,blue,capture->directory,file);snprintf(file,sizeof(file),"%s-current-kv.f32",name);writer_kv(s,position,capture->directory,file);
    REQUIRE(!qwen_stack_push(s,&child));s->observe_position=position+1;REQUIRE(snprintf(label,sizeof(label),"%s-next-l16",name)<(int)sizeof(label));REQUIRE(!qwen_push(s,*common));
    fprintf(f,",\"common_next\":");snprintf(file,sizeof(file),"%s-common-next.f32",name);readout(f,s,red,blue,capture->directory,file);snprintf(file,sizeof(file),"%s-next-kv.f32",name);writer_kv(s,position+1,capture->directory,file);REQUIRE(!qwen_stack_rollback(s,child));
    s->observe_position=-1;fprintf(f,",\"own\":");int last=generated(f,s,8);fprintf(f,",\"questions\":[");
    for(int q=0;q<2;q++){
        REQUIRE(!qwen_stack_push(s,&child));append_suffix(s,questions[q],question_lengths[q],last,0);if(q)fputc(',',f);fprintf(f,"{\"name\":\"%s\",\"first\":",q?"Niko":"Mira");readout(f,s,red,blue,NULL,NULL);fprintf(f,",\"output\":");generated(f,s,8);fputc('}',f);REQUIRE(!qwen_stack_rollback(s,child));
    }
    REQUIRE(!qwen_stack_rollback(s,branch));prefix_digest(s,position,after);REQUIRE(!strcmp(prefix,after));fprintf(f,"],\"prefix_intact\":true}");REQUIRE(!fflush(f));
}
#endif
