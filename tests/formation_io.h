#ifndef TCM_FORMATION_IO_H
#define TCM_FORMATION_IO_H
/* Assay-owned operands, derived during ordinary sequential layer execution. */
typedef struct { Capture capture; float *inherited,*carry,*new_write; int select_new; } Formation;
static void formation_port(void *context,int position,int layer,QwenPort id,const float *data,size_t count){
    (void)position;Formation *form=context;REQUIRE(layer==15);
    if(id!=QWEN_INPUT&&id!=QWEN_ATTENTION_WRITE&&id!=QWEN_RAW_RECEIVER&&id!=QWEN_MLP_RETURN&&id!=QWEN_OUTPUT)return;
    if(id==QWEN_INPUT)gpu_copy(form->inherited,data,count*4);
    char name[256];REQUIRE(snprintf(name,sizeof(name),"%s-%s.f32",form->capture.name,qwen_port_name(id))<(int)sizeof(name));
    float *h=read_floats(data,count);save(form->capture.directory,name,h,count*4);free(h);
}
static void project_away(const QwenSubspace *p,const float *input,float *output){
    int d=p->model->hidden;float *h=read_floats(input,(size_t)d);double dots[3]={0,0,0};const double *q[]={p->q0,p->q1,p->q2};
    for(int j=0;j<d;j++)for(int k=0;k<3;k++)dots[k]+=q[k][j]*(double)h[j];
    for(int j=0;j<d;j++){double v=h[j];for(int k=0;k<3;k++)v-=q[k][j]*dots[k];h[j]=(float)v;REQUIRE(isfinite(h[j]));}
    gpu_write(output,h,(size_t)d*4);free(h);
}
static void formation_prepare(Formation *form,QwenSession *s,const QwenSubspace *p,const Capture *capture){
    REQUIRE(s->layer==16&&p->rank==3);int d=p->model->hidden;
    project_away(p,form->inherited,form->inherited);project_away(p,s->x,form->carry);
    gpu_plane(form->new_write,form->carry,form->inherited,form->inherited,d,-1,0);
    const float *arrays[]={form->inherited,form->carry,form->new_write};const char *suffix[]={"inherited-operand","carried-operand","new-write-operand"};
    for(int j=0;j<3;j++){char file[160];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",capture->name,suffix[j])<(int)sizeof(file));float *h=read_floats(arrays[j],(size_t)d);save(capture->directory,file,h,(size_t)d*4);free(h);}
}
#endif
