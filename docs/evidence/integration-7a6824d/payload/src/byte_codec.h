#ifndef TCM_APPEND_CODEC_H
#define TCM_APPEND_CODEC_H
#include <math.h>
#include <stdint.h>
static int codec_row(const float *x,int8_t *codes,float *scale){
    float maximum=0;for(int d=0;d<128;d++){if(!isfinite(x[d]))return -1;maximum=fmaxf(maximum,fabsf(x[d]));}
    float s=maximum>0?maximum/127.0f:1.0f;if(!(s>0)||!isfinite(s))return -1;
    for(int d=0;d<128;d++){float value=roundf(x[d]/s);if(value>127)value=127;if(value< -127)value= -127;codes[d]=(int8_t)value;}*scale=s;return 0;
}
#endif
