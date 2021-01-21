#include <stdlib.h>
#include <stdio.h>
#include "../include/baipiaoocr_api.h"

#define BPOCR_DET_MODEL ""
#define BPOCR_CLS_MODEL ""
#define BPOCR_REC_MODEL ""
#define BPOCR_KEY_PATH  ""

#define THREAD_NUM   3
int main(int argc, char * argv[])
{


    BPHANDLE  Handle= BPOcrInit(BPOCR_DET_MODEL,BPOCR_CLS_MODEL,BPOCR_REC_MODEL,BPOCR_KEY_PATH,THREAD_NUM);
    if(!Handle)
    {
        printf("cannot initialize the OCR Engine.\n");
        return -1;
    }


    return 0;
}