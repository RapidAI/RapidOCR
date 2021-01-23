#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <memory.h>
#include <string>
#include "../include/baipiaoocr_api.h"

#define BPOCR_DET_MODEL "/ch_ppocr_mobile_v2.0_det_infer.onnx"
#define BPOCR_CLS_MODEL "/ch_ppocr_mobile_v2.0_cls_infer.onnx"
#define BPOCR_REC_MODEL "/ch_ppocr_mobile_v2.0_rec_infer.onnx"
#define BPOCR_KEY_PATH  "/ppocr_keys_v1.txt"

#define THREAD_NUM   3
#define MAX_PATH    260

#define  DEFAULT_MODEL_DIR  "/data/workprj/BaiPiaoOCR/models/"
int main(int argc, char * argv[])
{


#ifndef NDEBUG
//debug


#endif

char *szModelDir=NULL;

    if(argc == 1)
        szModelDir=DEFAULT_MODEL_DIR;
    else
    {
        szModelDir=argv[1]; 
    }
    
    char szDetModelPath[MAX_PATH]={0};
    char szClsModelPath[MAX_PATH]={0};
    char szRecModelPath[MAX_PATH]={0};
    char szKeylPath[MAX_PATH]={0};

    strcpy(szDetModelPath,szModelDir);
    strcpy(szClsModelPath,szModelDir);
    strcpy(szRecModelPath,szModelDir);
    strcpy(szKeylPath,szModelDir);

    strcat(szDetModelPath,BPOCR_DET_MODEL);
    strcat(szClsModelPath,BPOCR_CLS_MODEL);
    strcat(szRecModelPath,BPOCR_REC_MODEL);
    strcat(szKeylPath,BPOCR_KEY_PATH);

    BPHANDLE  Handle= BPOcrInit(szDetModelPath,szClsModelPath,szRecModelPath,szKeylPath,THREAD_NUM);
    if(!Handle)
    {
        printf("cannot initialize the OCR Engine.\n");
        return -1;
    }




    if(Handle)
    {
            BPOcrDeinit(Handle);
    }

    return 0;
}