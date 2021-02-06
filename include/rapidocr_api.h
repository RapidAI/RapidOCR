#pragma once
#ifdef __cplusplus
extern "C"
{
#endif 

#ifdef WIN32

#ifdef _BPOCR_EXPORT_API
#define _QM_OCR_API  __declspec(dllexport)
#else
#define _QM_OCR_API __declspec(dllimport)
#endif
#else

#define _QM_OCR_API 
#endif

typedef void* BPHANDLE;
typedef  char BOOL;
	
#ifndef NULL
#define NULL 0
#endif 
#define	TRUE	1
#define	FALSE	0


typedef struct __bpocr_param
{
	float unClipRatio;
	int maxSideLen;
	int padding;
	float  boxThresh;
	float boxScoreThresh;
	int numThread;
	int flagDoAngle; // 1 means do
	int flagMostAngle; // 1 means true
} BPOCR_PARAM;



/*
By default, nThreads should be the number of threads 
*/
_QM_OCR_API BPHANDLE   BPOcrInit(const char * szDetModel, const char * szClsModel, const char * szRecModel,const char *szKeyPath,int nThreads);

_QM_OCR_API BOOL  BPOcrDoOcr(BPHANDLE handle, const char* szImgPath, BOOL bAutoParam,BOOL bLongPic, BPOCR_PARAM *pParam);

_QM_OCR_API  int BPOcrGetLen(BPHANDLE handle);

_QM_OCR_API  BOOL BPOcrGetResult(BPHANDLE handle, char* szBuf, int nLen);

_QM_OCR_API void  BPOcrDeinit(BPHANDLE handle);




#ifdef __cplusplus
}
#endif
