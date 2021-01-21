#include "../include/precomp.h"
#ifdef __cplusplus
extern "C"
{
#endif 

	_QM_OCR_API BPHANDLE   BPOcrInit(const char* szDetModel, const char* szClsModel, const char* szRecModel,const char *szKeyPath, int nThreads)
	{

		omp_set_num_threads(nThreads);
		OcrLite * pOcrObj=new OcrLite;
		if(pOcrObj)
		{
			pOcrObj->setNumThread(nThreads);

			 pOcrObj->initModels(szDetModel, szClsModel, szRecModel, szKeyPath);
			 
			return pOcrObj;
		}
		else
		{
			return nullptr;
		}
		
	}

	_QM_OCR_API const char*  BPOcrDoOcr(BPHANDLE handle, const char* szImgPath, BOOL bAutoParam,BPOCR_PARAM *pParam)
	{

		OcrLite * pOcrObj=(	OcrLite *)handle;
		if(!pOcrObj)
			return NULL;

		return "";
	}

	_QM_OCR_API void  BPOcrDeinit(BPHANDLE handle)
	{

		OcrLite * pOcrObj=(	OcrLite *)handle;
		if(pOcrObj)
			delete pOcrObj;


	}


#ifdef __cplusplus
}
#endif 