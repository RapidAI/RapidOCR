#include "../include/precomp.h"
#ifdef __cplusplus
extern "C"
{
#endif 

	typedef struct
	{
		OcrLite OcrObj;
		std::string strRes;
	}OCR_OBJ;

	_QM_OCR_API BPHANDLE   BPOcrInit(const char* szDetModel, const char* szClsModel, const char* szRecModel,const char *szKeyPath, int nThreads)
	{

		omp_set_num_threads(nThreads);
		OCR_OBJ* pOcrObj=new OCR_OBJ;
		if(pOcrObj)
		{
			pOcrObj->OcrObj.setNumThread(nThreads);

			 pOcrObj->OcrObj.initModels(szDetModel, szClsModel, szRecModel, szKeyPath);
			 
			return pOcrObj;
		}
		else
		{
			return nullptr;
		}
		
	}

	
	_QM_OCR_API BOOL  BPOcrDoOcr(BPHANDLE handle, const char* szImgPath, BOOL bAutoParam, BOOL bLongPic,RAPIDOCR_PARAM *pParam)
	{

		OCR_OBJ* pOcrObj=(OCR_OBJ*)handle;
		if(!pOcrObj)
			return FALSE;

		RAPIDOCR_PARAM Param = *pParam;
		
		if (Param.boxScoreThresh == 0)
			Param.boxScoreThresh = 0.5;
		if (Param.boxThresh == 0)
			Param.boxThresh = 0.3;
		if (Param.flagDoAngle == 0)
			Param.flagDoAngle = 1;
		if (Param.flagMostAngle == 0)
			Param.flagMostAngle = 0;
		


		if (Param.maxSideLen == 0)
			Param.maxSideLen = 1024;
		if (Param.unClipRatio == 0)
			Param.unClipRatio = 1.5;



		if (bAutoParam)
		{

			cv::Mat oriImg = cv::imread(szImgPath);

			int nMax = oriImg.rows > oriImg.cols ? oriImg.rows : oriImg.cols;
			int nMin = oriImg.rows > oriImg.cols ? oriImg.cols: oriImg.rows ;

			if (nMax > nMin * 5) // 5 times
				bLongPic = true;
		}


		if (bLongPic)
		{
			
				Param.maxSideLen = 0;
			
				Param.unClipRatio = 3.0;
		}


		
		// 	-numThread %NUMBER_OF_PROCESSORS% ^
		// --padding 0 ^
		// --maxSideLen %MAX_SIDE_LEN% ^
		// --boxScoreThresh 0.5 ^
		// --boxThresh 0.3 ^
		// --unClipRatio %CLIP_RATE% ^
		// --doAngle 1 ^
		// --mostAngle 0

// 参数设置算法： https://github.com/znsoftm/RapidOCR/blob/219bf6295a9c3c5afa7f47bf5173ca02f1003521/python/ch_ppocr_mobile_v2_det/utils.py#L118

			

		
		std::string imgPath,imgDir,imgName;
		imgPath=szImgPath;
		imgDir=imgPath.substr(0, imgPath.find_last_of('\\') + 1);
		imgName =imgPath.substr(imgPath.find_last_of('\\') + 1);
		OcrResult result=pOcrObj->OcrObj.detect(imgDir.c_str(),imgName.c_str(),Param.padding,Param.maxSideLen,Param.boxScoreThresh,Param.boxThresh,Param.unClipRatio,Param.flagDoAngle?true:false,Param.flagMostAngle?true:false);
		if (result.strRes.length() > 0)
		{
	
			pOcrObj->strRes = result.strRes; 
			return TRUE;
		}
		else
			return FALSE;
	}


	_QM_OCR_API  int BPOcrGetLen(BPHANDLE handle)
	{
		OCR_OBJ* pOcrObj = (OCR_OBJ*)handle;
		if (!pOcrObj)
			return 0;
		return pOcrObj->strRes.size()+1;
	}

	_QM_OCR_API  BOOL BPOcrGetResult(BPHANDLE handle, char* szBuf, int nLen)
	{

		OCR_OBJ* pOcrObj = (OCR_OBJ*)handle;
		if (!pOcrObj)
			return FALSE;

		if (nLen > pOcrObj->strRes.size())
		{
			strncpy(szBuf, pOcrObj->strRes.c_str(), pOcrObj->strRes.size());
			szBuf[pOcrObj->strRes.size() - 1] = 0;
		}
		
		return pOcrObj->strRes.size();
	}

	_QM_OCR_API void  BPOcrDeinit(BPHANDLE handle)
	{

		OCR_OBJ* pOcrObj=(OCR_OBJ*)handle;
		if(pOcrObj)
			delete pOcrObj;


	}


#ifdef __cplusplus
}
#endif 