#pragma once
#include "FxBase.h"

class CTrModelGeneratedSeamlessSliding02 : public CFxBase
{
public:
	CTrModelGeneratedSeamlessSliding02(HINSTANCE hInst, const WCHAR* wszReferencePath);
	virtual ~CTrModelGeneratedSeamlessSliding02();

private:
	UINT GetPSShaderCodeSize();
	void* GetPSShaderCode();
	UINT GetPSParamSize();
	HRESULT InitPSParam(void* pPSParam);
	HRESULT UpdatePSParam(void* pPSParam);
	HRESULT SetExtraPSResource();


	LONGLONG m_llTotalDuration;
	LONGLONG m_llMaxDuration;
	XMFLOAT2 m_f2DirectionUpper;
	XMFLOAT2 m_f2DirectionLower;
	FLOAT* m_pfGaussianArray;
};
