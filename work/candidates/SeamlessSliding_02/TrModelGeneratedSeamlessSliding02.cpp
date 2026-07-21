#include <windows.h>
#include <atlbase.h>
#include "TrModelGeneratedSeamlessSliding02.h"
#include "TrModelGeneratedSeamlessSliding02_ps.h"

#define BREAK_IF_FAIL(hr) if		(FAILED(hr)) { break; }

#define MAX_SAMPLE_COUNT_PRODUCTION 30

typedef struct tagPS_SLSLIDE_PARAM
{
	INT      nSampleCount;
	INT      nTxIndex;
	INT      bSpeedUp;
	FLOAT    fMixRate;
	XMFLOAT2 f2AspectRatio;//(AspectX, AspectY)
	XMFLOAT2 f2Direction;//(DirectionX, DirectionY)
	XMFLOAT4 f4DistanceTable[MAX_SAMPLE_COUNT_PRODUCTION]; //.zw = padding
	tagPS_SLSLIDE_PARAM() :
		nSampleCount(5), nTxIndex(0), bSpeedUp(1), fMixRate(0.0), f2AspectRatio(1.0f, 1.0f), f2Direction(0.0f, 1.0f)
	{
		for (INT i = 0; i < MAX_SAMPLE_COUNT_PRODUCTION; i += 1)
		{
			f4DistanceTable[i] = XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
		}
	}

} PS_SLSLIDE_PARAM;

static_assert(sizeof(PS_SLSLIDE_PARAM) == 512, "PS parameter layout must match the shader constant buffer");

CTrModelGeneratedSeamlessSliding02::CTrModelGeneratedSeamlessSliding02(HINSTANCE hInst, const WCHAR* wszReferencePath)
	: CFxBase(hInst, wszReferencePath)
	, m_llTotalDuration(0)
	, m_llMaxDuration(0)
	, m_f2Direction(0.0f, 0.0f)
	, m_pfGaussianArray(NULL)
{
	m_llTotalDuration = 18000000;
	m_llMaxDuration = 18000000;
	m_f2Direction = XMFLOAT2(-1.2f, 0.0f);

	m_pfGaussianArray = new FLOAT[MAX_SAMPLE_COUNT_PRODUCTION];

	if (m_pfGaussianArray)
	{
		INT nDegree = MAX_SAMPLE_COUNT_PRODUCTION - 1;
		FLOAT fSigma = FLOAT(nDegree) / FLOAT(2.0);
		FLOAT fSigmaSqr = fSigma * fSigma;

		FLOAT fSum = 0.0;

		for (INT i = 0; i < MAX_SAMPLE_COUNT_PRODUCTION; i += 1)
		{
			FLOAT fValue = FLOAT((1.0 / sqrt(6.2831853 * fSigmaSqr)) * exp(FLOAT(i * i) / (-2.0 * fSigmaSqr)));
			m_pfGaussianArray[i] = fValue;
			fSum += fValue;
		}

		for (INT i = 0; i < MAX_SAMPLE_COUNT_PRODUCTION; i += 1)
		{
			m_pfGaussianArray[i] /= fSum;
		}
	}
}

CTrModelGeneratedSeamlessSliding02::~CTrModelGeneratedSeamlessSliding02()
{
	if (m_pfGaussianArray)
	{
		delete[] m_pfGaussianArray;
		m_pfGaussianArray = NULL;
	}
}

UINT CTrModelGeneratedSeamlessSliding02::GetPSShaderCodeSize()
{
	return sizeof(g_Tr_ModelGeneratedSeamlessSliding02_PS);
}

void* CTrModelGeneratedSeamlessSliding02::GetPSShaderCode()
{
	return (void*)(g_Tr_ModelGeneratedSeamlessSliding02_PS);
}

UINT CTrModelGeneratedSeamlessSliding02::GetPSParamSize()
{
	return sizeof(PS_SLSLIDE_PARAM);
}

HRESULT CTrModelGeneratedSeamlessSliding02::InitPSParam(void* pPSParam)
{
	if (!pPSParam)
		return E_INVALIDARG;

	PS_SLSLIDE_PARAM PSParam;
	memcpy_s(pPSParam, sizeof(PS_SLSLIDE_PARAM), &PSParam, sizeof(PS_SLSLIDE_PARAM));

	return S_OK;
}

HRESULT CTrModelGeneratedSeamlessSliding02::UpdatePSParam(void* pPSParam)
{
	if (!pPSParam)
		return E_INVALIDARG;

	PS_SLSLIDE_PARAM* pMyPSParam = (PS_SLSLIDE_PARAM*)pPSParam;
	pMyPSParam->f2Direction = m_f2Direction;
	pMyPSParam->bSpeedUp = 1;

	pMyPSParam->f2AspectRatio = XMFLOAT2(1.0f, 1.0f);
	if (m_nDstBufferWidth > m_nDstBufferHeight)
		pMyPSParam->f2AspectRatio.x = FLOAT(m_nDstBufferWidth) / FLOAT(m_nDstBufferHeight);
	else
		pMyPSParam->f2AspectRatio.y = FLOAT(m_nDstBufferHeight) / FLOAT(m_nDstBufferWidth);

	FLOAT fCurrPos = m_fProgress;
	if (m_llTotalDuration <= 0)
		m_llTotalDuration = m_llMaxDuration;

	FLOAT fCoeffDuration = FLOAT(m_llTotalDuration) / FLOAT(m_llMaxDuration);
	LONGLONG llTimeMid = m_llTotalDuration / 2 + 750000;
	LONGLONG llTimeRelative = LONGLONG(fCurrPos * FLOAT(m_llTotalDuration)) - llTimeMid;

	if (llTimeRelative < 0)
		pMyPSParam->nTxIndex = 0;
	else
		pMyPSParam->nTxIndex = 1;

	if (llTimeRelative < -2500000 * fCoeffDuration)
	{
		pMyPSParam->fMixRate = 0;
		pMyPSParam->nSampleCount = 1;
		pMyPSParam->f4DistanceTable[0] = XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
	}
	else if (llTimeRelative < 0)
	{
		if (llTimeRelative > -1000000 * fCoeffDuration)
		{
			pMyPSParam->bSpeedUp = 0;
			pMyPSParam->fMixRate = FLOAT(llTimeRelative + 1000000 * fCoeffDuration) / FLOAT(2000000 * fCoeffDuration);
		}
		else
			pMyPSParam->fMixRate = 0;

		FLOAT fRatio = FLOAT(llTimeRelative + 2500000 * fCoeffDuration) / FLOAT(2500000 * fCoeffDuration);
		fRatio = fRatio * fRatio;
		FLOAT fDistance = FLOAT(0.5) * fRatio;

		pMyPSParam->nSampleCount = 20;
		FLOAT fOffset = 0;
		FLOAT fStepDistance = FLOAT(0.011);
		FLOAT fStep = -fStepDistance * fRatio;

		for (INT i = 0; i < pMyPSParam->nSampleCount; i += 1)
		{
			pMyPSParam->f4DistanceTable[i] = XMFLOAT4(fDistance + fOffset, m_pfGaussianArray[i], 0.0f, 0.0f);

			fOffset += fStep;
			fStep *= FLOAT(1.05);
		}
	}
	else if (llTimeRelative < 2500000 * fCoeffDuration)
	{
		if (llTimeRelative < 1000000 * fCoeffDuration)
		{
			pMyPSParam->bSpeedUp = 0;
			pMyPSParam->fMixRate = FLOAT(llTimeRelative) / FLOAT(2000000 * fCoeffDuration) + FLOAT(0.5);
		}
		else
			pMyPSParam->fMixRate = 1;

		FLOAT fRatio = FLOAT(2500000 * fCoeffDuration - llTimeRelative) / FLOAT(2500000 * fCoeffDuration);
		fRatio = fRatio * fRatio;
		FLOAT fDistance = FLOAT(-0.5) * fRatio;

		pMyPSParam->nSampleCount = 20;
		FLOAT fOffset = 0;
		FLOAT fStepDistance = FLOAT(0.011);
		FLOAT fStep = -fStepDistance * FLOAT(fRatio * 0.7 + 0.3);

		for (INT i = 0; i < pMyPSParam->nSampleCount; i += 1)
		{
			pMyPSParam->f4DistanceTable[i] = XMFLOAT4(fDistance + fOffset, m_pfGaussianArray[i], 0.0f, 0.0f);

			fOffset += fStep;
			fStep /= FLOAT(1.05);
		}
	}
	else
	{
		pMyPSParam->fMixRate = 1;
		pMyPSParam->nSampleCount = 1;
		pMyPSParam->f4DistanceTable[0] = XMFLOAT4(0.0f, 1.0f, 0.0f, 0.0f);
	}

	return S_OK;
}

HRESULT CTrModelGeneratedSeamlessSliding02::SetExtraPSResource()
{
	HRESULT hr = S_OK;

	do
	{
		CComPtr<ID3D11ShaderResourceView> pPS_SRV_InputSrcA;
		CComPtr<ID3D11ShaderResourceView> pPS_SRV_InputSrcB;

		hr = m_pd3dDevice->CreateShaderResourceView(m_pTxInputSrcA, NULL, &pPS_SRV_InputSrcA);
		BREAK_IF_FAIL(hr);

		hr = m_pd3dDevice->CreateShaderResourceView(m_pTxInputSrcB, NULL, &pPS_SRV_InputSrcB);
		BREAK_IF_FAIL(hr);

		ID3D11ShaderResourceView* pPS_SRV[2] = { pPS_SRV_InputSrcA, pPS_SRV_InputSrcB };
		m_pDeferredContext->PSSetShaderResources(0, 2, pPS_SRV);
	} while (false);

	return hr;
}
