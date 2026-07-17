SamplerState samLinear : register(s0);
Texture2D TextureA : register(t0);
Texture2D TextureB : register(t1);
//--------------------------------------------------------------------------------------
// Constant Buffer Variables
//--------------------------------------------------------------------------------------
cbuffer PSParam : register(b0)
{
    int nSampleCount;
    int nTxIndex;
    int bSpeedUp;
    float fMixRate;
    float2 f2AspectRatio; //(AspectX, AspectY)
    float2 f2DirectionUpper; //(DirectionX, DirectionY) for the upper region
    float2 f2DirectionLower; //(DirectionX, DirectionY) for the lower region
    float2 f2Padding;
    float4 f4DistanceTable[30];
};
//--------------------------------------------------------------------------------------
// Define Shader Structure
//--------------------------------------------------------------------------------------
struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
    float Index : OBJINDEX;
};


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 Pixel_Shader(PS_INPUT input) : SV_TARGET
{
	// Return order -- float4(R, G, B, A)
	// f4.x = R , f4.y = G , f4.z = B , f4.w = A
    float2 f2TxCoord = input.Tex * f2AspectRatio;
    float2 f2TxCoordOffset;
    float4 f4Result0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 f4Result1 = float4(0.0, 0.0, 0.0, 0.0);
    float4 f4Result = float4(0.0, 0.0, 0.0, 0.0);
    float bandIndex = floor(input.Tex.y * 8.0);
    float2 f2Direction = fmod(bandIndex, 2.0) == 0.0 ? f2DirectionUpper : f2DirectionLower;

    if ((!bSpeedUp) || (bSpeedUp && nTxIndex == 0))
    {
        for (int i = 0; i < nSampleCount; i += 1)
        {
            f2TxCoordOffset = f2TxCoord - f2Direction * f4DistanceTable[i].x;
            f4Result0 += TextureA.Sample(samLinear, f2TxCoordOffset / f2AspectRatio) * f4DistanceTable[i].y;
        }
    }

    if ((!bSpeedUp) || (bSpeedUp && nTxIndex == 1))
    {
        for (int i = 0; i < nSampleCount; i += 1)
        {
            f2TxCoordOffset = f2TxCoord - f2Direction * f4DistanceTable[i].x;
            f4Result1 += TextureB.Sample(samLinear, f2TxCoordOffset / f2AspectRatio) * f4DistanceTable[i].y;
        }
    }

    if (!bSpeedUp)
    {
        f4Result = lerp(f4Result0, f4Result1, fMixRate);
        return f4Result;
    }
    else
    {
        if (nTxIndex == 0)
            return f4Result0;
        else
            return f4Result1;
    }
}
