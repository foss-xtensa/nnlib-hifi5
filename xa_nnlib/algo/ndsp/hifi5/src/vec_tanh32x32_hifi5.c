/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
/* ------------------------------------------------------------------------ */
/*  IntegrIT, Ltd.   www.integrIT.com, info@integrIT.com                    */
/*                                                                          */
/* DSP Library                                                              */
/*                                                                          */
/* This library contains copyrighted materials, trade secrets and other     */
/* proprietary information of IntegrIT, Ltd. This software is licensed for  */
/* use with Cadence processor cores only and must not be used for any other */
/* processors and platforms. The license to use these sources was given to  */
/* Cadence, Inc. under Terms and Condition of a Software License Agreement  */
/* between Cadence, Inc. and IntegrIT, Ltd.                                 */
/* ------------------------------------------------------------------------ */
/*          Copyright (C) 2015-2019 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
/*
  NatureDSP Signal Processing Library. Vector matematics
    Hyperbolic Tangent
    Optimized code for HiFi5
  IntegrIT, 2006-2019
*/

#include "NatureDSP_types.h"
#include "NatureDSP_Signal_math.h"
#include "common.h"

/*-------------------------------------------------------------------------
  Hyperbolic Tangent
  The functions compute the hyperbolic tangent of input argument. 32-bit
  fixed-point functions accept inputs in Q6.25 and form outputs in Q16.15
  format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output, Accuracy: 2 ULP
  Input:
  x[N]   input data, Q6.25 or floating point  
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q16.15 or floating point
-------------------------------------------------------------------------*/
void vec_tanh32x32(int32_t* restrict y, const int32_t* restrict x, int N)
{
  int n;
  static const int32_t ALIGN(32) polypow2[] = { 14685057, -114217091, 514075394, -1488269031, 2147475316 };

        ae_int32x4 * restrict pY = (      ae_int32x4 *)y;
  const ae_int32x4 * restrict pX = (const ae_int32x4 *)x;
  const ae_int32x4 * restrict pX1 = (const ae_int32x4 *)x;

  ae_int32x2 y0, y1, t0, t1, z0, z1;
  ae_int32x2 x0, x1, e0, e1, d0, d1;
  ae_valignx2 aX, aX1, aY;
  xtbool2 sign0, sign1;
  if (N<=0) return;

  if(N >= 4)
  {
      aX = AE_LA128_PP(pX);
      aX1 = AE_LA128_PP(pX1);
      aY = AE_ZALIGN128();

      for (n=0; n<(N>>2); n++)
      {
          AE_LA32X2X2_IP(x0, x1, aX, pX);
          sign0 = AE_LT32(x0, 0);
          sign1 = AE_LT32(x1, 0);
          AE_MULF2P32X4RAS(z0, z1, x0, x1, 1549082005, 1549082005);
          x0 = AE_ABS32S(z0);
          x1 = AE_ABS32S(z1);
          e0 = AE_SRAI32(x0, 23);
          e1 = AE_SRAI32(x1, 23);
#if 0
          x0 = AE_AND32(x0, AE_MOVDA32X2(0x007fffff, 0x007fffff));
          x1 = AE_AND32(x1, AE_MOVDA32X2(0x007fffff, 0x007fffff));
          x0 = AE_SLAI32(x0, 8);//Q31
          x1 = AE_SLAI32(x1, 8);
#else
          x0 = AE_MOVDEXT(x0, 23,8);
          x1 = AE_MOVDEXT(x1, 23,8);
#endif
          y0 = y1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 0);
          t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 1);
          AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
          t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 2);
          AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
          t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 3);
          AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
          t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 4);
          AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
          x0 = AE_SRAV32RS(y0, e0);//Q31
          x1 = AE_SRAV32RS(y1, e1);

          /* 0.96-x/2 */
          z0 = AE_SUB32(1030792151, AE_SRAI32(x0, 2));//Q30
          z1 = AE_SUB32(1030792151, AE_SRAI32(x1, 2));
          t0 = z0; t1 = z1;

          AE_MULAF2P32X4RAS(t0, t1, z0, z1, x0, x1);  //Q30+Q(30+31-31)=Q30
          d0 = AE_SUB32(1073741824, t0);//Q30
          d1 = AE_SUB32(1073741824, t1);
          t0 = AE_SRAI32(z0, 1); t1 = AE_SRAI32(z1, 1);//Q29
          AE_MULAF2P32X4RAS(t0, t1, z0, z1, d0, d1);z0 = t0; z1 = t1; //Q29+Q(30+30-31)=Q29

          AE_MULAF2P32X4RAS(t0, t1, z0, z1, x0, x1);//Q29+Q(29+31-31)
          d0 = AE_SUB32(536870912, t0);//Q29
          d1 = AE_SUB32(536870912, t1);
          //    t0 = AE_SRAI32(z0, 2); t1 = AE_SRAI32(z1, 2);//Q27
          AE_MULF2P32X4RAS(t0,t1,z0,z1,0x20000000,0x20000000);//Q27
          AE_MULAF2P32X4RAS(t0, t1, z0, z1, d0, d1); z0 = t0; z1 = t1;//Q27+Q(29+29-31)=Q27

          x0 = AE_SRAV32RS(x0, 12);//Q19
          x1 = AE_SRAV32RS(x1, 12);
          y0 = AE_SUB32(524288, x0);
          y1 = AE_SUB32(524288, x1);
          AE_MULF2P32X4RAS(z0, z1, z0, z1, y0, y1);//Q(27+19-31)=15

          AE_LA32X2X2_IP(x0, x1, aX1, pX1);
          sign0 = AE_LT32(x0, 0);
          sign1 = AE_LT32(x1, 0);
#if 0
          x0 = AE_NEG32S(z0);
          x1 = AE_NEG32S(z1);

          AE_MOVT32X2(z0, x0, sign0);
          AE_MOVT32X2(z1, x1, sign1);
#else
          z0=AE_MOVNEG32S_T(z0,x0);
          z1=AE_MOVNEG32S_T(z1,x1);
#endif
          AE_SA32X2X2_IP(z0, z1, aY, pY);
      }
      AE_SA128POS_FP(aY, pY);
      x += (N&~3);
      y += (N&~3);
      N &= 3;
  }

  if (N>0)
  {
    int32_t ALIGN(32) scratch[4];
    ae_int32x4 *pScr;
    pScr = (ae_int32x4*)scratch;
    pX = (const ae_int32x4*)x;
    pY = (      ae_int32x4*)y;
    AE_S32X2X2_I(0, 0, pScr, 0);
    __Pragma("no_unroll")
    for (n = 0; n<N; n++)
    {
      ae_int32x2 t;
      AE_L32_IP(t, castxcc(ae_int32, pX), sizeof(int32_t));
      AE_S32_L_IP(t, castxcc(ae_int32, pScr), sizeof(int32_t));
    }
    pScr = (ae_int32x4*)scratch;
    AE_L32X2X2_I(x0, x1, pScr, 0 * sizeof(ae_int32x4));
    sign0 = AE_LT32(x0, 0);
    sign1 = AE_LT32(x1, 0);
    AE_MULF2P32X4RAS(z0, z1, x0, x1, 1549082005, 1549082005);
    x0 = AE_ABS32S(z0);
    x1 = AE_ABS32S(z1);
    e0 = AE_SRAI32(x0, 23);
    e1 = AE_SRAI32(x1, 23);
#if 0
    x0 = AE_AND32(x0, AE_MOVDA32X2(0x007fffff, 0x007fffff));
    x1 = AE_AND32(x1, AE_MOVDA32X2(0x007fffff, 0x007fffff));
    x0 = AE_SLAI32(x0, 8);//Q31
    x1 = AE_SLAI32(x1, 8);
#else
    x0 = AE_MOVDEXT(x0, 23,8);
    x1 = AE_MOVDEXT(x1, 23,8);
#endif
    y0 = y1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 0);
    t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 1);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 2);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 3);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    t0 = t1 = AE_L32_I((const ae_int32 *)polypow2, 4 * 4);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    x0 = AE_SRAV32RS(y0, e0);//Q31
    x1 = AE_SRAV32RS(y1, e1);

    /* 0.96-x/2 */
    z0 = AE_SUB32(1030792151, AE_SRAI32(x0, 2));//Q30
    z1 = AE_SUB32(1030792151, AE_SRAI32(x1, 2));
    t0 = z0; t1 = z1;

    AE_MULAF2P32X4RAS(t0, t1, z0, z1, x0, x1);  //Q30+Q(30+31-31)=Q30
    d0 = AE_SUB32(1073741824, t0);//Q30
    d1 = AE_SUB32(1073741824, t1);
    t0 = AE_SRAI32(z0, 1); t1 = AE_SRAI32(z1, 1);//Q29
    AE_MULAF2P32X4RAS(t0, t1, z0, z1, d0, d1); z0 = t0; z1 = t1; //Q29+Q(30+30-31)=Q29

    AE_MULAF2P32X4RAS(t0, t1, z0, z1, x0, x1);//Q29+Q(29+31-31)
    d0 = AE_SUB32(536870912, t0);//Q29
    d1 = AE_SUB32(536870912, t1);
    t0 = AE_SRAI32(z0, 2); t1 = AE_SRAI32(z1, 2);//Q27
    AE_MULAF2P32X4RAS(t0, t1, z0, z1, d0, d1); z0 = t0; z1 = t1;//Q27+Q(29+29-31)=Q27

    x0 = AE_SRAV32RS(x0, 12);//Q19
    x1 = AE_SRAV32RS(x1, 12);
    y0 = AE_SUB32(524288, x0);
    y1 = AE_SUB32(524288, x1);
    AE_MULF2P32X4RAS(z0, z1, z0, z1, y0, y1);//Q(27+19-31)=15

    x0 = AE_NEG32S(z0);
    x1 = AE_NEG32S(z1);

    AE_MOVT32X2(z0, x0, sign0);
    AE_MOVT32X2(z1, x1, sign1);
    AE_S32X2X2_I(z0, z1, pScr, 0 * sizeof(ae_int32x4));
    __Pragma("no_unroll")
    for (n = 0; n<N; n++)
    {
      ae_int32x2 t;
      AE_L32_IP(t, castxcc(ae_int32, pScr), sizeof(int32_t));
      AE_S32_L_IP(t, castxcc(ae_int32, pY), sizeof(int32_t));
    }
  }
} /* vec_tanh32x32() */
