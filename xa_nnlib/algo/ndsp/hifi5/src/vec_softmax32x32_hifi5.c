/*******************************************************************************
* Copyright (c) 2022 Cadence Design Systems, Inc.
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
    Softmax
    Optimized code for HiFi5
  IntegrIT, 2006-2019
*/

#include "NatureDSP_types.h"
#include "NatureDSP_Signal_math.h"
#include "common.h"

//#include "vec_alog_table.h"
/*-------------------------------------------------------------------------
  Softmax
  The function computes the softmax (normalized exponential function) of 
  input data. 32-bit fixed-point functions accept inputs in Q6.25 and form 
  outputs in Q16.15 format. 

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB (see Note below)
  f      floating point input, floating point output

  Note: Accuracy of function may depend on amount of data and their 
  distribution. Given accuracy is achieved for N=2 for any pair of data 
  from input domain.

  Input:
  x[N]   input data, Q6.25 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

-------------------------------------------------------------------------*/
void vec_softmax32x32(int32_t * y, const int32_t * x, int N)
{

  /* polynomial coefficients for 2^x, in range -1...0 */
  const int32_t ALIGN(32) pow2poly[] = { 14685058, 114217091, 514075394, 1488269031, 2147475316 };
  int n;

        ae_int32x4 * restrict pY  = (      ae_int32x4 *)y;
  const ae_int32x4 * restrict pYr = (const ae_int32x4 *)y;
  const ae_int32x4 * restrict pX  = (const ae_int32x4 *)x;

  ae_int32x2 y0, y1, t0, t1;
  ae_int32x2 x0, x1, e0, e1;
  ae_int32x2 x2, x3;
  ae_int32x2 B0, B1, B, E_SUM, X, Y, E;
  ae_valignx2 aX, aYr, aY;
  aX = AE_LA128_PP(pX);
  
  aY = AE_ZALIGN128();

  if (N <= 0) return;
  B0 = B1 = B = AE_MOVDA32(MIN_INT32);
  for (n=0; n<(N>>3); n++)
  {
    AE_LA32X2X2_IP(x0, x1, aX, pX);
    AE_LA32X2X2_IP(x2, x3, aX, pX);
    B0 = AE_MAX32(x0, AE_MAX32(x1, B0));
    B1 = AE_MAX32(x2, AE_MAX32(x3, B1));
  }
  for (n=0; n<(N&7); n++)
  {
    AE_L32_IP(X, castxcc(ae_int32, pX), 4);
    B0 = AE_MAX32(X, B0);
  }
  B = AE_MAX32(B0, B1);

  X = AE_SEL32_LH(B, B);
  B = AE_MAX32(X, B);
  __Pragma("no_reorder");
  E_SUM = AE_ZERO32();
  pX = (const ae_int32x4 *)x;
  aX = AE_LA128_PP(pX);
  for (n=0; n<(N>>2); n++)
  {
    AE_LA32X2X2_IP(x0, x1, aX, pX);
    x0 = AE_SUB32S(x0, B);
    x1 = AE_SUB32S(x1, B);
    AE_MULF2P32X4RAS(x0, x1, x0, x1, 774541002, 774541002);
    e0 = AE_SRAI32(x0, 23);
    e1 = AE_SRAI32(x1, 23);
    e0 = AE_SUB32(7, e0);//e+1-8
    e1 = AE_SUB32(7, e1);
    x0 = AE_AND32(x0, AE_MOVDA32(0x7fffff));
    x1 = AE_AND32(x1, AE_MOVDA32(0x7fffff));
    x0 = AE_SUB32(x0, AE_MOVDA32(0x800000));
    x1 = AE_SUB32(x1, AE_MOVDA32(0x800000));
    x0 = AE_SLAI32(x0, 8);//Q31
    x1 = AE_SLAI32(x1, 8);

    y0 = y1 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 0);
    t0 = t1 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 1);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    t0 = t1 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 2);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    t0 = t1 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 3);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    t0 = t1 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 4);
    AE_MULAF2P32X4RAS(t0, t1, x0, x1, y0, y1); y0 = t0; y1 = t1;
    x0 = AE_SRAV32RS(y0, e0);//Q31
    x1 = AE_SRAV32RS(y1, e1);
    E_SUM = AE_ADD32S(E_SUM, x0);
    E_SUM = AE_ADD32S(E_SUM, x1);
    AE_SA32X2X2_IP(x0, x1, aY, pY);
  }
  AE_SA128POS_FP(aY, pY);
  for (n = 0; n<(N&3); n++)
  {
    AE_L32_IP(x0, castxcc(ae_int32, pX), 4);
    x0 = AE_SUB32S(x0, B);
    x0 = AE_MULFP32X2RAS(x0, 774541002);
    e0 = AE_SRAI32(x0, 23);
    e0 = AE_SUB32(7, e0);//e+1-8
    x0 = AE_AND32(x0, AE_MOVDA32(0x7fffff));
    x0 = AE_SUB32(x0, AE_MOVDA32(0x800000));
    x0 = AE_SLAI32(x0, 8);//Q31

    y0 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 0);
    t0 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 1);
    AE_MULAFP32X2RAS(t0, x0, y0); y0 = t0;
    t0 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 2);
    AE_MULAFP32X2RAS(t0, x0, y0); y0 = t0;
    t0 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 3);
    AE_MULAFP32X2RAS(t0, x0, y0); y0 = t0;
    t0 = AE_L32_I((const ae_int32 *)pow2poly, 4 * 4);
    AE_MULAFP32X2RAS(t0, x0, y0); y0 = t0;
    x0 = AE_SRAV32RS(y0, e0);//Q31
    x0 = AE_SEL32_HL(AE_ZERO32(), x0);
    E_SUM = AE_ADD32S(E_SUM, x0);
    AE_S32_L_IP(x0, castxcc(ae_int32, pY), 4);
  }
  x0 = AE_SEL32_LH(E_SUM, E_SUM);
  E_SUM = AE_ADD32S(E_SUM, x0);
  __Pragma("no_reorder");
  {
    unsigned nsa;
    xtbool2 isZero;
    ae_f32x2 t;
    X = E_SUM;

    nsa = AE_NSAZ32_L(X) - 8;
    X = AE_SLAA32S(X, nsa);
    nsa+=1;
    isZero = AE_EQ32(X, AE_ZERO32());
    /* first approximation */
    Y = AE_SUB32(AE_MOVDA32((int32_t)0xBB5000), X);

    t = AE_MOVF32X2_FROMINT32X2(AE_MOVDA32(0x400000)); AE_MULSFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); E = t;
    E = AE_ADD32(E, E);
    t = Y; AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(Y), AE_MOVF24X2_FROMINT32X2(E)); Y = t;

    t = AE_MOVF32X2_FROMINT32X2(AE_MOVDA32(0x400000)); AE_MULSFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(X), AE_MOVF24X2_FROMINT32X2(Y)); E = t;
    E = AE_ADD32(E, E);
    t = Y; AE_MULAFP24X2RA(t, AE_MOVF24X2_FROMINT32X2(Y), AE_MOVF24X2_FROMINT32X2(E)); Y = t;

    y0 = AE_SLAA32S(Y, nsa); /* Q23 */
  }
  __Pragma("no_reorder");
  pY  = (      ae_int32x4 *)y;
  pYr = (const ae_int32x4 *)y;
  aYr = AE_LA128_PP(pYr);
  aY = AE_ZALIGN128();
  __Pragma("ymemory(pYr)");
  __Pragma("ymemory(pY)");
  for (n=0; n<(N>>3); n++)
  {
    AE_LA32X2X2_IP(x0, x1, aYr, pYr);
    AE_LA32X2X2_IP(x2, x3, aYr, pYr);
    AE_MULF2P32X4RAS(x0, x1, x0, x1, y0, y0); 
    AE_MULF2P32X4RAS(x2, x3, x2, x3, y0, y0); 
    AE_SA32X2X2_IP(x0, x1, aY, pY);
    AE_SA32X2X2_IP(x2, x3, aY, pY);
  }
  AE_SA128POS_FP(aY, pY);
  for (n=0; n<(N&7); n++)
  {
    AE_L32_IP(x0, castxcc(ae_int32, pYr), 4);
    x0 = AE_MULFP32X2RAS(x0, y0); 
    AE_S32_L_IP(x0, castxcc(ae_int32, pY), 4);
  }

} /* vec_softmax32x32() */
