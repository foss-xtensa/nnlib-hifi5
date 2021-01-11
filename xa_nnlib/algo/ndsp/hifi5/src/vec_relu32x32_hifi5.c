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
    Rectifier functions
    Code optimized for HiFi5
  IntegrIT, 2006-2019
*/
#include "NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "common.h"

/*-------------------------------------------------------------------------
  Rectifier function
  The functions compute the rectifier linear unit function of input argument. 
  32-bit fixed-point functions accept inputs in Q6.25 and form outputs in 
  Q16.15 format. Parameter K allows to set upper threshold for proper 
  compression of output signal.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output. Accuracy 2 ULP
  Input:
  x[N]   input data, Q6.25 or floating point
  K      threshold, Q16.15 or floating point
  N      length of vectors
  Output:
  y[N]   result, Q16.15 or floating point

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q16.15 or floating point
-------------------------------------------------------------------------*/
void vec_relu32x32 (int32_t   * y, const int32_t   * x, int32_t   K, int N)
{
    const ae_int32x4 * restrict pX;
          ae_int32x4 * restrict pY;
    ae_valignx2 aX,aY;
    int n;
    if (N<=0) return;
    if(K<=0) K=0;
    pX=(const ae_int32x4 *)x;
    pY=(      ae_int32x4 *)y;
    aY=AE_ZALIGN128();
    aX=AE_LA128_PP(pX);
    for (n=0; n<(N>>4); n++)
    {
        ae_int32x2 x0,x1,x2,x3,x4,x5,x6,x7;
        AE_LA32X2X2_IP(x0,x1,aX,pX);
        AE_LA32X2X2_IP(x2,x3,aX,pX);
        AE_LA32X2X2_IP(x4,x5,aX,pX);
        AE_LA32X2X2_IP(x6,x7,aX,pX);
        AE_MUL2P32X4T(x0,x1,x0,x1,1<<22,1<<22);
        AE_MUL2P32X4T(x2,x3,x2,x3,1<<22,1<<22);
        AE_MUL2P32X4T(x4,x5,x4,x5,1<<22,1<<22);
        AE_MUL2P32X4T(x6,x7,x6,x7,1<<22,1<<22);
        AE_MINMAX32(x0,0,K); AE_MINMAX32(x1,0,K);
        AE_MINMAX32(x2,0,K); AE_MINMAX32(x3,0,K);
        AE_MINMAX32(x4,0,K); AE_MINMAX32(x5,0,K);
        AE_MINMAX32(x6,0,K); AE_MINMAX32(x7,0,K);
        AE_SA32X2X2_IP(x0,x1,aY,pY);
        AE_SA32X2X2_IP(x2,x3,aY,pY);
        AE_SA32X2X2_IP(x4,x5,aY,pY);
        AE_SA32X2X2_IP(x6,x7,aY,pY);
    }
    N&=15;
    __Pragma("no_unroll")
    __Pragma("loop_count max=3")
    for (n=0; n<(N>>2); n++)
    {
        ae_int32x2 x0,x1;
        AE_LA32X2X2_IP(x0,x1,aX,pX);
        AE_MUL2P32X4T(x0,x1,x0,x1,1<<22,1<<22);
        AE_MINMAX32(x0,0,K);
        AE_MINMAX32(x1,0,K);
        AE_SA32X2X2_IP(x0,x1,aY,pY);
    }
    AE_SA128POS_FP(aY,pY);
    N&=3;
    if (N)
    {
        __Pragma("no_unroll")
        __Pragma("loop_count max=3")
        for (n=0; n<(N); n++)
        {
            ae_int32x2 x0;
            AE_L32_IP(x0,castxcc(ae_int32,pX),sizeof(ae_int32));
            x0=AE_SRAI32(x0,10); // convert from Q25 to Q15
            AE_MINMAX32(x0,0,K);
            AE_S32_L_IP(x0,castxcc(ae_int32,pY),sizeof(ae_int32));
        }
    }
} /* vec_relu32x32() */
