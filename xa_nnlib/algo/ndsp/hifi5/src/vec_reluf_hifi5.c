/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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
    Code optimized for HiFi5 core
  IntegrIT, 2006-2019
*/
#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"

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
#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void,xa_nnlib_vec_reluf,(float32_t * restrict y, const float32_t * restrict x, float32_t K, int N))
#elif HAVE_VFPU
void xa_nnlib_vec_reluf     (float32_t * y, const float32_t * x, float32_t K, int N)
{
    int n,N0;
    xtfloatx2 x0,x1,x2,x3;
    ae_valignx2 aX,aY;
    const xtfloatx4* restrict pX=(const xtfloatx4*)x;
          xtfloatx4* restrict pY=(      xtfloatx4*)y;
    xtfloatx2 zero=XT_CONST_SX2(0);
    if (N<=7)
    {
    xtfloat* pX_t = castxcc(xtfloat,pX);
    xtfloat* pY_t = castxcc(xtfloat,pY);
        __Pragma("no_unroll")
        __Pragma("loop_count max=7")
        for (n=0; n<N; n++)
        {
            xtfloat t;
            XT_LSIP(t,pX_t,sizeof(xtfloat));
            t=XT_MIN_S(*(xtfloat*)&(K),t);
            t=XT_MAX_S(t,AE_MOVXTFLOAT_FROMXTFLOATX2(zero));
            XT_SSIP(t,pY_t,sizeof(xtfloat));
        }
        pX = (xtfloatx4*)pX_t;
        pY = (xtfloatx4*)pY_t;
        return;
    }
    aX=AE_LA128_PP(pX);
    aY=AE_ZALIGN128();
    AE_LASX2X2_IP(x0,x1,aX,pX);
    AE_LASX2X2_IP(x2,x3,aX,pX);
    x0=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x0),zero);
    x1=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x1),zero);
    x2=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x2),zero);
    x3=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x3),zero);
    AE_SASX2X2_IP(x0,x1,aY,pY);
    AE_SASX2X2_IP(x2,x3,aY,pY);
    AE_SA128POS_FP(aY,pY);
    N0=((N-1)&7)+1;
    N-=N0;
    if (N<=0) return;
    pX=(const xtfloatx4*)(x+N0);
    pY=(      xtfloatx4*)(y+N0);
    aX=AE_LA128_PP(pX);
    for (n=0; n<(N>>3); n++)
    {
        AE_LASX2X2_IP(x0,x1,aX,pX);
        AE_LASX2X2_IP(x2,x3,aX,pX);
        x0=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x0),zero);
        x1=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x1),zero);
        x2=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x2),zero);
        x3=XT_MAX_SX2(XT_MIN_SX2(AE_MOVXTFLOATX2_FROMXTFLOAT(*(xtfloat*)&K),x3),zero);
        AE_SASX2X2_IP(x0,x1,aY,pY);
        AE_SASX2X2_IP(x2,x3,aY,pY);
    }
    AE_SA128POS_FP(aY,pY);
} /* xa_nnlib_vec_reluf() */
#else
// code for scalar FPU
void xa_nnlib_vec_reluf     (float32_t * y, const float32_t * x, float32_t K, int N)
{
    const xtfloat* restrict pX=(const xtfloat*)x;
          xtfloat* restrict pY=(      xtfloat*)y;
    xtfloat t,zero=XT_CONST_S(0);
    xtbool bbig,bneg;
    int n;
    for(n=0; n<N; n++)
    {
        XT_LSIP(t,pX,sizeof(float32_t));
        bbig=XT_OLT_S(K,t);
        XT_MOVT_S(t,K,bbig);
        bneg=XT_OLT_S(t,zero);
        XT_MOVT_S(t,zero,bneg);
        XT_SSIP(t,pY,sizeof(float32_t));
    }
} /* xa_nnlib_vec_reluf() */
#endif
