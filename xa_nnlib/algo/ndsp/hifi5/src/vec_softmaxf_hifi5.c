/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
    Code optimized for HiFi5 core
  IntegrIT, 2006-2019
*/
#include "NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "common.h"
#include "common_fpu.h"
#include "inff_tbl.h"
#include "nanf_tbl.h"
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
#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void,vec_softmaxf,(float32_t * y, const float32_t * x,int N))
#elif HAVE_VFPU
void vec_softmaxf    (float32_t * y, const float32_t * x,int N)
{
    const xtfloatx4 * restrict pX;
          xtfloatx4 * restrict pY;
    const xtfloat   * restrict px;
          xtfloat   * restrict py;
    static const int32_t ALIGN(32) seq[]={0,1,2,3,4,5,6,7};
    int n;
    xtfloatx2 xmax;
    xtfloatx2 ysum;
    int N0;
    xtbool2 b01,b23,b45,b67;
    ae_int32x2 s01,s23,s45,s67;
    AE_L32X2X2_I(s01,s23,(const ae_int32x4*)seq,0*sizeof(ae_int32x4));
    AE_L32X2X2_I(s45,s67,(const ae_int32x4*)seq,1*sizeof(ae_int32x4));
    // separate case for small N
    if (N<=0) return;
    if (N<=7)
    {
        xtfloat xmax,ysum;
        /* compute maximum of x */
        xmax=minusInff.f;
        px=(const xtfloat *)x;
        py=(      xtfloat *)y;
        __Pragma("loop_count min=1,max=7")
        for (n=0; n<N; n++)
        {
            xtfloat t;
            XT_LSIP(t,px,sizeof(xtfloat));
            xmax=XT_MAX_S(xmax,t);
        }
        /* subtract maximum of x from input data */
        px=(const xtfloat*)x;
        __Pragma("loop_count min=1,max=7")
        for (n=0; n<N; n++)
        {
            xtfloat t;
            XT_LSIP(t,px,sizeof(xtfloat));
            t=XT_SUB_S(t,xmax);
            XT_SSIP(t,py,sizeof(xtfloat));
        }
        /* compute exp() */
        vec_antilognf(y,y,N);
        /* sum results */
        py=(xtfloat*)y;
        ysum=XT_CONST_S(0);
        __Pragma("loop_count min=1,max=7")
        for (n=0; n<N; n++)
        {
            xtfloat t;
            XT_LSIP(t,py,sizeof(xtfloat));
            ysum=XT_ADD_S(ysum,t);
        }
        /* normalize output */
        ysum=XT_RECIP_S(ysum);
        __Pragma("no_reorder")
        px=(xtfloat*)y;
        py=(xtfloat*)y;
        __Pragma("loop_count min=1,max=7")
        for (n=0; n<N; n++) 
        {
            xtfloat t;
            XT_LSIP(t,px,sizeof(xtfloat));
            t=XT_MUL_S(t,ysum);
            XT_SSIP(t,py,sizeof(xtfloat));
        }
        return;
    }
    // ok, here is N>=8
    NASSERT(N>=8);
    N0=((N-1)&7)+1;
    N-=N0;
    b01=AE_LT32(s01,N0);    // mask unnessesary elements on the first iteration
    b23=AE_LT32(s23,N0);
    b45=AE_LT32(s45,N0);
    b67=AE_LT32(s67,N0);

    /* compute maximum of x */
    {
        ae_valignx2 aX;
        xtfloatx2 x0,x1,x2,x3,max0,max1;
        pX=(const xtfloatx4 *)x;
        aX=AE_LA128_PP(pX);
        AE_LASX2X2_IP(x0,x1,aX,pX);
        AE_LASX2X2_IP(x2,x3,aX,pX);
        pX=(const xtfloatx4 *)(x+N0);
        aX=AE_LA128_PP(pX);
        max0=MAXNUM_SX2(x0,x1);
        max1=MAXNUM_SX2(x2,x3);
        for (n=0; n<(N>>3); n++) 
        {
            AE_LASX2X2_IP(x0,x1,aX,pX);
            AE_LASX2X2_IP(x2,x3,aX,pX);
            max0=MAXNUM_SX2(max0,MAXNUM_SX2(x0,x1));
            max1=MAXNUM_SX2(max1,MAXNUM_SX2(x2,x3));
        }
        max0=MAXNUM_SX2(max0,max1);
        max1=AE_SEL32_LH_SX2(max0,max0);
        xmax=MAXNUM_SX2(max1,max0);
    }

        /* subtract maximum of x from input data */
        pX=(const xtfloatx4*)x;
        pY=(      xtfloatx4*)y;
        {
            xtfloatx2 x0,x1,x2,x3;
            xtfloatx2 y0,y1,y2,y3;
            ae_valignx2 aX,aY;
            aX=AE_LA128_PP(pX);
            aY=AE_ZALIGN128();
            AE_LASX2X2_IP(x0,x1,aX,pX);
            AE_LASX2X2_IP(x2,x3,aX,pX);
            MOV_SX2X2(y0,y1,xmax,xmax);
            MOV_SX2X2(y2,y3,xmax,xmax);
            XT_MOVF_SX2(y0,0,b01);
            XT_MOVF_SX2(y1,0,b23);
            XT_MOVF_SX2(y2,0,b45);
            XT_MOVF_SX2(y3,0,b67);
            SUB_SX2X2(x0,x1,x0,x1,y0,y1);
            SUB_SX2X2(x2,x3,x2,x3,y2,y3);
            AE_SASX2X2_IP(x0,x1,aY,pY);
            AE_SASX2X2_IP(x2,x3,aY,pY);
            AE_SA128POS_FP(aY,pY);
            pX=(const xtfloatx4 *)(x+N0);
            pY=(      xtfloatx4 *)(y+N0);
            aX=AE_LA128_PP(pX);
            for (n=0; n<(N>>3); n++) 
            {
                AE_LASX2X2_IP(x0,x1,aX,pX);
                AE_LASX2X2_IP(x2,x3,aX,pX);
                SUB_SX2X2(x0,x1,x0,x1,xmax,xmax);
                SUB_SX2X2(x2,x3,x2,x3,xmax,xmax);
                AE_SASX2X2_IP(x0,x1,aY,pY);
                AE_SASX2X2_IP(x2,x3,aY,pY);
            }
            AE_SA128POS_FP(aY,pY);
        }
        /* compute exp() */
        vec_antilognf(y,y,N+N0);
        /* sum results */
        pY=(xtfloatx4*)y;
        {
            xtfloatx2 x0,x1,x2,x3;
            xtfloatx2 s0,s1,s2,s3;
            ae_valignx2 aY;
            aY=AE_LA128_PP(pY);
            AE_LASX2X2_IP(x0,x1,aY,pY);
            AE_LASX2X2_IP(x2,x3,aY,pY);
            CONST_SX2X2(s0,s1,0);
            CONST_SX2X2(s2,s3,0);
            XT_MOVF_SX2(x0,0,b01);
            XT_MOVF_SX2(x1,0,b23);
            XT_MOVF_SX2(x2,0,b45);
            XT_MOVF_SX2(x3,0,b67);
            ADD_SX2X2(s0,s1,x0,x1,s0,s1);
            ADD_SX2X2(s2,s3,x2,x3,s2,s3);
            AE_SA128POS_FP(aY,pY);
            pY=(      xtfloatx4 *)(y+N0);
            aY=AE_LA128_PP(pY);
            for (n=0; n<(N>>3); n++) 
            {
                AE_LASX2X2_IP(x0,x1,aY,pY);
                AE_LASX2X2_IP(x2,x3,aY,pY);
                ADD_SX2X2(s0,s1,x0,x1,s0,s1);
                ADD_SX2X2(s2,s3,x2,x3,s2,s3);
            }
            ADD_SX2X2(s0,s1,s0,s1,s2,s3);
            s0=ADD_SX2(s0,s1);
            ysum=ADD_HL_LH_S(s0,s0);
        }
        /* normalize output */
        ysum=XT_RECIP_S(ysum);
        __Pragma("no_reorder")
        pX=(xtfloatx4*)y;
        pY=(xtfloatx4*)y;
        {
            xtfloatx2 x0,x1,x2,x3;
            xtfloatx2 y0,y1,y2,y3;
            ae_valignx2 aX,aY;
            aX=AE_LA128_PP(pX);
            aY=AE_ZALIGN128();
            AE_LASX2X2_IP(x0,x1,aX,pX);
            AE_LASX2X2_IP(x2,x3,aX,pX);
            MOV_SX2X2(y0,y1,ysum,ysum);
            MOV_SX2X2(y2,y3,ysum,ysum);
            XT_MOVF_SX2(y0,XT_CONST_S(1),b01);
            XT_MOVF_SX2(y1,XT_CONST_S(1),b23);
            XT_MOVF_SX2(y2,XT_CONST_S(1),b45);
            XT_MOVF_SX2(y3,XT_CONST_S(1),b67);
            MUL_SX2X2(x0,x1,x0,x1,y0,y1);
            MUL_SX2X2(x2,x3,x2,x3,y2,y3);
            AE_SASX2X2_IP(x0,x1,aY,pY);
            AE_SASX2X2_IP(x2,x3,aY,pY);
            AE_SA128POS_FP(aY,pY);
            pX=(const xtfloatx4 *)(y+N0);
            pY=(      xtfloatx4 *)(y+N0);
            aX=AE_LA128_PP(pX);
            for (n=0; n<(N>>3); n++) 
            {
                AE_LASX2X2_IP(x0,x1,aX,pX);
                AE_LASX2X2_IP(x2,x3,aX,pX);
                MULQ_S(x0,x1,x0,x1,ysum);
                MULQ_S(x2,x3,x2,x3,ysum);
                AE_SASX2X2_IP(x0,x1,aY,pY);
                AE_SASX2X2_IP(x2,x3,aY,pY);
            }
            AE_SA128POS_FP(aY,pY);
        }
} /* vec_softmaxf() */
#else
// code for scalar FPU
void vec_softmaxf    (float32_t * y, const float32_t * x,int N)
{
    const xtfloat* restrict pX=(const xtfloat*)x;
          xtfloat* restrict pY=(      xtfloat*)y;
    int n;
    xtfloat xmax,ysum;
    if (N<0) return;
    /* compute maximum of x */
    xmax=minusInff.f;
    for (n=0; n<N; n++)
    {
        xtfloat t;
        XT_LSIP(t,pX,sizeof(xtfloat));
        XT_MOVT_S(xmax,t,XT_OLT_S(xmax,t));
    }
    /* subtract maximum of x from input data */
    pX=(const xtfloat*)x;
    for (n=0; n<N; n++)
    {
        xtfloat t;
        XT_LSIP(t,pX,sizeof(xtfloat));
        t=XT_SUB_S(t,xmax);
        XT_SSIP(t,pY,sizeof(xtfloat));
    }
    /* compute exp() */
    vec_antilognf(y,y,N);
    /* sum results */
    pY=(xtfloat*)y;
    ysum=XT_CONST_S(0);
    for (n=0; n<N; n++)
    {
        xtfloat t;
        XT_LSIP(t,pY,sizeof(xtfloat));
        ysum=XT_ADD_S(ysum,t);
    }
    /* normalize output */
    ysum=XT_RECIP_S(ysum);
    __Pragma("no_reorder")
    pX=(xtfloat*)y;
    pY=(xtfloat*)y;
    for (n=0; n<N; n++) 
    {
        xtfloat t;
        XT_LSIP(t,pX,sizeof(xtfloat));
        t=XT_MUL_S(t,ysum);
        XT_SSIP(t,pY,sizeof(xtfloat));
    }
} /* vec_softmaxf() */
#endif
