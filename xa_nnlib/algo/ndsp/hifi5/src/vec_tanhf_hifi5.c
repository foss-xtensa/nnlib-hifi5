/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
/* Copyright (c) 2019 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ("Cadence    */
/* Libraries") are the copyrighted works of Cadence Design Systems Inc.	    */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
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
    Code optimized for HiFi5 core
    IntegrIT, 2006-2019
*/
#include "NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "common.h"
#include "common_fpu.h"

/* Tables and constants. */
#include "tanhf_tbl.h"
#include "expf_tbl.h"
#include "nanf_tbl.h"
#include "pow2f_tbl.h"

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
#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void,vec_tanhf,(float32_t* y, const float32_t* x, int N))
#elif HAVE_VFPU
#define sz_f32    (int)sizeof(float32_t)
static void __tanhf(float32_t* restrict y, const float32_t* restrict x, int N);
void vec_tanhf(float32_t* restrict y, const float32_t* restrict x, int N)
{
    xtfloatx4 * restrict pX;
    xtfloatx4 * restrict pY;
    int n;
    if (N<=0) return;
    if (N&7)
    {
        float32_t ALIGN(32) xbuf[32],ybuf[32];
        pX=(xtfloatx4 *)xbuf;
        pY=(xtfloatx4 *)ybuf;
        AE_SSX2X2_I(XT_CONST_S(0),XT_CONST_S(0),pX,0*sizeof(xtfloatx4));
        AE_SSX2X2_I(XT_CONST_S(0),XT_CONST_S(0),pX,1*sizeof(xtfloatx4));
        AE_SSX2X2_I(XT_CONST_S(0),XT_CONST_S(0),pY,0*sizeof(xtfloatx4));
        AE_SSX2X2_I(XT_CONST_S(0),XT_CONST_S(0),pY,1*sizeof(xtfloatx4));
        for (n=0; n<(N&7); n++) 
        {
            xtfloat t;
            XT_LSIP(t,castxcc(xtfloat,x  ),sizeof(float32_t));
            XT_SSIP(t,castxcc(xtfloat,pX ),sizeof(float32_t));
        }
        pX=(xtfloatx4 *)xbuf;
        __tanhf((float32_t*)pY,(float32_t*)pX,8);
        for (n=0; n<(N&7); n++) 
        {
            xtfloat t;
            XT_LSIP(t,castxcc(xtfloat,pY),sizeof(float32_t));
            XT_SSIP(t,castxcc(xtfloat,y ),sizeof(float32_t));
        }
        N&=~7;
    }
    if (N<=0) return;
    __tanhf(y,x,N);
}

static void __tanhf(float32_t* restrict y, const float32_t* restrict x, int N)
{
    const ae_int32* restrict pPolytanhf = (const ae_int32*)polytanhf_tbl;
    const xtfloatx4 * restrict X;
        xtfloatx4 * restrict Y;
    const xtfloatx4 * restrict S_rd;
        xtfloatx4 * restrict S_wr;
    /* Current block index; overall number of blocks; number of values in the current block */
    ae_valignx2 X_va, Y_va;
    /* Block size, blkLen <= blkSize */
    const int blkSize = MAX_ALLOCA_SZ/2*2*sz_f32;
    xtfloatx2 one = XT_CONST_S(1);
    xtfloatx2 two = XT_CONST_S(2);
    xtfloatx2 half = XT_CONST_S(3);
    /* Allocate a fixed-size scratch area on the stack. */
    float32_t ALIGN(32) scr[2*blkSize];
    int n,M;
    NASSERT(N%8==0);
    NASSERT_ALIGN16(scr);
  /*
  * Data are processed in blocks of scratch area size. Further, the algorithm
  * implementation is splitted in order to feed the optimizing compiler with a
  * few loops of manageable size.
  */

    for (;N>0; N-=M,x+=M,y+=M)
    {
        M= XT_MIN( N, blkSize );

        X = (xtfloatx4*)(x);
        S_wr = (xtfloatx4*)scr;
        X_va = AE_LA128_PP(X);
        /* argumant reduction phase */
        __Pragma("loop_count factor=2")
        for ( n=0; n<(M>>2); n++ )
        {
                xtfloatx2 x0, x1, p0, p1, y0, y1, t0, t1;
            AE_LASX2X2_IP(x0, x1, X_va, X);
            ABS_SX2X2(x0, x1, x0, x1);
            MULQ_S(x0, x1, x0, x1, two);
            t0 = (xtfloatx2)80.f;
            x0 = XT_MIN_SX2(x0, t0);
            x1 = XT_MIN_SX2(x1, t0);
            /* scale input to 1/ln(2) */
            MULQ_S(p0, p1, x0, x1, log2_e[0].f);
            p0 = XT_FIROUND_SX2(p0);
            p1 = XT_FIROUND_SX2(p1);
            NEG_SX2X2(y0, y1, p0, p1);
            MADDQ_S(y0, y1, x0, x1, log2_e[0].f);
            MADDQ_S(y0, y1, x0, x1, log2_e[1].f);
            AE_SSX2X2_IP(y0, y1, S_wr, 4 * sz_f32);
            /* saturating p0 to the right values */
            t0 = (xtfloatx2) 129.f; p0 = XT_MIN_SX2(p0, t0); p1 = XT_MIN_SX2(p1, t0);
            t1 = (xtfloatx2)-151.f; p0 = XT_MAX_SX2(p0, t1); p1 = XT_MAX_SX2(p1, t1);
            AE_SSX2X2_IP(p0, p1, S_wr, 4 * sz_f32);
        }

        /* compute 2^x via polynomal appoximation */
        __Pragma("no_reorder")
        S_wr = (xtfloatx4*)scr;
        S_rd = (xtfloatx4*)scr;
        pPolytanhf = (const ae_int32*)pow2f_coef;
        __Pragma("loop_count factor=2")
        for (n = 0; n<(M >> 2); n++)
        {
            xtfloatx2 x0, x1, dx0, dx1, z0, z1;
            xtfloatx2 c0_0, c1_0, c0_1, c1_1, c0_2, c1_2;

            xtfloatx2 y0, y1, y2, y3, y4, y5, y6;
            ae_int32x2 tmp;
            AE_LSX2X2_XP(x0, x1, S_rd, +2*4 * sz_f32);
            MUL_SX2X2(dx0, dx1, x0, x1, x0, x1);
      
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y3 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_XP(tmp,pPolytanhf,-4*(int)sizeof(float32_t));   y4 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            y5 = pow2f_coef[5].f;
            y6 = pow2f_coef[6].f;
            c0_0 = c1_0 = y1;      MADDQ_S(c0_0, c1_0, x0, x1, y0);
            c0_1 = c1_1 = y3;      MADDQ_S(c0_1, c1_1, x0, x1, y2);
            c0_2 = c1_2 = y5;      MADDQ_S(c0_2, c1_2, x0, x1, y4);

            MADD_SX2X2(c0_1, c1_1, dx0, dx1, c0_0, c1_0);
            MADD_SX2X2(c0_2, c1_2, dx0, dx1, c0_1, c1_1);
            z0 = z1 = y6;
            MADD_SX2X2(z0, z1, x0, x1, c0_2, c1_2);
            AE_SSX2X2_IP(z0, z1, S_wr, 2*4 * sz_f32);
        }

        /* resulted scaling by 2^N and final Newton-Raphson phase */
        __Pragma("no_reorder")
        S_wr = (xtfloatx4*)scr;
        S_rd = (xtfloatx4*)scr;
        __Pragma("loop_count factor=2")
        for (n = 0; n<(M>> 2); n++)
        {
            xtfloatx2 r0, r1, z0, z1;
            xtfloatx2 eps0, eps1;
            xtfloatx2 p0, p1, y0, y1;
            xtfloatx2 s0_0,s0_1,s1_0,s1_1;

            ae_int32x2 t0, t1;
            AE_LSX2X2_XP(y0, y1, S_rd, +4 * sz_f32);
            AE_LSX2X2_XP(p0, p1, S_rd, +4 * sz_f32);

            /* Apply exponential part to the result */
            t0 = XT_TRUNC_SX2(p0, 0);
            t1 = XT_TRUNC_SX2(p1, 0);

            t0 = AE_SUB32S(t0, 1);
            t1 = AE_SUB32S(t1, 1);

            s0_0=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(AE_SRLI32(t0,1)));
            s0_1=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(AE_SRLI32(t1,1)));
            s1_0=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(AE_SUB32S(t0,AE_SRLI32(t0,1))));
            s1_1=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(AE_SUB32S(t1,AE_SRLI32(t1,1))));
            MUL_SX2X2(y0, y1, y0, y1, s1_0,s1_1);
            MUL_SX2X2(y0, y1, y0, y1, s0_0,s0_1);
            ADD_SX2X2(z0, z1, y0, y1, half, half);
            /* Initial approximation for 1/y */
            r0 = XT_RECIP0_SX2(z0);
            r1 = XT_RECIP0_SX2(z1);
            /* 2 Newton-Raphson iterations for 1/z  */
            eps0 = eps1 = one;
            MSUB_SX2X2(eps0, eps1, z0, z1, r0, r1);
            MADD_SX2X2(r0, r1, r0, r1, eps0, eps1);
            eps0 = eps1 = one;
            MSUB_SX2X2(eps0, eps1, z0, z1, r0, r1);
            MADD_SX2X2(r0, r1, r0, r1, eps0, eps1);
            SUB_SX2X2(z0, z1, one, one, r0, r1);
            AE_SSX2X2_IP(z0, z1, S_wr, 2 * 4 * sz_f32);
        }    
        /* next, compute output for smaller argument
        Use polynomial approximation for small input values. This branch is
        also used for a NaN on input.
        */
        __Pragma("no_reorder")
        S_wr = ((xtfloatx4*)scr)+1;
        S_rd = (xtfloatx4*)scr;
        X = (xtfloatx4*)(x);
        X_va = AE_LA128_PP(X);
        pPolytanhf=(const ae_int32*)polytanhf_tbl;
        __Pragma("loop_count factor=2")
        for (n = 0; n<(M >> 2); n++)
        {
            xtfloatx2 x0, x1, dx0, dx1, tx0, tx1, t0, t1;;
            xtfloatx2 z0, z1, tn0, tn1, tn2, tn3;
            ae_int32x2 tmp;
            AE_LASX2X2_IP(x0, x1, X_va, X);
            ABS_SX2X2(x0, x1, x0, x1);
            MUL_SX2X2(dx0, dx1, x0, x1, x0, x1);
            MUL_SX2X2(tx0, tx1, x0, x1, dx0, dx1);

            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_XP(tmp,pPolytanhf,-3*(int)sizeof(float32_t));   tn3 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            z0 = z1 = tn1;      MADDQ_S   (z0, z1, dx0, dx1,    tn0); t0 = z0; t1 = z1;
            z0 = z1 = tn2;      MADD_SX2X2(z0, z1, t0, t1, dx0, dx1); t0 = z0; t1 = z1;
            z0 = z1 = tn3;      MADD_SX2X2(z0, z1, t0, t1, dx0, dx1); t0 = z0; t1 = z1;
            z0 = x0; z1 = x1;
            MADD_SX2X2(z0, z1, t0, t1, tx0, tx1);
            AE_SSX2X2_IP(z0, z1, S_wr, 2 * 4 * sz_f32);
        }    
        /* final stage: select right output and apply sign */
        __Pragma("no_reorder")
        X = (xtfloatx4*)(x);
        Y = (xtfloatx4*)y;
        S_rd = (xtfloatx4*)scr;
        X_va = AE_LA128_PP(X);
        Y_va = AE_ZALIGN128();
        __Pragma("loop_count factor=2")
        for (n = 0; n<(M>> 2); n++)
        {
            xtbool2 b0big, b0sign, b1big, b1sign;
            xtfloatx2 x0, x1, z0, z1, z0big, z1big;
            ae_int32x2 ux0, ux1;
            AE_LASX2X2_IP(x0, x1, X_va, X);
            ux0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
            ux1 = XT_AE_MOVINT32X2_FROMXTFLOATX2(x1);
            b0sign = AE_LT32(ux0, 0);
            b1sign = AE_LT32(ux1, 0);
            ABS_SX2X2(x0, x1, x0, x1);
            b0big = XT_OLT_SX2(halfln3.f, x0);
            b1big = XT_OLT_SX2(halfln3.f, x1);
            AE_LSX2X2_XP(z0big, z1big, S_rd, +4 * sz_f32);
            AE_LSX2X2_XP(z0, z1, S_rd, +4 * sz_f32);
            XT_MOVT_SX2(z0, z0big, b0big);
            XT_MOVT_SX2(z1, z1big, b1big);
            /* apply sign */
            XT_MOVT_SX2(z0, XT_NEG_SX2(z0), b0sign);
            XT_MOVT_SX2(z1, XT_NEG_SX2(z1), b1sign);
            AE_SASX2X2_IP(z0, z1, Y_va, Y);
        }
        AE_SA128POS_FP( Y_va, Y );
    }
} /* vec_tanhf() */
#else
// code for scalar FPU
void vec_tanhf(float32_t* restrict y, const float32_t* restrict x, int N)
{
    xtfloat zero, one, two, half;
    int n;
    const xtfloat* restrict pX=(const xtfloat*)x;
          xtfloat* restrict pY=(      xtfloat*)y;
    zero = XT_CONST_S(0);
    one  = XT_CONST_S(1);
    two  = XT_CONST_S(2);
    half = XT_CONST_S(3);
    for (n = 0; n < N; n++) 
    {
        xtbool bsmall;
        xtfloat x,y;
        xtfloat z, r, eps, zsmall,xin;
        xtfloat p0, dy, y1;
        int32_t ux;
        int32_t e1, e2;
        XT_LSIP(x,pX,sizeof(float32_t));
        ux = XT_RFR(x); 
        ux = (ux & 0x80000000);
        x = XT_ABS_S(x);
        bsmall = XT_OLT_S(halfln3.f,x);
        xin=x;
        /* compute output for smaller argument */
        {
            /*
            * For a large input value tanh(x) is computed from exp(2*x)/2, using
            * the following identity: tanh(x) == 1 - 2/(exp(2*x)+1)
            */
            r = zero; XT_MADDN_S(r, two, x); x = r;
            {
                xtfloat t=(xtfloat)80.f;
                x = XT_MIN_S(x, t);
            }

            /* scale input to 1/ln(2) */
            p0 = XT_MUL_S(x, log2_e[0].f);
            #if defined(XT_FIROUND_S)
            p0 = XT_FIROUND_S(p0);
            #else
            p0 = XT_FLOAT_S(XT_ROUND_S(p0, 0), 0);
            #endif
            dy = XT_NEG_S(p0);
            XT_MADD_S(dy, x, log2_e[0].f);
            XT_MADD_S(dy, x, log2_e[1].f);
            /* compute 2^x */
            {
                float32_t y0, y2, y3, y4, y5, y6, dy2;
                dy2 = XT_MUL_S(dy, dy);
                y0 = pow2f_coef[0].f;
                y1 = pow2f_coef[1].f;
                y2 = pow2f_coef[2].f;
                y3 = pow2f_coef[3].f;
                y4 = pow2f_coef[4].f;
                y5 = pow2f_coef[5].f;
                y6 = pow2f_coef[6].f;
                XT_MADD_S(y1, y0, dy);
                XT_MADD_S(y3, y2, dy);
                XT_MADD_S(y5, y4, dy);

                XT_MADD_S(y3, y1, dy2);
                XT_MADD_S(y5, y3, dy2);
                XT_MADD_S(y6, y5, dy);
                y = y6;
            }

            /* resulted scaling */
            {
                xtfloat t;
                t=(xtfloat) 129.f;p0=XT_MIN_S(p0,t);
                t=(xtfloat)-151.f;p0=XT_MAX_S(p0,t);
            }

            /* Apply exponential part to the result */
            {
                uint32_t tmp, v1, v2;
                tmp = XT_TRUNC_S(p0, 0);
                tmp = tmp+254 - 30 - 1;
                v1 = (tmp>>1);
                v2 = (tmp-v1);
                e1 = (v1<<23);
                e2 = (v2<<23);
            }

            /*
            * Convert (y*2^(ex-30))/2 to floating-point p == exp(x)/2
            */
            r = XT_MUL_S(y, 1073741824.f);
            y = XT_MUL_S(r, XT_WFR(e2));
            y = XT_MUL_S(y, XT_WFR(e1));
            z = XT_ADD_S(y, half);
            /* Initial approximation for 1/y */
            r = XT_RECIP0_S(z);
            /* 2 Newton-Raphson iterations for 1/z  */
            eps = one; XT_MSUB_S(eps, z, r);
            XT_MADD_S(r, r, eps);
            eps = one; XT_MSUB_S(eps, z, r);
            XT_MADD_S(r, r, eps);
            zsmall = XT_SUB_S(one, r);
        }
        /* compute output for bigger argument */
        {
            /*
            * Use polynomial approximation for small input values. This branch is
            * also used for a NaN on input.
            */
            x=xin;
            float32_t x2, x3, tn0, tn1, tn2, tn3;
            x2 = XT_MUL_S(x, x);
            x3 = XT_MUL_S(x, x2);
            tn0 = polytanhf_tbl[0].f;
            tn1 = polytanhf_tbl[1].f;
            tn2 = polytanhf_tbl[2].f;
            tn3 = polytanhf_tbl[3].f;
            XT_MADD_S(tn1, tn0, x2);
            XT_MADD_S(tn2, tn1, x2);
            XT_MADD_S(tn3, tn2, x2);
            z = x;
            XT_MADD_S(z, tn3, x3);
        }
        XT_MOVT_S(z,zsmall,bsmall);
        /* apply sign */
        XT_MOVT_S(z,XT_NEG_S(z),AE_MOVBA(((uint32_t)ux)>>31));
        XT_SSIP(z,pY,sizeof(float32_t));
    }
}
#endif
