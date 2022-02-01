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
    Sigmoid
    Code optimized for HiFi5 core
  IntegrIT, 2006-2019
*/
#include "NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "common.h"
#include "common_fpu.h"

#define sz_f32    (int)sizeof(float32_t)

/*-------------------------------------------------------------------------
  Sigmoid
  The functions compute the sigmoid of input argument. 32-bit fixed-point 
  functions accept inputs in Q6.25 and form outputs in Q16.15 format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      floating point input, floating point output. Accuracy 2 ULP
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
DISCARD_FUN(void,vec_sigmoidf,(float32_t * y, const float32_t * x, int N))
#elif HAVE_VFPU

static void __sigmoidf(float32_t * y, const float32_t * x, int N, float32_t* scr)
{
  static const union ufloat32uint32 ALIGN(32) c[] = { { 0x3fb8aa3b }, { 0x32a57060 } };
  static const union ufloat32uint32 ALIGN(32) p[] = { { 0x39222a75 }, { 0x3aaf9334 }, { 0x3c1d94fc }, { 0x3d63578b }, { 0x3e75fdf0 }, { 0x3f317218 }, { 0x3f800000 } };

  const xtfloatx4 * restrict X;
        xtfloatx4 * restrict Y;
  const xtfloatx4 * restrict S_rd;
        xtfloatx4 * restrict S_wr;
  const ae_int32  * restrict pP = (const ae_int32  *)p;

  /* Current block index; overall number of blocks; number of values in the current block */
  int M;
  ae_valignx2 X_va, Y_va;

  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ/2*2*sz_f32;
  int n;
  NASSERT_ALIGN16(scr);
  NASSERT(N%8==0);

  for (;N>0; x+=M, y+=M, N-=M)
  {
    M= XT_MIN( N, blkSize );

    X = (xtfloatx4*)(x);
    S_wr = (xtfloatx4*)scr;

    X_va = AE_LA128_PP(X);
    /* argumant reduction phase */
    __Pragma("loop_count factor=2")
    for ( n=0; n<(M>>2); n++ )
    {
        xtfloatx2 x0, x1, p0, p1, y0, y1, d0, d1;
        xtbool2 s0, s1;
        ae_int32x2 n0, n1;
        AE_LASX2X2_IP(x0, x1, X_va, X);
        s0 = XT_OLT_SX2(x0, XT_CONST_S(0));
        s1 = XT_OLT_SX2(x1, XT_CONST_S(0));
        ABS_SX2X2(p0, p1, x0, x1);
        NEG_SX2X2(x0, x1, p0, p1);

        x0 = XT_MAX_SX2(-103.9721f, x0);
        x1 = XT_MAX_SX2(-103.9721f, x1);
        /* compute d+n=log2(e)*x */
        MULQ_S(p0, p1, x0, x1, c[0].f);
        y0 = XT_FIROUND_SX2(p0);
        y1 = XT_FIROUND_SX2(p1);
        NEG_SX2X2(d0, d1, y0, y1);
        MADDQ_S(d0, d1, x0, x1, c[0].f);
        MADDQ_S(d0, d1, x0, x1, c[1].f);

        n0 = XT_TRUNC_SX2(y0, 0);
        n1 = XT_TRUNC_SX2(y1, 0);
        AE_S32X2X2_IP(n0, n1, castxcc(ae_int32x4, S_wr), 4 * sz_f32);
        AE_SSX2X2_IP(d0, d1, S_wr, 4 * sz_f32);
    }
    // second phase: compute polynomial approximation
    __Pragma("no_reorder")
    S_wr = (xtfloatx4*)scr;
    S_rd = (xtfloatx4*)scr;
    X = (xtfloatx4*)(x);
    Y = (xtfloatx4*)y;

    S_wr = ((xtfloatx4*)scr)+1;
    Y_va = AE_ZALIGN128();
    X_va = AE_LA128_PP(X);
    __Pragma("loop_count factor=2")
    for (n = 0; n<(M >> 2); n++)
    {
        xtbool2 s0, s1;
        ae_int32x2 n0, n1;
        xtfloatx2 x0, x1, d0, d1, z0, z1, t0, t1;
        xtfloatx2 dd0, dd1, c0, c1;
        AE_LASX2X2_IP(x0, x1, X_va, X);
        AE_L32X2X2_IP(n0, n1, castxcc(ae_int32x4, S_rd), +4 * sz_f32);
        AE_LSX2X2_IP(d0, d1, S_rd, +4 * sz_f32);

        s0 = XT_OLT_SX2(x0, XT_CONST_S(0));
        s1 = XT_OLT_SX2(x1, XT_CONST_S(0));
        /* approx 2^d */
        MUL_SX2X2(dd0, dd1, d0, d1, d0, d1);
        { ae_int32x2 tmp; AE_L32_IP(tmp, pP, sizeof(float32_t)); c0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
        { ae_int32x2 tmp; AE_L32_IP(tmp, pP, sizeof(float32_t)); c1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
        { ae_int32x2 tmp; AE_L32_IP(tmp, pP, sizeof(float32_t)); t0 = t1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); } 
        MADDQ_S(t0, t1, dd0, dd1, c0); z0 = t0; z1 = t1;
        { ae_int32x2 tmp; AE_L32_IP(tmp, pP, sizeof(float32_t)); t0 = t1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
        MADDQ_S(t0, t1, dd0, dd1, c1); c0 = t0; c1 = t1;
        { ae_int32x2 tmp; AE_L32_IP(tmp, pP, sizeof(float32_t)); t0 = t1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
        MADD_SX2X2(t0, t1, dd0, dd1, z0, z1); z0 = t0; z1 = t1;
        { ae_int32x2 tmp; AE_L32_XP(tmp, pP, -5 * (int)sizeof(float32_t)); t0 = t1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp); }
        MADD_SX2X2(t0, t1, dd0, dd1, c0, c1); c0 = t0; c1 = t1;

        MADD_SX2X2(c0, c1, d0, d1, z0, z1); z0 = c0; z1 = c1;
        CONST_SX2X2(t0,t1,1); 
        MADD_SX2X2(t0, t1, d0, d1, z0, z1); z0 = t0; z1 = t1;

        t0=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(AE_MAX32(-127,n0)));
        t1=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(AE_MAX32(-127,n1)));
        MUL_SX2X2(c0, c1, z0, z1, t0, t1);
        /* save orignal sign as a sign of polynomial */
        XT_MOVT_SX2(z0, XT_NEG_SX2(z0), s0);
        XT_MOVT_SX2(z1, XT_NEG_SX2(z1), s1);

        AE_SASX2X2_IP(z0, z1, Y_va, Y);
        AE_SSX2X2_IP(c0, c1, S_wr, 2*4 * sz_f32);
    }
    AE_SA128POS_FP( Y_va, Y );

    // last phase: scale polynomial to 2^n and compute 1/(1+x)
    __Pragma("no_reorder")
    S_wr = ((xtfloatx4*)scr);
    S_rd = (xtfloatx4*)scr;
    X = (xtfloatx4*)(y);
    Y = (xtfloatx4*)y;

    S_wr = (xtfloatx4*)scr;
    Y_va = AE_ZALIGN128();
    X_va = AE_LA128_PP(X);
    __Pragma("loop_count factor=2")
    for (n = 0; n<(M >> 2); n++)
    {
        xtfloatx2 s0_0,s0_1,s1_0,s1_1;
        xtbool2 s0, s1;
        ae_int32x2 n0, n1;
        ae_int32x2 e0_0, e0_1, e1_0, e1_1;
        xtfloatx2 x0, x1, z0, z1;
        xtfloatx2 y0, y1;
        AE_LASX2X2_IP(z0, z1, X_va, X);
        AE_L32X2X2_IP(n0, n1, castxcc(ae_int32x4, S_rd), +4 * sz_f32);
        /* extract right sign */
        s0 = XT_OLT_SX2(z0, XT_CONST_S(0));
        s1 = XT_OLT_SX2(z1, XT_CONST_S(0));
        ABS_SX2X2(z0, z1, z0, z1);
        AE_LSX2X2_IP(x0, x1, S_rd, +4 * sz_f32);
        /* simplified ldexpf */

        e0_0 = AE_SRAI32(n0, 1);
        e0_1 = AE_SRAI32(n1, 1);
        e1_0 = AE_SUB32(n0, e0_0);
        e1_1 = AE_SUB32(n1, e0_1);

        s0_0=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(e0_0));
        s0_1=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(e0_1));
        s1_0=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(e1_0));
        s1_1=FLOATEXP_SX2(AE_MOVINT8X8_FROMINT32X2(e1_1));
        MUL_SX2X2(y0, y1, z0, z1, s0_0,s0_1);
        MUL_SX2X2(z0, z1, y0, y1, s1_0,s1_1);
        /* approx y=1/(1+x); */
        ADD_SX2X2(x0,x1,XT_CONST_S(1),XT_CONST_S(1),x0,x1);
        {
            xtfloatx2 t20,t21;
            y0 =XT_RECIP0_SX2(x0);
            y1 =XT_RECIP0_SX2(x1);
            CONST_SX2X2(t20,t21,1);
            MSUB_SX2X2(t20,t21,x0,x1,y0,y1);
            MADD_SX2X2(y0,y1,y0,y1,t20,t21);
            CONST_SX2X2(t20,t21,1);
            MSUB_SX2X2(t20,t21,x0,x1,y0,y1);
            MADD_SX2X2(y0,y1,y0,y1,t20,t21);
       }
        XT_MOVF_SX2(z0, XT_CONST_S(1), s0);
        XT_MOVF_SX2(z1, XT_CONST_S(1), s1);
        MUL_SX2X2(y0, y1, y0, y1, z0, z1);
        AE_SASX2X2_IP(y0, y1, Y_va, Y);
    }
    AE_SA128POS_FP(Y_va, Y);
  }
} /* __sigmoidf() */

void vec_sigmoidf    (float32_t * restrict y, const float32_t * restrict x, int N)
{
    xtfloatx4 * restrict pX;
    xtfloatx4 * restrict pY;
    int n;
    /* Block size, blkLen <= blkSize */
    const int blkSize = MAX_ALLOCA_SZ/2*2*sz_f32;
    /* Allocate a fixed-size scratch area on the stack. */
    float32_t ALIGN(32) scr[blkSize * 2];

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
        __sigmoidf((float32_t*)pY,(float32_t*)pX,8,scr);
        for (n=0; n<(N&7); n++) 
        {
            xtfloat t;
            XT_LSIP(t,castxcc(xtfloat,pY),sizeof(float32_t));
            XT_SSIP(t,castxcc(xtfloat,y ),sizeof(float32_t));
        }
        N&=~7;
    }
    if (N<=0) return;
    __sigmoidf(y,x,N,scr);
} /* vec_sigmoidf() */
#else
// code for scalar FPU
void vec_sigmoidf    (float32_t * y, const float32_t * x, int N)
{
  static const union ufloat32uint32 ALIGN(32) c[]={{0x3fb8aa3b},{0x32a57060}}; 
  static const union ufloat32uint32 ALIGN(32) p[]={{0x39222a75},{0x3aaf9334},{0x3c1d94fc},{0x3d63578b},{0x3e75fdf0},{0x3f317218},{0x3f800000}};
    const xtfloat * restrict pX=(const xtfloat *)x;
          xtfloat * restrict pY=(      xtfloat *)y;
    int n;
    for (n = 0; n < N; n++)
    {
        xtbool s;
        int32_t n,n0,n1;
        xtfloat x,s0,s1;
        xtfloat x0,y,z,d,t;
        XT_LSIP(x,pX,sizeof(float32_t));
        s=XT_OLT_S(x,0.f);
        x=XT_NEG_S(XT_ABS_S(x));
        XT_MOVT_S(x,-103.9721f,XT_OLT_S(x,-103.9721f));
        /* compute d+n=log2(e)*x */
        #if defined(XT_FIROUND_S)
            y=XT_FIROUND_S(XT_MUL_S(x,c[0].f));
        #else
            y=XT_FLOAT_S(XT_ROUND_S(XT_MUL_S(x,c[0].f),0),0);
        #endif
        d=XT_NEG_S(y);
        XT_MADDN_S(d,x,c[0].f);
        XT_MADDN_S(d,x,c[1].f);
        n=XT_TRUNC_S(y,0);
        /* approx 2^d */
        {
            xtfloat d2,z0,z1;
            d2=XT_MUL_S(d,d);
            z0=p[0].f;
            t =p[2].f; XT_MADDN_S(t,d2,z0); z0=t;
            t =p[4].f; XT_MADDN_S(t,d2,z0); z0=t;
            z1=p[1].f; 
            t =p[3].f; XT_MADDN_S(t,d2,z1); z1=t;
            t =p[5].f; XT_MADDN_S(t,d2,z1); z1=t;
            XT_MADDN_S(z1,z0,d);
            z=z1;
        }
        t=XT_CONST_S(1); XT_MADDN_S(t,d,z); z=t;
        /* compute approx x0 - it does not give right values on denorm values but it is ok for further computing 1/(1+x) */
        s0=XT_WFR((XT_MAX((n+127),0)<<23));
        x0=z;
        x0=XT_MUL_S(x0,s0);
        /* simplified ldexpf */
        n0=(n>>1);
        n1=(n-n0);
        n1=(n1+127);
        n0=(n0+127);
        n1=(n1<<23);
        n0=(n0<<23);
        s0=XT_WFR(n0);
        s1=XT_WFR(n1);
        x=XT_MUL_S(XT_MUL_S(z,s0),s1);
        /* approx y=1/(1+x); */
        y=XT_RECIP_S(XT_ADD_S(XT_CONST_S(1),x0));
        t=XT_MUL_S(y,x);
        XT_MOVT_S(y,t,s);
        XT_SSIP(y,pY,sizeof(float32_t));
   }
} /* vec_sigmoidf() */
#endif
