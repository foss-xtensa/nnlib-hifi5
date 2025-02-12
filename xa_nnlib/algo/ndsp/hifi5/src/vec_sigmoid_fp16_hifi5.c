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
/* Copyright (c) 2020 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ('Cadence    */
/* Libraries') are the copyrighted works of Cadence Design Systems Inc.     */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
/* ------------------------------------------------------------------------ */
/*  IntegrIT, Ltd.   www.integrIT.com, info@integrIT.com                    */
/*                                                                          */
/* NatureDSP_Baseband Library                                               */
/*                                                                          */
/* This library contains copyrighted materials, trade secrets and other     */
/* proprietary information of IntegrIT, Ltd. This software is licensed for  */
/* use with Cadence processor cores only and must not be used for any other */
/* processors and platforms. The license to use these sources was given to  */
/* Cadence, Inc. under Terms and Condition of a Software License Agreement  */
/* between Cadence, Inc. and IntegrIT, Ltd.                                 */
/* ------------------------------------------------------------------------ */
/*          Copyright (C) 2009-2020 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
/*
  NatureDSP Signal Processing Library. Vector matematics
    Sigmoid
    Code optimized for HiFi5 core
  IntegrIT, 2006-2019
*/
#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"

#define sz_f16    (int)sizeof(float16_t)

/*-------------------------------------------------------------------------
  Sigmoid
  The functions compute the sigmoid of input argument. 32-bit fixed-point 
  functions accept inputs in Q6.25 and form outputs in Q16.15 format.

  Precision:
  32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB.
  f      single precision floating-point. Accuracy 2 ULP
  fp16   half precision floating-point. Accuracy 2 ULP
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
#if !HAVE_HP_VFPU
DISCARD_FUN(void,xa_nnlib_vec_sigmoid_fp16,(float16_t * y, const float16_t * x, int N))
#else

#define sz_f16    (int)sizeof(float16_t)

static void __sigmoid_fp16(float16_t * y, const float16_t * x, int N, float16_t* scr)
{
  /*
  log2e=(log2(exp(1)));
  log2e0=double(it_half(log2e));
  log2e1=log2e-log2e0;
  log2e0=it_half(log2e0);
  log2e1=it_half(log2e1);
  */
  static const union ufloat16uint16 ALIGN(32) cnt[] = { { 0xcc29 }, /* -16.6355==-log(1/double(eps(it_half(0)))-1) */
                                                      { 0x3dc5 },   /* 1.4424 */
                                                      { 0x0d1e },   /* 0.00031233 */
                                                      { 0 } }; 

  /* polynonial coefficients for 2^x, x=-0.5...0.5 */
  static const union ufloat16uint16 ALIGN(32) p[] = { { 0x2b27 }, { 0x33c1 }, { 0x398c }, { 0x3c00 } };
  
  const xthalfx8 * restrict X;
        xthalfx8 * restrict Y;
  const xthalfx8 * restrict S_rd;
        xthalfx8 * restrict S_wr;
  const ae_int16 * restrict pP = (const ae_int16  *)p;
  const ae_int16 * restrict pC = (const ae_int16  *)cnt;

  /* Current block index; overall number of blocks; number of values in the current block */
  int M;
  ae_valignx2 X_va, Y_va;
  xthalfx4 x0, x1;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ/2*2*sz_f16;
  int n;
  NASSERT_ALIGN16(scr);
  NASSERT(N%16==0);

  for (;N>0; x+=M, y+=M, N-=M)
  {
    xthalfx4 minsigmoid_fp16 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pC, 0));
    //xthalfx4 log2e0 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pC, 2));
    //xthalfx4 log2e1 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pC, 4));

    xthalfx4 c0 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pP, 0));
    xthalfx4 c1 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pP, 2));
    xthalfx4 c2 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pP, 4));

    M= XT_MIN( N, blkSize );

    X = (xthalfx8*)(x);
    S_wr = (xthalfx8*)scr;

    X_va = AE_LA128_PP(X);
    /* argumant reduction phase */
    __Pragma("loop_count factor=2")
    for ( n=0; n<(M>>3); n++ )
    { 
      xthalfx4 p0, p1, y0, y1;
      xthalfx4 d0, d1;
      ae_int16x4 n0, n1;
      AE_LAHX4X2_IP(x0, x1, X_va, X);
      ABS_HX4X2(p0, p1, x0, x1);
      NEG_HX4X2(x0, x1, p0, p1);
      x0 = MAX_HX4(minsigmoid_fp16, x0);
      x1 = MAX_HX4(minsigmoid_fp16, x1);
      /* compute d+n=log2(e)*x */
      MULQ_H(p0, p1, x0, x1, AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pC, 2)));
      y0 = FIROUND_HX4(p0);
      y1 = FIROUND_HX4(p1);
      
      NEG_HX4X2(d0, d1, y0, y1);      
      MADDQ_H(d0, d1, x0, x1, AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pC, 2)));
      MADDQ_H(d0, d1, x0, x1, AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I(pC, 4)));
 
      n0 = TRUNC16_HX4(y0, 0);
      n1 = TRUNC16_HX4(y1, 0);
      AE_S16X4X2_IP(n0, n1, castxcc(ae_int16x8, S_wr), 8 * sz_f16);
      AE_SHX4X2_IP(d0, d1, S_wr, 8 * sz_f16);
    }
    /* second phase: compute polynomial approximation */
    __Pragma("no_reorder")
    S_wr = (xthalfx8*)scr;
    S_rd = (xthalfx8*)scr;
    X = (xthalfx8*)(x);
    Y = (xthalfx8*)y;

    S_wr = ((xthalfx8*)scr) + 1;
    Y_va = AE_ZALIGN128();
    X_va = AE_LA128_PP(X);
    __Pragma("loop_count factor=2")
    for (n = 0; n<(M >> 3); n++)
    {
      xtbool4 s0, s1;
      ae_int16x4 n0, n1;
      xthalfx4 d0, d1, z0, z1, y0, y1;
      AE_LAHX4X2_IP(x0, x1, X_va, X);
      AE_L16X4X2_IP(n0, n1, castxcc(ae_int16x8, S_rd), 8 * sz_f16);
      AE_LHX4X2_IP(d0, d1, S_rd, 8 * sz_f16);
      y0 = CONST_HX4(0);
      s0 = OLT_HX4(x0, y0);
      s1 = OLT_HX4(x1, y0);
      /* approx 2^d */
      y0 = y1 = c1;
      MADDQ_H(y0, y1, d0, d1, c0);
      z0 = z1 = c2;
      MADD_HX4X2(z0, z1, d0, d1, y0, y1); y0 = z0; y1 = z1;
      CONST_HX4X2(z0, z1, 1);
      MADD_HX4X2(z0, z1, d0, d1, y0, y1);
      /* save orignal sign as a sign of polynomial */
      NEG_HX4X2(y0, y1, z0, z1);
      MOVT_HX4(z0, y0, s0);
      MOVT_HX4(z1, y1, s1);
      AE_SAHX4X2_IP(z0, z1, Y_va, Y);
    }
    AE_SA128POS_FP(Y_va, Y);
    // last phase: scale polynomial to 2^n and compute 1/(1+x)
    __Pragma("no_reorder")
    S_wr = (xthalfx8*)scr;
    S_rd = (const xthalfx8*)scr;
    X    = (xthalfx8*)y;
    Y    = (xthalfx8*)y;
    Y_va = AE_ZALIGN128();
    X_va = AE_LA128_PP(X);
    __Pragma("loop_count factor=2")
    for (n = 0; n<(M >> 3); n++)
    {
     xthalfx4 s0_0, s0_1, s1_0, s1_1;
     xtbool4 s0, s1;
     ae_int16x4 n0, n1;
     ae_int16x4 e0_0, e0_1, e1_0, e1_1;
     xthalfx4 x0, x1, z0, z1;
     xthalfx4 y0, y1;
     AE_LAHX4X2_IP(z0, z1, X_va, X);
     AE_L16X4X2_IP(n0, n1, castxcc(ae_int16x8, S_rd), 2*8 * sz_f16);
     /* extract right sign */
     s0 = OLT_HX4(z0, CONST_HX4(0));
     s1 = OLT_HX4(z1, CONST_HX4(0));
     ABS_HX4X2(z0, z1, z0, z1);
     /* simplified ldexpf */
     e0_0 = AE_SRAI16(n0, 1);
     e0_1 = AE_SRAI16(n1, 1);
     e1_0 = AE_SUB16(n0, e0_0);
     e1_1 = AE_SUB16(n1, e0_1);
     n0 = AE_SLLI16S(AE_ADD16(e0_0, 15), 10);
     n1 = AE_SLLI16S(AE_ADD16(e1_0, 15), 10);
     s0_0 = AE_MOVXTHALFX4_FROMINT16X4(n0);
     s1_0 = AE_MOVXTHALFX4_FROMINT16X4(n1);
     n0 = AE_SLLI16S(AE_ADD16(e0_1, 15), 10);
     n1 = AE_SLLI16S(AE_ADD16(e1_1, 15), 10);
     s0_1 = AE_MOVXTHALFX4_FROMINT16X4(n0);
     s1_1 = AE_MOVXTHALFX4_FROMINT16X4(n1);
     
     MUL_HX4X2(y0, y1, z0, z1, s0_0, s0_1);
     MUL_HX4X2(z0, z1, y0, y1, s1_0, s1_1);
     /* approx y=1/(1+x); */
     ADD_HX4X2(x0, x1, CONST_HX4(1), CONST_HX4(1), z0, z1);
     {
       xthalfx4 t20, t21;
       y0 = RECIP0_HX4(x0);
       y1 = RECIP0_HX4(x1);
       CONST_HX4X2(t20, t21, 1);
       SUB_HX4X2(t20, t21, t20, t21, y0, y1);
       MSUB_HX4X2(t20, t21, z0, z1, y0, y1);
       MADD_HX4X2(y0, y1, y0, y1, t20, t21);
     }
     MOVF_HX4(z0, CONST_HX4(1), s0);
     MOVF_HX4(z1, CONST_HX4(1), s1);
     MUL_HX4X2(y0, y1, y0, y1, z0, z1);
     AE_SAHX4X2_IP(y0, y1, Y_va, Y);
    }
    AE_SA128POS_FP(Y_va, Y);

  }
} /* __sigmoid_fp16() */

void xa_nnlib_vec_sigmoid_fp16    (float16_t * restrict y, const float16_t * restrict x, int N)
{
#if 1
  xthalfx8  * restrict pX;
  xthalfx8  * restrict pY;
  int n;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ/2*2*sz_f16;
  /* Allocate a fixed-size scratch area on the stack. */
  float16_t ALIGN(32) scr[blkSize];

  if (N<=0) return;
  if (N & 15)
  {
    float16_t ALIGN(32) xbuf[16], ybuf[16];
    pX = (xthalfx8 *)xbuf;
    pY = (xthalfx8 *)ybuf;
    
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pX, 0 * sizeof(xthalfx8));
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pX, 1 * sizeof(xthalfx8));
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pY, 0 * sizeof(xthalfx8));
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pY, 1 * sizeof(xthalfx8));
    for (n=0; n<(N&15); n++) 
    {
      xthalf t;
      
      AE_LHIP(t, castxcc(xthalf, x), sizeof(float16_t));
      AE_SHIP(t, castxcc(xthalf, pX), sizeof(float16_t));
    }
    pX = (xthalfx8 *)xbuf;
    __sigmoid_fp16((float16_t*)pY,(float16_t*)pX,16,scr);
    for (n=0; n<(N&15); n++) 
    {
      xthalf t;
      AE_LHIP(t, castxcc(xthalf, pY), sizeof(float16_t));
      AE_SHIP(t, castxcc(xthalf, y), sizeof(float16_t));
    }
    N&=~15;
  }
  if (N<=0) return;
  __sigmoid_fp16(y,x,N,scr);
#else

  float16_t x_, y_;
  if (N <= 0) return;
  static const union ufloat16uint16
    log2e0 = { 0x3dc5 }, //1.4424
    log2e1 = { 0x0d1e }; //0.00031233 
  /* polynonial coefficients for 2^x, x=-0.5...0.5 */
  static const union ufloat16uint16
    pow2_fp16_coef[] = { { 0x2b27 }, { 0x33c1 }, { 0x398c }, { 0x3c00 } };
  static const union ufloat16uint16
    minsigmoid_fp16 = { 0xcc29 }, // -16.6355==-log(1/double(eps(it_half(0)))-1)
    one = { 0x3c00 },
    half = { 0x3800 };
  union ufloat16uint16 s0, s1;

  int i, s, n, n0, n1;
  float16_t z, d;
  for (i = 0; i<N; i++)
  {
    x_ = x[i];
    s = lt_f16(x_, 0);
    x_ = neg_f16(fabs_f16(x_));
    if (lt_f16(x_, minsigmoid_fp16.f)) x_ = minsigmoid_fp16.f;
    /* compute d+n=log2(e)*x */
    y_ = round_f16(mul_f16(x_, log2e0.f));
    d = fma_f16(neg_f16(x_), log2e0.f, y_);
    d = fma_f16(neg_f16(x_), log2e1.f, d);
    n = itrunc_f16(y_);
    /* approx 2^d */
    z = pow2_fp16_coef[0].f;
    z = fma_f16(neg_f16(d), z, pow2_fp16_coef[1].f);
    z = fma_f16(neg_f16(d), z, pow2_fp16_coef[2].f);
    z = fma_f16(neg_f16(d), z, pow2_fp16_coef[3].f);

    /* simplified ldexpf */
    n0 = n >> 1;
    n1 = n - n0;
    s1.u = ((uint16_t)(n1 + 15) << 10);
    s0.u = ((uint16_t)(n0 + 15) << 10);
    x_ = mul_f16(mul_f16(z, s0.f), s1.f);
    /* approx y=1/(1+x); */
    y_ = fma_f16(neg_f16(x_), half.f, one.f);
    d = fma_f16(neg_f16(x_), y_, sub_f16(one.f, y_)); y_ = fma_f16(y_, d, y_);
    d = fma_f16(neg_f16(x_), y_, sub_f16(one.f, y_)); y_ = fma_f16(y_, d, y_);
    if (s) y_ = mul_f16(y_, x_);
    y[i] = y_;
  }
#endif
} /* xa_nnlib_vec_sigmoid_fp16() */
#endif
