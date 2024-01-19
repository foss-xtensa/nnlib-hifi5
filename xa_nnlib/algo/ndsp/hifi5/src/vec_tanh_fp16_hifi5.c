/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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
    Hyperbolic Tangent
    Code optimized for HiFi5 core
    IntegrIT, 2006-2020
*/
#include "../include/NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common_fpu.h"
/* Tables and constants. */
#include "../include/tanh_fp16_tbl.h"
 
/*-------------------------------------------------------------------------
  Hyperbolic Tangent
  The functions compute the hyperbolic tangent of input argument. 32-bit
  fixed-point functions accept inputs in Q6.25 and form outputs in Q16.15
  format.

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
DISCARD_FUN(void,xa_nnlib_vec_tanh_fp16,(float16_t* y, const float16_t* x, int N))
#else
#define sz_f16    (int)sizeof(float16_t)

static void __tanh_fp16(float16_t * y, const float16_t * x, int N, float16_t* scr)
{
  const xthalf * restrict pT = (const xthalf  *)xa_nnlib_tanh_fp16_tbl;
  const xthalfx8 * restrict pX;
        xthalfx8 * restrict pY;
  const xthalfx8 * restrict S_rd;
        xthalfx8 * restrict S_wr;
  /* Current block index; overall number of blocks; number of values in the current block */
  ae_valignx2 X_va, Y_va;
  /* Block size, blkLen <= blkSize */
  xthalfx4 halfln3_fp16 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 0 * 2));
  xthalfx4 tanhf16_maxval = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 1 * 2));
  ae_int16x4 _23637 = AE_L16_X((ae_int16 *)pT, 8 * 2);
  ae_int16x4 _7 = AE_L16_X((ae_int16 *)pT, 9 * 2);
  /* Current block index; overall number of blocks; number of values in the current block */
  int n, M;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / 2 * 2 * sz_f16;
  NASSERT_ALIGN16(scr);
  NASSERT(N % 16 == 0);

  for (; N>0; x += M, y += M, N -= M)
  {
    M = XT_MIN(N, blkSize);
    pX   = (const xthalfx8*)(x);
    S_wr = (      xthalfx8*)(scr);
    X_va = AE_LA128_PP(pX);
   
    for (n = 0; n<(M >> 3); n++)
    {
      xthalfx4 x0, x1, z0, z1;
      ae_int16x4 a0, a1, n0, n1, a2, a3;
      ae_int16x4 y0, y1, y2, y3, t0, t1;
      ae_int32x2 d0, d1, d2, d3;
      AE_LAHX4X2_IP(x0, x1, X_va, pX);
      ABS_HX4X2(x0, x1, x0, x1);
      /* exact formula for x>0.54 */
      x0 = MINNUM_HX4(x0, tanhf16_maxval);
      x1 = MINNUM_HX4(x1, tanhf16_maxval);
      a0 = TRUNC16_HX4(x0, 12);
      a1 = TRUNC16_HX4(x1, 12);
      /* multiply by 1/ln(2) and convert to Q15 */
      AE_MUL16X4(d0, d1, a0, _23637);
      AE_MUL16X4(d2, d3, a1, _23637);
      d0 = AE_SRAI32R(d0, 10);
      d1 = AE_SRAI32R(d1, 10);
      d2 = AE_SRAI32R(d2, 10);
      d3 = AE_SRAI32R(d3, 10);
      /* exponential part  */
      n0 = AE_TRUNCI16X4F32S(d0, d1, 1);
      n1 = AE_TRUNCI16X4F32S(d2, d3, 1);
      d0 = AE_AND32(d0, 0x7fff);
      d1 = AE_AND32(d1, 0x7fff);
      d2 = AE_AND32(d2, 0x7fff);
      d3 = AE_AND32(d3, 0x7fff);
      /* mantissa          */
      a0 = AE_TRUNCI16X4F32S(d0, d1, 16);
      a1 = AE_TRUNCI16X4F32S(d2, d3, 16);
      /* compute 2^a, 0..1 in Q14 */
      a2 = AE_MULFP16X4S(a0, a0);
      a3 = AE_MULFP16X4S(a1, a1);
      y1 = AE_L16_I((ae_int16 *)pT, 4 * 2);
      y2 = AE_L16_I((ae_int16 *)pT, 5 * 2);
      y3 = AE_L16_I((ae_int16 *)pT, 6 * 2);
      t0 = AE_MULFP16X4S(a2, y1); t1 = AE_MULFP16X4S(a3, y1);
      y0 = AE_ADD16S(y3, t0); y1 = AE_ADD16S(y3, t1);

      y3 = AE_L16_I((ae_int16 *)pT, 7 * 2);
      t0 = AE_MULFP16X4S(a2, y2); t1 = AE_MULFP16X4S(a3, y2);
      y2 = AE_ADD16S(y3, t0); y3 = AE_ADD16S(y3, t1);

      t0 = AE_MULFP16X4S(a0, y0); t1 = AE_MULFP16X4S(a1, y1);
      y0 = AE_ADD16S(y2, t0);
      y1 = AE_ADD16S(y3, t1);

      z0 = FLOAT16_HX4(y0, 7);
      z1 = FLOAT16_HX4(y1, 7);
      n0 = AE_SLLI16S(AE_ADD16S(n0, _7), 10);
      n1 = AE_SLLI16S(AE_ADD16S(n1, _7), 10);
      MUL_HX4X2(z0, z1, z0, z1, AE_MOVXTHALFX4_FROMINT16X4(n0), AE_MOVXTHALFX4_FROMINT16X4(n1));
      /* convert exp(2x)/2 to tanh */
      ADD_HX4X2(z0, z1, z0, z1, CONST_HX4(3), CONST_HX4(3));
      z0 = RECIP_HX4(z0); z1 = RECIP_HX4(z1);
      SUB_HX4X2(z0, z1, CONST_HX4(1), CONST_HX4(1), z0, z1);
      AE_SHX4X2_IP(z0, z1, S_wr, 8 * sz_f16);
    }
    
    __Pragma("no_reorder");

    pX   = (const xthalfx8*)(x);
    S_rd = (const xthalfx8*)(scr);
    pY   = (      xthalfx8*)(y);
    X_va = AE_LA128_PP(pX);
    Y_va = AE_ZALIGN128();
    for (n = 0; n<(M >> 3); n++)
    {
      xthalfx4 x0, x1, y0, y1, z0, z1;
      xthalfx4 x2, x3, t0, t1;
      xtbool4 b0, b1;
      ae_int16x4 a0, a1;
      AE_LAHX4X2_IP(x0, x1, X_va, pX);
      AE_LHX4X2_IP(y0, y1, S_rd, 8 * sz_f16);
      a0 = AE_MOVINT16X4_FROMXTHALFX4(x0);
      a1 = AE_MOVINT16X4_FROMXTHALFX4(x1);
      ABS_HX4X2(x0, x1, x0, x1);

      b0 = ULT_HX4(x0, halfln3_fp16);
      b1 = ULT_HX4(x1, halfln3_fp16);
      /* polynomial for small x */
      MUL_HX4X2(x2, x3, x0, x1, x0, x1);
      z1 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 2 * 2));
      t0 = t1 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 3 * 2));
      MADD_HX4X2(t0, t1, x2, x3, z1, z1); z0 = t0; z1 = t1;
      MUL_HX4X2(z0, z1, x2, x3, z0, z1);
      t0 = x0; t1 = x1; MADD_HX4X2(t0, t1, z0, z1, x0, x1); z0 = t0; z1 = t1;
      /* select variant and apply sign */
      MOVT_HX4(y0, z0, b0);
      MOVT_HX4(y1, z1, b1);

      MOVT_HX4(y0, NEG_HX4(y0), AE_LT16(a0, AE_ZERO16()));
      MOVT_HX4(y1, NEG_HX4(y1), AE_LT16(a1, AE_ZERO16()));

      AE_SAHX4X2_IP(y0, y1, Y_va, pY);
    }
    AE_SA128POS_FP(Y_va, pY);
  }
} /* __tanh_fp16() */

void xa_nnlib_vec_tanh_fp16(float16_t* restrict y, const float16_t* restrict x, int N)
{
  #if 0
  const xthalf * restrict pT = (const xthalf  *)xa_nnlib_tanh_fp16_tbl;
  const xthalfx4 * restrict pX;
  const xthalfx4 * restrict pX_;
        xthalfx4 * restrict pY;
  /* Current block index; overall number of blocks; number of values in the current block */
  ae_valignx2 X_va, X__va, Y_va;
  /* Block size, blkLen <= blkSize */
  xthalfx4 one = CONST_HX4(1);
  xthalfx4 two = CONST_HX4(2);
  xthalfx4 half = CONST_HX4(3);
  xthalfx4 halfln3_fp16 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 0 * 2));
  xthalfx4 tanhf16_maxval = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 1 * 2));
  ae_int16x4 _23637 = AE_L16_X((ae_int16 *)pT, 8 * 2);
  ae_int16x4 _7 = AE_L16_X((ae_int16 *)pT, 9 * 2);

  /* Allocate a fixed-size scratch area on the stack. */
  int n,M;
  if (N <= 0) return;
  pX = (const xthalfx4*)(x);
  pY = (      xthalfx4*)(y);
  X_va = AE_LA128_PP(pX);
  Y_va = AE_ZALIGN128();

  for (n = 0; n<(N >> 3); n++)
  {
    xthalfx4 x0, x1, z0, z1;
    ae_int16x4 a0, a1, n0, n1, a2, a3;
    ae_int16x4 y0, y1, y2, y3, t0, t1, c1;
    ae_int32x2 d0, d1, d2, d3;
    AE_LAHX4X2_IP(x0, x1, X_va, pX);
    ABS_HX4X2(x0, x1, x0, x1);
    /* exact formula for x>0.54 */
    x0 = MINNUM_HX4(x0, tanhf16_maxval);
    x1 = MINNUM_HX4(x1, tanhf16_maxval);   
    a0 = TRUNC16_HX4(x0, 12);
    a1 = TRUNC16_HX4(x1, 12);
    /* multiply by 1/ln(2) and convert to Q15 */
    AE_MUL16X4(d0, d1, a0, _23637);
    AE_MUL16X4(d2, d3, a1, _23637);
    d0 = AE_SRAI32R(d0, 10);
    d1 = AE_SRAI32R(d1, 10);
    d2 = AE_SRAI32R(d2, 10);
    d3 = AE_SRAI32R(d3, 10);
    /* exponential part  */
    n0 = AE_TRUNCI16X4F32S(d0, d1, 1);
    n1 = AE_TRUNCI16X4F32S(d2, d3, 1);  
    d0 = AE_AND32(d0, 0x7fff);
    d1 = AE_AND32(d1, 0x7fff);
    d2 = AE_AND32(d2, 0x7fff);
    d3 = AE_AND32(d3, 0x7fff);
    /* mantissa          */
    a0 = AE_TRUNCI16X4F32S(d0, d1, 16);
    a1 = AE_TRUNCI16X4F32S(d2, d3, 16);
    /* compute 2^a, 0..1 in Q14 */
    a2 = AE_MULFP16X4S(a0, a0);
    a3 = AE_MULFP16X4S(a1, a1);
    y1 = AE_L16_I((ae_int16 *)pT, 4 * 2);
    y2 = AE_L16_I((ae_int16 *)pT, 5 * 2);
    y3 = AE_L16_I((ae_int16 *)pT, 6 * 2);
    t0 = AE_MULFP16X4S(a2, y1); t1 = AE_MULFP16X4S(a3, y1);
    y0 = AE_ADD16S(y3, t0); y1 = AE_ADD16S(y3, t1);

    y3 = AE_L16_I((ae_int16 *)pT, 7 * 2);
    t0 = AE_MULFP16X4S(a2, y2); t1 = AE_MULFP16X4S(a3, y2);
    y2 = AE_ADD16S(y3, t0); y3 = AE_ADD16S(y3, t1);
    
    t0 = AE_MULFP16X4S(a0, y0); t1 = AE_MULFP16X4S(a1, y1);  
    y0 = AE_ADD16S(y2, t0);
    y1 = AE_ADD16S(y3, t1);

    z0 = FLOAT16_HX4(y0, 7);
    z1 = FLOAT16_HX4(y1, 7);
    n0 = AE_SLLI16S(AE_ADD16S(n0, _7), 10);
    n1 = AE_SLLI16S(AE_ADD16S(n1, _7), 10);
    MUL_HX4X2(z0, z1, z0, z1, AE_MOVXTHALFX4_FROMINT16X4(n0), AE_MOVXTHALFX4_FROMINT16X4(n1));
    /* convert exp(2x)/2 to tanh */
    ADD_HX4X2(z0, z1, z0, z1, CONST_HX4(3), CONST_HX4(3));
    z0 = RECIP_HX4(z0); z1 = RECIP_HX4(z1);
    SUB_HX4X2(z0, z1, CONST_HX4(1), CONST_HX4(1), z0, z1);
    AE_SAHX4X2_IP(z0, z1, Y_va, pY);
  }
  AE_SA128POS_FP(Y_va, pY);

  __Pragma("no_reorder");
  
  pX  = (const xthalfx4*)(x);
  pX_ = (const xthalfx4*)(y);
  pY = (xthalfx4*)(y);
  X_va = AE_LA128_PP(pX);
  X__va = AE_LA128_PP(pX_);
  Y_va = AE_ZALIGN128();
  for (n = 0; n<(N >> 3); n++)
  {
    xthalfx4 x0, x1, y0, y1, z0, z1;
    xthalfx4 x2, x3, t0, t1;
    xtbool4 b0, b1;
    ae_int16x4 a0, a1;
    AE_LAHX4X2_IP(x0, x1, X_va, pX);
    AE_LAHX4X2_IP(y0, y1, X__va, pX_);
    a0 = AE_MOVINT16X4_FROMXTHALFX4(x0);
    a1 = AE_MOVINT16X4_FROMXTHALFX4(x1);
    ABS_HX4X2(x0, x1, x0, x1);

    b0 = ULT_HX4(x0, halfln3_fp16);
    b1 = ULT_HX4(x1, halfln3_fp16);
    /* polynomial for small x */
    MUL_HX4X2(x2, x3, x0, x1, x0, x1);
    z1 = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 2 * 2));
    t0 = t1 =  AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((ae_int16 *)pT, 3 * 2));
    MADD_HX4X2(t0, t1, x2, x3, z1, z1); z0 = t0; z1 = t1;
    MUL_HX4X2(z0, z1, x2, x3, z0, z1);
    t0 = x0; t1 = x1; MADD_HX4X2(t0, t1, z0, z1, x0, x1); z0 = t0; z1 = t1;
    /* select variant and apply sign */
    MOVT_HX4(z0, y0, b0);
    MOVT_HX4(z1, y1, b1);
    
    MOVT_HX4(z0, NEG_HX4(z0), AE_LT16(a0, AE_ZERO16()));
    MOVT_HX4(z1, NEG_HX4(z1), AE_LT16(a1, AE_ZERO16()));
 
    AE_SAHX4X2_IP(z0, z1, Y_va, pY);
  }
  AE_SA128POS_FP(Y_va, pY);


  #else
  xthalfx8  * restrict pX;
  xthalfx8  * restrict pY;
  int n;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / 2 * 2 * sz_f16;
  /* Allocate a fixed-size scratch area on the stack. */
  float16_t ALIGN(32) scr[blkSize];

  if (N <= 0) return;
  if (N & 15)
  {
    float16_t ALIGN(32) xbuf[16], ybuf[16];
    pX = (xthalfx8 *)xbuf;
    pY = (xthalfx8 *)ybuf;

    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pX, 0 * sizeof(xthalfx8));
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pX, 1 * sizeof(xthalfx8));
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pY, 0 * sizeof(xthalfx8));
    AE_SHX4X2_I(CONST_HX4(0), CONST_HX4(0), pY, 1 * sizeof(xthalfx8));
    for (n = 0; n<(N & 15); n++)
    {
      xthalf t;

      AE_LHIP(t, castxcc(xthalf, x), sizeof(float16_t));
      AE_SHIP(t, castxcc(xthalf, pX), sizeof(float16_t));
    }
    pX = (xthalfx8 *)xbuf;
    __tanh_fp16((float16_t*)pY, (float16_t*)pX, 16, scr);
    for (n = 0; n<(N & 15); n++)
    {
      xthalf t;
      AE_LHIP(t, castxcc(xthalf, pY), sizeof(float16_t));
      AE_SHIP(t, castxcc(xthalf, y), sizeof(float16_t));
    }
    N &= ~15;
  }
  if (N <= 0) return;
  __tanh_fp16(y, x, N, scr);
  #endif
} /* xa_nnlib_vec_tanh_fp16() */

#endif
