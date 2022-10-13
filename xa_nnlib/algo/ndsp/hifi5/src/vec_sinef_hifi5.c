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
  NatureDSP Signal Processing Library. Vector Mathematics
  Sine
    Code optimized for HiFi5
  IntegrIT, 2006-2019
*/

#include "NatureDSP_Signal_math.h"
/* Common helper macros. */
#include "common_fpu.h"
/* Tables */
#include "inv2pif_tbl.h"
#include "sinf_tbl.h"
/* sNaN/qNaN, single precision. */
#include "nanf_tbl.h"

#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void,vec_sinef,( float32_t * restrict y, const float32_t * restrict x, int N ))
#elif HAVE_VFPU

#define sz_f32    (int)sizeof(float32_t)

#if SINNCOSF_ALG==0
static void mysinef(float32_t* scr,
    float32_t* restrict y,
    const float32_t* restrict x,
    int N)
{

    const xtfloatx4* restrict X;
    const xtfloatx4* restrict X1;
    xtfloatx4* restrict Z;
    const xtfloatx4* restrict S_rd;
    xtfloatx4* restrict S_wr;
    xtfloatx4* restrict S_firound_wr;
    //const xtfloatx4* restrict T;
    const xtfloat* psintbl = (const xtfloat*)polysinf_tbl;
    const xtfloat* pcostbl = (const xtfloat*)polycosf_tbl;
    /* Current block index; overall number of blocks; number of values in the current block */
    ae_valignx2 X_va, X1_va, Z_va;

    /* Block size, blkLen <= blkSize */
    const int blkSize = MAX_ALLOCA_SZ / (2 * sz_f32);
    /* 2/pi splited into 24-bit chunks*/
    xtfloatx2 pi2fc0, pi2fc1, pi2fc2;
    /* init 24bit chunks of 2/pi*/
    pi2fc0 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOV32(0x3fc90fdb));
    pi2fc1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOV32(0xb33bbd2e));
    pi2fc2 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOV32(0xa6f72ced));

    int n, M;
    NASSERT_ALIGN16(scr);
    NASSERT(N % 8 == 0);

    /*
    * Data are processed in blocks of scratch area size. Further, the algorithm
    * implementation is splitted in order to feed the optimizing compiler with a
    * few loops of managable size.
    */

    Z_va = AE_ZALIGN128();
    for (; N > 0; N -= M, x += M, y += M)
    {
        M = XT_MIN(N, blkSize);


        X = (xtfloatx4*)(x);
        S_wr = (xtfloatx4*)scr;
        S_firound_wr = (xtfloatx4*)scr + 1;
        X = (xtfloatx4*)(x);
        X_va = AE_LA128_PP(X);
        /*
        *   Part I.  Argument reduction.
        */
        for (n = 0; n < (M >> 2); n++)
        {
            xtfloatx2 p0, p1;
            xtfloatx2 jf0, jf1;

            AE_LASX2X2_IP(p0, p1, X_va, X);

            ABS_SX2X2(p0, p1, p0, p1);
            MULQ_S(jf0, jf1, p0, p1, inv2pif.f);
            jf0 = FIROUND_SX2(jf0);
            jf1 = FIROUND_SX2(jf1);
            AE_SSX2X2_IP(jf0, jf1, S_firound_wr, 8 * sz_f32);

            MSUBQ_S(p0, p1, jf0, jf1, pi2fc0);
            MSUBQ_S(p0, p1, jf0, jf1, pi2fc1);
            MSUBQ_S(p0, p1, jf0, jf1, pi2fc2);

            AE_SSX2X2_IP(p0, p1, S_wr, 8 * sz_f32);
        }

        X = (xtfloatx4*)(x);
        X1 = (xtfloatx4*)(x);
        Z = (xtfloatx4*)(y);
        S_rd = (xtfloatx4*)scr;
        X_va = AE_LA128_PP(X);
        X1_va = AE_LA128_PP(X1);
        //T = (xtfloatx4*)my_polysincosf_tbl;

        for (n = 0; n < (M >> 2); n++)
        {
            /* Input value; reducted input value and its 2nd power; auxiliary var */
            xtfloatx2 xn0, p0, p20, t0;
            xtfloatx2 xn1, p1, p21, t1;
            /* Input value segment number; input and output signs; integer reprentation of output value */
            ae_int32x2 ji0, sx0, sy0;
            ae_int32x2 ji1, sx1, sy1;
            /* Cosine and sine approximations; output value */
            xtfloatx2 yc0, ys0;
            xtfloatx2 yc1, ys1;
            /* Polynomial coefficients for sine and cosine. */
            xtfloatx2 cf_s0, cf_s1, cf_s2;
            xtfloatx2 cf_c0, cf_c1, cf_c2;
            /* Cosine/sine selection; out-of-domain flags */
            xtbool2  b_ndom0;
            xtbool2  b_ndom1;

            AE_LASX2X2_IP(xn0, xn1, X_va, X);
            /* Determine the pi/2-wide segment the input value belongs to. */
            ABS_SX2X2(xn0, xn1, xn0, xn1);
            AE_LSX2X2_I(t0, t1, S_rd, 4 * sz_f32);
            ji0 = XT_TRUNC_SX2(t0, 0);         ji1 = XT_TRUNC_SX2(t1, 0);
            /*
             * Compute polynomial approximations of sine and cosine for the
             * reducted input value.
             */
            AE_LSX2X2_IP(p0, p1, S_rd, 8 * sz_f32);

            MUL_SX2X2(p20, p21, p0, p1, p0, p1);
            { ae_int32x2 t; AE_L32_XP(t, castxcc(ae_int32, psintbl), 0); cf_s0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(t); }
            cf_s1 = XT_LSI(psintbl, 1 * sz_f32);
            cf_s2 = XT_LSI(psintbl, 2 * sz_f32);
            { ae_int32x2 t; AE_L32_XP(t, castxcc(ae_int32, pcostbl), 0); cf_c0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(t); }
            cf_c1 = XT_LSI(pcostbl, 1 * sz_f32);
            cf_c2 = XT_LSI(pcostbl, 2 * sz_f32);


            t0 = t1 = cf_s1; MADDQ_S(t0, t1, p20, p21, cf_s0); ys0 = t0; ys1 = t1;
            t0 = t1 = cf_s2; MADD_SX2X2(t0, t1, ys0, ys1, p20, p21); ys0 = t0; ys1 = t1;
            MUL_SX2X2(ys0, ys1, ys0, ys1, p20, p21);
            MADD_SX2X2(p0, p1, ys0, ys1, p0, p1); ys0 = p0; ys1 = p1;

            t0 = t1 = cf_c1; MADDQ_S(t0, t1, p20, p21, cf_c0); yc0 = t0; yc1 = t1;
            t0 = t1 = cf_c2; MADD_SX2X2(t0, t1, yc0, yc1, p20, p21); yc0 = t0; yc1 = t1;
            CONST_SX2X2(t0, t1, 1); MADD_SX2X2(t0, t1, yc0, yc1, p20, p21);

            /* Select sine or cosine. */
            XT_MOVT_SX2(ys0, t0, AE_MOVBD1X2(ji0));
            XT_MOVT_SX2(ys1, t1, AE_MOVBD1X2(ji1));

            b_ndom0 = XT_OLT_SX2(sinf_maxval.f, xn0);
            b_ndom1 = XT_OLT_SX2(sinf_maxval.f, xn1);
            /* Set result to NaN for an out-of-domain input value. */
            XT_MOVT_SX2(ys0, qNaNf.f, b_ndom0);
            XT_MOVT_SX2(ys1, qNaNf.f, b_ndom1);

            /* Adjust the sign. */
            AE_LASX2X2_IP(xn0, xn1, X1_va, X1);
            /* Determine the input sign. */
            sx0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(xn0);
            sx1 = XT_AE_MOVINT32X2_FROMXTFLOATX2(xn1);
            {
                sy0 = AE_SLLI32(ji0, 30);   sy1 = AE_SLLI32(ji1, 30);
                sy0 = AE_XOR32(sx0, sy0);   sy1 = AE_XOR32(sx1, sy1);
                sy0 = AE_XOR32(XT_AE_MOVINT32X2_FROMXTFLOATX2(ys0), AE_AND32(sy0, 0x80000000));
                sy1 = AE_XOR32(XT_AE_MOVINT32X2_FROMXTFLOATX2(ys1), AE_AND32(sy1, 0x80000000));
                ys0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(sy0);
                ys1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(sy1);
            }
            /* Set result to NaN for an out-of-domain input value. */
            AE_SASX2X2_IP(ys0, ys1, Z_va, Z);
        }
        AE_SA128POS_FP(Z_va, Z);
    }
}

#else 
static void mysinef(   float32_t * scr,
                float32_t * restrict y,
          const float32_t * restrict x,
          int N )
{
  /*
    float32_t x2,y,ys,yc;
    int sx,n,j,k,ss;
    sx=takesignf(x);
    x=sx?-x:x;
    if(x>sinf_maxval.f) return 0;
    argument reduction 
    k = (int)STDLIB_MATH(floorf)(x*inv4pif.f);
    n = k + 1;
    j = n&~1;

    {
      float32_t dx, t, y = x, jj = (float32_t)j;
      const union ufloat32uint32 c[6] = {
        { 0x3f4a0000 },
        { 0xbb700000 },
        { 0xb6160000 },
        { 0x32080000 },
        { 0x2e060000 },
        { 0xa9b9ee5a } };
      y -= c[0].f*jj;
      y -= c[1].f*jj;
      y -= c[2].f*jj;
      t = y; y -= c[3].f*jj; t = (t - y); t -= c[3].f*jj; dx = t;
      t = y; y -= c[4].f*jj; t = (t - y); t -= c[4].f*jj; dx = (dx + t);
      t = y; y -= c[5].f*jj; t = (t - y); t -= c[5].f*jj; dx = (dx + t);
      y = (y + dx);
      x = y;
    }
    adjust signs 
    ss = sx ^ (((n) >> 2) & 1);
      compute sine/cosine via minmax polynomial  
    x2 = x*x;
    ys = polysinf_tbl[0].f;
    ys = ys*x2 + polysinf_tbl[1].f;
    ys = ys*x2 + polysinf_tbl[2].f;
    ys = ys*x2;
    ys = ys*x + x;
    yc = polycosf_tbl[0].f;
    yc = yc*x2 + polycosf_tbl[1].f;
    yc = yc*x2 + polycosf_tbl[2].f;
    yc = yc*x2 + 1.f;
    select sine/cosine 
    y = (n & 2) ? yc : ys;
    apply the sign 
    y = changesignf(y, ss);
    return y;
  */
    /* pi/2 splitted into 7-bit chunks. */
  static const union ufloat32uint32 ALIGN(32) c[6] = {
    { 0x3fca0000 }, { 0xbbf00000 },
    { 0xb6960000 }, { 0x32880000 },
    { 0x2e860000 }, { 0xaa39ee5a }
    };

  const xtfloatx4 * restrict X;
  const xtfloatx4 * restrict X1;
        xtfloatx4 * restrict Y;
  const xtfloatx4 * restrict S_rd;
        xtfloatx4 * restrict S_wr;
  const xtfloat   * restrict T;
  const xtfloat * psintbl=(const xtfloat *)polysinf_tbl;
  const xtfloat * pcostbl=(const xtfloat *)polycosf_tbl;

  ae_valignx2 X_va,X1_va, Y_va;

  /* Current block index; overall number of blocks; number of values in the current block */
  int blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ/sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */

  int n;
  NASSERT(N%8==0);
  NASSERT_ALIGN16( scr );

  /*
   * Data are processed in blocks of scratch area size. Further, the algorithm
   * implementation is splitted in order to feed the optimizing compiler with a
   * few loops of managable size.
   */
    blkLen =0;
    for ( ; N>0; N-=blkLen, x+=blkSize,y+=blkSize )
    {
        blkLen = XT_MIN(N,blkSize);

    /*
     * Part I, range reduction. Reference C code:
     *
     *   {
     *     float32_t xn, p, dp, t;
     *     int ji;
     *     float32_t jf;
     *   
     *     // pi/2 splitted into 7-bit chunks.
     *     static const union ufloat32uint32 c[6] = {
     *       { 0x3fca0000 }, { 0xbbf00000 },
     *       { 0xb6960000 }, { 0x32880000 },
     *       { 0x2e860000 }, { 0xaa39ee5a }
     *     };
     *   
     *     for ( n=0; n<blkLen; n++ )
     *     {
     *       xn = fabsf( x[blkIx*blkSize+n] );
     *   
     *       // Determine the pi/2-wide segment the input value belongs to.
     *       jf = roundf( xn*inv2pif.f );
     *   
     *       // Calculate the difference between the segment midpoint and input value.
     *       p = xn;
     *       p -= c[0].f*jf;
     *       p -= c[1].f*jf;
     *       p -= c[2].f*jf;
     *       t = p; p -= c[3].f*jf; t = t - p; t -= c[3].f*jf; dp = t;
     *       t = p; p -= c[4].f*jf; t = t - p; t -= c[4].f*jf; dp += t;
     *       t = p; p -= c[5].f*jf; t = t - p; t -= c[5].f*jf; dp += t;
     *       p += dp;
     *   
     *       scr[n] = p;
     *     }
     *   }
     */

        X    = (xtfloatx4*)(x);
        S_wr = (xtfloatx4*)scr;
        T    = (xtfloat  *)c;

        X_va = AE_LA128_PP( X );
        __Pragma( "loop_count min=1" );
        for ( n=0; n<(blkLen>>3); n++ )
        {
            /* pi/2 splitted into 7-bit chunks. */
            xtfloatx2 c0, c1, c2, c3, c4, c5;
            /* Scalar auxiliary var.  */
            xtfloat cs;
            xtfloatx2 xn0,xn1,xn2,xn3;
            xtfloatx2 jf0,jf1;
            xtfloatx2 jf2,jf3;
            xtfloatx2 p0,p1,dp0,dp1,t0,t1,r0,r1;
            xtfloatx2 p2,p3,dp2,dp3,t2,t3,r2,r3;

            /* For this particular loop, XP address update results in a better schedule if compared with IP. */
            XT_LSIP( cs, T, +1*sz_f32 ); c0 = cs;
            XT_LSIP( cs, T, +1*sz_f32 ); c1 = cs;
            XT_LSIP( cs, T, +1*sz_f32 ); c2 = cs;
            XT_LSIP( cs, T, +1*sz_f32 ); c3 = cs;
            XT_LSIP( cs, T, +1*sz_f32 ); c4 = cs;
            XT_LSXP( cs, T, -5*sz_f32 ); c5 = cs;

            AE_LASX2X2_IP( xn0,xn1, X_va, X );
            AE_LASX2X2_IP( xn2,xn3, X_va, X );
            /* Determine the pi/2-wide segment the input value belongs to. */
            ABS_SX2X2( xn0,xn1,xn0,xn1 ); 
            ABS_SX2X2( xn2,xn3,xn2,xn3 ); 
            MULQ_S(jf0,jf1,xn0,xn1,inv2pif.f);
            MULQ_S(jf2,jf3,xn2,xn3,inv2pif.f);
            jf0 = XT_FIROUND_SX2( jf0 );  jf1 = XT_FIROUND_SX2( jf1 );
            jf2 = XT_FIROUND_SX2( jf2 );  jf3 = XT_FIROUND_SX2( jf3 );
            /* Calculate the difference between the segment midpoint and input value. */
            p0 = xn0; p1 = xn1;
            p2 = xn2; p3 = xn3;
            MSUBQ_S(p0,p1,jf0,jf1,c0); 
            MSUBQ_S(p2,p3,jf2,jf3,c0);
            MSUBQ_S(p0,p1,jf0,jf1,c1); 
            MSUBQ_S(p2,p3,jf2,jf3,c1);
            MSUBQ_S(p0,p1,jf0,jf1,c2); 
            MSUBQ_S(p2,p3,jf2,jf3,c2);
            MULQ_S(r0,r1,jf0,jf1,c3); t0 = p0; t1 = p1; SUB_SX2X2(p0,p1,p0,p1,r0,r1); SUB_SX2X2(t0,t1,t0,t1,p0,p1); SUB_SX2X2(t0,t1,t0,t1,r0,r1);           dp0=t0; dp1=t1;
            MULQ_S(r2,r3,jf2,jf3,c3); t2 = p2; t3 = p3; SUB_SX2X2(p2,p3,p2,p3,r2,r3); SUB_SX2X2(t2,t3,t2,t3,p2,p3); SUB_SX2X2(t2,t3,t2,t3,r2,r3);           dp2=t2; dp3=t3;
            MULQ_S(r0,r1,jf0,jf1,c4); t0 = p0; t1 = p1; SUB_SX2X2(p0,p1,p0,p1,r0,r1); SUB_SX2X2(t0,t1,t0,t1,p0,p1); SUB_SX2X2(t0,t1,t0,t1,r0,r1); ADD_SX2X2(dp0,dp1,t0,t1,dp0,dp1);
            MULQ_S(r2,r3,jf2,jf3,c4); t2 = p2; t3 = p3; SUB_SX2X2(p2,p3,p2,p3,r2,r3); SUB_SX2X2(t2,t3,t2,t3,p2,p3); SUB_SX2X2(t2,t3,t2,t3,r2,r3); ADD_SX2X2(dp2,dp3,t2,t3,dp2,dp3);
            MULQ_S(r0,r1,jf0,jf1,c5); t0 = p0; t1 = p1; SUB_SX2X2(p0,p1,p0,p1,r0,r1); SUB_SX2X2(t0,t1,t0,t1,p0,p1); SUB_SX2X2(t0,t1,t0,t1,r0,r1); ADD_SX2X2(dp0,dp1,t0,t1,dp0,dp1);
            MULQ_S(r2,r3,jf2,jf3,c5); t2 = p2; t3 = p3; SUB_SX2X2(p2,p3,p2,p3,r2,r3); SUB_SX2X2(t2,t3,t2,t3,p2,p3); SUB_SX2X2(t2,t3,t2,t3,r2,r3); ADD_SX2X2(dp2,dp3,t2,t3,dp2,dp3);
            ADD_SX2X2(p0,p1,p0,p1,dp0,dp1);
            ADD_SX2X2(p2,p3,p2,p3,dp2,dp3);

            AE_SSX2X2_IP( p0,p1, S_wr, 4*sz_f32 );
            AE_SSX2X2_IP( p2,p3, S_wr, 4*sz_f32 );
        }
        __Pragma( "no_reorder" );

    /*
     * Part II, polynomial approximation. Reference C code:
     *
     *   {
     *     float32_t xn, yn, ys, yc, p, p2;
     *     int sx, sy;
     *     int ji;
     *   
     *     for ( n=0; n<blkLen; n++ )
     *     {
     *       xn = x[blkIx*blkSize+n];
     *   
     *       // Determine the pi/2-wide segment the input value belongs to.
     *       ji = (int)roundf( fabsf(xn)*inv2pif.f );
     *   
     *       // Adjust the sign.
     *       sx = takesignf( xn );
     *       sy = sx ^ ((ji>>1)&1);
     *   
     *       //
     *       // Compute sine/cosine approximation via minmax polynomials.
     *       //
     *   
     *       p = scr[n];
     *       p2 = p*p;
     *   
     *       ys = polysinf_tbl[0].f;
     *       ys = polysinf_tbl[1].f + ys*p2;
     *       ys = polysinf_tbl[2].f + ys*p2;
     *       ys = ys*p2;
     *       ys = ys*p + p;
     *   
     *       yc = polycosf_tbl[0].f;
     *       yc = polycosf_tbl[1].f + yc*p2;
     *       yc = polycosf_tbl[2].f + yc*p2;
     *       yc = yc*p2 + 1.f;
     *   
     *       // Select sine or cosine.
     *       yn = ( (ji&1) ? yc : ys );
     *       // Check for input domain.
     *       if ( fabsf(xn) > sinf_maxval.f ) yn = qNaNf.f;
     *       // Apply the sign.
     *       y[blkIx*blkSize+n] = changesignf( yn, sy );
     *     }
     *   }
     */
        X    = (xtfloatx4*)(x);
        X1   = (xtfloatx4*)(x);
        Y    = (xtfloatx4*)(y);
        S_rd = (xtfloatx4*)scr;

        X_va  = AE_LA128_PP( X );
        X1_va = AE_LA128_PP( X1 );
        Y_va  = AE_ZALIGN128();
        __Pragma( "loop_count factor=2" );
        for ( n=0; n<(blkLen>>2); n++ )
        {
            /* Input value; reducted input value and its 2nd power; auxiliary var */
            xtfloatx2 xn0, p0, p20, t0;
            xtfloatx2 xn1, p1, p21, t1;
            /* Input value segment number; input and output signs; integer reprentation of output value */
            ae_int32x2 ji0, sx0, sy0;
            ae_int32x2 ji1, sx1, sy1;
            /* Cosine and sine approximations; output value */
            xtfloatx2 yc0, ys0;
            xtfloatx2 yc1, ys1;
            /* Polynomial coefficients for sine and cosine. */
            xtfloatx2 cf_s0, cf_s1, cf_s2;
            xtfloatx2 cf_c0, cf_c1, cf_c2;
            /* Cosine/sine selection; out-of-domain flags */
            xtbool2  b_ndom0;
            xtbool2  b_ndom1;

            AE_LASX2X2_IP( xn0,xn1, X_va, X );
            /* Determine the pi/2-wide segment the input value belongs to. */
            ABS_SX2X2(xn0,xn1,xn0,xn1 );
            MULQ_S( t0,t1,xn0,xn1, inv2pif.f );
            t0 = XT_FIROUND_SX2( t0 );           t1 = XT_FIROUND_SX2( t1 );
            ji0 = XT_TRUNC_SX2( t0, 0 );         ji1 = XT_TRUNC_SX2( t1, 0 );
            /*
             * Compute polynomial approximations of sine and cosine for the
             * reducted input value.
             */
            AE_LSX2X2_IP( p0,p1, S_rd, 4*sz_f32 );
            MUL_SX2X2(p20,p21, p0,p1, p0,p1 );
            { ae_int32x2 t; AE_L32_XP(t,castxcc(ae_int32,psintbl),0); cf_s0=XT_AE_MOVXTFLOATX2_FROMINT32X2(t); }
            cf_s1 = XT_LSI( psintbl, 1*sz_f32 );
            cf_s2 = XT_LSI( psintbl, 2*sz_f32 );
            { ae_int32x2 t; AE_L32_XP(t,castxcc(ae_int32,pcostbl),0); cf_c0=XT_AE_MOVXTFLOATX2_FROMINT32X2(t); }
            cf_c1 = XT_LSI( pcostbl, 1*sz_f32 );
            cf_c2 = XT_LSI( pcostbl, 2*sz_f32 );


            t0=t1=cf_s1; MADDQ_S( t0,t1, p20,p21, cf_s0 ); ys0 = t0; ys1 = t1;
            t0=t1=cf_s2; MADD_SX2X2( t0,t1, ys0,ys1, p20,p21 ); ys0 = t0; ys1 = t1;
            MUL_SX2X2 (ys0,ys1,ys0,ys1, p20,p21 );
            MADD_SX2X2(p0,p1,  ys0,ys1, p0,p1   ); ys0 = p0; ys1 = p1;

            t0=t1=cf_c1; MADDQ_S   ( t0,t1, p20,p21, cf_c0   ); yc0 = t0; yc1 = t1;
            t0=t1=cf_c2; MADD_SX2X2( t0,t1, yc0,yc1, p20,p21 ); yc0 = t0; yc1 = t1;
            CONST_SX2X2(t0,t1,1); MADD_SX2X2( t0,t1, yc0,yc1, p20,p21 ); 

            /* Select sine or cosine. */
            XT_MOVT_SX2( ys0, t0, AE_MOVBD1X2(ji0));              
            XT_MOVT_SX2( ys1, t1, AE_MOVBD1X2(ji1));

            b_ndom0 = XT_OLT_SX2( sinf_maxval.f, xn0 );            
            b_ndom1 = XT_OLT_SX2( sinf_maxval.f, xn1 );
            /* Set result to NaN for an out-of-domain input value. */
            XT_MOVT_SX2( ys0, qNaNf.f, b_ndom0 );                  
            XT_MOVT_SX2( ys1, qNaNf.f, b_ndom1 );

            /* Adjust the sign. */
            AE_LASX2X2_IP( xn0,xn1, X1_va, X1 );
            /* Determine the input sign. */
            sx0 = XT_AE_MOVINT32X2_FROMXTFLOATX2( xn0);
            sx1 = XT_AE_MOVINT32X2_FROMXTFLOATX2( xn1 );
#if 0
            {
                xtfloatx2 mys0,mys1;
                sy0 = AE_SLLI32( ji0, 30 );   sy1 = AE_SLLI32( ji1, 30 );
                sy0 = AE_XOR32( sx0, sy0 );   sy1 = AE_XOR32( sx1, sy1 );
                NEG_SX2X2(mys0,mys1,ys0,ys1);
                XT_MOVT_SX2(ys0,mys0,AE_LT32(sy0,0));
                XT_MOVT_SX2(ys1,mys1,AE_LT32(sy1,0));
            }
#elif 1
            {
                sy0 = AE_SLLI32( ji0, 30 );   sy1 = AE_SLLI32( ji1, 30 );
                sy0 = AE_XOR32( sx0, sy0 );   sy1 = AE_XOR32( sx1, sy1 );
                sy0 = AE_XOR32(XT_AE_MOVINT32X2_FROMXTFLOATX2(ys0),AE_AND32(sy0,0x80000000)); 
                sy1 = AE_XOR32(XT_AE_MOVINT32X2_FROMXTFLOATX2(ys1),AE_AND32(sy1,0x80000000)); 
                ys0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(sy0);
                ys1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(sy1);
            }
#else
            {
                xtfloatx2 s0,s1;
                sy0 = AE_SLLI32( ji0, 30 );   sy1 = AE_SLLI32( ji1, 30 );
                sy0 = AE_XOR32( sx0, sy0 );   sy1 = AE_XOR32( sx1, sy1 );
                CONST_SX2X2(s0,s1,1);
                XT_MOVT_SX2(s0,-1.f,AE_LT32(sy0,0));
                XT_MOVT_SX2(s1,-1.f,AE_LT32(sy1,0));
                MUL_SX2X2(ys0,ys1,ys0,ys1,s0,s1);
            }
#endif
            /* Set result to NaN for an out-of-domain input value. */
            AE_SASX2X2_IP( ys0,ys1, Y_va, Y );
      }
      AE_SA128POS_FP( Y_va, Y );
    }
}

#endif

/*===========================================================================
  Vector matematics:
  vec_sine            sine    
===========================================================================*/

/*-------------------------------------------------------------------------
  Sine/Cosine 
  Fixed-point functions calculate sin(pi*x) or cos(pi*x) for numbers written 
  in Q31 or Q15 format. Return results in the same format. 
  Floating point functions compute sin(x) or cos(x)
  Two versions of functions available: regular version (vec_sine32x32, 
  vec_cosine32x32, , vec_sinef, vec_cosinef) 
  with arbitrary arguments and faster version (vec_sine32x32_fast, 
  vec_cosine32x32_fast) that apply some restrictions.
  NOTE:
  1.  Scalar floating point functions are compatible with standard ANSI C
      routines and set errno and exception flags accordingly
  2.  Floating point functions limit the range of allowable input values:
      [-102940.0, 102940.0] Whenever the input value does not belong to this
      range, the result is set to NaN.

  Precision: 
  32x32  32-bit inputs, 32-bit output. Accuracy: 1700 (7.9e-7)
  f      floating point. Accuracy 2 ULP

  Input:
  x[N]  input data,Q31 or floating point
  N     length of vectors
  Output:
  y[N]  output data,Q31 or floating point

  Restriction:
  Regular versions (vec_sine32x32, vec_cosine32x32, vec_sinef, 
  vec_cosinef):
  x,y - should not overlap

  Faster versions (vec_sine32x32_fast, vec_cosine32x32_fast):
  x,y - should not overlap
  x,y - aligned on 16-byte boundary
  N   - multiple of 2

  Scalar versions:
  ----------------
  return result in Q31 or floating point
-------------------------------------------------------------------------*/
void vec_sinef( float32_t * restrict y,
          const float32_t * restrict x,
          int N )
{
  const int blkSize = MAX_ALLOCA_SZ/sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */
  float32_t ALIGN(32) scr[blkSize];
  float32_t ALIGN(32) tmpIn[8],tmpOut[8];

  int M;
  if ( N<=0 ) return;
  M=N&~7;
  if ( M )
  {
      mysinef(scr,y,x,M); 
      y+=M;
      x+=M;
      N&=7;
  }
  if (N) 
  {     // processing the tail
      int off1,off2,off3,off4,off5,off6;
      xtfloat x0,x1,x2,x3,x4,x5,x6;
      off1=XT_MIN(N-1,1)<<2;
      off2=XT_MIN(N-1,2)<<2;
      off3=XT_MIN(N-1,3)<<2;
      off4=XT_MIN(N-1,4)<<2;
      off5=XT_MIN(N-1,5)<<2;
      off6=XT_MIN(N-1,6)<<2;
      x0=XT_LSI((const xtfloat*)x,0);
      x1=XT_LSX((const xtfloat*)x,off1);
      x2=XT_LSX((const xtfloat*)x,off2);
      x3=XT_LSX((const xtfloat*)x,off3);
      x4=XT_LSX((const xtfloat*)x,off4);
      x5=XT_LSX((const xtfloat*)x,off5);
      x6=XT_LSX((const xtfloat*)x,off6);
      XT_SSI(x0,(xtfloat*)tmpIn,0*sizeof(xtfloat));
      XT_SSI(x1,(xtfloat*)tmpIn,1*sizeof(xtfloat));
      XT_SSI(x2,(xtfloat*)tmpIn,2*sizeof(xtfloat));
      XT_SSI(x3,(xtfloat*)tmpIn,3*sizeof(xtfloat));
      XT_SSI(x4,(xtfloat*)tmpIn,4*sizeof(xtfloat));
      XT_SSI(x5,(xtfloat*)tmpIn,5*sizeof(xtfloat));
      XT_SSI(x6,(xtfloat*)tmpIn,6*sizeof(xtfloat));
      XT_SSI(XT_CONST_S(0),(xtfloat*)tmpIn,7*sizeof(xtfloat));
      mysinef(scr,tmpOut,tmpIn,8); 
      x0=XT_LSI((const xtfloat*)tmpOut,0*sizeof(xtfloat));
      x1=XT_LSI((const xtfloat*)tmpOut,1*sizeof(xtfloat));
      x2=XT_LSI((const xtfloat*)tmpOut,2*sizeof(xtfloat));
      x3=XT_LSI((const xtfloat*)tmpOut,3*sizeof(xtfloat));
      x4=XT_LSI((const xtfloat*)tmpOut,4*sizeof(xtfloat));
      x5=XT_LSI((const xtfloat*)tmpOut,5*sizeof(xtfloat));
      x6=XT_LSI((const xtfloat*)tmpOut,6*sizeof(xtfloat));
      XT_SSX(x6,(xtfloat*)y,off6);
      XT_SSX(x5,(xtfloat*)y,off5);
      XT_SSX(x4,(xtfloat*)y,off4);
      XT_SSX(x3,(xtfloat*)y,off3);
      XT_SSX(x2,(xtfloat*)y,off2);
      XT_SSX(x1,(xtfloat*)y,off1);
      XT_SSI(x0,(xtfloat*)y,0);
  }
}
#elif HAVE_FPU
#define sz_f32    (int)sizeof(float32_t)

/*===========================================================================
  Vector matematics:
  vec_sine            sine    
===========================================================================*/

/*-------------------------------------------------------------------------
  Sine/Cosine 
  Fixed-point functions calculate sin(pi*x) or cos(pi*x) for numbers written 
  in Q31 or Q15 format. Return results in the same format. 
  Floating point functions compute sin(x) or cos(x)
  Two versions of functions available: regular version (vec_sine32x32, 
  vec_cosine32x32, , vec_sinef, vec_cosinef) 
  with arbitrary arguments and faster version (vec_sine32x32_fast, 
  vec_cosine32x32_fast) that 
  apply some restrictions.
  NOTE:
  1.  Scalar floating point functions are compatible with standard ANSI C
      routines and set errno and exception flags accordingly
  2.  Floating point functions limit the range of allowable input values:
      [-102940.0, 102940.0] Whenever the input value does not belong to this
      range, the result is set to NaN.

  Precision: 
  32x32  32-bit inputs, 32-bit output. Accuracy: 1700 (7.9e-7)
  f      floating point. Accuracy 2 ULP

  Input:
  x[N]  input data,Q31 or floating point
  N     length of vectors
  Output:
  y[N]  output data,Q31 or floating point

  Restriction:
  Regular versions (vec_sine32x32, vec_cosine32x32, vec_sinef, vec_cosinef):
  x,y - should not overlap

  Faster versions (vec_sine32x32_fast, vec_cosine32x32_fast):
  x,y - should not overlap
  x,y - aligned on 16-byte boundary
  N   - multiple of 2

  Scalar versions:
  ----------------
  return result in Q31 or floating point
-------------------------------------------------------------------------*/

#if SINNCOSF_ALG==0
void vec_sinef( float32_t * restrict y, const float32_t * restrict x, int N )
{
    /* Reference code
    * const union ufloat32uint32 pi2fc[3] = {
    *     { 0x3fc90fdb }, { 0xb33bbd2e },
    *     { 0xa6f72ced }
    * };
    * float32_t x2, y, ys, yc;
    * int sx, j, ss, sc;
    * sx = takesignf(x);
    * x = sx ? -x : x;
    * argument reduction 
    * j = (int)roundf(x * inv2pif.f);
    * x = fmaf(pi2fc[0].f, -j, x);
    * x = fmaf(pi2fc[1].f, -j, x);
    * x = fmaf(pi2fc[2].f, -j, x);
    * adjust signs 
    * ss = sx ^ (((j) >> 1) & 1);
    * sc = ((j + 1) >> 1) & 1;
    * compute sine/cosine via minmax polynomial  
    * x2 = x * x;
    * ys = polysinf_tbl[0].f;
    * ys = ys * x2 + polysinf_tbl[1].f;
    * ys = ys * x2 + polysinf_tbl[2].f;
    * ys = ys * x2;
    * ys = ys * x + x;
    * yc = polycosf_tbl[0].f;
    * yc = yc * x2 + polycosf_tbl[1].f;
    * yc = yc * x2 + polycosf_tbl[2].f;
    * yc = yc * x2 + 1.f;
    * select sine/cosine 
    * y = (j & 1) ? yc : ys;
    *  apply the sign 
    * y = changesignf(y, ss);
    * return y;
    */
  const xtfloat *          X;
  const xtfloat *          S_rd;
  const xtfloat *          T;
        xtfloat * restrict S_wr;
  const xtfloat * restrict TBLS;
  const xtfloat * restrict TBLC;

  /* Current block index; overall number of blocks; number of values in the current block */
  int blkIx, blkNum, blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / (2*sz_f32);
  /* Allocate a fixed-size scratch area on the stack. */
  float32_t ALIGN(32) scr[2*blkSize];

  int n;
  if (N <= 0) return;

  NASSERT_ALIGN8(scr);
  /*
  * Data are processed in blocks of scratch area size. Further, the algorithm
  * implementation is splitted in order to feed the optimizing compiler with a
  * few loops of managable size.
  */

  blkNum = (N + blkSize - 1) / blkSize;

  for (blkIx = 0; blkIx<blkNum; blkIx++)
  {
    blkLen = XT_MIN(N - blkIx*blkSize, blkSize);
    /*
    * Part I, range reduction.
    */
    {
      /* Input value; reducted input value; correction term. */
      xtfloat xn, p;
      /* Auxiliary floating-point vars. */
      /* Input value segment number. */
      xtfloat jf;
      /*  24bit chunks of pi/2*/
      xtfloat c0, c1, c2;
      const union ufloat32uint32 c[3] = {
                  { 0x3fc90fdb }, { 0xb33bbd2e },
                  { 0xa6f72ced }
      };
      X = (xtfloat*)((uintptr_t)x + blkIx*blkSize*sz_f32);
      S_wr = (xtfloat*)scr;
      T = (xtfloat  *)c;
      __Pragma("loop_count min=1");
      for (n = 0; n<(blkLen );  n++)
      {
        XT_LSIP(xn, X, sz_f32);
        /*
        * Determine the pi/2-wide segment the input value belongs to.
        */
        xn = XT_ABS_S(xn);

        jf = XT_MUL_S(xn, inv2pif.f);

        jf = XT_ROUND_S(jf, 0);

        c0 = XT_LSI(T, 0 * sz_f32);
        c1 = XT_LSI(T, 1 * sz_f32);
        c2 = XT_LSI(T, 2 * sz_f32);

        p = xn;
        XT_MSUB_S(p, jf, c0);
        XT_MSUB_S(p, jf, c1);
        XT_MSUB_S(p, jf, c2);

        XT_SSIP(p, S_wr, sz_f32);
      }
    }
    __Pragma("no_reorder");
    /*
    * Part II, polynomial approximation, sign adjustment.
    */
    {
      /* Input value; reducted input value and its 2nd power; auxiliary var */
      xtfloat xn, p, p2;
      /* Polynomial coefficients for sine and cosine. */
      xtfloat s0, s1, s2, c0, c1, c2;
      /* Cosine/sine selection; out-of-domain flags */
      xtbool b_cs, b_ndom;
      int32_t * pY;
      X = (xtfloat*)((uintptr_t)x + blkIx*blkSize*sz_f32);
      pY = (int32_t*)((uintptr_t)y + blkIx*blkSize*sz_f32);
      S_rd = (xtfloat*)scr;
      TBLS = (const xtfloat *)polysinf_tbl;
      TBLC = (const xtfloat *)polycosf_tbl;
      __Pragma("loop_count min=1");
      for (n = 0; n<blkLen ; n++)
      {
        int32_t sx, ss, n0;
        xtfloat t0, y0, r0, _s, _c;
        int32_t j0;
        xn = XT_LSI(X, 0*sz_f32);

        /* Determine the pi/2-wide segment the input value belongs to. */
        xn = XT_ABS_S(xn);
        t0 = XT_MUL_S(xn, inv2pif.f);
        t0 = XT_ROUND_S(t0, 0);
        j0 = (int)XT_TRUNC_S(t0, 0);

        /* adjust signs  */
        ss = j0 << 30;
        /*
        * Compute polynomial approximations of sine and cosine for the
        * reducted input value.
        */
 
        s0= XT_LSI(TBLS, 0 * sz_f32);
        s1= XT_LSI(TBLS, 1 * sz_f32);
        s2= XT_LSI(TBLS, 2 * sz_f32);
        c0= XT_LSI(TBLC, 0 * sz_f32);
        c1= XT_LSI(TBLC, 1 * sz_f32);
        c2= XT_LSI(TBLC, 2 * sz_f32);
   
        XT_LSIP(p, S_rd, sz_f32);
        p2 = XT_MUL_S(p, p);
        y0 = s1; 
        XT_MADD_S(y0, s0, p2); r0 = y0; y0 = s2;
        XT_MADD_S(y0, r0, p2);
		y0 = XT_MUL_S(y0, p2);
        t0 = p;
        XT_MADD_S(t0, y0, p); _s = t0;
   
        y0 = c1; 
        XT_MADD_S(y0, c0, p2); r0 = y0; y0 = c2;
        XT_MADD_S(y0, r0, p2);
        t0 = XT_CONST_S(1);
        XT_MADD_S(t0, y0, p2); _c = t0;

        /* Select sine or cosine. */
        b_cs = AE_MOVBA(j0);
        XT_LSIP(xn, X, sz_f32);
        XT_MOVF_S(_c, _s, b_cs);
        /* Determine the input sign. */
        sx = XT_RFR(xn);
        xn = XT_ABS_S(xn);
        
        n0 = XT_RFR(_c);
        sx = XT_XOR(sx, ss);
        sx = sx & 0x80000000;
        n0 = XT_XOR(n0, sx);

        /* Set result to NaN for an out-of-domain input value. */
        b_ndom = XT_OLT_S(sinf_maxval.f, xn);

        {
          unsigned int t = n0;
          XT_MOVT(t, qNaNf.u, b_ndom); n0=t;
        }
        *pY++=n0;
      }
    }
  }
} /* vec_sinef() */

#else 
// Taken from Fusion
void vec_sinef( float32_t * restrict y, const float32_t * restrict x, int N )
{
  const xtfloat *          X;
  const xtfloat *          S_rd;
  const xtfloat *          T;
  const int     *          pT;
        xtfloat * restrict S_wr;
  const xtfloat * restrict TBLS;
  const xtfloat * restrict TBLC;

  /* Current block index; overall number of blocks; number of values in the current block */
  int blkIx, blkNum, blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */
  float32_t ALIGN(32) scr[blkSize];

  int n;
  if (N <= 0) return;

  NASSERT_ALIGN8(scr);
  /*
  * Data are processed in blocks of scratch area size. Further, the algorithm
  * implementation is splitted in order to feed the optimizing compiler with a
  * few loops of managable size.
  */

  blkNum = (N + blkSize - 1) / blkSize;

  for (blkIx = 0; blkIx<blkNum; blkIx++)
  {
    blkLen = XT_MIN(N - blkIx*blkSize, blkSize);
    /*
    * Part I, range reduction. Reference C code:
    *
    *   {
    *     float32_t xn, p, dp, t;
    *     int ji;
    *     float32_t jf;
    *
    *     static const union ufloat32uint32 c[6] = {
    *       { 0x3f4a0000 }, { 0xbb700000 },
    *       { 0xb6160000 }, { 0x32080000 },
    *       { 0x2e060000 }, { 0xa9b9ee5a }
    *     };
    *
    *     for ( n=0; n<blkLen; n++ )
    *     {
    *       xn = fabsf( x[blkIx*blkSize+n] );
    *
    *       // Determine the pi/2-wide segment the input value belongs to.
    *       ji = ( ( (int)floorf( xn*inv4pif.f ) + 1 ) & ~1 );
    *       jf = (float32_t)ji;
    *
    *       // Calculate the difference between the segment midpoint and input value.
    *       p = xn;
    *       p -= c[0].f*jf;
    *       p -= c[1].f*jf;
    *       p -= c[2].f*jf;
    *       t = p; p -= c[3].f*jf; t = t - p; t -= c[3].f*jf; dp = t;
    *       t = p; p -= c[4].f*jf; t = t - p; t -= c[4].f*jf; dp += t;
    *       t = p; p -= c[5].f*jf; t = t - p; t -= c[5].f*jf; dp += t;
    *       p += dp;
    *
    *       scr[n] = p;
    *     }
    *   }
    */
    {
      /* Input value; reducted input value; correction term. */
      xtfloat xn, p, dp;
      /* Auxiliary floating-point vars. */
      xtfloat t, r;
      /* Input value segment number. */
      ae_int32 ji, i0;
      xtfloat jf;
      /* pi/4 splitted into 7-bit chunks. */
      xtfloat c0, c1, c2, c3, c4, c5;

      static const  uint32_t ALIGN(32) c[6] = {
        0x3f4a0000, 0xbb700000,
        0xb6160000, 0x32080000,
        0x2e060000, 0xa9b9ee5a
      };
      /* 4/pi, 1, ~1 */
      static const uint32_t TAB[3] = { 0x3fa2f983, 0x00000001,
        0xFFFFFFFE
      };
      X = (xtfloat*)((uintptr_t)x + blkIx*blkSize*sz_f32);
      S_wr = (xtfloat*)scr;
      T = (xtfloat  *)c;
      pT = (int  *)TAB;

      __Pragma("loop_count min=1");
      for (n = 0; n<(blkLen );  n++)
      {
        XT_LSIP(xn, X, sz_f32);
        /*
        * Determine the pi/2-wide segment the input value belongs to.
        */
        xn = XT_ABS_S(xn);
        XT_LSIP(c0, castxcc(xtfloat,pT), sz_f32);
        t = XT_MUL_S(xn, c0);
        ji = XT_TRUNC_S(t, 0);
        i0=XT_L32I(pT,0); 
        ji = XT_ADD(ji, i0);
        i0=XT_L32I(pT, sz_f32); pT--;
        ji = XT_AND(ji, i0);
        jf = XT_FLOAT_S(ji, 0);

        /*
        * Calculate the difference between the segment midpoint and input value.
        */

        c0 = XT_LSI( T, 0 * sz_f32);
        c1 = XT_LSI( T, 1 * sz_f32);
        c2 = XT_LSI( T, 2 * sz_f32);
        c3 = XT_LSI( T, 3 * sz_f32);
        c4 = XT_LSI( T, 4 * sz_f32);
        c5 = XT_LSI( T, 5 * sz_f32);

        p = xn;
        XT_MSUB_S(p, jf, c0);
        XT_MSUB_S(p, jf, c1);
        XT_MSUB_S(p, jf, c2);

        r = XT_MUL_S(jf, c3); t = p; p = XT_SUB_S(p, r); t = XT_SUB_S(t, p); t = XT_SUB_S(t, r); dp = t;
        r = XT_MUL_S(jf, c4); t = p; p = XT_SUB_S(p, r); t = XT_SUB_S(t, p); t = XT_SUB_S(t, r); dp = XT_ADD_S(t, dp);
        r = XT_MUL_S(jf, c5); t = p; p = XT_SUB_S(p, r); t = XT_SUB_S(t, p); t = XT_SUB_S(t, r); dp = XT_ADD_S(t, dp);

        p = XT_ADD_S(p, dp);

        XT_SSIP(p, S_wr, sz_f32);
      }
    }
    __Pragma("no_reorder");
    /*
    * Part II, polynomial approximation. Reference C code:
    *
    *   {
    *     float32_t xn, yn, ys, yc, p, p2;
    *     int sx, sy;
    *     int ji;
    *
    *     for ( n=0; n<blkLen; n++ )
    *     {
    *       xn = x[blkIx*blkSize+n];
    *
    *       // Determine the pi/2-wide segment the input value belongs to.
    *       ji = (int)floorf( fabsf(xn)*inv4pif.f ) + 1;
    *
    *       // Adjust the sign.
    *       sx = takesignf( xn );
    *       sy = sx ^ ((ji>>2)&1);
    *
    *       //
    *       // Compute sine/cosine approximation via minmax polynomials.
    *       //
    *
    *       p = scr[n];
    *       p2 = p*p;
    *
    *       ys = polysinf_tbl[0].f;
    *       ys = polysinf_tbl[1].f + ys*p2;
    *       ys = polysinf_tbl[2].f + ys*p2;
    *       ys = ys*p2;
    *       ys = ys*p + p;
    *
    *       yc = polycosf_tbl[0].f;
    *       yc = polycosf_tbl[1].f + yc*p2;
    *       yc = polycosf_tbl[2].f + yc*p2;
    *       yc = yc*p2 + 1.f;
    *
    *       // Select sine or cosine.
    *       yn = ( (ji&2) ? yc : ys );
    *       // Check for input domain.
    *       if ( fabsf(xn) > sinf_maxval.f ) yn = qNaNf.f;
    *       // Apply the sign.
    *       y[blkIx*blkSize+n] = changesignf( yn, sy );
    *
    *       //
    *       // Perform additional analysis of input data for Error Handling.
    *       //
    *
    *       #if VEC_SINEF_ERRH != 0
    *       {
    *         if ( isnan(xn)    || fabsf(xn) > sinf_maxval.f ) i2_edom    = 1;
    *         if ( is_snanf(xn) || fabsf(xn) > sinf_maxval.f ) i2_fe_inv  = 1;
    *       }
    *       #endif
    *     }
    *   }
    */
    {
      /* Input value; reducted input value and its 2nd power; auxiliary var */
      xtfloat xn, p, p2;
      /* Polynomial coefficients for sine and cosine. */
      xtfloat s0, s1, s2, c0, c1, c2;
      /* Cosine/sine selection; out-of-domain flags */
      xtbool b_cs, b_ndom;
      int32_t * pY;
      X = (xtfloat*)((uintptr_t)x + blkIx*blkSize*sz_f32);
      pY = (int32_t*)((uintptr_t)y + blkIx*blkSize*sz_f32);
      S_rd = (xtfloat*)scr;
      TBLS = (const xtfloat *)polysinf_tbl;
      TBLC = (const xtfloat *)polycosf_tbl;
      __Pragma("loop_count min=1");
      for (n = 0; n<blkLen ; n++)
      {
        int32_t sx, ss, n0, tmp;
        xtfloat t0, y0, r0, _s, _c;
        int32_t j0;
        xn = XT_LSI(X, 0*sz_f32);

        /* Determine the pi/2-wide segment the input value belongs to. */
        xn = XT_ABS_S(xn);
        t0 = XT_MUL_S(xn, inv4pif.f);
        j0 = (int)XT_TRUNC_S(t0, 0);

        n0 = j0 + 1;
        /* adjust signs  */
        tmp = n0 & 4;
        ss = tmp << 29;
        /*
        * Compute polynomial approximations of sine and cosine for the
        * reducted input value.
        */
 
        s0= XT_LSI(TBLS, 0 * sz_f32);
        s1= XT_LSI(TBLS, 1 * sz_f32);
        s2= XT_LSI(TBLS, 2 * sz_f32);
        c0= XT_LSI(TBLC, 0 * sz_f32);
        c1= XT_LSI(TBLC, 1 * sz_f32);
        c2= XT_LSI(TBLC, 2 * sz_f32);
   
        XT_LSIP(p, S_rd, sz_f32);
        p2 = XT_MUL_S(p, p);
        y0 = s1; 
        XT_MADD_S(y0, s0, p2); r0 = y0; y0 = s2;
        XT_MADD_S(y0, r0, p2);
		y0 = XT_MUL_S(y0, p2);
        t0 = p;
        XT_MADD_S(t0, y0, p); _s = t0;
   
        y0 = c1; 
        XT_MADD_S(y0, c0, p2); r0 = y0; y0 = c2;
        XT_MADD_S(y0, r0, p2);
        t0 = XT_CONST_S(1);
        XT_MADD_S(t0, y0, p2); _c = t0;

        /* Select sine or cosine. */
        n0 = n0 & 2;
 
        b_cs = AE_INT64_EQ(AE_ZERO64(), AE_MOVINT64_FROMINT32(n0));
        XT_LSIP(xn, X, sz_f32);
        /* Determine the input sign. */
        sx = XT_RFR(xn);
        sx = sx & 0x80000000;
        xn = XT_ABS_S(xn);
         
        XT_MOVT_S(_c, _s, b_cs);
        n0 = XT_RFR(_c);
        sx = XT_XOR(sx, ss);
        n0 = XT_XOR(n0, sx);

        /* Set result to NaN for an out-of-domain input value. */
        b_ndom = XT_OLT_S(sinf_maxval.f, xn);

        {
          unsigned int t = n0;
          XT_MOVT(t, qNaNf.u, b_ndom); n0=t;
        }
        *pY++=n0;
      }
    }
  }
} /* vec_sinef() */

#endif

#endif
