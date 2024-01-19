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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

#define ALIGNMENT   16   /* 16 bytes alignment */

#if HAVE_VFPU

#if HAVE_VFPU
/*#define ENABLE_PRAGMA*/


/* 
 * x & v : matrices of dims [rows, cols1] and [rows, cols2]
 * y & w : column vector of length cols1 and cols2 respectively
 *     b : column vector of length rows
 * 
 * This kernel computes 
 *                          z = (x*y) + (v*w) + b
 *
 * If cols2 is specified as zero, this kernal will compute
 *                          z = (x*y) + b
 *   (without incurring the overhead of computing v*w)
 */
WORD32 static dual_mtx_vecmpyf_bias_add( FLOAT32 * z,
     const FLOAT32 * x,  const FLOAT32 * y, const FLOAT32 * v, const FLOAT32 * w,
     const FLOAT32 * b, int rows, int cols1, int cols2, int row_stride1, int row_stride2 )
{
  const xtfloatx4 *restrict px0;
  const xtfloatx4 *restrict px1;
  const xtfloatx4 *restrict px2;
  const xtfloatx4 *restrict px3;
  const xtfloatx4 *restrict px4;
  const xtfloatx4 *restrict px5;
  const xtfloatx4 *restrict px6;
  const xtfloatx4 *restrict px7;

  const xtfloatx4 *restrict pv0;
  const xtfloatx4 *restrict pv1;
  const xtfloatx4 *restrict pv2;
  const xtfloatx4 *restrict pv3;
  const xtfloatx4 *restrict pv4;
  const xtfloatx4 *restrict pv5;
  const xtfloatx4 *restrict pv6;
  const xtfloatx4 *restrict pv7;

  const xtfloatx4 *restrict py;
  const xtfloatx4 *restrict pw;
  const xtfloatx4 *restrict pb;
        xtfloatx4 *restrict pz;
        xtfloat   *restrict pz_;

  xtfloatx2 b0, b1;
  xtfloatx2 b2, b3;
  xtfloatx2 y0, y1, y2, y3;
  xtfloatx2 w0, w1;
  xtfloatx2 z0, z1;
  xtfloatx2 z2, z3;
  xtfloat z0_, b0_;

  xtfloatx2 x00, x01, x02, x03,
            x10, x11, x12, x13,
            x20, x21, x22, x23,
            x30, x31, x32, x33,
            x40, x41,
            x50, x51,
            x60, x61,
            x70, x71;

  xtfloatx2 v00, v01,
            v10, v11,
            v20, v21,
            v30, v31,
            v40, v41,
            v50, v51,
            v60, v61,
            v70, v71;

  xtfloatx2 acc00, acc01, acc02, acc03,
            acc10, acc11, acc12, acc13,
            acc20, acc21, acc22, acc23,
            acc30, acc31, acc32, acc33;
  xtfloatx2 acc40, acc41, acc50, acc51,
            acc60, acc61, acc70, acc71;
  int m, n, k;

  NASSERT(x);
  NASSERT(y);
  NASSERT(v);
  NASSERT(w);
  NASSERT(z);
  NASSERT((z != x) && (z != y) && (z != v) && (z != w) && (z != b));
  NASSERT_ALIGN(x,8);
  NASSERT_ALIGN(y,8);
  NASSERT_ALIGN(v,8);
  NASSERT_ALIGN(w,8);
  NASSERT_ALIGN(z,8);
  NASSERT_ALIGN(b,8);
  NASSERT(cols1%4==0);
  NASSERT(cols2%4==0);
  NASSERT(row_stride1%4==0);
  NASSERT(row_stride2%4==0);

  //if ((b == NULL) || (z == NULL))
  if (rows < 1)
  {
    return -2;
  }

  pz = (xtfloatx4 *)z;
  pb = (const xtfloatx4 *)(b);

  //if ((x != NULL) && (y != NULL) && (v != NULL) && (w != NULL))
  if (x && y && v && w && (cols1 > 0) && (cols2 > 0))           // Calculate z = (x*y) + (v*w) + b
  {
    /* Compute by 4 values */
#if defined(ENABLE_PRAGMA)
    __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
    for (m = 0; m < (rows>>3); m++)
    {
      px0 = (const xtfloatx4 *)(x+(8*m*row_stride1));
      pv0 = (const xtfloatx4 *)(v+(8*m*row_stride2));

      px1 = (const xtfloatx4 *)((FLOAT32 *)px0+row_stride1);
      px2 = (const xtfloatx4 *)((FLOAT32 *)px1+row_stride1);
      px3 = (const xtfloatx4 *)((FLOAT32 *)px2+row_stride1);
      px4 = (const xtfloatx4 *)((FLOAT32 *)px3+row_stride1);
      px5 = (const xtfloatx4 *)((FLOAT32 *)px4+row_stride1);
      px6 = (const xtfloatx4 *)((FLOAT32 *)px5+row_stride1);
      px7 = (const xtfloatx4 *)((FLOAT32 *)px6+row_stride1);

      py  = (const xtfloatx4 *)(y);

      pv1 = (const xtfloatx4 *)((FLOAT32 *)pv0+row_stride2);
      pv2 = (const xtfloatx4 *)((FLOAT32 *)pv1+row_stride2);
      pv3 = (const xtfloatx4 *)((FLOAT32 *)pv2+row_stride2);
      pv4 = (const xtfloatx4 *)((FLOAT32 *)pv3+row_stride2);
      pv5 = (const xtfloatx4 *)((FLOAT32 *)pv4+row_stride2);
      pv6 = (const xtfloatx4 *)((FLOAT32 *)pv5+row_stride2);
      pv7 = (const xtfloatx4 *)((FLOAT32 *)pv6+row_stride2);
      pw  = (const xtfloatx4 *)(w);

      acc00 = acc01 = acc10 = acc11 =
      acc20 = acc21 = acc30 = acc31 =  (xtfloatx2)0.0f;

      acc40 = acc41 = acc50 = acc51 =
      acc60 = acc61 = acc70 = acc71 =  (xtfloatx2)0.0f;

      b0 = b1 = (xtfloatx2)0.0f;
      b2 = b3 = (xtfloatx2)0.0f;
      if(b != NULL)
      {
        AE_LSX2X2_IP(b0,b1, pb,sizeof(xtfloatx4));
        AE_LSX2X2_IP(b2,b3, pb,sizeof(xtfloatx4));
      }

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
      for (n = 0; n < (cols1>>2); n++)
      {
        AE_LSX2X2_IP(x00,x01, px0,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x10,x11, px1,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x20,x21, px2,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x30,x31, px3,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x40,x41, px4,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x50,x51, px5,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x60,x61, px6,sizeof(xtfloatx4));
        AE_LSX2X2_IP(x70,x71, px7,sizeof(xtfloatx4));
        AE_LSX2X2_IP(y0,y1, py,sizeof(xtfloatx4));

        MADD_SX2X2(acc00,acc01,x00,x01,y0,y1);
        MADD_SX2X2(acc10,acc11,x10,x11,y0,y1);
        MADD_SX2X2(acc20,acc21,x20,x21,y0,y1);
        MADD_SX2X2(acc30,acc31,x30,x31,y0,y1);
        MADD_SX2X2(acc40,acc41,x40,x41,y0,y1);
        MADD_SX2X2(acc50,acc51,x50,x51,y0,y1);
        MADD_SX2X2(acc60,acc61,x60,x61,y0,y1);
        MADD_SX2X2(acc70,acc71,x70,x71,y0,y1);

      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;

      acc20 = acc20 + acc21;
      acc30 = acc30 + acc31;
      y0 = XT_SEL32_HL_SX2(acc20, acc30);
      y1 = XT_SEL32_LH_SX2(acc20, acc30);
      z1 = y0 + y1;

      acc40 = acc40 + acc41;
      acc50 = acc50 + acc51;
      y0 = XT_SEL32_HL_SX2(acc40, acc50);
      y1 = XT_SEL32_LH_SX2(acc40, acc50);
      z2 = y0 + y1;

      acc60 = acc60 + acc61;
      acc70 = acc70 + acc71;
      y0 = XT_SEL32_HL_SX2(acc60, acc70);
      y1 = XT_SEL32_LH_SX2(acc60, acc70);
      z3 = y0 + y1;


      acc00 = acc01 = acc10 = acc11 =
      acc20 = acc21 = acc30 = acc31 =  (xtfloatx2)0.0f;

      acc40 = acc41 = acc50 = acc51 =
      acc60 = acc61 = acc70 = acc71 =  (xtfloatx2)0.0f;
#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
#pragma no_unroll
      for (k = 0; k < (cols2>>2); k++)
      {
        AE_LSX2X2_IP(v00,v01, pv0,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v10,v11, pv1,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v20,v21, pv2,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v30,v31, pv3,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v40,v41, pv4,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v50,v51, pv5,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v60,v61, pv6,sizeof(xtfloatx4));
        AE_LSX2X2_IP(v70,v71, pv7,sizeof(xtfloatx4));
        AE_LSX2X2_IP(w0,w1, pw,sizeof(xtfloatx4));
        MADD_SX2X2(acc00,acc01,v00,v01,w0,w1);
        MADD_SX2X2(acc10,acc11,v10,v11,w0,w1);
        MADD_SX2X2(acc20,acc21,v20,v21,w0,w1);
        MADD_SX2X2(acc30,acc31,v30,v31,w0,w1);
        MADD_SX2X2(acc40,acc41,v40,v41,w0,w1);
        MADD_SX2X2(acc50,acc51,v50,v51,w0,w1);
        MADD_SX2X2(acc60,acc61,v60,v61,w0,w1);
        MADD_SX2X2(acc70,acc71,v70,v71,w0,w1);
      }
      acc00 = acc00 + acc01;
      acc10 = acc10 + acc11;
      w0 = XT_SEL32_HL_SX2(acc00, acc10);
      w1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = z0 + w0;
      z0 = z0 + w1;
      z0 = z0 + b0;

      acc20 = acc20 + acc21;
      acc30 = acc30 + acc31;
      w0 = XT_SEL32_HL_SX2(acc20, acc30);
      w1 = XT_SEL32_LH_SX2(acc20, acc30);
      z1 = z1 + w0;
      z1 = z1 + w1;
      z1 = z1 + b1;

      acc40 = acc40 + acc41;
      acc50 = acc50 + acc51;
      w0 = XT_SEL32_HL_SX2(acc40, acc50);
      w1 = XT_SEL32_LH_SX2(acc40, acc50);
      z2 = z2 + w0;
      z2 = z2 + w1;
      z2 = z2 + b2;

      acc60 = acc60 + acc61;
      acc70 = acc70 + acc71;
      w0 = XT_SEL32_HL_SX2(acc60, acc70);
      w1 = XT_SEL32_LH_SX2(acc60, acc70);
      z3 = z3 + w0;
      z3 = z3 + w1;
      z3 = z3 + b3;

      AE_SSX2X2_IP(z0, z1, pz,sizeof(xtfloatx4));
      AE_SSX2X2_IP(z2, z3, pz,sizeof(xtfloatx4));
    }

    /* Compute last (rows%8) output element */
    for (m = rows&(~7); m < rows; m++)
    {
      px0 = (const xtfloatx4 *)(x+m*row_stride1);
      py  = (const xtfloatx4 *)(y);
      pz_ = (xtfloat *)(z+m);
      pv0 = (const xtfloatx4 *)(v+m*row_stride2);
      pw  = (const xtfloatx4 *)(w);

      b0 = 0.0f;
      if(b != NULL)
      {
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
#pragma no_unroll
      for (n = 0; n < (cols1>>2); n++)
      {
        AE_LSX2X2_IP(x00,x01, px0,sizeof(xtfloatx4));
        AE_LSX2X2_IP(y0,y1, py,sizeof(xtfloatx4));
        MADD_SX2X2(acc00,acc01,x00,x01,y0,y1);
      }
      acc00 = acc00 + acc01;

      acc20 = acc21 = (xtfloatx2)0.0f;
#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
#pragma no_unroll
      for (k = 0; k < (cols2>>2); k++)
      {
        AE_LSX2X2_IP(v00,v01, pv0,sizeof(xtfloatx4));
        AE_LSX2X2_IP(w0,w1, pw,sizeof(xtfloatx4));
        MADD_SX2X2(acc20,acc21,v00,v01,w0,w1);
      }
      acc20 = acc20 + acc21;
      acc00 = acc00 + acc20;

      z0_ = XT_RADD_SX2(acc00);
      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
  //else if ((x != NULL) && (y != NULL))
  else if (cols1 > 0)                       // Calculate z = (x*y) + b
  {
    /* Compute 4 Rows at a time */
    for (m = 0; m < (rows>>2); m++)
    {
      px0 = (const xtfloatx4 *)(x+(4*m*row_stride1));
      px1 = (const xtfloatx4 *)((FLOAT32 *)px0+row_stride1);
      px2 = (const xtfloatx4 *)((FLOAT32 *)px1+row_stride1);
      px3 = (const xtfloatx4 *)((FLOAT32 *)px2+row_stride1);

      py  = (const xtfloatx4 *)(y);

      acc00 = acc01 = acc02 = acc03 = 
      acc10 = acc11 = acc12 = acc13 = 
      acc20 = acc21 = acc22 = acc23 = 
      acc30 = acc31 = acc32 = acc33 = (xtfloatx2)0.0f;

      acc40 = acc41 = acc50 = acc51 =
      acc60 = acc61 = acc70 = acc71 = (xtfloatx2)0.0f;

      b0 = b1 = (xtfloatx2)0.0f;
      if(b != NULL)
      {
        AE_LSX2X2_IP(b0,b1, pb,sizeof(xtfloatx4));
      }

      /* Compute for 8 colums per row, i.e. 4x8 * 8x4 */ 
      for (n = 0; n < (cols1>>3); n++)
      {
        AE_LSX2X2_I(x02, x03, px0, sizeof(xtfloatx4));  AE_LSX2X2_IP(x00, x01, px0, 2*sizeof(xtfloatx4));
        AE_LSX2X2_I(x12, x13, px1, sizeof(xtfloatx4));  AE_LSX2X2_IP(x10, x11, px1, 2*sizeof(xtfloatx4));
        AE_LSX2X2_I(x22, x23, px2, sizeof(xtfloatx4));  AE_LSX2X2_IP(x20, x21, px2, 2*sizeof(xtfloatx4));
        AE_LSX2X2_I(x32, x33, px3, sizeof(xtfloatx4));  AE_LSX2X2_IP(x30, x31, px3, 2*sizeof(xtfloatx4));
        AE_LSX2X2_I( y2,  y3,  py, sizeof(xtfloatx4));  AE_LSX2X2_IP( y0,  y1,  py, 2*sizeof(xtfloatx4));

        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
        MADD_SX2X2(acc10, acc11, x10, x11, y0, y1);
        MADD_SX2X2(acc20, acc21, x20, x21, y0, y1);
        MADD_SX2X2(acc30, acc31, x30, x31, y0, y1);

        MADD_SX2X2(acc02, acc03, x02, x03, y2, y3);
        MADD_SX2X2(acc12, acc13, x12, x13, y2, y3);
        MADD_SX2X2(acc22, acc23, x22, x23, y2, y3);
        MADD_SX2X2(acc32, acc33, x32, x33, y2, y3);
      }

      /* Compute for remaining cols1
       * Note : cols1 is a multiple of 4, this is a pre-requisite.
       *        So, if cols1%8 != 0,
       *        then remaining columns are exactly 4
       */
      if( (unsigned int)cols1 & 7 )
      {
        AE_LSX2X2_IP(x00, x01, px0, sizeof(xtfloatx4));
        AE_LSX2X2_IP(x10, x11, px1, sizeof(xtfloatx4));
        AE_LSX2X2_IP(x20, x21, px2, sizeof(xtfloatx4));
        AE_LSX2X2_IP(x30, x31, px3, sizeof(xtfloatx4));

        AE_LSX2X2_IP( y0,  y1,  py, sizeof(xtfloatx4));

        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
        MADD_SX2X2(acc10, acc11, x10, x11, y0, y1);
        MADD_SX2X2(acc20, acc21, x20, x21, y0, y1);
        MADD_SX2X2(acc30, acc31, x30, x31, y0, y1);
      }

      // z0.H = Sum of all elements in acc0X
      // z0.L = Sum of all elements in acc1X
      acc00 = acc00 + acc01 + acc02 + acc03;
      acc10 = acc10 + acc11 + acc12 + acc13;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;
      z0 = z0 + b0;

      // z1.H = Sum of all elements in acc2X
      // z1.L = Sum of all elements in acc3X
      acc20 = acc20 + acc21 + acc22 + acc23;
      acc30 = acc30 + acc31 + acc32 + acc33;
      y0 = XT_SEL32_HL_SX2(acc20, acc30);
      y1 = XT_SEL32_LH_SX2(acc20, acc30);
      z1 = y0 + y1;
      z1 = z1 + b1;

      AE_SSX2X2_IP(z0, z1, pz, sizeof(xtfloatx4));
    }

    /* Compute remaining rows */
    for (m = (rows&(~3)); m < rows; m++)
    {
      px0 = (const xtfloatx4 *)(x+m*row_stride1);
      py  = (const xtfloatx4 *)(y);
      pz_ = (xtfloat *)(z+m);

      b0 = 0.0f;
      if(b != NULL)
      {
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

#if defined(ENABLE_PRAGMA)
      __Pragma("loop_count min=1")
#endif /* ENABLE_PRAGMA */
#pragma no_unroll
      for (n = 0; n < (cols1>>2); n++)
      {
        AE_LSX2X2_IP(x00, x01, px0, sizeof(xtfloatx4));
        AE_LSX2X2_IP(y0, y1, py, sizeof(xtfloatx4));
        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
      }
      acc00 = acc00 + acc01;

      z0_ = XT_RADD_SX2(acc00);
      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
  else
  {
    return -1;
  }
} /* dual_mtx_vecmpyf_bias_add() */

void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloat*   px0
    ,xtfloat*   py
    ,WORD32     cols1
    ,WORD32     row_stride1
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  int align_offset = ((unsigned int)px0 & 0xf);

  pre_loop_count = (16 - align_offset) >> 2;
  pre_loop_count = (cols1 < pre_loop_count) ? cols1 : pre_loop_count;
  loop_count = (cols1 <= pre_loop_count)? 0 : (cols1 - pre_loop_count);
  post_loop_count = loop_count ? (loop_count & 7) : 0;
  loop_count >>= 3;

  xtfloatx2 acc00, acc01, acc02, acc03;
  xtfloatx2 acc10, acc11, acc12, acc13;
  xtfloatx2 acc20, acc21, acc22, acc23;
  xtfloatx2 acc30, acc31, acc32, acc33;
  xtfloatx2 x00, x01, x02, x03;
  xtfloatx2 x10, x11, x12, x13;
  xtfloatx2 x20, x21, x22, x23;
  xtfloatx2 x30, x31, x32, x33;
  xtfloatx2 y0, y1, y2, y3;

  xtfloat* px1 = px0 + 4*row_stride1; //next 4th row 
  xtfloat* px2 = px1 + 4*row_stride1; //next 4th row
  xtfloat* px3 = px2 + 4*row_stride1; //next 4th row 
  
  xtfloatx2 z0 = *out_0_0;
  xtfloatx2 z1 = *out_1_0;

  /* Pre loop computation */
  acc00 = 0.0f;
  acc20 = 0.0f;
  int k;

#pragma loop_count min=1
  for(k = 0; k < pre_loop_count; k++, px0++, px1++, px2++, px3++, py++)
  {
      x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(px0)), (xtfloatx2)(*(px1)));
      x20 = XT_SEL32_LL_SX2((xtfloatx2)(*(px2)), (xtfloatx2)(*(px3)));
      y0 = (xtfloatx2)(*(py));
      MADDQ_S(acc00, acc20, x00, x20, y0);
  }
  z0 = z0 + acc00;
  z1 = z1 + acc20;

  acc00 = acc01 = acc10 = acc11 = 0.0f;
  acc20 = acc21 = acc30 = acc31 = 0.0f;
  acc02 = acc03 = acc12 = acc13 = 0.0f;
  acc22 = acc23 = acc32 = acc33 = 0.0f;
  ae_valignx2 y_a = AE_LA128_PP(py);

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_LASX2X2_IP(y0, y1, y_a, (xtfloatx4 *)py);
    AE_LASX2X2_IP(y2, y3, y_a, (xtfloatx4 *)py);
    
    AE_LSX2X2_I(x02, x03, (xtfloatx4 *)px0, 16); AE_LSX2X2_IP(x00, x01, (xtfloatx4 *)px0, 32);
    AE_LSX2X2_I(x12, x13, (xtfloatx4 *)px1, 16); AE_LSX2X2_IP(x10, x11, (xtfloatx4 *)px1, 32);
    AE_LSX2X2_I(x22, x23, (xtfloatx4 *)px2, 16); AE_LSX2X2_IP(x20, x21, (xtfloatx4 *)px2, 32);
    AE_LSX2X2_I(x32, x33, (xtfloatx4 *)px3, 16); AE_LSX2X2_IP(x30, x31, (xtfloatx4 *)px3, 32);

    MADD_SX2X2(acc00,acc01,x00,x01,y0,y1);
    MADD_SX2X2(acc10,acc11,x10,x11,y0,y1);
    MADD_SX2X2(acc20,acc21,x20,x21,y0,y1);
    MADD_SX2X2(acc30,acc31,x30,x31,y0,y1);
    
    MADD_SX2X2(acc02,acc03,x02,x03,y2,y3);
    MADD_SX2X2(acc12,acc13,x12,x13,y2,y3);
    MADD_SX2X2(acc22,acc23,x22,x23,y2,y3);
    MADD_SX2X2(acc32,acc33,x32,x33,y2,y3);
  }
  acc00 = acc00 + acc01 + acc02 + acc03;
  acc10 = acc10 + acc11 + acc12 + acc13;
  y0 = XT_SEL32_HL_SX2(acc00, acc10);
  y1 = XT_SEL32_LH_SX2(acc00, acc10);
  z0 = z0 + y0;
  z0 = z0 + y1;

  acc20 = acc20 + acc21 + acc22 + acc23;
  acc30 = acc30 + acc31 + acc32 + acc33;
  y0 = XT_SEL32_HL_SX2(acc20, acc30);
  y1 = XT_SEL32_LH_SX2(acc20, acc30);
  z1 = z1 + y0;
  z1 = z1 + y1;

  //Remainder loop for cols1
  acc00 = 0.0f;
  acc20 = 0.0f;
  for(k = 0; k < post_loop_count; k++, px0++, px1++, px2++, px3++, py++)
  {
      x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(px0)), (xtfloatx2)(*(px1)));
      x20 = XT_SEL32_LL_SX2((xtfloatx2)(*(px2)), (xtfloatx2)(*(px3)));
      y0 = (xtfloatx2)(*(py));
      MADDQ_S(acc00, acc20, x00, x20, y0);
  }
  z0 = z0 + acc00;
  z1 = z1 + acc20;

  *out_0_0 = z0;
  *out_1_0 = z1;
}

WORD32 static dual_mtx_vecmpyf_bias_add_generic( FLOAT32 * z,
     const FLOAT32 * x,  const FLOAT32 * y, const FLOAT32 * v, const FLOAT32 * w,
     const FLOAT32 * b, int rows, int cols1, int cols2, int row_stride1, int row_stride2 )
{
  const xtfloatx4 *restrict px0;
  const xtfloatx4 *restrict px1;
  const xtfloatx4 *restrict px2;
  const xtfloatx4 *restrict pv0;
  const xtfloatx4 *restrict pv1;
  const xtfloatx4 *restrict pv2;
  const xtfloatx4 *restrict py;
  const xtfloatx4 *restrict pw;
  const xtfloat   *restrict pb;
        xtfloatx2 *restrict pz;
        xtfloat *restrict pz_;
  xtfloatx2 b0, b1;
  xtfloatx2 y0, y1;
  xtfloatx2 y2, y3;
  xtfloatx2 w0, w1;
  xtfloatx2 w2, w3;
  xtfloatx2 z0, z1;
  xtfloat z0_, b0_;
  xtfloatx2 x00, x01, x10, x11, x20, x21;
  xtfloatx2 x02, x03, x12, x13, x22, x23;
  xtfloatx2 v00, v01, v10, v11, v20, v21;
  xtfloatx2 v02, v03, v12, v13, v22, v23;
  xtfloatx2 acc00, acc01, acc10, acc11, acc20, acc21;
  xtfloatx2 acc02, acc03, acc12, acc13, acc22, acc23;
  ae_valignx2 x0_a, x1_a, x2_a, y_a;
  ae_valignx2 v0_a, v1_a, v2_a, w_a;
  int m=0, n, k;

  NASSERT(x);
  NASSERT(y);
  NASSERT(v);
  NASSERT(w);
  NASSERT(z);
  NASSERT((z != x) && (z != y) && (z != v) && (z != w) && (z != b));
  NASSERT_ALIGN(x,4);
  NASSERT_ALIGN(y,4);
  NASSERT_ALIGN(v,4);
  NASSERT_ALIGN(w,4);
  NASSERT_ALIGN(z,4);
  NASSERT_ALIGN(b,4);

  pz = (xtfloatx2 *)z;
  pb = (const xtfloat *)(b);

  if (y && w && (cols1 > 0))
  {
    int ii;
    for(m = 0; m < (rows & ~(16 - 1)); m += 16)
    {
      xtfloat* pz0 = (xtfloat*)((xtfloat *)pz + m);
      for(ii = 0; ii < 4; ii++)
      {
        /* Init out registers with bias */
        z0 = z1 = (xtfloatx2)0.0f;
        if(b != NULL)
        {
          z0 = XT_SEL32_LL_SX2((xtfloatx2)(pb[m+ii+0]), (xtfloatx2)(pb[m+ii+4]));
          z1 = XT_SEL32_LL_SX2((xtfloatx2)(pb[m+ii+8]), (xtfloatx2)(pb[m+ii+12]));
        }
        
        xtfloat *px = ((xtfloat *)x+((m+ii)*row_stride1));
        xtfloat *pvec1  = (xtfloat *)(y);
        xtfloat *pv = ((xtfloat *)v+((m+ii)*row_stride2));
        xtfloat *pvec2  = (xtfloat *)(w);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&z0
          ,&z1
          ,(xtfloat *)px
          ,(xtfloat *)pvec1
          ,cols1
          ,row_stride1
          );
        
        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&z0
          ,&z1
          ,(xtfloat *)pv
          ,(xtfloat *)pvec2
          ,cols2
          ,row_stride2
          );
        
        AE_SSX(z0, pz0, 16);
        AE_SSX(XT_SEL32_HH_SX2(z1,z1), pz0, 32);
        AE_SSX(z1, pz0, 48);
        AE_SSIP(XT_SEL32_HH_SX2(z0,z0), pz0, 4);
      }
    }
    xtfloat* pz0 = (xtfloat*)((xtfloat *)pz + m);
    int row_count = (rows - m) - (rows - m) % 3;
    for(ii = 0; ii < row_count; ii+= 3, m += 3)
    {
      px0 = (const xtfloatx4 *)(x+(m*row_stride1));
      pv0 = (const xtfloatx4 *)(v+(m*row_stride2));

      px1 = (const xtfloatx4 *)((FLOAT32 *)px0+row_stride1);
      px2 = (const xtfloatx4 *)((FLOAT32 *)px1+row_stride1);
      py  = (const xtfloatx4 *)(y);
      pv1 = (const xtfloatx4 *)((FLOAT32 *)pv0+row_stride2);
      pv2 = (const xtfloatx4 *)((FLOAT32 *)pv1+row_stride2);
      pw  = (const xtfloatx4 *)(w);

      x0_a = AE_LA128_PP(px0);
      x1_a = AE_LA128_PP(px1);
      x2_a = AE_LA128_PP(px2);
      y_a = AE_LA128_PP(py);

      acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = (xtfloatx2)0.0f;
      acc02 = acc03 = acc12 = acc13 = acc22 = acc23 = (xtfloatx2)0.0f;

      b0 = b1 = (xtfloatx2)0.0f;
      if(b != NULL)
      {
        b0 = XT_SEL32_LL_SX2((xtfloatx2)(pb[m+0]), (xtfloatx2)(pb[m+1]));
        b1 = (xtfloatx2)pb[m+2];
      }
      for (n = 0; n < (cols1>>3); n++)
      {
        AE_LASX2X2_IP(x00, x01, x0_a, px0);
        AE_LASX2X2_IP(x10, x11, x1_a, px1);
        AE_LASX2X2_IP(x20, x21, x2_a, px2);

        AE_LASX2X2_IP(y0, y1, y_a, py);

        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
        MADD_SX2X2(acc10, acc11, x10, x11, y0, y1);
        MADD_SX2X2(acc20, acc21, x20, x21, y0, y1);
        
        AE_LASX2X2_IP(x02, x03, x0_a, px0);
        AE_LASX2X2_IP(x12, x13, x1_a, px1);
        AE_LASX2X2_IP(x22, x23, x2_a, px2);

        AE_LASX2X2_IP(y2, y3, y_a, py);

        MADD_SX2X2(acc02, acc03, x02, x03, y2, y3);
        MADD_SX2X2(acc12, acc13, x12, x13, y2, y3);
        MADD_SX2X2(acc22, acc23, x22, x23, y2, y3);
      }
      acc00 = acc00 + acc01 + acc02 + acc03;
      acc10 = acc10 + acc11 + acc12 + acc13;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;

      acc20 = acc20 + acc21 + acc22 + acc23;
      z1 = RADD_SX2(acc20);

      acc00 = 0.0f;
      acc20 = 0.0f;
      for(k = 0; k < (cols1&7); k++)
      {
          x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(((xtfloat *)px0)+k)), (xtfloatx2)(*(((xtfloat *)px1)+k)));
          x20 = (xtfloatx2)(*(((xtfloat *)px2)+k));
          y0 = (xtfloatx2)(*((xtfloat *)py+k));
          MADDQ_S(acc00, acc20, x00, x20, y0);
      }
      z0 = z0 + acc00;
      z1 = z1 + acc20;

      v0_a = AE_LA128_PP(pv0);
      v1_a = AE_LA128_PP(pv1);
      v2_a = AE_LA128_PP(pv2);
      w_a = AE_LA128_PP(pw);

      acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = (xtfloatx2)0.0f;
      acc02 = acc03 = acc12 = acc13 = acc22 = acc23 = (xtfloatx2)0.0f;

      for (k = 0; k < (cols2>>3); k++)
      {
        AE_LASX2X2_IP(v00, v01, v0_a, pv0);
        AE_LASX2X2_IP(v10, v11, v1_a, pv1);
        AE_LASX2X2_IP(v20, v21, v2_a, pv2);

        AE_LASX2X2_IP(w0, w1, w_a, pw);

        MADD_SX2X2(acc00, acc01, v00, v01, w0, w1);
        MADD_SX2X2(acc10, acc11, v10, v11, w0, w1);
        MADD_SX2X2(acc20, acc21, v20, v21, w0, w1);
        
        AE_LASX2X2_IP(v02, v03, v0_a, pv0);
        AE_LASX2X2_IP(v12, v13, v1_a, pv1);
        AE_LASX2X2_IP(v22, v23, v2_a, pv2);

        AE_LASX2X2_IP(w2, w3, w_a, pw);

        MADD_SX2X2(acc02, acc03, v02, v03, w2, w3);
        MADD_SX2X2(acc12, acc13, v12, v13, w2, w3);
        MADD_SX2X2(acc22, acc23, v22, v23, w2, w3);
      }
      acc00 = acc00 + acc01 + acc02 + acc03;
      acc10 = acc10 + acc11 + acc12 + acc13;
      w0 = XT_SEL32_HL_SX2(acc00, acc10);
      w1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = z0 + w0;
      z0 = z0 + w1;

      acc20 = acc20 + acc21 + acc22 + acc23;
      w0 = RADD_SX2(acc20);
      z1 = z1 + w0;
      
      acc00 = 0.0f;
      acc20 = 0.0f;
      for(k = 0; k < (cols2&7); k++)
      {
          v00 = XT_SEL32_LL_SX2((xtfloatx2)(*(((xtfloat *)pv0)+k)), (xtfloatx2)(*(((xtfloat *)pv1)+k)));
          v20 = (xtfloatx2)(*(((xtfloat *)pv2)+k));
          w0 = (xtfloatx2)(*((xtfloat *)pw+k));
          MADDQ_S(acc00, acc20, v00, v20, w0);
      }
      z0 = z0 + acc00;
      z1 = z1 + acc20;

      /* Add bias */
      z0 = z0 + b0;
      z1 = z1 + b1;

      AE_SSIP(XT_SEL32_HH_SX2(z0,z0), (xtfloat *)pz0, 4);
      AE_SSIP(z0, (xtfloat *)pz0, 4);
      AE_SSIP(XT_SEL32_HH_SX2(z1,z1), (xtfloat *)pz0, 4);
    }

    /* Compute last (rows%3) output element */
    for (; m < rows; m++)
    {
      pz_ = (xtfloat*)((xtfloat *)pz + m);
      px0 = (const xtfloatx4 *)(x+m*row_stride1);
      py  = (const xtfloatx4 *)(y);
      pv0 = (const xtfloatx4 *)(v+m*row_stride2);
      pw  = (const xtfloatx4 *)(w);

      x0_a = AE_LA128_PP(px0);
      y_a = AE_LA128_PP(py);

      b0_ = 0.0f;
      if(b != NULL)
      {
        b0_ = b[m];
      }
      acc00 = acc01 = (xtfloatx2)0.0f;

      for (n = 0; n < (cols1>>2); n++)
      {
        AE_LASX2X2_IP(x00, x01, x0_a, px0);
        AE_LASX2X2_IP(y0, y1, y_a, py);

        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
      }
      acc00 = acc00 + acc01;
      z0_ = RADD_SX2(acc00);

      for(n = 0; n < (cols1&3); n++)
      {
          XT_MADD_S(z0_, *(((xtfloat *)px0)+n), *(((xtfloat *)py)+n));
      }

      v0_a = AE_LA128_PP(pv0);
      w_a = AE_LA128_PP(pw);
      acc00 = acc01 = (xtfloatx2)0.0f;
      for (k = 0; k < (cols2>>2); k++)
      {
        AE_LASX2X2_IP(v00, v01, v0_a, pv0);
        AE_LASX2X2_IP(w0, w1, w_a, pw);

        MADD_SX2X2(acc00, acc01, v00, v01, w0, w1);
      }
      acc00 = acc00 + acc01;
      z0_ = z0_ + XT_RADD_SX2(acc00);

      for(n = 0; n < (cols2&3); n++)
      {
          XT_MADD_S(z0_, *(((xtfloat *)pv0)+n), *(((xtfloat *)pw)+n));
      }

      z0_ = z0_ + b0_;

      AE_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
  else
  {
    int ii;
    for(m = 0; m < (rows & ~(16 - 1)); m += 16)
    {
      xtfloat* pz0 = (xtfloat*)((xtfloat *)pz + m);
      for(ii = 0; ii < 4; ii++)
      {
        /* Init out registers with bias */
        z0 = z1 = (xtfloatx2)0.0f;
        if(b != NULL)
        {
          z0 = XT_SEL32_LL_SX2((xtfloatx2)(pb[m+ii+0]), (xtfloatx2)(pb[m+ii+4]));
          z1 = XT_SEL32_LL_SX2((xtfloatx2)(pb[m+ii+8]), (xtfloatx2)(pb[m+ii+12]));
        }
        
        xtfloat *px = ((xtfloat *)x+((m+ii)*row_stride1));
        xtfloat *pvec1  = (xtfloat *)(y);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&z0
          ,&z1
          ,(xtfloat *)px
          ,(xtfloat *)pvec1
          ,cols1
          ,row_stride1
          );
        
        AE_SSX(XT_SEL32_LL_SX2(z0,z0), pz0, 16);
        AE_SSX(XT_SEL32_HH_SX2(z1,z1), pz0, 32);
        AE_SSX(XT_SEL32_LL_SX2(z1,z1), pz0, 48);
        AE_SSIP(XT_SEL32_HH_SX2(z0,z0), pz0, 4);
      }
    }
    xtfloat* pz0 = (xtfloat*)((xtfloat *)pz + m);
    int row_count = (rows - m) - (rows - m) % 3;
    for(ii = 0; ii < row_count; ii+= 3, m += 3)
    {
      px0 = (const xtfloatx4 *)(x+(m*row_stride1));

      px1 = (const xtfloatx4 *)((FLOAT32 *)px0+row_stride1);
      px2 = (const xtfloatx4 *)((FLOAT32 *)px1+row_stride1);
      py  = (const xtfloatx4 *)(y);

      x0_a = AE_LA128_PP(px0);
      x1_a = AE_LA128_PP(px1);
      x2_a = AE_LA128_PP(px2);
      y_a = AE_LA128_PP(py);

      acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = (xtfloatx2)0.0f;
      acc02 = acc03 = acc12 = acc13 = acc22 = acc23 = (xtfloatx2)0.0f;

      b0 = b1 = (xtfloatx2)0.0f;
      if(b != NULL)
      {
        b0 = XT_SEL32_LL_SX2((xtfloatx2)(pb[m+0]), (xtfloatx2)(pb[m+1]));
        b1 = (xtfloatx2)pb[m+2];
      }
      for (n = 0; n < (cols1>>3); n++)
      {
        AE_LASX2X2_IP(x00, x01, x0_a, px0);
        AE_LASX2X2_IP(x10, x11, x1_a, px1);
        AE_LASX2X2_IP(x20, x21, x2_a, px2);

        AE_LASX2X2_IP(y0, y1, y_a, py);

        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
        MADD_SX2X2(acc10, acc11, x10, x11, y0, y1);
        MADD_SX2X2(acc20, acc21, x20, x21, y0, y1);
        
        AE_LASX2X2_IP(x02, x03, x0_a, px0);
        AE_LASX2X2_IP(x12, x13, x1_a, px1);
        AE_LASX2X2_IP(x22, x23, x2_a, px2);

        AE_LASX2X2_IP(y2, y3, y_a, py);

        MADD_SX2X2(acc02, acc03, x02, x03, y2, y3);
        MADD_SX2X2(acc12, acc13, x12, x13, y2, y3);
        MADD_SX2X2(acc22, acc23, x22, x23, y2, y3);
      }
      acc00 = acc00 + acc01 + acc02 + acc03;
      acc10 = acc10 + acc11 + acc12 + acc13;
      y0 = XT_SEL32_HL_SX2(acc00, acc10);
      y1 = XT_SEL32_LH_SX2(acc00, acc10);
      z0 = y0 + y1;

      acc20 = acc20 + acc21 + acc22 + acc23;
      z1 = RADD_SX2(acc20);

      acc00 = 0.0f;
      acc20 = 0.0f;
      for(k = 0; k < (cols1&7); k++)
      {
          x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(((xtfloat *)px0)+k)), (xtfloatx2)(*(((xtfloat *)px1)+k)));
          x20 = (xtfloatx2)(*(((xtfloat *)px2)+k));
          y0 = (xtfloatx2)(*((xtfloat *)py+k));
          MADDQ_S(acc00, acc20, x00, x20, y0);
      }
      z0 = z0 + acc00;
      z1 = z1 + acc20;

      /* Add bias */
      z0 = z0 + b0;
      z1 = z1 + b1;

      AE_SSIP(XT_SEL32_HH_SX2(z0,z0), (xtfloat *)pz0, 4);
      AE_SSIP(z0, (xtfloat *)pz0, 4);
      AE_SSIP(XT_SEL32_HH_SX2(z1,z1), (xtfloat *)pz0, 4);
    }

    /* Compute last (rows%3) output element */
    pz_ = (xtfloat*)((xtfloat *)pz + m);
    for (; m < rows; m++)
    {
      px0 = (const xtfloatx4 *)(x+m*row_stride1);
      py  = (const xtfloatx4 *)(y);

      x0_a = AE_LA128_PP(px0);
      y_a = AE_LA128_PP(py);

      b0_ = 0.0f;
      if(b != NULL)
      {
        b0_ = b[m];
      }
      acc00 = acc01 = acc02 = acc03 =(xtfloatx2)0.0f;

      for (n = 0; n < (cols1>>3); n++)
      {
        AE_LASX2X2_IP(x00, x01, x0_a, px0);
        AE_LASX2X2_IP(y0, y1, y_a, py);
        AE_LASX2X2_IP(x02, x03, x0_a, px0);
        AE_LASX2X2_IP(y2, y3, y_a, py);

        MADD_SX2X2(acc00, acc01, x00, x01, y0, y1);
        MADD_SX2X2(acc02, acc03, x02, x03, y2, y3);
      }
      acc00 = acc00 + acc01 + acc02 + acc03;
      z0_ = XT_RADD_SX2(acc00);

      for(n = 0; n < (cols1&7); n++)
      {
          XT_MADD_S(z0_, *(((xtfloat *)px0)+n), *(((xtfloat *)py)+n));
      }

      z0_ = z0_ + b0_;

      XT_SSIP(z0_, pz_, sizeof(FLOAT32));
    }
    return 0;
  }
} /* dual_mtx_vecmpyf_bias_add_generic() */

#endif /* HAVE_VFPU */


/*-------------------------------------------------------------------------
  xa_nn_matXvec_f32xf32_f32_sigmoid
  This function computes the sigmoid operated over dual matrix vector
  multiplication with added bias vector value (the most fundamental DNN
  operation). The inputs and output are all 32 bit float numbers.

  Precision:
  f32xf32_f32  32-bit float inputs, 32-bit float output.

  Input:
  p_mat1         first matrix pointer,                32-bit float
  p_mat2         second matrix pointer,               32-bit float
  p_vec1         first vector pointer,                32-bit float
  p_vec2         second vector pointer,               32-bit float
  p_bias         bias vector pointer,                 32-bit float
  rows           number of rows,                      32 bit integer
  cols1          number of columns of first matrix,   32 bit integer
  cols2          number of columns of second matrix,  32 bit integer
  row_stride1    row offset of first matrix,          32 bit integer
  row_stride2    row offset of second matrix,         32 bit integer
  p_scratch      intermediate scratch vector pointer, 32-bit float
  Output:
  p_out          result vector pointer,               32-bit float

  Restriction:
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should hold
  valid addresses in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should not
  overlap in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should be 8 byte
  boundaries aligned in the memory space
  cols1, cols2, row_stride1, row_stride2 should be multiple of 4
-------------------------------------------------------------------------*/
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f32xf32_f32_sigmoid,(
    FLOAT32  *  p_out,
    FLOAT32  *  p_mat1,
    FLOAT32  *  p_mat2,
    FLOAT32  *  p_vec1,
    FLOAT32  *  p_vec2,
    FLOAT32  *  p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch))
#else
WORD32  xa_nn_matXvec_f32xf32_f32_sigmoid(
    FLOAT32  * __restrict__ p_out,
    FLOAT32  * __restrict__ p_mat1,
    FLOAT32  * __restrict__ p_mat2,
    FLOAT32  * __restrict__ p_vec1,
    FLOAT32  * __restrict__ p_vec2,
    FLOAT32  * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  WORD32 ret = 0, k;
  if(((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&3) == 0) && ((row_stride2&3) == 0) &&
     ((((unsigned)p_scratch)&15) == 0) && ((((unsigned)p_mat1)&15) == 0) && ((((unsigned)p_vec1)&15) == 0) &&
     ((((unsigned)p_mat2)&15) == 0) && ((((unsigned)p_vec2)&15) == 0) && ((((unsigned)p_bias)&15) == 0))
  {
    ret = dual_mtx_vecmpyf_bias_add(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }
  else
  {
    ret = dual_mtx_vecmpyf_bias_add_generic(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }

  if (0 == ret)
  {
    xa_nn_vec_sigmoid_f32_f32(p_out, p_scratch, rows);
  }
  else if (-1 == ret)
  {
    /* In erroneous case, populate output with zeros. */
    for (k = 0; k < rows; k++)
    {
      p_out[k] = 0.0f;
    }
  }

  return ret;
}
#endif /* !HAVE_VFPU */


/*-------------------------------------------------------------------------
  xa_nn_matXvec_f32xf32_f32_tanh
  This function computes the tanh operated over dual matrix vector
  multiplication with added bias vector value (the most fundamental DNN
  operation). The inputs and output are all 32 bit float numbers.

  Precision:
  f32xf32_f32  32-bit float inputs, 32-bit float output.

  Input:
  p_mat1         first matrix pointer,                32-bit float
  p_mat2         second matrix pointer,               32-bit float
  p_vec1         first vector pointer,                32-bit float
  p_vec2         second vector pointer,               32-bit float
  p_bias         bias vector pointer,                 32-bit float
  rows           number of rows,                      32 bit integer
  cols1          number of columns of first matrix,   32 bit integer
  cols2          number of columns of second matrix,  32 bit integer
  row_stride1    row offset of first matrix,          32 bit integer
  row_stride2    row offset of second matrix,         32 bit integer
  p_scratch      intermediate scratch vector pointer, 32-bit float
  Output:
  p_out          result vector pointer,               32-bit float

  Restriction:
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should hold
  valid addresses in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should not
  overlap in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias p_scratch should be 8 byte
  boundaries aligned in the memory space
  cols1, cols2, row_stride1, row_stride2 should be multiple of 4
-------------------------------------------------------------------------*/
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f32xf32_f32_tanh,(
    FLOAT32  *  p_out,
    FLOAT32  *  p_mat1,
    FLOAT32  *  p_mat2,
    FLOAT32  *  p_vec1,
    FLOAT32  *  p_vec2,
    FLOAT32  *  p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch))
#else
WORD32  xa_nn_matXvec_f32xf32_f32_tanh(
    FLOAT32  * __restrict__ p_out,
    FLOAT32  * __restrict__ p_mat1,
    FLOAT32  * __restrict__ p_mat2,
    FLOAT32  * __restrict__ p_vec1,
    FLOAT32  * __restrict__ p_vec2,
    FLOAT32  * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    FLOAT32  * __restrict__ p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND(((cols1&3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1&3) != 0), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND(((cols2&3) != 0), -1);
    XA_NNLIB_ARG_CHK_COND(((row_stride2&3) != 0), -1);
  }

  WORD32 ret = 0, k;
  if(((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&3) == 0) && ((row_stride2&3) == 0) &&
     ((((unsigned)p_scratch)&15) == 0) && ((((unsigned)p_mat1)&15) == 0) && ((((unsigned)p_vec1)&15) == 0) &&
     ((((unsigned)p_mat2)&15) == 0) && ((((unsigned)p_vec2)&15) == 0) && ((((unsigned)p_bias)&15) == 0))
  {
    ret = dual_mtx_vecmpyf_bias_add(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }
  else
  {
    ret = dual_mtx_vecmpyf_bias_add_generic(p_scratch, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }

  if (0 == ret)
  {
    xa_nn_vec_tanh_f32_f32(p_out, p_scratch, rows);
  }
  else if (-1 == ret)
  {
    /* In erroneous case, populate output with zeros. */
    for (k = 0; k < rows; k++)
    {
      p_out[k] = 0.0f;
    }
  }

  return ret;
}
#endif /* !HAVE_VFPU */


/*-------------------------------------------------------------------------
  xa_nn_matXvec_f32xf32_f32
  This function computes the dual matrix vector multiplication with added
  bias vector value (the most fundamental DNN operation). The inputs and
  output are all 32 bit float numbers.

  Precision:
  f32xf32_f32  32-bit float inputs, 32-bit float output.

  Input:
  p_mat1         first matrix pointer,                32-bit float
  p_mat2         second matrix pointer,               32-bit float
  p_vec1         first vector pointer,                32-bit float
  p_vec2         second vector pointer,               32-bit float
  p_bias         bias vector pointer,                 32-bit float
  rows           number of rows,                      32 bit integer
  cols1          number of columns of first matrix,   32 bit integer
  cols2          number of columns of second matrix,  32 bit integer
  row_stride1    row offset of first matrix,          32 bit integer
  row_stride2    row offset of second matrix,         32 bit integer
  Output:
  p_out          result vector pointer,               32-bit float

  Restriction:
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias should hold valid addresses
  in the memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias should not overlap in the
  memory space
  p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias should be 8 byte boundaries
  aligned in the memory space
  cols1, cols2, row_stride1, row_stride2 should be multiple of 4
-------------------------------------------------------------------------*/
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f32xf32_f32,(
    FLOAT32  *  p_out,
    const FLOAT32  *  p_mat1,
    const FLOAT32  *  p_mat2,
    const FLOAT32  *  p_vec1,
    const FLOAT32  *  p_vec2,
    const FLOAT32  *  p_bias,
    WORD32 rows, WORD32 cols1, WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2))
#else
WORD32  xa_nn_matXvec_f32xf32_f32(
    FLOAT32  * __restrict__ p_out,
    const FLOAT32  * __restrict__ p_mat1,
    const FLOAT32  * __restrict__ p_mat2,
    const FLOAT32  * __restrict__ p_vec1,
    const FLOAT32  * __restrict__ p_vec2,
    const FLOAT32  * __restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
  }

  WORD32 ret = 0, k;
  if(((cols1&3) == 0) && ((cols2&3) == 0) && ((row_stride1&3) == 0) && ((row_stride2&3) == 0) &&
     ((((unsigned)p_out)&15) == 0) && ((((unsigned)p_mat1)&15) == 0) && ((((unsigned)p_vec1)&15) == 0) &&
     ((((unsigned)p_mat2)&15) == 0) && ((((unsigned)p_vec2)&15) == 0) && ((((unsigned)p_bias)&15) == 0))
  {
    ret = dual_mtx_vecmpyf_bias_add(p_out, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }
  else
  {
    ret = dual_mtx_vecmpyf_bias_add_generic(p_out, p_mat1, p_vec1, p_mat2, p_vec2,
        p_bias, rows, cols1, cols2, row_stride1, row_stride2);
  }

  if (-1 == ret)
  {
    /* In erroneous case, populate output with zeros. */
    for (k = 0; k < rows; k++)
    {
      p_out[k] = 0.0f;
    }
  }

  return ret;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_dot_prod_f32xf32_f32,(
    FLOAT32  * p_out,
    FLOAT32  * p_inp1,
    FLOAT32  * p_inp2,
    WORD32 vec_length,
    WORD32 num_vecs))
#else
WORD32 xa_nn_dot_prod_f32xf32_f32(
         FLOAT32 * __restrict__ p_out,          /* pointer to output */
         const FLOAT32 * __restrict__ p_inp1,         /* pointer to input1 */
         const FLOAT32 * __restrict__ p_inp2,         /* pointer to input2 */
         WORD32 vec_length,
         WORD32 num_vecs)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((num_vecs <= 0), -1);

    xtfloatx4 *pt_inp1, *pt_inp2;
    float d_out;
    int i, j;

    if(((((unsigned)p_inp1)&15) == 0) && ((((unsigned)p_inp2)&15) == 0) && ((vec_length&3) == 0))
    {
        pt_inp1 = (xtfloatx4 *)p_inp1;
        pt_inp2 = (xtfloatx4 *)p_inp2;
        for(i = 0; i < num_vecs; i++)
        {
            xtfloatx2 d_acc0, d_acc1;
            d_acc0 = (xtfloatx2)0.0f;
            d_acc1 = (xtfloatx2)0.0f;

            for(j = 0; j < (vec_length>>2); j++)
            {
                xtfloatx2 d_inp1, d_inp2, d_inp3, d_inp4;
                AE_LSX2X2_IP(d_inp1, d_inp2, pt_inp1, 16);
                AE_LSX2X2_IP(d_inp3, d_inp4, pt_inp2, 16);
                MADD_SX2X2(d_acc0, d_acc1, d_inp1, d_inp2, d_inp3, d_inp4);
            }
            d_acc0 = XT_ADD_SX2(d_acc0, d_acc1);
            d_out = XT_ADD_S(XT_HIGH_S(d_acc0), XT_LOW_S(d_acc0));
            *(float *)(&p_out[i]) = d_out;
        }
    }
    else
    {
        ae_valignx2 inp1_a, inp2_a;

        for(i = 0; i < num_vecs; i++)
        {
            pt_inp1 = (xtfloatx4 *)(&p_inp1[i*vec_length]);
            pt_inp2 = (xtfloatx4 *)(&p_inp2[i*vec_length]);
            inp1_a = AE_LA128_PP(pt_inp1);
            inp2_a = AE_LA128_PP(pt_inp2);

            xtfloatx2 d_acc0, d_acc1;
            d_acc0 = (xtfloatx2)0.0f;
            d_acc1 = (xtfloatx2)0.0f;
            for(j = 0; j < (vec_length>>2); j++)
            {
                xtfloatx2 d_inp1, d_inp2, d_inp3, d_inp4;
                AE_LASX2X2_IP(d_inp1, d_inp2, inp1_a, pt_inp1);
                AE_LASX2X2_IP(d_inp3, d_inp4, inp2_a, pt_inp2);
                MADD_SX2X2(d_acc0, d_acc1, d_inp1, d_inp2, d_inp3, d_inp4);
            }
            d_acc0 = XT_ADD_SX2(d_acc0, d_acc1);
            d_out = XT_ADD_S(XT_HIGH_S(d_acc0), XT_LOW_S(d_acc0));

            float *pt_inp1u, *pt_inp2u;
            pt_inp1u = (float *)pt_inp1;
            pt_inp2u = (float *)pt_inp2;
            for(j = 0; j < (vec_length&3); j++)
            {
                XT_MADD_S(d_out, pt_inp1u[j], pt_inp2u[j]);
            }
            *(float *)(&p_out[i]) = d_out;
        }
    }

    return 0;
}
#endif /* #if !HAVE_VFPU */

#endif

