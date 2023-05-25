/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matmul_f32xf32_f32,(
    FLOAT32 * __restrict__ p_out,        
    const FLOAT32 * __restrict__ p_mat1, 
    const FLOAT32 * __restrict__ p_vec1, 
    const FLOAT32 * __restrict__ p_bias, 
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                   
    WORD32 vec_count,                     
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride))                      

#else
/* Using the 4 row 1 vec function defined in xa_nn_matXvec_f32.c for xa_nn_matXvec_f32() kernel */
extern void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloat*   px0
    ,xtfloat*   p_vec0
    ,WORD32     cols1
    ,WORD32     row_stride1
    );

static inline void spfunc_cols_mul4_out_offset1
    (xtfloat*   p_out
    ,const xtfloat*   p_mat1
    ,const xtfloat*   p_vec1
    ,const xtfloat*   p_bias
    ,WORD32     rows
    ,WORD32     vec_count
    ,WORD32     cols1
    ,WORD32     out_stride
    )
{
  int vec_itr, m_itr, c_itr;

  xtfloatx2 x00, x01, x10, x11;
  xtfloatx2 x20, x21, x30, x31;
  xtfloatx2 vec0_0, vec0_1;
  xtfloatx2 vec1_0, vec1_1;
  xtfloatx2 vec2_0, vec2_1;
  xtfloatx2 vec3_0, vec3_1;
  xtfloatx2 y0, y1, y2, y3;

  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xtfloat *p_out_0 = p_out + (vec_itr + 0) + (m_itr + 0)*out_stride;
      xtfloat *p_out_1 = p_out + (vec_itr + 0) + (m_itr + 1)*out_stride;
      xtfloat *p_out_2 = p_out + (vec_itr + 0) + (m_itr + 2)*out_stride;
      xtfloat *p_out_3 = p_out + (vec_itr + 0) + (m_itr + 3)*out_stride;
      xtfloat *px = (xtfloat *)(p_mat1+(m_itr*cols1));
      xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xtfloatx2 z0, z1, z2, z3;
      xtfloatx2 z4, z5, z6, z7;
      z0 = z1 = z2 = z3 = (xtfloatx2)0.0f;
      z4 = z5 = z6 = z7 = (xtfloatx2)0.0f;
      if(p_bias != NULL)
      {
        z1 = z0 = (xtfloatx2)(p_bias[m_itr+0]);
        z3 = z2 = (xtfloatx2)(p_bias[m_itr+1]);
        z5 = z4 = (xtfloatx2)(p_bias[m_itr+2]);
        z7 = z6 = (xtfloatx2)(p_bias[m_itr+3]);
      }
              
      xtfloatx2 acc_row0_vec0, acc_row0_vec1;
      xtfloatx2 acc_row1_vec0, acc_row1_vec1;
      xtfloatx2 acc_row2_vec0, acc_row2_vec1;
      xtfloatx2 acc_row3_vec0, acc_row3_vec1;
      xtfloatx2 acc_row0_vec2, acc_row0_vec3;
      xtfloatx2 acc_row1_vec2, acc_row1_vec3;
      xtfloatx2 acc_row2_vec2, acc_row2_vec3;
      xtfloatx2 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = 0.0f;
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = 0.0f;
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = 0.0f;
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = 0.0f;
  
#pragma loop_count min=1
      for(c_itr = 0; c_itr < cols1 >> 2; c_itr++, p_vec+=4, px+=4)
      {
        AE_LSX2X2_I(vec0_0, vec0_1, (xtfloatx4 *)p_vec, 0);
        AE_LSX2X2_X(vec1_0, vec1_1, (xtfloatx4 *)p_vec, sizeof(xtfloat)*cols1);
        AE_LSX2X2_X(vec2_0, vec2_1, (xtfloatx4 *)p_vec, sizeof(xtfloat)*2*cols1);
        AE_LSX2X2_X(vec3_0, vec3_1, (xtfloatx4 *)p_vec, sizeof(xtfloat)*3*cols1);
        
        AE_LSX2X2_I(x00, x01, (xtfloatx4 *)px, 0);
        AE_LSX2X2_X(x10, x11, (xtfloatx4 *)px, sizeof(xtfloat)*cols1);
        AE_LSX2X2_X(x20, x21, (xtfloatx4 *)px, sizeof(xtfloat)*2*cols1);
        AE_LSX2X2_X(x30, x31, (xtfloatx4 *)px, sizeof(xtfloat)*3*cols1);

        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        MADDQ_S(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
        MADDQ_S(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
        MADDQ_S(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
        MADDQ_S(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
        MADDQ_S(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
        MADDQ_S(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
        
        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
        MADDQ_S(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
        MADDQ_S(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
        MADDQ_S(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
        MADDQ_S(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
        MADDQ_S(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
        MADDQ_S(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      }
  
      y0 = XT_SEL32_HL_SX2(acc_row0_vec0, acc_row0_vec1);
      y1 = XT_SEL32_LH_SX2(acc_row0_vec0, acc_row0_vec1);
      z0 = z0 + y0;
      z0 = z0 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row0_vec2, acc_row0_vec3);
      y3 = XT_SEL32_LH_SX2(acc_row0_vec2, acc_row0_vec3);
      z1 = z1 + y2;
      z1 = z1 + y3;

      y0 = XT_SEL32_HL_SX2(acc_row1_vec0, acc_row1_vec1);
      y1 = XT_SEL32_LH_SX2(acc_row1_vec0, acc_row1_vec1);
      z2 = z2 + y0;
      z2 = z2 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row1_vec2, acc_row1_vec3);
      y3 = XT_SEL32_LH_SX2(acc_row1_vec2, acc_row1_vec3);
      z3 = z3 + y2;
      z3 = z3 + y3;

      y0 = XT_SEL32_HL_SX2(acc_row2_vec0, acc_row2_vec1);
      y1 = XT_SEL32_LH_SX2(acc_row2_vec0, acc_row2_vec1);
      z4 = z4 + y0;
      z4 = z4 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row2_vec2, acc_row2_vec3);
      y3 = XT_SEL32_LH_SX2(acc_row2_vec2, acc_row2_vec3);
      z5 = z5 + y2;
      z5 = z5 + y3;

      y0 = XT_SEL32_HL_SX2(acc_row3_vec0, acc_row3_vec1);
      y1 = XT_SEL32_LH_SX2(acc_row3_vec0, acc_row3_vec1);
      z6 = z6 + y0;
      z6 = z6 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row3_vec2, acc_row3_vec3);
      y3 = XT_SEL32_LH_SX2(acc_row3_vec2, acc_row3_vec3);
      z7 = z7 + y2;
      z7 = z7 + y3;

      AE_SSX2X2_IP(z0, z1, (xtfloatx4 *)p_out_0, 16);
      AE_SSX2X2_IP(z2, z3, (xtfloatx4 *)p_out_1, 16);
      AE_SSX2X2_IP(z4, z5, (xtfloatx4 *)p_out_2, 16);
      AE_SSX2X2_IP(z6, z7, (xtfloatx4 *)p_out_3, 16);
    }
  }
  for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xtfloat *p_out_0 = p_out + (vec_itr + 0) + (m_itr + 0)*out_stride;
      xtfloat *p_out_1 = p_out + (vec_itr + 0) + (m_itr + 1)*out_stride;
      xtfloat *p_out_2 = p_out + (vec_itr + 0) + (m_itr + 2)*out_stride;
      xtfloat *p_out_3 = p_out + (vec_itr + 0) + (m_itr + 3)*out_stride;
      xtfloat *px = (xtfloat *)(p_mat1+(m_itr*cols1));
      xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xtfloatx2 z0, z1;
      z0 = z1 = (xtfloatx2)0.0f;
      if(p_bias != NULL)
      {
        z0 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+0]), (xtfloatx2)(p_bias[m_itr+1]));
        z1 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+2]), (xtfloatx2)(p_bias[m_itr+3]));
      }
              
      xtfloatx2 acc_row0_vec0;
      xtfloatx2 acc_row1_vec0;
      xtfloatx2 acc_row2_vec0;
      xtfloatx2 acc_row3_vec0;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = 0.0f;
  
#pragma no_unroll
      for(c_itr = 0; c_itr < cols1 >> 2; c_itr++, p_vec+=4, px+=4)
      {
        AE_LSX2X2_I(vec0_0, vec0_1, (xtfloatx4 *)p_vec,  0);
        
        AE_LSX2X2_I(x00, x01, (xtfloatx4 *)px, 0);
        AE_LSX2X2_X(x10, x11, (xtfloatx4 *)px, sizeof(xtfloat)*cols1);
        AE_LSX2X2_X(x20, x21, (xtfloatx4 *)px, sizeof(xtfloat)*2*cols1);
        AE_LSX2X2_X(x30, x31, (xtfloatx4 *)px, sizeof(xtfloat)*3*cols1);

        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        
        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
      }
      acc_row0_vec0 = XT_RADD_SX2(acc_row0_vec0);
      acc_row1_vec0 = XT_RADD_SX2(acc_row1_vec0);
      acc_row2_vec0 = XT_RADD_SX2(acc_row2_vec0);
      acc_row3_vec0 = XT_RADD_SX2(acc_row3_vec0);
  
      AE_SSIP(acc_row0_vec0, p_out_0, 4);
      AE_SSIP(acc_row1_vec0, p_out_1, 4);
      AE_SSIP(acc_row2_vec0, p_out_2, 4);
      AE_SSIP(acc_row3_vec0, p_out_3, 4);
    }
  }

}

static inline void spfunc_cols_mul4_out_stride1
    (xtfloat*   p_out
    ,const xtfloat*   p_mat1
    ,const xtfloat*   p_vec1
    ,const xtfloat*   p_bias
    ,WORD32     rows
    ,WORD32     vec_count
    ,WORD32     cols1
    ,WORD32     out_offset
    )
{
  int vec_itr, m_itr, c_itr;

  xtfloatx2 x00, x01, x10, x11;
  xtfloatx2 x20, x21, x30, x31;
  xtfloatx2 vec0_0, vec0_1;
  xtfloatx2 vec1_0, vec1_1;
  xtfloatx2 vec2_0, vec2_1;
  xtfloatx2 vec3_0, vec3_1;
  xtfloatx2 y0, y1, y2, y3;

  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    xtfloat *p_out_0 = p_out + (vec_itr + 0)*out_offset;
    xtfloat *p_out_1 = p_out + (vec_itr + 1)*out_offset;
    xtfloat *p_out_2 = p_out + (vec_itr + 2)*out_offset;
    xtfloat *p_out_3 = p_out + (vec_itr + 3)*out_offset;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xtfloat *px = (xtfloat *)(p_mat1+(m_itr*cols1));
      xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xtfloatx2 z0, z1, z2, z3;
      xtfloatx2 z4, z5, z6, z7;
      z0 = z1 = z2 = z3 = (xtfloatx2)0.0f;
      z4 = z5 = z6 = z7 = (xtfloatx2)0.0f;
      if(p_bias != NULL)
      {
        z6 = z4 = z2 = z0 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+0]), (xtfloatx2)(p_bias[m_itr+1]));
        z7 = z5 = z3 = z1 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+2]), (xtfloatx2)(p_bias[m_itr+3]));
      }
              
      xtfloatx2 acc_row0_vec0, acc_row0_vec1;
      xtfloatx2 acc_row1_vec0, acc_row1_vec1;
      xtfloatx2 acc_row2_vec0, acc_row2_vec1;
      xtfloatx2 acc_row3_vec0, acc_row3_vec1;
      xtfloatx2 acc_row0_vec2, acc_row0_vec3;
      xtfloatx2 acc_row1_vec2, acc_row1_vec3;
      xtfloatx2 acc_row2_vec2, acc_row2_vec3;
      xtfloatx2 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = 0.0f;
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = 0.0f;
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = 0.0f;
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = 0.0f;
  
#pragma loop_count min=1
      for(c_itr = 0; c_itr < cols1 >> 2; c_itr++, p_vec+=4, px+=4)
      {
        AE_LSX2X2_I(vec0_0, vec0_1, (xtfloatx4 *)p_vec, 0);
        AE_LSX2X2_X(vec1_0, vec1_1, (xtfloatx4 *)p_vec, sizeof(xtfloat)*cols1);
        AE_LSX2X2_X(vec2_0, vec2_1, (xtfloatx4 *)p_vec, sizeof(xtfloat)*2*cols1);
        AE_LSX2X2_X(vec3_0, vec3_1, (xtfloatx4 *)p_vec, sizeof(xtfloat)*3*cols1);
        
        AE_LSX2X2_I(x00, x01, (xtfloatx4 *)px, 0);
        AE_LSX2X2_X(x10, x11, (xtfloatx4 *)px, sizeof(xtfloat)*cols1);
        AE_LSX2X2_X(x20, x21, (xtfloatx4 *)px, sizeof(xtfloat)*2*cols1);
        AE_LSX2X2_X(x30, x31, (xtfloatx4 *)px, sizeof(xtfloat)*3*cols1);

        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        MADDQ_S(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
        MADDQ_S(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
        MADDQ_S(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
        MADDQ_S(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
        MADDQ_S(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
        MADDQ_S(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
        
        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
        MADDQ_S(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
        MADDQ_S(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
        MADDQ_S(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
        MADDQ_S(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
        MADDQ_S(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
        MADDQ_S(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      }
  
      y0 = XT_SEL32_HL_SX2(acc_row0_vec0, acc_row1_vec0);
      y1 = XT_SEL32_LH_SX2(acc_row0_vec0, acc_row1_vec0);
      z0 = z0 + y0;
      z0 = z0 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row2_vec0, acc_row3_vec0);
      y3 = XT_SEL32_LH_SX2(acc_row2_vec0, acc_row3_vec0);
      z1 = z1 + y2;
      z1 = z1 + y3;

      y0 = XT_SEL32_HL_SX2(acc_row0_vec1, acc_row1_vec1);
      y1 = XT_SEL32_LH_SX2(acc_row0_vec1, acc_row1_vec1);
      z2 = z2 + y0;
      z2 = z2 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row2_vec1, acc_row3_vec1);
      y3 = XT_SEL32_LH_SX2(acc_row2_vec1, acc_row3_vec1);
      z3 = z3 + y2;
      z3 = z3 + y3;

      y0 = XT_SEL32_HL_SX2(acc_row0_vec2, acc_row1_vec2);
      y1 = XT_SEL32_LH_SX2(acc_row0_vec2, acc_row1_vec2);
      z4 = z4 + y0;
      z4 = z4 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row2_vec2, acc_row3_vec2);
      y3 = XT_SEL32_LH_SX2(acc_row2_vec2, acc_row3_vec2);
      z5 = z5 + y2;
      z5 = z5 + y3;

      y0 = XT_SEL32_HL_SX2(acc_row0_vec3, acc_row1_vec3);
      y1 = XT_SEL32_LH_SX2(acc_row0_vec3, acc_row1_vec3);
      z6 = z6 + y0;
      z6 = z6 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row2_vec3, acc_row3_vec3);
      y3 = XT_SEL32_LH_SX2(acc_row2_vec3, acc_row3_vec3);
      z7 = z7 + y2;
      z7 = z7 + y3;

      AE_SSX2X2_IP(z0, z1, (xtfloatx4 *)p_out_0, 16);
      AE_SSX2X2_IP(z2, z3, (xtfloatx4 *)p_out_1, 16);
      AE_SSX2X2_IP(z4, z5, (xtfloatx4 *)p_out_2, 16);
      AE_SSX2X2_IP(z6, z7, (xtfloatx4 *)p_out_3, 16);
    }
  }
  for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    xtfloat *p_out_0 = p_out + (vec_itr + 0)*out_offset;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xtfloat *px = (xtfloat *)(p_mat1+(m_itr*cols1));
      xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xtfloatx2 z0, z1;
      z0 = z1 = (xtfloatx2)0.0f;
      if(p_bias != NULL)
      {
        z0 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+0]), (xtfloatx2)(p_bias[m_itr+1]));
        z1 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+2]), (xtfloatx2)(p_bias[m_itr+3]));
      }
              
      xtfloatx2 acc_row0_vec0;
      xtfloatx2 acc_row1_vec0;
      xtfloatx2 acc_row2_vec0;
      xtfloatx2 acc_row3_vec0;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = 0.0f;
  
#pragma no_unroll
      for(c_itr = 0; c_itr < cols1 >> 2; c_itr++, p_vec+=4, px+=4)
      {
        AE_LSX2X2_I(vec0_0, vec0_1, (xtfloatx4 *)p_vec,  0);
        
        AE_LSX2X2_I(x00, x01, (xtfloatx4 *)px, 0);
        AE_LSX2X2_X(x10, x11, (xtfloatx4 *)px, sizeof(xtfloat)*cols1);
        AE_LSX2X2_X(x20, x21, (xtfloatx4 *)px, sizeof(xtfloat)*2*cols1);
        AE_LSX2X2_X(x30, x31, (xtfloatx4 *)px, sizeof(xtfloat)*3*cols1);

        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        
        MADDQ_S(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_S(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
      }
  
      y0 = XT_SEL32_HL_SX2(acc_row0_vec0, acc_row1_vec0);
      y1 = XT_SEL32_LH_SX2(acc_row0_vec0, acc_row1_vec0);
      z0 = z0 + y0;
      z0 = z0 + y1;

      y2 = XT_SEL32_HL_SX2(acc_row2_vec0, acc_row3_vec0);
      y3 = XT_SEL32_LH_SX2(acc_row2_vec0, acc_row3_vec0);
      z1 = z1 + y2;
      z1 = z1 + y3;

      AE_SSX2X2_IP(z0, z1, (xtfloatx4 *)p_out_0, 16);
    }
  }

}

static inline void _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloatx2* out_0_1
    ,xtfloatx2* out_1_1
    ,xtfloatx2* out_0_2
    ,xtfloatx2* out_1_2
    ,xtfloatx2* out_0_3
    ,xtfloatx2* out_1_3
    ,xtfloat*   px0
    ,xtfloat*   p_vec0
    ,WORD32     cols1
    ,WORD32     row_stride1
    ,WORD32     vec_offset
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  int align_offset = ((unsigned int)px0 & 0xf);

  pre_loop_count = (16 - align_offset) >> 2;
  pre_loop_count = (cols1 < pre_loop_count) ? cols1 : pre_loop_count;
  loop_count = (cols1 <= pre_loop_count)? 0 : (cols1 - pre_loop_count);
  post_loop_count = loop_count ? (loop_count & 3) : 0;
  loop_count >>= 2;

  xtfloatx2 acc_row0_vec0, acc_row0_vec1;
  xtfloatx2 acc_row1_vec0, acc_row1_vec1;
  xtfloatx2 acc_row2_vec0, acc_row2_vec1;
  xtfloatx2 acc_row3_vec0, acc_row3_vec1;
  xtfloatx2 acc_row0_vec2, acc_row0_vec3;
  xtfloatx2 acc_row1_vec2, acc_row1_vec3;
  xtfloatx2 acc_row2_vec2, acc_row2_vec3;
  xtfloatx2 acc_row3_vec2, acc_row3_vec3;
  xtfloatx2 x00, x01, x10, x11;
  xtfloatx2 x20, x21, x30, x31;
  xtfloatx2 vec0_0, vec0_1;
  xtfloatx2 vec1_0, vec1_1;
  xtfloatx2 vec2_0, vec2_1;
  xtfloatx2 vec3_0, vec3_1;
  xtfloatx2 y0, y1, y2, y3;

  xtfloat* px1 = px0 + 4*row_stride1; //next 4th row 
  xtfloat* px2 = px1 + 4*row_stride1; //next 4th row
  xtfloat* px3 = px2 + 4*row_stride1; //next 4th row 

  xtfloat *p_vec1  = (xtfloat *)(p_vec0 + vec_offset);
  xtfloat *p_vec2  = (xtfloat *)(p_vec1 + vec_offset);
  xtfloat *p_vec3  = (xtfloat *)(p_vec2 + vec_offset);
  
  xtfloatx2 z0 = *out_0_0;
  xtfloatx2 z1 = *out_1_0;
  xtfloatx2 z2 = *out_0_1;
  xtfloatx2 z3 = *out_1_1;
  xtfloatx2 z4 = *out_0_2;
  xtfloatx2 z5 = *out_1_2;
  xtfloatx2 z6 = *out_0_3;
  xtfloatx2 z7 = *out_1_3;

  /* Pre loop computation */
  acc_row0_vec0 = acc_row2_vec0 = 0.0f;
  acc_row0_vec1 = acc_row2_vec1 = 0.0f;
  acc_row0_vec2 = acc_row2_vec2 = 0.0f;
  acc_row0_vec3 = acc_row2_vec3 = 0.0f;
  int k;

#pragma loop_count min=1
  for(k = 0; k < pre_loop_count; k++, px0++, px1++, px2++, px3++, p_vec0++, p_vec1++, p_vec2++, p_vec3++)
  {
      x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(px0)), (xtfloatx2)(*(px1)));
      x20 = XT_SEL32_LL_SX2((xtfloatx2)(*(px2)), (xtfloatx2)(*(px3)));
      vec0_0 = (xtfloatx2)(*(p_vec0));
      vec1_0 = (xtfloatx2)(*(p_vec1));
      vec2_0 = (xtfloatx2)(*(p_vec2));
      vec3_0 = (xtfloatx2)(*(p_vec3));
      MADDQ_S(acc_row0_vec0, acc_row2_vec0, x00, x20, vec0_0);
      MADDQ_S(acc_row0_vec1, acc_row2_vec1, x00, x20, vec1_0);
      MADDQ_S(acc_row0_vec2, acc_row2_vec2, x00, x20, vec2_0);
      MADDQ_S(acc_row0_vec3, acc_row2_vec3, x00, x20, vec3_0);
  }
  z0 = z0 + acc_row0_vec0;
  z1 = z1 + acc_row2_vec0;
  z2 = z2 + acc_row0_vec1;
  z3 = z3 + acc_row2_vec1;
  z4 = z4 + acc_row0_vec2;
  z5 = z5 + acc_row2_vec2;
  z6 = z6 + acc_row0_vec3;
  z7 = z7 + acc_row2_vec3;

  acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = 0.0f;
  acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = 0.0f;
  acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = 0.0f;
  acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = 0.0f;
  
  ae_valignx2 vec0_a = AE_LA128_PP(p_vec0);
  ae_valignx2 vec1_a = AE_LA128_PP(p_vec1);
  ae_valignx2 vec2_a = AE_LA128_PP(p_vec2);
  ae_valignx2 vec3_a = AE_LA128_PP(p_vec3);

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_LASX2X2_IP(vec0_0, vec0_1, vec0_a, (xtfloatx4 *)p_vec0);
    AE_LASX2X2_IP(vec1_0, vec1_1, vec1_a, (xtfloatx4 *)p_vec1);
    AE_LASX2X2_IP(vec2_0, vec2_1, vec2_a, (xtfloatx4 *)p_vec2);
    AE_LASX2X2_IP(vec3_0, vec3_1, vec3_a, (xtfloatx4 *)p_vec3);
    
    AE_LSX2X2_IP(x00, x01, (xtfloatx4 *)px0, 16);
    AE_LSX2X2_IP(x10, x11, (xtfloatx4 *)px1, 16);
    AE_LSX2X2_IP(x20, x21, (xtfloatx4 *)px2, 16);
    AE_LSX2X2_IP(x30, x31, (xtfloatx4 *)px3, 16);

    MADDQ_S(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
    MADDQ_S(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
    MADDQ_S(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
    MADDQ_S(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
    MADDQ_S(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
    MADDQ_S(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
    MADDQ_S(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
    MADDQ_S(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
    
    MADDQ_S(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
    MADDQ_S(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
    MADDQ_S(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
    MADDQ_S(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
    MADDQ_S(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
    MADDQ_S(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
    MADDQ_S(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
    MADDQ_S(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
  }
  y0 = XT_SEL32_HL_SX2(acc_row0_vec0, acc_row1_vec0);
  y1 = XT_SEL32_LH_SX2(acc_row0_vec0, acc_row1_vec0);
  z0 = z0 + y0;
  z0 = z0 + y1;

  y2 = XT_SEL32_HL_SX2(acc_row2_vec0, acc_row3_vec0);
  y3 = XT_SEL32_LH_SX2(acc_row2_vec0, acc_row3_vec0);
  z1 = z1 + y2;
  z1 = z1 + y3;

  y0 = XT_SEL32_HL_SX2(acc_row0_vec1, acc_row1_vec1);
  y1 = XT_SEL32_LH_SX2(acc_row0_vec1, acc_row1_vec1);
  z2 = z2 + y0;
  z2 = z2 + y1;

  y2 = XT_SEL32_HL_SX2(acc_row2_vec1, acc_row3_vec1);
  y3 = XT_SEL32_LH_SX2(acc_row2_vec1, acc_row3_vec1);
  z3 = z3 + y2;
  z3 = z3 + y3;

  y0 = XT_SEL32_HL_SX2(acc_row0_vec2, acc_row1_vec2);
  y1 = XT_SEL32_LH_SX2(acc_row0_vec2, acc_row1_vec2);
  z4 = z4 + y0;
  z4 = z4 + y1;

  y2 = XT_SEL32_HL_SX2(acc_row2_vec2, acc_row3_vec2);
  y3 = XT_SEL32_LH_SX2(acc_row2_vec2, acc_row3_vec2);
  z5 = z5 + y2;
  z5 = z5 + y3;

  y0 = XT_SEL32_HL_SX2(acc_row0_vec3, acc_row1_vec3);
  y1 = XT_SEL32_LH_SX2(acc_row0_vec3, acc_row1_vec3);
  z6 = z6 + y0;
  z6 = z6 + y1;

  y2 = XT_SEL32_HL_SX2(acc_row2_vec3, acc_row3_vec3);
  y3 = XT_SEL32_LH_SX2(acc_row2_vec3, acc_row3_vec3);
  z7 = z7 + y2;
  z7 = z7 + y3;

  //Remainder loop for cols1
  acc_row0_vec0 = acc_row2_vec0 = 0.0f;
  acc_row0_vec1 = acc_row2_vec1 = 0.0f;
  acc_row0_vec2 = acc_row2_vec2 = 0.0f;
  acc_row0_vec3 = acc_row2_vec3 = 0.0f;
  for(k = 0; k < post_loop_count; k++, px0++, px1++, px2++, px3++, p_vec0++, p_vec1++, p_vec2++, p_vec3++)
  {
      x00 = XT_SEL32_LL_SX2((xtfloatx2)(*(px0)), (xtfloatx2)(*(px1)));
      x20 = XT_SEL32_LL_SX2((xtfloatx2)(*(px2)), (xtfloatx2)(*(px3)));
      vec0_0 = (xtfloatx2)(*(p_vec0));
      vec1_0 = (xtfloatx2)(*(p_vec1));
      vec2_0 = (xtfloatx2)(*(p_vec2));
      vec3_0 = (xtfloatx2)(*(p_vec3));
      MADDQ_S(acc_row0_vec0, acc_row2_vec0, x00, x20, vec0_0);
      MADDQ_S(acc_row0_vec1, acc_row2_vec1, x00, x20, vec1_0);
      MADDQ_S(acc_row0_vec2, acc_row2_vec2, x00, x20, vec2_0);
      MADDQ_S(acc_row0_vec3, acc_row2_vec3, x00, x20, vec3_0);
  }
  z0 = z0 + acc_row0_vec0;
  z1 = z1 + acc_row2_vec0;
  z2 = z2 + acc_row0_vec1;
  z3 = z3 + acc_row2_vec1;
  z4 = z4 + acc_row0_vec2;
  z5 = z5 + acc_row2_vec2;
  z6 = z6 + acc_row0_vec3;
  z7 = z7 + acc_row2_vec3;

  *out_0_0 = z0;
  *out_1_0 = z1;
  *out_0_1 = z2;
  *out_1_1 = z3;
  *out_0_2 = z4;
  *out_1_2 = z5;
  *out_0_3 = z6;
  *out_1_3 = z7;
}

static inline void _xa_nn_dot_product_1_row_4_vecs_unaligned
    (xtfloatx2* out_0_0
    ,xtfloatx2* out_1_0
    ,xtfloat*   py
    ,xtfloat*   pv0
    ,WORD32     cols1
    ,WORD32     vec_offset
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  int align_offset = ((unsigned int)py & 0xf);

  pre_loop_count = (16 - align_offset) >> 2;
  pre_loop_count = (cols1 < pre_loop_count) ? cols1 : pre_loop_count;
  loop_count = (cols1 <= pre_loop_count)? 0 : (cols1 - pre_loop_count);
  post_loop_count = loop_count ? (loop_count & 7) : 0;
  loop_count >>= 3;

  xtfloatx2 acc00, acc01, acc02, acc03;
  xtfloatx2 acc10, acc11, acc12, acc13;
  xtfloatx2 acc20, acc21, acc22, acc23;
  xtfloatx2 acc30, acc31, acc32, acc33;
  xtfloatx2 v00, v01, v02, v03;
  xtfloatx2 v10, v11, v12, v13;
  xtfloatx2 v20, v21, v22, v23;
  xtfloatx2 v30, v31, v32, v33;
  xtfloatx2 y0, y1, y2, y3;

  xtfloat* pv1 = pv0 + vec_offset; //next vec 
  xtfloat* pv2 = pv1 + vec_offset; //next vec
  xtfloat* pv3 = pv2 + vec_offset; //next vec 
  
  xtfloatx2 z0 = *out_0_0;
  xtfloatx2 z1 = *out_1_0;

  /* Pre loop computation */
  acc00 = 0.0f;
  acc20 = 0.0f;
  int k;

#pragma loop_count min=1
  for(k = 0; k < pre_loop_count; k++, pv0++, pv1++, pv2++, pv3++, py++)
  {
      v00 = XT_SEL32_LL_SX2((xtfloatx2)(*(pv0)), (xtfloatx2)(*(pv1)));
      v20 = XT_SEL32_LL_SX2((xtfloatx2)(*(pv2)), (xtfloatx2)(*(pv3)));
      y0 = (xtfloatx2)(*(py));
      MADD_SX2(acc00, v00, y0);
      MADD_SX2(acc20, v20, y0);
  }
  z0 = z0 + acc00;
  z1 = z1 + acc20;

  acc00 = acc01 = acc10 = acc11 = 0.0f;
  acc20 = acc21 = acc30 = acc31 = 0.0f;
  acc02 = acc03 = acc12 = acc13 = 0.0f;
  acc22 = acc23 = acc32 = acc33 = 0.0f;
  ae_valignx2 v0_a = AE_LA128_PP(pv0);
  ae_valignx2 v1_a = AE_LA128_PP(pv1);
  ae_valignx2 v2_a = AE_LA128_PP(pv2);
  ae_valignx2 v3_a = AE_LA128_PP(pv3);

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_LSX2X2_IP(y0, y1, (xtfloatx4 *)py, 16);
    AE_LSX2X2_IP(y2, y3, (xtfloatx4 *)py, 16);
    
    AE_LASX2X2_IP(v00, v01, v0_a, (xtfloatx4 *)pv0);
    AE_LASX2X2_IP(v10, v11, v1_a, (xtfloatx4 *)pv1);
    AE_LASX2X2_IP(v20, v21, v2_a, (xtfloatx4 *)pv2);
    AE_LASX2X2_IP(v30, v31, v3_a, (xtfloatx4 *)pv3);

    MADD_SX2X2(acc00,acc01,v00,v01,y0,y1);
    MADD_SX2X2(acc10,acc11,v10,v11,y0,y1);
    MADD_SX2X2(acc20,acc21,v20,v21,y0,y1);
    MADD_SX2X2(acc30,acc31,v30,v31,y0,y1);
    
    AE_LASX2X2_IP(v02, v03, v0_a, (xtfloatx4 *)pv0);
    AE_LASX2X2_IP(v12, v13, v1_a, (xtfloatx4 *)pv1);
    AE_LASX2X2_IP(v22, v23, v2_a, (xtfloatx4 *)pv2);
    AE_LASX2X2_IP(v32, v33, v3_a, (xtfloatx4 *)pv3);

    MADD_SX2X2(acc02,acc03,v02,v03,y2,y3);
    MADD_SX2X2(acc12,acc13,v12,v13,y2,y3);
    MADD_SX2X2(acc22,acc23,v22,v23,y2,y3);
    MADD_SX2X2(acc32,acc33,v32,v33,y2,y3);
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
  for(k = 0; k < post_loop_count; k++, pv0++, pv1++, pv2++, pv3++, py++)
  {
      v00 = XT_SEL32_LL_SX2((xtfloatx2)(*(pv0)), (xtfloatx2)(*(pv1)));
      v20 = XT_SEL32_LL_SX2((xtfloatx2)(*(pv2)), (xtfloatx2)(*(pv3)));
      y0 = (xtfloatx2)(*(py));
      MADD_SX2(acc00, v00, y0);
      MADD_SX2(acc20, v20, y0);
  }
  z0 = z0 + acc00;
  z1 = z1 + acc20;

  *out_0_0 = z0;
  *out_1_0 = z1;
}

WORD32 xa_nn_matmul_f32xf32_f32(
    FLOAT32 * __restrict__ p_out,          
    const FLOAT32 * __restrict__ p_mat1,   
    const FLOAT32 * __restrict__ p_vec1,   
    const FLOAT32 * __restrict__ p_bias,   
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                    
    WORD32 vec_count,                      
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride)                      
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    /* Special cases for conv2d_ptwise: 
       cols(inp_channels) multiple of 4
       rows(out_channels) multiple of 4 */
    if(((out_stride == 1 || out_offset == 1) &&
        ((out_offset & 0x3) == 0 || (out_stride & 0x3) == 0) && 
        (cols1 & 3) == 0) &&
        (row_stride1 == cols1) &&
        (vec_offset == cols1) &&
        ALIGNED_PTR(p_mat1, 16) &&
        ALIGNED_PTR(p_vec1, 16) &&
        ALIGNED_PTR(p_out, 16) &&
        ((rows & 0x3) == 0)
      )
    {
      /* NHWC out data format --> out_stride = 1 */
      /* NCHW out data format --> out_offset = 1 */
      if(out_stride == 1)
        spfunc_cols_mul4_out_stride1
          (p_out,
           p_mat1,
           p_vec1,
           p_bias,
           rows,
           vec_count,
           cols1,
           out_offset
          );
      else if(out_offset == 1)
        spfunc_cols_mul4_out_offset1
          (p_out,
           p_mat1,
           p_vec1,
           p_bias,
           rows,
           vec_count,
           cols1,
           out_stride
          );
      return 0;
    }
    else
    {
      for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
      {
        xtfloat *p_out_0 = p_out + (vec_itr + 0)*out_offset;
        xtfloat *p_out_1 = p_out + (vec_itr + 1)*out_offset;
        xtfloat *p_out_2 = p_out + (vec_itr + 2)*out_offset;
        xtfloat *p_out_3 = p_out + (vec_itr + 3)*out_offset;
        int ii;
        for(m_itr = 0; m_itr < (rows & ~(16 - 1)); m_itr += 16)
        {
          for(ii = 0; ii < 4; ii++)
          {
              xtfloat *p_out_0_ii = p_out_0 + (m_itr + ii) * out_stride;
              xtfloat *p_out_1_ii = p_out_1 + (m_itr + ii) * out_stride;
              xtfloat *p_out_2_ii = p_out_2 + (m_itr + ii) * out_stride;
              xtfloat *p_out_3_ii = p_out_3 + (m_itr + ii) * out_stride;
              /* Init out registers with bias */
              xtfloatx2 z0, z1, z2, z3;
              xtfloatx2 z4, z5, z6, z7;
              z0 = z1 = z2 = z3 = (xtfloatx2)0.0f;
              z4 = z5 = z6 = z7 = (xtfloatx2)0.0f;
              if(p_bias != NULL)
              {
                z6 = z4 = z2 = z0 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+ii+0]), (xtfloatx2)(p_bias[m_itr+ii+4]));
                z7 = z5 = z3 = z1 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+ii+8]), (xtfloatx2)(p_bias[m_itr+ii+12]));
              }
              
              xtfloat *p_mat = (xtfloat *)(p_mat1+((m_itr+ii)*row_stride1));
              xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*vec_offset));

              _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
                (&z0
                ,&z1
                ,&z2
                ,&z3
                ,&z4
                ,&z5
                ,&z6
                ,&z7
                ,(xtfloat *)p_mat
                ,(xtfloat *)p_vec
                ,cols1
                ,row_stride1
                ,vec_offset
                );
              
              XT_SSXP(XT_SEL32_HH_SX2(z0,z0), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z0, p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z1,z1), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z1, p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z2,z2), p_out_1_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z2, p_out_1_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z3,z3), p_out_1_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z3, p_out_1_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z4,z4), p_out_2_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z4, p_out_2_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z5,z5), p_out_2_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z5, p_out_2_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z6,z6), p_out_3_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z6, p_out_3_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z7,z7), p_out_3_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z7, p_out_3_ii, 4*out_stride*sizeof(xtfloat));
            }
          }
        p_out_0 = p_out_0 + (rows & (~15)) * out_stride;
        p_out_1 = p_out_1 + (rows & (~15)) * out_stride;
        p_out_2 = p_out_2 + (rows & (~15)) * out_stride;
        p_out_3 = p_out_3 + (rows & (~15)) * out_stride;
        
        //Remaining (rows % 16) rows
        for(m_itr = (rows & ~(15)); m_itr < rows; m_itr++)
        {
          /* Init out registers with bias */
          xtfloatx2 z0, z1;
          z0 = z1 = (xtfloatx2)0.0f;
          if(p_bias != NULL)
          {
            z0 = z1 = (xtfloatx2)(p_bias[m_itr]);
          }
          
          xtfloat *p_mat = (xtfloat *)(p_mat1+(m_itr*row_stride1));
          xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*vec_offset));

          _xa_nn_dot_product_1_row_4_vecs_unaligned
            (&z0
            ,&z1
            ,(xtfloat *)p_mat
            ,(xtfloat *)p_vec
            ,cols1
            ,vec_offset
            );
         
          AE_SSXP(XT_SEL32_HH_SX2(z0,z0), p_out_0, out_stride*sizeof(xtfloat));
          AE_SSXP(z0, p_out_1, out_stride*sizeof(xtfloat));
          AE_SSXP(XT_SEL32_HH_SX2(z1,z1), p_out_2, out_stride*sizeof(xtfloat));
          AE_SSXP(z1, p_out_3, out_stride*sizeof(xtfloat));
        }
      }
      /* Tail loop for vec unroll */
      for(vec_itr = (vec_count & ~(3)); vec_itr < vec_count; vec_itr++)
      {
        int ii;
        xtfloat *p_out_0 = p_out + (vec_itr*out_offset);
        for(m_itr = 0; m_itr < (rows & ~(16 - 1)); m_itr += 16)
        {
          for(ii = 0; ii < 4; ii++)
          {
              xtfloat *p_out_0_ii = p_out_0 + (m_itr + ii) * out_stride;
              /* Init out registers with bias */
              xtfloatx2 z0, z1;
              z0 = z1 = (xtfloatx2)0.0f;
              if(p_bias != NULL)
              {
                z0 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+ii+0]), (xtfloatx2)(p_bias[m_itr+ii+4]));
                z1 = XT_SEL32_LL_SX2((xtfloatx2)(p_bias[m_itr+ii+8]), (xtfloatx2)(p_bias[m_itr+ii+12]));
              }
              
              xtfloat *p_mat = (xtfloat *)(p_mat1+((m_itr+ii)*row_stride1));
              xtfloat *p_vec = (xtfloat *)(p_vec1+(vec_itr*vec_offset));

              _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
                (&z0
                ,&z1
                ,(xtfloat *)p_mat
                ,(xtfloat *)p_vec
                ,cols1
                ,row_stride1
                );
              
              XT_SSXP(XT_SEL32_HH_SX2(z0,z0), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z0, p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(XT_SEL32_HH_SX2(z1,z1), p_out_0_ii, 4*out_stride*sizeof(xtfloat));
              XT_SSXP(z1, p_out_0_ii, 4*out_stride*sizeof(xtfloat));
            }
        }

        p_out_0 = p_out_0 + (rows & (~15)) * out_stride;
        xtfloat bias = 0.0f;
        xtfloat *pbias = (xtfloat *) p_bias + m_itr;
        for(m_itr = (rows & ~(15)); m_itr < rows; m_itr++)
        {
          xtfloatx2 acc_row0_vec0;
          xtfloatx2 acc_row0_vec0B;
          xtfloatx2 vec_batch_0_0, vec_batch_0_1;
          xtfloatx2 vec_batch_0_2, vec_batch_0_3;
          xtfloatx2 mat1_0_0, mat1_0_1;
          xtfloatx2 mat1_0_2, mat1_0_3;
          
          xtfloat *p_vec_batch_0  = (xtfloat *)(p_vec1 + (vec_itr + 0)*vec_offset);
          xtfloat *p_mat1_0 = (xtfloat *) &p_mat1[(m_itr+0)*row_stride1];
          
          ae_valignx2 align_vec_batch_0 = AE_LA128_PP(p_vec_batch_0);
          ae_valignx2 align_mat1_0 = AE_LA128_PP(p_mat1_0);

          acc_row0_vec0 = 0.0f;
          acc_row0_vec0B = 0.0f;
          int cols1_count = cols1-(cols1&7);
          for(c_itr = 0; c_itr < cols1_count; c_itr+=8)
          {
              AE_LASX2X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec_batch_0, (xtfloatx4 *)p_vec_batch_0);
              AE_LASX2X2_IP(mat1_0_0, mat1_0_1, align_mat1_0, (xtfloatx4 *)p_mat1_0);
              
              MADD_SX2(acc_row0_vec0, vec_batch_0_0, mat1_0_0);
              MADD_SX2(acc_row0_vec0B, vec_batch_0_1, mat1_0_1);
              
              AE_LASX2X2_IP(vec_batch_0_2, vec_batch_0_3, align_vec_batch_0, (xtfloatx4 *)p_vec_batch_0);
              AE_LASX2X2_IP(mat1_0_2, mat1_0_3, align_mat1_0, (xtfloatx4 *)p_mat1_0);
              
              MADD_SX2(acc_row0_vec0, vec_batch_0_2, mat1_0_2);
              MADD_SX2(acc_row0_vec0B, vec_batch_0_3, mat1_0_3);
          }
          acc_row0_vec0 += acc_row0_vec0B;
          acc_row0_vec0 = XT_RADD_SX2(acc_row0_vec0);
          
          /* Remainder loop for cols1 */
          for(c_itr = cols1_count; c_itr < cols1; c_itr++, 
              p_vec_batch_0++, p_mat1_0++)
          {
              vec_batch_0_0 = (xtfloatx2)(*((xtfloat *)p_vec_batch_0));
              mat1_0_0 = (xtfloatx2)(*((xtfloat *)p_mat1_0));
              MADD_SX2(acc_row0_vec0, vec_batch_0_0, mat1_0_0);
          }
          if(p_bias!=NULL)
          {
            XT_LSIP(bias, pbias, 4);
            acc_row0_vec0 = ADD_S(acc_row0_vec0, bias);
          }
         
          XT_SSXP(acc_row0_vec0, p_out_0, out_stride*sizeof(xtfloat));
        }
      }
    }

    return 0;
}
#endif
