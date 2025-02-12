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
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matmul_f16xf16_f16,(
    WORD16 * __restrict__ p_out,          
    const WORD16 * __restrict__ p_mat1,   
    const WORD16 * __restrict__ p_vec1,   
    const WORD16 * __restrict__ p_bias,   
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                    
    WORD32 vec_count,                      
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride))
	
#else
#define DSELHX4(out0, out1, inp0, inp1, dsel){\
  ae_int16x4 out0_tmp, out1_tmp, inp0_tmp, inp1_tmp;\
  inp0_tmp = AE_MOVF16X4_FROMHALFX4(inp0);\
  inp1_tmp = AE_MOVF16X4_FROMHALFX4(inp1);\
  AE_DSEL16X4(out0_tmp, out1_tmp, inp0_tmp, inp1_tmp, dsel);\
  out0 = AE_MOVHALFX4_FROMF16X4(out0_tmp);\
  out1 = AE_MOVHALFX4_FROMF16X4(out1_tmp);\
}

static inline void spfunc_cols_mul4_out_stride1
    (xthalf*   p_out
    ,const xthalf*   p_mat1
    ,const xthalf*   p_vec1
    ,const xthalf*   p_bias
    ,WORD32     rows
    ,WORD32     vec_count
    ,WORD32     cols1
    ,WORD32     out_offset
    )
{
  int vec_itr, m_itr, c_itr;

  xthalfx4 x00, x01, x10, x11;
  xthalfx4 x20, x21, x30, x31;
  xthalfx4 vec0_0, vec0_1;
  xthalfx4 vec1_0, vec1_1;
  xthalfx4 vec2_0, vec2_1;
  xthalfx4 vec3_0, vec3_1;
  xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;
  ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));
  
  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    xthalf *p_out_0 = p_out + (vec_itr + 0)*out_offset;
    xthalf *p_out_1 = p_out + (vec_itr + 1)*out_offset;
    xthalf *p_out_2 = p_out + (vec_itr + 2)*out_offset;
    xthalf *p_out_3 = p_out + (vec_itr + 3)*out_offset;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xthalfx4 z0, z1, z2, z3;
      z0 = z1 = z2 = z3 = (xthalfx4)ZERO_HX4();
      if(p_bias != NULL)
      {
		 z0=AE_LHX4I((xthalfx4 *)&p_bias[m_itr],0);
		 z1=z2=z3=z0;
      }
              
      xthalfx4 acc_row0_vec0, acc_row0_vec1;
      xthalfx4 acc_row1_vec0, acc_row1_vec1;
      xthalfx4 acc_row2_vec0, acc_row2_vec1;
      xthalfx4 acc_row3_vec0, acc_row3_vec1;
      xthalfx4 acc_row0_vec2, acc_row0_vec3;
      xthalfx4 acc_row1_vec2, acc_row1_vec3;
      xthalfx4 acc_row2_vec2, acc_row2_vec3;
      xthalfx4 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = ZERO_HX4();
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = ZERO_HX4();
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = ZERO_HX4();

#pragma no_unroll      
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      {
        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec, 0);
        AE_LHX4X2_X(vec1_0, vec1_1, (xthalfx8 *)p_vec, sizeof(xthalf)*cols1);
        AE_LHX4X2_X(vec2_0, vec2_1, (xthalfx8 *)p_vec, sizeof(xthalf)*2*cols1);
        AE_LHX4X2_X(vec3_0, vec3_1, (xthalfx8 *)p_vec, sizeof(xthalf)*3*cols1);
        
        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
        
        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      }
      y0 = AE_SELH_7531(acc_row0_vec0, acc_row1_vec0);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row1_vec0);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec0, acc_row3_vec0);
      y3 = AE_SELH_6420(acc_row2_vec0, acc_row3_vec0);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z0=z0+y02;
	  z0=z0+y13;
	  
      y0 = AE_SELH_7531(acc_row0_vec1, acc_row1_vec1);
      y1 = AE_SELH_6420(acc_row0_vec1, acc_row1_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec1, acc_row3_vec1);
      y3 = AE_SELH_6420(acc_row2_vec1, acc_row3_vec1);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z1=z1+y02;
	  z1=z1+y13;
	  
	  y0 = AE_SELH_7531(acc_row0_vec2, acc_row1_vec2);
      y1 = AE_SELH_6420(acc_row0_vec2, acc_row1_vec2);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec2, acc_row3_vec2);
      y3 = AE_SELH_6420(acc_row2_vec2, acc_row3_vec2);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z2=z2+y02;
	  z2=z2+y13;
	  
	  
	  y0 = AE_SELH_7531(acc_row0_vec3, acc_row1_vec3);
      y1 = AE_SELH_6420(acc_row0_vec3, acc_row1_vec3);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec3, acc_row3_vec3);
      y3 = AE_SELH_6420(acc_row2_vec3, acc_row3_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z3=z3+y02;
	  z3=z3+y13;
	  
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
      AE_SHX4IP(z1,(xthalfx4 *)p_out_1, 8);
      AE_SHX4IP(z2,(xthalfx4 *)p_out_2, 8);
      AE_SHX4IP(z3,(xthalfx4 *)p_out_3, 8);
    }
  }
   for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    xthalf *p_out_0 = p_out + (vec_itr + 0)*out_offset;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));

      /* Init out registers with bias */
      xthalfx4 z0;
      z0 =(xthalfx4)ZERO_HX4();
      if(p_bias != NULL)
      {
        z0=AE_LHX4I((xthalfx4 *)&p_bias[m_itr],0);
      }

      xthalfx4 acc_row0_vec0,acc_row0_vec0_s;
      xthalfx4 acc_row1_vec0,acc_row1_vec0_s;
      xthalfx4 acc_row2_vec0,acc_row2_vec0_s;
      xthalfx4 acc_row3_vec0,acc_row3_vec0_s;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec0_s = acc_row1_vec0_s = acc_row2_vec0_s = acc_row3_vec0_s = ZERO_HX4();
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      {
        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec,  0);

        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        
        MADDQ_H(acc_row0_vec0_s, acc_row1_vec0_s, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0_s, acc_row3_vec0_s, x21, x31, vec0_1);
      }

      ADD_HX4X2(acc_row0_vec0,acc_row1_vec0,acc_row0_vec0,acc_row1_vec0,acc_row0_vec0_s,acc_row1_vec0_s);
      ADD_HX4X2(acc_row2_vec0,acc_row3_vec0,acc_row2_vec0,acc_row3_vec0,acc_row2_vec0_s,acc_row3_vec0_s);
      
      DSELHX4(y0,y1,acc_row0_vec0,acc_row1_vec0,dsel0);
      y01 = y0 + y1;
      DSELHX4(y2,y3,acc_row2_vec0,acc_row3_vec0,dsel0);
      y23 = y2 + y3;
      DSELHX4(y02,y13,y01,y23,dsel0);
      z0 = z0 + y02;
      z0 = z0 + y13;

      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
    }
  }

}

static inline void spfunc_aligned_cols_mul4_out_stride1
    (xthalf*   p_out
    ,const xthalf*   p_mat1
    ,const xthalf*   p_vec1
    ,const xthalf*   p_bias
    ,WORD32     rows
    ,WORD32     vec_count
    ,WORD32     cols1
    ,WORD32     out_offset
    ,WORD32     row_stride1
    ,WORD32     vec_offset
    )
{
  int vec_itr, m_itr, c_itr;

  xthalfx4 x00, x01, x10, x11;
  xthalfx4 x20, x21, x30, x31;
  xthalfx4 vec0_0, vec0_1;
  xthalfx4 vec1_0, vec1_1;
  xthalfx4 vec2_0, vec2_1;
  xthalfx4 vec3_0, vec3_1;
  xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;
  ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));
  
  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    xthalf *p_out_0 = p_out + (vec_itr + 0)*out_offset;
    xthalf *p_out_1 = p_out + (vec_itr + 1)*out_offset;
    xthalf *p_out_2 = p_out + (vec_itr + 2)*out_offset;
    xthalf *p_out_3 = p_out + (vec_itr + 3)*out_offset;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalf *px = (xthalf *)(p_mat1+(m_itr*row_stride1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*vec_offset));
  
      /* Init out registers with bias */
      xthalfx4 z0, z1, z2, z3;
      z0 = z1 = z2 = z3 = (xthalfx4)ZERO_HX4();
      if(p_bias != NULL)
      {
		 z0=AE_LHX4I((xthalfx4 *)&p_bias[m_itr],0);
		 z1=z2=z3=z0;
      }
              
      xthalfx4 acc_row0_vec0, acc_row0_vec1;
      xthalfx4 acc_row1_vec0, acc_row1_vec1;
      xthalfx4 acc_row2_vec0, acc_row2_vec1;
      xthalfx4 acc_row3_vec0, acc_row3_vec1;
      xthalfx4 acc_row0_vec2, acc_row0_vec3;
      xthalfx4 acc_row1_vec2, acc_row1_vec3;
      xthalfx4 acc_row2_vec2, acc_row2_vec3;
      xthalfx4 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = ZERO_HX4();
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = ZERO_HX4();
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = ZERO_HX4();
      
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      {
        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec, 0);
        AE_LHX4X2_X(vec1_0, vec1_1, (xthalfx8 *)p_vec, sizeof(xthalf)*vec_offset);
        AE_LHX4X2_X(vec2_0, vec2_1, (xthalfx8 *)p_vec, sizeof(xthalf)*2*vec_offset);
        AE_LHX4X2_X(vec3_0, vec3_1, (xthalfx8 *)p_vec, sizeof(xthalf)*3*vec_offset);
        
        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*row_stride1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*row_stride1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*row_stride1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
        
        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      }
      
      y0 = AE_SELH_7531(acc_row0_vec0, acc_row1_vec0);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row1_vec0);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec0, acc_row3_vec0);
      y3 = AE_SELH_6420(acc_row2_vec0, acc_row3_vec0);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  z0=z0+y02;
	  z0=z0+y13;
	  
      y0 = AE_SELH_7531(acc_row0_vec1, acc_row1_vec1);
      y1 = AE_SELH_6420(acc_row0_vec1, acc_row1_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec1, acc_row3_vec1);
      y3 = AE_SELH_6420(acc_row2_vec1, acc_row3_vec1);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  z1=z1+y02;
	  z1=z1+y13;
	  
	  y0 = AE_SELH_7531(acc_row0_vec2, acc_row1_vec2);
      y1 = AE_SELH_6420(acc_row0_vec2, acc_row1_vec2);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec2, acc_row3_vec2);
      y3 = AE_SELH_6420(acc_row2_vec2, acc_row3_vec2);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  z2=z2+y02;
	  z2=z2+y13;
	 
	  y0 = AE_SELH_7531(acc_row0_vec3, acc_row1_vec3);
      y1 = AE_SELH_6420(acc_row0_vec3, acc_row1_vec3);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec3, acc_row3_vec3);
      y3 = AE_SELH_6420(acc_row2_vec3, acc_row3_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z3=z3+y02;
	  z3=z3+y13;
	  
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
      AE_SHX4IP(z1,(xthalfx4 *)p_out_1, 8);
      AE_SHX4IP(z2,(xthalfx4 *)p_out_2, 8);
      AE_SHX4IP(z3,(xthalfx4 *)p_out_3, 8);
    }
  }
   for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    xthalf *p_out_0 = p_out + (vec_itr + 0)*out_offset;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalf *px = (xthalf *)(p_mat1+(m_itr*row_stride1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*vec_offset));

      /* Init out registers with bias */
      xthalfx4 z0;
      z0 = (xthalfx4)ZERO_HX4();
      if(p_bias != NULL)
      {
        z0=AE_LHX4I((xthalfx4 *)&p_bias[m_itr],0);
      }

      xthalfx4 acc_row0_vec0,acc_row0_vec0_s;
      xthalfx4 acc_row1_vec0,acc_row1_vec0_s;
      xthalfx4 acc_row2_vec0,acc_row2_vec0_s;
      xthalfx4 acc_row3_vec0,acc_row3_vec0_s;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec0_s = acc_row1_vec0_s = acc_row2_vec0_s = acc_row3_vec0_s = ZERO_HX4();

#pragma no_unroll
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      {
        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec,  0);

        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*row_stride1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*row_stride1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*row_stride1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        
        MADDQ_H(acc_row0_vec0_s, acc_row1_vec0_s, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0_s, acc_row3_vec0_s, x21, x31, vec0_1);
      }

      ADD_HX4X2(acc_row0_vec0,acc_row1_vec0,acc_row0_vec0,acc_row1_vec0,acc_row0_vec0_s,acc_row1_vec0_s);
      ADD_HX4X2(acc_row2_vec0,acc_row3_vec0,acc_row2_vec0,acc_row3_vec0,acc_row2_vec0_s,acc_row3_vec0_s);
      
      DSELHX4(y0,y1,acc_row0_vec0,acc_row1_vec0,dsel0);
      y01 = y0 + y1;
      DSELHX4(y2,y3,acc_row2_vec0,acc_row3_vec0,dsel0);
      y23 = y2 + y3;
      DSELHX4(y02,y13,y01,y23,dsel0);
      z0 = z0 + y02;
      z0 = z0 + y13;
      
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
    }
  }

}

static inline void spfunc_cols_mul4_out_offset1
    (xthalf*   p_out
    ,const xthalf*   p_mat1
    ,const xthalf*   p_vec1
    ,const xthalf*   p_bias
    ,WORD32     rows
    ,WORD32     vec_count
    ,WORD32     cols1
    ,WORD32     out_stride
    )
{
  int vec_itr, m_itr, c_itr;
  ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));

 if(cols1==8)
 { 
  if(p_bias!=NULL)
  { 
  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {

      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;
      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xthalfx4 z0, z1, z2, z3;
         
      ae_int16x4 z0_int,z1_int,z2_int,z3_int;
      z0_int=AE_L16_I((ae_int16 *)&p_bias[m_itr],0);
      z0=AE_MOVHALFX4_FROMF16X4(z0_int);
      z1_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+1],0);
      z1=AE_MOVHALFX4_FROMF16X4(z1_int);
      z2_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+2],0);
      z2=AE_MOVHALFX4_FROMF16X4(z2_int);
      z3_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+3],0);
      z3=AE_MOVHALFX4_FROMF16X4(z3_int);
              
      xthalfx4 acc_row0_vec0, acc_row0_vec1;
      xthalfx4 acc_row1_vec0, acc_row1_vec1;
      xthalfx4 acc_row2_vec0, acc_row2_vec1;
      xthalfx4 acc_row3_vec0, acc_row3_vec1;
      xthalfx4 acc_row0_vec2, acc_row0_vec3;
      xthalfx4 acc_row1_vec2, acc_row1_vec3;
      xthalfx4 acc_row2_vec2, acc_row2_vec3;
      xthalfx4 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = ZERO_HX4();
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = ZERO_HX4();
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = ZERO_HX4();
          
      xthalfx4 x00, x01, x10, x11;
      xthalfx4 x20, x21, x30, x31;
      xthalfx4 vec0_0, vec0_1;
      xthalfx4 vec1_0, vec1_1;
      xthalfx4 vec2_0, vec2_1;
      xthalfx4 vec3_0, vec3_1;

      AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec, 0);
      AE_LHX4X2_X(vec1_0, vec1_1, (xthalfx8 *)p_vec, sizeof(xthalf)*cols1);
      AE_LHX4X2_X(vec2_0, vec2_1, (xthalfx8 *)p_vec, sizeof(xthalf)*2*cols1);
      AE_LHX4X2_X(vec3_0, vec3_1, (xthalfx8 *)p_vec, sizeof(xthalf)*3*cols1);
      
      AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
      AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
      AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
      AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

      MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
      MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
      MADDQ_H(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
      MADDQ_H(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
      MADDQ_H(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
      MADDQ_H(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
      MADDQ_H(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
      MADDQ_H(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
      
      MADDQ_H(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
      MADDQ_H(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
      MADDQ_H(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
      MADDQ_H(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
      MADDQ_H(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
      MADDQ_H(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
      MADDQ_H(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
      MADDQ_H(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
    
      DSELHX4(y0,y1,acc_row0_vec0,acc_row0_vec1,dsel0);
      y01 = y0 + y1;
      DSELHX4(y2,y3,acc_row0_vec2,acc_row0_vec3,dsel0);
      y23 = y2 + y3;
      DSELHX4(y02,y13,y01,y23,dsel0);
      z0 = z0 + y02;
      z0 = z0 + y13;
      
      DSELHX4(y0,y1,acc_row1_vec0,acc_row1_vec1,dsel0);
      y01 = y0 + y1;
      DSELHX4(y2,y3,acc_row1_vec2,acc_row1_vec3,dsel0);
      y23 = y2 + y3;
      DSELHX4(y02,y13,y01,y23,dsel0);
      z1 = z1 + y02;
      z1 = z1 + y13;
      
      DSELHX4(y0,y1,acc_row2_vec0,acc_row2_vec1,dsel0);
      y01 = y0 + y1;
      DSELHX4(y2,y3,acc_row2_vec2,acc_row2_vec3,dsel0);
      y23 = y2 + y3;
      DSELHX4(y02,y13,y01,y23,dsel0);
      z2 = z2 + y02;
      z2 = z2 + y13;
      
      DSELHX4(y0,y1,acc_row3_vec0,acc_row3_vec1,dsel0);
      y01 = y0 + y1;
      DSELHX4(y2,y3,acc_row3_vec2,acc_row3_vec3,dsel0);
      y23 = y2 + y3;
      DSELHX4(y02,y13,y01,y23,dsel0);
      z3 = z3 + y02;
      z3 = z3 + y13;


     /* 
      y0 = AE_SELH_7531(acc_row0_vec0, acc_row0_vec1);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row0_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row0_vec2, acc_row0_vec3);
      y3 = AE_SELH_6420(acc_row0_vec2, acc_row0_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z0=z0+y02;
	  z0=z0+y13;
	  
      y0 = AE_SELH_7531(acc_row1_vec0, acc_row1_vec1);
      y1 = AE_SELH_6420(acc_row1_vec0, acc_row1_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row1_vec2, acc_row1_vec3);
      y3 = AE_SELH_6420(acc_row1_vec2, acc_row1_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z1=z1+y02;
	  z1=z1+y13;
	  
	  y0 = AE_SELH_7531(acc_row2_vec0, acc_row2_vec1);
      y1 = AE_SELH_6420(acc_row2_vec0, acc_row2_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec2, acc_row2_vec3);
      y3 = AE_SELH_6420(acc_row2_vec2, acc_row2_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z2=z2+y02;
	  z2=z2+y13;
	  
	  y0 = AE_SELH_7531(acc_row3_vec0, acc_row3_vec1);
      y1 = AE_SELH_6420(acc_row3_vec0, acc_row3_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row3_vec2, acc_row3_vec3);
      y3 = AE_SELH_6420(acc_row3_vec2, acc_row3_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z3=z3+y02;
	  z3=z3+y13;
	  */
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
      AE_SHX4IP(z1,(xthalfx4 *)p_out_1, 8);
      AE_SHX4IP(z2,(xthalfx4 *)p_out_2, 8);
      AE_SHX4IP(z3,(xthalfx4 *)p_out_3, 8);
    }
  }
}
else /*  bias==NULL condition */
{

for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {

      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;
      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xthalfx4 z0, z1, z2, z3;
      z0 = z1 = z2 = z3 = (xthalfx4)ZERO_HX4();
              
      xthalfx4 acc_row0_vec0, acc_row0_vec1;
      xthalfx4 acc_row1_vec0, acc_row1_vec1;
      xthalfx4 acc_row2_vec0, acc_row2_vec1;
      xthalfx4 acc_row3_vec0, acc_row3_vec1;
      xthalfx4 acc_row0_vec2, acc_row0_vec3;
      xthalfx4 acc_row1_vec2, acc_row1_vec3;
      xthalfx4 acc_row2_vec2, acc_row2_vec3;
      xthalfx4 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = ZERO_HX4();
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = ZERO_HX4();
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = ZERO_HX4();
          
      xthalfx4 x00, x01, x10, x11;
      xthalfx4 x20, x21, x30, x31;
      xthalfx4 vec0_0, vec0_1;
      xthalfx4 vec1_0, vec1_1;
      xthalfx4 vec2_0, vec2_1;
      xthalfx4 vec3_0, vec3_1;

      AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec, 0);
      AE_LHX4X2_X(vec1_0, vec1_1, (xthalfx8 *)p_vec, sizeof(xthalf)*cols1);
      AE_LHX4X2_X(vec2_0, vec2_1, (xthalfx8 *)p_vec, sizeof(xthalf)*2*cols1);
      AE_LHX4X2_X(vec3_0, vec3_1, (xthalfx8 *)p_vec, sizeof(xthalf)*3*cols1);
      
      AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
      AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
      AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
      AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

      MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
      MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
      MADDQ_H(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
      MADDQ_H(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
      MADDQ_H(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
      MADDQ_H(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
      MADDQ_H(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
      MADDQ_H(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
      
      MADDQ_H(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
      MADDQ_H(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
      MADDQ_H(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
      MADDQ_H(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
      MADDQ_H(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
      MADDQ_H(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
      MADDQ_H(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
      MADDQ_H(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      
      y0 = AE_SELH_7531(acc_row0_vec0, acc_row0_vec1);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row0_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row0_vec2, acc_row0_vec3);
      y3 = AE_SELH_6420(acc_row0_vec2, acc_row0_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z0=z0+y02;
	  z0=z0+y13;
	  
      y0 = AE_SELH_7531(acc_row1_vec0, acc_row1_vec1);
      y1 = AE_SELH_6420(acc_row1_vec0, acc_row1_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row1_vec2, acc_row1_vec3);
      y3 = AE_SELH_6420(acc_row1_vec2, acc_row1_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z1=z1+y02;
	  z1=z1+y13;
	  
	  y0 = AE_SELH_7531(acc_row2_vec0, acc_row2_vec1);
      y1 = AE_SELH_6420(acc_row2_vec0, acc_row2_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec2, acc_row2_vec3);
      y3 = AE_SELH_6420(acc_row2_vec2, acc_row2_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z2=z2+y02;
	  z2=z2+y13;
	  
	  y0 = AE_SELH_7531(acc_row3_vec0, acc_row3_vec1);
      y1 = AE_SELH_6420(acc_row3_vec0, acc_row3_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row3_vec2, acc_row3_vec3);
      y3 = AE_SELH_6420(acc_row3_vec2, acc_row3_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z3=z3+y02;
	  z3=z3+y13;
	  
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
      AE_SHX4IP(z1,(xthalfx4 *)p_out_1, 8);
      AE_SHX4IP(z2,(xthalfx4 *)p_out_2, 8);
      AE_SHX4IP(z3,(xthalfx4 *)p_out_3, 8);
    }
  }
}
  for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13,y0123;

      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));

      /* Init out registers with bias */

      xthalfx4 acc_row0_vec0,acc_row0_vec0_s;
      xthalfx4 acc_row1_vec0,acc_row1_vec0_s;
      xthalfx4 acc_row2_vec0,acc_row2_vec0_s;
      xthalfx4 acc_row3_vec0,acc_row3_vec0_s;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec0_s = acc_row1_vec0_s = acc_row2_vec0_s = acc_row3_vec0_s = ZERO_HX4();
        
      xthalfx4 x00, x01, x10, x11;
      xthalfx4 x20, x21, x30, x31;
      xthalfx4 vec0_0, vec0_1;

      AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec,  0);

      AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
      AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
      AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
      AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

      MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
      MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);

      MADDQ_H(acc_row0_vec0_s, acc_row1_vec0_s, x01, x11, vec0_1);
      MADDQ_H(acc_row2_vec0_s, acc_row3_vec0_s, x21, x31, vec0_1);
    
      ADD_HX4X2(acc_row0_vec0,acc_row1_vec0,acc_row0_vec0,acc_row1_vec0,acc_row0_vec0_s,acc_row1_vec0_s);
      ADD_HX4X2(acc_row2_vec0,acc_row3_vec0,acc_row2_vec0,acc_row3_vec0,acc_row2_vec0_s,acc_row3_vec0_s);

      
      y0 = AE_SELH_7531(acc_row0_vec0, acc_row1_vec0);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row1_vec0);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec0, acc_row3_vec0);
      y3 = AE_SELH_6420(acc_row2_vec0, acc_row3_vec0);
      y23 = y2 + y3;

	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  y0123=y13+y02;
      
      y0 = AE_SELH_6543(y0123, y0123);            
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y0),(ae_int16 *)p_out_0 ,2);			
	  y1 = AE_SELH_7362(y0123, y0123);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y1),(ae_int16 *)p_out_1,2);
      y2=AE_SELHX4IR(y0123,y0123,4);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y2),(ae_int16 *)p_out_2,2);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y0123),(ae_int16 *)p_out_3,2);
    }
  }
 }
 else /* cols1!=8 */
 {
  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {

      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;
      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));
  
      /* Init out registers with bias */
      xthalfx4 z0, z1, z2, z3;
//      xthalfx4 z4, z5, z6, z7;
      z0 = z1 = z2 = z3 = (xthalfx4)ZERO_HX4();
//      z4 = z5 = z6 = z7 = (xthalfx4)ZERO_HX4();
      if(p_bias != NULL)
      {
         ae_int16x4 z0_int,z1_int,z2_int,z3_int;
         z0_int=AE_L16_I((ae_int16 *)&p_bias[m_itr],0);
         z0=AE_MOVHALFX4_FROMF16X4(z0_int);
         z1_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+1],0);
         z1=AE_MOVHALFX4_FROMF16X4(z1_int);
         z2_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+2],0);
         z2=AE_MOVHALFX4_FROMF16X4(z2_int);
         z3_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+3],0);
         z3=AE_MOVHALFX4_FROMF16X4(z3_int);
      }
              
      xthalfx4 acc_row0_vec0, acc_row0_vec1;
      xthalfx4 acc_row1_vec0, acc_row1_vec1;
      xthalfx4 acc_row2_vec0, acc_row2_vec1;
      xthalfx4 acc_row3_vec0, acc_row3_vec1;
      xthalfx4 acc_row0_vec2, acc_row0_vec3;
      xthalfx4 acc_row1_vec2, acc_row1_vec3;
      xthalfx4 acc_row2_vec2, acc_row2_vec3;
      xthalfx4 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = ZERO_HX4();
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = ZERO_HX4();
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = ZERO_HX4();
          
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      {
        xthalfx4 x00, x01, x10, x11;
        xthalfx4 x20, x21, x30, x31;
        xthalfx4 vec0_0, vec0_1;
        xthalfx4 vec1_0, vec1_1;
        xthalfx4 vec2_0, vec2_1;
        xthalfx4 vec3_0, vec3_1;

        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec, 0);
        AE_LHX4X2_X(vec1_0, vec1_1, (xthalfx8 *)p_vec, sizeof(xthalf)*cols1);
        AE_LHX4X2_X(vec2_0, vec2_1, (xthalfx8 *)p_vec, sizeof(xthalf)*2*cols1);
        AE_LHX4X2_X(vec3_0, vec3_1, (xthalfx8 *)p_vec, sizeof(xthalf)*3*cols1);
        
        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
        
        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      }
      y0 = AE_SELH_7531(acc_row0_vec0, acc_row0_vec1);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row0_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row0_vec2, acc_row0_vec3);
      y3 = AE_SELH_6420(acc_row0_vec2, acc_row0_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  z0=z0+y02;
	  z0=z0+y13;
	  
      y0 = AE_SELH_7531(acc_row1_vec0, acc_row1_vec1);
      y1 = AE_SELH_6420(acc_row1_vec0, acc_row1_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row1_vec2, acc_row1_vec3);
      y3 = AE_SELH_6420(acc_row1_vec2, acc_row1_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  z1=z1+y02;
	  z1=z1+y13;
	  
	  y0 = AE_SELH_7531(acc_row2_vec0, acc_row2_vec1);
      y1 = AE_SELH_6420(acc_row2_vec0, acc_row2_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec2, acc_row2_vec3);
      y3 = AE_SELH_6420(acc_row2_vec2, acc_row2_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z2=z2+y02;
	  z2=z2+y13;
	  	  
	  y0 = AE_SELH_7531(acc_row3_vec0, acc_row3_vec1);
      y1 = AE_SELH_6420(acc_row3_vec0, acc_row3_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row3_vec2, acc_row3_vec3);
      y3 = AE_SELH_6420(acc_row3_vec2, acc_row3_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z3=z3+y02;
	  z3=z3+y13;
	  
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
      AE_SHX4IP(z1,(xthalfx4 *)p_out_1, 8);
      AE_SHX4IP(z2,(xthalfx4 *)p_out_2, 8);
      AE_SHX4IP(z3,(xthalfx4 *)p_out_3, 8);
    }
  }

   for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13,y0123;

      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*cols1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*cols1));

      /* Init out registers with bias */

      xthalfx4 acc_row0_vec0,acc_row0_vec0_s;
      xthalfx4 acc_row1_vec0,acc_row1_vec0_s;
      xthalfx4 acc_row2_vec0,acc_row2_vec0_s;
      xthalfx4 acc_row3_vec0,acc_row3_vec0_s;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec0_s = acc_row1_vec0_s = acc_row2_vec0_s = acc_row3_vec0_s = ZERO_HX4();
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      { 
        xthalfx4 x00, x01, x10, x11;
        xthalfx4 x20, x21, x30, x31;
        xthalfx4 vec0_0, vec0_1;

        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec,  0);

        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*cols1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*cols1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*cols1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);

        MADDQ_H(acc_row0_vec0_s, acc_row1_vec0_s, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0_s, acc_row3_vec0_s, x21, x31, vec0_1);
      }
      ADD_HX4X2(acc_row0_vec0,acc_row1_vec0,acc_row0_vec0,acc_row1_vec0,acc_row0_vec0_s,acc_row1_vec0_s);
      ADD_HX4X2(acc_row2_vec0,acc_row3_vec0,acc_row2_vec0,acc_row3_vec0,acc_row2_vec0_s,acc_row3_vec0_s);

      y0 = AE_SELH_7531(acc_row0_vec0, acc_row1_vec0);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row1_vec0);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec0, acc_row3_vec0);
      y3 = AE_SELH_6420(acc_row2_vec0, acc_row3_vec0);
      y23 = y2 + y3;

	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  y0123=y13+y02;
      
      y0 = AE_SELH_6543(y0123, y0123);            
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y0),(ae_int16 *)p_out_0 ,2);			
	  y1 = AE_SELH_7362(y0123, y0123);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y1),(ae_int16 *)p_out_1,2);
      y2=AE_SELHX4IR(y0123,y0123,4);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y2),(ae_int16 *)p_out_2,2);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y0123),(ae_int16 *)p_out_3,2);

    }
  }
 }
}

static inline void spfunc_aligned_cols_mul4_out_offset1
    (xthalf*   p_out
    ,const xthalf*   p_mat1
    ,const xthalf*   p_vec1
    ,const xthalf*   p_bias
    ,WORD32     rows
    ,WORD32     vec_count
    ,WORD32     cols1
    ,WORD32     out_stride
    ,WORD32     row_stride1
    ,WORD32     vec_offset
    )
{
  int vec_itr, m_itr, c_itr;

  
  for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {

      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;
      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*row_stride1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*vec_offset));
  
      /* Init out registers with bias */
      xthalfx4 z0, z1, z2, z3;
      //xthalfx4 z4, z5, z6, z7;
      z0 = z1 = z2 = z3 = (xthalfx4)ZERO_HX4();
      //z4 = z5 = z6 = z7 = (xthalfx4)ZERO_HX4();
      if(p_bias != NULL)
      {
         ae_int16x4 z0_int,z1_int,z2_int,z3_int;
         z0_int=AE_L16_I((ae_int16 *)&p_bias[m_itr],0);
         z0=AE_MOVHALFX4_FROMF16X4(z0_int);
         z1_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+1],0);
         z1=AE_MOVHALFX4_FROMF16X4(z1_int);
         z2_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+2],0);
         z2=AE_MOVHALFX4_FROMF16X4(z2_int);
         z3_int=AE_L16_I((ae_int16 *)&p_bias[m_itr+3],0);
         z3=AE_MOVHALFX4_FROMF16X4(z3_int);
      }
              
      xthalfx4 acc_row0_vec0, acc_row0_vec1;
      xthalfx4 acc_row1_vec0, acc_row1_vec1;
      xthalfx4 acc_row2_vec0, acc_row2_vec1;
      xthalfx4 acc_row3_vec0, acc_row3_vec1;
      xthalfx4 acc_row0_vec2, acc_row0_vec3;
      xthalfx4 acc_row1_vec2, acc_row1_vec3;
      xthalfx4 acc_row2_vec2, acc_row2_vec3;
      xthalfx4 acc_row3_vec2, acc_row3_vec3;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec1 = acc_row1_vec1 = acc_row2_vec1 = acc_row3_vec1 = ZERO_HX4();
      acc_row0_vec2 = acc_row1_vec2 = acc_row2_vec2 = acc_row3_vec2 = ZERO_HX4();
      acc_row0_vec3 = acc_row1_vec3 = acc_row2_vec3 = acc_row3_vec3 = ZERO_HX4();
          
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      {
        xthalfx4 x00, x01, x10, x11;
        xthalfx4 x20, x21, x30, x31;
        xthalfx4 vec0_0, vec0_1;
        xthalfx4 vec1_0, vec1_1;
        xthalfx4 vec2_0, vec2_1;
        xthalfx4 vec3_0, vec3_1;

        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec, 0);
        AE_LHX4X2_X(vec1_0, vec1_1, (xthalfx8 *)p_vec, sizeof(xthalf)*vec_offset);
        AE_LHX4X2_X(vec2_0, vec2_1, (xthalfx8 *)p_vec, sizeof(xthalf)*2*vec_offset);
        AE_LHX4X2_X(vec3_0, vec3_1, (xthalfx8 *)p_vec, sizeof(xthalf)*3*vec_offset);
        
        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*row_stride1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*row_stride1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*row_stride1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x00, x10, vec1_0);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x20, x30, vec1_0);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x00, x10, vec2_0);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x20, x30, vec2_0);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x00, x10, vec3_0);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x20, x30, vec3_0);
        
        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x21, x31, vec0_1);
        MADDQ_H(acc_row0_vec1, acc_row1_vec1, x01, x11, vec1_1);
        MADDQ_H(acc_row2_vec1, acc_row3_vec1, x21, x31, vec1_1);
        MADDQ_H(acc_row0_vec2, acc_row1_vec2, x01, x11, vec2_1);
        MADDQ_H(acc_row2_vec2, acc_row3_vec2, x21, x31, vec2_1);
        MADDQ_H(acc_row0_vec3, acc_row1_vec3, x01, x11, vec3_1);
        MADDQ_H(acc_row2_vec3, acc_row3_vec3, x21, x31, vec3_1);
      }

      y0 = AE_SELH_7531(acc_row0_vec0, acc_row0_vec1);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row0_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row0_vec2, acc_row0_vec3);
      y3 = AE_SELH_6420(acc_row0_vec2, acc_row0_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z0=z0+y02;
	  z0=z0+y13;
	  
      y0 = AE_SELH_7531(acc_row1_vec0, acc_row1_vec1);
      y1 = AE_SELH_6420(acc_row1_vec0, acc_row1_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row1_vec2, acc_row1_vec3);
      y3 = AE_SELH_6420(acc_row1_vec2, acc_row1_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z1=z1+y02;
	  z1=z1+y13;
	  
	  y0 = AE_SELH_7531(acc_row2_vec0, acc_row2_vec1);
      y1 = AE_SELH_6420(acc_row2_vec0, acc_row2_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec2, acc_row2_vec3);
      y3 = AE_SELH_6420(acc_row2_vec2, acc_row2_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z2=z2+y02;
	  z2=z2+y13;
	  	  
	  y0 = AE_SELH_7531(acc_row3_vec0, acc_row3_vec1);
      y1 = AE_SELH_6420(acc_row3_vec0, acc_row3_vec1);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row3_vec2, acc_row3_vec3);
      y3 = AE_SELH_6420(acc_row3_vec2, acc_row3_vec3);
      y23 = y2 + y3;
	  
	  y02 = AE_SELH_7531(y01, y23);
      y13= AE_SELH_6420(y01, y23);
	  z3=z3+y02;
	  z3=z3+y13;
	  
      AE_SHX4IP(z0,(xthalfx4 *)p_out_0, 8);
      AE_SHX4IP(z1,(xthalfx4 *)p_out_1, 8);
      AE_SHX4IP(z2,(xthalfx4 *)p_out_2, 8);
      AE_SHX4IP(z3,(xthalfx4 *)p_out_3, 8);
    }
  }
   for (vec_itr = (vec_count & ~(4-1)); vec_itr < vec_count; vec_itr++)
  {
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13,y0123;

      xthalf *p_out_0 = p_out + (vec_itr + 0)+(m_itr+0)*out_stride;
      xthalf *p_out_1 = p_out + (vec_itr + 0)+(m_itr+1)*out_stride;
      xthalf *p_out_2 = p_out + (vec_itr + 0)+(m_itr+2)*out_stride;
      xthalf *p_out_3 = p_out + (vec_itr + 0)+(m_itr+3)*out_stride;

      xthalf *px = (xthalf *)(p_mat1+(m_itr*row_stride1));
      xthalf *p_vec = (xthalf *)(p_vec1+(vec_itr*vec_offset));

      xthalfx4 acc_row0_vec0,acc_row0_vec0_s;
      xthalfx4 acc_row1_vec0,acc_row1_vec0_s;
      xthalfx4 acc_row2_vec0,acc_row2_vec0_s;
      xthalfx4 acc_row3_vec0,acc_row3_vec0_s;
      acc_row0_vec0 = acc_row1_vec0 = acc_row2_vec0 = acc_row3_vec0 = ZERO_HX4();
      acc_row0_vec0_s = acc_row1_vec0_s = acc_row2_vec0_s = acc_row3_vec0_s = ZERO_HX4();

#pragma no_unroll
      for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++, p_vec+=8, px+=8)
      { 
        xthalfx4 x00, x01, x10, x11;
        xthalfx4 x20, x21, x30, x31;
        xthalfx4 vec0_0, vec0_1;

        AE_LHX4X2_I(vec0_0, vec0_1, (xthalfx8 *)p_vec,  0);

        AE_LHX4X2_I(x00, x01, (xthalfx8 *)px, 0);
        AE_LHX4X2_X(x10, x11, (xthalfx8 *)px, sizeof(xthalf)*row_stride1);
        AE_LHX4X2_X(x20, x21, (xthalfx8 *)px, sizeof(xthalf)*2*row_stride1);
        AE_LHX4X2_X(x30, x31, (xthalfx8 *)px, sizeof(xthalf)*3*row_stride1);

        MADDQ_H(acc_row0_vec0, acc_row1_vec0, x00, x10, vec0_0);
        MADDQ_H(acc_row2_vec0, acc_row3_vec0, x20, x30, vec0_0);

        MADDQ_H(acc_row0_vec0_s, acc_row1_vec0_s, x01, x11, vec0_1);
        MADDQ_H(acc_row2_vec0_s, acc_row3_vec0_s, x21, x31, vec0_1);
      }

      ADD_HX4X2(acc_row0_vec0,acc_row1_vec0,acc_row0_vec0,acc_row1_vec0,acc_row0_vec0_s,acc_row1_vec0_s);
      ADD_HX4X2(acc_row2_vec0,acc_row3_vec0,acc_row2_vec0,acc_row3_vec0,acc_row2_vec0_s,acc_row3_vec0_s);

      y0 = AE_SELH_7531(acc_row0_vec0, acc_row1_vec0);
      y1 = AE_SELH_6420(acc_row0_vec0, acc_row1_vec0);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(acc_row2_vec0, acc_row3_vec0);
      y3 = AE_SELH_6420(acc_row2_vec0, acc_row3_vec0);
      y23 = y2 + y3;

	  y02 = AE_SELH_7531(y01, y23);
      y13 = AE_SELH_6420(y01, y23);
	  y0123=y13+y02;
      
      y0 = AE_SELH_6543(y0123, y0123);            
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y0),(ae_int16 *)p_out_0 ,2);			
	  y1 = AE_SELH_7362(y0123, y0123);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y1),(ae_int16 *)p_out_1,2);
      y2=AE_SELHX4IR(y0123,y0123,4);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y2),(ae_int16 *)p_out_2,2);
	  AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(y0123),(ae_int16 *)p_out_3,2);

    }
  }
}

WORD32 xa_nn_matmul_f16xf16_f16(
    WORD16 * __restrict__ p_out,          
    const WORD16 * __restrict__ p_mat1,   
    const WORD16 * __restrict__ p_vec1,   
    const WORD16 * __restrict__ p_bias,   
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
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
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
       cols(inp_channels) multiple of 8
       rows(out_channels) multiple of 4 */
    if(((out_stride == 1 || out_offset == 1) &&
        ((out_offset & 0x3) == 0 || (out_stride & 0x3) == 0) &&
        (cols1 & 7) == 0) &&
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
      {
       spfunc_cols_mul4_out_stride1
          ((xthalf *)p_out,
           (xthalf *)p_mat1,
           (xthalf *)p_vec1,
           (xthalf *)p_bias,
           rows,
           vec_count,
           cols1,
           out_offset
          );
      }
    else if(out_offset==1)
    
   { 
     spfunc_cols_mul4_out_offset1
          ((xthalf *)p_out,
           (xthalf *)p_mat1,
           (xthalf *)p_vec1,
           (xthalf *)p_bias,
           rows,
           vec_count,
           cols1,
           out_stride
          );
    }

    }
    else if(((out_offset == 1) ||(out_stride ==1)) &&
        ( ((out_stride & 0x3) == 0) || ((out_offset & 0x3)==0)) &&
        ((cols1 & 7) == 0) &&
        ((row_stride1 & 0x7)==0) &&
        ((vec_offset & 0x7)==0) &&
        ALIGNED_PTR(p_mat1, 16) &&
        ALIGNED_PTR(p_vec1, 16) &&
        ALIGNED_PTR(p_out, 16) &&
        ((rows & 0x3) == 0))
    {
    if(out_offset==1)
    {
    spfunc_aligned_cols_mul4_out_offset1
          ((xthalf *)p_out,
           (xthalf *)p_mat1,
           (xthalf *)p_vec1,
           (xthalf *)p_bias,
           rows,
           vec_count,
           cols1,
           out_stride,
           row_stride1,
           vec_offset
          );
    }
    else if(out_stride==1)
    {
        spfunc_aligned_cols_mul4_out_stride1
           ((xthalf *)p_out,
           (xthalf *)p_mat1,
           (xthalf *)p_vec1,
           (xthalf *)p_bias,
           rows,
           vec_count,
           cols1,
           out_offset,
           row_stride1,
           vec_offset
           );

    }

    }
    else
    {
        for (vec_itr = 0; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
        {
            xthalfx4 bias,bias1;
            ae_int16x4 bias_int,bias1_int;
            xthalf *pbias = (xthalf *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
            {
				xthalfx4 y0,y1,y2,y3,y01,y23,y13,y02,y0123;
                xthalfx4 acc_0_0,acc_0_1,acc_1_0,acc_1_1;
                acc_0_0=acc_0_1=acc_1_0=acc_1_1=ZERO_HX4(); 
                xthalfx4 acc_0_0_s,acc_0_1_s,acc_1_0_s,acc_1_1_s;
                acc_0_0_s=acc_0_1_s=acc_1_0_s=acc_1_1_s=ZERO_HX4(); 
                
                xthalfx4 vec_batch_0_0 ;
                xthalfx4 vec_batch_0_1 ;
                xthalfx4 vec_batch_1_0 ;
                xthalfx4 vec_batch_1_1 ;
                xthalfx4 mat1_0_0;
                xthalfx4 mat1_0_1;
                xthalfx4 mat1_1_0;
                xthalfx4 mat1_1_1;

                xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                xthalfx8 *p_vec_batch_1  = (xthalfx8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                xthalfx8 *p_mat1_0 = (xthalfx8 *) &p_mat1[(m_itr+0)*row_stride1];
                xthalfx8 *p_mat1_1 = (xthalfx8 *) &p_mat1[(m_itr+1)*row_stride1];

                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                ae_valignx2 align_mat_1 = AE_LA128_PP(p_mat1_1);

                int cols1_count = cols1- cols1%8;
                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
				
                    AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LAHX4X2_IP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1);
                    AE_LAHX4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    AE_LAHX4X2_IP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
                    MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
                    MADDQ_H(acc_0_0_s, acc_1_0_s, mat1_0_1, mat1_1_1, vec_batch_0_1);
                    MADDQ_H(acc_0_1, acc_1_1, mat1_0_0, mat1_1_0, vec_batch_1_0);
                    MADDQ_H(acc_0_1_s, acc_1_1_s, mat1_0_1, mat1_1_1, vec_batch_1_1);
                }

                if(cols1%8 !=0)
                {				
                    AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8)*2);
                    AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (cols1%8)*2);
                    AE_LAVHX4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8)*2);
                    AE_LAVHX4X2_XP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1, (cols1%8)*2);
                    MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
                    MADDQ_H(acc_0_0_s, acc_1_0_s, mat1_0_1, mat1_1_1, vec_batch_0_1);
                    MADDQ_H(acc_0_1, acc_1_1, mat1_0_0, mat1_1_0, vec_batch_1_0);
                    MADDQ_H(acc_0_1_s, acc_1_1_s, mat1_0_1, mat1_1_1, vec_batch_1_1);
                }
                ADD_HX4X2(acc_0_0,acc_1_0,acc_0_0,acc_1_0,acc_0_0_s,acc_1_0_s);
                ADD_HX4X2(acc_0_1,acc_1_1,acc_0_1,acc_1_1,acc_0_1_s,acc_1_1_s);
                if(p_bias!=NULL)
                {
					AE_L16_IP(bias_int,(ae_int16 *)pbias,2);
					AE_L16_IP(bias1_int,(ae_int16 *) pbias,2);
                    bias=AE_MOVHALFX4_FROMF16X4(bias_int);
                    bias1=AE_MOVHALFX4_FROMF16X4(bias1_int);
                }

				y0 = AE_SELH_7531(acc_0_0, acc_1_0);
				y1 = AE_SELH_6420(acc_0_0, acc_1_0);
				y01 = y0 + y1;

				y2 = AE_SELH_7531(acc_0_1, acc_1_1);
				y3 = AE_SELH_6420(acc_0_1, acc_1_1);
				y23 = y2 + y3;

				y02 = AE_SELH_7531(y01, y23);
				y13= AE_SELH_6420(y01, y23);
				y0123=y13+y02;

                bias=AE_SELH_7362(bias,bias1);
                y0123=y0123+bias;
				
				y0 = AE_SELH_6543(y0123, y0123);
                
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
				
				y1 = AE_SELH_7362(y0123, y0123);
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y1),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride),0);
				
                y2=AE_SELHX4IR(y0123,y0123,4);
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y2),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);
				
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0123),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride),0);
								

            }
            //Remaining row
            for(; m_itr < rows; m_itr++)
            {
				xthalfx4 y0,y1,y2,y3,y01,y23;
				
                xthalfx4 acc_0_0,acc_0_0_s;
                xthalfx4 acc_0_1,acc_0_1_s;
                acc_0_0=acc_0_0_s=acc_0_1=acc_0_1_s=ZERO_HX4();
                xthalfx4 vec_batch_0_0 ;
                xthalfx4 vec_batch_0_1 ;
                xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                xthalfx4 vec_batch_1_0 ;
                xthalfx4 vec_batch_1_1 ;
                xthalfx8 *p_vec_batch_1  = (xthalfx8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
                xthalfx4 mat1_0_0;
                xthalfx4 mat1_0_1;
                xthalfx8 *p_mat1_0 = (xthalfx8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                int cols1_count = cols1- cols1%8;

                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LAHX4X2_IP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1);
                    AE_LAHX4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_0, vec_batch_1_0,mat1_0_0);
                    MADDQ_H(acc_0_0_s, acc_0_1_s,vec_batch_0_1, vec_batch_1_1,mat1_0_1);
					
                }
                ADD_HX4X2(acc_0_0,acc_0_1,acc_0_0,acc_0_1,acc_0_0_s,acc_0_1_s);
                if(cols1%8 != 0)
                {
                    AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8 *2));
                    AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (cols1%8 *2));
                    AE_LAVHX4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8) *2);
                    MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_0, vec_batch_1_0,mat1_0_0);
                    MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_1, vec_batch_1_1,mat1_0_1);
                }

                if(p_bias!=NULL)
                {
					AE_L16_IP(bias_int,(ae_int16 *)pbias, 2);
                    bias=AE_MOVHALFX4_FROMF16X4(bias_int);
                }
				
				
				y0 = AE_SELH_7531(acc_0_0, acc_0_1);
				y1 = AE_SELH_6420(acc_0_0, acc_0_1);
				y01 = y0 + y1;

				y2 = AE_SELH_7531(y01, y01);
				y3 = AE_SELH_6420(y01, y01);
				y23 = y2 + y3;
				y23=y23+bias;
				y0 = AE_SELH_6543(y23, y23);
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
				
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y23),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);
				
            }

        }
        /* Tail loop for vec unroll */
        for(; vec_itr < vec_count; vec_itr++)
        {
            xthalfx4 bias,bias1;
            ae_int16x4 bias_int,bias1_int;
            xthalf *pbias = (xthalf *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
            {
				xthalfx4 y0,y1,y2,y3,y01,y23;
                xthalfx4 acc_0_0,acc_0_0_s;
                xthalfx4 acc_1_0,acc_1_0_s;
                acc_0_0=acc_1_0=acc_0_0_s=acc_1_0_s = ZERO_HX4();
                xthalfx4 vec_batch_0_0 ;
                xthalfx4 vec_batch_0_1 ;
                xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                xthalfx4 mat1_0_0;
                xthalfx4 mat1_0_1;
                xthalfx8 *p_mat1_0 = (xthalfx8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                xthalfx4 mat1_1_0;
                xthalfx4 mat1_1_1;

                xthalfx8 *p_mat1_1 = (xthalfx8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_valignx2 align_mat_1 = AE_LA128_PP(p_mat1_1);
                int cols1_count = cols1 - cols1%8;

                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LAHX4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    AE_LAHX4X2_IP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);		
				    MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
                    MADDQ_H(acc_0_0_s, acc_1_0_s, mat1_0_1, mat1_1_1, vec_batch_0_1);
                }
                ADD_HX4X2(acc_0_0,acc_1_0,acc_0_0,acc_1_0,acc_0_0_s,acc_1_0_s);

                if(cols1%8 != 0)
                {
                    AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8) * 2);
                    AE_LAVHX4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8) * 2);
                    AE_LAVHX4X2_XP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1, (cols1%8) * 2);
					MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
                    MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
                }
				
                if(p_bias!=NULL)
                {
                    AE_L16_IP(bias_int,(ae_int16 *)pbias,2);
                    AE_L16_IP(bias1_int,(ae_int16 *)pbias,2);
                    bias=AE_MOVHALFX4_FROMF16X4(bias_int);
                    bias1=AE_MOVHALFX4_FROMF16X4(bias1_int);
                }
				
				y0 = AE_SELH_7531(acc_0_0, acc_1_0);
				y1 = AE_SELH_6420(acc_0_0, acc_1_0);
				y01 = y0 + y1;

				y2 = AE_SELH_7531(y01, y01);
				y3 = AE_SELH_6420(y01, y01);
				y23 = y2 + y3;
				
				bias=AE_SELH_7362(bias,bias1);
                y23=y23+bias;
				y0 = AE_SELH_6543(y23, y23);
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
				
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y23),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride),0);
				
            }

            for(; m_itr < rows; m_itr++)
            {
				xthalfx4 y0,y1,y2,y3,y01,y23;
                xthalfx4 acc_0_0,acc_0_0_s;
                xthalfx4 acc_dummy_0_0,acc_dummy_0_0_s;
                acc_0_0=acc_0_0_s= acc_dummy_0_0=acc_dummy_0_0_s = ZERO_HX4();
                xthalfx4 vec_batch_0_0 ;
                xthalfx4 vec_batch_0_1 ;
                xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                xthalfx4 mat1_0_0;
                xthalfx4 mat1_0_1;
                xthalfx8 *p_mat1_0 = (xthalfx8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                int cols1_count = cols1 - cols1%8;

                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LAHX4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
					
				    MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_0, mat1_0_0, vec_batch_0_0);
                    MADDQ_H(acc_0_0_s, acc_dummy_0_0_s, mat1_0_1, mat1_0_1, vec_batch_0_1);
                }
                ADD_HX4X2(acc_0_0,acc_dummy_0_0 ,acc_0_0,acc_dummy_0_0,acc_0_0_s,acc_dummy_0_0_s);
                if(cols1%8 != 0)
                {
                    AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8 * 2));
                    AE_LAVHX4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8 * 2));
				    MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_0, mat1_0_0, vec_batch_0_0);
                    MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_1, mat1_0_1, vec_batch_0_1);
                }
                if(p_bias!=(void *)0)
                {

                    AE_L16_IP(bias_int,(ae_int16 *)pbias,2);
                    bias=AE_MOVHALFX4_FROMF16X4(bias_int);
                }
				
				y0 = AE_SELH_7531(acc_0_0, acc_0_0);
				y1 = AE_SELH_6420(acc_0_0, acc_0_0);
				y01 = y0 + y1;

				y2 = AE_SELH_7531(y01, y01);
				y3 = AE_SELH_6420(y01, y01);
				y23 = y2 + y3;
				
				y0=y23+bias;
				AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
				
            }
        }
 }
	
  return 0;
}

#endif
