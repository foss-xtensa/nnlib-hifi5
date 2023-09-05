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

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_matXvec_f16_circ,(
    WORD16 *__restrict__ p_out,
    WORD16 * __restrict__ p_mat,
    WORD16 * __restrict__ p_vec,
    WORD16 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 bias_row_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset))
#else /* #if !HAVE_HP_VFPU */

#define DSELHX4(out0, out1, inp0, inp1, dsel){\
  ae_int16x4 out0_tmp, out1_tmp, inp0_tmp, inp1_tmp;\
  inp0_tmp = AE_MOVF16X4_FROMHALFX4(inp0);\
  inp1_tmp = AE_MOVF16X4_FROMHALFX4(inp1);\
  AE_DSEL16X4(out0_tmp, out1_tmp, inp0_tmp, inp1_tmp, dsel);\
  out0 = AE_MOVHALFX4_FROMF16X4(out0_tmp);\
  out1 = AE_MOVHALFX4_FROMF16X4(out1_tmp);\
}

WORD32 xa_nn_matXvec_f16_circ(
    WORD16 *__restrict__ p_out,            /* output pointer */
    WORD16 *__restrict__ p_mat,            /* matrix: rows x cols */
    WORD16 *__restrict__ p_vec,            /* vec: cols x 1 */
    WORD16 *__restrict__ p_bias,           /* bias TBD: Need array? */
    WORD32 rows,                            /* Number of rows in matrix */
    WORD32 cols,                            /* Number of columns in matrix */
    WORD32 row_offset,                      /* row stride for matrix */
    WORD32 vec_count,                       /* number of vectors: 2, 4, 2n */
    WORD32 vec_offset,                      /* offset from current to next vector */
    WORD32 out_col_offset,
    WORD32 out_row_offset)
{

  WORD32 vec_itr, m_itr, c_itr;

  WORD32 out_offset = out_col_offset;
  WORD32 out_stride = out_row_offset;

  if( ((((unsigned)p_mat) & 15) == 0) && ((((unsigned)p_vec) & 15) == 0) && ((row_offset & 7) == 0) &&  ((cols & 7) == 0) && ((vec_offset & 7) == 0))
  {
    ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));
    /* Aligned case : row-unroll = 4 and vec-unroll = 4 */
    vec_itr = 0;
    for (; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      xthalfx4 bias,bias1,bias2,bias3;
      ae_int16x4 bias_int;

      for(m_itr = 0; m_itr < (rows & ~(4-1)); m_itr += 4)
      {
        xthalfx4 acc_row0_vec0 = ZERO_HX4(), acc_row0_vec1 = ZERO_HX4();
        xthalfx4 acc_row1_vec0 = ZERO_HX4(), acc_row1_vec1 = ZERO_HX4();
        xthalfx4 acc_row2_vec0 = ZERO_HX4(), acc_row2_vec1 = ZERO_HX4();
        xthalfx4 acc_row3_vec0 = ZERO_HX4(), acc_row3_vec1 = ZERO_HX4();
        xthalfx4 acc_row0_vec2 = ZERO_HX4(), acc_row0_vec3 = ZERO_HX4();
        xthalfx4 acc_row1_vec2 = ZERO_HX4(), acc_row1_vec3 = ZERO_HX4();
        xthalfx4 acc_row2_vec2 = ZERO_HX4(), acc_row2_vec3 = ZERO_HX4();
        xthalfx4 acc_row3_vec2 = ZERO_HX4(), acc_row3_vec3 = ZERO_HX4();

        xthalfx4 mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1;
        xthalfx4 vec0_0, vec0_1, vec1_0, vec1_1;
        xthalfx4 vec2_0, vec2_1, vec3_0, vec3_1;
        xthalfx4 y0, y1, y2, y3,y01,y02,y23,y13;

        xthalfx4 z0, z1, z2, z3;
        xthalfx4 z4, z5, z6, z7;
        z0 = z1 = z2 = z3 = ZERO_HX4();
        z4 = z5 = z6 = z7 = ZERO_HX4();
        if(p_bias != NULL)
        {
          z0 = AE_MOVHALFX4_FROMF16X4(AE_MOVDA16(p_bias[vec_itr]));
          z1 = AE_MOVHALFX4_FROMF16X4(AE_MOVDA16(p_bias[vec_itr+1]));
          z2 = AE_MOVHALFX4_FROMF16X4(AE_MOVDA16(p_bias[vec_itr+2]));
          z3 = AE_MOVHALFX4_FROMF16X4(AE_MOVDA16(p_bias[vec_itr+3]));
        }

        xthalfx8 *__restrict__ p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_1  = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_2  = (xthalfx8 *)(p_vec + (vec_itr + 2)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_3  = (xthalfx8 *)(p_vec + (vec_itr + 3)*vec_offset);

        xthalfx8 *__restrict__ p_mat0 = (xthalfx8 *) p_mat;
        xthalfx8 *__restrict__ p_mat1 = (xthalfx8 *) p_mat;
        xthalfx8 *__restrict__ p_mat2 = (xthalfx8 *) p_mat;
        xthalfx8 *__restrict__ p_mat3 = (xthalfx8 *) p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat0, (m_itr+0)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, (m_itr+1)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat2, (m_itr+2)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat3, (m_itr+3)*row_offset*sizeof(WORD16));

        AE_LHX4X2_XC(mat0_0, mat0_1, p_mat0, 8*sizeof(xthalf));
        AE_LHX4X2_XC(mat1_0, mat1_1, p_mat1, 8*sizeof(xthalf));
        AE_LHX4X2_XC(mat2_0, mat2_1, p_mat2, 8*sizeof(xthalf));
        AE_LHX4X2_XC(mat3_0, mat3_1, p_mat3, 8*sizeof(xthalf));

        for(c_itr = 0; c_itr < (cols>>3); c_itr++)
        {
          AE_LHX4X2_IP(vec0_0, vec0_1, p_vec_batch_0, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec1_0, vec1_1, p_vec_batch_1, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec2_0, vec2_1, p_vec_batch_2, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec3_0, vec3_1, p_vec_batch_3, 8*sizeof(xthalf));

          MADDQ_H(acc_row0_vec0, acc_row1_vec0, mat0_0, mat1_0, vec0_0);
          MADDQ_H(acc_row2_vec0, acc_row3_vec0, mat2_0, mat3_0, vec0_0);
          MADDQ_H(acc_row0_vec1, acc_row1_vec1, mat0_0, mat1_0, vec1_0);
          MADDQ_H(acc_row2_vec1, acc_row3_vec1, mat2_0, mat3_0, vec1_0);
          MADDQ_H(acc_row0_vec2, acc_row1_vec2, mat0_0, mat1_0, vec2_0);
          MADDQ_H(acc_row2_vec2, acc_row3_vec2, mat2_0, mat3_0, vec2_0);
          MADDQ_H(acc_row0_vec3, acc_row1_vec3, mat0_0, mat1_0, vec3_0);
          MADDQ_H(acc_row2_vec3, acc_row3_vec3, mat2_0, mat3_0, vec3_0);
          
          MADDQ_H(acc_row0_vec0, acc_row1_vec0, mat0_1, mat1_1, vec0_1);
          MADDQ_H(acc_row2_vec0, acc_row3_vec0, mat2_1, mat3_1, vec0_1);
          MADDQ_H(acc_row0_vec1, acc_row1_vec1, mat0_1, mat1_1, vec1_1);
          MADDQ_H(acc_row2_vec1, acc_row3_vec1, mat2_1, mat3_1, vec1_1);
          MADDQ_H(acc_row0_vec2, acc_row1_vec2, mat0_1, mat1_1, vec2_1);
          MADDQ_H(acc_row2_vec2, acc_row3_vec2, mat2_1, mat3_1, vec2_1);
          MADDQ_H(acc_row0_vec3, acc_row1_vec3, mat0_1, mat1_1, vec3_1);
          MADDQ_H(acc_row2_vec3, acc_row3_vec3, mat2_1, mat3_1, vec3_1);

          AE_LHX4X2_XC(mat0_0, mat0_1, p_mat0, 8*sizeof(xthalf));
          AE_LHX4X2_XC(mat1_0, mat1_1, p_mat1, 8*sizeof(xthalf));
          AE_LHX4X2_XC(mat2_0, mat2_1, p_mat2, 8*sizeof(xthalf));
          AE_LHX4X2_XC(mat3_0, mat3_1, p_mat3, 8*sizeof(xthalf));
        }

        DSELHX4(y0, y1, acc_row0_vec0, acc_row1_vec0, dsel0);
        y01 = y0 + y1;

        DSELHX4(y2, y3, acc_row2_vec0, acc_row3_vec0, dsel0);
        y23 = y2 + y3;
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z0=z0+y02;
        z0=z0+y13;
      
        DSELHX4(y0, y1, acc_row0_vec1, acc_row1_vec1, dsel0);
        y01 = y0 + y1;

        DSELHX4(y2, y3, acc_row2_vec1, acc_row3_vec1, dsel0);
        y23 = y2 + y3;
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z1=z1+y02;
        z1=z1+y13;
      
        DSELHX4(y0, y1, acc_row0_vec2, acc_row1_vec2, dsel0);
        y01 = y0 + y1;

        DSELHX4(y2, y3, acc_row2_vec2, acc_row3_vec2, dsel0);
        y23 = y2 + y3;
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z2=z2+y02;
        z2=z2+y13;
      
        DSELHX4(y0, y1, acc_row0_vec3, acc_row1_vec3, dsel0);
        y01 = y0 + y1;

        DSELHX4(y2, y3, acc_row2_vec3, acc_row3_vec3, dsel0);
        y23 = y2 + y3;
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z3=z3+y02;
        z3=z3+y13;

        AE_S16_0_I(AE_SEL16_6543(AE_MOVF16X4_FROMHALFX4(z0), AE_MOVF16X4_FROMHALFX4(z0)),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVF16X4_FROMHALFX4(z0), AE_MOVF16X4_FROMHALFX4(z0)),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVF16X4_FROMHALFX4(z0), AE_MOVF16X4_FROMHALFX4(z0)),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(z0)                     ,((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVF16X4_FROMHALFX4(z1), AE_MOVF16X4_FROMHALFX4(z1)),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVF16X4_FROMHALFX4(z1), AE_MOVF16X4_FROMHALFX4(z1)),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVF16X4_FROMHALFX4(z1), AE_MOVF16X4_FROMHALFX4(z1)),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(z1)                     ,((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVF16X4_FROMHALFX4(z2), AE_MOVF16X4_FROMHALFX4(z2)),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVF16X4_FROMHALFX4(z2), AE_MOVF16X4_FROMHALFX4(z2)),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVF16X4_FROMHALFX4(z2), AE_MOVF16X4_FROMHALFX4(z2)),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(z2)                     ,((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVF16X4_FROMHALFX4(z3), AE_MOVF16X4_FROMHALFX4(z3)),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVF16X4_FROMHALFX4(z3), AE_MOVF16X4_FROMHALFX4(z3)),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVF16X4_FROMHALFX4(z3), AE_MOVF16X4_FROMHALFX4(z3)),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(z3)                     ,((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 3)*out_stride),0);

      }

      //Remaining rows
      for(; m_itr < rows; m_itr++)
      {
        xthalfx4 y0,y1,y2,y3,y01,y23;
    
        xthalfx4 acc_0_0 = ZERO_HX4();
        xthalfx4 acc_0_1 = ZERO_HX4();
        xthalfx4 acc_1_0 = ZERO_HX4();
        xthalfx4 acc_1_1 = ZERO_HX4();

        xthalfx4 vec_batch_0_0, vec_batch_0_1, vec_batch_1_0, vec_batch_1_1;
        xthalfx4 vec_batch_2_0, vec_batch_2_1, vec_batch_3_0, vec_batch_3_1;

        xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *p_vec_batch_1  = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
        xthalfx8 *p_vec_batch_2  = (xthalfx8 *)(p_vec + (vec_itr + 2)*vec_offset);
        xthalfx8 *p_vec_batch_3  = (xthalfx8 *)(p_vec + (vec_itr + 3)*vec_offset);

        xthalfx4 mat1_0_0, mat1_0_1;
        xthalfx8 *p_mat1_0 = (xthalfx8 *) p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));
        ae_valignx2 align_mat_0;
        AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);

        for(c_itr = 0; c_itr < (cols >> 3); c_itr++)
        {
          AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec_batch_1_0, vec_batch_1_1, p_vec_batch_1, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec_batch_2_0, vec_batch_2_1, p_vec_batch_2, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec_batch_3_0, vec_batch_3_1, p_vec_batch_3, 8*sizeof(xthalf));

          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);

          MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_0, vec_batch_1_0, mat1_0_0);
          MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_1, vec_batch_1_1, mat1_0_1);
          MADDQ_H(acc_1_0, acc_1_1,vec_batch_2_0, vec_batch_3_0, mat1_0_0);
          MADDQ_H(acc_1_0, acc_1_1,vec_batch_2_1, vec_batch_3_1, mat1_0_1);

        }

        if(p_bias!=NULL)
        {
          bias_int = *(ae_int16 *)&p_bias[vec_itr];
          bias     = AE_MOVHALFX4_FROMF16X4(bias_int);
          bias_int = *(ae_int16 *)&p_bias[vec_itr+1];
          bias1    = AE_MOVHALFX4_FROMF16X4(bias_int);
          bias_int = *(ae_int16 *)&p_bias[vec_itr+2];
          bias2    = AE_MOVHALFX4_FROMF16X4(bias_int);
          bias_int = *(ae_int16 *)&p_bias[vec_itr+3];
          bias3    = AE_MOVHALFX4_FROMF16X4(bias_int);                    
        }

        DSELHX4(y0, y1, acc_0_0, acc_0_1, dsel0);
        y01 = y0 + y1;

        DSELHX4(y2, y3, y01, y01, dsel0);
        y23 = y2 + y3;

        y0 = AE_SELH_6543(y23, y23);
        y0 = y0 + bias;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
    
        y23 = y23 + bias1;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y23),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);

        DSELHX4(y0, y1, acc_1_0, acc_1_1, dsel0);
        y01 = y0 + y1;

        DSELHX4(y2, y3, y01, y01, dsel0);
        y23 = y2 + y3;

        y0 = AE_SELH_6543(y23, y23);
        y0 = y0 + bias2;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 0)*out_stride),0);
    
        y23 = y23 + bias3;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y23),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 0)*out_stride),0);
      }
    }
  }
  else
  {
    /* Unaligned case : row-unroll = 2 and vec-unroll = 2. This is necessary as there are only 4 valign registers */
    vec_itr = 0;
    for (; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
    {
      xthalfx4 bias,bias1;
      ae_int16x4 bias_int;

      for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
      {
        xthalfx4 y0,y1,y2,y3,y01,y23,y13,y02,y0123;

        xthalfx4 acc_0_0, acc_0_1, acc_1_0, acc_1_1;
        acc_0_0 = acc_0_1 = acc_1_0 = acc_1_1 = ZERO_HX4();
        xthalfx4 acc_0_0_s, acc_0_1_s, acc_1_0_s, acc_1_1_s;
        acc_0_0_s = acc_0_1_s = acc_1_0_s = acc_1_1_s = ZERO_HX4();
        
        xthalfx4 vec_batch_0_0, vec_batch_0_1, vec_batch_1_0, vec_batch_1_1;
        xthalfx4 mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1;

        xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *p_vec_batch_1  = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);

        xthalfx8 *p_mat1_0 = (xthalfx8 *) p_mat;
        xthalfx8 *p_mat1_1 = (xthalfx8 *) p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, (m_itr+1)*row_offset*sizeof(WORD16));

        ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
        ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
        ae_valignx2 align_mat_0, align_mat_1;
        AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);
        AE_LAHX4X2POS_PC(align_mat_1, p_mat1_1);

        int cols1_count = cols- cols%8;
        for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
        {
          AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
          AE_LAHX4X2_IP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1);
          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
          AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
          MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
          MADDQ_H(acc_0_0_s, acc_1_0_s, mat1_0_1, mat1_1_1, vec_batch_0_1);
          MADDQ_H(acc_0_1, acc_1_1, mat1_0_0, mat1_1_0, vec_batch_1_0);
          MADDQ_H(acc_0_1_s, acc_1_1_s, mat1_0_1, mat1_1_1, vec_batch_1_1);
        }

        if(cols%8 !=0)
        {    
          AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols%8)*2);
          AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (cols%8)*2);
          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
          AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
          MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
          MADDQ_H(acc_0_0_s, acc_1_0_s, mat1_0_1, mat1_1_1, vec_batch_0_1);
          MADDQ_H(acc_0_1, acc_1_1, mat1_0_0, mat1_1_0, vec_batch_1_0);
          MADDQ_H(acc_0_1_s, acc_1_1_s, mat1_0_1, mat1_1_1, vec_batch_1_1);
        }
        ADD_HX4X2(acc_0_0,acc_1_0,acc_0_0,acc_1_0,acc_0_0_s,acc_1_0_s);
        ADD_HX4X2(acc_0_1,acc_1_1,acc_0_1,acc_1_1,acc_0_1_s,acc_1_1_s);

        if(p_bias!=NULL)
        {
          bias_int = *(ae_int16 *)&p_bias[vec_itr];
          bias     = AE_MOVHALFX4_FROMF16X4(bias_int);
          bias_int = *(ae_int16 *)&p_bias[vec_itr+1];
          bias1    = AE_MOVHALFX4_FROMF16X4(bias_int);
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

        y0 = AE_SELH_6543(y0123, y0123);
        y0 = y0 + bias;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
    
        y1 = AE_SELH_7362(y0123, y0123);
        y1 = y1 + bias;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y1),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride),0);
    
        y2=AE_SELHX4IR(y0123,y0123,4);

        y2 = y2 + bias1;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y2),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);
    
        y0123 = y0123 + bias1;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0123),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride),0);
      }

      //Remaining row
      for(; m_itr < rows; m_itr++)
      {
        xthalfx4 y0,y1,y2,y3,y01,y23;
    
        xthalfx4 acc_0_0 = ZERO_HX4();
        xthalfx4 acc_0_1 = ZERO_HX4();
        xthalfx4 vec_batch_0_0, vec_batch_0_1, vec_batch_1_0, vec_batch_1_1;
        xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *p_vec_batch_1  = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
        ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
        ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
        xthalfx4 mat1_0_0, mat1_0_1;
        xthalfx8 *p_mat1_0 = (xthalfx8 *) p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));

        ae_valignx2 align_mat_0;
        AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);

        int cols1_count = cols- cols%8;

        for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
        {
          AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
          AE_LAHX4X2_IP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1);
          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
          MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_0, vec_batch_1_0,mat1_0_0);
          MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_1, vec_batch_1_1,mat1_0_1);
      
        }
        if(cols%8 != 0)
        {
          AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols%8 *2));
          AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (cols%8 *2));
          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
          MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_0, vec_batch_1_0,mat1_0_0);
          MADDQ_H(acc_0_0, acc_0_1,vec_batch_0_1, vec_batch_1_1,mat1_0_1);
        }

        if(p_bias!=NULL)
        {
          bias_int = *(ae_int16 *)&p_bias[vec_itr];
          bias     = AE_MOVHALFX4_FROMF16X4(bias_int);
          bias_int = *(ae_int16 *)&p_bias[vec_itr+1];
          bias1    = AE_MOVHALFX4_FROMF16X4(bias_int);
        }

        y0 = AE_SELH_7531(acc_0_0, acc_0_1);
        y1 = AE_SELH_6420(acc_0_0, acc_0_1);
        y01 = y0 + y1;

        y2 = AE_SELH_7531(y01, y01);
        y3 = AE_SELH_6420(y01, y01);
        y23 = y2 + y3;

        y0 = AE_SELH_6543(y23, y23);
        y0 = y0 + bias;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
    
        y23 = y23 + bias1;
        AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y23),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);
      }
    }
  }

  /* Tail loop for vec unroll. Vector counter continued from aligned/unaligned part as both have vec_unroll=2 */
  for(; vec_itr < vec_count; vec_itr++)
  {
    xthalfx4 bias;
    ae_int16x4 bias_int;
    xthalf *pbias = (xthalf *) p_bias;
    xthalfx4 bias1;

    m_itr = 0;
    for(; m_itr < (rows & ~(2-1)); m_itr += 2)
    {
      xthalfx4 y0,y1,y2,y3,y01,y23;
      xthalfx4 acc_0_0 = ZERO_HX4();
      xthalfx4 acc_1_0 = ZERO_HX4();
      xthalfx4 vec_batch_0_0;
      xthalfx4 vec_batch_0_1;
      xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
      ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
      xthalfx4 mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1 ;

      xthalfx8 *p_mat1_0 = (xthalfx8 *) p_mat;
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));
      ae_valignx2 align_mat_0;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);

      xthalfx8 *p_mat1_1 = (xthalfx8 *) p_mat;
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, (m_itr+1)*row_offset*sizeof(WORD16));
      ae_valignx2 align_mat_1;
      AE_LAHX4X2POS_PC(align_mat_1, p_mat1_1);

      int cols1_count = cols - cols%8;

      for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {
        AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);  
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
      }
      if(cols%8 != 0)
      {
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols%8) * 2);
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
      }
  
      if(p_bias!=NULL)
      {
        bias_int = *(ae_int16 *)&p_bias[vec_itr];
        bias  = AE_MOVHALFX4_FROMF16X4(bias_int);
        bias1 = AE_MOVHALFX4_FROMF16X4(bias_int);
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
      xthalfx4 acc_0_0 = ZERO_HX4();
      xthalfx4 acc_dummy_0_0  = ZERO_HX4();
      xthalfx4 vec_batch_0_0;
      xthalfx4 vec_batch_0_1;
      xthalfx8 *p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
      ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
      xthalfx4 mat1_0_0;
      xthalfx4 mat1_0_1;
      xthalfx8 *p_mat1_0 = (xthalfx8 *) p_mat;
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));
      ae_valignx2 align_mat_0;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);

      int cols1_count = cols - cols%8;

      for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {
        AE_LAHX4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
    
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_0, mat1_0_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_1, mat1_0_1, vec_batch_0_1);
      }

      if(cols%8 != 0)
      {
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols%8 * 2));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_0, mat1_0_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_1, mat1_0_1, vec_batch_0_1);
      }

      y0 = AE_SELH_7531(acc_0_0, acc_0_0);
      y1 = AE_SELH_6420(acc_0_0, acc_0_0);
      y01 = y0 + y1;

      y2 = AE_SELH_7531(y01, y01);
      y3 = AE_SELH_6420(y01, y01);
      y23 = y2 + y3;

      if(p_bias!=(void *)0)
      {
        pbias = (xthalf *)&p_bias[vec_itr];
        AE_L16_IP(bias_int,(ae_int16 *)pbias,2);
        bias=AE_MOVHALFX4_FROMF16X4(bias_int);
      }
      y0=y23+bias;
      AE_S16_0_I(AE_MOVF16X4_FROMHALFX4(y0),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
    }
  }
  
  return 0;

  /* Following serves as ref code 

  xthalf *p_out_tmp;
  xthalf *p_bias_f16 = (xthalf *)p_bias;
  xthalf *p_out_f16 = (xthalf *)p_out;
  
  for(vec_itr = 0; vec_itr < vec_count; vec_itr++)
  {
    for(m_itr = 0; m_itr < (rows); m_itr ++)
    {
      xthalf bias = p_bias_f16[vec_itr];
      xthalf acc_0_0 = ZERO_H();
      xthalf vec_batch_0;
      xthalf *p_vec_batch_0  = (xthalf *)(&p_vec[(vec_itr)*vec_offset]);
      xthalf mat_0;
      xthalf *p_mat_0 = (xthalf *) p_mat;
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat_0, (m_itr)*row_offset*sizeof(WORD16));

      for(c_itr = 0; c_itr < cols; c_itr++)
      {
        AE_LHIP(vec_batch_0, p_vec_batch_0, sizeof(xthalf));
        AE_LHXC(mat_0, p_mat_0, sizeof(xthalf));
        MADD_H(acc_0_0, vec_batch_0, mat_0);
      }
      acc_0_0 = ADD_H(acc_0_0,bias);
      p_out_tmp = &(p_out_f16[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]);
      AE_SHIP(acc_0_0, p_out_tmp,0);
    }
  }
  */
}
#endif /* #if !HAVE_HP_VFPU */
