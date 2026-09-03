/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_fpu.h"
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
    WORD32 out_row_offset,
    WORD16 out_activation_min,
    WORD16 out_activation_max))
#else /* #if !HAVE_HP_VFPU */

#define DSELHX4(out0, out1, inp0, inp1, dsel){\
  ae_int16x4 out0_tmp, out1_tmp, inp0_tmp, inp1_tmp;\
  inp0_tmp = AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(inp0));\
  inp1_tmp = AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(inp1));\
  AE_DSEL16X4(out0_tmp, out1_tmp, inp0_tmp, inp1_tmp, dsel);\
  out0 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(out0_tmp));\
  out1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(out1_tmp));\
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
    WORD32 out_row_offset,
    WORD16 out_activation_min,
    WORD16 out_activation_max)
{

  WORD32 vec_itr, m_itr, c_itr;

  WORD32 out_offset = out_col_offset;
  WORD32 out_stride = out_row_offset;
  ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));
  xthalfx4 activation_min = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_activation_min)));
  xthalfx4 activation_max = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_activation_max)));

#define CLAMP_HX4(value) MAX_HX4(MIN_HX4(value, activation_max), activation_min)

  if( ((((unsigned)p_mat) & 15) == 0) && ((((unsigned)p_vec) & 15) == 0) && ((row_offset & 7) == 0) &&  ((cols & 7) == 0) && ((vec_offset & 7) == 0))
  {
    /* Aligned case : row-unroll = 4 and vec-unroll = 4 */
    vec_itr = 0;
    for (; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      xthalfx4 bias,bias1,bias2,bias3;
      ae_int16 bias_int;

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
        //xthalfx4 z4, z5, z6, z7;
        z0 = z1 = z2 = z3 = ZERO_HX4();
        //z4 = z5 = z6 = z7 = ZERO_HX4();
        if(p_bias != NULL)
        {
          z0 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr])));
          z1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+1])));
          z2 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+2])));
          z3 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+3])));
        }

        xthalfx8 *__restrict__ p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_1  = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_2  = (xthalfx8 *)(p_vec + (vec_itr + 2)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_3  = (xthalfx8 *)(p_vec + (vec_itr + 3)*vec_offset);

        ae_int16x4 *p16x4_mat0 = (ae_int16x4 *)p_mat;
        ae_int16x4 *p16x4_mat1 = (ae_int16x4 *)p_mat;
        ae_int16x4 *p16x4_mat2 = (ae_int16x4 *)p_mat;
        ae_int16x4 *p16x4_mat3 = (ae_int16x4 *)p_mat;
        AE_ADDCIRC16X4_XC(p16x4_mat0, (m_itr+0)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC(p16x4_mat1, (m_itr+1)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC(p16x4_mat2, (m_itr+2)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC(p16x4_mat3, (m_itr+3)*row_offset*sizeof(WORD16));
        xthalfx8 *__restrict__ p_mat0 = (xthalfx8 *) p16x4_mat0;
        xthalfx8 *__restrict__ p_mat1 = (xthalfx8 *) p16x4_mat1;
        xthalfx8 *__restrict__ p_mat2 = (xthalfx8 *) p16x4_mat2;
        xthalfx8 *__restrict__ p_mat3 = (xthalfx8 *) p16x4_mat3;

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
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec0, acc_row3_vec0, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z0=ADD_HX4(z0,y02);
        z0=ADD_HX4(z0,y13);
      
        DSELHX4(y0, y1, acc_row0_vec1, acc_row1_vec1, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec1, acc_row3_vec1, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z1=ADD_HX4(z1,y02);
        z1=ADD_HX4(z1,y13);
      
        DSELHX4(y0, y1, acc_row0_vec2, acc_row1_vec2, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec2, acc_row3_vec2, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z2=ADD_HX4(z2, y02);
        z2=ADD_HX4(z2,y13);
      
        DSELHX4(y0, y1, acc_row0_vec3, acc_row1_vec3, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec3, acc_row3_vec3, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z3=ADD_HX4(z3,y02);
        z3=ADD_HX4(z3,y13);

        z0 = CLAMP_HX4(z0);
        z1 = CLAMP_HX4(z1);
        z2 = CLAMP_HX4(z2);
        z3 = CLAMP_HX4(z3);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))                     ,((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))                     ,((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))                     ,((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))                     ,((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 3)*out_stride),0);

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
        ae_int16x4 *p16x4_mat1_0 = (ae_int16x4 *)p_mat;
        AE_ADDCIRC16X4_XC(p16x4_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));
        xthalfx8 *p_mat1_0 = (xthalfx8 *) p16x4_mat1_0;
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
          bias     = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16(bias_int));
          bias_int = *(ae_int16 *)&p_bias[vec_itr+1];
          bias1    = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16(bias_int));
          bias_int = *(ae_int16 *)&p_bias[vec_itr+2];
          bias2    = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16(bias_int));
          bias_int = *(ae_int16 *)&p_bias[vec_itr+3];
          bias3    = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16(bias_int));                    
        }

        DSELHX4(y0, y1, acc_0_0, acc_0_1, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, y01, y01, dsel0);
        y23 = ADD_HX4(y2, y3);

        y0 = AE_SELH_6543(y23, y23);
        y0 = ADD_HX4(y0, bias);
  y0 = CLAMP_HX4(y0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)),((ae_int16 *)p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride),0);
    
        y23 = ADD_HX4(y23, bias1);
  y23 = CLAMP_HX4(y23);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)),((ae_int16 *)p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride),0);

        DSELHX4(y0, y1, acc_1_0, acc_1_1, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, y01, y01, dsel0);
        y23 = ADD_HX4(y2, y3);

        y0 = AE_SELH_6543(y23, y23);
        y0 = ADD_HX4(y0, bias2);
        y0 = CLAMP_HX4(y0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)),((ae_int16 *)p_out + (vec_itr + 2)*out_offset + (m_itr + 0)*out_stride),0);
    
        y23 = ADD_HX4(y23, bias3);
        y23 = CLAMP_HX4(y23);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)),((ae_int16 *)p_out + (vec_itr + 3)*out_offset + (m_itr + 0)*out_stride),0);
      }
    }
  }
  else
  {
    /* Unaligned case : row-unroll = 4 and vec-unroll = 4. This is necessary as there are only 4 valign registers (for mat)*/
    vec_itr = 0;
    for (; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      // Pre-calculate base output pointers for all 4 vectors
      ae_int16 *p_out_vec0 = (ae_int16 *)p_out + (vec_itr + 0)*out_offset;
      ae_int16 *p_out_vec1 = (ae_int16 *)p_out + (vec_itr + 1)*out_offset;
      ae_int16 *p_out_vec2 = (ae_int16 *)p_out + (vec_itr + 2)*out_offset;
      ae_int16 *p_out_vec3 = (ae_int16 *)p_out + (vec_itr + 3)*out_offset;

      int cols1_count = cols - cols%8;

      for(m_itr = 0; m_itr < (rows & ~(4-1)); m_itr += 4)
      {
        xthalfx4 acc_row0_vec0 = ZERO_HX4();
        xthalfx4 acc_row1_vec0 = ZERO_HX4();
        xthalfx4 acc_row2_vec0 = ZERO_HX4();
        xthalfx4 acc_row3_vec0 = ZERO_HX4();
        
        xthalfx4 acc_row0_vec1 = ZERO_HX4();
        xthalfx4 acc_row1_vec1 = ZERO_HX4();
        xthalfx4 acc_row2_vec1 = ZERO_HX4();
        xthalfx4 acc_row3_vec1 = ZERO_HX4();
        
        xthalfx4 acc_row0_vec2 = ZERO_HX4();
        xthalfx4 acc_row1_vec2 = ZERO_HX4();
        xthalfx4 acc_row2_vec2 = ZERO_HX4();
        xthalfx4 acc_row3_vec2 = ZERO_HX4();
        
        xthalfx4 acc_row0_vec3 = ZERO_HX4();
        xthalfx4 acc_row1_vec3 = ZERO_HX4();
        xthalfx4 acc_row2_vec3 = ZERO_HX4();
        xthalfx4 acc_row3_vec3 = ZERO_HX4();

        xthalfx4 mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1;
        xthalfx4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;

        // Load bias values
        xthalfx4 z0 = ZERO_HX4(), z1 = ZERO_HX4(), z2 = ZERO_HX4(), z3 = ZERO_HX4();
        if(p_bias != NULL)
        {
          z0 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+0])));
          z1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+1])));
          z2 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+2])));
          z3 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+3])));
        }

        xthalfx8 *__restrict__ p_vec_batch_0  = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_1  = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_2  = (xthalfx8 *)(p_vec + (vec_itr + 2)*vec_offset);
        xthalfx8 *__restrict__ p_vec_batch_3  = (xthalfx8 *)(p_vec + (vec_itr + 3)*vec_offset);

        ae_int16x4 *p16x4_mat0 = (ae_int16x4 *)p_mat;
        ae_int16x4 *p16x4_mat1 = (ae_int16x4 *)p_mat;
        ae_int16x4 *p16x4_mat2 = (ae_int16x4 *)p_mat;
        ae_int16x4 *p16x4_mat3 = (ae_int16x4 *)p_mat;
        AE_ADDCIRC16X4_XC(p16x4_mat0, (m_itr+0)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC(p16x4_mat1, (m_itr+1)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC(p16x4_mat2, (m_itr+2)*row_offset*sizeof(WORD16));
        AE_ADDCIRC16X4_XC(p16x4_mat3, (m_itr+3)*row_offset*sizeof(WORD16));
        xthalfx8 *__restrict__ p_mat0 = (xthalfx8 *) p16x4_mat0;
        xthalfx8 *__restrict__ p_mat1 = (xthalfx8 *) p16x4_mat1;
        xthalfx8 *__restrict__ p_mat2 = (xthalfx8 *) p16x4_mat2;
        xthalfx8 *__restrict__ p_mat3 = (xthalfx8 *) p16x4_mat3;

        ae_valignx2 align_mat_0, align_mat_1, align_mat_2, align_mat_3;
        AE_LAHX4X2POS_PC(align_mat_0, p_mat0);
        AE_LAHX4X2POS_PC(align_mat_1, p_mat1);
        AE_LAHX4X2POS_PC(align_mat_2, p_mat2);
        AE_LAHX4X2POS_PC(align_mat_3, p_mat3);

        for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
        {
          AE_LHX4X2_IP(vec0_0, vec0_1, p_vec_batch_0, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec1_0, vec1_1, p_vec_batch_1, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec2_0, vec2_1, p_vec_batch_2, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec3_0, vec3_1, p_vec_batch_3, 8*sizeof(xthalf));

          AE_LAHX4X2_IC(mat0_0, mat0_1, align_mat_0, p_mat0);
          AE_LAHX4X2_IC(mat1_0, mat1_1, align_mat_1, p_mat1);
          AE_LAHX4X2_IC(mat2_0, mat2_1, align_mat_2, p_mat2);
          AE_LAHX4X2_IC(mat3_0, mat3_1, align_mat_3, p_mat3);
          
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
        }

        ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
        ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
        ae_valignx2 align_vec2 = AE_LA128_PP(p_vec_batch_2);
        ae_valignx2 align_vec3 = AE_LA128_PP(p_vec_batch_3);
        int rem_elm = cols&0x7;
        ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
        ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
        if(rem_elm <= 4)
        {
          mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
          mask2 = AE_ZERO64();
        }
        else{
          mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
        }
        if(rem_elm)
        {
          AE_LAVHX4X2_XP(vec0_0, vec0_1, align_vec0, p_vec_batch_0, (rem_elm<<1));
          AE_LAVHX4X2_XP(vec1_0, vec1_1, align_vec1, p_vec_batch_1, (rem_elm<<1));
          AE_LAVHX4X2_XP(vec2_0, vec2_1, align_vec2, p_vec_batch_2, (rem_elm<<1));
          AE_LAVHX4X2_XP(vec3_0, vec3_1, align_vec3, p_vec_batch_3, (rem_elm<<1));
          AE_LAHX4X2_IC(mat0_0, mat0_1, align_mat_0, p_mat0);
          AE_LAHX4X2_IC(mat1_0, mat1_1, align_mat_1, p_mat1);
          AE_LAHX4X2_IC(mat2_0, mat2_1, align_mat_2, p_mat2);
          AE_LAHX4X2_IC(mat3_0, mat3_1, align_mat_3, p_mat3);
          mat0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat0_0)), mask1)));
          mat0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat0_1)), mask2)));
          mat1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0)), mask1)));
          mat1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_1)), mask2)));
          mat2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat2_0)), mask1)));
          mat2_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat2_1)), mask2)));
          mat3_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat3_0)), mask1)));
          mat3_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat3_1)), mask2)));
                    
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
        }

        xthalfx4 y0, y1, y2, y3, y01, y23, y02, y13;
        DSELHX4(y0, y1, acc_row0_vec0, acc_row1_vec0, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec0, acc_row3_vec0, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z0=ADD_HX4(z0,y02);
        z0=ADD_HX4(z0,y13);
      
        DSELHX4(y0, y1, acc_row0_vec1, acc_row1_vec1, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec1, acc_row3_vec1, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z1=ADD_HX4(z1,y02);
        z1=ADD_HX4(z1,y13);
      
        DSELHX4(y0, y1, acc_row0_vec2, acc_row1_vec2, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec2, acc_row3_vec2, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z2=ADD_HX4(z2, y02);
        z2=ADD_HX4(z2,y13);
      
        DSELHX4(y0, y1, acc_row0_vec3, acc_row1_vec3, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, acc_row2_vec3, acc_row3_vec3, dsel0);
        y23 = ADD_HX4(y2, y3);
      
        DSELHX4(y02, y13, y01, y23, dsel0);
        z3=ADD_HX4(z3,y02);
        z3=ADD_HX4(z3,y13);

        z0 = CLAMP_HX4(z0);
        z1 = CLAMP_HX4(z1);
        z2 = CLAMP_HX4(z2);
        z3 = CLAMP_HX4(z3);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))),(p_out_vec0 + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))),(p_out_vec0 + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))),(p_out_vec0 + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z0))                     ,(p_out_vec0 + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))),(p_out_vec1 + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))),(p_out_vec1 + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))),(p_out_vec1 + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z1))                     ,(p_out_vec1 + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))),(p_out_vec2 + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))),(p_out_vec2 + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))),(p_out_vec2 + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z2))                     ,(p_out_vec2 + (m_itr + 3)*out_stride),0);

        AE_S16_0_I(AE_SEL16_6543(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))),(p_out_vec3 + (m_itr + 0)*out_stride),0);
        AE_S16_0_I(AE_SEL16_5432(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))),(p_out_vec3 + (m_itr + 1)*out_stride),0);
        AE_S16_0_I(AE_SEL16_4321(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3)), AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))),(p_out_vec3 + (m_itr + 2)*out_stride),0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(z3))                     ,(p_out_vec3 + (m_itr + 3)*out_stride),0);
      }

      //Remaining rows
      for(; m_itr < rows; m_itr++)
      {
        xthalfx4 y0, y1, y2, y3, y01, y23;
        xthalfx4 acc_0_0 = ZERO_HX4();
        xthalfx4 acc_0_1 = ZERO_HX4();
        xthalfx4 acc_1_0 = ZERO_HX4();
        xthalfx4 acc_1_1 = ZERO_HX4();
        
        xthalfx4 vec_batch_0_0, vec_batch_0_1;
        xthalfx4 vec_batch_1_0, vec_batch_1_1;
        xthalfx4 vec_batch_2_0, vec_batch_2_1;
        xthalfx4 vec_batch_3_0, vec_batch_3_1;
        
        xthalfx8 *p_vec_batch_0 = (xthalfx8 *)(p_vec + (vec_itr + 0)*vec_offset);
        xthalfx8 *p_vec_batch_1 = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
        xthalfx8 *p_vec_batch_2 = (xthalfx8 *)(p_vec + (vec_itr + 2)*vec_offset);
        xthalfx8 *p_vec_batch_3 = (xthalfx8 *)(p_vec + (vec_itr + 3)*vec_offset);
        
        xthalfx4 mat1_0_0, mat1_0_1;
        
        ae_int16x4 *p16x4_mat1_0 = (ae_int16x4 *)p_mat;
        AE_ADDCIRC16X4_XC(p16x4_mat1_0, (m_itr+0)*row_offset*sizeof(WORD16));
        xthalfx8 *p_mat1_0 = (xthalfx8 *)p16x4_mat1_0;

        ae_valignx2 align_mat_0;
        AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);

        for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
        {
          AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec_batch_1_0, vec_batch_1_1, p_vec_batch_1, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec_batch_2_0, vec_batch_2_1, p_vec_batch_2, 8*sizeof(xthalf));
          AE_LHX4X2_IP(vec_batch_3_0, vec_batch_3_1, p_vec_batch_3, 8*sizeof(xthalf));
          
          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
          
          MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_0, vec_batch_1_0, mat1_0_0);
          MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_1, vec_batch_1_1, mat1_0_1);
          MADDQ_H(acc_1_0, acc_1_1, vec_batch_2_0, vec_batch_3_0, mat1_0_0);
          MADDQ_H(acc_1_0, acc_1_1, vec_batch_2_1, vec_batch_3_1, mat1_0_1);
        }
        ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
        ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
        ae_valignx2 align_vec2 = AE_LA128_PP(p_vec_batch_2);
        ae_valignx2 align_vec3 = AE_LA128_PP(p_vec_batch_3);
        int rem_elm = cols&0x7;
        ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
        ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
        if(rem_elm <= 4)
        {
            mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
            mask2 = AE_ZERO64();
        }
        else
        {
            mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
        }
        if(rem_elm)
        {
          AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (rem_elm<<1));
          AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (rem_elm<<1));
          AE_LAVHX4X2_XP(vec_batch_2_0, vec_batch_2_1, align_vec2, p_vec_batch_2, (rem_elm<<1));
          AE_LAVHX4X2_XP(vec_batch_3_0, vec_batch_3_1, align_vec3, p_vec_batch_3, (rem_elm<<1));
          AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
          mat1_0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_0)), mask1)));
          mat1_0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_1)), mask2)));
          
          MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_0, vec_batch_1_0, mat1_0_0);
          MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_1, vec_batch_1_1, mat1_0_1);
          MADDQ_H(acc_1_0, acc_1_1, vec_batch_2_0, vec_batch_3_0, mat1_0_0);
          MADDQ_H(acc_1_0, acc_1_1, vec_batch_2_1, vec_batch_3_1, mat1_0_1);
        }

        // Load bias for all 4 vectors
        xthalfx4 bias = ZERO_HX4(), bias1 = ZERO_HX4();
        xthalfx4 bias2 = ZERO_HX4(), bias3 = ZERO_HX4();
        
        if(p_bias != NULL)
        {
          bias = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+0])));
          bias1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+1])));
          bias2 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+2])));
          bias3 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+3])));
        }

        DSELHX4(y0, y1, acc_0_0, acc_0_1, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, y01, y01, dsel0);
        y23 = ADD_HX4(y2, y3);

        y0 = AE_SELH_6543(y23, y23);
        y0 = ADD_HX4(y0, bias);
  y0 = CLAMP_HX4(y0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)),(p_out_vec0 + (m_itr + 0)*out_stride),0);
    
        y23 = ADD_HX4(y23, bias1);
  y23 = CLAMP_HX4(y23);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)),(p_out_vec1 + (m_itr + 0)*out_stride),0);

        DSELHX4(y0, y1, acc_1_0, acc_1_1, dsel0);
        y01 = ADD_HX4(y0, y1);

        DSELHX4(y2, y3, y01, y01, dsel0);
        y23 = ADD_HX4(y2, y3);

        y0 = AE_SELH_6543(y23, y23);
        y0 = ADD_HX4(y0, bias2);
        y0 = CLAMP_HX4(y0);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)),(p_out_vec2 + (m_itr + 0)*out_stride),0);
    
        y23 = ADD_HX4(y23, bias3);
        y23 = CLAMP_HX4(y23);
        AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)),(p_out_vec3 + (m_itr + 0)*out_stride),0);
      }
    }
  }

  /* Tail loop for vec unroll */
  /* Handle remaining vectors in pairs when possible */
  for(; vec_itr < (vec_count & ~1); vec_itr += 2)
  {
    xthalfx4 bias = ZERO_HX4(), bias1 = ZERO_HX4();

    if(p_bias != NULL)
    {
      bias = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr])));
      bias1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr+1])));
    }

    /* Hoist cols aligned count once per vector pair */
    int cols1_count = cols - (cols & 7);

    /* Precompute vector base pointers and output base pointers to avoid repeated casts
      and pointer arithmetic inside the rows loop */
    xthalfx8 *p_vec_batch_0_base = (xthalfx8 *)(p_vec + (vec_itr)*vec_offset);
    xthalfx8 *p_vec_batch_1_base = (xthalfx8 *)(p_vec + (vec_itr + 1)*vec_offset);
    ae_int16 *p_out_vec0 = (ae_int16 *)p_out + vec_itr*out_offset;
    ae_int16 *p_out_vec1 = (ae_int16 *)p_out + (vec_itr+1)*out_offset;

    m_itr = 0;
    
    /* Process 4 rows at a time */
    for(; m_itr < (rows & ~3); m_itr += 4)
    {
      xthalfx4 acc_0_0 = ZERO_HX4(), acc_1_0 = ZERO_HX4(), acc_2_0 = ZERO_HX4(), acc_3_0 = ZERO_HX4();
      xthalfx4 acc_0_1 = ZERO_HX4(), acc_1_1 = ZERO_HX4(), acc_2_1 = ZERO_HX4(), acc_3_1 = ZERO_HX4();

      xthalfx8 *p_vec_batch_0 = p_vec_batch_0_base;
      xthalfx8 *p_vec_batch_1 = p_vec_batch_1_base;

      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat;
      ae_int16x4 *p16x4_mat_1 = (ae_int16x4 *)p_mat;
      ae_int16x4 *p16x4_mat_2 = (ae_int16x4 *)p_mat;
      ae_int16x4 *p16x4_mat_3 = (ae_int16x4 *)p_mat;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr + 0) * row_offset * sizeof(WORD16));
      AE_ADDCIRC16X4_XC(p16x4_mat_1, (m_itr + 1) * row_offset * sizeof(WORD16));
      AE_ADDCIRC16X4_XC(p16x4_mat_2, (m_itr + 2) * row_offset * sizeof(WORD16));
      AE_ADDCIRC16X4_XC(p16x4_mat_3, (m_itr + 3) * row_offset * sizeof(WORD16));
      
      xthalfx8 *p_mat_0 = (xthalfx8 *)p16x4_mat_0;
      xthalfx8 *p_mat_1 = (xthalfx8 *)p16x4_mat_1;
      xthalfx8 *p_mat_2 = (xthalfx8 *)p16x4_mat_2;
      xthalfx8 *p_mat_3 = (xthalfx8 *)p16x4_mat_3;

      ae_valignx2 align_mat_0, align_mat_1, align_mat_2, align_mat_3;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat_0);
      AE_LAHX4X2POS_PC(align_mat_1, p_mat_1);
      AE_LAHX4X2POS_PC(align_mat_2, p_mat_2);
      AE_LAHX4X2POS_PC(align_mat_3, p_mat_3);

      xthalfx4 mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1;
      xthalfx4 vec_batch_0_0, vec_batch_0_1;
      xthalfx4 vec_batch_1_0, vec_batch_1_1;
    
      for (c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {

        AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8 * sizeof(xthalf));
        AE_LHX4X2_IP(vec_batch_1_0, vec_batch_1_1, p_vec_batch_1, 8 * sizeof(xthalf));

        AE_LAHX4X2_IC(mat0_0, mat0_1, align_mat_0, p_mat_0);
        AE_LAHX4X2_IC(mat1_0, mat1_1, align_mat_1, p_mat_1);
        AE_LAHX4X2_IC(mat2_0, mat2_1, align_mat_2, p_mat_2);
        AE_LAHX4X2_IC(mat3_0, mat3_1, align_mat_3, p_mat_3);

        MADDQ_H(acc_0_0, acc_1_0, mat0_0, mat1_0, vec_batch_0_0);
        MADDQ_H(acc_2_0, acc_3_0, mat2_0, mat3_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat0_1, mat1_1, vec_batch_0_1);
        MADDQ_H(acc_2_0, acc_3_0, mat2_1, mat3_1, vec_batch_0_1);

        MADDQ_H(acc_0_1, acc_1_1, mat0_0, mat1_0, vec_batch_1_0);
        MADDQ_H(acc_2_1, acc_3_1, mat2_0, mat3_0, vec_batch_1_0);
        MADDQ_H(acc_0_1, acc_1_1, mat0_1, mat1_1, vec_batch_1_1);
        MADDQ_H(acc_2_1, acc_3_1, mat2_1, mat3_1, vec_batch_1_1);
      }

      ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
      ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
      int rem_elm = cols&0x7;
      ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      if(rem_elm <= 4)
      {
          mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
          mask2 = AE_ZERO64();
      }
      else
      {
          mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
      }
      if(rem_elm)
      {
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (rem_elm<<1));
        AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (rem_elm<<1));
        AE_LAHX4X2_IC(mat0_0, mat0_1, align_mat_0, p_mat_0);
        AE_LAHX4X2_IC(mat1_0, mat1_1, align_mat_1, p_mat_1);
        AE_LAHX4X2_IC(mat2_0, mat2_1, align_mat_2, p_mat_2);
        AE_LAHX4X2_IC(mat3_0, mat3_1, align_mat_3, p_mat_3);
        mat0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat0_0)), mask1)));
        mat0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat0_1)), mask2)));
        mat1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0)), mask1)));
        mat0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat0_1)), mask2)));
        mat2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat2_0)), mask1)));
        mat2_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat2_1)), mask2)));
        mat3_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat3_0)), mask1)));
        mat3_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat3_1)), mask2)));

        MADDQ_H(acc_0_0, acc_1_0, mat0_0, mat1_0, vec_batch_0_0);
        MADDQ_H(acc_2_0, acc_3_0, mat2_0, mat3_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat0_1, mat1_1, vec_batch_0_1);
        MADDQ_H(acc_2_0, acc_3_0, mat2_1, mat3_1, vec_batch_0_1);

        MADDQ_H(acc_0_1, acc_1_1, mat0_0, mat1_0, vec_batch_1_0);
        MADDQ_H(acc_2_1, acc_3_1, mat2_0, mat3_0, vec_batch_1_0);
        MADDQ_H(acc_0_1, acc_1_1, mat0_1, mat1_1, vec_batch_1_1);
        MADDQ_H(acc_2_1, acc_3_1, mat2_1, mat3_1, vec_batch_1_1);
      }
      // Process and store results
      xthalfx4 y0, y1, y01, y2, y3, y23;
      
      // Vec 0, Row 0
      y0 = AE_SELH_7531(acc_0_0, acc_1_0); y1 = AE_SELH_6420(acc_0_0, acc_1_0); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = AE_SELH_6543(y23, y23); y0 = ADD_HX4(y0, bias); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec0 + (m_itr+0)*out_stride), 0);
      // Vec 0, Row 1
      y23 = ADD_HX4(y23, bias); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), (p_out_vec0 + (m_itr+1)*out_stride), 0);
      
      // Vec 0, Row 2
      y0 = AE_SELH_7531(acc_2_0, acc_3_0); y1 = AE_SELH_6420(acc_2_0, acc_3_0); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = AE_SELH_6543(y23, y23); y0 = ADD_HX4(y0, bias); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec0 + (m_itr+2)*out_stride), 0);
      // Vec 0, Row 3
      y23 = ADD_HX4(y23, bias); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), (p_out_vec0 + (m_itr+3)*out_stride), 0);

      // Vec 1, Row 0
      y0 = AE_SELH_7531(acc_0_1, acc_1_1); y1 = AE_SELH_6420(acc_0_1, acc_1_1); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = AE_SELH_6543(y23, y23); y0 = ADD_HX4(y0, bias1); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec1 + (m_itr+0)*out_stride), 0);
      // Vec 1, Row 1
      y23 = ADD_HX4(y23, bias1); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), (p_out_vec1 + (m_itr+1)*out_stride), 0);

      // Vec 1, Row 2
      y0 = AE_SELH_7531(acc_2_1, acc_3_1); y1 = AE_SELH_6420(acc_2_1, acc_3_1); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = AE_SELH_6543(y23, y23); y0 = ADD_HX4(y0, bias1); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec1 + (m_itr+2)*out_stride), 0);
      // Vec 1, Row 3
      y23 = ADD_HX4(y23, bias1); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), (p_out_vec1 + (m_itr+3)*out_stride), 0);
    }
    
    /* Handle remaining rows in pairs */
    for(; m_itr < (rows & ~1); m_itr += 2)
    {
      xthalfx4 y0, y1, y2, y3, y01, y23;
      xthalfx4 acc_0_0 = ZERO_HX4();
      xthalfx4 acc_1_0 = ZERO_HX4();
      xthalfx4 acc_0_1 = ZERO_HX4();
      xthalfx4 acc_1_1 = ZERO_HX4();
      
      xthalfx8 *p_vec_batch_0 = p_vec_batch_0_base;
      xthalfx8 *p_vec_batch_1 = p_vec_batch_1_base;
      
      ae_int16x4 *p16x4_mat1_0 = (ae_int16x4 *)p_mat;
      ae_int16x4 *p16x4_mat1_1 = (ae_int16x4 *)p_mat;
      AE_ADDCIRC16X4_XC(p16x4_mat1_0, (m_itr)*row_offset*sizeof(WORD16));
      AE_ADDCIRC16X4_XC(p16x4_mat1_1, (m_itr+1)*row_offset*sizeof(WORD16));
      
      xthalfx8 *p_mat1_0 = (xthalfx8 *)p16x4_mat1_0;
      xthalfx8 *p_mat1_1 = (xthalfx8 *)p16x4_mat1_1;
      
      ae_valignx2 align_mat_0, align_mat_1;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);
      AE_LAHX4X2POS_PC(align_mat_1, p_mat1_1);
      
      xthalfx4 vec_batch_0_0, vec_batch_0_1;
      xthalfx4 vec_batch_1_0, vec_batch_1_1;
      xthalfx4 mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1;
      
      for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {
        AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8*sizeof(xthalf));
        AE_LHX4X2_IP(vec_batch_1_0, vec_batch_1_1, p_vec_batch_1, 8*sizeof(xthalf));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
        
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
        MADDQ_H(acc_0_1, acc_1_1, mat1_0_0, mat1_1_0, vec_batch_1_0);
        MADDQ_H(acc_0_1, acc_1_1, mat1_0_1, mat1_1_1, vec_batch_1_1);
      }

      ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
      ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
      int rem_elm = cols&0x7;
      ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      if(rem_elm <= 4)
      {
          mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
          mask2 = AE_ZERO64();
      }
      else
      {
          mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
      }
      if(rem_elm)
      {
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (rem_elm<<1));
        AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (rem_elm<<1));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
        mat1_0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_0)), mask1)));
        mat1_0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_1)), mask2)));
        mat1_1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_1_0)), mask1)));
        mat1_1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_1_1)), mask2)));
        
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
        MADDQ_H(acc_0_1, acc_1_1, mat1_0_0, mat1_1_0, vec_batch_1_0);
        MADDQ_H(acc_0_1, acc_1_1, mat1_0_1, mat1_1_1, vec_batch_1_1);
      }

      y0 = AE_SELH_7531(acc_0_0, acc_1_0); y1 = AE_SELH_6420(acc_0_0, acc_1_0); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = AE_SELH_6543(y23, y23); y0 = ADD_HX4(y0, bias); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec0 + m_itr*out_stride), 0);
      y23 = ADD_HX4(y23, bias); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), (p_out_vec0 + (m_itr+1)*out_stride), 0);
      
      y0 = AE_SELH_7531(acc_0_1, acc_1_1); y1 = AE_SELH_6420(acc_0_1, acc_1_1); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = AE_SELH_6543(y23, y23); y0 = ADD_HX4(y0, bias1); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec1 + m_itr*out_stride), 0);
      y23 = ADD_HX4(y23, bias1); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), (p_out_vec1 + (m_itr+1)*out_stride), 0);
    }
    
    // Handle remaining single row
    for(;m_itr < rows; m_itr++)
    {
      xthalfx4 y0, y1, y01, y2, y3, y23;
      xthalfx4 acc_0_0 = ZERO_HX4();
      xthalfx4 acc_0_1 = ZERO_HX4();
      
      xthalfx8 *p_vec_batch_0 = p_vec_batch_0_base;
      xthalfx8 *p_vec_batch_1 = p_vec_batch_1_base;
      
      ae_int16x4 *p16x4_mat1_0 = (ae_int16x4 *)p_mat;
      AE_ADDCIRC16X4_XC(p16x4_mat1_0, m_itr*row_offset*sizeof(WORD16));
      xthalfx8 *p_mat1_0 = (xthalfx8 *)p16x4_mat1_0;
      
      ae_valignx2 align_mat_0;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);
      
      xthalfx4 vec_batch_0_0, vec_batch_0_1, vec_batch_1_0, vec_batch_1_1;
      xthalfx4 mat1_0_0, mat1_0_1;
      
      for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {
        AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8*sizeof(xthalf));
        AE_LHX4X2_IP(vec_batch_1_0, vec_batch_1_1, p_vec_batch_1, 8*sizeof(xthalf));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        
        MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_0, vec_batch_1_0, mat1_0_0);
        MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_1, vec_batch_1_1, mat1_0_1);
      }

      ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
      ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
      int rem_elm = cols&0x7;
      ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      if(rem_elm <= 4)
      {
          mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
          mask2 = AE_ZERO64();
      }
      else
      {
          mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
      }
      if(rem_elm)
      {
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (rem_elm<<1));
        AE_LAVHX4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (rem_elm<<1));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        mat1_0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_0)), mask1)));
        mat1_0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_1)), mask2)));
        
        MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_0, vec_batch_1_0, mat1_0_0);
        MADDQ_H(acc_0_0, acc_0_1, vec_batch_0_1, vec_batch_1_1, mat1_0_1);
      }
      
      // Reduce and store vec 0
      y0 = AE_SELH_7531(acc_0_0, acc_0_0); y1 = AE_SELH_6420(acc_0_0, acc_0_0); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = ADD_HX4(y23, bias); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec0 + m_itr*out_stride), 0);

      // Reduce and store vec 1
      y0 = AE_SELH_7531(acc_0_1, acc_0_1); y1 = AE_SELH_6420(acc_0_1, acc_0_1); y01 = ADD_HX4(y0, y1);
      y2 = AE_SELH_7531(y01, y01); y3 = AE_SELH_6420(y01, y01); y23 = ADD_HX4(y2, y3);
      y0 = ADD_HX4(y23, bias1); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), (p_out_vec1 + m_itr*out_stride), 0);
    }
  }

  /* Final single vector if vec_count is odd */
  for(;vec_itr < vec_count; vec_itr++)
  {
    xthalfx4 bias = ZERO_HX4();
    
    if(p_bias != NULL)
    {
      bias = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(p_bias[vec_itr])));
    }
    
    int cols1_count = cols - cols%8;
    
    m_itr = 0;
    for(; m_itr < (rows & ~(2-1)); m_itr += 2)
    {
      xthalfx4 y0, y1, y2, y3, y01, y23;
      xthalfx4 acc_0_0 = ZERO_HX4();
      xthalfx4 acc_1_0 = ZERO_HX4();
      xthalfx4 vec_batch_0_0, vec_batch_0_1;
      
      xthalfx8 *p_vec_batch_0 = (xthalfx8 *)(p_vec + vec_itr*vec_offset);
      
      ae_int16x4 *p16x4_mat1_0 = (ae_int16x4 *)p_mat;
      ae_int16x4 *p16x4_mat1_1 = (ae_int16x4 *)p_mat;
      AE_ADDCIRC16X4_XC(p16x4_mat1_0, m_itr*row_offset*sizeof(WORD16));
      AE_ADDCIRC16X4_XC(p16x4_mat1_1, (m_itr+1)*row_offset*sizeof(WORD16));
      
      xthalfx8 *p_mat1_0 = (xthalfx8 *)p16x4_mat1_0;
      xthalfx8 *p_mat1_1 = (xthalfx8 *)p16x4_mat1_1;
      
      ae_valignx2 align_mat_0, align_mat_1;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);
      AE_LAHX4X2POS_PC(align_mat_1, p_mat1_1);
      
      xthalfx4 mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1;
      
      for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {
        AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8*sizeof(xthalf));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
        
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
      }
      ae_valignx2 align_vec = AE_LA128_PP(p_vec_batch_0);
      int rem_elm = cols&0x7;
      ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      if(rem_elm <= 4)
      {
          mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
          mask2 = AE_ZERO64();
      }
      else
      {
          mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
      }
      if(rem_elm != 0)
      {
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec, p_vec_batch_0, (rem_elm<<1));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        AE_LAHX4X2_IC(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
        mat1_0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_0)), mask1)));
        mat1_0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_1)), mask2)));
        mat1_1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_1_0)), mask1)));
        mat1_1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_1_1)), mask2)));
        
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_0, mat1_1_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_1_0, mat1_0_1, mat1_1_1, vec_batch_0_1);
      }
      
      y0 = AE_SELH_7531(acc_0_0, acc_1_0);
      y1 = AE_SELH_6420(acc_0_0, acc_1_0);
      y01 = ADD_HX4(y0, y1);
      
      y2 = AE_SELH_7531(y01, y01);
      y3 = AE_SELH_6420(y01, y01);
      y23 = ADD_HX4(y2, y3);
      
      y0 = AE_SELH_6543(y23, y23);
      y0 = ADD_HX4(y0, bias); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), ((ae_int16 *)p_out + vec_itr*out_offset + m_itr*out_stride), 0);
      
      y23 = ADD_HX4(y23, bias); y23 = CLAMP_HX4(y23);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y23)), ((ae_int16 *)p_out + vec_itr*out_offset + (m_itr+1)*out_stride), 0);
    }
    
    // Handle final single row
    for(; m_itr < rows; m_itr++)
    {
      xthalfx4 y0, y1, y2, y3, y01, y23;
      xthalfx4 acc_0_0 = ZERO_HX4();
      xthalfx4 acc_dummy_0_0 = ZERO_HX4();
      xthalfx4 vec_batch_0_0, vec_batch_0_1;
      
      xthalfx8 *p_vec_batch_0 = (xthalfx8 *)(p_vec + vec_itr*vec_offset);
      
      ae_int16x4 *p16x4_mat1_0 = (ae_int16x4 *)p_mat;
      AE_ADDCIRC16X4_XC(p16x4_mat1_0, m_itr*row_offset*sizeof(WORD16));
      xthalfx8 *p_mat1_0 = (xthalfx8 *)p16x4_mat1_0;
      
      ae_valignx2 align_mat_0;
      AE_LAHX4X2POS_PC(align_mat_0, p_mat1_0);
      
      xthalfx4 mat1_0_0, mat1_0_1;
      
      for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
      {
        AE_LHX4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec_batch_0, 8*sizeof(xthalf));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_0, mat1_0_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_1, mat1_0_1, vec_batch_0_1);
      }
      int rem_elm = cols&0x7;
      ae_int64 mask1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      ae_int64 mask2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0xffffffff, 0xffffffff));
      if(rem_elm <= 4)
      {
          mask1 = AE_SLAA64(mask1, ((4-rem_elm) * 16));
          mask2 = AE_ZERO64();
      }
      else
      {
          mask2 = AE_SLAA64(mask2, ((8-rem_elm) * 16));
      }
      if(rem_elm != 0)
      {
        ae_valignx2 align_vec = AE_LA128_PP(p_vec_batch_0);
        AE_LAVHX4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec, p_vec_batch_0, (rem_elm<<1));
        AE_LAHX4X2_IC(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
        mat1_0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_0)), mask1)));
        mat1_0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVINT16X4_FROMINT64(AE_AND64(AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4(mat1_0_1)), mask2)));
        
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_0, mat1_0_0, vec_batch_0_0);
        MADDQ_H(acc_0_0, acc_dummy_0_0, mat1_0_1, mat1_0_1, vec_batch_0_1);
      }
      
      y0 = AE_SELH_7531(acc_0_0, acc_0_0);
      y1 = AE_SELH_6420(acc_0_0, acc_0_0);
      y01 = ADD_HX4(y0, y1);
      
      y2 = AE_SELH_7531(y01, y01);
      y3 = AE_SELH_6420(y01, y01);
      y23 = ADD_HX4(y2, y3);
      
      y0 = ADD_HX4(y23, bias); y0 = CLAMP_HX4(y0);
      AE_S16_0_I(AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(y0)), ((ae_int16 *)p_out + vec_itr*out_offset + m_itr*out_stride), 0);
    }
  }
  
    #undef CLAMP_HX4
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
