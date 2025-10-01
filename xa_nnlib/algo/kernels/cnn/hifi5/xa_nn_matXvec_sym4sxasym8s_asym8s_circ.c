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
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

WORD32 xa_nn_matXvec_sym4sxasym8s_asym8s_circ(
    WORD8 * __restrict__ p_out,
    WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  int vec_itr, m_itr, c_itr;
  for(vec_itr = 0; vec_itr < vec_count; vec_itr++)
  {
    if((p_out_shift[vec_itr] > 31) || (p_out_shift[vec_itr] < -31))
    {
      return -1;
    }
  }
  if (!p_bias)
  {
    return -1;
  }

  ae_int32x2 max_int8 = SW_MOVDA32(127);
  ae_int32x2 min_int8 = SW_MOVDA32(-128);
  
  WORD8 *p_out_tmp;
  WORD8 *p_out_tmp0, *p_out_tmp1;

  vec_itr = 0;
  for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
  {
    int left_shift1, left_shift2, left_shift3, left_shift4;
#if TFLITE_SINGLE_ROUNDING
    left_shift1 = p_out_shift[vec_itr];
    left_shift2 = p_out_shift[vec_itr+1];
    left_shift3 = p_out_shift[vec_itr+2];
    left_shift4 = p_out_shift[vec_itr+3];
#if XCHAL_HAVE_HIFI5S
    left_shift1 = 31 - left_shift1;
    left_shift2 = 31 - left_shift2;
    left_shift3 = 31 - left_shift3;
    left_shift4 = 31 - left_shift4;
    left_shift1 = (left_shift1 << 16) | left_shift1;
    left_shift2 = (left_shift2 << 16) | left_shift2;
    left_shift3 = (left_shift3 << 16) | left_shift3;
    left_shift4 = (left_shift4 << 16) | left_shift4;
#endif    
#else
    int right_shift1, right_shift2, right_shift3, right_shift4;
    left_shift1 = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
    right_shift1 = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
    left_shift2 = p_out_shift[vec_itr+1]<0?0:p_out_shift[vec_itr+1];
    right_shift2 = p_out_shift[vec_itr+1]>0?0:-p_out_shift[vec_itr+1];
    left_shift3 = p_out_shift[vec_itr+2]<0?0:p_out_shift[vec_itr+2];
    right_shift3 = p_out_shift[vec_itr+2]>0?0:-p_out_shift[vec_itr+2];
    left_shift4 = p_out_shift[vec_itr+3]<0?0:p_out_shift[vec_itr+3];
    right_shift4 = p_out_shift[vec_itr+3]>0?0:-p_out_shift[vec_itr+3];
#endif     
    m_itr = 0; 

    for(; m_itr < (rows & ~(4-1)); m_itr += 4)
    {
      ae_int32x2 acc0 = AE_ZERO32();
      ae_int32x2 acc1 = AE_ZERO32();    
      ae_int32x2 acc2 = AE_ZERO32();
      ae_int32x2 acc3 = AE_ZERO32();   
      ae_int32x2 acc4 = AE_ZERO32();
      ae_int32x2 acc5 = AE_ZERO32();    
      ae_int32x2 acc6 = AE_ZERO32();
      ae_int32x2 acc7 = AE_ZERO32();             
      ae_int32x2 acc9, acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17, acc18, acc19, acc20, acc21, acc22, acc23, acc24; 

      ae_int8x8 mat0_0, mat0_1, mat0_2, mat0_3;
      ae_int8x8 mat1_0, mat1_1, mat1_2, mat1_3;
      ae_int8x8 mat2_0, mat2_1, mat2_2, mat2_3;
      ae_int8x8 mat3_0, mat3_1, mat3_2, mat3_3;

      ae_int8x8 vec0_0, vec0_1;
      ae_int8x8 vec1_0, vec1_1;
      ae_int8x8 vec2_0, vec2_1;
      ae_int8x8 vec3_0, vec3_1;

      ae_int16x4 mat_plus_zb0_0, mat_plus_zb0_1;
      ae_int16x4 mat_plus_zb0_2, mat_plus_zb0_3;
      ae_int16x4 mat_plus_zb0_4, mat_plus_zb0_5;
      ae_int16x4 mat_plus_zb0_6, mat_plus_zb0_7;

      ae_int16x4 mat_plus_zb1_0, mat_plus_zb1_1;
      ae_int16x4 mat_plus_zb1_2, mat_plus_zb1_3;
      ae_int16x4 mat_plus_zb1_4, mat_plus_zb1_5;
      ae_int16x4 mat_plus_zb1_6, mat_plus_zb1_7;

      ae_int16x4 mat_plus_zb2_0, mat_plus_zb2_1;
      ae_int16x4 mat_plus_zb2_2, mat_plus_zb2_3;
      ae_int16x4 mat_plus_zb2_4, mat_plus_zb2_5;
      ae_int16x4 mat_plus_zb2_6, mat_plus_zb2_7;

      ae_int16x4 mat_plus_zb3_0, mat_plus_zb3_1;
      ae_int16x4 mat_plus_zb3_2, mat_plus_zb3_3;
      ae_int16x4 mat_plus_zb3_4, mat_plus_zb3_5;
      ae_int16x4 mat_plus_zb3_6, mat_plus_zb3_7;          

      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);

      ae_int8x16 * __restrict__ p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 * __restrict__ p_vec_batch_1  = (ae_int8x16 *)(&p_vec1[(vec_itr+1)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 * __restrict__ p_vec_batch_2  = (ae_int8x16 *)(&p_vec1[(vec_itr+2)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 * __restrict__ p_vec_batch_3  = (ae_int8x16 *)(&p_vec1[(vec_itr+3)*PADDED_SIZE((vec_stride/2), 16)]);


      // ae_int8 *p_mat_0, *p_mat_1, *p_mat_2, *p_mat_3;
      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      ae_int8x16 * __restrict__ p_mat_0 = (ae_int8x16 *)p16x4_mat_0;
      ae_int16x4 *p16x4_mat_1 = (ae_int16x4 *)p_mat_0;
      AE_ADDCIRC16X4_XC(p16x4_mat_1, row_stride1*sizeof(WORD8)); 
      ae_int8x16 * __restrict__ p_mat_1 = (ae_int8x16 *)p16x4_mat_1;
      ae_int16x4 *p16x4_mat_2 = (ae_int16x4 *)p_mat_1;
      AE_ADDCIRC16X4_XC(p16x4_mat_2, row_stride1*sizeof(WORD8)); 
      ae_int8x16 * __restrict__ p_mat_2 = (ae_int8x16 *)p16x4_mat_2;
      ae_int16x4 *p16x4_mat_3 = (ae_int16x4 *)p_mat_2;
      AE_ADDCIRC16X4_XC(p16x4_mat_3, row_stride1*sizeof(WORD8)); 
      ae_int8x16 * __restrict__ p_mat_3 = (ae_int8x16 *)p16x4_mat_3;

      ae_valignx2 align_p_mat1_0;
      ae_valignx2 align_p_mat1_1;
      ae_valignx2 align_p_mat1_2;
      ae_valignx2 align_p_mat1_3;          

      AE_LA8X8X2POS_PC(align_p_mat1_0, p_mat_0);
      AE_LA8X8X2POS_PC(align_p_mat1_1, p_mat_1);
      AE_LA8X8X2POS_PC(align_p_mat1_2, p_mat_2);
      AE_LA8X8X2POS_PC(align_p_mat1_3, p_mat_3);

      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);
      ae_int4x16 vec_4b_0_0, vec_4b_0_1;
      ae_int4x16 vec_4b_1_0, vec_4b_1_1;
      ae_int4x16 vec_4b_2_0, vec_4b_2_1;
      ae_int4x16 vec_4b_3_0, vec_4b_3_1;

      ae_int16x4 dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604, 0x03010200));
      ae_int16x4 vec_interleaved0_16x4, vec_interleaved1_16x4, vec_interleaved2_16x4, vec_interleaved3_16x4, vec_interleaved4_16x4, vec_interleaved5_16x4, vec_interleaved6_16x4, vec_interleaved7_16x4;

        for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
        {
          AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, (16 * sizeof(WORD8)));
          vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
          vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);

          AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, (16 * sizeof(WORD8)));
          vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
          vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

          AE_L8X8X2_IP(vec2_0, vec2_1, p_vec_batch_2, (16 * sizeof(WORD8)));
          vec_4b_2_0 = AE_MOVINT4X16_FROMINT8X8(vec2_0);
          vec_4b_2_1 = AE_MOVINT4X16_FROMINT8X8(vec2_1);

          AE_L8X8X2_IP(vec3_0, vec3_1, p_vec_batch_3, (16 * sizeof(WORD8)));
          vec_4b_3_0 = AE_MOVINT4X16_FROMINT8X8(vec3_0);
          vec_4b_3_1 = AE_MOVINT4X16_FROMINT8X8(vec3_1);                        

          AE_LA8X8X2_IC(mat0_0, mat0_1, align_p_mat1_0, p_mat_0);
          AE_LA8X8X2_IC(mat0_2, mat0_3, align_p_mat1_0, p_mat_0);
          AE_LA8X8X2_IC(mat1_0, mat1_1, align_p_mat1_1, p_mat_1);
          AE_LA8X8X2_IC(mat1_2, mat1_3, align_p_mat1_1, p_mat_1);
          AE_LA8X8X2_IC(mat2_0, mat2_1, align_p_mat1_2, p_mat_2);
          AE_LA8X8X2_IC(mat2_2, mat2_3, align_p_mat1_2, p_mat_2);
          AE_LA8X8X2_IC(mat3_0, mat3_1, align_p_mat1_3, p_mat_3);
          AE_LA8X8X2_IC(mat3_2, mat3_3, align_p_mat1_3, p_mat_3);

          AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
          AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
          AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
          AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

          AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
          AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
          AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
          AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

          AE_SUBW8(mat_plus_zb2_0, mat_plus_zb2_1, mat2_0, mat_zb);
          AE_SUBW8(mat_plus_zb2_2, mat_plus_zb2_3, mat2_1, mat_zb);
          AE_SUBW8(mat_plus_zb2_4, mat_plus_zb2_5, mat2_2, mat_zb);
          AE_SUBW8(mat_plus_zb2_6, mat_plus_zb2_7, mat2_3, mat_zb);

          AE_SUBW8(mat_plus_zb3_0, mat_plus_zb3_1, mat3_0, mat_zb);
          AE_SUBW8(mat_plus_zb3_2, mat_plus_zb3_3, mat3_1, mat_zb);
          AE_SUBW8(mat_plus_zb3_4, mat_plus_zb3_5, mat3_2, mat_zb);
          AE_SUBW8(mat_plus_zb3_6, mat_plus_zb3_7, mat3_3, mat_zb);            

          AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
          AE_DSEL16X4(vec_interleaved1_16x4, vec_interleaved3_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_0), dsel);
          AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);
          AE_DSEL16X4(vec_interleaved5_16x4, vec_interleaved7_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_1), dsel);

          ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
          ae_int4x16 vec_interleaved1 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved1_16x4);

          ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
          ae_int4x16 vec_interleaved3 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved3_16x4);

          ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
          ae_int4x16 vec_interleaved5 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved5_16x4);

          ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);
          ae_int4x16 vec_interleaved7 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved7_16x4);

          AE_MULA8Q4X16(acc0, acc1, vec_interleaved0, vec_interleaved1, mat_plus_zb0_0, mat_plus_zb0_1);
          AE_MULA8Q4X16(acc0, acc1, vec_interleaved2, vec_interleaved3, mat_plus_zb0_2, mat_plus_zb0_3);
          AE_MULA8Q4X16(acc0, acc1, vec_interleaved4, vec_interleaved5, mat_plus_zb0_4, mat_plus_zb0_5);
          AE_MULA8Q4X16(acc0, acc1, vec_interleaved6, vec_interleaved7, mat_plus_zb0_6, mat_plus_zb0_7);

          AE_MULA8Q4X16(acc2, acc3, vec_interleaved0, vec_interleaved1, mat_plus_zb1_0, mat_plus_zb1_1);
          AE_MULA8Q4X16(acc2, acc3, vec_interleaved2, vec_interleaved3, mat_plus_zb1_2, mat_plus_zb1_3);
          AE_MULA8Q4X16(acc2, acc3, vec_interleaved4, vec_interleaved5, mat_plus_zb1_4, mat_plus_zb1_5);
          AE_MULA8Q4X16(acc2, acc3, vec_interleaved6, vec_interleaved7, mat_plus_zb1_6, mat_plus_zb1_7);    

          AE_MULA8Q4X16(acc4, acc5, vec_interleaved0, vec_interleaved1, mat_plus_zb2_0, mat_plus_zb2_1);
          AE_MULA8Q4X16(acc4, acc5, vec_interleaved2, vec_interleaved3, mat_plus_zb2_2, mat_plus_zb2_3);
          AE_MULA8Q4X16(acc4, acc5, vec_interleaved4, vec_interleaved5, mat_plus_zb2_4, mat_plus_zb2_5);
          AE_MULA8Q4X16(acc4, acc5, vec_interleaved6, vec_interleaved7, mat_plus_zb2_6, mat_plus_zb2_7);

          AE_MULA8Q4X16(acc6, acc7, vec_interleaved0, vec_interleaved1, mat_plus_zb3_0, mat_plus_zb3_1);
          AE_MULA8Q4X16(acc6, acc7, vec_interleaved2, vec_interleaved3, mat_plus_zb3_2, mat_plus_zb3_3);
          AE_MULA8Q4X16(acc6, acc7, vec_interleaved4, vec_interleaved5, mat_plus_zb3_4, mat_plus_zb3_5);
          AE_MULA8Q4X16(acc6, acc7, vec_interleaved6, vec_interleaved7, mat_plus_zb3_6, mat_plus_zb3_7);
        }
        if(cols_count != cols1)
        {
          AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, (16 * sizeof(WORD8)));
          vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift_0), rem_cols_shift_0));
          vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift_1), rem_cols_shift_1));          
          vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
          vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);

          AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, (16 * sizeof(WORD8)));
          vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift_0), rem_cols_shift_0));
          vec1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_1), rem_cols_shift_1), rem_cols_shift_1));          
          vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
          vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

          AE_L8X8X2_IP(vec2_0, vec2_1, p_vec_batch_2, (16 * sizeof(WORD8)));
          vec2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_0), rem_cols_shift_0), rem_cols_shift_0));
          vec2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_1), rem_cols_shift_1), rem_cols_shift_1));          
          vec_4b_2_0 = AE_MOVINT4X16_FROMINT8X8(vec2_0);
          vec_4b_2_1 = AE_MOVINT4X16_FROMINT8X8(vec2_1);

          AE_L8X8X2_IP(vec3_0, vec3_1, p_vec_batch_3, (16 * sizeof(WORD8)));
          vec3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_0), rem_cols_shift_0), rem_cols_shift_0));
          vec3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_1), rem_cols_shift_1), rem_cols_shift_1));          
          vec_4b_3_0 = AE_MOVINT4X16_FROMINT8X8(vec3_0);
          vec_4b_3_1 = AE_MOVINT4X16_FROMINT8X8(vec3_1);

          AE_LA8X8X2_IC(mat0_0, mat0_1, align_p_mat1_0, p_mat_0);
          AE_LA8X8X2_IC(mat0_2, mat0_3, align_p_mat1_0, p_mat_0);
          AE_LA8X8X2_IC(mat1_0, mat1_1, align_p_mat1_1, p_mat_1);
          AE_LA8X8X2_IC(mat1_2, mat1_3, align_p_mat1_1, p_mat_1);
          AE_LA8X8X2_IC(mat2_0, mat2_1, align_p_mat1_2, p_mat_2);
          AE_LA8X8X2_IC(mat2_2, mat2_3, align_p_mat1_2, p_mat_2);
          AE_LA8X8X2_IC(mat3_0, mat3_1, align_p_mat1_3, p_mat_3);
          AE_LA8X8X2_IC(mat3_2, mat3_3, align_p_mat1_3, p_mat_3);

          AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
          AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
          AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
          AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

          AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
          AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
          AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
          AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

          AE_SUBW8(mat_plus_zb2_0, mat_plus_zb2_1, mat2_0, mat_zb);
          AE_SUBW8(mat_plus_zb2_2, mat_plus_zb2_3, mat2_1, mat_zb);
          AE_SUBW8(mat_plus_zb2_4, mat_plus_zb2_5, mat2_2, mat_zb);
          AE_SUBW8(mat_plus_zb2_6, mat_plus_zb2_7, mat2_3, mat_zb);

          AE_SUBW8(mat_plus_zb3_0, mat_plus_zb3_1, mat3_0, mat_zb);
          AE_SUBW8(mat_plus_zb3_2, mat_plus_zb3_3, mat3_1, mat_zb);
          AE_SUBW8(mat_plus_zb3_4, mat_plus_zb3_5, mat3_2, mat_zb);
          AE_SUBW8(mat_plus_zb3_6, mat_plus_zb3_7, mat3_3, mat_zb);

          AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
          AE_DSEL16X4(vec_interleaved1_16x4, vec_interleaved3_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_0), dsel);
          AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);
          AE_DSEL16X4(vec_interleaved5_16x4, vec_interleaved7_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_1), dsel);

          ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
          ae_int4x16 vec_interleaved1 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved1_16x4);

          ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
          ae_int4x16 vec_interleaved3 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved3_16x4);

          ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
          ae_int4x16 vec_interleaved5 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved5_16x4);

          ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);
          ae_int4x16 vec_interleaved7 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved7_16x4);

          AE_MULA8Q4X16(acc0, acc1, vec_interleaved0, vec_interleaved1, mat_plus_zb0_0, mat_plus_zb0_1);
          AE_MULA8Q4X16(acc0, acc1, vec_interleaved2, vec_interleaved3, mat_plus_zb0_2, mat_plus_zb0_3);
          AE_MULA8Q4X16(acc0, acc1, vec_interleaved4, vec_interleaved5, mat_plus_zb0_4, mat_plus_zb0_5);
          AE_MULA8Q4X16(acc0, acc1, vec_interleaved6, vec_interleaved7, mat_plus_zb0_6, mat_plus_zb0_7);

          AE_MULA8Q4X16(acc2, acc3, vec_interleaved0, vec_interleaved1, mat_plus_zb1_0, mat_plus_zb1_1);
          AE_MULA8Q4X16(acc2, acc3, vec_interleaved2, vec_interleaved3, mat_plus_zb1_2, mat_plus_zb1_3);
          AE_MULA8Q4X16(acc2, acc3, vec_interleaved4, vec_interleaved5, mat_plus_zb1_4, mat_plus_zb1_5);
          AE_MULA8Q4X16(acc2, acc3, vec_interleaved6, vec_interleaved7, mat_plus_zb1_6, mat_plus_zb1_7);    

          AE_MULA8Q4X16(acc4, acc5, vec_interleaved0, vec_interleaved1, mat_plus_zb2_0, mat_plus_zb2_1);
          AE_MULA8Q4X16(acc4, acc5, vec_interleaved2, vec_interleaved3, mat_plus_zb2_2, mat_plus_zb2_3);
          AE_MULA8Q4X16(acc4, acc5, vec_interleaved4, vec_interleaved5, mat_plus_zb2_4, mat_plus_zb2_5);
          AE_MULA8Q4X16(acc4, acc5, vec_interleaved6, vec_interleaved7, mat_plus_zb2_6, mat_plus_zb2_7);  

          AE_MULA8Q4X16(acc6, acc7, vec_interleaved0, vec_interleaved1, mat_plus_zb3_0, mat_plus_zb3_1);
          AE_MULA8Q4X16(acc6, acc7, vec_interleaved2, vec_interleaved3, mat_plus_zb3_2, mat_plus_zb3_3);
          AE_MULA8Q4X16(acc6, acc7, vec_interleaved4, vec_interleaved5, mat_plus_zb3_4, mat_plus_zb3_5);
          AE_MULA8Q4X16(acc6, acc7, vec_interleaved6, vec_interleaved7, mat_plus_zb3_6, mat_plus_zb3_7);                
        }
               
      ae_int32x2 bias32x2_0 =  AE_MOVDA32X2(p_bias[vec_itr], p_bias[vec_itr+1]);
      ae_int32x2 bias32x2_1 =  AE_MOVDA32X2(p_bias[vec_itr+2], p_bias[vec_itr+3]);

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc1), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));
      acc2 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc2), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc3 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc3), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));
      acc4 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc4), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc5 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc5), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));
      acc6 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc6), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc7 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc7), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc9, acc13, acc0, acc2, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc10, acc14, acc0, acc2, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc11, acc15, acc1, acc3, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc12, acc16, acc1, acc3, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);

      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc17, acc21, acc4, acc6, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc18, acc22, acc4, acc6, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc19, acc23, acc5, acc7, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc20, acc24, acc5, acc7, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);
#else
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc9, acc13, acc0, acc2, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc10, acc14, acc0, acc2, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc11, acc15, acc1, acc3, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc12, acc16, acc1, acc3, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);

      MPY_BY_QUANT_MULT_X2X2_OUT32(acc17, acc21, acc4, acc6, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc18, acc22, acc4, acc6, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc19, acc23, acc5, acc7, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc20, acc24, acc5, acc7, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);
#endif

      ae_int16x4 out0, out1, out2, out3, out4, out5, out6, out7;
      out0 = AE_SAT16X4(acc9, acc10);
      out1 = AE_SAT16X4(acc11, acc12);

      out2 = AE_SAT16X4(acc13, acc14);
      out3 = AE_SAT16X4(acc15, acc16);

      out4 = AE_SAT16X4(acc17, acc18);
      out5 = AE_SAT16X4(acc19, acc20);

      out6 = AE_SAT16X4(acc21, acc22);
      out7 = AE_SAT16X4(acc23, acc24);

      out0 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out0), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out1 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out1), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out2 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out2), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out3 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out3), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out4 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out4), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out5 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out5), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out6 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out6), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out7 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out7), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));

      ae_int8x8 out_0 = AE_SAT8X8X16(out0, out1);
      ae_int8x8 out_1 = AE_SAT8X8X16(out2, out3);
      ae_int8x8 out_2 = AE_SAT8X8X16(out4, out5);
      ae_int8x8 out_3 = AE_SAT8X8X16(out6, out7);

      AE_SW_S8_7_X(out_0, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_4_X(out_0, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_3_X(out_0, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_S8_0_X(out_0, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+0)*out_row_offset);

      AE_SW_S8_7_X(out_1, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_SW_S8_4_X(out_1, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_SW_S8_3_X(out_1, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_S8_0_X(out_1, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+1)*out_row_offset);

      AE_SW_S8_7_X(out_2, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+2)*out_row_offset);
      AE_SW_S8_4_X(out_2, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+2)*out_row_offset);
      AE_SW_S8_3_X(out_2, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+2)*out_row_offset);
      AE_S8_0_X(out_2, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+2)*out_row_offset);

      AE_SW_S8_7_X(out_3, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+3)*out_row_offset);
      AE_SW_S8_4_X(out_3, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+3)*out_row_offset);
      AE_SW_S8_3_X(out_3, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+3)*out_row_offset);
      AE_S8_0_X(out_3, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+3)*out_row_offset);
    }

    for(; m_itr < (rows & ~(2-1)); m_itr += 2)
    {
      ae_int32x2 acc0 = AE_ZERO32();
      ae_int32x2 acc1 = AE_ZERO32();    
      ae_int32x2 acc2 = AE_ZERO32();
      ae_int32x2 acc3 = AE_ZERO32();   
      ae_int32x2 acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11; 

      ae_int8x8 mat0_0, mat0_1, mat0_2, mat0_3;
      ae_int8x8 mat1_0, mat1_1, mat1_2, mat1_3;

      ae_int8x8 vec0_0, vec0_1;
      ae_int8x8 vec1_0, vec1_1;
      ae_int8x8 vec2_0, vec2_1;
      ae_int8x8 vec3_0, vec3_1;

      ae_int16x4 mat_plus_zb0_0, mat_plus_zb0_1;
      ae_int16x4 mat_plus_zb0_2, mat_plus_zb0_3;
      ae_int16x4 mat_plus_zb0_4, mat_plus_zb0_5;
      ae_int16x4 mat_plus_zb0_6, mat_plus_zb0_7;

      ae_int16x4 mat_plus_zb1_0, mat_plus_zb1_1;
      ae_int16x4 mat_plus_zb1_2, mat_plus_zb1_3;
      ae_int16x4 mat_plus_zb1_4, mat_plus_zb1_5;
      ae_int16x4 mat_plus_zb1_6, mat_plus_zb1_7;

      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);

      ae_int8x16 * __restrict__ p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 * __restrict__ p_vec_batch_1  = (ae_int8x16 *)(&p_vec1[(vec_itr+1)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 * __restrict__ p_vec_batch_2  = (ae_int8x16 *)(&p_vec1[(vec_itr+2)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 * __restrict__ p_vec_batch_3  = (ae_int8x16 *)(&p_vec1[(vec_itr+3)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_mat_0, *p_mat_1;
      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      p_mat_0 = (ae_int8x16 *)p16x4_mat_0;
      ae_int16x4 *p16x4_mat_1 = (ae_int16x4 *)p_mat_0;
      AE_ADDCIRC16X4_XC(p16x4_mat_1, row_stride1*sizeof(WORD8)); 
      p_mat_1 = (ae_int8x16 *)p16x4_mat_1;

      ae_valignx2 align_p_mat1_0;
      ae_valignx2 align_p_mat1_1;       

      AE_LA8X8X2POS_PC(align_p_mat1_0, p_mat_0);
      AE_LA8X8X2POS_PC(align_p_mat1_1, p_mat_1);

      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);
      ae_int4x16 vec_4b_0_0, vec_4b_0_1;
      ae_int4x16 vec_4b_1_0, vec_4b_1_1;
      ae_int4x16 vec_4b_2_0, vec_4b_2_1;
      ae_int4x16 vec_4b_3_0, vec_4b_3_1;

      ae_int16x4 dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604, 0x03010200));
      ae_int16x4 vec_interleaved0_16x4, vec_interleaved1_16x4, vec_interleaved2_16x4, vec_interleaved3_16x4, vec_interleaved4_16x4, vec_interleaved5_16x4, vec_interleaved6_16x4, vec_interleaved7_16x4;

      for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, (16 * sizeof(WORD8)));
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);

        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, (16 * sizeof(WORD8)));
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_L8X8X2_IP(vec2_0, vec2_1, p_vec_batch_2, (16 * sizeof(WORD8)));
        vec_4b_2_0 = AE_MOVINT4X16_FROMINT8X8(vec2_0);
        vec_4b_2_1 = AE_MOVINT4X16_FROMINT8X8(vec2_1);

        AE_L8X8X2_IP(vec3_0, vec3_1, p_vec_batch_3, (16 * sizeof(WORD8)));
        vec_4b_3_0 = AE_MOVINT4X16_FROMINT8X8(vec3_0);
        vec_4b_3_1 = AE_MOVINT4X16_FROMINT8X8(vec3_1);                        

        AE_LA8X8X2_IC(mat0_0, mat0_1, align_p_mat1_0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, align_p_mat1_0, p_mat_0);
        AE_LA8X8X2_IC(mat1_0, mat1_1, align_p_mat1_1, p_mat_1);
        AE_LA8X8X2_IC(mat1_2, mat1_3, align_p_mat1_1, p_mat_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
        AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
        AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
        AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved1_16x4, vec_interleaved3_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);
        AE_DSEL16X4(vec_interleaved5_16x4, vec_interleaved7_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved1 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved1_16x4);

        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved3 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved3_16x4);

        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved5 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved5_16x4);

        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);
        ae_int4x16 vec_interleaved7 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved7_16x4);

        AE_MULA8Q4X16(acc0, acc1, vec_interleaved0, vec_interleaved1, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved2, vec_interleaved3, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved4, vec_interleaved5, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved6, vec_interleaved7, mat_plus_zb0_6, mat_plus_zb0_7);

        AE_MULA8Q4X16(acc2, acc3, vec_interleaved0, vec_interleaved1, mat_plus_zb1_0, mat_plus_zb1_1);
        AE_MULA8Q4X16(acc2, acc3, vec_interleaved2, vec_interleaved3, mat_plus_zb1_2, mat_plus_zb1_3);
        AE_MULA8Q4X16(acc2, acc3, vec_interleaved4, vec_interleaved5, mat_plus_zb1_4, mat_plus_zb1_5);
        AE_MULA8Q4X16(acc2, acc3, vec_interleaved6, vec_interleaved7, mat_plus_zb1_6, mat_plus_zb1_7);            
      }
      if(cols_count != cols1)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, (16 * sizeof(WORD8)));
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift_0), rem_cols_shift_0));
        vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);

        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, (16 * sizeof(WORD8)));
        vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift_0), rem_cols_shift_0));
        vec1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_L8X8X2_IP(vec2_0, vec2_1, p_vec_batch_2, (16 * sizeof(WORD8)));
        vec2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_0), rem_cols_shift_0), rem_cols_shift_0));
        vec2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_2_0 = AE_MOVINT4X16_FROMINT8X8(vec2_0);
        vec_4b_2_1 = AE_MOVINT4X16_FROMINT8X8(vec2_1);

        AE_L8X8X2_IP(vec3_0, vec3_1, p_vec_batch_3, (16 * sizeof(WORD8)));
        vec3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_0), rem_cols_shift_0), rem_cols_shift_0));
        vec3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_3_0 = AE_MOVINT4X16_FROMINT8X8(vec3_0);
        vec_4b_3_1 = AE_MOVINT4X16_FROMINT8X8(vec3_1);

        AE_LA8X8X2_IC(mat0_0, mat0_1, align_p_mat1_0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, align_p_mat1_0, p_mat_0);
        AE_LA8X8X2_IC(mat1_0, mat1_1, align_p_mat1_1, p_mat_1);
        AE_LA8X8X2_IC(mat1_2, mat1_3, align_p_mat1_1, p_mat_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
        AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
        AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
        AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved1_16x4, vec_interleaved3_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);
        AE_DSEL16X4(vec_interleaved5_16x4, vec_interleaved7_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved1 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved1_16x4);

        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved3 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved3_16x4);

        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved5 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved5_16x4);

        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);
        ae_int4x16 vec_interleaved7 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved7_16x4);

        AE_MULA8Q4X16(acc0, acc1, vec_interleaved0, vec_interleaved1, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved2, vec_interleaved3, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved4, vec_interleaved5, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved6, vec_interleaved7, mat_plus_zb0_6, mat_plus_zb0_7);

        AE_MULA8Q4X16(acc2, acc3, vec_interleaved0, vec_interleaved1, mat_plus_zb1_0, mat_plus_zb1_1);
        AE_MULA8Q4X16(acc2, acc3, vec_interleaved2, vec_interleaved3, mat_plus_zb1_2, mat_plus_zb1_3);
        AE_MULA8Q4X16(acc2, acc3, vec_interleaved4, vec_interleaved5, mat_plus_zb1_4, mat_plus_zb1_5);
        AE_MULA8Q4X16(acc2, acc3, vec_interleaved6, vec_interleaved7, mat_plus_zb1_6, mat_plus_zb1_7);                 
      }
               
      ae_int32x2 bias32x2_0 =  AE_MOVDA32X2(p_bias[vec_itr], p_bias[vec_itr+1]);
      ae_int32x2 bias32x2_1 =  AE_MOVDA32X2(p_bias[vec_itr+2], p_bias[vec_itr+3]);

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc1), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));
      acc2 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc2), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc3 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc3), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc4, acc8, acc0, acc2, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc5, acc9, acc0, acc2, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc6, acc10, acc1, acc3, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc7, acc11, acc1, acc3, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);
#else
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc4, acc8, acc0, acc2, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc5, acc9, acc0, acc2, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc6, acc10, acc1, acc3, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc7, acc11, acc1, acc3, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);
#endif

      ae_int16x4 out0, out1, out2, out3;
      out0 = AE_SAT16X4(acc4, acc5);
      out1 = AE_SAT16X4(acc6, acc7);

      out2 = AE_SAT16X4(acc8, acc9);
      out3 = AE_SAT16X4(acc10, acc11);

      out0 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out0), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out1 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out1), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out2 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out2), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out3 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out3), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));

      ae_int8x8 out_0 = AE_SAT8X8X16(out0, out1);
      ae_int8x8 out_1 = AE_SAT8X8X16(out2, out3);

      AE_SW_S8_7_X(out_0, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_4_X(out_0, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_3_X(out_0, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_S8_0_X(out_0, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+0)*out_row_offset);

      AE_SW_S8_7_X(out_1, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_SW_S8_4_X(out_1, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_SW_S8_3_X(out_1, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_S8_0_X(out_1, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+1)*out_row_offset);
    }

    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc0 = AE_ZERO32();
      ae_int32x2 acc1 = AE_ZERO32();    
      ae_int32x2 acc2 = AE_ZERO32();
      ae_int32x2 acc3 = AE_ZERO32();   
      ae_int32x2 acc4 = AE_ZERO32(); 
      ae_int32x2 acc5 = AE_ZERO32();           

      ae_int8x8 mat0_0, mat0_1, mat0_2, mat0_3;
      ae_int8x8 vec0_0, vec0_1;
      ae_int8x8 vec1_0, vec1_1;
      ae_int8x8 vec2_0, vec2_1;
      ae_int8x8 vec3_0, vec3_1;

      ae_int16x4 mat_plus_zb0_0, mat_plus_zb0_1;
      ae_int16x4 mat_plus_zb0_2, mat_plus_zb0_3;
      ae_int16x4 mat_plus_zb0_4, mat_plus_zb0_5;
      ae_int16x4 mat_plus_zb0_6, mat_plus_zb0_7;

      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);

      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_vec_batch_1  = (ae_int8x16 *)(&p_vec1[(vec_itr+1)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_vec_batch_2  = (ae_int8x16 *)(&p_vec1[(vec_itr+2)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_vec_batch_3  = (ae_int8x16 *)(&p_vec1[(vec_itr+3)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_mat_0;
      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;      
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      p_mat_0 = (ae_int8x16 *)p16x4_mat_0;

      ae_valignx2 align_p_mat1_0;

      AE_LA8X8X2POS_PC(align_p_mat1_0, p_mat_0);

      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);
      ae_int4x16 vec_4b_0_0, vec_4b_0_1;
      ae_int4x16 vec_4b_1_0, vec_4b_1_1;
      ae_int4x16 vec_4b_2_0, vec_4b_2_1;
      ae_int4x16 vec_4b_3_0, vec_4b_3_1;

      ae_int16x4 dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604, 0x03010200));
      ae_int16x4 vec_interleaved0_16x4, vec_interleaved1_16x4, vec_interleaved2_16x4, vec_interleaved3_16x4, vec_interleaved4_16x4, vec_interleaved5_16x4, vec_interleaved6_16x4, vec_interleaved7_16x4;

      for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, (16 * sizeof(WORD8)));
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);

        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, (16 * sizeof(WORD8)));
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_L8X8X2_IP(vec2_0, vec2_1, p_vec_batch_2, (16 * sizeof(WORD8)));
        vec_4b_2_0 = AE_MOVINT4X16_FROMINT8X8(vec2_0);
        vec_4b_2_1 = AE_MOVINT4X16_FROMINT8X8(vec2_1);

        AE_L8X8X2_IP(vec3_0, vec3_1, p_vec_batch_3, (16 * sizeof(WORD8)));
        vec_4b_3_0 = AE_MOVINT4X16_FROMINT8X8(vec3_0);
        vec_4b_3_1 = AE_MOVINT4X16_FROMINT8X8(vec3_1);                        

        AE_LA8X8X2_IC(mat0_0, mat0_1, align_p_mat1_0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, align_p_mat1_0, p_mat_0);


        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved1_16x4, vec_interleaved3_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);
        AE_DSEL16X4(vec_interleaved5_16x4, vec_interleaved7_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved1 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved1_16x4);

        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved3 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved3_16x4);

        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved5 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved5_16x4);

        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);
        ae_int4x16 vec_interleaved7 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved7_16x4);

        AE_MULA8Q4X16(acc0, acc1, vec_interleaved0, vec_interleaved1, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved2, vec_interleaved3, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved4, vec_interleaved5, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved6, vec_interleaved7, mat_plus_zb0_6, mat_plus_zb0_7);           
      }
      if(cols_count != cols1)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, (16 * sizeof(WORD8)));
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift_0), rem_cols_shift_0));
        vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);

        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, (16 * sizeof(WORD8)));
        vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift_0), rem_cols_shift_0));
        vec1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_L8X8X2_IP(vec2_0, vec2_1, p_vec_batch_2, (16 * sizeof(WORD8)));
        vec2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_0), rem_cols_shift_0), rem_cols_shift_0));
        vec2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_2_0 = AE_MOVINT4X16_FROMINT8X8(vec2_0);
        vec_4b_2_1 = AE_MOVINT4X16_FROMINT8X8(vec2_1);

        AE_L8X8X2_IP(vec3_0, vec3_1, p_vec_batch_3, (16 * sizeof(WORD8)));
        vec3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_0), rem_cols_shift_0), rem_cols_shift_0));
        vec3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_3_0 = AE_MOVINT4X16_FROMINT8X8(vec3_0);
        vec_4b_3_1 = AE_MOVINT4X16_FROMINT8X8(vec3_1);

        AE_LA8X8X2_IC(mat0_0, mat0_1, align_p_mat1_0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, align_p_mat1_0, p_mat_0);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);
        
        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved1_16x4, vec_interleaved3_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);
        AE_DSEL16X4(vec_interleaved5_16x4, vec_interleaved7_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_2_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_3_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved1 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved1_16x4);

        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved3 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved3_16x4);

        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved5 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved5_16x4);

        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);
        ae_int4x16 vec_interleaved7 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved7_16x4);

        AE_MULA8Q4X16(acc0, acc1, vec_interleaved0, vec_interleaved1, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved2, vec_interleaved3, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved4, vec_interleaved5, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, acc1, vec_interleaved6, vec_interleaved7, mat_plus_zb0_6, mat_plus_zb0_7);               
      }
             
      ae_int32x2 bias32x2_0 =  AE_MOVDA32X2(p_bias[vec_itr], p_bias[vec_itr+1]);
      ae_int32x2 bias32x2_1 =  AE_MOVDA32X2(p_bias[vec_itr+2], p_bias[vec_itr+3]);

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc1), AE_MOVF32X2_FROMINT32X2(bias32x2_1)));

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc2, acc0, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc3, acc0, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc4, acc1, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc5, acc1, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);
#else
      MPY_BY_QUANT_MULT_X2_OUT32(acc2, acc0, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2_OUT32(acc3, acc0, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
      MPY_BY_QUANT_MULT_X2_OUT32(acc4, acc1, p_out_multiplier[vec_itr+2], left_shift3, right_shift3);
      MPY_BY_QUANT_MULT_X2_OUT32(acc5, acc1, p_out_multiplier[vec_itr+3], left_shift4, right_shift4);
#endif

      ae_int16x4 out0, out1;
      out0 = AE_SAT16X4(acc2, acc3);
      out1 = AE_SAT16X4(acc4, acc5);
      out0 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out0), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out1 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out1), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      ae_int8x8 out = AE_SAT8X8X16(out0, out1);                 

      AE_SW_S8_7_X(out, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_4_X(out, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_3_X(out, (ae_int8 *)p_out, (vec_itr+2)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_S8_0_X(out, (ae_int8 *)p_out, (vec_itr+3)*out_col_offset + (m_itr+0)*out_row_offset);
    }
  }
  for(; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
  {
    int left_shift1, left_shift2;
#if TFLITE_SINGLE_ROUNDING      
    left_shift1 = p_out_shift[vec_itr];
    left_shift2 = p_out_shift[vec_itr+1];
#if XCHAL_HAVE_HIFI5S
    left_shift1 = 31 - left_shift1;
    left_shift2 = 31 - left_shift2;
    left_shift1 = (left_shift1 << 16) | left_shift1;
    left_shift2 = (left_shift2 << 16) | left_shift2; 
#endif    
#else
    int right_shift1, right_shift2;
    left_shift1 = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
    right_shift1 = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
    left_shift2 = p_out_shift[vec_itr+1]<0?0:p_out_shift[vec_itr+1];
    right_shift2 = p_out_shift[vec_itr+1]>0?0:-p_out_shift[vec_itr+1];      
#endif          
    for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
    {
      ae_int32x2 acc0 = AE_ZERO32();  
      ae_int32x2 acc1 = AE_ZERO32();    
      ae_int32x2 acc2, acc3, acc4, acc5; 
      ae_int32x2 dummy_acc;

      ae_int8x8 mat0_0, mat0_1, mat0_2, mat0_3;
      ae_int8x8 mat1_0, mat1_1, mat1_2, mat1_3;
      ae_int8x8 vec0_0, vec0_1;
      ae_int8x8 vec1_0, vec1_1;

      ae_int16x4 mat_plus_zb0_0, mat_plus_zb0_1;
      ae_int16x4 mat_plus_zb0_2, mat_plus_zb0_3;
      ae_int16x4 mat_plus_zb0_4, mat_plus_zb0_5;
      ae_int16x4 mat_plus_zb0_6, mat_plus_zb0_7;

      ae_int16x4 mat_plus_zb1_0, mat_plus_zb1_1;
      ae_int16x4 mat_plus_zb1_2, mat_plus_zb1_3;
      ae_int16x4 mat_plus_zb1_4, mat_plus_zb1_5;
      ae_int16x4 mat_plus_zb1_6, mat_plus_zb1_7;

      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);

      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_vec_batch_1  = (ae_int8x16 *)(&p_vec1[(vec_itr+1)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_mat_0, *p_mat_1;
      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      p_mat_0 = (ae_int8x16 *)p16x4_mat_0;
      ae_int16x4 *p16x4_mat_1 = (ae_int16x4 *)p_mat_0;
      AE_ADDCIRC16X4_XC(p16x4_mat_1, row_stride1*sizeof(WORD8)); 
      p_mat_1 = (ae_int8x16 *)p16x4_mat_1;

      ae_valignx2 mat_align0, mat_align1;

      AE_LA8X8X2POS_PC(mat_align0, p_mat_0);
      AE_LA8X8X2POS_PC(mat_align1, p_mat_1);

      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);
      ae_int4x16 vec_4b_0_0, vec_4b_0_1;
      ae_int4x16 vec_4b_1_0, vec_4b_1_1;

      ae_int16x4 dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604, 0x03010200));
      ae_int16x4 vec_interleaved0_16x4, vec_interleaved2_16x4, vec_interleaved4_16x4, vec_interleaved6_16x4;
      for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, 16 * sizeof(WORD8));
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);
        AE_LA8X8X2_IC(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, mat_align0, p_mat_0);

        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, 16 * sizeof(WORD8));
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);
        AE_LA8X8X2_IC(mat1_0, mat1_1, mat_align1, p_mat_1);
        AE_LA8X8X2_IC(mat1_2, mat1_3, mat_align1, p_mat_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
        AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
        AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
        AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);

        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved0, vec_interleaved0, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved2, vec_interleaved2, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved4, vec_interleaved4, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved6, vec_interleaved6, mat_plus_zb0_6, mat_plus_zb0_7);  

        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved0, vec_interleaved0, mat_plus_zb1_0, mat_plus_zb1_1);
        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved2, vec_interleaved2, mat_plus_zb1_2, mat_plus_zb1_3);
        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved4, vec_interleaved4, mat_plus_zb1_4, mat_plus_zb1_5);
        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved6, vec_interleaved6, mat_plus_zb1_6, mat_plus_zb1_7);
      }        
      if(cols_count != cols1)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, 16 * sizeof(WORD8));
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift_0), rem_cols_shift_0));
        vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);
        AE_LA8X8X2_IC(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, mat_align0, p_mat_0);
        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, 16 * sizeof(WORD8));
        vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift_0), rem_cols_shift_0));
        vec1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_LA8X8X2_IC(mat1_0, mat1_1, mat_align1, p_mat_1);
        AE_LA8X8X2_IC(mat1_2, mat1_3, mat_align1, p_mat_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
        AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
        AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
        AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);

        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved0, vec_interleaved0, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved2, vec_interleaved2, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved4, vec_interleaved4, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved6, vec_interleaved6, mat_plus_zb0_6, mat_plus_zb0_7);

        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved0, vec_interleaved0, mat_plus_zb1_0, mat_plus_zb1_1);
        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved2, vec_interleaved2, mat_plus_zb1_2, mat_plus_zb1_3);
        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved4, vec_interleaved4, mat_plus_zb1_4, mat_plus_zb1_5);
        AE_MULA8Q4X16(acc1, dummy_acc, vec_interleaved6, vec_interleaved6, mat_plus_zb1_6, mat_plus_zb1_7);
      }

      ae_int32x2 bias32x2_0 =  AE_MOVDA32X2(p_bias[vec_itr], p_bias[vec_itr+1]);

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc1), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc2, acc4, acc0, acc1, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc3, acc5, acc0, acc1, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
#else
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc2, acc4, acc0, acc1, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc3, acc5, acc0, acc1, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
#endif

      ae_int16x4 out0, out1;
      out0 = AE_SAT16X4(acc2, acc3);
      out1 = AE_SAT16X4(acc4, acc5);

      out0 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out0), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));
      out1 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out1), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));

      ae_int8x8 out_0 = AE_SAT8X8X16(out0, out0);
      ae_int8x8 out_1 = AE_SAT8X8X16(out1, out1);

      AE_SW_S8_7_X(out_0, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_4_X(out_0, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_7_X(out_1, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+1)*out_row_offset);
      AE_SW_S8_4_X(out_1, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+1)*out_row_offset);
    }
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc0 = AE_ZERO32();
      ae_int32x2 acc1 = AE_ZERO32();    
      ae_int32x2 acc2 = AE_ZERO32(); 

      ae_int32x2 dummy_acc;
      ae_int8x8 mat0_0, mat0_1, mat0_2, mat0_3;
      ae_int8x8 vec0_0, vec0_1;
      ae_int8x8 vec1_0, vec1_1;

      ae_int16x4 mat_plus_zb0_0, mat_plus_zb0_1;
      ae_int16x4 mat_plus_zb0_2, mat_plus_zb0_3;
      ae_int16x4 mat_plus_zb0_4, mat_plus_zb0_5;
      ae_int16x4 mat_plus_zb0_6, mat_plus_zb0_7;

      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);

      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_vec_batch_1  = (ae_int8x16 *)(&p_vec1[(vec_itr+1)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int8x16 *p_mat_0;

      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      p_mat_0 = (ae_int8x16 *)p16x4_mat_0;

      ae_valignx2 mat_align0;


      AE_LA8X8X2POS_PC(mat_align0, p_mat_0);

      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);
      ae_int4x16 vec_4b_0_0, vec_4b_0_1;
      ae_int4x16 vec_4b_1_0, vec_4b_1_1;

      ae_int16x4 dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604, 0x03010200));
      ae_int16x4 vec_interleaved0_16x4, vec_interleaved2_16x4, vec_interleaved4_16x4, vec_interleaved6_16x4;
      for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, 16 * sizeof(WORD8));
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);
        AE_LA8X8X2_IC(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, mat_align0, p_mat_0);

        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, 16 * sizeof(WORD8));
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);

        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved0, vec_interleaved0, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved2, vec_interleaved2, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved4, vec_interleaved4, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved6, vec_interleaved6, mat_plus_zb0_6, mat_plus_zb0_7);      
      }        
      if(cols_count != cols1)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, p_vec_batch_0, 16 * sizeof(WORD8));
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift_0), rem_cols_shift_0));
        vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);
        AE_LA8X8X2_IC(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, mat_align0, p_mat_0);
        AE_L8X8X2_IP(vec1_0, vec1_1, p_vec_batch_1, 16 * sizeof(WORD8));
        vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift_0), rem_cols_shift_0));
        vec1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_1_0 = AE_MOVINT4X16_FROMINT8X8(vec1_0);
        vec_4b_1_1 = AE_MOVINT4X16_FROMINT8X8(vec1_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_DSEL16X4(vec_interleaved0_16x4, vec_interleaved2_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_0), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_0), dsel);
        AE_DSEL16X4(vec_interleaved4_16x4, vec_interleaved6_16x4, AE_MOVINT16X4_FROMINT4X16(vec_4b_0_1), AE_MOVINT16X4_FROMINT4X16(vec_4b_1_1), dsel);

        ae_int4x16 vec_interleaved0 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved0_16x4);
        ae_int4x16 vec_interleaved2 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved2_16x4);
        ae_int4x16 vec_interleaved4 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved4_16x4);
        ae_int4x16 vec_interleaved6 = AE_MOVINT4X16_FROMINT16X4(vec_interleaved6_16x4);

        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved0, vec_interleaved0, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved2, vec_interleaved2, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved4, vec_interleaved4, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA8Q4X16(acc0, dummy_acc, vec_interleaved6, vec_interleaved6, mat_plus_zb0_6, mat_plus_zb0_7);
      }
     
      ae_int32x2 bias32x2_0 =  AE_MOVDA32X2(p_bias[vec_itr], p_bias[vec_itr+1]);

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(bias32x2_0)));

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc1, dummy_acc, acc0, acc0, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(acc2, dummy_acc, acc0, acc0, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
#else
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc1, dummy_acc, acc0, acc0, p_out_multiplier[vec_itr], left_shift1, right_shift1);
      MPY_BY_QUANT_MULT_X2X2_OUT32(acc2, dummy_acc, acc0, acc0, p_out_multiplier[vec_itr+1], left_shift2, right_shift2);
#endif
      ae_int16x4 out0;
      out0 = AE_SAT16X4(acc1, acc2);
      out0 = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(out0), AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(out_zero_bias))));

      ae_int8x8 out_0 = AE_SAT8X8X16(out0, out0);

      AE_SW_S8_7_X(out_0, (ae_int8 *)p_out, (vec_itr+0)*out_col_offset + (m_itr+0)*out_row_offset);
      AE_SW_S8_4_X(out_0, (ae_int8 *)p_out, (vec_itr+1)*out_col_offset + (m_itr+0)*out_row_offset); 
    }
  }
  for(; vec_itr < vec_count; vec_itr++)
  {
    int left_shift;
#if TFLITE_SINGLE_ROUNDING      
    left_shift = p_out_shift[vec_itr];
#if XCHAL_HAVE_HIFI5S
    left_shift = 31 - left_shift;
    left_shift = (left_shift << 16) | left_shift;
#endif
#else
    int right_shift;
    left_shift = p_out_shift[vec_itr]<0?0:p_out_shift[vec_itr];
    right_shift = p_out_shift[vec_itr]>0?0:-p_out_shift[vec_itr];
#endif   
    for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
    {
      ae_int32x2 acc0 = AE_ZERO32();
      ae_int32x2 acc1 = AE_ZERO32();    
      ae_int32x2 acc2 = AE_ZERO32();
      ae_int32x2 acc3 = AE_ZERO32();  

      ae_int32x2 dummy_acc;
      ae_int32x2 dummy_acc1;
      ae_int8x8 mat0_0, mat0_1, mat0_2, mat0_3;
      ae_int8x8 mat1_0, mat1_1, mat1_2, mat1_3;
      ae_int8x8 vec0_0, vec0_1;

      ae_int16x4 mat_plus_zb0_0, mat_plus_zb0_1;
      ae_int16x4 mat_plus_zb0_2, mat_plus_zb0_3;
      ae_int16x4 mat_plus_zb0_4, mat_plus_zb0_5;
      ae_int16x4 mat_plus_zb0_6, mat_plus_zb0_7;

      ae_int16x4 mat_plus_zb1_0, mat_plus_zb1_1;
      ae_int16x4 mat_plus_zb1_2, mat_plus_zb1_3;
      ae_int16x4 mat_plus_zb1_4, mat_plus_zb1_5;
      ae_int16x4 mat_plus_zb1_6, mat_plus_zb1_7;

      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);

      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);

      ae_int8x16 *p_mat_0, *p_mat_1;
      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      p_mat_0 = (ae_int8x16 *)p16x4_mat_0;
      ae_int16x4 *p16x4_mat_1 = (ae_int16x4 *)p_mat_0;
      AE_ADDCIRC16X4_XC(p16x4_mat_1, row_stride1*sizeof(WORD8)); 
      p_mat_1 = (ae_int8x16 *)p16x4_mat_1;

      ae_valignx2 mat_align0, mat_align1;
      ae_valignx2 vec_align0;

      AE_LA8X8X2POS_PC(mat_align0, p_mat_0);
      AE_LA8X8X2POS_PC(mat_align1, p_mat_1);
      
      vec_align0 = AE_LA128_PP(p_vec_batch_0);

      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);

      ae_int4x16 vec_4b_0_0, vec_4b_0_1;
      for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
      {
        AE_LA8X8X2_IP(vec0_0, vec0_1, vec_align0, p_vec_batch_0);
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);
        AE_LA8X8X2_IC(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, mat_align0, p_mat_0);

        AE_LA8X8X2_IC(mat1_0, mat1_1, mat_align1, p_mat_1);
        AE_LA8X8X2_IC(mat1_2, mat1_3, mat_align1, p_mat_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
        AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
        AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
        AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_0_0, vec_4b_0_0, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_0_0, vec_4b_0_0, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_0_1, vec_4b_0_1, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_0_1, vec_4b_0_1, mat_plus_zb0_6, mat_plus_zb0_7);

        AE_MULA4O4X16(acc2, dummy_acc, acc3, dummy_acc1, vec_4b_0_0, vec_4b_0_0, mat_plus_zb1_0, mat_plus_zb1_1);
        AE_MULA4O4X16(dummy_acc, acc2, dummy_acc1, acc3, vec_4b_0_0, vec_4b_0_0, mat_plus_zb1_2, mat_plus_zb1_3);
        AE_MULA4O4X16(acc2, dummy_acc, acc3, dummy_acc1, vec_4b_0_1, vec_4b_0_1, mat_plus_zb1_4, mat_plus_zb1_5);
        AE_MULA4O4X16(dummy_acc, acc2, dummy_acc1, acc3, vec_4b_0_1, vec_4b_0_1, mat_plus_zb1_6, mat_plus_zb1_7);        
      }        
      if(cols_count != cols1)
      {
        AE_LA8X8X2_IP(vec0_0, vec0_1, vec_align0, p_vec_batch_0);
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift_0), rem_cols_shift_0));
        vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift_1), rem_cols_shift_1));          
        vec_4b_0_0 = AE_MOVINT4X16_FROMINT8X8(vec0_0);
        vec_4b_0_1 = AE_MOVINT4X16_FROMINT8X8(vec0_1);
        AE_LA8X8X2_IC(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IC(mat0_2, mat0_3, mat_align0, p_mat_0);

        AE_LA8X8X2_IC(mat1_0, mat1_1, mat_align1, p_mat_1);
        AE_LA8X8X2_IC(mat1_2, mat1_3, mat_align1, p_mat_1);

        AE_SUBW8(mat_plus_zb0_0, mat_plus_zb0_1, mat0_0, mat_zb);
        AE_SUBW8(mat_plus_zb0_2, mat_plus_zb0_3, mat0_1, mat_zb);
        AE_SUBW8(mat_plus_zb0_4, mat_plus_zb0_5, mat0_2, mat_zb);
        AE_SUBW8(mat_plus_zb0_6, mat_plus_zb0_7, mat0_3, mat_zb);

        AE_SUBW8(mat_plus_zb1_0, mat_plus_zb1_1, mat1_0, mat_zb);
        AE_SUBW8(mat_plus_zb1_2, mat_plus_zb1_3, mat1_1, mat_zb);
        AE_SUBW8(mat_plus_zb1_4, mat_plus_zb1_5, mat1_2, mat_zb);
        AE_SUBW8(mat_plus_zb1_6, mat_plus_zb1_7, mat1_3, mat_zb);

        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_0_0, vec_4b_0_0, mat_plus_zb0_0, mat_plus_zb0_1);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_0_0, vec_4b_0_0, mat_plus_zb0_2, mat_plus_zb0_3);
        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_0_1, vec_4b_0_1, mat_plus_zb0_4, mat_plus_zb0_5);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_0_1, vec_4b_0_1, mat_plus_zb0_6, mat_plus_zb0_7);

        AE_MULA4O4X16(acc2, dummy_acc, acc3, dummy_acc1, vec_4b_0_0, vec_4b_0_0, mat_plus_zb1_0, mat_plus_zb1_1);
        AE_MULA4O4X16(dummy_acc, acc2, dummy_acc1, acc3, vec_4b_0_0, vec_4b_0_0, mat_plus_zb1_2, mat_plus_zb1_3);
        AE_MULA4O4X16(acc2, dummy_acc, acc3, dummy_acc1, vec_4b_0_1, vec_4b_0_1, mat_plus_zb1_4, mat_plus_zb1_5);
        AE_MULA4O4X16(dummy_acc, acc2, dummy_acc1, acc3, vec_4b_0_1, vec_4b_0_1, mat_plus_zb1_6, mat_plus_zb1_7);  
      }
      
      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S_HL_LH(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(acc1)));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S_HL_LH(AE_MOVF32X2_FROMINT32X2(acc2), AE_MOVF32X2_FROMINT32X2(acc3)));

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(SW_MOVDA32(p_bias[vec_itr]))));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc1), AE_MOVF32X2_FROMINT32X2(SW_MOVDA32(p_bias[vec_itr]))));

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc0, acc0, p_out_multiplier[vec_itr], left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc1, acc1, p_out_multiplier[vec_itr], left_shift, right_shift);
#else
      MPY_BY_QUANT_MULT_X2_OUT32(acc0, acc0, p_out_multiplier[vec_itr], left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(acc1, acc1, p_out_multiplier[vec_itr], left_shift, right_shift);
#endif

      acc0 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(SW_MOVDA32(out_zero_bias))));
      acc1 = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc1), AE_MOVF32X2_FROMINT32X2(SW_MOVDA32(out_zero_bias))));

      AE_MINMAX32(acc0, min_int8, max_int8);     
      AE_MINMAX32(acc1, min_int8, max_int8);       
      
      p_out_tmp0 = &(p_out[(vec_itr+0)*out_col_offset + (m_itr+0)*out_row_offset]);
      p_out_tmp1 = &(p_out[(vec_itr+0)*out_col_offset + (m_itr+1)*out_row_offset]);

      *p_out_tmp0 = (WORD8)AE_MOVAD32_H(acc0);
      *p_out_tmp1 = (WORD8)AE_MOVAD32_H(acc1);
    }
    for(; m_itr < (rows); m_itr ++)
    {
      ae_int32x2 acc = AE_ZERO32();
      ae_int32x2 acc0 = AE_ZERO32();
      ae_int32x2 acc1 = AE_ZERO32();     
      ae_int32x2 dummy_acc = AE_ZERO32();
      ae_int32x2 dummy_acc1 = AE_ZERO32();
      ae_int8x8 mat0, mat1, mat2, mat3;
      ae_int8x8 vec0, vec1;
      ae_int16x4 mat_plus_zb0, mat_plus_zb1;
      ae_int16x4 mat_plus_zb2, mat_plus_zb3;
      ae_int16x4 mat_plus_zb4, mat_plus_zb5;
      ae_int16x4 mat_plus_zb6, mat_plus_zb7;
      ae_int8x8 mat_zb = AE_MOVDA8(-mat1_offset);
      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)(&p_vec1[(vec_itr)*PADDED_SIZE((vec_stride/2), 16)]);
      ae_int16x4 *p16x4_mat_0 = (ae_int16x4 *)p_mat1;
      AE_ADDCIRC16X4_XC(p16x4_mat_0, (m_itr)*row_stride1*sizeof(WORD8)); 
      ae_int8x16 *p_mat_0 = (ae_int8x16 *)p16x4_mat_0;

      ae_valignx2 mat_align;
      ae_valignx2 vec_align;

      AE_LA8X8X2POS_PC(mat_align, p_mat_0);
      vec_align = AE_LA128_PP(p_vec_batch_0);


      int rem_cols_shift_0, rem_cols_shift_1;

      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

      int cols_count= cols1-(cols1&31);
      ae_int4x16 vec_4b_0, vec_4b_1;
      for(c_itr = 0; c_itr < cols_count >> 5; c_itr++)
      {
        AE_LA8X8X2_IP(vec0, vec1, vec_align, p_vec_batch_0);
        vec_4b_0 = AE_MOVINT4X16_FROMINT8X8(vec0);
        vec_4b_1 = AE_MOVINT4X16_FROMINT8X8(vec1);
        AE_LA8X8X2_IC(mat0, mat1, mat_align, p_mat_0);
        AE_LA8X8X2_IC(mat2, mat3, mat_align, p_mat_0);
        AE_SUBW8(mat_plus_zb0, mat_plus_zb1, mat0, mat_zb);
        AE_SUBW8(mat_plus_zb2, mat_plus_zb3, mat1, mat_zb);
        AE_SUBW8(mat_plus_zb4, mat_plus_zb5, mat2, mat_zb);
        AE_SUBW8(mat_plus_zb6, mat_plus_zb7, mat3, mat_zb);
        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_0, vec_4b_0, mat_plus_zb0, mat_plus_zb1);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_0, vec_4b_0, mat_plus_zb2, mat_plus_zb3);
        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_1, vec_4b_1, mat_plus_zb4, mat_plus_zb5);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_1, vec_4b_1, mat_plus_zb6, mat_plus_zb7);          
      }
      if(cols_count != cols1)
      {
        AE_LA8X8X2_IP(vec0, vec1, vec_align, p_vec_batch_0);
        vec0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0), rem_cols_shift_0), rem_cols_shift_0));
        vec1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1), rem_cols_shift_1), rem_cols_shift_1));
        vec_4b_0 = AE_MOVINT4X16_FROMINT8X8(vec0);
        vec_4b_1 = AE_MOVINT4X16_FROMINT8X8(vec1);
        AE_LA8X8X2_IC(mat0, mat1, mat_align, p_mat_0);
        AE_LA8X8X2_IC(mat2, mat3, mat_align, p_mat_0);
        AE_SUBW8(mat_plus_zb0, mat_plus_zb1, mat0, mat_zb);
        AE_SUBW8(mat_plus_zb2, mat_plus_zb3, mat1, mat_zb);
        AE_SUBW8(mat_plus_zb4, mat_plus_zb5, mat2, mat_zb);
        AE_SUBW8(mat_plus_zb6, mat_plus_zb7, mat3, mat_zb);
        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_0, vec_4b_0, mat_plus_zb0, mat_plus_zb1);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_0, vec_4b_0, mat_plus_zb2, mat_plus_zb3);
        AE_MULA4O4X16(acc0, dummy_acc, acc1, dummy_acc1, vec_4b_1, vec_4b_1, mat_plus_zb4, mat_plus_zb5);
        AE_MULA4O4X16(dummy_acc, acc0, dummy_acc1, acc1, vec_4b_1, vec_4b_1, mat_plus_zb6, mat_plus_zb7);     
      }
      acc = AE_MOVINT32X2_FROMF32X2(AE_ADD32S_HL_LH(AE_MOVF32X2_FROMINT32X2(acc0), AE_MOVF32X2_FROMINT32X2(acc1)));
      acc = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc), AE_MOVF32X2_FROMINT32X2(SW_MOVDA32(p_bias[vec_itr]))));
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc, acc, p_out_multiplier[vec_itr], left_shift, right_shift);
#else      
      MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[vec_itr], left_shift, right_shift);
#endif      
      acc = AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(acc), AE_MOVF32X2_FROMINT32X2(SW_MOVDA32(out_zero_bias))));
      AE_MINMAX32(acc, min_int8, max_int8);     
      p_out_tmp = &(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]);
      *p_out_tmp = (WORD8)AE_MOVAD32_H(acc);
    }
  }
  return 0;  

}
