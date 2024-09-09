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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

#ifdef AE_MULAZB8Q8X8
  #define MAT_VEC_MAC(...)  AE_MULAZB8Q8X8(__VA_ARGS__)
#else
  #define MAT_VEC_MAC(...)  AE_MULA8Q8X8(__VA_ARGS__)
#endif

#define PACK_32X2(dst1, src1, src2) \
  dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

#define PACK_32X2_per_chan(dst_0, dst_1, src_0, src_1, src_2, src_3) \
{\
  ae_int8x8 dst_temp1, dst_temp2; \
  dst_temp1 = AE_SEL8X8I(AE_MOVINT8X8_FROMINT16X4(src_0), AE_MOVINT8X8_FROMINT16X4(src_1), 25); \
  dst_temp2 = AE_SEL8X8I(AE_MOVINT8X8_FROMINT16X4(src_2), AE_MOVINT8X8_FROMINT16X4(src_3), 25); \
  AE_DSEL8X8(dst_0, dst_1, dst_temp1, dst_temp2, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x3175b9fd, 0x2064a8ec))); \
}

extern const long long g_sel_pattern[16];
extern const long long pre_loop_sel_pattern[16];
extern const long long post_loop_sel_pattern[16];

static void special_function_for_cols_mul_32
    (WORD8*        p_out_0
    ,const WORD8*  p_mat1
    ,const WORD8*  p_vec1_0
    ,const WORD32* p_bias_0
    ,WORD32        n_rows
    ,WORD32        n_vecs
    ,WORD32        cols
    ,const WORD32* p_out_mul
    ,const WORD32* p_out_shift
    ,WORD32        vec1_zero_bias
    ,WORD32        out_z_b
    ,WORD32        out_stride
    ,WORD32        row_offset
    ,WORD32        vec_offset
    ,WORD32        out_offset
    ,WORD32        out_activation_min
    ,WORD32        out_activation_max
    )
{
  WORD8 * __restrict__ p_dst_0;
  ae_int8x16 * __restrict__ p_vec_0;
  ae_int8x16 * __restrict__ p_vec_1;
  int c_itr;
  int m_itr = 0, vec_itr = 0;
  ae_int32x2 acc_buffer[4];

#if TFLITE_SINGLE_ROUNDING
  /* only one (original) shift value is needed in single rounding */
  int p_left_mult[4], p_out_mult[4];
#else
  int p_left_mult[4], p_right_mult[4], p_out_mult[4];
#endif

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;

  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3; 
  ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3; 
  ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;
  
  ae_int8x8 *p_mat1_0, *p_mat1_1, *p_mat1_2, *p_mat1_3;
  
  ae_valignx2 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2, align_p_mat1_3;

  for(m_itr = 0; m_itr < (n_rows & ~(4 - 1)); m_itr += 4)
  {
    ae_int32x2 acc_row0 = ZERO32;
    ae_int32x2 acc_row1 = ZERO32;
    
    ae_int32x2 d_bias0 = AE_ZERO32(), d_bias1 = AE_ZERO32();

#ifndef AE_MULAZB8Q8X8
    p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
    p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
    p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
    p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

    align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
    align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
    align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
    align_p_mat1_3 = AE_LA128_PP(p_mat1_3);
    
    ae_int8x8 vec_z_b = AE_MOVDA8(-vec1_zero_bias);

    #pragma loop_count min=1
    #pragma no_unroll
    for(c_itr = 0; c_itr < cols>>5; c_itr++)
    {
      /* Load 4 rows */
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
      AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
      AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16*)p_mat1_3);
      AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16*)p_mat1_3);

      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 , vec_z_b);
    }
#else
    AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec1_zero_bias, 0)));
#endif
   
    if(p_bias_0)
    {
      ae_valignx2 align_p_bias = AE_LA128_PP((ae_int32x4 *)p_bias_0);
      AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, (ae_int32x4 *)p_bias_0);
    }
    
    acc_row0 = AE_SUB32S(d_bias0, acc_row0);
    acc_row1 = AE_SUB32S(d_bias1, acc_row1);

    AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);

    p_dst_0 = (WORD8*)p_out_0 + (m_itr + 0) * out_stride;

#if TFLITE_SINGLE_ROUNDING
    p_left_mult[0] = p_out_shift[m_itr + 0];
    p_left_mult[1] = p_out_shift[m_itr + 1];
    p_left_mult[2] = p_out_shift[m_itr + 2];
    p_left_mult[3] = p_out_shift[m_itr + 3];
    
    ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
    AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_mult, 0);
    (void)r_mult_01;
    (void)r_mult_23;
#else /* #if TFLITE_SINGLE_ROUNDING */
    p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
    p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
    p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
    p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

    p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
    p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
    p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
    p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));
#endif /* #if TFLITE_SINGLE_ROUNDING */

    {
      ae_int32x2 d_temp0, d_temp1;
      ae_valignx2 align_p_mult = AE_LA128_PP((ae_int32x4 *)p_out_mul);
      AE_LA32X2X2_IP(d_temp0, d_temp1, align_p_mult, (ae_int32x4 *)p_out_mul);

      AE_S32X2X2_I(d_temp0, d_temp1, (ae_int32x4*)p_out_mult, 0);
    }
#if TFLITE_SINGLE_ROUNDING
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)p_out_mult, 0);
#endif

    for (vec_itr = 0; vec_itr < (n_vecs & ~(4 - 1)); vec_itr += 4)
    {
      ae_int32x2 acc_row0_vec0; 
      ae_int32x2 acc_row1_vec0;
      ae_int32x2 acc_row0_vec1;
      ae_int32x2 acc_row1_vec1; 
      ae_int32x2 acc_row0_vec2; 
      ae_int32x2 acc_row1_vec2; 
      ae_int32x2 acc_row0_vec3; 
      ae_int32x2 acc_row1_vec3;

      /* Initialize accumulators */
      AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buffer, 0);
      AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
      AE_L32X2X2_I(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buffer, 0);
      AE_L32X2X2_I(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buffer, 0);

      p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
      p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
      p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
      p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

      align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      p_vec_0  = (ae_int8x16 *)(p_vec1_0 + vec_itr * vec_offset);
      p_vec_1  = (ae_int8x16 *)((WORD8 *)p_vec_0 + 2*cols);

#pragma loop_count min=1
#pragma no_unroll
      for(c_itr = 0; c_itr < cols>>5; c_itr++)
      {
        /* Load 4 rows */
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16*)p_mat1_3);
        AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16*)p_mat1_3);

        /* Load  4 vectors  */
        AE_L8X8X2_X(vec1_batch_0, vec1_batch_1, p_vec_0, cols);
        AE_L8X8X2_X(vec1_batch_2, vec1_batch_3, p_vec_0, cols+16);
        AE_L8X8X2_X(vec3_batch_0, vec3_batch_1, p_vec_1, cols);
        AE_L8X8X2_X(vec3_batch_2, vec3_batch_3, p_vec_1, cols+16);

        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);
        AE_L8X8X2_IP(vec0_batch_2, vec0_batch_3, p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, p_vec_1, 16);
        AE_L8X8X2_IP(vec2_batch_2, vec2_batch_3, p_vec_1, 16);
        
        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec1_batch_2);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec2_batch_2);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec3_batch_2);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec1_batch_3);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec2_batch_3);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec3_batch_3);
      }   

      /* Apply quantization */
      ae_int16x4 out_0, out_1, out_2, out_3;
#if !TFLITE_SINGLE_ROUNDING
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_mult, 0);
      AE_L32X2X2_I(r_mult_01, r_mult_23, (ae_int32x4 *)p_right_mult, 0);
      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)p_out_mult, 0);
#endif /* #if TFLITE_SINGLE_ROUNDING */

      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_0, out_1, acc_row0_vec0, acc_row1_vec0, acc_row0_vec1, acc_row1_vec1, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_z_b);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_2, out_3, acc_row0_vec2, acc_row1_vec2, acc_row0_vec3, acc_row1_vec3, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_z_b);

      AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      /* Store output */
      STORE_16x4x2_8x4x2(out_0, out_1, p_dst_0, out_offset);
	  STORE_16x4x2_8x4x2(out_2, out_3, p_dst_0, out_offset);
    }
    
    for (; vec_itr < n_vecs; vec_itr++)
    {
      ae_int32x2 acc_row0_vec0; 
      ae_int32x2 acc_row1_vec0;

      /* Initialize accumulators */
      AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buffer, 0);

      p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
      p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
      p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
      p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

      align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      p_vec_0  = (ae_int8x16 *)(p_vec1_0 + vec_itr * vec_offset);

#pragma loop_count min=1
      for(c_itr = 0; c_itr < cols>>5; c_itr++)
      {
        /* Load 4 rows */
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16*)p_mat1_3);
        AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16*)p_mat1_3);

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec0_batch_2, vec0_batch_3, (ae_int8x16*)p_vec_0, 16);
        
        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3);
      }   

      /* Apply quantization */
      ae_int16x4 out_0;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_mult, 0);
#if TFLITE_SINGLE_ROUNDING
      (void)r_mult_01;
      (void)r_mult_23;
#else /* #if TFLITE_SINGLE_ROUNDING */
      AE_L32X2X2_I(r_mult_01, r_mult_23, (ae_int32x4 *)p_right_mult, 0);
#endif /* #if TFLITE_SINGLE_ROUNDING */
      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)p_out_mult, 0);

      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_z_b);

      AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      /* Store output */
      STORE_16x4_8x4(out_0, p_dst_0, out_offset);	
    }
  }
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_0_1
    ,ae_int32x2* out_0_2
    ,ae_int32x2* out_0_3
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_1_1
    ,ae_int32x2* out_1_2
    ,ae_int32x2* out_1_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset 
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) vec1_zero_bias;
  int c_itr = 0;
  int rem_cols = cols & 15;
  
#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift_0 = ((rem_cols)<=8)?(8-(rem_cols))*8:0;
  int rem_cols_shift_1 = ((rem_cols)>8)?(16-(rem_cols))*8:64;
#else
  ae_int8x8 scratch;
  int set_zero_mask_0 = 0x0FF >> rem_cols;
  int set_zero_mask_1 = rem_cols>8 ? 0x0FF>>(rem_cols-8) : 0x0FF;
#endif
  
  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1;
  ae_int8x8 vec2_batch_0, vec2_batch_1;
  ae_int8x8 vec3_batch_0, vec3_batch_1;

  ae_int8x8* p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
  ae_int8x8* p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
  ae_int8x8* p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;
  
  int cols_count = cols -(cols & 15);

  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);
    AE_L8X8X2_IP(vec1_batch_0, vec1_batch_1, (ae_int8x16 *)p_vec_1, 16);
    AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, (ae_int8x16 *)p_vec_2, 16);
    AE_L8X8X2_IP(vec3_batch_0, vec3_batch_1, (ae_int8x16 *)p_vec_3, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row0_1);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row1_1);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row2_1);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row3_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
  while(c_itr < cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

#ifndef AE_MULAZB8Q8X8
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));
    rem_cols_shift_0 = rem_cols_shift_1;
#else
    int vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, mat1_row0_0, mat1_row0_0, AE_MOVDA8(vec_z_b), set_zero_mask_0);
    AE_MOVT8X16_L(scratch, mat1_row1_0, mat1_row1_0, AE_MOVDA8(vec_z_b), set_zero_mask_0);
    AE_MOVT8X16_L(scratch, mat1_row2_0, mat1_row2_0, AE_MOVDA8(vec_z_b), set_zero_mask_0);
    AE_MOVT8X16_L(scratch, mat1_row3_0, mat1_row3_0, AE_MOVDA8(vec_z_b), set_zero_mask_0);
    set_zero_mask_0 = set_zero_mask_1;
#endif

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    c_itr += 8;
  }

  *out_0_0 = acc_row0_vec0;
  *out_0_1 = acc_row0_vec1;
  *out_0_2 = acc_row0_vec2;
  *out_0_3 = acc_row0_vec3;
       
  *out_1_0 = acc_row1_vec0;
  *out_1_1 = acc_row1_vec1;
  *out_1_2 = acc_row1_vec2;
  *out_1_3 = acc_row1_vec3;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      mat1_zero_bias
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) mat1_zero_bias;
  (VOID) vec1_zero_bias;
  int c_itr = 0;
  int rem_cols = cols & 15;
  
#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift_0 = ((rem_cols)<=8)?(8-(rem_cols))*8:0;
  int rem_cols_shift_1 = ((rem_cols)>8)?(16-(rem_cols))*8:64;
#else
  ae_int8x8 scratch;
  int set_zero_mask_0 = 0x0FF >> rem_cols;
  int set_zero_mask_1 = rem_cols>8 ? 0x0FF>>(rem_cols-8) : 0x0FF;
#endif

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;
  ae_int8x8 vec0_batch_0, vec0_batch_1; 

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  
  int cols_count=cols-(cols&15);

  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
  while(c_itr < cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    
#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift_0), rem_cols_shift_0));
    rem_cols_shift_0 = rem_cols_shift_1;
#else
    int vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), set_zero_mask_0);
    set_zero_mask_0 = set_zero_mask_1;
#endif

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    c_itr += 8;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) vec1_zero_bias;
  int c_itr = 0;
  int rem_cols = cols & 15;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift_0 = ((rem_cols)<=8)?(8-(rem_cols))*8:0;
  int rem_cols_shift_1 = ((rem_cols)>8)?(16-(rem_cols))*8:64;
#else
  ae_int8x8 scratch;
  int set_zero_mask_0 = 0x0FF >> rem_cols;
  int set_zero_mask_1 = rem_cols>8 ? 0x0FF>>(rem_cols-8) : 0x0FF;
#endif

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 mat1_row0_0, mat1_row0_1;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  int cols_count = cols - (cols & 15);

  #pragma no_unroll
  for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
  {
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0);
    MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 ,vec0_batch_1);
  }
 
  //Remainder loop for cols
  c_itr <<= 4;
  while(c_itr < cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    
#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift_0), rem_cols_shift_0));
    rem_cols_shift_0 = rem_cols_shift_1;
#else
    int vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), set_zero_mask_0);
    set_zero_mask_0 = set_zero_mask_1;
#endif

    MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0);
    c_itr += 8;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_0_1
    ,ae_int32x2* out_0_2
    ,ae_int32x2* out_0_3
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_1_1
    ,ae_int32x2* out_1_2
    ,ae_int32x2* out_1_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) vec1_zero_bias;
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1; 
  ae_int8x8 vec2_batch_0, vec2_batch_1; 
  ae_int8x8 vec3_batch_0, vec3_batch_1; 

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

#ifndef AE_MULAZB8Q8X8
  int pre_loop_shift;
#else
  int pre_loop_selector;
#endif

  int align_offset = ((unsigned int)p_mat1_0 & 0x7);
  pre_loop_count = 8 - align_offset;

#ifndef AE_MULAZB8Q8X8
  pre_loop_shift = align_offset * 8;
#else
  pre_loop_selector = 0x0FF << pre_loop_count;
#endif

  p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step
  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 15):((cols + align_offset) & 15);
  loop_count >>= 4;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift_0 = ((post_loop_count)<=8)?(8-(post_loop_count))*8:0;
  int rem_cols_shift_1 = ((post_loop_count)>8)?(16-(post_loop_count))*8:64;
#else
  ae_int8x8 scratch;
  int first_selector = 0x0FF >> post_loop_count;
  int   rem_selector = post_loop_count>8 ? 0x0FF>>(post_loop_count-8) : 0x0FF;
#endif

  int mask_start_end = ((cols + align_offset) < 16)?0:1;

  ae_int8x8* p_mat1_1 = p_mat1_0 + row_offset; //next 8th row 
  ae_int8x8* p_mat1_2 = p_mat1_1 + row_offset; //next 8th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + row_offset; //next 8th row 

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign align_p_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign align_p_vec_3 = AE_LA64_PP(p_vec_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  /* Pre loop computation */
  AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
  AE_L8X8_IP(mat1_row0_1, p_mat1_0, 8);
  AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
  AE_L8X8_IP(mat1_row1_1, p_mat1_1, 8);
  AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
  AE_L8X8_IP(mat1_row2_1, p_mat1_2, 8);
  AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);
  AE_L8X8_IP(mat1_row3_1, p_mat1_3, 8);

  AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
  AE_LA8X8_IP(vec0_batch_1, align_p_vec_0, (ae_int8x8 *)p_vec_0);
  AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
  AE_LA8X8_IP(vec1_batch_1, align_p_vec_1, (ae_int8x8 *)p_vec_1);
  AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
  AE_LA8X8_IP(vec2_batch_1, align_p_vec_2, (ae_int8x8 *)p_vec_2);
  AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);
  AE_LA8X8_IP(vec3_batch_1, align_p_vec_3, (ae_int8x8 *)p_vec_3);

  if(align_offset)
  {
#ifndef AE_MULAZB8Q8X8
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), pre_loop_shift), pre_loop_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), pre_loop_shift), pre_loop_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), pre_loop_shift), pre_loop_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), pre_loop_shift), pre_loop_shift));
#else
    int vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), pre_loop_selector);
    AE_MOVT8X16_L(scratch, vec1_batch_0, vec1_batch_0, AE_MOVDA8(vec_z_b), pre_loop_selector);
    AE_MOVT8X16_L(scratch, vec2_batch_0, vec2_batch_0, AE_MOVDA8(vec_z_b), pre_loop_selector);
    AE_MOVT8X16_L(scratch, vec3_batch_0, vec3_batch_0, AE_MOVDA8(vec_z_b), pre_loop_selector);
#endif
  }

  if(mask_start_end)
  {
    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row0_1);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row1_1);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row2_1);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row3_1);
  }

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row0_1, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row1_1, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row2_1, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);
    AE_L8X8_IP(mat1_row3_1, p_mat1_3, 8);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec0_batch_1, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec1_batch_1, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec2_batch_1, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);
    AE_LA8X8_IP(vec3_batch_1, align_p_vec_3, (ae_int8x8 *)p_vec_3);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row0_1);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row1_1);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row2_1);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row3_1);
  }

  //Remainder loop for cols
  c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_shift = rem_cols_shift_0;
#else
  int selector = first_selector;
#endif

  while(c_itr < post_loop_count)
  {
    if(mask_start_end)
    {
      AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
      AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
      AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
      AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

      AE_LA8X8_IP(vec0_batch_0, align_p_vec_0 ,(ae_int8x8*)p_vec_0);
      AE_LA8X8_IP(vec1_batch_0, align_p_vec_1 ,(ae_int8x8*)p_vec_1);
      AE_LA8X8_IP(vec2_batch_0, align_p_vec_2 ,(ae_int8x8*)p_vec_2);
      AE_LA8X8_IP(vec3_batch_0, align_p_vec_3 ,(ae_int8x8*)p_vec_3);
    }

#ifndef AE_MULAZB8Q8X8
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));
    rem_shift = rem_cols_shift_1;
#else
    int vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), selector);
    AE_MOVT8X16_L(scratch, vec1_batch_0, vec1_batch_0, AE_MOVDA8(vec_z_b), selector);
    AE_MOVT8X16_L(scratch, vec2_batch_0, vec2_batch_0, AE_MOVDA8(vec_z_b), selector);
    AE_MOVT8X16_L(scratch, vec3_batch_0, vec3_batch_0, AE_MOVDA8(vec_z_b), selector);
    selector = rem_selector;
#endif

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    c_itr += 8;

    if(!mask_start_end && (c_itr < post_loop_count))
    {
      mat1_row0_0 = mat1_row0_1;
      mat1_row1_0 = mat1_row1_1;
      mat1_row2_0 = mat1_row2_1;
      mat1_row3_0 = mat1_row3_1;
      vec0_batch_0 = vec0_batch_1;
      vec1_batch_0 = vec1_batch_1;
      vec2_batch_0 = vec2_batch_1;
      vec3_batch_0 = vec3_batch_1;
    }
  }

  *out_0_0 = acc_row0_vec0;
  *out_0_1 = acc_row0_vec1;
  *out_0_2 = acc_row0_vec2;
  *out_0_3 = acc_row0_vec3;
       
  *out_1_0 = acc_row1_vec0;
  *out_1_1 = acc_row1_vec1;
  *out_1_2 = acc_row1_vec2;
  *out_1_3 = acc_row1_vec3;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) vec1_zero_bias;
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#else
  ae_int8x8 scratch;                          // only a placeholder, no initialization needed
  const int selector = 0x0FF >> (cols & 7);
#endif

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8* p_mat1_1 = p_mat1_0 + row_offset; //next 8th row 
  ae_int8x8* p_mat1_2 = p_mat1_1 + row_offset; //next 8th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + row_offset; //next 8th row 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    
    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else
    int vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), selector);
#endif

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_0_1
    ,ae_int32x2* out_0_2
    ,ae_int32x2* out_0_3
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_1_1
    ,ae_int32x2* out_1_2
    ,ae_int32x2* out_1_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) vec1_zero_bias;
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#else
  ae_int8x8 scratch;
  int selector = 0x0FF >> (cols & 7);
#endif

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int8x8 vec0_batch_0; 
  ae_int8x8 vec1_batch_0; 
  ae_int8x8 vec2_batch_0; 
  ae_int8x8 vec3_batch_0; 
  ae_int8x8 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2, align_p_mat1_3; 

  ae_int8x8* p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
  ae_int8x8* p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
  ae_int8x8* p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign align_p_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign align_p_vec_3 = AE_LA64_PP(p_vec_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  AE_SW_PRIME_64(p_mat1_0, align_p_mat1_0);
  AE_SW_PRIME_64(p_mat1_1, align_p_mat1_1);
  AE_SW_PRIME_64(p_mat1_2, align_p_mat1_2);
  AE_SW_PRIME_64(p_mat1_3, align_p_mat1_3);

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);

    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);
  }  

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);

#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_cols_shift), rem_cols_shift));
    vec2_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_batch_0), rem_cols_shift), rem_cols_shift));
    vec3_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_batch_0), rem_cols_shift), rem_cols_shift));
#else
    ae_int32  vec_z_b = -vec1_zero_bias;                                                                                    //  For selector[bit7:bit0], if bitN is set,
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), selector);   //    then, copy 'vec_b_z' into vecX_batch_Y[byteN]
    AE_MOVT8X16_L(scratch, vec1_batch_0, vec1_batch_0, AE_MOVDA8(vec_z_b), selector);   //    else, preserve vecX_batch_Y[byteN] as is.
    AE_MOVT8X16_L(scratch, vec2_batch_0, vec2_batch_0, AE_MOVDA8(vec_z_b), selector);   //  We want, dot(vec, mat) where the extra trailing bytes in 'vecX_batch_Y' are zero.
    AE_MOVT8X16_L(scratch, vec3_batch_0, vec3_batch_0, AE_MOVDA8(vec_z_b), selector);   //  But, since AE_MULAZB automatically subtracts 'vec_z_b' from 'vec',
                                                                                        //    we set the trailing bytes to vec_z_b, so that the result (vec-vec_z_b) is zero.
#endif

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_0_1 = acc_row0_vec1;
  *out_0_2 = acc_row0_vec2;
  *out_0_3 = acc_row0_vec3;

  *out_1_0 = acc_row1_vec0;
  *out_1_1 = acc_row1_vec1;
  *out_1_2 = acc_row1_vec2;
  *out_1_3 = acc_row1_vec3;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      mat1_zero_bias
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) mat1_zero_bias;
  (VOID) vec1_zero_bias;
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#else
  ae_int8x8 scratch;                          // only a placeholder, no initialization needed
  int selector = 0x0FF >> (cols & 7);
#endif

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    
    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    MAT_VEC_MAC(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else
    ae_int32  vec_z_b = -vec1_zero_bias;
    AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), selector);
#endif

    MAT_VEC_MAC(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      vec1_zero_bias
    )
{
  (VOID) vec1_zero_bias;
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols1 & 7) * 8;
#else
  ae_int8x8 scratch;                          // only a placeholder, no initialization needed
  int selector = 0x0FF >> (cols1 & 7);
#endif

  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);

  int cols_count = cols1 - (cols1 & 7);

  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);

    MAT_VEC_MAC(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    
#ifndef AE_MULAZB8Q8X8
      vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else
      int vec_z_b = -vec1_zero_bias;
      AE_MOVT8X16_L(scratch, vec0_batch_0, vec0_batch_0, AE_MOVDA8(vec_z_b), selector);
#endif

    MAT_VEC_MAC(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
    
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 vec1_zero_bias,
    const WORD32* __restrict__ p_out_multiplier,
    const WORD32* __restrict__ p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min || out_activation_max > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  int itr = 0;
  for(itr=0; itr<rows; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  ae_int32x2 acc_buffer[4];
  WORD8 * __restrict__ p_dst_0;
  ae_int8* __restrict__ p_vec_0;

  /* Iterators used in for loops */
  int m_itr, vec_itr;
  int ii;

  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

  /* Shifts to match with Tensorflow */
  int p_left_shift[4], p_right_shift[4];
#if TFLITE_SINGLE_ROUNDING
  (void)p_right_shift[0];
#endif

#ifndef AE_MULAZB8Q8X8
  ae_int8x8 vec_z_b = AE_MOVDA8(-vec1_zero_bias);
#else
  int       vec_z_b = -vec1_zero_bias;
#endif

#undef VEC_UNROLL
#define VEC_UNROLL 4

  /* Special case for cols == 8 */
  if(
      (cols1 == 8) &&
      (row_stride1 == 8) &&
      (vec_offset == 8) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *pt_bias;
    ae_valignx2 align_p_bias;
    if(p_bias)
    {
      pt_bias = (ae_int32x4 *)p_bias;
      align_p_bias = AE_LA128_PP(pt_bias);
    }
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0;
      ae_int8x8 mat1_row1_0;
      ae_int8x8 mat1_row2_0;
      ae_int8x8 mat1_row3_0;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);

      /* Load 4 rows */
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row1_0, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row3_0, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      
      ae_int32x2 d_bias0 = AE_ZERO32(), d_bias1 = AE_ZERO32();
      if(p_bias)
      {
        AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, pt_bias);
      }
      
#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32, acc_row1 = ZERO32;
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      
      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);
      
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);
#else
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
      AE_S32X2X2_I(d_bias0, d_bias1, (ae_int32x4*)acc_buffer, 0);
#endif

#if TFLITE_SINGLE_ROUNDING
      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];
      
      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
      (void)r_mult_23;
      (void)r_mult_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_mult[4], p_right_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;

      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

#if TFLITE_SINGLE_ROUNDING
      AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);
#endif
      
      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;
      p_vec_0  = (ae_int8 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
      {
        ae_int32x2 acc_row0_vec0; 
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1; 
        ae_int32x2 acc_row0_vec2; 
        ae_int32x2 acc_row1_vec2; 
        ae_int32x2 acc_row0_vec3; 
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);

        ae_int8x8 vec0_batch_0; 
        ae_int8x8 vec1_batch_0; 
        ae_int8x8 vec2_batch_0; 
        ae_int8x8 vec3_batch_0; 

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec1_batch_0, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec3_batch_0, (ae_int8x16*)p_vec_0, 16);
        
        // KERNEL_4x8_MUL_8x4(acc_row0_vec0, acc_row0_vec1, acc_row1_vec0, acc_row1_vec1.......

        MAT_VEC_MAC(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0);
        MAT_VEC_MAC(acc_row0_vec1, acc_row1_vec1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0);
        MAT_VEC_MAC(acc_row0_vec2, acc_row1_vec2, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0);
        MAT_VEC_MAC(acc_row0_vec3, acc_row1_vec3, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

#if !TFLITE_SINGLE_ROUNDING
        AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_01, r_mult_23, (ae_int32x4 *)ptr_right_mult, 0);
        AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);
#endif

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_0, out_1, acc_row0_vec0, acc_row1_vec0, acc_row0_vec1, acc_row1_vec1, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);
        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_2, out_3, acc_row0_vec2, acc_row1_vec2, acc_row0_vec3, acc_row1_vec3, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        /* Store output */
        STORE_16x4x2_8x4x2(out_0, out_1, p_dst_0, out_offset);
        STORE_16x4x2_8x4x2(out_2, out_3, p_dst_0, out_offset);
      }
      /*
         for (; vec_itr < vec_count; vec_itr++)
         {
         }
       */
    }
    return 0;
  }

  /* Special case for cols == 16 */
  if(
      (cols1 == 16) &&
      (row_stride1 == 16) &&
      (vec_offset == 16) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *pt_bias;
    ae_valignx2 align_p_bias;
    if(p_bias)
    {
      pt_bias = (ae_int32x4 *)p_bias;
      align_p_bias = AE_LA128_PP(pt_bias);
    }
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0, mat1_row0_1;
      ae_int8x8 mat1_row1_0, mat1_row1_1;
      ae_int8x8 mat1_row2_0, mat1_row2_1;
      ae_int8x8 mat1_row3_0, mat1_row3_1;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);

      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);

      ae_int32x2 d_bias0 = AE_ZERO32(), d_bias1 = AE_ZERO32();
      if(p_bias)
      {
        AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, pt_bias);
      }
      
#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32, acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      
      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);
      
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);
#else
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
      AE_S32X2X2_I(d_bias0, d_bias1, (ae_int32x4*)acc_buffer, 0);
#endif

#if TFLITE_SINGLE_ROUNDING
      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
      (void)r_mult_23;
      (void)r_mult_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_mult[4], p_right_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

#if TFLITE_SINGLE_ROUNDING
      AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);
#endif
      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;

      p_vec_0  = (ae_int8 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
      {
        ae_int32x2 acc_row0_vec0; 
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1; 
        ae_int32x2 acc_row0_vec2; 
        ae_int32x2 acc_row1_vec2; 
        ae_int32x2 acc_row0_vec3; 
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);
        ae_int8x8 vec0_batch_0, vec0_batch_1; 
        ae_int8x8 vec1_batch_0, vec1_batch_1; 
        ae_int8x8 vec2_batch_0, vec2_batch_1; 
        ae_int8x8 vec3_batch_0, vec3_batch_1;

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_batch_0, vec1_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec3_batch_0, vec3_batch_1, (ae_int8x16*)p_vec_0, 16);

        MAT_VEC_MAC(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0);
        MAT_VEC_MAC(acc_row0_vec1, acc_row1_vec1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0);
        MAT_VEC_MAC(acc_row0_vec2, acc_row1_vec2, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0);
        MAT_VEC_MAC(acc_row0_vec3, acc_row1_vec3, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0);

        MAT_VEC_MAC(acc_row0_vec0, acc_row1_vec0, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_1);
        MAT_VEC_MAC(acc_row0_vec1, acc_row1_vec1, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_1);
        MAT_VEC_MAC(acc_row0_vec2, acc_row1_vec2, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_1);
        MAT_VEC_MAC(acc_row0_vec3, acc_row1_vec3, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_1);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

#if !TFLITE_SINGLE_ROUNDING
        AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_01, r_mult_23, (ae_int32x4 *)ptr_right_mult, 0);
        AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);
#endif

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_0, out_1, acc_row0_vec0, acc_row1_vec0, acc_row0_vec1, acc_row1_vec1, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);
        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_2, out_3, acc_row0_vec2, acc_row1_vec2, acc_row0_vec3, acc_row1_vec3, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        /* Store output */
        STORE_16x4x2_8x4x2(out_0, out_1, p_dst_0, out_offset);
        STORE_16x4x2_8x4x2(out_2, out_3, p_dst_0, out_offset);		
      }
      /*
      for (; vec_itr < vec_count; vec_itr++)
      {
      }
      */
    }
    return 0;
  }

  /* Special case for cols == 24 */
  if(
      (cols1 == 24) &&
      (row_stride1 == 24) &&
      (vec_offset == 24) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *pt_bias;
    ae_valignx2 align_p_bias;
    if(p_bias)
    {
      pt_bias = (ae_int32x4 *)p_bias;
      align_p_bias = AE_LA128_PP(pt_bias);
    }
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2;
      ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2;
      ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2;
      ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);

      AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row0_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row0_2, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IP(mat1_row1_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row1_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row1_2, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IP(mat1_row2_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row2_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row2_2, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IP(mat1_row3_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row3_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row3_2, align_p_mat1_0, p_mat1_0);
      
      ae_int32x2 d_bias0 = AE_ZERO32(), d_bias1 = AE_ZERO32();
      if(p_bias)
      {
        AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, pt_bias);
      }

#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32, acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 , vec_z_b);
      
      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);
      
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);
#else
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
      AE_S32X2X2_I(d_bias0, d_bias1, (ae_int32x4*)acc_buffer, 0);
#endif

#if TFLITE_SINGLE_ROUNDING
      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
      (void)r_mult_23;
      (void)r_mult_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_mult[4], p_right_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;

      p_vec_0  = (ae_int8 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
      {
        ae_int32x2 acc_row0_vec0; 
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1; 
        ae_int32x2 acc_row0_vec2; 
        ae_int32x2 acc_row1_vec2; 
        ae_int32x2 acc_row0_vec3; 
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);

        ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2; 
        ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2; 
        ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2; 
        ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2; 

        /* Load  4 vectors  */
        AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec0_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec0_batch_2, (ae_int8x8*)p_vec_0, 8);
        AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec1_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec1_batch_2, (ae_int8x8*)p_vec_0, 8);
        AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec2_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec2_batch_2, (ae_int8x8*)p_vec_0, 8);
        AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec3_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec3_batch_2, (ae_int8x8*)p_vec_0, 8);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec1_batch_2);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec2_batch_2);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec3_batch_2);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

#if !TFLITE_SINGLE_ROUNDING
        AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_01, r_mult_23, (ae_int32x4 *)ptr_right_mult, 0);
#endif
        AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_0, out_1, acc_row0_vec0, acc_row1_vec0, acc_row0_vec1, acc_row1_vec1, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);
        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_2, out_3, acc_row0_vec2, acc_row1_vec2, acc_row0_vec3, acc_row1_vec3, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        /* Store output */
        STORE_16x4x2_8x4x2(out_0, out_1, p_dst_0, out_offset);
        STORE_16x4x2_8x4x2(out_2, out_3, p_dst_0, out_offset);	
      }
      /*
      for (; vec_itr < vec_count; vec_itr++)
      {
      }
      */
    }
    return 0;
  } 

  /* Special case for cols == 32 */
  if(
      (cols1 == 32) &&
      (row_stride1 == 32) &&
      (vec_offset == 32) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
      ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
      ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
      ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);

      /* Load 4 rows */
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      
      ae_int32x2 d_bias0 = AE_ZERO32(), d_bias1 = AE_ZERO32();
      if(p_bias)
      {
        d_bias0 = AE_MOVDA32X2(p_bias[m_itr + 1], p_bias[m_itr + 0]);
        d_bias1 = AE_MOVDA32X2(p_bias[m_itr + 3], p_bias[m_itr + 2]);
      }

#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32, acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 , vec_z_b);
      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 , vec_z_b);
      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 , vec_z_b);

      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);

      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);
#else
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
      AE_S32X2X2_I(d_bias0, d_bias1, (ae_int32x4*)acc_buffer, 0);
#endif

#if TFLITE_SINGLE_ROUNDING
      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_32, l_mult_10, r_mult_32, r_mult_10;

      p_left_mult[3] = p_out_shift[m_itr + 0];
      p_left_mult[2] = p_out_shift[m_itr + 1];
      p_left_mult[1] = p_out_shift[m_itr + 2];
      p_left_mult[0] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_32, l_mult_10, (ae_int32x4 *)ptr_left_mult, 0);
      (void)r_mult_32;
      (void)r_mult_10;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_mult[4], p_right_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x2 l_mult_32, l_mult_10, r_mult_32, r_mult_10;

      p_left_mult[3] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[2] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[1] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[0] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[3] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[2] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[1] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[0] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_32, out_multiplier_10;
      p_out_mult[3] = p_out_multiplier[m_itr + 0];
      p_out_mult[2] = p_out_multiplier[m_itr + 1];
      p_out_mult[1] = p_out_multiplier[m_itr + 2];
      p_out_mult[0] = p_out_multiplier[m_itr + 3];

      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1);

      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
      {
        ae_int32x2 acc_row0_vec0; 
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1; 
        ae_int32x2 acc_row0_vec2; 
        ae_int32x2 acc_row1_vec2; 
        ae_int32x2 acc_row0_vec3; 
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);

        ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
        ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3; 
        ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3; 
        ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3; 

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec0_batch_2, vec0_batch_3, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_batch_0, vec1_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_batch_2, vec1_batch_3, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_2, vec2_batch_3, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec3_batch_0, vec3_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec3_batch_2, vec3_batch_3, (ae_int8x16*)p_vec_0, 16);

        MAT_VEC_MAC(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 , vec0_batch_0);
        MAT_VEC_MAC(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 , vec1_batch_0);
        MAT_VEC_MAC(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 , vec2_batch_0);
        MAT_VEC_MAC(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 , vec3_batch_0);

        MAT_VEC_MAC(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 , vec0_batch_1);
        MAT_VEC_MAC(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 , vec1_batch_1);
        MAT_VEC_MAC(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 , vec2_batch_1);
        MAT_VEC_MAC(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 , vec3_batch_1);

        MAT_VEC_MAC(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 , vec0_batch_2);
        MAT_VEC_MAC(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 , vec1_batch_2);
        MAT_VEC_MAC(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 , vec2_batch_2);
        MAT_VEC_MAC(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 , vec3_batch_2);

        MAT_VEC_MAC(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 , vec0_batch_3);
        MAT_VEC_MAC(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 , vec1_batch_3);
        MAT_VEC_MAC(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 , vec2_batch_3);
        MAT_VEC_MAC(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 , vec3_batch_3);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;
#if !TFLITE_SINGLE_ROUNDING
        AE_L32X2X2_IP(l_mult_32, l_mult_10, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_32, r_mult_10, (ae_int32x4 *)ptr_right_mult, 0);
#endif
        AE_L32X2X2_IP(out_multiplier_32, out_multiplier_10, (ae_int32x4 *)ptr_out_mult, 0);

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_0, out_1, acc_row1_vec0, acc_row0_vec0, acc_row1_vec1, acc_row0_vec1, out_multiplier_32, out_multiplier_10, l_mult_32, l_mult_10, r_mult_32, r_mult_10, out_zero_bias);
        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out_2, out_3, acc_row1_vec2, acc_row0_vec2, acc_row1_vec3, acc_row0_vec3, out_multiplier_32, out_multiplier_10, l_mult_32, l_mult_10, r_mult_32, r_mult_10, out_zero_bias);

        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        out32_0 = AE_SAT8X8X16(out_0, out_1);
        out32_1 = AE_SAT8X8X16(out_2, out_3);

        out32_0 = AE_MAX8(out32_0, AE_MOVDA8(out_activation_min));
        out32_0 = AE_MIN8(out32_0, AE_MOVDA8(out_activation_max));
        out32_1 = AE_MAX8(out32_1, AE_MOVDA8(out_activation_min));
        out32_1 = AE_MIN8(out32_1, AE_MOVDA8(out_activation_max));

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);

      }
      /*
         for (; vec_itr < vec_count; vec_itr++)
         {
         }
       */
    }
    return 0;
  } 

  /* Special case for cols == 64, 128, 256 */
  if(((cols1 & 0x1f) == 0) &&
     (cols1 <= 256) &&
     ((row_stride1 & 0x1f) == 0) &&
     ((vec_offset & 0x1f) == 0) &&
     ALIGNED_PTR(p_vec1, 16) &&
     ALIGNED_PTR(p_out, 4) &&
     (out_stride == 1) &&   // NHWC case
     ((out_offset & 0x3) == 0) &&   // NHWC case
     ((rows & 0x3) == 0)
    )
  {
    special_function_for_cols_mul_32
      (p_out,
       p_mat1,
       p_vec1,
       p_bias,
       rows,
       vec_count,
       cols1,
       p_out_multiplier,
       p_out_shift,
       vec1_zero_bias,
       out_zero_bias,
       out_stride,
       row_stride1,
       vec_offset,
       out_offset,
       out_activation_min,
       out_activation_max
      );

    return 0;
  }

  if(
      ALIGNED_PTR(p_mat1, 16) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ((row_stride1 & 15) == 0) &&
      ((vec_offset & 15) == 0)
      )
  {
#ifndef AE_MULAZB8Q8X8
    ae_int8x8 mat1_row0_0, mat1_row0_1;
    ae_int8x8 mat1_row1_0, mat1_row1_1;
    ae_int8x8 mat1_row2_0, mat1_row2_1;
    ae_int8x8 mat1_row3_0, mat1_row3_1;

    int rem_cols = cols1 & 15;
    int rem_cols_shift_0 = ((rem_cols) <= 8)?(8 - (rem_cols)) * 8:0;
    int rem_cols_shift_1 = ((rem_cols) > 8)?(16 - (rem_cols)) * 8:64;
#endif

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int32x2 acc_row0 = ZERO32, acc_row1 = ZERO32;
      
#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
      ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
      ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1);
      
      int c_itr = 0;
      int cols_count = cols1 - rem_cols;
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
        AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
        AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
        AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      }

      c_itr <<= 4;
      //Remainder loop for cols1
      while(c_itr < cols1)
      {
        AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);
        AE_L8X8_IP(mat1_row1_0, (ae_int8x8 *)p_mat1_1, 8);
        AE_L8X8_IP(mat1_row2_0, (ae_int8x8 *)p_mat1_2, 8);
        AE_L8X8_IP(mat1_row3_0, (ae_int8x8 *)p_mat1_3, 8);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);

        c_itr += 8;
        rem_cols_shift_0 = rem_cols_shift_1;
      }
#else
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, vec_z_b)));
#endif

      ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
      if(p_bias)
      {
        bias_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        bias_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }
      acc_row0 = AE_SUB32S(bias_01, acc_row0);
      acc_row1 = AE_SUB32S(bias_23, acc_row1);

      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);

      WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;
      WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + 1) * out_stride;
      WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + 2) * out_stride;
      WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + 3) * out_stride;

#if TFLITE_SINGLE_ROUNDING
      p_left_shift[0] = p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
      ae_int32x2 l_mult_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

      ae_int32x2 r_mult_23; 
      ae_int32x2 r_mult_01;
      (void)r_mult_23;
      (void)r_mult_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1] < 0 ? 0 : p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2] < 0 ? 0 : p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3] < 0 ? 0 : p_out_shift[m_itr + 3];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];
      p_right_shift[1] = p_out_shift[m_itr + 1] > 0 ? 0 : -p_out_shift[m_itr + 1];
      p_right_shift[2] = p_out_shift[m_itr + 2] > 0 ? 0 : -p_out_shift[m_itr + 2];
      p_right_shift[3] = p_out_shift[m_itr + 3] > 0 ? 0 : -p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
      ae_int32x2 l_mult_01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

      ae_int32x2 r_mult_23 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[2])), (0xFFFFFFFF << (31 - p_right_shift[3]))); 
      ae_int32x2 r_mult_01 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[0])), (0xFFFFFFFF << (31 - p_right_shift[1])));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(p_out_multiplier[m_itr + 2], p_out_multiplier[m_itr + 3]); 
      ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(p_out_multiplier[m_itr + 0], p_out_multiplier[m_itr + 1]);

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1;
        ae_int32x2 acc_row0_vec2;
        ae_int32x2 acc_row1_vec2;
        ae_int32x2 acc_row0_vec3;
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row1_vec0, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row0_vec3, (ae_int32x4*)acc_buffer, 16);
        AE_L32X2X2_I(acc_row1_vec2, acc_row1_vec3, (ae_int32x4*)acc_buffer, 16);

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1]; 

        _xa_nn_dot_product_4_rows_4_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,&acc_row1_vec0
           ,&acc_row1_vec1
           ,&acc_row1_vec2
           ,&acc_row1_vec3
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
           ,vec1_zero_bias
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[m_itr + 0], p_left_shift[0], p_right_shift[0], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[m_itr + 1], p_left_shift[1], p_right_shift[1], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[m_itr + 2], p_left_shift[2], p_right_shift[2], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[m_itr + 3], p_left_shift[3], p_right_shift[3], out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
      }

      // Remaining vectors
#ifdef AE_MULAZB8Q8X8
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
#endif
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,0
           ,vec1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_offset);
      }
    }

    // remaining rows
    for(; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0 = ZERO32;

#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row1 = ZERO32;
      ae_int8x8 mat1_row0_0;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

      int cols_count=cols1-(cols1&7);
      int rem_cols_shift = 64 - (cols1 & 7) * 8;

      int c_itr;
      #pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }
#else
      int mat_z_b = -vec1_zero_bias;
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, mat_z_b)));
#endif
    
      ae_int32x2 bias_0 = AE_ZERO32();
      if(p_bias)
      {
        bias_0 = AE_MOVDA32(p_bias[m_itr + 0]);
      }
      acc_row0 = AE_SUB32S(bias_0, acc_row0);

      AE_S32X2X2_I(acc_row0, acc_row0, (ae_int32x4*)acc_buffer, 0);

      WORD8* p_dst = (WORD8*)p_out + (m_itr + 0) * out_stride;

#if TFLITE_SINGLE_ROUNDING
      p_left_shift[0] = p_out_shift[m_itr + 0];
#else /* #if TFLITE_SINGLE_ROUNDING */
      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8x8* p_vec_0  = (ae_int8x8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8 *p_mat1_0 = (ae_int8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_vec_0
           ,p_mat1_0
           ,cols1
           ,vec_offset
           ,vec1_zero_bias
           ,0
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row0_vec1, p_out_multiplier[m_itr], p_left_shift[0], p_right_shift[0], out_zero_bias);
        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
      }

      // Remaining vectors
#ifdef AE_MULAZB8Q8X8
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
#endif
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );

        ae_int16x4 out_0;
        ae_int8x8 temp_vec0;
        MPY_BY_QUANT_MULT_X2_OUT16(out_0, acc_row0_vec0, p_out_multiplier[m_itr], p_left_shift[0], p_right_shift[0]);
        out_0 = AE_ADD16S(out_0, AE_MOVDA16(out_zero_bias));
        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
      }
    }
  }
  else if (p_mat1 && p_vec1)
  {
#ifndef AE_MULAZB8Q8X8
    ae_int8x8 mat1_row0_0, mat1_row0_1;
    ae_int8x8 mat1_row1_0, mat1_row1_1;
    ae_int8x8 mat1_row2_0, mat1_row2_1;
    ae_int8x8 mat1_row3_0, mat1_row3_1;

    int rem_cols = cols1 & 15;
    int rem_cols_shift_0 = ((rem_cols) <= 8)?(8 - (rem_cols)) * 8:0;
    int rem_cols_shift_1 = ((rem_cols) > 8)?(16 - (rem_cols)) * 8:64;
#endif

    for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
    {
      for(ii = 0; ii < 8; ii++)
      {
        ae_int32x2 acc_row0 = ZERO32; 
        ae_int32x2 acc_row1 = ZERO32;

#ifndef AE_MULAZB8Q8X8
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1]; 
        ae_int8x8* p_mat1_1 = p_mat1_0 + row_stride1; //next 8th row 
        ae_int8x8* p_mat1_2 = p_mat1_1 + row_stride1; //next 8th row
        ae_int8x8* p_mat1_3 = p_mat1_2 + row_stride1; //next 8th row
        
        ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
        ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
        ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
        ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);
        
        int c_itr=0;
        int cols_count=cols1 - rem_cols;
        for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
        {
          AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
          AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
          AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
          AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);
          AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
        }

        //Remainder loop for cols1
        if(cols_count!=cols1)
        {
          AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
          AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
          AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
          AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

          mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
          mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
          mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
          mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

          AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);

          if(rem_cols > 8)
          {
            mat1_row0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_1), rem_cols_shift_1), rem_cols_shift_1));
            mat1_row1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_1), rem_cols_shift_1), rem_cols_shift_1));
            mat1_row2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_1), rem_cols_shift_1), rem_cols_shift_1));
            mat1_row3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_1), rem_cols_shift_1), rem_cols_shift_1));

            AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
          }
        }
#else
        AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, vec_z_b)));
#endif

        ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
        if(p_bias)
        {
          bias_01 = AE_MOVDA32X2(p_bias[m_itr + ii +  0], p_bias[m_itr + ii +  8]);
          bias_23 = AE_MOVDA32X2(p_bias[m_itr + ii + 16], p_bias[m_itr + ii + 24]);
        }
        acc_row0 = AE_SUB32S(bias_01, acc_row0);
        acc_row1 = AE_SUB32S(bias_23, acc_row1);

        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);

        WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + ii +  0) * out_stride;
        WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + ii +  8) * out_stride;
        WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + ii + 16) * out_stride;
        WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + ii + 24) * out_stride;

#if TFLITE_SINGLE_ROUNDING
        p_left_shift[0] = p_out_shift[m_itr + ii +  0];
        p_left_shift[1] = p_out_shift[m_itr + ii +  8];
        p_left_shift[2] = p_out_shift[m_itr + ii + 16];
        p_left_shift[3] = p_out_shift[m_itr + ii + 24];

        ae_int32x2 l_mult_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
        ae_int32x2 l_mult_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

        ae_int32x2 r_mult_23; 
        ae_int32x2 r_mult_01;
        (void)r_mult_23;
        (void)r_mult_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
        p_left_shift[0] = p_out_shift[m_itr + ii +  0] < 0 ? 0 : p_out_shift[m_itr + ii +  0];
        p_left_shift[1] = p_out_shift[m_itr + ii +  8] < 0 ? 0 : p_out_shift[m_itr + ii +  8];
        p_left_shift[2] = p_out_shift[m_itr + ii + 16] < 0 ? 0 : p_out_shift[m_itr + ii + 16];
        p_left_shift[3] = p_out_shift[m_itr + ii + 24] < 0 ? 0 : p_out_shift[m_itr + ii + 24];

        p_right_shift[0] = p_out_shift[m_itr + ii +  0] > 0 ? 0 : -p_out_shift[m_itr + ii +  0];
        p_right_shift[1] = p_out_shift[m_itr + ii +  8] > 0 ? 0 : -p_out_shift[m_itr + ii +  8];
        p_right_shift[2] = p_out_shift[m_itr + ii + 16] > 0 ? 0 : -p_out_shift[m_itr + ii + 16];
        p_right_shift[3] = p_out_shift[m_itr + ii + 24] > 0 ? 0 : -p_out_shift[m_itr + ii + 24];

        ae_int32x2 l_mult_23 = AE_MOVDA32X2(1 << p_left_shift[2], 1 << p_left_shift[3]); 
        ae_int32x2 l_mult_01 = AE_MOVDA32X2(1 << p_left_shift[0], 1 << p_left_shift[1]);

        ae_int32x2 r_mult_23 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[2])), (0xFFFFFFFF << (31 - p_right_shift[3]))); 
        ae_int32x2 r_mult_01 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[0])), (0xFFFFFFFF << (31 - p_right_shift[1])));
#endif /* #if TFLITE_SINGLE_ROUNDING */

        ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(p_out_multiplier[m_itr + ii + 16], p_out_multiplier[m_itr + ii + 24]); 
        ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(p_out_multiplier[m_itr + ii + 0],  p_out_multiplier[m_itr + ii + 8]);

        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
          ae_int32x2 acc_row0_vec0;
          ae_int32x2 acc_row1_vec0;
          ae_int32x2 acc_row0_vec1;
          ae_int32x2 acc_row1_vec1;
          ae_int32x2 acc_row0_vec2;
          ae_int32x2 acc_row1_vec2;
          ae_int32x2 acc_row0_vec3;
          ae_int32x2 acc_row1_vec3;

          /* Initialize accumulators */
          AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);
          AE_L32X2X2_I(acc_row1_vec0, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
          AE_L32X2X2_I(acc_row0_vec2, acc_row0_vec3, (ae_int32x4*)acc_buffer, 16);
          AE_L32X2X2_I(acc_row1_vec2, acc_row1_vec3, (ae_int32x4*)acc_buffer, 16);

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1]; 

          _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row0_vec1
             ,&acc_row0_vec2
             ,&acc_row0_vec3
             ,&acc_row1_vec0
             ,&acc_row1_vec1
             ,&acc_row1_vec2
             ,&acc_row1_vec3
             ,p_mat1_0
             ,p_vec_0
             ,cols1
             ,row_stride1
             ,vec_offset
             ,vec1_zero_bias
            );

          ae_int16x4 out_0, out_1, out_2, out_3;

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[m_itr + ii +  0], p_left_shift[0], p_right_shift[0], out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[m_itr + ii +  8], p_left_shift[1], p_right_shift[1], out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[m_itr + ii + 16], p_left_shift[2], p_right_shift[2], out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[m_itr + ii + 24], p_left_shift[3], p_right_shift[3], out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
          AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
          AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
          AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);

          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        }

        // Remaining vectors
#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
#endif
        for (; vec_itr < vec_count; vec_itr++)
        {
          ae_int32x2 acc_row0_vec0 = acc_row0;
          ae_int32x2 acc_row1_vec0 = acc_row1;

          WORD8* p_dst = (WORD8*)p_out + vec_itr * out_offset + (m_itr + ii) * out_stride;
          ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8* p_mat1_0 = (ae_int8x8*) &p_mat1[(m_itr + ii + 0)* row_stride1]; 

          _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,p_mat1_0
             ,p_vec_0
             ,cols1
             ,row_stride1
             ,vec1_zero_bias
            );

          ae_int16x4 out_0;

          MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

          AE_SW_S8_6_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst,  0 * out_stride);
          AE_SW_S8_4_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst,  8 * out_stride);
          AE_SW_S8_2_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, 16 * out_stride);
          AE_S8_0_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, 24 * out_stride);
        }
      }
    }

    // Process loop for 4 rows and 4 vectors 
    for(; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
      ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
      ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1); 
      
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      int c_itr=0;
      int cols_count=cols1 - rem_cols;
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

        AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);

        if(rem_cols > 8)
        {
          mat1_row0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_1), rem_cols_shift_1), rem_cols_shift_1));
          mat1_row1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_1), rem_cols_shift_1), rem_cols_shift_1));
          mat1_row2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_1), rem_cols_shift_1), rem_cols_shift_1));
          mat1_row3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_1), rem_cols_shift_1), rem_cols_shift_1));

          AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
        }
      }
#else
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, vec_z_b)));
#endif

      ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
      if(p_bias)
      {
        bias_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
        bias_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }
      acc_row0 = AE_SUB32S(bias_01, acc_row0);
      acc_row1 = AE_SUB32S(bias_23, acc_row1);

      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);


      WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;
      WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + 1) * out_stride;
      WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + 2) * out_stride;
      WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + 3) * out_stride;

#if TFLITE_SINGLE_ROUNDING
      p_left_shift[0] = p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
      ae_int32x2 l_mult_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

      ae_int32x2 r_mult_23; 
      ae_int32x2 r_mult_01;
      (void)r_mult_23;
      (void)r_mult_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1] < 0 ? 0 : p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2] < 0 ? 0 : p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3] < 0 ? 0 : p_out_shift[m_itr + 3];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];
      p_right_shift[1] = p_out_shift[m_itr + 1] > 0 ? 0 : -p_out_shift[m_itr + 1];
      p_right_shift[2] = p_out_shift[m_itr + 2] > 0 ? 0 : -p_out_shift[m_itr + 2];
      p_right_shift[3] = p_out_shift[m_itr + 3] > 0 ? 0 : -p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
      ae_int32x2 l_mult_01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

      ae_int32x2 r_mult_23 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[2])), (0xFFFFFFFF << (31 - p_right_shift[3]))); 
      ae_int32x2 r_mult_01 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[0])), (0xFFFFFFFF << (31 - p_right_shift[1])));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(p_out_multiplier[m_itr + 2], p_out_multiplier[m_itr + 3]); 
      ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(p_out_multiplier[m_itr + 0], p_out_multiplier[m_itr + 1]);

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1;
        ae_int32x2 acc_row0_vec2;
        ae_int32x2 acc_row1_vec2;
        ae_int32x2 acc_row0_vec3;
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row1_vec0, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row0_vec3, (ae_int32x4*)acc_buffer, 16);
        AE_L32X2X2_I(acc_row1_vec2, acc_row1_vec3, (ae_int32x4*)acc_buffer, 16);

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1]; 

        _xa_nn_dot_product_4_rows_4_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,&acc_row1_vec0
           ,&acc_row1_vec1
           ,&acc_row1_vec2
           ,&acc_row1_vec3
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
           ,vec1_zero_bias
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[m_itr + 0], p_left_shift[0], p_right_shift[0], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[m_itr + 1], p_left_shift[1], p_right_shift[1], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[m_itr + 2], p_left_shift[2], p_right_shift[2], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[m_itr + 3], p_left_shift[3], p_right_shift[3], out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_MINMAX16(out_3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
      }

      // Remaining vectors
#ifdef AE_MULAZB8Q8X8
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
#endif
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,0
           ,vec1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_01, out_multiplier_23, l_mult_01, l_mult_23, r_mult_01, r_mult_23, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_offset);
      }
    }

    // remaining rows
    for(; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0 = ZERO32; 
      
#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);

      ae_int32x2 acc_row1 = ZERO32;
      
      int rem_cols_shift = 64 - (cols1 & 7) * 8;

      int c_itr=0;
      int cols_count=cols1-(cols1&7);
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }
#else
      ae_int32  mat_z_b = -vec1_zero_bias;
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, mat_z_b)));
#endif

      ae_int32x2 bias_0 = AE_ZERO32();
      if(p_bias)
      {
        bias_0 = AE_MOVDA32(p_bias[m_itr + 0]);
      }
      acc_row0 = AE_SUB32S(bias_0, acc_row0);
      AE_S32X2X2_I(acc_row0, acc_row0, (ae_int32x4*)acc_buffer, 0);

      WORD8* p_dst = (WORD8*)p_out + (m_itr + 0) * out_stride;

#if TFLITE_SINGLE_ROUNDING
      p_left_shift[0] = p_out_shift[m_itr + 0];
#else /* #if TFLITE_SINGLE_ROUNDING */
      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8x8* p_vec_0  = (ae_int8x8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8 *p_mat1_0 = (ae_int8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_vec_0
           ,p_mat1_0
           ,cols1
           ,vec_offset
           ,vec1_zero_bias
           ,0
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row0_vec1, p_out_multiplier[m_itr], p_left_shift[0], p_right_shift[0], out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
      }

      // Remaining vectors
#ifdef AE_MULAZB8Q8X8
      AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(vec_z_b, 0)));
#endif
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,vec1_zero_bias
          );

        ae_int16x4 out_0;
        ae_int8x8 temp_vec0;
        MPY_BY_QUANT_MULT_X2_OUT16(out_0, acc_row0_vec0, p_out_multiplier[m_itr], p_left_shift[0], p_right_shift[0]);
        out_0 = AE_ADD16S(out_0, AE_MOVDA16(out_zero_bias));
        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
      }
    }
  }
  else
  {
    return -1;
  }
    return 0;
}

WORD32 xa_nn_matmul_per_chan_sym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 vec1_zero_bias,
    const WORD32* __restrict__ p_out_multiplier,
    const WORD32* __restrict__ p_out_shift,
    WORD32 out_zero_bias)
{
  return xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(
              p_out,
              p_mat1,
              p_vec1,
              p_bias,
              rows,
              cols1,
              row_stride1,
              vec_count,
              vec_offset,
              out_stride,
              out_offset,
              vec1_zero_bias,
              p_out_multiplier,
              p_out_shift,
              out_zero_bias,
              -128,
              127,
              NULL);
}
