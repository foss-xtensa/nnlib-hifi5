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
#include "xa_nnlib_common_macros_hifi5.h"

#define ALIGNED_ADDR( addr, align ) \
  (void*)( ( (UWORD32)(addr) + ( (align) - 1 ) ) & ~( (align) - 1 ) )

#define AE_MOVINT4X16_FROMINT8X8_4R(out0, out1, out2, out3, out4, out5, out6, out7, in0, in1, in2, in3, in4, in5, in6, in7) { \
      out0 = AE_MOVINT4X16_FROMINT8X8(in0);\
      out1 = AE_MOVINT4X16_FROMINT8X8(in1);\
      out2 = AE_MOVINT4X16_FROMINT8X8(in2);\
      out3 = AE_MOVINT4X16_FROMINT8X8(in3);\
      out4 = AE_MOVINT4X16_FROMINT8X8(in4);\
      out5 = AE_MOVINT4X16_FROMINT8X8(in5);\
      out6 = AE_MOVINT4X16_FROMINT8X8(in6);\
      out7 = AE_MOVINT4X16_FROMINT8X8(in7);\
}

#define AE_MOVINT4X16_FROMINT8X8_2R(out0, out1, out2, out3, in0, in1, in2, in3) { \
      out0 = AE_MOVINT4X16_FROMINT8X8(in0);\
      out1 = AE_MOVINT4X16_FROMINT8X8(in1);\
      out2 = AE_MOVINT4X16_FROMINT8X8(in2);\
      out3 = AE_MOVINT4X16_FROMINT8X8(in3);\
}

#define AE_MOVINT4X16_FROMINT8X8_1R(out0, out1, in0, in1) { \
      out0 = AE_MOVINT4X16_FROMINT8X8(in0);\
      out1 = AE_MOVINT4X16_FROMINT8X8(in1);\
}

static WORD32 calculate_zero_point_x_vector(WORD32 vec_zero_bias, WORD32 mat_zero_bias, WORD8 *p_vec, WORD32 vec_len)
{
  WORD8 *vec_ptr = (WORD8 *)p_vec;
  ae_int64 acc = 0;
  if(mat_zero_bias != 0)
  {
    ae_int64 acc0 = 0, acc1 = 0;
    ae_int8x8 vec_zb = AE_MOVDA8(-vec_zero_bias);
    ae_int8x8 vec_zb0 = AE_MOVDA8(-vec_zero_bias);
    ae_int8x8 vec_zb1 = AE_MOVDA8(-vec_zero_bias);
    int rem_elems = (vec_len & 15);
    int rem_shift0, rem_shift1;
    rem_shift0 =  rem_elems >= 8 ? 0 : 64 - ((vec_len & 15) * 8);
    rem_shift1 =  rem_elems >= 8 ? 64 - (((vec_len & 15)-8) * 8) : 64;
    vec_zb0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec_zb0), rem_shift0), rem_shift0));
    vec_zb1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec_zb1), rem_shift1), rem_shift1));
    for(int i=0; i< vec_len >> 4; i++){
      ae_int8x8 vec0, vec1;
      ae_int16x4 vec_val0, vec_val1, vec_val2, vec_val3;
      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)vec_ptr, (16 * sizeof(WORD8)));
      AE_SUBW8(vec_val0, vec_val1, vec0, vec_zb);
      AE_SUBW8(vec_val2, vec_val3, vec1, vec_zb);
      AE_MULAAAA2Q16(acc0, acc1, vec_val0, vec_val1, AE_MOVDA16(mat_zero_bias), AE_MOVDA16(mat_zero_bias));
      AE_MULAAAA2Q16(acc0, acc1, vec_val2, vec_val3, AE_MOVDA16(mat_zero_bias), AE_MOVDA16(mat_zero_bias));
    }
    ae_valignx2 vec_align = AE_LA128_PP(vec_ptr);
    if(vec_len & 15)
    {
      ae_int8x8 vec0, vec1;
      ae_int16x4 vec_val0, vec_val1, vec_val2, vec_val3;
      AE_LAV8X8X2_XP(vec0, vec1, vec_align, (ae_int8x16 *)vec_ptr, (vec_len & 15));
      AE_SUBW8(vec_val0, vec_val1, vec0, vec_zb0);
      AE_SUBW8(vec_val2, vec_val3, vec1, vec_zb1);
      AE_MULAAAA2Q16(acc0, acc1, vec_val0, vec_val1, AE_MOVDA16(mat_zero_bias), AE_MOVDA16(mat_zero_bias));
      AE_MULAAAA2Q16(acc0, acc1, vec_val2, vec_val3, AE_MOVDA16(mat_zero_bias), AE_MOVDA16(mat_zero_bias));
    }
    acc = AE_ADD64S(acc0, acc1);
  }
  return AE_MOVINT32_FROMINT64(acc);
}

WORD32 xa_nn_matXvec_asym4sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    const WORD8 * __restrict__ p_vec1,
    const WORD8 * __restrict__ p_vec2,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 vec1_zero_bias,
    WORD32 vec2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias, 
    pVOID p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND(((row_stride1 % 2) != 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  if(p_mat2 != NULL || p_vec2 != NULL)
  {
    return -1;
  }

  WORD8 *p_vec_flipped = (WORD8 *)p_scratch;
  p_vec_flipped = ALIGNED_ADDR(p_vec_flipped, 16);

  WORD8 *p_vec_flip_process = (WORD8 *)p_vec_flipped;
  ae_int8x16 *p_vec_in = (ae_int8x16 *)p_vec1;

  ae_int8x8 vec0, vec1;
  ae_valignx2 vec_align, vec_flipped_align;
  vec_align = AE_LA128_PP(p_vec_in);
  vec_flipped_align = AE_ZALIGN128();

  /* Below code is for inverting order of vector by adding vector_zero_bias to the vector values. */
  for(int vec_itr=0; vec_itr < cols1 >> 4; vec_itr++)
  {
    AE_LA8X8X2_IP(vec0, vec1, vec_align, p_vec_in);
    AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT64(0xE6F7C4D5A2B38091));
    AE_SA8X8X2_IP(vec0, vec1, vec_flipped_align, (ae_int8x16 *)p_vec_flip_process);
  }
  if(cols1 & 15)
  {
    AE_LAV8X8X2_XP(vec0, vec1, vec_align, p_vec_in, (cols1 & 15));
    AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT64(0xE6F7C4D5A2B38091));
    AE_SA8X8X2_IP(vec0, vec1, vec_flipped_align, (ae_int8x16 *)p_vec_flip_process);
  }
  AE_SA128POS_FP(vec_flipped_align, p_vec_flip_process);

  WORD32 mat_zb_x_vec = calculate_zero_point_x_vector(vec1_zero_bias, mat1_zero_bias, p_vec_flipped, cols1);
  int m_itr = 0, c_itr = 0;
  int left_shift;
  int right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif

  WORD8 *out_ptr = p_out;

  for(; m_itr < (rows & ~(4-1)); m_itr += 4)
  {
    ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
    ae_valignx2 mat_align0, mat_align1, mat_align2, mat_align3;
    ae_int32x2 acc0 = AE_MOVDA32(0);
    ae_int32x2 acc1 = AE_MOVDA32(0);    
    if(p_bias != NULL)
    {
      acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr+1]);
      acc1 = AE_MOVDA32X2(p_bias[m_itr+2], p_bias[m_itr+3]);
    }
    ae_int8x16 *p_mat_0 = (ae_int8x16 *)(&p_mat1[(m_itr)*(row_stride1/2)*sizeof(WORD8)]);
    ae_int8x16 *p_mat_1 = (ae_int8x16 *)(&p_mat1[(m_itr+1)*(row_stride1/2)*sizeof(WORD8)]);
    ae_int8x16 *p_mat_2 = (ae_int8x16 *)(&p_mat1[(m_itr+2)*(row_stride1/2)*sizeof(WORD8)]);
    ae_int8x16 *p_mat_3 = (ae_int8x16 *)(&p_mat1[(m_itr+3)*(row_stride1/2)*sizeof(WORD8)]);

    mat_align0 = AE_LA128_PP(p_mat_0);
    mat_align1 = AE_LA128_PP(p_mat_1);
    mat_align2 = AE_LA128_PP(p_mat_2);
    mat_align3 = AE_LA128_PP(p_mat_3);

    WORD8 *p_vec_batch_0  = (WORD8 *)p_vec_flipped;

    ae_int8x8 mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1;
    ae_int4x16 mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b;
    ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat1_0_4b_interleaved, mat1_1_4b_interleaved, mat1_2_4b_interleaved, mat1_3_4b_interleaved;
    ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved, mat1_0_8b_interleaved, mat1_1_8b_interleaved, mat1_2_8b_interleaved, mat1_3_8b_interleaved;
    ae_int8x8 vec0, vec1, vec2, vec3;
    ae_int16x4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;
    ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT64(0xFBEAD9C873625140);
    int rem_cols_shift_0, rem_cols_shift_1;
    
    rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
    rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

    for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
      AE_LA8X8X2_IP(mat1_0, mat1_1, mat_align1, p_mat_1);
      AE_LA8X8X2_IP(mat2_0, mat2_1, mat_align2, p_mat_2);
      AE_LA8X8X2_IP(mat3_0, mat3_1, mat_align3, p_mat_3);
      AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b, mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1);

      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec2, vec3, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));

      AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
      AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
      AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
      AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b, mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat1_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat1_1_4b), dsel_hh_ll);
      AE_DSEL8X8(mat1_0_8b_interleaved, mat1_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat2_0_4b), AE_MOVINT8X8_FROMINT4X16(mat3_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat1_2_8b_interleaved, mat1_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat2_1_4b), AE_MOVINT8X8_FROMINT4X16(mat3_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat1_0_4b_interleaved, mat1_1_4b_interleaved, mat1_2_4b_interleaved, mat1_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved, mat1_0_8b_interleaved, mat1_1_8b_interleaved, mat1_2_8b_interleaved, mat1_3_8b_interleaved);
      AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec0_0, vec0_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec1_0, vec1_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec2_0, vec2_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec3_0, vec3_1);
    }
    if(cols1 & 31)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
      mat0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_0), rem_cols_shift_0), rem_cols_shift_0));
      mat0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_1), rem_cols_shift_1), rem_cols_shift_1));        
      AE_LA8X8X2_IP(mat1_0, mat1_1, mat_align1, p_mat_1);
      mat1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_0), rem_cols_shift_0), rem_cols_shift_0));
      mat1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_1), rem_cols_shift_1), rem_cols_shift_1));        
      AE_LA8X8X2_IP(mat2_0, mat2_1, mat_align2, p_mat_2);
      mat2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat2_0), rem_cols_shift_0), rem_cols_shift_0));
      mat2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat2_1), rem_cols_shift_1), rem_cols_shift_1));        
      AE_LA8X8X2_IP(mat3_0, mat3_1, mat_align3, p_mat_3);
      mat3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat3_0), rem_cols_shift_0), rem_cols_shift_0));
      mat3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat3_1), rem_cols_shift_1), rem_cols_shift_1));  

      AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b, mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat1_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat1_1_4b), dsel_hh_ll);
      AE_DSEL8X8(mat1_0_8b_interleaved, mat1_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat2_0_4b), AE_MOVINT8X8_FROMINT4X16(mat3_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat1_2_8b_interleaved, mat1_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat2_1_4b), AE_MOVINT8X8_FROMINT4X16(mat3_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat1_0_4b_interleaved, mat1_1_4b_interleaved, mat1_2_4b_interleaved, mat1_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved, mat1_0_8b_interleaved, mat1_1_8b_interleaved, mat1_2_8b_interleaved, mat1_3_8b_interleaved);
      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec2, vec3, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));

      AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
      AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
      AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
      AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);

      AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec0_0, vec0_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec1_0, vec1_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec2_0, vec2_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec3_0, vec3_1);
    }
    acc0 = AE_ADD32S(acc0, AE_MOVDA32(mat_zb_x_vec));
    acc1 = AE_ADD32S(acc1, AE_MOVDA32(mat_zb_x_vec));
    ae_int16x4 out;
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, acc1, out_multiplier, left_shift, right_shift, out_zero_bias)
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
    *out_ptr++ = (WORD8)AE_MOVAD16_3(out);
    *out_ptr++ = (WORD8)AE_MOVAD16_2(out);
    *out_ptr++ = (WORD8)AE_MOVAD16_1(out);
    *out_ptr++ = (WORD8)AE_MOVAD16_0(out);
  }

  for(; m_itr < (rows & ~(2-1)); m_itr += 2)
  {
    ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
    ae_valignx2 mat_align0, mat_align1;
    ae_int32x2 acc0 = AE_MOVDA32(0);
    ae_int32x2 dummy = AE_MOVDA32(0);
    if(p_bias != NULL)
    {
      acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr+1]);
    }
    ae_int32x2 acc1;
    ae_int8x16 *p_mat_0 = (ae_int8x16 *)(&p_mat1[(m_itr)*(row_stride1/2)*sizeof(WORD8)]);
    ae_int8x16 *p_mat_1 = (ae_int8x16 *)(&p_mat1[(m_itr+1)*(row_stride1/2)*sizeof(WORD8)]);

    mat_align0 = AE_LA128_PP(p_mat_0);
    mat_align1 = AE_LA128_PP(p_mat_1);

    WORD8 *p_vec_batch_0  = (WORD8 *)p_vec_flipped;

    ae_int8x8 mat0_0, mat0_1, mat1_0, mat1_1;
    ae_int4x16 mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b;
    ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved;
    ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved;
    ae_int8x8 vec0, vec1, vec2, vec3;
    ae_int16x4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;
    ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT64(0xFBEAD9C873625140);
    int rem_cols_shift_0, rem_cols_shift_1;
    
    rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
    rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

    for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
      AE_LA8X8X2_IP(mat1_0, mat1_1, mat_align1, p_mat_1);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat0_0, mat0_1, mat1_0, mat1_1);

      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec2, vec3, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));

      AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
      AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
      AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
      AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat0_0, mat0_1, mat1_0, mat1_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat1_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat1_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved);
      AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec0_0, vec0_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec1_0, vec1_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec2_0, vec2_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec3_0, vec3_1);
    }
    if(cols1 & 31)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
      mat0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_0), rem_cols_shift_0), rem_cols_shift_0));
      mat0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_1), rem_cols_shift_1), rem_cols_shift_1));        
      AE_LA8X8X2_IP(mat1_0, mat1_1, mat_align1, p_mat_1);
      mat1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_0), rem_cols_shift_0), rem_cols_shift_0));
      mat1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_1), rem_cols_shift_1), rem_cols_shift_1));          

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat0_0, mat0_1, mat1_0, mat1_1);

      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec2, vec3, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));

      AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
      AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
      AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
      AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat0_0, mat0_1, mat1_0, mat1_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat1_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat1_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved);
      AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec0_0, vec0_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec1_0, vec1_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec2_0, vec2_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec3_0, vec3_1);
    }
    acc0 = AE_ADD32S(acc0, AE_MOVDA32(mat_zb_x_vec));
    ae_int16x4 out;
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
    *out_ptr++ = (WORD8)AE_MOVAD16_3(out);
    *out_ptr++ = (WORD8)AE_MOVAD16_2(out);
  }

  for(; m_itr < (rows); m_itr ++)
  {
    ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
    ae_valignx2 mat_align0;
    ae_int32x2 acc0 = AE_MOVDA32(0);
    ae_int32x2 dummy = AE_MOVDA32(0);
    if(p_bias != NULL)
    {
      acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr]);
    }
    ae_int32x2 acc1;
    ae_int8x16 *p_mat_0 = (ae_int8x16 *)(&p_mat1[(m_itr)*(row_stride1/2)*sizeof(WORD8)]);

    mat_align0 = AE_LA128_PP(p_mat_0);

    WORD8 *p_vec_batch_0  = (WORD8 *)p_vec_flipped;

    ae_int8x8 mat0_0, mat0_1;
    ae_int4x16 mat0_0_4b, mat0_1_4b;
    ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved;
    ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved;
    ae_int8x8 vec0, vec1, vec2, vec3;
    ae_int16x4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;
    ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT64(0xFBEAD9C873625140);
    int rem_cols_shift_0, rem_cols_shift_1;
    
    rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
    rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

    for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec2, vec3, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));

      AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
      AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
      AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
      AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved);
      AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec0_0, vec0_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec1_0, vec1_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec2_0, vec2_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec3_0, vec3_1);
    }
    if(cols1 & 31)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
      mat0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_0), rem_cols_shift_0), rem_cols_shift_0));
      mat0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_1), rem_cols_shift_1), rem_cols_shift_1));                 

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_L8X8X2_IP(vec0, vec1, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec2, vec3, (ae_int8x16 *)p_vec_batch_0, (16 * sizeof(WORD8)));

      AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
      AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
      AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
      AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved);
      AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec0_0, vec0_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec1_0, vec1_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec2_0, vec2_1);
      AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec3_0, vec3_1);
    }
    acc0 = AE_ADD32S(acc0, AE_MOVDA32(mat_zb_x_vec));
    ae_int16x4 out;
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
    *out_ptr++ = (WORD8)AE_MOVAD16_3(out);
  }
  
  return 0;  
}

