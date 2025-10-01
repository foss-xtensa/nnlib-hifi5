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
  ae_int8x16 *vec_ptr = (ae_int8x16 *)p_vec;
  ae_int32x2 acc = ZERO32;
  WORD32 outval32 = 0;

  if(mat_zero_bias != 0)
  {
    ae_int32x2 acc0 = ZERO32, acc1 = ZERO32;
    ae_int8x8 matzb8 = AE_MOVDA8(mat_zero_bias);
    ae_int8x8 zero8 = AE_MOVDA8(0);

    for(int i=0; i< vec_len >> 4; i++){
      ae_int8x8 vec0, vec1;
      AE_L8X8X2_IP(vec0, vec1, vec_ptr, (16 * sizeof(WORD8)));
      AE_MULA8Q8X8(acc0, acc1, vec0, vec1, zero8, zero8, matzb8);
    }
    ae_valignx2 vec_align = AE_LA128_PP(vec_ptr);
    if(vec_len & 15)
    {
      ae_int8x8 vec0, vec1;
      AE_LAV8X8X2_XP(vec0, vec1, vec_align, vec_ptr, (vec_len & 15));
      AE_MULA8Q8X8(acc0, acc1, vec0, vec1, zero8, zero8, matzb8);
    }
    acc = AE_ADD32(acc0, acc1);
    acc = AE_ADD32_HL_LH(acc, acc);
    outval32 = AE_MOVAD32_L(acc);

    WORD32 vzb_contribution = vec_len*vec_zero_bias*mat_zero_bias;
    outval32 += vzb_contribution;
  }
  return outval32;
}

static void calculate_zero_point_x_4vectors(WORD32 *out0, WORD32 *out1, WORD32 *out2, WORD32 *out3, WORD32 vec_zero_bias, WORD32 mat_zero_bias, WORD8 *p_vec0, WORD8 *p_vec1, WORD8 *p_vec2, WORD8 *p_vec3, WORD32 vec_len)
{
  WORD32 outval0 = 0, outval1 = 0, outval2 = 0, outval3 = 0;
  ae_int8x16 *vec_ptr0 = (ae_int8x16 *)p_vec0;
  ae_int8x16 *vec_ptr1 = (ae_int8x16 *)p_vec1;
  ae_int8x16 *vec_ptr2 = (ae_int8x16 *)p_vec2;
  ae_int8x16 *vec_ptr3 = (ae_int8x16 *)p_vec3;

  WORD32 vzb_contribution = vec_len*vec_zero_bias*mat_zero_bias;

  ae_int32x2 accv0 = ZERO32;
  ae_int32x2 accv1 = ZERO32;
  ae_int32x2 accv2 = ZERO32;
  ae_int32x2 accv3 = ZERO32;

  if(mat_zero_bias != 0)
  {
    ae_int8x8 matzb8 = AE_MOVDA8(mat_zero_bias);

    for(int i=0; i< vec_len >> 4; i++){
      ae_int8x8 vec00, vec01, vec10, vec11, vec20, vec21, vec30, vec31;
      AE_L8X8X2_IP(vec00, vec01, vec_ptr0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec10, vec11, vec_ptr1, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec20, vec21, vec_ptr2, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec30, vec31, vec_ptr3, (16 * sizeof(WORD8)));
      AE_MULA8Q8X8(accv0, accv1, vec00, vec01, vec10, vec11, matzb8);
      AE_MULA8Q8X8(accv2, accv3, vec20, vec21, vec30, vec31, matzb8);
    }
    ae_valignx2 vec_align0 = AE_LA128_PP(vec_ptr0);
    ae_valignx2 vec_align1 = AE_LA128_PP(vec_ptr1);
    ae_valignx2 vec_align2 = AE_LA128_PP(vec_ptr2);
    ae_valignx2 vec_align3 = AE_LA128_PP(vec_ptr3);
    if(vec_len & 15)
    {
      ae_int8x8 vec00, vec01, vec10, vec11, vec20, vec21, vec30, vec31;
      AE_LAV8X8X2_XP(vec00, vec01, vec_align0, vec_ptr0, (vec_len & 15));
      AE_LAV8X8X2_XP(vec10, vec11, vec_align1, vec_ptr1, (vec_len & 15));
      AE_LAV8X8X2_XP(vec20, vec21, vec_align2, vec_ptr2, (vec_len & 15));
      AE_LAV8X8X2_XP(vec30, vec31, vec_align3, vec_ptr3, (vec_len & 15));
      AE_MULA8Q8X8(accv0, accv1, vec00, vec01, vec10, vec11, matzb8);
      AE_MULA8Q8X8(accv2, accv3, vec20, vec21, vec30, vec31, matzb8);
    }
    accv0 = AE_ADD32_HL_LH(accv0, accv0);
    accv1 = AE_ADD32_HL_LH(accv1, accv1);
    accv2 = AE_ADD32_HL_LH(accv2, accv2);
    accv3 = AE_ADD32_HL_LH(accv3, accv3);
    outval0 = AE_MOVAD32_L(accv0);
    outval1 = AE_MOVAD32_L(accv1);
    outval2 = AE_MOVAD32_L(accv2);
    outval3 = AE_MOVAD32_L(accv3);

    outval0 += vzb_contribution;
    outval1 += vzb_contribution;
    outval2 += vzb_contribution;
    outval3 += vzb_contribution;
  }
  *out0 = outval0;
  *out1 = outval1;
  *out2 = outval2;
  *out3 = outval3;
}

static inline void _xa_nn_dot_prod_4row_4vec_mat_unaligned_vec_aligned(
    ae_int32x2 *pacc0,
    ae_int32x2 *pacc1,
    ae_int32x2 *pacc2,
    ae_int32x2 *pacc3,
    ae_int32x2 *pacc4,
    ae_int32x2 *pacc5,
    ae_int32x2 *pacc6,
    ae_int32x2 *pacc7,
    WORD8      *p_mat,
    WORD32     row_stride,
    WORD8      *p_vec0,
    WORD8      *p_vec1,
    WORD8      *p_vec2,
    WORD8      *p_vec3,
    WORD32     cols,
    WORD32     vec1_zero_bias)
{
      ae_int32x2 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
      ae_int8x8 mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1;
      ae_int4x16 mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b;
      ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat1_0_4b_interleaved, mat1_1_4b_interleaved, mat1_2_4b_interleaved, mat1_3_4b_interleaved;
      ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved, mat1_0_8b_interleaved, mat1_1_8b_interleaved, mat1_2_8b_interleaved, mat1_3_8b_interleaved;

      ae_int8x8 vec00, vec01, vec02, vec03;
      ae_int8x8 vec10, vec11, vec12, vec13;
      ae_int8x8 vec20, vec21, vec22, vec33;
      ae_int8x8 vec30, vec31, vec32, vec23;
      ae_int16x4 vec00_0, vec00_1, vec01_0, vec01_1, vec02_0, vec02_1, vec03_0, vec03_1;
      ae_int16x4 vec10_0, vec10_1, vec11_0, vec11_1, vec12_0, vec12_1, vec13_0, vec13_1;
      ae_int16x4 vec20_0, vec20_1, vec21_0, vec21_1, vec22_0, vec22_1, vec23_0, vec23_1;
      ae_int16x4 vec30_0, vec30_1, vec31_0, vec31_1, vec32_0, vec32_1, vec33_0, vec33_1;
      ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xFBEAD9C8, 0x73625140));
      int rem_cols_shift_0, rem_cols_shift_1;

      acc0 = *pacc0; 
      acc1 = *pacc1; 
      acc2 = *pacc2; 
      acc3 = *pacc3; 
      acc4 = *pacc4; 
      acc5 = *pacc5; 
      acc6 = *pacc6; 
      acc7 = *pacc7; 

      ae_int8x16 *p_vec_batch_0 = (ae_int8x16 *)p_vec0;
      ae_int8x16 *p_vec_batch_1 = (ae_int8x16 *)p_vec1;
      ae_int8x16 *p_vec_batch_2 = (ae_int8x16 *)p_vec2;
      ae_int8x16 *p_vec_batch_3 = (ae_int8x16 *)p_vec3;
      ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);

      rem_cols_shift_0 = ((cols & 31) < 16) ? (64 - ((cols & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols & 31) < 16) ? 64 : (64 - (((cols & 31)-16) * 4));
  
      /* Prepare four rows from p_mat using row_stride */
      ae_int8x16 *p_mat_0 = (ae_int8x16 *)(p_mat);
      ae_int8x16 *p_mat_1 = (ae_int8x16 *)(&p_mat[1*(row_stride/2)*sizeof(WORD8)]);
      ae_int8x16 *p_mat_2 = (ae_int8x16 *)(&p_mat[2*(row_stride/2)*sizeof(WORD8)]);
      ae_int8x16 *p_mat_3 = (ae_int8x16 *)(&p_mat[3*(row_stride/2)*sizeof(WORD8)]);
  
      ae_valignx2 mat_align0 = AE_LA128_PP(p_mat_0);
      ae_valignx2 mat_align1 = AE_LA128_PP(p_mat_1);
      ae_valignx2 mat_align2 = AE_LA128_PP(p_mat_2);
      ae_valignx2 mat_align3 = AE_LA128_PP(p_mat_3);

      int c_itr;
      for(c_itr = 0; c_itr < (cols >> 5); c_itr++)
      {
        AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IP(mat1_0, mat1_1, mat_align1, p_mat_1);
        AE_LA8X8X2_IP(mat2_0, mat2_1, mat_align2, p_mat_2);
        AE_LA8X8X2_IP(mat3_0, mat3_1, mat_align3, p_mat_3);
        AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b, mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1);
  
        AE_L8X8X2_IP(vec00, vec01, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec02, vec03, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec10, vec11, p_vec_batch_1, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec12, vec13, p_vec_batch_1, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec20, vec21, p_vec_batch_2, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec22, vec23, p_vec_batch_2, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec30, vec31, p_vec_batch_3, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec32, vec33, p_vec_batch_3, (16 * sizeof(WORD8)));
  
        AE_SUBW8(vec00_0, vec00_1, vec00, vec_zb);
        AE_SUBW8(vec01_0, vec01_1, vec01, vec_zb);
        AE_SUBW8(vec02_0, vec02_1, vec02, vec_zb);
        AE_SUBW8(vec03_0, vec03_1, vec03, vec_zb);

        AE_SUBW8(vec10_0, vec10_1, vec10, vec_zb);
        AE_SUBW8(vec11_0, vec11_1, vec11, vec_zb);
        AE_SUBW8(vec12_0, vec12_1, vec12, vec_zb);
        AE_SUBW8(vec13_0, vec13_1, vec13, vec_zb);
  
        AE_SUBW8(vec20_0, vec20_1, vec20, vec_zb);
        AE_SUBW8(vec21_0, vec21_1, vec21, vec_zb);
        AE_SUBW8(vec22_0, vec22_1, vec22, vec_zb);
        AE_SUBW8(vec23_0, vec23_1, vec23, vec_zb);

        AE_SUBW8(vec30_0, vec30_1, vec30, vec_zb);
        AE_SUBW8(vec31_0, vec31_1, vec31, vec_zb);
        AE_SUBW8(vec32_0, vec32_1, vec32, vec_zb);
        AE_SUBW8(vec33_0, vec33_1, vec33, vec_zb);
  
        AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b, mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1);
  
        AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat1_0_4b), dsel_hh_ll);
        AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat1_1_4b), dsel_hh_ll);
        AE_DSEL8X8(mat1_0_8b_interleaved, mat1_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat2_0_4b), AE_MOVINT8X8_FROMINT4X16(mat3_0_4b), dsel_hh_ll);
        AE_DSEL8X8(mat1_2_8b_interleaved, mat1_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat2_1_4b), AE_MOVINT8X8_FROMINT4X16(mat3_1_4b), dsel_hh_ll);
  
        AE_MOVINT4X16_FROMINT8X8_4R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat1_0_4b_interleaved, mat1_1_4b_interleaved, mat1_2_4b_interleaved, mat1_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved, mat1_0_8b_interleaved, mat1_1_8b_interleaved, mat1_2_8b_interleaved, mat1_3_8b_interleaved);

        AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec00_0, vec00_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec01_0, vec01_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec02_0, vec02_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec03_0, vec03_1);

        AE_MULA8Q4X16(acc2, acc3, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec10_0, vec10_1);
        AE_MULA8Q4X16(acc2, acc3, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec11_0, vec11_1);
        AE_MULA8Q4X16(acc2, acc3, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec12_0, vec12_1);
        AE_MULA8Q4X16(acc2, acc3, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec13_0, vec13_1);

        AE_MULA8Q4X16(acc4, acc5, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec20_0, vec20_1);
        AE_MULA8Q4X16(acc4, acc5, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec21_0, vec21_1);
        AE_MULA8Q4X16(acc4, acc5, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec22_0, vec22_1);
        AE_MULA8Q4X16(acc4, acc5, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec23_0, vec23_1);

        AE_MULA8Q4X16(acc6, acc7, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec30_0, vec30_1);
        AE_MULA8Q4X16(acc6, acc7, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec31_0, vec31_1);
        AE_MULA8Q4X16(acc6, acc7, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec32_0, vec32_1);
        AE_MULA8Q4X16(acc6, acc7, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec33_0, vec33_1);
      }
      if(cols & 31)
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

        AE_L8X8X2_IP(vec00, vec01, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec02, vec03, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec10, vec11, p_vec_batch_1, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec12, vec13, p_vec_batch_1, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec20, vec21, p_vec_batch_2, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec22, vec23, p_vec_batch_2, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec30, vec31, p_vec_batch_3, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec32, vec33, p_vec_batch_3, (16 * sizeof(WORD8)));
  
        AE_SUBW8(vec00_0, vec00_1, vec00, vec_zb);
        AE_SUBW8(vec01_0, vec01_1, vec01, vec_zb);
        AE_SUBW8(vec02_0, vec02_1, vec02, vec_zb);
        AE_SUBW8(vec03_0, vec03_1, vec03, vec_zb);

        AE_SUBW8(vec10_0, vec10_1, vec10, vec_zb);
        AE_SUBW8(vec11_0, vec11_1, vec11, vec_zb);
        AE_SUBW8(vec12_0, vec12_1, vec12, vec_zb);
        AE_SUBW8(vec13_0, vec13_1, vec13, vec_zb);
  
        AE_SUBW8(vec20_0, vec20_1, vec20, vec_zb);
        AE_SUBW8(vec21_0, vec21_1, vec21, vec_zb);
        AE_SUBW8(vec22_0, vec22_1, vec22, vec_zb);
        AE_SUBW8(vec23_0, vec23_1, vec23, vec_zb);

        AE_SUBW8(vec30_0, vec30_1, vec30, vec_zb);
        AE_SUBW8(vec31_0, vec31_1, vec31, vec_zb);
        AE_SUBW8(vec32_0, vec32_1, vec32, vec_zb);
        AE_SUBW8(vec33_0, vec33_1, vec33, vec_zb);
  
        AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec00_0, vec00_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec01_0, vec01_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec02_0, vec02_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec03_0, vec03_1);

        AE_MULA8Q4X16(acc2, acc3, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec10_0, vec10_1);
        AE_MULA8Q4X16(acc2, acc3, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec11_0, vec11_1);
        AE_MULA8Q4X16(acc2, acc3, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec12_0, vec12_1);
        AE_MULA8Q4X16(acc2, acc3, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec13_0, vec13_1);

        AE_MULA8Q4X16(acc4, acc5, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec20_0, vec20_1);
        AE_MULA8Q4X16(acc4, acc5, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec21_0, vec21_1);
        AE_MULA8Q4X16(acc4, acc5, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec22_0, vec22_1);
        AE_MULA8Q4X16(acc4, acc5, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec23_0, vec23_1);

        AE_MULA8Q4X16(acc6, acc7, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec30_0, vec30_1);
        AE_MULA8Q4X16(acc6, acc7, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec31_0, vec31_1);
        AE_MULA8Q4X16(acc6, acc7, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec32_0, vec32_1);
        AE_MULA8Q4X16(acc6, acc7, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec33_0, vec33_1);
      }
    *pacc0 = acc0;
    *pacc1 = acc1;
    *pacc2 = acc2;
    *pacc3 = acc3;
    *pacc4 = acc4;
    *pacc5 = acc5;
    *pacc6 = acc6;
    *pacc7 = acc7;
}

static inline void _xa_nn_dot_prod_1row_4vec_mat_unaligned_vec_aligned(
    ae_int32x2 *pacc0,
    ae_int32x2 *pacc1,
    ae_int32x2 *pacc2,
    ae_int32x2 *pacc3,
    WORD8      *p_mat,
    WORD32     row_stride,
    WORD8      *p_vec0,
    WORD8      *p_vec1,
    WORD8      *p_vec2,
    WORD8      *p_vec3,
    WORD32     cols1,
    WORD32     vec1_zero_bias)
{
    ae_int32x2 dummy_acc;
    ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
    ae_int32x2 acc0, acc1, acc2, acc3;

    acc0 = *pacc0;
    acc1 = *pacc1;
    acc2 = *pacc2;
    acc3 = *pacc3;

    ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)p_vec0;
    ae_int8x16 *p_vec_batch_1  = (ae_int8x16 *)p_vec1;
    ae_int8x16 *p_vec_batch_2  = (ae_int8x16 *)p_vec2;
    ae_int8x16 *p_vec_batch_3  = (ae_int8x16 *)p_vec3;

    ae_int8x16 *p_mat_0 = (ae_int8x16 *)p_mat;

    ae_valignx2 mat_align0 = AE_LA128_PP(p_mat_0);
    ae_int8x8 mat0_0, mat0_1;
    ae_int4x16 mat0_0_4b, mat0_1_4b;
    ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved;
    ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved;
    ae_int8x8 vec00, vec01, vec02, vec03, vec10, vec11, vec12, vec13, vec20, vec21, vec22, vec23, vec30, vec31, vec32, vec33;
    ae_int16x4 vec00_0, vec00_1, vec01_0, vec01_1, vec02_0, vec02_1, vec03_0, vec03_1;
    ae_int16x4 vec10_0, vec10_1, vec11_0, vec11_1, vec12_0, vec12_1, vec13_0, vec13_1;
    ae_int16x4 vec20_0, vec20_1, vec21_0, vec21_1, vec22_0, vec22_1, vec23_0, vec23_1;
    ae_int16x4 vec30_0, vec30_1, vec31_0, vec31_1, vec32_0, vec32_1, vec33_0, vec33_1;

    ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xFBEAD9C8, 0x73625140));
    int rem_cols_shift_0, rem_cols_shift_1;
    
    rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
    rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));

    int c_itr;
    for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_L8X8X2_IP(vec00, vec01, p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec02, vec03, p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec10, vec11, p_vec_batch_1, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec12, vec13, p_vec_batch_1, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec20, vec21, p_vec_batch_2, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec22, vec23, p_vec_batch_2, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec30, vec31, p_vec_batch_3, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec32, vec33, p_vec_batch_3, (16 * sizeof(WORD8)));

      AE_SUBW8(vec00_0, vec00_1, vec00, vec_zb);
      AE_SUBW8(vec01_0, vec01_1, vec01, vec_zb);
      AE_SUBW8(vec02_0, vec02_1, vec02, vec_zb);
      AE_SUBW8(vec03_0, vec03_1, vec03, vec_zb);

      AE_SUBW8(vec10_0, vec10_1, vec10, vec_zb);
      AE_SUBW8(vec11_0, vec11_1, vec11, vec_zb);
      AE_SUBW8(vec12_0, vec12_1, vec12, vec_zb);
      AE_SUBW8(vec13_0, vec13_1, vec13, vec_zb);

      AE_SUBW8(vec20_0, vec20_1, vec20, vec_zb);
      AE_SUBW8(vec21_0, vec21_1, vec21, vec_zb);
      AE_SUBW8(vec22_0, vec22_1, vec22, vec_zb);
      AE_SUBW8(vec23_0, vec23_1, vec23, vec_zb);

      AE_SUBW8(vec30_0, vec30_1, vec30, vec_zb);
      AE_SUBW8(vec31_0, vec31_1, vec31, vec_zb);
      AE_SUBW8(vec32_0, vec32_1, vec32, vec_zb);
      AE_SUBW8(vec33_0, vec33_1, vec33, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved);

      AE_MULA8Q4X16(acc0, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec00_0, vec00_1);
      AE_MULA8Q4X16(acc0, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec01_0, vec01_1);
      AE_MULA8Q4X16(acc0, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec02_0, vec02_1);
      AE_MULA8Q4X16(acc0, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec03_0, vec03_1);

      AE_MULA8Q4X16(acc1, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec10_0, vec10_1);
      AE_MULA8Q4X16(acc1, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec11_0, vec11_1);
      AE_MULA8Q4X16(acc1, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec12_0, vec12_1);
      AE_MULA8Q4X16(acc1, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec13_0, vec13_1);

      AE_MULA8Q4X16(acc2, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec20_0, vec20_1);
      AE_MULA8Q4X16(acc2, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec21_0, vec21_1);
      AE_MULA8Q4X16(acc2, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec22_0, vec22_1);
      AE_MULA8Q4X16(acc2, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec23_0, vec23_1);

      AE_MULA8Q4X16(acc3, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec30_0, vec30_1);
      AE_MULA8Q4X16(acc3, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec31_0, vec31_1);
      AE_MULA8Q4X16(acc3, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec32_0, vec32_1);
      AE_MULA8Q4X16(acc3, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec33_0, vec33_1);
    }
    if(cols1 & 31)
    {
      AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
      mat0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_0), rem_cols_shift_0), rem_cols_shift_0));
      mat0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat0_1), rem_cols_shift_1), rem_cols_shift_1));                 

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_L8X8X2_IP(vec00, vec01, p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec02, vec03, p_vec_batch_0, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec10, vec11, p_vec_batch_1, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec12, vec13, p_vec_batch_1, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec20, vec21, p_vec_batch_2, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec22, vec23, p_vec_batch_2, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec30, vec31, p_vec_batch_3, (16 * sizeof(WORD8)));
      AE_L8X8X2_IP(vec32, vec33, p_vec_batch_3, (16 * sizeof(WORD8)));

      AE_SUBW8(vec00_0, vec00_1, vec00, vec_zb);
      AE_SUBW8(vec01_0, vec01_1, vec01, vec_zb);
      AE_SUBW8(vec02_0, vec02_1, vec02, vec_zb);
      AE_SUBW8(vec03_0, vec03_1, vec03, vec_zb);

      AE_SUBW8(vec10_0, vec10_1, vec10, vec_zb);
      AE_SUBW8(vec11_0, vec11_1, vec11, vec_zb);
      AE_SUBW8(vec12_0, vec12_1, vec12, vec_zb);
      AE_SUBW8(vec13_0, vec13_1, vec13, vec_zb);

      AE_SUBW8(vec20_0, vec20_1, vec20, vec_zb);
      AE_SUBW8(vec21_0, vec21_1, vec21, vec_zb);
      AE_SUBW8(vec22_0, vec22_1, vec22, vec_zb);
      AE_SUBW8(vec23_0, vec23_1, vec23, vec_zb);

      AE_SUBW8(vec30_0, vec30_1, vec30, vec_zb);
      AE_SUBW8(vec31_0, vec31_1, vec31, vec_zb);
      AE_SUBW8(vec32_0, vec32_1, vec32, vec_zb);
      AE_SUBW8(vec33_0, vec33_1, vec33, vec_zb);

      AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);

      AE_DSEL8X8(mat0_0_8b_interleaved, mat0_1_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), AE_MOVINT8X8_FROMINT4X16(mat0_0_4b), dsel_hh_ll);
      AE_DSEL8X8(mat0_2_8b_interleaved, mat0_3_8b_interleaved, AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), AE_MOVINT8X8_FROMINT4X16(mat0_1_4b), dsel_hh_ll);

      AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved);

      AE_MULA8Q4X16(acc0, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec00_0, vec00_1);
      AE_MULA8Q4X16(acc0, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec01_0, vec01_1);
      AE_MULA8Q4X16(acc0, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec02_0, vec02_1);
      AE_MULA8Q4X16(acc0, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec03_0, vec03_1);

      AE_MULA8Q4X16(acc1, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec10_0, vec10_1);
      AE_MULA8Q4X16(acc1, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec11_0, vec11_1);
      AE_MULA8Q4X16(acc1, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec12_0, vec12_1);
      AE_MULA8Q4X16(acc1, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec13_0, vec13_1);

      AE_MULA8Q4X16(acc2, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec20_0, vec20_1);
      AE_MULA8Q4X16(acc2, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec21_0, vec21_1);
      AE_MULA8Q4X16(acc2, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec22_0, vec22_1);
      AE_MULA8Q4X16(acc2, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec23_0, vec23_1);

      AE_MULA8Q4X16(acc3, dummy_acc, mat0_0_4b_interleaved, mat0_0_4b_interleaved, vec30_0, vec30_1);
      AE_MULA8Q4X16(acc3, dummy_acc, mat0_1_4b_interleaved, mat0_1_4b_interleaved, vec31_0, vec31_1);
      AE_MULA8Q4X16(acc3, dummy_acc, mat0_2_4b_interleaved, mat0_2_4b_interleaved, vec32_0, vec32_1);
      AE_MULA8Q4X16(acc3, dummy_acc, mat0_3_4b_interleaved, mat0_3_4b_interleaved, vec33_0, vec33_1);
    }
    *pacc0 = acc0;
    *pacc1 = acc1;
    *pacc2 = acc2;
    *pacc3 = acc3;
}

WORD32 xa_nn_matmul_asym4sxasym8s_asym8s(
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
    WORD32 mat1_zero_bias,
    WORD32 vec1_zero_bias,
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
  XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || mat1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1%2 != 0), -1);

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
  ae_f32x2 temp1, temp2, temp_multiplier, sraa32_temp;
#endif
  
  int vec_itr=0;
  for(; vec_itr < (vec_count&~3); vec_itr+=4)
  {
    WORD8 *p_vec_flipped0 = (WORD8 *)p_scratch;
    WORD8 *p_vec_flipped1 = (WORD8 *)p_scratch + cols1 + 32;
    WORD8 *p_vec_flipped2 = (WORD8 *)p_scratch + 2*(cols1 + 32);
    WORD8 *p_vec_flipped3 = (WORD8 *)p_scratch + 3*(cols1 + 32);
    p_vec_flipped0 = ALIGNED_ADDR(p_vec_flipped0, 16);
    p_vec_flipped1 = ALIGNED_ADDR(p_vec_flipped1, 16);
    p_vec_flipped2 = ALIGNED_ADDR(p_vec_flipped2, 16);
    p_vec_flipped3 = ALIGNED_ADDR(p_vec_flipped3, 16);
  
    ae_int8x16 *p_vec_flip_process0 = (ae_int8x16 *)p_vec_flipped0;
    ae_int8x16 *p_vec_flip_process1 = (ae_int8x16 *)p_vec_flipped1;
    ae_int8x16 *p_vec_flip_process2 = (ae_int8x16 *)p_vec_flipped2;
    ae_int8x16 *p_vec_flip_process3 = (ae_int8x16 *)p_vec_flipped3;
    ae_int8x16 *p_vec_in0 = (ae_int8x16 *)(p_vec1 + vec_itr*vec_offset);
    ae_int8x16 *p_vec_in1 = (ae_int8x16 *)(p_vec1 + (vec_itr+1)*vec_offset);
    ae_int8x16 *p_vec_in2 = (ae_int8x16 *)(p_vec1 + (vec_itr+2)*vec_offset);
    ae_int8x16 *p_vec_in3 = (ae_int8x16 *)(p_vec1 + (vec_itr+3)*vec_offset);
  
    ae_int8x8 vec0, vec1;
    ae_valignx2 vec_align0, vec_align1;
    ae_valignx2 vec_align2, vec_align3;
    vec_align0 = AE_LA128_PP(p_vec_in0);
    vec_align1 = AE_LA128_PP(p_vec_in1);
    vec_align2 = AE_LA128_PP(p_vec_in2);
    vec_align3 = AE_LA128_PP(p_vec_in3);
  
    /* Below code is for inverting order of vector */
    for(int vec_itr=0; vec_itr < cols1 >> 4; vec_itr++)
    {
      AE_LA8X8X2_IP(vec0, vec1, vec_align0, p_vec_in0);
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process0, 16);

      AE_LA8X8X2_IP(vec0, vec1, vec_align1, p_vec_in1);
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process1, 16);

      AE_LA8X8X2_IP(vec0, vec1, vec_align2, p_vec_in2);
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process2, 16);

      AE_LA8X8X2_IP(vec0, vec1, vec_align3, p_vec_in3);
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process3, 16);
    }
    if(cols1 & 15)
    {
      AE_LAV8X8X2_XP(vec0, vec1, vec_align0, p_vec_in0, (cols1 & 15));
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process0, 16);

      AE_LAV8X8X2_XP(vec0, vec1, vec_align1, p_vec_in1, (cols1 & 15));
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process1, 16);

      AE_LAV8X8X2_XP(vec0, vec1, vec_align2, p_vec_in2, (cols1 & 15));
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process2, 16);

      AE_LAV8X8X2_XP(vec0, vec1, vec_align3, p_vec_in3, (cols1 & 15));
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_S8X8X2_IP(vec0, vec1, p_vec_flip_process3, 16);
    }
  
    WORD32 mat_zb_x_vec0, mat_zb_x_vec1, mat_zb_x_vec2, mat_zb_x_vec3; 
    calculate_zero_point_x_4vectors(&mat_zb_x_vec0, &mat_zb_x_vec1, &mat_zb_x_vec2, &mat_zb_x_vec3, 
        vec1_zero_bias, mat1_zero_bias, p_vec_flipped0, p_vec_flipped1, p_vec_flipped2, p_vec_flipped3, cols1);

    WORD8 *out_ptr0 = p_out + vec_itr*out_offset;
    WORD8 *out_ptr1 = p_out + (vec_itr+1)*out_offset;
    WORD8 *out_ptr2 = p_out + (vec_itr+2)*out_offset;
    WORD8 *out_ptr3 = p_out + (vec_itr+3)*out_offset;

    int m_itr = 0;
    
    for(; m_itr < (rows & ~(4-1)); m_itr += 4)
    {
      ae_int32x2 acc0 = SW_MOVDA32(0);
      ae_int32x2 acc1 = SW_MOVDA32(0);    
      ae_int32x2 acc2 = SW_MOVDA32(0);
      ae_int32x2 acc3 = SW_MOVDA32(0);    
      ae_int32x2 acc4 = SW_MOVDA32(0);
      ae_int32x2 acc5 = SW_MOVDA32(0);    
      ae_int32x2 acc6 = SW_MOVDA32(0);
      ae_int32x2 acc7 = SW_MOVDA32(0);    
      if(p_bias != NULL)
      {
        acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr+1]);
        acc1 = AE_MOVDA32X2(p_bias[m_itr+2], p_bias[m_itr+3]);
        acc2 = acc0;
        acc3 = acc1;
        acc4 = acc0;
        acc5 = acc1;
        acc6 = acc0;
        acc7 = acc1;
      }
  
      WORD8 *p_vec_batch_0  = (WORD8 *)p_vec_flipped0;
      WORD8 *p_vec_batch_1  = (WORD8 *)p_vec_flipped1;
      WORD8 *p_vec_batch_2  = (WORD8 *)p_vec_flipped2;
      WORD8 *p_vec_batch_3  = (WORD8 *)p_vec_flipped3;
  
      _xa_nn_dot_prod_4row_4vec_mat_unaligned_vec_aligned(&acc0, &acc1, &acc2, &acc3, &acc4, &acc5, &acc6, &acc7, 
          (WORD8 *)&p_mat1[m_itr*row_stride1/2], row_stride1, p_vec_batch_0, p_vec_batch_1, p_vec_batch_2, p_vec_batch_3, cols1, vec1_zero_bias);

      acc0 = SW_ADD32S_INT32X2_INT32X2(acc0, SW_MOVDA32(mat_zb_x_vec0));
      acc1 = SW_ADD32S_INT32X2_INT32X2(acc1, SW_MOVDA32(mat_zb_x_vec0));
      acc2 = SW_ADD32S_INT32X2_INT32X2(acc2, SW_MOVDA32(mat_zb_x_vec1));
      acc3 = SW_ADD32S_INT32X2_INT32X2(acc3, SW_MOVDA32(mat_zb_x_vec1));
      acc4 = SW_ADD32S_INT32X2_INT32X2(acc4, SW_MOVDA32(mat_zb_x_vec2));
      acc5 = SW_ADD32S_INT32X2_INT32X2(acc5, SW_MOVDA32(mat_zb_x_vec2));
      acc6 = SW_ADD32S_INT32X2_INT32X2(acc6, SW_MOVDA32(mat_zb_x_vec3));
      acc7 = SW_ADD32S_INT32X2_INT32X2(acc7, SW_MOVDA32(mat_zb_x_vec3));

      ae_int16x4 out;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, acc1, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr0 = (WORD8)AE_MOVAD16_3(out); out_ptr0 += out_stride;
      *out_ptr0 = (WORD8)AE_MOVAD16_2(out); out_ptr0 += out_stride;
      *out_ptr0 = (WORD8)AE_MOVAD16_1(out); out_ptr0 += out_stride;
      *out_ptr0 = (WORD8)AE_MOVAD16_0(out); out_ptr0 += out_stride;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc2, acc3, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr1 = (WORD8)AE_MOVAD16_3(out); out_ptr1 += out_stride;
      *out_ptr1 = (WORD8)AE_MOVAD16_2(out); out_ptr1 += out_stride;
      *out_ptr1 = (WORD8)AE_MOVAD16_1(out); out_ptr1 += out_stride;
      *out_ptr1 = (WORD8)AE_MOVAD16_0(out); out_ptr1 += out_stride;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc4, acc5, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr2 = (WORD8)AE_MOVAD16_3(out); out_ptr2 += out_stride;
      *out_ptr2 = (WORD8)AE_MOVAD16_2(out); out_ptr2 += out_stride;
      *out_ptr2 = (WORD8)AE_MOVAD16_1(out); out_ptr2 += out_stride;
      *out_ptr2 = (WORD8)AE_MOVAD16_0(out); out_ptr2 += out_stride;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc6, acc7, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr3 = (WORD8)AE_MOVAD16_3(out); out_ptr3 += out_stride;
      *out_ptr3 = (WORD8)AE_MOVAD16_2(out); out_ptr3 += out_stride;
      *out_ptr3 = (WORD8)AE_MOVAD16_1(out); out_ptr3 += out_stride;
      *out_ptr3 = (WORD8)AE_MOVAD16_0(out); out_ptr3 += out_stride;
    }

    for(; m_itr < rows ; m_itr++ )
    {
      ae_int32x2 acc0 = SW_MOVDA32(0);
      ae_int32x2 acc1 = SW_MOVDA32(0);
      ae_int32x2 acc2 = SW_MOVDA32(0);
      ae_int32x2 acc3 = SW_MOVDA32(0);
      ae_int32x2 dummy = SW_MOVDA32(0);
      if(p_bias != NULL)
      {
        acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr]);
        acc1 = acc0;
        acc2 = acc0;
        acc3 = acc0;
      }
  
      WORD8 *p_vec_batch_0  = (WORD8 *)p_vec_flipped0;
      WORD8 *p_vec_batch_1  = (WORD8 *)p_vec_flipped1;
      WORD8 *p_vec_batch_2  = (WORD8 *)p_vec_flipped2;
      WORD8 *p_vec_batch_3  = (WORD8 *)p_vec_flipped3;

      _xa_nn_dot_prod_1row_4vec_mat_unaligned_vec_aligned(&acc0, &acc1, &acc2, &acc3,  
          (WORD8 *)&p_mat1[m_itr*row_stride1/2], row_stride1, p_vec_batch_0, p_vec_batch_1, p_vec_batch_2, p_vec_batch_3, cols1, vec1_zero_bias);
  
      ae_int16x4 out;

      acc0 = SW_ADD32S_INT32X2_INT32X2(acc0, SW_MOVDA32(mat_zb_x_vec0));
      acc1 = SW_ADD32S_INT32X2_INT32X2(acc1, SW_MOVDA32(mat_zb_x_vec1));
      acc2 = SW_ADD32S_INT32X2_INT32X2(acc2, SW_MOVDA32(mat_zb_x_vec2));
      acc3 = SW_ADD32S_INT32X2_INT32X2(acc3, SW_MOVDA32(mat_zb_x_vec3));

      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr0 = (WORD8)AE_MOVAD16_3(out); out_ptr0 += out_stride;

      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc1, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr1 = (WORD8)AE_MOVAD16_3(out); out_ptr1 += out_stride;

      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc2, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr2 = (WORD8)AE_MOVAD16_3(out); out_ptr2 += out_stride;

      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc3, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr3 = (WORD8)AE_MOVAD16_3(out); out_ptr3 += out_stride;
    }

  }

  for(; vec_itr < vec_count; vec_itr++)
  {
    WORD8 *p_vec_flipped = (WORD8 *)p_scratch;
    p_vec_flipped = ALIGNED_ADDR(p_vec_flipped, 16);
  
    ae_int8x16 *p_vec_flip_process = (ae_int8x16 *)p_vec_flipped;
    ae_int8x16 *p_vec_in = (ae_int8x16 *)(p_vec1 + vec_itr*vec_offset);
  
    ae_int8x8 vec0, vec1;
    ae_valignx2 vec_align, vec_flipped_align;
    vec_align = AE_LA128_PP(p_vec_in);
    vec_flipped_align = AE_ZALIGN128();
  
    /* Below code is for inverting order of vector by adding vector_zero_bias to the vector values. */
    for(int vec_itr=0; vec_itr < cols1 >> 4; vec_itr++)
    {
      AE_LA8X8X2_IP(vec0, vec1, vec_align, p_vec_in);
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_SA8X8X2_IP(vec0, vec1, vec_flipped_align, p_vec_flip_process);
    }
    if(cols1 & 15)
    {
      AE_LAV8X8X2_XP(vec0, vec1, vec_align, p_vec_in, (cols1 & 15));
      AE_DSEL8X8(vec0, vec1, vec0, vec1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xE6F7C4D5, 0xA2B38091)));
      AE_SA8X8X2_IP(vec0, vec1, vec_flipped_align, p_vec_flip_process);
    }
    AE_SA128POS_FP(vec_flipped_align, p_vec_flip_process);
  
    WORD32 mat_zb_x_vec = calculate_zero_point_x_vector(vec1_zero_bias, mat1_zero_bias, p_vec_flipped, cols1);
    int m_itr = 0, c_itr = 0;
  
    WORD8 *out_ptr = p_out + vec_itr*out_offset;
    //ae_f32x2 temp1, temp2, temp_multiplier, sraa32_temp;
    for(; m_itr < (rows & ~(4-1)); m_itr += 4)
    {
      ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
      ae_valignx2 mat_align0, mat_align1, mat_align2, mat_align3;
      ae_int32x2 acc0 = SW_MOVDA32(0);
      ae_int32x2 acc1 = SW_MOVDA32(0);
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
  
      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)p_vec_flipped;

      ae_int8x8 mat0_0, mat0_1, mat1_0, mat1_1, mat2_0, mat2_1, mat3_0, mat3_1;
      ae_int4x16 mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat2_0_4b, mat2_1_4b, mat3_0_4b, mat3_1_4b;
      ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved, mat1_0_4b_interleaved, mat1_1_4b_interleaved, mat1_2_4b_interleaved, mat1_3_4b_interleaved;
      ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved, mat1_0_8b_interleaved, mat1_1_8b_interleaved, mat1_2_8b_interleaved, mat1_3_8b_interleaved;
      ae_int8x8 vec0, vec1, vec2, vec3;
      ae_int16x4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;
      ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xFBEAD9C8, 0x73625140));
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
  
        AE_L8X8X2_IP(vec0, vec1, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec2, vec3, p_vec_batch_0, (16 * sizeof(WORD8)));
  
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
        AE_L8X8X2_IP(vec0, vec1, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec2, vec3, p_vec_batch_0, (16 * sizeof(WORD8)));
  
        AE_SUBW8(vec0_0, vec0_1, vec0, vec_zb);
        AE_SUBW8(vec1_0, vec1_1, vec1, vec_zb);
        AE_SUBW8(vec2_0, vec2_1, vec2, vec_zb);
        AE_SUBW8(vec3_0, vec3_1, vec3, vec_zb);
  
        AE_MULA8Q4X16(acc0, acc1, mat0_0_4b_interleaved, mat1_0_4b_interleaved, vec0_0, vec0_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_1_4b_interleaved, mat1_1_4b_interleaved, vec1_0, vec1_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_2_4b_interleaved, mat1_2_4b_interleaved, vec2_0, vec2_1);
        AE_MULA8Q4X16(acc0, acc1, mat0_3_4b_interleaved, mat1_3_4b_interleaved, vec3_0, vec3_1);
      }
      acc0 = SW_ADD32S_INT32X2_INT32X2(acc0, SW_MOVDA32(mat_zb_x_vec));
      acc1 = SW_ADD32S_INT32X2_INT32X2(acc1, SW_MOVDA32(mat_zb_x_vec));
      ae_int16x4 out;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, acc1, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr = (WORD8)AE_MOVAD16_3(out); out_ptr += out_stride;
      *out_ptr = (WORD8)AE_MOVAD16_2(out); out_ptr += out_stride;
      *out_ptr = (WORD8)AE_MOVAD16_1(out); out_ptr += out_stride;
      *out_ptr = (WORD8)AE_MOVAD16_0(out); out_ptr += out_stride;
    }
  
    for(; m_itr < (rows & ~(2-1)); m_itr += 2)
    {
      ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
      ae_valignx2 mat_align0, mat_align1;
      ae_int32x2 acc0 = SW_MOVDA32(0);
      ae_int32x2 dummy = SW_MOVDA32(0);
      if(p_bias != NULL)
      {
        acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr+1]);
      }
      ae_int32x2 acc1;
      ae_int8x16 *p_mat_0 = (ae_int8x16 *)(&p_mat1[(m_itr)*(row_stride1/2)*sizeof(WORD8)]);
      ae_int8x16 *p_mat_1 = (ae_int8x16 *)(&p_mat1[(m_itr+1)*(row_stride1/2)*sizeof(WORD8)]);
  
      mat_align0 = AE_LA128_PP(p_mat_0);
      mat_align1 = AE_LA128_PP(p_mat_1);
  
      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)p_vec_flipped;
  
      ae_int8x8 mat0_0, mat0_1, mat1_0, mat1_1;
      ae_int4x16 mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b;
      ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved;
      ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved;
      ae_int8x8 vec0, vec1, vec2, vec3;
      ae_int16x4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;
      ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xFBEAD9C8, 0x73625140));
      int rem_cols_shift_0, rem_cols_shift_1;
      
      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));
  
      for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
      {
        AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
        AE_LA8X8X2_IP(mat1_0, mat1_1, mat_align1, p_mat_1);
  
        AE_MOVINT4X16_FROMINT8X8_2R(mat0_0_4b, mat0_1_4b, mat1_0_4b, mat1_1_4b, mat0_0, mat0_1, mat1_0, mat1_1);
  
        AE_L8X8X2_IP(vec0, vec1, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec2, vec3, p_vec_batch_0, (16 * sizeof(WORD8)));
  
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
  
        AE_L8X8X2_IP(vec0, vec1, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec2, vec3, p_vec_batch_0, (16 * sizeof(WORD8)));
  
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
      acc0 = SW_ADD32S_INT32X2_INT32X2(acc0, SW_MOVDA32(mat_zb_x_vec));
      ae_int16x4 out;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr = (WORD8)AE_MOVAD16_3(out); out_ptr += out_stride;
      *out_ptr = (WORD8)AE_MOVAD16_2(out); out_ptr += out_stride;
    }
  
    for(; m_itr < (rows); m_itr ++)
    {
      ae_int8x8 vec_zb = AE_MOVDA8(-vec1_zero_bias);
      ae_valignx2 mat_align0;
      ae_int32x2 acc0 = SW_MOVDA32(0);
      ae_int32x2 dummy = SW_MOVDA32(0);
      if(p_bias != NULL)
      {
        acc0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr]);
      }
      ae_int32x2 acc1;
      ae_int8x16 *p_mat_0 = (ae_int8x16 *)(&p_mat1[(m_itr)*(row_stride1/2)*sizeof(WORD8)]);
  
      mat_align0 = AE_LA128_PP(p_mat_0);
  
      ae_int8x16 *p_vec_batch_0  = (ae_int8x16 *)p_vec_flipped;
  
      ae_int8x8 mat0_0, mat0_1;
      ae_int4x16 mat0_0_4b, mat0_1_4b;
      ae_int4x16 mat0_0_4b_interleaved, mat0_1_4b_interleaved, mat0_2_4b_interleaved, mat0_3_4b_interleaved;
      ae_int8x8 mat0_0_8b_interleaved, mat0_1_8b_interleaved, mat0_2_8b_interleaved, mat0_3_8b_interleaved;
      ae_int8x8 vec0, vec1, vec2, vec3;
      ae_int16x4 vec0_0, vec0_1, vec1_0, vec1_1, vec2_0, vec2_1, vec3_0, vec3_1;
      ae_int8x8 dsel_hh_ll = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xFBEAD9C8, 0x73625140));
      int rem_cols_shift_0, rem_cols_shift_1;
      
      rem_cols_shift_0 = ((cols1 & 31) < 16) ? (64 - ((cols1 & 31) * 4)) : 0;
      rem_cols_shift_1 = ((cols1 & 31) < 16) ? 64 : (64 - (((cols1 & 31)-16) * 4));
  
      for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
      {
        AE_LA8X8X2_IP(mat0_0, mat0_1, mat_align0, p_mat_0);
  
        AE_MOVINT4X16_FROMINT8X8_1R(mat0_0_4b, mat0_1_4b, mat0_0, mat0_1);
  
        AE_L8X8X2_IP(vec0, vec1, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec2, vec3, p_vec_batch_0, (16 * sizeof(WORD8)));
  
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
  
        AE_L8X8X2_IP(vec0, vec1, p_vec_batch_0, (16 * sizeof(WORD8)));
        AE_L8X8X2_IP(vec2, vec3, p_vec_batch_0, (16 * sizeof(WORD8)));
  
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
      acc0 = SW_ADD32S_INT32X2_INT32X2(acc0, SW_MOVDA32(mat_zb_x_vec));
      ae_int16x4 out;
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, acc0, dummy, out_multiplier, left_shift, right_shift, out_zero_bias)
      AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));
      *out_ptr = (WORD8)AE_MOVAD16_3(out); out_ptr += out_stride;
    }
  }
  
  return 0;
}
