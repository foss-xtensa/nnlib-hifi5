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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include <string.h>

#ifdef AE_MULAZB8Q8X8
  #define MAT_VEC_MAC(a0, a1, c0, c1, c2, c3, v0, cz, vz)  AE_MULAZB8Q8X8(a0, a1, c0, c1, c2, c3, v0)
#else
  #define MAT_VEC_MAC(a0, a1, c0, c1, c2, c3, v0, cz, vz) \
  { \
    ae_int16x4 va0, va1; \
    ae_int8x8 d_cz = AE_MOVDA8(cz); \
    AE_SUBW8(va0, va1, v0, AE_MOVDA8(vz)); \
    AE_MULA8Q8X16(a0, a1, c0, c1, c2, c3, va0, va1); \
    AE_MULA8Q8X16(a0, a1, d_cz, d_cz, d_cz, d_cz, AE_NEG16S(va0), AE_NEG16S(va1)); \
  }
#endif

#define PACK_32X2(dst1, src1, src2) \
  dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

extern const long long g_sel_pattern[16];

extern const long long pre_loop_sel_pattern[16];

extern const long long post_loop_sel_pattern[16];

static inline void special_function_for_cols_mul_32
    (WORD8*       p_out_0
    ,const WORD8* p_mat1_
    ,const WORD8* p_vec1_0
    ,const WORD32* p_bias_0
    ,WORD32        n_rows
    ,WORD32        n_vecs
    ,WORD32        cols
#ifndef AE_MULAZB8Q8X8
    ,WORD32        mat1_zb
    ,WORD32        vec1_zb
#endif
    ,WORD32        out_mul
    ,WORD32        l_shift
    ,WORD32        r_shift
    ,WORD32        out_z_b
    ,WORD32        out_offset_
    )
{
#if TFLITE_SINGLE_ROUNDING
    (void)r_shift;
#endif

    ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
    ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
    ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
    ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;

    ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
    ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3;
    ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3;
    ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;

    ae_int32x4 *pt_bias;
    ae_valignx2 bias_a;
    ae_int32x2 d_bias0, d_bias1;

    int m_itr = 0, vec_itr = 0;
    for (vec_itr = 0; vec_itr < (n_vecs & ~(4 - 1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out_0 + (vec_itr + 0) * out_offset_;
      WORD8* p_dst_1 = (WORD8*)p_out_0 + (vec_itr + 1) * out_offset_;
      WORD8* p_dst_2 = (WORD8*)p_out_0 + (vec_itr + 2) * out_offset_;
      WORD8* p_dst_3 = (WORD8*)p_out_0 + (vec_itr + 3) * out_offset_;

      if(p_bias_0)
      {
        pt_bias = (ae_int32x4 *)p_bias_0;
        bias_a = AE_LA128_PP(pt_bias);
      }

      for(m_itr = 0; m_itr < (n_rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1_[(m_itr + 0) * cols];
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1_0 + vec_itr * cols);

        d_bias0 = d_bias1 = AE_ZERO32();
        if(p_bias_0)
        {
          /* Load bias values */
          AE_LA32X2X2_IP(d_bias0, d_bias1, bias_a, pt_bias);
        }

        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1;
        ae_int32x2 acc_row0_vec2;
        ae_int32x2 acc_row1_vec2;
        ae_int32x2 acc_row0_vec3;
        ae_int32x2 acc_row1_vec3;

        /* Initialize accumulators with bias */
        acc_row0_vec0 = acc_row0_vec1 = d_bias0;
        acc_row0_vec2 = acc_row0_vec3 = d_bias0;
        acc_row1_vec0 = acc_row1_vec1 = d_bias1;
        acc_row1_vec2 = acc_row1_vec3 = d_bias1;

        int c_itr = 0;

#pragma loop_count min=1
        for(c_itr = 0; c_itr < cols>>5; c_itr++)
        {
          /* Load 4 rows */
          AE_L8X8X2_I(mat1_row0_0, mat1_row0_1, (ae_int8x16*)p_mat1_0, 0);
          AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16*)p_mat1_0, 16);
          AE_L8X8X2_X(mat1_row1_0, mat1_row1_1, (ae_int8x16*)p_mat1_0, cols);
          AE_L8X8X2_X(mat1_row1_2, mat1_row1_3, (ae_int8x16*)p_mat1_0, cols+16);
          AE_L8X8X2_X(mat1_row2_0, mat1_row2_1, (ae_int8x16*)p_mat1_0, 2*cols);
          AE_L8X8X2_X(mat1_row2_2, mat1_row2_3, (ae_int8x16*)p_mat1_0, 2*cols+16);
          AE_L8X8X2_X(mat1_row3_0, mat1_row3_1, (ae_int8x16*)p_mat1_0, 3*cols);
          AE_L8X8X2_X(mat1_row3_2, mat1_row3_3, (ae_int8x16*)p_mat1_0, 3*cols+16);

          /* Load  4 vectors  */
          AE_L8X8X2_I(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 0);
          AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, (ae_int8x16*)p_vec_0, 16);
          AE_L8X8X2_X(vec1_batch_0, vec1_batch_1, (ae_int8x16*)p_vec_0, cols);
          AE_L8X8X2_X(vec1_batch_2, vec1_batch_3, (ae_int8x16*)p_vec_0, cols+16);
          AE_L8X8X2_X(vec2_batch_0, vec2_batch_1, (ae_int8x16*)p_vec_0, 2*cols);
          AE_L8X8X2_X(vec2_batch_2, vec2_batch_3, (ae_int8x16*)p_vec_0, 2*cols+16);
          AE_L8X8X2_X(vec3_batch_0, vec3_batch_1, (ae_int8x16*)p_vec_0, 3*cols);
          AE_L8X8X2_X(vec3_batch_2, vec3_batch_3, (ae_int8x16*)p_vec_0, 3*cols+16);

          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);

          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1, mat1_zb, vec1_zb);

          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec1_batch_2, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec2_batch_2, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec3_batch_2, mat1_zb, vec1_zb);

          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec1_batch_3, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec2_batch_3, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec3_batch_3, mat1_zb, vec1_zb);

          p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 + 32);
          p_vec_0 += 32;
        }

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_mul, l_shift, r_shift, out_z_b);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, out_mul, l_shift, r_shift, out_z_b);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, out_mul, l_shift, r_shift, out_z_b);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, out_mul, l_shift, r_shift, out_z_b);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

        /* Store output */
        ae_int8x8 out32_0, out32_1;
        PACK_32X2(out32_0, out_0, out_1);
        PACK_32X2(out32_1, out_2, out_3);

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, 4);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_1, 4);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_2, 4);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_3, 4);
      }
    }
    for (vec_itr = (n_vecs & (~3)); vec_itr < n_vecs; vec_itr++)
    {
      WORD8* p_dst_0 = (WORD8*)p_out_0 + (vec_itr * out_offset_);
      if(p_bias_0)
      {
        pt_bias = (ae_int32x4 *)p_bias_0;
        bias_a = AE_LA128_PP(pt_bias);
      }

      for(m_itr = 0; m_itr < (n_rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1_[(m_itr + 0) * cols];
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1_0 + vec_itr * cols);

        d_bias0 = d_bias1 = AE_ZERO32();
        if(p_bias_0)
        {
          /* Load bias values */
          AE_LA32X2X2_IP(d_bias0, d_bias1, bias_a, pt_bias);
        }

        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        /* Initialize accumulators with bias */
        acc_row0_vec0 = d_bias0;
        acc_row1_vec0 = d_bias1;

        int c_itr = 0;

#pragma loop_count min=1
        for(c_itr = 0; c_itr < cols>>5; c_itr++)
        {
          /* Load 4 rows */
          AE_L8X8X2_I(mat1_row0_0, mat1_row0_1, (ae_int8x16*)p_mat1_0, 0);
          AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16*)p_mat1_0, 16);
          AE_L8X8X2_X(mat1_row1_0, mat1_row1_1, (ae_int8x16*)p_mat1_0, cols);
          AE_L8X8X2_X(mat1_row1_2, mat1_row1_3, (ae_int8x16*)p_mat1_0, cols+16);
          AE_L8X8X2_X(mat1_row2_0, mat1_row2_1, (ae_int8x16*)p_mat1_0, 2*cols);
          AE_L8X8X2_X(mat1_row2_2, mat1_row2_3, (ae_int8x16*)p_mat1_0, 2*cols+16);
          AE_L8X8X2_X(mat1_row3_0, mat1_row3_1, (ae_int8x16*)p_mat1_0, 3*cols);
          AE_L8X8X2_X(mat1_row3_2, mat1_row3_3, (ae_int8x16*)p_mat1_0, 3*cols+16);

          /* Load  4 vectors  */
          AE_L8X8X2_I(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 0);
          AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, (ae_int8x16*)p_vec_0, 16);

          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2, mat1_zb, vec1_zb);
          MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3, mat1_zb, vec1_zb);

          p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 + 32);
          p_vec_0 += 32;
        }

        /* Apply quantization */
        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_mul, l_shift, r_shift, out_z_b);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        /* Store output */
        ae_int8x8 out32_0;
        PACK_32X2(out32_0, out_0, out_0);

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, 4);
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
    ,WORD32      mat1_zb
#ifndef AE_MULAZB8Q8X8
    ,WORD32      vec1_zb
#endif
    )
{
    int c_itr = 0;

    ae_int8x8 mat_bias = AE_MOVDA8((WORD8)mat1_zb);
    int rem_cols = cols & 15;
    int rem_g8 = ((rem_cols & 15) > 8)?1:0;
    ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8 + 1]));
    ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8 + 1]));

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

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1, mat1_zb, vec1_zb);
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

        mat1_row0_0 = AE_SEL8X8(mat1_row0_0, mat_bias, sel1);
        mat1_row1_0 = AE_SEL8X8(mat1_row1_0, mat_bias, sel1);
        mat1_row2_0 = AE_SEL8X8(mat1_row2_0, mat_bias, sel1);
        mat1_row3_0 = AE_SEL8X8(mat1_row3_0, mat_bias, sel1);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);
        c_itr += 8;
        sel1 = sel2;
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
#ifndef AE_MULAZB8Q8X8
    ,WORD32      mat1_zb
#endif
    ,WORD32      vec1_zb
    )
{
  int c_itr = 0;

  ae_int8x8 vec_bias = AE_MOVDA8((WORD8)vec1_zb);
  int rem_cols = cols & 15;
  int rem_g8 = ((rem_cols & 15) > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8 + 1]));
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8 + 1]));

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

      MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);

      MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
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
      vec0_batch_0 = AE_SEL8X8(vec0_batch_0, vec_bias, sel1);

      MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
      c_itr += 8;
      sel1 = sel2;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
#ifndef AE_MULAZB8Q8X8
    ,WORD32      mat1_zb
#endif
    ,WORD32      vec1_zb
    )
{
    int c_itr = 0;
    ae_int8x8 vec0_batch_0, vec0_batch_1;
    ae_int8x8 mat1_row0_0, mat1_row0_1;

    ae_int32x2 acc_row0_vec0 = *out_0_0;
    ae_int32x2 acc_row0_vec1 = *out_1_0;

    ae_int8x8 vec_bias = AE_MOVDA8((WORD8)vec1_zb);
    int rem_cols = cols1 & 15;
    int rem_g8 = ((rem_cols & 15) > 8)?1:0;
    ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8 + 1]));
    ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8 + 1]));

    int cols_count = cols1 - (cols1 & 15);

    for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
    {
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

        AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 ,vec0_batch_1, mat1_zb, vec1_zb);

    }

    //Remainder loop for cols1
    c_itr <<= 4;
    while(c_itr < cols1)
    {
        AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);

        AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
        vec0_batch_0 = AE_SEL8X8(vec0_batch_0, vec_bias, sel1);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0, mat1_zb, vec1_zb);
        c_itr += 8;
        sel1 = sel2;
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
    ,WORD32      mat1_zb
#ifndef AE_MULAZB8Q8X8
    ,WORD32      vec1_zb
#endif
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  ae_int8x8 mat_bias = AE_MOVDA8((WORD8)mat1_zb);

  ae_int8x8 vec0_batch_0, vec0_batch_1;
  ae_int8x8 vec1_batch_0, vec1_batch_1;
  ae_int8x8 vec2_batch_0, vec2_batch_1;
  ae_int8x8 vec3_batch_0, vec3_batch_1;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  int align_offset = ((unsigned int)p_mat1_0 & 0x7);
  pre_loop_count = 8 - align_offset;
  ae_int8x8 pre_sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset & 7)], pre_loop_sel_pattern[2 * (align_offset & 7) + 1]));
  p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step
  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 15):((cols + align_offset) & 15);
  loop_count >>= 4;
  int mask_start_end = ((cols + align_offset) < 16)?0:1;

  int rem_g8 = (post_loop_count > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (post_loop_count & 7) * !rem_g8], post_loop_sel_pattern[2 * (post_loop_count & 7) * !rem_g8 + 1]));
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (post_loop_count & 7) * rem_g8], post_loop_sel_pattern[2 * (post_loop_count & 7) * rem_g8 + 1]));

  ae_int8x8* p_mat1_1 = p_mat1_0 + row_offset; //next 8th row
  ae_int8x8* p_mat1_2 = p_mat1_1 + row_offset; //next 8th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + row_offset; //next 8th row

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valignx2 alignx2_p_vec_0 = AE_LA128_PP(p_vec_0);
  ae_valignx2 alignx2_p_vec_1 = AE_LA128_PP(p_vec_1);
  ae_valignx2 alignx2_p_vec_2 = AE_LA128_PP(p_vec_2);
  ae_valignx2 alignx2_p_vec_3 = AE_LA128_PP(p_vec_3);

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

  AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, alignx2_p_vec_0, (ae_int8x16 *)p_vec_0);
  AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, alignx2_p_vec_1, (ae_int8x16 *)p_vec_1);
  AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, alignx2_p_vec_2, (ae_int8x16 *)p_vec_2);
  AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, alignx2_p_vec_3, (ae_int8x16 *)p_vec_3);

  if(align_offset)
  {
    mat1_row0_0 = AE_SEL8X8(mat1_row0_0, mat_bias, pre_sel1);
    mat1_row1_0 = AE_SEL8X8(mat1_row1_0, mat_bias, pre_sel1);
    mat1_row2_0 = AE_SEL8X8(mat1_row2_0, mat_bias, pre_sel1);
    mat1_row3_0 = AE_SEL8X8(mat1_row3_0, mat_bias, pre_sel1);
  }

  if(mask_start_end)
  {
      MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
      MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
      MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
      MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);

      MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
      MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1, mat1_zb, vec1_zb);
      MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1, mat1_zb, vec1_zb);
      MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1, mat1_zb, vec1_zb);
  }

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

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, alignx2_p_vec_0, (ae_int8x16 *)p_vec_0);
    AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, alignx2_p_vec_1, (ae_int8x16 *)p_vec_1);
    AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, alignx2_p_vec_2, (ae_int8x16 *)p_vec_2);
    AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, alignx2_p_vec_3, (ae_int8x16 *)p_vec_3);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1, mat1_zb, vec1_zb);
  }

  //Remainder loop for cols
  c_itr = 0;
  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign align_p_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign align_p_vec_3 = AE_LA64_PP(p_vec_3);

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

    mat1_row0_0 = AE_SEL8X8(mat1_row0_0, mat_bias, sel1);
    mat1_row1_0 = AE_SEL8X8(mat1_row1_0, mat_bias, sel1);
    mat1_row2_0 = AE_SEL8X8(mat1_row2_0, mat_bias, sel1);
    mat1_row3_0 = AE_SEL8X8(mat1_row3_0, mat_bias, sel1);
    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
    MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);

    c_itr += 8;
    sel1 = sel2;
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
#ifndef AE_MULAZB8Q8X8
    ,WORD32      mat1_zb
#endif
    ,WORD32      vec1_zb
    )
{
  int c_itr = 0;

  ae_int8x8 vec_bias = AE_MOVDA8((WORD8)vec1_zb);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols & 7)], post_loop_sel_pattern[2 * (cols & 7) + 1]));

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

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, vec_bias, sel1);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
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
    ,WORD32      mat1_zb
#ifndef AE_MULAZB8Q8X8
    ,WORD32      vec1_zb
#endif
    )
{
    int c_itr = 0;

    ae_int8x8 mat_bias = AE_MOVDA8((WORD8)mat1_zb);
    int rem_cols = cols & 15;
    int rem_g8 = ((rem_cols & 15) > 8)?1:0;
    ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8 + 1])); \
    ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8 + 1])); \

    ae_int8x8 mat1_row0_0, mat1_row0_1;
    ae_int8x8 mat1_row1_0, mat1_row1_1;
    ae_int8x8 mat1_row2_0, mat1_row2_1;
    ae_int8x8 mat1_row3_0, mat1_row3_1;

    ae_int8x8 vec0_batch_0, vec0_batch_1;
    ae_int8x8 vec1_batch_0, vec1_batch_1;
    ae_int8x8 vec2_batch_0, vec2_batch_1;
    ae_int8x8 vec3_batch_0, vec3_batch_1;
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

    int cols_count = cols -(cols & 15);
#pragma no_unroll
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
    {
        AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
        AE_LA8X8_IP(vec0_batch_1, align_p_vec_0, (ae_int8x8 *)p_vec_0);
        AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
        AE_LA8X8_IP(vec1_batch_1, align_p_vec_1, (ae_int8x8 *)p_vec_1);
        AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
        AE_LA8X8_IP(vec2_batch_1, align_p_vec_2, (ae_int8x8 *)p_vec_2);
        AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);
        AE_LA8X8_IP(vec3_batch_1, align_p_vec_3, (ae_int8x8 *)p_vec_3);

        AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
        AE_SW_LA8X8_IP(mat1_row0_1, align_p_mat1_0, p_mat1_0);
        AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
        AE_SW_LA8X8_IP(mat1_row1_1, align_p_mat1_1, p_mat1_1);
        AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
        AE_SW_LA8X8_IP(mat1_row2_1, align_p_mat1_2, p_mat1_2);
        AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);
        AE_SW_LA8X8_IP(mat1_row3_1, align_p_mat1_3, p_mat1_3);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1, mat1_zb, vec1_zb);
    }

    //Remainder loop for cols
    c_itr <<= 4;
    while(c_itr < cols)
    {
        AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
        AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
        AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
        AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

        AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
        AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
        AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
        AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);

        mat1_row0_0 = AE_SEL8X8(mat1_row0_0, mat_bias, sel1);
        mat1_row1_0 = AE_SEL8X8(mat1_row1_0, mat_bias, sel1);
        mat1_row2_0 = AE_SEL8X8(mat1_row2_0, mat_bias, sel1);
        mat1_row3_0 = AE_SEL8X8(mat1_row3_0, mat_bias, sel1);

        MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0, mat1_zb, vec1_zb);
        c_itr += 8;
        sel1 = sel2;
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
#ifndef AE_MULAZB8Q8X8
    ,WORD32      mat1_zb
#endif
    ,WORD32      vec1_zb
    )
{
  int c_itr = 0;

  ae_int8x8 vec_bias = AE_MOVDA8((WORD8)vec1_zb);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols & 7)], post_loop_sel_pattern[2 * (cols & 7) + 1]));

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

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, vec_bias, sel1);

    MAT_VEC_MAC(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, mat1_zb, vec1_zb);
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
#ifndef AE_MULAZB8Q8X8
    ,WORD32      mat1_zb
#endif
    ,WORD32      vec1_zb
    )
{
    int c_itr = 0;
    ae_int8x8 vec0_batch_0, vec0_batch_1;
    ae_int8x8 mat1_row0_0, mat1_row0_1;

    ae_int32x2 acc_row0_vec0 = *out_0_0;
    ae_int32x2 acc_row0_vec1 = *out_1_0;

    ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
    ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

    int rem_cols = (cols1 & 15);
    int rem_g8 = (rem_cols > 8)?1:0;

    ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * !rem_g8 + 1]));
    ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8], post_loop_sel_pattern[2 * (rem_cols & 7) * rem_g8 + 1]));

    ae_int8x8 vec_bias = AE_MOVDA8((WORD8)vec1_zb);
    int cols_count = cols1 - (cols1 & 15);

    for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
    {
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int8x16 *)p_vec_0);

        MAT_VEC_MAC(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0, mat1_zb, vec1_zb);
        MAT_VEC_MAC(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, vec0_batch_1, mat1_zb, vec1_zb);
    }

    //Remainder loop for cols1
    if(cols_count!=cols1)
    {
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int8x16 *)p_vec_0);

        vec0_batch_0 = AE_SEL8X8(vec0_batch_0, vec_bias, sel1);

        MAT_VEC_MAC(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0, mat1_zb, vec1_zb);

        if(rem_g8)
        {
            vec0_batch_1 = AE_SEL8X8(vec0_batch_1, vec_bias, sel2);
            MAT_VEC_MAC(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 ,vec0_batch_1, mat1_zb, vec1_zb);
        }
    }

    *out_0_0 = acc_row0_vec0;
    *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matmul_asym8sxasym8s_asym8s(
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
    WORD32 out_zero_bias)
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

  /* Iterators used in for loops */
  int m_itr, vec_itr;
  int ii;

  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  /*Load AE_BIASV8 and AE_BIASC8 state registers with mat1 and vec1 zero bias values*/
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec1_zero_bias, -mat1_zero_bias));
  ae_int64 biascv1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-mat1_zero_bias, -vec1_zero_bias));

  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

  /* Special case for cols multiple of 32 */
  if(((cols1 & 31) == 0) &&
      (row_stride1 == cols1) &&
      (vec_offset == cols1) &&
      ALIGNED_PTR(p_mat1, 16) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&
      ((out_offset & 0x3) == 0) &&
      ((rows & 0x3) == 0)
    )
  {
    AE_MOVZBVCDR(biasvc1);

    special_function_for_cols_mul_32
      (p_out,
       p_mat1,
       p_vec1,
       p_bias,
       rows,
       vec_count,
       cols1,
#ifndef AE_MULAZB8Q8X8
       -mat1_zero_bias,
       -vec1_zero_bias,
#endif
       out_multiplier,
       left_shift,
       right_shift,
       out_zero_bias,
       out_offset
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
    ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
    ae_int32x2 acc_row1_vec0, acc_row1_vec1, acc_row1_vec2, acc_row1_vec3;

    for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();

        if(p_bias)
        {
          bias_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
          bias_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
        }

        acc_row0_vec0 = bias_01;
        acc_row1_vec0 = bias_23;
        acc_row0_vec1 = bias_01;
        acc_row1_vec1 = bias_23;
        acc_row0_vec2 = bias_01;
        acc_row1_vec2 = bias_23;
        acc_row0_vec3 = bias_01;
        acc_row1_vec3 = bias_23;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        AE_MOVZBVCDR(biasvc1);
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
           ,-mat1_zero_bias
#ifndef AE_MULAZB8Q8X8
           ,-vec1_zero_bias
#endif
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier, left_shift, right_shift, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0, temp_vec1;
        temp_vec0 = AE_SAT8X8X16(out_0, out_1);
        temp_vec1 = AE_SAT8X8X16(out_2, out_3);

        AE_SW_S8_7_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_6_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_5_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);

        AE_SW_S8_7_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_6_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_5_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_4_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_3_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
        AE_SW_S8_2_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
        AE_SW_S8_1_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
        AE_S8_0_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
      }

      // Remaining vectors
      for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
      {
        ae_int32x2 bias_0 = AE_ZERO32();
        if(p_bias)
        {
          bias_0 = AE_MOVDA32(p_bias[m_itr]);
        }

        acc_row0_vec0 = bias_0;
        acc_row0_vec1 = bias_0;

        ae_int8x8* p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr + 0) * row_stride1];

        AE_MOVZBVCDR(biascv1);
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_vec_0
           ,p_mat1_0
           ,cols1
           ,vec_offset
#ifndef AE_MULAZB8Q8X8
           ,-vec1_zero_bias
#endif
           ,-mat1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row0_vec1, out_multiplier, left_shift, right_shift, out_zero_bias);
        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0;
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining rows
    for (vec_itr = (vec_count & (~3)); vec_itr < vec_count; vec_itr++)
    {
      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();

        if(p_bias)
        {
          bias_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
          bias_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
        }

        acc_row0_vec0 = bias_01;
        acc_row1_vec0 = bias_23;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
#ifndef AE_MULAZB8Q8X8
           ,-mat1_zero_bias
#endif
           ,-vec1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0;
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }

      // Remaining vectors
      for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
      {
        ae_int32x2 bias_0 = AE_ZERO32();
        if(p_bias)
        {
          bias_0 = AE_MOVDA32(p_bias[m_itr]);
        }

        acc_row0_vec0 = bias_0;
        acc_row0_vec1 = bias_0;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
#ifndef AE_MULAZB8Q8X8
           ,-mat1_zero_bias
#endif
           ,-vec1_zero_bias
          );

        ae_int8x8 temp_vec0;
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }
  else if (p_mat1 && p_vec1)
  {
    ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
    ae_int32x2 acc_row1_vec0, acc_row1_vec1, acc_row1_vec2, acc_row1_vec3;
    // for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
    for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;
      // for(ii = 0; ii < 8; ii++)
      for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
      {
        // for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
        for(ii = 0; ii < 8; ii++)
        {
          WORD8* p_dst_0_ii = p_dst_0 + (m_itr + ii) * out_stride;
          WORD8* p_dst_1_ii = p_dst_1 + (m_itr + ii) * out_stride;
          WORD8* p_dst_2_ii = p_dst_2 + (m_itr + ii) * out_stride;
          WORD8* p_dst_3_ii = p_dst_3 + (m_itr + ii) * out_stride;

          ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
          if(p_bias)
          {
            bias_01 = AE_MOVDA32X2(p_bias[ 0 + ii + m_itr], p_bias[ 8 + ii + m_itr]);
            bias_23 = AE_MOVDA32X2(p_bias[16 + ii + m_itr], p_bias[24 + ii + m_itr]);
          }

          acc_row0_vec0 = bias_01;
          acc_row1_vec0 = bias_23;
          acc_row0_vec1 = bias_01;
          acc_row1_vec1 = bias_23;
          acc_row0_vec2 = bias_01;
          acc_row1_vec2 = bias_23;
          acc_row0_vec3 = bias_01;
          acc_row1_vec3 = bias_23;

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1];

          AE_MOVZBVCDR(biasvc1);
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
             ,-mat1_zero_bias
#ifndef AE_MULAZB8Q8X8
             ,-vec1_zero_bias
#endif
            );

          ae_int16x4 out_0, out_1, out_2, out_3;

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          ae_int8x8 temp_vec0, temp_vec1;
          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          AE_SW_S8_7_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_SW_S8_6_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_SW_S8_5_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_SW_S8_4_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_1_ii, 8 * out_stride);
          AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1_ii, 8 * out_stride);
          AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_1_ii, 8 * out_stride);
          AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_1_ii, 8 * out_stride);

          AE_SW_S8_7_XP(temp_vec1, (ae_int8 *) p_dst_2_ii, 8 * out_stride);
          AE_SW_S8_6_XP(temp_vec1, (ae_int8 *) p_dst_2_ii, 8 * out_stride);
          AE_SW_S8_5_XP(temp_vec1, (ae_int8 *) p_dst_2_ii, 8 * out_stride);
          AE_SW_S8_4_XP(temp_vec1, (ae_int8 *) p_dst_2_ii, 8 * out_stride);
          AE_SW_S8_3_XP(temp_vec1, (ae_int8 *) p_dst_3_ii, 8 * out_stride);
          AE_SW_S8_2_XP(temp_vec1, (ae_int8 *) p_dst_3_ii, 8 * out_stride);
          AE_SW_S8_1_XP(temp_vec1, (ae_int8 *) p_dst_3_ii, 8 * out_stride);
          AE_S8_0_XP(temp_vec1, (ae_int8 *) p_dst_3_ii, 8 * out_stride);
        }
      }
      p_dst_0 = p_dst_0 + (rows & (~31)) * out_stride;
      p_dst_1 = p_dst_1 + (rows & (~31)) * out_stride;
      p_dst_2 = p_dst_2 + (rows & (~31)) * out_stride;
      p_dst_3 = p_dst_3 + (rows & (~31)) * out_stride;

      // Remaining vectors
      // for (; vec_itr < vec_count; vec_itr++)
      for(m_itr = (rows & (~31)); m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
        if(p_bias)
        {
          bias_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
          bias_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
        }

        acc_row0_vec0 = bias_01;
        acc_row1_vec0 = bias_23;
        acc_row0_vec1 = bias_01;
        acc_row1_vec1 = bias_23;
        acc_row0_vec2 = bias_01;
        acc_row1_vec2 = bias_23;
        acc_row0_vec3 = bias_01;
        acc_row1_vec3 = bias_23;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8* p_mat1_0 = (ae_int8x8*) &p_mat1[(m_itr)* row_stride1];

        AE_MOVZBVCDR(biasvc1);
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
           ,-mat1_zero_bias
#ifndef AE_MULAZB8Q8X8
           ,-vec1_zero_bias
#endif
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier, left_shift, right_shift, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0, temp_vec1;
        temp_vec0 = AE_SAT8X8X16(out_0, out_1);
        temp_vec1 = AE_SAT8X8X16(out_2, out_3);

        AE_SW_S8_7_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_6_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_5_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);

        AE_SW_S8_7_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_6_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_5_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_4_XP(temp_vec1, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_3_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
        AE_SW_S8_2_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
        AE_SW_S8_1_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
        AE_S8_0_XP(temp_vec1, (ae_int8 *) p_dst_3, out_stride);
      }

      // remaining rows
      for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
      {
        ae_int32x2 bias_0 = AE_ZERO32();
        if(p_bias)
        {
          bias_0 = AE_MOVDA32(p_bias[m_itr]);
        }
        acc_row0_vec0 = bias_0;
        acc_row0_vec1 = bias_0;
        ae_int8x8* p_vec_0  = (ae_int8x8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8 *p_mat1_0 = (ae_int8*) &p_mat1[m_itr * row_stride1];

        AE_MOVZBVCDR(biascv1);
        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_vec_0
           ,p_mat1_0
           ,cols1
           ,vec_offset
#ifndef AE_MULAZB8Q8X8
           ,-vec1_zero_bias
#endif
           ,-mat1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row0_vec1, out_multiplier, left_shift, right_shift, out_zero_bias);
        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0;
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_3, out_stride);
      }
    }

    for (vec_itr = (vec_count & (~3)); vec_itr < (vec_count); vec_itr++)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      // for(ii = 0; ii < 8; ii++)
      for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
      {
        // for (vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
        for(ii = 0; ii < 8; ii++)
        {
          WORD8* p_dst_0_ii = p_dst_0 + (m_itr + ii) * out_stride;

          ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
          if(p_bias)
          {
            bias_01 = AE_MOVDA32X2(p_bias[ 0 + ii + m_itr], p_bias[ 8 + ii + m_itr]);
            bias_23 = AE_MOVDA32X2(p_bias[16 + ii + m_itr], p_bias[24 + ii + m_itr]);
          }

          acc_row0_vec0 = bias_01;
          acc_row1_vec0 = bias_23;

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1];

          AE_MOVZBVCDR(biasvc1);
          _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,p_mat1_0
             ,p_vec_0
             ,cols1
             ,row_stride1
#ifndef AE_MULAZB8Q8X8
             ,-mat1_zero_bias
#endif
             ,-vec1_zero_bias
            );

          ae_int16x4 out_0;

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

          ae_int8x8 temp_vec0;
          temp_vec0 = AE_SAT8X8X16(out_0, out_0);

          AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
          AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_0_ii, 8 * out_stride);
        }
      }
      p_dst_0 = p_dst_0 + (rows & (~31)) * out_stride;

      // Remaining vectors
      // for (; vec_itr < vec_count; vec_itr++)
      for(m_itr = (rows & (~31)); m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 bias_01 = AE_ZERO32(), bias_23 = AE_ZERO32();
        if(p_bias)
        {
          bias_01 = AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]);
          bias_23 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
        }

        acc_row0_vec0 = bias_01;
        acc_row1_vec0 = bias_23;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8* p_mat1_0 = (ae_int8x8*) &p_mat1[(m_itr)* row_stride1];

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
#ifndef AE_MULAZB8Q8X8
           ,-mat1_zero_bias
#endif
           ,-vec1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0;
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
      }

      // remaining rows
      for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
      {
        ae_int32x2 bias_0 = AE_ZERO32();
        if(p_bias)
        {
          bias_0 = AE_MOVDA32(p_bias[m_itr]);
        }
        acc_row0_vec0 = bias_0;
        acc_row0_vec1 = bias_0;
        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*) &p_mat1[m_itr * row_stride1];

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
#ifndef AE_MULAZB8Q8X8
           ,-mat1_zero_bias
#endif
           ,-vec1_zero_bias
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        ae_int8x8 temp_vec0;
        temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
      }
    }
  }
  else
  {
    return -1;
  }
  return 0;
}
