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

#define MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out0, inp0, inp1, mult01, l_shift01) \
{ \
  ae_int32x2 d_red_mult01 = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(mult01, mult01)); \
  ae_int32x2 d_red_mult01_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(mult01, mult01)); \
  ae_int32x2 d_inp01_h = AE_ROUND32X2F64SASYM(inp0, inp1); \
  ae_int64 q0_l, q1_l; \
  AE_MUL32X2S_HH_LL(q0_l, q1_l, d_red_mult01, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(inp0), AE_MOVINT32X2_FROMINT64(inp1))); \
  AE_MULAFP32X2S_HH_LL(q0_l, q1_l, d_red_mult01_l16, AE_SLAI32(d_inp01_h, 15)); \
  q0_l = AE_SLAA64(q0_l, (AE_MOVAD32_H(l_shift01)+17)); \
  q1_l = AE_SLAA64(q1_l, (AE_MOVAD32_L(l_shift01)+17)); \
  out0 = AE_ROUND32X2F64SASYM(q0_l, q1_l); \
}

#define MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out0, inp0, inp1, mult, l_shift) \
{ \
  ae_int32x2 d_red_mult = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult))); \
  ae_int32x2 d_red_mult_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult)));  \
  ae_int32x2 d_inp01_h = AE_ROUND32X2F64SASYM(inp0, inp1); \
  ae_int64 q0_l, q1_l; \
  AE_MUL32X2S_HH_LL(q0_l, q1_l, d_red_mult, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(inp0), AE_MOVINT32X2_FROMINT64(inp1))); \
  AE_MULAFP32X2S_HH_LL(q0_l, q1_l, d_red_mult_l16, AE_SLAI32(d_inp01_h, 15)); \
  q0_l = AE_SLAA64(q0_l, (l_shift + 17)); \
  q1_l = AE_SLAA64(q1_l, (l_shift + 17)); \
  out0 = AE_ROUND32X2F64SASYM(q0_l, q1_l); \
}

extern const long long g_sel_pattern[16];
extern const long long pre_loop_sel_pattern[16];
extern const long long post_loop_sel_pattern[16];

static void special_function_for_cols_mul_32
    (WORD16*       p_out_0
    ,const WORD8*  p_mat1
    ,const WORD16* p_vec1_0
    ,const WORD64* p_bias_0
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
    )
{
  ae_int16x8 * __restrict__ p_vec_0;
  int c_itr;
  int m_itr = 0, vec_itr = 0;
  ae_int64 acc_buffer[4];
  ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3;

  ae_int16x4 vec0_batch_4, vec0_batch_5, vec0_batch_6, vec0_batch_7;
  ae_int16x4 vec1_batch_4, vec1_batch_5, vec1_batch_6, vec1_batch_7;

  ae_int8x8 *p_mat1_0, *p_mat1_1, *p_mat1_2, *p_mat1_3;

  ae_valignx2 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2, align_p_mat1_3;

  for(m_itr = 0; m_itr < (n_rows & ~(4 - 1)); m_itr += 4)
  {
    ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
    ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();

    if(p_bias_0)
    {
      d_bias0 = *(ae_int64 *)&p_bias_0[m_itr + 0];
      d_bias1 = *(ae_int64 *)&p_bias_0[m_itr + 1];
      d_bias2 = *(ae_int64 *)&p_bias_0[m_itr + 2];
      d_bias3 = *(ae_int64 *)&p_bias_0[m_itr + 3];
    }
    AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
    AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

    ae_int16x4 *p_ae_dst_0 = (ae_int16x4 *)((WORD16 *)p_out_0 + (m_itr + 0) * out_stride);

    ae_int32x2 l_mult_01, l_mult_23;
    l_mult_01 = AE_MOVDA32X2(p_out_shift[m_itr + 0], p_out_shift[m_itr + 1]);
    l_mult_23 = AE_MOVDA32X2(p_out_shift[m_itr + 2], p_out_shift[m_itr + 3]);

    ae_int32x2 out_multiplier_01, out_multiplier_23;
    ae_valignx2 align_p_mult = AE_LA128_PP((ae_int32x4 *)p_out_mul);
    AE_LA32X2X2_IP(out_multiplier_01, out_multiplier_23, align_p_mult, (ae_int32x4 *)p_out_mul);

#pragma loop_count min=1
    for (vec_itr = 0; vec_itr < n_vecs; vec_itr += 2)
    {
      ae_int32x2 acc_row0_vec0;
      ae_int32x2 acc_row1_vec0;
      ae_int32x2 acc_row0_vec1;
      ae_int32x2 acc_row1_vec1;

      ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;
      ae_int64 acc64_01, acc64_11, acc64_21, acc64_31;

      /* Initialize accumulators */
      /* This unusual sequence of I and IP is written to prevent compiler from converting them to mov */
      AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
      AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);
      AE_L64X2_IP(acc64_01, acc64_11, acc_buff, 0);
      AE_L64X2_I(acc64_21, acc64_31, acc_buff, 16);

      p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
      p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset);
      p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
      p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

      align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      p_vec_0  = (ae_int16x8 *)(p_vec1_0 + vec_itr * vec_offset);
      WORD32 next_vec_offset = vec_itr == n_vecs - 1 ? 0 : (vec_offset << 1);
      WORD32 next_dst_offset = vec_itr == n_vecs - 1 ? 0 : out_offset;

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
        AE_L16X4X2_X(vec1_batch_0, vec1_batch_1, p_vec_0, next_vec_offset);
        AE_L16X4X2_X(vec1_batch_2, vec1_batch_3, p_vec_0, next_vec_offset+16);
        AE_L16X4X2_X(vec1_batch_4, vec1_batch_5, p_vec_0, next_vec_offset+32);
        AE_L16X4X2_X(vec1_batch_6, vec1_batch_7, p_vec_0, next_vec_offset+48);

        AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_4, vec0_batch_5, p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_6, vec0_batch_7, p_vec_0, 16);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
        AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
        AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_2, vec1_batch_3);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
        AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec1_batch_4, vec1_batch_5);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);
        AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec1_batch_6, vec1_batch_7);
      }

      /* Apply quantization */
      ae_int16x4 out_0, out_1;

      MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);
      MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec1, acc64_01, acc64_11, out_multiplier_01, l_mult_01);

      MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);
      MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec1, acc64_21, acc64_31, out_multiplier_23, l_mult_23);

      out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
      out_1 = AE_SAT16X4(acc_row0_vec1, acc_row1_vec1);

      /* Store output */
      *p_ae_dst_0 = out_0;
      p_ae_dst_0 += (next_dst_offset >> 2);
      *p_ae_dst_0 = out_1;
      p_ae_dst_0 += (next_dst_offset >> 2);
    }
  }
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_aligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int64* out_20
    ,ae_int64* out_30
    ,ae_int64* out_01
    ,ae_int64* out_11
    ,ae_int64* out_21
    ,ae_int64* out_31
    ,ae_int64* out_02
    ,ae_int64* out_12
    ,ae_int64* out_22
    ,ae_int64* out_32
    ,ae_int64* out_03
    ,ae_int64* out_13
    ,ae_int64* out_23
    ,ae_int64* out_33
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;
  int rem_cols = cols & 15;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3;
  ae_int16x4 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3;
  ae_int16x4 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;

  ae_int8x16 *p_mat1_1 = (ae_int8x16 *)((ae_int8 *)p_mat1_0 + row_offset);
  ae_int8x16 *p_mat1_2 = (ae_int8x16 *)((ae_int8 *)p_mat1_1 + row_offset);
  ae_int8x16 *p_mat1_3 = (ae_int8x16 *)((ae_int8 *)p_mat1_2 + row_offset);

  ae_int16x8 *p_vec_1 = (ae_int16x8 *)((WORD16 *)p_vec_0 + vec_offset);
  ae_int16x8 *p_vec_2 = (ae_int16x8 *)((WORD16 *)p_vec_1 + vec_offset);
  ae_int16x8 *p_vec_3 = (ae_int16x8 *)((WORD16 *)p_vec_2 + vec_offset);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;
  ae_int64 acc64_20 = *out_20;
  ae_int64 acc64_30 = *out_30;

  ae_int64 acc64_01 = *out_01;
  ae_int64 acc64_11 = *out_11;
  ae_int64 acc64_21 = *out_21;
  ae_int64 acc64_31 = *out_31;

  ae_int64 acc64_02 = *out_02;
  ae_int64 acc64_12 = *out_12;
  ae_int64 acc64_22 = *out_22;
  ae_int64 acc64_32 = *out_32;

  ae_int64 acc64_03 = *out_03;
  ae_int64 acc64_13 = *out_13;
  ae_int64 acc64_23 = *out_23;
  ae_int64 acc64_33 = *out_33;

  for(c_itr = 0; c_itr < cols>>4; c_itr++)
  {
    AE_L16X4X2_I(vec0_batch_2, vec0_batch_3, p_vec_0, 16);
    AE_L16X4X2_I(vec1_batch_2, vec1_batch_3, p_vec_1, 16);
    AE_L16X4X2_I(vec2_batch_2, vec2_batch_3, p_vec_2, 16);
    AE_L16X4X2_I(vec3_batch_2, vec3_batch_3, p_vec_3, 16);

    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 32);
    AE_L16X4X2_IP(vec1_batch_0, vec1_batch_1, p_vec_1, 32);
    AE_L16X4X2_IP(vec2_batch_0, vec2_batch_1, p_vec_2, 32);
    AE_L16X4X2_IP(vec3_batch_0, vec3_batch_1, p_vec_3, 32);

    mat1_row0_1 = AE_L8X8_I((ae_int8x8 *)p_mat1_0, 8);
    mat1_row1_1 = AE_L8X8_I((ae_int8x8 *)p_mat1_1, 8);
    mat1_row2_1 = AE_L8X8_I((ae_int8x8 *)p_mat1_2, 8);
    mat1_row3_1 = AE_L8X8_I((ae_int8x8 *)p_mat1_3, 8);

    AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 16);
    AE_L8X8_IP(mat1_row1_0, (ae_int8x8 *)p_mat1_1, 16);
    AE_L8X8_IP(mat1_row2_0, (ae_int8x8 *)p_mat1_2, 16);
    AE_L8X8_IP(mat1_row3_0, (ae_int8x8 *)p_mat1_3, 16);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_2, vec1_batch_3);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_2, vec2_batch_3);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_2, vec3_batch_3);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    int rem_cols0, rem_cols1;
    ae_valignx2 a0, a1, a2, a3;
    a0 = AE_LA128_PP(p_mat1_0);
    a1 = AE_LA128_PP(p_mat1_1);
    a2 = AE_LA128_PP(p_mat1_2);
    a3 = AE_LA128_PP(p_mat1_3);

    rem_cols0 = (rem_cols > 8 ? 8 : rem_cols) << 1;
    rem_cols1 = ((rem_cols - 8) < 0 ? 0 : (rem_cols - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, a0, (ae_int8x16 *)p_mat1_0, rem_cols);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, a1, (ae_int8x16 *)p_mat1_1, rem_cols);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, a2, (ae_int8x16 *)p_mat1_2, rem_cols);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, a3, (ae_int8x16 *)p_mat1_3, rem_cols);

    a0 = AE_LA128_PP(p_vec_0);
    a1 = AE_LA128_PP(p_vec_1);
    a2 = AE_LA128_PP(p_vec_2);
    a3 = AE_LA128_PP(p_vec_3);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_0, a0, (ae_int16x8 *)p_vec_0, rem_cols0);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_0, a1, (ae_int16x8 *)p_vec_1, rem_cols0);
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_0, a2, (ae_int16x8 *)p_vec_2, rem_cols0);
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_0, a3, (ae_int16x8 *)p_vec_3, rem_cols0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, a0, (ae_int16x8 *)p_vec_0, rem_cols1);
    AE_LAV16X4X2_XP(vec1_batch_2, vec1_batch_3, a1, (ae_int16x8 *)p_vec_1, rem_cols1);
    AE_LAV16X4X2_XP(vec2_batch_2, vec2_batch_3, a2, (ae_int16x8 *)p_vec_2, rem_cols1);
    AE_LAV16X4X2_XP(vec3_batch_2, vec3_batch_3, a3, (ae_int16x8 *)p_vec_3, rem_cols1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_2, vec1_batch_3);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_2, vec2_batch_3);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_2, vec3_batch_3);
  }

  *out_00 = acc64_00;
  *out_10 = acc64_10;
  *out_20 = acc64_20;
  *out_30 = acc64_30;

  *out_01 = acc64_01;
  *out_11 = acc64_11;
  *out_21 = acc64_21;
  *out_31 = acc64_31;

  *out_02 = acc64_02;
  *out_12 = acc64_12;
  *out_22 = acc64_22;
  *out_32 = acc64_32;

  *out_03 = acc64_03;
  *out_13 = acc64_13;
  *out_23 = acc64_23;
  *out_33 = acc64_33;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int64* out_20
    ,ae_int64* out_30
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int c_itr = 0;
  int rem_cols = cols & 15;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;

  ae_int8x16 *p_mat1_1 = (ae_int8x16 *)((WORD8 *)p_mat1_0 + row_offset);
  ae_int8x16 *p_mat1_2 = (ae_int8x16 *)((WORD8 *)p_mat1_1 + row_offset);
  ae_int8x16 *p_mat1_3 = (ae_int8x16 *)((WORD8 *)p_mat1_2 + row_offset);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;
  ae_int64 acc64_20 = *out_20;
  ae_int64 acc64_30 = *out_30;

  for(c_itr = 0; c_itr < cols>>4; c_itr++)
  {
    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);

    AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat1_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat1_3, 16);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    int rem_cols0, rem_cols1;
    ae_valignx2 a0, a1, a2, a3;
    a0 = AE_LA128_PP(p_mat1_0);
    a1 = AE_LA128_PP(p_mat1_1);
    a2 = AE_LA128_PP(p_mat1_2);
    a3 = AE_LA128_PP(p_mat1_3);

    rem_cols0 = (rem_cols > 8 ? 8 : rem_cols) << 1;
    rem_cols1 = ((rem_cols - 8) < 0 ? 0 : (rem_cols - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, a0, p_mat1_0, rem_cols);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, a1, p_mat1_1, rem_cols);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, a2, p_mat1_2, rem_cols);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, a3, p_mat1_3, rem_cols);

    a0 = AE_LA128_PP((ae_int16x8 *)p_vec_0);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_0, a0, (ae_int16x8 *)p_vec_0, rem_cols0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, a0, (ae_int16x8 *)p_vec_0, rem_cols1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
  }

  *out_00 = acc64_00;
  *out_10 = acc64_10;
  *out_20 = acc64_20;
  *out_30 = acc64_30;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int64* out_00
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;
  int rem_cols = cols & 15;

  ae_int8x8 mat1_row0_0, mat1_row0_1;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = AE_ZERO64();

  for(c_itr = 0; c_itr < cols>>4; c_itr++)
  {
    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);

    AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat1_0, 16);

    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_2, vec0_batch_3, mat1_row0_1);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    int rem_cols0, rem_cols1;
    ae_valignx2 a0, a1;
    a0 = AE_LA128_PP(p_mat1_0);

    rem_cols0 = (rem_cols > 8 ? 8 : rem_cols) << 1;
    rem_cols1 = ((rem_cols - 8) < 0 ? 0 : (rem_cols - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, a0, p_mat1_0, rem_cols);

    a1 = AE_LA128_PP(p_vec_0);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_0, a1, p_vec_0, rem_cols0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, a1, p_vec_0, rem_cols1);

    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_2, vec0_batch_3, mat1_row0_1);
  }
  acc64_00 = AE_ADD64(acc64_00, acc64_10);

  *out_00 = acc64_00;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int64* out_20
    ,ae_int64* out_30
    ,ae_int64* out_01
    ,ae_int64* out_11
    ,ae_int64* out_21
    ,ae_int64* out_31
    ,ae_int64* out_02
    ,ae_int64* out_12
    ,ae_int64* out_22
    ,ae_int64* out_32
    ,ae_int64* out_03
    ,ae_int64* out_13
    ,ae_int64* out_23
    ,ae_int64* out_33
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{

  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3;
  ae_int16x4 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3;
  ae_int16x4 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  int align_offset = ((unsigned int)p_mat1_0 & 15);
  pre_loop_count = align_offset != 0 ? 16 - align_offset : 0;
  pre_loop_count = (cols < pre_loop_count) ? cols : pre_loop_count;

  loop_count = cols - pre_loop_count;
  post_loop_count = (loop_count & 15);

  ae_int8x16 *p_mat1_1 = (ae_int8x16 *)((WORD8 *)p_mat1_0 + 8 * row_offset); //next 8th row
  ae_int8x16 *p_mat1_2 = (ae_int8x16 *)((WORD8 *)p_mat1_1 + 8 * row_offset); //next 8th row
  ae_int8x16 *p_mat1_3 = (ae_int8x16 *)((WORD8 *)p_mat1_2 + 8 * row_offset); //next 8th row

  ae_int16x8 *p_vec_1 = (ae_int16x8 *)((WORD16 *)p_vec_0 + vec_offset);
  ae_int16x8 *p_vec_2 = (ae_int16x8 *)((WORD16 *)p_vec_1 + vec_offset);
  ae_int16x8 *p_vec_3 = (ae_int16x8 *)((WORD16 *)p_vec_2 + vec_offset);

  ae_valignx2 align_0;
  ae_valignx2 align_1;
  ae_valignx2 align_2;
  ae_valignx2 align_3;

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;
  ae_int64 acc64_20 = *out_20;
  ae_int64 acc64_30 = *out_30;

  ae_int64 acc64_01 = *out_01;
  ae_int64 acc64_11 = *out_11;
  ae_int64 acc64_21 = *out_21;
  ae_int64 acc64_31 = *out_31;

  ae_int64 acc64_02 = *out_02;
  ae_int64 acc64_12 = *out_12;
  ae_int64 acc64_22 = *out_22;
  ae_int64 acc64_32 = *out_32;

  ae_int64 acc64_03 = *out_03;
  ae_int64 acc64_13 = *out_13;
  ae_int64 acc64_23 = *out_23;
  ae_int64 acc64_33 = *out_33;

  /* Pre loop computation */
  if(pre_loop_count)
  {
    int pre_lc0, pre_lc1;
    align_0 = AE_LA128_PP(p_mat1_0);
    align_1 = AE_LA128_PP(p_mat1_1);
    align_2 = AE_LA128_PP(p_mat1_2);
    align_3 = AE_LA128_PP(p_mat1_3);

    pre_lc0 = (pre_loop_count > 8 ? 8 : pre_loop_count) << 1;
    pre_lc1 = ((pre_loop_count - 8 < 0) ? 0 : (pre_loop_count - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_0, p_mat1_0, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, align_1, p_mat1_1, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, align_2, p_mat1_2, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, align_3, p_mat1_3, pre_loop_count);

    align_0 = AE_LA128_PP(p_vec_0);
    align_1 = AE_LA128_PP(p_vec_1);
    align_2 = AE_LA128_PP(p_vec_2);
    align_3 = AE_LA128_PP(p_vec_3);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_0, p_vec_0, pre_lc0);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_1, p_vec_1, pre_lc0);
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_1, align_2, p_vec_2, pre_lc0);
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_1, align_3, p_vec_3, pre_lc0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_0, p_vec_0, pre_lc1);
    AE_LAV16X4X2_XP(vec1_batch_2, vec1_batch_3, align_1, p_vec_1, pre_lc1);
    AE_LAV16X4X2_XP(vec2_batch_2, vec2_batch_3, align_2, p_vec_2, pre_lc1);
    AE_LAV16X4X2_XP(vec3_batch_2, vec3_batch_3, align_3, p_vec_3, pre_lc1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_2, vec1_batch_3);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_2, vec2_batch_3);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_2, vec3_batch_3);
  }

  ae_valignx2 align_p_vec_0, align_p_vec_1, align_p_vec_2, align_p_vec_3;
  align_p_vec_0 = AE_LA128_PP(p_vec_0);
  align_p_vec_1 = AE_LA128_PP(p_vec_1);
  align_p_vec_2 = AE_LA128_PP(p_vec_2);
  align_p_vec_3 = AE_LA128_PP(p_vec_3);

#pragma no_unroll
  for(c_itr = 0; c_itr < (loop_count >> 4); c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);
    AE_L8X8_IP(mat1_row0_1, (ae_int8x8 *)p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, (ae_int8x8 *)p_mat1_1, 8);
    AE_L8X8_IP(mat1_row1_1, (ae_int8x8 *)p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, (ae_int8x8 *)p_mat1_2, 8);
    AE_L8X8_IP(mat1_row2_1, (ae_int8x8 *)p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, (ae_int8x8 *)p_mat1_3, 8);
    AE_L8X8_IP(mat1_row3_1, (ae_int8x8 *)p_mat1_3, 8);

    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1);
    AE_LA16X4X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2);
    AE_LA16X4X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3);

    AE_LA16X4X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_2, vec1_batch_3, align_p_vec_1, p_vec_1);
    AE_LA16X4X2_IP(vec2_batch_2, vec2_batch_3, align_p_vec_2, p_vec_2);
    AE_LA16X4X2_IP(vec3_batch_2, vec3_batch_3, align_p_vec_3, p_vec_3);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_2, vec1_batch_3);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_2, vec2_batch_3);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_2, vec3_batch_3);
  }

  if(post_loop_count)
  {
    int post_lc0, post_lc1;
    align_0 = AE_LA128_PP(p_mat1_0);
    align_1 = AE_LA128_PP(p_mat1_1);
    align_2 = AE_LA128_PP(p_mat1_2);
    align_3 = AE_LA128_PP(p_mat1_3);

    post_lc0 = (post_loop_count > 8 ? 8 : post_loop_count) << 1;
    post_lc1 = ((post_loop_count - 8 < 0) ? 0 : (post_loop_count - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_0, p_mat1_0, post_loop_count);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, align_1, p_mat1_1, post_loop_count);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, align_2, p_mat1_2, post_loop_count);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, align_3, p_mat1_3, post_loop_count);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, post_lc0);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1, post_lc0);
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2, post_lc0);
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3, post_lc0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec_0, p_vec_0, post_lc1);
    AE_LAV16X4X2_XP(vec1_batch_2, vec1_batch_3, align_p_vec_1, p_vec_1, post_lc1);
    AE_LAV16X4X2_XP(vec2_batch_2, vec2_batch_3, align_p_vec_2, p_vec_2, post_lc1);
    AE_LAV16X4X2_XP(vec3_batch_2, vec3_batch_3, align_p_vec_3, p_vec_3, post_lc1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_2, vec1_batch_3);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_2, vec2_batch_3);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_2, vec3_batch_3);
  }
  *out_00 = acc64_00;
  *out_10 = acc64_10;
  *out_20 = acc64_20;
  *out_30 = acc64_30;

  *out_01 = acc64_01;
  *out_11 = acc64_11;
  *out_21 = acc64_21;
  *out_31 = acc64_31;

  *out_02 = acc64_02;
  *out_12 = acc64_12;
  *out_22 = acc64_22;
  *out_32 = acc64_32;

  *out_03 = acc64_03;
  *out_13 = acc64_13;
  *out_23 = acc64_23;
  *out_33 = acc64_33;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int64* out_20
    ,ae_int64* out_30
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  int align_offset = ((unsigned int)p_mat1_0 & 15);
  pre_loop_count = align_offset != 0 ? 16 - align_offset : 0;
  pre_loop_count = (cols < pre_loop_count) ? cols : pre_loop_count;

  loop_count = cols - pre_loop_count;
  post_loop_count = (loop_count & 15);

  ae_int8x16* p_mat1_1 = (ae_int8x16 *)((WORD8 *)p_mat1_0 + 8 * row_offset); //next 8th row
  ae_int8x16* p_mat1_2 = (ae_int8x16 *)((WORD8 *)p_mat1_1 + 8 * row_offset); //next 8th row
  ae_int8x16* p_mat1_3 = (ae_int8x16 *)((WORD8 *)p_mat1_2 + 8 * row_offset); //next 8th row

  ae_valignx2 align_0;
  ae_valignx2 align_1;
  ae_valignx2 align_2;
  ae_valignx2 align_3;

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;
  ae_int64 acc64_20 = *out_20;
  ae_int64 acc64_30 = *out_30;

  /* Pre loop computation */
  if(pre_loop_count)
  {
    int pre_lc0, pre_lc1;
    align_0 = AE_LA128_PP(p_mat1_0);
    align_1 = AE_LA128_PP(p_mat1_1);
    align_2 = AE_LA128_PP(p_mat1_2);
    align_3 = AE_LA128_PP(p_mat1_3);

    pre_lc0 = (pre_loop_count > 8 ? 8 : pre_loop_count) << 1;
    pre_lc1 = ((pre_loop_count - 8 < 0) ? 0 : (pre_loop_count - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_0, p_mat1_0, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, align_1, p_mat1_1, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, align_2, p_mat1_2, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, align_3, p_mat1_3, pre_loop_count);

    align_0 = AE_LA128_PP(p_vec_0);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_0, p_vec_0, pre_lc0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_0, p_vec_0, pre_lc1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
  }

  ae_valignx2 align_p_vec_0;
  align_p_vec_0 = AE_LA128_PP(p_vec_0);

#pragma no_unroll
  for(c_itr = 0; c_itr < (loop_count >> 4); c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);
    AE_L8X8_IP(mat1_row0_1, (ae_int8x8 *)p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, (ae_int8x8 *)p_mat1_1, 8);
    AE_L8X8_IP(mat1_row1_1, (ae_int8x8 *)p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, (ae_int8x8 *)p_mat1_2, 8);
    AE_L8X8_IP(mat1_row2_1, (ae_int8x8 *)p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, (ae_int8x8 *)p_mat1_3, 8);
    AE_L8X8_IP(mat1_row3_1, (ae_int8x8 *)p_mat1_3, 8);

    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);

    AE_LA16X4X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec_0, p_vec_0);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
  }

  if(post_loop_count)
  {
    int post_lc0, post_lc1;
    align_0 = AE_LA128_PP(p_mat1_0);
    align_1 = AE_LA128_PP(p_mat1_1);
    align_2 = AE_LA128_PP(p_mat1_2);
    align_3 = AE_LA128_PP(p_mat1_3);

    post_lc0 = (post_loop_count > 8 ? 8 : post_loop_count) << 1;
    post_lc1 = ((post_loop_count - 8 < 0) ? 0 : (post_loop_count - 8)) << 1;

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_0, p_mat1_0, post_loop_count);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, align_1, p_mat1_1, post_loop_count);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, align_2, p_mat1_2, post_loop_count);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, align_3, p_mat1_3, post_loop_count);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, post_lc0);

    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec_0, p_vec_0, post_lc1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
  }
  *out_00 = acc64_00;
  *out_10 = acc64_10;
  *out_20 = acc64_20;
  *out_30 = acc64_30;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_unaligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int64* out_20
    ,ae_int64* out_30
    ,ae_int64* out_01
    ,ae_int64* out_11
    ,ae_int64* out_21
    ,ae_int64* out_31
    ,ae_int64* out_02
    ,ae_int64* out_12
    ,ae_int64* out_22
    ,ae_int64* out_32
    ,ae_int64* out_03
    ,ae_int64* out_13
    ,ae_int64* out_23
    ,ae_int64* out_33
    ,ae_int8x8*  p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec1_batch_0, vec1_batch_1;
  ae_int16x4 vec2_batch_0, vec2_batch_1;
  ae_int16x4 vec3_batch_0, vec3_batch_1;
  ae_int8x8 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2, align_p_mat1_3;

  ae_int8x8 *p_mat1_1 = (ae_int8x8 *)((WORD8 *)p_mat1_0 + row_offset);
  ae_int8x8 *p_mat1_2 = (ae_int8x8 *)((WORD8 *)p_mat1_1 + row_offset);
  ae_int8x8 *p_mat1_3 = (ae_int8x8 *)((WORD8 *)p_mat1_2 + row_offset);

  ae_int16x8 *p_vec_1 = (ae_int16x8 *)((WORD16 *)p_vec_0 + vec_offset);
  ae_int16x8 *p_vec_2 = (ae_int16x8 *)((WORD16 *)p_vec_1 + vec_offset);
  ae_int16x8 *p_vec_3 = (ae_int16x8 *)((WORD16 *)p_vec_2 + vec_offset);

  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
  ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
  ae_valignx2 align_p_vec_2 = AE_LA128_PP(p_vec_2);
  ae_valignx2 align_p_vec_3 = AE_LA128_PP(p_vec_3);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;
  ae_int64 acc64_20 = *out_20;
  ae_int64 acc64_30 = *out_30;

  ae_int64 acc64_01 = *out_01;
  ae_int64 acc64_11 = *out_11;
  ae_int64 acc64_21 = *out_21;
  ae_int64 acc64_31 = *out_31;

  ae_int64 acc64_02 = *out_02;
  ae_int64 acc64_12 = *out_12;
  ae_int64 acc64_22 = *out_22;
  ae_int64 acc64_32 = *out_32;

  ae_int64 acc64_03 = *out_03;
  ae_int64 acc64_13 = *out_13;
  ae_int64 acc64_23 = *out_23;
  ae_int64 acc64_33 = *out_33;

  AE_SW_PRIME_64(p_mat1_0, align_p_mat1_0);
  AE_SW_PRIME_64(p_mat1_1, align_p_mat1_1);
  AE_SW_PRIME_64(p_mat1_2, align_p_mat1_2);
  AE_SW_PRIME_64(p_mat1_3, align_p_mat1_3);

  int cols_count = cols -(cols & 7);
// #pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1);
    AE_LA16X4X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2);
    AE_LA16X4X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3);

    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, ((cols & 7) << 1));
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1, ((cols & 7) << 1));
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2, ((cols & 7) << 1));
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3, ((cols & 7) << 1));

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc64_01, acc64_11, acc64_21, acc64_31, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0, vec1_batch_1);
    AE_MULA8QW8X16(acc64_02, acc64_12, acc64_22, acc64_32, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0, vec2_batch_1);
    AE_MULA8QW8X16(acc64_03, acc64_13, acc64_23, acc64_33, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0, vec3_batch_1);
  }
  *out_00 = acc64_00;
  *out_10 = acc64_10;
  *out_20 = acc64_20;
  *out_30 = acc64_30;

  *out_01 = acc64_01;
  *out_11 = acc64_11;
  *out_21 = acc64_21;
  *out_31 = acc64_31;

  *out_02 = acc64_02;
  *out_12 = acc64_12;
  *out_22 = acc64_22;
  *out_32 = acc64_32;

  *out_03 = acc64_03;
  *out_13 = acc64_13;
  *out_23 = acc64_23;
  *out_33 = acc64_33;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int64* out_20
    ,ae_int64* out_30
    ,ae_int8x8*  p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int8x8 /*align_p_mat1_0, align_p_mat1_1, align_p_mat1_2,*/ align_p_mat1_3;
  ae_valign align_p_mat1_0, align_p_mat1_1, align_p_mat1_2;

  ae_int8x8 *p_mat1_1 = (ae_int8x8 *)((WORD8 *)p_mat1_0 + row_offset);
  ae_int8x8 *p_mat1_2 = (ae_int8x8 *)((WORD8 *)p_mat1_1 + row_offset);
  ae_int8x8 *p_mat1_3 = (ae_int8x8 *)((WORD8 *)p_mat1_2 + row_offset);

  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;
  ae_int64 acc64_20 = *out_20;
  ae_int64 acc64_30 = *out_30;

  // AE_SW_PRIME_64(p_mat1_0, align_p_mat1_0);
  // AE_SW_PRIME_64(p_mat1_1, align_p_mat1_1);
  // AE_SW_PRIME_64(p_mat1_2, align_p_mat1_2);
  align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  AE_SW_PRIME_64(p_mat1_3, align_p_mat1_3);

  int cols_count = cols -(cols & 7);
// #pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);

    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, ((cols & 7) << 1));

    AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
  }
  *out_00 = acc64_00;
  *out_10 = acc64_10;
  *out_20 = acc64_20;
  *out_30 = acc64_30;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_unaligned
    (ae_int64* out_00
    ,ae_int64* out_01
    ,ae_int64* out_02
    ,ae_int64* out_03
    ,ae_int8x8*  p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;

  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec1_batch_0, vec1_batch_1;
  ae_int16x4 vec2_batch_0, vec2_batch_1;
  ae_int16x4 vec3_batch_0, vec3_batch_1;
  ae_int8x8 align_p_mat1_0;

  ae_int16x8 *p_vec_1 = (ae_int16x8 *)((WORD16 *)p_vec_0 + vec_offset);
  ae_int16x8 *p_vec_2 = (ae_int16x8 *)((WORD16 *)p_vec_1 + vec_offset);
  ae_int16x8 *p_vec_3 = (ae_int16x8 *)((WORD16 *)p_vec_2 + vec_offset);

  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
  ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
  ae_valignx2 align_p_vec_2 = AE_LA128_PP(p_vec_2);
  ae_valignx2 align_p_vec_3 = AE_LA128_PP(p_vec_3);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = AE_ZERO64();

  ae_int64 acc64_01 = *out_01;
  ae_int64 acc64_11 = AE_ZERO64();

  ae_int64 acc64_02 = *out_02;
  ae_int64 acc64_12 = AE_ZERO64();

  ae_int64 acc64_03 = *out_03;
  ae_int64 acc64_13 = AE_ZERO64();

  AE_SW_PRIME_64(p_mat1_0, align_p_mat1_0);

  int cols_count = cols -(cols & 7);
// #pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1);
    AE_LA16X4X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2);
    AE_LA16X4X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3);

    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_01, acc64_11, vec1_batch_0, vec1_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_02, acc64_12, vec2_batch_0, vec2_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_03, acc64_13, vec3_batch_0, vec3_batch_1, mat1_row0_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, ((cols & 7) << 1));
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1, ((cols & 7) << 1));
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2, ((cols & 7) << 1));
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3, ((cols & 7) << 1));

    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_01, acc64_11, vec1_batch_0, vec1_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_02, acc64_12, vec2_batch_0, vec2_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_03, acc64_13, vec3_batch_0, vec3_batch_1, mat1_row0_0);
  }
  *out_00 = AE_ADD64(acc64_00, acc64_10);

  *out_01 = AE_ADD64(acc64_01, acc64_11);

  *out_02 = AE_ADD64(acc64_02, acc64_12);

  *out_03 = AE_ADD64(acc64_03, acc64_13);
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int64* out_00
    ,ae_int8x8*  p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 mat1_row0_0, mat1_row0_1;

  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec0_batch_2, vec0_batch_3;
  ae_valignx2 align_p_mat1_0;

  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = AE_ZERO64();

  align_p_mat1_0 = AE_LA128_PP((ae_int8x16 *)p_mat1_0);

  int cols_count = cols & (~15);
// #pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int16x8 *)p_vec_0);
    AE_LA16X4X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec_0, (ae_int16x8 *)p_vec_0);

    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_2, vec0_batch_3, mat1_row0_1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    int rem_cols0, rem_cols1;
    rem_cols0 = ((cols - cols_count > 8) ? 8 : (cols - cols_count)) << 1;
    rem_cols1 = ((cols - cols_count - 8 < 0) ? 0 : (cols - cols_count - 8)) << 1;
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int16x8 *)p_vec_0, rem_cols0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec_0, (ae_int16x8 *)p_vec_0, rem_cols1);

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0, (cols & 15));

    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc64_00, acc64_10, vec0_batch_2, vec0_batch_3, mat1_row0_1);
  }

  *out_00 = AE_ADD64(acc64_00, acc64_10);
}

WORD32 xa_nn_matmul_per_chan_sym8sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD16 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
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
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);

  int itr = 0;
  for(itr=0; itr<rows; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  ae_int64 acc_buffer[4];
  ae_int16* __restrict__ p_vec_0;

  /* Iterators used in for loops */
  int m_itr, vec_itr;

  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

#undef VEC_UNROLL
#define VEC_UNROLL 4

  /* Special case for cols == 8 */
  if(
      (cols1 == 8) &&
      (row_stride1 == 8) &&
      (vec_offset == 8) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 8) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x2) == 0)
    )
  {
    ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;

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

      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + 1];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + 2];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + 3];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_23, l_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;

      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

      AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

      ae_int16x4 *p_ae_dst_0 = (ae_int16x4 *)((WORD16*)p_out + (m_itr + 0) * out_stride);
      p_vec_0  = (ae_int16 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < (vec_count); vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

        /* This unusual sequence of I and IP is written to prevent compiler from converting them to mov */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

        ae_int16x4 vec0_batch_0, vec0_batch_1;

        /* Load  2 vectors  */
        AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8*)p_vec_0, 16);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

        /* Apply quantization */
        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

        /* Store output */
        *p_ae_dst_0 = out_0;
        p_ae_dst_0 += (out_offset >> 2);
      }
    }
    return 0;
  }

  /* Special case for cols == 16 */
  if(
      (cols1 == 16) &&
      (row_stride1 == 16) &&
      (vec_offset == 16) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 8) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;

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

      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + 1];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + 2];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + 3];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_23, l_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

      AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

      ae_int16x4 *p_ae_dst_0 = (ae_int16x4 *)((WORD16 *)p_out + (m_itr + 0) * out_stride);

      p_vec_0  = (ae_int16 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

        /* This unusual sequence of I and IP is written to prevent compiler from converting them to mov */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

        ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;

        /* Load  4 vectors  */
        AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8*)p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, (ae_int16x8*)p_vec_0, 16);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);

        /* Apply quantization */
        ae_int16x4 out_0; //, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

        /* Store output */
        *p_ae_dst_0 = out_0;
        p_ae_dst_0 += (out_offset >> 2);
      }
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
    ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;

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

      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + 1];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + 2];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + 3];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_23, l_mult_01;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

      ae_int16x4 *p_ae_dst_0 = (ae_int16x4 *)((WORD16*)p_out + (m_itr + 0) * out_stride);

      p_vec_0  = (ae_int16 *)(p_vec1);

      /* This takes 10 cycles per iteration, unrolling by 2 give 20, and unrolling by 4 gives 39,
         as gain is not much keeping unroll of 1 to have less code size */
#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

        /* This unusual sequence of I and IP is written to prevent compiler from converting them to mov */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

        ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3, vec0_batch_4, vec0_batch_5;

        /* Load  4 vectors  */
        AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8*)p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, (ae_int16x8*)p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_4, vec0_batch_5, (ae_int16x8*)p_vec_0, 16);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);

        /* Apply quantization */
        ae_int16x4 out_0; //out_1; //, out_2, out_3;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

        /* Store output */
        *p_ae_dst_0 = out_0;
        p_ae_dst_0 += (out_offset >> 2);
      }
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
    ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;

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

      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + 1];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + 2];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + 3];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

      int p_left_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x2 l_mult_01, l_mult_23;

      p_left_mult[0] = p_out_shift[m_itr + 0];
      p_left_mult[1] = p_out_shift[m_itr + 1];
      p_left_mult[2] = p_out_shift[m_itr + 2];
      p_left_mult[3] = p_out_shift[m_itr + 3];

      AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);

      int p_out_mult[4];
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      p_out_mult[0] = p_out_multiplier[m_itr + 0];
      p_out_mult[1] = p_out_multiplier[m_itr + 1];
      p_out_mult[2] = p_out_multiplier[m_itr + 2];
      p_out_mult[3] = p_out_multiplier[m_itr + 3];

      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

      ae_int16x4 *p_ae_dst_0 = (ae_int16x4 *)((WORD16*)p_out + (m_itr + 0) * out_stride);

      ae_int16* p_vec_0  = (ae_int16 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

        /* This unusual sequence of I and IP is written to prevent compiler from converting them to mov */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

        ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;

        ae_int16x4 vec0_batch_4, vec0_batch_5, vec0_batch_6, vec0_batch_7;

        /* Load  4 vectors  */
        AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8 *)p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, (ae_int16x8 *)p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_4, vec0_batch_5, (ae_int16x8 *)p_vec_0, 16);
        AE_L16X4X2_IP(vec0_batch_6, vec0_batch_7, (ae_int16x8 *)p_vec_0, 16);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);

        AE_MULA8QW8X16(acc64_00, acc64_10, acc64_20, acc64_30, mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);

        /* Apply quantization */
        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

        /* Store output */
        *p_ae_dst_0 = out_0;
        p_ae_dst_0 += (out_offset >> 2);
      }
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
    ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + 1];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + 2];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + 3];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

      ae_int16* p_dst_0 = (ae_int16*)p_out + (m_itr + 0) * out_stride;
      ae_int16* p_dst_1 = (ae_int16*)p_out + (m_itr + 1) * out_stride;
      ae_int16* p_dst_2 = (ae_int16*)p_out + (m_itr + 2) * out_stride;
      ae_int16* p_dst_3 = (ae_int16*)p_out + (m_itr + 3) * out_stride;

      ae_int32x2 l_mult_01 = AE_MOVDA32X2(p_out_shift[m_itr + 0], p_out_shift[m_itr + 1]);
      ae_int32x2 l_mult_23 = AE_MOVDA32X2(p_out_shift[m_itr + 2], p_out_shift[m_itr + 3]);

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

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;
        ae_int64 acc64_01, acc64_11, acc64_21, acc64_31;
        ae_int64 acc64_02, acc64_12, acc64_22, acc64_32;
        ae_int64 acc64_03, acc64_13, acc64_23, acc64_33;

        /* Initialize accumulators */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);
        AE_L64X2_IP(acc64_01, acc64_11, acc_buff, 0);
        AE_L64X2_I(acc64_21, acc64_31, acc_buff, 16);
        AE_L64X2_IP(acc64_02, acc64_12, acc_buff, 0);
        AE_L64X2_I(acc64_22, acc64_32, acc_buff, 16);
        AE_L64X2_IP(acc64_03, acc64_13, acc_buff, 0);
        AE_L64X2_I(acc64_23, acc64_33, acc_buff, 16);

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x16 *p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_4_vecs_aligned
          (&acc64_00
           ,&acc64_10
           ,&acc64_20
           ,&acc64_30
           ,&acc64_01
           ,&acc64_11
           ,&acc64_21
           ,&acc64_31
           ,&acc64_02
           ,&acc64_12
           ,&acc64_22
           ,&acc64_32
           ,&acc64_03
           ,&acc64_13
           ,&acc64_23
           ,&acc64_33
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec1, acc64_01, acc64_11, out_multiplier_01, l_mult_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec2, acc64_02, acc64_12, out_multiplier_01, l_mult_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec3, acc64_03, acc64_13, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec1, acc64_21, acc64_31, out_multiplier_23, l_mult_23);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec2, acc64_22, acc64_32, out_multiplier_23, l_mult_23);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec3, acc64_23, acc64_33, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        out_1 = AE_SAT16X4(acc_row0_vec1, acc_row1_vec1);
        out_2 = AE_SAT16X4(acc_row0_vec2, acc_row1_vec2);
        out_3 = AE_SAT16X4(acc_row0_vec3, acc_row1_vec3);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
        *p_dst_0 = AE_SEL16_6543(out_1, out_1);   p_dst_0 += out_offset;
        *p_dst_0 = AE_SEL16_6543(out_2, out_2);   p_dst_0 += out_offset;
        *p_dst_0 = AE_SEL16_6543(out_3, out_3);   p_dst_0 += out_offset;

        *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_1, out_1);   p_dst_1 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_2, out_2);   p_dst_1 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_3, out_3);   p_dst_1 += out_offset;

        *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_1, out_1);   p_dst_2 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_2, out_2);   p_dst_2 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_3, out_3);   p_dst_2 += out_offset;

        *p_dst_3 = (out_0);   p_dst_3 += out_offset;
        *p_dst_3 = (out_1);   p_dst_3 += out_offset;
        *p_dst_3 = (out_2);   p_dst_3 += out_offset;
        *p_dst_3 = (out_3);   p_dst_3 += out_offset;
      }

      // Remaining vectors
      for (vec_itr = (vec_count & (~3)); vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;
        acc64_00 = d_bias0;
        acc64_10 = d_bias1;
        acc64_20 = d_bias2;
        acc64_30 = d_bias3;

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x16 *p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc64_00
           ,&acc64_10
           ,&acc64_20
           ,&acc64_30
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;
        *p_dst_3 = (out_0);   p_dst_3 += out_offset;
      }
    }

    // remaining rows
    for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
    {
      ae_int64 d_bias0 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
      }
      AE_S64X2_I(d_bias0, d_bias0, acc_buff, 0);
      AE_S64X2_I(d_bias0, d_bias0, acc_buff, 16);

      ae_int16* p_dst_0 = (ae_int16*)p_out + (m_itr + 0) * out_stride;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        ae_int64 acc64_00;
        ae_int64 acc64_01;
        ae_int64 acc64_02;
        ae_int64 acc64_03;

        /* Initialize accumulators */
        AE_L64X2_IP(acc64_00, acc64_01, acc_buff, 0);
        AE_L64X2_I(acc64_02, acc64_03, acc_buff, 16);

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_1_rows_4_vecs_unaligned
          (&acc64_00
           ,&acc64_01
           ,&acc64_02
           ,&acc64_03
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec0, acc64_00, acc64_01, p_out_multiplier[m_itr + 0], p_out_shift[m_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec1, acc64_02, acc64_03, p_out_multiplier[m_itr + 0], p_out_shift[m_itr + 0]);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec1);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_0 = AE_SEL16_5432(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_0 = AE_SEL16_4321(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_0 = (out_0);   p_dst_0 += out_offset;
      }

      // Remaining vectors
      for (vec_itr = (vec_count & (~3)); vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;

        ae_int64 acc64_00;
        acc64_00 = d_bias0;

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x16 *p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc64_00
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec0, acc64_00, acc64_00, p_out_multiplier[m_itr + 0], p_out_shift[m_itr + 0]);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);

        *p_dst_0 = (out_0);   p_dst_0 += out_offset;
      }
    }
  }
  else if (p_mat1 && p_vec1)
  {
    int ii;
    ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;
    int p_left_shift[4];
    for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
    {
      for(ii = 0; ii < 8; ii++)
      {
        ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
        ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
        if(p_bias)
        {
          d_bias0 = *(ae_int64 *)&p_bias[m_itr + ii + 0];
          d_bias1 = *(ae_int64 *)&p_bias[m_itr + ii + 8];
          d_bias2 = *(ae_int64 *)&p_bias[m_itr + ii + 16];
          d_bias3 = *(ae_int64 *)&p_bias[m_itr + ii + 24];
        }
        AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
        AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

        WORD16* p_dst_0 = (WORD16*)p_out + (m_itr + ii +  0) * out_stride;
        WORD16* p_dst_1 = (WORD16*)p_out + (m_itr + ii +  8) * out_stride;
        WORD16* p_dst_2 = (WORD16*)p_out + (m_itr + ii + 16) * out_stride;
        WORD16* p_dst_3 = (WORD16*)p_out + (m_itr + ii + 24) * out_stride;

        p_left_shift[0] = p_out_shift[m_itr + ii +  0];
        p_left_shift[1] = p_out_shift[m_itr + ii +  8];
        p_left_shift[2] = p_out_shift[m_itr + ii + 16];
        p_left_shift[3] = p_out_shift[m_itr + ii + 24];

        ae_int32x2 l_mult_23;
        ae_int32x2 l_mult_01;
        AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_shift, 0);

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

          ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;
          ae_int64 acc64_01, acc64_11, acc64_21, acc64_31;
          ae_int64 acc64_02, acc64_12, acc64_22, acc64_32;
          ae_int64 acc64_03, acc64_13, acc64_23, acc64_33;

          /* Initialize accumulators */
          AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
          AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);
          AE_L64X2_IP(acc64_01, acc64_11, acc_buff, 0);
          AE_L64X2_I(acc64_21, acc64_31, acc_buff, 16);
          AE_L64X2_IP(acc64_02, acc64_12, acc_buff, 0);
          AE_L64X2_I(acc64_22, acc64_32, acc_buff, 16);
          AE_L64X2_IP(acc64_03, acc64_13, acc_buff, 0);
          AE_L64X2_I(acc64_23, acc64_33, acc_buff, 16);

          ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
          ae_int8x16 *p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr + ii + 0) * row_stride1];

          _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
            (&acc64_00
             ,&acc64_10
             ,&acc64_20
             ,&acc64_30
             ,&acc64_01
             ,&acc64_11
             ,&acc64_21
             ,&acc64_31
             ,&acc64_02
             ,&acc64_12
             ,&acc64_22
             ,&acc64_32
             ,&acc64_03
             ,&acc64_13
             ,&acc64_23
             ,&acc64_33
             ,p_mat1_0
             ,p_vec_0
             ,cols1
             ,row_stride1
             ,vec_offset
            );

          ae_int16x4 out_0, out_1, out_2, out_3;

          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);
          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec1, acc64_01, acc64_11, out_multiplier_01, l_mult_01);
          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec2, acc64_02, acc64_12, out_multiplier_01, l_mult_01);
          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec3, acc64_03, acc64_13, out_multiplier_01, l_mult_01);

          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);
          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec1, acc64_21, acc64_31, out_multiplier_23, l_mult_23);
          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec2, acc64_22, acc64_32, out_multiplier_23, l_mult_23);
          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec3, acc64_23, acc64_33, out_multiplier_23, l_mult_23);

          out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
          out_1 = AE_SAT16X4(acc_row0_vec1, acc_row1_vec1);
          out_2 = AE_SAT16X4(acc_row0_vec2, acc_row1_vec2);
          out_3 = AE_SAT16X4(acc_row0_vec3, acc_row1_vec3);

          *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
          *p_dst_0 = AE_SEL16_6543(out_1, out_1);   p_dst_0 += out_offset;
          *p_dst_0 = AE_SEL16_6543(out_2, out_2);   p_dst_0 += out_offset;
          *p_dst_0 = AE_SEL16_6543(out_3, out_3);   p_dst_0 += out_offset;

          *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
          *p_dst_1 = AE_SEL16_5432(out_1, out_1);   p_dst_1 += out_offset;
          *p_dst_1 = AE_SEL16_5432(out_2, out_2);   p_dst_1 += out_offset;
          *p_dst_1 = AE_SEL16_5432(out_3, out_3);   p_dst_1 += out_offset;

          *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;
          *p_dst_2 = AE_SEL16_4321(out_1, out_1);   p_dst_2 += out_offset;
          *p_dst_2 = AE_SEL16_4321(out_2, out_2);   p_dst_2 += out_offset;
          *p_dst_2 = AE_SEL16_4321(out_3, out_3);   p_dst_2 += out_offset;

          *p_dst_3 = (out_0);   p_dst_3 += out_offset;
          *p_dst_3 = (out_1);   p_dst_3 += out_offset;
          *p_dst_3 = (out_2);   p_dst_3 += out_offset;
          *p_dst_3 = (out_3);   p_dst_3 += out_offset;
        }

        // Remaining vectors
        for (; vec_itr < vec_count; vec_itr++)
        {
          ae_int32x2 acc_row0_vec0;
          ae_int32x2 acc_row1_vec0;

          ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

          /* Initialize accumulators */
          AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
          AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

          ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
          ae_int8x16 *p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr + ii + 0) * row_stride1];

          _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
            (&acc64_00
             ,&acc64_10
             ,&acc64_20
             ,&acc64_30
             ,p_mat1_0
             ,p_vec_0
             ,cols1
             ,row_stride1
            );

          ae_int16x4 out_0;

          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

          MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

          out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

          *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;

          *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;

          *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;

          *p_dst_3 = (out_0);                       p_dst_3 += out_offset;
        }
      }
    }

    // Process loop for 4 rows and 4 vectors
    for(; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + 1];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + 2];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + 3];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);

      WORD16* p_dst_0 = (WORD16*)p_out + (m_itr + 0) * out_stride;
      WORD16* p_dst_1 = (WORD16*)p_out + (m_itr + 1) * out_stride;
      WORD16* p_dst_2 = (WORD16*)p_out + (m_itr + 2) * out_stride;
      WORD16* p_dst_3 = (WORD16*)p_out + (m_itr + 3) * out_stride;

      int ALIGN(16) p_left_shift[4];
      p_left_shift[0] = p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_23;
      ae_int32x2 l_mult_01;
      AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_shift, 0);

      ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(p_out_multiplier[m_itr + 2], p_out_multiplier[m_itr + 3]);
      ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(p_out_multiplier[m_itr + 0], p_out_multiplier[m_itr + 1]);

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

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;
        ae_int64 acc64_01, acc64_11, acc64_21, acc64_31;
        ae_int64 acc64_02, acc64_12, acc64_22, acc64_32;
        ae_int64 acc64_03, acc64_13, acc64_23, acc64_33;

        /* Initialize accumulators */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);
        AE_L64X2_IP(acc64_01, acc64_11, acc_buff, 0);
        AE_L64X2_I(acc64_21, acc64_31, acc_buff, 16);
        AE_L64X2_IP(acc64_02, acc64_12, acc_buff, 0);
        AE_L64X2_I(acc64_22, acc64_32, acc_buff, 16);
        AE_L64X2_IP(acc64_03, acc64_13, acc_buff, 0);
        AE_L64X2_I(acc64_23, acc64_33, acc_buff, 16);

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_4_vecs_unaligned
          (&acc64_00
           ,&acc64_10
           ,&acc64_20
           ,&acc64_30
           ,&acc64_01
           ,&acc64_11
           ,&acc64_21
           ,&acc64_31
           ,&acc64_02
           ,&acc64_12
           ,&acc64_22
           ,&acc64_32
           ,&acc64_03
           ,&acc64_13
           ,&acc64_23
           ,&acc64_33
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec1, acc64_01, acc64_11, out_multiplier_01, l_mult_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec2, acc64_02, acc64_12, out_multiplier_01, l_mult_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec3, acc64_03, acc64_13, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec1, acc64_21, acc64_31, out_multiplier_23, l_mult_23);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec2, acc64_22, acc64_32, out_multiplier_23, l_mult_23);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec3, acc64_23, acc64_33, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);
        out_1 = AE_SAT16X4(acc_row0_vec1, acc_row1_vec1);
        out_2 = AE_SAT16X4(acc_row0_vec2, acc_row1_vec2);
        out_3 = AE_SAT16X4(acc_row0_vec3, acc_row1_vec3);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
        *p_dst_0 = AE_SEL16_6543(out_1, out_1);   p_dst_0 += out_offset;
        *p_dst_0 = AE_SEL16_6543(out_2, out_2);   p_dst_0 += out_offset;
        *p_dst_0 = AE_SEL16_6543(out_3, out_3);   p_dst_0 += out_offset;

        *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_1, out_1);   p_dst_1 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_2, out_2);   p_dst_1 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_3, out_3);   p_dst_1 += out_offset;

        *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_1, out_1);   p_dst_2 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_2, out_2);   p_dst_2 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_3, out_3);   p_dst_2 += out_offset;

        *p_dst_3 = (out_0);   p_dst_3 += out_offset;
        *p_dst_3 = (out_1);   p_dst_3 += out_offset;
        *p_dst_3 = (out_2);   p_dst_3 += out_offset;
        *p_dst_3 = (out_3);   p_dst_3 += out_offset;
      }

      // Remaining vectors
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;

        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

        /* Initialize accumulators */
        AE_L64X2_IP(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc64_00
           ,&acc64_10
           ,&acc64_20
           ,&acc64_30
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row0_vec0, acc64_00, acc64_10, out_multiplier_01, l_mult_01);

        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(acc_row1_vec0, acc64_20, acc64_30, out_multiplier_23, l_mult_23);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row1_vec0);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;

        *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;

        *p_dst_3 = (out_0);                       p_dst_3 += out_offset;
      }
    }

    // remaining rows
    for(; m_itr < rows; m_itr++)
    {
      ae_int64 d_bias0 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr];
      }
      AE_S64X2_I(d_bias0, d_bias0, acc_buff, 0);
      AE_S64X2_I(d_bias0, d_bias0, acc_buff, 16);

      WORD16* p_dst_0 = (WORD16*)p_out + (m_itr + 0) * out_stride;

      int left_shift;
      left_shift = p_out_shift[m_itr + 0];

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        ae_int64 acc64_00;
        ae_int64 acc64_01;
        ae_int64 acc64_02;
        ae_int64 acc64_03;

        /* Initialize accumulators */
        AE_L64X2_IP(acc64_00, acc64_01, acc_buff, 0);
        AE_L64X2_I(acc64_02, acc64_03, acc_buff, 16);

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_1_rows_4_vecs_unaligned
          (&acc64_00
           ,&acc64_01
           ,&acc64_02
           ,&acc64_03
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_offset
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec0, acc64_00, acc64_01, p_out_multiplier[m_itr + 0], left_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec1, acc64_02, acc64_03, p_out_multiplier[m_itr + 0], left_shift);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec1);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_0 = AE_SEL16_5432(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_0 = AE_SEL16_4321(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_0 = (out_0);   p_dst_0 += out_offset;
      }

      // Remaining vectors
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;

        ae_int64 acc64_00;

        /* Initialize accumulators */
        AE_L64_IP(acc64_00, (ae_int64 *)acc_buff, 0);

        ae_int16x8* p_vec_0  = (ae_int16x8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc64_00
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;

        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec0, acc64_00, acc64_00, p_out_multiplier[m_itr + 0], left_shift);

        out_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);

        *p_dst_0 = (out_0);   p_dst_0 += out_offset;
      }
    }
  }
  else
  {
    return -1;
  }
    return 0;
}
