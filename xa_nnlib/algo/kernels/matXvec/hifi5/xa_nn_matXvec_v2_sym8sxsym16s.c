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
static inline void _xa_nn_dot_product_8_rows_1_vecs_aligned
    (ae_int64* out_0_0
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    )
{
  int c_itr = 0;

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 mat1_row4_0, mat1_row4_1, mat1_row4_2, mat1_row4_3;
  ae_int8x8 mat1_row5_0, mat1_row5_1, mat1_row5_2, mat1_row5_3;
  ae_int8x8 mat1_row6_0, mat1_row6_1, mat1_row6_2, mat1_row6_3;
  ae_int8x8 mat1_row7_0, mat1_row7_1, mat1_row7_2, mat1_row7_3;
  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec0_batch_4, vec0_batch_5, vec0_batch_6, vec0_batch_7;

  ae_int8x16 *p_mat1_4 = (ae_int8x16*)((WORD8 *)p_mat1_0 + 4 * row_stride1);

  ae_int64 acc_row0_vec0;
  ae_int64 acc_row1_vec0;
  ae_int64 acc_row2_vec0;
  ae_int64 acc_row3_vec0;
  ae_int64 acc_row4_vec0;
  ae_int64 acc_row5_vec0;
  ae_int64 acc_row6_vec0;
  ae_int64 acc_row7_vec0;

  ae_int64x2 *p_out = (ae_int64x2 *)out_0_0;
  AE_L64X2_IP(acc_row0_vec0, acc_row1_vec0, p_out, 16);
  AE_L64X2_IP(acc_row2_vec0, acc_row3_vec0, p_out, 16);
  AE_L64X2_IP(acc_row4_vec0, acc_row5_vec0, p_out, 16);
  AE_L64X2_IP(acc_row6_vec0, acc_row7_vec0, p_out, 16);

  for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, p_mat1_0, 16);
    AE_L8X8X2_XP(mat1_row0_0, mat1_row0_1, p_mat1_0, row_stride1);

    AE_L8X8X2_I(mat1_row1_2, mat1_row1_3, p_mat1_0, 16);
    AE_L8X8X2_XP(mat1_row1_0, mat1_row1_1, p_mat1_0, row_stride1);

    AE_L8X8X2_I(mat1_row2_2, mat1_row2_3, p_mat1_0, 16);
    AE_L8X8X2_XP(mat1_row2_0, mat1_row2_1, p_mat1_0, row_stride1);

    AE_L8X8X2_I(mat1_row3_2, mat1_row3_3, p_mat1_0, 16);
    AE_L8X8X2_XP(mat1_row3_0, mat1_row3_1, p_mat1_0, 32 - 3*row_stride1);

    AE_L8X8X2_I(mat1_row4_2, mat1_row4_3, p_mat1_4, 16);
    AE_L8X8X2_XP(mat1_row4_0, mat1_row4_1, p_mat1_4, row_stride1);

    AE_L8X8X2_I(mat1_row5_2, mat1_row5_3, p_mat1_4, 16);
    AE_L8X8X2_XP(mat1_row5_0, mat1_row5_1, p_mat1_4, row_stride1);

    AE_L8X8X2_I(mat1_row6_2, mat1_row6_3, p_mat1_4, 16);
    AE_L8X8X2_XP(mat1_row6_0, mat1_row6_1, p_mat1_4, row_stride1);

    AE_L8X8X2_I(mat1_row7_2, mat1_row7_3, p_mat1_4, 16);
    AE_L8X8X2_XP(mat1_row7_0, mat1_row7_1, p_mat1_4, 32 - 3*row_stride1);

    AE_L16X4X2_I(vec0_batch_2, vec0_batch_3, p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 32);

    AE_L16X4X2_I(vec0_batch_6, vec0_batch_7, p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_4, vec0_batch_5, p_vec_0, 32);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);

    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_0, mat1_row5_0, mat1_row6_0, mat1_row7_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_1, mat1_row5_1, mat1_row6_1, mat1_row7_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_2, mat1_row5_2, mat1_row6_2, mat1_row7_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_3, mat1_row5_3, mat1_row6_3, mat1_row7_3, vec0_batch_6, vec0_batch_7);
  }
  //Remainder loop for cols1
  WORD32 rem_cols = (cols1 & 31);
  ae_valignx2 vec_align = AE_LA128_PP(p_vec_0);
  if(rem_cols > 0)
  {
    WORD32 rem_count8 = rem_cols >= 16 ? 16 : rem_cols;
    WORD32 rem_count8_a = rem_cols <= 16 ? 0 : (rem_cols - 16);
    WORD32 rem_vec0 = rem_count8 >= 8 ? 16 : rem_count8 << 1;
    WORD32 rem_vec1 = rem_count8 <= 8 ? 0 : (rem_count8 - 8) << 1;
    WORD32 rem_vec2 = rem_count8_a >= 8 ? 16 : rem_count8_a << 1;
    WORD32 rem_vec3 = rem_count8_a <= 8 ? 0 : (rem_count8_a - 8) << 1;

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, vec_align, p_vec_0, rem_vec0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, vec_align, p_vec_0, rem_vec1);
    AE_LAV16X4X2_XP(vec0_batch_4, vec0_batch_5, vec_align, p_vec_0, rem_vec2);
    AE_LAV16X4X2_XP(vec0_batch_6, vec0_batch_7, vec_align, p_vec_0, rem_vec3);

    ae_int8x16 *p_mat1_1 = (ae_int8x16 *)((WORD8 *)p_mat1_0 + row_stride1);
    ae_int8x16 *p_mat1_2 = (ae_int8x16 *)((WORD8 *)p_mat1_1 + row_stride1);
    ae_int8x16 *p_mat1_3 = (ae_int8x16 *)((WORD8 *)p_mat1_2 + row_stride1);

    ae_valignx2 mat1_0_align, mat1_1_align, mat1_2_align, mat1_3_align;
    ae_valignx2 mat1_4_align, mat1_5_align, mat1_6_align, mat1_7_align;

    mat1_0_align = AE_LA128_PP(p_mat1_0);
    mat1_1_align = AE_LA128_PP(p_mat1_1);
    mat1_2_align = AE_LA128_PP(p_mat1_2);
    mat1_3_align = AE_LA128_PP(p_mat1_3);

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, mat1_0_align, p_mat1_0, rem_count8);
    AE_LAV8X8X2_XP(mat1_row0_2, mat1_row0_3, mat1_0_align, p_mat1_0, rem_count8_a);

    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, mat1_1_align, p_mat1_1, rem_count8);
    AE_LAV8X8X2_XP(mat1_row1_2, mat1_row1_3, mat1_1_align, p_mat1_1, rem_count8_a);

    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, mat1_2_align, p_mat1_2, rem_count8);
    AE_LAV8X8X2_XP(mat1_row2_2, mat1_row2_3, mat1_2_align, p_mat1_2, rem_count8_a);

    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, mat1_3_align, p_mat1_3, rem_count8);
    AE_LAV8X8X2_XP(mat1_row3_2, mat1_row3_3, mat1_3_align, p_mat1_3, rem_count8_a);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);

    ae_int8x16 *p_mat1_5 = (ae_int8x16 *)((WORD8 *)p_mat1_4 + row_stride1);
    ae_int8x16 *p_mat1_6 = (ae_int8x16 *)((WORD8 *)p_mat1_5 + row_stride1);
    ae_int8x16 *p_mat1_7 = (ae_int8x16 *)((WORD8 *)p_mat1_6 + row_stride1);

    mat1_4_align = AE_LA128_PP(p_mat1_4);
    mat1_5_align = AE_LA128_PP(p_mat1_5);
    mat1_6_align = AE_LA128_PP(p_mat1_6);
    mat1_7_align = AE_LA128_PP(p_mat1_7);

    AE_LAV8X8X2_XP(mat1_row4_0, mat1_row4_1, mat1_4_align, p_mat1_4, rem_count8);
    AE_LAV8X8X2_XP(mat1_row4_2, mat1_row4_3, mat1_4_align, p_mat1_4, rem_count8_a);

    AE_LAV8X8X2_XP(mat1_row5_0, mat1_row5_1, mat1_5_align, p_mat1_5, rem_count8);
    AE_LAV8X8X2_XP(mat1_row5_2, mat1_row5_3, mat1_5_align, p_mat1_5, rem_count8_a);

    AE_LAV8X8X2_XP(mat1_row6_0, mat1_row6_1, mat1_6_align, p_mat1_6, rem_count8);
    AE_LAV8X8X2_XP(mat1_row6_2, mat1_row6_3, mat1_6_align, p_mat1_6, rem_count8_a);

    AE_LAV8X8X2_XP(mat1_row7_0, mat1_row7_1, mat1_7_align, p_mat1_7, rem_count8);
    AE_LAV8X8X2_XP(mat1_row7_2, mat1_row7_3, mat1_7_align, p_mat1_7, rem_count8_a);

    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_0, mat1_row5_0, mat1_row6_0, mat1_row7_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_1, mat1_row5_1, mat1_row6_1, mat1_row7_1, vec0_batch_2, vec0_batch_3);

    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_2, mat1_row5_2, mat1_row6_2, mat1_row7_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row4_vec0, acc_row5_vec0, acc_row6_vec0, acc_row7_vec0,
        mat1_row4_3, mat1_row5_3, mat1_row6_3, mat1_row7_3, vec0_batch_6, vec0_batch_7);
  }

  p_out = (ae_int64x2 *)out_0_0;
  AE_S64X2_IP(acc_row0_vec0, acc_row1_vec0, p_out, 16);
  AE_S64X2_IP(acc_row2_vec0, acc_row3_vec0, p_out, 16);
  AE_S64X2_IP(acc_row4_vec0, acc_row5_vec0, p_out, 16);
  AE_S64X2_IP(acc_row6_vec0, acc_row7_vec0, p_out, 16);
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int64* out_0_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int16*   p_vec_0
    ,WORD32      cols1
    )
{
  int rem_cols = (cols1 & 31);

  int c_itr = 0;

  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec0_batch_4, vec0_batch_5, vec0_batch_6, vec0_batch_7;
  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;

  ae_int64 acc_row0_vec0 = *out_0_0;
  ae_int64 acc_row1_vec0 = AE_ZERO64();

#pragma no_unroll
  for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 32);

    AE_L16X4X2_I(vec0_batch_2, vec0_batch_3, (ae_int16x8 *)p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8 *)p_vec_0, 32);
    AE_L16X4X2_I(vec0_batch_6, vec0_batch_7, (ae_int16x8 *)p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_4, vec0_batch_5, (ae_int16x8 *)p_vec_0, 32);

    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_2, vec0_batch_3, mat1_row0_1);
    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_4, vec0_batch_5, mat1_row0_2);
    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_6, vec0_batch_7, mat1_row0_3);
  }
  //Remainder loop for cols1
  ae_int16x8 *p_ae_vec_0 = (ae_int16x8 *)p_vec_0;
  ae_valignx2 vec_align = AE_LA128_PP(p_ae_vec_0);
  while(rem_cols > 0)
  {
    WORD32 rem_count8 = rem_cols >= 16 ? 16 : rem_cols;
    WORD32 rem_vec0 = rem_count8 >= 8 ? 16 : rem_count8 << 1;
    WORD32 rem_vec1 = rem_count8 <= 8 ? 0 : (rem_count8 - 8) << 1;

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, vec_align, p_ae_vec_0, rem_vec0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, vec_align, p_ae_vec_0, rem_vec1);

    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_2, vec0_batch_3, mat1_row0_1);

    rem_cols -= 16;
  }

  *out_0_0 = AE_ADD64(acc_row0_vec0, acc_row1_vec0);
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int64* out_0_0
    ,ae_int64* out_1_0
    ,ae_int64* out_2_0
    ,ae_int64* out_3_0
    ,ae_int8x16* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  int align_offset = ((unsigned int)p_mat1_0 & 0xf);
  pre_loop_count = align_offset == 0 ? 0 : 16 - align_offset;
  pre_loop_count = pre_loop_count > cols1 ? cols1 : pre_loop_count;

  loop_count = (cols1 < pre_loop_count) ? 0 : (cols1 - pre_loop_count);
  post_loop_count = (loop_count & 31);
  loop_count >>= 5;

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int16x4 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec0_batch_4, vec0_batch_5, vec0_batch_6, vec0_batch_7;

  ae_int8x16* p_mat1_1 = p_mat1_0 + row_stride1; //next 16th row
  ae_int8x16* p_mat1_2 = p_mat1_1 + row_stride1; //next 16th row
  ae_int8x16* p_mat1_3 = p_mat1_2 + row_stride1; //next 16th row

  ae_valignx2 align_p_vec0;

  ae_int64 acc_row0_vec0 = *out_0_0;
  ae_int64 acc_row1_vec0 = *out_1_0;
  ae_int64 acc_row2_vec0 = *out_2_0;
  ae_int64 acc_row3_vec0 = *out_3_0;

  /* Pre loop computation */
  if(pre_loop_count)
  {
    WORD32 pre_vec0 = pre_loop_count >= 8 ? 16 : pre_loop_count << 1;
    WORD32 pre_vec1 = pre_loop_count <= 8 ? 0 : (pre_loop_count - 8) << 1;
    ae_valignx2 mat_align0, mat_align1, mat_align2, mat_align3;

    mat_align0 = AE_LA128_PP(p_mat1_0);
    mat_align1 = AE_LA128_PP(p_mat1_1);
    mat_align2 = AE_LA128_PP(p_mat1_2);
    mat_align3 = AE_LA128_PP(p_mat1_3);

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, mat_align0, p_mat1_0, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, mat_align1, p_mat1_1, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, mat_align2, p_mat1_2, pre_loop_count);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, mat_align3, p_mat1_3, pre_loop_count);

    align_p_vec0 = AE_LA128_PP(p_vec_0);
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec0, p_vec_0, pre_vec0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec0, p_vec_0, pre_vec1);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
  }

  align_p_vec0 = AE_LA128_PP(p_vec_0);

  /* Keeping both loads of 3rd and 4th row IP for better slotting, there
     may be memory banking conflicts but loop cycles are 6, if 3rd and 4th
     row loads are kept I and IP, loop cycles are 8 */
#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, p_mat1_0, 16);
    AE_L8X8X2_I(mat1_row1_2, mat1_row1_3, p_mat1_1, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat1_0, 32);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat1_1, 32);

    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat1_3, 16);

    AE_L8X8X2_IP(mat1_row2_2, mat1_row2_3, p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_2, mat1_row3_3, p_mat1_3, 16);

    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, p_vec_0);
    AE_LA16X4X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec0, p_vec_0);
    AE_LA16X4X2_IP(vec0_batch_4, vec0_batch_5, align_p_vec0, p_vec_0);
    AE_LA16X4X2_IP(vec0_batch_6, vec0_batch_7, align_p_vec0, p_vec_0);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);
  }

  //Remainder loop for cols1
  if(post_loop_count > 0)
  {
    WORD32 post_count8 = post_loop_count >= 16 ? 16 : post_loop_count;
    WORD32 post_count8_a = post_loop_count <= 16 ? 0 : post_loop_count - 16;
    WORD32 post_vec0 = post_count8 >= 8 ? 16 : post_count8 << 1;
    WORD32 post_vec1 = post_count8 <= 8 ? 0 : (post_count8 - 8) << 1;
    WORD32 post_vec2 = post_count8_a >= 8 ? 16 : post_count8_a << 1;
    WORD32 post_vec3 = post_count8_a <= 8 ? 0 : (post_count8_a - 8) << 1;

    ae_valignx2 mat_align0, mat_align1, mat_align2, mat_align3;

    mat_align0 = AE_LA128_PP(p_mat1_0);
    mat_align1 = AE_LA128_PP(p_mat1_1);
    mat_align2 = AE_LA128_PP(p_mat1_2);
    mat_align3 = AE_LA128_PP(p_mat1_3);

    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, mat_align0, p_mat1_0, post_count8);
    AE_LAV8X8X2_XP(mat1_row0_2, mat1_row0_3, mat_align0, p_mat1_0, post_count8_a);

    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, mat_align1, p_mat1_1, post_count8);
    AE_LAV8X8X2_XP(mat1_row1_2, mat1_row1_3, mat_align1, p_mat1_1, post_count8_a);

    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, mat_align2, p_mat1_2, post_count8);
    AE_LAV8X8X2_XP(mat1_row2_2, mat1_row2_3, mat_align2, p_mat1_2, post_count8_a);

    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, mat_align3, p_mat1_3, post_count8);
    AE_LAV8X8X2_XP(mat1_row3_2, mat1_row3_3, mat_align3, p_mat1_3, post_count8_a);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec0, p_vec_0, post_vec0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec0, p_vec_0, post_vec1);
    AE_LAV16X4X2_XP(vec0_batch_4, vec0_batch_5, align_p_vec0, p_vec_0, post_vec2);
    AE_LAV16X4X2_XP(vec0_batch_6, vec0_batch_7, align_p_vec0, p_vec_0, post_vec3);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
  *out_2_0 = acc_row2_vec0;
  *out_3_0 = acc_row3_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int64* out_0_0
    ,ae_int64* out_1_0
    ,ae_int64* out_2_0
    ,ae_int64* out_3_0
    ,ae_int8x16*  p_mat1_0
    ,ae_int16*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int c_itr = 0;

  WORD32 loop_count = cols;

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;

  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec0_batch_2, vec0_batch_3;
  ae_int16x4 vec0_batch_4, vec0_batch_5;
  ae_int16x4 vec0_batch_6, vec0_batch_7;

  ae_int8x16 *p_mat1_1 = (ae_int8x16*)((WORD8 *)p_mat1_0 + row_offset);
  ae_int8x16 *p_mat1_2 = (ae_int8x16*)((WORD8 *)p_mat1_1 + row_offset);
  ae_int8x16 *p_mat1_3 = (ae_int8x16*)((WORD8 *)p_mat1_2 + row_offset);

  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
  ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
  ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

  ae_int64 acc_row0_vec0 = *out_0_0;
  ae_int64 acc_row1_vec0 = *out_1_0;
  ae_int64 acc_row2_vec0 = *out_2_0;
  ae_int64 acc_row3_vec0 = *out_3_0;

  align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
  align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
  align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

  for(c_itr = 0; c_itr < loop_count >> 5; c_itr++)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, p_mat1_1);
    AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, p_mat1_2);
    AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, p_mat1_3);

    AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, p_mat1_0);
    AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, p_mat1_1);
    AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, p_mat1_2);
    AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, p_mat1_3);

    AE_L16X4X2_I(vec0_batch_2, vec0_batch_3, (ae_int16x8 *)p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8 *)p_vec_0, 32);
    AE_L16X4X2_I(vec0_batch_6, vec0_batch_7, (ae_int16x8 *)p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_4, vec0_batch_5, (ae_int16x8 *)p_vec_0, 32);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);
  }

  WORD32 rem_count = (loop_count & 31);
  //Remainder loop for cols
  if (rem_count > 0)
  {
    WORD32 rem_count8 = rem_count >= 16 ? 16 : rem_count;
    WORD32 rem_count8_a = rem_count <= 16 ? 0 : rem_count - 16;
    WORD32 rem16_0 = rem_count8 >= 8 ? 16 : rem_count8 << 1;
    WORD32 rem16_1 = rem_count8 <= 8 ? 0 : (rem_count8 - 8) << 1;
    WORD32 rem16_2 = rem_count8_a >= 8 ? 16 : rem_count8_a << 1;
    WORD32 rem16_3 = rem_count8_a <= 8 ? 0 : (rem_count8_a - 8) << 1;
    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0, rem_count8);
    AE_LAV8X8X2_XP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, p_mat1_1, rem_count8);
    AE_LAV8X8X2_XP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, p_mat1_2, rem_count8);
    AE_LAV8X8X2_XP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, p_mat1_3, rem_count8);

    AE_LAV8X8X2_XP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, p_mat1_0, rem_count8_a);
    AE_LAV8X8X2_XP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, p_mat1_1, rem_count8_a);
    AE_LAV8X8X2_XP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, p_mat1_2, rem_count8_a);
    AE_LAV8X8X2_XP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, p_mat1_3, rem_count8_a);

    ae_valignx2 align_p_vec0 = AE_LA128_PP(p_vec_0);
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec0, p_vec_0, rem16_0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec0, p_vec_0, rem16_1);
    AE_LAV16X4X2_XP(vec0_batch_4, vec0_batch_5, align_p_vec0, p_vec_0, rem16_2);
    AE_LAV16X4X2_XP(vec0_batch_6, vec0_batch_7, align_p_vec0, p_vec_0, rem16_3);

    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0, vec0_batch_1);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_2, vec0_batch_3);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, vec0_batch_4, vec0_batch_5);
    AE_MULA8QW8X16(acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0,
        mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, vec0_batch_6, vec0_batch_7);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
  *out_2_0 = acc_row2_vec0;
  *out_3_0 = acc_row3_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int64* out_0_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int16*    p_vec_0
    ,WORD32      cols1
    )
{
  int c_itr = 0;
  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec0_batch_2, vec0_batch_3;
  ae_int8x8 mat1_row0_0, mat1_row0_1;

  ae_int64 acc_row0_vec0 = *out_0_0;
  ae_int64 acc_row1_vec0 = AE_ZERO64();

  ae_valign align1_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

  for(c_itr = 0; c_itr < cols1 >> 4; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align1_p_mat1_0, (ae_int8x8 *)p_mat1_0);
    AE_LA8X8_IP(mat1_row0_1, align1_p_mat1_0, (ae_int8x8 *)p_mat1_0);

    AE_L16X4X2_IP(vec0_batch_0, vec0_batch_1, (ae_int16x8 *)p_vec_0, 16);
    AE_L16X4X2_IP(vec0_batch_2, vec0_batch_3, (ae_int16x8 *)p_vec_0, 16);

    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_2, vec0_batch_3, mat1_row0_1);
  }

  WORD32 rem_count = (cols1 & 15);
  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  //Remainder loop for cols1
  if(rem_count != 0)
  {
    WORD32 rem16_0 = rem_count >= 8 ? 16 : rem_count << 1;
    WORD32 rem16_1 = rem_count <= 8 ? 0 : (rem_count - 8) << 1;
    AE_LAV8X8X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0, rem_count);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int16x8 *)p_vec_0, rem16_0);
    AE_LAV16X4X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec_0, (ae_int16x8 *)p_vec_0, rem16_1);

    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_0, vec0_batch_1, mat1_row0_0);
    AE_MULAAAA2Q16X8(acc_row0_vec0, acc_row1_vec0, vec0_batch_2, vec0_batch_3, mat1_row0_1);
  }

  *out_0_0 = AE_ADD64(acc_row0_vec0, acc_row1_vec0);
}


WORD32 xa_nn_matXvec_v2_sym8sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD16 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD8)*16, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16)*8, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16)*8, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64)*2, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 15), -1);
  /* Iterators used in for loops */
  int m_itr, ii;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }

  const WORD8 *p_mat1_0;
  const WORD16 *p_vec1_0;
  
  ae_int16x4 max_int16 = AE_MOVDA16(out_activation_max);
  ae_int16x4 min_int16 = AE_MOVDA16(out_activation_min);
  
  ae_int32x2 max_int32 = AE_MOVDA32(out_activation_max);
  ae_int32x2 min_int32 = AE_MOVDA32(out_activation_min);
  
  if((row_stride1 & 15) == 0)
  {
    for(m_itr = 0; m_itr < (rows & ~(8 - 1)); m_itr += 8)
    {
      ae_int64 ALIGN(16) acc_row_vec0[8];
      ae_int64 bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7;
      bias0 = AE_ZERO64();
      bias1 = AE_ZERO64();
      bias2 = AE_ZERO64();
      bias3 = AE_ZERO64();
      bias4 = AE_ZERO64();
      bias5 = AE_ZERO64();
      bias6 = AE_ZERO64();
      bias7 = AE_ZERO64();

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L64X2_I(bias2, bias3, (ae_int64x2 *)p_bias, 16);
        AE_L64X2_IP(bias0, bias1, (ae_int64x2 *)p_bias, 32);
        AE_L64X2_I(bias6, bias7, (ae_int64x2 *)p_bias, 16);
        AE_L64X2_IP(bias4, bias5, (ae_int64x2 *)p_bias, 32);
      }
      ae_int64x2 *p_acc = (ae_int64x2 *)acc_row_vec0;
      AE_S64X2_IP(bias0, bias1, p_acc, 16);
      AE_S64X2_IP(bias2, bias3, p_acc, 16);
      AE_S64X2_IP(bias4, bias5, p_acc, 16);
      AE_S64X2_IP(bias6, bias7, p_acc, 16);

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr*row_stride1));
      p_vec1_0 = (const WORD16 *)(p_vec1);

      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row_vec0[0]
         ,(ae_int8x16*)p_mat1_0
         ,(ae_int16x8*)p_vec1_0
         ,cols1
         ,row_stride1
        );

      ae_int16x4 out0, out1;
#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16_HIFI5S(out0, acc_row_vec0[0], acc_row_vec0[1], acc_row_vec0[2], acc_row_vec0[3], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16_HIFI5S(out1, acc_row_vec0[4], acc_row_vec0[5], acc_row_vec0[6], acc_row_vec0[7], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else      
      MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16(out0, acc_row_vec0[0], acc_row_vec0[1], acc_row_vec0[2], acc_row_vec0[3], out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16(out1, acc_row_vec0[4], acc_row_vec0[5], acc_row_vec0[6], acc_row_vec0[7], out_multiplier, out_shift);
#endif
      AE_MINMAX16(out0, min_int16, max_int16);
      AE_MINMAX16(out1, min_int16, max_int16);
      
      AE_S16X4X2_IP(out0, out1, (ae_int16x8 *)p_out, 16);
    }

    /* Compute last (rows % 8) output element */
    for (m_itr = (rows & (~(8 - 1))); m_itr < rows; m_itr++)
    {
      ae_int64 acc_row0_vec0 = AE_ZERO64();

      p_mat1_0 = (const WORD8 *)(p_mat1 + (m_itr * row_stride1));
      p_vec1_0 = (const WORD16 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L64_IP(acc_row0_vec0, (ae_int64 *)p_bias, 8);
      }

      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int16*)p_vec1_0
         ,cols1
        );

      ae_int32x2 out32_0;
      ae_int16x4 out0;
      MPY_BY_QUANT_MULT_ACC64_OUT32(out32_0, acc_row0_vec0, out_multiplier, out_shift);
      
      AE_MINMAX32(out32_0, min_int32, max_int32);
      
      out0 = AE_SAT16X4(out32_0, out32_0);
      *(ae_int16 *)p_out++ = out0;
    }
  }
  else
  {
    for(m_itr = 0; m_itr < (rows & ~(64 - 1)); m_itr += 64)
    {
      WORD16* p_dst_0 = (WORD16*)(p_out + m_itr);
      for(ii = 0; ii < 16; ii++)
      {
        ae_int64 acc_row0_vec0 = AE_ZERO64();
        ae_int64 acc_row1_vec0 = AE_ZERO64();
        ae_int64 acc_row2_vec0 = AE_ZERO64();
        ae_int64 acc_row3_vec0 = AE_ZERO64();

        if(bias_flag)
        {
          /* Load bias in the accumulator */
          ae_int64 *ae_p_bias = (ae_int64 *)&p_bias[m_itr + ii];
          acc_row0_vec0 = AE_L64_X(ae_p_bias, 0);
          acc_row1_vec0 = AE_L64_X(ae_p_bias, 16*8);
          acc_row2_vec0 = AE_L64_X(ae_p_bias, 32*8);
          acc_row3_vec0 = AE_L64_X(ae_p_bias, 48*8);
        }

        p_mat1_0 = (const WORD8 *)(p_mat1+((m_itr + ii) * row_stride1));
        p_vec1_0 = (const WORD16 *)(p_vec1);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,&acc_row2_vec0
           ,&acc_row3_vec0
           ,(ae_int8x16*)p_mat1_0
           ,(ae_int16x8*)p_vec1_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out0;
#if XCHAL_HAVE_HIFI5S
        MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16_HIFI5S(out0, acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else        
        MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16(out0, acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0, out_multiplier, out_shift);
#endif
        AE_MINMAX16(out0, min_int16, max_int16);
      
        *(ae_int16 *)(p_dst_0 + 48) = out0;
        *(ae_int16 *)(p_dst_0 + 32) = AE_SEL16_4321(out0, out0);
        *(ae_int16 *)(p_dst_0 + 16) = AE_SEL16_5432(out0, out0);
        *(ae_int16 *)p_dst_0++ = AE_SEL16_6543(out0, out0);
      }
    }

    WORD16* p_dst_0 = (WORD16*)(p_out + m_itr);
    for(m_itr = (rows & (~(64-1))); m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int64 acc_row0_vec0 = AE_ZERO64();
      ae_int64 acc_row1_vec0 = AE_ZERO64();
      ae_int64 acc_row2_vec0 = AE_ZERO64();
      ae_int64 acc_row3_vec0 = AE_ZERO64();

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        ae_int64 *p_ae_bias = (ae_int64 *)&p_bias[m_itr];
        acc_row0_vec0 = AE_L64_I(p_ae_bias, 0);
        acc_row1_vec0 = AE_L64_I(p_ae_bias, 1*8);
        acc_row2_vec0 = AE_L64_I(p_ae_bias, 2*8);
        acc_row3_vec0 = AE_L64_I(p_ae_bias, 3*8);
      }

        p_mat1_0 = (WORD8 *)(p_mat1 + (m_itr * row_stride1));
        p_vec1_0 = (WORD16 *)(p_vec1);

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,&acc_row2_vec0
           ,&acc_row3_vec0
           ,(ae_int8x16*)p_mat1_0
           ,(ae_int16*)p_vec1_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out0;
#if XCHAL_HAVE_HIFI5S
        MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16_HIFI5S(out0, acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else        
        MPY_BY_QUANT_MULT_ACC64_X2X2_OUT16(out0, acc_row0_vec0, acc_row1_vec0, acc_row2_vec0, acc_row3_vec0, out_multiplier, out_shift);
#endif
        AE_MINMAX16(out0, min_int16, max_int16);
    
        *(ae_int16 *)p_dst_0++ = AE_SEL16_6543(out0, out0);
        *(ae_int16 *)p_dst_0++ = AE_SEL16_5432(out0, out0);
        *(ae_int16 *)p_dst_0++ = AE_SEL16_4321(out0, out0);
        *(ae_int16 *)p_dst_0++ = out0;
    }
    /* Compute last (rows % 4) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int64 acc_row0_vec0 = AE_ZERO64();
      p_mat1_0 = (const WORD8 *)(p_mat1 + (m_itr * row_stride1));
      p_vec1_0 = (const WORD16 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = *(ae_int64 *)(&p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int16*)p_vec1_0
         ,cols1
        );

      ae_int32x2 out32_0;
      ae_int16x4 out0;
      MPY_BY_QUANT_MULT_ACC64_OUT32(out32_0, acc_row0_vec0, out_multiplier, out_shift);
      AE_MINMAX32(out32_0, min_int32, max_int32);
      out0 = AE_SAT16X4(out32_0, out32_0);
      *(ae_int16 *)p_dst_0++ = out0;
    }
  }
  return 0;
}

