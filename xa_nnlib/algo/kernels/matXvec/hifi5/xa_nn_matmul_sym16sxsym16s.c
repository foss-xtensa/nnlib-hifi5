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

static inline void _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
    (ae_int64* out
    ,ae_int16x8* p_mat1_0
    ,ae_int16x8* p_mat1_1
    ,ae_int16x8* p_mat1_2
    ,ae_int16x8* p_mat1_3
    ,ae_int16x8* p_vec_0
    ,ae_int16x8* p_vec_1
    ,ae_int16x8* p_vec_2
    ,ae_int16x8* p_vec_3
    ,WORD32      cols
    )
{
  
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec1_batch_0, vec1_batch_1;
  ae_int16x4 vec2_batch_0, vec2_batch_1;
  ae_int16x4 vec3_batch_0, vec3_batch_1;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 mat1_row2_0, mat1_row2_1;
  ae_int16x4 mat1_row3_0, mat1_row3_1;

  int align_offset = ((unsigned int)p_mat1_0 & 7);
  pre_loop_count = align_offset != 0 ? (8 - align_offset) >> 1 : 0;
  pre_loop_count = (cols < pre_loop_count) ? cols : pre_loop_count;

  loop_count = cols - pre_loop_count;
  post_loop_count = (loop_count & 7);

  ae_valignx2 align_0;
  ae_valignx2 align_1;
  ae_valignx2 align_2;
  ae_valignx2 align_3;

  ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;
  ae_int64 acc64_01, acc64_11, acc64_21, acc64_31;
  ae_int64 acc64_02, acc64_12, acc64_22, acc64_32;
  ae_int64 acc64_03, acc64_13, acc64_23, acc64_33;
  ae_int64x2 *p_out = (ae_int64x2 *)out;

  AE_L64X2_I(acc64_20, acc64_30, p_out, 16);
  AE_L64X2_IP(acc64_00, acc64_10, p_out, 32);

  AE_L64X2_I(acc64_21, acc64_31, p_out, 16);
  AE_L64X2_IP(acc64_01, acc64_11, p_out, 32);

  AE_L64X2_I(acc64_22, acc64_32, p_out, 16);
  AE_L64X2_IP(acc64_02, acc64_12, p_out, 32);

  AE_L64X2_I(acc64_23, acc64_33, p_out, 16);
  AE_L64X2_IP(acc64_03, acc64_13, p_out, 32);

  /* Pre loop computation */
  if(pre_loop_count)
  {
    align_0 = AE_LA128_PP(p_mat1_0);
    align_1 = AE_LA128_PP(p_mat1_1);
    align_2 = AE_LA128_PP(p_mat1_2);
    align_3 = AE_LA128_PP(p_mat1_3);

    AE_LAV16X4X2_XP(mat1_row0_0, mat1_row0_1, align_0, p_mat1_0, pre_loop_count*2);
    AE_LAV16X4X2_XP(mat1_row1_0, mat1_row1_1, align_1, p_mat1_1, pre_loop_count*2);
    AE_LAV16X4X2_XP(mat1_row2_0, mat1_row2_1, align_2, p_mat1_2, pre_loop_count*2);
    AE_LAV16X4X2_XP(mat1_row3_0, mat1_row3_1, align_3, p_mat1_3, pre_loop_count*2);

    align_0 = AE_LA128_PP(p_vec_0);
    align_1 = AE_LA128_PP(p_vec_1);
    align_2 = AE_LA128_PP(p_vec_2);
    align_3 = AE_LA128_PP(p_vec_3);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_0, p_vec_0, pre_loop_count*2);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_1, p_vec_1, pre_loop_count*2);
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_1, align_2, p_vec_2, pre_loop_count*2);
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_1, align_3, p_vec_3, pre_loop_count*2);

    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_20, acc64_30, vec0_batch_0, vec0_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_0, vec1_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_21, acc64_31, vec1_batch_0, vec1_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_02, acc64_12, vec2_batch_0, vec2_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_22, acc64_32, vec2_batch_0, vec2_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_03, acc64_13, vec3_batch_0, vec3_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_23, acc64_33, vec3_batch_0, vec3_batch_0, mat1_row2_0, mat1_row3_0);
  }

  ae_valignx2 align_p_vec_0, align_p_vec_1, align_p_vec_2, align_p_vec_3;
  align_p_vec_0 = AE_LA128_PP(p_vec_0);
  align_p_vec_1 = AE_LA128_PP(p_vec_1);
  align_p_vec_2 = AE_LA128_PP(p_vec_2);
  align_p_vec_3 = AE_LA128_PP(p_vec_3);

#pragma no_unroll
  for(c_itr = 0; c_itr < (loop_count >> 3); c_itr++)
  {
    mat1_row0_1 = AE_L16X4_I((ae_int16x4 *)p_mat1_0, 8);
    mat1_row1_1 = AE_L16X4_I((ae_int16x4 *)p_mat1_1, 8);
    mat1_row2_1 = AE_L16X4_I((ae_int16x4 *)p_mat1_2, 8);
    mat1_row3_1 = AE_L16X4_I((ae_int16x4 *)p_mat1_3, 8);

    AE_L16X4_IP(mat1_row0_0, (ae_int16x4 *)p_mat1_0, 16);
    AE_L16X4_IP(mat1_row1_0, (ae_int16x4 *)p_mat1_1, 16);
    AE_L16X4_IP(mat1_row2_0, (ae_int16x4 *)p_mat1_2, 16);
    AE_L16X4_IP(mat1_row3_0, (ae_int16x4 *)p_mat1_3, 16);

    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1);
    AE_LA16X4X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec_2, p_vec_2);
    AE_LA16X4X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec_3, p_vec_3);

    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_20, acc64_30, vec0_batch_0, vec0_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_0, vec1_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_21, acc64_31, vec1_batch_0, vec1_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_02, acc64_12, vec2_batch_0, vec2_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_22, acc64_32, vec2_batch_0, vec2_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_03, acc64_13, vec3_batch_0, vec3_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_23, acc64_33, vec3_batch_0, vec3_batch_0, mat1_row2_0, mat1_row3_0);

    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_1, vec0_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_20, acc64_30, vec0_batch_1, vec0_batch_1, mat1_row2_1, mat1_row3_1);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_1, vec1_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_21, acc64_31, vec1_batch_1, vec1_batch_1, mat1_row2_1, mat1_row3_1);
    
    AE_MULAAAA2Q16(acc64_02, acc64_12, vec2_batch_1, vec2_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_22, acc64_32, vec2_batch_1, vec2_batch_1, mat1_row2_1, mat1_row3_1);
    
    AE_MULAAAA2Q16(acc64_03, acc64_13, vec3_batch_1, vec3_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_23, acc64_33, vec3_batch_1, vec3_batch_1, mat1_row2_1, mat1_row3_1);
  }

  if(post_loop_count)
  {
    align_0 = AE_LA128_PP(p_mat1_0);
    align_1 = AE_LA128_PP(p_mat1_1);
    align_2 = AE_LA128_PP(p_mat1_2);
    align_3 = AE_LA128_PP(p_mat1_3);

    AE_LAV16X4X2_XP(mat1_row0_0, mat1_row0_1, align_0, p_mat1_0, post_loop_count*2);
    AE_LAV16X4X2_XP(mat1_row1_0, mat1_row1_1, align_1, p_mat1_1, post_loop_count*2);
    AE_LAV16X4X2_XP(mat1_row2_0, mat1_row2_1, align_2, p_mat1_2, post_loop_count*2);
    AE_LAV16X4X2_XP(mat1_row3_0, mat1_row3_1, align_3, p_mat1_3, post_loop_count*2);

    align_0 = AE_LA128_PP(p_vec_0);
    align_1 = AE_LA128_PP(p_vec_1);
    align_2 = AE_LA128_PP(p_vec_2);
    align_3 = AE_LA128_PP(p_vec_3);

    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_0, p_vec_0, post_loop_count*2);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_1, p_vec_1, post_loop_count*2);
    AE_LAV16X4X2_XP(vec2_batch_0, vec2_batch_1, align_2, p_vec_2, post_loop_count*2);
    AE_LAV16X4X2_XP(vec3_batch_0, vec3_batch_1, align_3, p_vec_3, post_loop_count*2);

    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_20, acc64_30, vec0_batch_0, vec0_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_0, vec1_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_21, acc64_31, vec1_batch_0, vec1_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_02, acc64_12, vec2_batch_0, vec2_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_22, acc64_32, vec2_batch_0, vec2_batch_0, mat1_row2_0, mat1_row3_0);
    
    AE_MULAAAA2Q16(acc64_03, acc64_13, vec3_batch_0, vec3_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_23, acc64_33, vec3_batch_0, vec3_batch_0, mat1_row2_0, mat1_row3_0);

    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_1, vec0_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_20, acc64_30, vec0_batch_1, vec0_batch_1, mat1_row2_1, mat1_row3_1);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_1, vec1_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_21, acc64_31, vec1_batch_1, vec1_batch_1, mat1_row2_1, mat1_row3_1);
    
    AE_MULAAAA2Q16(acc64_02, acc64_12, vec2_batch_1, vec2_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_22, acc64_32, vec2_batch_1, vec2_batch_1, mat1_row2_1, mat1_row3_1);
    
    AE_MULAAAA2Q16(acc64_03, acc64_13, vec3_batch_1, vec3_batch_1, mat1_row0_1, mat1_row1_1);
    AE_MULAAAA2Q16(acc64_23, acc64_33, vec3_batch_1, vec3_batch_1, mat1_row2_1, mat1_row3_1);
  }

  p_out = (ae_int64x2 *)out;

  AE_S64X2_I(acc64_20, acc64_30, p_out, 16);
  AE_S64X2_IP(acc64_00, acc64_10, p_out, 32);

  AE_S64X2_I(acc64_21, acc64_31, p_out, 16);
  AE_S64X2_IP(acc64_01, acc64_11, p_out, 32);

  AE_S64X2_I(acc64_22, acc64_32, p_out, 16);
  AE_S64X2_IP(acc64_02, acc64_12, p_out, 32);

  AE_S64X2_I(acc64_23, acc64_33, p_out, 16);
  AE_S64X2_IP(acc64_03, acc64_13, p_out, 32);
}

static inline void _xa_nn_dot_product_2_rows_2_vecs_unaligned
    (ae_int64* out
    ,ae_int16x8* p_mat1_0
    ,ae_int16x8* p_mat1_1
    ,ae_int16x8* p_vec_0
    ,ae_int16x8* p_vec_1
    ,WORD32      cols
    )
{
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec1_batch_0, vec1_batch_1;
  
  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
  ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
  
  ae_int64 acc64_00, acc64_01;
  ae_int64 acc64_10, acc64_11;

  ae_int64x2 *p_out = (ae_int64x2 *)out;
  
  AE_L64X2_IP(acc64_00, acc64_10, p_out, 16);
  AE_L64X2_IP(acc64_01, acc64_11, p_out, 16);
  
  int rem_cols = cols & 7;
  for(c_itr = 0; c_itr < cols>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1);
    
    AE_LA16X4X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA16X4X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, p_mat1_1);
    
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_1, vec0_batch_1, mat1_row0_1, mat1_row1_1);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_0, vec1_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_1, vec1_batch_1, mat1_row0_1, mat1_row1_1);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, rem_cols*2);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1, rem_cols*2);
    
    AE_LAV16X4X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0, rem_cols*2);
    AE_LAV16X4X2_XP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, p_mat1_1, rem_cols*2);
    
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_1, vec0_batch_1, mat1_row0_1, mat1_row1_1);
    
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_0, vec1_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_01, acc64_11, vec1_batch_1, vec1_batch_1, mat1_row0_1, mat1_row1_1);
  }
  
  p_out = (ae_int64x2 *)out;
  
  AE_S64X2_IP(acc64_00, acc64_10, p_out, 16);
  AE_S64X2_IP(acc64_01, acc64_11, p_out, 16);
}

static inline void _xa_nn_dot_product_2_rows_1_vecs_unaligned
    (ae_int64* out_00
    ,ae_int64* out_10
    ,ae_int16x8* p_mat1_0
    ,ae_int16x8* p_mat1_1
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    )
{
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 vec0_batch_0, vec0_batch_1;

  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = *out_10;

  int rem_cols = cols & 7;
  for(c_itr = 0; c_itr < cols>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA16X4X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, p_mat1_1);
    
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_1, vec0_batch_1, mat1_row0_1, mat1_row1_1);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, rem_cols*2);
    AE_LAV16X4X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0, rem_cols*2);
    AE_LAV16X4X2_XP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, p_mat1_1, rem_cols*2);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_0, mat1_row0_0, mat1_row1_0);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_1, vec0_batch_1, mat1_row0_1, mat1_row1_1);
  }

  *out_00 = acc64_00;
  *out_10 = acc64_10;
}

static inline void _xa_nn_dot_product_1_rows_2_vecs_unaligned
    (ae_int64* out_00
    ,ae_int64* out_01
    ,ae_int16x8* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,ae_int16x8* p_vec_1
    ,WORD32      cols
    )
{
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 vec0_batch_0, vec0_batch_1;
  ae_int16x4 vec1_batch_0, vec1_batch_1;
  
  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
  ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
  
  ae_int64 acc64_00, acc64_01;
  
  acc64_00 = *out_00;
  acc64_01 = *out_01;
  
  int rem_cols = cols & 7;
  for(c_itr = 0; c_itr < cols>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1);
    AE_LA16X4X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0);
    
    AE_MULAAAA2Q16(acc64_00, acc64_01, vec0_batch_0, vec1_batch_0, mat1_row0_0, mat1_row0_0);
    AE_MULAAAA2Q16(acc64_00, acc64_01, vec0_batch_1, vec1_batch_1, mat1_row0_1, mat1_row0_1);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, rem_cols*2);
    AE_LAV16X4X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec_1, p_vec_1, rem_cols*2);
    AE_LAV16X4X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0, rem_cols*2);
    
    AE_MULAAAA2Q16(acc64_00, acc64_01, vec0_batch_0, vec1_batch_0, mat1_row0_0, mat1_row0_0);
    AE_MULAAAA2Q16(acc64_00, acc64_01, vec0_batch_1, vec1_batch_1, mat1_row0_1, mat1_row0_1);
  }
  
  *out_00 = acc64_00;
  *out_01 = acc64_01;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int64* out_00
    ,ae_int16x8* p_mat1_0
    ,ae_int16x8* p_vec_0
    ,WORD32      cols
    )
{
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 vec0_batch_0, vec0_batch_1;

  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);;
  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

  ae_int64 acc64_00 = *out_00;
  ae_int64 acc64_10 = AE_ZERO64();

  int rem_cols = cols & 7;
  for(c_itr = 0; c_itr < cols>>3; c_itr++)
  {
    AE_LA16X4X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0);
    AE_LA16X4X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0, mat1_row0_1);
  }

  //Remainder loop for cols
  if(rem_cols)
  {
    AE_LAV16X4X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec_0, p_vec_0, rem_cols*2);
    AE_LAV16X4X2_XP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat1_0, rem_cols*2);
    AE_MULAAAA2Q16(acc64_00, acc64_10, vec0_batch_0, vec0_batch_1, mat1_row0_0, mat1_row0_1);
  }

  *out_00 = AE_ADD64(acc64_00, acc64_10);
}

WORD32 xa_nn_matmul_sym16sxsym16s_sym16s(
  WORD16 * __restrict__ p_out,
  const WORD16 * __restrict__ p_mat1,
  const WORD16 * __restrict__ p_vec1,
  const WORD64 * __restrict__ p_bias,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);
  
  ae_int64 acc_buffer[4];
  int m_itr, vec_itr, ii;
  ae_int64x2 *acc_buff = (ae_int64x2 *)acc_buffer;
  
  for(m_itr = 0; m_itr < (rows & ~(16-1)); m_itr += 16)
  {
    for(ii = 0; ii < 4; ii++)
    {
      ae_int64 d_bias0 = AE_ZERO64(), d_bias1 = AE_ZERO64();
      ae_int64 d_bias2 = AE_ZERO64(), d_bias3 = AE_ZERO64();
      if(p_bias)
      {
        d_bias0 = *(ae_int64 *)&p_bias[m_itr + ii + 0];
        d_bias1 = *(ae_int64 *)&p_bias[m_itr + ii + 4];
        d_bias2 = *(ae_int64 *)&p_bias[m_itr + ii + 8];
        d_bias3 = *(ae_int64 *)&p_bias[m_itr + ii + 12];
      }
      AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
      AE_S64X2_I(d_bias2, d_bias3, acc_buff, 16);
      
      WORD16* p_dst_0 = (WORD16*)p_out + (m_itr + ii +  0) * out_stride;
      WORD16* p_dst_1 = (WORD16*)p_out + (m_itr + ii +  4) * out_stride;
      WORD16* p_dst_2 = (WORD16*)p_out + (m_itr + ii +  8) * out_stride;
      WORD16* p_dst_3 = (WORD16*)p_out + (m_itr + ii + 12) * out_stride;
      for(vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr+=4)
      {
        ae_int64 ALIGN(16) acc64[16];
        ae_int64x2 *p_acc64 = (ae_int64x2 *)acc64;

        /* Initialize accumulators */
        AE_S64X2_IP(d_bias0, d_bias1, p_acc64, 16);
        AE_S64X2_IP(d_bias2, d_bias3, p_acc64, 16);
        AE_S64X2_IP(d_bias0, d_bias1, p_acc64, 16);
        AE_S64X2_IP(d_bias2, d_bias3, p_acc64, 16);
        AE_S64X2_IP(d_bias0, d_bias1, p_acc64, 16);
        AE_S64X2_IP(d_bias2, d_bias3, p_acc64, 16);
        AE_S64X2_IP(d_bias0, d_bias1, p_acc64, 16);
        AE_S64X2_IP(d_bias2, d_bias3, p_acc64, 16);
        
        ae_int16x8 *p_vec_0 = (ae_int16x8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int16x8 *p_vec_1 = (ae_int16x8 *)(p_vec1 + (vec_itr + 1 ) * vec_offset);
        ae_int16x8 *p_vec_2 = (ae_int16x8 *)(p_vec1 + (vec_itr + 2 ) * vec_offset);
        ae_int16x8 *p_vec_3 = (ae_int16x8 *)(p_vec1 + (vec_itr + 3 ) * vec_offset);
        
        ae_int16x8 *p_mat1_0 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1];
        ae_int16x8 *p_mat1_1 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 4) * row_stride1];
        ae_int16x8 *p_mat1_2 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 8) * row_stride1];
        ae_int16x8 *p_mat1_3 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 12) * row_stride1];

        _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
          (&acc64[0]
           ,p_mat1_0
           ,p_mat1_1
           ,p_mat1_2
           ,p_mat1_3
           ,p_vec_0
           ,p_vec_1
           ,p_vec_2
           ,p_vec_3
           ,cols1
          );
        
        ae_int32x2 acc_row01_vec0, acc_row23_vec0;
        ae_int32x2 acc_row01_vec1, acc_row23_vec1;
        ae_int32x2 acc_row01_vec2, acc_row23_vec2;
        ae_int32x2 acc_row01_vec3, acc_row23_vec3;
        ae_int16x4 out_0, out_1, out_2, out_3;

#if XCHAL_HAVE_HIFI5S
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64[0], acc64[1], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec0, acc64[2], acc64[3], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec1, acc64[4], acc64[5], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec1, acc64[6], acc64[7], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec2, acc64[8], acc64[9], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec2, acc64[10], acc64[11], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec3, acc64[12], acc64[13], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec3, acc64[14], acc64[15], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64[0], acc64[1], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec0, acc64[2], acc64[3], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec1, acc64[4], acc64[5], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec1, acc64[6], acc64[7], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec2, acc64[8], acc64[9], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec2, acc64[10], acc64[11], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec3, acc64[12], acc64[13], out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec3, acc64[14], acc64[15], out_multiplier, out_shift);
#endif

        out_0 = AE_SAT16X4(acc_row01_vec0, acc_row23_vec0);
        out_1 = AE_SAT16X4(acc_row01_vec1, acc_row23_vec1);
        out_2 = AE_SAT16X4(acc_row01_vec2, acc_row23_vec2);
        out_3 = AE_SAT16X4(acc_row01_vec3, acc_row23_vec3);
        
        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;
        *p_dst_3 = (out_0);                       p_dst_3 += out_offset;
        
        *p_dst_0 = AE_SEL16_6543(out_1, out_1);   p_dst_0 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_1, out_1);   p_dst_1 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_1, out_1);   p_dst_2 += out_offset;
        *p_dst_3 = (out_1);                       p_dst_3 += out_offset;
        
        *p_dst_0 = AE_SEL16_6543(out_2, out_2);   p_dst_0 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_2, out_2);   p_dst_1 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_2, out_2);   p_dst_2 += out_offset;
        *p_dst_3 = (out_2);                       p_dst_3 += out_offset;
        
        *p_dst_0 = AE_SEL16_6543(out_3, out_3);   p_dst_0 += out_offset;
        *p_dst_1 = AE_SEL16_5432(out_3, out_3);   p_dst_1 += out_offset;
        *p_dst_2 = AE_SEL16_4321(out_3, out_3);   p_dst_2 += out_offset;
        *p_dst_3 = (out_3); p_dst_3 += out_offset;
      }
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int64 acc64_00, acc64_10, acc64_20, acc64_30;

        /* Initialize accumulators */
        AE_L64X2_I(acc64_00, acc64_10, acc_buff, 0);
        AE_L64X2_I(acc64_20, acc64_30, acc_buff, 16);

        ae_int16* p_vec_0  = (ae_int16 *)(p_vec1 + vec_itr * vec_offset);
        
        ae_int16x8 *p_mat1_0 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1];
        ae_int16x8 *p_mat1_1 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 4) * row_stride1];
        ae_int16x8 *p_mat1_2 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 8) * row_stride1];
        ae_int16x8 *p_mat1_3 = (ae_int16x8 *) &p_mat1[(m_itr + ii + 12) * row_stride1];

        _xa_nn_dot_product_2_rows_1_vecs_unaligned
          (&acc64_00
           ,&acc64_10
           ,p_mat1_0
           ,p_mat1_1
           ,p_vec_0
           ,cols1
          );
          _xa_nn_dot_product_2_rows_1_vecs_unaligned
          (&acc64_20
           ,&acc64_30
           ,p_mat1_2
           ,p_mat1_3
           ,p_vec_0
           ,cols1
          );
        ae_int16x4 out_0;
        ae_int32x2 acc_row01_vec0, acc_row23_vec0;
        
#if XCHAL_HAVE_HIFI5S
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, out_shift);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, out_shift);
#endif
        out_0 = AE_SAT16X4(acc_row01_vec0, acc_row23_vec0);
        *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;

        *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;

        *p_dst_2 = AE_SEL16_4321(out_0, out_0);   p_dst_2 += out_offset;

        *p_dst_3 = (out_0);                       p_dst_3 += out_offset;
      }
    }
  }
  
  for(; m_itr < (rows & ~(2-1)); m_itr+=2)
  {
    ae_int64 d_bias0 = AE_ZERO64();
    ae_int64 d_bias1 = AE_ZERO64();
    if(p_bias)
    {
      d_bias0 = *(ae_int64 *)&p_bias[m_itr];
      d_bias1 = *(ae_int64 *)&p_bias[m_itr+1];
    }
    AE_S64X2_I(d_bias0, d_bias1, acc_buff, 0);
    
    WORD16* p_dst_0 = (WORD16*)p_out + m_itr * out_stride;
    WORD16* p_dst_1 = (WORD16*)p_out + (m_itr + 1) * out_stride;
    
    ae_int16x8* p_mat1_0 = (ae_int16x8 *)(p_mat1 + m_itr * row_stride1);
    ae_int16x8* p_mat1_1 = (ae_int16x8 *)(p_mat1 + (m_itr + 1) * row_stride1);
    
    for(vec_itr = 0; vec_itr < (vec_count & ~(2 - 1)); vec_itr+=2)
    {
      ae_int32x2 acc_row01_vec0;
      ae_int32x2 acc_row01_vec1;
      
      ae_int64 ALIGN(16) acc64[4];
      ae_int64x2 *p_acc64 = (ae_int64x2 *)acc64;
      
      AE_S64X2_IP(d_bias0, d_bias1, p_acc64, 16);
      AE_S64X2_IP(d_bias0, d_bias1, p_acc64, 16);
      
      ae_int16x8* p_vec_0 = (ae_int16x8 *)(p_vec1 + vec_itr * vec_offset);
      ae_int16x8* p_vec_1 = (ae_int16x8 *)(p_vec1 + (vec_itr + 1) * vec_offset);
      
      _xa_nn_dot_product_2_rows_2_vecs_unaligned
      (&acc64[0]
       ,p_mat1_0
       ,p_mat1_1
       ,p_vec_0
       ,p_vec_1
       ,cols1
      );

      ae_int16x4 out_0;
#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64[0], acc64[1], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec1, acc64[2], acc64[3], out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64[0], acc64[1], out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec1, acc64[2], acc64[3], out_multiplier, out_shift);
#endif

      out_0 = AE_SAT16X4(acc_row01_vec0, acc_row01_vec1);
      
      *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
      *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
      
      *p_dst_0 = AE_SEL16_4321(out_0, out_0);   p_dst_0 += out_offset;
      *p_dst_1 = (out_0);                       p_dst_1 += out_offset;
      
    }
    if(vec_itr < vec_count)
    {
      ae_int32x2 acc_row01_vec0;
      ae_int64 acc64_00, acc64_10;
      
      AE_L64X2_I(acc64_00, acc64_10, acc_buff, 0);
      
      ae_int16x8* p_vec_0 = (ae_int16x8 *)(p_vec1 + vec_itr * vec_offset);
      
      _xa_nn_dot_product_2_rows_1_vecs_unaligned
      (&acc64_00
       ,&acc64_10
       ,p_mat1_0
       ,p_mat1_1
       ,p_vec_0
       ,cols1
      );

      ae_int16x4 out_0;

#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, out_shift);
#endif
      out_0 = AE_SAT16X4(acc_row01_vec0, acc_row01_vec0);
      
      *p_dst_0 = AE_SEL16_6543(out_0, out_0);   p_dst_0 += out_offset;
      *p_dst_1 = AE_SEL16_5432(out_0, out_0);   p_dst_1 += out_offset;
    }
  }
  for(; m_itr < rows; m_itr++)
  {
    ae_int64 d_bias0 = AE_ZERO64();
    
    if(p_bias)
    {
      d_bias0 = *(ae_int64 *)&p_bias[m_itr];
    }
    AE_S64X2_I(d_bias0, d_bias0, acc_buff, 0);
    
    WORD16* p_dst_0 = (WORD16*)p_out + m_itr * out_stride;
    
    ae_int16x8* p_mat1_0 = (ae_int16x8 *)(p_mat1 + m_itr * row_stride1);
    
    for(vec_itr = 0; vec_itr < (vec_count & ~(2-1)); vec_itr+=2)
    {
      ae_int32x2 acc_row0_vec01;
      ae_int64 acc64_00, acc64_01;
      
      AE_L64X2_IP(acc64_00, acc64_01, (ae_int64x2 *)acc_buff, 0);
      
      ae_int16x8* p_vec_0 = (ae_int16x8 *)(p_vec1 + vec_itr * vec_offset);
      ae_int16x8* p_vec_1 = (ae_int16x8 *)(p_vec1 + (vec_itr + 1 ) * vec_offset);
      
      _xa_nn_dot_product_1_rows_2_vecs_unaligned
      (&acc64_00
       ,&acc64_01
       ,p_mat1_0
       ,p_vec_0
       ,p_vec_1
       ,cols1
      );

      ae_int16x4 out_0;

#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row0_vec01, acc64_00, acc64_01, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec01, acc64_00, acc64_01, out_multiplier, out_shift);
#endif
      out_0 = AE_SAT16X4(acc_row0_vec01, acc_row0_vec01);
      
      *p_dst_0 = AE_SEL16_4321(out_0, out_0);   p_dst_0 += out_offset;
      *p_dst_0 = (out_0);  p_dst_0 += out_offset;
    }
    
    for(; vec_itr < vec_count; vec_itr++)
    {
      ae_int32x2 acc_row0_vec0;
      ae_int64 acc64_00;
      
      AE_L64_IP(acc64_00, (ae_int64 *)acc_buff, 0);
      
      ae_int16x8* p_vec_0 = (ae_int16x8 *)(p_vec1 + vec_itr * vec_offset);
      
      _xa_nn_dot_product_1_rows_1_vecs_unaligned
      (&acc64_00
       ,p_mat1_0
       ,p_vec_0
       ,cols1
      );

      ae_int16x4 out_0;

#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row0_vec0, acc64_00, acc64_00, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row0_vec0, acc64_00, acc64_00, out_multiplier, out_shift);
#endif
      out_0 = AE_SAT16X4(acc_row0_vec0, acc_row0_vec0);

      *p_dst_0 = (out_0);
      
      p_dst_0 += out_offset;
    }
  }
  return 0;
}