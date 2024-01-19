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
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

extern const long long g_sel_pattern[16];

static inline void _xa_nn_dot_product_4_rows_4_vecs_aligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int64* out_4
    ,ae_int64* out_5
    ,ae_int64* out_6
    ,ae_int64* out_7
    ,ae_int64* out_8
    ,ae_int64* out_9
    ,ae_int64* out_10
    ,ae_int64* out_11
    ,ae_int64* out_12
    ,ae_int64* out_13
    ,ae_int64* out_14
    ,ae_int64* out_15
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 mat1_row2_0, mat1_row2_1;
  ae_int16x4 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;
  ae_int8x8 vec2_batch_0;
  ae_int8x8 vec3_batch_0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD16));

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;
  ae_int64 acc_2  = *out_2;
  ae_int64 acc_3  = *out_3;
  ae_int64 acc_4  = *out_4;
  ae_int64 acc_5  = *out_5;
  ae_int64 acc_6  = *out_6;
  ae_int64 acc_7  = *out_7;
  ae_int64 acc_8  = *out_8;
  ae_int64 acc_9  = *out_9;
  ae_int64 acc_10  = *out_10;
  ae_int64 acc_11  = *out_11;
  ae_int64 acc_12  = *out_12;
  ae_int64 acc_13  = *out_13;
  ae_int64 acc_14  = *out_14;
  ae_int64 acc_15  = *out_15;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  int cols_count = cols -(cols & 7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);
    AE_L16X4X2_XC(mat1_row2_0, mat1_row2_1, (ae_int16x8 *)p_mat1_2, 16);
    AE_L16X4X2_XC(mat1_row3_0, mat1_row3_1, (ae_int16x8 *)p_mat1_3, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

    AE_MULA8QW8X16(acc_0  ,  acc_1  ,  acc_2  ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4  ,  acc_5  ,  acc_6  ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
    AE_MULA8QW8X16(acc_8  ,  acc_9  ,  acc_10 ,  acc_11 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0 , mat1_row2_1);
    AE_MULA8QW8X16(acc_12 ,  acc_13 ,  acc_14 ,  acc_15 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row3_0 , mat1_row3_1);
  }

  //Remainder loop for cols
  int rem_shift = rem_cols_shift;
  if (cols_count != cols)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);
    AE_L16X4X2_XC(mat1_row2_0, mat1_row2_1, (ae_int16x8 *)p_mat1_2, 16);
    AE_L16X4X2_XC(mat1_row3_0, mat1_row3_1, (ae_int16x8 *)p_mat1_3, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_shift), rem_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_shift), rem_shift));
    vec2_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_batch_0), rem_shift), rem_shift));
    vec3_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_batch_0), rem_shift), rem_shift));

    AE_MULA8QW8X16(acc_0  ,  acc_1  ,  acc_2  ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4  ,  acc_5  ,  acc_6  ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
    AE_MULA8QW8X16(acc_8  ,  acc_9  ,  acc_10 ,  acc_11 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0 , mat1_row2_1);
    AE_MULA8QW8X16(acc_12 ,  acc_13 ,  acc_14 ,  acc_15 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row3_0 , mat1_row3_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
  *out_2  = acc_2;
  *out_3  = acc_3;
  *out_4  = acc_4;
  *out_5  = acc_5;
  *out_6  = acc_6;
  *out_7  = acc_7;
  *out_8  = acc_8;
  *out_9  = acc_9;
  *out_10  = acc_10;
  *out_11  = acc_11;
  *out_12  = acc_12;
  *out_13  = acc_13;
  *out_14  = acc_14;
  *out_15  = acc_15;
}

static inline void _xa_nn_dot_product_2_rows_4_vecs_aligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int64* out_4
    ,ae_int64* out_5
    ,ae_int64* out_6
    ,ae_int64* out_7
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;
  ae_int8x8 vec2_batch_0;
  ae_int8x8 vec3_batch_0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;
  ae_int64 acc_2  = *out_2;
  ae_int64 acc_3  = *out_3;
  ae_int64 acc_4  = *out_4;
  ae_int64 acc_5  = *out_5;
  ae_int64 acc_6  = *out_6;
  ae_int64 acc_7  = *out_7;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4 ,  acc_5 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  //Remainder loop for cols
  int rem_shift = rem_cols_shift;
  if (cols_count != cols)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_shift), rem_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_shift), rem_shift));
    vec2_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_batch_0), rem_shift), rem_shift));
    vec3_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_batch_0), rem_shift), rem_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4 ,  acc_5 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
  *out_2  = acc_2;
  *out_3  = acc_3;
  *out_4  = acc_4;
  *out_5  = acc_5;
  *out_6  = acc_6;
  *out_7  = acc_7;
}

static inline void _xa_nn_dot_product_2_rows_2_vecs_aligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;
  ae_int64 acc_2  = *out_2;
  ae_int64 acc_3  = *out_3;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    ae_int64 acc_4, acc_5, acc_6, acc_7; /* Dummy acc */
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_4 ,  acc_5  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_2 ,  acc_3 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  //Remainder loop for cols
  int rem_shift = rem_cols_shift;
  if (cols_count != cols)
  {
    ae_int64 acc_4, acc_5, acc_6, acc_7; /* Dummy acc */
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_shift), rem_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_shift), rem_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_4 ,  acc_5  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_2 ,  acc_3 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
  *out_2  = acc_2;
  *out_3  = acc_3;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7)*8;
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 mat1_row2_0, mat1_row2_1;
  ae_int16x4 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD16));

  ae_int64 acc_0 = *out_0;
  ae_int64 acc_1 = *out_1;
  ae_int64 acc_2 = *out_2;
  ae_int64 acc_3 = *out_3;

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);
    AE_L16X4X2_XC(mat1_row2_0, mat1_row2_1, (ae_int16x8 *)p_mat1_2, 16);
    AE_L16X4X2_XC(mat1_row3_0, mat1_row3_1, (ae_int16x8 *)p_mat1_3, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row1_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_1, acc_0, mat1_row1_0, mat1_row0_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_2, acc_3, mat1_row2_0, mat1_row3_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_3, acc_2, mat1_row3_0, mat1_row2_1, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L16X4X2_XC(mat1_row1_0, mat1_row1_1, (ae_int16x8 *)p_mat1_1, 16);
    AE_L16X4X2_XC(mat1_row2_0, mat1_row2_1, (ae_int16x8 *)p_mat1_2, 16);
    AE_L16X4X2_XC(mat1_row3_0, mat1_row3_1, (ae_int16x8 *)p_mat1_3, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row1_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_1, acc_0, mat1_row1_0, mat1_row0_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_2, acc_3, mat1_row2_0, mat1_row3_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_3, acc_2, mat1_row3_0, mat1_row2_1, vec0_batch_0);
  }

  *out_0 = acc_0;
  *out_1 = acc_1;
  *out_2 = acc_2;
  *out_3 = acc_3;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_aligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int16*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int c_itr = 0;

  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int16x4 vec0_batch_0, vec0_batch_1;

  /* p_vec needs to be accessed in circular fashion */
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset);
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset);
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset);

  ae_int64 acc_0 = *out_0;
  ae_int64 acc_1 = *out_1;
  ae_int64 acc_2 = *out_2;
  ae_int64 acc_3 = *out_3;

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L16X4X2_XC(vec0_batch_0, vec0_batch_1, (ae_int16x8 *)p_vec_0, 16);

    AE_MULA8QW8X16(acc_0 , acc_1 , acc_2 , acc_3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec0_batch_0 , vec0_batch_1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L16X4X2_XC(vec0_batch_0, vec0_batch_1, (ae_int16x8 *)p_vec_0, 16);
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift), rem_cols_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift), rem_cols_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8QW8X16(acc_0 , acc_1 , acc_2 , acc_3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec0_batch_0 , vec0_batch_1);
  }

  *out_0 = acc_0;
  *out_1 = acc_1;
  *out_2 = acc_2;
  *out_3 = acc_3;
}

static inline void _xa_nn_dot_product_1_rows_2_vecs_aligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      vec_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int16x4 mat1_row0_0, mat1_row0_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;

  int cols_count = cols -(cols & 7);

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    ae_int64 acc_2, acc_3; /* Dummy acc */
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
  }

  //Remainder loop for cols
  int rem_shift = rem_cols_shift;
  if (cols_count != cols)
  {
    ae_int64 acc_2, acc_3; /* Dummy acc */
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_shift), rem_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_shift), rem_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int64* out_0
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    )
{
  int c_itr = 0;

  ae_int8x8 vec0_batch_0;
  ae_int16x4 mat1_row0_0, mat1_row0_1;

  ae_int64 acc_0 = *out_0;

  int rem_cols_shift = 64 - (cols & 7) * 8;
  int cols_count=cols-(cols&7);
  ae_int64 acc_1 = AE_ZERO64();/* tmp acc */

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);
    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row0_1, vec0_batch_0);
  }
  acc_0 = AE_ADD64S(acc_0, acc_1);

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    ae_int64 acc_1 = AE_ZERO64();/* tmp acc */
    AE_L16X4X2_XC(mat1_row0_0, mat1_row0_1, (ae_int16x8 *)p_mat1_0, 16);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row0_1, vec0_batch_0);
    acc_0 = AE_ADD64S(acc_0, acc_1);
  }

  *out_0 = acc_0;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_unaligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int64* out_4
    ,ae_int64* out_5
    ,ae_int64* out_6
    ,ae_int64* out_7
    ,ae_int64* out_8
    ,ae_int64* out_9
    ,ae_int64* out_10
    ,ae_int64* out_11
    ,ae_int64* out_12
    ,ae_int64* out_13
    ,ae_int64* out_14
    ,ae_int64* out_15
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 mat1_row2_0, mat1_row2_1;
  ae_int16x4 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;
  ae_int8x8 vec2_batch_0;
  ae_int8x8 vec3_batch_0;
  ae_int8x8 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD16));

  ae_valignx2 align_p_mat1_0;
  ae_valignx2 align_p_mat1_1;
  ae_valignx2 align_p_mat1_2;
  ae_valignx2 align_p_mat1_3;

  AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
  AE_LA16X4X2POS_PC(align_p_mat1_1, (ae_int16x8 *)p_mat1_1);
  AE_LA16X4X2POS_PC(align_p_mat1_2, (ae_int16x8 *)p_mat1_2);
  AE_LA16X4X2POS_PC(align_p_mat1_3, (ae_int16x8 *)p_mat1_3);

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;
  ae_int64 acc_2  = *out_2;
  ae_int64 acc_3  = *out_3;
  ae_int64 acc_4  = *out_4;
  ae_int64 acc_5  = *out_5;
  ae_int64 acc_6  = *out_6;
  ae_int64 acc_7  = *out_7;
  ae_int64 acc_8  = *out_8;
  ae_int64 acc_9  = *out_9;
  ae_int64 acc_10  = *out_10;
  ae_int64 acc_11  = *out_11;
  ae_int64 acc_12  = *out_12;
  ae_int64 acc_13  = *out_13;
  ae_int64 acc_14  = *out_14;
  ae_int64 acc_15  = *out_15;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);
  AE_SW_PRIME_64(p_vec_1, align_p_vec1);
  AE_SW_PRIME_64(p_vec_2, align_p_vec2);
  AE_SW_PRIME_64(p_vec_3, align_p_vec3);

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);
    AE_LA16X4X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int16x8 *)p_mat1_2);
    AE_LA16X4X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int16x8 *)p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec1_batch_0, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4 ,  acc_5 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
    AE_MULA8QW8X16(acc_8  ,  acc_9  ,  acc_10 ,  acc_11 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0 , mat1_row2_1);
    AE_MULA8QW8X16(acc_12 ,  acc_13 ,  acc_14 ,  acc_15 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row3_0 , mat1_row3_1);
  }

  //Remainder loop for cols
  if (cols_count != cols)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);
    AE_LA16X4X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int16x8 *)p_mat1_2);
    AE_LA16X4X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int16x8 *)p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec1_batch_0, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_cols_shift), rem_cols_shift));
    vec2_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_batch_0), rem_cols_shift), rem_cols_shift));
    vec3_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4 ,  acc_5 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
    AE_MULA8QW8X16(acc_8  ,  acc_9  ,  acc_10 ,  acc_11 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0 , mat1_row2_1);
    AE_MULA8QW8X16(acc_12 ,  acc_13 ,  acc_14 ,  acc_15 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row3_0 , mat1_row3_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
  *out_2  = acc_2;
  *out_3  = acc_3;
  *out_4  = acc_4;
  *out_5  = acc_5;
  *out_6  = acc_6;
  *out_7  = acc_7;
  *out_8  = acc_8;
  *out_9  = acc_9;
  *out_10  = acc_10;
  *out_11  = acc_11;
  *out_12  = acc_12;
  *out_13  = acc_13;
  *out_14  = acc_14;
  *out_15  = acc_15;
}

static inline void _xa_nn_dot_product_2_rows_4_vecs_unaligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int64* out_4
    ,ae_int64* out_5
    ,ae_int64* out_6
    ,ae_int64* out_7
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;
  ae_int8x8 vec2_batch_0;
  ae_int8x8 vec3_batch_0;
  ae_int8x8 align_p_vec2, align_p_vec3;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));

  ae_valignx2 align_p_mat1_0;
  ae_valignx2 align_p_mat1_1;

  AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
  AE_LA16X4X2POS_PC(align_p_mat1_1, (ae_int16x8 *)p_mat1_1);

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;
  ae_int64 acc_2  = *out_2;
  ae_int64 acc_3  = *out_3;
  ae_int64 acc_4  = *out_4;
  ae_int64 acc_5  = *out_5;
  ae_int64 acc_6  = *out_6;
  ae_int64 acc_7  = *out_7;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valign align_p_vec0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec1 = AE_LA64_PP(p_vec_1);
  AE_SW_PRIME_64(p_vec_2, align_p_vec2);
  AE_SW_PRIME_64(p_vec_3, align_p_vec3);

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4 ,  acc_5 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  //Remainder loop for cols
  int rem_shift = rem_cols_shift;
  if (cols_count != cols)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_shift), rem_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_shift), rem_shift));
    vec2_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_batch_0), rem_shift), rem_shift));
    vec3_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_batch_0), rem_shift), rem_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_4 ,  acc_5 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
  *out_2  = acc_2;
  *out_3  = acc_3;
  *out_4  = acc_4;
  *out_5  = acc_5;
  *out_6  = acc_6;
  *out_7  = acc_7;
}

static inline void _xa_nn_dot_product_2_rows_2_vecs_unaligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));

  ae_valignx2 align_p_mat1_0;
  ae_valignx2 align_p_mat1_1;

  AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
  AE_LA16X4X2POS_PC(align_p_mat1_1, (ae_int16x8 *)p_mat1_1);

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;
  ae_int64 acc_2  = *out_2;
  ae_int64 acc_3  = *out_3;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;

  ae_valign align_p_vec0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec1 = AE_LA64_PP(p_vec_1);

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    ae_int64 acc_4, acc_5, acc_6, acc_7; /* Dummy acc */
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_4 ,  acc_5  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_2 ,  acc_3 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  //Remainder loop for cols
  int rem_shift = rem_cols_shift;
  if (cols_count != cols)
  {
    ae_int64 acc_4, acc_5, acc_6, acc_7; /* Dummy acc */
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_shift), rem_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_shift), rem_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_4 ,  acc_5  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
    AE_MULA8QW8X16(acc_2 ,  acc_3 ,  acc_6 ,  acc_7  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row1_0 , mat1_row1_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
  *out_2  = acc_2;
  *out_3  = acc_3;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int c_itr = 0;

  int rem_cols_shift = 64 - (cols & 7)*8;

  ae_int16x4 mat1_row0_0, mat1_row0_1;
  ae_int16x4 mat1_row1_0, mat1_row1_1;
  ae_int16x4 mat1_row2_0, mat1_row2_1;
  ae_int16x4 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 align_p_vec0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int16x4* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD16));
  ae_int16x4* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD16));

  ae_valignx2 align_p_mat1_0;
  ae_valignx2 align_p_mat1_1;
  ae_valignx2 align_p_mat1_2;
  ae_valignx2 align_p_mat1_3;

  AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
  AE_LA16X4X2POS_PC(align_p_mat1_1, (ae_int16x8 *)p_mat1_1);
  AE_LA16X4X2POS_PC(align_p_mat1_2, (ae_int16x8 *)p_mat1_2);
  AE_LA16X4X2POS_PC(align_p_mat1_3, (ae_int16x8 *)p_mat1_3);

  ae_int64 acc_0 = *out_0;
  ae_int64 acc_1 = *out_1;
  ae_int64 acc_2 = *out_2;
  ae_int64 acc_3 = *out_3;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);
    AE_LA16X4X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int16x8 *)p_mat1_2);
    AE_LA16X4X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int16x8 *)p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row1_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_1, acc_0, mat1_row1_0, mat1_row0_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_2, acc_3, mat1_row2_0, mat1_row3_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_3, acc_2, mat1_row3_0, mat1_row2_1, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA16X4X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int16x8 *)p_mat1_1);
    AE_LA16X4X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int16x8 *)p_mat1_2);
    AE_LA16X4X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int16x8 *)p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row1_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_1, acc_0, mat1_row1_0, mat1_row0_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_2, acc_3, mat1_row2_0, mat1_row3_1, vec0_batch_0);
    AE_MULAAAA2Q16X8(acc_3, acc_2, mat1_row3_0, mat1_row2_1, vec0_batch_0);
  }

  *out_0 = acc_0;
  *out_1 = acc_1;
  *out_2 = acc_2;
  *out_3 = acc_3;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_unaligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int64* out_2
    ,ae_int64* out_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int16*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int c_itr = 0;

  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int16x4 vec0_batch_0;
  ae_int16x4 vec0_batch_1;
  ae_valignx2 align_p_vec0;

  /* p_vec needs to be accessed in circular fashion */
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset);
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset);
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset);

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_int8x8 align_p_mat1_3;
  AE_SW_PRIME_64(p_mat1_3, align_p_mat1_3);

  ae_int64 acc_0 = *out_0;
  ae_int64 acc_1 = *out_1;
  ae_int64 acc_2 = *out_2;
  ae_int64 acc_3 = *out_3;

  AE_LA16X4X2POS_PC(align_p_vec0, (ae_int16x8 *)p_vec_0);

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_LA16X4X2_IC(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int16x8 *)p_vec_0);
    AE_MULA8QW8X16(acc_0 , acc_1 , acc_2 , acc_3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, vec0_batch_1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_LA16X4X2_IC(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int16x8 *)p_vec_0);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift), rem_cols_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift), rem_cols_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8QW8X16(acc_0 , acc_1 , acc_2 , acc_3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0, vec0_batch_1);
  }

  *out_0 = acc_0;
  *out_1 = acc_1;
  *out_2 = acc_2;
  *out_3 = acc_3;
}

static inline void _xa_nn_dot_product_1_rows_2_vecs_unaligned
    (ae_int64* out_0
    ,ae_int64* out_1
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      vec_offset
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int16x4 mat1_row0_0, mat1_row0_1;

  ae_int8x8 vec0_batch_0;
  ae_int8x8 vec1_batch_0;

  ae_valignx2 align_p_mat1_0;

  AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);

  ae_int64 acc_0  = *out_0;
  ae_int64 acc_1  = *out_1;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset;

  ae_valign align_p_vec0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec1 = AE_LA64_PP(p_vec_1);

  int cols_count = cols -(cols & 7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    ae_int64 acc_2, acc_3; /* Dummy acc */
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
  }

  //Remainder loop for cols
  if (cols_count != cols)
  {
    ae_int64 acc_2, acc_3; /* Dummy acc */
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8QW8X16(acc_0 ,  acc_1 ,  acc_2 ,  acc_3  , vec0_batch_0 , vec1_batch_0 , vec0_batch_0 , vec1_batch_0 , mat1_row0_0 , mat1_row0_1);
  }

  *out_0  = acc_0;
  *out_1  = acc_1;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int64* out_0
    ,ae_int16x4*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols & 7) * 8;
  ae_int8x8 vec0_batch_0;
  ae_int16x4 mat1_row0_0;
  ae_int16x4 mat1_row0_1;

  ae_int64 acc_0 = *out_0;

  ae_valignx2 align_p_mat1_0;
  ae_valign align_p_vec0;
  AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
  align_p_vec0 = AE_LA64_PP(p_vec_0);
  int cols_count=cols-(cols&7);
  ae_int64 acc_1 = AE_ZERO64(); /*tmp acc*/

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row0_1, vec0_batch_0);
  }
  acc_0 = AE_ADD64(acc_0, acc_1);

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    ae_int64 acc_1 = AE_ZERO64(); /*tmp acc*/
    AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULAAAA2Q16X8(acc_0, acc_1, mat1_row0_0, mat1_row0_1, vec0_batch_0);
    acc_0 = AE_ADD64S(acc_0, acc_1);
  }

  *out_0 = acc_0;
}

WORD32 xa_nn_matXvec_sym8sxsym16s_sym16s_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD64 * __restrict__ p_bias,
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
  (VOID) mat1_offset;
  (VOID) out_zero_bias;
  /* Iterators used in for loops */
  int m_itr, vec_itr;

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

  /* Special conv2d case when k_h*k_w*i_c < 9, o_c is multiple of 4 and out_data_format = NHWC */
  if(cols1 < 9 && (vec_count & 0x3) == 0 && (out_col_offset == 1))
  {
    int out_stride = out_row_offset*sizeof(WORD16);
    int out_offset = out_col_offset;

    WORD32 *p_out_mult = (WORD32 *)p_out_multiplier;
    ae_valignx2 alignx2_p_mult = AE_LA128_PP((ae_int32x4 *)p_out_multiplier);
    WORD32 *p_out_sh = (WORD32 *)p_out_shift;
    ae_valignx2 alignx2_p_sh = AE_LA128_PP((ae_int32x4 *)p_out_shift);
    
    // Process loop for 4 vectors 
    for(vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      ae_int64 bias_1 = p_bias[vec_itr + 1];
      ae_int64 bias_2 = p_bias[vec_itr + 2];
      ae_int64 bias_3 = p_bias[vec_itr + 3];
      
      WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;
      WORD16* p_dst_2 = (WORD16*)p_out + (vec_itr + 2) * out_offset;
      WORD16* p_dst_3 = (WORD16*)p_out + (vec_itr + 3) * out_offset;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1 /* Dummy reg */; 
      ae_int8x8 vec1_batch_0, vec1_batch_1 /* Dummy reg */; 
      ae_int8x8 vec2_batch_0, vec2_batch_1 /* Dummy reg */; 
      ae_int8x8 vec3_batch_0, vec3_batch_1 /* Dummy reg */;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      AE_LAV8X8X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0, cols1);
      AE_LAV8X8X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1, cols1);
      AE_LAV8X8X2_XP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2, cols1);
      AE_LAV8X8X2_XP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3, cols1);
      
      /* Load shifts and multiplier values */
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_LA32X2X2_IP(out_multiplier_01, out_multiplier_23, alignx2_p_mult, (ae_int32x4 *)p_out_mult);
      ae_int32x2 shift_01, shift_23;
      AE_LA32X2X2_IP(shift_01, shift_23, alignx2_p_sh, (ae_int32x4 *)p_out_sh);
#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 sh_0123 = AE_ADD16(AE_SAT16X4(shift_01, shift_23), AE_MOVDA16(33));
      shift_01 = AE_MOVINT32X2_FROMINT16X4(sh_0123);
      shift_23 = AE_SEL32_LL(shift_01, shift_01);
#endif
      
      for (m_itr = 0; m_itr < rows; m_itr++)
      {
        ae_int64 acc_0 = bias_0;
        ae_int64 acc_1 = bias_1;
        ae_int64 acc_2 = bias_2;
        ae_int64 acc_3 = bias_3;
        
        WORD16 *p_mat1_0 = (WORD16 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        ae_int16x4 mat1_row0_0, mat1_row0_1;
        ae_valignx2 align_p_mat1_0;

        AE_LA16X4X2POS_PC(align_p_mat1_0, (ae_int16x8 *)p_mat1_0);
        AE_LA16X4X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int16x8 *)p_mat1_0);

        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0, mat1_row0_1);

        /* Apply quantization */
#ifdef AE_TRUNCAV32X2F64S
        ae_int16x4 d0;
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2X2_OUT16(d0, acc_0, acc_1, acc_2, acc_3, out_multiplier_01, out_multiplier_23, shift_01, shift_23);
#else  /* AE_TRUNCAV32X2F64S */
        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_0, acc_0, acc_1, out_multiplier_01, shift_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_1, acc_2, acc_3, out_multiplier_23, shift_23);
        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
#endif
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_3, out_stride);
      }
    }
  }
  else if(((((unsigned)p_mat1) & 15) == 0) && ((((unsigned)p_vec1) & 7) == 0) && ((((unsigned)p_bias) & 3) == 0) &&
     ((row_stride1 & 7) == 0) && (vec_stride & 7) == 0)
  {
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset*sizeof(WORD16);
    int out_offset = out_col_offset;
    WORD32 *p_out_mult = (WORD32 *)p_out_multiplier;
    WORD32 *p_out_sh = (WORD32 *)p_out_shift;
    ae_valignx2 alignx2_p_mult = AE_LA128_PP((ae_int32x4 *)p_out_multiplier);
    ae_valignx2 alignx2_p_sh = AE_LA128_PP((ae_int32x4 *)p_out_shift);

    // Process loop for unroll of 4 vectors
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      ae_int64 bias_1 = p_bias[vec_itr + 1];
      ae_int64 bias_2 = p_bias[vec_itr + 2];
      ae_int64 bias_3 = p_bias[vec_itr + 3];

      WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;
      WORD16* p_dst_2 = (WORD16*)p_out + (vec_itr + 2) * out_offset;
      WORD16* p_dst_3 = (WORD16*)p_out + (vec_itr + 3) * out_offset;

      // Process loop for unroll of 4 rows
      m_itr = 0;
      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;
        ae_int64 acc_4;
        ae_int64 acc_5;
        ae_int64 acc_6;
        ae_int64 acc_7;
        ae_int64 acc_8;
        ae_int64 acc_9;
        ae_int64 acc_10;
        ae_int64 acc_11;
        ae_int64 acc_12;
        ae_int64 acc_13;
        ae_int64 acc_14;
        ae_int64 acc_15;

        acc_0 = acc_4 = acc_8  = acc_12 = bias_0;
        acc_1 = acc_5 = acc_9  = acc_13 = bias_1;
        acc_2 = acc_6 = acc_10 = acc_14 = bias_2;
        acc_3 = acc_7 = acc_11 = acc_15 = bias_3;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_4_rows_4_vecs_aligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,&acc_4
           ,&acc_5
           ,&acc_6
           ,&acc_7
           ,&acc_8
           ,&acc_9
           ,&acc_10
           ,&acc_11
           ,&acc_12
           ,&acc_13
           ,&acc_14
           ,&acc_15
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1, out_2, out_3;
        ae_int32x2 out_4, out_5, out_6, out_7;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_4, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_1, acc_5, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_2, acc_2, acc_6, p_out_multiplier[vec_itr + 2], p_out_shift[vec_itr + 2]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_3, acc_3, acc_7, p_out_multiplier[vec_itr + 3], p_out_shift[vec_itr + 3]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_4, acc_8 , acc_12, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_5, acc_9 , acc_13, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_6, acc_10, acc_14, p_out_multiplier[vec_itr + 2], p_out_shift[vec_itr + 2]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_7, acc_11, acc_15, p_out_multiplier[vec_itr + 3], p_out_shift[vec_itr + 3]);

        ae_int16x4 d0, d1, d2, d3;
        d0 = AE_SAT16X4(out_0, out_4);
        d1 = AE_SAT16X4(out_1, out_5);
        d2 = AE_SAT16X4(out_2, out_6);
        d3 = AE_SAT16X4(out_3, out_7);
        /* Store output */
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d0, d1), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d1), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d1), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d0, d2), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d2), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d2), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(                   d2, (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d0, d3), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d3), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d3), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(                   d3, (ae_int16*)p_dst_3, out_stride);
      }

      // Process loop for unroll of 2 rows
      for (; m_itr < (rows & ~(2 - 1)); m_itr += 2)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;
        ae_int64 acc_4;
        ae_int64 acc_5;
        ae_int64 acc_6;
        ae_int64 acc_7;

        acc_0 = acc_4 = bias_0;
        acc_1 = acc_5 = bias_1;
        acc_2 = acc_6 = bias_2;
        acc_3 = acc_7 = bias_3;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_2_rows_4_vecs_aligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,&acc_4
           ,&acc_5
           ,&acc_6
           ,&acc_7
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1, out_2, out_3;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_4, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_1, acc_5, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_2, acc_2, acc_6, p_out_multiplier[vec_itr + 2], p_out_shift[vec_itr + 2]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_3, acc_3, acc_7, p_out_multiplier[vec_itr + 3], p_out_shift[vec_itr + 3]);

        ae_int16x4 d0, d1;
        d0 = AE_SAT16X4(out_0, out_1);
        d1 = AE_SAT16X4(out_2, out_3);
        /* Store output */
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst_3, out_stride);
      }

      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_LA32X2X2_IP(out_multiplier_01, out_multiplier_23, alignx2_p_mult, (ae_int32x4 *)p_out_mult);
      ae_int32x2 shift_01, shift_23;
      AE_LA32X2X2_IP(shift_01, shift_23, alignx2_p_sh, (ae_int32x4 *)p_out_sh);
      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;

        acc_0 = bias_0;
        acc_1 = bias_1;
        acc_2 = bias_2;
        acc_3 = bias_3;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4* p_mat1_0 = (ae_int16x4*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_1_rows_4_vecs_aligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,(ae_int8x8*)p_vec_0
           ,(ae_int16*)p_mat1_0
           ,cols1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_0, acc_0, acc_1, out_multiplier_01, shift_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_1, acc_2, acc_3, out_multiplier_23, shift_23);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_3, out_stride);
      }
    }

    ae_valign align_p_mult = AE_LA64_PP((ae_int32x2 *)p_out_mult);
    ae_valign align_p_sh = AE_LA64_PP((ae_int32x2 *)p_out_sh);
    // Process loop for unroll of 2 vectors
    for(; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      ae_int64 bias_1 = p_bias[vec_itr + 1];

      WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(2 - 1)); m_itr += 2)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;

        acc_0 = acc_2 = bias_0;
        acc_1 = acc_3 = bias_1;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_2_rows_2_vecs_aligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_2, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_1, acc_3, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
        /* Store output */
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_1, out_stride);
      }

      ae_int32x2 out_multiplier_01;
      AE_LA32X2_IP(out_multiplier_01, align_p_mult, (ae_int32x2 *)p_out_mult);
      ae_int32x2 shift_01;
      AE_LA32X2_IP(shift_01, align_p_sh, (ae_int32x2 *)p_out_sh);
      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;

        acc_0 = bias_0;
        acc_1 = bias_1;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4* p_mat1_0 = (ae_int16x4*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_1_rows_2_vecs_aligned
           (&acc_0
           ,&acc_1
           ,(ae_int16x4*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,vec_stride
           );

        ae_int32x2 out_0;
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_0, acc_0, acc_1, out_multiplier_01, shift_01);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_0);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_1, out_stride);
      }
    }

    // remaining vectors
    for(; vec_itr < vec_count; vec_itr++)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      WORD16* p_dst = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;

        acc_0 = bias_0;
        acc_1 = bias_0;
        acc_2 = bias_0;
        acc_3 = bias_0;

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD16 *p_mat1_0 = (WORD16 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_4_rows_1_vecs_aligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,(ae_int16x4*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
           );

        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_1, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_2, acc_3, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));
        ae_int64 acc_0 = bias_0;

        _xa_nn_dot_product_1_rows_1_vecs_aligned
           (&acc_0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           );

        ae_int32x2 out_0;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_0, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        ae_int16x4 d0 = AE_SAT16X4(out_0, out_0);
        AE_S16_0_XP(d0, (ae_int16*)p_dst, out_stride);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset*sizeof(WORD16);
    int out_offset = out_col_offset;
    WORD32 *p_out_mult = (WORD32 *)p_out_multiplier;
    ae_valignx2 alignx2_p_mult = AE_LA128_PP((ae_int32x4 *)p_out_multiplier);
    WORD32 *p_out_sh = (WORD32 *)p_out_shift;
    ae_valignx2 alignx2_p_sh = AE_LA128_PP((ae_int32x4 *)p_out_shift);

    // Process loop for unroll of 4 vectors
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      ae_int64 bias_1 = p_bias[vec_itr + 1];
      ae_int64 bias_2 = p_bias[vec_itr + 2];
      ae_int64 bias_3 = p_bias[vec_itr + 3];

      WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;
      WORD16* p_dst_2 = (WORD16*)p_out + (vec_itr + 2) * out_offset;
      WORD16* p_dst_3 = (WORD16*)p_out + (vec_itr + 3) * out_offset;

      // Process loop for unroll of 4 rows
      m_itr = 0;
      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;
        ae_int64 acc_4;
        ae_int64 acc_5;
        ae_int64 acc_6;
        ae_int64 acc_7;
        ae_int64 acc_8;
        ae_int64 acc_9;
        ae_int64 acc_10;
        ae_int64 acc_11;
        ae_int64 acc_12;
        ae_int64 acc_13;
        ae_int64 acc_14;
        ae_int64 acc_15;

        acc_0 = acc_4 = acc_8  = acc_12 = bias_0;
        acc_1 = acc_5 = acc_9  = acc_13 = bias_1;
        acc_2 = acc_6 = acc_10 = acc_14 = bias_2;
        acc_3 = acc_7 = acc_11 = acc_15 = bias_3;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_4_rows_4_vecs_unaligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,&acc_4
           ,&acc_5
           ,&acc_6
           ,&acc_7
           ,&acc_8
           ,&acc_9
           ,&acc_10
           ,&acc_11
           ,&acc_12
           ,&acc_13
           ,&acc_14
           ,&acc_15
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1, out_2, out_3;
        ae_int32x2 out_4, out_5, out_6, out_7;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_4, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_1, acc_5, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_2, acc_2, acc_6, p_out_multiplier[vec_itr + 2], p_out_shift[vec_itr + 2]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_3, acc_3, acc_7, p_out_multiplier[vec_itr + 3], p_out_shift[vec_itr + 3]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_4, acc_8 , acc_12, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_5, acc_9 , acc_13, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_6, acc_10, acc_14, p_out_multiplier[vec_itr + 2], p_out_shift[vec_itr + 2]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_7, acc_11, acc_15, p_out_multiplier[vec_itr + 3], p_out_shift[vec_itr + 3]);

        ae_int16x4 d0, d1, d2, d3;
        d0 = AE_SAT16X4(out_0, out_4);
        d1 = AE_SAT16X4(out_1, out_5);
        d2 = AE_SAT16X4(out_2, out_6);
        d3 = AE_SAT16X4(out_3, out_7);
        /* Store output */
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d0, d1), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d1), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d1), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d0, d2), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d2), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d2), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(                   d2, (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d0, d3), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d3), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d3), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(                   d3, (ae_int16*)p_dst_3, out_stride);
      }
      // Process loop for unroll of 2 rows
      for (; m_itr < (rows & ~(2 - 1)); m_itr += 2)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;
        ae_int64 acc_4;
        ae_int64 acc_5;
        ae_int64 acc_6;
        ae_int64 acc_7;

        acc_0 = acc_4 = bias_0;
        acc_1 = acc_5 = bias_1;
        acc_2 = acc_6 = bias_2;
        acc_3 = acc_7 = bias_3;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_2_rows_4_vecs_unaligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,&acc_4
           ,&acc_5
           ,&acc_6
           ,&acc_7
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1, out_2, out_3;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_4, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_1, acc_5, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_2, acc_2, acc_6, p_out_multiplier[vec_itr + 2], p_out_shift[vec_itr + 2]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_3, acc_3, acc_7, p_out_multiplier[vec_itr + 3], p_out_shift[vec_itr + 3]);

        ae_int16x4 d0, d1;
        d0 = AE_SAT16X4(out_0, out_1);
        d1 = AE_SAT16X4(out_2, out_3);
        /* Store output */
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_6543(d1, d1), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d1, d1), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d1, d1), (ae_int16*)p_dst_3, out_stride);
        AE_S16_0_XP(                   d1, (ae_int16*)p_dst_3, out_stride);
      }

      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_LA32X2X2_IP(out_multiplier_01, out_multiplier_23, alignx2_p_mult, (ae_int32x4 *)p_out_mult);
      ae_int32x2 shift_01, shift_23;
      AE_LA32X2X2_IP(shift_01, shift_23, alignx2_p_sh, (ae_int32x4 *)p_out_sh);
      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;

        acc_0 = bias_0;
        acc_1 = bias_1;
        acc_2 = bias_2;
        acc_3 = bias_3;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4* p_mat1_0 = (ae_int16x4*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_1_rows_4_vecs_unaligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,(ae_int8x8*)p_vec_0
           ,(ae_int16*)p_mat1_0
           ,cols1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_0, acc_0, acc_1, out_multiplier_01, shift_01);
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_1, acc_2, acc_3, out_multiplier_23, shift_23);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_2, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_3, out_stride);
      }
    }

    ae_valign align_p_mult = AE_LA64_PP((ae_int32x2 *)p_out_mult);
    ae_valign align_p_sh = AE_LA64_PP((ae_int32x2 *)p_out_sh);
    // Process loop for unroll of 2 vectors
    for(; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      ae_int64 bias_1 = p_bias[vec_itr + 1];

      WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;

      m_itr = 0;

      // Process loop for unroll of 2 rows
      for (m_itr = 0; m_itr < (rows & ~(2 - 1)); m_itr += 2)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;

        acc_0 = acc_2 = bias_0;
        acc_1 = acc_3 = bias_1;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_2_rows_2_vecs_unaligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
           ,vec_stride
           );

        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_2, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_1, acc_3, p_out_multiplier[vec_itr + 1], p_out_shift[vec_itr + 1]);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
        /* Store output */
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_1, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_1, out_stride);
      }

      ae_int32x2 out_multiplier_01;
      AE_LA32X2_IP(out_multiplier_01, align_p_mult, (ae_int32x2 *)p_out_mult);
      ae_int32x2 shift_01;
      AE_LA32X2_IP(shift_01, align_p_sh, (ae_int32x2 *)p_out_sh);
      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;

        acc_0 = bias_0;
        acc_1 = bias_1;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4* p_mat1_0 = (ae_int16x4*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_1_rows_2_vecs_unaligned
           (&acc_0
           ,&acc_1
           ,(ae_int16x4*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,vec_stride
           );

        ae_int32x2 out_0;
        MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out_0, acc_0, acc_1, out_multiplier_01, shift_01);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_0);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst_0, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst_1, out_stride);
      }
    }
    // remaining vectors
    for(; vec_itr < vec_count; vec_itr++)
    {
      ae_int64 bias_0 = p_bias[vec_itr + 0];
      WORD16* p_dst = (WORD16*)p_out + (vec_itr + 0) * out_offset;
      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int64 acc_0;
        ae_int64 acc_1;
        ae_int64 acc_2;
        ae_int64 acc_3;

        acc_0 = bias_0;
        acc_1 = bias_0;
        acc_2 = bias_0;
        acc_3 = bias_0;

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD16 *p_mat1_0 = (WORD16 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
           (&acc_0
           ,&acc_1
           ,&acc_2
           ,&acc_3
           ,(ae_int16x4*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
           );

        ae_int32x2 out_0, out_1;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_1, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_1, acc_2, acc_3, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);

        ae_int16x4 d0;
        d0 = AE_SAT16X4(out_0, out_1);
        AE_S16_0_XP(AE_SEL16_6543(d0, d0), (ae_int16*)p_dst, out_stride);
        AE_S16_0_XP(AE_SEL16_5432(d0, d0), (ae_int16*)p_dst, out_stride);
        AE_S16_0_XP(AE_SEL16_4321(d0, d0), (ae_int16*)p_dst, out_stride);
        AE_S16_0_XP(                   d0, (ae_int16*)p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int16x4 *p_mat1_0 = (ae_int16x4*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD16));
        ae_int64 acc_0 = bias_0;

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
           (&acc_0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           );

        ae_int32x2 out_0;
        MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out_0, acc_0, acc_0, p_out_multiplier[vec_itr + 0], p_out_shift[vec_itr + 0]);
        ae_int16x4 d0 = AE_SAT16X4(out_0, out_0);
        AE_S16_0_XP(d0, (ae_int16*)p_dst, out_stride);
      }
    }
  }
  else
    return -1;
  return 0;
}
