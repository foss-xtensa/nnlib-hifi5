/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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

const long long g_sel_pattern[16] = {
 0xf7e6d5c4L, 0xb3a29180L,
 0xe7d6c5b4L, 0xa3928170L,
 0xd7c6b5a4L, 0x93827160L,
 0xc7b6a594L, 0x83726150L,
 0xb7a69584L, 0x73625140L,
 0xa7968574L, 0x63524130L,
 0x97867564L, 0x53423120L,
 0x87766554L, 0x43322110L
};

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
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec_offset 
    )
{
  int pre_loop_count, loop_count, post_loop_count, pre_loop_shift;
  int c_itr;

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
  pre_loop_shift = align_offset * 8;
  p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step 
  loop_count = cols1 - pre_loop_count;
  post_loop_count = loop_count & 0xf;
  loop_count >>= 4;

  int rem_cols_shift_0 = ((post_loop_count)<=8)?(8-(post_loop_count))*8:0;
  int rem_cols_shift_1 = ((post_loop_count)>8)?(16-(post_loop_count))*8:64;

  ae_int8x8* p_mat1_1 = p_mat1_0 + row_stride1; //next 8th row 
  ae_int8x8* p_mat1_2 = p_mat1_1 + row_stride1; //next 8th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + row_stride1; //next 8th row 

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valign align_p_vec0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec1 = AE_LA64_PP(p_vec_1);
  ae_valign align_p_vec2 = AE_LA64_PP(p_vec_2);
  ae_valign align_p_vec3 = AE_LA64_PP(p_vec_3);

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

  AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
  AE_LA8X8_IP(vec0_batch_1, align_p_vec0, (ae_int8x8 *)p_vec_0);
  AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8 *)p_vec_1);
  AE_LA8X8_IP(vec1_batch_1, align_p_vec1, (ae_int8x8 *)p_vec_1);
  AE_LA8X8_IP(vec2_batch_0, align_p_vec2, (ae_int8x8 *)p_vec_2);
  AE_LA8X8_IP(vec2_batch_1, align_p_vec2, (ae_int8x8 *)p_vec_2);
  AE_LA8X8_IP(vec3_batch_0, align_p_vec3, (ae_int8x8 *)p_vec_3);
  AE_LA8X8_IP(vec3_batch_1, align_p_vec3, (ae_int8x8 *)p_vec_3);

  mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), pre_loop_shift), pre_loop_shift));
  mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), pre_loop_shift), pre_loop_shift));
  mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), pre_loop_shift), pre_loop_shift));
  mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), pre_loop_shift), pre_loop_shift));

  AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
  AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
  AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

  AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
  AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
  AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
  AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

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

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8*)p_vec_0);
    AE_LA8X8_IP(vec0_batch_1, align_p_vec0, (ae_int8x8*)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1, (ae_int8x8*)p_vec_1);
    AE_LA8X8_IP(vec1_batch_1, align_p_vec1, (ae_int8x8*)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec2, (ae_int8x8*)p_vec_2);
    AE_LA8X8_IP(vec2_batch_1, align_p_vec2, (ae_int8x8*)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec3, (ae_int8x8*)p_vec_3);
    AE_LA8X8_IP(vec3_batch_1, align_p_vec3, (ae_int8x8*)p_vec_3);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols1
  c_itr = 0;
  int rem_shift = rem_cols_shift_0;;
  while(c_itr < post_loop_count)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0,(ae_int8x8*)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec1,(ae_int8x8*)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec2,(ae_int8x8*)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec3,(ae_int8x8*)p_vec_3);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    c_itr += 8;
    rem_shift = rem_cols_shift_1;
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
    ,WORD32      cols1
    ,WORD32      row_stride1
    )
{
  int rem_cols_shift = 64 - (cols1 & 7) * 8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + 8 * row_stride1); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + 8 * row_stride1); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + 8 * row_stride1); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols1-(cols1&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    )
{
  int rem_cols_shift = 64 - (cols1 & 7) * 8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols1-(cols1&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
    )
{
  int c_itr = 0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int rem_cols_shift = 64 - (cols1 & 7) * 8;
  int cols_count=cols1-(cols1&7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}
       
WORD32 xa_nn_matmul_8x8_8(
         WORD8 * __restrict__ p_out,          /* array of output pointers */
         const WORD8 * __restrict__ p_mat1,         /* matrix1: rows x cols1 */
         const WORD8 * __restrict__ p_vec1,         /* vec1: cols1 x 1 */
         const WORD8 * __restrict__ p_bias,         /* bias TBD: Need array? */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift,                       /* bias shift amount */
         WORD32 vec_count,                      /* number of vectors: 2, 4, 2n */
         WORD32 vec_offset,
         WORD32 out_offset,
         WORD32 out_stride)                      
{
  /* Iterators used in for loops */
  int m_itr, vec_itr, b_itr;
  int ii;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

#undef VEC_UNROLL
#define VEC_UNROLL 4

  acc_shift = acc_shift + 32;
  ae_int64 bias_array[32] = {0};

  if (p_mat1 && p_vec1)
  {
    int neg_bias = (bias_shift<=0)?1:0; 
    int rshift_bias = neg_bias?XT_MIN(16,-bias_shift):-1;
    ae_int32x2 lshift_mul_bias = neg_bias?AE_MOVDA32(1):AE_MOVDA32(1<<(bias_shift-1));

    ae_int16x4 _ae_int16x4_bias, _ae_int16x4_bias1; 
    ae_int8x8 _ae_int8x8_bias; 
    WORD8 *_WORD8_p_bias = (WORD8 *) p_bias; 

    for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
    {
      // TODO: Calculate shifted bias values and store on stack.
      if(p_bias)
      {
        WORD8 *p_bias_ua = _WORD8_p_bias + m_itr;
        ae_valign align_p_bias_ua;
        align_p_bias_ua = AE_LA64_PP(p_bias_ua);
        for(b_itr = 0; b_itr < 32; b_itr+=8)
        {
          AE_LA8X8_IP(_ae_int8x8_bias, align_p_bias_ua, (ae_int8x8 *)p_bias_ua);
          AE_CVTI16X4X2F8(_ae_int16x4_bias, _ae_int16x4_bias1, _ae_int8x8_bias, 0);
          _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
          _ae_int16x4_bias1 = AE_SRAA16S(_ae_int16x4_bias1, rshift_bias);
          bias_array[b_itr + 0] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);
          bias_array[b_itr + 1] = AE_MUL32X16_L2(lshift_mul_bias, _ae_int16x4_bias);
          bias_array[b_itr + 2] = AE_MUL32X16_L1(lshift_mul_bias, _ae_int16x4_bias);
          bias_array[b_itr + 3] = AE_MUL32X16_L0(lshift_mul_bias, _ae_int16x4_bias);
          bias_array[b_itr + 4] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias1);
          bias_array[b_itr + 5] = AE_MUL32X16_L2(lshift_mul_bias, _ae_int16x4_bias1);
          bias_array[b_itr + 6] = AE_MUL32X16_L1(lshift_mul_bias, _ae_int16x4_bias1);
          bias_array[b_itr + 7] = AE_MUL32X16_L0(lshift_mul_bias, _ae_int16x4_bias1);
        }
      }
      
      for(ii = 0; ii < 8; ii++)
      {
        WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + ii + 0) * out_stride;
        WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + ii + 8) * out_stride;
        WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + ii + 16) * out_stride;
        WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + ii + 24) * out_stride;
        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
          ae_int32x2 acc_row0_vec0 = ZERO32;
          ae_int32x2 acc_row1_vec0 = ZERO32;
          ae_int32x2 acc_row0_vec1 = ZERO32;
          ae_int32x2 acc_row1_vec1 = ZERO32;
          ae_int32x2 acc_row0_vec2 = ZERO32;
          ae_int32x2 acc_row1_vec2 = ZERO32;
          ae_int32x2 acc_row0_vec3 = ZERO32;
          ae_int32x2 acc_row1_vec3 = ZERO32;

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
            );

          ae_int64 acc64_0_0, acc64_0_1, acc64_0_2, acc64_0_3;
          ae_int64 acc64_1_0, acc64_1_1, acc64_1_2, acc64_1_3;
          ae_int64 acc64_2_0, acc64_2_1, acc64_2_2, acc64_2_3;
          ae_int64 acc64_3_0, acc64_3_1, acc64_3_2, acc64_3_3;

          ae_int8x8 temp_vec0, temp_vec1, temp_vec2, temp_vec3;

          acc64_0_0 = acc64_0_1 = acc64_0_2 = acc64_0_3 = bias_array[ 0 + ii];
          acc64_1_0 = acc64_1_1 = acc64_1_2 = acc64_1_3 = bias_array[ 8 + ii];
          acc64_2_0 = acc64_2_1 = acc64_2_2 = acc64_2_3 = bias_array[16 + ii];
          acc64_3_0 = acc64_3_1 = acc64_3_2 = acc64_3_3 = bias_array[24 + ii];
          
          AE_ACCW32(acc64_0_0, acc64_1_0, acc_row0_vec0, ZERO32);
          AE_ACCW32(acc64_0_1, acc64_1_1, acc_row0_vec1, ZERO32);
          AE_ACCW32(acc64_0_2, acc64_1_2, acc_row0_vec2, ZERO32);
          AE_ACCW32(acc64_0_3, acc64_1_3, acc_row0_vec3, ZERO32);
          AE_ACCW32(acc64_2_0, acc64_3_0, acc_row1_vec0, ZERO32);
          AE_ACCW32(acc64_2_1, acc64_3_1, acc_row1_vec1, ZERO32);
          AE_ACCW32(acc64_2_2, acc64_3_2, acc_row1_vec2, ZERO32);
          AE_ACCW32(acc64_2_3, acc64_3_3, acc_row1_vec3, ZERO32);

          acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_1_0,acc_shift));
          acc_row1_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_0,acc_shift), AE_SLAA64S(acc64_3_0,acc_shift));
          acc_row0_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_1,acc_shift), AE_SLAA64S(acc64_1_1,acc_shift));
          acc_row1_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_1,acc_shift), AE_SLAA64S(acc64_3_1,acc_shift));
          acc_row0_vec2 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_2,acc_shift), AE_SLAA64S(acc64_1_2,acc_shift));
          acc_row1_vec2 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_2,acc_shift), AE_SLAA64S(acc64_3_2,acc_shift));
          acc_row0_vec3 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_3,acc_shift), AE_SLAA64S(acc64_1_3,acc_shift));
          acc_row1_vec3 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_3,acc_shift), AE_SLAA64S(acc64_3_3,acc_shift));

          temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row1_vec0);
          temp_vec1 = AE_SAT8X4X32_L(acc_row0_vec1, acc_row1_vec1);
          temp_vec2 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row1_vec2);
          temp_vec3 = AE_SAT8X4X32_L(acc_row0_vec3, acc_row1_vec3);

          // TODO: Simplify output storage:
          // Try AE_S8_[n]_X using DSEL
#if 1
          AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_3_XP(temp_vec1, (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_3_XP(temp_vec2, (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_3_XP(temp_vec3, (ae_int8 *) p_dst_0, out_offset);
          
          AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_2_XP(temp_vec1, (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_2_XP(temp_vec2, (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_2_XP(temp_vec3, (ae_int8 *) p_dst_1, out_offset);
          
          AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_1_XP(temp_vec1, (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_1_XP(temp_vec2, (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_1_XP(temp_vec3, (ae_int8 *) p_dst_2, out_offset);
          
          AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_3, out_offset);
          AE_S8_0_XP(temp_vec1, (ae_int8 *) p_dst_3, out_offset);
          AE_S8_0_XP(temp_vec2, (ae_int8 *) p_dst_3, out_offset);
          AE_S8_0_XP(temp_vec3, (ae_int8 *) p_dst_3, out_offset);
#else
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 0*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,3));
          (*((WORD8 *) p_out +(vec_itr + 1)*out_offset + (m_itr + ii + 0*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec1,3));
          (*((WORD8 *) p_out +(vec_itr + 2)*out_offset + (m_itr + ii + 0*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec2,3));
          (*((WORD8 *) p_out +(vec_itr + 3)*out_offset + (m_itr + ii + 0*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec3,3));
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 1*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,2));
          (*((WORD8 *) p_out +(vec_itr + 1)*out_offset + (m_itr + ii + 1*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec1,2));
          (*((WORD8 *) p_out +(vec_itr + 2)*out_offset + (m_itr + ii + 1*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec2,2));
          (*((WORD8 *) p_out +(vec_itr + 3)*out_offset + (m_itr + ii + 1*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec3,2));
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 2*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,1));
          (*((WORD8 *) p_out +(vec_itr + 1)*out_offset + (m_itr + ii + 2*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec1,1));
          (*((WORD8 *) p_out +(vec_itr + 2)*out_offset + (m_itr + ii + 2*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec2,1));
          (*((WORD8 *) p_out +(vec_itr + 3)*out_offset + (m_itr + ii + 2*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec3,1));
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 3*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,0));
          (*((WORD8 *) p_out +(vec_itr + 1)*out_offset + (m_itr + ii + 3*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec1,0));
          (*((WORD8 *) p_out +(vec_itr + 2)*out_offset + (m_itr + ii + 3*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec2,0));
          (*((WORD8 *) p_out +(vec_itr + 3)*out_offset + (m_itr + ii + 3*8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec3,0));
#endif
        }

        // Remaining vectors
        // TODO: codesize considertations: remove inline for 4 row, 1 vec case ?
        for (; vec_itr < vec_count; vec_itr++)
        {
          ae_int32x2 acc_row0_vec0 = ZERO32;
          ae_int32x2 acc_row1_vec0 = ZERO32;

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
            );

          ae_int64 acc64_0_0;
          ae_int64 acc64_1_0;
          ae_int64 acc64_2_0;
          ae_int64 acc64_3_0;

          ae_int8x8 temp_vec0;

          acc64_0_0 = bias_array[0 + ii];
          acc64_1_0 = bias_array[8 + ii];
          acc64_2_0 = bias_array[16 + ii];
          acc64_3_0 = bias_array[24 + ii];

          AE_ACCW32(acc64_0_0, acc64_1_0, acc_row0_vec0, ZERO32);
          AE_ACCW32(acc64_2_0, acc64_3_0, acc_row1_vec0, ZERO32);

          acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_1_0,acc_shift));
          acc_row1_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_0,acc_shift), AE_SLAA64S(acc64_3_0,acc_shift));

          temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row1_vec0);

#if 1
          AE_SW_S8_3_X(temp_vec0, (ae_int8 *) p_dst,  0 * out_stride);
          AE_SW_S8_2_X(temp_vec0, (ae_int8 *) p_dst,  8 * out_stride);
          AE_SW_S8_1_X(temp_vec0, (ae_int8 *) p_dst, 16 * out_stride);
          AE_S8_0_X(temp_vec0, (ae_int8 *) p_dst, 24 * out_stride);
#else
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 0 * 8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,3));
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 1 * 8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,2));
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 2 * 8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,1));
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + ii + 3 * 8)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,0));
#endif
        }
      }
    }

    // remaining rows
    // TODO: Add the ~21 MAC/cycle loop for 4 rows and 4 vectors 
    {
      for(; m_itr < rows; m_itr++)
      {
        if(p_bias)
        {
          WORD8 *p_bias_ua = _WORD8_p_bias + m_itr;
          _ae_int16x4_bias = AE_MOVDA16(*p_bias_ua);
          _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
          bias_array[0] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);
        }
        WORD8* p_dst = (WORD8*)p_out + (m_itr + 0) * out_stride;
        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
          ae_int32x2 acc_row0_vec0 = ZERO32;
          ae_int32x2 acc_row0_vec1 = ZERO32;

          ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_offset);
          WORD8 *p_mat1_0 = (WORD8 *) &p_mat1[(m_itr)*row_stride1]; 

          _xa_nn_dot_product_4_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,&acc_row0_vec1
             ,(ae_int8x8*)p_vec_0
             ,(ae_int8*)p_mat1_0
             ,cols1
             ,vec_offset
            );

          ae_int64 acc64_0_0, acc64_0_1, acc64_0_2, acc64_0_3;
          ae_int8x8 temp_vec0;

          acc64_0_0 = bias_array[0];
          acc64_0_1 = acc64_0_0;
          acc64_0_2 = acc64_0_0;
          acc64_0_3 = acc64_0_0;

          AE_ACCW32(acc64_0_0, acc64_0_1, acc_row0_vec0, ZERO32);
          AE_ACCW32(acc64_0_2, acc64_0_3, acc_row0_vec1, ZERO32);

          acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_0_1,acc_shift));
          acc_row0_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_2,acc_shift), AE_SLAA64S(acc64_0_3,acc_shift));

          temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);

#if 1
          AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
          AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
          AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
          AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
#else
          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,3));
          (*((WORD8 *) p_out +(vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,2));
          (*((WORD8 *) p_out +(vec_itr + 2)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,1));
          (*((WORD8 *) p_out +(vec_itr + 3)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,0));
#endif
        }

        // Remaining vectors
        for (; vec_itr < vec_count; vec_itr++)
        {
          ae_int32x2 acc_row0_vec0 = ZERO32;
          ae_int32x2 acc_row0_vec1 = ZERO32;

          ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8 *p_mat1_0 = (ae_int8x8*) &p_mat1[m_itr * row_stride1]; 

          _xa_nn_dot_product_1_rows_1_vecs_unaligned
            (&acc_row0_vec0
             ,&acc_row0_vec1
             ,p_mat1_0
             ,p_vec_0
             ,cols1
            );

          ae_int64 acc64_0_0, dummy;
          ae_int8x8 temp_vec0;

          acc64_0_0 = bias_array[0];

          AE_ACCW32(acc64_0_0, dummy, acc_row0_vec0, ZERO32);

          acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_0_0,acc_shift));

          temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);

          (*((WORD8 *) p_out +(vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD8)(AE_MOVAD8(temp_vec0,3));
        }
      }
    }
  }
  else
  {
    return -1;
  }
  return 0;
}
