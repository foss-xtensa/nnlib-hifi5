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

#ifndef AE_MULAZB8Q8X8
#define MAT_VEC_MUL AE_MULA8Q8X8
#else
#define MAT_VEC_MUL AE_MULAZB8Q8X8
#endif


extern const long long g_sel_pattern[16];
extern const long long pre_loop_sel_pattern[16];
extern const long long post_loop_sel_pattern[16];

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
    )
{
  int rem_cols_shift_0 = ((cols & 15) <= 8)? (8 - (cols & 15)) * 8 : 0;
  int rem_cols_shift_1 = ((cols & 15) > 8)? (16 - (cols & 15)) * 8 : 64;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;
  
  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1; 
  ae_int8x8 vec2_batch_0, vec2_batch_1; 
  ae_int8x8 vec3_batch_0, vec3_batch_1; 

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;

  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  int cols_count = cols -(cols & 15);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row0_1, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row1_1, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row2_1, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);
    AE_L8X8_XC(mat1_row3_1, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec0_batch_1, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec1_batch_1, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec2_batch_1, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);
    AE_L8X8_IP(vec3_batch_1, (ae_int8x8 *)p_vec_3, 8);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
  int rem_shift = rem_cols_shift_0;
  while(c_itr < cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));
    
    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
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

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7)*8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int8x8 vec0_batch_0; 

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
#ifdef AE_MULAZB8Q8X8
    ,WORD32      vec1_zero_bias
#endif
    )
{
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#else
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
#endif
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 

  /* p_vec needs to be accessed in circular fashion */
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_XC(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_XC(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
#endif

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
    )
{
  int c_itr = 0;
  
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  int rem_cols_shift = 64 - (cols & 7) * 8;
  int cols_count=cols-(cols&7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    MAT_VEC_MUL(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

    MAT_VEC_MUL(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

// All functions imported from conv2d_std_8x8
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
#ifdef AE_MULAZB8Q8X8	
    ,WORD32      vec1_zero_bias
#endif
    )
{
#ifndef AE_MULAZB8Q8X8
  int pre_loop_count, loop_count, post_loop_count, pre_loop_shift;
#else
  int pre_loop_count, loop_count, post_loop_count;
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
#endif
  int c_itr;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1; 
  ae_int8x8 vec2_batch_0, vec2_batch_1; 
  ae_int8x8 vec3_batch_0, vec3_batch_1; 

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  int align_offset = ((unsigned int)p_vec_0 & 0x7);
  pre_loop_count = 8 - align_offset;
#ifndef AE_MULAZB8Q8X8  
  pre_loop_shift = align_offset * 8;
#else
  ae_int8x8 pre_sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset % 8)], pre_loop_sel_pattern[2 * (align_offset % 8) + 1])); 
  //TODO: circular access
  //p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
#endif
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, -align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step 
#ifndef AE_MULAZB8Q8X8 
  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 15):((cols + align_offset) & 15);
  loop_count >>= 4;

  int mask_start_end = ((cols + align_offset) < 16)?0:1;
  
  int rem_cols_shift_0 = (post_loop_count <= 8) ? (8 - post_loop_count) * 8 : 0;
  int rem_cols_shift_1 = (post_loop_count > 8) ? (16 - post_loop_count) * 8 : 64;

  ae_int8x8* p_mat1_1 = p_mat1_0; 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;  
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;  
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));
#else
  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 15):((cols + align_offset) & 15);
  loop_count >>= 4;
  
  int mask_start_end = ((cols + align_offset) < 16)?0:1;
  
  int rem_g8 = (post_loop_count > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (post_loop_count % 8) * !rem_g8], post_loop_sel_pattern[2 * (post_loop_count % 8) * !rem_g8 + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (post_loop_count % 8) * rem_g8], post_loop_sel_pattern[2 * (post_loop_count % 8) * rem_g8 + 1])); 

  ae_int8x8* p_mat1_1 = p_mat1_0; //next 8th row 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1,  1 * row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1; //next 8th row
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2,  1 * row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2; //next 8th row 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3,  1 * row_offset * sizeof(WORD8));
#endif

  // Process every 8th vector
  ae_int8* p_vec_1 = p_vec_0 + 8 * vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + 8 * vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + 8 * vec_offset;

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  /* Pre loop computation */
  AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
  AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
  AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
  AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
  AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
  AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
  AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
  AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

  AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
  AE_L8X8_IP(vec0_batch_1, (ae_int8x8 *)p_vec_0, 8);
  AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
  AE_L8X8_IP(vec1_batch_1, (ae_int8x8 *)p_vec_1, 8);
  AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
  AE_L8X8_IP(vec2_batch_1, (ae_int8x8 *)p_vec_2, 8);
  AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);
  AE_L8X8_IP(vec3_batch_1, (ae_int8x8 *)p_vec_3, 8);

  if(align_offset)
  {
#ifndef AE_MULAZB8Q8X8
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), pre_loop_shift), pre_loop_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), pre_loop_shift), pre_loop_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), pre_loop_shift), pre_loop_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), pre_loop_shift), pre_loop_shift));
#else
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, pre_sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, pre_sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, pre_sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, pre_sel1);
#endif
  }

  if(mask_start_end)
  {
    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
    AE_L8X8_IP(vec0_batch_1, (ae_int8x8*)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
    AE_L8X8_IP(vec1_batch_1, (ae_int8x8*)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
    AE_L8X8_IP(vec2_batch_1, (ae_int8x8*)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);
    AE_L8X8_IP(vec3_batch_1, (ae_int8x8*)p_vec_3, 8);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
#ifndef AE_MULAZB8Q8X8
  if(post_loop_count)
  {
    if(mask_start_end) 
    {
      AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
      AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
      AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
      
      AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
      AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
      AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

      AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
      AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
      AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
      AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);
      
      AE_L8X8_IP(vec0_batch_1, (ae_int8x8*)p_vec_0, 8);
      AE_L8X8_IP(vec1_batch_1, (ae_int8x8*)p_vec_1, 8);
      AE_L8X8_IP(vec2_batch_1, (ae_int8x8*)p_vec_2, 8);
      AE_L8X8_IP(vec3_batch_1, (ae_int8x8*)p_vec_3, 8);
    }

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

    mat1_row0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_1), rem_cols_shift_1), rem_cols_shift_1));
    mat1_row1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_1), rem_cols_shift_1), rem_cols_shift_1));
    mat1_row2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_1), rem_cols_shift_1), rem_cols_shift_1));
    mat1_row3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_1), rem_cols_shift_1), rem_cols_shift_1));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }
#else
  c_itr = 0;
  while(c_itr < post_loop_count)
  {
    if(mask_start_end) 
    {
      AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
      AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
      AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

      AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
      AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
      AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
      AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);
    }

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, sel1);
    
    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

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
#endif

  *out_0_0 = acc_row0_vec0;
  *out_0_1 = acc_row0_vec1;
  *out_0_2 = acc_row0_vec2;
  *out_0_3 = acc_row0_vec3;
       
  *out_1_0 = acc_row1_vec0;
  *out_1_1 = acc_row1_vec1;
  *out_1_2 = acc_row1_vec2;
  *out_1_3 = acc_row1_vec3;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
#ifdef AE_MULAZB8Q8X8
    ,WORD32      vec1_zero_bias
#endif
    )
{
#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#endif
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

#ifndef AE_MULAZB8Q8X8
  ae_int8x8 *p_mat1_1 = p_mat1_0 + row_offset; //next 8th vector
  ae_int8x8 *p_mat1_2 = p_mat1_1 + row_offset; //next 8th vector
  ae_int8x8 *p_mat1_3 = p_mat1_2 + row_offset; //next 8th vector
#else
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + 8 * row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + 8 * row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + 8 * row_offset);
#endif 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  /* p_vec needs to be accessed in circular fashion */
  AE_SW_PRIME_CIRC_64(p_vec_0, align_p_vec0);

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else 
	vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
#endif

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
#ifdef AE_MULAZB8Q8X8
    ,WORD32      vec1_zero_bias
#endif
    )
{
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift_0 = ((cols & 15) <= 8)? (8 - (cols & 15)) * 8 : 0;
  int rem_cols_shift_1 = ((cols & 15) > 8)? (16 - (cols & 15)) * 8 : 64;
#else
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  int rem_cols = cols & 15;
  int rem_g8 = ((rem_cols & 15) > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); \
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1]));
#endif

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1; 
  ae_int8x8 vec2_batch_0, vec2_batch_1; 
  ae_int8x8 vec3_batch_0, vec3_batch_1; 
  ae_int8x8 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;

  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);
  AE_SW_PRIME_64(p_vec_1, align_p_vec1);
  AE_SW_PRIME_64(p_vec_2, align_p_vec2);
  AE_SW_PRIME_64(p_vec_3, align_p_vec3);

  int cols_count = cols -(cols & 15);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec0_batch_1, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec1_batch_0, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec1_batch_1, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec2_batch_1, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);
    AE_SW_LA8X8_IP(vec3_batch_1, align_p_vec3, p_vec_3);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
#ifndef AE_MULAZB8Q8X8
  int rem_shift = rem_cols_shift_0;
#endif
  while(c_itr < cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec1_batch_0, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);

#ifndef AE_MULAZB8Q8X8
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));
#else
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, sel1);
#endif

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    MAT_VEC_MUL(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    MAT_VEC_MUL(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    MAT_VEC_MUL(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
    c_itr += 8;
#ifndef AE_MULAZB8Q8X8
    rem_shift = rem_cols_shift_1;
#else
    sel1 = sel2;
#endif
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
#ifdef AE_MULAZB8Q8X8
    ,WORD32      vec1_zero_bias
#endif
    )
{
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7)*8;
#else
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
#endif
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
#endif

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
#ifdef AE_MULAZB8Q8X8
    ,WORD32      vec1_zero_bias
#endif
    )
{
  int c_itr = 0;

#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#else
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
#endif  
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  /* p_vec needs to be accessed in circular fashion */
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_CIRC_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    
    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

#ifndef AE_MULAZB8Q8X8
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
#else
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
#endif

    MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
#ifdef AE_MULAZB8Q8X8
    ,WORD32      vec1_zero_bias
#endif
    )
{
  int c_itr = 0;
#ifndef AE_MULAZB8Q8X8
  int rem_cols_shift = 64 - (cols & 7) * 8;
#else
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
#endif  
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0;
  ae_valign align_p_vec0; 
  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  align_p_vec0 = AE_LA64_PP(p_vec_0);
  int cols_count=cols-(cols&7);
#ifndef AE_MULAZB8Q8X8
#pragma no_unroll
#endif
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

#ifndef AE_MULAZB8Q8X8
    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
#else
    MAT_VEC_MUL(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
#endif
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

#ifndef AE_MULAZB8Q8X8
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
#else
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    MAT_VEC_MUL(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
#endif
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matXvec_sym8sxasym8s_asym8s_circ(
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

#ifndef AE_MULAZB8Q8X8
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols1 & 7) * 8;
  ae_int8x8 mat_z_b = AE_MOVDA8(-mat1_offset);

  ae_int8x8 vec0_0, vec0_1;
  ae_int8x8 vec1_0, vec1_1;
  ae_int8x8 vec2_0, vec2_1;
  ae_int8x8 vec3_0, vec3_1;
#else
  ae_int32x2 bias_buffer[2];

  /*Load AE_BIASV8 and AE_BIASC8 state registers with mat1 and vec1 zero bias values*/
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, -mat1_offset));
  ae_int64 biascv1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-mat1_offset, 0));
#endif
 
  if(cols1 > 8 && cols1 < 16 && (vec_count & 0x3) == 0 && (out_col_offset == 1) && ((unsigned int)p_out & 0x3) == 0)
  {
#ifndef AE_MULAZB8Q8X8
    ae_int32x2 bias_buffer[2];
#else
    AE_MOVZBVCDR(biascv1);
    ae_int32x2 d_bias_0, d_bias_1;
#endif

    int out_stride = out_row_offset;

    ae_int32x4 *pt_bias = (ae_int32x4 *)p_bias;
    ae_valignx2 align_p_bias = AE_LA128_PP(pt_bias);
    
    ae_int32x4 *pt_out_mult = (ae_int32x4 *)p_out_multiplier;
    ae_valignx2 align_p_out_mult = AE_LA128_PP(pt_out_mult);
    // Process loop for 4 rows and 4 vectors 
    for(vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
#ifdef AE_MULAZB8Q8X8
      AE_LA32X2X2_IP(d_bias_0, d_bias_1, align_p_bias, (ae_int32x4 *)pt_bias);
      AE_S32X2X2_I(d_bias_0, d_bias_1, (ae_int32x4 *)bias_buffer, 0);
#endif    
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1; 
      ae_int8x8 vec1_batch_0, vec1_batch_1; 
      ae_int8x8 vec2_batch_0, vec2_batch_1; 
      ae_int8x8 vec3_batch_0, vec3_batch_1;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      AE_LAV8X8X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0, cols1);
      AE_LAV8X8X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1, cols1);
      AE_LAV8X8X2_XP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2, cols1);
      AE_LAV8X8X2_XP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3, cols1);
      
#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row0, acc_row1, vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat_z_b);
      AE_MULA8Q8X8(acc_row0, acc_row1, vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat_z_b);
 
      ae_int32x2 d_bias_0, d_bias_1;
      AE_LA32X2X2_IP(d_bias_0, d_bias_1, align_p_bias, (ae_int32x4 *)pt_bias);
      acc_row0 = AE_SUB32S(d_bias_0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias_1, acc_row1);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
#endif
      
      /* Shifts to match with Tensorflow */
#if TFLITE_SINGLE_ROUNDING
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[0] = p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3];

      ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]);
      ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

      ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
      ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]);
      
#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 ls0123 = AE_ADD16(AE_SAT16X4(ls_01, ls_23), AE_MOVDA16(17));
      ls_01 = AE_MOVINT32X2_FROMINT16X4(ls0123);
#endif
      (void)rs_01;
      (void)rs_23;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[0] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
      ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

      ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
      ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]); 
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 p_out_mult01, p_out_mult23;
      AE_LA32X2X2_IP(p_out_mult01, p_out_mult23, align_p_out_mult, (ae_int32x4 *)pt_out_mult);

#pragma concurrent
      for (m_itr = 0; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_valignx2 align_p_mat1_0;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0);
        MAT_VEC_MUL(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row0_1);

        /* Apply quantization */
        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult01, p_out_mult23, ls_01, ls_23, rs_01, rs_23, out_zero_bias); 

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        /* Store output */
		    STORE_16x4_8x4(out_0, p_dst_0, out_stride);
      }
    }
  }
  else if(cols1 == 27 && (vec_count & 0x3) == 0 && (out_col_offset == 1) && ((unsigned int)p_out & 0x3) == 0)
  {
#ifndef AE_MULAZB8Q8X8
    ae_int32x2 bias_buffer[2];
#else
    AE_MOVZBVCDR(biascv1);
#endif

    int out_stride = out_row_offset;

    int rem_cols = cols1 - 16;
    // Process loop for 4 rows and 4 vectors 
    for(vec_itr = 0; vec_itr < vec_count; vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
      ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3; 
      ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3; 
      ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3);
      
      AE_LAV8X8X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec0, (ae_int8x16 *)p_vec_0, rem_cols);
      AE_LAV8X8X2_XP(vec1_batch_2, vec1_batch_3, align_p_vec1, (ae_int8x16 *)p_vec_1, rem_cols);
      AE_LAV8X8X2_XP(vec2_batch_2, vec2_batch_3, align_p_vec2, (ae_int8x16 *)p_vec_2, rem_cols);
      AE_LAV8X8X2_XP(vec3_batch_2, vec3_batch_3, align_p_vec3, (ae_int8x16 *)p_vec_3, rem_cols);
      
#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_0, vec2_batch_0, vec1_batch_0, vec0_batch_0, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_1, vec2_batch_1, vec1_batch_1, vec0_batch_1, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_2, vec2_batch_2, vec1_batch_2, vec0_batch_2, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_3, vec2_batch_3, vec1_batch_3, vec0_batch_3, mat_z_b);
 
      ae_int32x2 d_bias_0, d_bias_1;
      d_bias_1 = AE_MOVDA32X2(p_bias[vec_itr + 3], p_bias[vec_itr + 2]);
      d_bias_0 = AE_MOVDA32X2(p_bias[vec_itr + 1], p_bias[vec_itr + 0]);
      
      acc_row0 = AE_SUB32S(d_bias_0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias_1, acc_row1);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
#else
      ae_int32x2 acc_row0 = AE_MOVDA32X2(p_bias[vec_itr + 1], p_bias[vec_itr + 0]);
      ae_int32x2 acc_row1 = AE_MOVDA32X2(p_bias[vec_itr + 3], p_bias[vec_itr + 2]);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
#endif
      
      /* Shifts to match with Tensorflow */
#if TFLITE_SINGLE_ROUNDING
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[3] = p_out_shift[vec_itr + 0];
      p_left_shift[2] = p_out_shift[vec_itr + 1];
      p_left_shift[1] = p_out_shift[vec_itr + 2];
      p_left_shift[0] = p_out_shift[vec_itr + 3];

      p_right_shift[3] = p_out_shift[vec_itr + 0];
      p_right_shift[2] = p_out_shift[vec_itr + 1];
      p_right_shift[1] = p_out_shift[vec_itr + 2];
      p_right_shift[0] = p_out_shift[vec_itr + 3];

      ae_int32x2 ls_32 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);
      ae_int32x2 ls_10 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]);

      ae_int32x2 rs_32 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
      ae_int32x2 rs_10 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]);

#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 ls3210 = AE_ADD16(AE_SAT16X4(ls_32, ls_10), AE_MOVDA16(17));
      ls_32 = AE_MOVINT32X2_FROMINT16X4(ls3210);
#endif

      (void)rs_32;
      (void)rs_10;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_shift[4], p_right_shift[4];

      p_left_shift[3] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[2] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[1] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[0] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[3] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[2] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[1] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[0] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      ae_int32x2 ls_32 = AE_SRAV32RS(AE_MOVI(1), AE_NEG32(AE_MOVDA32X2(p_left_shift[0], p_left_shift[1])));
      ae_int32x2 ls_10 = AE_SRAV32RS(AE_MOVI(1), AE_NEG32(AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]))); 

      ae_int32x2 rs_32 = AE_SRAV32RS(AE_MOVDA32(0x80000000), AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]));
      ae_int32x2 rs_10 = AE_SRAV32RS(AE_MOVDA32(0x80000000), AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]));
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 p_out_mult10, p_out_mult32;
      p_out_mult32 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 3], p_out_multiplier[vec_itr + 2]);
      p_out_mult10 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 1], p_out_multiplier[vec_itr + 0]);

      ae_int32x4 *pae_bias = (ae_int32x4 *)bias_buffer;
#pragma no_unroll
#pragma loop_count min=1
#pragma concurrent
      for (m_itr = 0; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, pae_bias, 0);

        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
        ae_valignx2 align_p_mat1_0;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IC(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);

        /* Apply quantization */
        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB_AV(out_0, acc_row1_vec0, acc_row0_vec0, p_out_mult32, p_out_mult10, ls_32, ls_10, rs_32, rs_10, out_zero_bias); 

        /* Store output */
        ae_int8x8 out32_0; 
        out32_0 = AE_SAT8X8X16(out_0, out_0);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }
    }
  }
  else if((cols1 == 40) && (vec_count & 0x3) == 0 && (out_col_offset == 1) && ((unsigned int)p_out & 0x3) == 0)
  {
    ae_int32x2 bias_buffer[2];
#ifdef AE_MULAZB8Q8X8
    AE_MOVZBVCDR(biascv1);
#endif

    int out_stride = out_row_offset;
    
    // Process loop for 4 rows and 4 vectors 
    for(vec_itr = 0; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1; 
      ae_int8x8 vec1_batch_0, vec1_batch_1; 
      ae_int8x8 vec2_batch_0, vec2_batch_1; 
      ae_int8x8 vec3_batch_0, vec3_batch_1;
      ae_int8x8 vec0_batch_2, vec0_batch_3; 
      ae_int8x8 vec1_batch_2, vec1_batch_3; 
      ae_int8x8 vec2_batch_2, vec2_batch_3; 
      ae_int8x8 vec3_batch_2, vec3_batch_3;
      ae_int8x8 vec0_batch_4, vec1_batch_4; 
      ae_int8x8 vec2_batch_4, vec3_batch_4;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3);
      
      AE_LA8X8X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_2, vec1_batch_3, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_2, vec2_batch_3, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_2, vec3_batch_3, align_p_vec3, (ae_int8x16 *)p_vec_3);
        
      ae_int8x8 temp0, temp1, temp2, temp3;
      AE_LAV8X8X2_XP(vec0_batch_4, temp0, align_p_vec0, (ae_int8x16 *)p_vec_0, 8);
      AE_LAV8X8X2_XP(vec1_batch_4, temp1, align_p_vec1, (ae_int8x16 *)p_vec_1, 8);
      AE_LAV8X8X2_XP(vec2_batch_4, temp2, align_p_vec2, (ae_int8x16 *)p_vec_2, 8);
      AE_LAV8X8X2_XP(vec3_batch_4, temp3, align_p_vec3, (ae_int8x16 *)p_vec_3, 8);
        
#ifndef AE_MULAZB8Q8X8
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_0, vec2_batch_0, vec1_batch_0, vec0_batch_0, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_1, vec2_batch_1, vec1_batch_1, vec0_batch_1, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_2, vec2_batch_2, vec1_batch_2, vec0_batch_2, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_3, vec2_batch_3, vec1_batch_3, vec0_batch_3, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_4, vec2_batch_4, vec1_batch_4, vec0_batch_4, mat_z_b);
      
      ae_int32x2 d_bias_0, d_bias_1;
      d_bias_1 = AE_MOVDA32X2(p_bias[vec_itr + 3], p_bias[vec_itr + 2]);
      d_bias_0 = AE_MOVDA32X2(p_bias[vec_itr + 1], p_bias[vec_itr + 0]);
      
      acc_row0 = AE_SUB32S(d_bias_0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias_1, acc_row1);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
#else
      ae_int32x2 d_bias_0, d_bias_1;
      ae_int32x2 bias_0, bias_1, bias_2, bias_3;
      bias_3 = AE_L32_I((ae_int32 *)p_bias, 12);
      bias_2 = AE_L32_I((ae_int32 *)p_bias, 8);
      bias_1 = AE_L32_I((ae_int32 *)p_bias, 4);
      AE_L32_IP(bias_0, (ae_int32 *)p_bias, 16);

      d_bias_0 = AE_SEL32_HH(bias_1, bias_0);
      d_bias_1 = AE_SEL32_HH(bias_3, bias_2);
      AE_S32X2X2_I(d_bias_0, d_bias_1, (ae_int32x4 *)bias_buffer, 0);
#endif
      
      /* Shifts to match with Tensorflow */
#if TFLITE_SINGLE_ROUNDING
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[3] = p_out_shift[vec_itr + 0];
      p_left_shift[2] = p_out_shift[vec_itr + 1];
      p_left_shift[1] = p_out_shift[vec_itr + 2];
      p_left_shift[0] = p_out_shift[vec_itr + 3];

      p_right_shift[3] = p_out_shift[vec_itr + 0];
      p_right_shift[2] = p_out_shift[vec_itr + 1];
      p_right_shift[1] = p_out_shift[vec_itr + 2];
      p_right_shift[0] = p_out_shift[vec_itr + 3];
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[3] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[2] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[1] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[0] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[3] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[2] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[1] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[0] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 p_out_mult10, p_out_mult32;
      p_out_mult32 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 3], p_out_multiplier[vec_itr + 2]);
      p_out_mult10 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 1], p_out_multiplier[vec_itr + 0]);

#pragma concurrent
      for (m_itr = 0; m_itr < (rows & ~(2 - 1)); m_itr += 2)
      {
        ae_int32x2 acc_row0_vec0, acc_row0_vec1;
        ae_int32x2 acc_row1_vec0, acc_row1_vec1;
        
        /* Initialize accumulators with bias - (ker * inp_zero_bias) */
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)bias_buffer, 0);

        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3, mat1_row0_4;
        ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3, mat1_row1_4;

        /* p_mat needs to be accessed in circular fashion */
        ae_int8x8* p_mat1_1 = p_mat1_0;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_stride1 * sizeof(WORD8));

        ae_valign align_p_mat1_0, align_p_mat1_1;

        AE_LA8X8POS_PC(align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8POS_PC(align_p_mat1_1, (ae_int8x8 *)p_mat1_1);

        AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        
        AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        
        AE_LA8X8_IC(mat1_row0_2, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_2, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        
        AE_LA8X8_IC(mat1_row0_3, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_3, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        
        AE_LA8X8_IC(mat1_row0_4, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_4, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        MAT_VEC_MUL(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row1_0);
        
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        MAT_VEC_MUL(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row1_1);

        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        MAT_VEC_MUL(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row1_2);
        
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);
        MAT_VEC_MUL(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row1_3);
        
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row0_4);
        MAT_VEC_MUL(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row1_4);
       
        /* Apply quantization */
        ae_int32x2 ls_32, ls_10, rs_32, rs_10;
        AE_L32X2X2_I(ls_32, ls_10, (ae_int32x4 *)p_left_shift, 0);
        AE_L32X2X2_I(rs_32, rs_10, (ae_int32x4 *)p_right_shift, 0);

        ae_int16x4 out_0, out_1;
        MPY_BY_QUANT_MULT_PER_CHAN_X2X2_X2_OUT16_ZB(out_0, out_1, acc_row1_vec0, acc_row0_vec0, acc_row1_vec1, acc_row0_vec1, p_out_mult32, p_out_mult10, ls_32, ls_10, rs_32, rs_10, out_zero_bias);

        /* Store output */
        ae_int8x8 out32_0;
        out32_0 = AE_SAT8X8X16(out_0, out_1);
        
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }

      //remaining rows
#pragma loop_count max=1
      for (m_itr = (rows & ~(2-1)); m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_int8x8 mat1_row0_2, mat1_row0_3;
        ae_int8x8 mat1_row0_4;
        ae_valign align_p_mat1_0;

        AE_LA8X8POS_PC(align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

        AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_2, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_3, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_4, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);
        MAT_VEC_MUL(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row0_4);

        /* Apply quantization */
        ae_int32x2 ls_32, ls_10, rs_32, rs_10;
        AE_L32X2X2_I(ls_32, ls_10, (ae_int32x4 *)p_left_shift, 0);
        AE_L32X2X2_I(rs_32, rs_10, (ae_int32x4 *)p_right_shift, 0);
        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB(out_0, acc_row1_vec0, acc_row0_vec0, p_out_mult32, p_out_mult10, ls_32, ls_10, rs_32, rs_10, out_zero_bias); 
        
        /* Store output */
        ae_int8x8 out32_0; 
        out32_0 = AE_SAT8X8X16(out_0, out_0);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }
    }
  }
  else if(((((unsigned)p_mat1) & 7) == 0) && ((((unsigned)p_vec1) & 7) == 0) && ((((unsigned)p_bias) & 3) == 0) &&
     ((row_stride1 & 15) == 0) && (vec_stride & 15) == 0) 
  {
    m_itr = 0, vec_itr = 0;

#ifndef AE_MULAZB8Q8X8
    ae_int32x2 acc_buffer[4];
#endif
    
    int out_stride = out_row_offset;
    int out_offset = out_col_offset;

    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_int8x8 *p_vec_1 = (ae_int8x8*)((WORD8 *)p_vec_0 + vec_stride); 
      ae_int8x8 *p_vec_2 = (ae_int8x8*)((WORD8 *)p_vec_1 + vec_stride); 
      ae_int8x8 *p_vec_3 = (ae_int8x8*)((WORD8 *)p_vec_2 + vec_stride); 

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&7);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_L8X8_IP(vec0_0, p_vec_0, 8);
        AE_L8X8_IP(vec1_0, p_vec_1, 8);
        AE_L8X8_IP(vec2_0, p_vec_2, 8);
        AE_L8X8_IP(vec3_0, p_vec_3, 8);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_L8X8_IP(vec0_0, p_vec_0, 8);
        AE_L8X8_IP(vec1_0, p_vec_1, 8);
        AE_L8X8_IP(vec2_0, p_vec_2, 8);
        AE_L8X8_IP(vec3_0, p_vec_3, 8);

        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift), rem_cols_shift));
        vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift), rem_cols_shift));
        vec2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_0), rem_cols_shift), rem_cols_shift));
        vec3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_0), rem_cols_shift), rem_cols_shift));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
      }

      acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]), acc_row0);
      acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]), acc_row1);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);
#endif
     
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
#if TFLITE_SINGLE_ROUNDING
      int p_right_shift[4], p_left_shift[4];

      p_left_shift[0] = p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3];

      ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);
      ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]);

      ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
      ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]);

#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 ls_0123 = AE_ADD16(AE_SAT16X4(ls_01, ls_23), AE_MOVDA16(17));
      ls_01 = AE_MOVINT32X2_FROMINT16X4(ls_0123);
#endif
      (void)rs_01;
      (void)rs_23;
#else /* #if TFLITE_SINGLE_ROUNDING */
      xtbool2 b0, b1;
      int p_right_shift[4], p_left_shift[4];
      b0 = AE_LT32(AE_MOVDA32X2(p_out_shift[vec_itr], p_out_shift[vec_itr + 1]), ZERO32);
      b1 = AE_LT32(AE_MOVDA32X2(p_out_shift[vec_itr + 2], p_out_shift[vec_itr + 3]), ZERO32);

      ae_int32x2 temp_0 = ZERO32, temp_1 = ZERO32;
      ae_int32x2 temp_2 = ZERO32, temp_3 = ZERO32;
      AE_MOVT32X2(temp_0, AE_MOVDA32X2(-p_out_shift[vec_itr], -p_out_shift[vec_itr + 1]), b0);
      AE_MOVT32X2(temp_1, AE_MOVDA32X2(-p_out_shift[vec_itr + 2], -p_out_shift[vec_itr + 3]), b1);
      AE_MOVF32X2(temp_2, AE_MOVDA32X2(p_out_shift[vec_itr], p_out_shift[vec_itr + 1]), b0);
      AE_MOVF32X2(temp_3, AE_MOVDA32X2(p_out_shift[vec_itr + 2], p_out_shift[vec_itr + 3]), b1);

      p_left_shift[0] = AE_MOVAD32_H(temp_2);
      p_left_shift[1] = AE_MOVAD32_L(temp_2);
      p_left_shift[2] = AE_MOVAD32_H(temp_3);
      p_left_shift[3] = AE_MOVAD32_L(temp_3);

      p_right_shift[0] = AE_MOVAD32_H(temp_0);
      p_right_shift[1] = AE_MOVAD32_L(temp_0);
      p_right_shift[2] = AE_MOVAD32_H(temp_1);
      p_right_shift[3] = AE_MOVAD32_L(temp_1);

      ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);
      ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]);

      ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
      ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]); 
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 p_out_mult01 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 0], p_out_multiplier[vec_itr + 1]);
      ae_int32x2 p_out_mult23 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 2], p_out_multiplier[vec_itr + 3]);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
#ifndef AE_MULAZB8Q8X8
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
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
        ae_int32x2 acc_row0_vec1 = AE_MOVDA32(p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec1 = acc_row0_vec1;
        ae_int32x2 acc_row0_vec2 = AE_MOVDA32(p_bias[vec_itr + 2]);
        ae_int32x2 acc_row1_vec2 = acc_row0_vec2;
        ae_int32x2 acc_row0_vec3 = AE_MOVDA32(p_bias[vec_itr + 3]);
        ae_int32x2 acc_row1_vec3 = acc_row0_vec3;
#endif
        
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biasvc1);
#endif
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
           ,vec_stride
          );

        ae_int16x4 out_0, out_1, out_2, out_3;
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], p_left_shift[0], p_right_shift[0], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[vec_itr + 1], p_left_shift[1], p_right_shift[1], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[vec_itr + 2], p_left_shift[2], p_right_shift[2], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[vec_itr + 3], p_left_shift[3], p_right_shift[3], out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

        /* Store output */
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]);
#endif

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biascv1);
#endif
        _xa_nn_dot_product_1_rows_4_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols1
           ,vec_stride
#ifdef AE_MULAZB8Q8X8
           ,-mat1_offset
#endif
          );

        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult01, p_out_mult23, ls_01, ls_23, rs_01, rs_23, out_zero_bias); 

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&7);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_L8X8_IP(vec0_0, p_vec_0, 8);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_L8X8_IP(vec0_0, p_vec_0, 8);
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift), rem_cols_shift));
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
      }
#endif

      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
      int temp_ls, temp_rs;
#if TFLITE_SINGLE_ROUNDING
      temp_ls = p_out_shift[vec_itr];
      temp_rs = p_out_shift[vec_itr];
      (void)temp_rs;
#else /* #if TFLITE_SINGLE_ROUNDING */
      temp_ls = p_out_shift[vec_itr] < 0 ? 0 : p_out_shift[vec_itr];
      temp_rs = -p_out_shift[vec_itr] < 0 ? 0 : -p_out_shift[vec_itr];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int32x2 acc_row0_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), AE_MOVDA32(AE_MOVAD32_H(acc_row0)));
        ae_int32x2 acc_row1_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), AE_MOVDA32(AE_MOVAD32_H(acc_row1)));
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
#endif

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biasvc1);
#endif
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], temp_ls, temp_rs, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int32x2 acc_row0_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), acc_row0);
        ae_int32x2 acc_row1_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), acc_row1);
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
#endif

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biasvc1);
#endif
        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );

        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_X2_OUT16(out_0, acc_row0_vec0, p_out_multiplier[vec_itr], temp_ls, temp_rs);
        out_0 = AE_ADD16S(out_0, AE_MOVDA16(out_zero_bias));
        ae_int8x8 temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        //TODO: AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset;
    int out_offset = out_col_offset;

#ifndef AE_MULAZB8Q8X8
    ae_int32x2 acc_buffer[4];
    int rem_cols = cols1 & 15;
#endif
    
    for(vec_itr = 0; vec_itr < (vec_count & ~(32 - 1)); vec_itr += 32)
    {
      int ii;
      for(ii = 0; ii < 8; ii++)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr + ii) * vec_stride];
        ae_int8x8 *p_vec_1 = p_vec_0 + vec_stride; 
        ae_int8x8 *p_vec_2 = p_vec_1 + vec_stride; 
        ae_int8x8 *p_vec_3 = p_vec_2 + vec_stride; 

        ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
        ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
        ae_valignx2 align_p_vec_2 = AE_LA128_PP(p_vec_2);
        ae_valignx2 align_p_vec_3 = AE_LA128_PP(p_vec_3);

        ae_int32x2 acc_row0 = ZERO32; 
        ae_int32x2 acc_row1 = ZERO32;

        int cols_count=cols1-(cols1&15);
#pragma no_unroll
        for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
        {
          AE_LA8X8X2_IP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0);
          AE_LA8X8X2_IP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1);
          AE_LA8X8X2_IP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2);
          AE_LA8X8X2_IP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
        }

        //Remainder loop for cols1
        if(cols_count!=cols1)
        {
          AE_LAV8X8X2_XP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0, rem_cols);
          AE_LAV8X8X2_XP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1, rem_cols);
          AE_LAV8X8X2_XP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2, rem_cols);
          AE_LAV8X8X2_XP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3, rem_cols);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);

          if(rem_cols > 8)
          {
            AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
          }
        }

        acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + ii +  0], p_bias[vec_itr + ii +  8]), acc_row0);
        acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + ii + 16], p_bias[vec_itr + ii + 24]), acc_row1);
        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);
#endif
        
        WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + ii +  0) * out_offset;
        WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + ii +  8) * out_offset;
        WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + ii + 16) * out_offset;
        WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + ii + 24) * out_offset;

        m_itr = 0;

        /* Shifts to match with Tensorflow */
#if TFLITE_SINGLE_ROUNDING
        int p_left_shift[4], p_right_shift[4];
        
        p_left_shift[0] = p_out_shift[vec_itr + ii +  0];
        p_left_shift[1] = p_out_shift[vec_itr + ii +  8];
        p_left_shift[2] = p_out_shift[vec_itr + ii + 16];
        p_left_shift[3] = p_out_shift[vec_itr + ii + 24];

        p_right_shift[0] = p_out_shift[vec_itr + ii +  0];
        p_right_shift[1] = p_out_shift[vec_itr + ii +  8];
        p_right_shift[2] = p_out_shift[vec_itr + ii + 16];
        p_right_shift[3] = p_out_shift[vec_itr + ii + 24];

        ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
        ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

        ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]); 
        ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);

#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 ls_0123 = AE_ADD16(AE_SAT16X4(ls_01, ls_23), AE_MOVDA16(17));
      ls_01 = AE_MOVINT32X2_FROMINT16X4(ls_0123);
#endif
        (void)rs_23;
        (void)rs_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
        int p_left_shift[4], p_right_shift[4];
        
        p_left_shift[0] = p_out_shift[vec_itr + ii +  0] < 0 ? 0 : p_out_shift[vec_itr + ii +  0];
        p_left_shift[1] = p_out_shift[vec_itr + ii +  8] < 0 ? 0 : p_out_shift[vec_itr + ii +  8];
        p_left_shift[2] = p_out_shift[vec_itr + ii + 16] < 0 ? 0 : p_out_shift[vec_itr + ii + 16];
        p_left_shift[3] = p_out_shift[vec_itr + ii + 24] < 0 ? 0 : p_out_shift[vec_itr + ii + 24];

        p_right_shift[0] = p_out_shift[vec_itr + ii +  0] > 0 ? 0 : -p_out_shift[vec_itr + ii +  0];
        p_right_shift[1] = p_out_shift[vec_itr + ii +  8] > 0 ? 0 : -p_out_shift[vec_itr + ii +  8];
        p_right_shift[2] = p_out_shift[vec_itr + ii + 16] > 0 ? 0 : -p_out_shift[vec_itr + ii + 16];
        p_right_shift[3] = p_out_shift[vec_itr + ii + 24] > 0 ? 0 : -p_out_shift[vec_itr + ii + 24];

        ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
        ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

        ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]); 
        ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
#endif /* #if TFLITE_SINGLE_ROUNDING */

        ae_int32x2 p_out_mult01 = AE_MOVDA32X2(p_out_multiplier[vec_itr + ii + 0], p_out_multiplier[vec_itr + ii + 8]);
        ae_int32x2 p_out_mult23 = AE_MOVDA32X2(p_out_multiplier[vec_itr + ii + 16], p_out_multiplier[vec_itr + ii + 24]);
        for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
        {
#ifndef AE_MULAZB8Q8X8
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
#else
          ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + ii + 0]);
          ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
          ae_int32x2 acc_row0_vec1 = AE_MOVDA32(p_bias[vec_itr + ii + 8]);
          ae_int32x2 acc_row1_vec1 = acc_row0_vec1;
          ae_int32x2 acc_row0_vec2 = AE_MOVDA32(p_bias[vec_itr + ii + 16]);
          ae_int32x2 acc_row1_vec2 = acc_row0_vec2;
          ae_int32x2 acc_row0_vec3 = AE_MOVDA32(p_bias[vec_itr + ii + 24]);
          ae_int32x2 acc_row1_vec3 = acc_row0_vec3;
#endif
          
          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + (vec_itr + ii) * vec_stride);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
          AE_MOVZBVCDR(biasvc1);
#endif
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
             ,vec_stride
#ifdef AE_MULAZB8Q8X8
             ,0
#endif
            );

            ae_int16x4 out_0, out_1, out_2, out_3;
            MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + ii + 0], p_left_shift[0], p_right_shift[0], out_zero_bias);
            MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[vec_itr + ii + 8], p_left_shift[1], p_right_shift[1], out_zero_bias);
            MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[vec_itr + ii + 16], p_left_shift[2], p_right_shift[2], out_zero_bias);
            MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[vec_itr + ii + 24], p_left_shift[3], p_right_shift[3], out_zero_bias);

            AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
            AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
            AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
            AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

            /* Store output */
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);
        }

        // Remaining rows 
        // TODO: codesize considerations: remove inline for 4 row, 1 vec case ?
        for (; m_itr < rows; m_itr++)
        {
#ifndef AE_MULAZB8Q8X8
          ae_int32x2 acc_row0_vec0 = acc_row0;
          ae_int32x2 acc_row1_vec0 = acc_row1;
#else
          ae_int32x2 acc_row0_vec0 = AE_MOVDA32X2(p_bias[vec_itr + ii + 0], p_bias[vec_itr + ii + 8]);
          ae_int32x2 acc_row1_vec0 = AE_MOVDA32X2(p_bias[vec_itr + ii + 16], p_bias[vec_itr + ii + 24]);
#endif

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + (vec_itr + ii) * vec_stride);
          ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
          AE_MOVZBVCDR(biascv1);
#endif
          _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,(ae_int8x8*)p_vec_0
             ,(ae_int8*)p_mat1_0
             ,cols1
             ,vec_stride
#ifdef AE_MULAZB8Q8X8
             ,-mat1_offset
#endif
            );

          ae_int16x4 out_0;
          MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult01, p_out_mult23, ls_01, ls_23, rs_01, rs_23, out_zero_bias); 

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
        }
      }
    }
    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_int8x8 *p_vec_1 = (ae_int8x8*)((WORD8 *)p_vec_0 + vec_stride); 
      ae_int8x8 *p_vec_2 = (ae_int8x8*)((WORD8 *)p_vec_1 + vec_stride); 
      ae_int8x8 *p_vec_3 = (ae_int8x8*)((WORD8 *)p_vec_2 + vec_stride); 

      ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
      ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
      ae_valignx2 align_p_vec_2 = AE_LA128_PP(p_vec_2);
      ae_valignx2 align_p_vec_3 = AE_LA128_PP(p_vec_3);

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

        int cols_count=cols1-(cols1&15);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
          AE_LA8X8X2_IP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0);
          AE_LA8X8X2_IP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1);
          AE_LA8X8X2_IP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2);
          AE_LA8X8X2_IP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LAV8X8X2_XP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0, rem_cols);
        AE_LAV8X8X2_XP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1, rem_cols);
        AE_LAV8X8X2_XP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2, rem_cols);
        AE_LAV8X8X2_XP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3, rem_cols);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);

        if(rem_cols > 8)
        {
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
        }
      }
      
      acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]), acc_row0);
      acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]), acc_row1);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);
#endif
        
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
#if TFLITE_SINGLE_ROUNDING
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[0] = p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3];

      ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
      ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

      ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]); 
      ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);

#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 ls_0123 = AE_ADD16(AE_SAT16X4(ls_01, ls_23), AE_MOVDA16(17));
      ls_01 = AE_MOVINT32X2_FROMINT16X4(ls_0123);
#endif
      (void)rs_23;
      (void)rs_01;
#else /* #if TFLITE_SINGLE_ROUNDING */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[0] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      ae_int32x2 ls_23 = AE_MOVDA32X2(p_left_shift[2], p_left_shift[3]); 
      ae_int32x2 ls_01 = AE_MOVDA32X2(p_left_shift[0], p_left_shift[1]);

      ae_int32x2 rs_23 = AE_MOVDA32X2(p_right_shift[2], p_right_shift[3]); 
      ae_int32x2 rs_01 = AE_MOVDA32X2(p_right_shift[0], p_right_shift[1]);
#endif /* #if TFLITE_SINGLE_ROUNDING */

      ae_int32x2 p_out_mult01 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 0], p_out_multiplier[vec_itr + 1]);
      ae_int32x2 p_out_mult23 = AE_MOVDA32X2(p_out_multiplier[vec_itr + 2], p_out_multiplier[vec_itr + 3]);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
#ifndef AE_MULAZB8Q8X8
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
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
        ae_int32x2 acc_row0_vec1 = AE_MOVDA32(p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec1 = acc_row0_vec1;
        ae_int32x2 acc_row0_vec2 = AE_MOVDA32(p_bias[vec_itr + 2]);
        ae_int32x2 acc_row1_vec2 = acc_row0_vec2;
        ae_int32x2 acc_row0_vec3 = AE_MOVDA32(p_bias[vec_itr + 3]);
        ae_int32x2 acc_row1_vec3 = acc_row0_vec3;
#endif
        
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biasvc1);
#endif
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
           ,vec_stride
#ifdef AE_MULAZB8Q8X8
           ,0
#endif
          );

        ae_int16x4 out_0, out_1, out_2, out_3;
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], p_left_shift[0], p_right_shift[0], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[vec_itr + 1], p_left_shift[1], p_right_shift[1], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[vec_itr + 2], p_left_shift[2], p_right_shift[2], out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[vec_itr + 3], p_left_shift[3], p_right_shift[3], out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
        AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

        /* Store output */
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]);
#endif

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biascv1);
#endif
        _xa_nn_dot_product_1_rows_4_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols1
           ,vec_stride
#ifdef AE_MULAZB8Q8X8
           ,-mat1_offset
#endif
          );

        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult01, p_out_mult23, ls_01, ls_23, rs_01, rs_23, out_zero_bias); 

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
#ifndef AE_MULAZB8Q8X8
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&15);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_LA8X8X2_IP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec0_1 , vec0_1 , vec0_1 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LAV8X8X2_XP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0, rem_cols);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
        
        if(rem_cols > 8)
        {
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec0_1 , vec0_1 , vec0_1 , mat_z_b);
        }
      }
      acc_row0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), acc_row0);
#endif

      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
      int temp_ls, temp_rs;
#if TFLITE_SINGLE_ROUNDING
      temp_ls = p_out_shift[vec_itr];
      temp_rs = p_out_shift[vec_itr];
      (void)temp_rs;
#else /* #if TFLITE_SINGLE_ROUNDING */
      temp_ls = p_out_shift[vec_itr] < 0 ? 0 : p_out_shift[vec_itr];
      temp_rs = -p_out_shift[vec_itr] < 0 ? 0 : -p_out_shift[vec_itr];
#endif /* #if TFLITE_SINGLE_ROUNDING */

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row0;
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
#endif

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biasvc1);
#endif
        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
#ifdef AE_MULAZB8Q8X8
           ,0
#endif
          );

        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], temp_ls, temp_rs, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
#ifndef AE_MULAZB8Q8X8
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row0;
#else
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = acc_row0_vec0;
#endif

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

#ifdef AE_MULAZB8Q8X8
        AE_MOVZBVCDR(biasvc1);
#endif
        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
#ifdef AE_MULAZB8Q8X8
           ,0
#endif
          );

        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_X2_OUT16(out_0, acc_row0_vec0, p_out_multiplier[vec_itr], temp_ls, temp_rs);
        out_0 = AE_ADD16S(out_0, AE_MOVDA16(out_zero_bias));
        ae_int8x8 temp_vec0 = AE_SAT8X8X16(out_0, out_0);

        //TODO: AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }
  else
    return -1;
  return 0;
}
