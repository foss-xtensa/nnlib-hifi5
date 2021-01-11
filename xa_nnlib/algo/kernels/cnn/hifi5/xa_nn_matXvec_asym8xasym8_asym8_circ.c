/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32(inp, left_shift); \
    inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out, inp, inp1, multiplier, l_shift, right_shift, out_off) \
    AE_MUL2P32X4S(inp, inp1, inp, inp1, l_shift, l_shift); \
    AE_MULF2P32X4RAS(inp, inp1, inp, inp1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);\
    inp1 = AE_SRAA32SYMS(inp1, right_shift);\
    out = AE_SAT16X4(inp, inp1); \
    out = AE_ADD16S(AE_MOVDA16(out_off), out); \
    AE_MINMAX16(out, AE_ZERO16(), AE_MOVDA16(255)); 

#define PACK_32X2(dst1, src1, src2) \
dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

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
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  int rem_cols = cols & 15;
  int rem_g8 = ((rem_cols & 15) > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1])); 
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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
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

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, sel1);
    
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
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
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_XC(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
  int c_itr = 0;
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  int cols_count=cols-(cols&7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
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
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  
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
  //pre_loop_shift = align_offset * 8;
  ae_int8x8 pre_sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset % 8)], pre_loop_sel_pattern[2 * (align_offset % 8) + 1])); 
  //TODO: circular access
  //p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, -align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step 
  loop_count = cols - pre_loop_count;
  post_loop_count = loop_count & 0xf;
  loop_count >>= 4;
  int rem_g8 = (post_loop_count > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (post_loop_count % 8) * !rem_g8], post_loop_sel_pattern[2 * (post_loop_count % 8) * !rem_g8 + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (post_loop_count % 8) * rem_g8], post_loop_sel_pattern[2 * (post_loop_count % 8) * rem_g8 + 1])); 

  ae_int8x8* p_mat1_1 = p_mat1_0; //next 8th row 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1,  1 * row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1; //next 8th row
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2,  1 * row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2; //next 8th row 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3,  1 * row_offset * sizeof(WORD8));

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
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, pre_sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, pre_sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, pre_sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, pre_sel1);
  }
  
  AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
  AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
  AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

  AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
  AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
  AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
  AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr = 0;
  while(c_itr < post_loop_count)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, sel1);
    
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

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

static inline void _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + 8 * row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + 8 * row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + 8 * row_offset); 

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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  int rem_cols = cols & 15;
  int rem_g8 = ((rem_cols & 15) > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); \
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1])); \
  
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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
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

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    vec1_batch_0 = AE_SEL8X8(vec1_batch_0, neg_vec_bias, sel1);
    vec2_batch_0 = AE_SEL8X8(vec2_batch_0, neg_vec_bias, sel1);
    vec3_batch_0 = AE_SEL8X8(vec3_batch_0, neg_vec_bias, sel1);
    
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
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
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
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

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
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
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols % 8)], post_loop_sel_pattern[2 * (cols % 8) + 1]));
  
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0;
  ae_valign align_p_vec0; 
  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  align_p_vec0 = AE_LA64_PP(p_vec_0);

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);

    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matXvec_asym8xasym8_asym8_circ(
    UWORD8 * __restrict__ p_out,
    UWORD8 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 vec1_offset,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
  /* Iterators used in for loops */
  int m_itr, vec_itr;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

  if((out_shift > 31) || (out_shift < -31))
  {
    return -1;
  }

  if (!p_bias)
  {
    return -1;
  }

  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;

  ae_int32x2 bias_buffer[2];

  /*Load AE_BIASV8 and AE_BIASC8 state registers with mat1 and vec1 zero bias values*/
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec1_offset, -mat1_offset));
  ae_int64 biascv1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-mat1_offset, -vec1_offset));
 
  ae_int32x2 max_uint8 = AE_MOVDA32(255);
  ae_int32x2 min_uint8 = AE_MOVDA32(0);
  
  ae_int32x2 l_mult = AE_MOVDA32(1 << left_shift);

  if(cols1 > 8 && cols1 < 16 && (vec_count & 0x3) == 0 && (out_col_offset == 1))
  {
    m_itr = 0, vec_itr = 0;

    AE_MOVZBVCDR(biascv1);

    int out_stride = out_row_offset;

    ae_int32x2 d_bias_0, d_bias_1;
    ae_int32x4 *pt_bias = (ae_int32x4 *)p_bias;
    ae_valignx2 align_p_bias = AE_LA128_PP(pt_bias);
    
    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      AE_LA32X2X2_IP(d_bias_0, d_bias_1, align_p_bias, (ae_int32x4 *)pt_bias);
      AE_S32X2X2_I(d_bias_0, d_bias_1, (ae_int32x4 *)bias_buffer, 0);
      
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)-vec1_offset);
      int rem_cols = cols1 & 15;
      ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8)], post_loop_sel_pattern[2 * (rem_cols % 8) + 1])); \
      
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

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3);
      
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel1);
      vec1_batch_1 = AE_SEL8X8(vec1_batch_1, neg_vec_bias, sel1);
      vec2_batch_1 = AE_SEL8X8(vec2_batch_1, neg_vec_bias, sel1);
      vec3_batch_1 = AE_SEL8X8(vec3_batch_1, neg_vec_bias, sel1);
        
      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
        ae_int32x2 acc_row1_vec0, acc_row1_vec1, acc_row1_vec2, acc_row1_vec3;
        
        /* Initialize accumulators with bias */
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)bias_buffer, 0);

        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_int8x8 mat1_row1_0, mat1_row1_1;
        ae_int8x8 mat1_row2_0, mat1_row2_1;
        ae_int8x8 mat1_row3_0, mat1_row3_1;

        /* p_mat needs to be accessed in circular fashion */
        ae_int8x8* p_mat1_1 = p_mat1_0;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_2 = p_mat1_1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_3 = p_mat1_2;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_stride1 * sizeof(WORD8));

        ae_valignx2 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2,align_p_mat1_3;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2POS_PC(align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2POS_PC(align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2POS_PC(align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0);
        AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0);
        AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0);
        AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0);

        AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row0_1);
        AE_MULAUUZB8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row1_1);
        AE_MULAUUZB8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row2_1);
        AE_MULAUUZB8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row3_1);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row0_vec1, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec2, acc_row0_vec3, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row1_vec0, acc_row1_vec1, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row1_vec2, acc_row1_vec3, out_multiplier, l_mult, right_shift, out_zero_bias);
        
        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        PACK_32X2(out32_0, out_0, out_1);
        PACK_32X2(out32_1, out_2, out_3);
        
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_valignx2 align_p_mat1_0;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0);
        AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row0_1);

        /* Apply quantization */
        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);
        
        /* Store output */
        ae_int8x8 out32_0; 
        PACK_32X2(out32_0, out_0, out_0);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }
    }
  }
  else if(((((unsigned)p_mat1) & 7) == 0) && ((((unsigned)p_vec1) & 7) == 0) && ((((unsigned)p_bias) & 3) == 0) &&
     ((row_stride1 & 15) == 0) && (vec_stride & 15) == 0) 
  {
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset;
    int out_offset = out_col_offset;

    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row0_vec1 = AE_MOVDA32(p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec1 = AE_MOVDA32(p_bias[vec_itr + 1]);
        ae_int32x2 acc_row0_vec2 = AE_MOVDA32(p_bias[vec_itr + 2]);
        ae_int32x2 acc_row1_vec2 = AE_MOVDA32(p_bias[vec_itr + 2]);
        ae_int32x2 acc_row0_vec3 = AE_MOVDA32(p_bias[vec_itr + 3]);
        ae_int32x2 acc_row1_vec3 = AE_MOVDA32(p_bias[vec_itr + 3]);
        
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

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
           ,vec_stride
           ,-vec1_offset
          );

        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier, l_mult, right_shift, out_zero_bias);
        
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
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        AE_MOVZBVCDR(biascv1);
        _xa_nn_dot_product_1_rows_4_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols1
           ,vec_stride
           ,-mat1_offset
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
           ,-vec1_offset
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,-vec1_offset
          );

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);

        ae_int8x8 temp_vec0 = AE_SATU8X4X32_L(acc_row0_vec0, acc_row0_vec0);

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

    for(vec_itr = 0; vec_itr < (vec_count & ~(32 - 1)); vec_itr += 32)
    {
      int ii;
      for(ii = 0; ii < 8; ii++)
      {
        WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + ii +  0) * out_offset;
        WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + ii +  8) * out_offset;
        WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + ii + 16) * out_offset;
        WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + ii + 24) * out_offset;

        m_itr = 0;

        for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
        {
          ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + ii + 0]);
          ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + ii + 0]);
          ae_int32x2 acc_row0_vec1 = AE_MOVDA32(p_bias[vec_itr + ii + 8]);
          ae_int32x2 acc_row1_vec1 = AE_MOVDA32(p_bias[vec_itr + ii + 8]);
          ae_int32x2 acc_row0_vec2 = AE_MOVDA32(p_bias[vec_itr + ii + 16]);
          ae_int32x2 acc_row1_vec2 = AE_MOVDA32(p_bias[vec_itr + ii + 16]);
          ae_int32x2 acc_row0_vec3 = AE_MOVDA32(p_bias[vec_itr + ii + 24]);
          ae_int32x2 acc_row1_vec3 = AE_MOVDA32(p_bias[vec_itr + ii + 24]);

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + (vec_itr + ii) * vec_stride);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

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
             ,vec_stride
             ,-vec1_offset
            );

            ae_int16x4 out_0, out_1, out_2, out_3;
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier, l_mult, right_shift, out_zero_bias);
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier, l_mult, right_shift, out_zero_bias);
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier, l_mult, right_shift, out_zero_bias);
            
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
          ae_int32x2 acc_row0_vec0 = AE_MOVDA32X2(p_bias[vec_itr + ii + 0], p_bias[vec_itr + ii + 8]);
          ae_int32x2 acc_row1_vec0 = AE_MOVDA32X2(p_bias[vec_itr + ii + 16], p_bias[vec_itr + ii + 24]);

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + (vec_itr + ii) * vec_stride);
          ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

          AE_MOVZBVCDR(biascv1);
          _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,(ae_int8x8*)p_vec_0
             ,(ae_int8*)p_mat1_0
             ,cols1
             ,vec_stride
             ,-mat1_offset
            );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);

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
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row0_vec1 = AE_MOVDA32(p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec1 = AE_MOVDA32(p_bias[vec_itr + 1]);
        ae_int32x2 acc_row0_vec2 = AE_MOVDA32(p_bias[vec_itr + 2]);
        ae_int32x2 acc_row1_vec2 = AE_MOVDA32(p_bias[vec_itr + 2]);
        ae_int32x2 acc_row0_vec3 = AE_MOVDA32(p_bias[vec_itr + 3]);
        ae_int32x2 acc_row1_vec3 = AE_MOVDA32(p_bias[vec_itr + 3]);
        
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

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
           ,vec_stride
           ,-vec1_offset
          );

        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier, l_mult, right_shift, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier, l_mult, right_shift, out_zero_bias);
        
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
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        AE_MOVZBVCDR(biascv1);
        _xa_nn_dot_product_1_rows_4_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols1
           ,vec_stride
           ,-mat1_offset
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
           ,-vec1_offset
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);
        ae_int32x2 acc_row1_vec0 = AE_MOVDA32(p_bias[vec_itr + 0]);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        AE_MOVZBVCDR(biasvc1);
        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,-vec1_offset
          );

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);

        ae_int8x8 temp_vec0 = AE_SATU8X4X32_L(acc_row0_vec0, acc_row0_vec0);

        //TODO: AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }
  else
    return -1;
  return 0;
}
