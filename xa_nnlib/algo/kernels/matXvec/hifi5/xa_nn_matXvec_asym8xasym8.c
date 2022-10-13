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

extern const long long post_loop_sel_pattern[16];

static inline void _xa_nn_dot_product_8_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_2_0
    ,ae_int32x2* out_3_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec1_zero_bias
    )
{
  int rem_cols = (cols1 & 31);
  int rem_g16 = (rem_cols > 16)?1:0;
  int rem_ge16 = (rem_cols < 16)?0:1;
  int rem_cols_16 = rem_g16?((cols1 - 16) & 15):(cols1 & 15);
  int rem_g8 = (rem_cols_16 > 8)?1:0;
  int rem_cols8 = rem_cols_16 % 8;  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16) + 1])); 
  ae_int8x8 sel4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16) + 1])); 
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  
  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 mat1_row4_0, mat1_row4_1, mat1_row4_2, mat1_row4_3;
  ae_int8x8 mat1_row5_0, mat1_row5_1, mat1_row5_2, mat1_row5_3;
  ae_int8x8 mat1_row6_0, mat1_row6_1, mat1_row6_2, mat1_row6_3;
  ae_int8x8 mat1_row7_0, mat1_row7_1, mat1_row7_2, mat1_row7_3;
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1); 
  ae_int8x8 *p_mat1_4 = (ae_int8x8*)((WORD8 *)p_mat1_3 + row_stride1); 
  ae_int8x8 *p_mat1_5 = (ae_int8x8*)((WORD8 *)p_mat1_4 + row_stride1); 
  ae_int8x8 *p_mat1_6 = (ae_int8x8*)((WORD8 *)p_mat1_5 + row_stride1); 
  ae_int8x8 *p_mat1_7 = (ae_int8x8*)((WORD8 *)p_mat1_6 + row_stride1); 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row2_vec0 = *out_2_0;
  ae_int32x2 acc_row3_vec0 = *out_3_0;

  for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_I(mat1_row1_2, mat1_row1_3, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_I(mat1_row2_2, mat1_row2_3, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_I(mat1_row3_2, mat1_row3_3, (ae_int8x16 *)p_mat1_3, 16);
    AE_L8X8X2_I(mat1_row4_2, mat1_row4_3, (ae_int8x16 *)p_mat1_4, 16);
    AE_L8X8X2_I(mat1_row5_2, mat1_row5_3, (ae_int8x16 *)p_mat1_5, 16);
    AE_L8X8X2_I(mat1_row6_2, mat1_row6_3, (ae_int8x16 *)p_mat1_6, 16);
    AE_L8X8X2_I(mat1_row7_2, mat1_row7_3, (ae_int8x16 *)p_mat1_7, 16);

    AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, (ae_int8x16 *)p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 32);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 32);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 32);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 32);
    AE_L8X8X2_IP(mat1_row4_0, mat1_row4_1, (ae_int8x16 *)p_mat1_4, 32);
    AE_L8X8X2_IP(mat1_row5_0, mat1_row5_1, (ae_int8x16 *)p_mat1_5, 32);
    AE_L8X8X2_IP(mat1_row6_0, mat1_row6_1, (ae_int8x16 *)p_mat1_6, 32);
    AE_L8X8X2_IP(mat1_row7_0, mat1_row7_1, (ae_int8x16 *)p_mat1_7, 32);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 32);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_0 , mat1_row5_0 , mat1_row6_0 , mat1_row7_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULAUUZB8Q8X8(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_1 , mat1_row5_1 , mat1_row6_1 , mat1_row7_1 ,vec0_batch_1);
    
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
    AE_MULAUUZB8Q8X8(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_2 , mat1_row5_2 , mat1_row6_2 , mat1_row7_2 ,vec0_batch_2);
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3);
    AE_MULAUUZB8Q8X8(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_3 , mat1_row5_3 , mat1_row6_3 , mat1_row7_3 ,vec0_batch_3);
  }
  //Remainder loop for cols1
  int flag_itr = 0;
  while(rem_cols > 0)
  {
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);
    AE_L8X8X2_IP(mat1_row4_0, mat1_row4_1, (ae_int8x16 *)p_mat1_4, 16);
    AE_L8X8X2_IP(mat1_row5_0, mat1_row5_1, (ae_int8x16 *)p_mat1_5, 16);
    AE_L8X8X2_IP(mat1_row6_0, mat1_row6_1, (ae_int8x16 *)p_mat1_6, 16);
    AE_L8X8X2_IP(mat1_row7_0, mat1_row7_1, (ae_int8x16 *)p_mat1_7, 16);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_0 , mat1_row5_0 , mat1_row6_0 , mat1_row7_0 ,vec0_batch_0);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
      AE_MULAUUZB8Q8X8(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_1 , mat1_row5_1 , mat1_row6_1 , mat1_row7_1 ,vec0_batch_1);
      flag_itr = 1;
    }
    rem_cols -= 16;
    sel1 = sel3; 
    sel2 = sel4;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
  *out_2_0 = acc_row2_vec0;
  *out_3_0 = acc_row3_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      vec1_zero_bias
    )
{
  int rem_cols = (cols1 & 31);
  int rem_g16 = (rem_cols > 16)?1:0;
  int rem_ge16 = (rem_cols < 16)?0:1;
  int rem_cols_16 = rem_g16?((cols1 - 16) & 15):(cols1 & 15);
  int rem_g8 = (rem_cols_16 > 8)?1:0;
  int rem_cols8 = rem_cols_16 % 8;  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16) + 1])); 
  ae_int8x8 sel4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16) + 1])); 
  
  int c_itr = 0;
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

#pragma no_unroll
  for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 32);

    AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, (ae_int8x16 *)p_vec_0, 16);
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 32);


    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, vec0_batch_1);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_2, mat1_row0_2, mat1_row0_2, mat1_row0_2, vec0_batch_2);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_3, mat1_row0_3, mat1_row0_3, mat1_row0_3, vec0_batch_3);
  }
  //Remainder loop for cols1
  int flag_itr = 0;
  while(rem_cols > 0)
  {
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, vec0_batch_1);
      flag_itr = 1;
    }
    rem_cols -= 16;
    sel1 = sel3; 
    sel2 = sel4;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

static inline void _xa_nn_dot_product_3_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec1_zero_bias
    )
{
  int rem_cols = (cols1 & 15);
  int rem_g8 = (rem_cols > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); \
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1])); \
  int c_itr = 0;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 vec0_batch_0, vec0_batch_1; 

  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 

  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
  ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
  ae_valignx2 align_p_vec0 = AE_LA128_PP(p_vec_0);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count = cols1 - (cols1 & 15);
  for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
    AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
    AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);

    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , neg_vec_bias ,vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , neg_vec_bias ,vec0_batch_1);
  }

  //Remainder loop for cols1
  if(cols_count != cols1)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
    AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
    AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, neg_vec_bias, vec0_batch_0);
    
    if(rem_g8)
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , neg_vec_bias ,vec0_batch_1);
    }
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
    ,WORD32      vec1_zero_bias
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
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); \
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1])); \
  ae_int8x8 neg_vec_bias = AE_MOVDA8((UWORD8)vec1_zero_bias);
  int cols_count = cols1 - (cols1 & 15);
  for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int8x16 *)p_vec_0);

    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, vec0_batch_1);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int8x16 *)p_vec_0);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_MULAUUZB8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
    
    if(rem_g8)
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_MULAUUZB8Q8X8(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 ,vec0_batch_1);
    }
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matXvec_asym8xasym8_asym8(
    UWORD8 * __restrict__ p_out,
    const UWORD8 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_mat2,
    const UWORD8 * __restrict__ p_vec1,
    const UWORD8 * __restrict__ p_vec2,
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
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -255 || mat1_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -255 || vec1_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    XA_NNLIB_ARG_CHK_COND((mat2_zero_bias < -255 || mat2_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((vec2_zero_bias < -255 || vec2_zero_bias > 0), -1);
  }

  /* Iterators used in for loops */
  int m_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  /*Load AE_BIASV8 and AE_BIASC8 state registers with mat1 and vec1 zero bias values*/
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec1_zero_bias, -mat1_zero_bias));
  ae_int64 biasvc2 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec2_zero_bias, -mat2_zero_bias));

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }

  const WORD8 *p_mat1_0;
  const WORD8 *p_mat2_0;
  const WORD8 *p_vec1_0;
  const WORD8 *p_vec2_0;
  ae_int8x8 out8_0, out8_1;  
  ae_int8x8 out8_2, out8_3; 

  ae_int32x2 max_uint8 = AE_MOVDA32(255);
  ae_int32x2 min_uint8 = AE_MOVDA32(0);
  
  if(p_mat2 && p_vec2 && ((((unsigned)p_out) & 15) == 0) && ((((unsigned)p_mat1) & 15) == 0) && ((((unsigned)p_mat2) & 15) == 0) &&
     ((((unsigned)p_vec1) & 15) == 0) && ((((unsigned)p_vec2) & 15) == 0) && ((((unsigned)p_bias) & 3) == 0) &&
     ((row_stride1 & 15) == 0) && ((row_stride2 & 15) == 0))
  {
    for(m_itr = 0; m_itr < (rows & ~(8 - 1)); m_itr += 8)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      ae_int32x2 acc_row2_vec0 = ZERO32;
      ae_int32x2 acc_row3_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4 *)p_bias, 16);
        AE_L32X2X2_IP(acc_row2_vec0, acc_row3_vec0, (ae_int32x4 *)p_bias, 16);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr*row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      p_mat2_0 = (const WORD8 *)(p_mat2+(m_itr*row_stride2));
      p_vec2_0 = (const WORD8 *)(p_vec2);

      AE_MOVZBVCDR(biasvc1);
      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,-vec1_zero_bias
        );

      AE_MOVZBVCDR(biasvc2);
      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,row_stride2
         ,-vec2_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row2_vec0, acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row3_vec0, acc_row3_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
      acc_row2_vec0 = AE_ADD32S(acc_row2_vec0, out_zero_bias);
      acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
      
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      AE_MINMAX32(acc_row1_vec0, min_uint8, max_uint8);
      AE_MINMAX32(acc_row2_vec0, min_uint8, max_uint8);
      AE_MINMAX32(acc_row3_vec0, min_uint8, max_uint8);

      out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
      out8_1 = AE_MOVINT8X8_FROMINT32X2(acc_row1_vec0);
      out8_2 = AE_MOVINT8X8_FROMINT32X2(acc_row2_vec0);
      out8_3 = AE_MOVINT8X8_FROMINT32X2(acc_row3_vec0);
      AE_SW_S8_4_IP(out8_0, (ae_int8 *) p_out, 1);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_out, 1);
      AE_SW_S8_4_IP(out8_1, (ae_int8 *) p_out, 1);
      AE_S8_0_IP(out8_1, (ae_int8 *) p_out, 1);
      AE_SW_S8_4_IP(out8_2, (ae_int8 *) p_out, 1);
      AE_S8_0_IP(out8_2, (ae_int8 *) p_out, 1);
      AE_SW_S8_4_IP(out8_3, (ae_int8 *) p_out, 1);
      AE_S8_0_IP(out8_3, (ae_int8 *) p_out, 1);
    }

    /* Compute last (rows % 8) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      p_mat1_0 = (const WORD8 *)(p_mat1 + (m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      p_mat2_0 = (const WORD8 *)(p_mat2 + (m_itr * row_stride2));
      p_vec2_0 = (const WORD8 *)(p_vec2);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L32_IP(acc_row0_vec0, (ae_int32 *)p_bias, 4);
      }

      AE_MOVZBVCDR(biasvc1);
      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,-vec1_zero_bias
        );

      AE_MOVZBVCDR(biasvc2);
      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,-vec2_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      *p_out++ = (UWORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }  
  else if (p_mat2 && p_vec2)
  {
    int row_count = rows - rows % 3;
    for(m_itr = 0; m_itr < row_count; m_itr += 3)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32(p_bias[m_itr + 2]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      p_mat2_0 = (const WORD8 *)(p_mat2+(m_itr * row_stride2));
      p_vec2_0 = (const WORD8 *)(p_vec2);

      AE_MOVZBVCDR(biasvc1);
      _xa_nn_dot_product_3_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,-vec1_zero_bias
        );

      AE_MOVZBVCDR(biasvc2);
      _xa_nn_dot_product_3_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,row_stride2
         ,-vec2_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      AE_MINMAX32(acc_row1_vec0, min_uint8, max_uint8);
      
      out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
      out8_1 = AE_MOVINT8X8_FROMINT32X2(acc_row1_vec0);
      AE_SW_S8_4_IP(out8_0, (ae_int8 *) p_out, 1);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_out, 1);
      AE_SW_S8_4_IP(out8_1, (ae_int8 *) p_out, 1);
    }

    /* Compute last (rows % 3) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      p_mat1_0 = (const WORD8 *)(p_mat1 + (m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      p_mat2_0 = (const WORD8 *)(p_mat2 + (m_itr * row_stride2));
      p_vec2_0 = (const WORD8 *)(p_vec2);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      AE_MOVZBVCDR(biasvc1);
      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,-vec1_zero_bias
        );

      AE_MOVZBVCDR(biasvc2);
      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,-vec2_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      *p_out++ = (UWORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else if(((((unsigned)p_out) & 15) == 0) && ((((unsigned)p_mat1) & 15) == 0) &&
    ((((unsigned)p_vec1) & 15) == 0) && ((((unsigned)p_bias) & 3) == 0) &&
    ((row_stride1 & 15) == 0))
  {
    AE_MOVZBVCDR(biasvc1);
    for(m_itr = 0; m_itr < (rows & ~(8 - 1)); m_itr += 8)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      ae_int32x2 acc_row2_vec0 = ZERO32;
      ae_int32x2 acc_row3_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4 *)p_bias, 16);
        AE_L32X2X2_IP(acc_row2_vec0, acc_row3_vec0, (ae_int32x4 *)p_bias, 16);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,-vec1_zero_bias
        );

      ae_int16x4 out_0, out_1;
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, acc_row2_vec0, acc_row3_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);

      AE_MINMAX16(out_0, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out_1, AE_ZERO16(), AE_MOVDA16(255));

      ae_int8x8 temp_vec0;
      temp_vec0 = AE_SATU8X8X16(out_0, out_1);

      AE_S8X8_IP(temp_vec0, (ae_int8x8 *) p_out, 8);
    }

    /* Compute last (rows % 8) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L32_IP(acc_row0_vec0, (ae_int32 *)p_bias, 4);
      }

      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,-vec1_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      *p_out++ = (UWORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else if (p_mat1 && p_vec1)
  {
    AE_MOVZBVCDR(biasvc1);
    ae_int8x8 out8_0, out8_1;  
    int row_count = rows - rows % 3;
    for(m_itr = 0; m_itr < row_count; m_itr += 3)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32(p_bias[m_itr + 2]);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr*row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_3_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,-vec1_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      AE_MINMAX32(acc_row1_vec0, min_uint8, max_uint8);
      
      out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
      out8_1 = AE_MOVINT8X8_FROMINT32X2(acc_row1_vec0);
      AE_SW_S8_4_IP(out8_0, (ae_int8 *) p_out, 1);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_out, 1);
      AE_SW_S8_4_IP(out8_1, (ae_int8 *) p_out, 1);
    }

    /* Compute last (rows%3) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,-vec1_zero_bias
        );

      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_uint8, max_uint8);
      *p_out++ = (UWORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else
  {
    return -1;
  }

  return 0;
}
