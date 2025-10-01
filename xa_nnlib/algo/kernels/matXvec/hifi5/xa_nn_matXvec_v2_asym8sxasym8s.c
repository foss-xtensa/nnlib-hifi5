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
// Defining macros for HiFi5 RI.6 and HiFi5 RI.5(< RI.6) compatibility
#ifndef AE_MULAZB8Q8X8 // HiFI5 RI.5 

  #define KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(vec_idx) \
  { \
    ae_int16x4 wvec0_ ## vec_idx, wvec1_ ## vec_idx; \
    ae_int8x8 d_mat1_zb = AE_MOVDA8(-mat_zero_bias); \
    AE_SUBW8(wvec0_ ## vec_idx, wvec1_ ## vec_idx, vec0_batch_ ## vec_idx, neg_vec_bias); \
    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_ ## vec_idx, mat1_row1_ ## vec_idx, mat1_row2_ ## vec_idx, mat1_row3_ ## vec_idx, wvec0_ ## vec_idx, wvec1_ ## vec_idx); \
    AE_MULA8Q8X16(acc_row2_vec0, acc_row3_vec0, mat1_row4_ ## vec_idx, mat1_row5_ ## vec_idx, mat1_row6_ ## vec_idx, mat1_row7_ ## vec_idx, wvec0_ ## vec_idx, wvec1_ ## vec_idx); \
    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, d_mat1_zb, d_mat1_zb, d_mat1_zb, d_mat1_zb, AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec0_ ## vec_idx))), AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec1_ ## vec_idx)))); \
    AE_MULA8Q8X16(acc_row2_vec0, acc_row3_vec0, d_mat1_zb, d_mat1_zb, d_mat1_zb, d_mat1_zb, AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec0_ ## vec_idx))), AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec1_ ## vec_idx)))); \
  }
  
  #define KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(vec_idx) \
  { \
    ae_int16x4 wvec0_ ## vec_idx, wvec1_ ## vec_idx; \
    ae_int8x8 d_mat1_zb = AE_MOVDA8(-mat_zero_bias); \
    AE_SUBW8(wvec0_ ## vec_idx, wvec1_ ## vec_idx, vec0_batch_ ## vec_idx, neg_vec_bias); \
    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_ ## vec_idx, mat1_row1_ ## vec_idx, mat1_row2_ ## vec_idx, mat1_row3_ ## vec_idx, wvec0_ ## vec_idx, wvec1_ ## vec_idx); \
    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, d_mat1_zb, d_mat1_zb, d_mat1_zb, d_mat1_zb, AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec0_ ## vec_idx))), AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec1_ ## vec_idx)))); \
  }
  
  #define KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(vec_idx) \
  { \
    ae_int16x4 wvec0_ ## vec_idx, wvec1_ ## vec_idx; \
    ae_int8x8 d_mat1_zb = AE_MOVDA8(-mat_zero_bias); \
    AE_SUBW8(wvec0_ ## vec_idx, wvec1_ ## vec_idx, vec0_batch_ ## vec_idx, neg_vec_bias); \
    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_ ## vec_idx, mat1_row0_ ## vec_idx, mat1_row0_ ## vec_idx, mat1_row0_ ## vec_idx, wvec0_ ## vec_idx, wvec1_ ## vec_idx); \
    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, d_mat1_zb, d_mat1_zb, d_mat1_zb, d_mat1_zb, AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec0_ ## vec_idx))), AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(wvec1_ ## vec_idx)))); \
  }
  
#else // HiFi5 RI.6

  #define KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(vec_idx) \
    AE_MULAZB8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_ ## vec_idx, mat1_row1_ ## vec_idx, mat1_row2_ ## vec_idx, mat1_row3_ ## vec_idx, vec0_batch_ ## vec_idx); \
    AE_MULAZB8Q8X8(acc_row2_vec0, acc_row3_vec0, mat1_row4_ ## vec_idx, mat1_row5_ ## vec_idx, mat1_row6_ ## vec_idx, mat1_row7_ ## vec_idx, vec0_batch_ ## vec_idx);
  
  #define KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(vec_idx) \
    AE_MULAZB8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_ ## vec_idx, mat1_row1_ ## vec_idx, mat1_row2_ ## vec_idx, mat1_row3_ ## vec_idx, vec0_batch_ ## vec_idx); \
  
  #define KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(vec_idx) \
    AE_MULAZB8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_ ## vec_idx, mat1_row0_ ## vec_idx, mat1_row0_ ## vec_idx, mat1_row0_ ## vec_idx, vec0_batch_ ## vec_idx);

#endif  //AE_MULAZB8Q8X8


extern const long long pre_loop_sel_pattern[16]; 
extern const long long post_loop_sel_pattern[16]; 
extern const long long g_sel_pattern[16];

static inline void _xa_nn_dot_product_8_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_2_0
    ,ae_int32x2* out_3_0
    ,ae_int8x8*  pt_mat_0
    ,ae_int8*    pt_vec_0
    ,WORD32      cols
    ,WORD32      row_stride
    ,WORD32      vec_zero_bias
    ,WORD32      mat_zero_bias
    )
{
  int rem_cols = (cols & 31);
  int rem_g16 = (rem_cols > 16)?1:0;
  int rem_ge16 = (rem_cols < 16)?0:1;
  int rem_cols_16 = rem_g16?((cols - 16) & 15):(cols & 15);
  int rem_g8 = (rem_cols_16 > 8)?1:0;
  int rem_cols8 = rem_cols_16 % 8;  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16) + 1])); 
  ae_int8x8 sel4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16) + 1])); 
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec_zero_bias);

  #ifdef AE_MULAZB8Q8X8
    ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec_zero_bias, -mat_zero_bias));
    AE_MOVZBVCDR(biasvc1);
  #endif

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 mat1_row4_0, mat1_row4_1, mat1_row4_2, mat1_row4_3;
  ae_int8x8 mat1_row5_0, mat1_row5_1, mat1_row5_2, mat1_row5_3;
  ae_int8x8 mat1_row6_0, mat1_row6_1, mat1_row6_2, mat1_row6_3;
  ae_int8x8 mat1_row7_0, mat1_row7_1, mat1_row7_2, mat1_row7_3;
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  
  ae_int8x16 *p_vec_0 = (ae_int8x16*)pt_vec_0;

  ae_int8x16 *p_mat_0 = (ae_int8x16*)pt_mat_0;
  ae_int8x16 *p_mat_1 = (ae_int8x16*)((WORD8 *)p_mat_0 + row_stride); 
  ae_int8x16 *p_mat_2 = (ae_int8x16*)((WORD8 *)p_mat_1 + row_stride); 
  ae_int8x16 *p_mat_3 = (ae_int8x16*)((WORD8 *)p_mat_2 + row_stride); 
  ae_int8x16 *p_mat1_4 = (ae_int8x16*)((WORD8 *)p_mat_3 + row_stride); 
  ae_int8x16 *p_mat1_5 = (ae_int8x16*)((WORD8 *)p_mat1_4 + row_stride); 
  ae_int8x16 *p_mat1_6 = (ae_int8x16*)((WORD8 *)p_mat1_5 + row_stride); 
  ae_int8x16 *p_mat1_7 = (ae_int8x16*)((WORD8 *)p_mat1_6 + row_stride); 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row2_vec0 = *out_2_0;
  ae_int32x2 acc_row3_vec0 = *out_3_0;

  for(c_itr = 0; c_itr < cols >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, p_mat_0, 16);
    AE_L8X8X2_I(mat1_row1_2, mat1_row1_3, p_mat_1, 16);
    AE_L8X8X2_I(mat1_row2_2, mat1_row2_3, p_mat_2, 16);
    AE_L8X8X2_I(mat1_row3_2, mat1_row3_3, p_mat_3, 16);
    AE_L8X8X2_I(mat1_row4_2, mat1_row4_3, p_mat1_4, 16);
    AE_L8X8X2_I(mat1_row5_2, mat1_row5_3, p_mat1_5, 16);
    AE_L8X8X2_I(mat1_row6_2, mat1_row6_3, p_mat1_6, 16);
    AE_L8X8X2_I(mat1_row7_2, mat1_row7_3, p_mat1_7, 16);

    AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 32);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat_1, 32);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat_2, 32);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat_3, 32);
    AE_L8X8X2_IP(mat1_row4_0, mat1_row4_1, p_mat1_4, 32);
    AE_L8X8X2_IP(mat1_row5_0, mat1_row5_1, p_mat1_5, 32);
    AE_L8X8X2_IP(mat1_row6_0, mat1_row6_1, p_mat1_6, 32);
    AE_L8X8X2_IP(mat1_row7_0, mat1_row7_1, p_mat1_7, 32);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 32);

    KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(0);
    KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(1);
    KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(2);
    KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(3);
  }
  //Remainder loop for cols
  int flag_itr = 0;
  while(rem_cols > 0)
  {
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat_3, 16);
    AE_L8X8X2_IP(mat1_row4_0, mat1_row4_1, p_mat1_4, 16);
    AE_L8X8X2_IP(mat1_row5_0, mat1_row5_1, p_mat1_5, 16);
    AE_L8X8X2_IP(mat1_row6_0, mat1_row6_1, p_mat1_6, 16);
    AE_L8X8X2_IP(mat1_row7_0, mat1_row7_1, p_mat1_7, 16);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(0);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      KERNEL_ROW8_VEC1_ASYM8S_ASYM8S(1);
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
    ,ae_int8x8*  pt_mat_0
    ,ae_int8*    pt_vec_0
    ,WORD32      cols
    ,WORD32      vec_zero_bias
    ,WORD32      mat_zero_bias
    )
{
  int rem_cols = (cols & 31);
  int rem_g16 = (rem_cols > 16)?1:0;
  int rem_ge16 = (rem_cols < 16)?0:1;
  int rem_cols_16 = rem_g16?((cols - 16) & 15):(cols & 15);
  int rem_g8 = (rem_cols_16 > 8)?1:0;
  int rem_cols8 = rem_cols_16 % 8;  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16) + 1])); 
  ae_int8x8 sel4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16) + 1])); 

  int c_itr = 0;
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec_zero_bias);

  #ifdef AE_MULAZB8Q8X8
    ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec_zero_bias, -mat_zero_bias));
    AE_MOVZBVCDR(biasvc1);
  #endif

  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  ae_int8x16 *p_mat_0 = (ae_int8x16 *)pt_mat_0;
  ae_int8x16 *p_vec_0 = (ae_int8x16 *)pt_vec_0;

#pragma no_unroll
  for(c_itr = 0; c_itr < cols >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, p_mat_0, 16);
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 32);

    AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, p_vec_0, 16);
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 32);

    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(0);
    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(1);
    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(2);
    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(3);
  }
  //Remainder loop for cols
  int flag_itr = 0;
  while(rem_cols > 0)
  {
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 16);
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(0);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(1);
      flag_itr = 1;
    }
    rem_cols -= 16;
    sel1 = sel3; 
    sel2 = sel4;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  pt_mat_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_stride1
    ,WORD32      vec_zero_bias
    ,WORD32      mat_zero_bias
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  int align_offset = ((unsigned int)pt_mat_0 & 0xf);
  pre_loop_count = 16 - align_offset;
  int pre_rem_g8 = (align_offset > 8)?1:0;
  ae_int8x8 pre_sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset % 8) * !pre_rem_g8], pre_loop_sel_pattern[2 * (align_offset % 8) * !pre_rem_g8 + 1])); 
  ae_int8x8 pre_sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset % 8) * pre_rem_g8], pre_loop_sel_pattern[2 * (align_offset % 8) * pre_rem_g8 + 1])); 
  ae_int8x16 *p_mat_0 = (ae_int8x16 *)((ae_int8 *)pt_mat_0 - align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 31):((cols + align_offset) & 31);
  loop_count >>= 5;

  int mask_start_end = ((cols + align_offset) > 16)?1:0;

  int rem_g16 = (post_loop_count > 16)?1:0;
  int rem_ge16 = (post_loop_count < 16)?0:1;
  int rem_cols_16 = rem_g16?((post_loop_count - 16) & 15):(post_loop_count & 15);
  int rem_g8 = (rem_cols_16 > 8)?1:0;
  int rem_cols8 = rem_cols_16 % 8;  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16) + 1])); 
  ae_int8x8 sel4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16) + 1])); 
  
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec_zero_bias);
  
  #ifdef AE_MULAZB8Q8X8
    ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec_zero_bias, -mat_zero_bias));
    AE_MOVZBVCDR(biasvc1);
  #endif

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 

  ae_int8x16* p_mat_1 = p_mat_0 + row_stride1; //next 16th row 
  ae_int8x16* p_mat_2 = p_mat_1 + row_stride1; //next 16th row
  ae_int8x16* p_mat_3 = p_mat_2 + row_stride1; //next 16th row 

   ae_int8x16 *pt_vec_0 =  (ae_int8x16 *)p_vec_0;
  
  ae_valignx2 align_p_vec0 = AE_LA128_PP(pt_vec_0);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  /* Pre loop computation */
  AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 16);
  AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat_1, 16);
  AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat_2, 16);
  AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat_3, 16);

  AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, pt_vec_0);
  if(align_offset)
  {
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, pre_sel1);
  }
  if(pre_rem_g8)
  {
    vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, pre_sel2);
  }
  if(mask_start_end)
  {
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(0); 
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(1); 
  }

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, p_mat_0, 16);
    AE_L8X8X2_I(mat1_row1_2, mat1_row1_3, p_mat_1, 16);
    AE_L8X8X2_I(mat1_row2_2, mat1_row2_3, p_mat_2, 16);
    AE_L8X8X2_I(mat1_row3_2, mat1_row3_3, p_mat_3, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 32);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat_1, 32);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat_2, 32);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat_3, 32);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, pt_vec_0);
    AE_LA8X8X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec0, pt_vec_0);
    
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(0);
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(1);
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(2);
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(3);
  }

  //Remainder loop for cols
  int flag_itr = 0;
  while(post_loop_count > 0)
  {
    if(mask_start_end)
    {
      AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, p_mat_0, 16);
      AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, p_mat_1, 16);
      AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, p_mat_2, 16);
      AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, p_mat_3, 16);

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, pt_vec_0);
    } 
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(0); 
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(1); 
      flag_itr = 1;
    }
    post_loop_count -= 16;
    sel1 = sel3; 
    sel2 = sel4;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat_0
    ,ae_int8x8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_zero_bias
    ,WORD32      mat_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec_zero_bias);
  #ifdef AE_MULAZB8Q8X8
    ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec_zero_bias, -mat_zero_bias));
    AE_MOVZBVCDR(biasvc1);
  #endif
  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols & 7)], post_loop_sel_pattern[2 * (cols & 7) + 1]));
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 

  ae_int8x8 *p_mat_1 = (ae_int8x8*)((WORD8 *)p_mat_0 + row_offset); 
  ae_int8x8 *p_mat_2 = (ae_int8x8*)((WORD8 *)p_mat_1 + row_offset); 
  ae_int8x8 *p_mat_3 = (ae_int8x8*)((WORD8 *)p_mat_2 + row_offset); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat_3);
    
    AE_L8X8_IP(vec0_batch_0, p_vec_0, 8);
    
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat_3);

    AE_L8X8_IP(vec0_batch_0, p_vec_0, 8);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    KERNEL_ROW4_VEC1_ASYM8S_ASYM8S(0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  pt_mat_0
    ,ae_int8x8*    pt_vec_0
    ,WORD32      cols
    ,WORD32      vec_zero_bias
    ,WORD32      mat_zero_bias
    )
{
  int c_itr = 0;
  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 mat1_row0_0, mat1_row0_1;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  ae_int8x16 *p_vec_0 =  (ae_int8x16 *)pt_vec_0;
  ae_int8x16 *p_mat_0 = (ae_int8x16 *)pt_mat_0;
  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat_0);

  int rem_cols = (cols & 15);
  int rem_g8 = (rem_cols > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); \
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1])); \
  
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec_zero_bias);
  #ifdef AE_MULAZB8Q8X8
    ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-vec_zero_bias, -mat_zero_bias));
    AE_MOVZBVCDR(biasvc1);
  #endif
  
  int cols_count = cols - (cols & 15);
  for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat_0);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);

    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(0);
    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, p_mat_0);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(0);
    
    if(rem_g8)
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      KERNEL_ROW1_VEC1_ASYM8S_ASYM8S(1);
    }
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

WORD32 xa_nn_matXvec_v2_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat,
    const WORD8 * __restrict__ p_vec,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_stride,
    WORD32 mat_zero_bias,
    WORD32 vec_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)    
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, 2*sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat, 2*sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, 2*sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, 2*sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride < cols), -1);
  XA_NNLIB_ARG_CHK_COND((vec_zero_bias < -127 || vec_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((mat_zero_bias < -127 || mat_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  /* Iterators used in for loops */
  int m_itr, ii;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
#if XCHAL_HAVE_HIFI5S
  int left_shift_out32 = 31 - left_shift; 
  left_shift_out32 = (left_shift_out32 << 16) | left_shift_out32;
#endif // XCHAL_HAVE_HIFI5S
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }

  const WORD8 *p_mat_0;
  const WORD8 *p_vec_0;

  ae_int32x2 max_int8 = SW_MOVDA32(out_activation_max);
  ae_int32x2 min_int8 = SW_MOVDA32(out_activation_min);
  
  if((row_stride & 15) == 0)
  {
    ae_int32x4 *pt32x4_bias = (ae_int32x4 *)p_bias;
    ae_int8x8 * pt_out = (ae_int8x8 *) p_out;
    for(m_itr = 0; m_itr < (rows & ~(8 - 1)); m_itr += 8)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      ae_int32x2 acc_row2_vec0 = ZERO32;
      ae_int32x2 acc_row3_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, pt32x4_bias, 16);
        AE_L32X2X2_IP(acc_row2_vec0, acc_row3_vec0, pt32x4_bias, 16);
      }

      p_mat_0 = (const WORD8 *)(p_mat+(m_itr * row_stride));
      p_vec_0 = (const WORD8 *)(p_vec);

      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,(ae_int8x8*)p_mat_0
         ,(ae_int8*)p_vec_0
         ,cols
         ,row_stride
         ,vec_zero_bias
         ,mat_zero_bias
        );

      ae_int16x4 out0, out1;
      
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out1, acc_row2_vec0, acc_row3_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);

      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

#ifndef AE_S8X4UX2_IP
      ae_int8x8 out8_0; 
      out8_0 = AE_SAT8X8X16(out0, out1);
      AE_S8X8_IP(out8_0, pt_out, 8);
#else
      AE_S8X4UX2_IP(out0, out1, pt_out, 8);
#endif
    }
    ae_int32 *pt32_bias = (ae_int32 *)pt32x4_bias;
    p_out = (WORD8 *)pt_out;
    /* Compute last (rows % 8) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      p_mat_0 = (const WORD8 *)(p_mat+(m_itr * row_stride));
      p_vec_0 = (const WORD8 *)(p_vec);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_L32_IP(acc_row0_vec0, pt32_bias, 4);
      }

      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat_0
         ,(ae_int8*)p_vec_0
         ,cols
         ,vec_zero_bias
         ,mat_zero_bias
        );
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift_out32, right_shift);
#else
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
#endif      
      acc_row0_vec0 = SW_ADD32S_INT32X2_INT32X2(acc_row0_vec0, SW_MOVDA32(out_zero_bias));
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }
  else
  {
    ae_int8x8 out8_0, out8_1;  
    for(m_itr = 0; m_itr < (rows & ~(64 - 1)); m_itr += 64)
    {
      ae_int8* p_dst_0 = (ae_int8*)(p_out + m_itr);
      for(ii = 0; ii < 16; ii++)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row1_vec0 = ZERO32;

        if(bias_flag)
        {
          /* Load bias in the accumulator */
          acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + ii +  0], p_bias[m_itr + ii + 16]);
          acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + ii + 32], p_bias[m_itr + ii + 48]);
        }

        p_mat_0 = (WORD8 *)(p_mat+((m_itr + ii) * row_stride));
        p_vec_0 = (WORD8 *)(p_vec);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat_0
           ,(ae_int8*)p_vec_0
           ,cols
           ,row_stride
           ,vec_zero_bias
           ,mat_zero_bias
          );
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift_out32, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift_out32, right_shift);
#else       
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32(acc_row1_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift);
#endif        
        acc_row0_vec0 = SW_ADD32S_INT32X2_INT32X2(acc_row0_vec0, SW_MOVDA32(out_zero_bias));
        acc_row1_vec0 = SW_ADD32S_INT32X2_INT32X2(acc_row1_vec0, SW_MOVDA32(out_zero_bias));
        AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32(acc_row1_vec0, min_int8, max_int8);
        
        out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
        out8_1 = AE_MOVINT8X8_FROMINT32X2(acc_row1_vec0);
        AE_S8_0_X(out8_0, p_dst_0, 16);
        AE_SW_S8_4_X(out8_1, p_dst_0, 32);
        AE_S8_0_X(out8_1,  p_dst_0, 48);
        AE_SW_S8_4_IP(out8_0,  p_dst_0, 1);
      }
    }

    ae_int8* p_dst_0 = (ae_int8*)(p_out + m_itr);
    for(; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

        p_mat_0 = (WORD8 *)(p_mat + (m_itr * row_stride));
        p_vec_0 = (WORD8 *)(p_vec);

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat_0
           ,(ae_int8x8*)p_vec_0
           ,cols
           ,row_stride
           ,vec_zero_bias
           ,mat_zero_bias
          );
      
        ae_int16x4 out_0;
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, left_shift, right_shift, out_zero_bias);

        AE_MINMAX16(out_0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
 
        out8_0 = AE_MOVINT8X8_FROMINT16X4(out_0);
        AE_SW_S8_6_XP(out8_0, p_dst_0, 1);
        AE_SW_S8_4_XP(out8_0, p_dst_0, 1);
        AE_SW_S8_2_XP(out8_0, p_dst_0, 1);
        AE_S8_0_XP(out8_0, p_dst_0, 1);
    }

    p_dst_0 = (ae_int8*)(p_out + m_itr);
    /* Compute last (rows % 4) output element */
    for (; m_itr < rows; m_itr++)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;
      
      p_mat_0 = (const WORD8 *)(p_mat+(m_itr * row_stride));
      p_vec_0 = (const WORD8 *)(p_vec);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = SW_MOVDA32(p_bias[m_itr]);
      }

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat_0
         ,(ae_int8x8*)p_vec_0
         ,cols
         ,vec_zero_bias
         ,mat_zero_bias
        );
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
      MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift_out32, right_shift);
#else
      MPY_BY_QUANT_MULT_X2_OUT32(acc_row0_vec0, acc_row0_vec0, out_multiplier, left_shift, right_shift);
#endif      
      acc_row0_vec0 = SW_ADD32S_INT32X2_INT32X2(acc_row0_vec0, SW_MOVDA32(out_zero_bias));
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
      AE_S8_0_IP(out8_0, p_dst_0, 1);
    }
  }
  return 0;
}
