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
#include "xa_nnlib_common_macros_hifi5.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32(inp, left_shift); \
    inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out, inp1, inp2, multiplier, l_shift, r_shift, out_off) \
    AE_MUL2P32X4S(inp1, inp2, inp1, inp2, l_shift, l_shift); \
    AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    inp1 = AE_SRAA32SYMS(inp1, r_shift); \
    inp2 = AE_SRAA32SYMS(inp2, r_shift); \
    out = AE_SAT16X4(inp1, inp2); \
    out = AE_ADD16S(AE_MOVDA16(out_off), out); \
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127));

#define PACK_32X2(dst1, src1, src2) \
    dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

extern const long long pre_loop_sel_pattern[16]; 
extern const long long post_loop_sel_pattern[16]; 
extern const long long g_sel_pattern[16];

static inline void special_function_for_cols_mul_32_unaligned
    (WORD8*        p_out_0
    ,const WORD8*  p_mat1
    ,const WORD8*  p_vec1
    ,const WORD32* p_bias
    ,WORD32        rows
    ,WORD32        cols
    ,WORD32        out_multiplier
    ,WORD32        left_shift
    ,WORD32        right_shift
    ,WORD32        out_zero_bias
    ,WORD32        row_stride1
    ,WORD32        bias_flag
    ,WORD32        vec1_zero_bias
    )
{
  int m_itr = 0;
  int c_itr = 0;

  ae_int32x2 l_mult = AE_MOVDA32(1 << left_shift);
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int8x8 align_p_vec0;
  ae_int16x4 wvec0_0_0, wvec0_0_1;
  ae_int16x4 wvec0_1_0, wvec0_1_1; 
  ae_int16x4 wvec0_2_0, wvec0_2_1;
  ae_int16x4 wvec0_3_0, wvec0_3_1; 

  for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
  {
    ae_int32x2 acc_row0_vec0 = ZERO32;
    ae_int32x2 acc_row1_vec0 = ZERO32;

    if(bias_flag)
    {
      /* Load bias in the accumulator */
      acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
      acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
    }

    const WORD8 *p_mat1_0 = (WORD8 *)(p_mat1 + (m_itr * row_stride1));
    const WORD8 *p_vec1_0 = (WORD8 *)(p_vec1);

    ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
    ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
    ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1); 

    ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
    ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
    ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
    ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

    AE_SW_PRIME_64(p_vec1_0, align_p_vec0);

    for(c_itr = 0; c_itr < cols >> 5; c_itr++)
    {
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
      AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
      AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);
      AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);
    
      AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec1_0);
      AE_SW_LA8X8_IP(vec0_batch_1, align_p_vec0, p_vec1_0);
      AE_SW_LA8X8_IP(vec0_batch_2, align_p_vec0, p_vec1_0);
      AE_SW_LA8X8_IP(vec0_batch_3, align_p_vec0, p_vec1_0);
    
      AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias);
      AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias);
      AE_SUBW8(wvec0_2_0, wvec0_2_1, vec0_batch_2, neg_vec_bias);
      AE_SUBW8(wvec0_3_0, wvec0_3_1, vec0_batch_3, neg_vec_bias);

      AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, wvec0_0_0, wvec0_0_1);
      AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, wvec0_1_0, wvec0_1_1);
      AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_2, mat1_row1_2, mat1_row2_2, mat1_row3_2, wvec0_2_0, wvec0_2_1);
      AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_3, mat1_row1_3, mat1_row2_3, mat1_row3_3, wvec0_3_0, wvec0_3_1);
    }

    ae_int16x4 out_0;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);

    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, out_0);
    AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out_0, 4);
  }
}

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

  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);
  
  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 mat1_row4_0, mat1_row4_1, mat1_row4_2, mat1_row4_3;
  ae_int8x8 mat1_row5_0, mat1_row5_1, mat1_row5_2, mat1_row5_3;
  ae_int8x8 mat1_row6_0, mat1_row6_1, mat1_row6_2, mat1_row6_3;
  ae_int8x8 mat1_row7_0, mat1_row7_1, mat1_row7_2, mat1_row7_3;
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  
  ae_int16x4 wvec0_0_0, wvec0_1_0, wvec0_2_0, wvec0_3_0; 
  ae_int16x4 wvec0_0_1, wvec0_1_1, wvec0_2_1, wvec0_3_1; 

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

    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
    AE_SUBW8(wvec0_2_0, wvec0_2_1, vec0_batch_2, neg_vec_bias); 
    AE_SUBW8(wvec0_3_0, wvec0_3_1, vec0_batch_3, neg_vec_bias); 

    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_0 , mat1_row5_0 , mat1_row6_0 , mat1_row7_0 ,wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,wvec0_1_0, wvec0_1_1);
    AE_MULA8Q8X16(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_1 , mat1_row5_1 , mat1_row6_1 , mat1_row7_1 ,wvec0_1_0, wvec0_1_1);
    
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,wvec0_2_0, wvec0_2_1);
    AE_MULA8Q8X16(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_2 , mat1_row5_2 , mat1_row6_2 , mat1_row7_2 ,wvec0_2_0, wvec0_2_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,wvec0_3_0, wvec0_3_1);
    AE_MULA8Q8X16(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_3 , mat1_row5_3 , mat1_row6_3 , mat1_row7_3 ,wvec0_3_0, wvec0_3_1);
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
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_0 , mat1_row5_0 , mat1_row6_0 , mat1_row7_0 ,wvec0_0_0, wvec0_0_1);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
      AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,wvec0_1_0, wvec0_1_1);
      AE_MULA8Q8X16(acc_row2_vec0 , acc_row3_vec0 , mat1_row4_1 , mat1_row5_1 , mat1_row6_1 , mat1_row7_1 ,wvec0_1_0, wvec0_1_1);
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
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);
  
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;

  ae_int16x4 wvec0_0_0, wvec0_1_0, wvec0_2_0, wvec0_3_0; 
  ae_int16x4 wvec0_0_1, wvec0_1_1, wvec0_2_1, wvec0_3_1; 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

#pragma no_unroll
  for(c_itr = 0; c_itr < cols1 >> 5; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 32);

    AE_L8X8X2_I(vec0_batch_2, vec0_batch_3, (ae_int8x16 *)p_vec_0, 16);
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 32);

    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
    AE_SUBW8(wvec0_2_0, wvec0_2_1, vec0_batch_2, neg_vec_bias); 
    AE_SUBW8(wvec0_3_0, wvec0_3_1, vec0_batch_3, neg_vec_bias); 

    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, wvec0_1_0, wvec0_1_1);
    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_2, mat1_row0_2, mat1_row0_2, mat1_row0_2, wvec0_2_0, wvec0_2_1);
    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_3, mat1_row0_3, mat1_row0_3, mat1_row0_3, wvec0_3_0, wvec0_3_1);
  }
  //Remainder loop for cols1
  int flag_itr = 0;
  while(rem_cols > 0)
  {
    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, wvec0_0_0, wvec0_0_1);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
      AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, wvec0_1_0, wvec0_1_1);
      flag_itr = 1;
    }
    rem_cols -= 16;
    sel1 = sel3; 
    sel2 = sel4;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec1_zero_bias
    )
{
  int pre_loop_count, loop_count, post_loop_count;
  int c_itr;

  int align_offset = ((unsigned int)p_mat1_0 & 0xf);
  pre_loop_count = 16 - align_offset;
  int pre_rem_g8 = (align_offset > 8)?1:0;
  ae_int8x8 pre_sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset % 8) * !pre_rem_g8], pre_loop_sel_pattern[2 * (align_offset % 8) * !pre_rem_g8 + 1])); 
  ae_int8x8 pre_sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(pre_loop_sel_pattern[2 * (align_offset % 8) * pre_rem_g8], pre_loop_sel_pattern[2 * (align_offset % 8) * pre_rem_g8 + 1])); 
  p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  loop_count = (cols1 < pre_loop_count)?0:(cols1 - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 31):((cols1 + align_offset) & 31);
  loop_count >>= 5;

  int mask_start_end = ((cols1 + align_offset) > 16)?1:0;

  int rem_g16 = (post_loop_count > 16)?1:0;
  int rem_ge16 = (post_loop_count < 16)?0:1;
  int rem_cols_16 = rem_g16?((post_loop_count - 16) & 15):(post_loop_count & 15);
  int rem_g8 = (rem_cols_16 > 8)?1:0;
  int rem_cols8 = rem_cols_16 % 8;  
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && !rem_g16) + 1])); 
  ae_int8x8 sel3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (!rem_g8 && rem_g16) + 1])); 
  ae_int8x8 sel4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16)], post_loop_sel_pattern[2 * (rem_cols8) * (rem_g8 && rem_g16) + 1])); 
  
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int16x4 wvec0_0_0, wvec0_1_0; 
  ae_int16x4 wvec0_0_1, wvec0_1_1; 
  ae_int16x4 wvec0_2_0, wvec0_3_0; 
  ae_int16x4 wvec0_2_1, wvec0_3_1; 

  ae_int8x8* p_mat1_1 = p_mat1_0 + 2*row_stride1; //next 16th row 
  ae_int8x8* p_mat1_2 = p_mat1_1 + 2*row_stride1; //next 16th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + 2*row_stride1; //next 16th row 
  
  ae_valignx2 align_p_vec0 = AE_LA128_PP(p_vec_0);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  /* Pre loop computation */
  AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
  AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
  AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
  AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

  AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
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
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,wvec0_1_0, wvec0_1_1);
  }

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_L8X8X2_I(mat1_row0_2, mat1_row0_3, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_I(mat1_row1_2, mat1_row1_3, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_I(mat1_row2_2, mat1_row2_3, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_I(mat1_row3_2, mat1_row3_3, (ae_int8x16 *)p_mat1_3, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 32);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 32);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 32);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 32);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
    AE_LA8X8X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec0, (ae_int8x16 *)p_vec_0);
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
    AE_SUBW8(wvec0_2_0, wvec0_2_1, vec0_batch_2, neg_vec_bias); 
    AE_SUBW8(wvec0_3_0, wvec0_3_1, vec0_batch_3, neg_vec_bias); 

    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,wvec0_1_0, wvec0_1_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,wvec0_2_0, wvec0_2_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,wvec0_3_0, wvec0_3_1);
  }

  //Remainder loop for cols1
  int flag_itr = 0;
  while(post_loop_count > 0)
  {
    if(mask_start_end)
    {
      AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
      AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
      AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
      AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
    } 
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,wvec0_0_0, wvec0_0_1);
    
    if(((rem_g8 || rem_ge16) && !flag_itr) || (rem_g8 && flag_itr))
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
      AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,wvec0_1_0, wvec0_1_1);
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
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (cols & 7)], post_loop_sel_pattern[2 * (cols & 7) + 1]));
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;
  ae_int16x4 wvec0_0_0, wvec0_0_1;

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
    
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias);

    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, wvec0_0_0, wvec0_0_1);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias);

    AE_MULA8Q8X16(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, wvec0_0_0, wvec0_0_1);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_16x_cols_mat_unalined_vec_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    ,WORD32      row_stride1
    ,WORD32      vec1_zero_bias
    )
{
  int c_itr = 0;

  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;
  ae_int8x8 vec0_batch_0, vec0_batch_1;

  ae_int16x4 wvec0_0_0, wvec0_1_0;
  ae_int16x4 wvec0_0_1, wvec0_1_1;

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1);
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1);
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1);

  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
  ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
  ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

#pragma no_unroll
  for(c_itr = 0; c_itr < cols1 >> 4; c_itr++)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
    AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
    AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
    AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias);
    AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias);

    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,wvec0_1_0, wvec0_1_1);
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
  ae_int16x4 wvec0_0_0, wvec0_1_0; 
  ae_int16x4 wvec0_0_1, wvec0_1_1; 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
  ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);

  int rem_cols = (cols1 & 15);
  int rem_g8 = (rem_cols > 8)?1:0;
  ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * !rem_g8 + 1])); \
  ae_int8x8 sel2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8], post_loop_sel_pattern[2 * (rem_cols % 8) * rem_g8 + 1])); \
  ae_int8x8 neg_vec_bias = AE_MOVDA8((WORD8)-vec1_zero_bias);
  int cols_count = cols1 - (cols1 & 15);
  for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int8x16 *)p_vec_0);

    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 
    AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
    
    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, wvec0_0_0, wvec0_0_1);
    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, wvec0_1_0, wvec0_1_1);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

    AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec_0, (ae_int8x16 *)p_vec_0);

    vec0_batch_0 = AE_SEL8X8(vec0_batch_0, neg_vec_bias, sel1);
    AE_SUBW8(wvec0_0_0, wvec0_0_1, vec0_batch_0, neg_vec_bias); 

    AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, wvec0_0_0, wvec0_0_1);
    
    if(rem_g8)
    {
      vec0_batch_1 = AE_SEL8X8(vec0_batch_1, neg_vec_bias, sel2);
      AE_SUBW8(wvec0_1_0, wvec0_1_1, vec0_batch_1, neg_vec_bias); 
      AE_MULA8Q8X16(acc_row0_vec0, acc_row0_vec1, mat1_row0_1, mat1_row0_1, mat1_row0_1, mat1_row0_1, wvec0_1_0, wvec0_1_1);
    }
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matXvec_sym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    const WORD8 * __restrict__ p_vec1,
    const WORD8 * __restrict__ p_vec2,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 cols2,
    WORD32 row_stride1,
    WORD32 row_stride2,
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
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
    XA_NNLIB_ARG_CHK_COND((vec2_zero_bias < -127 || vec2_zero_bias > 128), -1);
  }

  /* Iterators used in for loops */
  int m_itr, ii;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;

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

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);
  
  if(p_mat2 && p_vec2 && ((((unsigned)p_out) & 15) == 0) && ((((unsigned)p_mat1) & 15) == 0) && ((((unsigned)p_mat2) & 15) == 0) &&
     ((((unsigned)p_vec1) & 15) == 0) && ((((unsigned)p_vec2) & 15) == 0) && ((((unsigned)p_bias) & 15) == 0) &&
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

      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );

      _xa_nn_dot_product_8_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,&acc_row2_vec0
         ,&acc_row3_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,row_stride2
         ,vec2_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row3_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
      acc_row2_vec0 = AE_ADD32S(acc_row2_vec0, out_zero_bias);
      acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
      
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      AE_MINMAX32(acc_row1_vec0, min_int8, max_int8);
      AE_MINMAX32(acc_row2_vec0, min_int8, max_int8);
      AE_MINMAX32(acc_row3_vec0, min_int8, max_int8);

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

      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      _xa_nn_dot_product_1_rows_1_vecs_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,vec2_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }  
  else if (p_mat2 && p_vec2)
  {
    for(m_itr = 0; m_itr < (rows & ~(64 - 1)); m_itr += 64)
    {
      WORD8* p_dst_0 = (WORD8*)(p_out + m_itr);
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

        p_mat1_0 = (const WORD8 *)(p_mat1+((m_itr + ii) * row_stride1));
        p_vec1_0 = (const WORD8 *)(p_vec1);

        p_mat2_0 = (const WORD8 *)(p_mat2+((m_itr + ii) * row_stride2));
        p_vec2_0 = (const WORD8 *)(p_vec2);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec1_0
           ,cols1
           ,row_stride1
           ,vec1_zero_bias
          );

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat2_0
           ,(ae_int8*)p_vec2_0
           ,cols2
           ,row_stride2
           ,vec2_zero_bias
          );

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32(acc_row1_vec0, min_int8, max_int8);
        
        out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
        out8_1 = AE_MOVINT8X8_FROMINT32X2(acc_row1_vec0);
        AE_S8_0_X(out8_0, (ae_int8 *) p_dst_0, 16);
        AE_SW_S8_4_X(out8_1, (ae_int8 *) p_dst_0, 32);
        AE_S8_0_X(out8_1, (ae_int8 *) p_dst_0, 48);
        AE_SW_S8_4_IP(out8_0, (ae_int8 *) p_dst_0, 1);
      }
    }

    WORD8* p_dst_0 = (WORD8*)(p_out + m_itr);
    /* Compute last (rows % 64) output element */
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

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,vec1_zero_bias
        );

      _xa_nn_dot_product_1_rows_1_vecs_unaligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat2_0
         ,(ae_int8*)p_vec2_0
         ,cols2
         ,vec2_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_dst_0, 1);
    }
  }

  else if(((((unsigned)p_out) & 15) == 0) && ((((unsigned)p_mat1) & 15) == 0) &&
    ((((unsigned)p_vec1) & 15) == 0) && ((((unsigned)p_bias) & 15) == 0) &&
    ((row_stride1 & 15) == 0))
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
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row3_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
      acc_row2_vec0 = AE_ADD32S(acc_row2_vec0, out_zero_bias);
      acc_row3_vec0 = AE_ADD32S(acc_row3_vec0, out_zero_bias);
      
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      AE_MINMAX32(acc_row1_vec0, min_int8, max_int8);
      AE_MINMAX32(acc_row2_vec0, min_int8, max_int8);
      AE_MINMAX32(acc_row3_vec0, min_int8, max_int8);

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
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      *p_out++ = (WORD8)AE_MOVAD32_L(acc_row0_vec0);
    }
  }

  else if((cols1 == 64) && (row_stride1 == 64) && ((rows & 0x3) == 0) && p_mat1 && p_vec1 && ((((unsigned)p_out) & 3) == 0))
  {
    special_function_for_cols_mul_32_unaligned
      (p_out,
       p_mat1,
       p_vec1,
       p_bias,
       rows,
       cols1,
       out_multiplier,
       left_shift,
       right_shift,
       out_zero_bias,
       row_stride1,
       bias_flag,
       vec1_zero_bias
      ); 
  }

  else if (p_mat1 && p_vec1)
  {
    ae_int8x8 out8_0, out8_1;  
    for(m_itr = 0; m_itr < (rows & ~(64 - 1)); m_itr += 64)
    {
      WORD8* p_dst_0 = (WORD8*)(p_out + m_itr);
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

        p_mat1_0 = (WORD8 *)(p_mat1+((m_itr + ii) * row_stride1));
        p_vec1_0 = (WORD8 *)(p_vec1);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec1_0
           ,cols1
           ,row_stride1
           ,vec1_zero_bias
          );
       
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row1_vec0 = AE_ADD32S(acc_row1_vec0, out_zero_bias);
        AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
        AE_MINMAX32(acc_row1_vec0, min_int8, max_int8);
        
        out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
        out8_1 = AE_MOVINT8X8_FROMINT32X2(acc_row1_vec0);
        AE_S8_0_X(out8_0, (ae_int8 *) p_dst_0, 16);
        AE_SW_S8_4_X(out8_1, (ae_int8 *) p_dst_0, 32);
        AE_S8_0_X(out8_1, (ae_int8 *) p_dst_0, 48);
        AE_SW_S8_4_IP(out8_0, (ae_int8 *) p_dst_0, 1);
      }
    }

    WORD8* p_dst_0 = (WORD8*)(p_out + m_itr);
    ae_int32x2 l_mult = AE_MOVDA32(1 << left_shift);
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

        p_mat1_0 = (WORD8 *)(p_mat1 + (m_itr * row_stride1));
        p_vec1_0 = (WORD8 *)(p_vec1);

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec1_0
           ,cols1
           ,row_stride1
           ,vec1_zero_bias
          );
      
        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier, l_mult, right_shift, out_zero_bias);
 
        out8_0 = AE_MOVINT8X8_FROMINT16X4(out_0);
        AE_SW_S8_6_XP(out8_0, (ae_int8 *) p_dst_0, 1);
        AE_SW_S8_4_XP(out8_0, (ae_int8 *) p_dst_0, 1);
        AE_SW_S8_2_XP(out8_0, (ae_int8 *) p_dst_0, 1);
        AE_S8_0_XP(out8_0, (ae_int8 *) p_dst_0, 1);
    }

    p_dst_0 = (WORD8*)(p_out + m_itr);
    /* Compute last (rows % 4) output element */
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
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
      AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);
      out8_0 = AE_MOVINT8X8_FROMINT32X2(acc_row0_vec0);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_dst_0, 1);
    }
  }
  else
  {
    return -1;
  }

  return 0;
}

WORD32 xa_nn_matXvec_out_stride_sym8sxasym8s_16(
    WORD16 * __restrict__ p_out,
    const WORD8  * __restrict__ p_mat1,
    const WORD8  * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 out_stride,
    WORD32 vec1_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift)
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
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);

  /* Iterators used in for loops */
  int m_itr, ii;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;
  int out_stride_by_2;

  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
  out_stride_by_2 = (out_stride<<1);

  int bias_flag = 0;
  if(p_bias != NULL)
  {
    bias_flag = 1;
  }

  const WORD8 *p_mat1_0;
  const WORD8 *p_vec1_0;
  ae_int16x4 out16_0, out16_1;
  ae_int16x4 out16_2, out16_3;

  ae_int32x2 max_int16 = AE_MOVDA32(0x7fff);
  ae_int32x2 min_int16 = AE_MOVDA32(0xffff8000L);

  if(((((unsigned)p_vec1) & 15) == 0) && ((row_stride1 & 15) == 0) && ((rows&3) == 0) && ((cols1 & 15) == 0))
  {
    ae_valignx2 align_bias;
    if(bias_flag)
    align_bias = AE_LA128_PP(p_bias);

    for(m_itr = 0; m_itr < (rows); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        AE_LA32X2X2_IP(acc_row0_vec0, acc_row1_vec0, align_bias, (ae_int32x4 *)p_bias);
      }

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      _xa_nn_dot_product_4_rows_16x_cols_mat_unalined_vec_aligned
        (&acc_row0_vec0
         ,&acc_row1_vec0
         ,(ae_int8x8*)p_mat1_0
         ,(ae_int8*)p_vec1_0
         ,cols1
         ,row_stride1
         ,vec1_zero_bias
        );
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);

      AE_MINMAX32(acc_row0_vec0, min_int16, max_int16);
      AE_MINMAX32(acc_row1_vec0, min_int16, max_int16);

      out16_0 = AE_MOVINT16X4_FROMINT32X2(acc_row0_vec0);
      out16_1 = AE_MOVINT16X4_FROMINT32X2(acc_row1_vec0);

      AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_1, out16_1), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_1, (ae_int16 *) p_out, out_stride_by_2);
    }
  }
  else if(((((unsigned)p_out) & 15) == 0) && ((((unsigned)p_mat1) & 15) == 0) &&
      ((((unsigned)p_vec1) & 15) == 0) && ((((unsigned)p_bias) & 15) == 0) &&
      ((row_stride1 & 15) == 0))
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
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row2_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row3_vec0, out_multiplier, left_shift, right_shift);

      AE_MINMAX32(acc_row0_vec0, min_int16, max_int16);
      AE_MINMAX32(acc_row1_vec0, min_int16, max_int16);
      AE_MINMAX32(acc_row2_vec0, min_int16, max_int16);
      AE_MINMAX32(acc_row3_vec0, min_int16, max_int16);

      out16_0 = AE_MOVINT16X4_FROMINT32X2(acc_row0_vec0);
      out16_1 = AE_MOVINT16X4_FROMINT32X2(acc_row1_vec0);
      out16_2 = AE_MOVINT16X4_FROMINT32X2(acc_row2_vec0);
      out16_3 = AE_MOVINT16X4_FROMINT32X2(acc_row3_vec0);

      AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_1, out16_1), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_1, (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_2, out16_2), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_2, (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(AE_SEL16_5432(out16_3, out16_3), (ae_int16 *) p_out, out_stride_by_2);
      AE_S16_0_XP(out16_3, (ae_int16 *) p_out, out_stride_by_2);
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
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      AE_MINMAX32(acc_row0_vec0, min_int16, max_int16);
      out16_0 = AE_MOVINT16X4_FROMINT32X2(acc_row0_vec0);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_out, out_stride_by_2);
    }
  }	
  else if (p_mat1 && p_vec1)
  {
    for(m_itr = 0; m_itr < (rows & ~(64 - 1)); m_itr += 64)
    {
      WORD16* p_dst_0 = (WORD16*)(p_out + m_itr*out_stride);
      for(ii = 0; ii < 16; ii++)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row1_vec0 = ZERO32;

        if(bias_flag)
        {
          /* Load bias in the accumulator */
          acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr + ii +  0], p_bias[m_itr + ii + 16]);
          acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + ii +  32], p_bias[m_itr + ii + 48]);
        }

        p_mat1_0 = (WORD8 *)(p_mat1+((m_itr + ii) * row_stride1));
        p_vec1_0 = (WORD8 *)(p_vec1);

        _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec1_0
           ,cols1
           ,row_stride1
           ,vec1_zero_bias
          );

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);
        AE_MINMAX32(acc_row0_vec0, min_int16, max_int16);
        AE_MINMAX32(acc_row1_vec0, min_int16, max_int16);

        out16_0 = AE_MOVINT16X4_FROMINT32X2(acc_row0_vec0);
        out16_1 = AE_MOVINT16X4_FROMINT32X2(acc_row1_vec0);

        AE_S16_0_X(out16_0, (ae_int16 *) p_dst_0, 16*out_stride_by_2);
        AE_S16_0_X(AE_SEL16_5432(out16_1, out16_1), (ae_int16 *) p_dst_0, 32*out_stride_by_2);
        AE_S16_0_X(out16_1, (ae_int16 *) p_dst_0, 48*out_stride_by_2);
        AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_dst_0, out_stride_by_2);
      }
    }

#if 1
    WORD16* p_dst_0 = (WORD16*)(p_out + m_itr*out_stride);

    for(; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int32x2 acc_row0_vec0 = ZERO32;
      ae_int32x2 acc_row1_vec0 = ZERO32;

      p_mat1_0 = (const WORD8 *)(p_mat1+(m_itr * row_stride1));
      p_vec1_0 = (const WORD8 *)(p_vec1);

      if(bias_flag)
      {
        /* Load bias in the accumulator */
        acc_row0_vec0 = AE_MOVDA32X2(p_bias[m_itr], p_bias[m_itr + 1]);
        acc_row1_vec0 = AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]);
      }

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec1_0
           ,cols1
           ,row_stride1
           ,vec1_zero_bias
          );
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row1_vec0, out_multiplier, left_shift, right_shift);

      AE_MINMAX32(acc_row0_vec0, min_int16, max_int16);
      AE_MINMAX32(acc_row1_vec0, min_int16, max_int16);

      out16_0 = AE_MOVINT16X4_FROMINT32X2(acc_row0_vec0);
      out16_1 = AE_MOVINT16X4_FROMINT32X2(acc_row1_vec0);

        AE_S16_0_XP(AE_SEL16_5432(out16_0, out16_0), (ae_int16 *) p_dst_0, out_stride_by_2);
        AE_S16_0_XP(out16_0, (ae_int16 *) p_dst_0, out_stride_by_2);
        AE_S16_0_XP(AE_SEL16_5432(out16_1, out16_1), (ae_int16 *) p_dst_0, out_stride_by_2);
        AE_S16_0_XP(out16_1, (ae_int16 *) p_dst_0, out_stride_by_2);
    }
#endif

    p_dst_0 = (WORD16*)(p_out + m_itr*out_stride);
    /* Compute last (rows % 64) output element */
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
         ,vec1_zero_bias
        );

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, out_multiplier, left_shift, right_shift);
      AE_MINMAX32(acc_row0_vec0, min_int16, max_int16);	
      out16_0 = AE_MOVINT16X4_FROMINT32X2(acc_row0_vec0);
      AE_S16_0_XP(out16_0, (ae_int16 *) p_dst_0, out_stride_by_2);
    }
  }
  else
  {
    return -1;
  }

  return 0;
}

