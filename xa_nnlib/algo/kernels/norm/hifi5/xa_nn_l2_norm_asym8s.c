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
#include "xa_type_def.h"
#include "common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros_hifi5.h"

//output: output_inv_sqrt (ae_int32x2), output_shift (int)
//input:  input (ae_int32x2) , reverse_shift (int)
#define GET_INV_SQRT_QUANTIZED_MULTIPLIER_EXP(output_inv_sqrt, output_shift, input, reverse_shift){\
  ae_int32x2 CT_Q31_minus_1, CT_Q31, CT_Q29, CT_ONE;\
  CT_Q31_minus_1 = AE_MOVDA32(Q31_minus_1);\
  CT_Q31 = AE_MOVDA32(Q31);\
  CT_Q29 = AE_MOVDA32(Q29);\
  CT_ONE = AE_MOVDA32(1);\
\
  xtbool2 b1, b2;\
  b1 = AE_LE32(input, CT_ONE);\
\
  if(AE_MOVAB2(b1))\
  {\
    output_inv_sqrt = AE_MOV32(CT_Q31_minus_1);\
    output_shift = 0;\
  }\
  else\
  {\
    output_shift = 11;\
    b2 = AE_LT32(input, CT_Q29);\
    while(!AE_MOVAB2(b2))\
    {\
      input = AE_SRAI32(input, 2);\
      ++output_shift;\
      b2 = AE_LT32(input, CT_Q29);\
    }\
\
    int max_left_shift_bits, max_left_shift_bit_pairs, left_shift_bit_pairs;\
    max_left_shift_bits = AE_NSA32_L(input);\
    max_left_shift_bit_pairs = max_left_shift_bits / 2;\
    left_shift_bit_pairs = max_left_shift_bit_pairs - 1;\
    output_shift -= left_shift_bit_pairs;\
    input = AE_SLAA32(input, (2*left_shift_bit_pairs));\
\
    ae_int32x2 fixedpoint_input, fixedpoint_half_input, fixedpoint_half_three, x, x2, x3, y1, y2;\
    fixedpoint_input = AE_SRAI32(input, 1);\
    fixedpoint_half_input = AE_SRAI32R(fixedpoint_input, 1);\
    fixedpoint_half_three = AE_MOVDA32(FIXED_POINT_HALF_THREE);\
    x = AE_MOVDA32(FIXED_POINT_ONE);\
\
    int i = 0;\
    for(i=0; i<5; i++)\
    {\
      x2 = AE_MULFP32X2RS(x, x);\
      x3 = AE_MULFP32X2RS(x2, x);\
      x3 = AE_SLAI32S(x3, 6);\
\
      y1 = AE_MULFP32X2RS(fixedpoint_half_three, x);\
      y2 = AE_MULFP32X2RS(fixedpoint_half_input, x3);\
\
      x = AE_SUB32S(y1, y2);\
      x = AE_SLAI32S(x, 3);\
    }\
\
    ae_int32x2 fixedpoint_half_sqrt_2;\
    fixedpoint_half_sqrt_2 = AE_MOVDA32(FIXED_POINT_HALF_SQRT_2);\
    output_inv_sqrt = AE_MULFP32X2RS(x, fixedpoint_half_sqrt_2);\
    if(output_shift < 0)\
    {\
      output_inv_sqrt = AE_SLAA32S(output_inv_sqrt, -output_shift);\
      output_shift = 0;\
    }\
    output_shift *= reverse_shift;\
\
  }\
}

static const int Q31_minus_1 = 0x7fffffff;
static const int Q31         = 0x80000000;
static const int Q29         = 0x20000000;
static const int FIXED_POINT_HALF_THREE = 0x18000000;
static const int FIXED_POINT_ONE = 0x10000000;
static const int FIXED_POINT_HALF_SQRT_2 = 0x5a82799a;

WORD32 xa_nn_l2_norm_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_inp,
                            WORD32 zero_point,
                            WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < -128) || (zero_point > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  ae_int8x16 *p_in  = (ae_int8x16 *)p_inp;
  ae_int8x16 *p_o   = (ae_int8x16 *)p_out;

  int output_scale = 7;
  int reverse_shift = -1;

  int i = 0;
  int rem_length = (num_elm & 15);
  int rem_length_shift_0 = ((rem_length) <= 4)?(4 - (rem_length)) * 16:0;
  int rem_length_shift_1 = (((rem_length) > 4) && ((rem_length) <= 8))?(8 - (rem_length)) * 16:64;
  int rem_length_shift_2 = (((rem_length) > 8) && ((rem_length) <= 12))?(12 - (rem_length)) * 16:0;
  int rem_length_shift_3 = ((rem_length) > 12)?(16 - (rem_length)) * 16:64;

  ae_valignx2 align_src_hf5, align_dst_hf5;
  align_src_hf5 = AE_LA128_PP(p_in);
  align_dst_hf5 = AE_ZALIGN128();

  ae_int8x8 m1, m2, z_8x8;
  ae_int16x4 z76, z54, z32, z10;
  ae_int32x2 acc;
  ae_int64 acc_0 = 0, acc_1 = 0;
  z_8x8 = AE_MOVDA8(zero_point);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_hf5, p_in);
    AE_SUBW8(z76, z54, m1, z_8x8);
    AE_SUBW8(z32, z10, m2, z_8x8);

    AE_MULAAAA2Q16(acc_0, acc_1, z76, z54, z76, z54);
    AE_MULAAAA2Q16(acc_0, acc_1, z32, z10, z32, z10);
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_hf5, p_in, rem_length);
    AE_SUBW8(z76, z54, m1, z_8x8);
    if(rem_length < 8)
    {
      z76 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z76), rem_length_shift_0), rem_length_shift_0));
      z54 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z54), rem_length_shift_1), rem_length_shift_1));
    }
    AE_MULAAAA2Q16(acc_0, acc_1, z76, z54, z76, z54);

    if(rem_length > 8)
    {
      AE_SUBW8(z32, z10, m2, z_8x8);
      z32 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z32), rem_length_shift_2), rem_length_shift_2));
      z10 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z10), rem_length_shift_3), rem_length_shift_3));
      AE_MULAAAA2Q16(acc_0, acc_1, z32, z10, z32, z10);
    }
  }

  acc_0 = AE_ADD64(acc_0, acc_1);
  acc = AE_SAT32X2(acc_0, acc_0);

  ae_int32x2 inv_l2norm_multiplier;
  int inv_l2norm_shift;
  GET_INV_SQRT_QUANTIZED_MULTIPLIER_EXP(inv_l2norm_multiplier, inv_l2norm_shift, acc, reverse_shift);

  int shift = inv_l2norm_shift + output_scale;
#if TFLITE_SINGLE_ROUNDING 
  int left_shift  = shift;
  int right_shift;
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift  = shift<0 ? 0 : shift;
  int right_shift = shift>0 ? 0 :-shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */
  
  ae_int32x2 y76, y54, y32, y10, x76, x54, x32, x10;

  p_in  = (ae_int8x16 *)p_inp;
  align_src_hf5 = AE_LA128_PP(p_in);

  ae_int16x4 one_16x4 = AE_MOVDA16(1);
  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_hf5, p_in);
    AE_SUBW8(z76, z54, m1, z_8x8);
    AE_SUBW8(z32, z10, m2, z_8x8);

    AE_MUL16X4(y76, y54, z76, one_16x4);
    AE_MUL16X4(y32, y10, z54, one_16x4);
    AE_MUL16X4(x76, x54, z32, one_16x4);
    AE_MUL16X4(x32, x10, z10, one_16x4);

    MPY_BY_QUANT_MULT_X2X2_OUT16(z76, y76, y54, inv_l2norm_multiplier, left_shift, right_shift); 
    MPY_BY_QUANT_MULT_X2X2_OUT16(z54, y32, y10, inv_l2norm_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2X2_OUT16(z32, x76, x54, inv_l2norm_multiplier, left_shift, right_shift); 
    MPY_BY_QUANT_MULT_X2X2_OUT16(z10, x32, x10, inv_l2norm_multiplier, left_shift, right_shift);

    m1 = AE_SAT8X8X16(z76, z54);
    m2 = AE_SAT8X8X16(z32, z10);

    AE_SA8X8X2_IP(m1, m2, align_dst_hf5, p_o);
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_hf5, p_in, rem_length);
    AE_SUBW8(z76, z54, m1, z_8x8);
    if(rem_length < 8)
    {
      z76 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z76), rem_length_shift_0), rem_length_shift_0));
      z54 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z54), rem_length_shift_1), rem_length_shift_1));
    }

    AE_MUL16X4(y76, y54, z76, one_16x4);
    AE_MUL16X4(y32, y10, z54, one_16x4);

    MPY_BY_QUANT_MULT_X2X2_OUT16(z76, y76, y54, inv_l2norm_multiplier, left_shift, right_shift); 
    MPY_BY_QUANT_MULT_X2X2_OUT16(z54, y32, y10, inv_l2norm_multiplier, left_shift, right_shift);

    m1 = AE_SAT8X8X16(z76, z54);

    if(rem_length >8)
    {
      AE_SUBW8(z32, z10, m2, z_8x8);

      z32 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z32), rem_length_shift_2), rem_length_shift_2));
      z10 = AE_MOVINT16X4_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT16X4(z10), rem_length_shift_3), rem_length_shift_3));

      AE_MUL16X4(x76, x54, z32, one_16x4);
      AE_MUL16X4(x32, x10, z10, one_16x4);

      MPY_BY_QUANT_MULT_X2X2_OUT16(z32, x76, x54, inv_l2norm_multiplier, left_shift, right_shift); 
      MPY_BY_QUANT_MULT_X2X2_OUT16(z10, x32, x10, inv_l2norm_multiplier, left_shift, right_shift);

      m2 = AE_SAT8X8X16(z32, z10);
    }

    AE_SAV8X8X2_XP(m1, m2, align_dst_hf5, p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst_hf5, p_o);

  return 0;
}

