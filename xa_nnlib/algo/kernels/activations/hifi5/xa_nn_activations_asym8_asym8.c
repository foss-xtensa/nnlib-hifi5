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
#include "xa_type_def.h"
#include "common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"

#define ALIGNMENT   16   /* 16 bytes alignment */

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define SUB_128(inp){\
  ae_int64 temp;\
  temp = AE_MOVINT64_FROMINT8X8(inp);\
  temp = AE_XOR(temp, offset_xor);\
  inp = AE_MOVINT8X8_FROMINT64(temp);\
}

#define ADD_128(inp){\
  ae_int64 temp;\
  temp = AE_MOVINT64_FROMINT8X8(inp);\
  temp = AE_XOR(temp, offset_xor);\
  inp = AE_MOVINT8X8_FROMINT64(temp);\
}

#define LIMIT(out, inp, min, max){\
  SUB_128(inp);\
  out = AE_MIN8(inp, max);\
  out = AE_MAX8(out, min);\
  ADD_128(out);\
}

#define MultiplyByQuantizedMultiplierGreaterThanOne(y, x, multiplier, lsh) {\
  y = AE_SLAA32(x, lsh);\
  y = AE_MULFP32X2RAS(y, multiplier);\
}

#define MultiplyByQuantizedMultiplierGreaterThanOneX2(y, z, l, m, multiplier, lsh) {\
  y = AE_SLAA32(l, lsh);\
  z = AE_SLAA32(m, lsh);\
  AE_MULF2P32X4RAS(y, z, y, z, multiplier, multiplier);\
}

#define MultiplyByQuantizedMultiplierSmallerThanOneExp(prod, val, multiplier, lsh) {\
  ae_int64 temp64_h, temp64_l;\
  prod = AE_MULFP32X2RAS(val, multiplier);\
  temp64_h = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(prod, ZERO));\
  temp64_l = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(prod, ZERO));\
  temp64_h = AE_SLAA64S(temp64_h, lsh);\
  temp64_l = AE_SLAA64S(temp64_l, lsh);\
  prod = AE_ROUND32X2F64SSYM(temp64_h, temp64_l);\
}

#define ROUNDING_HALF_SUM(s, a){\
  ae_int64 max32;\
  ae_int64 r=-1;\
  xtbool br;\
  max32 = Q31_minus_1;\
  s = AE_ADD64(max32, a);\
  br = AE_LE64((ae_int64)0, s);\
  AE_MOVT64(r, (ae_int64)1, br);\
  s = AE_SRAI64(AE_ADD64(s,r), 1);\
}

#define EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(y_out, inp)\
{\
  ae_int32x2 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6;\
\
  x1_in = AE_ADD32S(inp, CT_1_BY_8);\
  x2 = AE_MULFP32X2RS(x1_in, x1_in);\
  AE_MULF2P32X4RAS(x3, x4, x2, x2, x1_in, x2);\
  x4_by_4 = AE_SRAI32R(x4, 2);\
  y1 = AE_ADD32S(x4_by_4, x3);\
  y2 = AE_MULFP32X2RS(y1, CT_1_BY_3);\
  y3 = AE_ADD32S(y2, x2);\
  y4 = AE_SRAI32R(y3, 1);\
\
  y5 = AE_ADD32S(x1_in, y4); \
  y6 = AE_MULFP32X2RS(y5, CT);\
  y_out = AE_ADD32S(y6, CT);\
}

#define GEMMLOWP_EXP_BARREL_SHIFTER(out, exponent, FixedPointMultiplier, remainder)\
{\
  int shift_amount;\
  ae_int32x2 out1,  mask, scale;\
  xtbool2 b;\
\
  shift_amount = 27 + exponent;\
  scale = AE_SLAA32(ONE, shift_amount);\
\
  mask = AE_AND32(remainder,  scale);\
\
  b = AE_LT32(zero, mask);\
\
  out1 = AE_MULFP32X2RS(out, FixedPointMultiplier);\
  AE_MOVT32X2(out, out1, b);\
}

// calculates exp for inp < 0
#define EXP_Q26(y, inp)\
{\
  xtbool2 b;\
  ae_int32x2 x_in, x2, remainder;\
  ae_int32x2 a_mod_quater_minus_q_1_by_4;\
\
  x2 = AE_AND32(inp, mask_6fs);\
  a_mod_quater_minus_q_1_by_4 = AE_SUB32(x2, q_1_by_4);\
  x_in = AE_SLAI32(a_mod_quater_minus_q_1_by_4, 4);\
\
  EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(y, x_in)\
\
  remainder = AE_SUB32(a_mod_quater_minus_q_1_by_4, inp);\
\
  GEMMLOWP_EXP_BARREL_SHIFTER(y,-2, 1672461947, remainder);\
  GEMMLOWP_EXP_BARREL_SHIFTER(y,-1, 1302514674, remainder);\
  GEMMLOWP_EXP_BARREL_SHIFTER(y, 0, 790015084,  remainder);\
  GEMMLOWP_EXP_BARREL_SHIFTER(y, 1, 290630308,  remainder);\
  GEMMLOWP_EXP_BARREL_SHIFTER(y, 2, 39332535,   remainder);\
  GEMMLOWP_EXP_BARREL_SHIFTER(y, 3, 720401,     remainder);\
  GEMMLOWP_EXP_BARREL_SHIFTER(y, 4, 242,        remainder);\
\
  b = AE_EQ32(inp, zero);\
  AE_MOVT32X2(y, AE_MOVDA32(Q31_minus_1), b);\
}

#define EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCLX2(y_out1, y_out2, inp1, inp2)\
{\
  ae_int32x2 x1_in, x2, x3, x4, x4_by_4, y1, y2, y3, y4, y5, y6;\
  ae_int32x2 l1_in, l2, l3, l4, l4_by_4, m1, m2, m3, m4, m5, m6;\
\
  x1_in = AE_ADD32S(inp1, CT_1_BY_8);\
  l1_in = AE_ADD32S(inp2, CT_1_BY_8);\
  AE_MULF2P32X4RAS(x2, l2, x1_in, l1_in, x1_in, l1_in);\
  AE_MULF2P32X4RAS(x3, x4, x2, x2, x1_in, x2);\
  AE_MULF2P32X4RAS(l3, l4, l2, l2, l1_in, l2);\
  x4_by_4 = AE_SRAI32R(x4, 2);\
  l4_by_4 = AE_SRAI32R(l4, 2);\
  y1 = AE_ADD32S(x4_by_4, x3);\
  m1 = AE_ADD32S(l4_by_4, l3);\
  AE_MULF2P32X4RAS(y2, m2, y1, m1, CT_1_BY_3, CT_1_BY_3);\
  y3 = AE_ADD32S(y2, x2);\
  m3 = AE_ADD32S(m2, l2);\
  y4 = AE_SRAI32R(y3, 1);\
  m4 = AE_SRAI32R(m3, 1);\
\
  y5 = AE_ADD32S(x1_in, y4); \
  m5 = AE_ADD32S(l1_in, m4); \
  AE_MULF2P32X4RAS(y6, m6, y5, m5, CT, CT);\
  y_out1 = AE_ADD32S(y6, CT);\
  y_out2 = AE_ADD32S(m6, CT);\
}

#define GEMMLOWP_EXP_BARREL_SHIFTERX2(out_1, out_2, exponent, FixedPointMultiplier, remainder1, remainder2)\
{\
  int shift_amount;\
  ae_int32x2 out1,  mask1, scale;\
  ae_int32x2 out2,  mask2;\
  xtbool2 b1, b2;\
\
  shift_amount = 27 + exponent;\
  scale = AE_SLAA32(ONE, shift_amount);\
\
  mask1 = AE_AND32(remainder1,  scale);\
  mask2 = AE_AND32(remainder2,  scale);\
\
  b1 = AE_LT32(zero, mask1);\
  b2 = AE_LT32(zero, mask2);\
\
  AE_MULF2P32X4RAS(out1, out2, out_1, out_2, FixedPointMultiplier, FixedPointMultiplier);\
  AE_MOVT32X2(out_1, out1, b1);\
  AE_MOVT32X2(out_2, out2, b2);\
}

#define EXP_Q26X2(y1, y2, inp1, inp2)\
{\
  xtbool2 b;\
  ae_int32x2 x_in1, x_in2, x2, remainder1, remainder2;\
  ae_int32x2 a_mod_quater_minus_q_1_by_4_first;\
  ae_int32x2 a_mod_quater_minus_q_1_by_4_second;\
\
  x2 = AE_AND32(inp1, mask_6fs);\
  a_mod_quater_minus_q_1_by_4_first = AE_SUB32(x2, q_1_by_4);\
  x_in1 = AE_SLAI32(a_mod_quater_minus_q_1_by_4_first, 4);\
\
  x2 = AE_AND32(inp2, mask_6fs);\
  a_mod_quater_minus_q_1_by_4_second = AE_SUB32(x2, q_1_by_4);\
  x_in2 = AE_SLAI32(a_mod_quater_minus_q_1_by_4_second, 4);\
\
  EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCLX2(y1, y2, x_in1, x_in2)\
\
  remainder1 = AE_SUB32(a_mod_quater_minus_q_1_by_4_first, inp1);\
  remainder2 = AE_SUB32(a_mod_quater_minus_q_1_by_4_second, inp2);\
\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, -2, 1672461947, remainder1, remainder2);\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, -1, 1302514674, remainder1, remainder2);\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, 0, 790015084,   remainder1, remainder2);\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, 1, 290630308,   remainder1, remainder2);\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, 2, 39332535,    remainder1, remainder2);\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, 3, 720401,      remainder1, remainder2);\
  GEMMLOWP_EXP_BARREL_SHIFTERX2(y1, y2, 4, 242,         remainder1, remainder2);\
\
  b = AE_EQ32(inp1, zero);\
  AE_MOVT32X2(y1, AE_MOVDA32(Q31_minus_1), b);\
\
  b = AE_EQ32(inp2, zero);\
  AE_MOVT32X2(y2, AE_MOVDA32(Q31_minus_1), b);\
}

//extern ae_int32x2 one_over_one_plus_x_for_x_in_0_1(ae_int64 a);

//output: y1, y2 (ae_int32x2)
//input:  a1, a2 (ae_int32x2)
#define ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y1, y2, a1, a2){\
  ae_int64 s1, s2, s3, s4;\
  ae_int64 t1, t2, t3, t4;\
  ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
  ae_int32x2 half_den34, m2, x2, half_denominator_times_x2;\
  ae_int32x2 one_minus_half_denominator_times_x1;\
  ae_int32x2 one_minus_half_denominator_times_x2;\
  ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
  int i;\
\
  CT_48_by_7 = AE_MOVDA32(constant_48_over_17);\
  CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);\
  CT_F2_ONE = AE_MOVDA32(F2_ONE);\
\
  s1 = AE_MUL32_HH(a1, ONE);\
  s2 = AE_MUL32_LL(a1, ONE);\
  s3 = AE_MUL32_HH(a2, ONE);\
  s4 = AE_MUL32_LL(a2, ONE);\
\
  ROUNDING_HALF_SUM(t1, s1)\
  ROUNDING_HALF_SUM(t2, s2)\
  ROUNDING_HALF_SUM(t3, s3)\
  ROUNDING_HALF_SUM(t4, s4)\
\
  half_den12 = AE_MOVINT32X2_FROMINT64(t1);\
  half_den34 = AE_MOVINT32X2_FROMINT64(t2);\
  half_den12 = AE_SEL32_LL(half_den12, half_den34);\
\
  half_den34 = AE_MOVINT32X2_FROMINT64(t3);\
  m1 = AE_MOVINT32X2_FROMINT64(t4);\
  half_den34 = AE_SEL32_LL(half_den34, m1);\
\
  AE_MULF2P32X4RAS(m2, m1, half_den34, half_den12, CT_neg_32_by_7, CT_neg_32_by_7);\
  x1 = AE_ADD32S(m1, CT_48_by_7);\
  x2 = AE_ADD32S(m2, CT_48_by_7);\
\
  for(i=0; i<3; i++)\
  {\
    AE_MULF2P32X4RAS(half_denominator_times_x1, half_denominator_times_x2, x1, x2, half_den12, half_den34);\
    one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
    one_minus_half_denominator_times_x2 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x2);\
    AE_MULF2P32X4RAS(m1, m2, x1, x2, one_minus_half_denominator_times_x1, one_minus_half_denominator_times_x2);\
    m1 = AE_SLAI32S(m1, 2);\
    x1 = AE_ADD32S(x1, m1);\
  \
    m2 = AE_SLAI32S(m2, 2);\
    x2 = AE_ADD32S(x2, m2);\
  \
  }\
\
  y1 = AE_SLAI32S(x1, 1);\
  y2 = AE_SLAI32S(x2, 1);\
\
}

//output: y1 (ae_int32x2)
//input:  a1 (ae_int32x2)
#define ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1(y1, a1){\
  ae_int64 s1;\
  ae_int64 t1;\
  ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
  ae_int32x2 one_minus_half_denominator_times_x1;\
  ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
  int i;\
\
  CT_48_by_7 = AE_MOVDA32(constant_48_over_17);\
  CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);\
  CT_F2_ONE = AE_MOVDA32(F2_ONE);\
\
  s1 = AE_MUL32_HH(a1, ONE);\
\
  ROUNDING_HALF_SUM(t1, s1)\
\
  half_den12 = AE_MOVINT32X2_FROMINT64(t1);\
\
  m1 = AE_MULFP32X2RS(half_den12, CT_neg_32_by_7);\
  x1 = AE_ADD32S(m1, CT_48_by_7);\
\
  for(i=0; i<3; i++)\
  {\
    half_denominator_times_x1 = AE_MULFP32X2RS(x1, half_den12);\
    one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
    m1 = AE_MULFP32X2RS(x1, one_minus_half_denominator_times_x1);\
    m1 = AE_SLAI32S(m1, 2);\
    x1 = AE_ADD32S(x1, m1);\
  }\
\
  y1 = AE_SLAI32S(x1, 1);\
\
}

static const int CONSTANT_TERM =  (0x70f5a894);
static const int CONSTANT_1_OVER_3 = (0x2aaaaaab);
static const int CONSTANT_1_OVER_8 = (0x10000000);
static const int ONE_QUATER_Q26 = (0x2000000); // Q5.27
static const int MASK = (0x1ffffff);
static const int Q31_minus_1 = 0x7fffffff;
static const int constant_48_over_17 = 1515870810;
static const int constant_neg_32_over_17 = -1010580540;
static const int F2_ONE = 0x20000000;

WORD32 xa_nn_vec_sigmoid_asym8_asym8(UWORD8 *p_out,
                      const UWORD8 *p_vec,
                            WORD32 zero_point,
                            WORD32 input_range_radius,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(UWORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < 0) || (zero_point > 255)), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_left_shift < -31) || (input_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_range_radius < 0) || (input_range_radius > 255)), -1);

  int i;
  int rem_length = (vec_length & 15);
  ae_int32x2 x76, x54, x32, x10, y76, y54, y32, y10, l76, l54, l32, l10, m76, m54, m32, m10;
  ae_int32x2 z, mul, zero;
  ae_int32x2 mask_6fs = AE_MOVDA32(MASK);
  ae_int32x2 q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
  ae_int32x2 CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
  ae_int32x2 CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
  ae_int32x2 CT = AE_MOVDA32(CONSTANT_TERM);
  ae_int32x2 ONE = AE_MOVDA32(1);
  ae_int16x4 CONST_255_16x4 = AE_MOVDA16(255);
  ae_int16x4 CONST_256_16x4 = AE_MOVDA16(256);
  ae_int32x2 radius, minus_radius;
  ae_int16x4 radius_16, minus_radius_16;
  xtbool4 a7654, a3210, b7654, b3210, c7654, c3210, d7654, d3210, e7654, e3210, f7654, f3210;
  ae_int32x2 dequantized_y76, dequantized_y54, dequantized_y32, dequantized_y10, dequantized_x76, dequantized_x54, dequantized_x32, dequantized_x10;
  ae_int32x2 exp_y76, exp_y54, exp_y32, exp_y10, exp_x76, exp_x54, exp_x32, exp_x10;
  ae_valignx2 align_src_hf5, align_dst_hf5;
  ae_int8x8 m0, m1, m2, z_8x8;
  ae_int16x4 z76, z54, z32, z10, zero_16x4;
    
  UWORD8 *p_in  = (UWORD8 *)p_vec;
  UWORD8 *p_o = (UWORD8 *)p_out;

  align_src_hf5 = AE_LA128_PP((ae_int8x16 *)p_in);
  align_dst_hf5 = AE_ZALIGN128();

  radius = AE_MOVDA32(input_range_radius);
  minus_radius = AE_NEG32(radius);

  radius_16 = AE_MOVDA16(input_range_radius);
  minus_radius_16 = AE_NEG16S(radius_16);

  z = AE_MOVDA32(zero_point);
  z_8x8 = AE_MOVDA8(zero_point);
  mul = AE_MOVDA32(input_multiplier);
  zero = AE_ZERO32();
  zero_16x4 = AE_ZERO16();

  for(i=0; i<(vec_length >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
    AE_SUBW8U(z76, z54, m1, z_8x8);
    AE_SUBW8U(z32, z10, m2, z_8x8);

    // set flag if z <= minus_radius
    a7654 = AE_LE16(z76, minus_radius_16);
    a3210 = AE_LE16(z54, minus_radius_16);
    b7654 = AE_LE16(z32, minus_radius_16);
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    c7654 = AE_LT16(z76, radius_16);
    c3210 = AE_LT16(z54, radius_16);
    d7654 = AE_LT16(z32, radius_16);
    d3210 = AE_LT16(z10, radius_16);

    //set flag if z < 0
    e7654 = AE_LT16(z76, zero_16x4);
    e3210 = AE_LT16(z54, zero_16x4);
    f7654 = AE_LT16(z32, zero_16x4);
    f3210 = AE_LT16(z10, zero_16x4);

    AE_CVTI32X4F16(y76, y54, z76, 0);
    AE_CVTI32X4F16(y32, y10, z54, 0);
    AE_CVTI32X4F16(x76, x54, z32, 0);
    AE_CVTI32X4F16(x32, x10, z10, 0);

    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, mul, input_left_shift)

    // Computing Absolute value
    y76 = AE_ABS32(dequantized_y76);
    y54 = AE_ABS32(dequantized_y54);
    y32 = AE_ABS32(dequantized_y32);
    y10 = AE_ABS32(dequantized_y10);
    y76 = AE_NEG32(y76);
    y54 = AE_NEG32(y54);
    y32 = AE_NEG32(y32);
    y10 = AE_NEG32(y10);

    x76 = AE_ABS32(dequantized_x76);
    x54 = AE_ABS32(dequantized_x54);
    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x76 = AE_NEG32(x76);
    x54 = AE_NEG32(x54);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
    EXP_Q26X2(exp_y76, exp_y54, y76, y54);
    EXP_Q26X2(exp_y32, exp_y10, y32, y10);
    EXP_Q26X2(exp_x76, exp_x54, x76, x54);
    EXP_Q26X2(exp_x32, exp_x10, x32, x10);

    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y76, y54, exp_y76, exp_y54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, exp_y32, exp_y10)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x76, x54, exp_x76, exp_x54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    l76 = AE_SRAA32RS(y76, 23);
    l54 = AE_SRAA32RS(y54, 23);
    l32 = AE_SRAA32RS(y32, 23);
    l10 = AE_SRAA32RS(y10, 23);
    m76 = AE_SRAA32RS(x76, 23);
    m54 = AE_SRAA32RS(x54, 23);
    m32 = AE_SRAA32RS(x32, 23);
    m10 = AE_SRAA32RS(x10, 23);
    // Due to rounding operation, sometimes value gets set to 256.
    // We need to saturate it to 255. 
    // SATU8X8X16 used before store operation takes care of this.
      
    z76 = AE_CVT16X4(l76, l54);
    z54 = AE_CVT16X4(l32, l10);
    z32 = AE_CVT16X4(m76, m54);
    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
    AE_MOVT16X4(z76, AE_SUB16S(CONST_256_16x4, z76), e7654); 
    AE_MOVT16X4(z54, AE_SUB16S(CONST_256_16x4, z54), e3210); 
    AE_MOVT16X4(z32, AE_SUB16S(CONST_256_16x4, z32), f7654); 
    AE_MOVT16X4(z10, AE_SUB16S(CONST_256_16x4, z10), f3210); 

    // if(inp_centered <= -radius) output = 0
    AE_MOVT16X4(z76, AE_ZERO16(), a7654); 
    AE_MOVT16X4(z54, AE_ZERO16(), a3210); 
    AE_MOVT16X4(z32, AE_ZERO16(), b7654); 
    AE_MOVT16X4(z10, AE_ZERO16(), b3210); 

    // if(inp_centered >= radius) output = 255
    AE_MOVF16X4(z76, CONST_255_16x4, c7654); 
    AE_MOVF16X4(z54, CONST_255_16x4, c3210); 
    AE_MOVF16X4(z32, CONST_255_16x4, d7654); 
    AE_MOVF16X4(z10, CONST_255_16x4, d3210); 

    m0 = AE_SATU8X8X16(z76, z54);
    m1 = AE_SATU8X8X16(z32, z10);

    AE_SA8X8X2_IP(m0, m1, align_dst_hf5, (ae_int8x16 *)p_o);
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in, rem_length);
    AE_SUBW8U(z76, z54, m1, z_8x8);
    AE_SUBW8U(z32, z10, m2, z_8x8);

    // set flag if z <= minus_radius
    a7654 = AE_LE16(z76, minus_radius_16);
    a3210 = AE_LE16(z54, minus_radius_16);
    b7654 = AE_LE16(z32, minus_radius_16);
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    c7654 = AE_LT16(z76, radius_16);
    c3210 = AE_LT16(z54, radius_16);
    d7654 = AE_LT16(z32, radius_16);
    d3210 = AE_LT16(z10, radius_16);

    //set flag if z < 0
    e7654 = AE_LT16(z76, zero_16x4);
    e3210 = AE_LT16(z54, zero_16x4);
    f7654 = AE_LT16(z32, zero_16x4);
    f3210 = AE_LT16(z10, zero_16x4);

    AE_CVTI32X4F16(y76, y54, z76, 0);
    AE_CVTI32X4F16(y32, y10, z54, 0);
    AE_CVTI32X4F16(x76, x54, z32, 0);
    AE_CVTI32X4F16(x32, x10, z10, 0);

    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, mul, input_left_shift)

    // Computing Absolute value
    y76 = AE_ABS32(dequantized_y76);
    y54 = AE_ABS32(dequantized_y54);
    y32 = AE_ABS32(dequantized_y32);
    y10 = AE_ABS32(dequantized_y10);
    y76 = AE_NEG32(y76);
    y54 = AE_NEG32(y54);
    y32 = AE_NEG32(y32);
    y10 = AE_NEG32(y10);

    x76 = AE_ABS32(dequantized_x76);
    x54 = AE_ABS32(dequantized_x54);
    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x76 = AE_NEG32(x76);
    x54 = AE_NEG32(x54);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
    EXP_Q26X2(exp_y76, exp_y54, y76, y54);
    EXP_Q26X2(exp_y32, exp_y10, y32, y10);
    EXP_Q26X2(exp_x76, exp_x54, x76, x54);
    EXP_Q26X2(exp_x32, exp_x10, x32, x10);

    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y76, y54, exp_y76, exp_y54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, exp_y32, exp_y10)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x76, x54, exp_x76, exp_x54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    l76 = AE_SRAA32RS(y76, 23);
    l54 = AE_SRAA32RS(y54, 23);
    l32 = AE_SRAA32RS(y32, 23);
    l10 = AE_SRAA32RS(y10, 23);
    m76 = AE_SRAA32RS(x76, 23);
    m54 = AE_SRAA32RS(x54, 23);
    m32 = AE_SRAA32RS(x32, 23);
    m10 = AE_SRAA32RS(x10, 23);
    // Due to rounding operation, sometimes value gets set to 256.
    // We need to saturate it to 255. 
    // SATU8X8X16 used before store operation takes care of this.
      
    z76 = AE_CVT16X4(l76, l54);
    z54 = AE_CVT16X4(l32, l10);
    z32 = AE_CVT16X4(m76, m54);
    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
    AE_MOVT16X4(z76, AE_SUB16S(CONST_256_16x4, z76), e7654); 
    AE_MOVT16X4(z54, AE_SUB16S(CONST_256_16x4, z54), e3210); 
    AE_MOVT16X4(z32, AE_SUB16S(CONST_256_16x4, z32), f7654); 
    AE_MOVT16X4(z10, AE_SUB16S(CONST_256_16x4, z10), f3210); 

    // if(inp_centered <= -radius) output = 0
    AE_MOVT16X4(z76, AE_ZERO16(), a7654); 
    AE_MOVT16X4(z54, AE_ZERO16(), a3210); 
    AE_MOVT16X4(z32, AE_ZERO16(), b7654); 
    AE_MOVT16X4(z10, AE_ZERO16(), b3210); 

    // if(inp_centered >= radius) output = 255
    AE_MOVF16X4(z76, CONST_255_16x4, c7654); 
    AE_MOVF16X4(z54, CONST_255_16x4, c3210); 
    AE_MOVF16X4(z32, CONST_255_16x4, d7654); 
    AE_MOVF16X4(z10, CONST_255_16x4, d3210); 

    m0 = AE_SATU8X8X16(z76, z54);
    m1 = AE_SATU8X8X16(z32, z10);

    AE_SAV8X8X2_XP(m0, m1, align_dst_hf5, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst_hf5, p_o);

  return 0;
}

WORD32 xa_nn_vec_sigmoid_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_vec,
                            WORD32 zero_point,
                            WORD32 input_range_radius,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < -128) || (zero_point > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_left_shift < -31) || (input_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_range_radius < 0) || (input_range_radius > 255)), -1);

  int i;
  int rem_length = (vec_length & 15);
  ae_int32x2 x76, x54, x32, x10, y76, y54, y32, y10, l76, l54, l32, l10, m76, m54, m32, m10;
  ae_int32x2 z, mul, zero;
  ae_int32x2 mask_6fs = AE_MOVDA32(MASK);
  ae_int32x2 q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
  ae_int32x2 CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
  ae_int32x2 CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
  ae_int32x2 CT = AE_MOVDA32(CONSTANT_TERM);
  ae_int32x2 ONE = AE_MOVDA32(1);
  ae_int16x4 CONST_255_16x4 = AE_MOVDA16(255);
  ae_int16x4 CONST_256_16x4 = AE_MOVDA16(256);
  ae_int32x2 radius, minus_radius;
  ae_int16x4 radius_16, minus_radius_16;
  xtbool4 a7654, a3210, b7654, b3210, c7654, c3210, d7654, d3210, e7654, e3210, f7654, f3210;
  ae_int32x2 dequantized_y76, dequantized_y54, dequantized_y32, dequantized_y10, dequantized_x76, dequantized_x54, dequantized_x32, dequantized_x10;
  ae_int32x2 exp_y76, exp_y54, exp_y32, exp_y10, exp_x76, exp_x54, exp_x32, exp_x10;
  ae_valignx2 align_src_hf5, align_dst_hf5;
  ae_int8x8 m0, m1, m2, z_8x8;
  ae_int16x4 z76, z54, z32, z10, zero_16x4;
  
  /* Second operand for XOR instruction used in SUB_128 and ADD_128*/
  ae_int64 offset_xor = AE_MOVINT64_FROMINT8X8(AE_MOVDA8(128));
  
  WORD8 *p_in  = (WORD8 *)p_vec;
  WORD8 *p_o = (WORD8 *)p_out;

  align_src_hf5 = AE_LA128_PP((ae_int8x16 *)p_in);
  align_dst_hf5 = AE_ZALIGN128();

  radius = AE_MOVDA32(input_range_radius);
  minus_radius = AE_NEG32(radius);

  radius_16 = AE_MOVDA16(input_range_radius);
  minus_radius_16 = AE_NEG16S(radius_16);

  z = AE_MOVDA32(zero_point);
  z_8x8 = AE_MOVDA8(zero_point);
  mul = AE_MOVDA32(input_multiplier);
  zero = AE_ZERO32();
  zero_16x4 = AE_ZERO16();

  for(i=0; i<(vec_length >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
    AE_SUBW8(z76, z54, m1, z_8x8);
    AE_SUBW8(z32, z10, m2, z_8x8);

    // set flag if z <= minus_radius
    a7654 = AE_LE16(z76, minus_radius_16);
    a3210 = AE_LE16(z54, minus_radius_16);
    b7654 = AE_LE16(z32, minus_radius_16);
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    c7654 = AE_LT16(z76, radius_16);
    c3210 = AE_LT16(z54, radius_16);
    d7654 = AE_LT16(z32, radius_16);
    d3210 = AE_LT16(z10, radius_16);

    //set flag if z < 0
    e7654 = AE_LT16(z76, zero_16x4);
    e3210 = AE_LT16(z54, zero_16x4);
    f7654 = AE_LT16(z32, zero_16x4);
    f3210 = AE_LT16(z10, zero_16x4);

    AE_CVTI32X4F16(y76, y54, z76, 0);
    AE_CVTI32X4F16(y32, y10, z54, 0);
    AE_CVTI32X4F16(x76, x54, z32, 0);
    AE_CVTI32X4F16(x32, x10, z10, 0);

    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, mul, input_left_shift)

    // Computing Absolute value
    y76 = AE_ABS32(dequantized_y76);
    y54 = AE_ABS32(dequantized_y54);
    y32 = AE_ABS32(dequantized_y32);
    y10 = AE_ABS32(dequantized_y10);
    y76 = AE_NEG32(y76);
    y54 = AE_NEG32(y54);
    y32 = AE_NEG32(y32);
    y10 = AE_NEG32(y10);

    x76 = AE_ABS32(dequantized_x76);
    x54 = AE_ABS32(dequantized_x54);
    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x76 = AE_NEG32(x76);
    x54 = AE_NEG32(x54);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
    EXP_Q26X2(exp_y76, exp_y54, y76, y54);
    EXP_Q26X2(exp_y32, exp_y10, y32, y10);
    EXP_Q26X2(exp_x76, exp_x54, x76, x54);
    EXP_Q26X2(exp_x32, exp_x10, x32, x10);

    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y76, y54, exp_y76, exp_y54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, exp_y32, exp_y10)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x76, x54, exp_x76, exp_x54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    l76 = AE_SRAA32RS(y76, 23);
    l54 = AE_SRAA32RS(y54, 23);
    l32 = AE_SRAA32RS(y32, 23);
    l10 = AE_SRAA32RS(y10, 23);
    m76 = AE_SRAA32RS(x76, 23);
    m54 = AE_SRAA32RS(x54, 23);
    m32 = AE_SRAA32RS(x32, 23);
    m10 = AE_SRAA32RS(x10, 23);
    // Due to rounding operation, sometimes value gets set to 256.
    // We need to saturate it to 255. 
    // SATU8X8X16 used before store operation takes care of this.
      
    z76 = AE_CVT16X4(l76, l54);
    z54 = AE_CVT16X4(l32, l10);
    z32 = AE_CVT16X4(m76, m54);
    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
    AE_MOVT16X4(z76, AE_SUB16S(CONST_256_16x4, z76), e7654); 
    AE_MOVT16X4(z54, AE_SUB16S(CONST_256_16x4, z54), e3210); 
    AE_MOVT16X4(z32, AE_SUB16S(CONST_256_16x4, z32), f7654); 
    AE_MOVT16X4(z10, AE_SUB16S(CONST_256_16x4, z10), f3210); 

    // if(inp_centered <= -radius) output = 0
    AE_MOVT16X4(z76, AE_ZERO16(), a7654); 
    AE_MOVT16X4(z54, AE_ZERO16(), a3210); 
    AE_MOVT16X4(z32, AE_ZERO16(), b7654); 
    AE_MOVT16X4(z10, AE_ZERO16(), b3210); 

    // if(inp_centered >= radius) output = 255
    AE_MOVF16X4(z76, CONST_255_16x4, c7654); 
    AE_MOVF16X4(z54, CONST_255_16x4, c3210); 
    AE_MOVF16X4(z32, CONST_255_16x4, d7654); 
    AE_MOVF16X4(z10, CONST_255_16x4, d3210); 

    m0 = AE_SATU8X8X16(z76, z54);
    m1 = AE_SATU8X8X16(z32, z10);

    SUB_128(m0) 
    SUB_128(m1) 
    AE_SA8X8X2_IP(m0, m1, align_dst_hf5, (ae_int8x16 *)p_o);
  }
  AE_SA128POS_FP(align_dst_hf5, p_o);

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in, rem_length);
    AE_SUBW8(z76, z54, m1, z_8x8);
    AE_SUBW8(z32, z10, m2, z_8x8);

    // set flag if z <= minus_radius
    a7654 = AE_LE16(z76, minus_radius_16);
    a3210 = AE_LE16(z54, minus_radius_16);
    b7654 = AE_LE16(z32, minus_radius_16);
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    c7654 = AE_LT16(z76, radius_16);
    c3210 = AE_LT16(z54, radius_16);
    d7654 = AE_LT16(z32, radius_16);
    d3210 = AE_LT16(z10, radius_16);

    //set flag if z < 0
    e7654 = AE_LT16(z76, zero_16x4);
    e3210 = AE_LT16(z54, zero_16x4);
    f7654 = AE_LT16(z32, zero_16x4);
    f3210 = AE_LT16(z10, zero_16x4);

    AE_CVTI32X4F16(y76, y54, z76, 0);
    AE_CVTI32X4F16(y32, y10, z54, 0);
    AE_CVTI32X4F16(x76, x54, z32, 0);
    AE_CVTI32X4F16(x32, x10, z10, 0);

    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, mul, input_left_shift)

    // Computing Absolute value
    y76 = AE_ABS32(dequantized_y76);
    y54 = AE_ABS32(dequantized_y54);
    y32 = AE_ABS32(dequantized_y32);
    y10 = AE_ABS32(dequantized_y10);
    y76 = AE_NEG32(y76);
    y54 = AE_NEG32(y54);
    y32 = AE_NEG32(y32);
    y10 = AE_NEG32(y10);

    x76 = AE_ABS32(dequantized_x76);
    x54 = AE_ABS32(dequantized_x54);
    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x76 = AE_NEG32(x76);
    x54 = AE_NEG32(x54);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute sigmoid/logistic i.e. one_over_one_plus_x(exp(x))
    EXP_Q26X2(exp_y76, exp_y54, y76, y54);
    EXP_Q26X2(exp_y32, exp_y10, y32, y10);
    EXP_Q26X2(exp_x76, exp_x54, x76, x54);
    EXP_Q26X2(exp_x32, exp_x10, x32, x10);

    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y76, y54, exp_y76, exp_y54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, exp_y32, exp_y10)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x76, x54, exp_x76, exp_x54)
    ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    l76 = AE_SRAA32RS(y76, 23);
    l54 = AE_SRAA32RS(y54, 23);
    l32 = AE_SRAA32RS(y32, 23);
    l10 = AE_SRAA32RS(y10, 23);
    m76 = AE_SRAA32RS(x76, 23);
    m54 = AE_SRAA32RS(x54, 23);
    m32 = AE_SRAA32RS(x32, 23);
    m10 = AE_SRAA32RS(x10, 23);
    // Due to rounding operation, sometimes value gets set to 256.
    // We need to saturate it to 255. 
    // SATU8X8X16 used before store operation takes care of this.
      
    z76 = AE_CVT16X4(l76, l54);
    z54 = AE_CVT16X4(l32, l10);
    z32 = AE_CVT16X4(m76, m54);
    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = 1 - sigmoid(abs(dequantized_input))
    AE_MOVT16X4(z76, AE_SUB16S(CONST_256_16x4, z76), e7654); 
    AE_MOVT16X4(z54, AE_SUB16S(CONST_256_16x4, z54), e3210); 
    AE_MOVT16X4(z32, AE_SUB16S(CONST_256_16x4, z32), f7654); 
    AE_MOVT16X4(z10, AE_SUB16S(CONST_256_16x4, z10), f3210); 

    // if(inp_centered <= -radius) output = 0
    AE_MOVT16X4(z76, AE_ZERO16(), a7654); 
    AE_MOVT16X4(z54, AE_ZERO16(), a3210); 
    AE_MOVT16X4(z32, AE_ZERO16(), b7654); 
    AE_MOVT16X4(z10, AE_ZERO16(), b3210); 

    // if(inp_centered >= radius) output = 255
    AE_MOVF16X4(z76, CONST_255_16x4, c7654); 
    AE_MOVF16X4(z54, CONST_255_16x4, c3210); 
    AE_MOVF16X4(z32, CONST_255_16x4, d7654); 
    AE_MOVF16X4(z10, CONST_255_16x4, d3210); 

    m0 = AE_SATU8X8X16(z76, z54);
    m1 = AE_SATU8X8X16(z32, z10);

    SUB_128(m0) 
    SUB_128(m1) 
    AE_SAV8X8X2_XP(m0, m1, align_dst_hf5, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst_hf5, p_o);

  return 0;
}

/*
 * inp: p_vec: 1 byte aligned input pointer
 * out: p_out: 1 byte aligned output pointer*/
WORD32 xa_nn_vec_activation_min_max_asym8_asym8(UWORD8 * __restrict__ p_out,
                                      const  UWORD8 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
  int i;
  ae_int8x8 x, y, z, m, min, max;
  ae_valignx2 align_src, align_dst;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

  UWORD8 *p_o = p_out;
  UWORD8 *p_v = (UWORD8 *)p_vec;

  /* Second operand for XOR instruction used in SUB_128 and ADD_128*/
  ae_int64 offset_xor = AE_MOVINT64_FROMINT8X8(AE_MOVDA8(128));

  min  = AE_MOVDA8(activation_min);
  max  = AE_MOVDA8(activation_max);

  SUB_128(min)
  SUB_128(max)

  align_src = AE_LA128_PP((ae_int8x16 *)p_v);
  align_dst = AE_ZALIGN128();

  if((activation_max >= (int)255) && (activation_min <= (int)0))
  {
    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

      AE_SA8X8X2_IP(x, y, align_dst, (ae_int8x16 *)p_o);
    }
    AE_SA128POS_FP(align_dst, p_o);

    for(i=0; i < (vec_length & 15); i++)
    {
      AE_L8_IP(x, (ae_int8 *)p_v, sizeof(ae_int8));

      AE_S8_0_IP(x, (ae_int8 *)p_o, sizeof(ae_int8));
    }
  }
  else if((activation_max < (int)255) && (activation_min <= 0))
  {
    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

      /* Subtract 128 to change UNSIGNED RANGE to SIGNED RANGE*/
      SUB_128(x)
      SUB_128(y)

      z = AE_MIN8(x, max);
      m = AE_MIN8(y, max);

      /* ADD 128 to change SIGNED RANGE to UNSIGNED RANGE*/
      ADD_128(z)
      ADD_128(m)

      AE_SA8X8X2_IP(z, m, align_dst, (ae_int8x16 *)p_o);
    }

    AE_SA128POS_FP(align_dst, p_o);

    for(i=0; i < (vec_length & 15); i++)
    {
      AE_L8_IP(x, (ae_int8 *)p_v, sizeof(ae_int8));
            
      SUB_128(x)
          
      y = AE_MIN8(x, max);

      ADD_128(y)

      AE_S8_0_IP(y, (ae_int8 *)p_o, sizeof(ae_int8));
    }
  }
  else if((activation_max >= (int)255) && (activation_min > 0))
  {
    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

      SUB_128(x)
      SUB_128(y)

      z = AE_MAX8(x, min);
      m = AE_MAX8(y, min);

      ADD_128(z)
      ADD_128(m)

      AE_SA8X8X2_IP(z, m, align_dst, (ae_int8x16 *)p_o);
    }

    AE_SA128POS_FP(align_dst, p_o);

    for(i=0; i < (vec_length & 15); i++)
    {
      AE_L8_IP(x, (ae_int8 *)p_v, sizeof(ae_int8));

      SUB_128(x)

      y = AE_MAX8(x, min);

      ADD_128(y)

      AE_S8_0_IP(y, (ae_int8 *)p_o, sizeof(ae_int8));
    }
  }
  else
  {
    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

      LIMIT(z, x, min, max)
      LIMIT(m, y, min, max)

      AE_SA8X8X2_IP(z, m, align_dst, (ae_int8x16 *)p_o);
    }

    AE_SA128POS_FP(align_dst, p_o);

    for(i=0; i < (vec_length & 15); i++)
    {
      AE_L8_IP(x, (ae_int8 *)p_v, sizeof(ae_int8));

      LIMIT(y, x, min, max)

      AE_S8_0_IP(y, (ae_int8 *)p_o, sizeof(ae_int8));
    }
  }

  return 0;
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out, inp1, inp2, multiplier, left_shift, right_shift) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, left_shift, left_shift); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_NEG32(AE_MOVDA32(multiplier)), AE_NEG32(AE_MOVDA32(multiplier))); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, right_shift, right_shift); \
  out = AE_SAT16X4(inp1, inp2); \
}

WORD32 xa_nn_vec_relu_asym8u_asym8u( UWORD8 * __restrict__ p_out,
                    const   UWORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 quantized_activation_min,
                            WORD32 quantized_activation_max,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < 0) || (inp_zero_bias > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < 0) || (out_zero_bias > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_min < 0) || (quantized_activation_min > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_max < 0) || (quantized_activation_max > 255)), -1);
  XA_NNLIB_ARG_CHK_COND((quantized_activation_max < quantized_activation_min), -1);

  int rem_length = (vec_length & 15);

  WORD8 *p_o = (WORD8 *)p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int8x8 inp_zb = AE_MOVDA8(inp_zero_bias);
  ae_int16x4 act_min = AE_MOVDA16(quantized_activation_min);
  ae_int16x4 act_max = AE_MOVDA16(quantized_activation_max);
  ae_int16x4 one = AE_MOVDA16(1);
  
  int left_shift  = out_shift<0?0: out_shift;
  left_shift = (1 << left_shift);
  int right_shift = out_shift>0?0:-out_shift;
  right_shift = (0XFFFFFFFF << (31 - right_shift));
  
  ae_valignx2 align_src  = AE_LA128_PP((ae_int8x16 *)p_v);
  ae_valignx2 align_dst = AE_ZALIGN128();

  for(i=0; i<(vec_length >> 4); i++)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_v0_0, d_v0_1;
    ae_int16x4 d_v1_0, d_v1_1;
    ae_int32x2 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    ae_int32x2 d_w2_0, d_w2_1, d_w3_0, d_w3_1;
    
    AE_LA8X8X2_IP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v);
    
    AE_SUBW8U(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8U(d_v1_0, d_v1_1, d_inp1, inp_zb); 

    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    AE_MUL16X4(d_w1_0, d_w1_1, d_v0_1, one); 
    AE_MUL16X4(d_w2_0, d_w2_1, d_v1_0, one); 
    AE_MUL16X4(d_w3_0, d_w3_1, d_v1_1, one); 

    ae_int16x4 out0, out1, out2, out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out1, d_w1_0, d_w1_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out2, d_w2_0, d_w2_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out3, d_w3_0, d_w3_1, out_multiplier, left_shift, right_shift); 

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);

    //Clamp the output in the quantized activation range
    AE_MINMAX16(out0, act_min, act_max);
    AE_MINMAX16(out1, act_min, act_max);
    AE_MINMAX16(out2, act_min, act_max);
    AE_MINMAX16(out3, act_min, act_max);

    ae_int8x8 out8_0 = AE_SATU8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SATU8X8X16(out2, out3);
  
    AE_SA8X8X2_IP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o);
  }
 
  //remainder loop
  if(rem_length)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_v0_0, d_v0_1;
    ae_int16x4 d_v1_0, d_v1_1;
    ae_int32x2 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    ae_int32x2 d_w2_0, d_w2_1, d_w3_0, d_w3_1;
    
    AE_LAV8X8X2_XP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v, rem_length);
    AE_SUBW8U(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8U(d_v1_0, d_v1_1, d_inp1, inp_zb); 

    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    AE_MUL16X4(d_w1_0, d_w1_1, d_v0_1, one); 
    AE_MUL16X4(d_w2_0, d_w2_1, d_v1_0, one); 
    AE_MUL16X4(d_w3_0, d_w3_1, d_v1_1, one); 

    ae_int16x4 out0, out1, out2, out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out1, d_w1_0, d_w1_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out2, d_w2_0, d_w2_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out3, d_w3_0, d_w3_1, out_multiplier, left_shift, right_shift); 

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);
 
    //Clamp the output in the quantized activation range
    AE_MINMAX16(out0, act_min, act_max);
    AE_MINMAX16(out1, act_min, act_max);
    AE_MINMAX16(out2, act_min, act_max);
    AE_MINMAX16(out3, act_min, act_max);

    ae_int8x8 out8_0 = AE_SATU8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SATU8X8X16(out2, out3);
  
    AE_SAV8X8X2_XP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_vec_relu_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 quantized_activation_min,
                            WORD32 quantized_activation_max,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_min < -128) || (quantized_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((quantized_activation_max < -128) || (quantized_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((quantized_activation_max < quantized_activation_min), -1);

  int rem_length = (vec_length & 15);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int8x8 inp_zb = AE_MOVDA8(inp_zero_bias);
  ae_int16x4 act_min = AE_MOVDA16(quantized_activation_min);
  ae_int16x4 act_max = AE_MOVDA16(quantized_activation_max);
  ae_int16x4 one = AE_MOVDA16(1);
  
  int left_shift  = out_shift<0?0: out_shift;
  left_shift = (1 << left_shift);
  int right_shift = out_shift>0?0:-out_shift;
  right_shift = (0XFFFFFFFF << (31 - right_shift));
  
  ae_valignx2 align_src  = AE_LA128_PP((ae_int8x16 *)p_v);
  ae_valignx2 align_dst = AE_ZALIGN128();

  for(i=0; i<(vec_length >> 4); i++)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_v0_0, d_v0_1;
    ae_int16x4 d_v1_0, d_v1_1;
    ae_int32x2 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    ae_int32x2 d_w2_0, d_w2_1, d_w3_0, d_w3_1;
    
    AE_LA8X8X2_IP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v);
    
    AE_SUBW8(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8(d_v1_0, d_v1_1, d_inp1, inp_zb); 

    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    AE_MUL16X4(d_w1_0, d_w1_1, d_v0_1, one); 
    AE_MUL16X4(d_w2_0, d_w2_1, d_v1_0, one); 
    AE_MUL16X4(d_w3_0, d_w3_1, d_v1_1, one); 

    ae_int16x4 out0, out1, out2, out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out1, d_w1_0, d_w1_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out2, d_w2_0, d_w2_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out3, d_w3_0, d_w3_1, out_multiplier, left_shift, right_shift); 

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);

    //Clamp the output in the quantized activation range
    AE_MINMAX16(out0, act_min, act_max);
    AE_MINMAX16(out1, act_min, act_max);
    AE_MINMAX16(out2, act_min, act_max);
    AE_MINMAX16(out3, act_min, act_max);

    ae_int8x8 out8_0 = AE_SAT8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SAT8X8X16(out2, out3);
  
    AE_SA8X8X2_IP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o);
  }
 
  //remainder loop
  if(rem_length)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_v0_0, d_v0_1;
    ae_int16x4 d_v1_0, d_v1_1;
    ae_int32x2 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    ae_int32x2 d_w2_0, d_w2_1, d_w3_0, d_w3_1;
    
    AE_LAV8X8X2_XP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v, rem_length);
    AE_SUBW8(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8(d_v1_0, d_v1_1, d_inp1, inp_zb); 

    // Multiply with out multiplier
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    AE_MUL16X4(d_w1_0, d_w1_1, d_v0_1, one); 
    AE_MUL16X4(d_w2_0, d_w2_1, d_v1_0, one); 
    AE_MUL16X4(d_w3_0, d_w3_1, d_v1_1, one); 

    ae_int16x4 out0, out1, out2, out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out1, d_w1_0, d_w1_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out2, d_w2_0, d_w2_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out3, d_w3_0, d_w3_1, out_multiplier, left_shift, right_shift); 

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);
 
    //Clamp the output in the quantized activation range
    AE_MINMAX16(out0, act_min, act_max);
    AE_MINMAX16(out1, act_min, act_max);
    AE_MINMAX16(out2, act_min, act_max);
    AE_MINMAX16(out3, act_min, act_max);

    ae_int8x8 out8_0 = AE_SAT8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SAT8X8X16(out2, out3);
  
    AE_SAV8X8X2_XP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_vec_prelu_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                    const   WORD8 * __restrict__ p_vec_alpha,
                            WORD32 inp_zero_bias,
                            WORD32 alpha_zero_bias,
                            WORD32 alpha_multiplier,
                            WORD32 alpha_shift,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec_alpha, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -127) || (inp_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_zero_bias < -127) || (alpha_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_shift < -31) || (alpha_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((alpha_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int rem_length = (vec_length & 15);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;
  WORD8 *p_v_a = (WORD8 *)p_vec_alpha;

  ae_int8x8 inp_zb = AE_MOVDA8(-inp_zero_bias);
  ae_int8x8 alpha_zb = AE_MOVDA8(-alpha_zero_bias);
  ae_int16x4 one = AE_MOVDA16(1);
  ae_int16x4 zero = AE_ZERO16();
  
  int left_shift  = out_shift<0?0: out_shift;
  left_shift = (1 << left_shift);
  int right_shift = out_shift>0?0:-out_shift;
  right_shift = (0XFFFFFFFF << (31 - right_shift));
  
  int a_left_shift  = alpha_shift<0?0: alpha_shift;
  a_left_shift = (1 << a_left_shift);
  int a_right_shift = alpha_shift>0?0:-alpha_shift;
  a_right_shift = (0XFFFFFFFF << (31 - a_right_shift));

  ae_valignx2 align_src  = AE_LA128_PP((ae_int8x16 *)p_v);
  ae_valignx2 align_src1 = AE_LA128_PP((ae_int8x16 *)p_v_a);
  ae_valignx2 align_dst = AE_ZALIGN128();

  for(i=0; i<(vec_length >> 4); i++)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int8x8 d_a_inp0, d_a_inp1;
    ae_int16x4 d_v0_0, d_v0_1;
    ae_int16x4 d_v1_0, d_v1_1;
    ae_int32x2 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    ae_int32x2 d_w2_0, d_w2_1, d_w3_0, d_w3_1;
    
    AE_LA8X8X2_IP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v);
    AE_LA8X8X2_IP(d_a_inp0, d_a_inp1, align_src1, (ae_int8x16*)p_v_a);
    
    AE_SUBW8(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8(d_v1_0, d_v1_1, d_inp1, inp_zb); 

    //Checking for input values less than inp_zero_bias
    xtbool4 sel0 = AE_LT16(d_v0_0, zero);
    xtbool4 sel1 = AE_LT16(d_v0_1, zero);
    xtbool4 sel2 = AE_LT16(d_v1_0, zero);
    xtbool4 sel3 = AE_LT16(d_v1_1, zero);
    
    // Multiply with out multiplier for input values >= 0
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    AE_MUL16X4(d_w1_0, d_w1_1, d_v0_1, one); 
    AE_MUL16X4(d_w2_0, d_w2_1, d_v1_0, one); 
    AE_MUL16X4(d_w3_0, d_w3_1, d_v1_1, one); 

    ae_int16x4 out0, out1, out2, out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out1, d_w1_0, d_w1_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out2, d_w2_0, d_w2_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out3, d_w3_0, d_w3_1, out_multiplier, left_shift, right_shift); 

    // Add alpha zero bias and multiply with alpha multiplier for input values < 0
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1, d_alpha_w1_0, d_alpha_w1_1;
    ae_int32x2 d_alpha_w2_0, d_alpha_w2_1, d_alpha_w3_0, d_alpha_w3_1;
    ae_int16x4 d_alpha_v0_0, d_alpha_v0_1, d_alpha_v1_0, d_alpha_v1_1;
    AE_SUBW8(d_alpha_v0_0, d_alpha_v0_1, d_a_inp0, alpha_zb); 
    AE_SUBW8(d_alpha_v1_0, d_alpha_v1_1, d_a_inp1, alpha_zb); 
   
    AE_MUL16X4(d_alpha_w0_0, d_alpha_w0_1, d_v0_0, d_alpha_v0_0);
    AE_MUL16X4(d_alpha_w1_0, d_alpha_w1_1, d_v0_1, d_alpha_v0_1);
    AE_MUL16X4(d_alpha_w2_0, d_alpha_w2_1, d_v1_0, d_alpha_v1_0);
    AE_MUL16X4(d_alpha_w3_0, d_alpha_w3_1, d_v1_1, d_alpha_v1_1);

    ae_int16x4 a_out0, a_out1, a_out2, a_out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out0, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out1, d_alpha_w1_0, d_alpha_w1_1, alpha_multiplier, a_left_shift, a_right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out2, d_alpha_w2_0, d_alpha_w2_1, alpha_multiplier, a_left_shift, a_right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out3, d_alpha_w3_0, d_alpha_w3_1, alpha_multiplier, a_left_shift, a_right_shift); 

    AE_MOVT16X4(out0, a_out0, sel0);
    AE_MOVT16X4(out1, a_out1, sel1);
    AE_MOVT16X4(out2, a_out2, sel2);
    AE_MOVT16X4(out3, a_out3, sel3);

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);
  
    ae_int8x8 out8_0 = AE_SAT8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SAT8X8X16(out2, out3);
  
    AE_SA8X8X2_IP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o);
  }
 
  //remainder loop
  if(rem_length)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int8x8 d_a_inp0, d_a_inp1;
    ae_int16x4 d_v0_0, d_v0_1;
    ae_int16x4 d_v1_0, d_v1_1;
    ae_int32x2 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    ae_int32x2 d_w2_0, d_w2_1, d_w3_0, d_w3_1;
    
    AE_LAV8X8X2_XP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v, rem_length);
    AE_LAV8X8X2_XP(d_a_inp0, d_a_inp1, align_src1, (ae_int8x16*)p_v_a, rem_length);
    
    AE_SUBW8(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8(d_v1_0, d_v1_1, d_inp1, inp_zb); 

    //Checking for input values less than 0.
    xtbool4 sel0 = AE_LT16(d_v0_0, zero);
    xtbool4 sel1 = AE_LT16(d_v0_1, zero);
    xtbool4 sel2 = AE_LT16(d_v1_0, zero);
    xtbool4 sel3 = AE_LT16(d_v1_1, zero);

    // Multiply with out multiplier for input values >= 0
    AE_MUL16X4(d_w0_0, d_w0_1, d_v0_0, one); 
    AE_MUL16X4(d_w1_0, d_w1_1, d_v0_1, one); 
    AE_MUL16X4(d_w2_0, d_w2_1, d_v1_0, one); 
    AE_MUL16X4(d_w3_0, d_w3_1, d_v1_1, one); 

    ae_int16x4 out0, out1, out2, out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out0, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out1, d_w1_0, d_w1_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out2, d_w2_0, d_w2_1, out_multiplier, left_shift, right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out3, d_w3_0, d_w3_1, out_multiplier, left_shift, right_shift); 

    // Add alpha zero bias and multiply with alpha multiplier for input values < 0
    ae_int32x2 d_alpha_w0_0, d_alpha_w0_1, d_alpha_w1_0, d_alpha_w1_1;
    ae_int32x2 d_alpha_w2_0, d_alpha_w2_1, d_alpha_w3_0, d_alpha_w3_1;
    ae_int16x4 d_alpha_v0_0, d_alpha_v0_1, d_alpha_v1_0, d_alpha_v1_1;
    AE_SUBW8(d_alpha_v0_0, d_alpha_v0_1, d_a_inp0, alpha_zb); 
    AE_SUBW8(d_alpha_v1_0, d_alpha_v1_1, d_a_inp1, alpha_zb); 
   
    AE_MUL16X4(d_alpha_w0_0, d_alpha_w0_1, d_v0_0, d_alpha_v0_0);
    AE_MUL16X4(d_alpha_w1_0, d_alpha_w1_1, d_v0_1, d_alpha_v0_1);
    AE_MUL16X4(d_alpha_w2_0, d_alpha_w2_1, d_v1_0, d_alpha_v1_0);
    AE_MUL16X4(d_alpha_w3_0, d_alpha_w3_1, d_v1_1, d_alpha_v1_1);
    
    ae_int16x4 a_out0, a_out1, a_out2, a_out3;
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out0, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out1, d_alpha_w1_0, d_alpha_w1_1, alpha_multiplier, a_left_shift, a_right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out2, d_alpha_w2_0, d_alpha_w2_1, alpha_multiplier, a_left_shift, a_right_shift); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(a_out3, d_alpha_w3_0, d_alpha_w3_1, alpha_multiplier, a_left_shift, a_right_shift); 

    AE_MOVT16X4(out0, a_out0, sel0);
    AE_MOVT16X4(out1, a_out1, sel1);
    AE_MOVT16X4(out2, a_out2, sel2);
    AE_MOVT16X4(out3, a_out3, sel3);

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);
  
    ae_int8x8 out8_0 = AE_SAT8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SAT8X8X16(out2, out3);
  
    AE_SAV8X8X2_XP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(inp1, multiplier, left_shift, right_shift, ext_shift) \
{\
  inp1 = AE_MULP16X16X4S(inp1, AE_MOVDA16(left_shift)); \
  inp1 = AE_MULFP16X4RAS(inp1, multiplier); \
  inp1 = AE_MULP16X16X4S(inp1, AE_MOVDA16(ext_shift)); \
  inp1 = AE_SRAA16SYMS(inp1, right_shift); \
}

WORD32 xa_nn_vec_hard_swish_asym8s_asym8s( WORD8 * __restrict__ p_out,
                            const   WORD8 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD16 reluish_multiplier,
                            WORD32 reluish_shift,
                            WORD16 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((reluish_shift < -31) || (reluish_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((reluish_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int rem_length = (vec_length & 15);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  ae_int8x8 inp_zb = AE_MOVDA8(inp_zero_bias);
  ae_int16x4 hires_mul = AE_MOVDA16(128);
  
  int r_left_shift  = reluish_shift > 0 ? reluish_shift-1 : 0;
  r_left_shift = (1 << r_left_shift);
  int ext_lsh = reluish_shift > 0 ? 1 : 0;
  ext_lsh = (1 << ext_lsh);
  int r_right_shift = reluish_shift>0?0:-reluish_shift;

  /* Second operand for XOR instruction for ADD_32768*/
  ae_int64 offset_xor = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(32768));
  
  ae_valignx2 align_src = AE_LA128_PP((ae_int8x16 *)p_v);
  ae_valignx2 align_dst = AE_ZALIGN128();

  for(i=0; i<(vec_length >> 4); i++)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_v0_0, d_v0_1, d_v1_0, d_v1_1;
    ae_int16x4 d_r0_0, d_r0_1, d_r1_0, d_r1_1;
    ae_int16x4 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    
    AE_LA8X8X2_IP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v);
    
    AE_SUBW8(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8(d_v1_0, d_v1_1, d_inp1, inp_zb);

    //Shifting the result to MSB bits
    d_r0_0 = AE_MULP16X16X4S(d_v0_0, hires_mul);
    d_r0_1 = AE_MULP16X16X4S(d_v0_1, hires_mul);
    d_r1_0 = AE_MULP16X16X4S(d_v1_0, hires_mul);
    d_r1_1 = AE_MULP16X16X4S(d_v1_1, hires_mul);

    // Multiply shifted result with out multiplier 
    d_w0_0 = AE_MULFP16X4RAS(d_r0_0, out_multiplier); 
    d_w0_1 = AE_MULFP16X4RAS(d_r0_1, out_multiplier); 
    d_w1_0 = AE_MULFP16X4RAS(d_r1_0, out_multiplier); 
    d_w1_1 = AE_MULFP16X4RAS(d_r1_1, out_multiplier);

    // Multiply the shifted result with reluish multiplier and apply reluish shift
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r0_0, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r0_1, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r1_0, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r1_1, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 

    // Bring the output in [0,1] range from [-1, 1] range.
    d_r0_0 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r0_0), offset_xor)); 
    d_r0_1 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r0_1), offset_xor)); 
    d_r1_0 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r1_0), offset_xor)); 
    d_r1_1 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r1_1), offset_xor));
    d_r0_0 = AE_SRLI16(d_r0_0, 1);
    d_r0_1 = AE_SRLI16(d_r0_1, 1);
    d_r1_0 = AE_SRLI16(d_r1_0, 1);
    d_r1_1 = AE_SRLI16(d_r1_1, 1);

    // Multiply inp*relu6(inp+3)
    ae_int16x4 out0, out1, out2, out3;
    out0 = AE_MULFP16X4S(d_w0_0, d_r0_0); 
    out1 = AE_MULFP16X4S(d_w0_1, d_r0_1); 
    out2 = AE_MULFP16X4S(d_w1_0, d_r1_0); 
    out3 = AE_MULFP16X4S(d_w1_1, d_r1_1);

    out0 = AE_SRAA16SYMS(out0, -out_shift); 
    out1 = AE_SRAA16SYMS(out1, -out_shift); 
    out2 = AE_SRAA16SYMS(out2, -out_shift); 
    out3 = AE_SRAA16SYMS(out3, -out_shift);

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);
  
    ae_int8x8 out8_0 = AE_SAT8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SAT8X8X16(out2, out3);
  
    AE_SA8X8X2_IP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o);
  }
 
  //remainder loop
  if(rem_length)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_v0_0, d_v0_1, d_v1_0, d_v1_1;
    ae_int16x4 d_r0_0, d_r0_1, d_r1_0, d_r1_1;
    ae_int16x4 d_w0_0, d_w0_1, d_w1_0, d_w1_1;
    
    AE_LAV8X8X2_XP(d_inp0, d_inp1, align_src, (ae_int8x16*)p_v, rem_length);
    
    AE_SUBW8(d_v0_0, d_v0_1, d_inp0, inp_zb); 
    AE_SUBW8(d_v1_0, d_v1_1, d_inp1, inp_zb);

    //Shifting the result to MSB bits
    d_r0_0 = AE_MULP16X16X4S(d_v0_0, hires_mul);
    d_r0_1 = AE_MULP16X16X4S(d_v0_1, hires_mul);
    d_r1_0 = AE_MULP16X16X4S(d_v1_0, hires_mul);
    d_r1_1 = AE_MULP16X16X4S(d_v1_1, hires_mul);

    // Multiply shifted result with out multiplier 
    d_w0_0 = AE_MULFP16X4RAS(d_r0_0, out_multiplier); 
    d_w0_1 = AE_MULFP16X4RAS(d_r0_1, out_multiplier); 
    d_w1_0 = AE_MULFP16X4RAS(d_r1_0, out_multiplier); 
    d_w1_1 = AE_MULFP16X4RAS(d_r1_1, out_multiplier);

    // Multiply the shifted result with reluish multiplier and apply reluish shift
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r0_0, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r0_1, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r1_0, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4_Q15(d_r1_1, AE_MOVDA16(reluish_multiplier), r_left_shift, r_right_shift, ext_lsh); 

    // Bring the output in [0,1] range from [-1, 1] range.
    d_r0_0 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r0_0), offset_xor)); 
    d_r0_1 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r0_1), offset_xor)); 
    d_r1_0 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r1_0), offset_xor)); 
    d_r1_1 = AE_MOVINT16X4_FROMINT64(AE_XOR(AE_MOVINT64_FROMINT16X4(d_r1_1), offset_xor));
    d_r0_0 = AE_SRLI16(d_r0_0, 1);
    d_r0_1 = AE_SRLI16(d_r0_1, 1);
    d_r1_0 = AE_SRLI16(d_r1_0, 1);
    d_r1_1 = AE_SRLI16(d_r1_1, 1);

    // Multiply inp*relu6(inp+3)
    ae_int16x4 out0, out1, out2, out3;
    out0 = AE_MULFP16X4S(d_w0_0, d_r0_0); 
    out1 = AE_MULFP16X4S(d_w0_1, d_r0_1); 
    out2 = AE_MULFP16X4S(d_w1_0, d_r1_0); 
    out3 = AE_MULFP16X4S(d_w1_1, d_r1_1);

    out0 = AE_SRAA16SYMS(out0, -out_shift); 
    out1 = AE_SRAA16SYMS(out1, -out_shift); 
    out2 = AE_SRAA16SYMS(out2, -out_shift); 
    out3 = AE_SRAA16SYMS(out3, -out_shift);

    out0 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out0);
    out1 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out1);
    out2 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out2);
    out3 = AE_ADD16S(AE_MOVDA16(out_zero_bias), out3);
  
    ae_int8x8 out8_0 = AE_SAT8X8X16(out0, out1);
    ae_int8x8 out8_1 = AE_SAT8X8X16(out2, out3);
  
    AE_SAV8X8X2_XP(out8_0, out8_1, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);
  
  return 0;
}

//output: y1 (ae_int32x2)
//input:  a1 (ae_int32x2)
#define ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1(y1, a1){\
  ae_int64 s1;\
  ae_int64 t1;\
  ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
  ae_int32x2 one_minus_half_denominator_times_x1;\
  ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
  int i;\
\
  CT_48_by_7 = AE_MOVDA32(constant_48_over_17);\
  CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);\
  CT_F2_ONE = AE_MOVDA32(F2_ONE);\
\
  s1 = AE_MUL32_HH(a1, ONE);\
\
  ROUNDING_HALF_SUM(t1, s1)\
\
  half_den12 = AE_MOVINT32X2_FROMINT64(t1);\
\
  m1 = AE_MULFP32X2RS(half_den12, CT_neg_32_by_7);\
  x1 = AE_ADD32S(m1, CT_48_by_7);\
\
  for(i=0; i<3; i++)\
  {\
    half_denominator_times_x1 = AE_MULFP32X2RS(x1, half_den12);\
    one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
    m1 = AE_MULFP32X2RS(x1, one_minus_half_denominator_times_x1);\
    m1 = AE_SLAI32S(m1, 2);\
    x1 = AE_ADD32S(x1, m1);\
  }\
\
  x1 = AE_SUB32S(x1, CT_F2_ONE);\
  y1 = AE_SLAI32S(x1, 2);\
\
}

//output: y1, y2 (ae_int32x2)
//input:  a1, a2 (ae_int32x2)
#define ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y1, y2, a1, a2){\
  ae_int64 s1, s2, s3, s4;\
  ae_int64 t1, t2, t3, t4;\
  ae_int32x2 half_den12, m1, x1, half_denominator_times_x1;\
  ae_int32x2 half_den34, m2, x2, half_denominator_times_x2;\
  ae_int32x2 one_minus_half_denominator_times_x1;\
  ae_int32x2 one_minus_half_denominator_times_x2;\
  ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;\
  int i;\
\
  CT_48_by_7 = AE_MOVDA32(constant_48_over_17);\
  CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);\
  CT_F2_ONE = AE_MOVDA32(F2_ONE);\
\
  s1 = AE_MUL32_HH(a1, ONE);\
  s2 = AE_MUL32_LL(a1, ONE);\
  s3 = AE_MUL32_HH(a2, ONE);\
  s4 = AE_MUL32_LL(a2, ONE);\
\
  ROUNDING_HALF_SUM(t1, s1)\
  ROUNDING_HALF_SUM(t2, s2)\
  ROUNDING_HALF_SUM(t3, s3)\
  ROUNDING_HALF_SUM(t4, s4)\
\
  half_den12 = AE_MOVINT32X2_FROMINT64(t1);\
  half_den34 = AE_MOVINT32X2_FROMINT64(t2);\
  half_den12 = AE_SEL32_LL(half_den12, half_den34);\
\
  half_den34 = AE_MOVINT32X2_FROMINT64(t3);\
  m1 = AE_MOVINT32X2_FROMINT64(t4);\
  half_den34 = AE_SEL32_LL(half_den34, m1);\
\
  AE_MULF2P32X4RAS(m2, m1, half_den34, half_den12, CT_neg_32_by_7, CT_neg_32_by_7);\
  x1 = AE_ADD32S(m1, CT_48_by_7);\
  x2 = AE_ADD32S(m2, CT_48_by_7);\
\
  for(i=0; i<3; i++)\
  {\
    AE_MULF2P32X4RAS(half_denominator_times_x1, half_denominator_times_x2, x1, x2, half_den12, half_den34);\
    one_minus_half_denominator_times_x1 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x1);\
    one_minus_half_denominator_times_x2 = AE_SUB32S(CT_F2_ONE, half_denominator_times_x2);\
    AE_MULF2P32X4RAS(m1, m2, x1, x2, one_minus_half_denominator_times_x1, one_minus_half_denominator_times_x2);\
    m1 = AE_SLAI32S(m1, 2);\
    x1 = AE_ADD32S(x1, m1);\
  \
    m2 = AE_SLAI32S(m2, 2);\
    x2 = AE_ADD32S(x2, m2);\
  \
  }\
\
  x1 = AE_SUB32S(x1, CT_F2_ONE);\
  x2 = AE_SUB32S(x2, CT_F2_ONE);\
  y1 = AE_SLAI32S(x1, 2);\
  y2 = AE_SLAI32S(x2, 2);\
\
}

WORD32 xa_nn_vec_tanh_asym8s_asym8s(WORD8 *p_out,
                      const WORD8 *p_vec,
                            WORD32 zero_point,
                            WORD32 input_range_radius,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((zero_point < -128) || (zero_point > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_left_shift < -31) || (input_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_range_radius < 0) || (input_range_radius > 255)), -1);

  int i;
  int rem_length = (vec_length & 15);
  ae_int32x2 x76, x54, x32, x10, y76, y54, y32, y10, l76, l54, l32, l10, m76, m54, m32, m10;
  ae_int32x2 z, mul, zero;
  ae_int32x2 mask_6fs = AE_MOVDA32(MASK);
  ae_int32x2 q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
  ae_int32x2 CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
  ae_int32x2 CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
  ae_int32x2 CT = AE_MOVDA32(CONSTANT_TERM);
  ae_int32x2 ONE = AE_MOVDA32(1);
  ae_int16x4 CONST_127_16x4 = AE_MOVDA16(127);
  ae_int16x4 CONST_MINUS_128_16x4 = AE_MOVDA16(-128);
  ae_int32x2 radius, minus_radius;
  ae_int16x4 radius_16, minus_radius_16;
  xtbool4 a7654, a3210, b7654, b3210, c7654, c3210, d7654, d3210, e7654, e3210, f7654, f3210;
  ae_int32x2 dequantized_y76, dequantized_y54, dequantized_y32, dequantized_y10, dequantized_x76, dequantized_x54, dequantized_x32, dequantized_x10;
  ae_int32x2 exp_y76, exp_y54, exp_y32, exp_y10, exp_x76, exp_x54, exp_x32, exp_x10;
  ae_valignx2 align_src_hf5, align_dst_hf5;
  ae_int8x8 m0, m1, m2, z_8x8;
  ae_int16x4 z76, z54, z32, z10, zero_16x4;
  
  WORD8 *p_in  = (WORD8 *)p_vec;
  WORD8 *p_o = (WORD8 *)p_out;

  align_src_hf5 = AE_LA128_PP((ae_int8x16 *)p_in);
  align_dst_hf5 = AE_ZALIGN128();

  radius = AE_MOVDA32(input_range_radius);
  minus_radius = AE_NEG32(radius);

  radius_16 = AE_MOVDA16(input_range_radius);
  minus_radius_16 = AE_NEG16S(radius_16);

  z = AE_MOVDA32(zero_point);
  z_8x8 = AE_MOVDA8(zero_point);
  mul = AE_MOVDA32(input_multiplier);
  zero = AE_ZERO32();
  zero_16x4 = AE_ZERO16();

  for(i=0; i<(vec_length >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
    AE_SUBW8(z76, z54, m1, z_8x8);
    AE_SUBW8(z32, z10, m2, z_8x8);

    // set flag if z <= minus_radius
    a7654 = AE_LE16(z76, minus_radius_16);
    a3210 = AE_LE16(z54, minus_radius_16);
    b7654 = AE_LE16(z32, minus_radius_16);
    b3210 = AE_LE16(z10, minus_radius_16);

    // set flag if z < radius
    c7654 = AE_LT16(z76, radius_16);
    c3210 = AE_LT16(z54, radius_16);
    d7654 = AE_LT16(z32, radius_16);
    d3210 = AE_LT16(z10, radius_16);

    //set flag if z < 0
    e7654 = AE_LT16(z76, zero_16x4);
    e3210 = AE_LT16(z54, zero_16x4);
    f7654 = AE_LT16(z32, zero_16x4);
    f3210 = AE_LT16(z10, zero_16x4);

    AE_CVTI32X4F16(y76, y54, z76, 0);
    AE_CVTI32X4F16(y32, y10, z54, 0);
    AE_CVTI32X4F16(x76, x54, z32, 0);
    AE_CVTI32X4F16(x32, x10, z10, 0);

    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, mul, input_left_shift)

    // Computing Absolute value
    y76 = AE_ABS32(dequantized_y76);
    y54 = AE_ABS32(dequantized_y54);
    y32 = AE_ABS32(dequantized_y32);
    y10 = AE_ABS32(dequantized_y10);
    y76 = AE_NEG32(y76);
    y54 = AE_NEG32(y54);
    y32 = AE_NEG32(y32);
    y10 = AE_NEG32(y10);

    x76 = AE_ABS32(dequantized_x76);
    x54 = AE_ABS32(dequantized_x54);
    x32 = AE_ABS32(dequantized_x32);
    x10 = AE_ABS32(dequantized_x10);
    x76 = AE_NEG32(x76);
    x54 = AE_NEG32(x54);
    x32 = AE_NEG32(x32);
    x10 = AE_NEG32(x10);

    // Compute tanh i.e. one_minus_x_over_one_plus_x(exp(-2x))
    y76 = AE_SLAI32S(y76, 1);
    y54 = AE_SLAI32S(y54, 1);
    y32 = AE_SLAI32S(y32, 1);
    y10 = AE_SLAI32S(y10, 1);
    x76 = AE_SLAI32S(x76, 1);
    x54 = AE_SLAI32S(x54, 1);
    x32 = AE_SLAI32S(x32, 1);
    x10 = AE_SLAI32S(x10, 1);

    EXP_Q26X2(exp_y76, exp_y54, y76, y54);
    EXP_Q26X2(exp_y32, exp_y10, y32, y10);
    EXP_Q26X2(exp_x76, exp_x54, x76, x54);
    EXP_Q26X2(exp_x32, exp_x10, x32, x10);

    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y76, y54, exp_y76, exp_y54)
    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, exp_y32, exp_y10)
    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x76, x54, exp_x76, exp_x54)
    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

    // Downscale to 8 bit
    l76 = AE_SRAA32RS(y76, 24);
    l54 = AE_SRAA32RS(y54, 24);
    l32 = AE_SRAA32RS(y32, 24);
    l10 = AE_SRAA32RS(y10, 24);
    m76 = AE_SRAA32RS(x76, 24);
    m54 = AE_SRAA32RS(x54, 24);
    m32 = AE_SRAA32RS(x32, 24);
    m10 = AE_SRAA32RS(x10, 24);
    // Due to rounding operation, sometimes value gets set to 128.
    // We need to saturate it to 127. 
    // SAT8X8X16 used before store operation takes care of this.
      
    z76 = AE_CVT16X4(l76, l54);
    z54 = AE_CVT16X4(l32, l10);
    z32 = AE_CVT16X4(m76, m54);
    z10 = AE_CVT16X4(m32, m10);

    // if(inp_centered < 0) output = - tanh(abs(dequantized_input))
    AE_MOVT16X4(z76, AE_NEG16S(z76), e7654); 
    AE_MOVT16X4(z54, AE_NEG16S(z54), e3210); 
    AE_MOVT16X4(z32, AE_NEG16S(z32), f7654); 
    AE_MOVT16X4(z10, AE_NEG16S(z10), f3210); 

    // if(inp_centered <= -radius) output = -128
    AE_MOVT16X4(z76, CONST_MINUS_128_16x4, a7654); 
    AE_MOVT16X4(z54, CONST_MINUS_128_16x4, a3210); 
    AE_MOVT16X4(z32, CONST_MINUS_128_16x4, b7654); 
    AE_MOVT16X4(z10, CONST_MINUS_128_16x4, b3210); 

    // if(inp_centered >= radius) output = 127
    AE_MOVF16X4(z76, CONST_127_16x4, c7654); 
    AE_MOVF16X4(z54, CONST_127_16x4, c3210); 
    AE_MOVF16X4(z32, CONST_127_16x4, d7654); 
    AE_MOVF16X4(z10, CONST_127_16x4, d3210); 

    m0 = AE_SAT8X8X16(z76, z54);
    m1 = AE_SAT8X8X16(z32, z10);

    AE_SA8X8X2_IP(m0, m1, align_dst_hf5, (ae_int8x16 *)p_o);
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in, rem_length);
    AE_SUBW8(z76, z54, m1, z_8x8);

    // set flag if z <= minus_radius
    a7654 = AE_LE16(z76, minus_radius_16);
    a3210 = AE_LE16(z54, minus_radius_16);

    // set flag if z < radius
    c7654 = AE_LT16(z76, radius_16);
    c3210 = AE_LT16(z54, radius_16);

    //set flag if z < 0
    e7654 = AE_LT16(z76, zero_16x4);
    e3210 = AE_LT16(z54, zero_16x4);

    AE_CVTI32X4F16(y76, y54, z76, 0);
    AE_CVTI32X4F16(y32, y10, z54, 0);

    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, mul, input_left_shift)
    MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, mul, input_left_shift)

    // Computing Absolute value
    y76 = AE_ABS32(dequantized_y76);
    y54 = AE_ABS32(dequantized_y54);
    y32 = AE_ABS32(dequantized_y32);
    y10 = AE_ABS32(dequantized_y10);
    y76 = AE_NEG32(y76);
    y54 = AE_NEG32(y54);
    y32 = AE_NEG32(y32);
    y10 = AE_NEG32(y10);

    // Compute tanh i.e. one_minus_x_over_one_plus_x(exp(-2x))
    y76 = AE_SLAI32S(y76, 1);
    y54 = AE_SLAI32S(y54, 1);
    y32 = AE_SLAI32S(y32, 1);
    y10 = AE_SLAI32S(y10, 1);

    EXP_Q26X2(exp_y76, exp_y54, y76, y54);
    EXP_Q26X2(exp_y32, exp_y10, y32, y10);

    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y76, y54, exp_y76, exp_y54)
    ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(y32, y10, exp_y32, exp_y10)

    // Downscale to 8 bit
    l76 = AE_SRAA32RS(y76, 24);
    l54 = AE_SRAA32RS(y54, 24);
    l32 = AE_SRAA32RS(y32, 24);
    l10 = AE_SRAA32RS(y10, 24);
    // Due to rounding operation, sometimes value gets set to 128.
    // We need to saturate it to 127. 
    // SAT8X8X16 used before store operation takes care of this.
      
    z76 = AE_CVT16X4(l76, l54);
    z54 = AE_CVT16X4(l32, l10);

    // if(inp_centered < 0) output = - tanh(abs(dequantized_input))
    AE_MOVT16X4(z76, AE_NEG16S(z76), e7654); 
    AE_MOVT16X4(z54, AE_NEG16S(z54), e3210); 

    // if(inp_centered < -radius) output = -128
    AE_MOVT16X4(z76, CONST_MINUS_128_16x4, a7654); 
    AE_MOVT16X4(z54, CONST_MINUS_128_16x4, a3210); 

    // if(inp_centered > radius) output = 127
    AE_MOVF16X4(z76, CONST_127_16x4, c7654); 
    AE_MOVF16X4(z54, CONST_127_16x4, c3210); 

    m0 = AE_SAT8X8X16(z76, z54);

    if(rem_length >8)
    {
      AE_SUBW8(z32, z10, m2, z_8x8);

      // set flag if z <= minus_radius
      b7654 = AE_LE16(z32, minus_radius_16);
      b3210 = AE_LE16(z10, minus_radius_16);

      // set flag if z < radius
      d7654 = AE_LT16(z32, radius_16);
      d3210 = AE_LT16(z10, radius_16);

      //set flag if z < 0
      f7654 = AE_LT16(z32, zero_16x4);
      f3210 = AE_LT16(z10, zero_16x4);

      AE_CVTI32X4F16(x76, x54, z32, 0);
      AE_CVTI32X4F16(x32, x10, z10, 0);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, mul, input_left_shift)
      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, mul, input_left_shift)

      // Computing Absolute value
      x76 = AE_ABS32(dequantized_x76);
      x54 = AE_ABS32(dequantized_x54);
      x32 = AE_ABS32(dequantized_x32);
      x10 = AE_ABS32(dequantized_x10);
      x76 = AE_NEG32(x76);
      x54 = AE_NEG32(x54);
      x32 = AE_NEG32(x32);
      x10 = AE_NEG32(x10);

      // Compute tanh i.e. one_minus_x_over_one_plus_x(exp(-2x))
      x76 = AE_SLAI32S(x76, 1);
      x54 = AE_SLAI32S(x54, 1);
      x32 = AE_SLAI32S(x32, 1);
      x10 = AE_SLAI32S(x10, 1);

      EXP_Q26X2(exp_x76, exp_x54, x76, x54);
      EXP_Q26X2(exp_x32, exp_x10, x32, x10);

      ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x76, x54, exp_x76, exp_x54)
      ONE_MINUS_X_OVER_ONE_PLUS_X_FOR_X_IN_0_1_32X2(x32, x10, exp_x32, exp_x10)

      // Downscale to 8 bit
      m76 = AE_SRAA32RS(x76, 24);
      m54 = AE_SRAA32RS(x54, 24);
      m32 = AE_SRAA32RS(x32, 24);
      m10 = AE_SRAA32RS(x10, 24);
      // Due to rounding operation, sometimes value gets set to 128.
      // We need to saturate it to 127. 
      // SAT8X8X16 used before store operation takes care of this.
      
      z32 = AE_CVT16X4(m76, m54);
      z10 = AE_CVT16X4(m32, m10);

      // if(inp_centered < 0) output = - tanh(abs(dequantized_input))
      AE_MOVT16X4(z32, AE_NEG16S(z32), f7654); 
      AE_MOVT16X4(z10, AE_NEG16S(z10), f3210); 

      // if(inp_centered <= -radius) output = -128
      AE_MOVT16X4(z32, CONST_MINUS_128_16x4, b7654); 
      AE_MOVT16X4(z10, CONST_MINUS_128_16x4, b3210); 

      // if(inp_centered >= radius) output = 127
      AE_MOVF16X4(z32, CONST_127_16x4, d7654); 
      AE_MOVF16X4(z10, CONST_127_16x4, d3210); 

      m1 = AE_SAT8X8X16(z32, z10);
    }

    AE_SAV8X8X2_XP(m0, m1, align_dst_hf5, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst_hf5, p_o);

  return 0;
}

#if 0
enum ActivationFn {
    kActivationNone = 0,
    kActivationRelu,
    kActivationRelu1,
    kActivationRelu6,
    kActivationTanh,
    kActivationSignBit,
    kActivationSigmoid,
};

#define QUANTIZE(y, f){\
    xtfloat recip_scale, prod;\
    recip_scale = XT_FLOAT_S(input_scale, 0);\
    recip_scale = XT_RECIP_S(recip_scale);\
    prod = XT_MUL_S(recip_scale, XT_FLOAT_S(f, 0));\
    prod = XT_FIROUND_S(prod);\
    y = XT_ADD_S(prod, XT_FLOAT_S(input_offset, 0));\
}

#define CALCULATE_ACTIVATION_RANGE_ASYM8(activation){\
    if (activation == kActivationRelu)\
    {\
        QUANTIZE(y, 0.0)\
        act_min = XT_MAX_S(0, y);\
        act_max = 255;\
    }\
    else if (activation == kActivationRelu6) \
    {\
       QUANTIZE(y, 0.0)\
       act_min = XT_MAX_S(0, y);\
       QUANTIZE(y, 6.0)\
       act_max = XT_MIN_S(255, y);\
    }\
    else if (activation == kActivationRelu1) \
    {\
       QUANTIZE(y, -1.0)\
       act_min = XT_MAX_S(0, y);\
       QUANTIZE(y, 1.0)\
       act_max = XT_MIN_S(255, y);\
    }\
    else if (activation == kActivationNone)\
    {\
        act_min = 0;\
        act_max = 255;\
    }\
}


WORD32 xa_nn_vec_relu_asym8(
    UWORD8       * __restrict__ p_out,
    const UWORD8 * __restrict__ p_inp,
    WORD32       input_offset,
    WORD32       input_scale,
    WORD32       vec_length)
{
  xtfloat y, act_max, act_min;

  // Calculating act_min and act_max
  CALCULATE_ACTIVATION_RANGE_ASYM8(kActivationRelu)

  relu_asym8(p_out,
             p_inp,
             act_min,
             act_max,
             vec_length);
  return 0;
}


WORD32 xa_nn_vec_relu1_asym8(
    UWORD8       * __restrict__ p_out,
    const UWORD8 * __restrict__ p_inp,
    WORD32       input_offset,
    WORD32       input_scale,
    WORD32       vec_length)
{
  xtfloat y, act_max, act_min;
  // Calculating act_min and act_max
  CALCULATE_ACTIVATION_RANGE_ASYM8(kActivationRelu1)

  relu_asym8(p_out,
             p_inp,
             act_min,
             act_max,
             vec_length);

  return 0;
}

WORD32 xa_nn_vec_relu6_asym8(
    UWORD8       * __restrict__ p_out,
    const UWORD8 * __restrict__ p_inp,
    WORD32       input_offset,
    WORD32       input_scale,
    WORD32       vec_length)
{
  xtfloat y, act_max, act_min;
  // Calculating act_min and act_max
  CALCULATE_ACTIVATION_RANGE_ASYM8(kActivationRelu6)

  relu_asym8(p_out,
             p_inp,
             act_min,
             act_max,
             vec_length);

  return 0;
}
#endif

