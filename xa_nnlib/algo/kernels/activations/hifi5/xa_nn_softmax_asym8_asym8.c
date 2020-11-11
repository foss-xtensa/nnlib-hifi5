/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
//#include "xa_nn_basic_state.h"
#include "xa_type_def.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

#include "xa_nnlib_common_macros_hifi5.h"

#define ALIGNMENT   16   /* 16 bytes alignment */
#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define SUB_128(inp){\
        ae_int64 temp;\
        temp = AE_MOVINT64_FROMINT8X8(inp);\
        temp = AE_XOR(temp, offset_xor);\
        inp = AE_MOVINT8X8_FROMINT64(temp);\
}

#define SUB_32768(inp){\
        ae_int64 temp;\
        temp = AE_MOVINT64_FROMINT16X4(inp);\
        temp = AE_XOR(temp, offset_xor_16);\
        inp = AE_MOVINT16X4_FROMINT64(temp);\
}

static const int CONSTANT_TERM =  (0x70f5a894);
static const int CONSTANT_1_OVER_3 = (0x2aaaaaab);
static const int CONSTANT_1_OVER_8 = (0x10000000);
static const int ONE_QUATER_Q26 = (0x1000000); // Q6.26
static const int MASK = (0xffffff);
static const int Q31 = 0x7fffffff;
static const int constant_48_over_17 = 1515870810;
static const int constant_neg_32_over_17 = -1010580540;
static const int F2_ONE = 0x20000000;

#define MultiplyByQuantizedMultiplierGreaterThanOne(y, x, multiplier, lsh) {\
    y = AE_SLAA32(x, lsh);\
    y = AE_MULFP32X2RAS(y, multiplier);\
}

#define MultiplyByQuantizedMultiplierGreaterThanOneX2(y, z, l, m, multiplier, lsh) {\
    y = AE_SLAA32(l, lsh);\
    z = AE_SLAA32(m, lsh);\
    AE_MULF2P32X4RAS(y, z, y, z, multiplier, multiplier);\
}

#define ROUNDING_HALF_SUM(s, a){\
    ae_int64 max32;\
    ae_int64 r=-1;\
    xtbool br;\
    max32 = Q31;\
    s = AE_ADD64(max32, a);\
    br = AE_LE64((ae_int64)0, s);\
    AE_MOVT64(r, (ae_int64)1, br);\
    s = AE_SRAI64(AE_ADD64(s,r), 1);\
}


static ae_int32x2 one_over_one_plus_x_for_x_in_0_1(ae_int64 a)
{
    ae_int64 s;
    ae_int32x2 half_den, m, x, half_denominator_times_x;
    ae_int32x2 one_minus_half_denominator_times_x;
    ae_int32x2 CT_48_by_7, CT_neg_32_by_7, CT_F2_ONE;
    int i;

    CT_48_by_7 = AE_MOVDA32(constant_48_over_17);
    CT_neg_32_by_7 = AE_MOVDA32(constant_neg_32_over_17);
    CT_F2_ONE = AE_MOVDA32(F2_ONE);

    ROUNDING_HALF_SUM(s, a)

    half_den = AE_MOVINT32X2_FROMINT64(s);
    half_den = AE_SEL32_LL(half_den, half_den); // half denominator


    // Computation of x
    m = AE_MULFP32X2RS(half_den, CT_neg_32_by_7);
    x = AE_ADD32S(m, CT_48_by_7);

    for(i=0; i<3; i++)
    {
        half_denominator_times_x = AE_MULFP32X2RS(x, half_den);
        one_minus_half_denominator_times_x = AE_SUB32S(CT_F2_ONE, half_denominator_times_x);
        m = AE_MULFP32X2RS(x, one_minus_half_denominator_times_x);
        m = AE_SLAI32S(m, 2);
        x = AE_ADD32S(x, m);
    }

    x = AE_SLAI32S(x, 1);

    return x;
}
ae_int32x2 GetReciprocal(ae_int64 x, int x_integerbits, int *lsh)
{
    int headroom_plus_one;
    ae_int64 shifted_sum_minus_one, CT_Q31;
    ae_int64 shifted_sum;
    ae_int32x2 scale;

    headroom_plus_one = AE_NSA64(x) - 31;
    *lsh = x_integerbits - headroom_plus_one;


    CT_Q31 = Q31;
    shifted_sum = AE_SLAA64(x, headroom_plus_one);

    shifted_sum_minus_one = AE_SUB64(shifted_sum, CT_Q31);
    scale = one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one);
    return scale;
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
    shift_amount = 26 + exponent;\
    scale = AE_SLAA32(ONE, shift_amount);\
\
    mask = AE_AND32(remainder,  scale);\
\
    b = AE_LT32(z, mask);\
\
    out1 = AE_MULFP32X2RS(out, FixedPointMultiplier);\
    AE_MOVT32X2(out, out1, b);\
}

#define EXP_Q26(y, inp)\
{\
    xtbool2 b;\
    ae_int32x2 x_in, x2, remainder;\
    ae_int32x2 a_mod_quater_minus_q_1_by_4;\
\
    x2 = AE_AND32(inp, mask_6fs);\
    a_mod_quater_minus_q_1_by_4 = AE_SUB32(x2, q_1_by_4);\
    x_in = AE_SLAI32(a_mod_quater_minus_q_1_by_4, 5);\
\
    EXP_ON_INTERVAL_BETWEEN_NEGATIVE_ONE_QUARTER_AND_0_EXCL(y, x_in)\
\
    remainder = AE_SUB32(a_mod_quater_minus_q_1_by_4, inp);\
\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,-2, 1672461947, remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,-1, 1302514674, remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,0, 790015084,   remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,1, 290630308,   remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,2, 39332535,    remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,3, 720401,      remainder);\
    GEMMLOWP_EXP_BARREL_SHIFTER(y,4, 242,         remainder);\
\
    b = AE_EQ32(inp, z);\
    AE_MOVT32X2(y, AE_MOVDA32(Q31), b);\
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
    shift_amount = 26 + exponent;\
    scale = AE_SLAA32(ONE, shift_amount);\
\
    mask1 = AE_AND32(remainder1,  scale);\
    mask2 = AE_AND32(remainder2,  scale);\
\
    b1 = AE_LT32(z, mask1);\
    b2 = AE_LT32(z, mask2);\
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
    x_in1 = AE_SLAI32(a_mod_quater_minus_q_1_by_4_first, 5);\
\
    x2 = AE_AND32(inp2, mask_6fs);\
    a_mod_quater_minus_q_1_by_4_second = AE_SUB32(x2, q_1_by_4);\
    x_in2 = AE_SLAI32(a_mod_quater_minus_q_1_by_4_second, 5);\
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
    b = AE_EQ32(inp1, z);\
    AE_MOVT32X2(y1, AE_MOVDA32(Q31), b);\
\
    b = AE_EQ32(inp2, z);\
    AE_MOVT32X2(y2, AE_MOVDA32(Q31), b);\
}


WORD32 xa_nn_vec_softmax_asym8_asym8( UWORD8 * __restrict__ p_out,
                    const   UWORD8 * __restrict__ p_vec,
                            WORD32  diffmin,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length,
                            pVOID   p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

    int i;
    int shift_bits_reciprocal;
    xtbool2 f76, f54, f32, f10, g76, g54, g32, g10;
    UWORD8 *p_in = (UWORD8 *)p_vec;
    WORD32 *p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);
    ae_int32x2 y76, y54, y32, y10, x76, x54, x32, x10, diff_min, multiplier;
    ae_int32x2 dequantized_y76, dequantized_y54, dequantized_y32, dequantized_y10, dequantized_x76, dequantized_x54, dequantized_x32, dequantized_x10;
    ae_int32x2 unsat_out76, unsat_out54, unsat_out32, unsat_out10, ONE;
    ae_int32x2 exp_y76, exp_y54, exp_y32, exp_y10, exp_x76, exp_x54, exp_x32, exp_x10, sum_exp, recip_sum_exp;
    ae_int16x4 z76, z54, z32, z10;

    ae_int64 sum_exp_64;
    ae_valign align_src, align_dst;
    
    ae_int8x8 m0, m1, m2, m3, max;
    ae_int16x4 max_16;
    ae_valignx2 align_src_hf5, align_dst_hf5;
    /* Second operand for XOR instruction used in SUB_128 and ADD_128*/
    ae_int64 offset_xor = AE_MOVINT64_FROMINT8X8(AE_MOVDA8(128));

    align_src_hf5 = AE_LA128_PP((ae_int8x16 *)p_in);
    align_dst_hf5 = AE_ZALIGN128();

    ae_int32x2 z = AE_ZERO32();
    ae_int32x2 CT, CT_1_BY_3, CT_1_BY_8;
    ae_int32x2 mask_6fs, q_1_by_4;
    CT = AE_MOVDA32(CONSTANT_TERM);
    CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    mask_6fs = AE_MOVDA32(MASK);
    q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ONE = AE_MOVDA32(1);

    // Calculating Max
    {
        m0 = AE_MOVDA8(0x80);
        for(i=0; i<(vec_length >> 4); i++)
        {
          AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
          SUB_128(m1)
          SUB_128(m2)
          m0 = AE_MAX8(m0, m1);
          m0 = AE_MAX8(m0, m2);
        }

        align_src = AE_LA64_PP((ae_int8x8 *)p_in);

        for(i=0; i < ((vec_length & 15) >> 3); i++)
        {
          AE_LA8X8_IP(m1, align_src, (ae_int8x8 *)p_in);
          SUB_128(m1)
          m0 = AE_MAX8(m0, m1);
        }

        for(i=0; i < (vec_length & 7); i++)
        {
          AE_L8_IP(m1, (ae_int8 *)p_in, sizeof(ae_int8));
          SUB_128(m1)
          m0 = AE_MAX8(m0, m1);
        }

        if(vec_length < 8)
        {
          max = AE_MOVDA8((AE_MOVAD8(m0, 0) +  128));
          max_16 = AE_MOVDA16((AE_MOVAD8(m0, 0) +  128));
        }
        else
        {
          ae_int16x4 temp1, temp2;
          ae_int32x2 temp3, temp4;
          AE_CVTI16X4X2F8(temp1, temp2, m0, 0);
          temp2 = AE_MAX16(temp1, temp2);

          AE_CVTI32X4F16(temp3, temp4, temp2, 0);
          temp4 = AE_MAX32(temp3, temp4);

          temp3 = AE_SEL32_LH(temp4, temp4);
          temp3 = AE_MAX32(temp3, temp4);

          max = AE_MOVDA8((AE_MOVAD32_L(temp3) +  128));
          max_16 = AE_MOVDA16((AE_MOVAD32_L(temp3) +  128));
        }
    }

    diff_min = AE_MOVDA32(diffmin);
    multiplier = AE_MOVDA32(input_beta_multiplier);
    sum_exp = z; // setting to zero

    p_in = (UWORD8 *)p_vec;

    align_dst = AE_ZALIGN64(); // zero alignment reg

    align_src = AE_LA64_PP((ae_int8x8 *)p_in);

    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
      AE_SUBW8U(z76, z54, m1, max);
      AE_SUBW8U(z32, z10, m2, max);
      AE_CVTI32X4F16(y76, y54, z76, 0);
      AE_CVTI32X4F16(y32, y10, z54, 0);
      AE_CVTI32X4F16(x76, x54, z32, 0);
      AE_CVTI32X4F16(x32, x10, z10, 0);

      f76 = AE_LE32(diff_min, y76);
      f54 = AE_LE32(diff_min, y54);
      f32 = AE_LE32(diff_min, y32);
      f10 = AE_LE32(diff_min, y10);
      g76 = AE_LE32(diff_min, x76);
      g54 = AE_LE32(diff_min, x54);
      g32 = AE_LE32(diff_min, x32);
      g10 = AE_LE32(diff_min, x10);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_y76, exp_y54, dequantized_y76, dequantized_y54);
      AE_MOVF32X2(exp_y76, AE_ZERO32(), f76);

      AE_MOVF32X2(exp_y54, AE_ZERO32(), f54);
      AE_SA32X2X2_IP(exp_y76, exp_y54, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_y76 = AE_SRAA32RS(exp_y76, (int)12);
      exp_y54 = AE_SRAA32RS(exp_y54, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_y32, exp_y10, dequantized_y32, dequantized_y10);
      AE_MOVF32X2(exp_y32, AE_ZERO32(), f32);

      AE_MOVF32X2(exp_y10, AE_ZERO32(), f10);
      AE_SA32X2X2_IP(exp_y32, exp_y10, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
      exp_y10 = AE_SRAA32RS(exp_y10, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_x76, exp_x54, dequantized_x76, dequantized_x54);
      AE_MOVF32X2(exp_x76, AE_ZERO32(), g76);

      AE_MOVF32X2(exp_x54, AE_ZERO32(), g54);
      AE_SA32X2X2_IP(exp_x76, exp_x54, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_x76 = AE_SRAA32RS(exp_x76, (int)12);
      exp_x54 = AE_SRAA32RS(exp_x54, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_x32, exp_x10, dequantized_x32, dequantized_x10);
      AE_MOVF32X2(exp_x32, AE_ZERO32(), g32);

      AE_MOVF32X2(exp_x10, AE_ZERO32(), g10);
      AE_SA32X2X2_IP(exp_x32, exp_x10, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_x32 = AE_SRAA32RS(exp_x32, (int)12);
      exp_x10 = AE_SRAA32RS(exp_x10, (int)12);

      sum_exp = AE_ADD32S(sum_exp, exp_y76);
      sum_exp = AE_ADD32S(sum_exp, exp_y54);
      sum_exp = AE_ADD32S(sum_exp, exp_y32);
      sum_exp = AE_ADD32S(sum_exp, exp_y10);
      sum_exp = AE_ADD32S(sum_exp, exp_x76);
      sum_exp = AE_ADD32S(sum_exp, exp_x54);
      sum_exp = AE_ADD32S(sum_exp, exp_x32);
      sum_exp = AE_ADD32S(sum_exp, exp_x10);

    }
    sum_exp = AE_ADD32S_HL_LH(sum_exp, sum_exp);
    AE_SA128POS_FP(align_dst_hf5, p_exp); // finalize the stream
    
   // remainder loop
    for(i=0; i < (vec_length & 15); i++)
    {
        int rem_x;

        rem_x = (WORD32) *p_in++;
        rem_x = rem_x -  AE_MOVAD16_0(max_16);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MultiplyByQuantizedMultiplierGreaterThanOne(dequantized_y32, y32, multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, AE_ZERO32(), f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp_64 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(sum_exp), 32);
    recip_sum_exp = GetReciprocal(sum_exp_64, 12, &shift_bits_reciprocal);

    p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);

    for(i=0; i<(vec_length >> 3); i++)
    {
        AE_L32X2X2_IP(exp_y76, exp_y54, (ae_int32x4 *)p_exp, 4*sizeof(WORD32));
        AE_L32X2X2_IP(exp_y32, exp_y10, (ae_int32x4 *)p_exp, 4*sizeof(WORD32));

        AE_MULF2P32X4RAS(unsat_out76, unsat_out54, exp_y76, exp_y54, recip_sum_exp, recip_sum_exp);
        unsat_out76 = AE_SRAA32RS(unsat_out76, shift_bits_reciprocal + 31 - 8);
        unsat_out54 = AE_SRAA32RS(unsat_out54, shift_bits_reciprocal + 31 - 8);

        AE_MULF2P32X4RAS(unsat_out32, unsat_out10, exp_y32, exp_y10, recip_sum_exp, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);
        unsat_out10 = AE_SRAA32RS(unsat_out10, shift_bits_reciprocal + 31 - 8);

        m0 = AE_SATU8X4X32_L(unsat_out76, unsat_out54); 
        m1 = AE_SATU8X4X32_L(unsat_out32, unsat_out10);
        m2 = AE_SEL8X8I(m0, m1, 3);
        AE_SA8X8_IP(m2, align_dst, (ae_int8x8 *)p_out);
    }
    AE_SA64POS_FP(align_dst, p_out);

    // remainder loop
    __Pragma("no_unroll");
    for(i=0; i < (vec_length & 7); i++)
    {
        AE_L32_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);

        m3 = AE_SATU8X4X32_L(unsat_out32, unsat_out32);
        AE_S8_0_IP(m3, (ae_int8 *) p_out, 1);
    }

    return 0;
}

WORD32 xa_nn_vec_softmax_asym8s_asym8s( WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32  diffmin,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length,
                            pVOID   p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

    int i;
    int shift_bits_reciprocal;
    xtbool2 f76, f54, f32, f10, g76, g54, g32, g10;
    WORD8 *p_in = (WORD8 *)p_vec;
    WORD32 *p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);
    ae_int32x2 y76, y54, y32, y10, x76, x54, x32, x10, diff_min, multiplier;
    ae_int32x2 dequantized_y76, dequantized_y54, dequantized_y32, dequantized_y10, dequantized_x76, dequantized_x54, dequantized_x32, dequantized_x10;
    ae_int32x2 unsat_out76, unsat_out54, unsat_out32, unsat_out10, ONE;
    ae_int32x2 exp_y76, exp_y54, exp_y32, exp_y10, exp_x76, exp_x54, exp_x32, exp_x10, sum_exp, recip_sum_exp;
    ae_int16x4 z76, z54, z32, z10;

    ae_int64 sum_exp_64;
    ae_valign align_src, align_dst;
    
    ae_int8x8 m0, m1, m2, m3, max;
    ae_int16x4 max_16;
    ae_valignx2 align_src_hf5, align_dst_hf5;
    /* Second operand for XOR instruction used in SUB_128 and ADD_128*/
    ae_int64 offset_xor = AE_MOVINT64_FROMINT8X8(AE_MOVDA8(128));

    align_src_hf5 = AE_LA128_PP((ae_int8x16 *)p_in);
    align_dst_hf5 = AE_ZALIGN128();

    ae_int32x2 z = AE_ZERO32();
    ae_int32x2 CT, CT_1_BY_3, CT_1_BY_8;
    ae_int32x2 mask_6fs, q_1_by_4;
    CT = AE_MOVDA32(CONSTANT_TERM);
    CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    mask_6fs = AE_MOVDA32(MASK);
    q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ONE = AE_MOVDA32(1);

    // Calculating Max
    {
        m0 = AE_MOVDA8(0x80);
        for(i=0; i<(vec_length >> 4); i++)
        {
          AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
          m0 = AE_MAX8(m0, m1);
          m0 = AE_MAX8(m0, m2);
        }

        align_src = AE_LA64_PP((ae_int8x8 *)p_in);

        for(i=0; i < ((vec_length & 15) >> 3); i++)
        {
          AE_LA8X8_IP(m1, align_src, (ae_int8x8 *)p_in);
          m0 = AE_MAX8(m0, m1);
        }

        for(i=0; i < (vec_length & 7); i++)
        {
          AE_L8_IP(m1, (ae_int8 *)p_in, sizeof(ae_int8));
          m0 = AE_MAX8(m0, m1);
        }

        if(vec_length < 8)
        {
          max = AE_MOVDA8((AE_MOVAD8(m0, 0)));
          max_16 = AE_MOVDA16((AE_MOVAD8(m0, 0)));
        }
        else
        {
          ae_int16x4 temp1, temp2;
          ae_int32x2 temp3, temp4;
          AE_CVTI16X4X2F8(temp1, temp2, m0, 0);
          temp2 = AE_MAX16(temp1, temp2);

          AE_CVTI32X4F16(temp3, temp4, temp2, 0);
          temp4 = AE_MAX32(temp3, temp4);

          temp3 = AE_SEL32_LH(temp4, temp4);
          temp3 = AE_MAX32(temp3, temp4);

          max = AE_MOVDA8((AE_MOVAD32_L(temp3)));
          max_16 = AE_MOVDA16((AE_MOVAD32_L(temp3)));
        }
    }

    diff_min = AE_MOVDA32(diffmin);
    multiplier = AE_MOVDA32(input_beta_multiplier);
    sum_exp = z; // setting to zero

    p_in = (WORD8 *)p_vec;

    align_dst = AE_ZALIGN64(); // zero alignment reg

    align_src = AE_LA64_PP((ae_int8x8 *)p_in);

    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
      AE_SUBW8(z76, z54, m1, max);
      AE_SUBW8(z32, z10, m2, max);
      AE_CVTI32X4F16(y76, y54, z76, 0);
      AE_CVTI32X4F16(y32, y10, z54, 0);
      AE_CVTI32X4F16(x76, x54, z32, 0);
      AE_CVTI32X4F16(x32, x10, z10, 0);

      f76 = AE_LE32(diff_min, y76);
      f54 = AE_LE32(diff_min, y54);
      f32 = AE_LE32(diff_min, y32);
      f10 = AE_LE32(diff_min, y10);
      g76 = AE_LE32(diff_min, x76);
      g54 = AE_LE32(diff_min, x54);
      g32 = AE_LE32(diff_min, x32);
      g10 = AE_LE32(diff_min, x10);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_y76, exp_y54, dequantized_y76, dequantized_y54);
      AE_MOVF32X2(exp_y76, AE_ZERO32(), f76);

      AE_MOVF32X2(exp_y54, AE_ZERO32(), f54);
      AE_SA32X2X2_IP(exp_y76, exp_y54, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_y76 = AE_SRAA32RS(exp_y76, (int)12);
      exp_y54 = AE_SRAA32RS(exp_y54, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_y32, exp_y10, dequantized_y32, dequantized_y10);
      AE_MOVF32X2(exp_y32, AE_ZERO32(), f32);

      AE_MOVF32X2(exp_y10, AE_ZERO32(), f10);
      AE_SA32X2X2_IP(exp_y32, exp_y10, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
      exp_y10 = AE_SRAA32RS(exp_y10, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_x76, exp_x54, dequantized_x76, dequantized_x54);
      AE_MOVF32X2(exp_x76, AE_ZERO32(), g76);

      AE_MOVF32X2(exp_x54, AE_ZERO32(), g54);
      AE_SA32X2X2_IP(exp_x76, exp_x54, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_x76 = AE_SRAA32RS(exp_x76, (int)12);
      exp_x54 = AE_SRAA32RS(exp_x54, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_x32, exp_x10, dequantized_x32, dequantized_x10);
      AE_MOVF32X2(exp_x32, AE_ZERO32(), g32);

      AE_MOVF32X2(exp_x10, AE_ZERO32(), g10);
      AE_SA32X2X2_IP(exp_x32, exp_x10, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_x32 = AE_SRAA32RS(exp_x32, (int)12);
      exp_x10 = AE_SRAA32RS(exp_x10, (int)12);

      sum_exp = AE_ADD32S(sum_exp, exp_y76);
      sum_exp = AE_ADD32S(sum_exp, exp_y54);
      sum_exp = AE_ADD32S(sum_exp, exp_y32);
      sum_exp = AE_ADD32S(sum_exp, exp_y10);
      sum_exp = AE_ADD32S(sum_exp, exp_x76);
      sum_exp = AE_ADD32S(sum_exp, exp_x54);
      sum_exp = AE_ADD32S(sum_exp, exp_x32);
      sum_exp = AE_ADD32S(sum_exp, exp_x10);

    }
    sum_exp = AE_ADD32S_HL_LH(sum_exp, sum_exp);
    AE_SA128POS_FP(align_dst_hf5, p_exp); // finalize the stream
   
   // remainder loop
    for(i=0; i < (vec_length & 15); i++)
    {
        int rem_x;

        rem_x = (WORD32) *p_in++;
        rem_x = rem_x -  AE_MOVAD16_0(max_16);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MultiplyByQuantizedMultiplierGreaterThanOne(dequantized_y32, y32, multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, AE_ZERO32(), f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp_64 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(sum_exp), 32);
    recip_sum_exp = GetReciprocal(sum_exp_64, 12, &shift_bits_reciprocal);

    p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);

    for(i=0; i<(vec_length >> 3); i++)
    {
        AE_L32X2X2_IP(exp_y76, exp_y54, (ae_int32x4 *)p_exp, 4*sizeof(WORD32));
        AE_L32X2X2_IP(exp_y32, exp_y10, (ae_int32x4 *)p_exp, 4*sizeof(WORD32));

        AE_MULF2P32X4RAS(unsat_out76, unsat_out54, exp_y76, exp_y54, recip_sum_exp, recip_sum_exp);
        unsat_out76 = AE_SRAA32RS(unsat_out76, shift_bits_reciprocal + 31 - 8);
        unsat_out54 = AE_SRAA32RS(unsat_out54, shift_bits_reciprocal + 31 - 8);

        AE_MULF2P32X4RAS(unsat_out32, unsat_out10, exp_y32, exp_y10, recip_sum_exp, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);
        unsat_out10 = AE_SRAA32RS(unsat_out10, shift_bits_reciprocal + 31 - 8);

        m0 = AE_SATU8X4X32_L(unsat_out76, unsat_out54); 
        m1 = AE_SATU8X4X32_L(unsat_out32, unsat_out10);
        m2 = AE_SEL8X8I(m0, m1, 3);
        SUB_128(m2)
        AE_SA8X8_IP(m2, align_dst, (ae_int8x8 *)p_out);
    }
    AE_SA64POS_FP(align_dst, p_out);

    // remainder loop
    __Pragma("no_unroll");
    for(i=0; i < (vec_length & 7); i++)
    {
        AE_L32_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 8);

        m3 = AE_SATU8X4X32_L(unsat_out32, unsat_out32);
        SUB_128(m3)
        AE_S8_0_IP(m3, (ae_int8 *) p_out, 1);
    }

    return 0;
}

WORD32 xa_nn_vec_softmax_asym8s_16( WORD16 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_vec,
                            WORD32  diffmin,
                            WORD32  input_beta_left_shift,
                            WORD32  input_beta_multiplier,
                            WORD32  vec_length,
                            pVOID   p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

    int i;
    int shift_bits_reciprocal;
    xtbool2 f76, f54, f32, f10, g76, g54, g32, g10;
    WORD8 *p_in = (WORD8 *)p_vec;
    WORD32 *p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);
    ae_int32x2 y76, y54, y32, y10, x76, x54, x32, x10, diff_min, multiplier;
    ae_int32x2 dequantized_y76, dequantized_y54, dequantized_y32, dequantized_y10, dequantized_x76, dequantized_x54, dequantized_x32, dequantized_x10;
    ae_int32x2 unsat_out76, unsat_out54, unsat_out32, unsat_out10, ONE;
    ae_int32x2 exp_y76, exp_y54, exp_y32, exp_y10, exp_x76, exp_x54, exp_x32, exp_x10, sum_exp, recip_sum_exp;
    ae_int16x4 z76, z54, z32, z10;

    ae_int64 sum_exp_64;
    ae_valign align_src, align_dst;
    
    ae_int8x8 m0, m1, m2, max;
    ae_int16x4 n0, n1, n2, max_16;

    ae_valignx2 align_src_hf5, align_dst_hf5;
    /* Second operand for XOR instruction used in SUB_32768 and ADD_32768*/
    ae_int64 offset_xor_16 = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(32768));

    align_src_hf5 = AE_LA128_PP((ae_int8x16 *)p_in);
    align_dst_hf5 = AE_ZALIGN128();

    ae_int32x2 z = AE_ZERO32();
    ae_int32x2 CT, CT_1_BY_3, CT_1_BY_8;
    ae_int32x2 mask_6fs, q_1_by_4;
    CT = AE_MOVDA32(CONSTANT_TERM);
    CT_1_BY_3 = AE_MOVDA32(CONSTANT_1_OVER_3);
    CT_1_BY_8 = AE_MOVDA32(CONSTANT_1_OVER_8);
    mask_6fs = AE_MOVDA32(MASK);
    q_1_by_4 = AE_MOVDA32(ONE_QUATER_Q26);
    ONE = AE_MOVDA32(1);

    // Calculating Max
    {
        m0 = AE_MOVDA8(0x80);
        for(i=0; i<(vec_length >> 4); i++)
        {
          AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
          m0 = AE_MAX8(m0, m1);
          m0 = AE_MAX8(m0, m2);
        }

        align_src = AE_LA64_PP((ae_int8x8 *)p_in);

        for(i=0; i < ((vec_length & 15) >> 3); i++)
        {
          AE_LA8X8_IP(m1, align_src, (ae_int8x8 *)p_in);
          m0 = AE_MAX8(m0, m1);
        }

        for(i=0; i < (vec_length & 7); i++)
        {
          AE_L8_IP(m1, (ae_int8 *)p_in, sizeof(ae_int8));
          m0 = AE_MAX8(m0, m1);
        }

        if(vec_length < 8)
        {
          max = AE_MOVDA8((AE_MOVAD8(m0, 0)));
          max_16 = AE_MOVDA16((AE_MOVAD8(m0, 0)));
        }
        else
        {
          ae_int16x4 temp1, temp2;
          ae_int32x2 temp3, temp4;
          AE_CVTI16X4X2F8(temp1, temp2, m0, 0);
          temp2 = AE_MAX16(temp1, temp2);

          AE_CVTI32X4F16(temp3, temp4, temp2, 0);
          temp4 = AE_MAX32(temp3, temp4);

          temp3 = AE_SEL32_LH(temp4, temp4);
          temp3 = AE_MAX32(temp3, temp4);

          max = AE_MOVDA8((AE_MOVAD32_L(temp3)));
          max_16 = AE_MOVDA16((AE_MOVAD32_L(temp3)));
        }
    }

    diff_min = AE_MOVDA32(diffmin);
    multiplier = AE_MOVDA32(input_beta_multiplier);
    sum_exp = z; // setting to zero

    p_in = (WORD8 *)p_vec;

    align_dst = AE_ZALIGN64(); // zero alignment reg

    align_src = AE_LA64_PP((ae_int8x8 *)p_in);

    for(i=0; i<(vec_length >> 4); i++)
    {
      AE_LA8X8X2_IP(m1, m2, align_src_hf5, (ae_int8x16 *)p_in);
      AE_SUBW8(z76, z54, m1, max);
      AE_SUBW8(z32, z10, m2, max);
      AE_CVTI32X4F16(y76, y54, z76, 0);
      AE_CVTI32X4F16(y32, y10, z54, 0);
      AE_CVTI32X4F16(x76, x54, z32, 0);
      AE_CVTI32X4F16(x32, x10, z10, 0);

      f76 = AE_LE32(diff_min, y76);
      f54 = AE_LE32(diff_min, y54);
      f32 = AE_LE32(diff_min, y32);
      f10 = AE_LE32(diff_min, y10);
      g76 = AE_LE32(diff_min, x76);
      g54 = AE_LE32(diff_min, x54);
      g32 = AE_LE32(diff_min, x32);
      g10 = AE_LE32(diff_min, x10);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y76, dequantized_y54, y76, y54, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_y76, exp_y54, dequantized_y76, dequantized_y54);
      AE_MOVF32X2(exp_y76, AE_ZERO32(), f76);

      AE_MOVF32X2(exp_y54, AE_ZERO32(), f54);
      AE_SA32X2X2_IP(exp_y76, exp_y54, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_y76 = AE_SRAA32RS(exp_y76, (int)12);
      exp_y54 = AE_SRAA32RS(exp_y54, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_y32, dequantized_y10, y32, y10, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_y32, exp_y10, dequantized_y32, dequantized_y10);
      AE_MOVF32X2(exp_y32, AE_ZERO32(), f32);

      AE_MOVF32X2(exp_y10, AE_ZERO32(), f10);
      AE_SA32X2X2_IP(exp_y32, exp_y10, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
      exp_y10 = AE_SRAA32RS(exp_y10, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x76, dequantized_x54, x76, x54, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_x76, exp_x54, dequantized_x76, dequantized_x54);
      AE_MOVF32X2(exp_x76, AE_ZERO32(), g76);

      AE_MOVF32X2(exp_x54, AE_ZERO32(), g54);
      AE_SA32X2X2_IP(exp_x76, exp_x54, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_x76 = AE_SRAA32RS(exp_x76, (int)12);
      exp_x54 = AE_SRAA32RS(exp_x54, (int)12);

      MultiplyByQuantizedMultiplierGreaterThanOneX2(dequantized_x32, dequantized_x10, x32, x10, multiplier, input_beta_left_shift)
      EXP_Q26X2(exp_x32, exp_x10, dequantized_x32, dequantized_x10);
      AE_MOVF32X2(exp_x32, AE_ZERO32(), g32);

      AE_MOVF32X2(exp_x10, AE_ZERO32(), g10);
      AE_SA32X2X2_IP(exp_x32, exp_x10, align_dst_hf5, (ae_int32x4 *)p_exp);
      exp_x32 = AE_SRAA32RS(exp_x32, (int)12);
      exp_x10 = AE_SRAA32RS(exp_x10, (int)12);

      sum_exp = AE_ADD32S(sum_exp, exp_y76);
      sum_exp = AE_ADD32S(sum_exp, exp_y54);
      sum_exp = AE_ADD32S(sum_exp, exp_y32);
      sum_exp = AE_ADD32S(sum_exp, exp_y10);
      sum_exp = AE_ADD32S(sum_exp, exp_x76);
      sum_exp = AE_ADD32S(sum_exp, exp_x54);
      sum_exp = AE_ADD32S(sum_exp, exp_x32);
      sum_exp = AE_ADD32S(sum_exp, exp_x10);

    }
    sum_exp = AE_ADD32S_HL_LH(sum_exp, sum_exp);
    AE_SA128POS_FP(align_dst_hf5, p_exp); // finalize the stream
   
   // remainder loop
    for(i=0; i < (vec_length & 15); i++)
    {
        int rem_x;

        rem_x = (WORD32) *p_in++;
        rem_x = rem_x -  AE_MOVAD16_0(max_16);
        y32 = AE_MOVDA32(rem_x);
        f32 = AE_LE32(diff_min, y32);

        MultiplyByQuantizedMultiplierGreaterThanOne(dequantized_y32, y32, multiplier, input_beta_left_shift)
        EXP_Q26(exp_y32, dequantized_y32);
        AE_MOVF32X2(exp_y32, AE_ZERO32(), f32);
        AE_S32_L_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        exp_y32 = AE_SRAA32RS(exp_y32, (int)12);
        sum_exp = AE_ADD32S(sum_exp, exp_y32);
    }

    sum_exp_64 = AE_SRAI64(AE_MOVINT64_FROMINT32X2(sum_exp), 32);
    recip_sum_exp = GetReciprocal(sum_exp_64, 12, &shift_bits_reciprocal);

    p_exp = (WORD32 *)ALIGN_PTR(p_scratch, ALIGNMENT);

    for(i=0; i<(vec_length >> 3); i++)
    {
        AE_L32X2X2_IP(exp_y76, exp_y54, (ae_int32x4 *)p_exp, 4*sizeof(WORD32));
        AE_L32X2X2_IP(exp_y32, exp_y10, (ae_int32x4 *)p_exp, 4*sizeof(WORD32));

        AE_MULF2P32X4RAS(unsat_out76, unsat_out54, exp_y76, exp_y54, recip_sum_exp, recip_sum_exp);
        unsat_out76 = AE_SRAA32RS(unsat_out76, shift_bits_reciprocal + 31 - 16);
        unsat_out54 = AE_SRAA32RS(unsat_out54, shift_bits_reciprocal + 31 - 16);

        AE_MULF2P32X4RAS(unsat_out32, unsat_out10, exp_y32, exp_y10, recip_sum_exp, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 16);
        unsat_out10 = AE_SRAA32RS(unsat_out10, shift_bits_reciprocal + 31 - 16);

        n0 = AE_SATU16X4(unsat_out76, unsat_out54); 
        n1 = AE_SATU16X4(unsat_out32, unsat_out10);
        SUB_32768(n0)
        SUB_32768(n1)
        AE_SA16X4X2_IP(n0, n1, align_dst_hf5, (ae_int16x8 *)p_out);
    }
    AE_SA128POS_FP(align_dst_hf5, p_out); // finalize the stream

    // remainder loop
    __Pragma("no_unroll");
    for(i=0; i < (vec_length & 7); i++)
    {
        AE_L32_IP(exp_y32, (ae_int32 *)p_exp, sizeof(WORD32));

        unsat_out32 = AE_MULFP32X2RAS(exp_y32, recip_sum_exp);
        unsat_out32 = AE_SRAA32RS(unsat_out32, shift_bits_reciprocal + 31 - 16);

        n2 = AE_SATU16X4(unsat_out32, unsat_out32);
        SUB_32768(n2)
        AE_S16_0_IP(n2, (ae_int16 *) p_out, 2);
    }

    return 0;
}
int get_softmax_scratch_size(int inp_precision, int out_precision, int length)
{
    int size_of_one_elm_in_bytes, total_bytes;
    (void) out_precision;

    /* This function returns scratch size required by softmax implementation in bytes
       scratch memory is needed to save exponents of inputs computed in the function,
       every exponent is computed as 32 bit (4 bytes) number currently*/
    switch(inp_precision)
    {
        case 8:
            size_of_one_elm_in_bytes = 4;
            break;
        case 16:
            size_of_one_elm_in_bytes = 4;
            break;
        case 32:
            size_of_one_elm_in_bytes = 4;
            break;
        case -1:
            size_of_one_elm_in_bytes = 4;
            break;
        case -3:
            size_of_one_elm_in_bytes = 4;
            break;
        case -4:
            size_of_one_elm_in_bytes = 4;
            break;
    }

    total_bytes = size_of_one_elm_in_bytes*length;
    total_bytes = ALIGNED_SIZE(total_bytes, ALIGNMENT);

    return total_bytes;
}










