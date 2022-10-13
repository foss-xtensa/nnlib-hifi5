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
#ifndef __XA_NNLIB_QUANT_MACROS_HIFI5_H__
#define __XA_NNLIB_QUANT_MACROS_HIFI5_H__

#if TFLITE_SINGLE_ROUNDING

#define MPY_BY_QUANT_MULT_X2_OUT32(out, inp, multiplier, left_shift, right_shift) \
{ \
  ae_int64 out64_0, out64_1; \
  ae_int64 INT64_ONE = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0,1)); \
  ae_int64 round_val = AE_SLAA64S(INT64_ONE, 30 - left_shift); \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp, AE_MOVDA32(multiplier)); \
  out64_0 = AE_ADD64S(out64_0, round_val); \
  out64_1 = AE_ADD64S(out64_1, round_val); \
  out = AE_TRUNCA32X2F64S(out64_0, out64_1, 1 + left_shift); \
}

#define MPY_BY_QUANT_MULT_SLS_X2_OUT32(out, inp, multiplier, left_shift, right_shift) \
  MPY_BY_QUANT_MULT_X2_OUT32(out, inp, multiplier, left_shift, right_shift)

#define MPY_BY_QUANT_MULT_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) \
{ \
  ae_int64 out64_0, out64_1, out64_2, out64_3; \
  ae_int64 INT64_ONE = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0,1)); \
  ae_int64 round_val = AE_SLAA64S(INT64_ONE, 30 - left_shift); \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp1, AE_MOVDA32(multiplier)); \
  AE_MUL32X2S_HH_LL(out64_2, out64_3, inp2, AE_MOVDA32(multiplier)); \
  out64_0 = AE_ADD64S(out64_0, round_val); \
  out64_1 = AE_ADD64S(out64_1, round_val); \
  out64_2 = AE_ADD64S(out64_2, round_val); \
  out64_3 = AE_ADD64S(out64_3, round_val); \
  out1 = AE_TRUNCA32X2F64S(out64_0, out64_1, 1 + left_shift); \
  out2 = AE_TRUNCA32X2F64S(out64_2, out64_3, 1 + left_shift); \
}

#define MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) \
  MPY_BY_QUANT_MULT_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, left_shift, right_shift)

#define MPY_BY_QUANT_MULT_X2_OUT16(out, inp, multiplier, left_shift, right_shift) \
{ \
  ae_int64 out64_0, out64_1; \
  ae_int32x2 out32_0; \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp, AE_MOVDA32(multiplier)); \
  out32_0 = AE_TRUNCA32X2F64S(out64_0, out64_1, left_shift + 17); \
  out = AE_ROUND16X4F32SASYM(out32_0, out32_0); \
}

#define MPY_BY_QUANT_MULT_X2X2_OUT16(out, inp1, inp2, multiplier, l_shift, r_shift) \
{ \
  ae_int64 out64_0, out64_1, out64_2, out64_3; \
  ae_int32x2 out32_0, out32_1; \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp1, AE_MOVDA32(multiplier)); \
  AE_MUL32X2S_HH_LL(out64_2, out64_3, inp2, AE_MOVDA32(multiplier)); \
  out32_0 = AE_TRUNCA32X2F64S(out64_0, out64_1, l_shift + 17); \
  out32_1 = AE_TRUNCA32X2F64S(out64_2, out64_3, l_shift + 17); \
  out = AE_ROUND16X4F32SASYM(out32_0, out32_1); \
}

#define MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out, inp1, inp2, multiplier, l_shift, r_shift, out_off) \
{ \
  MPY_BY_QUANT_MULT_X2X2_OUT16(out, inp1, inp2, multiplier, l_shift, r_shift) \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
}

#define MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB MPY_BY_QUANT_MULT_X2X2_OUT16_ZB

#ifdef AE_TRUNCAV32X2F64S
#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV(out, inp1, inp2, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
{ \
  ae_int64 out64_0, out64_1, out64_2, out64_3; \
  ae_int32x2 out32_0, out32_1; \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp1, mult_01); \
  AE_MUL32X2S_HH_LL(out64_2, out64_3, inp2, mult_23); \
  out32_0 = AE_TRUNCAV32X2F64S(out64_0, out64_1, AE_MOVAD32_H(ls_01)); \
  out32_1 = AE_TRUNCAV32X2F64S(out64_2, out64_3, AE_MOVAD32_L(ls_01)); \
  out = AE_ROUND16X4F32SASYM(out32_0, out32_1); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
}
#else
#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB
#endif

#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
{ \
  ae_int64 out64_0, out64_1, out64_2, out64_3; \
  ae_int32x2 out32_0, out32_1, out32_2, out32_3; \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp1, mult_01); \
  AE_MUL32X2S_HH_LL(out64_2, out64_3, inp2, mult_23); \
  out32_0 = AE_TRUNCA32F64S(out64_0, AE_MOVAD32_H(ls_01) + 17); \
  out32_1 = AE_TRUNCA32F64S(out64_1, AE_MOVAD32_L(ls_01) + 17); \
  out32_2 = AE_TRUNCA32F64S(out64_2, AE_MOVAD32_H(ls_23) + 17); \
  out32_3 = AE_TRUNCA32F64S(out64_3, AE_MOVAD32_L(ls_23) + 17); \
  out32_0 = AE_SEL32_LL(out32_0, out32_1); \
  out32_1 = AE_SEL32_LL(out32_2, out32_3); \
  out = AE_ROUND16X4F32SASYM(out32_0, out32_1); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
}

#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_X2_OUT16_ZB(out, out_a, inp1, inp2, inp1_a, inp2_a, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
{ \
  ae_int64 out64_0, out64_1, out64_2, out64_3; \
  ae_int64 out64_0_a, out64_1_a, out64_2_a, out64_3_a; \
  ae_int32x2 out32_0, out32_1, out32_2, out32_3; \
  AE_MUL32X2S_HH_LL(out64_0, out64_1, inp1, mult_01); \
  AE_MUL32X2S_HH_LL(out64_2, out64_3, inp2, mult_23); \
  AE_MUL32X2S_HH_LL(out64_0_a, out64_1_a, inp1_a, mult_01); \
  AE_MUL32X2S_HH_LL(out64_2_a, out64_3_a, inp2_a, mult_23); \
  out32_0 = AE_TRUNCA32X2F64S(out64_0, out64_0_a, AE_MOVAD32_H(ls_01) + 17); \
  out32_1 = AE_TRUNCA32X2F64S(out64_1, out64_1_a, AE_MOVAD32_L(ls_01) + 17); \
  out32_2 = AE_TRUNCA32X2F64S(out64_2, out64_2_a, AE_MOVAD32_H(ls_23) + 17); \
  out32_3 = AE_TRUNCA32X2F64S(out64_3, out64_3_a, AE_MOVAD32_L(ls_23) + 17); \
  out = AE_ROUND16X4F32SASYM(AE_SEL32_HH(out32_0, out32_1), AE_SEL32_HH(out32_2, out32_3)); \
  out_a = AE_ROUND16X4F32SASYM(AE_SEL32_LL(out32_0, out32_1), AE_SEL32_LL(out32_2, out32_3)); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
  out_a = AE_ADD16S(AE_MOVDA16(out_off), out_a); \
}

#define MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) \
  MPY_BY_QUANT_MULT_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) 

#define MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB_AV(out, inp1, inp2, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV(out, inp1, inp2, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off)

#define MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off)

#define MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out, out_a, inp1, inp2, inp1_a, inp2_a, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_X2X2_X2_OUT16_ZB(out, out_a, inp1, inp2, inp1_a, inp2_a, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off)

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod, val, multiplier, lsh) \
{ \
  ae_int32x2 mult_ls0, mult_ls1; \
  xtbool2 b0 = AE_EQ32(AE_MOVDA32(lsh), AE_ZERO32()); \
  mult_ls0 = mult_ls1 = AE_MOVDA32(multiplier); \
  AE_MOVF32X2(mult_ls0, AE_ZERO32(), b0); \
  AE_MOVT32X2(mult_ls1, AE_ZERO32(), b0); \
  prod = AE_MULFP32X2RAS(val, mult_ls0); \
  AE_MULAFP32X2TS(prod, val, mult_ls1); \
  prod = AE_SRAA32RS(prod, -lsh); \
}

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32(prod0, prod1, val0, val1, multiplier, lsh) \
{ \
  ae_int32x2 mult_ls0, mult_ls1; \
  xtbool2 b0 = AE_EQ32(AE_MOVDA32(lsh), AE_ZERO32()); \
  mult_ls0 = mult_ls1 = AE_MOVDA32(multiplier); \
  AE_MOVF32X2(mult_ls0, AE_ZERO32(), b0); \
  AE_MOVT32X2(mult_ls1, AE_ZERO32(), b0); \
  AE_MULF2P32X4RAS(prod0, prod1, val0, val1, mult_ls0, mult_ls0); \
  AE_MULAFP32X2TS(prod0, val0, mult_ls1); \
  AE_MULAFP32X2TS(prod1, val1, mult_ls1); \
  prod0 = AE_SRAA32RS(prod0, -lsh); \
  prod1 = AE_SRAA32RS(prod1, -lsh); \
}

#define MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(acc0, acc1, val0, val1, multiplier, lsh) \
{ \
  ae_int32x2 mult_ls0, mult_ls1; \
  ae_int32x2 out0, out1; \
  xtbool2 b0 = AE_EQ32(AE_MOVDA32(lsh), AE_ZERO32()); \
  mult_ls0 = mult_ls1 = AE_MOVDA32(multiplier); \
  AE_MOVF32X2(mult_ls0, AE_ZERO32(), b0); \
  AE_MOVT32X2(mult_ls1, AE_ZERO32(), b0); \
  out0 = AE_MULFP32X2TS(val0, mult_ls1); \
  out1 = AE_MULFP32X2TS(val1, mult_ls1); \
  AE_MULSF2P32X4RAS(out0, out1, val0, val1, mult_ls0, mult_ls0); \
  AE_MULAF2P32X4RAS(acc0, acc1, out0, out1, AE_SRLA32(AE_MOVDA32(0x80000000), -lsh), AE_SRLA32(AE_MOVDA32(0x80000000), -lsh)); \
}

#define MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(acc0, acc1, val0, val1, multiplier, lsh) \
{ \
  ae_int32x2 mult_ls0, mult_ls1; \
  ae_int32x2 out0, out1; \
  xtbool2 b0 = AE_EQ32(AE_MOVDA32(lsh), AE_ZERO32()); \
  mult_ls0 = mult_ls1 = AE_MOVDA32(multiplier); \
  AE_MOVF32X2(mult_ls0, AE_ZERO32(), b0); \
  AE_MOVT32X2(mult_ls1, AE_ZERO32(), b0); \
  out0 = AE_MULFP32X2TS(val0, mult_ls1); \
  out1 = AE_MULFP32X2TS(val1, mult_ls1); \
  AE_MULSF2P32X4RAS(out0, out1, val0, val1, mult_ls0, mult_ls0); \
  AE_MULSF2P32X4RAS(acc0, acc1, out0, out1, AE_SRLA32(AE_MOVDA32(0x80000000), -lsh), AE_SRLA32(AE_MOVDA32(0x80000000), -lsh)); \
}

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(prod0, prod1, val0, val1, multiplier, lsh, out_off) \
    prod0 = prod1 = AE_MOVDA32(out_off); \
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(prod0, prod1, val0, val1, multiplier, lsh)

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(prod1, prod2, val1, val2, multiplier1, multiplier2, lsh1, lsh2) \
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod1, val1, multiplier1, lsh1) \
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod2, val2, multiplier2, lsh2)

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, val0, val1, multiplier, lsh, out_off) \
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out0, val0, val1, multiplier, lsh, lsh, out_off)

#define MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(y, x, multiplier, lsh) \
    MPY_BY_QUANT_MULT_X2_OUT32(y, x, multiplier, lsh, lsh)

#define MPY_BY_QUANT_MULT_GT_ONE_X2X2_OUT32(y, z, l, m, multiplier, lsh) \
    MPY_BY_QUANT_MULT_X2X2_OUT32(y, z, l, m, multiplier, lsh, lsh)

#else /* #if TFLITE_SINGLE_ROUNDING */

#define MPY_BY_QUANT_MULT_X2_OUT32(out, inp, multiplier, left_shift, right_shift) \
  out = AE_SLAA32(inp, left_shift); \
  out = AE_MULFP32X2RAS(out, AE_MOVDA32(multiplier)); \
  out = AE_SRAA32SYMS(out, right_shift);

#define MPY_BY_QUANT_MULT_SLS_X2_OUT32(out, inp, multiplier, left_shift, right_shift) \
  out = AE_SLAA32S(inp, left_shift); \
  out = AE_MULFP32X2RAS(out, AE_MOVDA32(multiplier)); \
  out = AE_SRAA32SYMS(out, right_shift);

#define MPY_BY_QUANT_MULT_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) \
{ \
  ae_int32x2 d_ls = AE_MOVDA32(1<<left_shift); \
  AE_MUL2P32X4(out1, out2, inp1, inp2, d_ls, d_ls); \
  AE_MULF2P32X4RAS(out1, out2, out1, out2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  out1 = AE_SRAA32SYMS(out1, right_shift); \
  out2 = AE_SRAA32SYMS(out2, right_shift); \
}

#define MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) \
{ \
  ae_int32x2 d_ls = AE_MOVDA32(1<<left_shift); \
  AE_MUL2P32X4S(out1, out2, inp1, inp2, d_ls, d_ls); \
  AE_MULF2P32X4RAS(out1, out2, out1, out2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  out1 = AE_SRAA32SYMS(out1, right_shift); \
  out2 = AE_SRAA32SYMS(out2, right_shift); \
}

#define MPY_BY_QUANT_MULT_X2_OUT16(out, inp, multiplier, left_shift, right_shift) \
{ \
  ae_int32x2 out32_0; \
  out32_0 = AE_SLAA32(inp, left_shift); \
  out32_0 = AE_MULFP32X2RAS(out32_0, AE_MOVDA32(multiplier)); \
  out32_0 = AE_SRAA32SYMS(out32_0, right_shift); \
  out = AE_SAT16X4(out32_0, out32_0); \
}

#define MPY_BY_QUANT_MULT_X2X2_OUT16(out, inp1, inp2, multiplier, l_shift, r_shift) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, AE_MOVDA32(1 << l_shift), AE_MOVDA32(1 << l_shift)); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_NEG32(AE_MOVDA32(multiplier)), AE_NEG32(AE_MOVDA32(multiplier))); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, AE_SRAA32(AE_MOVDA32(0x80000000), r_shift), AE_SRAA32(AE_MOVDA32(0x80000000), r_shift)); \
  out = AE_SAT16X4(inp1, inp2); \
}

#define MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out, inp1, inp2, multiplier, l_shift, r_shift, out_off) \
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, AE_SLAA32(AE_MOVI(1), l_shift), AE_SLAA32(AE_MOVI(1), l_shift)); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, AE_SRAA32S(AE_MOVDA32(0x80000000), r_shift), AE_SRAA32S(AE_MOVDA32(0x80000000), r_shift));      \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_SUB16S(AE_MOVDA16(out_off), out);

#define MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out, inp1, inp2, multiplier, l_shift, r_shift, out_off) \
  inp1 = AE_SLAA32S(inp1, l_shift); \
  inp2 = AE_SLAA32S(inp2, l_shift); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, AE_SRAA32S(AE_MOVDA32(0x80000000), r_shift), AE_SRAA32S(AE_MOVDA32(0x80000000), r_shift));      \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_SUB16S(AE_MOVDA16(out_off), out);

#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB_AV MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB

#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, AE_SRAV32RS(AE_MOVI(1), AE_NEG32(ls_01)), AE_SRAV32RS(AE_MOVI(1), AE_NEG32(ls_23))); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, mult_01, mult_23); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, AE_SRAV32RS(AE_MOVDA32(0x80000000), rs_01), AE_SRAV32RS(AE_MOVDA32(0x80000000), rs_23)); \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_SUB16S(AE_MOVDA16(out_off), out); \
}

#define MPY_BY_QUANT_MULT_PER_CHAN_X2X2_X2_OUT16_ZB(out, out_a, inp1, inp2, inp1_a, inp2_a, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_X2X2_OUT16_ZB(out_a, inp1_a, inp2_a, mult_01, mult_23, ls_01, ls_23, rs_01, rs_23, out_off)

#define MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB_AV MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB

#define MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, ls_mult_01, ls_mult_23); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, mult_01, mult_23); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, rs_mult_01, rs_mult_23); \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_SUB16S(AE_MOVDA16(out_off), out); \
}

#define MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_X2_OUT16_ZB(out, out_a, inp1, inp2, inp1_a, inp2_a, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out, inp1, inp2, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off) \
  MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_a, inp1_a, inp2_a, mult_01, mult_23, ls_mult_01, ls_mult_23, rs_mult_01, rs_mult_23, out_off)

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(prod, val, multiplier, lsh) {\
    prod = AE_MULFP32X2RAS(val, AE_MOVDA32(multiplier));\
    prod = AE_SRAA32SYMS(prod, -lsh);\
}

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32(prod0, prod1, val0, val1, multiplier, lsh) { \
    AE_MULF2P32X4RAS(prod0, prod1, val0, val1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    prod0 = AE_SRAA32SYMS(prod0, -lsh); \
    prod1 = AE_SRAA32SYMS(prod1, -lsh); \
}

#define MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(acc0, acc1, val0, val1, multiplier, lsh) { \
    AE_MULF2P32X4RAS(val0, val1, val0, val1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    AE_MULSF2P32X4RS(acc0, acc1, val0, val1, AE_SLAA32S(AE_MOVDA32(0x80000000), lsh), AE_SLAA32S(AE_MOVDA32(0x80000000), lsh)); \
}

#define MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(acc0, acc1, val0, val1, multiplier, lsh) { \
    AE_MULF2P32X4RAS(val0, val1, val0, val1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    AE_MULAF2P32X4RS(acc0, acc1, val0, val1, AE_SLAA32S(AE_MOVDA32(0x80000000), lsh), AE_SLAA32S(AE_MOVDA32(0x80000000), lsh)); \
}

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out1, out2, inp1, inp2, multiplier, l_shift, out_off) \
  AE_MULF2P32X4RAS(out1, out2, inp1, inp2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  AE_MULF2P32X4RS(out1, out2, out1, out2, AE_SLAA32S(AE_MOVDA32(0x80000000), l_shift), AE_SLAA32S(AE_MOVDA32(0x80000000), l_shift)); \
  out1 = AE_SUB32S(AE_MOVDA32(out_off), out1); \
  out2 = AE_SUB32S(AE_MOVDA32(out_off), out2);

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(prod1, prod2, val1, val2, multiplier1, multiplier2, lsh1, lsh2) {\
    AE_MULF2P32X4RAS(prod1, prod2, val1, val2, AE_MOVDA32(multiplier1), AE_MOVDA32(multiplier2));\
    prod1 = AE_SRAA32SYMS(prod1, -lsh1);\
    prod2 = AE_SRAA32SYMS(prod2, -lsh2);\
}

#define MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, inp1, inp2, multiplier, l_shift, out_off) \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, AE_SLAA32S(AE_MOVDA32(0x80000000), l_shift), AE_SLAA32S(AE_MOVDA32(0x80000000), l_shift)); \
  out1 = AE_SAT16X4(inp1, inp2); \
  out1 = AE_SUB16S(AE_MOVDA16(out_off), out1);

#define MPY_BY_QUANT_MULT_GT_ONE_X2_OUT32(y, x, multiplier, lsh) {\
    y = AE_SLAA32(x, lsh);\
    y = AE_MULFP32X2RAS(y, AE_MOVDA32(multiplier));\
}

#define MPY_BY_QUANT_MULT_GT_ONE_X2X2_OUT32(y, z, l, m, multiplier, lsh) {\
    y = AE_SLAA32(l, lsh);\
    z = AE_SLAA32(m, lsh);\
    AE_MULF2P32X4RAS(y, z, y, z, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier));\
}

#define MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(out1, out2, inp1, inp2, multiplier, left_shift, right_shift) \
{\
  int rsh_mul = (0XFFFFFFFF << (31 - right_shift)); \
  out1 = AE_SLAA32S(inp1, left_shift); \
  out1 = AE_MULFP32X2RAS(out1, AE_NEG32(AE_MOVDA32(multiplier))); \
  out1 = AE_MULFP32X2RS(out1, rsh_mul); \
  out2 = AE_SLAA32S(inp2, left_shift); \
  out2 = AE_MULFP32X2RAS(out2, AE_NEG32(AE_MOVDA32(multiplier))); \
  out2 = AE_MULFP32X2RS(out2, rsh_mul); \
}

#endif /* #if TFLITE_SINGLE_ROUNDING */

#endif /* #ifndef __XA_NNLIB_QUANT_MACROS_HIFI5_H__ */
