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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

WORD32 xa_nn_elm_equal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;
  ae_int16x4 x76, x54, x32, x10, y76, y54, y32, y10;
  ae_int32x2 a76, a54, a32, a10, b76, b54, b32, b10, c76, c54, c32, c10, d76, d54, d32, d10;
  xtbool2 e76, e54, e32, e10, f76, f54, f32, f10;
  ae_int32x2 dequantized_a76, dequantized_a54, dequantized_a32, dequantized_a10, dequantized_b76, dequantized_b54, dequantized_b32, dequantized_b10, dequantized_c76, dequantized_c54, dequantized_c32, dequantized_c10, dequantized_d76, dequantized_d54, dequantized_d32, dequantized_d10;

  ae_int8x8 inp1_z_b = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 inp2_z_b = AE_MOVDA8(-inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP((ae_int8x16 *)p_in1);
  align_src_in2 = AE_LA128_PP((ae_int8x16 *)p_in2);
  align_dst     = AE_ZALIGN128();

  ae_int32x2 out_76_H = AE_ZERO32();
  ae_int32x2 out_54_H = AE_ZERO32();
  ae_int32x2 out_32_H = AE_ZERO32();
  ae_int32x2 out_10_H = AE_ZERO32();
  ae_int32x2 out_76_L = AE_ZERO32();
  ae_int32x2 out_54_L = AE_ZERO32();
  ae_int32x2 out_32_L = AE_ZERO32();
  ae_int32x2 out_10_L = AE_ZERO32();
  ae_int32x2 one_32x2  = AE_MOVDA32(1);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(x32, x10, m2, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);
    AE_SUBW8(y32, y10, m4, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(b76, b54, x32, left_shift);
    AE_CVTA32X4F16S(b32, b10, x10, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);
    AE_CVTA32X4F16S(d76, d54, y32, left_shift);
    AE_CVTA32X4F16S(d32, d10, y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_EQ32(dequantized_a76, dequantized_c76);
    e54 = AE_EQ32(dequantized_a54, dequantized_c54);
    e32 = AE_EQ32(dequantized_a32, dequantized_c32);
    e10 = AE_EQ32(dequantized_a10, dequantized_c10);
    f76 = AE_EQ32(dequantized_b76, dequantized_d76);
    f54 = AE_EQ32(dequantized_b54, dequantized_d54);
    f32 = AE_EQ32(dequantized_b32, dequantized_d32);
    f10 = AE_EQ32(dequantized_b10, dequantized_d10);

    AE_MOVT32X2(out_76_H, one_32x2, e76);
    AE_MOVT32X2(out_54_H, one_32x2, e54);
    AE_MOVT32X2(out_32_H, one_32x2, e32);
    AE_MOVT32X2(out_10_H, one_32x2, e10);
    AE_MOVT32X2(out_76_L, one_32x2, f76);
    AE_MOVT32X2(out_54_L, one_32x2, f54);
    AE_MOVT32X2(out_32_L, one_32x2, f32);
    AE_MOVT32X2(out_10_L, one_32x2, f10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);
    y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
    y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
    y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
    y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);
    x32 = AE_SEL16I(y76, y54, 8);
    x10 = AE_SEL16I(y32, y10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
    m2 = AE_SAT8X8X16(x32, x10);
    AE_SA8X8X2_IP(m1, m2, align_dst, (ae_int8x16 *)p_o);

    out_76_H = AE_ZERO32();
    out_54_H = AE_ZERO32();
    out_32_H = AE_ZERO32();
    out_10_H = AE_ZERO32();
    out_76_L = AE_ZERO32();
    out_54_L = AE_ZERO32();
    out_32_L = AE_ZERO32();
    out_10_L = AE_ZERO32();
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2, rem_length);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_EQ32(dequantized_a76, dequantized_c76);
    e54 = AE_EQ32(dequantized_a54, dequantized_c54);
    e32 = AE_EQ32(dequantized_a32, dequantized_c32);
    e10 = AE_EQ32(dequantized_a10, dequantized_c10);

    AE_MOVT32X2(out_76_H, one_32x2, e76);
    AE_MOVT32X2(out_54_H, one_32x2, e54);
    AE_MOVT32X2(out_32_H, one_32x2, e32);
    AE_MOVT32X2(out_10_H, one_32x2, e10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
   
    if(rem_length > 8)
    {
      AE_SUBW8(x32, x10, m2, inp1_z_b);
      AE_SUBW8(y32, y10, m4, inp2_z_b);

      AE_CVTA32X4F16S(b76, b54, x32, left_shift);
      AE_CVTA32X4F16S(b32, b10, x10, left_shift);
      AE_CVTA32X4F16S(d76, d54, y32, left_shift);
      AE_CVTA32X4F16S(d32, d10, y10, left_shift);

      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

      f76 = AE_EQ32(dequantized_b76, dequantized_d76);
      f54 = AE_EQ32(dequantized_b54, dequantized_d54);
      f32 = AE_EQ32(dequantized_b32, dequantized_d32);
      f10 = AE_EQ32(dequantized_b10, dequantized_d10);

      AE_MOVT32X2(out_76_L, one_32x2, f76);
      AE_MOVT32X2(out_54_L, one_32x2, f54);
      AE_MOVT32X2(out_32_L, one_32x2, f32);
      AE_MOVT32X2(out_10_L, one_32x2, f10);

      y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
      y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
      y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
      y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

      x32 = AE_SEL16I(y76, y54, 8);
      x10 = AE_SEL16I(y32, y10, 8);

      m2 = AE_SAT8X8X16(x32, x10);
    } 
    AE_SAV8X8X2_XP(m1, m2, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}


WORD32 xa_nn_elm_notequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;
  ae_int16x4 x76, x54, x32, x10, y76, y54, y32, y10;
  ae_int32x2 a76, a54, a32, a10, b76, b54, b32, b10, c76, c54, c32, c10, d76, d54, d32, d10;
  xtbool2 e76, e54, e32, e10, f76, f54, f32, f10;
  ae_int32x2 dequantized_a76, dequantized_a54, dequantized_a32, dequantized_a10, dequantized_b76, dequantized_b54, dequantized_b32, dequantized_b10, dequantized_c76, dequantized_c54, dequantized_c32, dequantized_c10, dequantized_d76, dequantized_d54, dequantized_d32, dequantized_d10;

  ae_int8x8 inp1_z_b = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 inp2_z_b = AE_MOVDA8(-inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP((ae_int8x16 *)p_in1);
  align_src_in2 = AE_LA128_PP((ae_int8x16 *)p_in2);
  align_dst     = AE_ZALIGN128();

  ae_int32x2 out_76_H = AE_ZERO32();
  ae_int32x2 out_54_H = AE_ZERO32();
  ae_int32x2 out_32_H = AE_ZERO32();
  ae_int32x2 out_10_H = AE_ZERO32();
  ae_int32x2 out_76_L = AE_ZERO32();
  ae_int32x2 out_54_L = AE_ZERO32();
  ae_int32x2 out_32_L = AE_ZERO32();
  ae_int32x2 out_10_L = AE_ZERO32();
  ae_int32x2 one_32x2  = AE_MOVDA32(1);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(x32, x10, m2, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);
    AE_SUBW8(y32, y10, m4, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(b76, b54, x32, left_shift);
    AE_CVTA32X4F16S(b32, b10, x10, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);
    AE_CVTA32X4F16S(d76, d54, y32, left_shift);
    AE_CVTA32X4F16S(d32, d10, y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_EQ32(dequantized_a76, dequantized_c76);
    e54 = AE_EQ32(dequantized_a54, dequantized_c54);
    e32 = AE_EQ32(dequantized_a32, dequantized_c32);
    e10 = AE_EQ32(dequantized_a10, dequantized_c10);
    f76 = AE_EQ32(dequantized_b76, dequantized_d76);
    f54 = AE_EQ32(dequantized_b54, dequantized_d54);
    f32 = AE_EQ32(dequantized_b32, dequantized_d32);
    f10 = AE_EQ32(dequantized_b10, dequantized_d10);

    AE_MOVF32X2(out_76_H, one_32x2, e76);
    AE_MOVF32X2(out_54_H, one_32x2, e54);
    AE_MOVF32X2(out_32_H, one_32x2, e32);
    AE_MOVF32X2(out_10_H, one_32x2, e10);
    AE_MOVF32X2(out_76_L, one_32x2, f76);
    AE_MOVF32X2(out_54_L, one_32x2, f54);
    AE_MOVF32X2(out_32_L, one_32x2, f32);
    AE_MOVF32X2(out_10_L, one_32x2, f10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);
    y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
    y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
    y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
    y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);
    x32 = AE_SEL16I(y76, y54, 8);
    x10 = AE_SEL16I(y32, y10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
    m2 = AE_SAT8X8X16(x32, x10);
    AE_SA8X8X2_IP(m1, m2, align_dst, (ae_int8x16 *)p_o);

    out_76_H = AE_ZERO32();
    out_54_H = AE_ZERO32();
    out_32_H = AE_ZERO32();
    out_10_H = AE_ZERO32();
    out_76_L = AE_ZERO32();
    out_54_L = AE_ZERO32();
    out_32_L = AE_ZERO32();
    out_10_L = AE_ZERO32();
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2, rem_length);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_EQ32(dequantized_a76, dequantized_c76);
    e54 = AE_EQ32(dequantized_a54, dequantized_c54);
    e32 = AE_EQ32(dequantized_a32, dequantized_c32);
    e10 = AE_EQ32(dequantized_a10, dequantized_c10);

    AE_MOVF32X2(out_76_H, one_32x2, e76);
    AE_MOVF32X2(out_54_H, one_32x2, e54);
    AE_MOVF32X2(out_32_H, one_32x2, e32);
    AE_MOVF32X2(out_10_H, one_32x2, e10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
   
    if(rem_length > 8)
    {
      AE_SUBW8(x32, x10, m2, inp1_z_b);
      AE_SUBW8(y32, y10, m4, inp2_z_b);

      AE_CVTA32X4F16S(b76, b54, x32, left_shift);
      AE_CVTA32X4F16S(b32, b10, x10, left_shift);
      AE_CVTA32X4F16S(d76, d54, y32, left_shift);
      AE_CVTA32X4F16S(d32, d10, y10, left_shift);

      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

      f76 = AE_EQ32(dequantized_b76, dequantized_d76);
      f54 = AE_EQ32(dequantized_b54, dequantized_d54);
      f32 = AE_EQ32(dequantized_b32, dequantized_d32);
      f10 = AE_EQ32(dequantized_b10, dequantized_d10);

      AE_MOVF32X2(out_76_L, one_32x2, f76);
      AE_MOVF32X2(out_54_L, one_32x2, f54);
      AE_MOVF32X2(out_32_L, one_32x2, f32);
      AE_MOVF32X2(out_10_L, one_32x2, f10);

      y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
      y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
      y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
      y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

      x32 = AE_SEL16I(y76, y54, 8);
      x10 = AE_SEL16I(y32, y10, 8);

      m2 = AE_SAT8X8X16(x32, x10);
    } 
    AE_SAV8X8X2_XP(m1, m2, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_greater_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;
  ae_int16x4 x76, x54, x32, x10, y76, y54, y32, y10;
  ae_int32x2 a76, a54, a32, a10, b76, b54, b32, b10, c76, c54, c32, c10, d76, d54, d32, d10;
  xtbool2 e76, e54, e32, e10, f76, f54, f32, f10;
  ae_int32x2 dequantized_a76, dequantized_a54, dequantized_a32, dequantized_a10, dequantized_b76, dequantized_b54, dequantized_b32, dequantized_b10, dequantized_c76, dequantized_c54, dequantized_c32, dequantized_c10, dequantized_d76, dequantized_d54, dequantized_d32, dequantized_d10;

  ae_int8x8 inp1_z_b = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 inp2_z_b = AE_MOVDA8(-inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP((ae_int8x16 *)p_in1);
  align_src_in2 = AE_LA128_PP((ae_int8x16 *)p_in2);
  align_dst     = AE_ZALIGN128();

  ae_int32x2 out_76_H = AE_ZERO32();
  ae_int32x2 out_54_H = AE_ZERO32();
  ae_int32x2 out_32_H = AE_ZERO32();
  ae_int32x2 out_10_H = AE_ZERO32();
  ae_int32x2 out_76_L = AE_ZERO32();
  ae_int32x2 out_54_L = AE_ZERO32();
  ae_int32x2 out_32_L = AE_ZERO32();
  ae_int32x2 out_10_L = AE_ZERO32();
  ae_int32x2 one_32x2  = AE_MOVDA32(1);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(x32, x10, m2, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);
    AE_SUBW8(y32, y10, m4, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(b76, b54, x32, left_shift);
    AE_CVTA32X4F16S(b32, b10, x10, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);
    AE_CVTA32X4F16S(d76, d54, y32, left_shift);
    AE_CVTA32X4F16S(d32, d10, y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LE32(dequantized_a76, dequantized_c76);
    e54 = AE_LE32(dequantized_a54, dequantized_c54);
    e32 = AE_LE32(dequantized_a32, dequantized_c32);
    e10 = AE_LE32(dequantized_a10, dequantized_c10);
    f76 = AE_LE32(dequantized_b76, dequantized_d76);
    f54 = AE_LE32(dequantized_b54, dequantized_d54);
    f32 = AE_LE32(dequantized_b32, dequantized_d32);
    f10 = AE_LE32(dequantized_b10, dequantized_d10);

    AE_MOVF32X2(out_76_H, one_32x2, e76);
    AE_MOVF32X2(out_54_H, one_32x2, e54);
    AE_MOVF32X2(out_32_H, one_32x2, e32);
    AE_MOVF32X2(out_10_H, one_32x2, e10);
    AE_MOVF32X2(out_76_L, one_32x2, f76);
    AE_MOVF32X2(out_54_L, one_32x2, f54);
    AE_MOVF32X2(out_32_L, one_32x2, f32);
    AE_MOVF32X2(out_10_L, one_32x2, f10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);
    y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
    y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
    y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
    y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);
    x32 = AE_SEL16I(y76, y54, 8);
    x10 = AE_SEL16I(y32, y10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
    m2 = AE_SAT8X8X16(x32, x10);
    AE_SA8X8X2_IP(m1, m2, align_dst, (ae_int8x16 *)p_o);

    out_76_H = AE_ZERO32();
    out_54_H = AE_ZERO32();
    out_32_H = AE_ZERO32();
    out_10_H = AE_ZERO32();
    out_76_L = AE_ZERO32();
    out_54_L = AE_ZERO32();
    out_32_L = AE_ZERO32();
    out_10_L = AE_ZERO32();
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2, rem_length);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LE32(dequantized_a76, dequantized_c76);
    e54 = AE_LE32(dequantized_a54, dequantized_c54);
    e32 = AE_LE32(dequantized_a32, dequantized_c32);
    e10 = AE_LE32(dequantized_a10, dequantized_c10);

    AE_MOVF32X2(out_76_H, one_32x2, e76);
    AE_MOVF32X2(out_54_H, one_32x2, e54);
    AE_MOVF32X2(out_32_H, one_32x2, e32);
    AE_MOVF32X2(out_10_H, one_32x2, e10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
   
    if(rem_length > 8)
    {
      AE_SUBW8(x32, x10, m2, inp1_z_b);
      AE_SUBW8(y32, y10, m4, inp2_z_b);

      AE_CVTA32X4F16S(b76, b54, x32, left_shift);
      AE_CVTA32X4F16S(b32, b10, x10, left_shift);
      AE_CVTA32X4F16S(d76, d54, y32, left_shift);
      AE_CVTA32X4F16S(d32, d10, y10, left_shift);

      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

      f76 = AE_LE32(dequantized_b76, dequantized_d76);
      f54 = AE_LE32(dequantized_b54, dequantized_d54);
      f32 = AE_LE32(dequantized_b32, dequantized_d32);
      f10 = AE_LE32(dequantized_b10, dequantized_d10);

      AE_MOVF32X2(out_76_L, one_32x2, f76);
      AE_MOVF32X2(out_54_L, one_32x2, f54);
      AE_MOVF32X2(out_32_L, one_32x2, f32);
      AE_MOVF32X2(out_10_L, one_32x2, f10);

      y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
      y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
      y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
      y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

      x32 = AE_SEL16I(y76, y54, 8);
      x10 = AE_SEL16I(y32, y10, 8);

      m2 = AE_SAT8X8X16(x32, x10);
    } 
    AE_SAV8X8X2_XP(m1, m2, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_greaterequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;
  ae_int16x4 x76, x54, x32, x10, y76, y54, y32, y10;
  ae_int32x2 a76, a54, a32, a10, b76, b54, b32, b10, c76, c54, c32, c10, d76, d54, d32, d10;
  xtbool2 e76, e54, e32, e10, f76, f54, f32, f10;
  ae_int32x2 dequantized_a76, dequantized_a54, dequantized_a32, dequantized_a10, dequantized_b76, dequantized_b54, dequantized_b32, dequantized_b10, dequantized_c76, dequantized_c54, dequantized_c32, dequantized_c10, dequantized_d76, dequantized_d54, dequantized_d32, dequantized_d10;

  ae_int8x8 inp1_z_b = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 inp2_z_b = AE_MOVDA8(-inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP((ae_int8x16 *)p_in1);
  align_src_in2 = AE_LA128_PP((ae_int8x16 *)p_in2);
  align_dst     = AE_ZALIGN128();

  ae_int32x2 out_76_H = AE_ZERO32();
  ae_int32x2 out_54_H = AE_ZERO32();
  ae_int32x2 out_32_H = AE_ZERO32();
  ae_int32x2 out_10_H = AE_ZERO32();
  ae_int32x2 out_76_L = AE_ZERO32();
  ae_int32x2 out_54_L = AE_ZERO32();
  ae_int32x2 out_32_L = AE_ZERO32();
  ae_int32x2 out_10_L = AE_ZERO32();
  ae_int32x2 one_32x2  = AE_MOVDA32(1);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(x32, x10, m2, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);
    AE_SUBW8(y32, y10, m4, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(b76, b54, x32, left_shift);
    AE_CVTA32X4F16S(b32, b10, x10, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);
    AE_CVTA32X4F16S(d76, d54, y32, left_shift);
    AE_CVTA32X4F16S(d32, d10, y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LT32(dequantized_a76, dequantized_c76);
    e54 = AE_LT32(dequantized_a54, dequantized_c54);
    e32 = AE_LT32(dequantized_a32, dequantized_c32);
    e10 = AE_LT32(dequantized_a10, dequantized_c10);
    f76 = AE_LT32(dequantized_b76, dequantized_d76);
    f54 = AE_LT32(dequantized_b54, dequantized_d54);
    f32 = AE_LT32(dequantized_b32, dequantized_d32);
    f10 = AE_LT32(dequantized_b10, dequantized_d10);

    AE_MOVF32X2(out_76_H, one_32x2, e76);
    AE_MOVF32X2(out_54_H, one_32x2, e54);
    AE_MOVF32X2(out_32_H, one_32x2, e32);
    AE_MOVF32X2(out_10_H, one_32x2, e10);
    AE_MOVF32X2(out_76_L, one_32x2, f76);
    AE_MOVF32X2(out_54_L, one_32x2, f54);
    AE_MOVF32X2(out_32_L, one_32x2, f32);
    AE_MOVF32X2(out_10_L, one_32x2, f10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);
    y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
    y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
    y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
    y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);
    x32 = AE_SEL16I(y76, y54, 8);
    x10 = AE_SEL16I(y32, y10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
    m2 = AE_SAT8X8X16(x32, x10);
    AE_SA8X8X2_IP(m1, m2, align_dst, (ae_int8x16 *)p_o);

    out_76_H = AE_ZERO32();
    out_54_H = AE_ZERO32();
    out_32_H = AE_ZERO32();
    out_10_H = AE_ZERO32();
    out_76_L = AE_ZERO32();
    out_54_L = AE_ZERO32();
    out_32_L = AE_ZERO32();
    out_10_L = AE_ZERO32();
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2, rem_length);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LT32(dequantized_a76, dequantized_c76);
    e54 = AE_LT32(dequantized_a54, dequantized_c54);
    e32 = AE_LT32(dequantized_a32, dequantized_c32);
    e10 = AE_LT32(dequantized_a10, dequantized_c10);

    AE_MOVF32X2(out_76_H, one_32x2, e76);
    AE_MOVF32X2(out_54_H, one_32x2, e54);
    AE_MOVF32X2(out_32_H, one_32x2, e32);
    AE_MOVF32X2(out_10_H, one_32x2, e10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
   
    if(rem_length > 8)
    {
      AE_SUBW8(x32, x10, m2, inp1_z_b);
      AE_SUBW8(y32, y10, m4, inp2_z_b);

      AE_CVTA32X4F16S(b76, b54, x32, left_shift);
      AE_CVTA32X4F16S(b32, b10, x10, left_shift);
      AE_CVTA32X4F16S(d76, d54, y32, left_shift);
      AE_CVTA32X4F16S(d32, d10, y10, left_shift);

      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

      f76 = AE_LT32(dequantized_b76, dequantized_d76);
      f54 = AE_LT32(dequantized_b54, dequantized_d54);
      f32 = AE_LT32(dequantized_b32, dequantized_d32);
      f10 = AE_LT32(dequantized_b10, dequantized_d10);

      AE_MOVF32X2(out_76_L, one_32x2, f76);
      AE_MOVF32X2(out_54_L, one_32x2, f54);
      AE_MOVF32X2(out_32_L, one_32x2, f32);
      AE_MOVF32X2(out_10_L, one_32x2, f10);

      y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
      y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
      y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
      y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

      x32 = AE_SEL16I(y76, y54, 8);
      x10 = AE_SEL16I(y32, y10, 8);

      m2 = AE_SAT8X8X16(x32, x10);
    } 
    AE_SAV8X8X2_XP(m1, m2, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_less_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;
  ae_int16x4 x76, x54, x32, x10, y76, y54, y32, y10;
  ae_int32x2 a76, a54, a32, a10, b76, b54, b32, b10, c76, c54, c32, c10, d76, d54, d32, d10;
  xtbool2 e76, e54, e32, e10, f76, f54, f32, f10;
  ae_int32x2 dequantized_a76, dequantized_a54, dequantized_a32, dequantized_a10, dequantized_b76, dequantized_b54, dequantized_b32, dequantized_b10, dequantized_c76, dequantized_c54, dequantized_c32, dequantized_c10, dequantized_d76, dequantized_d54, dequantized_d32, dequantized_d10;

  ae_int8x8 inp1_z_b = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 inp2_z_b = AE_MOVDA8(-inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP((ae_int8x16 *)p_in1);
  align_src_in2 = AE_LA128_PP((ae_int8x16 *)p_in2);
  align_dst     = AE_ZALIGN128();

  ae_int32x2 out_76_H = AE_ZERO32();
  ae_int32x2 out_54_H = AE_ZERO32();
  ae_int32x2 out_32_H = AE_ZERO32();
  ae_int32x2 out_10_H = AE_ZERO32();
  ae_int32x2 out_76_L = AE_ZERO32();
  ae_int32x2 out_54_L = AE_ZERO32();
  ae_int32x2 out_32_L = AE_ZERO32();
  ae_int32x2 out_10_L = AE_ZERO32();
  ae_int32x2 one_32x2  = AE_MOVDA32(1);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(x32, x10, m2, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);
    AE_SUBW8(y32, y10, m4, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(b76, b54, x32, left_shift);
    AE_CVTA32X4F16S(b32, b10, x10, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);
    AE_CVTA32X4F16S(d76, d54, y32, left_shift);
    AE_CVTA32X4F16S(d32, d10, y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LT32(dequantized_a76, dequantized_c76);
    e54 = AE_LT32(dequantized_a54, dequantized_c54);
    e32 = AE_LT32(dequantized_a32, dequantized_c32);
    e10 = AE_LT32(dequantized_a10, dequantized_c10);
    f76 = AE_LT32(dequantized_b76, dequantized_d76);
    f54 = AE_LT32(dequantized_b54, dequantized_d54);
    f32 = AE_LT32(dequantized_b32, dequantized_d32);
    f10 = AE_LT32(dequantized_b10, dequantized_d10);

    AE_MOVT32X2(out_76_H, one_32x2, e76);
    AE_MOVT32X2(out_54_H, one_32x2, e54);
    AE_MOVT32X2(out_32_H, one_32x2, e32);
    AE_MOVT32X2(out_10_H, one_32x2, e10);
    AE_MOVT32X2(out_76_L, one_32x2, f76);
    AE_MOVT32X2(out_54_L, one_32x2, f54);
    AE_MOVT32X2(out_32_L, one_32x2, f32);
    AE_MOVT32X2(out_10_L, one_32x2, f10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);
    y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
    y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
    y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
    y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);
    x32 = AE_SEL16I(y76, y54, 8);
    x10 = AE_SEL16I(y32, y10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
    m2 = AE_SAT8X8X16(x32, x10);
    AE_SA8X8X2_IP(m1, m2, align_dst, (ae_int8x16 *)p_o);

    out_76_H = AE_ZERO32();
    out_54_H = AE_ZERO32();
    out_32_H = AE_ZERO32();
    out_10_H = AE_ZERO32();
    out_76_L = AE_ZERO32();
    out_54_L = AE_ZERO32();
    out_32_L = AE_ZERO32();
    out_10_L = AE_ZERO32();
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2, rem_length);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LT32(dequantized_a76, dequantized_c76);
    e54 = AE_LT32(dequantized_a54, dequantized_c54);
    e32 = AE_LT32(dequantized_a32, dequantized_c32);
    e10 = AE_LT32(dequantized_a10, dequantized_c10);

    AE_MOVT32X2(out_76_H, one_32x2, e76);
    AE_MOVT32X2(out_54_H, one_32x2, e54);
    AE_MOVT32X2(out_32_H, one_32x2, e32);
    AE_MOVT32X2(out_10_H, one_32x2, e10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
   
    if(rem_length > 8)
    {
      AE_SUBW8(x32, x10, m2, inp1_z_b);
      AE_SUBW8(y32, y10, m4, inp2_z_b);

      AE_CVTA32X4F16S(b76, b54, x32, left_shift);
      AE_CVTA32X4F16S(b32, b10, x10, left_shift);
      AE_CVTA32X4F16S(d76, d54, y32, left_shift);
      AE_CVTA32X4F16S(d32, d10, y10, left_shift);

      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

      f76 = AE_LT32(dequantized_b76, dequantized_d76);
      f54 = AE_LT32(dequantized_b54, dequantized_d54);
      f32 = AE_LT32(dequantized_b32, dequantized_d32);
      f10 = AE_LT32(dequantized_b10, dequantized_d10);

      AE_MOVT32X2(out_76_L, one_32x2, f76);
      AE_MOVT32X2(out_54_L, one_32x2, f54);
      AE_MOVT32X2(out_32_L, one_32x2, f32);
      AE_MOVT32X2(out_10_L, one_32x2, f10);

      y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
      y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
      y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
      y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

      x32 = AE_SEL16I(y76, y54, 8);
      x10 = AE_SEL16I(y32, y10, 8);

      m2 = AE_SAT8X8X16(x32, x10);
    } 
    AE_SAV8X8X2_XP(m1, m2, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_lessequal_asym8sxasym8s(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_shift,
                            WORD32  inp1_multiplier,
                    const   WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_shift < -31) || (inp1_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_shift < -31) || (inp2_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((inp1_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp2_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;
  ae_int16x4 x76, x54, x32, x10, y76, y54, y32, y10;
  ae_int32x2 a76, a54, a32, a10, b76, b54, b32, b10, c76, c54, c32, c10, d76, d54, d32, d10;
  xtbool2 e76, e54, e32, e10, f76, f54, f32, f10;
  ae_int32x2 dequantized_a76, dequantized_a54, dequantized_a32, dequantized_a10, dequantized_b76, dequantized_b54, dequantized_b32, dequantized_b10, dequantized_c76, dequantized_c54, dequantized_c32, dequantized_c10, dequantized_d76, dequantized_d54, dequantized_d32, dequantized_d10;

  ae_int8x8 inp1_z_b = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8 inp2_z_b = AE_MOVDA8(-inp2_zero_bias);

  WORD8 *p_in1  = (WORD8 *)p_inp1;
  WORD8 *p_in2  = (WORD8 *)p_inp2;
  WORD8 *p_o    = (WORD8 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP((ae_int8x16 *)p_in1);
  align_src_in2 = AE_LA128_PP((ae_int8x16 *)p_in2);
  align_dst     = AE_ZALIGN128();

  ae_int32x2 out_76_H = AE_ZERO32();
  ae_int32x2 out_54_H = AE_ZERO32();
  ae_int32x2 out_32_H = AE_ZERO32();
  ae_int32x2 out_10_H = AE_ZERO32();
  ae_int32x2 out_76_L = AE_ZERO32();
  ae_int32x2 out_54_L = AE_ZERO32();
  ae_int32x2 out_32_L = AE_ZERO32();
  ae_int32x2 out_10_L = AE_ZERO32();
  ae_int32x2 one_32x2  = AE_MOVDA32(1);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(x32, x10, m2, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);
    AE_SUBW8(y32, y10, m4, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(b76, b54, x32, left_shift);
    AE_CVTA32X4F16S(b32, b10, x10, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);
    AE_CVTA32X4F16S(d76, d54, y32, left_shift);
    AE_CVTA32X4F16S(d32, d10, y10, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LE32(dequantized_a76, dequantized_c76);
    e54 = AE_LE32(dequantized_a54, dequantized_c54);
    e32 = AE_LE32(dequantized_a32, dequantized_c32);
    e10 = AE_LE32(dequantized_a10, dequantized_c10);
    f76 = AE_LE32(dequantized_b76, dequantized_d76);
    f54 = AE_LE32(dequantized_b54, dequantized_d54);
    f32 = AE_LE32(dequantized_b32, dequantized_d32);
    f10 = AE_LE32(dequantized_b10, dequantized_d10);

    AE_MOVT32X2(out_76_H, one_32x2, e76);
    AE_MOVT32X2(out_54_H, one_32x2, e54);
    AE_MOVT32X2(out_32_H, one_32x2, e32);
    AE_MOVT32X2(out_10_H, one_32x2, e10);
    AE_MOVT32X2(out_76_L, one_32x2, f76);
    AE_MOVT32X2(out_54_L, one_32x2, f54);
    AE_MOVT32X2(out_32_L, one_32x2, f32);
    AE_MOVT32X2(out_10_L, one_32x2, f10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);
    y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
    y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
    y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
    y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);
    x32 = AE_SEL16I(y76, y54, 8);
    x10 = AE_SEL16I(y32, y10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
    m2 = AE_SAT8X8X16(x32, x10);
    AE_SA8X8X2_IP(m1, m2, align_dst, (ae_int8x16 *)p_o);

    out_76_H = AE_ZERO32();
    out_54_H = AE_ZERO32();
    out_32_H = AE_ZERO32();
    out_10_H = AE_ZERO32();
    out_76_L = AE_ZERO32();
    out_54_L = AE_ZERO32();
    out_32_L = AE_ZERO32();
    out_10_L = AE_ZERO32();
  }

  // remainder loop
  if(rem_length)
  {
    AE_LAV8X8X2_XP(m1, m2, align_src_in1, (ae_int8x16 *)p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, (ae_int8x16 *)p_in2, rem_length);

    AE_SUBW8(x76, x54, m1, inp1_z_b);
    AE_SUBW8(y76, y54, m3, inp2_z_b);

    AE_CVTA32X4F16S(a76, a54, x76, left_shift);
    AE_CVTA32X4F16S(a32, a10, x54, left_shift);
    AE_CVTA32X4F16S(c76, c54, y76, left_shift);
    AE_CVTA32X4F16S(c32, c10, y54, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a76, dequantized_c76, a76, c76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a54, dequantized_c54, a54, c54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a32, dequantized_c32, a32, c32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_a10, dequantized_c10, a10, c10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

    e76 = AE_LE32(dequantized_a76, dequantized_c76);
    e54 = AE_LE32(dequantized_a54, dequantized_c54);
    e32 = AE_LE32(dequantized_a32, dequantized_c32);
    e10 = AE_LE32(dequantized_a10, dequantized_c10);

    AE_MOVT32X2(out_76_H, one_32x2, e76);
    AE_MOVT32X2(out_54_H, one_32x2, e54);
    AE_MOVT32X2(out_32_H, one_32x2, e32);
    AE_MOVT32X2(out_10_H, one_32x2, e10);

    x76 = AE_MOVINT16X4_FROMINT32X2(out_76_H);
    x54 = AE_MOVINT16X4_FROMINT32X2(out_54_H);
    x32 = AE_MOVINT16X4_FROMINT32X2(out_32_H);
    x10 = AE_MOVINT16X4_FROMINT32X2(out_10_H);

    x76 = AE_SEL16I(x76, x54, 8);
    x54 = AE_SEL16I(x32, x10, 8);

    m1 = AE_SAT8X8X16(x76, x54);
   
    if(rem_length > 8)
    {
      AE_SUBW8(x32, x10, m2, inp1_z_b);
      AE_SUBW8(y32, y10, m4, inp2_z_b);

      AE_CVTA32X4F16S(b76, b54, x32, left_shift);
      AE_CVTA32X4F16S(b32, b10, x10, left_shift);
      AE_CVTA32X4F16S(d76, d54, y32, left_shift);
      AE_CVTA32X4F16S(d32, d10, y10, left_shift);

      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b76, dequantized_d76, b76, d76, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b54, dequantized_d54, b54, d54, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b32, dequantized_d32, b32, d32, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32_X2(dequantized_b10, dequantized_d10, b10, d10, inp1_multiplier, inp2_multiplier, inp1_shift, inp2_shift);

      f76 = AE_LE32(dequantized_b76, dequantized_d76);
      f54 = AE_LE32(dequantized_b54, dequantized_d54);
      f32 = AE_LE32(dequantized_b32, dequantized_d32);
      f10 = AE_LE32(dequantized_b10, dequantized_d10);

      AE_MOVT32X2(out_76_L, one_32x2, f76);
      AE_MOVT32X2(out_54_L, one_32x2, f54);
      AE_MOVT32X2(out_32_L, one_32x2, f32);
      AE_MOVT32X2(out_10_L, one_32x2, f10);

      y76 = AE_MOVINT16X4_FROMINT32X2(out_76_L);
      y54 = AE_MOVINT16X4_FROMINT32X2(out_54_L);
      y32 = AE_MOVINT16X4_FROMINT32X2(out_32_L);
      y10 = AE_MOVINT16X4_FROMINT32X2(out_10_L);

      x32 = AE_SEL16I(y76, y54, 8);
      x10 = AE_SEL16I(y32, y10, 8);

      m2 = AE_SAT8X8X16(x32, x10);
    } 
    AE_SAV8X8X2_XP(m1, m2, align_dst, (ae_int8x16 *)p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}
