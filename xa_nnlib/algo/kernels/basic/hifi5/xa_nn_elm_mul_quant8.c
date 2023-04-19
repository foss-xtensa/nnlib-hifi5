/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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

WORD32 xa_nn_elm_mul_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < 0) || (out_zero_bias > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -255) || (inp1_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -255) || (inp2_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < 0) || (out_activation_min > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < 0) || (out_activation_max > 255)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);


    int i;
    UWORD8 *out = p_out;
    WORD8 *p_i1 = (WORD8 *)p_inp1;
    WORD8 *p_i2 = (WORD8 *)p_inp2;
    ae_f16x4 x1, x2;
    ae_int32x2 temp;
    ae_f16x4 temp16X4, zero_bias1, zero_bias2;
    ae_f32x2 op_zero_bias, activation_min, activation_max;
#if TFLITE_SINGLE_ROUNDING
    int left_shift = out_shift;
    int right_shift = out_shift;
    /* Single rounding doesn't need two shifts */
    (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
    int left_shift = out_shift < 0 ? 0 : out_shift;
    int right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    // Taking input zero_bias into 16X4 variable
    temp = AE_MOVDA32X2(inp1_zero_bias, inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32X2(inp2_zero_bias, inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);


    // Taking into 32x2 variable
    op_zero_bias = AE_MOVDA32X2(out_zero_bias, out_zero_bias);

    activation_min = AE_MOVDA32X2(out_activation_min, out_activation_min);
    activation_max = AE_MOVDA32X2(out_activation_max, out_activation_max);

    if(((((unsigned)p_i1)&3) == 0) && ((((unsigned)p_i2)&3) == 0))
    {
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 prod32, prod10;
            ae_f32x2 clamped_out32, clamped_out10;
            ae_f32x2 unclamped_out32, unclamped_out10;


            AE_L8X4F_IP(x1, p_i1, 4*sizeof(WORD8));
            AE_L8X4F_IP(x2, p_i2, 4*sizeof(WORD8));

            x1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x1), 8));
            x2 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x2), 8));

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            AE_MUL16X4(prod32, prod10, v1, v2);

            // unclamped result
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
            unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);
            unclamped_out10 = AE_ADD32S(unclamped_out10, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
            CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out32, clamped_out10)
        }
    }
    else
    {
        ALIGN_REGISTER_TYPE i1_a, i2_a;

        PRIME_8X4U(p_i1, i1_a);
        PRIME_8X4U(p_i2, i2_a);
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 prod32, prod10;
            ae_f32x2 clamped_out32, clamped_out10;
            ae_f32x2 unclamped_out32, unclamped_out10;


            AE_LA8X4U_IP(x1, i1_a, p_i1);
            AE_LA8X4U_IP(x2, i2_a, p_i2);

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            AE_MUL16X4(prod32, prod10, v1, v2);

            // unclamped result
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
            MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out10, prod10, out_multiplier, left_shift, right_shift)
            unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);
            unclamped_out10 = AE_ADD32S(unclamped_out10, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)
            CLAMP_VAL(clamped_out10, unclamped_out10, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out32, clamped_out10)
        }
    }

    p_i1 = (WORD8 *)p_inp1 + (num_elm & ~3);
    p_i2 = (WORD8 *)p_inp2 + (num_elm & ~3);

    // Remainder Loop
    for(i=0; i < (num_elm & 3); i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 prod32, prod10;
        ae_f32x2 clamped_out32;
        ae_f32x2 unclamped_out32;

        WORD16 i1, i2;

        i1 = (WORD16) *((UWORD8 *)p_i1 + i);
        i2 = (WORD16) *((UWORD8 *)p_i2 + i);

        x1 = AE_MOVDA16(i1);
        x2 = AE_MOVDA16(i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        AE_MUL16X4(prod32, prod10, v1, v2);

        // unclamped result
        MPY_BY_QUANT_MULT_SLS_X2_OUT32(unclamped_out32, prod32, out_multiplier, left_shift, right_shift)
        unclamped_out32 = AE_ADD32S(unclamped_out32, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out32, unclamped_out32, activation_min, activation_max)

        // Store Output
        i1 = (WORD16)AE_MOVAD32_H(clamped_out32);
        *out++ = (UWORD8) i1;
    }

    return 0;
}

static void internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  out_lc,
                            WORD32  in_lc)
{
  int i, j;
  WORD8 * __restrict__ p_a;
  WORD8 * __restrict__ p_b;
  WORD8 *__restrict__ p_c;

#if TFLITE_SINGLE_ROUNDING
  int l_shift = out_shift;
  int r_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int l_shift = out_shift >= 0 ?   out_shift : 0;
  int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int8x8 a0_7, b0_7;

  const ae_int8x8 za = AE_MOVDA8(-inp1_zero_bias);
  const ae_int8x8 zb = AE_MOVDA8(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 raw_mul0_1, raw_mul2_3, raw_mul4_5, raw_mul6_7;

  ae_int16x4 out0, out1;

  int num_simd8_ops;
  int num_scalar_ops;

  if(out_lc == 1)
  {
    num_simd8_ops = in_lc >> 3;
    num_scalar_ops = in_lc & 7;
  }
  else
  {
    num_simd8_ops = (in_lc >> 4) << 1;
    num_scalar_ops = in_lc & 15;
  }

  ae_valign va_a, va_b, va_c;

#pragma loop_count min=1
  for(i = 0; i < out_lc; i++)
  {
    p_a = (WORD8 *)&p_inp1[i * in_lc];
    p_b = (WORD8 *)p_inp2;
    p_c = (WORD8 *)&p_out[i * in_lc];

    va_a = AE_LA64_PP(p_a);
    va_b = AE_LA64_PP(p_b);
    va_c = AE_ZALIGN64();
    for(j = 0; j < num_simd8_ops; j++)
    {
      AE_LA8X8_IP(a0_7, va_a, (ae_int8x8 *)p_a);
      AE_LA8X8_IP(b0_7, va_b, (ae_int8x8 *)p_b);

      // Add input zero bias
      AE_SUBW8(a0_3, a4_7, a0_7, za);
      AE_SUBW8(b0_3, b4_7, b0_7, zb);

      AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
      AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);

      // Raw Output
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out0, raw_mul0_1, raw_mul2_3, out_multiplier, l_shift, r_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out1, raw_mul4_5, raw_mul6_7, out_multiplier, l_shift, r_shift, out_zero_bias);

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      ae_int8x8 res = AE_SAT8X8X16(out0, out1);

      AE_SA8X8_IP(res, va_c, (ae_int8x8 *)p_c);
    }
    AE_SA64POS_FP(va_c, p_c);
  }

  if(num_scalar_ops != 0)
  {
    ae_int8x8 a8_15, b8_15;
    ae_int16x4 out2, out3;
    ae_valignx2 va_ax2, va_bx2, va_cx2;
    ae_int16x4 a8_11, a12_15, b8_11, b12_15;
    ae_int32x2 raw_mul8_9, raw_mul10_11, raw_mul12_13, raw_mul14_15;

    p_b = (WORD8 *)&p_inp2[num_simd8_ops << 3];
    va_bx2 = AE_LA128_PP((ae_int8x16 *)p_b);
    AE_LAV8X8X2_XP(b0_7, b8_15, va_bx2, (ae_int8x16 *)p_b, num_scalar_ops);
    AE_SUBW8(b0_3, b4_7, b0_7, zb);
    AE_SUBW8(b8_11, b12_15, b8_15, zb);

#pragma loop_count min=1
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD8 *)&p_inp1[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD8 *)&p_out[i * in_lc + (num_simd8_ops << 3)];

      va_ax2 = AE_LA128_PP((ae_int8x16 *)p_a);
      va_cx2 = AE_ZALIGN128();

      AE_LAV8X8X2_XP(a0_7, a8_15, va_ax2, (ae_int8x16 *)p_a, num_scalar_ops);

      // Add input zero bias
      AE_SUBW8(a0_3, a4_7, a0_7, za);
      AE_SUBW8(a8_11, a12_15, a8_15, za);

      AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0_3);
      AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b4_7);
      AE_MUL16X4(raw_mul8_9, raw_mul10_11, a8_11, b8_11);
      AE_MUL16X4(raw_mul12_13, raw_mul14_15, a12_15, b12_15);

      // Raw Output
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out0, raw_mul0_1, raw_mul2_3, out_multiplier, l_shift, r_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out1, raw_mul4_5, raw_mul6_7, out_multiplier, l_shift, r_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out2, raw_mul8_9, raw_mul10_11, out_multiplier, l_shift, r_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out3, raw_mul12_13, raw_mul14_15, out_multiplier, l_shift, r_shift, out_zero_bias);

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      ae_int8x8 res0 = AE_SAT8X8X16(out0, out1);
      ae_int8x8 res1 = AE_SAT8X8X16(out2, out3);

      AE_SAV8X8X2_XP(res0, res1, va_cx2, (ae_int8x16 *)p_c, num_scalar_ops);
      AE_SA128POS_FP(va_cx2, (ae_int8x16 *)p_c);
    }
  }
}

static void internal_elm_mul_broadcast_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  num_elm)
{
  int i;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c =          p_out;

#if TFLITE_SINGLE_ROUNDING
  int l_shift = out_shift;
  int r_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)r_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int l_shift = out_shift >= 0 ?   out_shift : 0;
  int r_shift = out_shift <  0 ?  -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int8x8 a0_7, b;

  ae_int8x8  za = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8  zb = AE_MOVDA8(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0, b1;

  ae_int32x2 raw_mul0_1, raw_mul2_3, raw_mul4_5, raw_mul6_7;

  ae_int16x4 out0, out1;

  const int num_simd8_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;

  ae_valign va_a = AE_LA64_PP(p_a);
  ae_valign va_c = AE_ZALIGN64();

  b = AE_MOVDA8(p_b[0]);
  AE_SUBW8(b0, b1, b, zb);

  for(i=0; i<num_simd8_ops; i++)
  {
    AE_LA8X8_IP(a0_7, va_a, (ae_int8x8 *)p_a);

    // Add input zero bias
    AE_SUBW8(a0_3, a4_7, a0_7, za);

    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
    AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b0);

    // Raw Output
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out0, raw_mul0_1, raw_mul2_3, out_multiplier, l_shift, r_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out1, raw_mul4_5, raw_mul6_7, out_multiplier, l_shift, r_shift, out_zero_bias);

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    ae_int8x8 res = AE_SAT8X8X16(out0, out1);

    AE_SA8X8_IP(res, va_c, (ae_int8x8 *)p_c);
  }
  AE_SA64POS_FP(va_c, p_c);

  if(num_scalar_ops != 0)
  {
    ae_valignx2 va_ax2, va_cx2;

    va_ax2 = AE_LA128_PP((ae_int8x16 *)p_a);
    va_cx2 = AE_ZALIGN128();

    ae_int8x8 at0;

    AE_LAV8X8X2_XP(a0_7, at0, va_ax2, (ae_int8x16 *)p_a, num_scalar_ops);

    // Add input zero bias
    AE_SUBW8(a0_3, a4_7, a0_7, za);

    AE_MUL16X4(raw_mul0_1, raw_mul2_3, a0_3, b0);
    AE_MUL16X4(raw_mul4_5, raw_mul6_7, a4_7, b0);

    // Raw Output
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out0, raw_mul0_1, raw_mul2_3, out_multiplier, l_shift, r_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16_ZB(out1, raw_mul4_5, raw_mul6_7, out_multiplier, l_shift, r_shift, out_zero_bias);

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    ae_int8x8 res = AE_SAT8X8X16(out0, out1);

    AE_SAV8X8X2_XP(res, res, va_cx2, (ae_int8x16 *)p_c, num_scalar_ops);
    AE_SA128POS_FP(va_cx2, (ae_int8x16 *)p_c);
  }
}

WORD32 xa_nn_elm_mul_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
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
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(
      p_out,
      out_zero_bias,
      out_shift,
      out_multiplier,
      out_activation_min,
      out_activation_max,
      p_inp1,
      inp1_zero_bias,
      p_inp2,
      inp2_zero_bias,
      1,
      num_elm);

  return 0;
}

WORD32 xa_nn_elm_mul_broadcast_4D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD8 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                      const WORD8 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                            WORD32  inp2_zero_bias)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  /* Check shapes */
  int i;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i] && p_inp1_shape[i] != 1 && p_inp2_shape[i] != 1) ||
       (p_out_shape[i] != (p_inp1_shape[i] > p_inp2_shape[i] ? p_inp1_shape[i] : p_inp2_shape[i])))
    {
      return -1;
    }
  }

  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    // inp1_strides[i] = inp1_strides[i + 1] * p_inp1_shape[i + 1];
    // inp2_strides[i] = inp2_strides[i + 1] * p_inp2_shape[i + 1];
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast = 0;
  int inp1_const = 1, inp2_const = 1;
  for(i = 0; i < 4; i++)
  {
    if(p_inp1_shape[i] != p_inp2_shape[i])
    {
      if(p_inp1_shape[i] == 1)
        inp1_strides[i] = 0;
      else
        inp2_strides[i] = 0;

      need_broadcast = 1;
    }
    if(p_inp1_shape[i] != 1)
      inp1_const &= 0;
    if(p_inp2_shape[i] != 1)
      inp2_const &= 0;
  }
  int itr0, itr1, itr2;

  WORD8 *p_out_tmp = p_out;
  const WORD8 *__restrict__ p_inp1_tmp = p_inp1;
  const WORD8 *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0)
  {
    internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(
                p_out,
                out_zero_bias,
                out_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1,
                inp1_zero_bias,
                p_inp2,
                inp2_zero_bias,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    WORD32 inp1_zb;
    WORD32 inp2_zb;

    inp1_zb = inp1_zero_bias;
    inp2_zb = inp2_zero_bias;

    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp1_zb = inp2_zero_bias;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[2];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }
    else if(inp2_strides[2] == 0)
    {
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }

    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const WORD8 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD8 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_mul_broadcast_2D_asym8sxasym8s_asym8s(
            p_out_tmp,
            out_zero_bias,
            out_shift,
            out_multiplier,
            out_activation_min,
            out_activation_max,
            p_inp1_tmp0,
            inp1_zb,
            p_inp2_tmp0,
            inp2_zb,
            out_lc,
            in_lc);
        p_out_tmp += in_lc * out_lc;
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  else if(inp1_const == 1 || inp2_const == 1)
  {
    WORD32 inp1_zb;
    WORD32 inp2_zb;
    inp1_zb = inp1_zero_bias;
    inp2_zb = inp2_zero_bias;
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp1_zb = inp2_zero_bias;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }

    internal_elm_mul_broadcast_asym8sxasym8s_asym8s(
        p_out_tmp,
        out_zero_bias,
        out_shift,
        out_multiplier,
        out_activation_min,
        out_activation_max,
        p_inp1_tmp,
        inp1_zb,
        p_inp2_tmp,
        inp2_zb,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else
  {
    WORD32 inp1_zb;
    WORD32 inp2_zb;
    inp1_zb = inp1_zero_bias;
    inp2_zb = inp2_zero_bias;
    if(inp1_strides[3] == 0)
    {
      inp2_zb = inp1_zero_bias;
      inp1_zb = inp2_zero_bias;
      const WORD8 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;

      int tmp_strides[3];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      tmp_strides[2] = inp1_strides[2];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      inp1_strides[2] = inp2_strides[2];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      inp2_strides[2] = tmp_strides[2];
    }
    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const WORD8 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const WORD8 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const WORD8 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const WORD8 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_mul_broadcast_asym8sxasym8s_asym8s(
                p_out_tmp,
                out_zero_bias,
                out_shift,
                out_multiplier,
                out_activation_min,
                out_activation_max,
                p_inp1_tmp1,
                inp1_zb,
                p_inp2_tmp1,
                inp2_zb,
                p_out_shape[3]);
          }
          p_out_tmp += p_out_shape[3];
          p_inp1_tmp1 += inp1_strides[2];
          p_inp2_tmp1 += inp2_strides[2];
        }
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  return 0;
}
