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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_common_bcast_macro.h"

WORD32 xa_nn_elm_sub_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
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
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_multiplier < 0) || (inp1_multiplier < 0) || (inp2_multiplier < 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < 0) || (out_activation_min > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < 0) || (out_activation_max > 255)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    UWORD8 *out = p_out;
    WORD8 *p_i1 = (WORD8 *)p_inp1;
    WORD8 *p_i2 = (WORD8 *)p_inp2;

    ae_int16x4 x1, x2;
    ae_int32x2 temp;
    ae_int16x4 zero_bias1, zero_bias2;
    ae_int16x4 temp16X4;
    ae_int32x2 op_zero_bias, activation_min, activation_max;

    // Taking zero_bias into 16X4 variable
    temp = SW_MOVDA32(inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = AE_SEL16_6420(temp16X4, temp16X4);

    temp = SW_MOVDA32(inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = AE_SEL16_6420(temp16X4, temp16X4);

    op_zero_bias = SW_MOVDA32(out_zero_bias);

    activation_min = SW_MOVDA32(out_activation_min);
    activation_max = SW_MOVDA32(out_activation_max);

    if(((((unsigned)p_i1)&3) == 0) && ((((unsigned)p_i2)&3) == 0))
    {
        for(i=0;i < num_elm>>2;i++)
        {
            ae_int16x4 v1, v2;
            ae_int32x2 shifted_v1, shifted_v2;
            ae_int32x2 shifted_v3, shifted_v4;
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_int32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 clamped_out12, clamped_out34;

            ae_f16x4 x1_t;
            ae_f16x4 x2_t;
            AE_L8X4F_IP(x1_t, p_i1, 4*sizeof(WORD8));
            AE_L8X4F_IP(x2_t, p_i2, 4*sizeof(WORD8));

            x1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMF16X4(x1_t), 8));
            x2 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMF16X4(x2_t), 8));

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            shifted_v1 = AE_SEXT32X2D16_32(v1);
            shifted_v2 = AE_SEXT32X2D16_10(v1);
            shifted_v3 = AE_SEXT32X2D16_32(v2);
            shifted_v4 = AE_SEXT32X2D16_10(v2);

            shifted_v1 = AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(shifted_v1), left_shift));
            shifted_v2 = AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(shifted_v2), left_shift));
            shifted_v3 = AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(shifted_v3), left_shift));
            shifted_v4 = AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(shifted_v4), left_shift));


            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, shifted_v2, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, shifted_v4, inp2_multiplier, inp2_left_shift)

            // Raw Difference
            raw_sum12 = AE_MOVINT32X2_FROMF32X2(AE_SUB32S(scaled_v1, scaled_v3));
            raw_sum34 = AE_MOVINT32X2_FROMF32X2(AE_SUB32S(scaled_v2, scaled_v4));

            // Raw Output
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift)
            raw_out12 = AE_ADD32S(raw_out12, AE_MOVF32X2_FROMINT32X2(op_zero_bias));
            raw_out34 = AE_ADD32S(raw_out34, AE_MOVF32X2_FROMINT32X2(op_zero_bias));

            // clamped_out
            CLAMP_VAL(clamped_out12, AE_MOVINT32X2_FROMF32X2(raw_out12), activation_min, activation_max)
            CLAMP_VAL(clamped_out34, AE_MOVINT32X2_FROMF32X2(raw_out34), activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out12, clamped_out34)
        }
    }
    else
    {
        ALIGN_REGISTER_TYPE i1_a, i2_a;

        PRIME_8X4U(p_i1, i1_a);
        PRIME_8X4U(p_i2, i2_a);
        for(i=0;i < num_elm>>2;i++)
        {
            ae_int16x4 v1, v2;
            ae_int32x2 shifted_v1, shifted_v2;
            ae_int32x2 shifted_v3, shifted_v4;
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_int32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 clamped_out12, clamped_out34;


            AE_LA8X4U_IP(x1, i1_a, p_i1);
            AE_LA8X4U_IP(x2, i2_a, p_i2);

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            shifted_v1 = AE_SEXT32X2D16_32(v1);
            shifted_v2 = AE_SEXT32X2D16_10(v1);
            shifted_v3 = AE_SEXT32X2D16_32(v2);
            shifted_v4 = AE_SEXT32X2D16_10(v2);

            shifted_v1 = SW_SLAA32S_INT32X2_INT32X2(shifted_v1, left_shift);
            shifted_v2 = SW_SLAA32S_INT32X2_INT32X2(shifted_v2, left_shift);
            shifted_v3 = SW_SLAA32S_INT32X2_INT32X2(shifted_v3, left_shift);
            shifted_v4 = SW_SLAA32S_INT32X2_INT32X2(shifted_v4, left_shift);


            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v2, shifted_v2, inp1_multiplier, inp1_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v4, shifted_v4, inp2_multiplier, inp2_left_shift)

            // Raw Difference
            raw_sum12 = AE_MOVINT32X2_FROMF32X2(AE_SUB32S(scaled_v1, scaled_v3));
            raw_sum34 = AE_MOVINT32X2_FROMF32X2(AE_SUB32S(scaled_v2, scaled_v4));

            // Raw Output
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
            MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out34, raw_sum34, out_multiplier, out_left_shift)
            raw_out12 = AE_ADD32S(raw_out12, AE_MOVF32X2_FROMINT32X2(op_zero_bias));
            raw_out34 = AE_ADD32S(raw_out34, AE_MOVF32X2_FROMINT32X2(op_zero_bias));

            // clamped_out
            CLAMP_VAL(clamped_out12, AE_MOVINT32X2_FROMF32X2(raw_out12), activation_min, activation_max)
            CLAMP_VAL(clamped_out34, AE_MOVINT32X2_FROMF32X2(raw_out34), activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out12, clamped_out34)
        }
    }
    // Remainder Loop
    for(i=0; i < (num_elm & 3); i++)
    {
        ae_int16x4 v1, v2;
        ae_int32x2 shifted_v1;
        ae_int32x2 shifted_v3;
        ae_f32x2 scaled_v1;
        ae_f32x2 scaled_v3;
        ae_int32x2 raw_sum12;
        ae_f32x2 raw_out12;
        ae_f32x2 clamped_out12;

        WORD16 i1, i2;

        i1 = (WORD16) *((UWORD8 *)p_i1 + i);
        i2 = (WORD16) *((UWORD8 *)p_i2 + i);

        x1 = AE_MOVDA16(i1);
        x2 = AE_MOVDA16(i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        shifted_v1 = AE_SEXT32X2D16_32(v1);
        shifted_v3 = AE_SEXT32X2D16_32(v2);

        shifted_v1 = SW_SLAA32S_INT32X2_INT32X2(shifted_v1, left_shift);
        shifted_v3 = SW_SLAA32S_INT32X2_INT32X2(shifted_v3, left_shift);

        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v1, shifted_v1, inp1_multiplier, inp1_left_shift)
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_v3, shifted_v3, inp2_multiplier, inp2_left_shift)

        // Raw Difference
        raw_sum12 = AE_MOVINT32X2_FROMF32X2(AE_SUB32S(scaled_v1, scaled_v3));

        // Raw Output
        MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(raw_out12, raw_sum12, out_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, AE_MOVF32X2_FROMINT32X2(op_zero_bias));

        // clamped_out
        CLAMP_VAL(clamped_out12, AE_MOVINT32X2_FROMF32X2(raw_out12), activation_min, activation_max)

        // Store Output
        i1 = (WORD16)AE_MOVAD32_H(AE_MOVINT32X2_FROMF32X2(clamped_out12));
        *out++ = (UWORD8) i1;
    }

    return 0;
}

static void internal_elm_sub_broadcast_2D_asym8sxasym8s_asym8s(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  out_zero_bias = args->out_zero_bias;
  WORD32  out_left_shift = args->out_shift;
  WORD32  out_multiplier = args->out_multiplier;
  WORD32  out_activation_min = args->out_activation_min;
  WORD32  out_activation_max = args->out_activation_max;
  WORD32  inp1_zero_bias = args->inp1_zero_bias;
  WORD32  inp1_left_shift = args->inp1_left_shift;
  WORD32  inp1_multiplier = args->inp1_multiplier;
  WORD32  inp2_zero_bias = args->inp2_zero_bias;
  WORD32  inp2_left_shift = args->inp2_left_shift;
  WORD32  inp2_multiplier = args->inp2_multiplier;
  WORD32  left_shift = args->left_shift;
  WORD32  out_lc = args->out_lc;
  WORD32  in_lc = args->in_lc;
  
  int i, j;
  WORD8 *p_inp1_8 = (WORD8*) p_inp1;
  WORD8 *p_inp2_8 = (WORD8*) p_inp2;
  WORD8 *p_out_8 = (WORD8*) p_out;
  WORD8 * __restrict__ p_a;
  WORD8 * __restrict__ p_b;
  WORD8 *__restrict__ p_c;

  ae_int8x8 a0_7, b0_7;

  const ae_int8x8 za = AE_MOVDA8(-inp1_zero_bias);
  const ae_int8x8 zb = AE_MOVDA8(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_f32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;

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
    p_a = (WORD8 *)&p_inp1_8[i * in_lc];
    p_b = (WORD8 *)p_inp2_8;
    p_c = (WORD8 *)&p_out_8[i * in_lc];

    ae_int8x8 *p8x8_a = (ae_int8x8 *)p_a;
    ae_int8x8 *p8x8_b = (ae_int8x8 *)p_b;
    ae_int8x8 *p8x8_c = (ae_int8x8 *)p_c;
    va_a = AE_LA64_PP(p8x8_a);
    va_b = AE_LA64_PP(p8x8_b);
    va_c = AE_ZALIGN64();
    
    for(j = 0; j < num_simd8_ops; j++)
    {
      AE_LA8X8_IP(a0_7, va_a, p8x8_a);
      AE_LA8X8_IP(b0_7, va_b, p8x8_b);

      // Add input zero bias
      AE_SUBW8(a0_3, a4_7, a0_7, za);
      AE_SUBW8(b0_3, b4_7, b0_7, zb);

      ae_f32x2 shifted_a0_1_t;
      ae_f32x2 shifted_a2_3_t;
      ae_f32x2 shifted_a4_5_t;
      ae_f32x2 shifted_a6_7_t;
      
      ae_f32x2 shifted_b0_1_t;
      ae_f32x2 shifted_b2_3_t;
      ae_f32x2 shifted_b4_5_t;
      ae_f32x2 shifted_b6_7_t;
      
      // LSH (and promote to 32-bit)
      AE_CVTA32X4F16S(shifted_a0_1_t, shifted_a2_3_t, a0_3, left_shift);
      AE_CVTA32X4F16S(shifted_a4_5_t, shifted_a6_7_t, a4_7, left_shift);

      AE_CVTA32X4F16S(shifted_b0_1_t, shifted_b2_3_t, b0_3, left_shift);
      AE_CVTA32X4F16S(shifted_b4_5_t, shifted_b6_7_t, b4_7, left_shift);

      shifted_a0_1 = AE_MOVINT32X2_FROMF32X2(shifted_a0_1_t);
      shifted_a2_3 = AE_MOVINT32X2_FROMF32X2(shifted_a2_3_t);
      shifted_a4_5 = AE_MOVINT32X2_FROMF32X2(shifted_a4_5_t);
      shifted_a6_7 = AE_MOVINT32X2_FROMF32X2(shifted_a6_7_t);
      
      shifted_b0_1 = AE_MOVINT32X2_FROMF32X2(shifted_b0_1_t);
      shifted_b2_3 = AE_MOVINT32X2_FROMF32X2(shifted_b2_3_t);
      shifted_b4_5 = AE_MOVINT32X2_FROMF32X2(shifted_b4_5_t);
      shifted_b6_7 = AE_MOVINT32X2_FROMF32X2(shifted_b6_7_t);      
      raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = AE_MOVF32X2_FROMINT32X2(AE_ZERO32());
      // Scaled input
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);

      // Raw Output
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift, out_zero_bias);

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      ae_int8x8 res = AE_SAT8X8X16(out0, out1);

      AE_SA8X8_IP(res, va_c, p8x8_c);
    }
    AE_SA64POS_FP(va_c, p8x8_c);
    p_a = (WORD8 *)p8x8_a;
    p_b = (WORD8 *)p8x8_b;
    p_c = (WORD8 *)p8x8_c;
  }

  if(num_scalar_ops != 0)
  {
    ae_int8x8 a8_15, b8_15;
    ae_int16x4 out2, out3;
    ae_valignx2 va_ax2, va_bx2, va_cx2;
    ae_int16x4 a8_11, a12_15, b8_11, b12_15;
    ae_int32x2 shifted_a8_9, shifted_a10_11, shifted_a12_13, shifted_a14_15;
    ae_int32x2 shifted_b8_9, shifted_b10_11, shifted_b12_13, shifted_b14_15;
    ae_f32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;
    ae_f32x2 scaled_b8_9, scaled_b10_11, scaled_b12_13, scaled_b14_15;
    ae_f32x2 raw_sum8_9, raw_sum10_11, raw_sum12_13, raw_sum14_15;

    p_b = (WORD8 *)&p_inp2_8[num_simd8_ops << 3];
    ae_int8x16 *p8x16_b = (ae_int8x16 *)p_b;

    va_bx2 = AE_LA128_PP(p8x16_b);
    AE_LAV8X8X2_XP(b0_7, b8_15, va_bx2, p8x16_b, num_scalar_ops);
    p_b = (WORD8*)p8x16_b;
    AE_SUBW8(b0_3, b4_7, b0_7, zb);
    AE_SUBW8(b8_11, b12_15, b8_15, zb);

    ae_f32x2 shifted_b0_1_t;
    ae_f32x2 shifted_b2_3_t;
    ae_f32x2 shifted_b4_5_t;
    ae_f32x2 shifted_b6_7_t;
    
    ae_f32x2 shifted_b8_9_t;
    ae_f32x2 shifted_b10_11_t;
    ae_f32x2 shifted_b12_13_t;
    ae_f32x2 shifted_b14_15_t;
        
    AE_CVTA32X4F16S(shifted_b0_1_t, shifted_b2_3_t, b0_3, left_shift);
    AE_CVTA32X4F16S(shifted_b4_5_t, shifted_b6_7_t, b4_7, left_shift);
    AE_CVTA32X4F16S(shifted_b8_9_t, shifted_b10_11_t, b8_11, left_shift);
    AE_CVTA32X4F16S(shifted_b12_13_t, shifted_b14_15_t, b12_15, left_shift);

    shifted_b0_1 = AE_MOVINT32X2_FROMF32X2(shifted_b0_1_t);
    shifted_b2_3 = AE_MOVINT32X2_FROMF32X2(shifted_b2_3_t);
    shifted_b4_5 = AE_MOVINT32X2_FROMF32X2(shifted_b4_5_t);
    shifted_b6_7 = AE_MOVINT32X2_FROMF32X2(shifted_b6_7_t);
      
    shifted_b8_9 = AE_MOVINT32X2_FROMF32X2(shifted_b8_9_t);
    shifted_b10_11 = AE_MOVINT32X2_FROMF32X2(shifted_b10_11_t);
    shifted_b12_13 = AE_MOVINT32X2_FROMF32X2(shifted_b12_13_t);
    shifted_b14_15 = AE_MOVINT32X2_FROMF32X2(shifted_b14_15_t);
      
    scaled_b0_1 = scaled_b2_3 = scaled_b4_5 = scaled_b6_7 = AE_MOVF32X2_FROMINT32X2(AE_ZERO32());
    scaled_b8_9 = scaled_b10_11 = scaled_b12_13 = scaled_b14_15 = AE_MOVF32X2_FROMINT32X2(AE_ZERO32());
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(scaled_b0_1, scaled_b2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(scaled_b4_5, scaled_b6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(scaled_b8_9, scaled_b10_11, shifted_b8_9, shifted_b10_11, inp2_multiplier, inp2_left_shift);
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(scaled_b12_13, scaled_b14_15, shifted_b12_13, shifted_b14_15, inp2_multiplier, inp2_left_shift);

#pragma loop_count min=1
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD8 *)&p_inp1_8[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD8 *)&p_out_8[i * in_lc + (num_simd8_ops << 3)];

      ae_int8x16 *p8x16_a = (ae_int8x16 *)p_a;
      ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
      va_ax2 = AE_LA128_PP(p8x16_a);
      va_cx2 = AE_ZALIGN128();

      AE_LAV8X8X2_XP(a0_7, a8_15, va_ax2, p8x16_a, num_scalar_ops);

      // Add input zero bias
      AE_SUBW8(a0_3, a4_7, a0_7, za);
      AE_SUBW8(a8_11, a12_15, a8_15, za);

      ae_f32x2 shifted_a0_1_t;
      ae_f32x2 shifted_a2_3_t;
      ae_f32x2 shifted_a4_5_t;
      ae_f32x2 shifted_a6_7_t;
      
      ae_f32x2 shifted_a8_9_t;
      ae_f32x2 shifted_a10_11_t;
      ae_f32x2 shifted_a12_13_t;
      ae_f32x2 shifted_a14_15_t;
      // LSH (and promote to 32-bit)
      AE_CVTA32X4F16S(shifted_a0_1_t, shifted_a2_3_t, a0_3, left_shift);
      AE_CVTA32X4F16S(shifted_a4_5_t, shifted_a6_7_t, a4_7, left_shift);
      AE_CVTA32X4F16S(shifted_a8_9_t, shifted_a10_11_t, a8_11, left_shift);
      AE_CVTA32X4F16S(shifted_a12_13_t, shifted_a14_15_t, a12_15, left_shift);

      shifted_a0_1 = AE_MOVINT32X2_FROMF32X2(shifted_a0_1_t);
      shifted_a2_3 = AE_MOVINT32X2_FROMF32X2(shifted_a2_3_t);
      shifted_a4_5 = AE_MOVINT32X2_FROMF32X2(shifted_a4_5_t);
      shifted_a6_7 = AE_MOVINT32X2_FROMF32X2(shifted_a6_7_t);
      
      shifted_a8_9 = AE_MOVINT32X2_FROMF32X2(shifted_a8_9_t);
      shifted_a10_11 = AE_MOVINT32X2_FROMF32X2(shifted_a10_11_t);
      shifted_a12_13 = AE_MOVINT32X2_FROMF32X2(shifted_a12_13_t);
      shifted_a14_15 = AE_MOVINT32X2_FROMF32X2(shifted_a14_15_t);
      raw_sum0_1 = scaled_b0_1;
      raw_sum2_3 = scaled_b2_3;
      raw_sum4_5 = scaled_b4_5;
      raw_sum6_7 = scaled_b6_7;
      raw_sum8_9 = scaled_b8_9;
      raw_sum10_11 = scaled_b10_11;
      raw_sum12_13 = scaled_b12_13;
      raw_sum14_15 = scaled_b14_15;
      // Scaled input
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum8_9, raw_sum10_11, shifted_a8_9, shifted_a10_11, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum12_13, raw_sum14_15, shifted_a12_13, shifted_a14_15, inp1_multiplier, inp1_left_shift);

      // Raw Output
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out2, raw_sum8_9, raw_sum10_11, out_multiplier, out_left_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out3, raw_sum12_13, raw_sum14_15, out_multiplier, out_left_shift, out_zero_bias);

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out2, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out3, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      ae_int8x8 res0 = AE_SAT8X8X16(out0, out1);
      ae_int8x8 res1 = AE_SAT8X8X16(out2, out3);

      AE_SAV8X8X2_XP(res0, res1, va_cx2, p8x16_c, num_scalar_ops);
      AE_SA128POS_FP(va_cx2, p8x16_c);
    }
  }
}

static void internal_elm_sub_broadcast_asym8sxasym8s_asym8s(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  out_zero_bias = args->out_zero_bias;
  WORD32  out_left_shift = args->out_shift;
  WORD32  out_multiplier = args->out_multiplier;
  WORD32  out_activation_min = args->out_activation_min;
  WORD32  out_activation_max = args->out_activation_max;
  WORD32  inp1_zero_bias = args->inp1_zero_bias;
  WORD32  inp1_left_shift = args->inp1_left_shift;
  WORD32  inp1_multiplier = args->inp1_multiplier;
  WORD32  inp2_zero_bias = args->inp2_zero_bias;
  WORD32  inp2_left_shift = args->inp2_left_shift;
  WORD32  inp2_multiplier = args->inp2_multiplier;
  WORD32  left_shift = args->left_shift;
  WORD32  num_elm = args->num_elm;
  int i;
  WORD8 * __restrict__ p_a = (WORD8 *)p_inp1;
  WORD8 * __restrict__ p_b = (WORD8 *)p_inp2;
  WORD8 *__restrict__ p_c =          p_out;

  ae_int8x8 a0_7, b;

  ae_int8x8  za = AE_MOVDA8(-inp1_zero_bias);
  ae_int8x8  zb = AE_MOVDA8(-inp2_zero_bias);
  WORD32 a_ls, a_mult, b_ls, b_mult;
  a_ls = inp1_left_shift;
  a_mult = inp1_multiplier;
  b_ls = inp2_left_shift;
  b_mult = inp2_multiplier;

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0, b1;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0;//, shifted_b1;
  ae_f32x2 scaled_b0;

  ae_f32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;

  ae_int16x4 out0, out1;

  const int num_simd8_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;

  ae_int8x8 *p8x8_a = (ae_int8x8 *)p_a;    
  ae_int8x8 *p8x8_c = (ae_int8x8 *)p_c;
  ae_valign va_a = AE_LA64_PP(p8x8_a);
  ae_valign va_c = AE_ZALIGN64();

  b = AE_MOVDA8(p_b[0]);
  AE_SUBW8(b0, b1, b, zb);
  
  ae_f32x2 shifted_b0_t;
  ae_f32x2 shifted_b1_t;
  AE_CVTA32X4F16S(shifted_b0_t, shifted_b1_t, b0, left_shift);
  
  shifted_b0 = AE_MOVINT32X2_FROMF32X2(shifted_b0_t);
  //shifted_b1 = AE_MOVINT32X2_FROMF32X2(shifted_b1_t);
  
  MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);
  scaled_b0 = AE_NEG32S(scaled_b0);
  
  for(i=0; i<num_simd8_ops; i++)
  {
    AE_LA8X8_IP(a0_7, va_a, p8x8_a);

    // Add input zero bias
    AE_SUBW8(a0_3, a4_7, a0_7, za);

    ae_f32x2 shifted_a0_1_t;
    ae_f32x2 shifted_a2_3_t;
    ae_f32x2 shifted_a4_5_t;
    ae_f32x2 shifted_a6_7_t;
    // LSH (and promote to 32-bit)
    AE_CVTA32X4F16S(shifted_a0_1_t, shifted_a2_3_t, a0_3, left_shift);
    AE_CVTA32X4F16S(shifted_a4_5_t, shifted_a6_7_t, a4_7, left_shift);

    shifted_a0_1 = AE_MOVINT32X2_FROMF32X2(shifted_a0_1_t);
    shifted_a2_3 = AE_MOVINT32X2_FROMF32X2(shifted_a2_3_t);
    shifted_a4_5 = AE_MOVINT32X2_FROMF32X2(shifted_a4_5_t);
    shifted_a6_7 = AE_MOVINT32X2_FROMF32X2(shifted_a6_7_t);
    raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = scaled_b0;
    // Scaled input
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, a_mult, a_ls);

    // Raw Output
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift, out_zero_bias);

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    ae_int8x8 res = AE_SAT8X8X16(out0, out1);

    AE_SA8X8_IP(res, va_c,p8x8_c);
  }
  AE_SA64POS_FP(va_c, p8x8_c);

  ae_int8x16 *p8x16_a = (ae_int8x16 *)p8x8_a;
  ae_int8x16 *p8x16_c = (ae_int8x16 *)p8x8_c;
  if(num_scalar_ops != 0)
  {
    ae_valignx2 va_ax2, va_cx2;

    va_ax2 = AE_LA128_PP(p8x16_a);
    va_cx2 = AE_ZALIGN128();

    ae_int8x8 at0;

    AE_LAV8X8X2_XP(a0_7, at0, va_ax2, p8x16_a, num_scalar_ops);

    // Add input zero bias
    AE_SUBW8(a0_3, a4_7, a0_7, za);

    ae_f32x2 shifted_a0_1_t;
    ae_f32x2 shifted_a2_3_t;
    ae_f32x2 shifted_a4_5_t;
    ae_f32x2 shifted_a6_7_t;
    
    // LSH (and promote to 32-bit)
    AE_CVTA32X4F16S(shifted_a0_1_t, shifted_a2_3_t, a0_3, left_shift);
    AE_CVTA32X4F16S(shifted_a4_5_t, shifted_a6_7_t, a4_7, left_shift);

    shifted_a0_1 = AE_MOVINT32X2_FROMF32X2(shifted_a0_1_t);
    shifted_a2_3 = AE_MOVINT32X2_FROMF32X2(shifted_a2_3_t);
    shifted_a4_5 = AE_MOVINT32X2_FROMF32X2(shifted_a4_5_t);
    shifted_a6_7 = AE_MOVINT32X2_FROMF32X2(shifted_a6_7_t);
    raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = scaled_b0;
    // Scaled input
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, a_mult, a_ls);

    // Raw Output
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out0, raw_sum0_1, raw_sum2_3, out_multiplier, out_left_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT16_ZB(out1, raw_sum4_5, raw_sum6_7, out_multiplier, out_left_shift, out_zero_bias);

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    ae_int8x8 res = AE_SAT8X8X16(out0, out1);

    AE_SAV8X8X2_XP(res, res, va_cx2, p8x16_c, num_scalar_ops);
    AE_SA128POS_FP(va_cx2, p8x16_c);
  }
  p_a = (WORD8 *)p8x16_a;
  p_c = (WORD8 *)p8x16_c;
}

WORD32 xa_nn_elm_sub_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
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
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  bcast_args_t args = {0};
  args.out_zero_bias = out_zero_bias;
  args.out_shift = out_left_shift;
  args.out_multiplier = out_multiplier;
  args.out_activation_min = out_activation_min;
  args.out_activation_max = out_activation_max;
  args.inp1_zero_bias = inp1_zero_bias;
  args.inp1_left_shift = inp1_left_shift;
  args.inp1_multiplier = inp1_multiplier;
  args.inp2_zero_bias = inp2_zero_bias;
  args.inp2_left_shift = inp2_left_shift;
  args.inp2_multiplier = inp2_multiplier;
  args.left_shift = left_shift;
  args.num_elm = num_elm;
  args.out_lc = 1;
  args.in_lc = num_elm;

  internal_elm_sub_broadcast_2D_asym8sxasym8s_asym8s(
      p_out,
      p_inp1,
      p_inp2,
      &args);

  return 0;
}

WORD32 xa_nn_elm_sub_broadcast_4D_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD8 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD8 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift)
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
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

  bcast_args_t args = {0};
  args.out_zero_bias = out_zero_bias;
  args.out_shift = out_left_shift;
  args.out_multiplier = out_multiplier;
  args.out_activation_min = out_activation_min;
  args.out_activation_max = out_activation_max;
  args.inp1_zero_bias = inp1_zero_bias;
  args.inp1_left_shift = inp1_left_shift;
  args.inp1_multiplier = inp1_multiplier;
  args.inp2_zero_bias = inp2_zero_bias;
  args.inp2_left_shift = inp2_left_shift;
  args.inp2_multiplier = inp2_multiplier;
  args.left_shift = left_shift;
  args.out_elm_size = args.inp_elm_size = 1;
  args.multiplier_sign = -1;

  return CALL_BCAST(internal_elm_sub_broadcast_2D_asym8sxasym8s_asym8s, 
            internal_elm_sub_broadcast_asym8sxasym8s_asym8s,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
