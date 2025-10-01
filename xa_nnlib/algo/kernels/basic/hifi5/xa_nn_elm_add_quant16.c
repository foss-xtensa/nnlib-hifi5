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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_quant_macros_hifi5.h"
#include "xa_nnlib_common_bcast_macro.h"

#define SW_SLAA32S_INT32X2_INT32X2(inp1, inp2) AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(inp1), inp2))
#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)


static void internal_elm_add_broadcast_2D_asym16sxasym16s_asym16s(void * __restrict__ p_out,
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
  WORD16 * __restrict__ p_inp1_16 = (WORD16*)p_inp1;
  WORD16 * __restrict__ p_inp2_16 = (WORD16*)p_inp2;
  WORD16 *__restrict__ p_out_16 = (WORD16*)p_out;

  WORD16 * __restrict__ p_a;
  WORD16 * __restrict__ p_b;
  WORD16 *__restrict__ p_c;

  const ae_int16x4 za = AE_MOVDA16(-inp1_zero_bias);
  const ae_int16x4 zb = AE_MOVDA16(-inp2_zero_bias);

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_f32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;

  ae_int32x2 d_left_shift = SW_SLAA32S_INT32X2_INT32X2(SW_MOVDA32(-1), left_shift);

  ae_f32x2 out32_0, out32_1, out32_2, out32_3;

  ae_int16x4 out0, out1;

  int num_simd8_ops;
  int num_scalar_ops;

  num_simd8_ops = in_lc >> 3;
  num_scalar_ops = in_lc & 7;

  ae_valignx2 va_a, va_b, va_c;

#pragma loop_count min=1
  for(i = 0; i < out_lc; i++)
  {
    p_a = (WORD16 *)&p_inp1_16[i * in_lc];
    p_b = (WORD16 *)p_inp2_16;
    p_c = (WORD16 *)&p_out_16[i * in_lc];

    ae_int16x8 *p16x8_a = (ae_int16x8 *)p_a;
    ae_int16x8 *p16x8_b = (ae_int16x8 *)p_b;
    ae_int16x8 *p16x8_c = (ae_int16x8 *)p_c;
    
    va_a = AE_LA128_PP((ae_int8x16 *)p_a);
    va_b = AE_LA128_PP((ae_int8x16 *)p_b);
    va_c = AE_ZALIGN128();
    for(j = 0; j < num_simd8_ops; j++)
    {
      AE_LA16X4X2_IP(a0_3, a4_7, va_a, p16x8_a);
      AE_LA16X4X2_IP(b0_3, b4_7, va_b, p16x8_b);

      // Add input zero bias
      AE_SUBW16(shifted_a0_1, shifted_a2_3, za, a0_3);
      AE_SUBW16(shifted_a4_5, shifted_a6_7, za, a4_7);
      AE_SUBW16(shifted_b0_1, shifted_b2_3, zb, b0_3);
      AE_SUBW16(shifted_b4_5, shifted_b6_7, zb, b4_7);

      // LSH
      AE_MUL2P32X4S(shifted_a0_1, shifted_a2_3, shifted_a0_1, shifted_a2_3, d_left_shift, d_left_shift);
      AE_MUL2P32X4S(shifted_a4_5, shifted_a6_7, shifted_a4_5, shifted_a6_7, d_left_shift, d_left_shift);
      AE_MUL2P32X4S(shifted_b0_1, shifted_b2_3, shifted_b0_1, shifted_b2_3, d_left_shift, d_left_shift);
      AE_MUL2P32X4S(shifted_b4_5, shifted_b6_7, shifted_b4_5, shifted_b6_7, d_left_shift, d_left_shift);

      raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = AE_MOVF32X2_FROMINT32X2(AE_ZERO32());
      // Scaled input
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);

      // Raw Output
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_0, out32_1, AE_MOVINT32X2_FROMF32X2(raw_sum0_1), AE_MOVINT32X2_FROMF32X2(raw_sum2_3), out_multiplier, out_left_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_2, out32_3, AE_MOVINT32X2_FROMF32X2(raw_sum4_5), AE_MOVINT32X2_FROMF32X2(raw_sum6_7), out_multiplier, out_left_shift, out_zero_bias);

      out0 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_0), AE_MOVINT32X2_FROMF32X2(out32_1));
      out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_2), AE_MOVINT32X2_FROMF32X2(out32_3));
      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      AE_SA16X4X2_IP(out0, out1, va_c, p16x8_c);
    }
    AE_SA128POS_FP(va_c, p16x8_c);
    p_a = (WORD16 *)p16x8_a;
    p_b = (WORD16 *)p16x8_b;
    p_c = (WORD16 *)p16x8_c;
  }

  if(num_scalar_ops != 0)
  {
    ae_f32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;

    p_b = (WORD16 *)&p_inp2_16[num_simd8_ops << 3];
    ae_int16x8 *p16x8_b = (ae_int16x8 *)p_b;
    va_b = AE_LA128_PP((ae_int8x16 *)p_b);
    AE_LAV16X4X2_XP(b0_3, b4_7, va_b, p16x8_b, num_scalar_ops<<1);
    p_b = (WORD16 *)p16x8_b;
    AE_SUBW16(shifted_b0_1, shifted_b2_3, zb, b0_3);
    AE_SUBW16(shifted_b4_5, shifted_b6_7, zb, b4_7);

    AE_MUL2P32X4S(shifted_b0_1, shifted_b2_3, shifted_b0_1, shifted_b2_3, d_left_shift, d_left_shift);
    AE_MUL2P32X4S(shifted_b4_5, shifted_b6_7, shifted_b4_5, shifted_b6_7, d_left_shift, d_left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32(scaled_b0_1, scaled_b2_3, AE_MOVF32X2_FROMINT32X2(shifted_b0_1), AE_MOVF32X2_FROMINT32X2(shifted_b2_3), inp2_multiplier, inp2_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32(scaled_b4_5, scaled_b6_7, AE_MOVF32X2_FROMINT32X2(shifted_b4_5), AE_MOVF32X2_FROMINT32X2(shifted_b6_7), inp2_multiplier, inp2_left_shift);

#pragma loop_count min=1
    for(i = 0; i < out_lc; i++)
    {
      p_a = (WORD16 *)&p_inp1_16[i * in_lc + (num_simd8_ops << 3)];
      p_c = (WORD16 *)&p_out_16[i * in_lc + (num_simd8_ops << 3)];

      ae_int16x8 *p16x8_a = (ae_int16x8 *)p_a;      
      ae_int16x8 *p16x8_c = (ae_int16x8 *)p_c;
      va_a = AE_LA128_PP(p16x8_a);
      va_c = AE_ZALIGN128();

      AE_LAV16X4X2_XP(a0_3, a4_7, va_a, p16x8_a, num_scalar_ops<<1);

      // Add input zero bias
      AE_SUBW16(shifted_a0_1, shifted_a2_3, za, a0_3);
      AE_SUBW16(shifted_a4_5, shifted_a6_7, za, a4_7);

      // LSH
      AE_MUL2P32X4S(shifted_a0_1, shifted_a2_3, shifted_a0_1, shifted_a2_3, d_left_shift, d_left_shift);
      AE_MUL2P32X4S(shifted_a4_5, shifted_a6_7, shifted_a4_5, shifted_a6_7, d_left_shift, d_left_shift);

      raw_sum0_1 = scaled_b0_1;
      raw_sum2_3 = scaled_b2_3;
      raw_sum4_5 = scaled_b4_5;
      raw_sum6_7 = scaled_b6_7;
      // Scaled input
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);

      // Raw Output
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_0, out32_1, AE_MOVINT32X2_FROMF32X2(raw_sum0_1), AE_MOVINT32X2_FROMF32X2(raw_sum2_3), out_multiplier, out_left_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_2, out32_3, AE_MOVINT32X2_FROMF32X2(raw_sum4_5), AE_MOVINT32X2_FROMF32X2(raw_sum6_7), out_multiplier, out_left_shift, out_zero_bias);

      out0 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_0), AE_MOVINT32X2_FROMF32X2(out32_1));
      out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_2), AE_MOVINT32X2_FROMF32X2(out32_3));

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      AE_SAV16X4X2_XP(out0, out1, va_c, p16x8_c, num_scalar_ops<<1);
      AE_SA128POS_FP(va_c, p16x8_c);
    }
  }
}

static void internal_elm_add_broadcast_asym16sxasym16s_asym16s(void * __restrict__ p_out,
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
  ae_int16x8 * __restrict__ p_a = (ae_int16x8 *)p_inp1;
  WORD16 * __restrict__ p_b = (WORD16 *)p_inp2;
  ae_int16x8 *__restrict__ p_c =  (ae_int16x8*)p_out;

  ae_int16x4  za = AE_MOVDA16(-inp1_zero_bias);
  ae_int16x4  zb = AE_MOVDA16(-inp2_zero_bias);
  WORD32 a_ls, a_mult, b_ls, b_mult;
  a_ls = inp1_left_shift;
  a_mult = inp1_multiplier;
  b_ls = inp2_left_shift;
  b_mult = inp2_multiplier;

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b;

  ae_int32x2 d_left_shift = SW_SLAA32S_INT32X2_INT32X2(SW_MOVDA32(-1), left_shift);

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0, shifted_b1;
  ae_f32x2 scaled_b0;

  ae_f32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;

  ae_f32x2 out32_0, out32_1, out32_2, out32_3;
  ae_int16x4 out0, out1;

  const int num_simd8_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;

  ae_valignx2 va_a = AE_LA128_PP(p_a);
  ae_valignx2 va_c = AE_ZALIGN128();

  b = AE_MOVDA16(p_b[0]);
  AE_SUBW16(shifted_b0, shifted_b1, b, zb);
  shifted_b0 = SW_SLAA32S_INT32X2_INT32X2(shifted_b0, left_shift);
  MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);

  for(i=0; i<num_simd8_ops; i++)
  {
    AE_LA16X4X2_IP(a0_3, a4_7, va_a, p_a);

    // Add input zero bias
    AE_SUBW16(shifted_a0_1, shifted_a2_3, za, a0_3);
    AE_SUBW16(shifted_a4_5, shifted_a6_7, za, a4_7);

    // LSH (and promote to 32-bit)
    AE_MUL2P32X4S(shifted_a0_1, shifted_a2_3, shifted_a0_1, shifted_a2_3, d_left_shift, d_left_shift);
    AE_MUL2P32X4S(shifted_a4_5, shifted_a6_7, shifted_a4_5, shifted_a6_7, d_left_shift, d_left_shift);

    raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = scaled_b0;
    // Scaled input
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, a_mult, a_ls);

    // Raw Output
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_0, out32_1, AE_MOVINT32X2_FROMF32X2(raw_sum0_1), AE_MOVINT32X2_FROMF32X2(raw_sum2_3), out_multiplier, out_left_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_2, out32_3, AE_MOVINT32X2_FROMF32X2(raw_sum4_5), AE_MOVINT32X2_FROMF32X2(raw_sum6_7), out_multiplier, out_left_shift, out_zero_bias);

    out0 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_0), AE_MOVINT32X2_FROMF32X2(out32_1));
    out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_2), AE_MOVINT32X2_FROMF32X2(out32_3));

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    AE_SA16X4X2_IP(out0, out1, va_c, p_c);
  }
  AE_SA128POS_FP(va_c, p_c);

  if(num_scalar_ops != 0)
  {
    va_a = AE_LA128_PP((ae_int8x16 *)p_a);
    va_c = AE_ZALIGN128();

    AE_LAV16X4X2_XP(a0_3, a4_7, va_a, p_a, num_scalar_ops<<1);

    // Add input zero bias
    AE_SUBW16(shifted_a0_1, shifted_a2_3, za, a0_3);
    AE_SUBW16(shifted_a4_5, shifted_a6_7, za, a4_7);

    // LSH (and promote to 32-bit)
    AE_MUL2P32X4S(shifted_a0_1, shifted_a2_3, shifted_a0_1, shifted_a2_3, d_left_shift, d_left_shift);
    AE_MUL2P32X4S(shifted_a4_5, shifted_a6_7, shifted_a4_5, shifted_a6_7, d_left_shift, d_left_shift);

    raw_sum0_1 = raw_sum2_3 = raw_sum4_5 = raw_sum6_7 = scaled_b0;
    // Scaled input
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum0_1, raw_sum2_3, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
    MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_sum4_5, raw_sum6_7, shifted_a4_5, shifted_a6_7, a_mult, a_ls);

    // Raw Output
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_0, out32_1, AE_MOVINT32X2_FROMF32X2(raw_sum0_1), AE_MOVINT32X2_FROMF32X2(raw_sum2_3), out_multiplier, out_left_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32_ZB(out32_2, out32_3, AE_MOVINT32X2_FROMF32X2(raw_sum4_5), AE_MOVINT32X2_FROMF32X2(raw_sum6_7), out_multiplier, out_left_shift, out_zero_bias);

    out0 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_0), AE_MOVINT32X2_FROMF32X2(out32_1));
    out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(out32_2), AE_MOVINT32X2_FROMF32X2(out32_3));

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    AE_SAV16X4X2_XP(out0, out1, va_c, p_c, num_scalar_ops<<1);
    AE_SA128POS_FP(va_c, p_c);
  }
}

WORD32 xa_nn_elm_add_broadcast_4D_asym16sxasym16s_asym16s(WORD16 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD16 * __restrict__ p_inp2,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -32767) || (inp1_zero_bias > 32768)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -32767) || (inp2_zero_bias > 32768)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -32768) || (out_activation_min > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < -32768) || (out_activation_max > 32767)), -1);
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
  args.out_elm_size = args.inp_elm_size = 2;
  args.multiplier_sign = 1;

  return CALL_BCAST(internal_elm_add_broadcast_2D_asym16sxasym16s_asym16s, 
            internal_elm_add_broadcast_asym16sxasym16s_asym16s,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
