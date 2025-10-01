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

static void internal_elm_squared_diff_broadcast_2D_sym16sxsym16s_sym16s(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  out_left_shift = args->out_shift;
  WORD32  out_multiplier = args->out_multiplier;
  WORD32  out_activation_min = args->out_activation_min;
  WORD32  out_activation_max = args->out_activation_max;
  WORD32  inp1_left_shift = args->inp1_left_shift;
  WORD32  inp1_multiplier = args->inp1_multiplier;
  WORD32  inp2_left_shift = args->inp2_left_shift;
  WORD32  inp2_multiplier = args->inp2_multiplier;
  WORD32  left_shift = args->left_shift;
  WORD32  out_lc = args->out_lc;
  WORD32  in_lc = args->in_lc;
#if TFLITE_SINGLE_ROUNDING
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift;
  out_rs = out_left_shift;
  (void)out_rs;
#else
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift > 0 ? out_left_shift : 0;
  out_rs = out_left_shift < 0 ? -out_left_shift : 0;
#endif
  int i, j;
  WORD16 * __restrict__ p_inp1_16 = (WORD16*)p_inp1;
  WORD16 * __restrict__ p_inp2_16 = (WORD16*)p_inp2;
  WORD16 *__restrict__ p_out_16 = (WORD16*)p_out;
  ae_int16x8 * __restrict__ p_a;
  ae_int16x8 * __restrict__ p_b;
  ae_int16x8 *__restrict__ p_c;

  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

  ae_f32x2 raw_diff0_1, raw_diff2_3, raw_diff4_5, raw_diff6_7;

  ae_int16x4 out0, out1;

  int num_simd8_ops;
  int num_scalar_ops;

  num_simd8_ops = in_lc >> 3;
  num_scalar_ops = in_lc & 7;

  ae_valignx2 va_a, va_b, va_c;

#pragma loop_count min=1
  for(i = 0; i < out_lc; i++)
  {
    p_a = (ae_int16x8 *)&p_inp1_16[i * in_lc];
    p_b = (ae_int16x8 *)p_inp2_16;
    p_c = (ae_int16x8 *)&p_out_16[i * in_lc];

    va_a = AE_LA128_PP(p_a);
    va_b = AE_LA128_PP(p_b);
    va_c = AE_ZALIGN128();
    for(j = 0; j < num_simd8_ops; j++)
    {
      AE_LA16X4X2_IP(a0_3, a4_7, va_a, p_a);
      AE_LA16X4X2_IP(b0_3, b4_7, va_b, p_b);

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
    
      raw_diff0_1 = raw_diff2_3 = raw_diff4_5 = raw_diff6_7 = AE_MOVF32X2_FROMINT32X2(AE_ZERO32());
      // Scaled input
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MACC_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_b0_1, shifted_b2_3, inp2_multiplier, inp2_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_b4_5, shifted_b6_7, inp2_multiplier, inp2_left_shift);

      ae_int32x2 raw_diff0_1_i32x2, raw_diff2_3_i32x2, raw_diff4_5_i32x2, raw_diff6_7_i32x2;
      AE_MUL2P32X4S(raw_diff0_1_i32x2, raw_diff2_3_i32x2, AE_MOVINT32X2_FROMF32X2(raw_diff0_1), AE_MOVINT32X2_FROMF32X2(raw_diff2_3), AE_MOVINT32X2_FROMF32X2(raw_diff0_1), AE_MOVINT32X2_FROMF32X2(raw_diff2_3));
      AE_MUL2P32X4S(raw_diff4_5_i32x2, raw_diff6_7_i32x2, AE_MOVINT32X2_FROMF32X2(raw_diff4_5), AE_MOVINT32X2_FROMF32X2(raw_diff6_7), AE_MOVINT32X2_FROMF32X2(raw_diff4_5), AE_MOVINT32X2_FROMF32X2(raw_diff6_7));

      raw_diff0_1 = AE_MOVF32X2_FROMINT32X2(raw_diff0_1_i32x2);
      raw_diff2_3 = AE_MOVF32X2_FROMINT32X2(raw_diff2_3_i32x2);
      raw_diff4_5 = AE_MOVF32X2_FROMINT32X2(raw_diff4_5_i32x2);
      raw_diff6_7 = AE_MOVF32X2_FROMINT32X2(raw_diff6_7_i32x2);

      // Raw Output
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out0, raw_diff0_1_i32x2, raw_diff2_3_i32x2, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out1, raw_diff4_5_i32x2, raw_diff6_7_i32x2, out_multiplier, out_ls, out_rs);

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      AE_SA16X4X2_IP(out0, out1, va_c, p_c);
    }
    AE_SA128POS_FP(va_c, p_c);
  }

  if(num_scalar_ops != 0)
  {
    ae_f32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;

    p_b = (ae_int16x8 *)&p_inp2_16[num_simd8_ops << 3];
    va_b = AE_LA128_PP(p_b);
    AE_LAV16X4X2_XP(b0_3, b4_7, va_b, p_b, (num_scalar_ops << 1));

    ae_f32x2 shifted_b0_1_t;
    ae_f32x2 shifted_b2_3_t;
    ae_f32x2 shifted_b4_5_t;
    ae_f32x2 shifted_b6_7_t;
    
    AE_CVTA32X4F16S(shifted_b0_1_t, shifted_b2_3_t, b0_3, left_shift);
    AE_CVTA32X4F16S(shifted_b4_5_t, shifted_b6_7_t, b4_7, left_shift);

    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32(scaled_b0_1, scaled_b2_3, shifted_b0_1_t, shifted_b2_3_t, inp2_multiplier, inp2_left_shift);
    MPY_BY_QUANT_MULT_ST_ONE_EXP_X2X2_OUT32(scaled_b4_5, scaled_b6_7, shifted_b4_5_t, shifted_b6_7_t, inp2_multiplier, inp2_left_shift);

#pragma loop_count min=1
    for(i = 0; i < out_lc; i++)
    {
      p_a = (ae_int16x8 *)&p_inp1_16[i * in_lc + (num_simd8_ops << 3)];
      p_c = (ae_int16x8 *)&p_out_16[i * in_lc + (num_simd8_ops << 3)];

      va_a = AE_LA128_PP(p_a);
      va_c = AE_ZALIGN128();

      AE_LAV16X4X2_XP(a0_3, a4_7, va_a, p_a, (num_scalar_ops << 1));

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
      
      raw_diff0_1 = scaled_b0_1;
      raw_diff2_3 = scaled_b2_3;
      raw_diff4_5 = scaled_b4_5;
      raw_diff6_7 = scaled_b6_7;

      // Scaled input
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1, raw_diff2_3, shifted_a0_1, shifted_a2_3, inp1_multiplier, inp1_left_shift);
      MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5, raw_diff6_7, shifted_a4_5, shifted_a6_7, inp1_multiplier, inp1_left_shift);

      ae_int32x2 raw_diff0_1_i32x2, raw_diff2_3_i32x2, raw_diff4_5_i32x2, raw_diff6_7_i32x2;
      AE_MUL2P32X4S(raw_diff0_1_i32x2, raw_diff2_3_i32x2, AE_MOVINT32X2_FROMF32X2(raw_diff0_1), AE_MOVINT32X2_FROMF32X2(raw_diff2_3), AE_MOVINT32X2_FROMF32X2(raw_diff0_1), AE_MOVINT32X2_FROMF32X2(raw_diff2_3));
      AE_MUL2P32X4S(raw_diff4_5_i32x2, raw_diff6_7_i32x2, AE_MOVINT32X2_FROMF32X2(raw_diff4_5), AE_MOVINT32X2_FROMF32X2(raw_diff6_7), AE_MOVINT32X2_FROMF32X2(raw_diff4_5), AE_MOVINT32X2_FROMF32X2(raw_diff6_7));

      // Raw Output
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out0, raw_diff0_1_i32x2, raw_diff2_3_i32x2, out_multiplier, out_ls, out_rs);
      MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out1, raw_diff4_5_i32x2, raw_diff6_7_i32x2, out_multiplier, out_ls, out_rs);

      // Clamp output
      AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
      AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

      AE_SAV16X4X2_XP(out0, out1, va_c, p_c, (num_scalar_ops << 1));
      AE_SA128POS_FP(va_c, p_c);
    }
  }
}

static void internal_elm_squared_diff_broadcast_sym16sxsym16s_sym16s(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  out_left_shift = args->out_shift;
  WORD32  out_multiplier = args->out_multiplier;
  WORD32  out_activation_min = args->out_activation_min;
  WORD32  out_activation_max = args->out_activation_max;
  WORD32  inp1_left_shift = args->inp1_left_shift;
  WORD32  inp1_multiplier = args->inp1_multiplier;
  WORD32  inp2_left_shift = args->inp2_left_shift;
  WORD32  inp2_multiplier = args->inp2_multiplier;
  WORD32  left_shift = args->left_shift;
  WORD32  num_elm = args->num_elm;

#if TFLITE_SINGLE_ROUNDING
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift;
  out_rs = out_left_shift;
  (void)out_rs;
#else
  WORD32 out_ls, out_rs;
  out_ls = out_left_shift > 0 ? out_left_shift : 0;
  out_rs = out_left_shift < 0 ? -out_left_shift : 0;
#endif
  int i;
  ae_int16x8 * __restrict__ p_a = (ae_int16x8 *)p_inp1;
  ae_int16x8 * __restrict__ p_c = (ae_int16x8 *)p_out;

  WORD32 a_ls, a_mult, b_ls, b_mult;
  a_ls = inp1_left_shift;
  a_mult = inp1_multiplier;
  b_ls = inp2_left_shift;
  b_mult = inp2_multiplier;

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0;

  ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
  ae_int32x2 shifted_b0;//, shifted_b1;
  ae_f32x2 scaled_b0;

  ae_int32x2 raw_diff0_1, raw_diff2_3, raw_diff4_5, raw_diff6_7;

  ae_int16x4 out0, out1;

  const int num_simd8_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;

  ae_valignx2 va_a = AE_LA128_PP(p_a);
  ae_valignx2 va_c = AE_ZALIGN128();

  b0 = AE_MOVDA16(((WORD16*)p_inp2)[0]);
  ae_f32x2 shifted_b0_t;
  ae_f32x2 shifted_b1_t;
  
  AE_CVTA32X4F16S(shifted_b0_t, shifted_b1_t, b0, left_shift);
  
  shifted_b0 = AE_MOVINT32X2_FROMF32X2(shifted_b0_t);
  //shifted_b1 = AE_MOVINT32X2_FROMF32X2(shifted_b1_t);
  
  MPY_BY_QUANT_MULT_ST_ONE_EXP_X2_OUT32(scaled_b0, shifted_b0, b_mult, b_ls);

  ae_int32x2 ALIGN(16) scaled_b[2];
  scaled_b[0] = scaled_b[1] = AE_MOVINT32X2_FROMF32X2(scaled_b0);

  ae_int32x4 *p_scaled_b = (ae_int32x4 *)scaled_b;
  for(i=0; i<num_simd8_ops; i++)
  {
    AE_LA16X4X2_IP(a0_3, a4_7, va_a, p_a);

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
    
    AE_L32X2X2_I(raw_diff0_1, raw_diff2_3, p_scaled_b, 0);
    AE_L32X2X2_IP(raw_diff4_5, raw_diff6_7, p_scaled_b, 0);

    ae_f32x2 raw_diff0_1_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff0_1); 
    ae_f32x2 raw_diff2_3_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff2_3); 
    ae_f32x2 raw_diff4_5_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff4_5); 
    ae_f32x2 raw_diff6_7_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff6_7); 
    // Scaled input
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1_f32x2, raw_diff2_3_f32x2, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5_f32x2, raw_diff6_7_f32x2, shifted_a4_5, shifted_a6_7, a_mult, a_ls);
   
    raw_diff0_1 = AE_MOVINT32X2_FROMF32X2(raw_diff0_1_f32x2); 
    raw_diff2_3 = AE_MOVINT32X2_FROMF32X2(raw_diff2_3_f32x2); 
    raw_diff4_5 = AE_MOVINT32X2_FROMF32X2(raw_diff4_5_f32x2); 
    raw_diff6_7 = AE_MOVINT32X2_FROMF32X2(raw_diff6_7_f32x2);  

    AE_MUL2P32X4S(raw_diff0_1, raw_diff2_3, raw_diff0_1, raw_diff2_3, raw_diff0_1, raw_diff2_3);
    AE_MUL2P32X4S(raw_diff4_5, raw_diff6_7, raw_diff4_5, raw_diff6_7, raw_diff4_5, raw_diff6_7);

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out0, raw_diff0_1, raw_diff2_3, out_multiplier, out_ls, out_rs);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out1, raw_diff4_5, raw_diff6_7, out_multiplier, out_ls, out_rs);

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    AE_SA16X4X2_IP(out0, out1, va_c, p_c);
  }
  AE_SA128POS_FP(va_c, p_c);

  if(num_scalar_ops != 0)
  {
    va_a = AE_LA128_PP(p_a);
    va_c = AE_ZALIGN128();

    AE_LAV16X4X2_XP(a0_3, a4_7, va_a, p_a, (num_scalar_ops << 1));

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
    
    raw_diff0_1 = raw_diff2_3 = raw_diff4_5 = raw_diff6_7 = AE_MOVINT32X2_FROMF32X2(scaled_b0);

    ae_f32x2 raw_diff0_1_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff0_1); 
    ae_f32x2 raw_diff2_3_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff2_3); 
    ae_f32x2 raw_diff4_5_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff4_5); 
    ae_f32x2 raw_diff6_7_f32x2 = AE_MOVF32X2_FROMINT32X2(raw_diff6_7); 

    // Scaled input
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff0_1_f32x2, raw_diff2_3_f32x2, shifted_a0_1, shifted_a2_3, a_mult, a_ls);
    MPY_BY_QUANT_MSUB_ST_ONE_EXP_X2X2_OUT32(raw_diff4_5_f32x2, raw_diff6_7_f32x2, shifted_a4_5, shifted_a6_7, a_mult, a_ls);

    raw_diff0_1 = AE_MOVINT32X2_FROMF32X2(raw_diff0_1_f32x2); 
    raw_diff2_3 = AE_MOVINT32X2_FROMF32X2(raw_diff2_3_f32x2); 
    raw_diff4_5 = AE_MOVINT32X2_FROMF32X2(raw_diff4_5_f32x2); 
    raw_diff6_7 = AE_MOVINT32X2_FROMF32X2(raw_diff6_7_f32x2);  

    AE_MUL2P32X4S(raw_diff0_1, raw_diff2_3, raw_diff0_1, raw_diff2_3, raw_diff0_1, raw_diff2_3);
    AE_MUL2P32X4S(raw_diff4_5, raw_diff6_7, raw_diff4_5, raw_diff6_7, raw_diff4_5, raw_diff6_7);

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out0, raw_diff0_1, raw_diff2_3, out_multiplier, out_ls, out_rs);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT16(out1, raw_diff4_5, raw_diff6_7, out_multiplier, out_ls, out_rs);

    // Clamp output
    AE_MINMAX16(out0, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
    AE_MINMAX16(out1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));

    AE_SAV16X4X2_XP(out0, out1, va_c, p_c, (num_scalar_ops << 1));
    AE_SA128POS_FP(va_c, p_c);
  }
}

WORD32 xa_nn_elm_squared_diff_broadcast_4D_sym16sxsym16s_sym16s(WORD16 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                      const WORD16 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
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
  XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND((left_shift != 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_min < -32768) || (out_activation_min > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_activation_max < out_activation_min) || (out_activation_max > 32767)), -1);

  bcast_args_t args = {0};
  args.out_shift = out_left_shift;
  args.out_multiplier = out_multiplier;
  args.out_activation_min = out_activation_min;
  args.out_activation_max = out_activation_max;
  args.out_elm_size = args.inp_elm_size = 2;
  args.multiplier_sign = 1;
  args.inp2_left_shift = inp2_left_shift;
  args.inp2_multiplier = inp2_multiplier;
  args.left_shift = left_shift;
  args.inp1_left_shift = inp1_left_shift;
  args.inp1_multiplier = inp1_multiplier;

  return CALL_BCAST(internal_elm_squared_diff_broadcast_2D_sym16sxsym16s_sym16s, 
            internal_elm_squared_diff_broadcast_sym16sxsym16s_sym16s,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
