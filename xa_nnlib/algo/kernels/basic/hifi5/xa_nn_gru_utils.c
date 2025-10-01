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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_quant_macros_hifi5.h"

#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#define SW_ADD32S_INT32X2_INT32X2(inp1, inp2) AE_MOVINT32X2_FROMF32X2(AE_ADD32S(AE_MOVF32X2_FROMINT32X2(inp1), AE_MOVF32X2_FROMINT32X2(inp2)));

WORD32 xa_nn_gru_hidden_state_update_8(WORD8* p_hidden_state,
                                      const WORD16* p_update_gate,
                                     const WORD16* p_modulated_state,
                                      WORD32 update_to_modulated_state_multiplier,
                                      WORD32 update_to_modulated_state_shift,
                                      WORD32 update_to_hidden_state_multiplier,
                                      WORD32 update_to_hidden_state_shift,
                                      WORD32 out_multiplier,
                                      WORD32 out_shift,
                                      WORD32 hidden_zero_bias,
                                      WORD32 num_elms)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_hidden_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_update_gate, -1);
  XA_NNLIB_ARG_CHK_PTR(p_modulated_state, -1);
  /* Pointer alignment checks */ 
  XA_NNLIB_ARG_CHK_ALIGN(p_hidden_state, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_update_gate, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_modulated_state, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((update_to_modulated_state_shift < -31 || update_to_modulated_state_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((update_to_hidden_state_shift < -31 || update_to_hidden_state_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((num_elms <= 0),-1);
  XA_NNLIB_ARG_CHK_COND((hidden_zero_bias < -128 || hidden_zero_bias > 127), -1);

  int left_shift_utm, right_shift_utm;
  int left_shift_uth, right_shift_uth;
  int left_shift, right_shift;

  #if TFLITE_SINGLE_ROUNDING
    left_shift_utm = update_to_modulated_state_shift;
    right_shift_utm = update_to_modulated_state_shift;

    left_shift_uth = update_to_hidden_state_shift;
    right_shift_uth = update_to_hidden_state_shift;
  
    left_shift = out_shift;
    right_shift = out_shift;

  #if XCHAL_HAVE_HIFI5S
    left_shift_utm = 31 - left_shift_utm;
    left_shift_utm = (left_shift_utm << 16) | left_shift_utm;

    left_shift_uth = 31 - left_shift_uth;
    left_shift_uth = (left_shift_uth << 16) | left_shift_uth;

    left_shift = 31 - left_shift;
    left_shift = (left_shift << 16) | left_shift;
  #endif  
    /* Single rounding macro doesn't need two shifts so this is not used */
    (void)right_shift_utm;
    (void)right_shift_uth;
    (void)right_shift;

  #else /* #if TFLITE_SINGLE_ROUNDING */
    left_shift_utm = update_to_modulated_state_shift > 0 ? update_to_modulated_state_shift : 0;
    right_shift_utm = update_to_modulated_state_shift < 0 ? -update_to_modulated_state_shift : 0;

    left_shift_uth = update_to_hidden_state_shift > 0 ? update_to_hidden_state_shift : 0;
    right_shift_uth = update_to_hidden_state_shift < 0 ? -update_to_hidden_state_shift : 0;
  
    left_shift = out_shift > 0 ? out_shift : 0;
    right_shift = out_shift < 0 ? -out_shift : 0;
  #endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int32x2 int16_min = SW_MOVDA32(-32768);
  ae_int32x2 int16_max = SW_MOVDA32(32767);
  ae_int32x2 d_out_zero_bias = SW_MOVDA32(hidden_zero_bias);
  ae_int8x8 d_hidden_zero_bias = AE_MOVDA8(hidden_zero_bias);
  ae_int16x4 d_int16_max = AE_MOVDA16(32767);
  ae_int16x4 d_one16 = AE_MOVDA16(1);

  const ae_int8x8 * ptr_hidden_in = (const ae_int8x8 *)p_hidden_state;
  const ae_int16x8 * ptr_update = (const ae_int16x8 *)p_update_gate;
  const ae_int16x8 * ptr_modulated = (const ae_int16x8 *)p_modulated_state;
  ae_int8x8 * ptr_hidden_out = (ae_int8x8 *)p_hidden_state;

  ae_valign a_hid_in = AE_LA64_PP(ptr_hidden_in);
  ae_valignx2 a_up = AE_LA128_PP(ptr_update);
  ae_valignx2 a_mod = AE_LA128_PP(ptr_modulated);
  ae_valign a_hid_out = AE_ZALIGN64();

  ae_int8x8 d_hidden_in, d_hidden_out;
  ae_int16x4 d_hidden_in1, d_hidden_in2;
  ae_int16x4 d_hidden_out1, d_hidden_out2;
  ae_int16x4 d_update1, d_update2, d_modulated1, d_modulated2;
  ae_int16x4 d_one_minus_update1, d_one_minus_update2;
  
  ae_int32x2 d_update_times_modulated11, d_update_times_modulated12, d_update_times_modulated21, d_update_times_modulated22;
  ae_int32x2 q_update_times_modulated11, q_update_times_modulated12, q_update_times_modulated21, q_update_times_modulated22;

  ae_int32x2 d_update_times_hidden11, d_update_times_hidden12, d_update_times_hidden21, d_update_times_hidden22;
  ae_int32x2 q_update_times_hidden11, q_update_times_hidden12, q_update_times_hidden21, q_update_times_hidden22;

  ae_int32x2 d_final_sum11, d_final_sum12, d_final_sum21, d_final_sum22;
  ae_int32x2 q_final_sum11, q_final_sum12, q_final_sum21, q_final_sum22;
  
  WORD32 itr;
  for(itr = 0; itr < (num_elms >> 3); itr++)
  {

    AE_LA8X8_IP(d_hidden_in, a_hid_in, ptr_hidden_in);
    AE_SUBW8(d_hidden_in1, d_hidden_in2, d_hidden_in, d_hidden_zero_bias);

    AE_LA16X4X2_IP(d_update1, d_update2, a_up, ptr_update);
    AE_LA16X4X2_IP(d_modulated1, d_modulated2, a_mod, ptr_modulated);

    //update_gate * hidden_state
    AE_MUL16X4S(d_update_times_hidden11, d_update_times_hidden12, d_update1, d_hidden_in1);
    AE_MUL16X4S(d_update_times_hidden21, d_update_times_hidden22, d_update2, d_hidden_in2);

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_hidden11, q_update_times_hidden12, d_update_times_hidden11, d_update_times_hidden12, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_hidden21, q_update_times_hidden22, d_update_times_hidden21, d_update_times_hidden22, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#else       
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_hidden11, q_update_times_hidden12, d_update_times_hidden11, d_update_times_hidden12, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_hidden21, q_update_times_hidden22, d_update_times_hidden21, d_update_times_hidden22, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#endif  

    AE_MINMAX32(q_update_times_hidden11, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden12, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden21, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden22, int16_min, int16_max);

    //(1.0 - update_gate) * modulated_state
    d_one_minus_update1 = AE_MOVINT16X4_FROMF16X4(AE_SUB16S(AE_MOVF16X4_FROMINT16X4(d_int16_max), AE_MOVF16X4_FROMINT16X4(d_update1)));
    d_one_minus_update2 = AE_MOVINT16X4_FROMF16X4(AE_SUB16S(AE_MOVF16X4_FROMINT16X4(d_int16_max), AE_MOVF16X4_FROMINT16X4(d_update2)));
    d_update1           = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(d_one_minus_update1), AE_MOVF16X4_FROMINT16X4(d_one16)));
    d_update2           = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(d_one_minus_update2), AE_MOVF16X4_FROMINT16X4(d_one16)));

    AE_MUL16X4S(d_update_times_modulated11, d_update_times_modulated12, d_update1, d_modulated1);
    AE_MUL16X4S(d_update_times_modulated21, d_update_times_modulated22, d_update2, d_modulated2);

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_modulated11, q_update_times_modulated12, d_update_times_modulated11, d_update_times_modulated12, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_modulated21, q_update_times_modulated22, d_update_times_modulated21, d_update_times_modulated22, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#else        
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_modulated11, q_update_times_modulated12, d_update_times_modulated11, d_update_times_modulated12, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_modulated21, q_update_times_modulated22, d_update_times_modulated21, d_update_times_modulated22, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#endif  
    
    AE_MINMAX32(q_update_times_modulated11, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated12, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated21, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated22, int16_min, int16_max);

    //(update_gate * hidden_state) + ((1.0 - update_gate) * modulated_state)
    d_final_sum11 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated11, q_update_times_hidden11);
    d_final_sum12 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated12, q_update_times_hidden12);
    d_final_sum21 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated21, q_update_times_hidden21);
    d_final_sum22 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated22, q_update_times_hidden22);
    
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_final_sum11, q_final_sum12, d_final_sum11, d_final_sum12, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_final_sum21, q_final_sum22, d_final_sum21, d_final_sum22, out_multiplier, left_shift, right_shift);
#else        
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_final_sum11, q_final_sum12, d_final_sum11, d_final_sum12, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_final_sum21, q_final_sum22, d_final_sum21, d_final_sum22, out_multiplier, left_shift, right_shift);
#endif  

    q_final_sum11 = SW_ADD32S_INT32X2_INT32X2(q_final_sum11, d_out_zero_bias);
    q_final_sum12 = SW_ADD32S_INT32X2_INT32X2(q_final_sum12, d_out_zero_bias);
    q_final_sum21 = SW_ADD32S_INT32X2_INT32X2(q_final_sum21, d_out_zero_bias);
    q_final_sum22 = SW_ADD32S_INT32X2_INT32X2(q_final_sum22, d_out_zero_bias);

    //Saturate to 8-bit
    d_hidden_out1 = AE_SAT16X4(q_final_sum11, q_final_sum12);
    d_hidden_out2 = AE_SAT16X4(q_final_sum21, q_final_sum22);

    d_hidden_out = AE_SAT8X8X16(d_hidden_out1, d_hidden_out2);
    AE_SA8X8_IP(d_hidden_out, a_hid_out, ptr_hidden_out);
  }
  AE_SA64POS_FP(a_hid_out, ptr_hidden_out);
  WORD32 rem_elm = num_elms & 7;
  if(rem_elm)
  {
    const ae_int8x16 *ptr8x16_hidden_in = (ae_int8x16 *)ptr_hidden_in;
    ae_int8x16 *ptr8x16_hidden_out = (ae_int8x16 *)ptr_hidden_out;
    ae_int8x8 temp;
    ae_valignx2 a2_hid_in = AE_LA128_PP(ptr8x16_hidden_in);
    ae_valignx2 a2_hid_out = AE_ZALIGN128();

    AE_LAV8X8X2_XP(d_hidden_in, temp, a2_hid_in, ptr8x16_hidden_in, rem_elm);
    AE_SUBW8(d_hidden_in1, d_hidden_in2, d_hidden_in, d_hidden_zero_bias);
    
    AE_LA16X4X2_IP(d_update1, d_update2, a_up, ptr_update);
    AE_LA16X4X2_IP(d_modulated1, d_modulated2, a_mod, ptr_modulated);

    //update_gate * hidden_state
    AE_MUL16X4S(d_update_times_hidden11, d_update_times_hidden12, d_update1, d_hidden_in1);
    AE_MUL16X4S(d_update_times_hidden21, d_update_times_hidden22, d_update2, d_hidden_in2);

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_hidden11, q_update_times_hidden12, d_update_times_hidden11, d_update_times_hidden12, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_hidden21, q_update_times_hidden22, d_update_times_hidden21, d_update_times_hidden22, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#else        
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_hidden11, q_update_times_hidden12, d_update_times_hidden11, d_update_times_hidden12, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_hidden21, q_update_times_hidden22, d_update_times_hidden21, d_update_times_hidden22, update_to_hidden_state_multiplier, left_shift_uth, right_shift_uth);
#endif  

    AE_MINMAX32(q_update_times_hidden11, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden12, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden21, int16_min, int16_max);
    AE_MINMAX32(q_update_times_hidden22, int16_min, int16_max);

    //(1.0 - update_gate) * modulated_state
    d_one_minus_update1 = AE_MOVINT16X4_FROMF16X4(AE_SUB16S(AE_MOVF16X4_FROMINT16X4(d_int16_max), AE_MOVF16X4_FROMINT16X4(d_update1)));
    d_one_minus_update2 = AE_MOVINT16X4_FROMF16X4(AE_SUB16S(AE_MOVF16X4_FROMINT16X4(d_int16_max), AE_MOVF16X4_FROMINT16X4(d_update2)));
    d_update1           = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(d_one_minus_update1), AE_MOVF16X4_FROMINT16X4(d_one16)));
    d_update2           = AE_MOVINT16X4_FROMF16X4(AE_ADD16S(AE_MOVF16X4_FROMINT16X4(d_one_minus_update2), AE_MOVF16X4_FROMINT16X4(d_one16)));

    AE_MUL16X4S(d_update_times_modulated11, d_update_times_modulated12, d_update1, d_modulated1);
    AE_MUL16X4S(d_update_times_modulated21, d_update_times_modulated22, d_update2, d_modulated2);

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_modulated11, q_update_times_modulated12, d_update_times_modulated11, d_update_times_modulated12, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_update_times_modulated21, q_update_times_modulated22, d_update_times_modulated21, d_update_times_modulated22, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#else        
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_modulated11, q_update_times_modulated12, d_update_times_modulated11, d_update_times_modulated12, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_update_times_modulated21, q_update_times_modulated22, d_update_times_modulated21, d_update_times_modulated22, update_to_modulated_state_multiplier, left_shift_utm, right_shift_utm);
#endif  
    
    AE_MINMAX32(q_update_times_modulated11, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated12, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated21, int16_min, int16_max);
    AE_MINMAX32(q_update_times_modulated22, int16_min, int16_max);

    //(update_gate * hidden_state) + ((1.0 - update_gate) * modulated_state)
    d_final_sum11 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated11, q_update_times_hidden11);
    d_final_sum12 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated12, q_update_times_hidden12);
    d_final_sum21 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated21, q_update_times_hidden21);
    d_final_sum22 = SW_ADD32S_INT32X2_INT32X2(q_update_times_modulated22, q_update_times_hidden22);

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_final_sum11, q_final_sum12, d_final_sum11, d_final_sum12, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(q_final_sum21, q_final_sum22, d_final_sum21, d_final_sum22, out_multiplier, left_shift, right_shift);
#else        
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_final_sum11, q_final_sum12, d_final_sum11, d_final_sum12, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2X2_OUT32(q_final_sum21, q_final_sum22, d_final_sum21, d_final_sum22, out_multiplier, left_shift, right_shift);
#endif  

    q_final_sum11 = SW_ADD32S_INT32X2_INT32X2(q_final_sum11, d_out_zero_bias);
    q_final_sum12 = SW_ADD32S_INT32X2_INT32X2(q_final_sum12, d_out_zero_bias);
    q_final_sum21 = SW_ADD32S_INT32X2_INT32X2(q_final_sum21, d_out_zero_bias);
    q_final_sum22 = SW_ADD32S_INT32X2_INT32X2(q_final_sum22, d_out_zero_bias);

    //Saturate to 8-bit
    d_hidden_out1 = AE_SAT16X4(q_final_sum11, q_final_sum12);
    d_hidden_out2 = AE_SAT16X4(q_final_sum21, q_final_sum22);

    d_hidden_out = AE_SAT8X8X16(d_hidden_out1, d_hidden_out2);
    
    AE_SAV8X8X2_XP(d_hidden_out, temp, a2_hid_out, ptr8x16_hidden_out, rem_elm);
    AE_SA128POS_FP(a2_hid_out, ptr8x16_hidden_out);
  }

  return 0;
}                                      
