/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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

WORD32 xa_nn_renorm_asym8s_asym8s(WORD8 * __restrict__ p_out,
                              const WORD8 * __restrict__ p_inp,
                              WORD32 num_elm,
                              WORD32 renorm_scale,
                              WORD32 renorm_shift,
                              WORD32 input_zero_bias,
                              WORD32 output_zero_bias)
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */  
  XA_NNLIB_ARG_CHK_COND((renorm_shift<0 || renorm_shift >= 24),-1);
  XA_NNLIB_ARG_CHK_COND((renorm_scale<0 || renorm_scale > 65535),-1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -128 || input_zero_bias > 127),-1);
  XA_NNLIB_ARG_CHK_COND((output_zero_bias < -128 || output_zero_bias > 127),-1); 
  
  WORD32 zero_in_out = (input_zero_bias * renorm_scale) - ((WORD32)output_zero_bias << renorm_shift);
  ae_int32x2 renorm_scale32 = AE_MOVDA32(renorm_scale);
  ae_int32x2 zero_point32 = AE_MOVDA32(zero_in_out);
  
  ae_int16x4 d_inp1, d_inp2;
  ae_int32x2 zero_point32_00, zero_point32_01, zero_point32_10, zero_point32_11;
  ae_int32x2 out32_00, out32_01, out32_10, out32_11;
  ae_int16x4 out16_0, out16_1;
  ae_int8x8 d_out;
  
  const WORD8 *inp_ptr = p_inp;
  ae_valign align_inp = AE_LA64_PP(inp_ptr);
  ae_int8x8 *out_ptr = (ae_int8x8 *)p_out;
  ae_valign align_out = AE_ZALIGN64();
  ae_int32x2 shift_one = AE_SRAA32(AE_MOVDA32(0x80000000), renorm_shift);
  
  int i;
  for(i=0; i<(num_elm & ~(8-1)); i+=8)
  {
    AE_LA8X4S_IP(d_inp1, align_inp, inp_ptr);
    AE_LA8X4S_IP(d_inp2, align_inp, inp_ptr);
    
    AE_MOVD32X4(zero_point32_00, zero_point32_01, zero_point32, zero_point32);
    AE_MOVD32X4(zero_point32_10, zero_point32_11, zero_point32, zero_point32);

    AE_MULSP32X16X2_H(zero_point32_00, renorm_scale32, d_inp1);
    AE_MULSP32X16X2_L(zero_point32_01, renorm_scale32, d_inp1);
    AE_MULSP32X16X2_H(zero_point32_10, renorm_scale32, d_inp2);
    AE_MULSP32X16X2_L(zero_point32_11, renorm_scale32, d_inp2);
    
    AE_MULF2P32X4RAS(out32_00, out32_01, zero_point32_00, zero_point32_01, shift_one, shift_one);
    AE_MULF2P32X4RAS(out32_10, out32_11, zero_point32_10, zero_point32_11, shift_one, shift_one);
    
    out16_0 = AE_SAT16X4(out32_00, out32_01);
    out16_1 = AE_SAT16X4(out32_10, out32_11);
    d_out = AE_SAT8X8X16(out16_0, out16_1);
    
    AE_SA8X8_IP(d_out, align_out, out_ptr);
  }
  
  AE_SA64POS_FP(align_out, out_ptr);
  
  if(i<num_elm)
  {
    ae_int8x8 extra_var, d_inp_8;
    ae_valignx2 align_inpx2 = AE_LA128_PP(inp_ptr);
    ae_valignx2 align_outx2 = AE_ZALIGN128();
  
    AE_LAV8X8X2_XP(d_inp_8, extra_var, align_inpx2, (ae_int8x16 *)inp_ptr, num_elm-i);
    AE_CVTA16X4X2F8(d_inp1, d_inp2, d_inp_8, 0);
    
    zero_point32_00 = AE_MOV32(zero_point32);
    zero_point32_01 = AE_MOV32(zero_point32);
    zero_point32_10 = AE_MOV32(zero_point32);
    zero_point32_11 = AE_MOV32(zero_point32);

    AE_MULSP32X16X2_H(zero_point32_00, renorm_scale32, d_inp1);
    AE_MULSP32X16X2_L(zero_point32_01, renorm_scale32, d_inp1);
    AE_MULSP32X16X2_H(zero_point32_10, renorm_scale32, d_inp2);
    AE_MULSP32X16X2_L(zero_point32_11, renorm_scale32, d_inp2);
    
    AE_MULF2P32X4RAS(out32_00, out32_01, zero_point32_00, zero_point32_01, shift_one, shift_one);
    AE_MULF2P32X4RAS(out32_10, out32_11, zero_point32_10, zero_point32_11, shift_one, shift_one);
    
    out16_0 = AE_SAT16X4(out32_00, out32_01);
    out16_1 = AE_SAT16X4(out32_10, out32_11);
    d_out = AE_SAT8X8X16(out16_0, out16_1);
      
    AE_SAV8X8X2_XP(d_out, extra_var, align_outx2, (ae_int8x16 *)out_ptr, num_elm-i); 
    AE_SA128POS_FP(align_outx2, out_ptr);
  }
  return 0;
}                        
                        