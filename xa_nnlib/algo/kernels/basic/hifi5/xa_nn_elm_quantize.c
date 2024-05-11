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
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include <math.h>
#define PACK_32X2(dst1, src1, src2) \
dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e0c0a08, 0x06040200)));

WORD32 xa_nn_elm_requantize_asym8u_asym8s(WORD8 * __restrict__ p_out, 
                                    const UWORD8 * __restrict__ p_inp, 
                                    WORD32 inp_zero_bias, 
                                    WORD32 out_zero_bias, 
                                    WORD32 out_shift, 
                                    WORD32 out_multiplier, 
                                    WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < 0) || (inp_zero_bias > 255)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int8x8 *p_i = (ae_int8x8 *)p_inp;
  WORD8 *p_o = p_out;

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();
  ae_int8x8 d_inp_zero_bias   = AE_MOVDA8(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int8x8 d_inp0, d_out0;
  ae_int16x4 ONE = AE_MOVDA16(1);
  ae_int16x4 d_inp16_0,d_inp16_1;
  ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;
  ae_int16x4 d_out0_16, d_out1_16;

  for(i = 0; i < num_elm >> 3; i++)
  {
    AE_LA8X8_IP(d_inp0, align_inp, p_i);
    AE_SUBW8U(d_inp16_0, d_inp16_1, d_inp0, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp16_0 , ONE);
    AE_MUL16X4(d_inp32_2 , d_inp32_3 , d_inp16_1 , ONE);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out0_16 , d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out1_16 , d_inp32_2, d_inp32_3, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    d_out0 = AE_SAT8X8X16(d_out0_16 , d_out1_16);
    AE_SA8X8_IP(d_out0 ,align_dst, (ae_int8x8 *)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  ae_valignx2 align_inpx2 = AE_LA128_PP(p_i);
  ae_valignx2 align_dstx2 = AE_ZALIGN128();

  if(num_elm & 7){
    ae_int8x8 d_inp0_8x8, d_inp1_8x8, d_out0_8x8, d_out1_8x8;
    AE_LAV8X8X2_XP(d_inp0_8x8, d_inp1_8x8, align_inpx2, (ae_int8x16 *)p_i, num_elm & 7);
    AE_SUBW8U(d_inp16_0, d_inp16_1, d_inp0_8x8, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp16_0 , ONE);
    AE_MUL16X4(d_inp32_2 , d_inp32_3 , d_inp16_1 , ONE);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out0_16 , d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out1_16 , d_inp32_2, d_inp32_3, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    d_out0_8x8 = AE_SAT8X8X16(d_out0_16 , d_out1_16);
    d_out1_8x8 = d_inp1_8x8;
    AE_SAV8X8X2_XP(d_out0_8x8, d_out1_8x8, align_dstx2, (ae_int8x16 *)p_o, num_elm & 7);
  }
  AE_SA128POS_FP(align_dstx2, p_o);

  return 0;
}

WORD32 xa_nn_elm_requantize_asym16s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD8 *out = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);

  ae_int16x4 out_0, out_1;

  for(i = 0; i < (num_elm >> 3); i++)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1;
    ae_int32x2 d_inp32_2, d_inp32_3;
    AE_LA16X4X2_IP(d_inp0, d_inp1, align_inp, (ae_int16x8 *)p_i);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    AE_SUBW16(d_inp32_2, d_inp32_3, d_inp1, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, d_inp32_2, d_inp32_3, d_out_multiplier, left_shift, right_shift, out_zero_bias);

    AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
    AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));

    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, out_1);

    AE_SA8X8_IP(out32_0, align_dst, (ae_int8x8 *)out);
  }
  AE_SA64POS_FP(align_dst, out);

  /*Remainder loop*/
#pragma loop_count max=7
  for(i = 0; i < (num_elm & 7); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L16_IP(d_inp0, (ae_int16 *)p_i, 2);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift, out_zero_bias);

    AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, out_0);

    AE_S8_0_IP(out32_0, (ae_int8 *)out, 1);
  }

  return 0;
}

WORD32 xa_nn_elm_requantize_asym16s_asym16s(WORD16 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  ae_int16x8 *p_o = (ae_int16x8 *)p_out;
  const ae_int16x8 *p_i = (const ae_int16x8 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valignx2 align_dst = AE_ZALIGN128();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);

  ae_int16x4 out_0, out_1;

  for(i = 0; i < (num_elm >> 3); i++)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1;
    ae_int32x2 d_inp32_2, d_inp32_3;
    AE_LA16X4X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    AE_SUBW16(d_inp32_2, d_inp32_3, d_inp1, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_2, d_inp32_2, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_3, d_inp32_3, d_out_multiplier, left_shift, right_shift);

    d_inp32_0 = AE_ADD32S(d_inp32_0, AE_MOVDA32(out_zero_bias));
    d_inp32_1 = AE_ADD32S(d_inp32_1, AE_MOVDA32(out_zero_bias));
    d_inp32_2 = AE_ADD32S(d_inp32_2, AE_MOVDA32(out_zero_bias));
    d_inp32_3 = AE_ADD32S(d_inp32_3, AE_MOVDA32(out_zero_bias));

    out_0 = AE_SAT16X4(d_inp32_0, d_inp32_1);
    out_1 = AE_SAT16X4(d_inp32_2, d_inp32_3);

    AE_SA16X4X2_IP(out_0, out_1, align_dst, p_o);
  }

  /*Remainder */
  if((num_elm & 7) != 0)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1;
    ae_int32x2 d_inp32_2, d_inp32_3;
    AE_LAV16X4X2_XP(d_inp0, d_inp1, align_inp, p_i, ((num_elm & 7) << 1));
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    AE_SUBW16(d_inp32_2, d_inp32_3, d_inp1, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_2, d_inp32_2, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_3, d_inp32_3, d_out_multiplier, left_shift, right_shift);

    d_inp32_0 = AE_ADD32S(d_inp32_0, AE_MOVDA32(out_zero_bias));
    d_inp32_1 = AE_ADD32S(d_inp32_1, AE_MOVDA32(out_zero_bias));
    d_inp32_2 = AE_ADD32S(d_inp32_2, AE_MOVDA32(out_zero_bias));
    d_inp32_3 = AE_ADD32S(d_inp32_3, AE_MOVDA32(out_zero_bias));

    out_0 = AE_SAT16X4(d_inp32_0, d_inp32_1);
    out_1 = AE_SAT16X4(d_inp32_2, d_inp32_3);

    AE_SAV16X4X2_XP(out_0, out_1, align_dst, p_o, ((num_elm & 7) << 1));
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_requantize_asym16s_asym32s(WORD32 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  //XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD32 *p_o = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  ae_int32x2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_LA16X4_IP(d_inp0, align_inp, (ae_int16x4 *)p_i);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);

    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_1, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    d_out1 = AE_ADD32S(d_inp32_1, d_out_zero_bias);
    AE_SA32X2_IP(d_out0, align_dst, (ae_int32x2*)p_o);
    AE_SA32X2_IP(d_out1, align_dst, (ae_int32x2*)p_o);

  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L16_IP(d_inp0, (ae_int16 *)p_i, 2);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    AE_S32_L_IP(d_out0, (ae_int32*)p_o, 4);
  }
  return 0;
}


WORD32 xa_nn_elm_requantize_asym8s_asym32s(WORD32 * __restrict__ p_out,
                                           const WORD8 * __restrict__ p_inp,
                                           WORD32  inp_zero_bias,
                                           WORD32  out_zero_bias,
                                           WORD32  out_shift,
                                           WORD32  out_multiplier,
                                           WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  if(out_shift < -31)
  {
    out_multiplier = 0;
    out_shift = 0;
  }
  int i;
  ae_int32x4 *p_o = (ae_int32x4 *)p_out;
  ae_int8x16 *p_i = (ae_int8x16 *)p_inp;

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valignx2 align_dst = AE_ZALIGN128();

  ae_int8x8 d_inp_zero_bias = AE_MOVDA8(inp_zero_bias);
  ae_int16x4 ONE = AE_MOVDA16(1);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  ae_int32x2 d_out0, d_out1, d_out2, d_out3;
  ae_int32x2 d_out4, d_out5, d_out6, d_out7;

  for(i = 0; i < (num_elm >> 4); i++)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_inp16_0, d_inp16_1, d_inp16_2, d_inp16_3;
    ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;
    ae_int32x2 d_inp32_4, d_inp32_5, d_inp32_6, d_inp32_7;

    AE_LA8X8X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_SUBW8(d_inp16_0, d_inp16_1, d_inp0, d_inp_zero_bias);
    AE_SUBW8(d_inp16_2, d_inp16_3, d_inp1, d_inp_zero_bias);

    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16_0, ONE);
    AE_MUL16X4(d_inp32_2, d_inp32_3, d_inp16_1, ONE);
    AE_MUL16X4(d_inp32_4, d_inp32_5, d_inp16_2, ONE);
    AE_MUL16X4(d_inp32_6, d_inp32_7, d_inp16_3, ONE);

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(d_inp32_0, d_inp32_1, d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(d_inp32_2, d_inp32_3, d_inp32_2, d_inp32_3, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(d_inp32_4, d_inp32_5, d_inp32_4, d_inp32_5, d_out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(d_inp32_6, d_inp32_7, d_inp32_6, d_inp32_7, d_out_multiplier, left_shift, right_shift);

    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    d_out1 = AE_ADD32S(d_inp32_1, d_out_zero_bias);
    d_out2 = AE_ADD32S(d_inp32_2, d_out_zero_bias);
    d_out3 = AE_ADD32S(d_inp32_3, d_out_zero_bias);
    d_out4 = AE_ADD32S(d_inp32_4, d_out_zero_bias);
    d_out5 = AE_ADD32S(d_inp32_5, d_out_zero_bias);
    d_out6 = AE_ADD32S(d_inp32_6, d_out_zero_bias);
    d_out7 = AE_ADD32S(d_inp32_7, d_out_zero_bias);

    AE_SA32X2X2_IP(d_out0, d_out1, align_dst, p_o);
    AE_SA32X2X2_IP(d_out2, d_out3, align_dst, p_o);
    AE_SA32X2X2_IP(d_out4, d_out5, align_dst, p_o);
    AE_SA32X2X2_IP(d_out6, d_out7, align_dst, p_o);

  }
  AE_SA128POS_FP(align_dst, p_o);

  /*Remainder loop*/
#pragma no_unroll
  for(i = 0; i < (num_elm & 15); i++)
  {
    ae_int8x8 d_inp0;
    ae_int16x4 d_inp16_0, d_inp16_1;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L8_IP(d_inp0, (ae_int8 *)p_i, 1);
    AE_SUBW8(d_inp16_0, d_inp16_1, d_inp0, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16_0, ONE);
    MPY_BY_QUANT_MULT_X2_OUT32(d_inp32_0, d_inp32_0, d_out_multiplier, left_shift, right_shift);
    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    AE_S32_L_IP(d_out0, (ae_int32*)p_o, 4);
  }
  return 0;
}

WORD32 xa_nn_elm_requantize_asym8s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD8 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  /* Single rounding doesn't need two shifts */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int8x8 *p_i = (ae_int8x8 *)p_inp;
  WORD8 *p_o = p_out;

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();
  ae_int8x8 d_inp_zero_bias   = AE_MOVDA8(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int8x8 d_inp0, d_out0;
  ae_int16x4 ONE = AE_MOVDA16(1);
  ae_int16x4 d_inp16_0,d_inp16_1;
  ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;
  ae_int16x4 d_out0_16, d_out1_16;

  for(i = 0; i < num_elm >> 3; i++)
  {
    AE_LA8X8_IP(d_inp0, align_inp, p_i);
    AE_SUBW8(d_inp16_0, d_inp16_1, d_inp0, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp16_0 , ONE);
    AE_MUL16X4(d_inp32_2 , d_inp32_3 , d_inp16_1 , ONE);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out0_16 , d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out1_16 , d_inp32_2, d_inp32_3, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    d_out0 = AE_SAT8X8X16(d_out0_16 , d_out1_16);
    AE_SA8X8_IP(d_out0 ,align_dst, (ae_int8x8 *)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  ae_valignx2 align_inpx2 = AE_LA128_PP(p_i);
  ae_valignx2 align_dstx2 = AE_ZALIGN128();

  if(num_elm & 7){
    ae_int8x8 d_inp0_8x8, d_inp1_8x8, d_out0_8x8, d_out1_8x8;
    AE_LAV8X8X2_XP(d_inp0_8x8, d_inp1_8x8, align_inpx2, (ae_int8x16 *)p_i, num_elm & 7);
    AE_SUBW8(d_inp16_0, d_inp16_1, d_inp0_8x8, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0 , d_inp32_1 , d_inp16_0 , ONE);
    AE_MUL16X4(d_inp32_2 , d_inp32_3 , d_inp16_1 , ONE);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out0_16 , d_inp32_0, d_inp32_1, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_out1_16 , d_inp32_2, d_inp32_3, d_out_multiplier, left_shift, right_shift, out_zero_bias);
    d_out0_8x8 = AE_SAT8X8X16(d_out0_16 , d_out1_16);
    d_out1_8x8 = d_inp1_8x8;
    AE_SAV8X8X2_XP(d_out0_8x8, d_out1_8x8, align_dstx2, (ae_int8x16 *)p_o, num_elm & 7);
  }
  AE_SA128POS_FP(align_dstx2, p_o);

  return 0;
}

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_dequantize_asym8s_f32,
                               (FLOAT32 * __restrict__ p_out,
                               const WORD8 * __restrict__ p_inp,
                               WORD32  inp_zero_bias,
                               FLOAT32  inp_scale,
                               WORD32  num_elm))
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_elm_dequantize_asym8s_f32(FLOAT32 * __restrict__ p_out,
                                       const WORD8 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32  inp_scale,
                                       WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -128) || (inp_zero_bias > 127)), -1);

  int i;
  xtfloatx4 *p_o = (xtfloatx4 *)p_out;
  ae_int8x16 *p_i = (ae_int8x16 *)p_inp;

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valignx2 align_dst = AE_ZALIGN128();

  ae_int8x8 d_inp_zero_bias = AE_MOVDA8(inp_zero_bias);
  ae_int16x4 ONE = AE_MOVDA16(1);
  xtfloat *inp_scale_ptr = &inp_scale;
  xtfloat d_inp_scale;
  AE_LSIP(d_inp_scale, inp_scale_ptr, sizeof(FLOAT32));;
  xtfloatx2 d_out0, d_out1, d_out2, d_out3;
  xtfloatx2 d_out4, d_out5, d_out6, d_out7;

  for(i = 0; i < (num_elm >> 4); i++)
  {
    ae_int8x8 d_inp0, d_inp1;
    ae_int16x4 d_inp16_0, d_inp16_1, d_inp16_2, d_inp16_3;
    ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;
    ae_int32x2 d_inp32_4, d_inp32_5, d_inp32_6, d_inp32_7;

    AE_LA8X8X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_SUBW8(d_inp16_0, d_inp16_1, d_inp0, d_inp_zero_bias);
    AE_SUBW8(d_inp16_2, d_inp16_3, d_inp1, d_inp_zero_bias);

    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16_0, ONE);
    AE_MUL16X4(d_inp32_2, d_inp32_3, d_inp16_1, ONE);
    AE_MUL16X4(d_inp32_4, d_inp32_5, d_inp16_2, ONE);
    AE_MUL16X4(d_inp32_6, d_inp32_7, d_inp16_3, ONE);

    MULQ_S(d_out0, d_out1, d_inp32_0, d_inp32_1, d_inp_scale);
    MULQ_S(d_out2, d_out3, d_inp32_2, d_inp32_3, d_inp_scale);
    MULQ_S(d_out4, d_out5, d_inp32_4, d_inp32_5, d_inp_scale);
    MULQ_S(d_out6, d_out7, d_inp32_6, d_inp32_7, d_inp_scale);

    AE_SASX2X2_IP(d_out0, d_out1, align_dst, p_o);
    AE_SASX2X2_IP(d_out2, d_out3, align_dst, p_o);
    AE_SASX2X2_IP(d_out4, d_out5, align_dst, p_o);
    AE_SASX2X2_IP(d_out6, d_out7, align_dst, p_o);

  }
  AE_SA128POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 15); i++)
  {
    ae_int8x8 d_inp0;
    ae_int16x4 d_inp16_0, d_inp16_1;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L8_IP(d_inp0, (ae_int8 *)p_i, 1);
    AE_SUBW8(d_inp16_0, d_inp16_1, d_inp0, d_inp_zero_bias);
    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16_0, ONE);
    MULQ_S(d_out0, d_out1, d_inp32_0, d_inp32_0, d_inp_scale);
    AE_SSIP(d_out0, (xtfloat *)p_o, sizeof(FLOAT32));
  }
  return 0;
}
#endif /* #if !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_quantize_f32_asym8s,
                               (WORD8 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               FLOAT32 out_scale,
                               WORD32  out_zero_bias,
                               WORD32  num_elm))
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_elm_quantize_f32_asym8s(WORD8 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_scale == 0.0f || !isfinite(out_scale)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);

  int i;
  xtfloatx4 *p_i = (xtfloatx4 *)p_inp;
  ae_int8x8 *p_o = (ae_int8x8 *)p_out;

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_out_zero_bias = AE_MOVDA16(out_zero_bias);
  xtfloat *out_scale_ptr = &out_scale;
  xtfloatx2 d_out_scale = (xtfloatx2)*out_scale_ptr;
  xtfloatx2 d_one = FLOAT_SX2(AE_MOVDA32(1),0);
  xtfloatx2 d_one_over_out_scale = XT_DIV_SX2(d_one, d_out_scale);
  for(i = 0; i < (num_elm >> 3); i++)
  {
    xtfloatx2 d_inp0, d_inp1, d_inp2, d_inp3;
    xtfloatx2 d_inp0_t, d_inp1_t, d_inp2_t, d_inp3_t;
    ae_int8x8 d_out0;
    ae_int16x4 d_out16_0, d_out16_1;
    ae_int32x2 d_out32_0, d_out32_1, d_out32_2, d_out32_3;

    AE_LASX2X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_LASX2X2_IP(d_inp2, d_inp3, align_inp, p_i);

    MUL_SX2X2(d_inp0_t, d_inp1_t, d_inp0, d_inp1, d_one_over_out_scale, d_one_over_out_scale);
    MUL_SX2X2(d_inp2_t, d_inp3_t, d_inp2, d_inp3, d_one_over_out_scale, d_one_over_out_scale);

    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);
    d_inp2_t = XT_FIROUND_SX2(d_inp2_t);
    d_inp3_t = XT_FIROUND_SX2(d_inp3_t);

    d_out32_0 = XT_TRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_TRUNC_SX2(d_inp1_t, 0);
    d_out32_2 = XT_TRUNC_SX2(d_inp2_t, 0);
    d_out32_3 = XT_TRUNC_SX2(d_inp3_t, 0);

    d_out16_0 = AE_SAT16X4(d_out32_0, d_out32_1);
    d_out16_1 = AE_SAT16X4(d_out32_2, d_out32_3);

    d_out16_0 = AE_ADD16S(d_out16_0, d_out_zero_bias);
    d_out16_1 = AE_ADD16S(d_out16_1, d_out_zero_bias);

    d_out0 = AE_SAT8X8X16(d_out16_0, d_out16_1);

    AE_SA8X8_IP(d_out0, align_dst, p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  WORD32 rem_elm = (num_elm & 7);
  if(rem_elm > 0)
  {
    xtfloatx2 d_inp0, d_inp1, d_inp2, d_inp3;
    xtfloatx2 d_inp0_t, d_inp1_t, d_inp2_t, d_inp3_t;
    ae_int8x8 d_out0;
    ae_int16x4 d_out16_0, d_out16_1;
    ae_int32x2 d_out32_0, d_out32_1, d_out32_2, d_out32_3;
    ae_int16x4 d_tmp0, d_tmp1, d_tmp2, d_tmp3;
    ae_valignx2 align_dstx2;

    WORD32 rem_elm0, rem_elm1;
    rem_elm0 = rem_elm >= 4 ? 16 : rem_elm << 2;
    rem_elm1 = rem_elm <= 4 ? 0 : (rem_elm - 4) << 2;
    ae_int16x8 *p_i16 = (ae_int16x8 *)p_i;
    align_inp = AE_LA128_PP(p_i16);
    align_dstx2 = AE_ZALIGN128();
    AE_LAV16X4X2_XP(d_tmp0, d_tmp1, align_inp, p_i16, rem_elm0);
    AE_LAV16X4X2_XP(d_tmp2, d_tmp3, align_inp, p_i16, rem_elm1);

    d_tmp0 = AE_SEL16_2301(d_tmp0, d_tmp0);
    d_tmp1 = AE_SEL16_2301(d_tmp1, d_tmp1);
    d_tmp2 = AE_SEL16_2301(d_tmp2, d_tmp2);
    d_tmp3 = AE_SEL16_2301(d_tmp3, d_tmp3);

    d_inp0 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp0));
    d_inp1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp1));
    d_inp2 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp2));
    d_inp3 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp3));

    MUL_SX2X2(d_inp0_t, d_inp1_t, d_inp0, d_inp1, d_one_over_out_scale, d_one_over_out_scale);
    MUL_SX2X2(d_inp2_t, d_inp3_t, d_inp2, d_inp3, d_one_over_out_scale, d_one_over_out_scale);

    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);
    d_inp2_t = XT_FIROUND_SX2(d_inp2_t);
    d_inp3_t = XT_FIROUND_SX2(d_inp3_t);

    d_out32_0 = XT_TRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_TRUNC_SX2(d_inp1_t, 0);
    d_out32_2 = XT_TRUNC_SX2(d_inp2_t, 0);
    d_out32_3 = XT_TRUNC_SX2(d_inp3_t, 0);

    d_out16_0 = AE_SAT16X4(d_out32_0, d_out32_1);
    d_out16_1 = AE_SAT16X4(d_out32_2, d_out32_3);

    d_out16_0 = AE_ADD16S(d_out16_0, d_out_zero_bias);
    d_out16_1 = AE_ADD16S(d_out16_1, d_out_zero_bias);

    d_out0 = AE_SAT8X8X16(d_out16_0, d_out16_1);

    AE_SAV8X8X2_XP(d_out0, d_out0, align_dstx2, (ae_int8x16 *)p_o, rem_elm);
    AE_SA128POS_FP(align_dstx2, (ae_int8x16 *)p_o);
  }
  return 0;
}
#endif /* #if !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_dequantize_asym16s_f32,
                               (FLOAT32 * __restrict__ p_out,
                               const WORD16 * __restrict__ p_inp,
                               WORD32  inp_zero_bias,
                               FLOAT32  inp_scale,
                               WORD32  num_elm))
#else
WORD32 xa_nn_elm_dequantize_asym16s_f32(FLOAT32 * __restrict__ p_out,
                                       const WORD16 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32 inp_scale,
                                       WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);

  int i;
  xtfloatx4 *p_o = (xtfloatx4 *)p_out;
  ae_int16x8 *p_i = (ae_int16x8 *)p_inp;

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valignx2 align_dst = AE_ZALIGN128();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  xtfloat *inp_scale_ptr = &inp_scale;
  xtfloat d_inp_scale;
  AE_LSIP(d_inp_scale, inp_scale_ptr, sizeof(FLOAT32));;
  xtfloatx2 d_out0, d_out1, d_out2, d_out3;

  for(i = 0; i < (num_elm >> 3); i++)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;

    AE_LA16X4X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    AE_SUBW16(d_inp32_2, d_inp32_3, d_inp1, d_inp_zero_bias);

    d_out0 = FLOAT_SX2(d_inp32_0, 0);
    d_out1 = FLOAT_SX2(d_inp32_1, 0);
    d_out2 = FLOAT_SX2(d_inp32_2, 0);
    d_out3 = FLOAT_SX2(d_inp32_3, 0);

    MULQ_S(d_out0, d_out1, d_out0, d_out1, d_inp_scale);
    MULQ_S(d_out2, d_out3, d_out2, d_out3, d_inp_scale);

    AE_SASX2X2_IP(d_out0, d_out1, align_dst, p_o);
    AE_SASX2X2_IP(d_out2, d_out3, align_dst, p_o);
  }
  AE_SA128POS_FP(align_dst, p_o);

  WORD32 rem_elm = num_elm & 7;
  /*Remainder loop*/
  if(rem_elm > 0)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1, d_inp32_2, d_inp32_3;

    AE_LAV16X4X2_XP(d_inp0, d_inp1, align_inp, p_i, rem_elm << 1);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    AE_SUBW16(d_inp32_2, d_inp32_3, d_inp1, d_inp_zero_bias);

    d_out0 = FLOAT_SX2(d_inp32_0, 0);
    d_out1 = FLOAT_SX2(d_inp32_1, 0);
    d_out2 = FLOAT_SX2(d_inp32_2, 0);
    d_out3 = FLOAT_SX2(d_inp32_3, 0);

    MULQ_S(d_out0, d_out1, d_out0, d_out1, d_inp_scale);
    MULQ_S(d_out2, d_out3, d_out2, d_out3, d_inp_scale);

    ae_int16x4 d_tmp0, d_tmp1, d_tmp2, d_tmp3;

    d_tmp0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d_out0));
    d_tmp1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d_out1));
    d_tmp2 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d_out2));
    d_tmp3 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d_out3));

    d_tmp0 = AE_SEL16_2301(d_tmp0, d_tmp0);
    d_tmp1 = AE_SEL16_2301(d_tmp1, d_tmp1);
    d_tmp2 = AE_SEL16_2301(d_tmp2, d_tmp2);
    d_tmp3 = AE_SEL16_2301(d_tmp3, d_tmp3);

    ae_int16x8 *p_o16 = (ae_int16x8 *)p_o;
    align_dst = AE_ZALIGN128();
    WORD32 rem_elm0, rem_elm1;
    rem_elm0 = rem_elm >= 4 ? 16 : rem_elm << 2;
    rem_elm1 = rem_elm <= 4 ? 0 : (rem_elm - 4) << 2;
    AE_SAV16X4X2_XP(d_tmp0, d_tmp1, align_dst, p_o16, rem_elm0);
    AE_SAV16X4X2_XP(d_tmp2, d_tmp3, align_dst, p_o16, rem_elm1);
    AE_SA128POS_FP(align_dst, p_o16);
  }
  return 0;
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_elm_quantize_f32_asym16s,
                               (WORD16 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               FLOAT32 out_scale,
                               WORD32  out_zero_bias,
                               WORD32  num_elm))
#else /* #if !HAVE_VFPU */
WORD32 xa_nn_elm_quantize_f32_asym16s(WORD16 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_scale == 0.0f || !isfinite(out_scale)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);

  int i;
  xtfloatx4 *p_i = (xtfloatx4 *)p_inp;
  ae_int16x8 *p_o = (ae_int16x8 *)p_out;

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valignx2 align_dst = AE_ZALIGN128();

  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  xtfloat *out_scale_ptr = &out_scale;
  xtfloatx2 d_out_scale = (xtfloatx2)*out_scale_ptr;
  xtfloatx2 d_one = FLOAT_SX2(AE_MOVDA32(1),0);
  xtfloatx2 d_one_over_out_scale = XT_DIV_SX2(d_one, d_out_scale);

  for(i = 0; i < (num_elm >> 3); i++)
  {
    xtfloatx2 d_inp0, d_inp1, d_inp2, d_inp3;
    xtfloatx2 d_inp0_t, d_inp1_t, d_inp2_t, d_inp3_t;
    ae_int16x4 d_out0, d_out1;
    ae_int32x2 d_out32_0, d_out32_1, d_out32_2, d_out32_3;


    AE_LASX2X2_IP(d_inp0, d_inp1, align_inp, p_i);
    AE_LASX2X2_IP(d_inp2, d_inp3, align_inp, p_i);

    MUL_SX2X2(d_inp0_t, d_inp1_t, d_inp0, d_inp1, d_one_over_out_scale, d_one_over_out_scale);
    MUL_SX2X2(d_inp2_t, d_inp3_t, d_inp2, d_inp3, d_one_over_out_scale, d_one_over_out_scale);

    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);
    d_inp2_t = XT_FIROUND_SX2(d_inp2_t);
    d_inp3_t = XT_FIROUND_SX2(d_inp3_t);

    d_out32_0 = XT_TRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_TRUNC_SX2(d_inp1_t, 0);
    d_out32_2 = XT_TRUNC_SX2(d_inp2_t, 0);
    d_out32_3 = XT_TRUNC_SX2(d_inp3_t, 0);

    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    d_out32_1 = AE_ADD32S(d_out32_1, d_out_zero_bias);
    d_out32_2 = AE_ADD32S(d_out32_2, d_out_zero_bias);
    d_out32_3 = AE_ADD32S(d_out32_3, d_out_zero_bias);

    d_out0 = AE_SAT16X4(d_out32_0, d_out32_1);
    d_out1 = AE_SAT16X4(d_out32_2, d_out32_3);

    AE_SA16X4X2_IP(d_out0, d_out1, align_dst, p_o);
  }

  WORD32 rem_elm = num_elm & 7;
  if(rem_elm > 0)
  {
    xtfloatx2 d_inp0, d_inp1, d_inp2, d_inp3;
    xtfloatx2 d_inp0_t, d_inp1_t, d_inp2_t, d_inp3_t;
    ae_int16x4 d_out0, d_out1;
    ae_int32x2 d_out32_0, d_out32_1, d_out32_2, d_out32_3;
    ae_int16x4 d_tmp0, d_tmp1, d_tmp2, d_tmp3;

    WORD32 rem_elm0, rem_elm1;
    rem_elm0 = rem_elm >= 4 ? 16 : rem_elm << 2;
    rem_elm1 = rem_elm <= 4 ? 0 : (rem_elm - 4) << 2;
    ae_int16x8 *p_i16 = (ae_int16x8 *)p_i;
    align_inp = AE_LA128_PP(p_i16);
    AE_LAV16X4X2_XP(d_tmp0, d_tmp1, align_inp, p_i16, rem_elm0);
    AE_LAV16X4X2_XP(d_tmp2, d_tmp3, align_inp, p_i16, rem_elm1);

    d_tmp0 = AE_SEL16_2301(d_tmp0, d_tmp0);
    d_tmp1 = AE_SEL16_2301(d_tmp1, d_tmp1);
    d_tmp2 = AE_SEL16_2301(d_tmp2, d_tmp2);
    d_tmp3 = AE_SEL16_2301(d_tmp3, d_tmp3);

    d_inp0 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp0));
    d_inp1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp1));
    d_inp2 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp2));
    d_inp3 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_tmp3));

    MUL_SX2X2(d_inp0_t, d_inp1_t, d_inp0, d_inp1, d_one_over_out_scale, d_one_over_out_scale);
    MUL_SX2X2(d_inp2_t, d_inp3_t, d_inp2, d_inp3, d_one_over_out_scale, d_one_over_out_scale);

    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);
    d_inp2_t = XT_FIROUND_SX2(d_inp2_t);
    d_inp3_t = XT_FIROUND_SX2(d_inp3_t);

    d_out32_0 = XT_TRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_TRUNC_SX2(d_inp1_t, 0);
    d_out32_2 = XT_TRUNC_SX2(d_inp2_t, 0);
    d_out32_3 = XT_TRUNC_SX2(d_inp3_t, 0);

    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    d_out32_1 = AE_ADD32S(d_out32_1, d_out_zero_bias);
    d_out32_2 = AE_ADD32S(d_out32_2, d_out_zero_bias);
    d_out32_3 = AE_ADD32S(d_out32_3, d_out_zero_bias);

    d_out0 = AE_SAT16X4(d_out32_0, d_out32_1);
    d_out1 = AE_SAT16X4(d_out32_2, d_out32_3);

    AE_SAV16X4X2_XP(d_out0, d_out1, align_dst, p_o, rem_elm << 1);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}
#endif
