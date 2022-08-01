/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_quant_macros_hifi5.h"

WORD32 xa_nn_vec_leaky_relu_asym16s_asym16s( WORD16 * __restrict__ p_out,
                    const   WORD16 * __restrict__ p_vec,
                            WORD32 inp_zero_bias,
                            WORD32 alpha_multiplier,
                            WORD32 alpha_shift,
                            WORD32 out_multiplier,
                            WORD32 out_shift,
                            WORD32 out_zero_bias,
                            WORD32 vec_length)
{
  int i;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND(((alpha_shift < -31) || (alpha_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((alpha_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -32768) || (out_zero_bias > 32767)), -1);

  int rem_length = (vec_length & 7);

  WORD16 *p_o = p_out;
  WORD16 *p_v = (WORD16 *)p_vec;
 
  ae_int16x4 inp_zb_16 = AE_MOVDA16(inp_zero_bias);
#if TFLITE_SINGLE_ROUNDING
  int left_shift  = out_shift;
  int right_shift  = out_shift;
  int a_left_shift  = alpha_shift;
  int a_right_shift  = alpha_shift;
  (void)a_right_shift;
  (void)right_shift;
#else
  int left_shift  = out_shift<0?0: out_shift;
  int right_shift = out_shift>0?0:-out_shift;
  int a_left_shift  = alpha_shift<0?0: alpha_shift;
  int a_right_shift = alpha_shift>0?0:-alpha_shift;
#endif

  ae_valignx2 align_src  = AE_LA128_PP((ae_int16x8 *)p_v);
  ae_valignx2 align_dst = AE_ZALIGN128(); // zero alignment reg

  ae_int16x4 d_inp0, d_inp1;
  ae_int32x2 d_w0_0, d_w0_1, d_w0_2, d_w0_3;
  ae_int32x2 d_alpha_w0_0, d_alpha_w0_1;
  ae_int32x2 d_alpha_w0_2, d_alpha_w0_3;

#pragma concurrent
  for(i=0; i<(vec_length >> 3); i++)
  {
    AE_LA16X4X2_IP(d_inp0, d_inp1, align_src, (ae_int16x8 *)p_v);
    
    AE_SUBW16(d_w0_0, d_w0_1, d_inp0, inp_zb_16);
    AE_SUBW16(d_w0_2, d_w0_3, d_inp1, inp_zb_16);

    xtbool2 sel0 = AE_LT32(d_w0_0, AE_ZERO32());
    xtbool2 sel1 = AE_LT32(d_w0_1, AE_ZERO32());
    xtbool2 sel2 = AE_LT32(d_w0_2, AE_ZERO32());
    xtbool2 sel3 = AE_LT32(d_w0_3, AE_ZERO32());

    d_alpha_w0_0 = d_w0_0; d_alpha_w0_1 = d_w0_1;
    d_alpha_w0_2 = d_w0_2; d_alpha_w0_3 = d_w0_3;

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_w0_0, d_w0_1, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_w0_2, d_w0_3, d_w0_2, d_w0_3, out_multiplier, left_shift, right_shift);

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_alpha_w0_0, d_alpha_w0_1, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_alpha_w0_2, d_alpha_w0_3, d_alpha_w0_2, d_alpha_w0_3, alpha_multiplier, a_left_shift, a_right_shift);

    AE_MOVT32X2(d_w0_0, d_alpha_w0_0, sel0);
    AE_MOVT32X2(d_w0_1, d_alpha_w0_1, sel1);
    AE_MOVT32X2(d_w0_2, d_alpha_w0_2, sel2);
    AE_MOVT32X2(d_w0_3, d_alpha_w0_3, sel3);

    d_w0_0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_0);
    d_w0_1 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_1);
    d_w0_2 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_2);
    d_w0_3 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_3);

    ae_int16x4 out0,out1;
    out0 = AE_SAT16X4(d_w0_0, d_w0_1);
    out1 = AE_SAT16X4(d_w0_2, d_w0_3);

    AE_SA16X4X2_IP(out0, out1, align_dst, (ae_int16x8 *)p_o);
  }

  if( rem_length ){

    AE_LAV16X4X2_XP(d_inp0, d_inp1, align_src, (ae_int16x8 *)p_v, rem_length * 2);
    AE_SUBW16(d_w0_0, d_w0_1, d_inp0, inp_zb_16);
    AE_SUBW16(d_w0_2, d_w0_3, d_inp1, inp_zb_16);

    xtbool2 sel0 = AE_LT32(d_w0_0, AE_ZERO32());
    xtbool2 sel1 = AE_LT32(d_w0_1, AE_ZERO32());
    xtbool2 sel2 = AE_LT32(d_w0_2, AE_ZERO32());
    xtbool2 sel3 = AE_LT32(d_w0_3, AE_ZERO32());

    d_alpha_w0_0 = d_w0_0; d_alpha_w0_1 = d_w0_1;
    d_alpha_w0_2 = d_w0_2; d_alpha_w0_3 = d_w0_3;

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_w0_0, d_w0_1, d_w0_0, d_w0_1, out_multiplier, left_shift, right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_w0_2, d_w0_3, d_w0_2, d_w0_3, out_multiplier, left_shift, right_shift);

    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_alpha_w0_0, d_alpha_w0_1, d_alpha_w0_0, d_alpha_w0_1, alpha_multiplier, a_left_shift, a_right_shift);
    MPY_BY_QUANT_MULT_SLS_X2X2_OUT32_ALT(d_alpha_w0_2, d_alpha_w0_3, d_alpha_w0_2, d_alpha_w0_3, alpha_multiplier, a_left_shift, a_right_shift);

    AE_MOVT32X2(d_w0_0, d_alpha_w0_0, sel0);
    AE_MOVT32X2(d_w0_1, d_alpha_w0_1, sel1);
    AE_MOVT32X2(d_w0_2, d_alpha_w0_2, sel2);
    AE_MOVT32X2(d_w0_3, d_alpha_w0_3, sel3);

    d_w0_0 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_0);
    d_w0_1 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_1);
    d_w0_2 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_2);
    d_w0_3 = AE_ADD32S(AE_MOVDA32(out_zero_bias), d_w0_3);

    ae_int16x4 out0,out1;
    out0 = AE_SAT16X4(d_w0_0, d_w0_1);
    out1 = AE_SAT16X4(d_w0_2, d_w0_3);

    AE_SAV16X4X2_XP(out0, out1, align_dst, (ae_int16x8 *)p_o, rem_length * 2);
  }

  AE_SA128POS_FP(align_dst, p_o); // finalize the stream
  return 0;
}
