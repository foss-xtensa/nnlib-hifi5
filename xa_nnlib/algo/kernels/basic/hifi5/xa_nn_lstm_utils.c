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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_quant_macros_hifi5.h"

WORD32 xa_nn_elm_add_16x16_16(WORD16 * __restrict__ p_out,
                        const WORD16 * __restrict__ p_inp1,
                        const WORD16 * __restrict__ p_inp2,
                              WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm < 0), -1);

  int i;
  ae_int16x8 * __restrict__ p_a;
  ae_int16x8 * __restrict__ p_b;
  ae_int16x8 *__restrict__ p_c;

  // intermediate results and scratch registers
  ae_int16x4 a0_3, a4_7, b0_3, b4_7;

  ae_int16x4 out0, out1;

  int num_simd8_ops;
  int num_scalar_ops;

  num_simd8_ops = num_elm >> 3;
  num_scalar_ops = num_elm & 7;

  ae_valignx2 va_a, va_b, va_c;

  p_a = (ae_int16x8 *)p_inp1;
  p_b = (ae_int16x8 *)p_inp2;
  p_c = (ae_int16x8 *)p_out;

  va_a = AE_LA128_PP(p_a);
  va_b = AE_LA128_PP(p_b);
  va_c = AE_ZALIGN128();
  for(i = 0; i < num_simd8_ops; i++)
  {
    AE_LA16X4X2_IP(a0_3, a4_7, va_a, p_a);
    AE_LA16X4X2_IP(b0_3, b4_7, va_b, p_b);

    out0 = AE_ADD16S(a0_3, b0_3);
    out1 = AE_ADD16S(a4_7, b4_7);

    AE_SA16X4X2_IP(out0, out1, va_c, p_c);
  }

  if(num_scalar_ops != 0)
  {
    AE_LAV16X4X2_XP(a0_3, a4_7, va_a, p_a, (num_scalar_ops << 1));
    AE_LAV16X4X2_XP(b0_3, b4_7, va_b, p_b, (num_scalar_ops << 1));

    out0 = AE_ADD16S(a0_3, b0_3);
    out1 = AE_ADD16S(a4_7, b4_7);

    AE_SAV16X4X2_XP(out0, out1, va_c, p_c, (num_scalar_ops << 1));
  }
  AE_SA128POS_FP(va_c, p_c);
  return 0;
}

WORD32 xa_nn_lstm_cell_state_update_16(WORD16* p_cell_state,
                                           const WORD16* p_forget_gate,
                                           const WORD16* p_cell_gate,
                                           const WORD16* p_input_gate,
                                           WORD32 cell_to_forget_shift,
                                           WORD32 cell_to_input_shift,
                                           WORD32 quantized_cell_clip,
                                           WORD32 num_elms)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_cell_state, -1);
  XA_NNLIB_ARG_CHK_PTR(p_forget_gate, -1);
  XA_NNLIB_ARG_CHK_PTR(p_cell_gate, -1);
  XA_NNLIB_ARG_CHK_PTR(p_input_gate, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_cell_state, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_forget_gate, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_cell_gate, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_input_gate, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((cell_to_forget_shift < -31 || cell_to_forget_shift > -1), -1);
  XA_NNLIB_ARG_CHK_COND((cell_to_input_shift < -31 || cell_to_input_shift > -1), -1);
  XA_NNLIB_ARG_CHK_COND((num_elms < 0), -1);

  WORD32 ctof_right_shift, ctoi_right_shift;

#if TFLITE_SINGLE_ROUNDING
  ctof_right_shift = -cell_to_forget_shift;
  ctoi_right_shift = -cell_to_input_shift;
  ae_int32x2 d_ctof_rs = AE_MOVDA32(1 << (31 - ctof_right_shift));
  ae_int32x2 d_ctoi_rs = AE_MOVDA32(1 << (31 - ctoi_right_shift));
#else
  ctof_right_shift = -cell_to_forget_shift - 1;
  ctoi_right_shift = -cell_to_input_shift - 1;
  ae_int32x2 d_1_rs = AE_MOVDA32(1 << 30);
#endif

  const ae_int16x8 *p16x8_cs_r, *p16x8_fg_r;
  const ae_int16x8 *p16x8_cg_r, *p16x8_ig_r;

  ae_int16x8* p16x8_cs_w;

  ae_valignx2 align_cs_r, align_fg_r;
  ae_valignx2 align_cg_r, align_ig_r;
  ae_valignx2 align_cs_w;

  ae_int16x4 d_cs_r_0, d_cs_r_1;
  ae_int16x4 d_fg_0, d_fg_1;
  ae_int16x4 d_cg_0, d_cg_1;
  ae_int16x4 d_ig_0, d_ig_1;
  ae_int16x4 d_cs_w_0, d_cs_w_1;
  ae_int32x2 d_mul_0, d_mul_1, d_mul_2, d_mul_3;
  ae_int32x2 d_mul_4, d_mul_5, d_mul_6, d_mul_7;

  ae_int16x4 d_min, d_max;

  int i = 0;
  p16x8_cs_r = (const ae_int16x8*)p_cell_state;
  p16x8_fg_r = (const ae_int16x8*)p_forget_gate;
  p16x8_cg_r = (const ae_int16x8*)p_cell_gate;
  p16x8_ig_r = (const ae_int16x8*)p_input_gate;

  p16x8_cs_w = (ae_int16x8*)p_cell_state;

  align_fg_r = AE_LA128_PP(p16x8_fg_r);
  align_cg_r = AE_LA128_PP(p16x8_cg_r);
  align_ig_r = AE_LA128_PP(p16x8_ig_r);

  if (quantized_cell_clip > 0) {
    d_min = AE_MOVDA16(-quantized_cell_clip);
    d_max = AE_MOVDA16(quantized_cell_clip);
  } else {
    d_min = AE_MOVDA16(-32768);
    d_max = AE_MOVDA16(32767);
  }
  int pre_loop_count = ((16 - (((unsigned)p_cell_state)&15))&15)>>1;
  pre_loop_count = pre_loop_count > num_elms ? num_elms : pre_loop_count;
  int core_loop_count = num_elms - pre_loop_count;
  int post_loop_count = core_loop_count & 7;

  if(pre_loop_count)
  {
    align_cs_r = AE_LA128_PP(p16x8_cs_r);

    AE_LAV16X4X2_XP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r, (pre_loop_count << 1));
    AE_LAV16X4X2_XP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r, (pre_loop_count << 1));
    AE_LAV16X4X2_XP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r, (pre_loop_count << 1));
    AE_LAV16X4X2_XP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r, (pre_loop_count << 1));

    AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
    AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);

#if TFLITE_SINGLE_ROUNDING
    AE_MULF2P32X4RAS(d_mul_0, d_mul_1, d_mul_0, d_mul_1, d_ctof_rs, d_ctof_rs);
    AE_MULF2P32X4RAS(d_mul_2, d_mul_3, d_mul_2, d_mul_3, d_ctof_rs, d_ctof_rs);
    d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
    d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);
#else
    AE_MULF2P32X4RAS(d_mul_0, d_mul_1, d_mul_0, d_mul_1, d_1_rs, d_1_rs);
    AE_MULF2P32X4RAS(d_mul_2, d_mul_3, d_mul_2, d_mul_3, d_1_rs, d_1_rs);
    d_mul_0 = AE_SRAA32SYMS(d_mul_0, ctof_right_shift);
    d_mul_1 = AE_SRAA32SYMS(d_mul_1, ctof_right_shift);
    d_mul_2 = AE_SRAA32SYMS(d_mul_2, ctof_right_shift);
    d_mul_3 = AE_SRAA32SYMS(d_mul_3, ctof_right_shift);
    d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
    d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);
#endif

    AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
    AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);

#if TFLITE_SINGLE_ROUNDING
    AE_MULF2P32X4RAS(d_mul_4, d_mul_5, d_mul_4, d_mul_5, d_ctoi_rs, d_ctoi_rs);
    AE_MULF2P32X4RAS(d_mul_6, d_mul_7, d_mul_6, d_mul_7, d_ctoi_rs, d_ctoi_rs);
    d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
    d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);
#else
    AE_MULF2P32X4RAS(d_mul_4, d_mul_5, d_mul_4, d_mul_5, d_1_rs, d_1_rs);
    AE_MULF2P32X4RAS(d_mul_6, d_mul_7, d_mul_6, d_mul_7, d_1_rs, d_1_rs);
    d_mul_4 = AE_SRAA32SYMS(d_mul_4, ctoi_right_shift);
    d_mul_5 = AE_SRAA32SYMS(d_mul_5, ctoi_right_shift);
    d_mul_6 = AE_SRAA32SYMS(d_mul_6, ctoi_right_shift);
    d_mul_7 = AE_SRAA32SYMS(d_mul_7, ctoi_right_shift);
    d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
    d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);
#endif

    d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
    d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

    AE_MINMAX16(d_cs_w_0, d_min, d_max);
    AE_MINMAX16(d_cs_w_1, d_min, d_max);

    align_cs_w = AE_ZALIGN128();
    AE_SAV16X4X2_XP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w, (pre_loop_count << 1));
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);
  }

#pragma concurrent
  for (i = 0; i < (core_loop_count >> 3); i++)
  {
    AE_L16X4X2_IP(d_cs_r_0, d_cs_r_1, p16x8_cs_r, 16);
    AE_LA16X4X2_IP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r);
    AE_LA16X4X2_IP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r);
    AE_LA16X4X2_IP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r);

    AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
    AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);

#if TFLITE_SINGLE_ROUNDING
    AE_MULF2P32X4RAS(d_mul_0, d_mul_1, d_mul_0, d_mul_1, d_ctof_rs, d_ctof_rs);
    AE_MULF2P32X4RAS(d_mul_2, d_mul_3, d_mul_2, d_mul_3, d_ctof_rs, d_ctof_rs);
    d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
    d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);
#else
    AE_MULF2P32X4RAS(d_mul_0, d_mul_1, d_mul_0, d_mul_1, d_1_rs, d_1_rs);
    AE_MULF2P32X4RAS(d_mul_2, d_mul_3, d_mul_2, d_mul_3, d_1_rs, d_1_rs);
    d_mul_0 = AE_SRAA32SYMS(d_mul_0, ctof_right_shift);
    d_mul_1 = AE_SRAA32SYMS(d_mul_1, ctof_right_shift);
    d_mul_2 = AE_SRAA32SYMS(d_mul_2, ctof_right_shift);
    d_mul_3 = AE_SRAA32SYMS(d_mul_3, ctof_right_shift);
    d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
    d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);
#endif

    AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
    AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);

#if TFLITE_SINGLE_ROUNDING
    AE_MULF2P32X4RAS(d_mul_4, d_mul_5, d_mul_4, d_mul_5, d_ctoi_rs, d_ctoi_rs);
    AE_MULF2P32X4RAS(d_mul_6, d_mul_7, d_mul_6, d_mul_7, d_ctoi_rs, d_ctoi_rs);
    d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
    d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);
#else
    AE_MULF2P32X4RAS(d_mul_4, d_mul_5, d_mul_4, d_mul_5, d_1_rs, d_1_rs);
    AE_MULF2P32X4RAS(d_mul_6, d_mul_7, d_mul_6, d_mul_7, d_1_rs, d_1_rs);
    d_mul_4 = AE_SRAA32SYMS(d_mul_4, ctoi_right_shift);
    d_mul_5 = AE_SRAA32SYMS(d_mul_5, ctoi_right_shift);
    d_mul_6 = AE_SRAA32SYMS(d_mul_6, ctoi_right_shift);
    d_mul_7 = AE_SRAA32SYMS(d_mul_7, ctoi_right_shift);
    d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
    d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);
#endif

    d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
    d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

    AE_MINMAX16(d_cs_w_0, d_min, d_max);
    AE_MINMAX16(d_cs_w_1, d_min, d_max);

    AE_S16X4X2_IP(d_cs_w_0, d_cs_w_1, p16x8_cs_w, 16);
  }

  if(post_loop_count)
  {
    align_cs_r = AE_LA128_PP(p16x8_cs_r);

    AE_LAV16X4X2_XP(d_cs_r_0, d_cs_r_1, align_cs_r, p16x8_cs_r, (post_loop_count << 1));
    AE_LAV16X4X2_XP(d_fg_0, d_fg_1, align_fg_r, p16x8_fg_r, (post_loop_count << 1));
    AE_LAV16X4X2_XP(d_cg_0, d_cg_1, align_cg_r, p16x8_cg_r, (post_loop_count << 1));
    AE_LAV16X4X2_XP(d_ig_0, d_ig_1, align_ig_r, p16x8_ig_r, (post_loop_count << 1));

    AE_MUL16X4(d_mul_0, d_mul_1, d_cs_r_0, d_fg_0);
    AE_MUL16X4(d_mul_2, d_mul_3, d_cs_r_1, d_fg_1);

#if TFLITE_SINGLE_ROUNDING
    AE_MULF2P32X4RAS(d_mul_0, d_mul_1, d_mul_0, d_mul_1, d_ctof_rs, d_ctof_rs);
    AE_MULF2P32X4RAS(d_mul_2, d_mul_3, d_mul_2, d_mul_3, d_ctof_rs, d_ctof_rs);
    d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
    d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);
#else
    AE_MULF2P32X4RAS(d_mul_0, d_mul_1, d_mul_0, d_mul_1, d_1_rs, d_1_rs);
    AE_MULF2P32X4RAS(d_mul_2, d_mul_3, d_mul_2, d_mul_3, d_1_rs, d_1_rs);
    d_mul_0 = AE_SRAA32SYMS(d_mul_0, ctof_right_shift);
    d_mul_1 = AE_SRAA32SYMS(d_mul_1, ctof_right_shift);
    d_mul_2 = AE_SRAA32SYMS(d_mul_2, ctof_right_shift);
    d_mul_3 = AE_SRAA32SYMS(d_mul_3, ctof_right_shift);
    d_cs_w_0 = AE_SAT16X4(d_mul_0, d_mul_1);
    d_cs_w_1 = AE_SAT16X4(d_mul_2, d_mul_3);
#endif

    AE_MUL16X4(d_mul_4, d_mul_5, d_cg_0, d_ig_0);
    AE_MUL16X4(d_mul_6, d_mul_7, d_cg_1, d_ig_1);

#if TFLITE_SINGLE_ROUNDING
    AE_MULF2P32X4RAS(d_mul_4, d_mul_5, d_mul_4, d_mul_5, d_ctoi_rs, d_ctoi_rs);
    AE_MULF2P32X4RAS(d_mul_6, d_mul_7, d_mul_6, d_mul_7, d_ctoi_rs, d_ctoi_rs);
    d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
    d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);
#else
    AE_MULF2P32X4RAS(d_mul_4, d_mul_5, d_mul_4, d_mul_5, d_1_rs, d_1_rs);
    AE_MULF2P32X4RAS(d_mul_6, d_mul_7, d_mul_6, d_mul_7, d_1_rs, d_1_rs);
    d_mul_4 = AE_SRAA32SYMS(d_mul_4, ctoi_right_shift);
    d_mul_5 = AE_SRAA32SYMS(d_mul_5, ctoi_right_shift);
    d_mul_6 = AE_SRAA32SYMS(d_mul_6, ctoi_right_shift);
    d_mul_7 = AE_SRAA32SYMS(d_mul_7, ctoi_right_shift);
    d_cg_0 = AE_SAT16X4(d_mul_4, d_mul_5);
    d_cg_1 = AE_SAT16X4(d_mul_6, d_mul_7);
#endif

    d_cs_w_0 = AE_ADD16S(d_cs_w_0, d_cg_0);
    d_cs_w_1 = AE_ADD16S(d_cs_w_1, d_cg_1);

    AE_MINMAX16(d_cs_w_0, d_min, d_max);
    AE_MINMAX16(d_cs_w_1, d_min, d_max);

    align_cs_w = AE_ZALIGN128();
    AE_SAV16X4X2_XP(d_cs_w_0, d_cs_w_1, align_cs_w, p16x8_cs_w, (post_loop_count << 1));
    AE_SA128POS_FP(align_cs_w, p16x8_cs_w);
  }
  return 0;
}
