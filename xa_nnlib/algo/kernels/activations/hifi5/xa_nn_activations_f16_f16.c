/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "../../../ndsp/hifi5/include/NatureDSP_Signal_math.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include <xtensa/tie/xt_hifi2.h>

#if HAVE_VFPU

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_sigmoid_f16_f16,(
    WORD16       *  p_out,
    const WORD16 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_sigmoid_f16_f16(
    WORD16       * __restrict__ p_out,        /* result, floating point */
    const WORD16 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_sigmoid_fp16(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_HP_VFPU */

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_tanh_f16_f16,(
    WORD16        *  p_out,
    const WORD16  *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_tanh_f16_f16(
    WORD16       * __restrict__ p_out,        /* result, floating point */
    const WORD16 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_tanh_fp16(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_HP_VFPU */
#endif

#if !HAVE_HP_VFPU

DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_softmax_f16_f16,(
    WORD16       *  p_out,
    const WORD16 *  p_vec,
    WORD16 *        pscale_beta,
    WORD32          vec_length)                  )

#else

/* This number sequence is used for handling out-of-bound elements in tail loops */
static const WORD16 seq_0_7[8] = {0, 1, 2, 3, 4, 5, 6, 7};

#define ADD_HX4T(OUT, IN1, IN2, MASK) { \
  OUT = IN1; \
  xthalfx4 add_temp = ADD_HX4(IN1, IN2); \
  MOVT_HX4(OUT, add_temp, MASK); \
}

#define RADD_HX4(OUTSUM, VEC) {\
  ae_int16x4 VEC_INT = AE_MOVINT16X4_FROMXTHALFX4(VEC); \
  VEC_INT = AE_SEL16I(VEC_INT, VEC_INT, 0); \
  xthalfx4 sum_temp = ADD_HX4(VEC, AE_MOVXTHALFX4_FROMINT16X4(VEC_INT)); \
  ae_int16x4 sum_temp_int = AE_MOVINT16X4_FROMXTHALFX4(sum_temp); \
  sum_temp_int = AE_SEL16I(sum_temp_int, sum_temp_int, 4); \
  OUTSUM = ADD_HX4(sum_temp, AE_MOVXTHALFX4_FROMINT16X4(sum_temp_int)); \
}

/* Decomposed LUTs using multiplicative factors to reduce table size.
 * For an index computed as: idx = 8*n2 + n1  (0 <= n1,n2 < 8)
 * the original 64-entry value exp[idx] equals:
 *   exp[idx] = exp_lut_n1[n1] * exp_lut_n2[n2]
 * Store the two 8-entry vectors per sign (n1 and n2), total 16 entries.
 */
static const uint16_t __attribute__((aligned(16))) exp_neg_lut_n1[8] = { 0x3C00, 0x3A3B, 0x38DA, 0x378F, 0x35E3, 0x3496, 0x3324, 0x3190 };
static const uint16_t __attribute__((aligned(16))) exp_neg_lut_n2[8] = { 0x3C00, 0x3055, 0x24B0, 0x1914, 0x0D7F, 0x02FA, 0x0067, 0x000E };
static const uint16_t __attribute__((aligned(16))) exp_pos_lut_n1[8] = { 0x3C00, 0x3D23, 0x3E98, 0x403C, 0x4170, 0x42FB, 0x447B, 0x45C1 };
static const uint16_t __attribute__((aligned(16))) exp_pos_lut_n2[8] = { 0x3C00, 0x4764, 0x52D3, 0x5E4E, 0x69D2, 0x7561, 0x7BFF, 0x7BFF };

/* LUT tables for log computation.
 * For mantissa m in [1, 2), take top 3 fraction bits as index (0-7).
 * Grid points: m_grid = 1 + idx*0.125 = {1.0, 1.125, 1.25, ..., 1.875}
 * log_lut[i]  = log(1 + i*0.125) in fp16
 * log_inv_lut[i] = 1/(1 + i*0.125) in fp16
 */
static const uint16_t __attribute__((aligned(16))) log_lut[8] = {
    0x0000, 0x2F8A, 0x3324, 0x3518, 0x367D, 0x37C5, 0x387A, 0x3907
};
static const uint16_t __attribute__((aligned(16))) log_inv_lut[8] = {
    0x3C00, 0x3B1C, 0x3A66, 0x39D1, 0x3955, 0x38EC, 0x3892, 0x3844
};

/* COMPUTE_EXP_NEG_HX4X2: computes exp(x) for two xthalfx4 vectors, negative inputs only.
 * Used in softmax where x = (elem - max) * beta <= 0.
 * Positive LUT path, sign-selection, and upper saturation are removed for cycle savings.
 * exp(-|x|) = exp_neg_lut_n1[n1] * exp_neg_lut_n2[n2] * poly(0.125 - residual) * e^(-1/8)
 * where idx = floor(|x|*4)&63, n1 = idx&7, n2 = idx>>3, residual = |x| - idx*0.25 in [0,0.25)
 */
  #define COMPUTE_EXP_NEG_HX4X2(hfvecResultTemp0, hfvecResultTemp1, hfvecTemp0, hfvecTemp1) \
  {                                                                                          \
    xthalfx4 __hf_x0 = (hfvecTemp0);                                                        \
    xthalfx4 __hf_x1 = (hfvecTemp1);                                                        \
    xthalfx4 __hf_abs0, __hf_abs1, __hf_res0, __hf_res1;                                    \
    xthalfx4 __hf_inp_min_1by8_0, __hf_inp_min_1by8_1;                                      \
    xthalfx4 __hf_zeroes, __hf_minus16F, __hf_oneF, __hf_1by8F, __hf_exp_neg_1by8;          \
    xtbool4  __hf_b0, __hf_b1;                                                              \
                                                                                             \
    /* constants */                                                                          \
    __hf_zeroes      = CONST_HX4(0);                                                        \
    __hf_minus16F    = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xCC00)); /* -16.0f */         \
    __hf_oneF        = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */           \
    __hf_1by8F       = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3000)); /* 0.125f */         \
    __hf_exp_neg_1by8= AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3B0F)); /* e^(-1/8) */       \
                                                                                             \
    __hf_res0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */                  \
    __hf_res1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */                  \
                                                                                             \
    /* abs inp (x <= 0, so abs = -x) */                                                      \
    ABS_HX4X2(__hf_abs0, __hf_abs1, __hf_x0, __hf_x1);                                      \
                                                                                             \
    /* LUT-based range reduction: idx = floor(|x|*4) & 0x3F                              */  \
    /* Decomposed as idx = 8*n2 + n1, so exp(-idx*0.25) = neg_lut_n1[n1] * neg_lut_n2[n2] */ \
    ae_int16x4 __hf_idx0 = AE_AND16(TRUNC16_HX4(__hf_abs0, 2), AE_MOVDA16(0x003F));         \
    ae_int16x4 __hf_idx1 = AE_AND16(TRUNC16_HX4(__hf_abs1, 2), AE_MOVDA16(0x003F));         \
    /* Residual: abs - idx*0.25, in [0, 0.25) */                                             \
    xthalfx4 __hf_idx_f0 = FLOAT16_HX4(__hf_idx0, 2);                                       \
    xthalfx4 __hf_idx_f1 = FLOAT16_HX4(__hf_idx1, 2);                                       \
    SUB_HX4X2(__hf_abs0, __hf_abs1, __hf_abs0, __hf_abs1, __hf_idx_f0, __hf_idx_f1);        \
    /* n1 = idx & 7, n2 = idx >> 3; XOR inverts bit order for AE_SEL16X4 addressing */       \
    ae_int16x4 __hf_n1_0 = AE_XOR16(__hf_idx0,               AE_MOVDA16(0xFFFF));            \
    ae_int16x4 __hf_n1_1 = AE_XOR16(__hf_idx1,               AE_MOVDA16(0xFFFF));            \
    ae_int16x4 __hf_n2_0 = AE_XOR16(AE_SRLI16(__hf_idx0, 3), AE_MOVDA16(0xFFFF));            \
    ae_int16x4 __hf_n2_1 = AE_XOR16(AE_SRLI16(__hf_idx1, 3), AE_MOVDA16(0xFFFF));            \
                                                                                             \
    /* LUT lookups (negative path only) */                                                   \
    ae_int16x4 __hf_tbl0 = *(const ae_int16x4 *)exp_neg_lut_n1;                              \
    ae_int16x4 __hf_tbl1 = *(const ae_int16x4 *)(exp_neg_lut_n1 + 4);                        \
    ae_int16x4 __hf_n1_val0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_0);                  \
    ae_int16x4 __hf_n1_val1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_1);                  \
    __hf_tbl0 = *(const ae_int16x4 *)exp_neg_lut_n2;                                         \
    __hf_tbl1 = *(const ae_int16x4 *)(exp_neg_lut_n2 + 4);                                   \
    ae_int16x4 __hf_n2_val0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_0);                  \
    ae_int16x4 __hf_n2_val1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_1);                  \
    MUL_HX4X2(__hf_res0, __hf_res1,                                                          \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n1_val0),                                     \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n1_val1),                                     \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n2_val0),                                     \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n2_val1));                                    \
                                                                                             \
    /* Polynomial residual for [0, 0.25) interval.                                       */  \
    /* inp_min_1by8 = 0.125 - abs_residual (always negated since input is always neg).   */  \
    /* When abs_residual==0 (exact multiple of 0.25): outTemp is overridden to 1.0.      */  \
    __hf_b0 = OEQ_HX4(__hf_abs0, __hf_zeroes); __hf_b1 = OEQ_HX4(__hf_abs1, __hf_zeroes);  \
    SUB_HX4X2(__hf_inp_min_1by8_0, __hf_inp_min_1by8_1, __hf_1by8F, __hf_1by8F, __hf_abs0, __hf_abs1); \
                                                                                             \
    /* polynomial evaluation */                                                              \
    xthalfx4 __hf_hfC0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x2955)); /* 1/24 */       \
    xthalfx4 __hf_hfC0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x2955));                   \
    xthalfx4 __hf_hfC1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155)); /* 1/6 */        \
    xthalfx4 __hf_hfC1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155));                   \
    xthalfx4 __hf_hfC2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3800)); /* 0.5 */        \
    xthalfx4 __hf_hfC2_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3800));                   \
    xthalfx4 __hf_hfC3_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0 */        \
    xthalfx4 __hf_hfC3_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                   \
    xthalfx4 __hf_outTemp_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                \
    xthalfx4 __hf_outTemp_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                \
                                                                                             \
    MADD_HX4X2(__hf_hfC1_0, __hf_hfC1_1, __hf_hfC0_0, __hf_hfC0_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_hfC2_0, __hf_hfC2_1, __hf_hfC1_0, __hf_hfC1_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_hfC3_0, __hf_hfC3_1, __hf_hfC2_0, __hf_hfC2_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_hfC3_0, __hf_hfC3_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MUL_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_outTemp_0, __hf_outTemp_1, __hf_exp_neg_1by8, __hf_exp_neg_1by8); \
    /* if abs_residual == 0: no polynomial correction needed, exp = LUT result exactly */ \
    MOVT_HX4(__hf_outTemp_0, __hf_oneF, __hf_b0); MOVT_HX4(__hf_outTemp_1, __hf_oneF, __hf_b1); \
    MUL_HX4X2(__hf_res0, __hf_res1, __hf_res0, __hf_res1, __hf_outTemp_0, __hf_outTemp_1); \
                                                                                             \
    /* saturation: if inp <= -16 => 0 */                                                     \
    __hf_b0 = OLE_HX4(__hf_x0, __hf_minus16F); __hf_b1 = OLE_HX4(__hf_x1, __hf_minus16F);  \
    MOVT_HX4(__hf_res0, __hf_zeroes, __hf_b0); MOVT_HX4(__hf_res1, __hf_zeroes, __hf_b1);  \
                                                                                             \
    (hfvecResultTemp0) = __hf_res0; (hfvecResultTemp1) = __hf_res1;                          \
  }



WORD32 xa_nn_vec_softmax_f16_f16(
  WORD16       * __restrict__ p_out,  /* float16 output buffer of length vec_length */
  const WORD16 * __restrict__ p_vec,  /* float16 input buffer of length vec_length */
  WORD16 *      pscale_beta,          /* Scalar float16 value, passed via pointer */
  WORD32        vec_length)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_PTR(pscale_beta, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((*pscale_beta <= 0), -1);

  int i;
  xthalfx8 * __restrict__ p_vec_fp16 = (xthalfx8 *)p_vec;
  xthalfx8 * __restrict__ p_out_fp16 = (xthalfx8 *)p_out;

  xthalfx4 max_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xFC00));
  ae_valignx2 va_in = AE_LA128_PP((void *)p_vec_fp16);

  for(i = 0; i < (vec_length & ~7); i+=8)
  {
    xthalfx4 vec0, vec1;
    AE_LAHX4X2_IP(vec0, vec1, va_in, p_vec_fp16);
    max_vec = MAX_HX4(max_vec, vec0);
    max_vec = MAX_HX4(max_vec, vec1);
  }
  int rem_len = vec_length & 7;
  if(rem_len > 0)
  {
    xthalfx4 vec0, vec1;
    AE_LAVHX4X2_XP(vec0, vec1, va_in, p_vec_fp16, rem_len*sizeof(WORD16));

    /* Handle out-of-bound elements */
    ae_int16x4 seq0, seq1;
    AE_L16X4X2_I(seq0, seq1, (void *)seq_0_7, 0);
    xtbool4 b0 = AE_LT16(seq0, AE_MOVDA16(rem_len));
    xtbool4 b1 = AE_LT16(seq1, AE_MOVDA16(rem_len));
    MOVF_HX4(vec0, max_vec, b0);
    MOVF_HX4(vec1, max_vec, b1);

    max_vec = MAX_HX4(max_vec, vec0);
    max_vec = MAX_HX4(max_vec, vec1);
  }
  xthalf max_val = RMAXNUM_H(max_vec);

  p_vec_fp16 = (xthalfx8 *)p_vec;

  ae_valignx2 va_out = AE_ZALIGN128();
  xthalfx4 sum_v = CONST_HX4(0);
  xthalfx4 max_val_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*(WORD16*)&max_val));
  xthalfx4 beta_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*pscale_beta));

  va_in = AE_LA128_PP((void *)p_vec_fp16);

  for(i = 0; i < vec_length; i+=8)
  {
    int rem_len = vec_length - i;
    rem_len = (rem_len > 8) ? 8 : rem_len;
  
    xthalfx4 elem0_f16, elem1_f16, exp_in0, exp_in1, exp_out0, exp_out1;
    AE_LAVHX4X2_XP(elem0_f16, elem1_f16, va_in, p_vec_fp16, rem_len*sizeof(WORD16));

    SUB_HX4X2(exp_in0, exp_in1, elem0_f16, elem1_f16, max_val_vec, max_val_vec);
    MUL_HX4X2(exp_in0, exp_in1, exp_in0, exp_in1, beta_vec, beta_vec);

    COMPUTE_EXP_NEG_HX4X2(exp_out0, exp_out1, exp_in0, exp_in1);

    AE_SAVHX4X2_XP(exp_out0, exp_out1, va_out, p_out_fp16, rem_len*sizeof(WORD16));

    ae_int16x4 seq0, seq1;
    AE_L16X4X2_I(seq0, seq1, (void *)seq_0_7, 0);
    xtbool4 b0 = AE_LT16(seq0, AE_MOVDA16(rem_len));
    xtbool4 b1 = AE_LT16(seq1, AE_MOVDA16(rem_len));

    ADD_HX4T(sum_v, sum_v, exp_out0, b0);
    ADD_HX4T(sum_v, sum_v, exp_out1, b1);
  }
  AE_SA128POS_FP(va_out, (void *)p_out_fp16);
  xthalfx4 outsum;
  RADD_HX4(outsum, sum_v);

  xthalfx4 inv_sum_vec = RECIP_HX4(outsum);
  p_vec_fp16 = (xthalfx8 *)p_out; // Input exponents are stored at p_out
  p_out_fp16 = (xthalfx8 *)p_out;

  va_in = AE_LA128_PP((void *)p_vec_fp16);

  for(i = 0; i < vec_length; i+=8)
  {
    int rem_len = vec_length - i;
    rem_len = (rem_len > 8) ? 8 : rem_len;

    xthalfx4 out0_f16, out1_f16;
    AE_LAVHX4X2_XP(out0_f16, out1_f16, va_in, p_vec_fp16, rem_len*sizeof(WORD16));
    MUL_HX4X2(out0_f16, out1_f16, out0_f16, out1_f16, inv_sum_vec, inv_sum_vec);
    AE_SAVHX4X2_XP(out0_f16, out1_f16, va_out, p_out_fp16, rem_len*sizeof(WORD16));
  }
  AE_SA128POS_FP(va_out, (void *)p_out_fp16);

  return 0;
}
#endif

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_leaky_relu_f16_f16,(
    WORD16       *  p_out,
    const WORD16 *  p_vec,
    WORD16 *        slope,
    WORD32          vec_length)                  )
#else

  WORD32 xa_nn_vec_leaky_relu_f16_f16(
    WORD16       * __restrict__ p_out,  /* float16 output buffer of length vec_length */
    const WORD16 * __restrict__ p_vec,  /* float16 input buffer of length vec_length */
    WORD16 *      slope,          /* Scalar float16 value, passed via pointer */
    WORD32        vec_length)
  {
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(slope, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((*slope <= 0), -1);
    
    
    const xthalfx8 * restrict pX; 
    xthalfx8 * restrict pY; 
    pX = (const xthalfx8 *)p_vec;
    pY = (xthalfx8 *)p_out;
    

    xthalfx4 slope_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((void*)slope,0));

    xthalfx4 x0, x1;
    xthalfx4 x0_0, x1_0, zeroes;

    zeroes = CONST_HX4(0);

    ae_valignx2 X_va, Y_va;
    X_va = AE_LA128_PP(pX);
    Y_va = AE_ZALIGN128();

      
    for (int n = 0; n<(vec_length>>3); n++)
    {
      AE_LAHX4X2_IP(x0, x1, X_va, pX);
      xtbool4 b0, b1;
      b0 = OLT_HX4(x0, zeroes);
      b1 = OLT_HX4(x1, zeroes);
      MUL_HX4X2(x0_0, x1_0, x0, x1, slope_vec, slope_vec);
      //p_out_fp16[i] = (p_vec_fp16[i] < 0.0f) ? (xthalf)((float)p_vec_fp16[i] * (float)slope_val) : p_vec_fp16[i];
      MOVT_HX4(x0, x0_0, b0);
      MOVT_HX4(x1, x1_0, b1);      
      AE_SAHX4X2_IP(x0, x1, Y_va, pY);
    }

    int rem_val = vec_length & 7;
    if(rem_val){

      AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
      xtbool4 b0, b1;
      b0 = OLT_HX4(x0, zeroes);
      b1 = OLT_HX4(x1, zeroes);
      MUL_HX4X2(x0_0, x1_0, x0, x1, slope_vec, slope_vec);
      //p_out_fp16[i] = (p_vec_fp16[i] < 0.0f) ? (xthalf)((float)p_vec_fp16[i] * (float)slope_val) : p_vec_fp16[i];
      MOVT_HX4(x0, x0_0, b0);
      MOVT_HX4(x1, x1_0, b1);      
      AE_SAVHX4X2_XP(x0, x1, Y_va, pY, rem_val*sizeof(WORD16));
    }

    AE_SA128POS_FP(Y_va, pY);

    return 0;
  }
#endif

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_prelu_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,  
      WORD16 *      slope,         
      WORD32        vec_length)                 )
#else

  WORD32 xa_nn_vec_prelu_f16_f16(
        WORD16       * __restrict__ p_out,  
        const WORD16 * __restrict__ p_vec,  
        WORD16 *      slope,         
        WORD32        vec_length)
  {
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    XA_NNLIB_ARG_CHK_PTR(slope, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

    const xthalfx8 * restrict pX;
    const xthalfx8 * restrict pslope;
    xthalfx8 * restrict pY; 
    pX = (const xthalfx8 *)p_vec;
    pY = (xthalfx8 *)p_out;
    pslope = (const xthalfx8 *)slope;

    xthalfx4 x0, x1, slope_vec0, slope_vec1;
    xthalfx4 x0_0, x1_0, zeroes;

    zeroes = CONST_HX4(0);

    ae_valignx2 X_va, Y_va, slope_va;
    X_va = AE_LA128_PP(pX);
    slope_va = AE_LA128_PP(pslope);
    Y_va = AE_ZALIGN128();

    
    for (int n = 0; n<(vec_length>>3); n++)
    {
      AE_LAHX4X2_IP(x0, x1, X_va, pX);
      AE_LAHX4X2_IP(slope_vec0, slope_vec1, slope_va, pslope);
      xtbool4 b0, b1;
      b0 = OLT_HX4(x0, zeroes);
      b1 = OLT_HX4(x1, zeroes);
      MUL_HX4X2(x0_0, x1_0, x0, x1, slope_vec0, slope_vec1);
      MOVT_HX4(x0, x0_0, b0);
      MOVT_HX4(x1, x1_0, b1);      
      AE_SAHX4X2_IP(x0, x1, Y_va, pY);
    }

    int rem_val = vec_length & 7;
    if(rem_val){

      AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
      AE_LAVHX4X2_XP (slope_vec0, slope_vec1, slope_va, pslope,rem_val*sizeof(WORD16));
      xtbool4 b0, b1;
      b0 = OLT_HX4(x0, zeroes);
      b1 = OLT_HX4(x1, zeroes);
      MUL_HX4X2(x0_0, x1_0, x0, x1, slope_vec0, slope_vec1);
      MOVT_HX4(x0, x0_0, b0);
      MOVT_HX4(x1, x1_0, b1);      
      AE_SAVHX4X2_XP(x0, x1, Y_va, pY, rem_val*sizeof(WORD16));
    }

    AE_SA128POS_FP(Y_va, pY);
    return 0;
  }
#endif

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_hardswish_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,         
      WORD32        vec_length)                  )
#else

  WORD32 xa_nn_vec_hardswish_f16_f16(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,         
      WORD32        vec_length)

  {
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    
    
    const xthalfx8 * restrict pX; 
    xthalfx8 * restrict pY; 
    pX = (const xthalfx8 *)p_vec;
    pY = (xthalfx8 *)p_out;

    xthalfx4 x0, x1, x0_plus3, x1_plus3;
    xthalfx4 zeroes, three, six, inv_six;
    xthalfx4 x0_0, x1_0, y0, y1;

    zeroes = CONST_HX4(0);
    three = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x4200));
    six = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x4600));
    inv_six = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155));

    ae_valignx2 X_va, Y_va;
    X_va = AE_LA128_PP(pX);
    Y_va = AE_ZALIGN128();

      
    for (int n = 0; n<(vec_length>>3); n++)
    {
      AE_LAHX4X2_IP(x0, x1, X_va, pX);
      ADD_HX4X2(x0_plus3, x1_plus3, x0, x1, three, three);
      xtbool4 b0, b1;

      //max(0, x + 3) -> stored in xi_0
      b0 = OLT_HX4(x0_plus3, zeroes); // if x0_plus3 < 0, b0 = 1
      b1 = OLT_HX4(x1_plus3, zeroes);
      MOVT_HX4(x0_plus3, zeroes, b0); // move zero into locations wherever x0_plus3 < 0
      MOVT_HX4(x1_plus3, zeroes, b1);     
      
      //min(6,xi) -> stored in xi_0
      b0 = OLT_HX4(six,x0_plus3);  // if x0_plus3 > 6, b0 = 1
      b1 = OLT_HX4(six,x1_plus3);  
      MOVT_HX4(x0_plus3, six, b0); //move six into locations wherever max(0,x0_plus3) > 6
      MOVT_HX4(x1_plus3, six, b1);

      MUL_HX4X2(x0_0, x1_0, x0_plus3, x1_plus3, inv_six, inv_six);
      MUL_HX4X2(y0, y1, x0, x1, x0_0, x1_0);
      AE_SAHX4X2_IP(y0, y1, Y_va, pY);
    }

    int rem_val = vec_length & 7;
    if(rem_val){

      AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
      ADD_HX4X2(x0_plus3, x1_plus3, x0, x1, three, three);
      xtbool4 b0, b1;
      //w = x .* min(6, max(0, x + 3)) / 6;

      //max(0, x + 3) -> stored in xi_0
      b0 = OLT_HX4(x0_plus3, zeroes); // if x0_plus3 < 0, b0 = 1
      b1 = OLT_HX4(x1_plus3, zeroes);
      MOVT_HX4(x0_plus3, zeroes, b0); // move zero into locations wherever x0_plus3 < 0
      MOVT_HX4(x1_plus3, zeroes, b1);     
      
      //min(6,xi) -> stored in xi_0
      b0 = OLT_HX4(six,x0_plus3);  // if x0_plus3 > 6, b0 = 1
      b1 = OLT_HX4(six,x1_plus3);  
      MOVT_HX4(x0_plus3, six, b0); //move six into locations wherever max(0,x0_plus3) > 6
      MOVT_HX4(x1_plus3, six, b1);

      MUL_HX4X2(x0_0, x1_0, x0_plus3, x1_plus3, inv_six, inv_six);
      MUL_HX4X2(y0, y1, x0, x1, x0_0, x1_0);  
      AE_SAVHX4X2_XP(y0, y1, Y_va, pY, rem_val*sizeof(WORD16));
    }

    AE_SA128POS_FP(Y_va, pY);
    return 0;
  }
#endif

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_exp_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,         
      WORD32        vec_length)                  )
#else
    
/*
 * COMPUTE_EXP_FULLRANGE_LUT (HiFi5)
 * Inlined implementation of the previous `__exp_fp16` helper so the
 * macro is self-contained and can be modified to use decomposed LUTs
 * later.  This preserves the original range-splitting, polynomial
 * residual and saturation behaviour.
 */
#define COMPUTE_EXP_FULLRANGE_LUT(result0, result1, x0, x1)                             \
  {                                                                                  \
    xthalfx4 __hf_x0 = (x0);                                                            \
    xthalfx4 __hf_x1 = (x1);                                                            \
    xthalfx4 __hf_abs0, __hf_abs1, __hf_res0, __hf_res1, __hf_const_big, __hf_const_sm; \
    xthalfx4 __hf_exp_plus_1by8_0, __hf_exp_plus_1by8_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1, __hf_temp0, __hf_temp1; \
    xthalfx4 __hf_zeroes, __hf_minus16F, __hf_elevenF, __hf_FP16MAX, __hf_oneF, __hf_1by8F; \
    xtbool4  __hf_b0, __hf_b1, __hf_b0_pos, __hf_b1_pos, __hf_b0_neg, __hf_b1_neg;      \
                                                                                         \
    /* constants */                                                                      \
    __hf_zeroes  = CONST_HX4(0);                                                        \
    __hf_minus16F= AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xCC00)); /* -16.0f */        \
    __hf_elevenF = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x4980)); /* 11.0f */         \
    __hf_oneF    = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */          \
    __hf_1by8F   = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3000)); /* 0.125f */        \
    __hf_FP16MAX = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x7BFF)); /* 65504 */         \
                                                                                         \
    __hf_res0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */            \
    __hf_res1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */            \
                                                                                         \
    /* abs inp calculation */                                                            \
    ABS_HX4X2(__hf_abs0, __hf_abs1, __hf_x0, __hf_x1);                                 \
                                                                                         \
    /* expPlusOneByEight selection based on sign */                                      \
    __hf_b0_pos = OLE_HX4(__hf_zeroes, __hf_x0); __hf_b0_neg = OLT_HX4(__hf_x0, __hf_zeroes); \
    __hf_b1_pos = OLE_HX4(__hf_zeroes, __hf_x1); __hf_b1_neg = OLT_HX4(__hf_x1, __hf_zeroes); \
    __hf_const_sm = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3B0F)); /* 0.88232421875 */  \
    __hf_const_big= AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C88)); /* 1.1328125 */      \
    MOVT_HX4(__hf_exp_plus_1by8_0, __hf_const_sm, __hf_b0_neg); MOVT_HX4(__hf_exp_plus_1by8_1, __hf_const_sm, __hf_b1_neg); \
    MOVT_HX4(__hf_exp_plus_1by8_0, __hf_const_big, __hf_b0_pos); MOVT_HX4(__hf_exp_plus_1by8_1, __hf_const_big, __hf_b1_pos); \
    /* LUT-based range reduction: idx = floor(|x|*4) & 0x3F                              */ \
    /* idx encodes which 0.25-step bucket |x| falls in (0..63).                        */ \
    /* Decomposed as idx = 8*n2 + n1, so exp(+-idx*0.25) = lut_n1[n1] * lut_n2[n2].   */ \
    ae_int16x4 __hf_idx0 = AE_AND16(TRUNC16_HX4(__hf_abs0, 2), AE_MOVDA16(0x003F));     \
    ae_int16x4 __hf_idx1 = AE_AND16(TRUNC16_HX4(__hf_abs1, 2), AE_MOVDA16(0x003F));     \
    /* Residual: abs - idx*0.25, stays in [0, 0.25) for the polynomial */               \
    xthalfx4 __hf_idx_f0 = FLOAT16_HX4(__hf_idx0, 2);                                   \
    xthalfx4 __hf_idx_f1 = FLOAT16_HX4(__hf_idx1, 2);                                   \
    SUB_HX4X2(__hf_abs0, __hf_abs1, __hf_abs0, __hf_abs1, __hf_idx_f0, __hf_idx_f1);   \
    /* n1 = idx & 7 (low 3 bits), n2 = idx >> 3 (high 3 bits).                        */ \
    /* XOR with 0xFFFF inverts bit order for AE_SEL16X4 element addressing.            */ \
    ae_int16x4 __hf_n1_0 = AE_XOR16(__hf_idx0,               AE_MOVDA16(0xFFFF));        \
    ae_int16x4 __hf_n1_1 = AE_XOR16(__hf_idx1,               AE_MOVDA16(0xFFFF));        \
    ae_int16x4 __hf_n2_0 = AE_XOR16(AE_SRLI16(__hf_idx0, 3), AE_MOVDA16(0xFFFF));        \
    ae_int16x4 __hf_n2_1 = AE_XOR16(AE_SRLI16(__hf_idx1, 3), AE_MOVDA16(0xFFFF));        \
    /* LUT lookups for negative exponents */                                             \
    ae_int16x4 __hf_tbl0 = *(const ae_int16x4 *)exp_neg_lut_n1;                          \
    ae_int16x4 __hf_tbl1 = *(const ae_int16x4 *)(exp_neg_lut_n1 + 4);                    \
    ae_int16x4 __hf_neg_n1_0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_0);             \
    ae_int16x4 __hf_neg_n1_1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_1);             \
    __hf_tbl0 = *(const ae_int16x4 *)exp_neg_lut_n2;                                     \
    __hf_tbl1 = *(const ae_int16x4 *)(exp_neg_lut_n2 + 4);                                \
    ae_int16x4 __hf_neg_n2_0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_0);             \
    ae_int16x4 __hf_neg_n2_1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_1);             \
    /* LUT lookups for positive exponents */                                             \
    __hf_tbl0 = *(const ae_int16x4 *)exp_pos_lut_n1;                                     \
    __hf_tbl1 = *(const ae_int16x4 *)(exp_pos_lut_n1 + 4);                                \
    ae_int16x4 __hf_pos_n1_0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_0);             \
    ae_int16x4 __hf_pos_n1_1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_1);             \
    __hf_tbl0 = *(const ae_int16x4 *)exp_pos_lut_n2;                                     \
    __hf_tbl1 = *(const ae_int16x4 *)(exp_pos_lut_n2 + 4);                                \
    ae_int16x4 __hf_pos_n2_0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_0);             \
    ae_int16x4 __hf_pos_n2_1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_1);             \
    /* Select neg or pos factor based on sign, then multiply n1*n2 = exp(+-idx*0.25) */ \
    ae_int16x4 __hf_sel_n1_0 = __hf_pos_n1_0; AE_MOVT16X4(__hf_sel_n1_0, __hf_neg_n1_0, __hf_b0_neg); \
    ae_int16x4 __hf_sel_n1_1 = __hf_pos_n1_1; AE_MOVT16X4(__hf_sel_n1_1, __hf_neg_n1_1, __hf_b1_neg); \
    ae_int16x4 __hf_sel_n2_0 = __hf_pos_n2_0; AE_MOVT16X4(__hf_sel_n2_0, __hf_neg_n2_0, __hf_b0_neg); \
    ae_int16x4 __hf_sel_n2_1 = __hf_pos_n2_1; AE_MOVT16X4(__hf_sel_n2_1, __hf_neg_n2_1, __hf_b1_neg); \
    MUL_HX4X2(__hf_res0, __hf_res1,                                                      \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_sel_n1_0),                                \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_sel_n1_1),                                \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_sel_n2_0),                                \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_sel_n2_1));                               \
    /* if absInp == 0 */                                                                 \
    __hf_b0 = OEQ_HX4(__hf_abs0, __hf_zeroes); __hf_b1 = OEQ_HX4(__hf_abs1, __hf_zeroes); \
    SUB_HX4X2(__hf_inp_min_1by8_0, __hf_inp_min_1by8_1, __hf_abs0, __hf_abs1, __hf_1by8F, __hf_1by8F); \
    MOVT_HX4(__hf_inp_min_1by8_0, __hf_zeroes, __hf_b0); MOVT_HX4(__hf_inp_min_1by8_1, __hf_zeroes, __hf_b1); \
    NEG_HX4X2(__hf_temp0, __hf_temp1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MOVT_HX4(__hf_inp_min_1by8_0, __hf_temp0, __hf_b0_neg); MOVT_HX4(__hf_inp_min_1by8_1, __hf_temp1, __hf_b1_neg); \
                                                                                         \
    /* polynomial evaluation */                                                          \
    xthalfx4 __hf_hfC0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x2955)); /* 1/24 */     \
    xthalfx4 __hf_hfC0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x2955));                    \
    xthalfx4 __hf_hfC1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155)); /* 1/6 */      \
    xthalfx4 __hf_hfC1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155));                    \
    xthalfx4 __hf_hfC2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3800)); /* 0.5 */      \
    xthalfx4 __hf_hfC2_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3800));                    \
    xthalfx4 __hf_hfC3_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0 */      \
    xthalfx4 __hf_hfC3_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                    \
    xthalfx4 __hf_outTemp_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                \
    xthalfx4 __hf_outTemp_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                \
                                                                                         \
    MADD_HX4X2(__hf_hfC1_0, __hf_hfC1_1, __hf_hfC0_0, __hf_hfC0_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_hfC2_0, __hf_hfC2_1, __hf_hfC1_0, __hf_hfC1_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_hfC3_0, __hf_hfC3_1, __hf_hfC2_0, __hf_hfC2_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_hfC3_0, __hf_hfC3_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MUL_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_outTemp_0, __hf_outTemp_1, __hf_exp_plus_1by8_0, __hf_exp_plus_1by8_1); \
    MOVT_HX4(__hf_outTemp_0, __hf_oneF, __hf_b0); MOVT_HX4(__hf_outTemp_1, __hf_oneF, __hf_b1); \
    MUL_HX4X2(__hf_res0, __hf_res1, __hf_res0, __hf_res1, __hf_outTemp_0, __hf_outTemp_1); \
                                                                                         \
    /* saturation: if inp <= -16 => 0; if inp >= 11 => FP16MAX */                        \
    __hf_b0 = OLE_HX4(__hf_x0, __hf_minus16F); __hf_b1 = OLE_HX4(__hf_x1, __hf_minus16F); \
    MOVT_HX4(__hf_res0, __hf_zeroes, __hf_b0); MOVT_HX4(__hf_res1, __hf_zeroes, __hf_b1); \
    __hf_b0 = OLE_HX4(__hf_elevenF, __hf_x0); __hf_b1 = OLE_HX4(__hf_elevenF, __hf_x1); \
    MOVT_HX4(__hf_res0, __hf_FP16MAX, __hf_b0); MOVT_HX4(__hf_res1, __hf_FP16MAX, __hf_b1); \
                                                                                         \
    /* write back to caller-provided results */                                           \
    (result0) = __hf_res0; (result1) = __hf_res1;                                         \
  }

WORD32 xa_nn_vec_exp_f16_f16(
    WORD16       * __restrict__ p_out,  
    const WORD16 * __restrict__ p_vec,         
    WORD32        vec_length)
{
        /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

    const xthalfx8 * __restrict__ pX; xthalfx8 * __restrict__ pY; 
    pX = (const xthalfx8 *)p_vec; pY = (xthalfx8 *)p_out;

    ae_valignx2 X_va, Y_va;
    X_va = AE_LA128_PP(pX);
    Y_va = AE_ZALIGN128();

    xthalfx4 x0, x1, result0, result1;

    for (int n = 0; n<(vec_length>>3); n++)
    {
        AE_LAHX4X2_IP(x0, x1, X_va, pX);
        COMPUTE_EXP_FULLRANGE_LUT(result0, result1, x0, x1);
        AE_SAHX4X2_IP(result0, result1, Y_va, pY);
    }

    int rem_val = vec_length & 7;
    if(rem_val){

        AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
        COMPUTE_EXP_FULLRANGE_LUT(result0, result1, x0, x1);
        AE_SAVHX4X2_XP(result0, result1, Y_va, pY, rem_val*sizeof(WORD16));
    }

    AE_SA128POS_FP(Y_va, pY);

    return 0;
}
#endif /* !HAVE_HP_VFPU */

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_sqrt_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,       
      WORD32        vec_length)                )
#else
      WORD32 xa_nn_vec_sqrt_f16_f16(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,         
      WORD32        vec_length)
      {
          /* NULL pointer checks */
        XA_NNLIB_ARG_CHK_PTR(p_out, -1);
        XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
        /* Basic Parameter checks */
        XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
        
        
        const xthalfx8 * restrict pX; 
        xthalfx8 * restrict pY; 
        pX = (const xthalfx8 *)p_vec;
        pY = (xthalfx8 *)p_out;

        xthalfx4 x0, x1, y0, y1;

        ae_valignx2 X_va, Y_va;
        X_va = AE_LA128_PP(pX);
        Y_va = AE_ZALIGN128();

        for (int n = 0; n<(vec_length>>3); n++)
        {
          AE_LAHX4X2_IP(x0, x1, X_va, pX);
          y0 = FSQRT_HX4(x0);
          y1 = FSQRT_HX4(x1); 
          AE_SAHX4X2_IP(y0, y1, Y_va, pY);
        }

        int rem_val = vec_length & 7;
        if(rem_val){

          AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
          y0 = FSQRT_HX4(x0);
          y1 = FSQRT_HX4(x1);       
          AE_SAVHX4X2_XP(y0, y1, Y_va, pY, rem_val*sizeof(WORD16));
        }

        AE_SA128POS_FP(Y_va, pY);

        return 0;
      }
#endif /* !HAVE_HP_VFPU */

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_selu_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,   
      WORD16* alpha,
      WORD16* lambda,     
      WORD32  vec_length)                )
#else
  
 #define COMPUTE_EXPM1_NEG_HX4X2(hfvecResultTemp0, hfvecResultTemp1, hfvecTemp0, hfvecTemp1) \
  {                                                                                          \
    xthalfx4 __hf_x0 = (hfvecTemp0);                                                        \
    xthalfx4 __hf_x1 = (hfvecTemp1);                                                        \
    xthalfx4 __hf_abs0, __hf_abs1, __hf_res0, __hf_res1;                                    \
    xthalfx4 __hf_inp_min_1by8_0, __hf_inp_min_1by8_1;                                      \
    xthalfx4 __hf_zeroes, __hf_minus16F, __hf_oneF;          \
    xtbool4  __hf_b0, __hf_b1;                                                              \
                                                                                             \
    /* constants */                                                                          \
    __hf_zeroes      = CONST_HX4(0);                                                        \
    __hf_minus16F    = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xCC00)); /* -16.0f */         \
    __hf_oneF        = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */           \
                                                                                             \
    __hf_res0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */                  \
    __hf_res1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f */                  \
                                                                                             \
    /* abs inp (x <= 0, so abs = -x) */                                                      \
    ABS_HX4X2(__hf_abs0, __hf_abs1, __hf_x0, __hf_x1);                                      \
                                                                                             \
    /* LUT-based range reduction: idx = floor(|x|*4) & 0x3F                              */  \
    /* Decomposed as idx = 8*n2 + n1, so exp(-idx*0.25) = neg_lut_n1[n1] * neg_lut_n2[n2] */ \
    ae_int16x4 __hf_idx0 = AE_AND16(TRUNC16_HX4(__hf_abs0, 2), AE_MOVDA16(0x003F));         \
    ae_int16x4 __hf_idx1 = AE_AND16(TRUNC16_HX4(__hf_abs1, 2), AE_MOVDA16(0x003F));         \
    /* Residual: abs - idx*0.25, in [0, 0.25) */                                             \
    xthalfx4 __hf_idx_f0 = FLOAT16_HX4(__hf_idx0, 2);                                       \
    xthalfx4 __hf_idx_f1 = FLOAT16_HX4(__hf_idx1, 2);                                       \
    SUB_HX4X2(__hf_abs0, __hf_abs1, __hf_abs0, __hf_abs1, __hf_idx_f0, __hf_idx_f1);        \
    /* n1 = idx & 7, n2 = idx >> 3; XOR inverts bit order for AE_SEL16X4 addressing */       \
    ae_int16x4 __hf_n1_0 = AE_XOR16(__hf_idx0,               AE_MOVDA16(0xFFFF));            \
    ae_int16x4 __hf_n1_1 = AE_XOR16(__hf_idx1,               AE_MOVDA16(0xFFFF));            \
    ae_int16x4 __hf_n2_0 = AE_XOR16(AE_SRLI16(__hf_idx0, 3), AE_MOVDA16(0xFFFF));            \
    ae_int16x4 __hf_n2_1 = AE_XOR16(AE_SRLI16(__hf_idx1, 3), AE_MOVDA16(0xFFFF));            \
                                                                                             \
    /* LUT lookups (negative path only) */                                                   \
    ae_int16x4 __hf_tbl0 = *(const ae_int16x4 *)exp_neg_lut_n1;                              \
    ae_int16x4 __hf_tbl1 = *(const ae_int16x4 *)(exp_neg_lut_n1 + 4);                        \
    ae_int16x4 __hf_n1_val0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_0);                  \
    ae_int16x4 __hf_n1_val1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n1_1);                  \
    __hf_tbl0 = *(const ae_int16x4 *)exp_neg_lut_n2;                                         \
    __hf_tbl1 = *(const ae_int16x4 *)(exp_neg_lut_n2 + 4);                                   \
    ae_int16x4 __hf_n2_val0 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_0);                  \
    ae_int16x4 __hf_n2_val1 = AE_SEL16X4(__hf_tbl0, __hf_tbl1, __hf_n2_1);                  \
    MUL_HX4X2(__hf_res0, __hf_res1,                                                          \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n1_val0),                                     \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n1_val1),                                     \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n2_val0),                                     \
               AE_MOVXTHALFX4_FROMINT16X4(__hf_n2_val1));                                    \
                                                                                             \
    /* Polynomial residual for [0, 0.25) interval.                                       */  \
    /* inp_min_1by8 = 0.125 - abs_residual (always negated since input is always neg).   */  \
    /* When abs_residual==0 (exact multiple of 0.25): outTemp is overridden to 0.      */  \
    __hf_b0 = OEQ_HX4(__hf_abs0, __hf_zeroes); __hf_b1 = OEQ_HX4(__hf_abs1, __hf_zeroes);  \
    NEG_HX4X2(__hf_inp_min_1by8_0, __hf_inp_min_1by8_1, __hf_abs0, __hf_abs1);/* r is now in the range of (-0.25,0] */ \
                                                                                             \
    /* polynomial evaluation */                                                              \
    xthalfx4 __hf_hfC0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x2955)); /* 1/24 */       \
    xthalfx4 __hf_hfC0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x2955));                   \
    xthalfx4 __hf_hfC1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155)); /* 1/6 */        \
    xthalfx4 __hf_hfC1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3155));                   \
    xthalfx4 __hf_hfC2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3800)); /* 0.5 */        \
    xthalfx4 __hf_hfC2_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3800));                   \
    xthalfx4 __hf_hfC3_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0 */        \
    xthalfx4 __hf_hfC3_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                   \
    xthalfx4 __hf_outTemp_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                \
    xthalfx4 __hf_outTemp_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                \
                                                                                             \
    MADD_HX4X2(__hf_hfC1_0, __hf_hfC1_1, __hf_hfC0_0, __hf_hfC0_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_hfC2_0, __hf_hfC2_1, __hf_hfC1_0, __hf_hfC1_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MADD_HX4X2(__hf_hfC3_0, __hf_hfC3_1, __hf_hfC2_0, __hf_hfC2_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    MUL_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_hfC3_0, __hf_hfC3_1, __hf_inp_min_1by8_0, __hf_inp_min_1by8_1); \
    /* MUL_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_outTemp_0, __hf_outTemp_1, __hf_exp_neg_1by8, __hf_exp_neg_1by8); */ \
    /* if abs_residual == 0: no polynomial correction needed, exp = LUT result exactly */ \
    MOVT_HX4(__hf_outTemp_0, __hf_zeroes, __hf_b0); MOVT_HX4(__hf_outTemp_1, __hf_zeroes, __hf_b1); \
        /* saturation: if inp <= -16 => -1 */                                                     \
    __hf_b0 = OLE_HX4(__hf_x0, __hf_minus16F); __hf_b1 = OLE_HX4(__hf_x1, __hf_minus16F);  \
    xthalfx4 __hf_minus1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xBC00));  \
    MOVT_HX4(__hf_outTemp_0, __hf_minus1, __hf_b0); MOVT_HX4(__hf_outTemp_1, __hf_minus1, __hf_b1);  \
    MUL_HX4X2(__hf_outTemp_0, __hf_outTemp_1, __hf_res0, __hf_res1, __hf_outTemp_0, __hf_outTemp_1); /* Multiply with A = exp(-idx*0.25) i.e. we have A*(exp(r)-1) */ \
    SUB_HX4X2(__hf_res0, __hf_res1, __hf_res0, __hf_res1, __hf_oneF, __hf_oneF); /* compute (A-1)*/                                 \
    ADD_HX4X2(__hf_res0, __hf_res1, __hf_res0, __hf_res1, __hf_outTemp_0, __hf_outTemp_1); /* Compute A*(exp(r)-1) + (A-1) = A*exp(r) -1 */ \
    (hfvecResultTemp0) = __hf_res0; (hfvecResultTemp1) = __hf_res1;                          \
  }
  
      WORD32 xa_nn_vec_selu_f16_f16(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,
      WORD16* alpha,
      WORD16* lambda,          
      WORD32  vec_length)
      {
            /* NULL pointer checks */
        XA_NNLIB_ARG_CHK_PTR(p_out, -1);
        XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
        XA_NNLIB_ARG_CHK_PTR(alpha, -1);
        XA_NNLIB_ARG_CHK_PTR(lambda, -1);
        /* Basic Parameter checks */
        XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
        XA_NNLIB_ARG_CHK_COND((*alpha <= 0), -1);
        XA_NNLIB_ARG_CHK_COND((*lambda <= 0), -1);
        
        
        const xthalfx8 * restrict pX; 
        xthalfx8 * restrict pY; 
        pX = (const xthalfx8 *)p_vec;
        pY = (xthalfx8 *)p_out;
        

        xthalfx4 alpha_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((void*)alpha,0));
        xthalfx4 lambda_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_L16_I((void*)lambda,0));

        xthalfx4 x0, x1, y0, y1;
        xthalfx4 x0_1, x1_1, zeroes;

        zeroes = CONST_HX4(0);

        ae_valignx2 X_va, Y_va;
        X_va = AE_LA128_PP(pX);
        Y_va = AE_ZALIGN128();

        for (int n = 0; n<(vec_length>>3); n++)
        {
          AE_LAHX4X2_IP(x0, x1, X_va, pX);
          
          xtbool4 b0, b1; 
          // w(neg_idx) = lambda .* (alpha .* exp(x(neg_idx)) - alpha);
          b0 = OLE_HX4(x0, zeroes);
          b1 = OLE_HX4(x1, zeroes);
          x0_1 = CONST_HX4(1); x1_1 = CONST_HX4(1);
          MOVT_HX4(x0_1, x0, b0);
          MOVT_HX4(x1_1, x1, b1);
          COMPUTE_EXPM1_NEG_HX4X2(y0, y1, x0_1, x1_1);
          MUL_HX4X2(y0, y1, y0, y1, alpha_vec, alpha_vec); 
          
          // w(pos_idx) = lambda .* x(pos_idx);
          b0 = ULT_HX4(zeroes, x0);
          b1 = ULT_HX4(zeroes, x1);
          MOVT_HX4(y0, x0, b0);
          MOVT_HX4(y1, x1, b1);
          MUL_HX4X2(y0, y1, y0, y1, lambda_vec, lambda_vec);


          AE_SAHX4X2_IP(y0, y1, Y_va, pY);
        }

        int rem_val = vec_length & 7;
        if(rem_val){

          AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
  
          xtbool4 b0, b1; 
          // w(neg_idx) = lambda .* (alpha .* exp(x(neg_idx)) - alpha);
          b0 = OLE_HX4(x0, zeroes);
          b1 = OLE_HX4(x1, zeroes);
          x0_1 = CONST_HX4(1); x1_1 = CONST_HX4(1);
          MOVT_HX4(x0_1, x0, b0);
          MOVT_HX4(x1_1, x1, b1);
          COMPUTE_EXPM1_NEG_HX4X2(y0, y1, x0_1, x1_1);
          MUL_HX4X2(y0, y1, y0, y1, alpha_vec, alpha_vec); 
          
          // w(pos_idx) = lambda .* x(pos_idx);
          b0 = ULT_HX4(zeroes, x0);
          b1 = ULT_HX4(zeroes, x1);
          MOVT_HX4(y0, x0, b0);
          MOVT_HX4(y1, x1, b1);
          MUL_HX4X2(y0, y1, y0, y1, lambda_vec, lambda_vec);

          AE_SAVHX4X2_XP(y0, y1, Y_va, pY, rem_val*sizeof(WORD16));
        }

        AE_SA128POS_FP(Y_va, pY);
        return 0;
      }
#endif /* !HAVE_HP_VFPU */


#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_logsoftmax_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,       
      WORD32        vec_length)                )
#else

  #define COMPUTE_LOG_POSITIVE_HX4X2(result0, result1, x0, x1)                                \
    {                                                                                           \
      xthalfx4 __hfl_wx0    = (x0);                                                        \
      xthalfx4 __hfl_wx1    = (x1); /* working copies that get inverted/clamped */         \
      xthalfx4 __hfl_res0, __hfl_res1;                                                         \
      xthalfx4 __hfl_mant0, __hfl_mant1;                                                       \
      xthalfx4 __hfl_y0, __hfl_y1, __hfl_y2_0, __hfl_y2_1;                                    \
      xthalfx4 __hfl_mp1_0, __hfl_mp1_1, __hfl_mm1_0, __hfl_mm1_1;                            \
      xthalfx4 __hfl_oneF, __hfl_FP16MAX, __hfl_log2F;                                         \
      xtbool4  __hfl_b0_xLT1, __hfl_b1_xLT1, __hfl_b0, __hfl_b1;                              \
                                                                                                \
      /* constants */                                                                           \
      __hfl_oneF   = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0f              */   \
      __hfl_FP16MAX= AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x7BFF)); /* 65504.0f          */   \
      __hfl_log2F  = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x398C)); /* ln(2) ~0.6934f16  */   \
                                                                                                \
      /* if x < 1: invert to 1/x (log(x) = -log(1/x)); clamp inverted value to FP16_MAX */    \
      __hfl_b0_xLT1 = OLT_HX4(x0, __hfl_oneF);                                          \
      __hfl_b1_xLT1 = OLT_HX4(x1, __hfl_oneF);                                          \
      xthalfx4 __hfl_inv0 = RECIP_HX4(x0);                                              \
      xthalfx4 __hfl_inv1 = RECIP_HX4(x1);                                              \
      MOVT_HX4(__hfl_wx0, __hfl_inv0, __hfl_b0_xLT1);                                         \
      MOVT_HX4(__hfl_wx1, __hfl_inv1, __hfl_b1_xLT1);                                         \
      __hfl_b0 = OLT_HX4(__hfl_FP16MAX, __hfl_wx0); /* wx > 65504 after inversion */          \
      __hfl_b1 = OLT_HX4(__hfl_FP16MAX, __hfl_wx1);                                           \
      MOVT_HX4(__hfl_wx0, __hfl_FP16MAX, __hfl_b0);                                           \
      MOVT_HX4(__hfl_wx1, __hfl_FP16MAX, __hfl_b1);                                           \
                                                                                                \
      /* Extract FP16 exponent N (= exp_bits - 15) and mantissa m = x / 2^N in [1, 2) */      \
      ae_int16x4 __hfl_hx0    = AE_MOVINT16X4_FROMXTHALFX4(__hfl_wx0);                        \
      ae_int16x4 __hfl_hx1    = AE_MOVINT16X4_FROMXTHALFX4(__hfl_wx1);                         \
      ae_int16x4 __hfl_exp0   = AE_AND16(AE_SRLI16(__hfl_hx0, 10), AE_MOVDA16(0x001F));       \
      ae_int16x4 __hfl_exp1   = AE_AND16(AE_SRLI16(__hfl_hx1, 10), AE_MOVDA16(0x001F));       \
      __hfl_b0 = AE_EQ16(__hfl_exp0, AE_MOVDA16(0x001F)); \
      __hfl_b1 = AE_EQ16(__hfl_exp1, AE_MOVDA16(0x001F)); \
      ae_int16x4 __hfl_N0     = AE_SUB16(__hfl_exp0, AE_MOVDA16(15)); /* N = exp - bias */     \
      ae_int16x4 __hfl_N1     = AE_SUB16(__hfl_exp1, AE_MOVDA16(15));                          \
      ae_int16x4 __hfl_mbits0 = AE_OR16(AE_AND16(__hfl_hx0, AE_MOVDA16(0x03FF)), AE_MOVDA16(0x3C00)); \
      ae_int16x4 __hfl_mbits1 = AE_OR16(AE_AND16(__hfl_hx1, AE_MOVDA16(0x03FF)), AE_MOVDA16(0x3C00)); \
      __hfl_mant0 = AE_MOVXTHALFX4_FROMINT16X4(__hfl_mbits0); /* m in [1, 2) */               \
      __hfl_mant1 = AE_MOVXTHALFX4_FROMINT16X4(__hfl_mbits1);                                  \
                                                                                                \
      /* result = N * ln(2); FLOAT16_HX4(N, 0) converts integer N to float16 */               \
      MUL_HX4X2(__hfl_res0, __hfl_res1,                                                        \
                FLOAT16_HX4(__hfl_N0, 0), FLOAT16_HX4(__hfl_N1, 0),                          \
                __hfl_log2F, __hfl_log2F);                                                    \
                                                                                                \
      /* y = (m - 1) / (m + 1), in [0, 1/3) for m in [1, 2) */                               \
      ADD_HX4X2(__hfl_mp1_0, __hfl_mp1_1, __hfl_mant0, __hfl_mant1, __hfl_oneF, __hfl_oneF); \
      SUB_HX4X2(__hfl_mm1_0, __hfl_mm1_1, __hfl_mant0, __hfl_mant1, __hfl_oneF, __hfl_oneF); \
      MUL_HX4X2(__hfl_y0, __hfl_y1,                                                            \
                __hfl_mm1_0, __hfl_mm1_1,                                                     \
                RECIP_HX4(__hfl_mp1_0), RECIP_HX4(__hfl_mp1_1));                             \
      /* y2 = y*y; double y -> 2y for the polynomial */                                        \
      MUL_HX4X2(__hfl_y2_0, __hfl_y2_1, __hfl_y0, __hfl_y1, __hfl_y0, __hfl_y1);            \
      ADD_HX4X2(__hfl_y0, __hfl_y1, __hfl_y0, __hfl_y1, __hfl_y0, __hfl_y1);                 \
                                                                                                \
      /* Horner evaluation of ln(m) = 2y*(1 + y2*(1/3 + y2*(1/5 + y2/7)))               */   \
      xthalfx4 __hfl_c0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3092)); /* 1/7 */        \
      xthalfx4 __hfl_c0_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3092));                   \
      xthalfx4 __hfl_c1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3266)); /* 1/5 */        \
      xthalfx4 __hfl_c1_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3266));                   \
      xthalfx4 __hfl_c2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3555)); /* 1/3 */        \
      xthalfx4 __hfl_c2_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3555));                   \
      xthalfx4 __hfl_c3_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0 */        \
      xthalfx4 __hfl_c3_1 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00));                   \
      MADD_HX4X2(__hfl_c1_0, __hfl_c1_1, __hfl_c0_0, __hfl_c0_1, __hfl_y2_0, __hfl_y2_1);  \
      MADD_HX4X2(__hfl_c2_0, __hfl_c2_1, __hfl_c1_0, __hfl_c1_1, __hfl_y2_0, __hfl_y2_1);  \
      MADD_HX4X2(__hfl_c3_0, __hfl_c3_1, __hfl_c2_0, __hfl_c2_1, __hfl_y2_0, __hfl_y2_1);  \
      MADD_HX4X2(__hfl_res0, __hfl_res1, __hfl_c3_0, __hfl_c3_1, __hfl_y0, __hfl_y1);       \
                                                                                                \
      /* if x was < 1: negate result (log(1/x) = -log(x)) */                                  \
      xthalfx4 __hfl_neg0, __hfl_neg1;                                                         \
      NEG_HX4X2(__hfl_neg0, __hfl_neg1, __hfl_res0, __hfl_res1);                              \
      MOVT_HX4(__hfl_res0, __hfl_neg0, __hfl_b0_xLT1);                                        \
      MOVT_HX4(__hfl_res1, __hfl_neg1, __hfl_b1_xLT1);                                        \
      MOVT_HX4(__hfl_res0,__hfl_wx0,__hfl_b0);                                                \
      MOVT_HX4(__hfl_res1,__hfl_wx1,__hfl_b1);                                                \
      /* log(0) = -inf: booleans from original x0/x1, short live range */                     \
      xthalfx4 __hfl_neginf = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xFC00));                 \
      xtbool4 __hfl_b0_xEQ0 = OEQ_HX4(x0, AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x0000))); \
      xtbool4 __hfl_b1_xEQ0 = OEQ_HX4(x1, AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x0000))); \
      MOVT_HX4(__hfl_res0, __hfl_neginf, __hfl_b0_xEQ0);                                      \
      MOVT_HX4(__hfl_res1, __hfl_neginf, __hfl_b1_xEQ0);                                      \
      /* log(+inf) = +inf: booleans from original x0/x1, short live range */                  \
      xthalfx4 __hfl_posinf = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x7C00));                 \
      xtbool4 __hfl_b0_xIsInf = OEQ_HX4(x0, __hfl_posinf);                             \
      xtbool4 __hfl_b1_xIsInf = OEQ_HX4(x1, __hfl_posinf);                             \
      MOVT_HX4(__hfl_res0, __hfl_posinf, __hfl_b0_xIsInf);                                    \
      MOVT_HX4(__hfl_res1, __hfl_posinf, __hfl_b1_xIsInf);                                    \
      (result0) = __hfl_res0; (result1) = __hfl_res1;                                          \
    }

      WORD32 xa_nn_vec_logsoftmax_f16_f16(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,         
      WORD32        vec_length)
      {
        /* NULL pointer checks */
        XA_NNLIB_ARG_CHK_PTR(p_out, -1);
        XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
        /* Basic Parameter checks */
        XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
        
        int i;
        xthalfx8 * __restrict__ p_vec_fp16 = (xthalfx8 *)p_vec;
        xthalfx8 * __restrict__ p_out_fp16 = (xthalfx8 *)p_out;

        xthalfx4 max_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xFC00)); //-inf
        ae_valignx2 va_in = AE_LA128_PP((void *)p_vec_fp16);

        /* ---------- find max element in input vector for numerical stability of exp calculation ------------- */
        for(i = 0; i < (vec_length & ~7); i+=8)
        {
          xthalfx4 vec0, vec1;
          AE_LAHX4X2_IP(vec0, vec1, va_in, p_vec_fp16);
          max_vec = MAX_HX4(max_vec, vec0);
          max_vec = MAX_HX4(max_vec, vec1);
        }
        int rem_len = vec_length & 7;
        if(rem_len > 0)
        {
          xthalfx4 vec0, vec1;
          AE_LAVHX4X2_XP(vec0, vec1, va_in, p_vec_fp16, rem_len*sizeof(WORD16));

          /* Handle out-of-bound elements */
          ae_int16x4 seq0, seq1;
          AE_L16X4X2_I(seq0, seq1, (void *)seq_0_7, 0);
          xtbool4 b0 = AE_LT16(seq0, AE_MOVDA16(rem_len));
          xtbool4 b1 = AE_LT16(seq1, AE_MOVDA16(rem_len));
          MOVF_HX4(vec0, max_vec, b0);
          MOVF_HX4(vec1, max_vec, b1);

          max_vec = MAX_HX4(max_vec, vec0);
          max_vec = MAX_HX4(max_vec, vec1);
        }
        xthalf max_val = RMAXNUM_H(max_vec);
        xthalfx4 max_val_vec = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*(WORD16*)&max_val));
        /* -----------------------------------------------------------------------------  */ 

        p_vec_fp16 = (xthalfx8 *)p_vec;
        ae_valignx2 va_out = AE_ZALIGN128();
        xthalfx4 sum_v = CONST_HX4(0);
        
        va_in = AE_LA128_PP((void *)p_vec_fp16);

        /* --------------- sum_over_j(exp(xj-xmax)) -------------- */
        for(i = 0; i < vec_length; i+=8)
        {
          int rem_len = vec_length - i;
          rem_len = (rem_len > 8) ? 8 : rem_len;
        
          xthalfx4 elem0_f16, elem1_f16, exp_in0, exp_in1, exp_out0, exp_out1;
          AE_LAVHX4X2_XP(elem0_f16, elem1_f16, va_in, p_vec_fp16, rem_len*sizeof(WORD16));
          SUB_HX4X2(exp_in0, exp_in1, elem0_f16, elem1_f16, max_val_vec, max_val_vec); //(xi - xmax)
          COMPUTE_EXP_NEG_HX4X2(exp_out0, exp_out1, exp_in0, exp_in1); //(xi - xmax) <= 0 always

          ae_int16x4 seq0, seq1;
          AE_L16X4X2_I(seq0, seq1, (void *)seq_0_7, 0);
          xtbool4 b0 = AE_LT16(seq0, AE_MOVDA16(rem_len));
          xtbool4 b1 = AE_LT16(seq1, AE_MOVDA16(rem_len));

          ADD_HX4T(sum_v, sum_v, exp_out0, b0);
          ADD_HX4T(sum_v, sum_v, exp_out1, b1);
        }
        /* -----------------------------------------------------------------------------  */ 
        AE_SA128POS_FP(va_out, (void *)p_out_fp16);
        xthalfx4 outsum;
        RADD_HX4(outsum, sum_v);

        xthalfx4 logsum, __logsum_dummy;
        COMPUTE_LOG_POSITIVE_HX4X2(logsum, __logsum_dummy, outsum, outsum);
        logsum = ADD_HX4(logsum, max_val_vec); // logsum = log(sum) + max_val

        p_vec_fp16 = (xthalfx8 *)p_vec;
        p_out_fp16 = (xthalfx8 *)p_out;
        va_in = AE_LA128_PP((void *)p_vec_fp16);

        for(i = 0; i < vec_length; i+=8)
        {
          int rem_len = vec_length - i;
          rem_len = (rem_len > 8) ? 8 : rem_len;

          xthalfx4 out0_f16, out1_f16;
          AE_LAVHX4X2_XP(out0_f16, out1_f16, va_in, p_vec_fp16, rem_len*sizeof(WORD16));
          SUB_HX4X2(out0_f16, out1_f16, out0_f16, out1_f16, logsum, logsum); //xi - (logsum + xmax) = (xi - xmax) - log(sum(exp(xj-xmax)))
          AE_SAVHX4X2_XP(out0_f16, out1_f16, va_out, p_out_fp16, rem_len*sizeof(WORD16));
        }
        AE_SA128POS_FP(va_out, (void *)p_out_fp16);

        return 0;
      }
#endif /* !HAVE_HP_VFPU */

#if !HAVE_HP_VFPU
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_log_f16_f16,(
      WORD16       * __restrict__ p_out,  
      const WORD16 * __restrict__ p_vec,         
      WORD32        vec_length)                  )
#else

/* COMPUTE_LOG_POSITIVE_FULLRANGE: LUT-based log for two xthalfx4 vectors.
 * Handles full range including negative -> NaN, zero -> -inf, inf/NaN passthrough.
 * Uses 8-entry LUT (3-bit mantissa index) + 3-term residual polynomial.
 * Eliminates all RECIP instructions.  Peak register pressure ~12/16.
 *
 * Algorithm:
 *   log(x) = N*ln2 + log(m_grid) + log(1 + u)
 * where:
 *   N        = exponent of x (possibly after subnormal normalization)
 *   m        = mantissa in [1, 2)
 *   idx      = top 3 mantissa fraction bits (0-7)
 *   m_grid   = 1 + idx*0.125 (grid point from LUT)
 *   residual = m - m_grid in [0, 0.125)
 *   u        = residual * inv_lut[idx]  (avoids RECIP)
 *   log(1+u) ~  u*(1 + u*(-1/2 + u/3))  (3-term Horner)
 *
 * For x in [0.75, 1.25]: near-1 polynomial (Phase 1) overrides to
 * avoid catastrophic cancellation in N*ln2 + log(m).
 */
  #define COMPUTE_LOG_POSITIVE_FULLRANGE(result0, result1, x0, x1)                            \
    {                                                                                           \
      xthalfx4 __hfl_oneF = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3C00)); /* 1.0 */        \
                                                                                                \
      /* --- Phase 1: Near-1 polynomial (computed first, low register pressure) ---         \
       * log(1+d) = d*(1 + d*(-1/2 + d*(1/3 + d*(-1/4)))) for |d| <= 0.25                 */  \
      xthalfx4 __hfl_d0, __hfl_d1;                                                            \
      SUB_HX4X2(__hfl_d0, __hfl_d1, x0, x1, __hfl_oneF, __hfl_oneF);                         \
      xthalfx4 __hfl_lc0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xB400)); /* -1/4 */     \
      xthalfx4 __hfl_lc0_1 = __hfl_lc0_0;                                                     \
      xthalfx4 __hfl_lc1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3555)); /* 1/3 */      \
      xthalfx4 __hfl_lc1_1 = __hfl_lc1_0;                                                     \
      xthalfx4 __hfl_lc2_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xB800)); /* -1/2 */     \
      xthalfx4 __hfl_lc2_1 = __hfl_lc2_0;                                                     \
      xthalfx4 __hfl_lc3_0 = __hfl_oneF;                                                      \
      xthalfx4 __hfl_lc3_1 = __hfl_oneF;                                                      \
      MADD_HX4X2(__hfl_lc1_0, __hfl_lc1_1, __hfl_lc0_0, __hfl_lc0_1, __hfl_d0, __hfl_d1);  \
      MADD_HX4X2(__hfl_lc2_0, __hfl_lc2_1, __hfl_lc1_0, __hfl_lc1_1, __hfl_d0, __hfl_d1);  \
      MADD_HX4X2(__hfl_lc3_0, __hfl_lc3_1, __hfl_lc2_0, __hfl_lc2_1, __hfl_d0, __hfl_d1);  \
      xthalfx4 __hfl_log1p_0, __hfl_log1p_1;                                                  \
      MUL_HX4X2(__hfl_log1p_0, __hfl_log1p_1, __hfl_lc3_0, __hfl_lc3_1, __hfl_d0, __hfl_d1);\
      /* --- End Phase 1: only log1p_0/1 survive --- */                                       \
                                                                                                \
      /* --- Phase 2: LUT-based main path --- */                                              \
      xthalfx4 __hfl_wx0 = (x0);                                                              \
      xthalfx4 __hfl_wx1 = (x1);                                                              \
      xthalfx4 __hfl_log2F = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x398C)); /* ln2 */       \
                                                                                                \
      /* Subnormal normalization: multiply by 2^14 if exponent field == 0 */                  \
      ae_int16x4 __hfl_xbits0 = AE_MOVINT16X4_FROMXTHALFX4(__hfl_wx0);                        \
      ae_int16x4 __hfl_xbits1 = AE_MOVINT16X4_FROMXTHALFX4(__hfl_wx1);                        \
      xtbool4 __hfl_b0_sub = AE_EQ16(AE_AND16(AE_SRLI16(__hfl_xbits0, 10),                   \
                                               AE_MOVDA16(0x001F)), AE_MOVDA16(0));            \
      xtbool4 __hfl_b1_sub = AE_EQ16(AE_AND16(AE_SRLI16(__hfl_xbits1, 10),                   \
                                               AE_MOVDA16(0x001F)), AE_MOVDA16(0));            \
      xthalfx4 __hfl_scale = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x7400)); /* 2^14 */      \
      xthalfx4 __hfl_sx0, __hfl_sx1;                                                          \
      MUL_HX4X2(__hfl_sx0, __hfl_sx1, __hfl_wx0, __hfl_wx1, __hfl_scale, __hfl_scale);       \
      MOVT_HX4(__hfl_wx0, __hfl_sx0, __hfl_b0_sub);                                           \
      MOVT_HX4(__hfl_wx1, __hfl_sx1, __hfl_b1_sub);                                           \
                                                                                                \
      /* Bit extraction: exponent N and mantissa m in [1,2) */                                \
      ae_int16x4 __hfl_hx0 = AE_MOVINT16X4_FROMXTHALFX4(__hfl_wx0);                           \
      ae_int16x4 __hfl_hx1 = AE_MOVINT16X4_FROMXTHALFX4(__hfl_wx1);                           \
      ae_int16x4 __hfl_exp0 = AE_AND16(AE_SRLI16(__hfl_hx0, 10), AE_MOVDA16(0x001F));        \
      ae_int16x4 __hfl_exp1 = AE_AND16(AE_SRLI16(__hfl_hx1, 10), AE_MOVDA16(0x001F));        \
      ae_int16x4 __hfl_N0 = AE_SUB16(__hfl_exp0, AE_MOVDA16(15));                             \
      ae_int16x4 __hfl_N1 = AE_SUB16(__hfl_exp1, AE_MOVDA16(15));                             \
      ae_int16x4 __hfl_mbits0 = AE_OR16(AE_AND16(__hfl_hx0, AE_MOVDA16(0x03FF)),             \
                                         AE_MOVDA16(0x3C00));                                  \
      ae_int16x4 __hfl_mbits1 = AE_OR16(AE_AND16(__hfl_hx1, AE_MOVDA16(0x03FF)),             \
                                         AE_MOVDA16(0x3C00));                                  \
                                                                                                \
      /* Grid point: zero bottom 7 mantissa bits */                                           \
      ae_int16x4 __hfl_mgrid0 = AE_AND16(__hfl_mbits0, AE_MOVDA16(0xFF80));                   \
      ae_int16x4 __hfl_mgrid1 = AE_AND16(__hfl_mbits1, AE_MOVDA16(0xFF80));                   \
                                                                                                \
      /* LUT index: top 3 mantissa fraction bits (bits [9:7] of mantissa field) */            \
      ae_int16x4 __hfl_idx0 = AE_AND16(AE_SRLI16(__hfl_hx0, 7), AE_MOVDA16(0x0007));        \
      ae_int16x4 __hfl_idx1 = AE_AND16(AE_SRLI16(__hfl_hx1, 7), AE_MOVDA16(0x0007));        \
      ae_int16x4 __hfl_sel0 = AE_XOR16(__hfl_idx0, AE_MOVDA16(0xFFFF));                       \
      ae_int16x4 __hfl_sel1 = AE_XOR16(__hfl_idx1, AE_MOVDA16(0xFFFF));                       \
                                                                                                \
      /* LUT lookups via AE_SEL16X4 */                                                        \
      ae_int16x4 __hfl_tbl0 = *(const ae_int16x4 *)log_lut;                                   \
      ae_int16x4 __hfl_tbl1 = *(const ae_int16x4 *)(log_lut + 4);                             \
      ae_int16x4 __hfl_logv0 = AE_SEL16X4(__hfl_tbl0, __hfl_tbl1, __hfl_sel0);               \
      ae_int16x4 __hfl_logv1 = AE_SEL16X4(__hfl_tbl0, __hfl_tbl1, __hfl_sel1);               \
      __hfl_tbl0 = *(const ae_int16x4 *)log_inv_lut;                                          \
      __hfl_tbl1 = *(const ae_int16x4 *)(log_inv_lut + 4);                                    \
      ae_int16x4 __hfl_invv0 = AE_SEL16X4(__hfl_tbl0, __hfl_tbl1, __hfl_sel0);               \
      ae_int16x4 __hfl_invv1 = AE_SEL16X4(__hfl_tbl0, __hfl_tbl1, __hfl_sel1);               \
                                                                                                \
      /* residual = m - m_grid (exact in fp16 since bottom 7 bits) */                         \
      xthalfx4 __hfl_m0 = AE_MOVXTHALFX4_FROMINT16X4(__hfl_mbits0);                           \
      xthalfx4 __hfl_m1 = AE_MOVXTHALFX4_FROMINT16X4(__hfl_mbits1);                           \
      xthalfx4 __hfl_mg0 = AE_MOVXTHALFX4_FROMINT16X4(__hfl_mgrid0);                          \
      xthalfx4 __hfl_mg1 = AE_MOVXTHALFX4_FROMINT16X4(__hfl_mgrid1);                          \
      xthalfx4 __hfl_r0, __hfl_r1;                                                            \
      SUB_HX4X2(__hfl_r0, __hfl_r1, __hfl_m0, __hfl_m1, __hfl_mg0, __hfl_mg1);              \
                                                                                                \
      /* u = residual * (1/m_grid) from LUT - replaces RECIP */                              \
      xthalfx4 __hfl_u0, __hfl_u1;                                                            \
      MUL_HX4X2(__hfl_u0, __hfl_u1, __hfl_r0, __hfl_r1,                                      \
                AE_MOVXTHALFX4_FROMINT16X4(__hfl_invv0),                                      \
                AE_MOVXTHALFX4_FROMINT16X4(__hfl_invv1));                                     \
                                                                                                \
      /* Residual polynomial: log(1+u) = u*(1 + u*(-1/2 + u/3)), 3-term Horner */           \
      xthalfx4 __hfl_pc0_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3555)); /* 1/3 */      \
      xthalfx4 __hfl_pc0_1 = __hfl_pc0_0;                                                     \
      xthalfx4 __hfl_pc1_0 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xB800)); /* -1/2 */     \
      xthalfx4 __hfl_pc1_1 = __hfl_pc1_0;                                                     \
      xthalfx4 __hfl_pc2_0 = __hfl_oneF; /* 1.0 */                                            \
      xthalfx4 __hfl_pc2_1 = __hfl_oneF;                                                      \
      MADD_HX4X2(__hfl_pc1_0, __hfl_pc1_1, __hfl_pc0_0, __hfl_pc0_1, __hfl_u0, __hfl_u1);  \
      MADD_HX4X2(__hfl_pc2_0, __hfl_pc2_1, __hfl_pc1_0, __hfl_pc1_1, __hfl_u0, __hfl_u1);  \
      xthalfx4 __hfl_corr0, __hfl_corr1;                                                      \
      MUL_HX4X2(__hfl_corr0, __hfl_corr1, __hfl_pc2_0, __hfl_pc2_1, __hfl_u0, __hfl_u1);   \
                                                                                                \
      /* result = N*ln2 + log_lut_val + correction */                                         \
      xthalfx4 __hfl_res0, __hfl_res1;                                                        \
      MUL_HX4X2(__hfl_res0, __hfl_res1,                                                       \
                FLOAT16_HX4(__hfl_N0, 0), FLOAT16_HX4(__hfl_N1, 0),                          \
                __hfl_log2F, __hfl_log2F);                                                    \
      ADD_HX4X2(__hfl_res0, __hfl_res1, __hfl_res0, __hfl_res1,                              \
                AE_MOVXTHALFX4_FROMINT16X4(__hfl_logv0),                                      \
                AE_MOVXTHALFX4_FROMINT16X4(__hfl_logv1));                                     \
      ADD_HX4X2(__hfl_res0, __hfl_res1, __hfl_res0, __hfl_res1, __hfl_corr0, __hfl_corr1);  \
                                                                                                \
      /* --- Phase 3: Selection and special cases --- */                                     \
      ae_int16x4 __hfl_xo0 = AE_MOVINT16X4_FROMXTHALFX4(x0);                                 \
      ae_int16x4 __hfl_xo1 = AE_MOVINT16X4_FROMXTHALFX4(x1);                                 \
      ae_int16x4 __hfl_oe0 = AE_AND16(AE_SRLI16(__hfl_xo0, 10), AE_MOVDA16(0x001F));        \
      ae_int16x4 __hfl_oe1 = AE_AND16(AE_SRLI16(__hfl_xo1, 10), AE_MOVDA16(0x001F));        \
                                                                                                \
      /* Subnormal compensation: subtract 14*ln2 (recompute condition from original exp) */   \
      xthalfx4 __hfl_14ln2 = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x48DA));                 \
      xtbool4 __hfl_b0_sub2 = AE_EQ16(__hfl_oe0, AE_MOVDA16(0));                              \
      xtbool4 __hfl_b1_sub2 = AE_EQ16(__hfl_oe1, AE_MOVDA16(0));                              \
      xthalfx4 __hfl_adj0, __hfl_adj1;                                                        \
      SUB_HX4X2(__hfl_adj0, __hfl_adj1, __hfl_res0, __hfl_res1, __hfl_14ln2, __hfl_14ln2);  \
      MOVT_HX4(__hfl_res0, __hfl_adj0, __hfl_b0_sub2);                                        \
      MOVT_HX4(__hfl_res1, __hfl_adj1, __hfl_b1_sub2);                                        \
                                                                                                \
      /* Near-1 override: select log1p for x in [0.75, 1.25] using two FP compares */        \
      xthalfx4 __hfl_near_lo = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3A00)); /* 0.75 */   \
      xthalfx4 __hfl_near_hi = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x3D00)); /* 1.25 */   \
      xthalfx4 __hfl_nr0 = __hfl_log1p_0;                                                    \
      xthalfx4 __hfl_nr1 = __hfl_log1p_1;                                                    \
      MOVT_HX4(__hfl_nr0, __hfl_res0, OLT_HX4((x0), __hfl_near_lo));  /* x<0.75: Phase2 */ \
      MOVT_HX4(__hfl_nr1, __hfl_res1, OLT_HX4((x1), __hfl_near_lo));                        \
      MOVT_HX4(__hfl_nr0, __hfl_res0, OLT_HX4(__hfl_near_hi, (x0))); /* x>1.25: Phase2 */  \
      MOVT_HX4(__hfl_nr1, __hfl_res1, OLT_HX4(__hfl_near_hi, (x1)));                        \
      __hfl_res0 = __hfl_nr0;                                                                 \
      __hfl_res1 = __hfl_nr1;                                                                 \
                                                                                                \
      /* log(0) = -inf */                                                                     \
      xthalfx4 __hfl_neginf = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0xFC00));                 \
      xtbool4 __hfl_b0_z = OEQ_HX4(x0, AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0)));         \
      xtbool4 __hfl_b1_z = OEQ_HX4(x1, AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0)));         \
      MOVT_HX4(__hfl_res0, __hfl_neginf, __hfl_b0_z);                                         \
      MOVT_HX4(__hfl_res1, __hfl_neginf, __hfl_b1_z);                                         \
                                                                                                \
      /* log(+inf/NaN) = x: propagate via exp==31 check */                                    \
      xtbool4 __hfl_b0_sp = AE_EQ16(__hfl_oe0, AE_MOVDA16(0x001F));                           \
      xtbool4 __hfl_b1_sp = AE_EQ16(__hfl_oe1, AE_MOVDA16(0x001F));                           \
      MOVT_HX4(__hfl_res0, x0, __hfl_b0_sp);                                                  \
      MOVT_HX4(__hfl_res1, x1, __hfl_b1_sp);                                                  \
                                                                                                \
      /* log(x < 0) = NaN; ordered compare: -0 is not < 0, so log(-0)=-inf is unaffected */   \
      xthalfx4 __hfl_qnan = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(0x7E00));                   \
      xtbool4 __hfl_b0_xNeg = OLT_HX4(x0, CONST_HX4(0));                                     \
      xtbool4 __hfl_b1_xNeg = OLT_HX4(x1, CONST_HX4(0));                                     \
      MOVT_HX4(__hfl_res0, __hfl_qnan, __hfl_b0_xNeg);                                        \
      MOVT_HX4(__hfl_res1, __hfl_qnan, __hfl_b1_xNeg);                                        \
                                                                                                \
      (result0) = __hfl_res0; (result1) = __hfl_res1;                                          \
    }

WORD32 xa_nn_vec_log_f16_f16(
    WORD16       * __restrict__ p_out,  
    const WORD16 * __restrict__ p_vec,         
    WORD32        vec_length)
{
        /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

    const xthalfx8 * __restrict__ pX; xthalfx8 * __restrict__ pY; 
    pX = (const xthalfx8 *)p_vec; pY = (xthalfx8 *)p_out;

    ae_valignx2 X_va, Y_va;
    X_va = AE_LA128_PP(pX);
    Y_va = AE_ZALIGN128();

    xthalfx4 x0, x1, result0, result1;

    for (int n = 0; n<(vec_length>>3); n++)
    {
        AE_LAHX4X2_IP(x0, x1, X_va, pX);
        COMPUTE_LOG_POSITIVE_FULLRANGE(result0, result1, x0, x1);
        AE_SAHX4X2_IP(result0, result1, Y_va, pY);
    }

    int rem_val = vec_length & 7;
    if(rem_val){

        AE_LAVHX4X2_XP (x0, x1, X_va, pX,rem_val*sizeof(WORD16));
        COMPUTE_LOG_POSITIVE_FULLRANGE(result0, result1, x0, x1);
        AE_SAVHX4X2_XP(result0, result1, Y_va, pY, rem_val*sizeof(WORD16));
    }

    AE_SA128POS_FP(Y_va, pY);

    return 0;
}
#endif /* !HAVE_HP_VFPU */
