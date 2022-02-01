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
#ifndef __XA_NNLIB_COMMON_MACROS_H__
#define __XA_NNLIB_COMMON_MACROS_H__

#ifndef NULL
#define NULL (void *)0
#endif /* NULL */

/* Macros for memcpy */
#define MEMCPY_8b(out, inp, N) \
{ \
  int itr; \
  ae_int8x8 di0, di1; \
  ae_int8x16 *pae_i, *pae_o; \
  ae_valignx2 i_a, o_a; \
  pae_i = (ae_int8x16 *)(inp); \
  pae_o = (ae_int8x16 *)(out); \
  i_a = AE_LA128_PP(pae_i); \
  o_a = AE_ZALIGN128(); \
  for(itr = 0; itr < ((N)>>4); itr++) \
  { \
    AE_LA8X8X2_IP(di0, di1, i_a, pae_i); \
    AE_SA8X8X2_IP(di0, di1, o_a, pae_o); \
  } \
  AE_LAV8X8X2_XP(di0, di1, i_a, pae_i, ((N)&15)); \
  AE_SAV8X8X2_XP(di0, di1, o_a, pae_o, ((N)&15)); \
  AE_SA128POS_FP(o_a, pae_o); \
}

#define DUAL_MEMCPY_2D_8b_CONT_OUT(out0, out1, inp0, inp1, rows, cols, inp_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int8x8 di0_0, di0_1; \
  ae_int8x8 di1_0, di1_1; \
  ae_int8x16 *pae_in0, *pae_out0; \
  ae_valignx2 in0_a, out0_a; \
  ae_int8x16 *pae_in1, *pae_out1; \
  ae_valignx2 in1_a, out1_a; \
  pae_out0 = (ae_int8x16 *)(out0); \
  out0_a = AE_ZALIGN128(); \
  pae_out1 = (ae_int8x16 *)(out1); \
  out1_a = AE_ZALIGN128(); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_in0 = (ae_int8x16 *)(&(inp0)[itr_r * inp_row_offset]); \
    in0_a = AE_LA128_PP(pae_in0); \
    pae_in1 = (ae_int8x16 *)(&(inp1)[itr_r * inp_row_offset]); \
    in1_a = AE_LA128_PP(pae_in1); \
__Pragma("no_unroll") \
    for(itr_c = 0; itr_c < ((cols)>>4); itr_c++) \
    { \
      AE_LA8X8X2_IP(di0_0, di0_1, in0_a, pae_in0); \
      AE_SA8X8X2_IP(di0_0, di0_1, out0_a, pae_out0); \
      AE_LA8X8X2_IP(di1_0, di1_1, in1_a, pae_in1); \
      AE_SA8X8X2_IP(di1_0, di1_1, out1_a, pae_out1); \
    } \
    AE_LAV8X8X2_XP(di0_0, di0_1, in0_a, pae_in0, ((cols)&15)); \
    AE_SAV8X8X2_XP(di0_0, di0_1, out0_a, pae_out0, ((cols)&15)); \
    AE_LAV8X8X2_XP(di1_0, di1_1, in1_a, pae_in1, ((cols)&15)); \
    AE_SAV8X8X2_XP(di1_0, di1_1, out1_a, pae_out1, ((cols)&15)); \
  } \
  AE_SA128POS_FP(out0_a, pae_out0); \
  AE_SA128POS_FP(out1_a, pae_out1); \
}

#define MEMCPY_2D_8b_CONT_OUT(out0, inp0, rows, cols, inp_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int8x8 di0_0, di0_1; \
  ae_int8x16 *pae_in0, *pae_out0; \
  ae_valignx2 in0_a, out0_a; \
  pae_out0 = (ae_int8x16 *)(out0); \
  out0_a = AE_ZALIGN128(); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_in0 = (ae_int8x16 *)(&(inp0)[itr_r * inp_row_offset]); \
    in0_a = AE_LA128_PP(pae_in0); \
    for(itr_c = 0; itr_c < ((cols)>>4); itr_c++) \
    { \
      AE_LA8X8X2_IP(di0_0, di0_1, in0_a, pae_in0); \
      AE_SA8X8X2_IP(di0_0, di0_1, out0_a, pae_out0); \
    } \
    AE_LAV8X8X2_XP(di0_0, di0_1, in0_a, pae_in0, ((cols)&15)); \
    AE_SAV8X8X2_XP(di0_0, di0_1, out0_a, pae_out0, ((cols)&15)); \
  } \
  AE_SA128POS_FP(out0_a, pae_out0); \
}

#define DUAL_MEMCPY_2D_8b_CONT_INP(out0, out1, inp0, inp1, rows, cols, out_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int8x8 di0_0, di0_1; \
  ae_int8x8 di1_0, di1_1; \
  ae_int8x16 *pae_in0, *pae_out0; \
  ae_valignx2 in0_a, out0_a; \
  ae_int8x16 *pae_in1, *pae_out1; \
  ae_valignx2 in1_a, out1_a; \
  pae_in0 = (ae_int8x16 *)(inp0); \
  in0_a = AE_LA128_PP(pae_in0); \
  pae_in1 = (ae_int8x16 *)(inp1); \
  in1_a = AE_LA128_PP(pae_in1); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_out0 = (ae_int8x16 *)(&(out0)[itr_r * out_row_offset]); \
    out0_a = AE_ZALIGN128(); \
    pae_out1 = (ae_int8x16 *)(&(out1)[itr_r * out_row_offset]); \
    out1_a = AE_ZALIGN128(); \
__Pragma("no_unroll") \
    for(itr_c = 0; itr_c < ((cols)>>4); itr_c++) \
    { \
      AE_LA8X8X2_IP(di0_0, di0_1, in0_a, pae_in0); \
      AE_SA8X8X2_IP(di0_0, di0_1, out0_a, pae_out0); \
      AE_LA8X8X2_IP(di1_0, di1_1, in1_a, pae_in1); \
      AE_SA8X8X2_IP(di1_0, di1_1, out1_a, pae_out1); \
    } \
    AE_LAV8X8X2_XP(di0_0, di0_1, in0_a, pae_in0, ((cols)&15)); \
    AE_SAV8X8X2_XP(di0_0, di0_1, out0_a, pae_out0, ((cols)&15)); \
    AE_SA128POS_FP(out0_a, pae_out0); \
    AE_LAV8X8X2_XP(di1_0, di1_1, in1_a, pae_in1, ((cols)&15)); \
    AE_SAV8X8X2_XP(di1_0, di1_1, out1_a, pae_out1, ((cols)&15)); \
    AE_SA128POS_FP(out1_a, pae_out1); \
  } \
}

#define MEMCPY_2D_8b_CONT_INP(out0, inp0, rows, cols, out_row_offset) \
{ \
  int itr_r, itr_c; \
  ae_int8x8 di0_0, di0_1; \
  ae_int8x16 *pae_in0, *pae_out0; \
  ae_valignx2 in0_a, out0_a; \
  pae_in0 = (ae_int8x16 *)(inp0); \
  in0_a = AE_LA128_PP(pae_in0); \
  for(itr_r = 0; itr_r < rows; itr_r++) \
  { \
    pae_out0 = (ae_int8x16 *)(&(out0)[itr_r * out_row_offset]); \
    out0_a = AE_ZALIGN128(); \
    for(itr_c = 0; itr_c < ((cols)>>4); itr_c++) \
    { \
      AE_LA8X8X2_IP(di0_0, di0_1, in0_a, pae_in0); \
      AE_SA8X8X2_IP(di0_0, di0_1, out0_a, pae_out0); \
    } \
    AE_LAV8X8X2_XP(di0_0, di0_1, in0_a, pae_in0, ((cols)&15)); \
    AE_SAV8X8X2_XP(di0_0, di0_1, out0_a, pae_out0, ((cols)&15)); \
    AE_SA128POS_FP(out0_a, pae_out0); \
  } \
}

/* Macro for zero value */
#define ZERO64   AE_ZERO64()
#define ZERO16X4 AE_MOVDA16(0)
#define ZERO16   (0)
#define ZERO32   (0)

/* Macro for 1 */
#define ONE16X4 AE_MOVDA16(1)

/* Value of ROW_UNROLL currently supported are 1,2,4,8 only */
#ifndef ROW_UNROLL
#define ROW_UNROLL 8
#endif
#define VEC_UNROLL 2

#define ACC_LSH_AFTER_FIRST_MATXVEC 0

/* Increment in bytes required for particular load
 * instructions. */
#define INCREMENT_IN_BYTES_FOR_WORD8     1
#define INCREMENT_IN_BYTES_FOR_INT16     2
#define INCREMENT_IN_BYTES_FOR_INT32     (INCREMENT_IN_BYTES_FOR_INT16   * 2)
#define INCREMENT_IN_BYTES_FOR_WORD8X4   (INCREMENT_IN_BYTES_FOR_WORD8   * 4)
#define INCREMENT_IN_BYTES_FOR_WORD8X8   (INCREMENT_IN_BYTES_FOR_WORD8   * 8)
#define INCREMENT_IN_BYTES_FOR_INT16X4   (INCREMENT_IN_BYTES_FOR_INT16   * 4)
#define INCREMENT_IN_BYTES_FOR_INT64     INCREMENT_IN_BYTES_FOR_INT16X4
#define INCREMENT_IN_BYTES_FOR_FLOAT32   4
#define INCREMENT_IN_BYTES_FOR_FLOAT32x2 (INCREMENT_IN_BYTES_FOR_FLOAT32 * 2)

/* Limit effective bias_shift and acc_shift to [-63 ... 63] */
#define LIMIT_VARIABLE(_var, _left_limit, _right_limit) \
  _var = _var > _right_limit ? _right_limit : _var < _left_limit ? _left_limit : _var;

#define LIMIT_ACC_LSH \
  LIMIT_VARIABLE(acc_shift, -63, 63); \

#define LIMIT_BIAS_LSH \
  LIMIT_VARIABLE(bias_shift, -63, 63); \

#define BW(_datatype) sizeof(_datatype)

#define ADJUST_VAR_AxB(A, B) \
  (((8 * (4 - (BW(A) + BW(B))))))

#define ADJUST_VAR_C(C) \
  (((64 - (8 * BW(C)))))

#define ADJUST_ACC_LSH_AxB_C(A, B, C) \
  acc_shift = acc_shift - ADJUST_VAR_AxB(A, B) + ADJUST_VAR_C(C); \
  LIMIT_ACC_LSH;

#define ADJUST_BIAS_LSH_AxB(A, B) \
  bias_shift = bias_shift + ADJUST_VAR_AxB(A, B); \
  LIMIT_BIAS_LSH;

#define ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(A, B, C) \
  ADJUST_ACC_LSH_AxB_C(A, B, C); \
  ADJUST_BIAS_LSH_AxB(A, B); \

/* ==================================================================================================== */
#define SETUP_BIAS_f32 \
  xtfloat _xtfloat_bias = (xtfloat)0.0f; \
  xtfloat *_xtfloat_p_bias = (xtfloat *) p_bias; \

#define SETUP_BIAS_ASYM8b \
  WORD32 _WORD32_bias; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD32 *_WORD32_p_bias = (WORD32 *) p_bias; \

#define SETUP_BIAS_8b_UNALIGNED_SUPPORT \
  ae_int64 sat_1;\
  ae_int8x8 _ae_int8_bias; \
  ae_int8 *_ae_int8_p_bias = (ae_int8 *) p_bias; \

#define SETUP_BIAS_8b \
  ae_int64 sat_1,sat_2;\
  ae_int8x8 _ae_int8_bias; \
  ae_int8x8 _ae_int8_bias_1; \
  ae_int8 *_ae_int8_p_bias = (ae_int8 *) p_bias; \

#define SETUP_BIAS_32b \
  ae_int64 sat_1;\
  ae_int64 sat_2;\
  ae_int32x2 _ae_int32x2_bias = ZERO32; \
  ae_int32x2 *_ae_int32x2_p_bias = (ae_int32x2 *) p_bias; \

#define SETUP_BIAS_32b_SINGLE \
  ae_int64 sat_1;\
  ae_int32x2 _ae_int32_bias = ZERO32; \
  ae_int32 *_ae_int32_p_bias = (ae_int32 *) p_bias; \

#define SETUP_BIAS_16b \
ae_int16 _ae_int16_bias = ZERO16; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int16 *_ae_int16_p_bias = (ae_int16 *) p_bias; \

#define SETUP_BIAS_64b \
  ae_int64 _ae_int64_bias = ZERO64; \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int64 *_ae_int64_p_bias = (ae_int64 *) p_bias; \

//#define SETUP_ACC_FOR_8bx8b(idx)   SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_8bx8b(idx)   SETUP_ACC_32x32b(idx)
#define SETUP_ACC_FOR_8bx16b(idx)  SETUP_ACC_64b_8x16(idx)
#define SETUP_ACC_FOR_16bx8b(idx)  SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_16bx16b(idx) SETUP_ACC_64b(idx)

#define SETUP_ACC_FOR_ASYM8bxASYM8b(idx) \
  ae_int64 _ae_int64_acc_ ## idx = ZERO64; \

/*------------------ time batching macros ----------------- */

#define SETUP_ACC_BATCH_ROW_FOR_16bx8b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_8bx16b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_8bx8b  SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_ROW_FOR_16bx16b

#define SETUP_ACC_BATCH_FOR_16bx8b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_8bx16b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_8bx8b  SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_FOR_16bx16b

#define SETUP_ACC_BATCH_ROW_FOR_16bx16b(idx_row)\
  SETUP_ACC_BATCH_VEC_UNROLL(idx_row);\

#define SETUP_ACC_BATCH_FOR_16bx16b(idx_row,idx_vec) \
  ae_int64 _ae_int64_acc_ ##idx_row ##_ ##idx_vec = ZERO64; \
  ae_int64 _ae_int64_acc_1_ ##idx_row ##_ ##idx_vec = ZERO64; \

#define SETUP_ACC_BATCH_ROW_FOR_f32(idx_row)\
  SETUP_ACC_BATCH_VEC_UNROLL(idx_row);\

#define SETUP_ACC_BATCH_FOR_f32(idx_row,idx_vec) \
  xtfloatx2 _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
  xtfloatx2 _xtfloatx2_acc_1_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
  xtfloat _xtfloat_acc_ ##idx_row ##_ ##idx_vec = (xtfloat) 0.0f;\
/*---------------------------------------------------------*/

#define SETUP_ACC_32x32b(idx) \
  ae_int32x2 _ae_int32x2_acc_ ## idx = ZERO32; \

#define SETUP_ACC_64b_8x16(idx) \
  ae_int64 _ae_int64_acc_ ## idx = ZERO64; \

#define SETUP_ACC_64b(idx) \
  ae_int64 _ae_int64_acc_ ## idx = ZERO64; \
  ae_int64 _ae_int64_acc_1_ ## idx = ZERO64; \

#define SETUP_VEC1_8X16b_UNALIGNED \
  ae_int16x4 _ae_int16x4_vec1; \
  ae_int16x4 _ae_int16x4_vec1_1; \
  ae_int16x4 * _ae_int16x4_p_vec1 = (ae_int16x4 *) p_vec1; \
  ae_valign _align_ae_int16x4_p_vec1 = AE_LA64_PP(_ae_int16x4_p_vec1);

#define SETUP_VEC2_8X16b_UNALIGNED \
  ae_int16x4 _ae_int16x4_vec2; \
  ae_int16x4 _ae_int16x4_vec2_1; \
  ae_int16x4 * _ae_int16x4_p_vec2 = (ae_int16x4 *) p_vec2; \
  ae_valign _align_ae_int16x4_p_vec2 = AE_LA64_PP(_ae_int16x4_p_vec2);

#define SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(vec_idx) \
  ae_int16x4 _ae_int16x4_vec1_ ## vec_idx; \
  ae_int16x4 _ae_int16x4_vec1_1_ ## vec_idx; \
  ae_int16x8 * _ae_int16x8_p_vec1_ ## vec_idx = (ae_int16x8 *) (p_vec1[vec_itr + vec_idx]); \
  ae_valignx2 _align_ae_int16x8_p_vec1_ ## vec_idx = AE_LA128_PP(_ae_int16x8_p_vec1_ ## vec_idx);

#define SETUP_VEC1_16b_UNALIGNED_BATCH(vec_idx) \
  ae_int16x4 _ae_int16x4_vec1_ ## vec_idx; \
  ae_int16x4 * _ae_int16x4_p_vec1_ ## vec_idx = (ae_int16x4 *) (p_vec1[vec_itr + vec_idx]); \
  ae_valign _align_ae_int16x4_p_vec1_ ## vec_idx = AE_LA64_PP(_ae_int16x4_p_vec1_ ## vec_idx);

#define SETUP_VEC1_16b_UNALIGNED_ONE_VECTOR \
  ae_int16x4 _ae_int16x4_vec1;; \
  ae_int16x4 * _ae_int16x4_p_vec1 = (ae_int16x4 *) (p_vec1[vec_itr]); \
  ae_valign _align_ae_int16x4_p_vec1 = AE_LA64_PP(_ae_int16x4_p_vec1);

#define SETUP_VEC1_16b_UNALIGNED \
  ae_int16x4 _ae_int16x4_vec1;; \
  ae_int16x4 * _ae_int16x4_p_vec1 = (ae_int16x4 *) p_vec1; \
  ae_valign _align_ae_int16x4_p_vec1 = AE_LA64_PP(_ae_int16x4_p_vec1);

#define SETUP_VEC2_16b_UNALIGNED \
  ae_int16x4 _ae_int16x4_vec2;; \
  ae_int16x4 * _ae_int16x4_p_vec2 = (ae_int16x4 *) p_vec2; \
  ae_valign _align_ae_int16x4_p_vec2 = AE_LA64_PP(_ae_int16x4_p_vec2);

#define SETUP_VEC1_8b_SINGLE \
  ae_int8x8 _ae_int8x8_vec1;; \
  ae_int8x8 * _ae_int8x8_p_vec1 = (ae_int8x8 *) p_vec1; \

#define SETUP_VEC2_8b_SINGLE \
  ae_int8x8 _ae_int8x8_vec2;; \
  ae_int8x8 * _ae_int8x8_p_vec2 = (ae_int8x8 *) p_vec2; \

#define SETUP_VEC1_8b_UNALIGNED_BATCH(idx) \
  ae_int8x8 _ae_int8x8_vec1_ ## idx; \
  ae_int8x8 * _ae_int8x8_p_vec1_ ## idx = (ae_int8x8 *) (p_vec1[vec_itr + idx]); \
  ae_valign _align_ae_int8x8_p_vec1_ ## idx = AE_LA64_PP(_ae_int8x8_p_vec1_ ## idx);

#define SETUP_VEC1_8b_UNALIGNED \
  ae_int8x8 _ae_int8x8_vec1;; \
  ae_int8x8 * _ae_int8x8_p_vec1 = (ae_int8x8 *) p_vec1; \
  ae_valign _align_ae_int8x8_p_vec1 = AE_LA64_PP(_ae_int8x8_p_vec1);

#define SETUP_VEC2_8b_UNALIGNED \
  ae_int8x8 _ae_int8x8_vec2;; \
  ae_int8x8 * _ae_int8x8_p_vec2 = (ae_int8x8 *) p_vec2; \
  ae_valign _align_ae_int8x8_p_vec2 = AE_LA64_PP(_ae_int8x8_p_vec2);

#define SETUP_VEC1_8b \
  ae_int8x8 _ae_int8x8_vec1;; \
  ae_int8x8 _ae_int8x8_vec1_1; \
  ae_int8x16 * _ae_int8x16_p_vec1 = (ae_int8x16 *) p_vec1; \

#define SETUP_VEC1_8b_16 \
  ae_int8x8 _ae_int8x8_vec1;; \
  ae_int8x8 _ae_int8x8_vec1_1; \
  ae_int8x8 _ae_int8x8_vec1_2;; \
  ae_int8x8 _ae_int8x8_vec1_3; \
  ae_int8x16 * _ae_int8x16_p_vec1 = (ae_int8x16 *) p_vec1; \

#define SETUP_VEC2_8b_16 \
  ae_int8x8 _ae_int8x8_vec2 ; \
  ae_int8x8 _ae_int8x8_vec2_1; \
  ae_int8x8 _ae_int8x8_vec2_2; \
  ae_int8x8 _ae_int8x8_vec2_3; \
  ae_int8x16 * _ae_int8x16_p_vec2 = (ae_int8x16 *) p_vec2; \

#define SETUP_VEC2_8b \
  ae_int8x8 _ae_int8x8_vec2 ; \
  ae_int8x8 _ae_int8x8_vec2_1; \
  ae_int8x16 *_ae_int8x16_p_vec2 = (ae_int8x16 *) p_vec2; \

#define SETUP_VEC1_16b \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec1_1 = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_vec1 = (ae_int16x8 *) p_vec1; \


#define SETUP_VEC2_16b \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec2_1 = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_vec2 = (ae_int16x8 *) p_vec2; \

#define SETUP_VEC1_16b_16 \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec1_1 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec1_2 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec1_3 = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_vec1 = (ae_int16x8 *) p_vec1; \


#define SETUP_VEC2_16b_16 \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec2_1 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec2_2 = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec2_3 = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_vec2 = (ae_int16x8 *) p_vec2; \

#define SETUP_VEC1_ASYM8b \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  WORD8 *_WORD8_p_vec1 = (WORD8 *) p_vec1; \

#define SETUP_VEC2_ASYM8b \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  WORD8 *_WORD8_p_vec2 = (WORD8 *) p_vec2; \

/*------------------ time batching macros ----------------- */

#define SETUP_VEC_BATCH_8b(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  WORD8 *_WORD8_p_vec_batch_ ##idx_vec  = (WORD8 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_BATCH_16b(idx_vec)\
  ae_int16x4 _ae_int16x4_vec_batch_ ##idx_vec  = ZERO16X4; \
  ae_int16x4 _ae_int16x4_vec_batch_1_ ##idx_vec  = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_vec_batch_ ##idx_vec  = (ae_int16x8 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_BATCH_f32(idx_vec)\
  xtfloatx2 _xtfloatx2_vec_batch_ ##idx_vec  = (xtfloatx2)0.0f ; \
  xtfloatx2 _xtfloatx2_vec_batch_1_ ##idx_vec  = (xtfloatx2)0.0f ; \
  xtfloatx4 *_xtfloatx4_p_vec_batch_ ##idx_vec  = (xtfloatx4 *)(p_vec1[vec_itr + idx_vec]); \

#define SETUP_VEC_BATCH_ASYM8b SETUP_VEC_BATCH_8b
/*---------------------------------------------------------*/

#define SETUP_MAT1_8b_UNALIGNED(idx) \
  ae_int8x8 _ae_int8x8_mat1_ ## idx ; \
  ae_int8x8 * _ae_int8x8_p_mat1_ ## idx = (ae_int8x8 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ae_valign _align_ae_int8x8_p_mat1_ ## idx = AE_LA64_PP(_ae_int8x8_p_mat1_ ##idx);

#define SETUP_MAT2_8b_UNALIGNED(idx) \
  ae_int8x8 _ae_int8x8_mat2_ ## idx ; \
  ae_int8x8 * _ae_int8x8_p_mat2_ ## idx = (ae_int8x8 *) &p_mat2[(m_itr+idx)*row_stride2]; \
  ae_valign _align_ae_int8x8_p_mat2_ ## idx = AE_LA64_PP(_ae_int8x8_p_mat2_ ##idx);

#define SETUP_MAT1_16b_UNALIGNED_16x8(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx ; \
  ae_int16x4 _ae_int16x4_mat1_1_ ## idx ; \
  ae_int16x8 * _ae_int16x8_p_mat1_ ## idx = (ae_int16x8 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ae_valignx2 _align_ae_int16x8_p_mat1_ ## idx = AE_LA128_PP(_ae_int16x8_p_mat1_ ##idx);

#define SETUP_MAT2_16b_UNALIGNED_16x8(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx ; \
  ae_int16x4 _ae_int16x4_mat2_1_ ## idx ; \
  ae_int16x8 * _ae_int16x8_p_mat2_ ## idx = (ae_int16x8 *) &p_mat2[(m_itr+idx)*row_stride2]; \
  ae_valignx2 _align_ae_int16x8_p_mat2_ ## idx = AE_LA128_PP(_ae_int16x8_p_mat2_ ##idx);


#define SETUP_MAT1_16b_UNALIGNED(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx ; \
  ae_int16x4 * _ae_int16x4_p_mat1_ ## idx = (ae_int16x4 *) &p_mat1[(m_itr+idx)*row_stride1]; \
  ae_valign _align_ae_int16x4_p_mat1_ ## idx = AE_LA64_PP(_ae_int16x4_p_mat1_ ##idx);

#define SETUP_MAT2_16b_UNALIGNED(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx ; \
  ae_int16x4 * _ae_int16x4_p_mat2_ ## idx = (ae_int16x4 *) &p_mat2[(m_itr+idx)*row_stride2]; \
  ae_valign _align_ae_int16x4_p_mat2_ ## idx = AE_LA64_PP(_ae_int16x4_p_mat2_ ##idx);

#define SETUP_VEC_BATCH_ASYM8b SETUP_VEC_BATCH_8b
#define SETUP_VEC_OFFSET_BATCH_ASYM8b SETUP_VEC_OFFSET_BATCH_8b
#define SETUP_VEC_OFFSET_BATCH_ASYM8b_UNALIGNED SETUP_VEC_OFFSET_BATCH_8b_UNALIGNED
/*---------------------------------------------------------*/

#define SETUP_MAT1_8b(idx) \
  ae_int8x8 _ae_int8x8_mat1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat1_1_ ## idx; \
  ae_int8x16 * _ae_int8x16_p_mat1_ ## idx = (ae_int8x16 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT1_8b_16_BATCH(idx) \
  ae_int8x8 _ae_int8x8_mat1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat1_1_ ## idx; \


#define SETUP_MAT1_8b_8x16(idx) \
  ae_int8x8 _ae_int8x8_mat1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat1_1_ ## idx; \


#define SETUP_MAT2_8b_8x16(idx) \
  ae_int8x8 _ae_int8x8_mat2_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_1_ ## idx; \


#define SETUP_MAT2_8b(idx) \
  ae_int8x8 _ae_int8x8_mat2_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_1_ ## idx ; \
  ae_int8x16 * _ae_int8x16_p_mat2_ ## idx = (ae_int8x16 *) &p_mat2[(m_itr+idx)*row_stride2]; \

#define SETUP_MAT1_16b(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  ae_int16x4 _ae_int16x4_mat1_1_ ## idx = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_mat1_ ## idx = (ae_int16x8 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT2_16b(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx = ZERO16X4; \
  ae_int16x4 _ae_int16x4_mat2_1_ ## idx = ZERO16X4; \
  ae_int16x8 *_ae_int16x8_p_mat2_ ## idx = (ae_int16x8 *) &p_mat2[(m_itr+idx)*row_stride2]; \

#define SETUP_MAT1_f32(idx) \
  xtfloatx2 _xtfloatx2_mat1_ ## idx = (xtfloatx2)0.0f; \
  xtfloatx2 _xtfloatx2_mat1_1_ ## idx = (xtfloatx2)0.0f; \
  xtfloatx4 *_xtfloatx4_p_mat1_ ## idx = (xtfloatx4 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT1_ASYM8b(idx) \
  ae_int16x4 _ae_int16x4_mat1_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat1_ ## idx = (WORD8 *) &p_mat1[(m_itr+idx)*row_stride1]; \

#define SETUP_MAT2_ASYM8b(idx) \
  ae_int16x4 _ae_int16x4_mat2_ ## idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat2_ ## idx = (WORD8 *) &p_mat2[(m_itr+idx)*row_stride2]; \

/* ====================================================================== */

#define LOAD_VEC1_8X16b_UNALIGNED \
        AE_LA16X4_IP(_ae_int16x4_vec1,_align_ae_int16x4_p_vec1, _ae_int16x4_p_vec1);\
        AE_LA16X4_IP(_ae_int16x4_vec1_1,_align_ae_int16x4_p_vec1, _ae_int16x4_p_vec1);

#define LOAD_VEC2_8X16b_UNALIGNED \
        AE_LA16X4_IP(_ae_int16x4_vec2,_align_ae_int16x4_p_vec2, _ae_int16x4_p_vec2);\
        AE_LA16X4_IP(_ae_int16x4_vec2_1,_align_ae_int16x4_p_vec2, _ae_int16x4_p_vec2);


#define LOAD_VEC2_16b_UNALIGNED \
    AE_LA16X4_IP(_ae_int16x4_vec2,_align_ae_int16x4_p_vec2, _ae_int16x4_p_vec2);

#define LOAD_VEC2_16b_UNALIGNED_SINGLE_ELEMENT \
    AE_L16_IP(_ae_int16x4_vec2,(ae_int16* ) _ae_int16x4_p_vec2,2);

#define LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(idx) \
    AE_LA16X4X2_IP(_ae_int16x4_vec1_ ## idx,_ae_int16x4_vec1_1_ ## idx,_align_ae_int16x8_p_vec1_ ## idx, _ae_int16x8_p_vec1_ ## idx);

#define LOAD_VEC1_16b_UNALIGNED_BATCH(idx) \
    AE_LA16X4_IP(_ae_int16x4_vec1_ ## idx,_align_ae_int16x4_p_vec1_ ## idx, _ae_int16x4_p_vec1_ ## idx);

#define LOAD_VEC1_16b_UNALIGNED \
    AE_LA16X4_IP(_ae_int16x4_vec1,_align_ae_int16x4_p_vec1, _ae_int16x4_p_vec1);

#define LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(idx) \
    AE_L16_IP(_ae_int16x4_vec1_ ## idx,(ae_int16* ) _ae_int16x8_p_vec1_ ## idx,2);

#define LOAD_VEC1_16b_UNALIGNED_SINGLE_ELEMENT_BATCH(idx) \
    AE_L16_IP(_ae_int16x4_vec1_ ## idx,(ae_int16* ) _ae_int16x4_p_vec1_ ## idx,2);

#define LOAD_VEC1_16b_UNALIGNED_SINGLE_ELEMENT \
    AE_L16_IP(_ae_int16x4_vec1,(ae_int16* ) _ae_int16x4_p_vec1,2);

#define LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(idx) \
    AE_L8_IP(_ae_int8x8_vec1_ ## idx,(ae_int8 *)_ae_int8x8_p_vec1_ ##idx,1);

#define LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT \
    AE_L8_IP(_ae_int8x8_vec1,(ae_int8 *)_ae_int8x8_p_vec1,1);

#define LOAD_VEC2_8b_UNALIGNED_SINGLE_ELEMENT \
    AE_L8_IP(_ae_int8x8_vec2,(ae_int8 *)_ae_int8x8_p_vec2,1);

#define LOAD_VEC1_8b_UNALIGNED_BATCH(idx) \
  AE_LA8X8_IP(_ae_int8x8_vec1_ ## idx, _align_ae_int8x8_p_vec1_ ## idx ,_ae_int8x8_p_vec1_ ## idx); \

#define LOAD_VEC1_8b_UNALIGNED \
  AE_LA8X8_IP(_ae_int8x8_vec1, _align_ae_int8x8_p_vec1 ,_ae_int8x8_p_vec1); \

#define LOAD_VEC2_8b_UNALIGNED \
  AE_LA8X8_IP(_ae_int8x8_vec2, _align_ae_int8x8_p_vec2 ,_ae_int8x8_p_vec2); \

#define LOAD_VEC1_8b_SINGLE \
  AE_L8X8_IP(_ae_int8x8_vec1,_ae_int8x8_p_vec1, 8); \

#define LOAD_VEC2_8b_SINGLE \
  AE_L8X8_IP(_ae_int8x8_vec2,_ae_int8x8_p_vec2, 8); \

#define LOAD_VEC1_8b \
  AE_L8X8X2_IP(_ae_int8x8_vec1,_ae_int8x8_vec1_1, _ae_int8x16_p_vec1, 16); \

#define LOAD_VEC1_8b_16 \
    AE_L8X8X2_IP(_ae_int8x8_vec1,_ae_int8x8_vec1_1, _ae_int8x16_p_vec1, 16); \

#define LOAD_VEC1_16b_16 \
  AE_L16X4X2_IP(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 4*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC1_16b_16_1 \
  AE_L16X4X2_I (_ae_int16x4_vec1_2,_ae_int16x4_vec1_3, _ae_int16x8_p_vec1, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC2_16b_16 \
  AE_L16X4X2_IP(_ae_int16x4_vec2,_ae_int16x4_vec2_1, _ae_int16x8_p_vec2, 4*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC2_16b_16_1 \
  AE_L16X4X2_I (_ae_int16x4_vec2_2,_ae_int16x4_vec2_3, _ae_int16x8_p_vec2, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC1_8b_16_1 \
  AE_L8X8X2_IP(_ae_int8x8_vec1_2,_ae_int8x8_vec1_3, _ae_int8x16_p_vec1,16 ); \

#define LOAD_VEC2_8b_16 \
  AE_L8X8X2_IP(_ae_int8x8_vec2,_ae_int8x8_vec2_1, _ae_int8x16_p_vec2, 16); \

#define LOAD_VEC2_8b_16_1 \
  AE_L8X8X2_IP(_ae_int8x8_vec2_2,_ae_int8x8_vec2_3, _ae_int8x16_p_vec2, 16); \

#define LOAD_VEC2_8b \
  AE_L8X8X2_IP(_ae_int8x8_vec2,_ae_int8x8_vec2_1, _ae_int8x16_p_vec2, 16); \

#define LOAD_VEC1_16b \
  AE_L16X4X2_IP(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC2_16b \
  AE_L16X4X2_IP(_ae_int16x4_vec2,_ae_int16x4_vec2_1, _ae_int16x8_p_vec2, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC1_ASYM8b \
  AE_L8X4F_IP(_ae_int16x4_vec1, _WORD8_p_vec1, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec1 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec1), 8)); \
  _ae_int16x4_vec1 = AE_ADD16(_ae_int16x4_vec1, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_VEC2_ASYM8b \
  AE_L8X4F_IP(_ae_int16x4_vec2, _WORD8_p_vec2, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec2 = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec2), 8)); \
  _ae_int16x4_vec2 = AE_ADD16(_ae_int16x4_vec2, AE_MOVDA16(vec2_zero_bias)); \
/*------------------ time batching macros ----------------- */
#define LOAD_VEC_BATCH_f32(idx_vec) \
  AE_LSX2X2_IP(_xtfloatx2_vec_batch_ ##idx_vec,_xtfloatx2_vec_batch_1_ ##idx_vec ,_xtfloatx4_p_vec_batch_ ##idx_vec, 2*INCREMENT_IN_BYTES_FOR_FLOAT32x2); \

#define LOAD_VEC_BATCH_8b(idx_vec) \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \

#define LOAD_VEC_BATCH_16b(idx_vec) \
  AE_L16X4X2_IP(_ae_int16x4_vec_batch_ ##idx_vec,_ae_int16x4_vec_batch_1_ ##idx_vec, _ae_int16x8_p_vec_batch_ ##idx_vec, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_VEC_BATCH_ASYM8b(idx_vec) \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_ ##idx_vec, _WORD8_p_vec_batch_ ##idx_vec, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_vec_batch_ ##idx_vec  = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_batch_ ##idx_vec), 8)); \
  _ae_int16x4_vec_batch_ ##idx_vec = AE_ADD16(_ae_int16x4_vec_batch_ ##idx_vec, AE_MOVDA16(vec1_zero_bias)); \

#define LOAD_BIAS_8b_FOR_8bx8b \
  _WORD8_bias = *_WORD8_p_bias++; \
  _WORD16_bias = _WORD8_bias; \
  *((WORD16 *) _ae_int16_p_bias) = _WORD16_bias; \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \

#define LOAD_BIAS_16b_FOR_8bx16b \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \

#define LOAD_BIAS_16b_FOR_16bx8b LOAD_BIAS_16b_FOR_8bx16b

#define LOAD_BIAS_16b_FOR_16bx16b \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \

#define LOAD_BIAS_f32 \
  AE_LSIP(_xtfloat_bias, _xtfloat_p_bias, INCREMENT_IN_BYTES_FOR_FLOAT32); \

#define LOAD_BIAS_ASYM8b \
  _WORD32_bias = *_WORD32_p_bias++; \
  _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \

#define LOAD_BIAS_ASYM8b_MATMUL \
  if(p_bias!=NULL)\
  {\
    _WORD32_bias = *_WORD32_p_bias++; \
    _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \
  }
/*---------------------------------------------------------*/
#define LOAD_ROW_MAT1_16b_UNALIGNED_16x8(idx) \
    AE_LA16X4X2_IP(_ae_int16x4_mat1_ ## idx,_ae_int16x4_mat1_1_ ## idx,_align_ae_int16x8_p_mat1_ ## idx,_ae_int16x8_p_mat1_ ## idx);

#define LOAD_ROW_MAT2_16b_UNALIGNED_16x8(idx) \
    AE_LA16X4X2_IP(_ae_int16x4_mat2_ ## idx,_ae_int16x4_mat2_1_ ## idx,_align_ae_int16x8_p_mat2_ ## idx,_ae_int16x8_p_mat2_ ## idx);

#define LOAD_ROW_MAT1_16b_UNALIGNED(idx) \
    AE_LA16X4_IP(_ae_int16x4_mat1_ ## idx,_align_ae_int16x4_p_mat1_ ## idx,_ae_int16x4_p_mat1_ ## idx);

#define LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(idx) \
    AE_L16_IP(_ae_int16x4_mat1_ ## idx,(ae_int16 *)_ae_int16x8_p_mat1_ ## idx,2);

#define LOAD_ROW_MAT2_16b_UNALIGNED_SINGLE_ELEMENT_16x8(idx) \
    AE_L16_IP(_ae_int16x4_mat2_ ## idx,(ae_int16 *)_ae_int16x8_p_mat2_ ## idx,2);

#define LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT(idx) \
    AE_L16_IP(_ae_int16x4_mat1_ ## idx,(ae_int16 *)_ae_int16x4_p_mat1_ ## idx,2);

#define LOAD_ROW_MAT2_16b_UNALIGNED(idx) \
    AE_LA16X4_IP(_ae_int16x4_mat2_ ## idx,_align_ae_int16x4_p_mat2_ ## idx,_ae_int16x4_p_mat2_ ## idx);

#define LOAD_ROW_MAT2_16b_UNALIGNED_SINGLE_ELEMENT(idx) \
    AE_L16_IP(_ae_int16x4_mat2_ ## idx,(ae_int16 *)_ae_int16x4_p_mat2_ ## idx,2);


#define LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(idx) \
  AE_L8_IP(_ae_int8x8_mat1_ ## idx,(ae_int8 * )_ae_int8x8_p_mat1_ ## idx,1); \

#define LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(idx) \
  AE_L8_IP(_ae_int8x8_mat2_ ## idx,(ae_int8 * )_ae_int8x8_p_mat2_ ## idx,1); \

#define LOAD_ROW_MAT1_8b_UNALIGNED(idx) \
  AE_LA8X8_IP(_ae_int8x8_mat1_ ## idx, _align_ae_int8x8_p_mat1_ ## idx ,_ae_int8x8_p_mat1_ ## idx); \

#define LOAD_ROW_MAT2_8b_UNALIGNED(idx) \
  AE_LA8X8_IP(_ae_int8x8_mat2_ ## idx, _align_ae_int8x8_p_mat2_ ## idx ,_ae_int8x8_p_mat2_ ## idx); \

#define LOAD_ROW_MAT1_8b(idx) \
  AE_L8X8X2_IP(_ae_int8x8_mat1_ ## idx, _ae_int8x8_mat1_1_ ## idx,_ae_int8x16_p_mat1_ ## idx, 16); \

#define LOAD_ROW_MAT1_8b_16(idx) \
  AE_L8X8X2_IP(_ae_int8x8_mat1_ ## idx, _ae_int8x8_mat1_1_ ## idx,_ae_int8x16_p_mat1_ ## idx, 16); \

#define LOAD_ROW_MAT1_8b_16_1(idx) \
  AE_L8X8X2_IP(_ae_int8x8_mat1_2_ ## idx, _ae_int8x8_mat1_3_ ## idx,_ae_int8x16_p_mat1_ ## idx, 16); \

#define LOAD_ROW_MAT2_8b(idx) \
  AE_L8X8X2_IP(_ae_int8x8_mat2_ ## idx, _ae_int8x8_mat2_1_ ## idx,_ae_int8x16_p_mat2_ ## idx, 16); \

#define LOAD_ROW_MAT2_8b_16(idx) \
  AE_L8X8X2_IP(_ae_int8x8_mat2_ ## idx, _ae_int8x8_mat2_1_ ## idx,_ae_int8x16_p_mat2_ ## idx, 16); \

#define LOAD_ROW_MAT2_8b_16_1(idx) \
  AE_L8X8X2_IP(_ae_int8x8_mat2_2_ ## idx, _ae_int8x8_mat2_3_ ## idx,_ae_int8x16_p_mat2_ ## idx,16); \

#define LOAD_ROW_MAT1_16b(idx) \
  AE_L16X4X2_IP(_ae_int16x4_mat1_ ## idx,_ae_int16x4_mat1_1_ ## idx,_ae_int16x8_p_mat1_ ## idx, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_ROW_MAT2_16b(idx) \
  AE_L16X4X2_IP(_ae_int16x4_mat2_ ## idx,_ae_int16x4_mat2_1_ ## idx ,_ae_int16x8_p_mat2_ ## idx, 2*INCREMENT_IN_BYTES_FOR_INT16X4); \

#define LOAD_ROW_MAT1_f32(idx) \
  AE_LSX2X2_IP(_xtfloatx2_mat1_ ## idx,_xtfloatx2_mat1_1_ ## idx, _xtfloatx4_p_mat1_ ## idx, 2*INCREMENT_IN_BYTES_FOR_FLOAT32x2); \

#define LOAD_ROW_MAT1_ASYM8b(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat1_ ##idx, _WORD8_p_mat1_ ##idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat1_ ##idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx), 8)); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8b_UNALIGNED(idx) \
  AE_LA8X4F_IP(_ae_int16x4_mat1_ ## idx, _align_WORD8_p_mat1_ ## idx, _WORD8_p_mat1_ ## idx); \
  _ae_int16x4_mat1_ ##idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx), 8)); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT1_ASYM8b_SINGLE_UNALIGNED(idx) \
  _ae_int16x4_mat1_ ## idx = AE_MOVDA16(((short)*(_WORD8_p_mat1_ ## idx)) << 8); \
  _WORD8_p_mat1_ ## idx++;\
  _ae_int16x4_mat1_ ##idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_ ##idx), 8)); \
  _ae_int16x4_mat1_ ##idx = AE_ADD16(_ae_int16x4_mat1_ ##idx, AE_MOVDA16(mat1_zero_bias)); \

#define LOAD_ROW_MAT2_ASYM8b(idx) \
  AE_L8X4F_IP(_ae_int16x4_mat2_ ## idx, _WORD8_p_mat2_ ## idx, INCREMENT_IN_BYTES_FOR_WORD8X4); \
  _ae_int16x4_mat2_ ## idx = AE_MOVF16X4_FROMF64(AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat2_ ## idx), 8)); \
  _ae_int16x4_mat2_ ##idx = AE_ADD16(_ae_int16x4_mat2_ ##idx, AE_MOVDA16(mat2_zero_bias)); \

#define KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(idx,idx1)\
            AE_MULAAAA2Q16X8(_ae_int64_acc_ ## idx ,_ae_int64_acc_ ## idx1,_ae_int16x4_mat1_ ## idx ,_ae_int16x4_mat1_1_ ## idx1,_ae_int8x8_vec1);\
            AE_MULAAAA2Q16X8(_ae_int64_acc_ ## idx1 ,_ae_int64_acc_ ## idx,_ae_int16x4_mat1_ ## idx1,_ae_int16x4_mat1_1_ ## idx ,_ae_int8x8_vec1);


#define KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(idx,idx1)\
            AE_MULAAAA2Q16X8(_ae_int64_acc_ ## idx ,_ae_int64_acc_ ## idx1,_ae_int16x4_mat2_ ## idx ,_ae_int16x4_mat2_1_ ## idx1,_ae_int8x8_vec2);\
            AE_MULAAAA2Q16X8(_ae_int64_acc_ ## idx1 ,_ae_int64_acc_ ## idx,_ae_int16x4_mat2_ ## idx1,_ae_int16x4_mat2_1_ ## idx ,_ae_int8x8_vec2);

#define KERNEL_MAT1_VEC1_16b_8b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0)),_ae_int16x4_mat1_0);\

#define KERNEL_MAT2_VEC2_16b_8b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0)),_ae_int16x4_mat2_0);\

#define KERNEL_MAT1_VEC1_16b_8b_UNALIGNED_SINGLE_ELEMENT \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0)),_ae_int16x4_mat1_0);\
   AE_MULA16_00(_ae_int64_acc_1,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0)),_ae_int16x4_mat1_1);

#define KERNEL_MAT2_VEC2_16b_8b_UNALIGNED_SINGLE_ELEMENT \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0)),_ae_int16x4_mat2_0);\
   AE_MULA16_00(_ae_int64_acc_1,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0)),_ae_int16x4_mat2_1);

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_SINGLE_ROW \
    AE_MULAAAAQ16(_ae_int64_acc_0,_ae_int16x4_mat1_0,_ae_int16x4_vec1); \

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW \
    AE_MULA16_00(_ae_int64_acc_0,_ae_int16x4_mat1_0,_ae_int16x4_vec1); \

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW_BATCH \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1);\
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1);\

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW_8x16_batch \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1_0);\

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1);\

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_BATCH_SINGLE_ROW \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1_0);\
   AE_MULA16_00(_ae_int64_acc_2,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1_1);\

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_BATCH \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1_0);\
   AE_MULA16_00(_ae_int64_acc_1,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),_ae_int16x4_vec1_0);\
   AE_MULA16_00(_ae_int64_acc_2,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1_1);\
   AE_MULA16_00(_ae_int64_acc_3,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),_ae_int16x4_vec1_1);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_VECTOR \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1_0);\
   AE_MULA16_00(_ae_int64_acc_1,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),_ae_int16x4_vec1_0);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT \
   AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),_ae_int16x4_vec1);\
   AE_MULA16_00(_ae_int64_acc_1,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),_ae_int16x4_vec1);\
   AE_MULA16_00(_ae_int64_acc_2,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_2,0)),_ae_int16x4_vec1);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_BATCH_SINGLE_ROW \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat1_0,zero_temp, zero_temp,zero_temp,_ae_int16x4_vec1_0,_ae_int16x4_vec1_1_0);\
        AE_MULA8QW8X16(_ae_int64_acc_2,_ae_int64_acc_3,_ae_int64_acc_0,_ae_int64_acc_1,_ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,_ae_int16x4_vec1_1,_ae_int16x4_vec1_1_1);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_BATCH \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat1_0, _ae_int8x8_mat1_1, zero_temp,zero_temp,_ae_int16x4_vec1_0,_ae_int16x4_vec1_1_0);\
        AE_MULA8QW8X16(_ae_int64_acc_2,_ae_int64_acc_3,_ae_int64_acc_0,_ae_int64_acc_1,_ae_int8x8_mat1_0, _ae_int8x8_mat1_1, zero_temp,zero_temp,_ae_int16x4_vec1_1,_ae_int16x4_vec1_1_1);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_VECTOR \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat1_0, _ae_int8x8_mat1_1, zero_temp,zero_temp,_ae_int16x4_vec1_0,_ae_int16x4_vec1_1_0);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ROW_8x16_BATCH \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat1_0,zero_temp, zero_temp,zero_temp,_ae_int16x4_vec1_0,_ae_int16x4_vec1_1_0);


#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ROW \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat1_0,zero_temp, zero_temp,zero_temp,_ae_int16x4_vec1,_ae_int16x4_vec1_1);

#define KERNEL_MAT1_VEC1_8b_16b_UNALIGNED \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat1_0, _ae_int8x8_mat1_1, _ae_int8x8_mat1_2, zero_temp,_ae_int16x4_vec1,_ae_int16x4_vec1_1);

#define KERNEL_MAT2_VEC2_8b_16b_UNALIGNED \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat2_0, _ae_int8x8_mat2_1, _ae_int8x8_mat2_2 ,zero_temp,_ae_int16x4_vec2,_ae_int16x4_vec2_1);

#define KERNEL_MAT2_VEC2_8b_16b_UNALIGNED_SINGLE_ROW \
        AE_MULA8QW8X16(_ae_int64_acc_0,_ae_int64_acc_1,_ae_int64_acc_2,_ae_int64_acc_3,_ae_int8x8_mat2_0, zero_temp, zero_temp,zero_temp,_ae_int16x4_vec2,_ae_int16x4_vec2_1);

#define KERNEL_MAT2_VEC2_8b_16b_UNALIGNED_SINGLE_ELEMENT \
                  AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_0,0)),_ae_int16x4_vec2);\
                  AE_MULA16_00(_ae_int64_acc_1,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_1,0)),_ae_int16x4_vec2);\
                  AE_MULA16_00(_ae_int64_acc_2,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_2,0)),_ae_int16x4_vec2);\

#define KERNEL_MAT2_VEC2_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW \
                  AE_MULA16_00(_ae_int64_acc_0,AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_0,0)),_ae_int16x4_vec2);\

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_BATCH_SINGLE_ROW(acc1,idx) \
    AE_MULAAAAQ16(_ae_int64_acc_ ## acc1,_ae_int16x4_mat1_0,_ae_int16x4_vec1_ ## idx); \

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_BATCH(acc1,acc2,idx) \
    AE_MULAAAAQ16(_ae_int64_acc_ ## acc1,_ae_int16x4_mat1_0,_ae_int16x4_vec1_ ## idx); \
    AE_MULAAAAQ16(_ae_int64_acc_ ## acc2,_ae_int16x4_mat1_1,_ae_int16x4_vec1_ ## idx);

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED \
    AE_MULAAAAQ16(_ae_int64_acc_0,_ae_int16x4_mat1_0,_ae_int16x4_vec1); \
    AE_MULAAAAQ16(_ae_int64_acc_1,_ae_int16x4_mat1_1,_ae_int16x4_vec1);

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_SINGLE_ELEMENT_BATCH_SINGLE_ROW(acc1,idx) \
    AE_MULA16_00(_ae_int64_acc_ ## acc1,_ae_int16x4_mat1_0,_ae_int16x4_vec1_ ## idx); \

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_SINGLE_ELEMENT_BATCH(acc1,acc2,idx) \
    AE_MULA16_00(_ae_int64_acc_ ## acc1,_ae_int16x4_mat1_0,_ae_int16x4_vec1_ ## idx); \
    AE_MULA16_00(_ae_int64_acc_ ## acc2,_ae_int16x4_mat1_1,_ae_int16x4_vec1_ ## idx);

#define KERNEL_MAT1_VEC1_16b_16b_UNALIGNED_SINGLE_ELEMENT \
    AE_MULA16_00(_ae_int64_acc_0,_ae_int16x4_mat1_0,_ae_int16x4_vec1); \
    AE_MULA16_00(_ae_int64_acc_1,_ae_int16x4_mat1_1,_ae_int16x4_vec1);

#define KERNEL_MAT2_VEC2_16b_16b_UNALIGNED_SINGLE_ROW \
    AE_MULAAAAQ16(_ae_int64_acc_0,_ae_int16x4_mat2_0,_ae_int16x4_vec2); \

#define KERNEL_MAT2_VEC2_16b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW \
    AE_MULA16_00(_ae_int64_acc_0,_ae_int16x4_mat2_0,_ae_int16x4_vec2); \

#define KERNEL_MAT2_VEC2_16b_16b_UNALIGNED \
    AE_MULAAAAQ16(_ae_int64_acc_0,_ae_int16x4_mat2_0,_ae_int16x4_vec2); \
    AE_MULAAAAQ16(_ae_int64_acc_1,_ae_int16x4_mat2_1,_ae_int16x4_vec2);

#define KERNEL_MAT2_VEC2_16b_16b_UNALIGNED_SINGLE_ELEMENT \
    AE_MULA16_00(_ae_int64_acc_0,_ae_int16x4_mat2_0,_ae_int16x4_vec2); \
    AE_MULA16_00(_ae_int64_acc_1,_ae_int16x4_mat2_1,_ae_int16x4_vec2);




#define KERNEL_MAT2_VEC2_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT \
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0)));

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT_BATCH_SINGLE_VECTOR \
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_0,0)));\

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT_BATCH \
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_0,0)));\
          _ae_int32x2_acc_2= _ae_int32x2_acc_2 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_1,0)));

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT \
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0)));


#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_VECTOR_BATCH \
    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat1_0 ,zero_temp , zero_temp,_ae_int8x8_vec1_0);\

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_BATCH \
    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat1_0 ,zero_temp , zero_temp,_ae_int8x8_vec1_0);\
    AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 ,zero_temp,_ae_int8x8_mat1_0 ,zero_temp , zero_temp,_ae_int8x8_vec1_1);

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW \
    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat1_0 ,zero_temp , zero_temp,_ae_int8x8_vec1);

#define KERNEL_MAT2_VEC2_8b_8b_UNALIGNED_SINGLE_ROW \
    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat2_0 ,zero_temp , zero_temp,_ae_int8x8_vec2);

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_BATCH_SINGLE_ROW \
                    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat1_0 , zero_temp, _ae_int8x8_mat1_1, _ae_int8x8_vec1_0);\

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_BATCH \
                    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat1_0 , zero_temp, _ae_int8x8_mat1_1, _ae_int8x8_vec1_0);\
                    AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 ,zero_temp,_ae_int8x8_mat1_0 , zero_temp, _ae_int8x8_mat1_1, _ae_int8x8_vec1_1);



#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED \
    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat1_0 ,_ae_int8x8_mat1_2, _ae_int8x8_mat1_1,_ae_int8x8_vec1);

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT_BATCH_SINGLE_ROW \
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_0,0)));\
          _ae_int32x2_acc_1= _ae_int32x2_acc_1 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_0,0)));\

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT_BATCH \
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_0,0)));\
          _ae_int32x2_acc_1= _ae_int32x2_acc_1 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_0,0)));\
          _ae_int32x2_acc_2= _ae_int32x2_acc_2 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_1,0)));\
          _ae_int32x2_acc_3= _ae_int32x2_acc_3 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1_1,0)));\

#define KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT \
          ae_int32x2 temp;\
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0)));\
          temp = AE_SEL32_LL(AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_2,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0))),AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat1_1,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec1,0))));\
          _ae_int32x2_acc_1= _ae_int32x2_acc_1 + temp;\

#define KERNEL_MAT2_VEC2_8b_8b_UNALIGNED_SINGLE_ELEMENT \
          ae_int32x2 temp;\
          _ae_int32x2_acc_0= _ae_int32x2_acc_0 + AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_0,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0)));\
          temp = AE_SEL32_LL(AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_2,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0))),AE_MUL16S(AE_MOVDA16(AE_MOVAD8(_ae_int8x8_mat2_1,0)),AE_MOVDA16(AE_MOVAD8(_ae_int8x8_vec2,0))));\
          _ae_int32x2_acc_1= _ae_int32x2_acc_1 + temp;\


#define KERNEL_MAT2_VEC2_8b_8b_UNALIGNED \
    AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 ,zero_temp,_ae_int8x8_mat2_0 ,_ae_int8x8_mat2_2 , _ae_int8x8_mat2_1,_ae_int8x8_vec2);

#define KERNEL_MAT1_VEC1_8b_8b(idx) \
  AE_MUL8Q8X8(_ae_int32x2_acc_ ## idx,_ae_int32x2_acc_ ## idx,\
              _ae_int8x8_mat1_ ## idx, _ae_int8x8_mat1_ ## idx,\
              _ae_int8x8_mat1_ ## idx, _ae_int8x8_mat1_ ## idx,\
              _ae_int8x8_vec1 );\

#define KERNEL_MAT2_VEC2_8b_8b(idx) \
  AE_MUL8Q8X8(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_16b_8b(idx) \
  LOAD_ROW_MAT1_16b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_16b_8b(idx) \
  LOAD_ROW_MAT2_16b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_8b_16b(idx) \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_8b_16b(idx) \
  LOAD_ROW_MAT2_8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \

#define KERNEL_MAT1_VEC1_16b_16b(idx) \
  LOAD_ROW_MAT1_16b(idx); \
  AE_MULAAAA2Q16(_ae_int64_acc_ ## idx,_ae_int64_acc_1_ ## idx, _ae_int16x4_vec1, _ae_int16x4_vec1_1 ,_ae_int16x4_mat1_ ## idx,_ae_int16x4_mat1_1_ ## idx); \

#define KERNEL_MAT2_VEC2_16b_16b(idx) \
  LOAD_ROW_MAT2_16b(idx); \
  AE_MULAAAA2Q16(_ae_int64_acc_ ## idx,_ae_int64_acc_1_ ## idx, _ae_int16x4_vec2,_ae_int16x4_vec2_1, _ae_int16x4_mat2_ ## idx, _ae_int16x4_mat2_1_ ## idx); \

#define kernel_mat1_16_vec2_16_8x8_acc_32_DUAL \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_0 , _ae_int8x8_mat1_1 , _ae_int8x8_mat1_2 , _ae_int8x8_mat1_3 , _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat1_4 , _ae_int8x8_mat1_5 , _ae_int8x8_mat1_6 , _ae_int8x8_mat1_7 , _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat1_8 , _ae_int8x8_mat1_9 , _ae_int8x8_mat1_10, _ae_int8x8_mat1_11, _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat1_12, _ae_int8x8_mat1_13, _ae_int8x8_mat1_14, _ae_int8x8_mat1_15, _ae_int8x8_vec1); \
         \
          AE_MULA8Q8X8(_ae_int32x2_acc_8 ,  _ae_int32x2_acc_9 , _ae_int8x8_mat1_1_0 , _ae_int8x8_mat1_1_1 , _ae_int8x8_mat1_1_2 , _ae_int8x8_mat1_1_3 , _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_10,  _ae_int32x2_acc_11, _ae_int8x8_mat1_1_4 , _ae_int8x8_mat1_1_5 , _ae_int8x8_mat1_1_6 , _ae_int8x8_mat1_1_7 , _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_12,  _ae_int32x2_acc_13, _ae_int8x8_mat1_1_8 , _ae_int8x8_mat1_1_9 , _ae_int8x8_mat1_1_10, _ae_int8x8_mat1_1_11, _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_14,  _ae_int32x2_acc_15, _ae_int8x8_mat1_1_12, _ae_int8x8_mat1_1_13, _ae_int8x8_mat1_1_14, _ae_int8x8_mat1_1_15, _ae_int8x8_vec1_1); \

#define kernel_mat1_16_vec2_16_8x8_acc_32_DUAL_1 \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_2_0 , _ae_int8x8_mat1_2_1 , _ae_int8x8_mat1_2_2 , _ae_int8x8_mat1_2_3 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat1_2_4 , _ae_int8x8_mat1_2_5 , _ae_int8x8_mat1_2_6 , _ae_int8x8_mat1_2_7 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat1_2_8 , _ae_int8x8_mat1_2_9 , _ae_int8x8_mat1_2_10, _ae_int8x8_mat1_2_11, _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat1_2_12, _ae_int8x8_mat1_2_13, _ae_int8x8_mat1_2_14, _ae_int8x8_mat1_2_15, _ae_int8x8_vec1_2); \
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_8 ,  _ae_int32x2_acc_9 , _ae_int8x8_mat1_3_0 , _ae_int8x8_mat1_3_1 , _ae_int8x8_mat1_3_2 , _ae_int8x8_mat1_3_3 , _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_10,  _ae_int32x2_acc_11, _ae_int8x8_mat1_3_4 , _ae_int8x8_mat1_3_5 , _ae_int8x8_mat1_3_6 , _ae_int8x8_mat1_3_7 , _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_12,  _ae_int32x2_acc_13, _ae_int8x8_mat1_3_8 , _ae_int8x8_mat1_3_9 , _ae_int8x8_mat1_3_10, _ae_int8x8_mat1_3_11, _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_14,  _ae_int32x2_acc_15, _ae_int8x8_mat1_3_12, _ae_int8x8_mat1_3_13, _ae_int8x8_mat1_3_14, _ae_int8x8_mat1_3_15, _ae_int8x8_vec1_3);

#define kernel_mat2_16_vec2_16_8x8_acc_32_DUAL \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_0 , _ae_int8x8_mat2_1 , _ae_int8x8_mat2_2 , _ae_int8x8_mat2_3 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat2_4 , _ae_int8x8_mat2_5 , _ae_int8x8_mat2_6 , _ae_int8x8_mat2_7 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat2_8 , _ae_int8x8_mat2_9 , _ae_int8x8_mat2_10, _ae_int8x8_mat2_11, _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat2_12, _ae_int8x8_mat2_13, _ae_int8x8_mat2_14, _ae_int8x8_mat2_15, _ae_int8x8_vec2); \
         \
          AE_MULA8Q8X8(_ae_int32x2_acc_8 ,  _ae_int32x2_acc_9 , _ae_int8x8_mat2_1_0 , _ae_int8x8_mat2_1_1 , _ae_int8x8_mat2_1_2 , _ae_int8x8_mat2_1_3 , _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_10,  _ae_int32x2_acc_11, _ae_int8x8_mat2_1_4 , _ae_int8x8_mat2_1_5 , _ae_int8x8_mat2_1_6 , _ae_int8x8_mat2_1_7 , _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_12,  _ae_int32x2_acc_13, _ae_int8x8_mat2_1_8 , _ae_int8x8_mat2_1_9 , _ae_int8x8_mat2_1_10, _ae_int8x8_mat2_1_11, _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_14,  _ae_int32x2_acc_15, _ae_int8x8_mat2_1_12, _ae_int8x8_mat2_1_13, _ae_int8x8_mat2_1_14, _ae_int8x8_mat2_1_15, _ae_int8x8_vec2_1);

#define kernel_mat2_16_vec2_16_8x8_acc_32_DUAL_1 \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_2_0 , _ae_int8x8_mat2_2_1 , _ae_int8x8_mat2_2_2 , _ae_int8x8_mat2_2_3 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat2_2_4 , _ae_int8x8_mat2_2_5 , _ae_int8x8_mat2_2_6 , _ae_int8x8_mat2_2_7 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat2_2_8 , _ae_int8x8_mat2_2_9 , _ae_int8x8_mat2_2_10, _ae_int8x8_mat2_2_11, _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat2_2_12, _ae_int8x8_mat2_2_13, _ae_int8x8_mat2_2_14, _ae_int8x8_mat2_2_15, _ae_int8x8_vec2_2); \
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_8 ,  _ae_int32x2_acc_9 , _ae_int8x8_mat2_3_0 , _ae_int8x8_mat2_3_1 , _ae_int8x8_mat2_3_2 , _ae_int8x8_mat2_3_3 , _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_10,  _ae_int32x2_acc_11, _ae_int8x8_mat2_3_4 , _ae_int8x8_mat2_3_5 , _ae_int8x8_mat2_3_6 , _ae_int8x8_mat2_3_7 , _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_12,  _ae_int32x2_acc_13, _ae_int8x8_mat2_3_8 , _ae_int8x8_mat2_3_9 , _ae_int8x8_mat2_3_10, _ae_int8x8_mat2_3_11, _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_14,  _ae_int32x2_acc_15, _ae_int8x8_mat2_3_12, _ae_int8x8_mat2_3_13, _ae_int8x8_mat2_3_14, _ae_int8x8_mat2_3_15, _ae_int8x8_vec2_3);


#define kernel_mat1_16_vec2_16_8x8_acc_32 \
          AE_MULA8Q8X8(_ae_int32x2_acc_0, _ae_int32x2_acc_1,_ae_int8x8_mat1_0, _ae_int8x8_mat1_1, \
                      _ae_int8x8_mat1_2, _ae_int8x8_mat1_3,_ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2, _ae_int32x2_acc_3,_ae_int8x8_mat1_4, _ae_int8x8_mat1_5, \
                      _ae_int8x8_mat1_6, _ae_int8x8_mat1_7,_ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4, _ae_int32x2_acc_5,_ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, \
                      _ae_int8x8_mat1_1_2, _ae_int8x8_mat1_1_3,_ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6, _ae_int32x2_acc_7,_ae_int8x8_mat1_1_4, _ae_int8x8_mat1_1_5,\
                      _ae_int8x8_mat1_1_6, _ae_int8x8_mat1_1_7,_ae_int8x8_vec1_1);

#define kernel_mat2_16_vec2_16_8x8_acc_32 \
          AE_MULA8Q8X8(_ae_int32x2_acc_0, _ae_int32x2_acc_1,_ae_int8x8_mat2_0, _ae_int8x8_mat2_1,\
                      _ae_int8x8_mat2_2, _ae_int8x8_mat2_3,_ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2, _ae_int32x2_acc_3,_ae_int8x8_mat2_4, _ae_int8x8_mat2_5,\
                      _ae_int8x8_mat2_6, _ae_int8x8_mat2_7,_ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4, _ae_int32x2_acc_5,_ae_int8x8_mat2_1_0, _ae_int8x8_mat2_1_1,\
                      _ae_int8x8_mat2_1_2, _ae_int8x8_mat2_1_3,_ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6, _ae_int32x2_acc_7,_ae_int8x8_mat2_1_4, _ae_int8x8_mat2_1_5, \
                      _ae_int8x8_mat2_1_6, _ae_int8x8_mat2_1_7,_ae_int8x8_vec2_1);

#define adding_conjugate_accumulators_32_DUAL \
        _ae_int32x2_acc_0 = _ae_int32x2_acc_0 + _ae_int32x2_acc_8;\
        _ae_int32x2_acc_1 = _ae_int32x2_acc_1 + _ae_int32x2_acc_9;\
        _ae_int32x2_acc_2 = _ae_int32x2_acc_2 + _ae_int32x2_acc_10;\
        _ae_int32x2_acc_3 = _ae_int32x2_acc_3 + _ae_int32x2_acc_11;\
        _ae_int32x2_acc_4 = _ae_int32x2_acc_4 + _ae_int32x2_acc_12;\
        _ae_int32x2_acc_5 = _ae_int32x2_acc_5 + _ae_int32x2_acc_13;\
        _ae_int32x2_acc_6 = _ae_int32x2_acc_6 + _ae_int32x2_acc_14;\
        _ae_int32x2_acc_7 = _ae_int32x2_acc_7 + _ae_int32x2_acc_15;

#define adding_conjugate_accumulators_32 \
        _ae_int32x2_acc_0 = _ae_int32x2_acc_0 + _ae_int32x2_acc_4;\
        _ae_int32x2_acc_1 = _ae_int32x2_acc_1 + _ae_int32x2_acc_5;\
        _ae_int32x2_acc_2 = _ae_int32x2_acc_2 + _ae_int32x2_acc_6;\
        _ae_int32x2_acc_3 = _ae_int32x2_acc_3 + _ae_int32x2_acc_7;

#define KERNEL_MAT1_VEC1_ASYM8b_ASYM8b(idx) \
  LOAD_ROW_MAT1_ASYM8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec1, _ae_int16x4_mat1_ ## idx); \

#define KERNEL_MAT2_VEC2_ASYM8b_ASYM8b(idx) \
  LOAD_ROW_MAT2_ASYM8b(idx); \
  AE_MULAAAAQ16(_ae_int64_acc_ ## idx, _ae_int16x4_vec2, _ae_int16x4_mat2_ ## idx); \
/*------------------ time batching macros ----------------- */

#define KERNEL_MAT1_VEC_BATCH_ROW_8b_8b  KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_16b_8b KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_8b_16b KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b     KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_8b      KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_16b_8b     KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_16b     KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b(idx_row,idx_vec) \
  AE_MULAAAAQ16(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec, _ae_int16x4_mat1_ ##idx_row); \

#define KERNEL_MAT1_VEC_BATCH_ROW_16b_16b(idx_row)\
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row);\

#define KERNEL_MAT1_VEC_BATCH_16b_16b(idx_row,idx_vec) \
  AE_MULAAAA2Q16(_ae_int64_acc_ ##idx_row ##_ ##idx_vec,_ae_int64_acc_1_ ##idx_row ##_ ##idx_vec, _ae_int16x4_vec_batch_ ##idx_vec,_ae_int16x4_vec_batch_1_ ##idx_vec, _ae_int16x4_mat1_ ##idx_row,_ae_int16x4_mat1_1_ ##idx_row); \

#define KERNEL_MAT1_VEC_BATCH_ROW_f32(idx_row)\
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row);\

#define KERNEL_MAT1_VEC_BATCH_f32(idx_row,idx_vec) \
    MADD_SX2X2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec,_xtfloatx2_acc_1_ ##idx_row ##_ ##idx_vec, _xtfloatx2_vec_batch_ ##idx_vec,_xtfloatx2_vec_batch_1_ ##idx_vec, _xtfloatx2_mat1_ ##idx_row, _xtfloatx2_mat1_1_ ##idx_row); \

/*---------------------------------------------------------*/
#define ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE \
        AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        _ae_int64_acc_0 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_0,ZERO32));\
        _ae_int64_acc_0 = AE_SRAA64(_ae_int64_acc_0,32);\
        sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias,0)),ZERO32));\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_0  = AE_ADD64S(_ae_int64_acc_0,sat_1);\

#define ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(idx,idx1) \
        AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias,0)),ZERO32));\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx,32);\
        _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx,sat_1);\
        _ae_int64_acc_ ## idx1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx1,ZERO32));\
        _ae_int64_acc_ ## idx1 = AE_SRAA64(_ae_int64_acc_ ## idx1,32);\
        _ae_int64_acc_ ## idx1 = AE_ADD64S(_ae_int64_acc_ ## idx1,sat_1);\

#define ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(idx) \
        AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        _ae_int64_acc_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx,32);\
        sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias,0)),ZERO32));\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx,sat_1);\

#define ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(idx) \
        AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        _ae_int64_acc_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx, ZERO32));\
        _ae_int64_acc_1 = AE_SRAA64(_ae_int64_acc_1,32);\
        sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias,0)),ZERO32));\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1,sat_1);\
        \
        AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        _ae_int64_acc_2 = AE_MOVF64_FROMF32X2(AE_SEL32_HH(_ae_int32x2_acc_ ## idx, ZERO32));\
        _ae_int64_acc_2 = AE_SRAA64(_ae_int64_acc_2,32);\
        sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias,0)),ZERO32));\
        sat_1 = AE_SRAA64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_2 = AE_ADD64S(_ae_int64_acc_2,sat_1);\

#define ADD_BIAS_8b_ACC_FOR_8bx8b(idx) \
        AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        AE_L8_IP(_ae_int8_bias_1,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);\
        sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias,0)),ZERO32));\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        sat_2 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(AE_MOVDA32(AE_MOVAD8(_ae_int8_bias_1,0)),ZERO32));\
        sat_2 = AE_SRAI64(sat_2,32);\
        sat_2 = AE_SLAA64S(sat_2,bias_shift);\
        _ae_int64_acc_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_HH(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx,32);\
        _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx,sat_1);\
        \
        _ae_int64_acc_1_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_1_ ## idx = AE_SRAA64(_ae_int64_acc_1_ ## idx,32);\
        _ae_int64_acc_1_ ## idx = AE_ADD64S(_ae_int64_acc_1_ ## idx,sat_2);\

#define ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_EXTRA_ROW(idx) \
        AE_L32_IP(_ae_int32x2_bias,(ae_int32*) _ae_int32x2_p_bias, INCREMENT_IN_BYTES_FOR_INT32); \
        sat_1 = AE_MOVF64_FROMF32X2(_ae_int32x2_bias);\
        sat_1 = AE_SRAA64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx,32);\
        _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx,sat_1);\

#define ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_1(idx) \
        AE_L32_IP(_ae_int32_bias, _ae_int32_p_bias, INCREMENT_IN_BYTES_FOR_INT32); \
        sat_1 = AE_MOVF64_FROMF32X2(_ae_int32_bias);\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_1 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_1 = AE_SRAA64(_ae_int64_acc_1,32);\
        _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1,sat_1);\
        \
        AE_L32_IP(_ae_int32_bias, _ae_int32_p_bias, INCREMENT_IN_BYTES_FOR_INT32); \
        sat_1 = AE_MOVF64_FROMF32X2(_ae_int32_bias);\
        sat_1 = AE_SRAI64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_2 = AE_MOVF64_FROMF32X2(AE_SEL32_HH(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_2 = AE_SRAA64(_ae_int64_acc_2,32);\
        _ae_int64_acc_2 = AE_ADD64S(_ae_int64_acc_2,sat_1);\


#define ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(idx) \
        AE_L32_IP(_ae_int32_bias, _ae_int32_p_bias, INCREMENT_IN_BYTES_FOR_INT32); \
        sat_1 = AE_MOVF64_FROMF32X2(_ae_int32_bias);\
        sat_1 = AE_SRAA64(sat_1,32);\
        sat_1 = AE_SLAA64S(sat_1,bias_shift);\
        _ae_int64_acc_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
        _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx,32);\
        _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx,sat_1);\

#define ADD_BIAS_32b_ACC_FOR_8bx8b(idx) \
  AE_L32X2_IP(_ae_int32x2_bias, _ae_int32x2_p_bias, 2*INCREMENT_IN_BYTES_FOR_INT32); \
  sat_1 = AE_MOVF64_FROMF32X2(AE_SEL32_HH(_ae_int32x2_bias,ZERO32));\
  sat_1 = AE_SRAI64(sat_1,32);\
  sat_1 = AE_SLAA64S(sat_1,bias_shift);\
  sat_2 = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_bias,ZERO32));\
  sat_2 = AE_SRAI64(sat_2,32);\
  sat_2 = AE_SLAA64S(sat_2,bias_shift);\
  _ae_int64_acc_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_HH(_ae_int32x2_acc_ ## idx,ZERO32));\
  _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx,32);\
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx,sat_1);\
  \
  _ae_int64_acc_1_ ## idx = AE_MOVF64_FROMF32X2(AE_SEL32_LL(_ae_int32x2_acc_ ## idx,ZERO32));\
  _ae_int64_acc_1_ ## idx = AE_SRAA64(_ae_int64_acc_1_ ## idx,32);\
  _ae_int64_acc_1_ ## idx = AE_ADD64S(_ae_int64_acc_1_ ## idx,sat_2);\

#define ADD_BIAS_16b_ACC_FOR_8bx16b_NEW(idx) \
  AE_L16_IP(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  /* Saturate 16b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(AE_MOVDA64(AE_MOVAD16_0(_ae_int16_bias)), bias_shift); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_16b_ACC_FOR_8bx16b_BATCH(idx,idx1) \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  /* Saturate 16b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \
  _ae_int64_acc_ ## idx1 = AE_ADD64S(_ae_int64_acc_ ## idx1, _ae_int64_sat_bias); \


#define ADD_BIAS_16b_ACC_FOR_8bx16b(idx) \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  /* Saturate 16b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_16b_ACC_FOR_16bx8b ADD_BIAS_16b_ACC_FOR_8bx16b

#define ADD_BIAS_64b_ACC_FOR_8bx16b(idx) \
  ae_int64_loadip(_ae_int64_bias, _ae_int64_p_bias, INCREMENT_IN_BYTES_FOR_INT64); \
  /* Saturate 64b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int64_bias), bias_shift); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_16b_ACC_FOR_16bx16b(idx) \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16); \
  /* Saturate 16b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift); \
  _ae_int64_acc_ ## idx =_ae_int64_acc_ ## idx+_ae_int64_acc_1_ ## idx;\
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_64b_ACC_FOR_16bx16b(idx) \
  ae_int64_loadip(_ae_int64_bias, _ae_int64_p_bias, INCREMENT_IN_BYTES_FOR_INT64); \
  /* Saturate 64b bias after shift to 64b */ \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int64_bias), bias_shift); \
  _ae_int64_acc_ ## idx =_ae_int64_acc_ ## idx+_ae_int64_acc_1_ ## idx;\
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

#define ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx) \
    /* Load 32b bias */ \
  _WORD32_bias = *_WORD32_p_bias++; \
  _ae_int64_sat_bias = AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \
  _ae_int64_acc_ ## idx = AE_ADD64S(_ae_int64_acc_ ## idx, _ae_int64_sat_bias); \

/*------------------ time batching macros ----------------- */
#define ADD_BIAS_BATCH_ROW_8b_ACC_FOR_8bx8b(idx_row)\
  LOAD_BIAS_8b_FOR_8bx8b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_8bx16b(idx_row)\
  LOAD_BIAS_16b_FOR_8bx16b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx8b(idx_row)\
  LOAD_BIAS_16b_FOR_16bx8b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx16b(idx_row)\
  LOAD_BIAS_16b_FOR_16bx16b; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx_row) \
  LOAD_BIAS_ASYM8b \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row); \

#define ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b_MATMUL(idx_row) \
  LOAD_BIAS_ASYM8b_MATMUL \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row); \

#define ADD_BIAS_BATCH_8b_ACC_FOR_8bx8b    ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b
#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx8b    ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b
#define ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b    ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b

#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b(idx_row,idx_vec)\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec=_ae_int64_acc_1_ ##idx_row ##_ ##idx_vec+_ae_int64_acc_ ##idx_row ##_ ##idx_vec;\
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_ADD64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, _ae_int64_sat_bias); \

#define ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b     ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b

#define ADD_BIAS_BATCH_ROW_ACC_FOR_f32(idx_row)\
  LOAD_BIAS_f32; \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);\

#define ADD_BIAS_BATCH_ACC_FOR_f32(idx_row,idx_vec)\
  _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec =_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec+_xtfloatx2_acc_1_ ##idx_row ##_ ##idx_vec;\
  _xtfloat_acc_ ##idx_row ##_ ##idx_vec = RADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec);\
  _xtfloat_acc_ ##idx_row ##_ ##idx_vec = ADD_S(_xtfloat_acc_ ##idx_row ##_ ##idx_vec, _xtfloat_bias); \

#define STORE_ACC_8bx8b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *) p_scratch + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_8bx8b_AT_OUT_8b(idx) \
  AE_ROUND16X4F32SSYM(0, AE_TRUNCA32F64S(_ae_int64_acc_ ## idx,, acc_shift)); \
  _ae_int16_tmp_var_ ## idx = AE_SLAI16S(_ae_f16x4_tmp_var_ ## idx, 8); \
  (*((WORD8 *) p_out + m_itr + idx)) = (((*((UWORD16 *)&_ae_int16_tmp_var_ ## idx)) & 0xFF00) >> 8); \

#define STORE_ACC_8bx8b_AT_OUT_16b(idx) \
  (*((ae_int16 *) p_out + m_itr + idx)) = \
  AE_ROUND16X4F32SSYM(0, AE_TRUNCA32F64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1 \
          AE_MULA8Q8X8(_ae_int32x2_acc_1, _ae_int32x2_acc_0,zero_temp,zero_temp,zero_temp,_ae_int8x8_mat1_0,_ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_1, _ae_int32x2_acc_0,zero_temp,zero_temp,zero_temp,_ae_int8x8_mat1_1_0,_ae_int8x8_vec1_1);

#define KERNEL_8x8_NOT_UNROLLED_MAT2_VEC2 \
          AE_MULA8Q8X8(_ae_int32x2_acc_0, _ae_int32x2_acc_0,zero_temp,zero_temp,zero_temp,_ae_int8x8_mat2_0,_ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0, _ae_int32x2_acc_0,zero_temp,zero_temp,zero_temp,_ae_int8x8_mat2_1_0,_ae_int8x8_vec2_1);

#define STORE_ACC_8bx8b_AT_OUT_8_SINGLE \
        ae_int8x8 temp;\
        ae_int32x2 temp_32;\
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);\
        temp_32  = AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_0);\
        temp_32 = AE_SLAI32S(temp_32,24); \
        temp_32 = AE_SRAI32(temp_32,24); \
        temp = AE_MOVINT8X8_FROMINT32X2(temp_32);\
        AE_S8_0_IP(temp,output_ptr,sizeof(ae_int8));

#define STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(idx) \
        _ae_int64_acc_ ## idx = AE_SLAA64S(_ae_int64_acc_ ## idx,acc_shift);\
        temp_32  = AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_ ## idx);\
        temp_32 = AE_SLAI32S(temp_32,24); \
        temp_32 = AE_SRAI32(temp_32,24); \
        temp_output = AE_MOVINT8X8_FROMINT32X2(temp_32);\
        AE_S8_0_IP(temp_output,output_ptr,sizeof(ae_int8));

#define ZER0_8x8_Temp_Variable \
        ae_int8x8 zero_temp = AE_MOVINT8X8_FROMINT32X2(0);

#define SETUP_OUTPUT_STORE_8_UNALIGNED_SUPPORT \
        ae_int8 * output_ptr; \
        output_ptr=(ae_int8 *)(p_out+m_itr);\
        ae_int8x8 temp_output;\
        ae_int32 temp_32;\


#define SETUP_OUTPUT_STORE_8x8 \
        ae_int8 * output_ptr; \
        output_ptr=(ae_int8 *)(p_out+m_itr);

#define SETUP_OUTPUT_STORE_8 \
        ae_int8 * output_ptr; \
        output_ptr=(ae_int8 *)(p_out+m_itr);

#define SETUP_OUTPUT_STORE_32x4 \
        ae_int32x4 * output_ptr; \
        output_ptr=(ae_int32x4 *)(p_out+m_itr);

#define SETUP_OUTPUT_STORE_32x4_SCRATCH \
        ae_int32x4 * output_ptr; \
        output_ptr=(ae_int32x4 *)((ae_int32 *)p_scratch+m_itr);

#define SETUP_OUTPUT_STORE_32_SCRATCH \
        ae_int32 * output_ptr; \
        output_ptr=(ae_int32 *)((ae_int32 *)p_scratch+m_itr);

#define SETUP_OUTPUT_STORE_32 \
        ae_int32 * output_ptr; \
        output_ptr=(ae_int32 *)(p_out+m_itr);

#define SETUP_OUTPUT_STORE_16x4 \
        ae_int16 * output_ptr;\
        output_ptr=(ae_int16 *)(p_out+m_itr);

#define SETUP_OUTPUT_STORE_16 \
        ae_int16 * output_ptr;\
        output_ptr=(ae_int16 *)(p_out+m_itr);

#define SETUP_OUTPUT_STORE_16_UNALIGNED_SUPPORT \
        ae_int16 * output_ptr;\
        output_ptr=(ae_int16 *)(p_out+m_itr); \
        ae_int32x2 temp_32;\
        ae_int16x4 temp_output;\

#define STORE_ACC_8bx8b_AT_OUT_8x8x2 \
        ae_int32x2 temp32_1, temp32_2, temp32_3, temp32_4;\
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);\
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);\
        _ae_int64_acc_2 = AE_SLAA64S(_ae_int64_acc_2 ,acc_shift);\
        _ae_int64_acc_3 = AE_SLAA64S(_ae_int64_acc_3 ,acc_shift);\
        _ae_int64_acc_4 = AE_SLAA64S(_ae_int64_acc_4 ,acc_shift);\
        _ae_int64_acc_5 = AE_SLAA64S(_ae_int64_acc_5 ,acc_shift);\
        _ae_int64_acc_6 = AE_SLAA64S(_ae_int64_acc_6 ,acc_shift);\
        _ae_int64_acc_7 = AE_SLAA64S(_ae_int64_acc_7 ,acc_shift);\
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);\
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);\
        _ae_int64_acc_1_2 = AE_SLAA64S(_ae_int64_acc_1_2 ,acc_shift);\
        _ae_int64_acc_1_3 = AE_SLAA64S(_ae_int64_acc_1_3 ,acc_shift);\
        _ae_int64_acc_1_4 = AE_SLAA64S(_ae_int64_acc_1_4 ,acc_shift);\
        _ae_int64_acc_1_5 = AE_SLAA64S(_ae_int64_acc_1_5 ,acc_shift);\
        _ae_int64_acc_1_6 = AE_SLAA64S(_ae_int64_acc_1_6 ,acc_shift);\
        _ae_int64_acc_1_7 = AE_SLAA64S(_ae_int64_acc_1_7 ,acc_shift);\
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);\
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);\
        temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_2, _ae_int64_acc_1_2);\
        temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_3, _ae_int64_acc_1_3);\
        temp32_1 = AE_SLAI32S(temp32_1,24); \
        temp32_1 = AE_SRAI32(temp32_1,24); \
        temp32_2 = AE_SLAI32S(temp32_2,24); \
        temp32_2 = AE_SRAI32(temp32_2,24); \
        temp32_3 = AE_SLAI32S(temp32_3,24); \
        temp32_3 = AE_SRAI32(temp32_3,24); \
        temp32_4 = AE_SLAI32S(temp32_4,24); \
        temp32_4 = AE_SRAI32(temp32_4,24); \
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_3,temp32_3)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_3,temp32_3)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_4,temp32_4)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_4,temp32_4)),output_ptr,sizeof(ae_int8));\
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_4, _ae_int64_acc_1_4);\
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_5, _ae_int64_acc_1_5);\
        temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_6, _ae_int64_acc_1_6);\
        temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_7, _ae_int64_acc_1_7);\
        temp32_1 = AE_SLAI32S(temp32_1,24); \
        temp32_1 = AE_SRAI32(temp32_1,24); \
        temp32_2 = AE_SLAI32S(temp32_2,24); \
        temp32_2 = AE_SRAI32(temp32_2,24); \
        temp32_3 = AE_SLAI32S(temp32_3,24); \
        temp32_3 = AE_SRAI32(temp32_3,24); \
        temp32_4 = AE_SLAI32S(temp32_4,24); \
        temp32_4 = AE_SRAI32(temp32_4,24); \
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_3,temp32_3)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_3,temp32_3)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_4,temp32_4)),output_ptr,sizeof(ae_int8));\
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_4,temp32_4)),output_ptr,sizeof(ae_int8));\

#define STORE_ACC_8bx8b_AT_OUT_8x8 \
        ae_int8x8 temp;\
        ae_int16x4 temp1,temp2;\
        temp1=AE_TRUNCA16X4F32S(_ae_int32x2_acc_0,_ae_int32x2_acc_1, 24+acc_shift);\
        temp2=AE_TRUNCA16X4F32S(_ae_int32x2_acc_2,_ae_int32x2_acc_3, 24+acc_shift);\
        temp=AE_ROUND8X8F16SSYM(temp1,temp2);\
        AE_S8X8_IP(temp,output_ptr,sizeof(ae_int8x8));

#define STORE_ACC_8bx8b_AT_OUT_16x4x2 \
        ae_int32x2 temp32_1, temp32_2, temp32_3, temp32_4;\
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);\
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);\
        _ae_int64_acc_2 = AE_SLAA64S(_ae_int64_acc_2 ,acc_shift);\
        _ae_int64_acc_3 = AE_SLAA64S(_ae_int64_acc_3 ,acc_shift);\
        _ae_int64_acc_4 = AE_SLAA64S(_ae_int64_acc_4 ,acc_shift);\
        _ae_int64_acc_5 = AE_SLAA64S(_ae_int64_acc_5 ,acc_shift);\
        _ae_int64_acc_6 = AE_SLAA64S(_ae_int64_acc_6 ,acc_shift);\
        _ae_int64_acc_7 = AE_SLAA64S(_ae_int64_acc_7 ,acc_shift);\
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);\
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);\
        _ae_int64_acc_1_2 = AE_SLAA64S(_ae_int64_acc_1_2 ,acc_shift);\
        _ae_int64_acc_1_3 = AE_SLAA64S(_ae_int64_acc_1_3 ,acc_shift);\
        _ae_int64_acc_1_4 = AE_SLAA64S(_ae_int64_acc_1_4 ,acc_shift);\
        _ae_int64_acc_1_5 = AE_SLAA64S(_ae_int64_acc_1_5 ,acc_shift);\
        _ae_int64_acc_1_6 = AE_SLAA64S(_ae_int64_acc_1_6 ,acc_shift);\
        _ae_int64_acc_1_7 = AE_SLAA64S(_ae_int64_acc_1_7 ,acc_shift);\
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);\
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);\
        temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_2, _ae_int64_acc_1_2);\
        temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_3, _ae_int64_acc_1_3);\
        temp32_1 = AE_SLAI32S(temp32_1, 16); \
        temp32_1 = AE_SRAI32(temp32_1,16); \
        temp32_2 = AE_SLAI32S(temp32_2, 16); \
        temp32_2 = AE_SRAI32(temp32_2,16); \
        temp32_3 = AE_SLAI32S(temp32_3, 16); \
        temp32_3 = AE_SRAI32(temp32_3,16); \
        temp32_4 = AE_SLAI32S(temp32_4, 16); \
        temp32_4 = AE_SRAI32(temp32_4,16); \
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_3,temp32_3)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_3,temp32_3)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_4,temp32_4)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_4,temp32_4)),output_ptr,sizeof(ae_int16));\
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_4, _ae_int64_acc_1_4);\
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_5, _ae_int64_acc_1_5);\
        temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_6, _ae_int64_acc_1_6);\
        temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_7, _ae_int64_acc_1_7);\
        temp32_1 = AE_SLAI32S(temp32_1, 16); \
        temp32_1 = AE_SRAI32(temp32_1,16); \
        temp32_2 = AE_SLAI32S(temp32_2, 16); \
        temp32_2 = AE_SRAI32(temp32_2,16); \
        temp32_3 = AE_SLAI32S(temp32_3, 16); \
        temp32_3 = AE_SRAI32(temp32_3,16); \
        temp32_4 = AE_SLAI32S(temp32_4, 16); \
        temp32_4 = AE_SRAI32(temp32_4,16); \
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_3,temp32_3)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_3,temp32_3)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_4,temp32_4)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_4,temp32_4)),output_ptr,sizeof(ae_int16));\

#define STORE_ACC_8bx8b_AT_OUT_16x4 \
        ae_int16x4 temp1,temp2;\
        _ae_int32x2_acc_0 = AE_SLAA32S(_ae_int32x2_acc_0,16+acc_shift);\
        _ae_int32x2_acc_1 = AE_SLAA32S(_ae_int32x2_acc_1,16+acc_shift);\
        _ae_int32x2_acc_2 = AE_SLAA32S(_ae_int32x2_acc_2,16+acc_shift);\
        _ae_int32x2_acc_3 = AE_SLAA32S(_ae_int32x2_acc_3,16+acc_shift);\
        temp1=AE_ROUND16X4F32SSYM(_ae_int32x2_acc_0,_ae_int32x2_acc_1);\
        temp2=AE_ROUND16X4F32SSYM(_ae_int32x2_acc_2,_ae_int32x2_acc_3);\
        AE_S16X4_IP(temp1,output_ptr,sizeof(ae_int16x4));\
        AE_S16X4_IP(temp2,output_ptr,sizeof(ae_int16x4));

#define STORE_ACC_8bx8b_AT_OUT_16_SINGLE \
        ae_int16x4 temp1;\
        ae_int32x2 temp_32;\
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);\
        temp_32  = AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_0);\
        temp_32 = AE_SLAI32S(temp_32,16); \
        temp_32 = AE_SRAI32(temp_32,16); \
        temp1 = AE_MOVINT16X4_FROMINT32X2(temp_32);\
        AE_S16_0_IP(temp1,output_ptr,sizeof(ae_int16));\

#define STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(idx) \
        _ae_int64_acc_ ## idx = AE_SLAA64S(_ae_int64_acc_ ## idx,acc_shift);\
        temp_32  = AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_ ## idx);\
        temp_32 = AE_SLAA32S(temp_32,16); \
        temp_32 = AE_SRAA32(temp_32,16); \
        temp_output = AE_MOVINT16X4_FROMINT32X2(temp_32);\
        AE_S16_0_IP(temp_output,output_ptr,sizeof(ae_int16));\

#define STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(idx_row,idx_vec,acc) \
        _ae_int64_acc_ ## acc = AE_SLAA64S(_ae_int64_acc_ ## acc,acc_shift);\
  (*((ae_int32 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_ ## acc); \

#define STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(idx) \
        _ae_int64_acc_ ## idx = AE_SLAA64S(_ae_int64_acc_ ## idx,acc_shift);\
        AE_S32_L_IP(AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_ ## idx),output_ptr,sizeof(ae_int32));

#define STORE_ACC_8bx8b_AT_OUT_32_SINGLE \
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);\
        AE_S32_L_IP(AE_ROUND32X2F64SSYM(ZERO32,_ae_int64_acc_0),output_ptr,sizeof(ae_int32));

#define STORE_ACC_8bx16b_AT_OUT_16x4(one,two,three,four)\
        ae_int32x2 temp1 ## one,temp2 ## one;\
        _ae_int64_acc_ ## one   =  AE_SLAA64S(_ae_int64_acc_ ## one  ,acc_shift);\
        _ae_int64_acc_ ## two   =  AE_SLAA64S(_ae_int64_acc_ ## two  ,acc_shift);\
        _ae_int64_acc_ ## three =  AE_SLAA64S(_ae_int64_acc_ ## three,acc_shift);\
        _ae_int64_acc_ ## four  =  AE_SLAA64S(_ae_int64_acc_ ## four ,acc_shift);\
        temp1 ## one = AE_ROUND32X2F64SSYM(_ae_int64_acc_ ## one,_ae_int64_acc_ ## two); \
        temp2 ## one = AE_ROUND32X2F64SSYM(_ae_int64_acc_ ## three,_ae_int64_acc_ ## four); \
        temp1 ## one =  AE_SLAI32S(temp1 ## one,16);\
        temp1 ## one =  AE_SRAI32(temp1 ## one,16);\
        temp2 ## one =  AE_SLAI32S(temp2 ## one,16);\
        temp2 ## one =  AE_SRAI32(temp2 ## one,16);\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp1 ## one,temp1 ## one)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp1 ## one,temp1 ## one)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp2 ## one, temp2 ## one)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp2 ## one, temp2 ## one)),output_ptr,sizeof(ae_int16));\


#define STORE_ACC_8bx16b_AT_OUT_16x4x4\
        ae_int32x2 temp1,temp2;\
        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);\
        _ae_int64_acc_1 =  AE_SLAA64S(_ae_int64_acc_1,acc_shift);\
        _ae_int64_acc_2 =  AE_SLAA64S(_ae_int64_acc_2,acc_shift);\
        _ae_int64_acc_3 =  AE_SLAA64S(_ae_int64_acc_3,acc_shift);\
        _ae_int64_acc_4 =  AE_SLAA64S(_ae_int64_acc_4,acc_shift);\
        _ae_int64_acc_5 =  AE_SLAA64S(_ae_int64_acc_5,acc_shift);\
        _ae_int64_acc_6 =  AE_SLAA64S(_ae_int64_acc_6,acc_shift);\
        _ae_int64_acc_7 =  AE_SLAA64S(_ae_int64_acc_7,acc_shift);\
        _ae_int64_acc_8  = AE_SLAA64S(_ae_int64_acc_8 ,acc_shift);\
        _ae_int64_acc_9  = AE_SLAA64S(_ae_int64_acc_9 ,acc_shift);\
        _ae_int64_acc_10 = AE_SLAA64S(_ae_int64_acc_10,acc_shift);\
        _ae_int64_acc_11 = AE_SLAA64S(_ae_int64_acc_11,acc_shift);\
        _ae_int64_acc_12 = AE_SLAA64S(_ae_int64_acc_12,acc_shift);\
        _ae_int64_acc_13 = AE_SLAA64S(_ae_int64_acc_13,acc_shift);\
        _ae_int64_acc_14 = AE_SLAA64S(_ae_int64_acc_14,acc_shift);\
        _ae_int64_acc_15 = AE_SLAA64S(_ae_int64_acc_15,acc_shift);\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_0,_ae_int64_acc_1); \
        temp2 = AE_ROUND32X2F64SSYM(_ae_int64_acc_2,_ae_int64_acc_3); \
        temp1 =  AE_SLAI32S(temp1,16);\
        temp1 =  AE_SRAI32(temp1,16);\
        temp2 =  AE_SLAI32S(temp2,16);\
        temp2 =  AE_SRAI32(temp2,16);\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_4,_ae_int64_acc_5);\
        temp2 = AE_ROUND32X2F64SSYM(_ae_int64_acc_6,_ae_int64_acc_7);\
        temp1 =  AE_SLAI32S(temp1,16);\
        temp1 =  AE_SRAI32(temp1,16);\
        temp2 =  AE_SLAI32S(temp2,16);\
        temp2 =  AE_SRAI32(temp2,16);\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_8,_ae_int64_acc_9);\
        temp2 = AE_ROUND32X2F64SSYM(_ae_int64_acc_10,_ae_int64_acc_11);\
        temp1 =  AE_SLAI32S(temp1,16);\
        temp1 =  AE_SRAI32(temp1,16);\
        temp2 =  AE_SLAI32S(temp2,16);\
        temp2 =  AE_SRAI32(temp2,16);\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_12,_ae_int64_acc_13);\
        temp2 = AE_ROUND32X2F64SSYM(_ae_int64_acc_14,_ae_int64_acc_15);\
        temp1 =  AE_SLAI32S(temp1,16);\
        temp1 =  AE_SRAI32(temp1,16);\
        temp2 =  AE_SLAI32S(temp2,16);\
        temp2 =  AE_SRAI32(temp2,16);\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp1,temp1)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp2,temp2)),output_ptr,sizeof(ae_int16));\
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp2,temp2)),output_ptr,sizeof(ae_int16));\


#define STORE_ACC_8bx8b_AT_OUT_32x4 \
        _ae_int32x2_acc_0 = AE_SLAA32S(_ae_int32x2_acc_0,acc_shift);\
        _ae_int32x2_acc_1 = AE_SLAA32S(_ae_int32x2_acc_1,acc_shift);\
        _ae_int32x2_acc_2 = AE_SLAA32S(_ae_int32x2_acc_2,acc_shift);\
        _ae_int32x2_acc_3 = AE_SLAA32S(_ae_int32x2_acc_3,acc_shift);\
        AE_S32X2X2_IP(_ae_int32x2_acc_0,_ae_int32x2_acc_1, output_ptr, sizeof(ae_int32x4));\
        AE_S32X2X2_IP(_ae_int32x2_acc_2,_ae_int32x2_acc_3, output_ptr, sizeof(ae_int32x4));\

#define STORE_ACC_8bx8b_AT_OUT_32x4x2 \
        ae_int32x2 temp32_1, temp32_2, temp32_3, temp32_4;\
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);\
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);\
        _ae_int64_acc_2 = AE_SLAA64S(_ae_int64_acc_2 ,acc_shift);\
        _ae_int64_acc_3 = AE_SLAA64S(_ae_int64_acc_3 ,acc_shift);\
        _ae_int64_acc_4 = AE_SLAA64S(_ae_int64_acc_4 ,acc_shift);\
        _ae_int64_acc_5 = AE_SLAA64S(_ae_int64_acc_5 ,acc_shift);\
        _ae_int64_acc_6 = AE_SLAA64S(_ae_int64_acc_6 ,acc_shift);\
        _ae_int64_acc_7 = AE_SLAA64S(_ae_int64_acc_7 ,acc_shift);\
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);\
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);\
        _ae_int64_acc_1_2 = AE_SLAA64S(_ae_int64_acc_1_2 ,acc_shift);\
        _ae_int64_acc_1_3 = AE_SLAA64S(_ae_int64_acc_1_3 ,acc_shift);\
        _ae_int64_acc_1_4 = AE_SLAA64S(_ae_int64_acc_1_4 ,acc_shift);\
        _ae_int64_acc_1_5 = AE_SLAA64S(_ae_int64_acc_1_5 ,acc_shift);\
        _ae_int64_acc_1_6 = AE_SLAA64S(_ae_int64_acc_1_6 ,acc_shift);\
        _ae_int64_acc_1_7 = AE_SLAA64S(_ae_int64_acc_1_7 ,acc_shift);\
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);\
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);\
        temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_2, _ae_int64_acc_1_2);\
        temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_3, _ae_int64_acc_1_3);\
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));\
        AE_S32X2X2_IP(temp32_3,temp32_4, output_ptr, sizeof(ae_int32x4));\
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_4, _ae_int64_acc_1_4);\
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_5, _ae_int64_acc_1_5);\
        temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_6, _ae_int64_acc_1_6);\
        temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_7, _ae_int64_acc_1_7);\
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));\
        AE_S32X2X2_IP(temp32_3,temp32_4, output_ptr, sizeof(ae_int32x4));\

#define STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx) \
  _ae_int32x2_acc_ ## idx = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ## idx, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) p_out + m_itr + idx)) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ## idx); \

/* ==================================================================================================== */
#define STORE_ACC_8bx16b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *) p_scratch + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_8bx16b_AT_OUT_16b(idx) \
  ae_int32 temp32 ## idx;\
  _ae_int64_acc_ ## idx =  AE_SLAA64S(_ae_int64_acc_ ## idx,acc_shift);\
  temp32 ## idx =  AE_ROUND32F64SSYM(_ae_int64_acc_ ## idx);\
  (*((ae_int16 *) p_out + m_itr + idx)) = AE_SAT16X4(0,temp32 ## idx); \

#define STORE_ACC_16bx8b_AT_OUT_16b STORE_ACC_8bx16b_AT_OUT_16b

#define STORE_ACC_8bx16b_AT_OUT_64b_64x2x8 \
        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);\
        _ae_int64_acc_1 =  AE_SLAA64S(_ae_int64_acc_1,acc_shift);\
        _ae_int64_acc_2 =  AE_SLAA64S(_ae_int64_acc_2,acc_shift);\
        _ae_int64_acc_3 =  AE_SLAA64S(_ae_int64_acc_3,acc_shift);\
        _ae_int64_acc_4 =  AE_SLAA64S(_ae_int64_acc_4,acc_shift);\
        _ae_int64_acc_5 =  AE_SLAA64S(_ae_int64_acc_5,acc_shift);\
        _ae_int64_acc_6 =  AE_SLAA64S(_ae_int64_acc_6,acc_shift);\
        _ae_int64_acc_7 =  AE_SLAA64S(_ae_int64_acc_7,acc_shift);\
        _ae_int64_acc_8  = AE_SLAA64S(_ae_int64_acc_8 ,acc_shift);\
        _ae_int64_acc_9  = AE_SLAA64S(_ae_int64_acc_9 ,acc_shift);\
        _ae_int64_acc_10 = AE_SLAA64S(_ae_int64_acc_10,acc_shift);\
        _ae_int64_acc_11 = AE_SLAA64S(_ae_int64_acc_11,acc_shift);\
        _ae_int64_acc_12 = AE_SLAA64S(_ae_int64_acc_12,acc_shift);\
        _ae_int64_acc_13 = AE_SLAA64S(_ae_int64_acc_13,acc_shift);\
        _ae_int64_acc_14 = AE_SLAA64S(_ae_int64_acc_14,acc_shift);\
        _ae_int64_acc_15 = AE_SLAA64S(_ae_int64_acc_15,acc_shift);\
        AE_S64X2_IP(_ae_int64_acc_0,_ae_int64_acc_1,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_2,_ae_int64_acc_3,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_4,_ae_int64_acc_5,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_6,_ae_int64_acc_7,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_8,_ae_int64_acc_9,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_10,_ae_int64_acc_11,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_12,_ae_int64_acc_13,output_ptr,sizeof(ae_int64x2));\
        AE_S64X2_IP(_ae_int64_acc_14,_ae_int64_acc_15,output_ptr,sizeof(ae_int64x2));\

#define STORE_ACC_8bx16b_AT_OUT_32b_32x4x4 \
        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);\
        _ae_int64_acc_1 =  AE_SLAA64S(_ae_int64_acc_1,acc_shift);\
        _ae_int64_acc_2 =  AE_SLAA64S(_ae_int64_acc_2,acc_shift);\
        _ae_int64_acc_3 =  AE_SLAA64S(_ae_int64_acc_3,acc_shift);\
        _ae_int64_acc_4 =  AE_SLAA64S(_ae_int64_acc_4,acc_shift);\
        _ae_int64_acc_5 =  AE_SLAA64S(_ae_int64_acc_5,acc_shift);\
        _ae_int64_acc_6 =  AE_SLAA64S(_ae_int64_acc_6,acc_shift);\
        _ae_int64_acc_7 =  AE_SLAA64S(_ae_int64_acc_7,acc_shift);\
        _ae_int64_acc_8  = AE_SLAA64S(_ae_int64_acc_8 ,acc_shift);\
        _ae_int64_acc_9  = AE_SLAA64S(_ae_int64_acc_9 ,acc_shift);\
        _ae_int64_acc_10 = AE_SLAA64S(_ae_int64_acc_10,acc_shift);\
        _ae_int64_acc_11 = AE_SLAA64S(_ae_int64_acc_11,acc_shift);\
        _ae_int64_acc_12 = AE_SLAA64S(_ae_int64_acc_12,acc_shift);\
        _ae_int64_acc_13 = AE_SLAA64S(_ae_int64_acc_13,acc_shift);\
        _ae_int64_acc_14 = AE_SLAA64S(_ae_int64_acc_14,acc_shift);\
        _ae_int64_acc_15 = AE_SLAA64S(_ae_int64_acc_15,acc_shift);\
        ae_int32x2 temp,temp1;\
        temp  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0,_ae_int64_acc_1);\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_2,_ae_int64_acc_3);\
        AE_S32X2X2_IP(temp,temp1,output_ptr,sizeof(ae_int32x4));\
        temp  = AE_ROUND32X2F64SSYM(_ae_int64_acc_4,_ae_int64_acc_5);\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_6,_ae_int64_acc_7);\
        AE_S32X2X2_IP(temp,temp1,output_ptr,sizeof(ae_int32x4));\
        temp  = AE_ROUND32X2F64SSYM(_ae_int64_acc_8,_ae_int64_acc_9);\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_10,_ae_int64_acc_11);\
        AE_S32X2X2_IP(temp,temp1,output_ptr,sizeof(ae_int32x4));\
        temp  = AE_ROUND32X2F64SSYM(_ae_int64_acc_12,_ae_int64_acc_13);\
        temp1 = AE_ROUND32X2F64SSYM(_ae_int64_acc_14,_ae_int64_acc_15);\
        AE_S32X2X2_IP(temp,temp1,output_ptr,sizeof(ae_int32x4));


#define STORE_ACC_8bx16b_AT_OUT_32b(idx) \
  (*((ae_int32 *) p_out + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_8bx16b_AT_OUT_64b(idx) \
  (*((ae_int64 *) p_out + m_itr + idx)) = \
  AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift); \

/* ==================================================================================================== */
#define STORE_ACC_16bx16b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *) p_scratch + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_16bx16b_AT_OUT_16b(idx) \
  ae_int32x2 temp_ ## idx;\
  _ae_int64_acc_ ## idx =  AE_SLAA64S(_ae_int64_acc_## idx,acc_shift);\
  temp_ ## idx = AE_ROUND32F64SSYM(_ae_int64_acc_ ## idx);\
  (*((ae_int16 *) p_out + m_itr + idx)) = AE_SAT16X4(0, temp_ ## idx); \

#define STORE_ACC_16bx16b_AT_OUT_32b(idx) \
  (*((ae_int32 *) p_out + m_itr + idx)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift)); \

#define STORE_ACC_16bx16b_AT_OUT_64b(idx) \
  (*((ae_int64 *) p_out + m_itr + idx)) = \
  AE_SLAA64S(_ae_int64_acc_ ## idx, acc_shift); \

/*------------------ time batching macros ----------------- */
#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_32b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_8b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_8bx8b_AT_OUT_32b(idx_row,idx_vec) \
  (*((ae_int32 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)); \

#define STORE_ACC_BATCH_8bx8b_AT_OUT_8b(idx_row,idx_vec) \
  ae_int16 _ae_int16_tmp_var_ ## idx_row ##_ ##idx_vec; \
  ae_f16x4 _ae_f16x4_tmp_var_ ## idx_row ##_ ##idx_vec = \
  AE_ROUND16X4F32SSYM(0, AE_TRUNCA32F64S(_ae_int64_acc_ ## idx_row ##_ ##idx_vec, acc_shift)); \
  _ae_int16_tmp_var_ ## idx_row ##_ ##idx_vec = AE_SLAA16S(_ae_f16x4_tmp_var_ ## idx_row ##_ ##idx_vec, 8); \
  (*((WORD8 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = (((*((UWORD16 *)&_ae_int16_tmp_var_ ## idx_row ##_ ##idx_vec)) & 0xFF00) >> 8); \

#define STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_ROW_16bx8b_AT_OUT_16b STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_8bx16b_AT_OUT_64b(idx_row,idx_vec) \
  (*((ae_int64 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_SLAA64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, acc_shift); \

#define STORE_ACC_BATCH_16bx8b_AT_OUT_16b(idx_row,idx_vec) \
  (*((ae_int16 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_ROUND16X4F32SSYM(0, AE_TRUNCA32F64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, acc_shift)); \

#define STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_64b(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_16b STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(idx_row,idx_vec,acc) \
  (*((ae_int64 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_SLAA64S(_ae_int64_acc_ ## acc, acc_shift); \

#define STORE_ACC_BATCH_16bx16b_AT_OUT_64b(idx_row,idx_vec) \
  (*((ae_int64 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_SLAA64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, acc_shift); \

#define STORE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row,idx_vec) \
  (*((ae_int16 *) p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
  AE_ROUND16X4F32SSYM(0, AE_TRUNCA32F64S(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, acc_shift)); \

#define STORE_ACC_BATCH_ROW_AT_OUT_f32(idx_row)\
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);\

#define STORE_ACC_BATCH_AT_OUT_f32(idx_row,idx_vec) \
  /*p_out value stored in a tmp pointer to make it inout for ISA */\
  p_out_tmp = (p_out[vec_itr + idx_vec] + m_itr + idx_row);\
  AE_SSIP(_xtfloat_acc_ ##idx_row ##_ ##idx_vec,p_out_tmp,0); \

#define STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row); \

#define STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row,idx_vec) \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MIN32(AE_MAX32(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *) (p_out[vec_itr + idx_vec] + m_itr + idx_row))) = (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec); \

/*---------------------------------------------------------*/
/* Specific macros needed for extra calculations involved
  for ASYM8b */

/* This is written to match with Tensorflow */
#define ADJUST_ACC_ASYM8b(idx) \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */ \
  ae_int32x2 _ae_int32x2_acc_ ## idx = AE_SLAA32(AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ## idx), left_shift); \
  _ae_int32x2_acc_ ## idx = AE_MULFP32X2RAS(_ae_int32x2_acc_ ## idx, AE_MOVDA32(out_multiplier)); \
  /* Shift by out_shift, same as Tensorflow */ \
  _ae_int64_acc_ ## idx = AE_SLAI64(AE_MOVINT64_FROMINT32X2(_ae_int32x2_acc_ ## idx), 32); \
  _ae_int64_acc_ ## idx = AE_SRAA64(_ae_int64_acc_ ## idx, right_shift); \
  _ae_int32x2_acc_ ## idx = AE_ROUND32F64SSYM(_ae_int64_acc_ ## idx); \
  /* Add output zero point */ \
  (_ae_int32x2_acc_ ## idx) = AE_ADD32S(_ae_int32x2_acc_ ## idx, AE_MOVDA32(out_zero_bias)); \

/* For time batching */
#define ADJUST_ACC_BATCH_ROW_ASYM8b(idx_row) \
  ADJUST_ACC_BATCH_VEC_UNROLL(idx_row); \

/* For time batching */
#define ADJUST_ACC_BATCH_ASYM8b(idx_row, idx_vec) \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */ \
  ae_int32x2 _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_SLAA32(AE_MOVINT32X2_FROMINT64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec), left_shift); \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_MULFP32X2RAS(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(out_multiplier)); \
  /* Shift by out_shift, same as Tensorflow */ \
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_SLAI64(AE_MOVINT64_FROMINT32X2(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec), 32); \
  _ae_int64_acc_ ##idx_row ##_ ##idx_vec = AE_SRAA64(_ae_int64_acc_ ##idx_row ##_ ##idx_vec, right_shift); \
  _ae_int32x2_acc_ ##idx_row ##_ ##idx_vec = AE_ROUND32F64SSYM(_ae_int64_acc_ ##idx_row ##_ ##idx_vec); \
  /* Add output zero point */ \
  (_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec) = AE_ADD32S(_ae_int32x2_acc_ ##idx_row ##_ ##idx_vec, AE_MOVDA32(out_zero_bias)); \

/*---------------------------------------------------------*/
/* ==================================================================================================== */
#if (ROW_UNROLL == 1)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)
#define SETUP_ROW_SUM_MAT2   UNROLL_SETUP_ROW_SUM_MAT2(0)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)
#define LOAD_MAT2            UNROLL_LOAD_MAT2(0)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)
#define ADJUST_MAC_MAT1_VEC1 UNROLL_ADJUST_MAC_MAT1_VEC1(0)
#define ADJUST_MAC_MAT2_VEC2 UNROLL_ADJUST_MAC_MAT2_VEC2(0)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)
#define STORE_ACC            UNROLL_STORE_ACC(0)

#elif (ROW_UNROLL == 2)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)   UNROLL_SETUP_ROW_SUM_MAT1(1)
#define SETUP_ROW_SUM_MAT2   UNROLL_SETUP_ROW_SUM_MAT2(0)   UNROLL_SETUP_ROW_SUM_MAT2(1)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)           UNROLL_SETUP_MAT2(1)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)            UNROLL_LOAD_MAT1(1)
#define LOAD_MAT2            UNROLL_LOAD_MAT2(0)            UNROLL_LOAD_MAT2(1)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)     UNROLL_KERNEL_MAT1_VEC1(1)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)     UNROLL_KERNEL_MAT2_VEC2(1)
#define ADJUST_MAC_MAT1_VEC1 UNROLL_ADJUST_MAC_MAT1_VEC1(0) UNROLL_ADJUST_MAC_MAT1_VEC1(1)
#define ADJUST_MAC_MAT2_VEC2 UNROLL_ADJUST_MAC_MAT2_VEC2(0) UNROLL_ADJUST_MAC_MAT2_VEC2(1)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)

#elif (ROW_UNROLL == 4)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)   UNROLL_SETUP_ROW_SUM_MAT1(1)   UNROLL_SETUP_ROW_SUM_MAT1(2)   UNROLL_SETUP_ROW_SUM_MAT1(3)
#define SETUP_ROW_SUM_MAT2   UNROLL_SETUP_ROW_SUM_MAT2(0)   UNROLL_SETUP_ROW_SUM_MAT2(1)   UNROLL_SETUP_ROW_SUM_MAT2(2)   UNROLL_SETUP_ROW_SUM_MAT2(3)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)           UNROLL_SETUP_MAT2(1)           UNROLL_SETUP_MAT2(2)           UNROLL_SETUP_MAT2(3)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)            UNROLL_LOAD_MAT1(1)            UNROLL_LOAD_MAT1(2)            UNROLL_LOAD_MAT1(3)
#define LOAD_MAT2            UNROLL_LOAD_MAT2(0)            UNROLL_LOAD_MAT2(1)            UNROLL_LOAD_MAT2(2)            UNROLL_LOAD_MAT2(3)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)     UNROLL_KERNEL_MAT1_VEC1(1)     UNROLL_KERNEL_MAT1_VEC1(2)     UNROLL_KERNEL_MAT1_VEC1(3)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)     UNROLL_KERNEL_MAT2_VEC2(1)     UNROLL_KERNEL_MAT2_VEC2(2)     UNROLL_KERNEL_MAT2_VEC2(3)
#define ADJUST_MAC_MAT1_VEC1 UNROLL_ADJUST_MAC_MAT1_VEC1(0) UNROLL_ADJUST_MAC_MAT1_VEC1(1) UNROLL_ADJUST_MAC_MAT1_VEC1(2) UNROLL_ADJUST_MAC_MAT1_VEC1(3)
#define ADJUST_MAC_MAT2_VEC2 UNROLL_ADJUST_MAC_MAT2_VEC2(0) UNROLL_ADJUST_MAC_MAT2_VEC2(1) UNROLL_ADJUST_MAC_MAT2_VEC2(2) UNROLL_ADJUST_MAC_MAT2_VEC2(3)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)         UNROLL_ADD_BIAS_ACC(2)         UNROLL_ADD_BIAS_ACC(3)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)           UNROLL_ADJUST_ACC(2)           UNROLL_ADJUST_ACC(3)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)            UNROLL_STORE_ACC(2)            UNROLL_STORE_ACC(3)

#elif (ROW_UNROLL == 8)
#define SETUP_ACC            UNROLL_SETUP_ACC(0)            UNROLL_SETUP_ACC(1)            UNROLL_SETUP_ACC(2)            UNROLL_SETUP_ACC(3)            UNROLL_SETUP_ACC(4)            UNROLL_SETUP_ACC(5)            UNROLL_SETUP_ACC(6)            UNROLL_SETUP_ACC(7)
#define SETUP_ROW_SUM_MAT1   UNROLL_SETUP_ROW_SUM_MAT1(0)   UNROLL_SETUP_ROW_SUM_MAT1(1)   UNROLL_SETUP_ROW_SUM_MAT1(2)   UNROLL_SETUP_ROW_SUM_MAT1(3)   UNROLL_SETUP_ROW_SUM_MAT1(4)   UNROLL_SETUP_ROW_SUM_MAT1(5)   UNROLL_SETUP_ROW_SUM_MAT1(6)   UNROLL_SETUP_ROW_SUM_MAT1(7)
#define SETUP_ROW_SUM_MAT2   UNROLL_SETUP_ROW_SUM_MAT2(0)   UNROLL_SETUP_ROW_SUM_MAT2(1)   UNROLL_SETUP_ROW_SUM_MAT2(2)   UNROLL_SETUP_ROW_SUM_MAT2(3)   UNROLL_SETUP_ROW_SUM_MAT2(4)   UNROLL_SETUP_ROW_SUM_MAT2(5)   UNROLL_SETUP_ROW_SUM_MAT2(6)   UNROLL_SETUP_ROW_SUM_MAT2(7)
#define SETUP_MAT1           UNROLL_SETUP_MAT1(0)           UNROLL_SETUP_MAT1(1)           UNROLL_SETUP_MAT1(2)           UNROLL_SETUP_MAT1(3)           UNROLL_SETUP_MAT1(4)           UNROLL_SETUP_MAT1(5)           UNROLL_SETUP_MAT1(6)           UNROLL_SETUP_MAT1(7)
#define SETUP_MAT2           UNROLL_SETUP_MAT2(0)           UNROLL_SETUP_MAT2(1)           UNROLL_SETUP_MAT2(2)           UNROLL_SETUP_MAT2(3)           UNROLL_SETUP_MAT2(4)           UNROLL_SETUP_MAT2(5)           UNROLL_SETUP_MAT2(6)           UNROLL_SETUP_MAT2(7)
#define LOAD_MAT1            UNROLL_LOAD_MAT1(0)        UNROLL_LOAD_MAT1(1)        UNROLL_LOAD_MAT1(2)        UNROLL_LOAD_MAT1(3)        UNROLL_LOAD_MAT1(4)        UNROLL_LOAD_MAT1(5)        UNROLL_LOAD_MAT1(6)        UNROLL_LOAD_MAT1(7)
#define LOAD_MAT2            UNROLL_LOAD_MAT2(0)        UNROLL_LOAD_MAT2(1)        UNROLL_LOAD_MAT2(2)        UNROLL_LOAD_MAT2(3)        UNROLL_LOAD_MAT2(4)        UNROLL_LOAD_MAT2(5)        UNROLL_LOAD_MAT2(6)        UNROLL_LOAD_MAT2(7)
#define KERNEL_MAT1_VEC1     UNROLL_KERNEL_MAT1_VEC1(0)     UNROLL_KERNEL_MAT1_VEC1(1)     UNROLL_KERNEL_MAT1_VEC1(2)     UNROLL_KERNEL_MAT1_VEC1(3)     UNROLL_KERNEL_MAT1_VEC1(4)     UNROLL_KERNEL_MAT1_VEC1(5)     UNROLL_KERNEL_MAT1_VEC1(6)     UNROLL_KERNEL_MAT1_VEC1(7)
#define KERNEL_MAT2_VEC2     UNROLL_KERNEL_MAT2_VEC2(0)     UNROLL_KERNEL_MAT2_VEC2(1)     UNROLL_KERNEL_MAT2_VEC2(2)     UNROLL_KERNEL_MAT2_VEC2(3)     UNROLL_KERNEL_MAT2_VEC2(4)     UNROLL_KERNEL_MAT2_VEC2(5)     UNROLL_KERNEL_MAT2_VEC2(6)     UNROLL_KERNEL_MAT2_VEC2(7)
#define ADJUST_MAC_MAT1_VEC1 UNROLL_ADJUST_MAC_MAT1_VEC1(0) UNROLL_ADJUST_MAC_MAT1_VEC1(1) UNROLL_ADJUST_MAC_MAT1_VEC1(2) UNROLL_ADJUST_MAC_MAT1_VEC1(3) UNROLL_ADJUST_MAC_MAT1_VEC1(4) UNROLL_ADJUST_MAC_MAT1_VEC1(5) UNROLL_ADJUST_MAC_MAT1_VEC1(6) UNROLL_ADJUST_MAC_MAT1_VEC1(7)
#define ADJUST_MAC_MAT2_VEC2 UNROLL_ADJUST_MAC_MAT2_VEC2(0) UNROLL_ADJUST_MAC_MAT2_VEC2(1) UNROLL_ADJUST_MAC_MAT2_VEC2(2) UNROLL_ADJUST_MAC_MAT2_VEC2(3) UNROLL_ADJUST_MAC_MAT2_VEC2(4) UNROLL_ADJUST_MAC_MAT2_VEC2(5) UNROLL_ADJUST_MAC_MAT2_VEC2(6) UNROLL_ADJUST_MAC_MAT2_VEC2(7)
#define ADD_BIAS_ACC         UNROLL_ADD_BIAS_ACC(0)         UNROLL_ADD_BIAS_ACC(1)         UNROLL_ADD_BIAS_ACC(2)         UNROLL_ADD_BIAS_ACC(3)         UNROLL_ADD_BIAS_ACC(4)         UNROLL_ADD_BIAS_ACC(5)         UNROLL_ADD_BIAS_ACC(6)         UNROLL_ADD_BIAS_ACC(7)
#define ADJUST_ACC           UNROLL_ADJUST_ACC(0)           UNROLL_ADJUST_ACC(1)           UNROLL_ADJUST_ACC(2)           UNROLL_ADJUST_ACC(3)           UNROLL_ADJUST_ACC(4)           UNROLL_ADJUST_ACC(5)           UNROLL_ADJUST_ACC(6)           UNROLL_ADJUST_ACC(7)
#define STORE_ACC            UNROLL_STORE_ACC(0)            UNROLL_STORE_ACC(1)            UNROLL_STORE_ACC(2)            UNROLL_STORE_ACC(3)            UNROLL_STORE_ACC(4)            UNROLL_STORE_ACC(5)            UNROLL_STORE_ACC(6)            UNROLL_STORE_ACC(7)
#elif (ROW_UNROLL == 16)

#define SETUP_ACC UNROLL_SETUP_ACC(0) UNROLL_SETUP_ACC(1) UNROLL_SETUP_ACC(2) UNROLL_SETUP_ACC(3) UNROLL_SETUP_ACC(4) UNROLL_SETUP_ACC(5) UNROLL_SETUP_ACC(6) UNROLL_SETUP_ACC(7) \
                  UNROLL_SETUP_ACC(8) UNROLL_SETUP_ACC(9) UNROLL_SETUP_ACC(10) UNROLL_SETUP_ACC(11) UNROLL_SETUP_ACC(12) UNROLL_SETUP_ACC(13) UNROLL_SETUP_ACC(14) UNROLL_SETUP_ACC(15)

#define SETUP_MAT1 UNROLL_SETUP_MAT1(0) UNROLL_SETUP_MAT1(1) UNROLL_SETUP_MAT1(2) UNROLL_SETUP_MAT1(3) UNROLL_SETUP_MAT1(4) UNROLL_SETUP_MAT1(5) UNROLL_SETUP_MAT1(6) UNROLL_SETUP_MAT1(7) \
                  UNROLL_SETUP_MAT1(8) UNROLL_SETUP_MAT1(9) UNROLL_SETUP_MAT1(10) UNROLL_SETUP_MAT1(11) UNROLL_SETUP_MAT1(12) UNROLL_SETUP_MAT1(13) UNROLL_SETUP_MAT1(14) UNROLL_SETUP_MAT1(15)

#define SETUP_MAT2 UNROLL_SETUP_MAT2(0) UNROLL_SETUP_MAT2(1) UNROLL_SETUP_MAT2(2) UNROLL_SETUP_MAT2(3) UNROLL_SETUP_MAT2(4) UNROLL_SETUP_MAT2(5) UNROLL_SETUP_MAT2(6) UNROLL_SETUP_MAT2(7) \
                  UNROLL_SETUP_MAT2(8) UNROLL_SETUP_MAT2(9) UNROLL_SETUP_MAT2(10) UNROLL_SETUP_MAT2(11) UNROLL_SETUP_MAT2(12) UNROLL_SETUP_MAT2(13) UNROLL_SETUP_MAT2(14) UNROLL_SETUP_MAT2(15)

#define ADD_BIAS_ACC UNROLL_ADD_BIAS_ACC(0) UNROLL_ADD_BIAS_ACC(1) UNROLL_ADD_BIAS_ACC(2) UNROLL_ADD_BIAS_ACC(3) UNROLL_ADD_BIAS_ACC(4) UNROLL_ADD_BIAS_ACC(5) UNROLL_ADD_BIAS_ACC(6) UNROLL_ADD_BIAS_ACC(7) \
                  UNROLL_ADD_BIAS_ACC(8) UNROLL_ADD_BIAS_ACC(9) UNROLL_ADD_BIAS_ACC(10) UNROLL_ADD_BIAS_ACC(11) UNROLL_ADD_BIAS_ACC(12) UNROLL_ADD_BIAS_ACC(13) UNROLL_ADD_BIAS_ACC(14) UNROLL_ADD_BIAS_ACC(15)

#endif /* (ROW_UNROLL == 1) */

#if (ROW_UNROLL == 4 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH                           UNROLL_SETUP_VEC_BATCH(0)               UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_ACC_BATCH                           UNROLL_ROW_SETUP_ACC_BATCH(0)           UNROLL_ROW_SETUP_ACC_BATCH(1)       UNROLL_ROW_SETUP_ACC_BATCH(2)       UNROLL_ROW_SETUP_ACC_BATCH(3)
#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row)       UNROLL_SETUP_ACC_BATCH(idx_row,0)       UNROLL_SETUP_ACC_BATCH(idx_row,1)
#define SETUP_ACC_BATCH_TAIL                      UNROLL_SETUP_ACC_BATCH(0,0)             UNROLL_SETUP_ACC_BATCH(1,0)         UNROLL_SETUP_ACC_BATCH(2,0)         UNROLL_SETUP_ACC_BATCH(3,0)

#define LOAD_VEC_BATCH                            UNROLL_LOAD_VEC_BATCH(0)                UNROLL_LOAD_VEC_BATCH(1)
#define LOAD_BATCH_MAT1                           UNROLL_LOAD_ROW_MAT1(0)                 UNROLL_LOAD_ROW_MAT1(1)             UNROLL_LOAD_ROW_MAT1(2)             UNROLL_LOAD_ROW_MAT1(3)

#define KERNEL_MAT1_VEC_BATCH                     UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0)     UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(1) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(2) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(3)
#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row) UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row,0) UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row,1)
#define KERNEL_MAT1_VEC_BATCH_TAIL                UNROLL_KERNEL_MAT1_VEC_BATCH(0,0)       UNROLL_KERNEL_MAT1_VEC_BATCH(1,0)   UNROLL_KERNEL_MAT1_VEC_BATCH(2,0)   UNROLL_KERNEL_MAT1_VEC_BATCH(3,0)

#define ADD_BIAS_ACC_BATCH                        UNROLL_ROW_ADD_BIAS_ACC(0)              UNROLL_ROW_ADD_BIAS_ACC(1)          UNROLL_ROW_ADD_BIAS_ACC(2)          UNROLL_ROW_ADD_BIAS_ACC(3)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row)    UNROLL_ADD_BIAS_ACC_BATCH(idx_row,0)    UNROLL_ADD_BIAS_ACC_BATCH(idx_row,1)
#define ADD_BIAS_ACC_BATCH_TAIL                   LOAD_BIAS                               UNROLL_ADD_BIAS_ACC_BATCH(0,0)      LOAD_BIAS                           UNROLL_ADD_BIAS_ACC_BATCH(1,0)      LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(2,0) LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(3,0)

#define STORE_ACC_BATCH                           UNROLL_ROW_STORE_ACC(0)                 UNROLL_ROW_STORE_ACC(1)             UNROLL_ROW_STORE_ACC(2)             UNROLL_ROW_STORE_ACC(3)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row)       UNROLL_STORE_ACC_BATCH(idx_row,0)       UNROLL_STORE_ACC_BATCH(idx_row,1)
#define STORE_ACC_BATCH_TAIL                      UNROLL_STORE_ACC_BATCH(0,0)             UNROLL_STORE_ACC_BATCH(1,0)         UNROLL_STORE_ACC_BATCH(2,0)         UNROLL_STORE_ACC_BATCH(3,0)

#define ADJUST_ACC_BATCH_TAIL                     UNROLL_ADJUST_ACC_BATCH(0, 0)           UNROLL_ADJUST_ACC_BATCH(1, 0)       UNROLL_ADJUST_ACC_BATCH(2, 0)       UNROLL_ADJUST_ACC_BATCH(3, 0)
#define ADJUST_ACC_BATCH                          UNROLL_ROW_ADJUST_ACC(0)                UNROLL_ROW_ADJUST_ACC(1)                UNROLL_ROW_ADJUST_ACC(2)            UNROLL_ROW_ADJUST_ACC(3)
#define ADJUST_ACC_BATCH_VEC_UNROLL(idx_row)      UNROLL_ADJUST_ACC_BATCH(idx_row,0)      UNROLL_ADJUST_ACC_BATCH(idx_row,1)

#endif /* (ROW_UNROLL == 4 && VEC_UNROLL == 2)*/

#define AE_SW_PRIME_64(p_char, tmp) \
    WORD8 *p_char_align_##p_char =  (WORD8 *)((unsigned int)p_char & ~0x7); \
    int sel_idx_##p_char = (unsigned int)p_char - (unsigned int)p_char_align_##p_char; \
    ae_int8x8 sel_##p_char = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(g_sel_pattern[2 * sel_idx_##p_char], g_sel_pattern[2 * sel_idx_##p_char + 1])); \
    AE_L8X8_IP(tmp, (ae_int8x8 *)p_char_align_##p_char, 8); 

#define AE_SW_LA8X8_IP(d, tmp, p_char) { \
      ae_int8x8 d_tmp; \
      AE_L8X8_IP(d_tmp, (ae_int8x8 *)p_char_align_##p_char, 8);\
      AE_DSEL8X8(d, tmp, tmp, d_tmp, sel_##p_char); \
    }

// Circular buffer size needs to be multiple of 8 
#define AE_SW_PRIME_CIRC_64(p_char, tmp) \
    WORD8 *p_char_align_##p_char =  (WORD8 *)((unsigned int)p_char & ~0x7); \
    int sel_idx_##p_char = (unsigned int)p_char - (unsigned int)p_char_align_##p_char; \
    ae_int8x8 sel_##p_char = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(g_sel_pattern[2 * sel_idx_##p_char], g_sel_pattern[2 * sel_idx_##p_char + 1])); \
    AE_L8X8_XC(tmp, (ae_int8x8 *)p_char_align_##p_char, 8); 
        
#define AE_SW_LA8X8_IC(d, tmp, p_char) { \
      ae_int8x8 d_tmp; \
      AE_L8X8_XC(d_tmp, (ae_int8x8 *)p_char_align_##p_char, 8);\
      AE_DSEL8X8(d, tmp, tmp, d_tmp, sel_##p_char); \
    }
       
#define STORE8X8_1 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x00776655L, 0x44332211L))                                  
#define STORE8X8_2 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x11007766L, 0x55443322L))                                  
#define STORE8X8_3 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x22110077L, 0x66554433L))
#define STORE8X8_4 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x33221100L, 0x77665544L))                                  
#define STORE8X8_5 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x44332211L, 0x00776655L))                                  
#define STORE8X8_6 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x55443322L, 0x11007766L))                                  
#define STORE8X8_7 AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x00112233L, 0x44556677L))

#define AE_SW_S8_1_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_1); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_2_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_2); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_3_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_3); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_4_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_4); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_5_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_5); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_6_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_6); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_7_XP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_7); \
     AE_S8_0_XP(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_1_X(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_1); \
     AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_2_X(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_2); \
     AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_3_X(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_3); \
     AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_4_X(d, p_char, offset) { \
    ae_int8x8 d_tmp;\
    d_tmp = AE_SEL8X8(d, d, STORE8X8_4); \
    AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
    }

#define AE_SW_S8_5_X(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_5); \
     AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_6_X(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_6); \
     AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_7_X(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_7); \
     AE_S8_0_X(d_tmp , (ae_int8 *) p_char, offset);\
     }

#define AE_SW_S8_4_IP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_4); \
     AE_S8_0_IP(d_tmp , (ae_int8 *) p_char, offset);\
     }
 
#define AE_SW_S8_6_IP(d, p_char, offset) { \
     ae_int8x8 d_tmp;\
     d_tmp = AE_SEL8X8(d, d, STORE8X8_6); \
     AE_S8_0_IP(d_tmp , (ae_int8 *) p_char, offset);\
     }

/* Alignment checking */
#define ALIGNED_PTR(ptr, alignment) ((((unsigned int)ptr & (alignment - 1))) == 0)

#endif /* __XA_NNLIB_COMMON_MACROS_H__ */
