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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

#ifndef AE_LAV32X2X2_XP
#define AE_SW_LAV32X2X2_XP(out0, out1, align_out, p_out, off) \
{ \
      ae_int16x4 d_out16_0, d_out16_1; \
      ae_int16x8 *p16x8_out = (ae_int16x8 *)p_out;\
      AE_LAV16X4X2_XP(d_out16_0, d_out16_1, align_out, p16x8_out, off); \
      d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
      d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
      out0 = AE_MOVINT32X2_FROMINT16X4(d_out16_0); \
      out1 = AE_MOVINT32X2_FROMINT16X4(d_out16_1); \
      p_out = (ae_int32x4*)p16x8_out;\
}
#else
#define AE_SW_LAV32X2X2_XP  AE_LAV32X2X2_XP
#endif

#define IND_MAX       1023
#define S32_MIN       (-(((int64_t) 1) << 31))
#define S32_MAX       ((((int64_t) 1) << 31) - 1)
#define SCHAR_MAX 0x7f
#define SCHAR_MIN (-SCHAR_MAX - 1)
#define VAR_MAX_16B   ((1 << 14) - (1 << 4))
#define ZERO32   AE_ZERO32()
#define ONES8   AE_MOVDA8(1)

WORD32 xa_nn_instance_norm_3D_8_8_nhwc(
  WORD8 *p_out,
  const WORD8 *p_inp,
  const WORD16 *p_alpha,
  const WORD32 *p_beta,
  const WORD32 *p_rsqrt,
  WORD32 input_height,
  WORD32 input_width,
  WORD32 input_channels,
  WORD32 output_shift,   /* Shift value to bring the final value to 8b */
  WORD32 mean_shift,  /* set to a S to do the division */
  WORD32 mean_scale,  /*Scale = (1<<S) /H*W */
  WORD32 sq_acc_shift, /*  set to a shift value of accumulation of squares to 32 bits*/
  /*WORD32 relu_flag,*/
  WORD32 min_val,     /* minimum Value for clamping if reluFlag is set to 1 */ 
  WORD32 max_val)
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_alpha, -1);
  XA_NNLIB_ARG_CHK_PTR(p_beta, -1);
  XA_NNLIB_ARG_CHK_PTR(p_rsqrt, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0 || input_channels <= 0),-1);
  XA_NNLIB_ARG_CHK_COND((output_shift <= -24 || output_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((mean_shift < -32 || mean_shift >= -2), -1);
  XA_NNLIB_ARG_CHK_COND((mean_shift >= sq_acc_shift), -1);
  XA_NNLIB_ARG_CHK_COND((mean_scale <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((sq_acc_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((min_val < -128 || min_val > 127), -1);
  XA_NNLIB_ARG_CHK_COND((max_val < min_val || max_val > 127), -1);

  WORD32 itr_c, itr_h, itr_w;
  WORD32 index[8];
  ae_int16x4 d_input_val0, d_input_val1, d16_mean0, d16_mean1;
  ae_int32x2 d_tmp0, d_tmp1, d_tmp2, d_tmp3;
  ae_int64 d64_tmp0, d64_tmp1, d64_tmp2, d64_tmp3;
  ae_int8x8 d8_mean;

  ae_int16x8 *pae_alpha = (ae_int16x8 *)p_alpha;
  ae_int32x4 *pae_beta = (ae_int32x4 *)p_beta;

  ae_valignx2 alpha_a, beta_a;

  alpha_a = AE_LA128_PP(pae_alpha);
  beta_a = AE_LA128_PP(pae_beta);

  ae_int32x2 d_mean_scale = AE_MOVDA32X2(mean_scale, mean_scale);

  for(itr_c = 0; itr_c < input_channels; itr_c+=8)
  {
    ae_int32x2 d_mean0 = AE_ZERO32();
    ae_int32x2 d_mean1 = AE_ZERO32();
    ae_int32x2 d_mean2 = AE_ZERO32();
    ae_int32x2 d_mean3 = AE_ZERO32();
    ae_int64 d_var0 = AE_ZERO64();
    ae_int64 d_var1 = AE_ZERO64();
    ae_int64 d_var2 = AE_ZERO64();
    ae_int64 d_var3 = AE_ZERO64();
    ae_int64 d_var4 = AE_ZERO64();
    ae_int64 d_var5 = AE_ZERO64();
    ae_int64 d_var6 = AE_ZERO64();
    ae_int64 d_var7 = AE_ZERO64();

    const ae_int8x16 *pae_inp = (const ae_int8x16 *)&p_inp[itr_c];
    ae_int8x16 *pae_out;
    ae_valignx2 inp_a, out_a;
    WORD32 ne8 = input_channels - itr_c >= 8 ? 8 : input_channels - itr_c;

    WORD32 loop_count = input_height * input_width;
    itr_h = 0;
    while(itr_h < input_height * input_width)
    {
      loop_count = input_height * input_width - itr_h < 1 << 17 ? input_height * input_width - itr_h : 1 << 17;
      d_tmp0 = AE_ZERO32();
      d_tmp1 = AE_ZERO32();
      d_tmp2 = AE_ZERO32();
      d_tmp3 = AE_ZERO32();
      for(itr_w = 0; itr_w < loop_count; itr_w++)
      {
        ae_int8x8 d0, d1;
        ae_f16x4 df0, df1;
        inp_a = AE_LA128_PP(pae_inp);
        AE_LAV8X8X2_XP(d0, d1, inp_a, pae_inp, ne8);
        AE_CVTA16X4X2F8(df0, df1, d0, 0);
        d_input_val0 = AE_MOVINT16X4_FROMF16X4(df0);
        d_input_val1 = AE_MOVINT16X4_FROMF16X4(df1);
        AE_MULA16X4(d_mean0, d_mean1, d_input_val0, AE_MOVDA16(1));
        AE_MULA16X4(d_mean2, d_mean3, d_input_val1, AE_MOVDA16(1));
        AE_MULA16X4(d_tmp0, d_tmp1, d_input_val0, d_input_val0);
        AE_MULA16X4(d_tmp2, d_tmp3, d_input_val1, d_input_val1);
        pae_inp = (const ae_int8x16 *)((WORD8 *)pae_inp + input_channels - ne8);
      }
      AE_ACCW32(d_var0, d_var1, d_tmp0, AE_ZERO32());
      AE_ACCW32(d_var2, d_var3, d_tmp1, AE_ZERO32());
      AE_ACCW32(d_var4, d_var5, d_tmp2, AE_ZERO32());
      AE_ACCW32(d_var6, d_var7, d_tmp3, AE_ZERO32());
      itr_h += itr_w;
    };
    // mean_4_times = ROUNDASYM(mean * mean_scale, mean_shift - 2);
    // mean_s8 = MINMAX32(ROUNDASYM(mean * mean_scale, mean_shift), -128, 127);
    AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_mean0, d_mean_scale);
    d64_tmp2 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift + 2);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times0 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp2, d64_tmp3);

    d64_tmp0 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift);

    d_mean0 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp0, d64_tmp1);

    AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_mean1, d_mean_scale);
    d64_tmp2 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift + 2);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times1 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp2, d64_tmp3);

    d64_tmp0 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift);

    d_mean1 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp0, d64_tmp1);

    AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_mean2, d_mean_scale);
    d64_tmp2 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift + 2);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times2 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp2, d64_tmp3);

    d64_tmp0 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift);

    d_mean2 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp0, d64_tmp1);

    AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_mean3, d_mean_scale);
    d64_tmp2 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift + 2);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift + 2);
    ae_int32x2 d_mean_4_times3 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp2, d64_tmp3);

    d64_tmp0 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + mean_shift);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + mean_shift);

    d_mean3 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp0, d64_tmp1);

    d16_mean0 = AE_SAT16X4(d_mean0, d_mean1);
    d16_mean1 = AE_SAT16X4(d_mean2, d_mean3);
    d8_mean = AE_SAT8X8X16(d16_mean0, d16_mean1);

    // var_shift = var >> sq_acc_shift;
    // UWORD32 upper_limit = ((1 << 14) - (1 << 4));
    // var = MINMAX32(ROUNDASYM(var_shift * mean_scale, mean_shift - sq_acc_shift));
    // UWORD32 index = MINMAX(((var << 4) - (mean_4_times * mean_4_times)) >> 4, 0, upper_limit);
    d_var0 = AE_SRAA64(d_var0, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var0), d_mean_scale);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + (mean_shift - sq_acc_shift));

    d_var1 = AE_SRAA64(d_var1, -sq_acc_shift);
    d64_tmp2 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var1), d_mean_scale);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp2, 32 + (mean_shift - sq_acc_shift));

    ae_int32x2 d_var_scaled0 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp1, d64_tmp3);

    d_var2 = AE_SRAA64(d_var2, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var2), d_mean_scale);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + (mean_shift - sq_acc_shift));

    d_var3 = AE_SRAA64(d_var3, -sq_acc_shift);
    d64_tmp2 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var3), d_mean_scale);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp2, 32 + (mean_shift - sq_acc_shift));

    ae_int32x2 d_var_scaled1 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp1, d64_tmp3);

    d_var4 = AE_SRAA64(d_var4, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var4), d_mean_scale);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + (mean_shift - sq_acc_shift));

    d_var5 = AE_SRAA64(d_var5, -sq_acc_shift);
    d64_tmp2 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var5), d_mean_scale);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp2, 32 + (mean_shift - sq_acc_shift));

    ae_int32x2 d_var_scaled2 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp1, d64_tmp3);

    d_var6 = AE_SRAA64(d_var6, -sq_acc_shift);
    d64_tmp0 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var6), d_mean_scale);
    d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + (mean_shift - sq_acc_shift));

    d_var7 = AE_SRAA64(d_var7, -sq_acc_shift);
    d64_tmp2 = AE_MUL32U_LL(AE_MOVINT32X2_FROMINT64(d_var7), d_mean_scale);
    d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp2, 32 + (mean_shift - sq_acc_shift));

    ae_int32x2 d_var_scaled3 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp1, d64_tmp3);

    ae_int32x2 d_ls4 = AE_MOVDA32X2(1 << 4, 1 << 4);
    AE_MUL32X2S_HH_LL(d_var0, d_var1, d_var_scaled0, d_ls4);
    AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_mean_4_times0, d_mean_4_times0);
    d_var0 = AE_SUB64(d_var0, d64_tmp0);
    d_var1 = AE_SUB64(d_var1, d64_tmp1);

    AE_MUL32X2S_HH_LL(d_var2, d_var3, d_var_scaled1, d_ls4);
    AE_MUL32X2S_HH_LL(d64_tmp2, d64_tmp3, d_mean_4_times1, d_mean_4_times1);
    d_var2 = AE_SUB64(d_var2, d64_tmp2);
    d_var3 = AE_SUB64(d_var3, d64_tmp3);

    AE_MUL32X2S_HH_LL(d_var4, d_var5, d_var_scaled2, d_ls4);
    AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_mean_4_times2, d_mean_4_times2);
    d_var4 = AE_SUB64(d_var4, d64_tmp0);
    d_var5 = AE_SUB64(d_var5, d64_tmp1);

    AE_MUL32X2S_HH_LL(d_var6, d_var7, d_var_scaled3, d_ls4);
    AE_MUL32X2S_HH_LL(d64_tmp2, d64_tmp3, d_mean_4_times3, d_mean_4_times3);
    d_var6 = AE_SUB64(d_var6, d64_tmp2);
    d_var7 = AE_SUB64(d_var7, d64_tmp3);

    ae_int32x2 d_const = AE_MOVDA32X2((1 << 14) - (1 << 4), (1 << 14) - (1 << 4));
    ae_int32x2 d_index[4];
    d_index[0] = AE_TRUNCA32X2F64S(d_var0, d_var1, 28);
    AE_MINMAX32(d_index[0], AE_ZERO32(), d_const);
    index[0] = AE_MOVAD32_H(d_index[0]);
    index[1] = AE_MOVAD32_L(d_index[0]);

    d_index[1] = AE_TRUNCA32X2F64S(d_var2, d_var3, 28);
    AE_MINMAX32(d_index[1], AE_ZERO32(), d_const);
    index[2] = AE_MOVAD32_H(d_index[1]);
    index[3] = AE_MOVAD32_L(d_index[1]);

    d_index[2] = AE_TRUNCA32X2F64S(d_var4, d_var5, 28);
    AE_MINMAX32(d_index[2], AE_ZERO32(), d_const);
    index[4] = AE_MOVAD32_H(d_index[2]);
    index[5] = AE_MOVAD32_L(d_index[2]);

    d_index[3] = AE_TRUNCA32X2F64S(d_var6, d_var7, 28);
    AE_MINMAX32(d_index[3], AE_ZERO32(), d_const);
    index[6] = AE_MOVAD32_H(d_index[3]);
    index[7] = AE_MOVAD32_L(d_index[3]);

    WORD32 lut_key_shift[8] = {0};
    WORD32 itr;
    ae_int32x2 d_var_scale[4];
    for(itr = 0; itr < 4; itr++)
    {
      lut_key_shift[2*itr+0] = (21 - AE_NSAZ32_L(AE_SEL32_HH(d_index[itr], d_index[itr])) + 1) & (~1);
      lut_key_shift[2*itr+0] = lut_key_shift[2*itr+0] < 2 ? 2 : lut_key_shift[2*itr+0] > 20 ? 20 : lut_key_shift[2*itr+0];
      lut_key_shift[2*itr+0] = index[2*itr+0] < 1 << 10 ? 0 : lut_key_shift[2*itr+0];

      lut_key_shift[2*itr+1] = (21 - AE_NSAZ32_L(d_index[itr]) + 1) & (~1);
      lut_key_shift[2*itr+1] = lut_key_shift[2*itr+1] < 2 ? 2 : lut_key_shift[2*itr+1] > 20 ? 20 : lut_key_shift[2*itr+1];
      lut_key_shift[2*itr+1] = index[2*itr+1] < 1 << 10 ? 0 : lut_key_shift[2*itr+1];

      d_tmp0 = SW_SRAA32S_INT32_INT32X2((*(ae_int32 *)(&p_rsqrt[index[2*itr+0] >> lut_key_shift[2*itr+0]])), (lut_key_shift[2*itr+0] >> 1));
      d_tmp1 = SW_SRAA32S_INT32_INT32X2((*(ae_int32 *)(&p_rsqrt[index[2*itr+1] >> lut_key_shift[2*itr+1]])), (lut_key_shift[2*itr+1] >> 1));
      d_var_scale[itr] = AE_SEL32_LL(d_tmp0, d_tmp1);
    }

    pae_inp = (const ae_int8x16 *)&p_inp[itr_c];
    pae_out = (ae_int8x16 *)&p_out[itr_c];

    ae_int16x4 d_alpha_val0, d_alpha_val1;
    ae_int32x2 d_beta_val0, d_beta_val1, d_beta_val2, d_beta_val3;

    WORD32 ne16, ne32_0, ne32_1;
    ne16 = input_channels - itr_c > 8 ? 16 : ((input_channels - itr_c)<<1);
    ne32_0 = input_channels - itr_c > 4 ? 16 : ((input_channels - itr_c)<<2);
    ne32_1 = input_channels - itr_c <= 4 ? 0 : ((input_channels - itr_c - 4)<<2);
    AE_LAV16X4X2_XP(d_alpha_val0, d_alpha_val1, alpha_a, pae_alpha, ne16);
    AE_SW_LAV32X2X2_XP(d_beta_val0, d_beta_val1, beta_a, pae_beta, ne32_0);
    AE_SW_LAV32X2X2_XP(d_beta_val2, d_beta_val3, beta_a, pae_beta, ne32_1);

    ae_int32x2 d_ls24 = AE_MOVDA32X2(1 << 24, 1 << 24);
#pragma concurrent
    for(itr_h = 0; itr_h < input_height * input_width; itr_h++)
    {
        // input_val = alpha_val * (input_val - mean_s8);
        // input_val = input_val * var_scale;
        // input_val += betaVal << 24;
        // PRIME_8X4F(pt_inp, inp_a);
        // AE_LA8X4F_IP(d_input_val, inp_a, pt_inp);
        ae_int8x8 d0, d1;
        inp_a = AE_LA128_PP(pae_inp);
        AE_LAV8X8X2_XP(d0, d1, inp_a, pae_inp, ne8);
        AE_SUBW8(d_input_val0, d_input_val1, d0, d8_mean);
        AE_MUL16X4(d_tmp0, d_tmp1, d_input_val0, d_alpha_val0);
        AE_MUL16X4(d_tmp2, d_tmp3, d_input_val1, d_alpha_val1);

        AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_tmp0, d_var_scale[0]);
        AE_MUL32X2S_HH_LL(d64_tmp2, d64_tmp3, d_tmp1, d_var_scale[1]);

        AE_MULA32X2S_HH_LL(d64_tmp0, d64_tmp1, d_beta_val0, d_ls24);
        AE_MULA32X2S_HH_LL(d64_tmp2, d64_tmp3, d_beta_val1, d_ls24);

        d64_tmp0 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + (output_shift - 24));
        d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + (output_shift - 24));
        d64_tmp2 = SW_SLAA64S_INT64_INT64(d64_tmp2, 32 + (output_shift - 24));
        d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp3, 32 + (output_shift - 24));

        d_tmp0 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp0, d64_tmp1);
        d_tmp1 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp2, d64_tmp3);
        ae_int16x4 d16_tmp0 = AE_SAT16X4(d_tmp0, d_tmp1);
        AE_MINMAX16(d16_tmp0, AE_MOVDA16(min_val), AE_MOVDA16(max_val));

        AE_MUL32X2S_HH_LL(d64_tmp0, d64_tmp1, d_tmp2, d_var_scale[2]);
        AE_MUL32X2S_HH_LL(d64_tmp2, d64_tmp3, d_tmp3, d_var_scale[3]);

        AE_MULA32X2S_HH_LL(d64_tmp0, d64_tmp1, d_beta_val2, d_ls24);
        AE_MULA32X2S_HH_LL(d64_tmp2, d64_tmp3, d_beta_val3, d_ls24);

        d64_tmp0 = SW_SLAA64S_INT64_INT64(d64_tmp0, 32 + (output_shift - 24));
        d64_tmp1 = SW_SLAA64S_INT64_INT64(d64_tmp1, 32 + (output_shift - 24));
        d64_tmp2 = SW_SLAA64S_INT64_INT64(d64_tmp2, 32 + (output_shift - 24));
        d64_tmp3 = SW_SLAA64S_INT64_INT64(d64_tmp3, 32 + (output_shift - 24));

        d_tmp2 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp0, d64_tmp1);
        d_tmp3 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d64_tmp2, d64_tmp3);
        ae_int16x4 d16_tmp1 = AE_SAT16X4(d_tmp2, d_tmp3);
        AE_MINMAX16(d16_tmp1, AE_MOVDA16(min_val), AE_MOVDA16(max_val));

        ae_int8x8 d8_res = AE_SAT8X8X16(d16_tmp0, d16_tmp1);

        out_a = AE_ZALIGN128();
        AE_SAV8X8X2_XP(d8_res, d8_res, out_a, pae_out, ne8);
        AE_SA128POS_FP(out_a, pae_out);
        pae_inp = (const ae_int8x16 *)((WORD8 *)pae_inp + input_channels - ne8);
        pae_out = (ae_int8x16 *)((WORD8 *)pae_out + input_channels - ne8);
    }
  }
  return 0;
}
